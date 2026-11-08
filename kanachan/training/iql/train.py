#!/usr/bin/env python3

import re
import datetime
import math
from pathlib import Path
import os
import logging
import sys
from typing import Any
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
from tensordict import TensorDict  # type: ignore
from tensordict.nn import TensorDictModule, TensorDictSequential  # type: ignore
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.amp.grad_scaler import GradScaler
from torch.distributed import (
    init_process_group,
    broadcast,
    ReduceOp,
    all_reduce,
)
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter
import kanachan.training.iql.config  # pylint: disable=unused-import  # noqa: F401
from kanachan.constants import MAX_NUM_ACTION_CANDIDATES
from kanachan.training.common import (
    get_distributed_environment,
    average_tensor,
    with_last,
    rescale_gradients,
    is_gradient_nonfinite,
    backoff_gradient_scale,
    get_gradient,
)
from kanachan.training.core.offline_rl import DataLoader, EpisodeReplayBuffer
from kanachan.training.core.checkpoint import (
    SnapshotWriter,
    TrainingState,
    load_training_state,
    save_training_state,
)
from kanachan.training.core.progress import TrainingProgress
from kanachan.training.core.optimizers import move_optimizer_state, consolidate_optimizer_state
from kanachan.nn import TwinQActor
from kanachan.nn.decision_factory import (
    make_backbone, make_candidate_head, make_state_head, candidate_head_keys,
    network_description,
)
from kanachan.nn.decision_value import transfer_bc_features
from kanachan.training.core import decision_checkpoint
from kanachan.training.core.final_rank import final_rank_loss
from kanachan.nn.functional import compute_td_target, expectile_loss
from kanachan.model_loader import dump_object, load_model
import kanachan.training.core.config as _config


def _training(
    *,
    training_data: Path,
    contiguous_training_data: bool,
    rewrite_rooms: int | None,
    rewrite_grades: int | None,
    replay_buffer_size: int,
    num_workers: int,
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype,
    value_network: nn.Module,
    q1_source_network: nn.Module,
    q2_source_network: nn.Module,
    q1_target_network: nn.Module,
    q2_target_network: nn.Module,
    reward_plugin: Path,
    discount_factor: float,
    expectile: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    v_max_gradient_norm: float,
    q_max_gradient_norm: float,
    value_optimizer: Optimizer,
    q1_optimizer: Optimizer,
    q2_optimizer: Optimizer,
    value_lr_scheduler: LRScheduler | None,
    q1_lr_scheduler: LRScheduler | None,
    q2_lr_scheduler: LRScheduler | None,
    target_update_interval: int,
    target_update_rate: float,
    snapshot_interval: int,
    num_samples: int,
    summary_writer: SummaryWriter,
    snapshot_writer: SnapshotWriter,
    initial_optimizer_steps: int = 0,
    initial_grad_scaler_state: dict | None = None,
    initial_rng_state: dict | None = None,
    final_rank_weight: float = 0.0,
) -> None:
    start_time = datetime.datetime.now()

    move_optimizer_state(value_optimizer)
    move_optimizer_state(q1_optimizer)
    move_optimizer_state(q2_optimizer)

    world_size, rank, local_rank = get_distributed_environment()

    is_amp_enabled = device.type == "cuda" and dtype != amp_dtype

    # Load the reward plugin.
    with open(reward_plugin, encoding="UTF-8") as file_pointer:
        exec(file_pointer.read(), globals())  # pylint: disable=exec-used

    if replay_buffer_size >= 1:
        data_loader = EpisodeReplayBuffer(
            training_data=training_data,
            contiguous_training_data=contiguous_training_data,
            num_skip_samples=num_samples,
            rewrite_rooms=rewrite_rooms,
            rewrite_grades=rewrite_grades,
            # pylint: disable=undefined-variable
            get_reward=get_reward,  # type: ignore # noqa: F821
            discount_factor=discount_factor,
            dtype=dtype,
            max_size=replay_buffer_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(num_workers >= 1),
        )
    else:
        data_loader = DataLoader(
            path=training_data,
            num_skip_samples=num_samples,
            rewrite_rooms=rewrite_rooms,
            rewrite_grades=rewrite_grades,
            # pylint: disable=undefined-variable
            get_reward=get_reward,  # type: ignore # noqa: F821
            dtype=dtype,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(num_workers >= 1),
            drop_last=(world_size >= 2),
        )

    last_snapshot: int | None = None
    if snapshot_interval > 0:
        last_snapshot = num_samples

    is_amp_enabled = device.type == "cuda" and dtype != amp_dtype
    autocast_kwargs: dict[str, Any] = {
        "device_type": device.type,
        "dtype": amp_dtype,
        "enabled": is_amp_enabled,
        # Reusing a source model under no_grad can cache detached AMP weights.
        "cache_enabled": False,
    }
    grad_scaler = GradScaler("cuda", enabled=is_amp_enabled)
    if initial_grad_scaler_state is not None and grad_scaler.is_enabled():
        grad_scaler.load_state_dict(initial_grad_scaler_state)

    batch_count = 0
    pending_samples = 0
    epoch_start_samples = num_samples
    configured_batch_size = batch_size
    progress = TrainingProgress(
        summary_writer,
        batch_size,
        gradient_accumulation_steps,
        target_update_interval,
        target_update_rate,
        initial_optimizer_steps=initial_optimizer_steps,
    )

    decision_checkpoint.restore(initial_rng_state)
    for data, is_last in with_last(data_loader):
        data = data.to(device=device)
        assert isinstance(data, TensorDict)

        local_batch_size: int = data.size(0)  # type: ignore

        action = data["action"]
        assert action.device == device
        assert action.dtype == torch.int32
        assert action.dim() == 1
        assert action.size(0) == data.size(0)

        done = data["next", "done"]
        assert done.device == device
        assert done.dtype == torch.bool
        assert done.dim() == 1
        assert done.size(0) == data.size(0)

        def _compute_q_target(
            q_target_model: nn.Module,
        ) -> tuple[torch.Tensor, float]:
            copy = TensorDict(
                {
                    "sparse": data["sparse"].detach().clone(),
                    "numeric": data["numeric"].detach().clone(),
                    "progression": data["progression"].detach().clone(),
                    "candidates": data["candidates"].detach().clone(),
                },
                batch_size=data.size(0),
                device=device,
            )
            with torch.autocast(**autocast_kwargs):
                q_target_model(copy)
            q = copy["action_value"]
            assert q.dim() == 2
            assert q.size(0) == data.size(0)
            assert q.size(1) == MAX_NUM_ACTION_CANDIDATES
            q = q[torch.arange(local_batch_size, device=device), action.long()]
            assert q.dim() == 1
            assert q.size(0) == data.size(0)

            q_batch_mean = q.detach().clone().mean()
            if world_size >= 2:
                average_tensor(q_batch_mean)

            return q, q_batch_mean.item()

        # Get the Q target value to compute the loss of the V model.
        with torch.no_grad():
            q1, q1_mean = _compute_q_target(q1_target_network)
            q2, q2_mean = _compute_q_target(q2_target_network)

            q = torch.minimum(q1, q2)
            q = q.detach().clone()
            assert q.dim() == 1
            assert q.size(0) == data.size(0)

        # Backprop for the V model.
        copy = TensorDict(
            {
                "sparse": data["sparse"].detach().clone(),
                "numeric": data["numeric"].detach().clone(),
                "progression": data["progression"].detach().clone(),
                "candidates": data["candidates"].detach().clone(),
            },
            batch_size=data.size(0),
            device=device,
        )
        with torch.autocast(**autocast_kwargs):
            value_network(copy)
        value = copy["state_value"]
        assert value.dim() == 1
        assert value.size(0) == data.size(0)

        value_mean = value.detach().clone().mean()
        if world_size >= 2:
            average_tensor(value_mean)

        value_loss = expectile_loss(q, value, expectile)
        if final_rank_weight > 0:
            auxiliary_loss = final_rank_loss(copy["final_rank_logits"], data)
            value_loss = value_loss + final_rank_weight * auxiliary_loss
            if rank == 0:
                summary_writer.add_scalar("Final Rank Loss", auxiliary_loss.item(), num_samples)
                summary_writer.add_scalar(
                    "Final Rank Labeled Samples", data["next", "end_of_game"].sum().item(), num_samples
                )
        if not math.isfinite(value_loss.item()):
            raise RuntimeError("Value loss becomes non-finite.")

        _value_loss = value_loss.detach().clone()
        if world_size >= 2:
            average_tensor(_value_loss)
        value_loss_to_display = _value_loss.item()

        value_loss *= local_batch_size / (configured_batch_size * gradient_accumulation_steps)
        grad_scaler.scale(value_loss).backward()

        # Get the reward to compute the loss of the Q source models.
        reward = data["next", "reward"]

        # Get V to compute the loss of the Q source models.
        # A terminal observation in the RL format is entirely padding. Never
        # feed it to the decision decoder (even if the resulting V is masked).
        discount = data.get(("next", "discount"), discount_factor)
        discount_tensor = torch.as_tensor(discount, device=device)
        bootstrap = ~done & (discount_tensor != 0)
        copy = TensorDict(
            {
                key: data["next", key][bootstrap].detach().clone()
                for key in ("sparse", "numeric", "progression", "candidates")
            },
            batch_size=int(bootstrap.sum().item()),
            device=device,
        )
        with torch.no_grad(), torch.autocast(**autocast_kwargs):
            value = torch.zeros_like(reward)
            if copy.size(0) > 0:
                # No-gradient evaluation must bypass DDP: ranks may have
                # different numbers of nonterminal samples, including zero.
                value_evaluator = (
                    value_network.module
                    if isinstance(value_network, DistributedDataParallel) else value_network
                )
                value_evaluator(copy)
                value = value.to(copy["state_value"].dtype)
                value[bootstrap] = copy["state_value"]
            target = compute_td_target(
                reward, value, done, discount
            )

        def _backprop(q_source_network: nn.Module) -> float:
            copy = TensorDict(
                {
                    "sparse": data["sparse"].detach().clone(),
                    "numeric": data["numeric"].detach().clone(),
                    "progression": data["progression"].detach().clone(),
                    "candidates": data["candidates"].detach().clone(),
                },
                batch_size=data.size(0),
                device=device,
            )
            with torch.autocast(**autocast_kwargs):
                q_source_network(copy)
            q = copy["action_value"]
            assert q.dim() == 2
            assert q.size(0) == data.size(0)
            assert q.size(1) == MAX_NUM_ACTION_CANDIDATES
            q = q[torch.arange(local_batch_size, device=device), action.long()]

            loss_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            q_loss = target.to(loss_dtype) - q.to(loss_dtype)
            q_loss = q_loss**2.0
            q_loss = torch.mean(q_loss)
            if not math.isfinite(q_loss.item()):
                raise RuntimeError("Q loss becomes non-finite.")

            _q_loss = q_loss.detach().clone()
            if world_size >= 2:
                average_tensor(_q_loss)
            q_loss_to_display = _q_loss.item()

            q_loss *= local_batch_size / (configured_batch_size * gradient_accumulation_steps)
            grad_scaler.scale(q_loss).backward()

            return q_loss_to_display

        # Backprop for the Q1 source network.
        q1_loss_to_display = _backprop(q1_source_network)

        # Backprop for the Q2 source network.
        q2_loss_to_display = _backprop(q2_source_network)

        num_samples += local_batch_size * world_size
        pending_samples += local_batch_size * world_size
        batch_count += 1

        def _step(
            network: nn.Module,
            max_gradient_norm: float,
            optimizer: Optimizer,
            scheduler,
        ) -> float:
            gradient = get_gradient(network)
            # pylint: disable=not-callable
            gradient_norm = float(torch.linalg.vector_norm(gradient).item())
            nn.utils.clip_grad_norm_(
                network.parameters(),
                max_gradient_norm,
                error_if_nonfinite=True,
            )
            grad_scaler.step(optimizer)
            if scheduler is not None:
                scheduler.step()

            return gradient_norm

        if batch_count % gradient_accumulation_steps == 0 or is_last:
            normalization = (
                configured_batch_size * world_size * gradient_accumulation_steps
                / pending_samples
            )
            rescale_gradients(value_network, normalization)
            rescale_gradients(q1_source_network, normalization)
            rescale_gradients(q2_source_network, normalization)
            pending_samples = 0
            grad_scaler.unscale_(value_optimizer)
            grad_scaler.unscale_(q1_optimizer)
            grad_scaler.unscale_(q2_optimizer)
            is_grad_nonfinite = (
                is_gradient_nonfinite(value_network)
                or is_gradient_nonfinite(q1_source_network)
                or is_gradient_nonfinite(q2_source_network)
            )
            is_grad_nonfinite = torch.where(
                is_grad_nonfinite,
                torch.ones_like(is_grad_nonfinite),
                torch.zeros_like(is_grad_nonfinite),
            )
            if world_size >= 2:
                all_reduce(is_grad_nonfinite)
            if is_grad_nonfinite.item() >= 1:
                logging.warning(
                    "Skip an optimization step because of non-finite "
                    "values in the gradient."
                )
                backoff_gradient_scale(grad_scaler)
                value_optimizer.zero_grad()
                q1_optimizer.zero_grad()
                q2_optimizer.zero_grad()
                continue

            v_gradient_norm = _step(
                value_network,
                v_max_gradient_norm,
                value_optimizer,
                value_lr_scheduler,
            )
            q1_gradient_norm = _step(
                q1_source_network,
                q_max_gradient_norm,
                q1_optimizer,
                q1_lr_scheduler,
            )
            q2_gradient_norm = _step(
                q2_source_network,
                q_max_gradient_norm,
                q2_optimizer,
                q2_lr_scheduler,
            )
            grad_scaler.update()
            progress.optimizer_step(num_samples)

            value_optimizer.zero_grad()
            q1_optimizer.zero_grad()
            q2_optimizer.zero_grad()

            if progress.total_optimizer_steps % target_update_interval == 0:
                with torch.no_grad():
                    for _param, _target_param in zip(
                        q1_source_network.parameters(),
                        q1_target_network.parameters(),
                    ):
                        _target_param.data *= 1.0 - target_update_rate
                        _target_param.data += (
                            target_update_rate * _param.data.detach().clone()
                        )
                    for _param, _target_param in zip(
                        q2_source_network.parameters(),
                        q2_target_network.parameters(),
                    ):
                        _target_param.data *= 1.0 - target_update_rate
                        _target_param.data += (
                            target_update_rate * _param.data.detach().clone()
                        )

                progress.target_update(num_samples)

            if rank == 0:
                logging.info(
                    "sample = %d, V loss = %E, Q1 loss = %E, Q2 loss = %E, "
                    "V gradient norm = %E, Q1 gradient norm = %E, "
                    "Q2 gradient norm = %E",
                    num_samples,
                    value_loss_to_display,
                    q1_loss_to_display,
                    q2_loss_to_display,
                    v_gradient_norm,
                    q1_gradient_norm,
                    q2_gradient_norm,
                )
                summary_writer.add_scalars(
                    "Q", {"Q1": q1_mean, "Q2": q2_mean}, num_samples
                )
                summary_writer.add_scalars(
                    "Q Gradient Norm",
                    {"Q1": q1_gradient_norm, "Q2": q2_gradient_norm},
                    num_samples,
                )
                summary_writer.add_scalars(
                    "Q Loss",
                    {"Q1": q1_loss_to_display, "Q2": q2_loss_to_display},
                    num_samples,
                )
                summary_writer.add_scalar(
                    "Value", value_mean.item(), num_samples
                )
                summary_writer.add_scalar(
                    "Value Gradient Norm", v_gradient_norm, num_samples
                )
                summary_writer.add_scalar(
                    "Value Loss", value_loss_to_display, num_samples
                )
        else:
            if rank == 0:
                logging.info(
                    "sample = %d, value loss = %E, Q1 loss = %E, Q2 loss = %E",
                    num_samples,
                    value_loss_to_display,
                    q1_loss_to_display,
                    q2_loss_to_display,
                )
                summary_writer.add_scalars(
                    "Q", {"Q1": q1_mean, "Q2": q2_mean}, num_samples
                )
                summary_writer.add_scalars(
                    "Q Loss",
                    {"Q1": q1_loss_to_display, "Q2": q2_loss_to_display},
                    num_samples,
                )
                summary_writer.add_scalar(
                    "Value", value_mean.item(), num_samples
                )
                summary_writer.add_scalar(
                    "Value Loss", value_loss_to_display, num_samples
                )

        if (
            last_snapshot is not None
            and num_samples - last_snapshot >= snapshot_interval
            and pending_samples == 0
            and (
                num_workers <= 1
                or (num_samples - epoch_start_samples)
                % (configured_batch_size * world_size * num_workers)
                == 0
            )
        ):
            consolidate_optimizer_state(value_optimizer, q1_optimizer, q2_optimizer)
            if rank == 0:
                snapshot_writer(
                    num_samples, optimizer_steps=progress.total_optimizer_steps,
                    grad_scaler_state=grad_scaler.state_dict() if grad_scaler.is_enabled() else None,
                )
            last_snapshot = num_samples

    progress.finish(0)
    elapsed_time = datetime.datetime.now() - start_time
    consolidate_optimizer_state(value_optimizer, q1_optimizer, q2_optimizer)

    if rank == 0:
        logging.info(
            "A training has finished (elapsed time = %s).", elapsed_time
        )
        snapshot_writer(
            None,
            resume_num_samples=num_samples - pending_samples,
            optimizer_steps=progress.total_optimizer_steps,
            cursor_is_exact=(num_workers <= 1 or pending_samples == 0),
            grad_scaler_state=grad_scaler.state_dict() if grad_scaler.is_enabled() else None,
        )


@hydra.main(version_base=None, config_name="config")
def _main(config: DictConfig) -> None:
    (
        world_size,
        rank,
        local_rank,
        device,
        dtype,
        amp_dtype,
    ) = _config.device.validate(config)

    if not config.training_data.exists():
        errmsg = f"{config.training_data}: Does not exist."
        raise RuntimeError(errmsg)
    if not config.training_data.is_file():
        errmsg = f"{config.training_data}: Not a file."
        raise RuntimeError(errmsg)

    if isinstance(config.rewrite_rooms, str):
        config.rewrite_rooms = {
            "bronze": 0,
            "silver": 1,
            "gold": 2,
            "jade": 3,
            "throne": 4,
        }[config.rewrite_rooms]
    if config.rewrite_rooms is not None and (
        config.rewrite_rooms < 0 or 4 < config.rewrite_rooms
    ):
        errmsg = (
            f"{config.rewrite_rooms}: "
            "`rewrite_rooms` must be an integer within the range [0, 4]."
        )
        raise RuntimeError(errmsg)

    if isinstance(config.rewrite_grades, str):
        config.rewrite_grades = {
            "novice1": 0,
            "novice2": 1,
            "novice3": 2,
            "adept1": 3,
            "adept2": 4,
            "adept3": 5,
            "expert1": 6,
            "expert2": 7,
            "expert3": 8,
            "master1": 9,
            "master2": 10,
            "master3": 11,
            "saint1": 12,
            "saint2": 13,
            "saint3": 14,
            "celestial": 15,
        }[config.rewrite_grades]
    if config.rewrite_grades is not None and (
        config.rewrite_grades < 0 or 15 < config.rewrite_grades
    ):
        errmsg = (
            f"{config.rewrite_grades}: "
            "`rewrite_grades` must be an integer within the range [0, 15]."
        )
        raise RuntimeError(errmsg)

    if config.num_workers is None:
        config.num_workers = 0 if device.type == "cpu" else 2
    if config.num_workers < 0:
        raise RuntimeError("num_workers must be a non-negative integer.")

    if config.replay_buffer_size < 0:
        errmsg = (
            f"{config.replay_buffer_size}: `replay_buffer_size` must be"
            " a non-negative integer."
        )
        raise RuntimeError(errmsg)
    if config.contiguous_training_data and config.replay_buffer_size == 0:
        errmsg = "Use `replay_buffer_size` for `contiguous_training_data`."
        raise RuntimeError(errmsg)

    _config.encoder.validate(config.encoder)

    _config.decoder.validate(config.decoder, config.encoder.dimension)

    if config.initial_model_prefix is not None and config.decoder.load_from is not None:
        raise RuntimeError("initial_model_prefix conflicts with decoder.load_from.")

    if config.decoder.load_from is not None:
        raise RuntimeError(
            "decoder.load_from is not supported for separate Q/V heads. "
            "Use encoder.load_from for BC features, then initialize the AWR "
            "policy with the BC encoder and decoder when extracting a policy."
        )

    if config.bc_initial_model is not None:
        if config.encoder.architecture != "dense_decoder":
            raise ValueError("bc_initial_model requires a dense decision decoder.")
        if config.initial_model_prefix is not None or config.encoder.load_from is not None:
            raise ValueError("bc_initial_model conflicts with resume or encoder.load_from.")
        if not config.bc_initial_model.is_file():
            raise ValueError("bc_initial_model must be an existing model file.")
    if not math.isfinite(config.final_rank_weight) or config.final_rank_weight < 0:
        raise ValueError("final_rank_weight must be finite and nonnegative.")
    if config.final_rank_weight > 0 and config.encoder.architecture != "dense_decoder":
        raise ValueError("The final-rank auxiliary head requires a dense decision decoder.")
    if not config.double_q_learning:
        raise ValueError("IQL requires two critics; double_q_learning=false is not supported.")

    if config.initial_model_prefix is not None:
        if config.encoder.load_from is not None:
            errmsg = (
                "`initial_model_prefix` conflicts with `encoder.load_from`."
            )
            raise RuntimeError(errmsg)
        if not config.initial_model_prefix.exists():
            errmsg = f"{config.initial_model_prefix}: Does not exist."
            raise RuntimeError(errmsg)
        if not config.initial_model_prefix.is_dir():
            errmsg = f"{config.initial_model_prefix}: Not a directory."
            raise RuntimeError(errmsg)

    if config.initial_model_index is not None:
        if config.initial_model_prefix is None:
            errmsg = (
                "`initial_model_index` must be combined with"
                " `initial_model_prefix`."
            )
            raise RuntimeError(errmsg)
        if config.initial_model_index < 0:
            errmsg = (
                f"{config.initial_model_index}: An invalid initial model"
                " index."
            )
            raise RuntimeError(errmsg)

    num_samples = 0
    initial_optimizer_steps = 0
    initial_grad_scaler_state = None
    value_encoder_snapshot_path: Path | None = None
    value_decoder_snapshot_path: Path | None = None
    q1_source_encoder_snapshot_path: Path | None = None
    q1_source_decoder_snapshot_path: Path | None = None
    q2_source_encoder_snapshot_path: Path | None = None
    q2_source_decoder_snapshot_path: Path | None = None
    q1_target_encoder_snapshot_path: Path | None = None
    q1_target_decoder_snapshot_path: Path | None = None
    q2_target_encoder_snapshot_path: Path | None = None
    q2_target_decoder_snapshot_path: Path | None = None
    value_optimizer_snapshot_path: Path | None = None
    q1_optimizer_snapshot_path: Path | None = None
    q2_optimizer_snapshot_path: Path | None = None
    value_lr_scheduler_snapshot_path: Path | None = None
    q1_lr_scheduler_snapshot_path: Path | None = None
    q2_lr_scheduler_snapshot_path: Path | None = None

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None

        if config.initial_model_index is None:
            for child in os.listdir(config.initial_model_prefix):
                match = re.search(
                    "^(?:(?:(?:value|q[12]-source|q[12]-target)-(?:encoder|decoder))|value-optimizer|q[12]-optimizer|value-lr-scheduler|q[12]-lr-scheduler)(?:\\.(\\d+))?\\.pth$",
                    child,
                )
                if match is None:
                    continue
                if match[1] is None:
                    config.initial_model_index = sys.maxsize
                    continue
                if (
                    config.initial_model_index is None
                    or int(match[1]) > config.initial_model_index
                ):
                    config.initial_model_index = int(match[1])
                    continue
        if config.initial_model_index is None:
            errmsg = f"{config.initial_model_prefix}: No model snapshot found."
            raise RuntimeError(errmsg)

        if config.initial_model_index == sys.maxsize:
            config.initial_model_index = 0
            infix = ""
        else:
            num_samples = config.initial_model_index
            infix = f".{num_samples}"

        training_state = load_training_state(
            config.initial_model_prefix,
            infix,
            num_samples,
            restart_data=config.optimizer.initialize,
            start_new_epoch=config.start_new_epoch,
        )
        num_samples = training_state.num_samples
        initial_optimizer_steps = training_state.optimizer_steps
        initial_grad_scaler_state = training_state.grad_scaler_state
        if config.replay_buffer_size > 0 and num_samples > 0:
            if not config.optimizer.initialize:
                raise RuntimeError(
                    "Replay checkpoints do not store the shuffle buffer. "
                    "Use optimizer.initialize=true to reuse the weights and restart the data."
                )
            num_samples = 0
            initial_optimizer_steps = 0

        value_encoder_snapshot_path = Path(
            config.initial_model_prefix / f"value-encoder{infix}.pth"
        )
        if not value_encoder_snapshot_path.exists():
            errmsg = f"{value_encoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not value_encoder_snapshot_path.is_file():
            errmsg = f"{value_encoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        value_decoder_snapshot_path = Path(
            config.initial_model_prefix / f"value-decoder{infix}.pth"
        )
        if not value_decoder_snapshot_path.exists():
            errmsg = f"{value_decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not value_decoder_snapshot_path.is_file():
            errmsg = f"{value_decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q1_source_encoder_snapshot_path = Path(
            config.initial_model_prefix / f"q1-source-encoder{infix}.pth"
        )
        if not q1_source_encoder_snapshot_path.exists():
            errmsg = f"{q1_source_encoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q1_source_encoder_snapshot_path.is_file():
            errmsg = f"{q1_source_encoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q1_source_decoder_snapshot_path = Path(
            config.initial_model_prefix / f"q1-source-decoder{infix}.pth"
        )
        if not q1_source_decoder_snapshot_path.exists():
            errmsg = f"{q1_source_decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q1_source_decoder_snapshot_path.is_file():
            errmsg = f"{q1_source_decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q2_source_encoder_snapshot_path = Path(
            config.initial_model_prefix / f"q2-source-encoder{infix}.pth"
        )
        if not q2_source_encoder_snapshot_path.exists():
            errmsg = f"{q2_source_encoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q2_source_encoder_snapshot_path.is_file():
            errmsg = f"{q2_source_encoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q2_source_decoder_snapshot_path = Path(
            config.initial_model_prefix / f"q2-source-decoder{infix}.pth"
        )
        if not q2_source_decoder_snapshot_path.exists():
            errmsg = f"{q2_source_decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q2_source_decoder_snapshot_path.is_file():
            errmsg = f"{q2_source_decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q1_target_encoder_snapshot_path = Path(
            config.initial_model_prefix / f"q1-target-encoder{infix}.pth"
        )
        if not q1_target_encoder_snapshot_path.exists():
            errmsg = f"{q1_target_encoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q1_target_encoder_snapshot_path.is_file():
            errmsg = f"{q1_target_encoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q1_target_decoder_snapshot_path = Path(
            config.initial_model_prefix / f"q1-target-decoder{infix}.pth"
        )
        if not q1_target_decoder_snapshot_path.exists():
            errmsg = f"{q1_target_decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q1_target_decoder_snapshot_path.is_file():
            errmsg = f"{q1_target_decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q2_target_encoder_snapshot_path = Path(
            config.initial_model_prefix / f"q2-target-encoder{infix}.pth"
        )
        if not q2_target_encoder_snapshot_path.exists():
            errmsg = f"{q2_target_encoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q2_target_encoder_snapshot_path.is_file():
            errmsg = f"{q2_target_encoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q2_target_decoder_snapshot_path = Path(
            config.initial_model_prefix / f"q2-target-decoder{infix}.pth"
        )
        if not q2_target_decoder_snapshot_path.exists():
            errmsg = f"{q2_target_decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q2_target_decoder_snapshot_path.is_file():
            errmsg = f"{q2_target_decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        value_optimizer_snapshot_path = Path(
            config.initial_model_prefix / f"value-optimizer{infix}.pth"
        )
        if (
            not value_optimizer_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            value_optimizer_snapshot_path = None

        q1_optimizer_snapshot_path = Path(
            config.initial_model_prefix / f"q1-optimizer{infix}.pth"
        )
        if (
            not q1_optimizer_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            q1_optimizer_snapshot_path = None

        q2_optimizer_snapshot_path = Path(
            config.initial_model_prefix / f"q2-optimizer{infix}.pth"
        )
        if (
            not q2_optimizer_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            q2_optimizer_snapshot_path = None

        value_lr_scheduler_snapshot_path = Path(
            config.initial_model_prefix / f"value-lr-scheduler{infix}.pth"
        )
        if value_optimizer_snapshot_path is None or (
            config.optimizer.warmup_steps == 0
            and config.optimizer.annealing_steps == 0
        ):
            value_lr_scheduler_snapshot_path = None
        else:
            if not value_lr_scheduler_snapshot_path.exists():
                errmsg = f"{value_lr_scheduler_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not value_lr_scheduler_snapshot_path.is_file():
                errmsg = f"{value_lr_scheduler_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

        q1_lr_scheduler_snapshot_path = Path(
            config.initial_model_prefix / f"q1-lr-scheduler{infix}.pth"
        )
        if q1_optimizer_snapshot_path is None or (
            config.optimizer.warmup_steps == 0
            and config.optimizer.annealing_steps == 0
        ):
            q1_lr_scheduler_snapshot_path = None
        else:
            # Older versions wrote these two files with underscores.
            legacy_scheduler_path = (
                config.initial_model_prefix / f"q1_lr_scheduler{infix}.pth"
            )
            if (
                not q1_lr_scheduler_snapshot_path.exists()
                and legacy_scheduler_path.is_file()
            ):
                q1_lr_scheduler_snapshot_path = legacy_scheduler_path
            if not q1_lr_scheduler_snapshot_path.exists():
                errmsg = f"{q1_lr_scheduler_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not q1_lr_scheduler_snapshot_path.is_file():
                errmsg = f"{q1_lr_scheduler_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

        q2_lr_scheduler_snapshot_path = Path(
            config.initial_model_prefix / f"q2-lr-scheduler{infix}.pth"
        )
        if q2_optimizer_snapshot_path is None or (
            config.optimizer.warmup_steps == 0
            and config.optimizer.annealing_steps == 0
        ):
            q2_lr_scheduler_snapshot_path = None
        else:
            # Older versions wrote these two files with underscores.
            legacy_scheduler_path = (
                config.initial_model_prefix / f"q2_lr_scheduler{infix}.pth"
            )
            if (
                not q2_lr_scheduler_snapshot_path.exists()
                and legacy_scheduler_path.is_file()
            ):
                q2_lr_scheduler_snapshot_path = legacy_scheduler_path
            if not q2_lr_scheduler_snapshot_path.exists():
                errmsg = f"{q2_lr_scheduler_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not q2_lr_scheduler_snapshot_path.is_file():
                errmsg = f"{q2_lr_scheduler_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

    if not config.reward_plugin.exists():
        errmsg = f"{config.reward_plugin}: Does not exist."
        raise RuntimeError(errmsg)
    if not config.reward_plugin.is_file():
        errmsg = f"{config.reward_plugin}: Not a file."
        raise RuntimeError(errmsg)

    if not math.isfinite(config.discount_factor) or not 0.0 <= config.discount_factor <= 1.0:
        errmsg = (
            f"{config.discount_factor}: An invalid value for"
            " `discount_factor`."
        )
        raise RuntimeError(errmsg)

    if not math.isfinite(config.expectile) or config.expectile <= 0.0 or 1.0 <= config.expectile:
        errmsg = f"{config.expectile}: An invalid value for `expectile`."
        raise RuntimeError(errmsg)

    if config.batch_size <= 0:
        errmsg = (
            f"{config.batch_size}: `batch_size` must be a positive integer."
        )
        raise RuntimeError(errmsg)
    if config.batch_size % world_size != 0:
        errmsg = (
            f"`batch_size` must be divisible by the world size ({world_size})."
        )
        raise RuntimeError(errmsg)

    if config.gradient_accumulation_steps <= 0:
        errmsg = (
            f"{config.gradient_accumulation_steps}: "
            "`gradient_accumulation_steps` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if not math.isfinite(config.q_max_gradient_norm) or config.q_max_gradient_norm <= 0.0:
        errmsg = (
            f"{config.q_max_gradient_norm}: `q_max_gradient_norm` must be a "
            "positive real number."
        )
        raise RuntimeError(errmsg)

    if not math.isfinite(config.v_max_gradient_norm) or config.v_max_gradient_norm <= 0.0:
        errmsg = (
            f"{config.v_max_gradient_norm}: `v_max_gradient_norm` must be a "
            "positive real number."
        )
        raise RuntimeError(errmsg)

    _config.optimizer.validate(config.optimizer)

    if config.target_update_interval <= 0:
        errmsg = (
            f"{config.target_update_interval}: "
            "`target_update_interval` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if not math.isfinite(config.target_update_rate) or config.target_update_rate <= 0.0 or config.target_update_rate > 1.0:
        errmsg = (
            f"{config.target_update_rate}: `target_update_rate` must be a "
            "real value within the range (0.0, 1.0]."
        )
        raise RuntimeError(errmsg)

    if config.snapshot_interval < 0:
        errmsg = (
            f"{config.snapshot_interval}: `snapshot_interval` must be a"
            " non-negative integer."
        )
        raise RuntimeError(errmsg)

    decision_recipe = decision_checkpoint.recipe(config, "iql", dtype=dtype, amp_dtype=amp_dtype)
    initial_rng_state = decision_checkpoint.validate_resume(config, decision_recipe)

    output_prefix = Path(HydraConfig.get().runtime.output_dir)

    if rank == 0:
        logging.info("Model type: implicit Q-learning (IQL)")

        _config.device.dump(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            device=device,
            dtype=dtype,
            amp_dtype=amp_dtype,
        )

        logging.info("Training data: %s", config.training_data)
        if num_samples > 0:
            logging.info(
                "# of training samples consumed so far: %d", num_samples
            )
        logging.info(
            "Training data is contiguous: %s", config.contiguous_training_data
        )
        if config.rewrite_rooms is not None:
            logging.info(
                "Rewrite the rooms in the training data to: %d",
                config.rewrite_rooms,
            )
        if config.rewrite_grades is not None:
            logging.info(
                "Rewrite the grades in the training data to: %d",
                config.rewrite_grades,
            )
        logging.info("# of workers: %d", config.num_workers)
        if config.replay_buffer_size > 0:
            logging.info("Replay buffer size: %d", config.replay_buffer_size)

        _config.encoder.dump(config.encoder)

        _config.decoder.dump(config.decoder)

        if config.initial_model_prefix is not None:
            logging.info(
                "Initial model prefix: %s", config.initial_model_prefix
            )
            logging.info("Initlal model index: %d", config.initial_model_index)
            if config.optimizer.initialize:
                logging.info("(Will not load optimizer)")

        logging.info("Reward plugin: %s", config.reward_plugin)
        logging.info("Double Q-learning: %s", config.double_q_learning)
        logging.info("Discount factor: %f", config.discount_factor)
        logging.info("Expectile: %f", config.expectile)
        logging.info("Checkpointing: %s", config.checkpointing)
        logging.info("Batch size: %d", config.batch_size)
        logging.info(
            "# of steps for gradient accumulation: %d",
            config.gradient_accumulation_steps,
        )
        logging.info(
            "Virtual batch size: %d",
            config.batch_size * config.gradient_accumulation_steps,
        )
        logging.info(
            "Norm threshold for gradient clipping on Q: %E",
            config.q_max_gradient_norm,
        )
        logging.info(
            "Norm threshold for gradient clipping on V: %E",
            config.v_max_gradient_norm,
        )

        _config.optimizer.dump(config.optimizer)

        logging.info(
            "Target update interval: %d", config.target_update_interval
        )
        logging.info("Target update rate: %f", config.target_update_rate)

        if config.initial_model_prefix is not None:
            logging.info(
                "Initial value encoder snapshot: %s",
                value_encoder_snapshot_path,
            )
            logging.info(
                "Initial Q1 source encoder network snapshot: %s",
                q1_source_encoder_snapshot_path,
            )
            logging.info(
                "Initial Q1 source decoder network snapshot: %s",
                q1_source_decoder_snapshot_path,
            )
            logging.info(
                "Initial Q2 source encoder network snapshot: %s",
                q2_source_encoder_snapshot_path,
            )
            logging.info(
                "Initial Q2 source decoder network snapshot: %s",
                q2_source_decoder_snapshot_path,
            )
            logging.info(
                "Initial Q1 target encoder network snapshot: %s",
                q1_target_encoder_snapshot_path,
            )
            logging.info(
                "Initial Q1 target decoder network snapshot: %s",
                q1_target_decoder_snapshot_path,
            )
            logging.info(
                "Initial Q2 target encoder network snapshot: %s",
                q2_target_encoder_snapshot_path,
            )
            logging.info(
                "Initial Q2 target decoder network snapshot: %s",
                q2_target_decoder_snapshot_path,
            )
            if value_optimizer_snapshot_path is not None:
                logging.info(
                    "Initial value optimizer snapshot: %s",
                    value_optimizer_snapshot_path,
                )
                logging.info(
                    "Initial value LR scheduler snapshot: %s",
                    value_lr_scheduler_snapshot_path,
                )
            if q1_optimizer_snapshot_path is not None:
                logging.info(
                    "Initial Q1 optimizer snapshot: %s",
                    q1_optimizer_snapshot_path,
                )
                logging.info(
                    "Initial Q1 LR scheduler snapshot: %s",
                    q1_lr_scheduler_snapshot_path,
                )
            if q2_optimizer_snapshot_path is not None:
                logging.info(
                    "Initial Q2 optimizer snapshot: %s",
                    q2_optimizer_snapshot_path,
                )
                logging.info(
                    "Initial Q2 LR scheduler snapshot: %s",
                    q2_lr_scheduler_snapshot_path,
                )

        logging.info("Output prefix: %s", output_prefix)
        if config.snapshot_interval == 0:
            logging.info("Snapshot interval: N/A")
        else:
            logging.info("Snapshot interval: %d", config.snapshot_interval)

    if world_size >= 2:
        init_process_group(backend="nccl" if device.type == "cuda" else "gloo")

    value_encoder = make_backbone(
        config.encoder, checkpointing=config.checkpointing, dtype=dtype
    )
    value_encoder_tdm = TensorDictModule(
        value_encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["encode"],  # type: ignore
    )
    value_decoder = make_state_head(
        config.decoder, input_dimension=config.encoder.dimension,
        architecture=config.encoder.architecture, dtype=dtype,
        final_rank=config.final_rank_weight > 0,
    )
    value_decoder.zero_output()
    value_output_keys = ["state_value"]
    if config.final_rank_weight > 0:
        value_output_keys.append("final_rank_logits")
    value_decoder_tdm = TensorDictModule(
        value_decoder,
        in_keys=["encode"],  # type: ignore
        out_keys=value_output_keys,  # type: ignore
    )
    value_network = TensorDictSequential(value_encoder_tdm, value_decoder_tdm)
    if world_size >= 2:
        value_network.to(device=device)
        for _param in value_network.parameters():
            broadcast(_param.data, src=0)
        value_network.to(device="cpu")

    q1_source_encoder = make_backbone(
        config.encoder, checkpointing=config.checkpointing, dtype=dtype
    )
    q1_source_encoder_tdm = TensorDictModule(
        q1_source_encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["encode"],  # type: ignore
    )
    q1_source_decoder = make_candidate_head(
        config.decoder, input_dimension=config.encoder.dimension,
        architecture=config.encoder.architecture, dtype=dtype,
    )
    q1_source_decoder.zero_output()
    q1_source_decoder_tdm = TensorDictModule(
        q1_source_decoder,
        in_keys=candidate_head_keys(config.encoder.architecture, config.decoder),  # type: ignore
        out_keys=["action_value"],  # type: ignore
    )
    q1_source_network = TensorDictSequential(
        q1_source_encoder_tdm, q1_source_decoder_tdm
    )
    if world_size >= 2:
        q1_source_network.to(device=device)
        for _param in q1_source_network.parameters():
            broadcast(_param.data, src=0)
        q1_source_network.to(device="cpu")

    q2_source_encoder = make_backbone(
        config.encoder, checkpointing=config.checkpointing, dtype=dtype
    )
    q2_source_encoder_tdm = TensorDictModule(
        q2_source_encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["encode"],  # type: ignore
    )
    q2_source_decoder = make_candidate_head(
        config.decoder, input_dimension=config.encoder.dimension,
        architecture=config.encoder.architecture, dtype=dtype,
    )
    q2_source_decoder.zero_output()
    q2_source_decoder_tdm = TensorDictModule(
        q2_source_decoder,
        in_keys=candidate_head_keys(config.encoder.architecture, config.decoder),  # type: ignore
        out_keys=["action_value"],  # type: ignore
    )
    q2_source_network = TensorDictSequential(
        q2_source_encoder_tdm, q2_source_decoder_tdm
    )
    if world_size >= 2:
        q2_source_network.to(device=device)
        for _param in q2_source_network.parameters():
            broadcast(_param.data, src=0)
        q2_source_network.to(device="cpu")

    q1_target_encoder = make_backbone(
        config.encoder, checkpointing=config.checkpointing, dtype=dtype
    )
    q1_target_encoder_tdm = TensorDictModule(
        q1_target_encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["encode"],  # type: ignore
    )
    q1_target_decoder = make_candidate_head(
        config.decoder, input_dimension=config.encoder.dimension,
        architecture=config.encoder.architecture, dtype=dtype,
    )
    q1_target_decoder_tdm = TensorDictModule(
        q1_target_decoder,
        in_keys=candidate_head_keys(config.encoder.architecture, config.decoder),  # type: ignore
        out_keys=["action_value"],  # type: ignore
    )
    q1_target_network = TensorDictSequential(
        q1_target_encoder_tdm, q1_target_decoder_tdm
    )
    with torch.no_grad():
        for _param, _param_target in zip(
            q1_source_network.parameters(), q1_target_network.parameters()
        ):
            _param_target.data = _param.data.detach().clone()

    q2_target_encoder = make_backbone(
        config.encoder, checkpointing=config.checkpointing, dtype=dtype
    )
    q2_target_encoder_tdm = TensorDictModule(
        q2_target_encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["encode"],  # type: ignore
    )
    q2_target_decoder = make_candidate_head(
        config.decoder, input_dimension=config.encoder.dimension,
        architecture=config.encoder.architecture, dtype=dtype,
    )
    q2_target_decoder_tdm = TensorDictModule(
        q2_target_decoder,
        in_keys=candidate_head_keys(config.encoder.architecture, config.decoder),  # type: ignore
        out_keys=["action_value"],  # type: ignore
    )
    q2_target_network = TensorDictSequential(
        q2_target_encoder_tdm, q2_target_decoder_tdm
    )
    with torch.no_grad():
        for _param, _param_target in zip(
            q2_source_network.parameters(), q2_target_network.parameters()
        ):
            _param_target.data = _param.data.detach().clone()

    model_to_save = TwinQActor(
        q1_target_network, q2_target_network, return_values=True,
        tie_break_by_action_id=config.encoder.architecture == "dense_decoder",
    )
    model_to_save_tdm = TensorDictModule(
        model_to_save,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["action", "action_value"],  # type: ignore
    )

    if config.bc_initial_model is not None:
        teacher = load_model(config.bc_initial_model, map_location=torch.device("cpu"))
        for body, head in (
            (value_encoder, value_decoder),
            (q1_source_encoder, q1_source_decoder), (q2_source_encoder, q2_source_decoder),
            (q1_target_encoder, q1_target_decoder), (q2_target_encoder, q2_target_decoder),
        ):
            transfer_bc_features(body, head, teacher)
        # Identical transferred features plus two zero Q outputs would keep
        # both deterministic critics identical forever. Give each its own
        # fresh readout; never copy the BC action logits into a value function.
        for head in (q1_source_decoder, q2_source_decoder):
            nn.init.normal_(head.layers[-1].weight, std=0.02)
            nn.init.zeros_(head.layers[-1].bias)
        q1_target_network.load_state_dict(q1_source_network.state_dict())
        q2_target_network.load_state_dict(q2_source_network.state_dict())
        del teacher

    value_optimizer, value_lr_scheduler = _config.optimizer.create(
        device.type, config.optimizer, value_network
    )
    q1_optimizer, q1_lr_scheduler = _config.optimizer.create(
        device.type, config.optimizer, q1_source_network
    )
    q2_optimizer, q2_lr_scheduler = _config.optimizer.create(
        device.type, config.optimizer, q2_source_network
    )

    if config.encoder.load_from is not None:
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        encoder_state_dict = torch.load(
            config.encoder.load_from, map_location="cpu", weights_only=True
        )
        value_encoder.load_state_dict(encoder_state_dict)
        q1_source_encoder.load_state_dict(encoder_state_dict)
        q2_source_encoder.load_state_dict(encoder_state_dict)
        q1_target_encoder.load_state_dict(encoder_state_dict)
        q2_target_encoder.load_state_dict(encoder_state_dict)

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        assert value_encoder_snapshot_path is not None
        assert value_decoder_snapshot_path is not None
        assert q1_source_encoder_snapshot_path is not None
        assert q1_source_decoder_snapshot_path is not None
        assert q2_source_encoder_snapshot_path is not None
        assert q2_source_decoder_snapshot_path is not None
        assert q1_target_encoder_snapshot_path is not None
        assert q1_target_decoder_snapshot_path is not None
        assert q2_target_encoder_snapshot_path is not None
        assert q2_target_decoder_snapshot_path is not None

        value_encoder_state_dict = torch.load(
            value_encoder_snapshot_path, map_location="cpu", weights_only=True
        )
        value_encoder.load_state_dict(value_encoder_state_dict)
        value_decoder_state_dict = torch.load(
            value_decoder_snapshot_path, map_location="cpu", weights_only=True
        )
        value_decoder.load_state_dict(value_decoder_state_dict)

        q1_source_encoder_state_dict = torch.load(
            q1_source_encoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q1_source_encoder.load_state_dict(q1_source_encoder_state_dict)
        q1_source_decoder_state_dict = torch.load(
            q1_source_decoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q1_source_decoder.load_state_dict(q1_source_decoder_state_dict)

        q2_source_encoder_state_dict = torch.load(
            q2_source_encoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q2_source_encoder.load_state_dict(q2_source_encoder_state_dict)
        q2_source_decoder_state_dict = torch.load(
            q2_source_decoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q2_source_decoder.load_state_dict(q2_source_decoder_state_dict)

        q1_target_encoder_state_dict = torch.load(
            q1_target_encoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q1_target_encoder.load_state_dict(q1_target_encoder_state_dict)
        q1_target_decoder_state_dict = torch.load(
            q1_target_decoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q1_target_decoder.load_state_dict(q1_target_decoder_state_dict)

        q2_target_encoder_state_dict = torch.load(
            q2_target_encoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q2_target_encoder.load_state_dict(q2_target_encoder_state_dict)
        q2_target_decoder_state_dict = torch.load(
            q2_target_decoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q2_target_decoder.load_state_dict(q2_target_decoder_state_dict)

        if value_optimizer_snapshot_path is not None:
            value_optimizer_state_dict = torch.load(
                value_optimizer_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            value_optimizer.load_state_dict(value_optimizer_state_dict)
        if value_lr_scheduler_snapshot_path is not None:
            assert value_lr_scheduler is not None
            value_lr_scheduler_state_dict = torch.load(
                value_lr_scheduler_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            value_lr_scheduler.load_state_dict(value_lr_scheduler_state_dict)

        if q1_optimizer_snapshot_path is not None:
            q1_optimizer_state_dict = torch.load(
                q1_optimizer_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            q1_optimizer.load_state_dict(q1_optimizer_state_dict)
        if q1_lr_scheduler_snapshot_path is not None:
            assert q1_lr_scheduler is not None
            q1_lr_scheduler_state_dict = torch.load(
                q1_lr_scheduler_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            q1_lr_scheduler.load_state_dict(q1_lr_scheduler_state_dict)

        if q2_optimizer_snapshot_path is not None:
            q2_optimizer_state_dict = torch.load(
                q2_optimizer_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            q2_optimizer.load_state_dict(q2_optimizer_state_dict)
        if q2_lr_scheduler_snapshot_path is not None:
            assert q2_lr_scheduler is not None
            q2_lr_scheduler_state_dict = torch.load(
                q2_lr_scheduler_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            q2_lr_scheduler.load_state_dict(q2_lr_scheduler_state_dict)

    value_network.requires_grad_(True)
    value_network.train()
    value_network.to(device=device, dtype=dtype)
    if world_size >= 2:
        value_network = DistributedDataParallel(value_network)
        value_network = nn.SyncBatchNorm.convert_sync_batchnorm(value_network)

    q1_source_network.requires_grad_(True)
    q1_source_network.train()
    q1_source_network.to(device=device, dtype=dtype)
    if world_size >= 2:
        q1_source_network = DistributedDataParallel(q1_source_network)
        q1_source_network = nn.SyncBatchNorm.convert_sync_batchnorm(
            q1_source_network
        )

    q2_source_network.requires_grad_(True)
    q2_source_network.train()
    q2_source_network.to(device=device, dtype=dtype)
    if world_size >= 2:
        q2_source_network = DistributedDataParallel(q2_source_network)
        q2_source_network = nn.SyncBatchNorm.convert_sync_batchnorm(
            q2_source_network
        )

    q1_target_network.requires_grad_(False)
    q1_target_network.eval()
    q1_target_network.to(device=device, dtype=dtype)

    q2_target_network.requires_grad_(False)
    q2_target_network.eval()
    q2_target_network.to(device=device, dtype=dtype)

    if world_size >= 2:
        # BC transfer can reinitialize Q after the earlier broadcasts. DDP
        # has now synchronized the sources; synchronize the independent
        # targets too, before they supervise V. Broadcast loaded targets on
        # resume instead of copying sources over their saved Polyak lag.
        with torch.no_grad():
            for target_network in (q1_target_network, q2_target_network):
                for tensor in (
                    *target_network.parameters(), *target_network.buffers()
                ):
                    broadcast(tensor, src=0)

    snapshots_path = output_prefix / "snapshots"

    def snapshot_writer(
        num_samples: int | None = None,
        *,
        resume_num_samples: int | None = None,
        optimizer_steps: int = 0,
        cursor_is_exact: bool = True,
        grad_scaler_state: dict | None = None,
    ) -> None:
        snapshots_path.mkdir(parents=True, exist_ok=True)

        infix = "" if num_samples is None else f".{num_samples}"

        torch.save(
            value_encoder.state_dict(),
            snapshots_path / f"value-encoder{infix}.pth",
        )
        torch.save(
            value_decoder.state_dict(),
            snapshots_path / f"value-decoder{infix}.pth",
        )
        torch.save(
            q1_source_encoder.state_dict(),
            snapshots_path / f"q1-source-encoder{infix}.pth",
        )
        torch.save(
            q1_source_decoder.state_dict(),
            snapshots_path / f"q1-source-decoder{infix}.pth",
        )
        torch.save(
            q2_source_encoder.state_dict(),
            snapshots_path / f"q2-source-encoder{infix}.pth",
        )
        torch.save(
            q2_source_decoder.state_dict(),
            snapshots_path / f"q2-source-decoder{infix}.pth",
        )
        torch.save(
            q1_target_encoder.state_dict(),
            snapshots_path / f"q1-target-encoder{infix}.pth",
        )
        torch.save(
            q1_target_decoder.state_dict(),
            snapshots_path / f"q1-target-decoder{infix}.pth",
        )
        torch.save(
            q2_target_encoder.state_dict(),
            snapshots_path / f"q2-target-encoder{infix}.pth",
        )
        torch.save(
            q2_target_decoder.state_dict(),
            snapshots_path / f"q2-target-decoder{infix}.pth",
        )
        torch.save(
            value_optimizer.state_dict(),
            snapshots_path / f"value-optimizer{infix}.pth",
        )
        if value_lr_scheduler is not None:
            torch.save(
                value_lr_scheduler.state_dict(),
                snapshots_path / f"value-lr-scheduler{infix}.pth",
            )
        torch.save(
            q1_optimizer.state_dict(),
            snapshots_path / f"q1-optimizer{infix}.pth",
        )
        if q1_lr_scheduler is not None:
            torch.save(
                q1_lr_scheduler.state_dict(),
                snapshots_path / f"q1-lr-scheduler{infix}.pth",
            )
        torch.save(
            q2_optimizer.state_dict(),
            snapshots_path / f"q2-optimizer{infix}.pth",
        )
        if q2_lr_scheduler is not None:
            torch.save(
                q2_lr_scheduler.state_dict(),
                snapshots_path / f"q2-lr-scheduler{infix}.pth",
            )

        state = dump_object(
            model_to_save_tdm,
            [dump_object(model_to_save, [
                network_description(q1_target_network, config, dtype=dtype),
                network_description(q2_target_network, config, dtype=dtype),
            ], {"return_values": True,
                "tie_break_by_action_id": config.encoder.architecture == "dense_decoder"})],
            {"in_keys": ["sparse", "numeric", "progression", "candidates"],
             "out_keys": ["action", "action_value"]},
        )
        torch.save(state, snapshots_path / f"model{infix}.kanachan")
        value_to_save = value_network.module if isinstance(value_network, DistributedDataParallel) else value_network
        torch.save(network_description(
            value_to_save, config, dtype=dtype, state=True,
            final_rank=config.final_rank_weight > 0,
        ), snapshots_path / f"value-model{infix}.kanachan")

        cursor = (
            resume_num_samples
            if resume_num_samples is not None
            else num_samples
        )
        assert cursor is not None
        save_training_state(
            snapshots_path,
            infix,
            TrainingState(
                cursor,
                optimizer_steps,
                cursor_is_exact and config.get("replay_buffer_size", 0) == 0,
                grad_scaler_state,
            ),
        )
        decision_checkpoint.save(snapshots_path, infix, decision_recipe, algorithm="iql")

    tensorboard_path = output_prefix / "tensorboard"
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    with SummaryWriter(log_dir=tensorboard_path) as summary_writer:
        _training(
            training_data=config.training_data,
            contiguous_training_data=config.contiguous_training_data,
            rewrite_rooms=config.rewrite_rooms,
            rewrite_grades=config.rewrite_grades,
            replay_buffer_size=config.replay_buffer_size,
            num_workers=config.num_workers,
            device=device,
            dtype=dtype,
            amp_dtype=amp_dtype,
            value_network=value_network,
            q1_source_network=q1_source_network,
            q2_source_network=q2_source_network,
            q1_target_network=q1_target_network,
            q2_target_network=q2_target_network,
            reward_plugin=config.reward_plugin,
            discount_factor=config.discount_factor,
            expectile=config.expectile,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            v_max_gradient_norm=config.v_max_gradient_norm,
            q_max_gradient_norm=config.q_max_gradient_norm,
            value_optimizer=value_optimizer,
            q1_optimizer=q1_optimizer,
            q2_optimizer=q2_optimizer,
            value_lr_scheduler=value_lr_scheduler,
            q1_lr_scheduler=q1_lr_scheduler,
            q2_lr_scheduler=q2_lr_scheduler,
            target_update_interval=config.target_update_interval,
            target_update_rate=config.target_update_rate,
            snapshot_interval=config.snapshot_interval,
            num_samples=num_samples,
            initial_optimizer_steps=initial_optimizer_steps,
            initial_grad_scaler_state=initial_grad_scaler_state,
            initial_rng_state=initial_rng_state,
            final_rank_weight=config.final_rank_weight,
            summary_writer=summary_writer,
            snapshot_writer=snapshot_writer,
        )


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
    sys.exit(0)
