#!/usr/bin/env python3

import re
import datetime
from contextlib import ExitStack
import math
from pathlib import Path
import os
import logging
import sys
from typing import Any, Callable
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import Optimizer
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp.grad_scaler import GradScaler
from torch.distributed import (
    init_process_group,
    broadcast,
    ReduceOp,
    all_reduce,
)
from torch.utils.tensorboard.writer import SummaryWriter
from tensordict import TensorDict  # type: ignore
from tensordict.nn import TensorDictModule, TensorDictSequential  # type: ignore
from kanachan.constants import MAX_NUM_ACTION_CANDIDATES, NUM_TYPES_OF_ACTIONS
from kanachan.training.common import (
    get_distributed_environment,
    average_tensor,
    with_last,
    rescale_gradients,
    checked_clip_gradients,
    backoff_gradient_scale,
)
from kanachan.training.core.checkpoint import (
    SnapshotWriter,
    TrainingState,
    load_training_state,
    save_training_state,
)
from kanachan.training.core.progress import TrainingProgress
from kanachan.training.core.optimizers import move_optimizer_state, consolidate_optimizer_state
import kanachan.training.core.config as _config

# pylint: disable=unused-import
import kanachan.training.cql.config  # noqa: F401
from kanachan.training.core.offline_rl import DataLoader, EpisodeReplayBuffer
from kanachan.nn import Encoder, Decoder, QRDecoder, QDecoder, DecodeConverter
from kanachan.nn.qr_decoder import compute_td_error
from kanachan.nn.functional import compute_td_target
from kanachan.training.core.rl.rewards import PlacementReward
from kanachan.model_loader import dump_object, dump_model, load_model
from kanachan.nn.decision_factory import (
    make_backbone, backbone_kwargs, make_candidate_head,
    candidate_head_kwargs, candidate_head_keys,
)
from kanachan.training.core.rl.discrete_cql import (
    td_components, mc_components, td_penalty, conservative_penalty, q_statistics,
)
from kanachan.training.core import decision_checkpoint


def _training(
    *,
    training_data: Path,
    contiguous_training_data: bool,
    rewrite_rooms: int | None,
    rewrite_grades: int | None,
    num_workers: int,
    replay_buffer_size: int,
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype,
    source_network: nn.Module,
    target_network: nn.Module | None,
    reward_plugin: Path | None,
    discount_factor: float,
    td_error_computation: Callable[[TensorDict], None] | None,
    alpha: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_gradient_norm: float,
    optimizer: Optimizer,
    scheduler: lr_scheduler.LRScheduler | None,
    target_update_interval: int,
    target_update_rate: float,
    snapshot_interval: int,
    num_samples: int,
    summary_writer: SummaryWriter,
    snapshot_writer: SnapshotWriter,
    initial_optimizer_steps: int = 0,
    initial_grad_scaler_state: dict | None = None,
    reward_model: Path | None = None,
    replay_state: dict | None = None,
    cql_temperature: float = 1.0,
    initial_rng_state: dict | None = None,
    q_monitor_interval: int = 100,
    data_cache: Path | None = None,
    data_cache_bytes: int = 256 * 1024 * 1024,
    reward_scheme: str = "placement",
    round_score_unit: float = 10000.0,
    round_reward_weight: float = 0.2,
    target_kind: str = "td",
    regression_loss: str = "mse",
    replay_buffer_factory=None,
) -> None:
    if target_kind not in ("td", "mc"):
        raise ValueError("CQL target_kind must be mc or td.")
    if target_kind == "mc" and (
        target_network is not None or not contiguous_training_data or replay_buffer_size < 1
        or reward_plugin is not None or reward_model is not None
    ):
        raise ValueError("MC CQL requires complete episode replay, observed rewards and no target network.")
    if target_kind == "td" and td_error_computation is None:
        raise ValueError("TD CQL requires its TD target computation.")
    if target_kind == "mc" and not all(p.requires_grad for p in source_network.parameters()):
        raise ValueError("MC CQL trains every online-network parameter.")
    if regression_loss not in ("mse", "huber"):
        raise ValueError("Unknown CQL regression loss.")
    from kanachan.nn.decision_value import q_only

    mc_source = q_only(source_network) if target_kind == "mc" else None
    start_time = datetime.datetime.now()

    move_optimizer_state(optimizer)

    world_size, rank, local_rank = get_distributed_environment()

    # Load the reward plugin.
    from kanachan.training.online.state import file_digest

    reward_identity = (
        "plugin:" + file_digest(reward_plugin) if reward_plugin is not None else
        "placement-v1:" + (file_digest(reward_model) if reward_model is not None else "terminal")
    )
    if reward_plugin is None:
        from kanachan.training.core.rl.rewards import configured_reward

        get_reward = configured_reward(
            reward_scheme, shaping_model=reward_model,
            score_unit=round_score_unit, round_weight=round_reward_weight,
        )
        if reward_scheme != "placement":
            reward_identity = f"round-and-placement-v1:unit={round_score_unit:.17g}:weight={round_reward_weight:.17g}"
    else:
        namespace: dict[str, Any] = globals().copy()
        with open(reward_plugin, encoding="UTF-8") as file_pointer:
            exec(file_pointer.read(), namespace)  # pylint: disable=exec-used
        get_reward = namespace["get_reward"]

    if replay_buffer_size >= 1:
        data_loader = (replay_buffer_factory or EpisodeReplayBuffer)(
            training_data=training_data,
            contiguous_training_data=contiguous_training_data,
            num_skip_samples=num_samples if replay_state is None else replay_state["input_samples"],
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
            ranks=[r for r in range(world_size)],
            main_rank=0,
            reward_identity=reward_identity,
            data_cache=data_cache, data_cache_bytes=data_cache_bytes,
        )
        if replay_state is not None:
            data_loader.load_state_dict(replay_state)
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

    last_snapshot: int | None = None
    if snapshot_interval > 0:
        last_snapshot = num_samples

    batch_count = 0
    pending_samples = 0
    epoch_start_samples = num_samples
    configured_batch_size = batch_size
    progress = TrainingProgress(
        summary_writer,
        batch_size,
        gradient_accumulation_steps,
        target_update_interval if target_network is not None else None,
        target_update_rate,
        initial_optimizer_steps=initial_optimizer_steps,
    )

    decision_checkpoint.restore(initial_rng_state)
    checkpointable_replay = bool(getattr(data_loader, "can_checkpoint", False))
    batches = data_loader.iter_with_last() if checkpointable_replay else with_last(data_loader)
    for data, is_last in batches:
        # The reader supplies CPU tensors. Check ranges before device transfer
        # rather than synchronizing CUDA twice for these existing validations.
        assert ((data["action"] >= 0) & (data["action"] < MAX_NUM_ACTION_CANDIDATES)).all().item()
        if target_kind == "mc" and not torch.equal(data["next", "done"], data["next", "end_of_game"]):
            raise ValueError("MC observed returns must terminate only at the end of the half-game.")
        data = data.to(device=device, non_blocking=True)
        local_batch_size = int(data.size(0))
        assert isinstance(data, TensorDict)

        action: Tensor = data["action"]
        assert isinstance(action, Tensor)
        assert action.device == device
        assert action.dtype == torch.int32
        assert action.dim() == 1
        assert action.size(0) == local_batch_size
        next_samples = num_samples + local_batch_size * world_size
        batch_index = (next_samples + configured_batch_size * world_size - 1) // (
            configured_batch_size * world_size
        )
        monitor = q_monitor_interval > 0 and (is_last or batch_index % q_monitor_interval == 0)
        logged_return = (
            data.get("action_value", None) if monitor and contiguous_training_data else None
        )
        if logged_return is not None:
            logged_return = logged_return.detach().clone()

        with torch.autocast(**autocast_kwargs):
            if target_kind == "mc":
                values, chosen, expected = mc_components(mc_source, data)
                data["action_value"] = values
                data["mc_target"] = expected
                data["td_error"] = 2 * td_penalty(chosen, expected, regression_loss)
            else:
                td_error_computation(data)

        td_error: Tensor = data["td_error"]
        assert isinstance(td_error, Tensor)
        assert td_error.device == device
        assert td_error.dtype == dtype
        assert td_error.dim() == 1
        assert td_error.size(0) == local_batch_size
        td_error = td_error.mean()

        q: Tensor = data["action_value"]
        assert isinstance(q, Tensor)
        assert q.device == device
        assert q.dtype == dtype
        assert q.dim() == 2
        assert q.size(0) == local_batch_size
        assert q.size(1) == MAX_NUM_ACTION_CANDIDATES

        q_sa = q[torch.arange(local_batch_size, device=device), action]
        assert q_sa.device == device
        assert q_sa.dtype == dtype
        assert q_sa.dim() == 1
        assert q_sa.size(0) == local_batch_size

        # Compute the regularization term.
        masked_q = q.masked_fill(
            data["candidates"] >= NUM_TYPES_OF_ACTIONS, -torch.inf
        )
        per_sample_regularizer = conservative_penalty(
            masked_q, q_sa, cql_temperature
        )
        loss_regularizer = alpha * per_sample_regularizer.mean()
        loss_td_error = 0.5 * td_error
        loss = loss_regularizer + loss_td_error
        regression_tag = "Loss (MC Return)" if target_kind == "mc" else "Loss (TD Error)"
        metrics = {
            "Q": q_sa.detach().mean(), "Loss (Regularizer)": loss_regularizer.detach(),
            regression_tag: loss_td_error.detach(), "Loss": loss.detach(),
        }
        if monitor:
            diagnostics = q_statistics(
                masked_q, q_sa, data["candidates"], temperature=cql_temperature,
                gap=per_sample_regularizer, logged_return=logged_return,
                target=data.get("mc_target" if target_kind == "mc" else "td_target", None),
                done=data["next", "done"], target_kind=target_kind,
            )
            if world_size > 1:
                diagnostics = {f"{key} (rank average)": value for key, value in diagnostics.items()}
            metrics.update(diagnostics)
        packed_metrics = torch.stack(list(metrics.values()))
        if world_size >= 2:
            average_tensor(packed_metrics)
        displayed = dict(zip(metrics, packed_metrics.cpu().tolist(), strict=True))
        q_to_display = displayed["Q"]
        loss_regularizer_to_display = displayed["Loss (Regularizer)"]
        loss_td_error_to_display = displayed[regression_tag]
        loss_to_display = displayed["Loss"]

        if not math.isfinite(loss_to_display):
            errmsg = "Loss becomes non-finite."
            raise RuntimeError(errmsg)

        loss *= local_batch_size / (configured_batch_size * gradient_accumulation_steps)
        grad_scaler.scale(loss).backward()

        num_samples += local_batch_size * world_size
        pending_samples += local_batch_size * world_size
        batch_count += 1

        if batch_count % gradient_accumulation_steps == 0 or is_last:
            normalization = (
                configured_batch_size * world_size * gradient_accumulation_steps
                / pending_samples
            )
            rescale_gradients(source_network, normalization)
            pending_samples = 0
            grad_scaler.unscale_(optimizer)
            norm = checked_clip_gradients(source_network, max_gradient_norm)
            if norm is None:
                logging.warning(
                    "Skip an optimization step because of non-finite "
                    "values in the gradient."
                )
                backoff_gradient_scale(grad_scaler)
                optimizer.zero_grad()
                continue
            gradient_norm = float(norm)

            grad_scaler.step(optimizer)
            grad_scaler.update()
            progress.optimizer_step(num_samples)
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            if (
                target_network is not None
                and progress.total_optimizer_steps % target_update_interval
                == 0
            ):
                assert target_update_interval >= 1
                assert target_update_rate > 0.0
                with torch.no_grad():
                    for _u, _v in zip(
                        source_network.parameters(),
                        target_network.parameters(),
                    ):
                        _v.data *= 1.0 - target_update_rate
                        _v.data += target_update_rate * _u.data

                progress.target_update(num_samples)

            if rank == 0:
                logging.info(
                    "sample = %s, loss = %s, Q = %E, gradient norm = %s",
                    num_samples,
                    loss_to_display,
                    q_to_display,
                    gradient_norm,
                )
                summary_writer.add_scalar("Q", q_to_display, num_samples)
                summary_writer.add_scalar(
                    "Loss (Regularizer)",
                    loss_regularizer_to_display,
                    num_samples,
                )
                summary_writer.add_scalar(
                    regression_tag, loss_td_error_to_display, num_samples
                )
                summary_writer.add_scalar("Loss", loss_to_display, num_samples)
                summary_writer.add_scalar(
                    "Gradient Norm", gradient_norm, num_samples
                )
                if scheduler is not None:
                    summary_writer.add_scalar(
                        "LR", scheduler.get_last_lr()[0], num_samples
                    )
        else:
            if rank == 0:
                logging.info(
                    "sample = %s, loss = %s, Q = %E",
                    num_samples,
                    loss_to_display,
                    q_to_display,
                )
                summary_writer.add_scalar("Q", q_to_display, num_samples)
                summary_writer.add_scalar(
                    "Loss (Regularizer)",
                    loss_regularizer_to_display,
                    num_samples,
                )
                summary_writer.add_scalar(
                    regression_tag, loss_td_error_to_display, num_samples
                )
                summary_writer.add_scalar("Loss", loss_to_display, num_samples)

        if rank == 0 and monitor:
            for key, value in displayed.items():
                if key not in ("Q", "Loss (Regularizer)", regression_tag, "Loss"):
                    summary_writer.add_scalar(key, value, num_samples)

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
            consolidate_optimizer_state(optimizer)
            if rank == 0:
                snapshot_writer(
                    num_samples, optimizer_steps=progress.total_optimizer_steps,
                    grad_scaler_state=grad_scaler.state_dict() if grad_scaler.is_enabled() else None,
                    **({"replay_state": data_loader.state_dict()} if checkpointable_replay else {}),
                )
            last_snapshot = num_samples

    progress.finish(0)
    elapsed_time = datetime.datetime.now() - start_time
    consolidate_optimizer_state(optimizer)

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
            **({"replay_state": data_loader.state_dict()} if checkpointable_replay else {}),
        )


@hydra.main(version_base=None, config_name="config")
def _main(config: DictConfig) -> None:
    from torch._dynamo import config as dynamo_config

    flex = (config.encoder.architecture == "dense_decoder"
            and getattr(config.encoder, "attention_backend", "sdpa") == "flex")
    # The dense CQL loop can stay eager while its FlexAttention kernel compiles.
    dynamo_config.disable = ((not config.enable_compile and not flex)
                             or os.environ.get("TORCHDYNAMO_DISABLE") == "1")
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
            "`rewrite_rooms` must be an integer within the range `[0, 4]`."
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
            "`rewrite_grades` must be an integer within the range `[0, 15]`."
        )
        raise RuntimeError(errmsg)

    if config.num_workers is None:
        config.num_workers = 0 if device.type == "cpu" else 2
    if config.num_workers < 0:
        raise RuntimeError("num_workers must be a non-negative integer.")

    if config.replay_buffer_size < 0:
        errmsg = (
            f"{config.replay_buffer_size}: `replay_buffer_size` must be "
            "a non-negative integer."
        )
        raise RuntimeError(errmsg)
    if config.contiguous_training_data and config.replay_buffer_size == 0:
        errmsg = "Use `replay_buffer_size` for `contiguous_training_data`."
        raise RuntimeError(errmsg)

    _config.encoder.validate(config.encoder)

    _config.decoder.validate(config.decoder, config.encoder.dimension)

    if config.initial_model_prefix is not None and config.decoder.load_from is not None:
        raise RuntimeError("initial_model_prefix conflicts with decoder.load_from.")

    if config.num_qr_intervals is not None and config.num_qr_intervals <= 0:
        errmsg = (
            f"{config.num_qr_intervals}: `num_qr_intervals` must be a "
            "positive integer."
        )
        raise RuntimeError(errmsg)

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
                "`initial_model_index` must be combined with "
                "`initial_model_prefix`."
            )
            raise RuntimeError(errmsg)
        if config.initial_model_index < 0:
            errmsg = (
                f"{config.initial_model_index}: "
                "An invalid initial model index."
            )
            raise RuntimeError(errmsg)

    num_samples = 0
    initial_optimizer_steps = 0
    initial_grad_scaler_state = None
    encoder_snapshot_path: Path | None = None
    decoder_snapshot_path: Path | None = None
    source_encoder_snapshot_path: Path | None = None
    source_decoder_snapshot_path: Path | None = None
    replay_state = None
    target_encoder_snapshot_path: Path | None = None
    target_decoder_snapshot_path: Path | None = None
    optimizer_snapshot_path: Path | None = None
    scheduler_snapshot_path: Path | None = None

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None

        if config.initial_model_index is None:
            for child in os.listdir(config.initial_model_prefix):
                if config.double_q_learning:
                    match = re.search(
                        "^(?:(?:source|target)-(?:encoder|decoder)|optimizer|lr-scheduler)(?:\\.(\\d+))?\\.pth$",
                        child,
                    )
                else:
                    match = re.search(
                        "^(?:encoder|decoder|optimizer|lr-scheduler)(?:\\.(\\d+))?\\.pth$",
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
            replay_path = config.initial_model_prefix / f"replay-state{infix}.pth"
            if not config.optimizer.initialize and replay_path.is_file():
                replay_state = torch.load(replay_path, map_location="cpu", weights_only=True)
            elif not config.optimizer.initialize:
                raise RuntimeError(
                    "Replay checkpoints do not store the shuffle buffer. "
                    "Use optimizer.initialize=true to reuse the weights and restart the data."
                )
            else:
                num_samples = 0
                initial_optimizer_steps = 0

        if config.double_q_learning:
            source_encoder_snapshot_path = Path(
                config.initial_model_prefix / f"source-encoder{infix}.pth"
            )
            if not source_encoder_snapshot_path.exists():
                errmsg = f"{source_encoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not source_encoder_snapshot_path.is_file():
                errmsg = f"{source_encoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

            source_decoder_snapshot_path = Path(
                config.initial_model_prefix / f"source-decoder{infix}.pth"
            )
            if not source_decoder_snapshot_path.exists():
                errmsg = f"{source_decoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not source_decoder_snapshot_path.is_file():
                errmsg = f"{source_decoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

            target_encoder_snapshot_path = Path(
                config.initial_model_prefix / f"target-encoder{infix}.pth"
            )
            if not target_encoder_snapshot_path.exists():
                errmsg = f"{target_encoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not target_encoder_snapshot_path.is_file():
                errmsg = f"{target_encoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

            target_decoder_snapshot_path = Path(
                config.initial_model_prefix / f"target-decoder{infix}.pth"
            )
            if not target_decoder_snapshot_path.exists():
                errmsg = f"{target_decoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not target_decoder_snapshot_path.is_file():
                errmsg = f"{target_decoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)
        else:
            encoder_snapshot_path = Path(
                config.initial_model_prefix / f"encoder{infix}.pth"
            )
            if not encoder_snapshot_path.exists():
                errmsg = f"{encoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not encoder_snapshot_path.is_file():
                errmsg = f"{encoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

            decoder_snapshot_path = Path(
                config.initial_model_prefix / f"decoder{infix}.pth"
            )
            if not decoder_snapshot_path.exists():
                errmsg = f"{decoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not decoder_snapshot_path.is_file():
                errmsg = f"{decoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

        optimizer_snapshot_path = Path(
            config.initial_model_prefix / f"optimizer{infix}.pth"
        )
        if (
            not optimizer_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            optimizer_snapshot_path = None

        scheduler_snapshot_path = Path(
            config.initial_model_prefix / f"lr-scheduler{infix}.pth"
        )
        if (
            not scheduler_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            scheduler_snapshot_path = None

    if config.reward_plugin is not None and not config.reward_plugin.exists():
        errmsg = f"{config.reward_plugin}: Does not exist."
        raise RuntimeError(errmsg)
    if config.reward_plugin is not None and not config.reward_plugin.is_file():
        errmsg = f"{config.reward_plugin}: Not a file."
        raise RuntimeError(errmsg)

    if not math.isfinite(config.discount_factor) or not 0.0 <= config.discount_factor <= 1.0:
        errmsg = (
            f"{config.discount_factor}: "
            "An invalid value for `discount_factor`."
        )
        raise RuntimeError(errmsg)

    if config.reward_plugin is None and config.discount_factor != 1.0:
        raise ValueError("Built-in placement rewards require discount_factor=1. Use --config-name placement.")
    if type(config.q_monitor_interval) is not int or config.q_monitor_interval < 0:
        raise ValueError("q_monitor_interval must be a nonnegative number of batches.")
    if config.data_cache is not None and (
        config.replay_buffer_size < 1 or config.data_cache_bytes < 65536
    ):
        raise ValueError("data_cache requires replay and a byte budget of at least 65536.")
    if (
        config.td_loss not in ("mse", "huber")
        or config.export_network not in ("online", "target")
        or not math.isfinite(config.cql_temperature)
        or config.cql_temperature <= 0
    ):
        raise ValueError("Invalid scalar CQL loss, temperature or export network.")
    for removed_option in ("mc_weight", "critic_warmup_steps"):
        if removed_option in config:
            raise ValueError(
                f"{removed_option} is no longer supported: choose an explicit MC or TD "
                "target and train every online-network parameter."
            )
    if config.reward_scheme not in ("placement", "round_and_placement"):
        raise ValueError("Unknown reward_scheme.")
    if config.target_kind not in ("td", "mc"):
        raise ValueError("CQL target_kind must be mc or td.")
    if config.target_kind == "mc" and (
        config.double_q_learning or config.num_qr_intervals is not None
        or not config.contiguous_training_data or config.replay_buffer_size < 1
        or config.reward_plugin is not None or config.reward_model is not None
        or config.export_network != "online"
    ):
        raise ValueError("MC CQL requires scalar online export, complete episode replay, observed rewards and double_q_learning=false.")
    if (not math.isfinite(config.round_score_unit) or config.round_score_unit <= 0
            or not math.isfinite(config.round_reward_weight) or config.round_reward_weight < 0):
        raise ValueError("Invalid round reward scale or weight.")
    if config.reward_scheme == "round_and_placement" and (
        not config.contiguous_training_data or config.replay_buffer_size < 1
        or config.reward_plugin is not None or config.reward_model is not None
    ):
        raise ValueError("Round rewards need complete episode replay and cannot be combined with reward_plugin or reward_model.")
    if config.bc_initial_model is not None:
        if config.encoder.architecture != "dense_decoder" or config.encoder.load_from is not None or config.decoder.load_from is not None:
            raise ValueError("bc_initial_model requires the dense decoder and conflicts with partial snapshot loading.")
        if not config.bc_initial_model.is_file():
            raise ValueError("The BC feature reference model does not exist.")
    if config.encoder.architecture == "dense_decoder":
        if config.num_qr_intervals is not None or (config.target_kind == "td" and not config.double_q_learning):
            raise ValueError("Dense decoder CQL requires scalar MC returns or Double DQN.")
        if config.decoder.load_from is not None:
            raise ValueError(
                "Transfer only encoder.load_from from BC; initialize a new Q head. "
                "Use initial_model_prefix for a complete CQL resume."
            )
    if config.num_qr_intervals is not None and config.export_network != "target":
        raise ValueError("Quantile CQL retains its legacy target export.")
    if config.reward_model is not None:
        if config.reward_plugin is not None:
            raise ValueError("Specify reward_model or reward_plugin, not both.")
        if not config.contiguous_training_data:
            raise ValueError("reward_model requires contiguous_training_data=true.")
        if not config.reward_model.is_file():
            raise ValueError(f"{config.reward_model}: EOR model not found.")

    if not math.isfinite(config.kappa) or config.kappa < 0.0:
        errmsg = f"{config.kappa}: `kappa` must be a non-negative real value."
        raise RuntimeError(errmsg)

    if not math.isfinite(config.alpha) or config.alpha < 0.0:
        errmsg = f"{config.alpha}: `alpha` must be a non-negative real value."
        raise RuntimeError(errmsg)

    if config.batch_size <= 0:
        errmsg = (
            f"{config.batch_size}: `batch_size` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if config.gradient_accumulation_steps <= 0:
        errmsg = (
            f"{config.gradient_accumulation_steps}: "
            "`gradient_accumulation_steps` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if config.max_gradient_norm <= 0.0:
        errmsg = (
            f"{config.max_gradient_norm}: "
            "`max_gradient_norm` must be a positive real value."
        )
        raise RuntimeError(errmsg)

    _config.optimizer.validate(config.optimizer)

    if config.target_update_interval < 0:
        errmsg = (
            f"{config.target_update_interval}: "
            "`target_update_interval` must be a non-negative integer."
        )
        raise RuntimeError(errmsg)
    if config.double_q_learning and config.target_update_interval == 0:
        errmsg = (
            "`target_update_interval` must be a positive integer "
            "for double Q-learning."
        )
        raise RuntimeError(errmsg)

    if config.target_update_rate < 0.0 or config.target_update_rate > 1.0:
        errmsg = (
            f"{config.target_update_rate}: `target_update_rate` must be "
            "a real value within the range `[0.0, 1.0]`."
        )
        raise RuntimeError(errmsg)
    if config.double_q_learning and config.target_update_rate == 0.0:
        errmsg = (
            "`target_update_rate` must be a positive real value "
            "for double Q-learning."
        )
        raise RuntimeError(errmsg)

    if config.snapshot_interval < 0:
        errmsg = (
            f"{config.snapshot_interval}: "
            "`snapshot_interval` must be a non-negative integer."
        )
        raise RuntimeError(errmsg)

    output_prefix = Path(HydraConfig.get().runtime.output_dir)

    if rank == 0:
        logging.info("Model type: conservative Q-learning (CQL)")

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

        if config.num_qr_intervals is None:
            logging.info("# of QR intervals: (N/A)")
        else:
            logging.info("# of QR intervals: %d", config.num_qr_intervals)
        logging.info("Dueling network: %s", config.dueling_network)

        if config.initial_model_prefix is not None:
            logging.info(
                "Initial model prefix: %s", config.initial_model_prefix
            )
            logging.info("Initlal model index: %d", config.initial_model_index)
            if config.optimizer.initialize:
                logging.info("(Will not load optimizer)")

        logging.info("Reward plugin: %s", config.reward_plugin)
        logging.info("Double Q-learning: %s", config.double_q_learning)
        logging.info("CQL target kind: %s", config.target_kind)
        logging.info("Discount factor: %f", config.discount_factor)
        logging.info("Kappa: %f", config.kappa)
        logging.info("Alpha: %f", config.alpha)
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
            "Norm threshold for gradient clipping: %E",
            config.max_gradient_norm,
        )

        _config.optimizer.dump(config.optimizer)

        if config.double_q_learning:
            logging.info(
                "Target network update interval: %d",
                config.target_update_interval,
            )
            logging.info(
                "Target network update rate: %f", config.target_update_rate
            )
        else:
            logging.info(
                "Target network update interval: "
                "(ignored since double Q-learning is disabled)"
            )
            logging.info(
                "Target network update rate: "
                "(ignored since double Q-learning is disabled)"
            )

        if config.initial_model_prefix is not None:
            if config.double_q_learning:
                logging.info(
                    "Initial source encoder snapshot: %s",
                    source_encoder_snapshot_path,
                )
                logging.info(
                    "Initial source decoder snapshot: %s",
                    source_decoder_snapshot_path,
                )
                logging.info(
                    "Initial target encoder snapshot: %s",
                    target_encoder_snapshot_path,
                )
                logging.info(
                    "Initial target decoder snapshot: %s",
                    target_decoder_snapshot_path,
                )
            else:
                logging.info(
                    "Initial encoder snapshot: %s", encoder_snapshot_path
                )
                logging.info(
                    "Initial decoder snapshot: %s", decoder_snapshot_path
                )
            if optimizer_snapshot_path is not None:
                logging.info(
                    "Initial optimizer snapshot: %s", optimizer_snapshot_path
                )
            if scheduler_snapshot_path is not None:
                logging.info(
                    "Initial LR scheduler snapshot: %s",
                    scheduler_snapshot_path,
                )

        logging.info("Output prefix: %s", output_prefix)
        if config.snapshot_interval == 0:
            logging.info("Snapshot interval: N/A")
        else:
            logging.info("Snapshot interval: %d", config.snapshot_interval)

    if world_size >= 2:
        init_process_group(backend="nccl" if device.type == "cuda" else "gloo")

    decision_recipe = decision_checkpoint.recipe(
        config, "cql", dtype=dtype, amp_dtype=amp_dtype
    )
    initial_rng_state = decision_checkpoint.validate_resume(config, decision_recipe)
    encoder = make_backbone(
        config.encoder, checkpointing=config.checkpointing, dtype=dtype
    )
    encoder_tdm = TensorDictModule(
        encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["encode"],  # type: ignore
    )
    decoder: nn.Module
    if config.num_qr_intervals is None:
        decoder = make_candidate_head(
            config.decoder,
            input_dimension=config.encoder.dimension,
            architecture=config.encoder.architecture,
            dtype=dtype,
        )
        decoder_tdm = TensorDictModule(
            decoder,
            in_keys=candidate_head_keys(config.encoder.architecture, config.decoder),
            out_keys=["action_value"],  # type: ignore
        )
    else:
        decoder = QRDecoder(
            input_dimension=config.encoder.dimension,
            dimension=config.decoder.dimension,
            activation_function=config.decoder.activation_function,
            dropout=config.decoder.dropout,
            layer_normalization=config.decoder.layer_normalization,
            num_layers=config.decoder.num_layers,
            num_qr_intervals=config.num_qr_intervals,
            dueling_network=config.dueling_network,
            noise_init_std=None,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        decoder_tdm = TensorDictModule(
            decoder,
            in_keys=["candidates", "encode"],  # type: ignore
            out_keys=["qr_action_value"],  # type: ignore
        )
    decoder.zero_output()
    network = TensorDictSequential([encoder_tdm, decoder_tdm])
    if world_size >= 2:
        network.to(device=device)
        for _param in network.parameters():
            broadcast(_param.data, src=0)
        network.to(device=torch.device("cpu"))

    target_encoder: Encoder | None = None
    target_encoder_tdm: TensorDictModule | None = None
    target_decoder: nn.Module | None = None
    target_decoder_tdm: TensorDictModule | None = None
    target_network: TensorDictSequential | None = None
    if config.double_q_learning:
        target_encoder = make_backbone(
            config.encoder, checkpointing=config.checkpointing, dtype=dtype
        )
        target_encoder_tdm = TensorDictModule(
            target_encoder,
            in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
            out_keys=["encode"],  # type: ignore
        )
        if config.num_qr_intervals is None:
            target_decoder = make_candidate_head(
                config.decoder,
                input_dimension=config.encoder.dimension,
                architecture=config.encoder.architecture,
                dtype=dtype,
            )
            target_decoder_tdm = TensorDictModule(
                target_decoder,
                in_keys=candidate_head_keys(config.encoder.architecture, config.decoder),
                out_keys=["action_value"],  # type: ignore
            )
        else:
            target_decoder = QRDecoder(
                input_dimension=config.encoder.dimension,
                dimension=config.decoder.dimension,
                activation_function=config.decoder.activation_function,
                dropout=config.decoder.dropout,
                layer_normalization=config.decoder.layer_normalization,
                num_layers=config.decoder.num_layers,
                num_qr_intervals=config.num_qr_intervals,
                dueling_network=config.dueling_network,
                noise_init_std=None,
                device=torch.device("cpu"),
                dtype=dtype,
            )
            target_decoder_tdm = TensorDictModule(
                target_decoder,
                in_keys=["candidates", "encode"],  # type: ignore
                out_keys=["qr_action_value"],  # type: ignore
            )
        target_network = TensorDictSequential(
            [target_encoder_tdm, target_decoder_tdm]
        )
        with torch.no_grad():
            for _param, _target_param in zip(
                network.parameters(), target_network.parameters()
            ):
                _target_param.data = _param.data.detach().clone()

    q_decoder: QDecoder | None = None
    q_decoder_tdm: TensorDictModule | None = None
    if config.num_qr_intervals is None:
        argmax_layer = DecodeConverter(
            "argmax", tie_break_by_action_id=config.encoder.architecture == "dense_decoder"
        )
        argmax_layer_tdm = TensorDictModule(
            argmax_layer,
            in_keys=["candidates", "action_value"],  # type: ignore
            out_keys=["action"],  # type: ignore
        )
        if config.double_q_learning and config.export_network == "target":
            assert target_encoder_tdm is not None
            assert target_decoder_tdm is not None
            network_to_save = TensorDictSequential(
                [
                    target_encoder_tdm,
                    target_decoder_tdm,
                    argmax_layer_tdm,
                ]
            )
        else:
            network_to_save = TensorDictSequential(
                [encoder_tdm, decoder_tdm, argmax_layer_tdm]
            )
    else:
        q_decoder = QDecoder()
        q_decoder_tdm = TensorDictModule(
            q_decoder,
            in_keys=["qr_action_value"],  # type: ignore
            out_keys=["action_value"],  # type: ignore
        )
        argmax_layer = DecodeConverter("argmax")
        argmax_layer_tdm = TensorDictModule(
            argmax_layer,
            in_keys=["candidates", "action_value"],  # type: ignore
            out_keys=["action"],  # type: ignore
        )
        if config.double_q_learning:
            assert target_encoder_tdm is not None
            assert target_decoder_tdm is not None
            network_to_save = TensorDictSequential(
                [
                    target_encoder_tdm,
                    target_decoder_tdm,
                    q_decoder_tdm,
                    argmax_layer_tdm,
                ]
            )
        else:
            network_to_save = TensorDictSequential(
                [encoder_tdm, decoder_tdm, q_decoder_tdm, argmax_layer_tdm]
            )

    network.requires_grad_(True)
    network.train()
    network = network.to(device=device, dtype=dtype)
    if world_size >= 2:
        network = nn.SyncBatchNorm.convert_sync_batchnorm(network)
        network = DistributedDataParallel(network)

    if config.double_q_learning:
        assert target_network is not None
        target_network.requires_grad_(False)
        target_network.eval()
        target_network = target_network.to(device=device, dtype=dtype)

        # Online exports share the trainable source modules. Do not freeze
        # them merely because a frozen training target also exists.
        if config.export_network == "target":
            network_to_save.requires_grad_(False)
            network_to_save.eval()
        network_to_save = network_to_save.to(device=device, dtype=dtype)
    else:
        network_to_save.requires_grad_(True)
        network_to_save.train()
        network_to_save = network_to_save.to(device=device, dtype=dtype)

    optimizer, scheduler = _config.optimizer.create(
        device.type, config.optimizer, network
    )

    if config.bc_initial_model is not None and config.initial_model_prefix is None:
        from kanachan.nn.decision_value import transfer_bc_features

        bc_model = load_model(config.bc_initial_model, map_location=torch.device("cpu")).float()
        transfer_bc_features(encoder, decoder, bc_model)
        if target_encoder is not None and target_decoder is not None:
            transfer_bc_features(target_encoder, target_decoder, bc_model)
        del bc_model

    if config.encoder.load_from is not None:
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None
        encoder_state_dict = torch.load(
            config.encoder.load_from, map_location="cpu", weights_only=True
        )
        encoder.load_state_dict(encoder_state_dict)
        encoder.to(device=device, dtype=dtype)

        if config.double_q_learning:
            assert target_encoder is not None
            target_encoder.load_state_dict(encoder_state_dict)
            target_encoder.to(device=device, dtype=dtype)

    if config.decoder.load_from is not None:
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None
        decoder_state_dict = torch.load(
            config.decoder.load_from, map_location="cpu", weights_only=True
        )
        decoder.load_state_dict(decoder_state_dict)
        decoder.to(device=device, dtype=dtype)

        if config.double_q_learning:
            assert target_decoder is not None
            target_decoder.load_state_dict(decoder_state_dict)
            target_decoder.to(device=device, dtype=dtype)

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        if config.double_q_learning:
            assert source_encoder_snapshot_path is not None
            assert source_decoder_snapshot_path is not None
            assert target_encoder_snapshot_path is not None
            assert target_encoder is not None
            assert target_decoder_snapshot_path is not None
            assert target_decoder is not None
            assert target_network is not None

            source_encoder_state_dict = torch.load(
                source_encoder_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            encoder.load_state_dict(source_encoder_state_dict)
            encoder.to(device=device, dtype=dtype)

            source_decoder_state_dict = torch.load(
                source_decoder_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            decoder.load_state_dict(source_decoder_state_dict)
            decoder.to(device=device, dtype=dtype)

            target_encoder_state_dict = torch.load(
                target_encoder_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            target_encoder.load_state_dict(target_encoder_state_dict)
            target_encoder.to(device=device, dtype=dtype)

            target_decoder_state_dict = torch.load(
                target_decoder_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            target_decoder.load_state_dict(target_decoder_state_dict)
            target_decoder.to(device=device, dtype=dtype)
        else:
            assert encoder_snapshot_path is not None
            assert decoder_snapshot_path is not None

            encoder_state_dict = torch.load(
                encoder_snapshot_path, map_location="cpu", weights_only=True
            )
            encoder.load_state_dict(encoder_state_dict)
            encoder.to(device=device, dtype=dtype)

            decoder_state_dict = torch.load(
                decoder_snapshot_path, map_location="cpu", weights_only=True
            )
            decoder.load_state_dict(decoder_state_dict)
            decoder.to(device=device, dtype=dtype)

        if optimizer_snapshot_path is not None:
            optimizer_state_dict = torch.load(
                optimizer_snapshot_path, map_location="cpu", weights_only=True
            )
            optimizer.load_state_dict(optimizer_state_dict)

        if scheduler is not None and scheduler_snapshot_path is not None:
            schedular_state_dict = torch.load(
                scheduler_snapshot_path, map_location="cpu", weights_only=True
            )
            scheduler.load_state_dict(schedular_state_dict)

    from kanachan.nn.decision_value import q_only
    # Keep the DDP forward for distributed training; optimize the single-rank
    # dense path without changing exported parameter names or optimizer IDs.
    td_source = q_only(network) if world_size == 1 else network
    td_target = q_only(target_network) if target_network is not None else None

    def td_error_computation(data: TensorDict) -> None:
        if config.num_qr_intervals is not None:
            compute_td_error(
                source_network=network,
                target_network=target_network
                if config.double_q_learning
                else None,
                data=data,
                discount_factor=config.discount_factor,
                kappa=config.kappa,
            )
            return

        if config.encoder.architecture == "dense_decoder":
            values, chosen, expected = td_components(
                td_source, td_target if config.double_q_learning else None,
                data, default_discount=config.discount_factor,
            )
            data["action_value"] = values
            data["td_target"] = expected
            # The existing optimizer loop applies 0.5 to this field.
            data["td_error"] = 2 * td_penalty(chosen, expected, config.td_loss)
            return

        batch_size = int(data.batch_size[0])

        # Compute a^*.
        copy = TensorDict(
            {
                "sparse": data["next", "sparse"].detach().clone(),
                "numeric": data["next", "numeric"].detach().clone(),
                "progression": data["next", "progression"].detach().clone(),
                "candidates": data["next", "candidates"].detach().clone(),
            },
            batch_size=batch_size,
            device=data.device,
        )
        with torch.no_grad():
            network(copy)

        a_star = copy["action_value"].argmax(dim=1).int()
        assert isinstance(a_star, Tensor)
        assert a_star.device == device
        assert a_star.dtype == torch.int32
        assert a_star.dim() == 1
        assert a_star.size(0) == batch_size
        assert (0 <= a_star).all().item()
        assert (a_star < MAX_NUM_ACTION_CANDIDATES).all().item()

        copy = TensorDict(
            {
                "sparse": data["sparse"].detach().clone(),
                "numeric": data["numeric"].detach().clone(),
                "progression": data["progression"].detach().clone(),
                "candidates": data["candidates"].detach().clone(),
            },
            batch_size=batch_size,
            device=data.device,
        )
        network(copy)
        data["action_value"] = copy["action_value"]

        copy = TensorDict(
            {
                "sparse": data["next", "sparse"].detach().clone(),
                "numeric": data["next", "numeric"].detach().clone(),
                "progression": data["next", "progression"].detach().clone(),
                "candidates": data["next", "candidates"].detach().clone(),
            },
            batch_size=batch_size,
            device=data.device,
        )
        with torch.no_grad():
            if config.double_q_learning:
                assert target_network is not None
                target_network(copy)
            else:
                network(copy)
        data["next", "action_value"] = copy["action_value"]

        action = data["action"]
        assert isinstance(action, Tensor)
        assert action.device == device
        assert action.dtype == torch.int32
        assert action.dim() == 1
        assert action.size(0) == batch_size
        assert (0 <= action).all().item()
        assert (action < MAX_NUM_ACTION_CANDIDATES).all().item()

        reward = data["next", "reward"]
        assert isinstance(reward, Tensor)
        assert reward.device == device
        assert reward.dtype == dtype
        assert reward.dim() == 1
        assert reward.size(0) == batch_size

        done = data["next", "done"]
        assert isinstance(done, Tensor)
        assert done.device == device
        assert done.dtype == torch.bool
        assert done.dim() == 1
        assert done.size(0) == batch_size

        q_sa = data["action_value"][
            torch.arange(batch_size, device=device), action
        ]

        next_q_sa = data["next", "action_value"][
            torch.arange(batch_size, device=device), a_star
        ]

        target = compute_td_target(
            reward, next_q_sa, done, data.get(("next", "discount"), config.discount_factor)
        )

        td_error = 2 * td_penalty(q_sa, target, config.td_loss)

        data["td_error"] = td_error

    snapshots_path = output_prefix / "snapshots"

    def snapshot_writer(
        num_samples: int | None = None,
        *,
        resume_num_samples: int | None = None,
        optimizer_steps: int = 0,
        cursor_is_exact: bool = True,
        grad_scaler_state: dict | None = None,
        replay_state: dict | None = None,
    ) -> None:
        snapshots_path.mkdir(parents=True, exist_ok=True)

        infix = "" if num_samples is None else f".{num_samples}"

        if config.double_q_learning:
            assert target_encoder is not None
            assert target_decoder is not None
            torch.save(
                encoder.state_dict(),
                snapshots_path / f"source-encoder{infix}.pth",
            )
            torch.save(
                decoder.state_dict(),
                snapshots_path / f"source-decoder{infix}.pth",
            )
            torch.save(
                target_encoder.state_dict(),
                snapshots_path / f"target-encoder{infix}.pth",
            )
            torch.save(
                target_decoder.state_dict(),
                snapshots_path / f"target-decoder{infix}.pth",
            )
        else:
            torch.save(
                encoder.state_dict(), snapshots_path / f"encoder{infix}.pth"
            )
            torch.save(
                decoder.state_dict(), snapshots_path / f"decoder{infix}.pth"
            )
        torch.save(
            optimizer.state_dict(), snapshots_path / f"optimizer{infix}.pth"
        )
        if scheduler is not None:
            torch.save(
                scheduler.state_dict(),
                snapshots_path / f"lr-scheduler{infix}.pth",
            )

        if config.double_q_learning and config.export_network == "target":
            assert target_encoder is not None
            assert target_decoder is not None
            encoder_to_save = target_encoder
            encoder_tdm_to_save = target_encoder_tdm
            decoder_to_save = target_decoder
            decoder_tdm_to_save = target_decoder_tdm
        else:
            encoder_to_save = encoder
            encoder_tdm_to_save = encoder_tdm
            decoder_to_save = decoder
            decoder_tdm_to_save = decoder_tdm
        if config.num_qr_intervals is None:
            network_state = dump_model(
                network_to_save,
                [
                    dump_object(
                        encoder_tdm_to_save,
                        [
                            dump_object(
                                encoder_to_save,
                                [],
                                backbone_kwargs(
                                    config.encoder, checkpointing=config.checkpointing, dtype=dtype
                                ),
                            ),
                        ],
                        {
                            "in_keys": [
                                "sparse",
                                "numeric",
                                "progression",
                                "candidates",
                            ],
                            "out_keys": ["encode"],
                        },
                    ),
                    dump_object(
                        decoder_tdm_to_save,
                        [
                            dump_object(
                                decoder_to_save,
                                [],
                                candidate_head_kwargs(
                                    config.decoder, input_dimension=config.encoder.dimension,
                                    architecture=config.encoder.architecture, dtype=dtype,
                                ),
                            ),
                        ],
                        {
                            "in_keys": candidate_head_keys(config.encoder.architecture, config.decoder),
                            "out_keys": ["action_value"],
                        },
                    ),
                    dump_object(
                        argmax_layer_tdm,
                        [
                            dump_object(
                                argmax_layer,
                                ["argmax"],
                                {"tie_break_by_action_id": config.encoder.architecture == "dense_decoder"},
                            ),
                        ],
                        {
                            "in_keys": ["candidates", "action_value"],
                            "out_keys": ["action"],
                        },
                    ),
                ],
                {},
            )
        else:
            assert q_decoder is not None
            assert q_decoder_tdm is not None
            network_state = dump_model(
                network_to_save,
                [
                    dump_object(
                        encoder_tdm_to_save,
                        [
                            dump_object(
                                encoder_to_save,
                                [],
                                {
                                    "position_encoder": config.encoder.position_encoder,
                                    "dimension": config.encoder.dimension,
                                    "num_heads": config.encoder.num_heads,
                                    "dim_feedforward": config.encoder.dim_feedforward,
                                    "activation_function": config.encoder.activation_function,
                                    "dropout": config.encoder.dropout,
                                    "layer_normalization": config.encoder.layer_normalization,
                                    "architecture": config.encoder.architecture,
                                    "num_layers": config.encoder.num_layers,
                                    "checkpointing": config.checkpointing,
                                    "device": torch.device("cpu"),
                                    "dtype": dtype,
                                },
                            )
                        ],
                        {
                            "in_keys": [
                                "sparse",
                                "numeric",
                                "progression",
                                "candidates",
                            ],
                            "out_keys": ["encode"],
                        },
                    ),
                    dump_object(
                        decoder_tdm_to_save,
                        [
                            dump_object(
                                decoder_to_save,
                                [],
                                {
                                    "input_dimension": config.encoder.dimension,
                                    "dimension": config.decoder.dimension,
                                    "activation_function": config.decoder.activation_function,
                                    "dropout": config.decoder.dropout,
                                    "layer_normalization": config.decoder.layer_normalization,
                                    "num_layers": config.decoder.num_layers,
                                    "num_qr_intervals": config.num_qr_intervals,
                                    "dueling_network": config.dueling_network,
                                    "noise_init_std": None,
                                    "device": torch.device("cpu"),
                                    "dtype": dtype,
                                },
                            )
                        ],
                        {
                            "in_keys": ["candidates", "encode"],
                            "out_keys": ["qr_action_value"],
                        },
                    ),
                    dump_object(
                        q_decoder_tdm,
                        [dump_object(q_decoder, [], {})],
                        {
                            "in_keys": ["qr_action_value"],
                            "out_keys": ["action_value"],
                        },
                    ),
                    dump_object(
                        argmax_layer_tdm,
                        [dump_object(argmax_layer, ["argmax"], {})],
                        {
                            "in_keys": ["candidates", "action_value"],
                            "out_keys": ["action"],
                        },
                    ),
                ],
                {},
            )
        torch.save(network_state, snapshots_path / f"model{infix}.kanachan")

        cursor = (
            resume_num_samples
            if resume_num_samples is not None
            else num_samples
        )
        assert cursor is not None
        if replay_state is not None:
            from kanachan.training.online.state import atomic_save

            atomic_save(snapshots_path / f"replay-state{infix}.pth", replay_state)
        save_training_state(
            snapshots_path,
            infix,
            TrainingState(
                cursor,
                optimizer_steps,
                cursor_is_exact and (config.get("replay_buffer_size", 0) == 0 or replay_state is not None),
                grad_scaler_state,
            ),
        )
        decision_checkpoint.save(
            snapshots_path, infix, decision_recipe, algorithm="cql"
        )

    tensorboard_path = output_prefix / "tensorboard"
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    if config.candidate_feature_workers < 0 or config.prepared_mc_bytes < 1 or config.prepared_mc_epochs < 1:
        raise ValueError("Invalid prepared MC workers, RAM budget or epoch count.")
    if config.candidate_feature_workers and (
        config.target_kind != "mc" or config.encoder.architecture != "dense_decoder"
        or config.decoder.candidate_features != "discard_v1" or world_size != 1
        or config.num_workers != 0 or not config.contiguous_training_data
        or config.start_new_epoch
    ):
        raise ValueError("Prepared afterstates require sequential single-process dense MC+CQL without start_new_epoch.")
    with SummaryWriter(log_dir=tensorboard_path) as summary_writer, ExitStack() as replay_resources:
        replay_buffer_factory = None
        if config.candidate_feature_workers:
            from kanachan.training.core.offline_rl.prepared_mc import PreparedMCCache

            def replay_buffer_factory(**arguments):
                if arguments["num_skip_samples"] != 0:
                    raise ValueError("Prepared MC cache must regenerate from its complete source.")
                return replay_resources.enter_context(PreparedMCCache(
                    lambda: EpisodeReplayBuffer(**arguments), workers=config.candidate_feature_workers,
                    max_bytes=config.prepared_mc_bytes, seed=config.prepared_mc_seed,
                    epochs=config.prepared_mc_epochs, pin_memory=device.type == "cuda",
                ))
        torch.autograd.set_detect_anomaly(
            False
        )  # `True` for debbing purpose only.
        _training(
            training_data=config.training_data,
            contiguous_training_data=config.contiguous_training_data,
            rewrite_rooms=config.rewrite_rooms,
            rewrite_grades=config.rewrite_grades,
            num_workers=config.num_workers,
            replay_buffer_size=config.replay_buffer_size,
            device=device,
            dtype=dtype,
            amp_dtype=amp_dtype,
            source_network=network,
            target_network=target_network,
            reward_plugin=config.reward_plugin,
            reward_model=config.reward_model,
            reward_scheme=config.reward_scheme,
            round_score_unit=config.round_score_unit,
            round_reward_weight=config.round_reward_weight,
            target_kind=config.target_kind,
            regression_loss=config.td_loss,
            replay_state=replay_state,
            discount_factor=config.discount_factor,
            td_error_computation=td_error_computation,
            alpha=config.alpha,
            cql_temperature=config.cql_temperature,
            q_monitor_interval=config.q_monitor_interval,
            data_cache=config.data_cache, data_cache_bytes=config.data_cache_bytes,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_gradient_norm=config.max_gradient_norm,
            optimizer=optimizer,
            scheduler=scheduler,
            target_update_interval=config.target_update_interval,
            target_update_rate=config.target_update_rate,
            snapshot_interval=config.snapshot_interval,
            num_samples=num_samples,
            initial_optimizer_steps=initial_optimizer_steps,
            initial_grad_scaler_state=initial_grad_scaler_state,
            initial_rng_state=initial_rng_state,
            replay_buffer_factory=replay_buffer_factory,
            summary_writer=summary_writer,
            snapshot_writer=snapshot_writer,
        )  # pylint: disable=missing-kwoa


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
    sys.exit(0)
