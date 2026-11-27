import datetime
from contextlib import ExitStack
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tensordict import TensorDict  # type: ignore
from tensordict.nn import (  # type: ignore
    TensorDictModule,
    TensorDictSequential,
)
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.distributed import (
    ReduceOp,
    all_reduce,
    broadcast,
    init_process_group,
)
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter

# pylint: disable=unused-import
import kanachan.training.bc.config  # noqa: F401
import kanachan.training.core.config as _config
from kanachan.constants import MAX_NUM_ACTION_CANDIDATES, NUM_TYPES_OF_ACTIONS
from kanachan.model_loader import dump_model, dump_object
from kanachan.nn.decision_factory import (
    make_backbone, backbone_kwargs, backbone_in_keys, make_candidate_head,
    candidate_head_kwargs, candidate_head_keys,
    load_candidate_head_state,
)
from kanachan.training.core import decision_checkpoint
from kanachan.nn import DecodeConverter, Decoder, Encoder
from kanachan.training.common import (
    get_distributed_environment,
    average_tensor,
    with_last,
    rescale_gradients,
    checked_clip_gradients,
    backoff_gradient_scale,
)
from kanachan.training.core.bc import DataLoader
from kanachan.training.core.checkpoint import (
    SnapshotWriter,
    TrainingState,
    load_training_state,
    save_training_state,
)
from kanachan.training.core.progress import TrainingProgress
from kanachan.training.core.optimizers import (
    adamw_parameter_groups,
    consolidate_optimizer_state,
    move_optimizer_state,
    validate_optimizer_parameter_coverage,
)
from kanachan.training.online.state import capture_rng


def run_v2_trajectory_bc(
    config: DictConfig,
    *,
    world_size: int,
    rank: int,
    local_rank: int,
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    """Production Trajectory V2 BC training entrypoint using disk-spooled source-v3 rounds."""
    from kanachan.training.bc.streaming_v2 import SpoolingRoundBCDataSource
    from kanachan.training.bc.trajectory_v2 import (
        TrajectoryBCModel,
        _expected_inactive_bc_param_ids,
        save_trajectory_bc_checkpoint,
        train_trajectory_bc_steps,
    )

    if config.encoder.architecture != "structured_dual_v2":
        raise ValueError(
            f"training_mode='v2_trajectory' requires encoder.architecture='structured_dual_v2', "
            f"got {config.encoder.architecture!r}."
        )
    if config.rewrite_rooms is not None or config.rewrite_grades is not None:
        raise ValueError("v2_trajectory mode does not rewrite source-v3 room/grade sparse tokens.")
    if config.initial_model_prefix is not None or config.initial_model_index is not None:
        raise ValueError(
            "training_mode='v2_trajectory' uses resume_from=<ckpt.pt>, not initial_model_prefix/initial_model_index."
        )
    if config.encoder.load_from is not None or config.decoder.load_from is not None:
        raise ValueError(
            "training_mode='v2_trajectory' does not support legacy encoder.load_from/decoder.load_from; use resume_from."
        )
    if int(config.candidate_feature_workers) > 0:
        raise ValueError(
            "v2_trajectory derives structured_v2 features inside SpoolingRoundBCDataSource; candidate_feature_workers > 0 is unsupported."
        )
    if int(config.prefetch_batches) < 0:
        raise ValueError(f"prefetch_batches must be >= 0, got {config.prefetch_batches}.")
    if int(config.spool_num_buckets) < 1 or int(config.spool_max_bucket_bytes) < 512:
        raise ValueError("Invalid spool_num_buckets (<1) or spool_max_bucket_bytes (<512).")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}.")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be >= 1.")
    if config.max_gradient_norm <= 0.0 or not math.isfinite(config.max_gradient_norm):
        raise ValueError(f"Invalid max_gradient_norm={config.max_gradient_norm}.")
    if config.snapshot_interval < 0:
        raise ValueError(f"snapshot_interval must be >= 0, got {config.snapshot_interval}.")
    if not math.isfinite(config.next_event_weight) or config.next_event_weight < 0.0:
        raise ValueError(f"Invalid next_event_weight={config.next_event_weight}.")
    if not math.isfinite(config.final_rank_weight) or config.final_rank_weight < 0.0:
        raise ValueError(f"Invalid final_rank_weight={config.final_rank_weight}.")
    if float(config.final_rank_weight) > 0.0 and not bool(config.use_final_rank_head):
        raise ValueError("final_rank_weight > 0 requires use_final_rank_head=True.")
    if bool(config.allow_unlabeled) and bool(config.use_final_rank_head) and float(config.final_rank_weight) > 0.0:
        raise ValueError("allow_unlabeled=True conflicts with use_final_rank_head=True and final_rank_weight > 0.")
    if not math.isfinite(config.label_smoothing) or not 0.0 <= config.label_smoothing < 1.0:
        raise ValueError(f"Invalid label_smoothing={config.label_smoothing}.")
    _config.optimizer.validate(config.optimizer)
    if config.optimizer.type != "adamw":
        raise ValueError(
            f"v2_trajectory requires optimizer.type='adamw', got {config.optimizer.type!r}."
        )

    if world_size >= 2 and not torch.distributed.is_initialized():
        init_process_group(backend="nccl" if device.type == "cuda" else "gloo")

    output_prefix = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = output_prefix / "snapshots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(config.spool_cache_dir) if config.spool_cache_dir is not None else (
        output_prefix / "bc_spool_cache"
    )

    data_source = SpoolingRoundBCDataSource(
        config.training_data,
        cache_dir=cache_dir,
        num_hash_buckets=int(config.spool_num_buckets),
        max_bucket_bytes=int(config.spool_max_bucket_bytes),
        max_rounds=config.max_rounds,
        allow_unlabeled=bool(config.allow_unlabeled),
        allow_incomplete=bool(config.allow_incomplete),
        enable_early_shape_hint=bool(getattr(config.encoder, "enable_early_shape_hint", False)),
        suit_augmentation=bool(config.suit_augmentation),
        aug_seed=int(config.aug_seed),
        sampler_seed=int(config.sampler_seed),
        batch_size=int(config.batch_size),
        drop_last=bool(config.drop_last),
    )
    sampler = None
    try:
        sampler = data_source.create_sampler(
            batch_size=int(config.batch_size),
            seed=int(config.sampler_seed),
            rank=rank,
            world_size=world_size,
            drop_last=bool(config.drop_last),
        )
        model = TrajectoryBCModel(
            preset=(getattr(config.encoder, "preset", None) or "custom"),
            dimension=int(config.encoder.dimension),
            num_heads=int(config.encoder.num_heads),
            dim_feedforward=(
                int(config.encoder.dim_feedforward)
                if config.encoder.dim_feedforward is not None
                else None
            ),
            n_d1_layers=int(config.encoder.n_d1_layers),
            n_d2_layers=int(config.encoder.n_d2_layers),
            use_rel_bias=bool(config.encoder.use_rel_bias),
            cand_self_attn=bool(config.encoder.cand_self_attn),
            head_type=str(config.encoder.head_type),
            enable_early_shape_hint=bool(getattr(config.encoder, "enable_early_shape_hint", False)),
            rope_base=float(config.encoder.rope_base),
            use_next_event_head=bool(config.use_next_event_head),
            next_event_weight=float(config.next_event_weight),
            use_final_rank_head=bool(config.use_final_rank_head),
            final_rank_weight=float(config.final_rank_weight),
            label_smoothing=float(config.label_smoothing),
            checkpointing=bool(config.checkpointing),
            device=device,
            dtype=dtype,
        )
        raw_model = model
        train_model: nn.Module = model
        if world_size >= 2:
            if device.type == "cuda":
                train_model = DistributedDataParallel(
                    model, device_ids=[local_rank], output_device=local_rank
                )
            else:
                train_model = DistributedDataParallel(model)

        optimizer, scheduler = _config.optimizer.create(device.type, config.optimizer, raw_model)
        validate_optimizer_parameter_coverage(
            raw_model,
            optimizer,
            inactive_param_ids=_expected_inactive_bc_param_ids(raw_model),
            context="run_v2_trajectory_bc",
        )
        scaler = GradScaler(
            "cuda",
            enabled=(device.type == "cuda" and amp_dtype == torch.float16),
        )
        training_cfg = {
            "batch_size": int(config.batch_size),
            "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
            "max_gradient_norm": float(config.max_gradient_norm),
            "suit_augmentation": bool(config.suit_augmentation),
            "aug_seed": int(config.aug_seed),
            "sampler_seed": int(config.sampler_seed),
            "drop_last": bool(config.drop_last),
            "allow_unlabeled": bool(config.allow_unlabeled),
            "allow_incomplete": bool(config.allow_incomplete),
            "max_rounds": int(config.max_rounds) if config.max_rounds is not None else None,
            "prefetch_batches": int(config.prefetch_batches),
            "dtype": str(dtype),
            "amp_dtype": str(amp_dtype),
            "world_size": int(world_size),
            "optimizer_type": str(config.optimizer.type),
            "use_zero": bool(config.optimizer.use_zero),
            "warmup_steps": int(getattr(config.optimizer, "warmup_steps", 0)),
            "annealing_steps": int(getattr(config.optimizer, "annealing_steps", 0)),
            "annealing_steps_factor": int(getattr(config.optimizer, "annealing_steps_factor", 1)),
            "warmup_start_lr": float(getattr(config.optimizer, "warmup_start_lr", 0.0)),
        }
        recipe = raw_model.recipe(
            data_sha256=data_source.data_sha256,
            source_data_sha256=data_source.source_data_sha256,
            preprocess_recipe_sha256=data_source.preprocess_recipe_sha256,
            ordered_sample_sha256=data_source.ordered_sample_sha256,
            lr=float(config.optimizer.learning_rate),
            weight_decay=float(config.optimizer.weight_decay),
            training_config=training_cfg,
        )
        arch_meta = {
            "training_mode": "v2_trajectory",
            "architecture": recipe["architecture"],
            "preset": recipe["backbone"]["preset"],
            "use_next_event_head": recipe["use_next_event_head"],
            "use_final_rank_head": recipe["use_final_rank_head"],
            "data_sha256": recipe["data_sha256"],
            "source_data_sha256": data_source.source_data_sha256,
            "preprocess_recipe_sha256": data_source.preprocess_recipe_sha256,
            "ordered_sample_sha256": data_source.ordered_sample_sha256,
            "training_config": training_cfg,
            "world_size": world_size,
        }

        if rank == 0:
            logging.info("Model type: behavioral cloning V2 trajectory (v2_trajectory)")
            logging.info("Architecture: %s (preset=%s)", arch_meta["architecture"], arch_meta["preset"])
            _config.device.dump(
                world_size=world_size,
                rank=rank,
                local_rank=local_rank,
                device=device,
                dtype=dtype,
                amp_dtype=amp_dtype,
            )
            logging.info("Spool cache dir: %s (%d rounds, %d games)", cache_dir, len(data_source), data_source.num_games)
            _config.encoder.dump(config.encoder)
            _config.optimizer.dump(config.optimizer)

        tensorboard_path = output_prefix / "tensorboard"
        if rank == 0:
            tensorboard_path.mkdir(parents=True, exist_ok=True)
        with (SummaryWriter(log_dir=tensorboard_path) if rank == 0 else ExitStack()) as summary_writer:
            if rank == 0:
                summary_writer.add_text("architecture/summary", json.dumps(arch_meta, sort_keys=True), 0)
            result = train_trajectory_bc_steps(
                train_model,  # type: ignore[arg-type]
                data_source,
                optimizer,
                batch_size=int(config.batch_size),
                max_steps=config.max_steps,
                device=device,
                amp_dtype=amp_dtype,
                suit_augmentation=bool(config.suit_augmentation),
                aug_seed=int(config.aug_seed),
                sampler_seed=int(config.sampler_seed),
                grad_clip=float(config.max_gradient_norm),
                scheduler=scheduler,
                grad_scaler=scaler,
                ckpt_dir=ckpt_dir,
                checkpoint_every=int(config.snapshot_interval),
                resume_from=config.resume_from,
                data_sha256=data_source.data_sha256,
                recipe=recipe,
                sampler=sampler,
                grad_accum_steps=int(config.gradient_accumulation_steps),
                prefetch_batches=int(config.prefetch_batches),
                prefetch_workers=max(1, int(config.num_workers or 1)),
                drop_last=bool(config.drop_last),
            )
            if rank == 0:
                for k, v in result.get("metrics", {}).items():
                    if math.isfinite(float(v)):
                        summary_writer.add_scalar(f"bc_v2/{k}", float(v), int(result["step"]))
                summary_writer.flush()

        final_step = int(result["step"])
        final_ckpt = ckpt_dir / f"traj-bc-{final_step}.pt"
        need_final = [False]
        if rank == 0:
            need_final[0] = not final_ckpt.exists()
        if world_size >= 2:
            torch.distributed.broadcast_object_list(need_final, src=0)
        if need_final[0]:
            final_aug_gen: torch.Generator | None = None
            if bool(config.suit_augmentation):
                sync_seed = (
                    int(config.aug_seed)
                    + int(rank) * 1_000_003
                    + int(sampler.epoch) * 1_000_003
                    + int(sampler.cursor) * 10_007
                ) & 0x7FFF_FFFF_FFFF_FFFF
                final_aug_gen = torch.Generator(device="cpu").manual_seed(sync_seed)
            save_trajectory_bc_checkpoint(
                ckpt_dir,
                raw_model,
                optimizer,
                sampler=sampler,
                step=final_step,
                recipe=recipe,
                scheduler=scheduler,
                grad_scaler=scaler,
                aug_generator=final_aug_gen,
                numerical_stats={
                    "optimizer_steps": int(result["optimizer_steps"]),
                    "skipped_steps": int(result["skipped_steps"]),
                    "nonfinite_loss_steps": int(result["nonfinite_loss_steps"]),
                    "nonfinite_grad_steps": int(result["nonfinite_grad_steps"]),
                },
            )
        if rank == 0:
            raw_model.dump_actor(ckpt_dir / "model.kanachan")
        if world_size >= 2:
            torch.distributed.barrier()
        return result
    finally:
        if sampler is not None:
            sampler.close()
        data_source.close()


def _training(
    *,
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype,
    training_data: Path,
    num_workers: int,
    rewrite_rooms: int | None,
    rewrite_grades: int | None,
    num_samples: int,
    network_tdm: nn.Module,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_gradient_norm: float,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    snapshot_interval: int,
    summary_writer: SummaryWriter,
    snapshot_writer: SnapshotWriter,
    initial_optimizer_steps: int = 0,
    initial_grad_scaler_state: dict | None = None,
    initial_rng_state: dict | None = None,
    data_loader_factory=None,
) -> None:
    start_time = datetime.datetime.now()

    move_optimizer_state(optimizer)

    world_size, rank, local_rank = get_distributed_environment()

    # Prepare the training data loader. Note that this data loader must iterate
    # the training data set only once.
    data_loader = (data_loader_factory or DataLoader)(
        path=training_data,
        num_skip_samples=num_samples,
        rewrite_rooms=rewrite_rooms,
        rewrite_grades=rewrite_grades,
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
    }
    grad_scaler = GradScaler("cuda", enabled=is_amp_enabled)
    if initial_grad_scaler_state is not None and grad_scaler.is_enabled():
        grad_scaler.load_state_dict(initial_grad_scaler_state)

    last_snapshot = None
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
        initial_optimizer_steps=initial_optimizer_steps,
    )

    loss_function = nn.NLLLoss()

    decision_checkpoint.restore(initial_rng_state)
    for data, is_last in with_last(data_loader):
        data = data.to(device=device, non_blocking=True)
        local_batch_size = int(data.size(0))

        action: Tensor = data["action"]
        assert isinstance(action, Tensor)
        assert action.device == device
        assert action.dtype == torch.int32
        assert action.dim() == 1
        assert action.size(0) == local_batch_size

        copy = data.detach().clone()
        assert isinstance(copy, TensorDict)
        with torch.autocast(**autocast_kwargs):
            network_tdm(copy)

        log_probs: Tensor = copy["log_probs"]
        assert isinstance(log_probs, Tensor)
        assert log_probs.device == device
        assert log_probs.dtype == dtype
        assert log_probs.dim() == 2
        assert log_probs.size(0) == local_batch_size
        assert log_probs.size(1) == MAX_NUM_ACTION_CANDIDATES

        has_choice = (data["candidates"] < NUM_TYPES_OF_ACTIONS).sum(dim=-1) >= 2
        per_sample_loss = nn.functional.nll_loss(
            log_probs, action.long(), reduction="none"
        )
        choice_mask = has_choice.to(per_sample_loss.dtype)
        loss: Tensor = (per_sample_loss * choice_mask).sum() / choice_mask.sum().clamp_min(1.0)

        _loss = loss.detach().clone()
        if world_size >= 2:
            average_tensor(_loss)
        loss_to_display = float(_loss.item())

        if not math.isfinite(loss_to_display):
            errmsg = "Training loss becomes non-finite."
            raise RuntimeError(errmsg)

        loss = loss * (local_batch_size / (configured_batch_size * gradient_accumulation_steps))
        grad_scaler.scale(loss).backward()  # type: ignore

        num_samples += local_batch_size * world_size
        pending_samples += local_batch_size * world_size
        batch_count += 1

        if batch_count % gradient_accumulation_steps == 0 or is_last:
            normalization = (
                configured_batch_size * world_size * gradient_accumulation_steps
                / pending_samples
            )
            rescale_gradients(network_tdm, normalization)
            pending_samples = 0
            grad_scaler.unscale_(optimizer)
            norm = checked_clip_gradients(network_tdm, max_gradient_norm)
            if norm is None:
                if rank == 0:
                    logging.warning(
                        "Skip an optimization step "
                        "because of non-finite values in the gradient."
                    )
                backoff_gradient_scale(grad_scaler)
                optimizer.zero_grad()
                continue

            gradient_norm = float(norm)

            old_scale = grad_scaler.get_scale() if grad_scaler.is_enabled() else None
            grad_scaler.step(optimizer)
            grad_scaler.update()
            step_skipped = old_scale is not None and grad_scaler.get_scale() < old_scale
            if not step_skipped:
                progress.optimizer_step(num_samples)
                if scheduler is not None:
                    scheduler.step()

            optimizer.zero_grad()

            if rank == 0:
                logging.info(
                    "sample = %d, loss = %E, gradient norm = %E",
                    num_samples,
                    loss_to_display,
                    gradient_norm,
                )
            summary_writer.add_scalar("Loss", loss_to_display, num_samples)
            summary_writer.add_scalar(
                "Gradient norm", gradient_norm, num_samples
            )
            if scheduler is not None:
                summary_writer.add_scalar(
                    "LR", scheduler.get_last_lr()[0], num_samples
                )
        else:
            if rank == 0:
                logging.info(
                    "sample = %d, loss = %E", num_samples, loss_to_display
                )
            summary_writer.add_scalar("Loss", loss_to_display, num_samples)

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
                )
            last_snapshot = num_samples

    progress.finish(0)
    elapsed_time = datetime.datetime.now() - start_time
    consolidate_optimizer_state(optimizer)

    if rank == 0:
        logging.info(
            "A training epoch has finished (elapsed time = %s).", elapsed_time
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

    training_mode = str(getattr(config, "training_mode", "single_step"))
    if training_mode not in ("single_step", "v2_trajectory"):
        raise ValueError(
            f"Unknown BC training_mode={training_mode!r}; expected 'single_step' or 'v2_trajectory'."
        )

    if not config.training_data.exists():
        errmsg = f"{config.training_data}: Does not exist."
        raise RuntimeError(errmsg)
    if training_mode == "single_step" and not config.training_data.is_file():
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

    _config.encoder.validate(config.encoder)

    if training_mode == "v2_trajectory":
        run_v2_trajectory_bc(
            config,
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            device=device,
            dtype=dtype,
            amp_dtype=amp_dtype,
        )
        return

    _config.decoder.validate(config.decoder, config.encoder.dimension)

    if config.initial_model_prefix is not None and config.decoder.load_from is not None:
        raise RuntimeError("initial_model_prefix conflicts with decoder.load_from.")

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
                f"{config.initial_model_index}: An invalid initial"
                " model index."
            )
            raise RuntimeError(errmsg)

    num_samples = 0
    initial_optimizer_steps = 0
    initial_grad_scaler_state = None
    encoder_snapshot_path: Path | None = None
    decoder_snapshot_path: Path | None = None
    optimizer_snapshot_path: Path | None = None
    scheduler_snapshot_path: Path | None = None

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None

        if config.initial_model_index is None:
            for child in os.listdir(config.initial_model_prefix):
                match = re.search(
                    "^(?:encoder|decoder|optimizer|scheduler)(?:\\.(\\d+))?\\.pth$",
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

        encoder_snapshot_path = (
            config.initial_model_prefix / f"encoder{infix}.pth"
        )
        assert encoder_snapshot_path is not None
        if not encoder_snapshot_path.exists():
            errmsg = f"{encoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not encoder_snapshot_path.is_file():
            errmsg = f"{encoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        decoder_snapshot_path = (
            config.initial_model_prefix / f"decoder{infix}.pth"
        )
        assert decoder_snapshot_path is not None
        if not decoder_snapshot_path.exists():
            errmsg = f"{decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)

        optimizer_snapshot_path = (
            config.initial_model_prefix / f"optimizer{infix}.pth"
        )
        assert optimizer_snapshot_path is not None
        if (
            not optimizer_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            optimizer_snapshot_path = None

        scheduler_snapshot_path = (
            config.initial_model_prefix / f"scheduler{infix}.pth"
        )
        assert scheduler_snapshot_path is not None
        if (
            not scheduler_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            scheduler_snapshot_path = None

    if config.batch_size <= 0:
        errmsg = (
            f"{config.batch_size}: `batch_size` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if config.gradient_accumulation_steps < 1:
        errmsg = (
            f"{config.gradient_accumulation_steps}:"
            " `gradient_accumulation_steps` must be an integer greater"
            " than 0."
        )
        raise RuntimeError(errmsg)

    if config.max_gradient_norm <= 0.0:
        errmsg = (
            f"{config.max_gradient_norm}: `max_gradient_norm` must be a"
            " positive real value."
        )
        raise RuntimeError(errmsg)

    _config.optimizer.validate(config.optimizer)

    if config.snapshot_interval < 0:
        errmsg = (
            f"{config.snapshot_interval}: `snapshot_interval` must be a"
            " non-negative integer."
        )
        raise RuntimeError(errmsg)

    if config.candidate_feature_workers < 0 or config.candidate_prefetch_batches < 1:
        raise ValueError("Candidate feature workers must be nonnegative and prefetch batches positive.")
    is_dense_prefetch = (
        config.encoder.architecture == "dense_decoder"
        and config.decoder.candidate_features == "discard_v1"
    )
    is_v2_prefetch = (
        config.encoder.architecture == "structured_dual_v2"
        and getattr(config.decoder, "candidate_features", "none") in ("none", "structured_v2")
    )
    if config.candidate_feature_workers and (
        not (is_dense_prefetch or is_v2_prefetch)
        or config.num_workers != 0
        or world_size != 1
    ):
        raise ValueError(
            "Ordered candidate prefetch requires single-process discard_v1/structured_v2 BC with num_workers=0."
        )

    output_prefix = Path(HydraConfig.get().runtime.output_dir)

    if rank == 0:
        logging.info("Model type: behavioral cloning (BC)")

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

        _config.encoder.dump(config.encoder)

        _config.decoder.dump(config.decoder)

        if config.initial_model_prefix is not None:
            logging.info(
                "Initial model prefix: %s", config.initial_model_prefix
            )
            logging.info("Initlal model index: %d", config.initial_model_index)
            if config.optimizer.initialize:
                logging.info("(Will not load optimizer)")

        logging.info("Checkpointing: %s", config.checkpointing)
        logging.info("Batch size: %d", config.batch_size)
        logging.info(
            "# of steps for gradient accumulation: %d",
            config.gradient_accumulation_steps,
        )
        logging.info(
            "Norm threshold for gradient clipping: %E",
            config.max_gradient_norm,
        )

        _config.optimizer.dump(config.optimizer)

        if config.initial_model_prefix is not None:
            logging.info("Initial encoder snapshot: %s", encoder_snapshot_path)
            logging.info("Initial decoder snapshot: %s", decoder_snapshot_path)
            if optimizer_snapshot_path is not None:
                logging.info(
                    "Initial optimizer snapshot: %s", optimizer_snapshot_path
                )
            if scheduler_snapshot_path is not None:
                logging.info(
                    "Initial scheduler snapshot: %s", scheduler_snapshot_path
                )

        logging.info("Output prefix: %s", output_prefix)
        if config.snapshot_interval == 0:
            logging.info("Snapshot interval: N/A")
        else:
            logging.info("Snapshot interval: %d", config.snapshot_interval)

    if world_size >= 2:
        init_process_group(backend="nccl" if device.type == "cuda" else "gloo")

    decision_recipe = decision_checkpoint.recipe(
        config, "bc", dtype=dtype, amp_dtype=amp_dtype
    )
    initial_rng_state = decision_checkpoint.validate_resume(config, decision_recipe)
    use_v2_prefetch = (
        config.encoder.architecture == "structured_dual_v2"
        and config.candidate_feature_workers > 0
    )
    head_extra = (
        {"head_type": getattr(config.encoder, "head_type", "factorized")}
        if config.encoder.architecture == "structured_dual_v2"
        else {}
    )
    encoder = make_backbone(
        config.encoder, checkpointing=config.checkpointing, dtype=dtype
    )
    encoder_tdm = TensorDictModule(
        encoder,
        in_keys=backbone_in_keys(config.encoder, precomputed=use_v2_prefetch),  # type: ignore
        out_keys=["encode"],  # type: ignore
    )
    decoder = make_candidate_head(
        config.decoder,
        input_dimension=config.encoder.dimension,
        architecture=config.encoder.architecture,
        dtype=dtype,
        **head_extra,
    )
    decoder.zero_output()
    decoder_tdm = TensorDictModule(
        decoder,
        in_keys=candidate_head_keys(
            config.encoder.architecture, config.decoder,
            precomputed=config.candidate_feature_workers > 0,
        ),
        out_keys=["decode"],  # type: ignore
    )
    decode_converter = DecodeConverter("log_probs")
    decode_converter_tdm = TensorDictModule(
        decode_converter,
        in_keys=["candidates", "decode"],  # type: ignore
        out_keys=["log_probs"],  # type: ignore
    )
    network_tdm = TensorDictSequential(
        [encoder_tdm, decoder_tdm, decode_converter_tdm]
    )
    if world_size >= 2:
        network_tdm.to(device=device)
        for _param in network_tdm.parameters():
            broadcast(_param.data, src=0)
        network_tdm.to(device="cpu")

    network_tdm.requires_grad_(True)
    network_tdm.train()
    network_tdm = network_tdm.to(device=device, dtype=dtype)
    if world_size >= 2:
        network_tdm = nn.SyncBatchNorm.convert_sync_batchnorm(network_tdm)
        network_tdm = DistributedDataParallel(network_tdm)
        assert isinstance(network_tdm, nn.Module)

    argmax_layer = DecodeConverter(
        "argmax",
        tie_break_by_action_id=config.encoder.architecture in ("dense_decoder", "structured_dual_v2"),
    )
    argmax_tdm = TensorDictModule(
        argmax_layer,
        in_keys=["candidates", "log_probs"],  # type: ignore
        out_keys=["action"],  # type: ignore
    )
    network_tdm_to_save = TensorDictSequential(
        [encoder_tdm, decoder_tdm, decode_converter_tdm, argmax_tdm]
    )

    network_tdm_to_save.requires_grad_(True)
    network_tdm_to_save.train()
    network_tdm_to_save = network_tdm_to_save.to(device=device, dtype=dtype)

    optimizer, scheduler = _config.optimizer.create(
        device.type, config.optimizer, network_tdm
    )

    if config.encoder.load_from is not None:
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        encoder_state_dict = torch.load(
            config.encoder.load_from, map_location="cpu", weights_only=True
        )
        encoder.load_state_dict(encoder_state_dict)
        encoder.to(device=device, dtype=dtype)

    if config.decoder.load_from is not None:
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        decoder_state_dict = torch.load(
            config.decoder.load_from, map_location="cpu", weights_only=True
        )
        load_candidate_head_state(decoder, decoder_state_dict)
        decoder.to(device=device, dtype=dtype)

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None

        assert encoder_snapshot_path is not None
        encoder_state_dict = torch.load(
            encoder_snapshot_path, map_location="cpu", weights_only=True
        )
        encoder.load_state_dict(encoder_state_dict)
        encoder.to(device=device, dtype=dtype)

        assert decoder_snapshot_path is not None
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

        if scheduler_snapshot_path is not None:
            assert scheduler is not None
            scheduler_state_dict = torch.load(
                scheduler_snapshot_path, map_location="cpu", weights_only=True
            )
            scheduler.load_state_dict(scheduler_state_dict)

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
                snapshots_path / f"scheduler{infix}.pth",
            )

        network_tdm_state = dump_model(
            network_tdm_to_save,
            [
                dump_object(
                    encoder_tdm,
                    [
                        dump_object(
                            encoder,
                            [],
                            backbone_kwargs(
                                config.encoder, checkpointing=config.checkpointing, dtype=dtype
                            ),
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
                    decoder_tdm,
                    [
                        dump_object(
                            decoder,
                            [],
                            candidate_head_kwargs(
                                config.decoder, input_dimension=config.encoder.dimension,
                                architecture=config.encoder.architecture, dtype=dtype,
                                **head_extra,
                            ),
                        )
                    ],
                    {
                        "in_keys": candidate_head_keys(config.encoder.architecture, config.decoder),
                        "out_keys": ["decode"],
                    },
                ),
                dump_object(
                    decode_converter_tdm,
                    [dump_object(decode_converter, ["log_probs"], {})],
                    {
                        "in_keys": ["candidates", "decode"],
                        "out_keys": ["log_probs"],
                    },
                ),
                dump_object(
                    argmax_tdm,
                    [dump_object(argmax_layer, ["argmax"], {
                        "tie_break_by_action_id": config.encoder.architecture in ("dense_decoder", "structured_dual_v2"),
                    })],
                    {
                        "in_keys": ["candidates", "log_probs"],
                        "out_keys": ["action"],
                    },
                ),
            ],
            {},
        )
        torch.save(
            network_tdm_state, snapshots_path / f"model{infix}.kanachan"
        )

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
        decision_checkpoint.save(
            snapshots_path, infix, decision_recipe, algorithm="bc"
        )

    tensorboard_path = output_prefix / "tensorboard"
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=tensorboard_path) as summary_writer, ExitStack() as loader_resources:
        if rank == 0:
            summary_writer.add_text(
                "architecture/summary",
                json.dumps(
                    {
                        "training_mode": "single_step",
                        "architecture": str(config.encoder.architecture),
                        "preset": getattr(config.encoder, "preset", None),
                    },
                    sort_keys=True,
                ),
                0,
            )
        data_loader_factory = None
        if config.candidate_feature_workers:
            from kanachan.training.core.bc.prefetch import CandidateFeaturePrefetch

            prefetch_mode = (
                "structured_v2"
                if config.encoder.architecture == "structured_dual_v2"
                else "discard_v1"
            )
            early_hint = getattr(config.encoder, "enable_early_shape_hint", False)

            def data_loader_factory(**arguments):
                return loader_resources.enter_context(CandidateFeaturePrefetch(
                    DataLoader(**arguments, raw_batches=True), workers=config.candidate_feature_workers,
                    batches=config.candidate_prefetch_batches,
                    pin_memory=device.type == "cuda",
                    raw_batches=True, rewrite_rooms=arguments["rewrite_rooms"],
                    rewrite_grades=arguments["rewrite_grades"],
                    mode=prefetch_mode,
                    enable_early_shape_hint=early_hint,
                ))

        torch.autograd.set_detect_anomaly(
            False  # `True` for debbing purpose only.
        )
        _training(
            device=device,
            dtype=dtype,
            amp_dtype=amp_dtype,
            training_data=config.training_data,
            num_workers=config.num_workers,
            rewrite_rooms=config.rewrite_rooms,
            rewrite_grades=config.rewrite_grades,
            num_samples=num_samples,
            initial_optimizer_steps=initial_optimizer_steps,
            initial_grad_scaler_state=initial_grad_scaler_state,
            initial_rng_state=initial_rng_state,
            data_loader_factory=data_loader_factory,
            network_tdm=network_tdm,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_gradient_norm=config.max_gradient_norm,
            optimizer=optimizer,
            scheduler=scheduler,
            snapshot_interval=config.snapshot_interval,
            summary_writer=summary_writer,
            snapshot_writer=snapshot_writer,
        )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    _main()
    sys.exit(0)
