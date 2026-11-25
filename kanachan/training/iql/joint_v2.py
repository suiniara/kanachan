"""Stage-2 Shared-Backbone Joint IQL + AWR Training & Checkpoint Recovery (§4.4B)."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel

from kanachan.constants import MAX_NUM_ACTION_CANDIDATES, NUM_TYPES_OF_ACTIONS
from kanachan.nn.structured_dual_decoder import (
    ARCHITECTURE_ID,
    structured_v2_schema,
)
from kanachan.nn.structured_iql import JointStructuredIQLModule
from kanachan.structured_v2 import (
    CAND_DYN_DIM,
    EVENT_BYTES,
    GLOBAL_FEATS_DIM,
    KANACHAN_ACTION_MANIFEST_SHA256,
    PAD_ACTION_ID,
    PLAYER_FEATS_DIM,
    TILE_FEATS_DIM,
    derive_structured_v2_batch,
)
from kanachan.training.core.optimizers import move_optimizer_state
from kanachan.training.grp import FrozenGRP
from kanachan.training.online.state import atomic_save, capture_rng, cpu_state, restore_rng


def derive_next_structured_v2_batch(
    next_obs: TensorDict | dict[str, Tensor],
    done: Tensor,
    *,
    enable_early_shape_hint: bool = False,
) -> dict[str, Tensor]:
    """Precompute CPU ``v2_*`` features for RL ``next`` observations safely across terminal rows."""
    done_cpu = done.detach().to(device="cpu", dtype=torch.bool)
    non_term = ~done_cpu
    bsz = int(done_cpu.shape[0])
    lmax = 1
    sub_feats: dict[str, Tensor] | None = None
    active_idx: Tensor | None = None

    if bool(non_term.any().item()):
        active_idx = non_term.nonzero(as_tuple=False).squeeze(-1)
        sub_feats = derive_structured_v2_batch(
            next_obs["sparse"][active_idx],
            next_obs["numeric"][active_idx],
            next_obs["progression"][active_idx],
            next_obs["candidates"][active_idx],
            enable_early_shape_hint=enable_early_shape_hint,
            include_room_level=False,
        )
        lmax = max(1, int(sub_feats["v2_events"].shape[1]))

    cand_ids = torch.full((bsz, MAX_NUM_ACTION_CANDIDATES), PAD_ACTION_ID, dtype=torch.int32)
    cand_ids[:, 0] = 0
    cand_mask = torch.zeros((bsz, MAX_NUM_ACTION_CANDIDATES), dtype=torch.bool)
    cand_mask[:, 0] = True
    out: dict[str, Tensor] = {
        "v2_tile_feats": torch.zeros((bsz, 34, TILE_FEATS_DIM), dtype=torch.float32),
        "v2_events": torch.zeros((bsz, lmax, EVENT_BYTES), dtype=torch.uint8),
        "v2_event_len": torch.ones((bsz,), dtype=torch.int32),
        "v2_global_feats": torch.zeros((bsz, GLOBAL_FEATS_DIM), dtype=torch.float32),
        "v2_player_feats": torch.zeros((bsz, 4, PLAYER_FEATS_DIM), dtype=torch.float32),
        "v2_cand_ids": cand_ids,
        "v2_cand_mask": cand_mask,
        "v2_cand_mode": torch.zeros((bsz, MAX_NUM_ACTION_CANDIDATES), dtype=torch.uint8),
        "v2_cand_dyn": torch.zeros((bsz, MAX_NUM_ACTION_CANDIDATES, CAND_DYN_DIM), dtype=torch.float32),
    }
    if sub_feats is not None and active_idx is not None:
        for key, val in sub_feats.items():
            out[key][active_idx] = val
    return out


def attach_structured_v2_transition_features(
    batch: TensorDict,
    *,
    enable_early_shape_hint: bool = False,
) -> TensorDict:
    """Attach CPU-precomputed ``v2_*`` features to both ``batch`` and ``batch['next']``."""
    out = batch.clone(recurse=False)
    if "v2_tile_feats" not in out:
        cur_feats = derive_structured_v2_batch(
            out["sparse"],
            out["numeric"],
            out["progression"],
            out["candidates"],
            enable_early_shape_hint=enable_early_shape_hint,
            include_room_level=False,
        )
        for key, val in cur_feats.items():
            out[key] = val
    if "next" in out and "v2_tile_feats" not in out["next"]:
        nxt_feats = derive_next_structured_v2_batch(
            out["next"],
            out["next", "done"],
            enable_early_shape_hint=enable_early_shape_hint,
        )
        for key, val in nxt_feats.items():
            out["next", key] = val
    return out


def build_joint_iql_recipe(
    module: JointStructuredIQLModule,
    *,
    base_lr: float,
    weight_decay: float,
    decoder1_lr_scale: float,
    grp: FrozenGRP | None = None,
    data_sha256: str = "",
) -> dict[str, Any]:
    iql_cfg: dict[str, Any] = {
        "expectile": module.expectile,
        "discount_factor": module.discount_factor,
        "awr_inv_temp_beta": module.awr_inv_temp_beta,
        "awr_max_weight": module.awr_max_weight,
        "kl_coef": module.kl_coef,
        "keep_bc_anchor": module.bc_anchor_backbone is not None,
        "target_ema_tau": module.target_ema_tau,
        "final_rank_weight": module.final_rank_weight,
        "freeze_decoder1": module.freeze_decoder1,
        "freeze_shared_tile_emb": module.freeze_shared_tile_emb,
        "decoder1_lr_scale": float(decoder1_lr_scale),
        "use_cql_quantile_head": module.quantile_q_head is not None,
    }
    if module.quantile_q_head is not None:
        iql_cfg["num_quantiles"] = int(module.quantile_q_head.num_quantiles)
        iql_cfg["qr_double_q"] = bool(module.qr_double_q)
        iql_cfg["quantile_cql_alpha"] = float(module.quantile_cql_alpha)
        iql_cfg["cql_temperature"] = float(module.cql_temperature)
    return {
        "format_version": 1,
        "architecture": ARCHITECTURE_ID,
        "manifest_sha256": KANACHAN_ACTION_MANIFEST_SHA256,
        "schema": structured_v2_schema(),
        "backbone": {
            "preset": module.backbone.preset,
            "dimension": module.backbone.dimension,
            "num_heads": module.backbone.num_heads,
            "dim_feedforward": module.backbone.dim_feedforward,
            "n_d1_layers": module.backbone.n_d1_layers,
            "n_d2_layers": module.backbone.n_d2_layers,
            "use_rel_bias": module.backbone.use_rel_bias,
            "cand_self_attn": module.backbone.cand_self_attn,
            "head_type": module.backbone.head_type,
            "enable_early_shape_hint": module.backbone.enable_early_shape_hint,
            "include_room_level": False,
            "rope_base": module.backbone.rope_base,
        },
        "iql": iql_cfg,
        "grp_sha256": grp.grp_sha256 if grp is not None else None,
        "ruleset_sha256": grp.ruleset_sha256 if grp is not None else None,
        "data_sha256": data_sha256,
        "base_lr": float(base_lr),
        "weight_decay": float(weight_decay),
    }


def build_joint_iql_optimizer(
    module: JointStructuredIQLModule,
    *,
    base_lr: float,
    weight_decay: float,
    decoder1_lr_scale: float = 0.25,
    device: torch.device | None = None,
    fused: bool | None = None,
) -> torch.optim.AdamW:
    """Construct AdamW with parameter decay/lr grouping and fused CUDA guard support."""
    dev = device or next(module.parameters()).device
    use_fused = (dev.type == "cuda") if fused is None else bool(fused)
    groups = module.parameter_groups(
        base_lr=base_lr,
        weight_decay=weight_decay,
        decoder1_lr_scale=decoder1_lr_scale,
    )
    return torch.optim.AdamW(groups, fused=use_fused)


def save_joint_iql_checkpoint(
    ckpt_dir: Path,
    module: JointStructuredIQLModule,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    cursor: int,
    recipe: dict[str, Any],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_scaler: GradScaler | None = None,
    numerical_stats: dict[str, int] | None = None,
) -> Path:
    """Rank-0 exclusive checkpoint/actor write with per-rank RNG gather and error broadcast."""
    raw = module.module if isinstance(module, DistributedDataParallel) else module
    assert isinstance(raw, JointStructuredIQLModule)
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1

    local_rng = capture_rng()
    if is_dist:
        rng_by_rank: list[Any] = [None] * world_size
        dist.all_gather_object(rng_by_rank, local_rng)
    else:
        rng_by_rank = [local_rng]

    ckpt_path = ckpt_dir / f"joint-iql-{step}.pt"
    actor_path = ckpt_dir / f"model.{step}.kanachan"
    err_holder: list[str | None] = [None]

    if rank == 0:
        try:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            stats_clean = {str(k): int(v) for k, v in (numerical_stats or {}).items()}
            payload = {
                "step": int(step),
                "cursor": int(cursor),
                "world_size": int(world_size),
                "recipe": recipe,
                "numerical_stats": stats_clean,
                "model": cpu_state(raw.state_dict()),
                "optimizer": cpu_state(optimizer.state_dict()),
                "scheduler": cpu_state(scheduler.state_dict()) if scheduler is not None else None,
                "grad_scaler": cpu_state(grad_scaler.state_dict())
                if grad_scaler is not None and grad_scaler.is_enabled()
                else None,
                "rng": rng_by_rank[0],
                "rng_by_rank": rng_by_rank,
            }
            atomic_save(ckpt_path, payload)
            raw.dump_actor(actor_path)
        except BaseException as exc:  # noqa: BLE001
            err_holder[0] = f"{type(exc).__name__}: {exc}"

    if is_dist:
        # Synchronizes all ranks after Rank-0 write AND propagates failures without hanging.
        dist.broadcast_object_list(err_holder, src=0)
    if err_holder[0] is not None:
        raise RuntimeError(f"Rank-0 checkpoint save failed: {err_holder[0]}")
    return ckpt_path


def load_joint_iql_checkpoint(
    ckpt_path: Path,
    module: JointStructuredIQLModule,
    optimizer: torch.optim.Optimizer,
    *,
    expected_recipe: dict[str, Any],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_scaler: GradScaler | None = None,
    map_location: torch.device | str = "cpu",
    stats_out: dict[str, int] | None = None,
) -> tuple[int, int]:
    """Fail-closed restore preserving Target Q Polyak lag, DDP sync, and per-rank RNG."""
    del map_location  # Load on CPU first, then place optimizer state beside module parameters.
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if payload.get("recipe") != expected_recipe:
        raise ValueError("Joint IQL resume recipe/manifest/GRP differs (Fail-Closed).")
    raw = module.module if isinstance(module, DistributedDataParallel) else module
    assert isinstance(raw, JointStructuredIQLModule)
    raw.load_state_dict(payload["model"], strict=True)
    # Preserve Target Q EMA lag across DDP ranks!
    raw.sync_targets_ddp()
    optimizer.load_state_dict(payload["optimizer"])
    move_optimizer_state(optimizer)
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if grad_scaler is not None and grad_scaler.is_enabled() and payload.get("grad_scaler") is not None:
        grad_scaler.load_state_dict(payload["grad_scaler"])

    if stats_out is not None:
        stats_out.clear()
        for k, v in (payload.get("numerical_stats") or {}).items():
            stats_out[str(k)] = int(v)

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1
    if payload.get("rng_by_rank") is not None:
        rng_list = payload["rng_by_rank"]
        if len(rng_list) != world_size:
            raise ValueError(
                f"Checkpoint world_size={len(rng_list)} does not match current world_size={world_size}."
            )
        restore_rng(rng_list[rank])
    elif payload.get("rng") is not None:
        if world_size > 1:
            raise ValueError("Multi-rank DDP resume requires per-rank 'rng_by_rank' state.")
        restore_rng(payload["rng"])
    return int(payload["step"]), int(payload["cursor"])


def _compute_grad_norm_and_nonfinite(
    grads: list[Tensor],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Compute scalar FP32 total L2 gradient norm and non-finite flag with zero host sync."""
    if not grads:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return zero, torch.zeros((), device=device, dtype=torch.bool)
    per_norms = [torch.linalg.vector_norm(g, ord=2, dtype=torch.float32) for g in grads]
    total_norm = torch.linalg.vector_norm(torch.stack(per_norms), ord=2)
    return total_norm, ~torch.isfinite(total_norm)


def train_joint_iql_batches(
    module: JointStructuredIQLModule,
    batches: Sequence[TensorDict],
    optimizer: torch.optim.Optimizer,
    *,
    recipe: dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    max_grad_norm: float = 1.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_scaler: GradScaler | None = None,
    ckpt_dir: Path | None = None,
    checkpoint_every: int = 0,
    resume_from: Path | None = None,
    ddp_wrapped: DistributedDataParallel | None = None,
    log_interval: int = 50,
    check_finite_interval: int = 50,
    max_nonfinite_steps: int | None = None,
    loss_transform: Callable[[Tensor, int], Tensor] | None = None,
    post_backward_hook: Callable[[torch.nn.Module, torch.optim.Optimizer, int], None] | None = None,
) -> dict[str, Any]:
    """Train JointStructuredIQLModule with strict pre-update numerical safety guards."""
    raw = module.module if isinstance(module, DistributedDataParallel) else module
    assert isinstance(raw, JointStructuredIQLModule)
    raw.sync_targets_ddp()

    is_amp = amp_dtype is not None and device.type == "cuda" and amp_dtype != torch.float32
    scaler = grad_scaler or GradScaler("cuda", enabled=(is_amp and amp_dtype == torch.float16))

    start_step = 0
    cursor = 0
    stats_restored: dict[str, int] = {}
    if resume_from is not None:
        start_step, cursor = load_joint_iql_checkpoint(
            resume_from,
            raw,
            optimizer,
            expected_recipe=recipe,
            scheduler=scheduler,
            grad_scaler=scaler,
            map_location=device,
            stats_out=stats_restored,
        )

    runner = ddp_wrapped if ddp_wrapped is not None else module
    runner.train()

    is_dist = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_dist else 1
    can_gpu_fused_guard = (
        device.type == "cuda"
        and bool(optimizer.param_groups)
        and all(bool(g.get("fused", False)) for g in optimizer.param_groups)
        and isinstance(optimizer, torch.optim.AdamW)
        and scheduler is None
    )
    one_f32 = torch.ones((), device=device, dtype=torch.float32)
    zero_f32 = torch.zeros((), device=device, dtype=torch.float32)
    clip_f32 = torch.full((), float(max_grad_norm), device=device, dtype=torch.float32)

    optimizer_steps_t = torch.full(
        (), int(stats_restored.get("optimizer_steps", 0)), device=device, dtype=torch.int64
    )
    skipped_steps_t = torch.full(
        (), int(stats_restored.get("skipped_steps", 0)), device=device, dtype=torch.int64
    )
    nonfinite_loss_steps_t = torch.full(
        (), int(stats_restored.get("nonfinite_loss_steps", 0)), device=device, dtype=torch.int64
    )
    nonfinite_grad_steps_t = torch.full(
        (), int(stats_restored.get("nonfinite_grad_steps", 0)), device=device, dtype=torch.int64
    )

    step = start_step
    last_metrics: dict[str, float] = {}
    last_stats: dict[str, int] = {
        "optimizer_steps": int(stats_restored.get("optimizer_steps", 0)),
        "skipped_steps": int(stats_restored.get("skipped_steps", 0)),
        "nonfinite_loss_steps": int(stats_restored.get("nonfinite_loss_steps", 0)),
        "nonfinite_grad_steps": int(stats_restored.get("nonfinite_grad_steps", 0)),
    }
    last_out: dict[str, Tensor] | None = None

    def _flush_metrics(out_map: dict[str, Tensor]) -> tuple[dict[str, float], dict[str, int]]:
        keys = list(out_map.keys())
        packed = torch.stack(
            [
                optimizer_steps_t.double(),
                skipped_steps_t.double(),
                nonfinite_loss_steps_t.double(),
                nonfinite_grad_steps_t.double(),
            ]
            + [out_map[k].detach().double() for k in keys]
        ).cpu().tolist()
        stats = {
            "optimizer_steps": int(round(packed[0])),
            "skipped_steps": int(round(packed[1])),
            "nonfinite_loss_steps": int(round(packed[2])),
            "nonfinite_grad_steps": int(round(packed[3])),
        }
        metrics_f = {k: float(v) for k, v in zip(keys, packed[4:], strict=True)}
        if max_nonfinite_steps is not None and stats["skipped_steps"] > int(max_nonfinite_steps):
            raise RuntimeError(
                f"Exceeded max_nonfinite_steps={max_nonfinite_steps}: skipped={stats['skipped_steps']} "
                f"(nonfinite_loss={stats['nonfinite_loss_steps']}, "
                f"nonfinite_grad={stats['nonfinite_grad_steps']})."
            )
        return metrics_f, stats

    for idx in range(cursor, len(batches)):
        batch = batches[idx].to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=bool(is_amp),
            cache_enabled=False,
        ):
            out = runner(batch, amp_dtype=amp_dtype)
            loss = out["loss"]
            if loss_transform is not None:
                loss = loss_transform(loss, step)

        loss_bad = (~torch.isfinite(loss.detach())).reshape(())
        if loss.requires_grad:
            scaler.scale(loss).backward()
        if post_backward_hook is not None:
            post_backward_hook(runner, optimizer, step)
        if scaler.is_enabled():
            scaler.unscale_(optimizer)

        params_with_grad: list[Tensor] = []
        grads: list[Tensor] = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

        grad_norm, grad_bad = _compute_grad_norm_and_nonfinite(grads, device=device)
        if scaler.is_enabled():
            opt_state = scaler._per_optimizer_states[id(optimizer)]
            found_map = opt_state["found_inf_per_device"]
            scaler_bad = zero_f32 > 1.0
            for v in found_map.values():
                scaler_bad = scaler_bad | (v.to(device=device, dtype=torch.float32) > 0.0)
            grad_bad = grad_bad | scaler_bad
        step_bad = loss_bad | grad_bad

        if is_dist and world_size > 1:
            sync_dev = torch.device("cpu") if dist.get_backend() == "gloo" else device
            flags_sync = torch.stack([loss_bad, grad_bad]).to(device=sync_dev, dtype=torch.int32)
            dist.all_reduce(flags_sync, op=dist.ReduceOp.MAX)
            flags_dev = (flags_sync > 0).to(device=device)
            loss_bad = flags_dev[0]
            grad_bad = flags_dev[1]
            step_bad = loss_bad | grad_bad

        if params_with_grad and max_grad_norm > 0.0:
            safe_norm = torch.where(step_bad, clip_f32, grad_norm)
            torch.nn.utils.clip_grads_with_norm_(
                params_with_grad, float(max_grad_norm), safe_norm
            )

        found_inf_f32 = step_bad.to(dtype=torch.float32)
        if scaler.is_enabled():
            opt_state = scaler._per_optimizer_states[id(optimizer)]
            found_map = opt_state["found_inf_per_device"]
            if found_map:
                for d_key in list(found_map.keys()):
                    cur = found_map[d_key]
                    found_map[d_key] = torch.maximum(
                        cur, found_inf_f32.to(device=cur.device, dtype=cur.dtype)
                    )
            else:
                found_map[device] = found_inf_f32

        if can_gpu_fused_guard:
            if scaler.is_enabled():
                optimizer.grad_scale = one_f32  # type: ignore[attr-defined]
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.grad_scale = one_f32  # type: ignore[attr-defined]
                optimizer.found_inf = found_inf_f32  # type: ignore[attr-defined]
                try:
                    optimizer.step()
                finally:
                    if hasattr(optimizer, "grad_scale"):
                        del optimizer.grad_scale  # type: ignore[attr-defined]
                    if hasattr(optimizer, "found_inf"):
                        del optimizer.found_inf  # type: ignore[attr-defined]
            raw.update_targets(valid=~step_bad)
        else:
            bad_host = bool(step_bad.item())
            if not bad_host:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                raw.update_targets()
            elif scaler.is_enabled():
                scaler.update()

        optimizer.zero_grad(set_to_none=True)

        step += 1
        cursor = idx + 1
        last_out = out
        step_bad_i = step_bad.to(dtype=torch.int64)
        skipped_steps_t.add_(step_bad_i)
        optimizer_steps_t.add_(1 - step_bad_i)
        nonfinite_loss_steps_t.add_(loss_bad.to(dtype=torch.int64))
        nonfinite_grad_steps_t.add_(grad_bad.to(dtype=torch.int64))

        need_ckpt = ckpt_dir is not None and checkpoint_every > 0 and step % checkpoint_every == 0
        need_sync = (
            device.type == "cpu"
            or idx == len(batches) - 1
            or need_ckpt
            or (check_finite_interval > 0 and step % check_finite_interval == 0)
            or (log_interval > 0 and step % log_interval == 0)
        )
        if need_sync:
            last_metrics, last_stats = _flush_metrics(out)
        if need_ckpt:
            assert ckpt_dir is not None
            save_joint_iql_checkpoint(
                ckpt_dir,
                raw,
                optimizer,
                step=step,
                cursor=cursor,
                recipe=recipe,
                scheduler=scheduler,
                grad_scaler=scaler,
                numerical_stats=last_stats,
            )

    if last_out is not None and not last_metrics:
        last_metrics, last_stats = _flush_metrics(last_out)

    return {
        "step": step,
        "optimizer_steps": last_stats["optimizer_steps"],
        "skipped_steps": last_stats["skipped_steps"],
        "nonfinite_loss_steps": last_stats["nonfinite_loss_steps"],
        "nonfinite_grad_steps": last_stats["nonfinite_grad_steps"],
        "cursor": cursor,
        "metrics": last_metrics,
    }


__all__ = [
    "attach_structured_v2_transition_features",
    "build_joint_iql_optimizer",
    "build_joint_iql_recipe",
    "derive_next_structured_v2_batch",
    "load_joint_iql_checkpoint",
    "save_joint_iql_checkpoint",
    "train_joint_iql_batches",
]
