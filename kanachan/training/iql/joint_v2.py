"""Stage-2 Shared-Backbone Joint IQL + AWR Training & Checkpoint Recovery (§4.4B)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
from tensordict import TensorDict
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel

from kanachan.nn.structured_dual_decoder import (
    ARCHITECTURE_ID,
    structured_v2_schema,
)
from kanachan.nn.structured_iql import JointStructuredIQLModule
from kanachan.structured_v2 import KANACHAN_ACTION_MANIFEST_SHA256
from kanachan.training.grp import FrozenGRP
from kanachan.training.online.state import atomic_save, capture_rng, restore_rng


def build_joint_iql_recipe(
    module: JointStructuredIQLModule,
    *,
    base_lr: float,
    weight_decay: float,
    decoder1_lr_scale: float,
    grp: FrozenGRP | None = None,
    data_sha256: str = "",
) -> dict[str, Any]:
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
        "iql": {
            "expectile": module.expectile,
            "discount_factor": module.discount_factor,
            "awr_inv_temp_beta": module.awr_inv_temp_beta,
            "awr_max_weight": module.awr_max_weight,
            "kl_coef": module.kl_coef,
            "target_ema_tau": module.target_ema_tau,
            "final_rank_weight": module.final_rank_weight,
            "freeze_decoder1": module.freeze_decoder1,
            "freeze_shared_tile_emb": module.freeze_shared_tile_emb,
            "decoder1_lr_scale": float(decoder1_lr_scale),
            "use_cql_quantile_head": module.quantile_q_head is not None,
        },
        "grp_sha256": grp.grp_sha256 if grp is not None else None,
        "ruleset_sha256": grp.ruleset_sha256 if grp is not None else None,
        "data_sha256": data_sha256,
        "base_lr": float(base_lr),
        "weight_decay": float(weight_decay),
    }


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
) -> Path:
    raw = module.module if isinstance(module, DistributedDataParallel) else module
    assert isinstance(raw, JointStructuredIQLModule)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "cursor": int(cursor),
        "recipe": recipe,
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler": grad_scaler.state_dict() if grad_scaler is not None and grad_scaler.is_enabled() else None,
        "rng": capture_rng(),
    }
    ckpt_path = ckpt_dir / f"joint-iql-{step}.pt"
    atomic_save(ckpt_path, payload)
    raw.dump_actor(ckpt_dir / f"model.{step}.kanachan")
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
) -> tuple[int, int]:
    """Fail-closed restore preserving Target Q Polyak lag + DDP Rank-0 Target sync."""
    payload = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    if payload.get("recipe") != expected_recipe:
        raise ValueError("Joint IQL resume recipe/manifest/GRP differs (Fail-Closed).")
    raw = module.module if isinstance(module, DistributedDataParallel) else module
    assert isinstance(raw, JointStructuredIQLModule)
    raw.load_state_dict(payload["model"], strict=True)
    # Preserve Target Q EMA lag across DDP ranks!
    raw.sync_targets_ddp()
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if grad_scaler is not None and grad_scaler.is_enabled() and payload.get("grad_scaler") is not None:
        grad_scaler.load_state_dict(payload["grad_scaler"])
    if payload.get("rng") is not None:
        restore_rng(payload["rng"])
    return int(payload["step"]), int(payload["cursor"])


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
) -> dict[str, Any]:
    """Train JointStructuredIQLModule over ordered TensorDict transition batches."""
    raw = module.module if isinstance(module, DistributedDataParallel) else module
    assert isinstance(raw, JointStructuredIQLModule)
    raw.sync_targets_ddp()

    start_step = 0
    cursor = 0
    if resume_from is not None:
        start_step, cursor = load_joint_iql_checkpoint(
            resume_from,
            raw,
            optimizer,
            expected_recipe=recipe,
            scheduler=scheduler,
            grad_scaler=grad_scaler,
            map_location=device,
        )

    is_amp = amp_dtype is not None and device.type == "cuda" and amp_dtype != torch.float32
    scaler = grad_scaler or GradScaler("cuda", enabled=(is_amp and amp_dtype == torch.float16))
    runner = ddp_wrapped if ddp_wrapped is not None else module
    runner.train()

    step = start_step
    last_metrics: dict[str, float] = {}
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

        if not math.isfinite(float(loss.item())):
            raise RuntimeError("Joint Structured IQL loss became non-finite.")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        trainable = [p for p in runner.parameters() if p.requires_grad]
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        old_scale = scaler.get_scale() if scaler.is_enabled() else None
        scaler.step(optimizer)
        scaler.update()
        step_skipped = old_scale is not None and scaler.get_scale() < old_scale
        if not step_skipped:
            if scheduler is not None:
                scheduler.step()
            raw.update_targets()

        step += 1
        cursor = idx + 1
        last_metrics = {k: float(v.item()) for k, v in out.items()}
        if ckpt_dir is not None and checkpoint_every > 0 and step % checkpoint_every == 0:
            save_joint_iql_checkpoint(
                ckpt_dir, raw, optimizer, step=step, cursor=cursor,
                recipe=recipe, scheduler=scheduler, grad_scaler=scaler,
            )

    return {"step": step, "cursor": cursor, "metrics": last_metrics}


__all__ = [
    "build_joint_iql_recipe",
    "load_joint_iql_checkpoint",
    "save_joint_iql_checkpoint",
    "train_joint_iql_batches",
]
