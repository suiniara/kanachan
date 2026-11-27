"""Stage-2 Shared-Backbone Joint IQL + AWR Training & Checkpoint Recovery (§4.4B)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch import Tensor
from torch.amp.grad_scaler import GradScaler, OptState
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
from kanachan.training.core.optimizers import (
    OptimizerCoverageError,
    consolidate_optimizer_state,
    dummy_ddp_zero_loss,
    move_optimizer_state,
    sync_ddp_step_status,
    validate_optimizer_parameter_coverage,
)
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
    source_data_sha256: str = "",
    preprocess_recipe_sha256: str = "",
    training_config: Mapping[str, Any] | None = None,
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
    recipe_obj: dict[str, Any] = {
        "format_version": 2,
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
        "grp_rank_payoffs": list(grp.rank_payoffs) if grp is not None else None,
        "grp_calibration_report": dict(grp.calibration_report) if grp is not None else None,
        "data_sha256": data_sha256,
        "source_data_sha256": source_data_sha256,
        "preprocess_recipe_sha256": preprocess_recipe_sha256,
        "base_lr": float(base_lr),
        "weight_decay": float(weight_decay),
    }
    if training_config is not None:
        recipe_obj["training_config"] = dict(training_config)
    return recipe_obj


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
    opt = torch.optim.AdamW(groups, fused=use_fused)
    validate_optimizer_parameter_coverage(
        module,
        opt,
        inactive_param_ids=module.static_inactive_param_ids(),
        context="build_joint_iql_optimizer",
    )
    return opt


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
    stream_state: Any | None = None,
) -> Path:
    """Rank-0 exclusive checkpoint/actor write with per-rank RNG/stream gather and error broadcast."""
    raw = module.module if isinstance(module, DistributedDataParallel) else module
    assert isinstance(raw, JointStructuredIQLModule)
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1

    if is_dist and world_size > 1:
        consolidate_optimizer_state(optimizer)

    local_rng = capture_rng()
    if is_dist:
        rng_by_rank: list[Any] = [None] * world_size
        stream_by_rank: list[Any] = [None] * world_size
        dist.all_gather_object(rng_by_rank, local_rng)
        dist.all_gather_object(stream_by_rank, stream_state)
    else:
        rng_by_rank = [local_rng]
        stream_by_rank = [stream_state]

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
                "stream_state_by_rank": stream_by_rank
                if any(s is not None for s in stream_by_rank)
                else None,
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
    stream_state_out: list[Any] | None = None,
) -> tuple[int, int]:
    """Fail-closed restore preserving Target Q Polyak lag, DDP sync, and per-rank RNG/stream."""
    del map_location  # Load on CPU first, then place optimizer state beside module parameters.
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    saved_recipe = payload.get("recipe")
    if not isinstance(saved_recipe, dict):
        raise ValueError("Joint IQL checkpoint missing valid recipe dictionary (Fail-Closed).")
    if saved_recipe.get("source_data_sha256") != expected_recipe.get("source_data_sha256"):
        raise ValueError("Joint IQL resume source_data_sha256 differs (Fail-Closed).")
    if saved_recipe.get("preprocess_recipe_sha256") != expected_recipe.get("preprocess_recipe_sha256"):
        raise ValueError("Joint IQL resume preprocess_recipe_sha256 differs (Fail-Closed).")
    if saved_recipe != expected_recipe:
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
    if "world_size" in payload and int(payload["world_size"]) != world_size:
        raise ValueError(
            f"Checkpoint world_size={payload['world_size']} does not match current world_size={world_size}."
        )
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

    if stream_state_out is not None and payload.get("stream_state_by_rank") is not None:
        ss_list = payload["stream_state_by_rank"]
        if len(ss_list) != world_size:
            raise ValueError(
                f"Checkpoint world_size={len(ss_list)} does not match current world_size={world_size}."
            )
        stream_state_out[:] = [ss_list[rank]]
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


# ---------------------------------------------------------------------------
# PyTorch 2.6.0 GradScaler & Autograd Safety Helpers
# ---------------------------------------------------------------------------
# Relied-upon PyTorch 2.6.0 (torch/amp/grad_scaler.py & torch/optim/adamw.py) contract:
# 1. `scaler._per_optimizer_states` maps `id(optimizer)` ->
#    `{"stage": OptState, "found_inf_per_device": dict[torch.device, Tensor]}`.
# 2. `scaler.unscale_(optimizer)` transitions `stage` from `OptState.READY` to
#    `OptState.UNSCALED` and populates `found_inf_per_device` with float32 inf flags.
# 3. For optimizers with `_step_supports_amp_scaling == True` (CUDA `AdamW(fused=True)`),
#    `scaler.step(optimizer)` checks `stage == OptState.UNSCALED`, assigns
#    `optimizer.grad_scale` (preserving pre-set `one_f32`) and `optimizer.found_inf`
#    (sum of `found_inf_per_device`), executes `optimizer.step()`, sets
#    `stage = OptState.STEPPED`, and deletes both attributes afterward.
# 4. For non-fused optimizers (`_step_supports_amp_scaling` unset/False),
#    `scaler._maybe_opt_step` checks `sum(v.item() for v in found_inf_per_device.values())`
#    and skips `optimizer.step()` when non-zero; calling `scaler.update()` directly
#    on host-confirmed bad steps skips `optimizer.step()` uniformly across both paths.
# 5. `scaler.update()` reduces all `found_inf_per_device` tensors into `_scale` and
#    `_growth_tracker` via `torch._amp_update_scale_`, then resets `_per_optimizer_states`
#    back to `OptState.READY` (`defaultdict(_refresh_per_optimizer_state)`).


def expected_inactive_iql_param_ids(module: JointStructuredIQLModule) -> set[int]:
    """Return parameter IDs statically bypassed by preset architecture flags."""
    return module.static_inactive_param_ids()


def _validate_grad_scaler_optimizer_state(
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
    *,
    require_unscaled: bool = True,
) -> dict[torch.device, Tensor]:
    """Validate PyTorch 2.6.0 `GradScaler._per_optimizer_states` structure fail-closed."""
    per_states = getattr(scaler, "_per_optimizer_states", None)
    if not isinstance(per_states, Mapping):
        raise RuntimeError(
            "Unsupported GradScaler implementation: missing '_per_optimizer_states' mapping "
            f"(torch={torch.__version__})."
        )
    opt_key = id(optimizer)
    if opt_key not in per_states:
        raise RuntimeError(
            "GradScaler optimizer state missing for optimizer; ensure scaler.unscale_(optimizer) ran."
        )
    opt_state = per_states[opt_key]
    if not isinstance(opt_state, Mapping) or "found_inf_per_device" not in opt_state or "stage" not in opt_state:
        raise RuntimeError(
            "Unexpected GradScaler optimizer state layout; expected 'stage' and "
            f"'found_inf_per_device' keys, got {list(opt_state.keys()) if isinstance(opt_state, Mapping) else type(opt_state)}."
        )
    if require_unscaled and opt_state["stage"] is not OptState.UNSCALED:
        raise RuntimeError(
            f"GradScaler optimizer stage must be OptState.UNSCALED before step/update, got {opt_state['stage']!r}."
        )
    found_map = opt_state["found_inf_per_device"]
    if not isinstance(found_map, dict):
        raise RuntimeError(
            f"Expected 'found_inf_per_device' to be dict, got {type(found_map)}."
        )
    if require_unscaled:
        if not found_map:
            raise RuntimeError("GradScaler 'found_inf_per_device' is empty after unscale_.")
        for dev_key, flag_t in found_map.items():
            if not isinstance(flag_t, Tensor):
                raise RuntimeError(
                    f"GradScaler found_inf entry for {dev_key!r} is not a Tensor: {type(flag_t)}."
                )
    return found_map


def _grad_scaler_detect_nonfinite(
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    zero_f32: Tensor,
) -> Tensor:
    """Read per-device overflow flags from `GradScaler` with strict layout validation."""
    found_map = _validate_grad_scaler_optimizer_state(scaler, optimizer, require_unscaled=True)
    scaler_bad = zero_f32 > 1.0
    for val in found_map.values():
        val_f = val.to(device=device, dtype=torch.float32)
        scaler_bad = scaler_bad | (val_f > 0.0) | (~torch.isfinite(val_f))
    return scaler_bad


def _grad_scaler_inject_step_bad(
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
    *,
    step_bad: Tensor,
    device: torch.device,
) -> None:
    """Propagate tensorized loss/grad/DDP failure flags into `GradScaler` before step/update."""
    found_map = _validate_grad_scaler_optimizer_state(scaler, optimizer, require_unscaled=True)
    found_inf_f32 = step_bad.to(dtype=torch.float32)
    if found_map:
        for dev_key in list(found_map.keys()):
            cur = found_map[dev_key]
            found_map[dev_key] = torch.maximum(
                cur, found_inf_f32.to(device=cur.device, dtype=cur.dtype)
            )
    else:
        found_map[device] = found_inf_f32


def _grad_scaler_reset_optimizer_state(
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Clear per-optimizer `GradScaler` tracking on pre-update fatal errors."""
    if not scaler.is_enabled():
        return
    per_states = getattr(scaler, "_per_optimizer_states", None)
    if isinstance(per_states, dict):
        per_states.pop(id(optimizer), None)


def _collect_reachable_autograd_param_ids(loss: Tensor) -> tuple[set[int], bool]:
    """Walk `loss.grad_fn` with zero CUDA sync to find reachable `AccumulateGrad` parameters."""
    root = loss.grad_fn
    if root is None:
        return set(), False
    seen: set[Any] = set()
    stack: list[Any] = [root]
    reachable: set[int] = set()
    has_reentrant_ckpt = False
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        if "CheckpointFunction" in type(fn).__name__:
            has_reentrant_ckpt = True
        var = getattr(fn, "variable", None)
        if var is not None:
            reachable.add(id(var))
        for nxt, _ in fn.next_functions:
            if nxt is not None and nxt not in seen:
                stack.append(nxt)
    return reachable, has_reentrant_ckpt


def _optimizer_step_with_grad_scaler(
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
    *,
    one_f32: Tensor,
) -> None:
    """Execute `scaler.step(optimizer)` with safe `grad_scale` / `found_inf` cleanup."""
    if not scaler.is_enabled():
        optimizer.step()
        return
    supports_amp = bool(getattr(optimizer, "_step_supports_amp_scaling", False))
    if supports_amp:
        optimizer.grad_scale = one_f32  # type: ignore[attr-defined]
    try:
        scaler.step(optimizer)
    finally:
        if hasattr(optimizer, "grad_scale"):
            del optimizer.grad_scale  # type: ignore[attr-defined]
        if hasattr(optimizer, "found_inf"):
            del optimizer.found_inf  # type: ignore[attr-defined]


def train_joint_iql_batches(
    module: JointStructuredIQLModule,
    batches: Sequence[TensorDict] | Any,
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
    max_steps: int | None = None,
    grad_accum_steps: int = 1,
) -> dict[str, Any]:
    """Train JointStructuredIQLModule with strict autograd, DDP, streaming, and pre-update safety guards."""
    import contextlib

    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1.")

    raw = module.module if isinstance(module, DistributedDataParallel) else module
    assert isinstance(raw, JointStructuredIQLModule)
    raw.sync_targets_ddp()

    stream_grp = getattr(batches, "grp", None)
    if stream_grp is not None:
        if (
            stream_grp.grp_sha256 != recipe.get("grp_sha256")
            or stream_grp.ruleset_sha256 != recipe.get("ruleset_sha256")
        ):
            raise ValueError(
                "StreamingIQLDataset FrozenGRP identity disagrees with Joint IQL recipe (Fail-Closed)."
            )
    stream_src_sha = getattr(batches, "source_data_sha256", None)
    if stream_src_sha and recipe.get("source_data_sha256") and stream_src_sha != recipe.get("source_data_sha256"):
        raise ValueError(
            "StreamingIQLDataset source_data_sha256 disagrees with Joint IQL recipe (Fail-Closed)."
        )
    stream_prep_sha = getattr(batches, "preprocess_recipe_sha256", None)
    if stream_prep_sha and recipe.get("preprocess_recipe_sha256") and stream_prep_sha != recipe.get("preprocess_recipe_sha256"):
        raise ValueError(
            "StreamingIQLDataset preprocess_recipe_sha256 disagrees with Joint IQL recipe (Fail-Closed)."
        )
    stream_data_sha = getattr(batches, "data_sha256", None)
    if stream_data_sha and recipe.get("data_sha256") and stream_data_sha != recipe.get("data_sha256"):
        raise ValueError(
            "StreamingIQLDataset data_sha256 disagrees with Joint IQL recipe (Fail-Closed)."
        )

    is_amp = amp_dtype is not None and device.type == "cuda" and amp_dtype != torch.float32
    scaler = grad_scaler or GradScaler("cuda", enabled=(is_amp and amp_dtype == torch.float16))

    is_stream = hasattr(batches, "next_batch")
    start_step = 0
    cursor = 0
    stats_restored: dict[str, int] = {}
    stream_restored: list[Any] = []
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
            stream_state_out=stream_restored,
        )
        if is_stream and stream_restored:
            batches.load_state_dict(stream_restored[0])

    runner = ddp_wrapped if ddp_wrapped is not None else module
    ddp_runner = isinstance(runner, DistributedDataParallel)
    runner.train()

    inactive_ids = expected_inactive_iql_param_ids(raw)
    validate_optimizer_parameter_coverage(
        raw,
        optimizer,
        inactive_param_ids=inactive_ids,
        context="train_joint_iql_batches",
    )
    expected_active_named = tuple(
        (n, p)
        for n, p in raw.named_parameters()
        if p.requires_grad and id(p) not in inactive_ids
    )
    if not expected_active_named:
        raise OptimizerCoverageError(
            "[train_joint_iql_batches] No trainable parameters remain active in model/optimizer."
        )

    is_dist = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_dist else 1
    can_gpu_fused_guard = (
        device.type == "cuda"
        and bool(optimizer.param_groups)
        and all(bool(g.get("fused", False)) for g in optimizer.param_groups)
        and bool(getattr(optimizer, "_step_supports_amp_scaling", False))
        and isinstance(optimizer, torch.optim.AdamW)
        and scheduler is None
    )
    one_f32 = torch.ones((), device=device, dtype=torch.float32)
    zero_f32 = torch.zeros((), device=device, dtype=torch.float32)
    zero_b = torch.zeros((), device=device, dtype=torch.bool)
    one_b = torch.ones((), device=device, dtype=torch.bool)
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
    fatal_steps_t = torch.zeros((), device=device, dtype=torch.int64)

    step = start_step
    last_metrics: dict[str, float] = {}
    last_stats: dict[str, int] = {
        "optimizer_steps": int(stats_restored.get("optimizer_steps", 0)),
        "skipped_steps": int(stats_restored.get("skipped_steps", 0)),
        "nonfinite_loss_steps": int(stats_restored.get("nonfinite_loss_steps", 0)),
        "nonfinite_grad_steps": int(stats_restored.get("nonfinite_grad_steps", 0)),
    }
    last_out: dict[str, Tensor] | None = None
    last_fatal_msg = ""

    def _flush_metrics(out_map: dict[str, Tensor]) -> tuple[dict[str, float], dict[str, int]]:
        keys = list(out_map.keys())
        packed = torch.stack(
            [
                optimizer_steps_t.double(),
                skipped_steps_t.double(),
                nonfinite_loss_steps_t.double(),
                nonfinite_grad_steps_t.double(),
                fatal_steps_t.double(),
            ]
            + [out_map[k].detach().double() for k in keys]
        ).cpu().tolist()
        if int(round(packed[4])) > 0:
            raise RuntimeError(
                last_fatal_msg
                or "Fatal autograd/disconnected-graph error synchronized across DDP ranks."
            )
        stats = {
            "optimizer_steps": int(round(packed[0])),
            "skipped_steps": int(round(packed[1])),
            "nonfinite_loss_steps": int(round(packed[2])),
            "nonfinite_grad_steps": int(round(packed[3])),
        }
        metrics_f = {k: float(v) for k, v in zip(keys, packed[5:], strict=True)}
        if max_nonfinite_steps is not None and stats["skipped_steps"] > int(max_nonfinite_steps):
            raise RuntimeError(
                f"Exceeded max_nonfinite_steps={max_nonfinite_steps}: skipped={stats['skipped_steps']} "
                f"(nonfinite_loss={stats['nonfinite_loss_steps']}, "
                f"nonfinite_grad={stats['nonfinite_grad_steps']})."
            )
        return metrics_f, stats

    seq_len = len(batches) if (not is_stream and isinstance(batches, Sequence)) else None
    seq_cursor = cursor

    def _pull_microbatch() -> tuple[TensorDict | None, bool, bool]:
        """Pull next microbatch and signal ``(batch, is_definite_end, any_rank_ghost)``."""
        nonlocal seq_cursor
        if is_stream:
            b = batches.next_batch()
            if b is None and max_steps is not None:
                batches.reset_pass()
                b = batches.next_batch()
            if is_dist and world_size > 1:
                sync_dev = torch.device("cpu") if dist.get_backend() == "gloo" else device
                flags_t = torch.tensor(
                    [1 if b is not None else 0, 1 if b is None else 0],
                    device=sync_dev,
                    dtype=torch.int32,
                )
                dist.all_reduce(flags_t, op=dist.ReduceOp.MAX)
                if int(flags_t[0].item()) == 0:
                    return None, True, False
                any_ghost_mb = int(flags_t[1].item()) > 0
                if b is None:
                    b = batches.ghost_batch()
                seq_cursor += 1
                return b, False, any_ghost_mb
            if b is None:
                return None, True, False
            seq_cursor += 1
            return b, False, False

        assert seq_len is not None
        if seq_cursor >= seq_len:
            if max_steps is None:
                return None, True, False
            seq_cursor = 0
            if seq_len == 0:
                return None, True, False
        b = batches[seq_cursor]
        seq_cursor += 1
        at_seq_end = max_steps is None and seq_cursor >= seq_len
        return b, at_seq_end, False

    try:
        while max_steps is None or step < max_steps:
            micro_group: list[TensorDict] = []
            reached_end = False
            any_ghost_group = False
            for _ in range(int(grad_accum_steps)):
                b, mb_end, mb_ghost = _pull_microbatch()
                if b is None:
                    reached_end = True
                    break
                micro_group.append(b)
                any_ghost_group = any_ghost_group or mb_ghost
                if mb_end:
                    reached_end = True
                    break
            if not micro_group:
                break

            local_w_sum = 0.0
            any_bw_key = False
            for bcpu in micro_group:
                has_bw = "batch_weight" in bcpu.keys()
                any_bw_key = any_bw_key or has_bw
                bw = float(bcpu["batch_weight"].mean().item()) if has_bw else 1.0
                local_w_sum += bw

            if is_dist and world_size > 1 and (
                int(grad_accum_steps) > 1 or any_ghost_group or ((not is_stream) and any_bw_key)
            ):
                sync_dev = torch.device("cpu") if dist.get_backend() == "gloo" else device
                w_vec = torch.tensor([local_w_sum], device=sync_dev, dtype=torch.float64)
                dist.all_reduce(w_vec, op=dist.ReduceOp.SUM)
                global_w_sum = max(1e-8, float(w_vec.item()))
                loss_scale: float | None = float(world_size) / global_w_sum
            elif len(micro_group) > 1 or local_w_sum <= 0.0:
                loss_scale = 1.0 / max(1e-8, local_w_sum) if local_w_sum > 0.0 else 0.0
            else:
                loss_scale = None

            optimizer.zero_grad(set_to_none=True)

            local_fatal = False
            skipped_forward = bool(last_fatal_msg)
            last_forward_armed_reducer = False
            local_fatal_msg = last_fatal_msg
            loss_bad = zero_b
            out: dict[str, Tensor] = last_out if last_out is not None else {}

            if skipped_forward:
                local_fatal = True
            else:
                for m_idx, batch_cpu in enumerate(micro_group):
                    batch = batch_cpu.to(device=device, non_blocking=True)
                    is_last_micro = m_idx == len(micro_group) - 1
                    sync_ctx = (
                        runner.no_sync()
                        if (ddp_runner and not is_last_micro)
                        else contextlib.nullcontext()
                    )
                    with sync_ctx:
                        with torch.autocast(
                            device_type=device.type,
                            dtype=amp_dtype or torch.float32,
                            enabled=bool(is_amp),
                            cache_enabled=False,
                        ):
                            out = runner(batch, amp_dtype=amp_dtype, loss_scale=loss_scale)
                        if ddp_runner and is_last_micro:
                            last_forward_armed_reducer = True
                        loss = out["loss"]
                        if loss_transform is not None:
                            loss = loss_transform(loss, step)

                        if (
                            not isinstance(loss, Tensor)
                            or loss.numel() != 1
                            or not loss.requires_grad
                            or loss.grad_fn is None
                        ):
                            local_fatal = True
                            local_fatal_msg = (
                                "Training loss has no autograd graph (requires_grad=False or grad_fn is None); "
                                "aborting before backward/optimizer update."
                            )
                            break

                        reachable_ids, has_reentrant_ckpt = _collect_reachable_autograd_param_ids(loss)
                        opt_reach = any(id(p) in reachable_ids for _, p in expected_active_named)
                        if not opt_reach:
                            local_fatal = True
                            local_fatal_msg = (
                                "Disconnected computation graph detected before backward: trainable parameters "
                                "not connected to loss (<no optimizer parameters connected to loss>)."
                            )
                            break
                        if not has_reentrant_ckpt:
                            pre_disc = [
                                n for n, p in expected_active_named if id(p) not in reachable_ids
                            ]
                            if pre_disc:
                                local_fatal = True
                                local_fatal_msg = (
                                    "Disconnected computation graph detected before backward: trainable parameters "
                                    f"not connected to loss ({pre_disc[:8]}{'...' if len(pre_disc) > 8 else ''})."
                                )
                                break

                        loss_bad = loss_bad | (~torch.isfinite(loss.detach())).reshape(())
                        scaler.scale(loss).backward()

            if local_fatal:
                last_fatal_msg = local_fatal_msg
                optimizer.zero_grad(set_to_none=True)
                if not (is_dist and world_size > 1):
                    _grad_scaler_reset_optimizer_state(scaler, optimizer)
                    raise RuntimeError(local_fatal_msg)
                # Participate in DDP Reducer AllReduce cleanly via zero-grad dummy backward.
                dummy_ddp_zero_loss(
                    runner, device, prepare_reducer=(ddp_runner and not last_forward_armed_reducer)
                ).backward()
                loss_bad = zero_b
                grad_bad = zero_b
                grad_norm = zero_f32
                params_with_grad: list[Tensor] = []
            else:
                post_disc = [n for n, p in expected_active_named if p.grad is None]
                if post_disc:
                    local_fatal = True
                    local_fatal_msg = (
                        "Disconnected computation graph after backward: trainable parameters received "
                        f"no gradients ({post_disc[:8]}{'...' if len(post_disc) > 8 else ''})."
                    )
                    last_fatal_msg = local_fatal_msg
                    if not (is_dist and world_size > 1):
                        optimizer.zero_grad(set_to_none=True)
                        _grad_scaler_reset_optimizer_state(scaler, optimizer)
                        raise RuntimeError(local_fatal_msg)
                    loss_bad = zero_b
                    grad_bad = zero_b
                    grad_norm = zero_f32
                    params_with_grad = []
                else:
                    if post_backward_hook is not None:
                        post_backward_hook(runner, optimizer, step)
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)

                    params_with_grad = []
                    grads: list[Tensor] = []
                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            if p.grad is not None:
                                params_with_grad.append(p)
                                grads.append(p.grad)

                    grad_norm, grad_bad = _compute_grad_norm_and_nonfinite(grads, device=device)
                    if scaler.is_enabled():
                        scaler_bad = _grad_scaler_detect_nonfinite(
                            scaler, optimizer, device=device, zero_f32=zero_f32
                        )
                        grad_bad = grad_bad | scaler_bad

            fatal_any, loss_bad, grad_bad = sync_ddp_step_status(
                one_b if local_fatal else zero_b,
                loss_bad,
                grad_bad,
                is_dist=is_dist,
                world_size=world_size,
                device=device,
            )
            step_bad = loss_bad | grad_bad | fatal_any

            if not can_gpu_fused_guard:
                fatal_host, bad_host = torch.stack([fatal_any, step_bad]).cpu().tolist()
                if fatal_host:
                    optimizer.zero_grad(set_to_none=True)
                    _grad_scaler_reset_optimizer_state(scaler, optimizer)
                    suffix = (
                        " [propagated across DDP]"
                        if (is_dist and world_size > 1 and not local_fatal)
                        else ""
                    )
                    raise RuntimeError(
                        (last_fatal_msg or "Fatal autograd/disconnected-graph error detected") + suffix
                    )
                if params_with_grad and max_grad_norm > 0.0:
                    safe_norm = torch.where(step_bad, clip_f32, grad_norm)
                    torch.nn.utils.clip_grads_with_norm_(
                        params_with_grad, float(max_grad_norm), safe_norm
                    )
                if scaler.is_enabled():
                    _grad_scaler_inject_step_bad(scaler, optimizer, step_bad=step_bad, device=device)
                if not bad_host:
                    if scaler.is_enabled():
                        _optimizer_step_with_grad_scaler(scaler, optimizer, one_f32=one_f32)
                        scaler.update()
                    else:
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    raw.update_targets()
                elif scaler.is_enabled():
                    scaler.update()
                count_bad = step_bad
            else:
                fatal_steps_t.add_(fatal_any.to(dtype=torch.int64))
                sticky_bad = step_bad | (fatal_steps_t > 0)
                if params_with_grad and max_grad_norm > 0.0:
                    safe_norm = torch.where(sticky_bad, clip_f32, grad_norm)
                    torch.nn.utils.clip_grads_with_norm_(
                        params_with_grad, float(max_grad_norm), safe_norm
                    )
                if scaler.is_enabled():
                    _grad_scaler_inject_step_bad(
                        scaler, optimizer, step_bad=sticky_bad, device=device
                    )
                    _optimizer_step_with_grad_scaler(scaler, optimizer, one_f32=one_f32)
                    scaler.update()
                else:
                    found_inf_f32 = sticky_bad.to(dtype=torch.float32)
                    optimizer.grad_scale = one_f32  # type: ignore[attr-defined]
                    optimizer.found_inf = found_inf_f32  # type: ignore[attr-defined]
                    try:
                        optimizer.step()
                    finally:
                        if hasattr(optimizer, "grad_scale"):
                            del optimizer.grad_scale  # type: ignore[attr-defined]
                        if hasattr(optimizer, "found_inf"):
                            del optimizer.found_inf  # type: ignore[attr-defined]
                raw.update_targets(valid=~sticky_bad)
                count_bad = sticky_bad

            optimizer.zero_grad(set_to_none=True)

            step += 1
            cursor = seq_cursor
            last_out = out
            step_bad_i = count_bad.to(dtype=torch.int64)
            skipped_steps_t.add_(step_bad_i)
            optimizer_steps_t.add_(1 - step_bad_i)
            nonfinite_loss_steps_t.add_(loss_bad.to(dtype=torch.int64))
            nonfinite_grad_steps_t.add_(grad_bad.to(dtype=torch.int64))

            at_end = (max_steps is not None and step >= max_steps) or reached_end
            need_ckpt = ckpt_dir is not None and checkpoint_every > 0 and step % checkpoint_every == 0
            need_sync = (
                device.type == "cpu"
                or at_end
                or need_ckpt
                or (check_finite_interval > 0 and step % check_finite_interval == 0)
                or (log_interval > 0 and step % log_interval == 0)
            )
            if need_sync:
                last_metrics, last_stats = _flush_metrics(out)
            if need_ckpt:
                assert ckpt_dir is not None
                cur_stream_state = batches.state_dict() if is_stream else None
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
                    stream_state=cur_stream_state,
                )
            if reached_end and max_steps is None:
                break
    finally:
        if is_stream and hasattr(batches, "_stop_prefetcher"):
            batches._stop_prefetcher(rollback_queued=True)

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
    "expected_inactive_iql_param_ids",
    "load_joint_iql_checkpoint",
    "save_joint_iql_checkpoint",
    "train_joint_iql_batches",
]
