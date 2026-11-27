"""Content-based SHA-256 dataset and preprocessing recipe fingerprints (§3)."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch.distributed as dist

from kanachan.nn.structured_dual_decoder import structured_v2_schema
from kanachan.structured_v2 import KANACHAN_ACTION_MANIFEST_SHA256, STRUCTURED_V2_SCHEMA_ID

SOURCE_V3_BC_FORMAT_VERSION = "source_v3_bc_v1"
SOURCE_V3_RL_FORMAT_VERSION = "source_v3_rl_v1"


def sha256_file_bytes(path: Path, *, chunk_size: int = 1024 * 1024) -> tuple[int, str]:
    """Stream-hash exact on-disk bytes (compressed bytes for .gz) without full file load."""
    c_size = int(chunk_size)
    if c_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    p = Path(path)
    hasher = hashlib.sha256()
    buf = bytearray(c_size)
    mv = memoryview(buf)
    total_bytes = 0
    with p.open("rb") as fp:
        while True:
            n = fp.readinto(mv)
            if not n:
                break
            hasher.update(mv[:n])
            total_bytes += n
    return total_bytes, hasher.hexdigest()


def compute_reward_plugin_sha256(
    reward_fn: Any | None = None,
    reward_plugin_path: Path | None = None,
) -> str | None:
    """Compute content SHA-256 of a custom reward plugin file or Python callable."""
    if reward_plugin_path is not None:
        _, digest = sha256_file_bytes(Path(reward_plugin_path))
        return digest
    if reward_fn is None:
        return None
    h = hashlib.sha256()
    target = reward_fn
    if not inspect.isfunction(target) and callable(target) and hasattr(target, "__call__"):
        h.update(f"{type(target).__module__}.{type(target).__qualname__}".encode("utf-8"))
        try:
            src = inspect.getsource(type(target))
            h.update(src.encode("utf-8"))
            return h.hexdigest()
        except (OSError, TypeError):
            pass
    if inspect.isfunction(target):
        h.update(f"{target.__module__}.{target.__qualname__}".encode("utf-8"))
        try:
            src = inspect.getsource(target)
            h.update(src.encode("utf-8"))
        except (OSError, TypeError):
            code = getattr(target, "__code__", None)
            if code is not None:
                h.update(code.co_code)
                h.update(repr(code.co_consts).encode("utf-8"))
                h.update(repr(code.co_names).encode("utf-8"))
        return h.hexdigest()
    h.update(repr(target).encode("utf-8"))
    return h.hexdigest()


def compute_source_data_fingerprint(
    files: Sequence[Path],
    *,
    source_format: str,
    schema_id: str = STRUCTURED_V2_SCHEMA_ID,
    manifest_sha256: str = KANACHAN_ACTION_MANIFEST_SHA256,
    max_items: int | None = None,
    shared_across_ranks: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Compute ordered byte-content SHA-256 identity independent of filesystem path or mtime.

    When `shared_across_ranks=True` inside an initialized process group, Rank 0 hashes once
    and broadcasts `(source_data_sha256, file_entries)` to peer ranks.
    """
    ordered_files = [Path(f) for f in files]
    use_ddp = bool(shared_across_ranks and dist.is_available() and dist.is_initialized())
    rank = dist.get_rank() if use_ddp else 0

    def _compute_local() -> tuple[str, list[dict[str, Any]]]:
        entries: list[dict[str, Any]] = []
        for idx, fpath in enumerate(ordered_files):
            sz, hex_digest = sha256_file_bytes(fpath)
            entries.append({"index": idx, "size": sz, "sha256": hex_digest})
        payload = {
            "source_format": str(source_format),
            "schema_id": str(schema_id),
            "manifest_sha256": str(manifest_sha256),
            "max_items": int(max_items) if max_items is not None else None,
            "files": entries,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return digest, entries

    if not use_ddp:
        return _compute_local()

    box: list[Any] = [None, None]
    if rank == 0:
        try:
            box[0] = _compute_local()
        except BaseException as exc:  # noqa: BLE001
            box[1] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(box, src=0)
    if box[1] is not None:
        raise RuntimeError(f"Rank-0 source content fingerprint computation failed: {box[1]}")
    res_sha, res_entries = box[0]
    return str(res_sha), list(res_entries)


def compute_bc_spool_recipe_fingerprint(
    *,
    allow_unlabeled: bool,
    allow_incomplete: bool,
    enable_early_shape_hint: bool,
    max_rounds: int | None,
    schema_version: str | None = None,
) -> str:
    """Hash BC parsing / disk-spool layout parameters."""
    payload = {
        "kind": "bc_spool_recipe_v1",
        "schema": structured_v2_schema(),
        "schema_version": str(schema_version or SOURCE_V3_BC_FORMAT_VERSION),
        "manifest_sha256": KANACHAN_ACTION_MANIFEST_SHA256,
        "allow_unlabeled": bool(allow_unlabeled),
        "allow_incomplete": bool(allow_incomplete),
        "enable_early_shape_hint": bool(enable_early_shape_hint),
        "max_rounds": int(max_rounds) if max_rounds is not None else None,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def compute_bc_preprocess_fingerprint(
    *,
    allow_unlabeled: bool,
    allow_incomplete: bool,
    enable_early_shape_hint: bool,
    max_rounds: int | None,
    suit_augmentation: bool = False,
    aug_seed: int = 1337,
    sampler_seed: int = 42,
    batch_size: int | None = None,
    bucket_mult: int = 8,
    drop_last: bool = False,
    schema_version: str | None = None,
) -> str:
    """Hash full BC preprocessing + deterministic replay/augmentation configuration."""
    payload = {
        "kind": "bc_preprocess_recipe_v1",
        "schema": structured_v2_schema(),
        "schema_version": str(schema_version or SOURCE_V3_BC_FORMAT_VERSION),
        "manifest_sha256": KANACHAN_ACTION_MANIFEST_SHA256,
        "allow_unlabeled": bool(allow_unlabeled),
        "allow_incomplete": bool(allow_incomplete),
        "enable_early_shape_hint": bool(enable_early_shape_hint),
        "max_rounds": int(max_rounds) if max_rounds is not None else None,
        "suit_augmentation": bool(suit_augmentation),
        "aug_seed": int(aug_seed) if suit_augmentation else None,
        "sampler_seed": int(sampler_seed),
        "batch_size": int(batch_size) if batch_size is not None else None,
        "bucket_mult": int(bucket_mult),
        "drop_last": bool(drop_last),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def compute_iql_preprocess_fingerprint(
    *,
    enable_early_shape_hint: bool,
    rewrite_rooms: int | None,
    rewrite_grades: int | None,
    reward_scheme: str,
    reward_plugin_sha256: str | None,
    score_unit: float,
    round_weight: float,
    discount_factor: float,
    grp: Any | None,
    reservoir_size: int,
    warmup_transitions: int,
    batch_size: int,
    drop_last: bool,
    seed: int,
    suit_augmentation: bool,
    aug_seed: int,
    schema_version: str | None = None,
) -> str:
    """Hash full IQL preprocessing, reward/GRP identity, reservoir & replay configuration."""
    grp_meta = None
    if grp is not None:
        grp_meta = {
            "grp_sha256": getattr(grp, "grp_sha256", None),
            "ruleset_sha256": getattr(grp, "ruleset_sha256", None),
            "rank_payoffs": [float(x) for x in getattr(grp, "rank_payoffs", ())],
        }
    payload = {
        "kind": "iql_preprocess_recipe_v1",
        "schema": structured_v2_schema(),
        "schema_version": str(schema_version or SOURCE_V3_RL_FORMAT_VERSION),
        "manifest_sha256": KANACHAN_ACTION_MANIFEST_SHA256,
        "enable_early_shape_hint": bool(enable_early_shape_hint),
        "rewrite_rooms": int(rewrite_rooms) if rewrite_rooms is not None else None,
        "rewrite_grades": int(rewrite_grades) if rewrite_grades is not None else None,
        "reward_scheme": str(reward_scheme),
        "reward_plugin_sha256": reward_plugin_sha256,
        "score_unit": float(score_unit),
        "round_weight": float(round_weight),
        "discount_factor": float(discount_factor),
        "grp": grp_meta,
        "reservoir_size": int(reservoir_size),
        "warmup_transitions": int(warmup_transitions),
        "batch_size": int(batch_size),
        "drop_last": bool(drop_last),
        "seed": int(seed),
        "suit_augmentation": bool(suit_augmentation),
        "aug_seed": int(aug_seed) if suit_augmentation else None,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def combine_dataset_identity(
    source_data_sha256: str,
    preprocess_recipe_sha256: str,
) -> str:
    """Combine content identity and recipe identity into a single canonical digest."""
    payload = {
        "source_data_sha256": str(source_data_sha256),
        "preprocess_recipe_sha256": str(preprocess_recipe_sha256),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "SOURCE_V3_BC_FORMAT_VERSION",
    "SOURCE_V3_RL_FORMAT_VERSION",
    "combine_dataset_identity",
    "compute_bc_preprocess_fingerprint",
    "compute_bc_spool_recipe_fingerprint",
    "compute_iql_preprocess_fingerprint",
    "compute_reward_plugin_sha256",
    "compute_source_data_fingerprint",
    "sha256_file_bytes",
]
