"""Portable dataset, hardware, and strict-acceptance test environment helpers (§9)."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import torch
import torch.distributed as dist

VAL_BC_ENV_VAR = "KANACHAN_VAL_BC_DATA"
VAL_RL_ENV_VAR = "KANACHAN_VAL_RL_DATA"
STRICT_ACCEPTANCE_ENV_VAR = "KANACHAN_STRICT_ACCEPTANCE"
DISTRIBUTED_TEST_ENV_VAR = "KANACHAN_TEST_DISTRIBUTED"
SKIP_CUDA_ENV_VAR = "KANACHAN_SKIP_CUDA"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def is_strict_acceptance() -> bool:
    """Return True when skipped external-data/hardware tests must fail closed."""
    return _env_flag(STRICT_ACCEPTANCE_ENV_VAR)


def is_cuda_skipped() -> bool:
    """Return True when CUDA execution is explicitly suppressed (e.g. CPU CI)."""
    return _env_flag(SKIP_CUDA_ENV_VAR)


def is_distributed_test_enabled() -> bool:
    """Return True when multi-rank Gloo tests are enabled (or required strictly)."""
    return _env_flag(DISTRIBUTED_TEST_ENV_VAR) or is_strict_acceptance()


def preferred_test_device() -> torch.device:
    """Return CUDA device unless suppressed by ``KANACHAN_SKIP_CUDA=1`` or unavailable."""
    if not is_cuda_skipped() and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_val_data_path(kind: str) -> tuple[Path | None, str]:
    """Resolve ``(path_or_none, env_var_name)`` for ``kind`` in ``{'bc', 'rl'}``."""
    norm = kind.strip().lower()
    if norm == "bc":
        env_var = VAL_BC_ENV_VAR
    elif norm == "rl":
        env_var = VAL_RL_ENV_VAR
    else:
        raise ValueError(f"Unknown validation dataset kind={kind!r}; expected 'bc' or 'rl'.")
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return None, env_var
    return Path(raw), env_var


def resolve_val_bc_path() -> Path | None:
    path, _ = resolve_val_data_path("bc")
    return path


def resolve_val_rl_path() -> Path | None:
    path, _ = resolve_val_data_path("rl")
    return path


def require_or_skip_real_data(
    case: unittest.TestCase,
    kind: str,
    *,
    context: str = "",
) -> Path:
    """Resolve real validation data path; fail on invalid/strict missing, skip otherwise."""
    path, env_var = resolve_val_data_path(kind)
    ctx_suffix = f" ({context})" if context else ""
    if path is None:
        msg = (
            f"Real {kind.upper()} validation dataset not configured{ctx_suffix}: "
            f"set {env_var}=<path-to-validation.{kind}.tsv.gz>."
        )
        if is_strict_acceptance():
            case.fail(f"[{STRICT_ACCEPTANCE_ENV_VAR}=1] {msg}")
        case.skipTest(msg)
    if not path.is_file():
        case.fail(
            f"{env_var}={str(path)!r} does not point to an existing file{ctx_suffix}."
        )
    return path


def require_or_skip_cuda(
    case: unittest.TestCase,
    *,
    context: str = "",
) -> torch.device:
    """Return CUDA device, failing closed under strict mode if unavailable/skipped."""
    ctx_suffix = f" ({context})" if context else ""
    if is_cuda_skipped():
        msg = f"CUDA tests suppressed via {SKIP_CUDA_ENV_VAR}=1{ctx_suffix}."
        if is_strict_acceptance():
            case.fail(f"[{STRICT_ACCEPTANCE_ENV_VAR}=1] {msg}")
        case.skipTest(msg)
    if not torch.cuda.is_available():
        msg = f"CUDA device not available{ctx_suffix}."
        if is_strict_acceptance():
            case.fail(f"[{STRICT_ACCEPTANCE_ENV_VAR}=1] {msg}")
        case.skipTest(msg)
    return torch.device("cuda")


def require_or_skip_distributed(
    case: unittest.TestCase,
    *,
    context: str = "",
) -> None:
    """Require `torch.distributed` availability and opt-in flag (or strict mode)."""
    ctx_suffix = f" ({context})" if context else ""
    if not dist.is_available():
        msg = f"torch.distributed is not available{ctx_suffix}."
        if is_strict_acceptance():
            case.fail(f"[{STRICT_ACCEPTANCE_ENV_VAR}=1] {msg}")
        case.skipTest(msg)
    if not is_distributed_test_enabled():
        case.skipTest(f"Set {DISTRIBUTED_TEST_ENV_VAR}=1 to run multi-rank Gloo DDP test{ctx_suffix}.")


__all__ = [
    "DISTRIBUTED_TEST_ENV_VAR",
    "SKIP_CUDA_ENV_VAR",
    "STRICT_ACCEPTANCE_ENV_VAR",
    "VAL_BC_ENV_VAR",
    "VAL_RL_ENV_VAR",
    "is_cuda_skipped",
    "is_distributed_test_enabled",
    "is_strict_acceptance",
    "preferred_test_device",
    "require_or_skip_cuda",
    "require_or_skip_distributed",
    "require_or_skip_real_data",
    "resolve_val_bc_path",
    "resolve_val_data_path",
    "resolve_val_rl_path",
]
