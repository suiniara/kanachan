"""Audit C++ structured_v2 encoder against real source-v3 data & Python reference."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from kanachan.constants import (
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_ACTIONS,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_SPARSE_FEATURES,
    SEAT_INDEX,
    SEAT_OFFSET,
)
from kanachan.structured_v2 import (
    EVENT_BYTES,
    PAD_ACTION_ID,
    derive_structured_v2_batch,
    permute_raw_four_fields,
    permute_structured_batch,
)
from kanachan.structured_v2_ref import derive_structured_v2_batch_py
from kanachan.training.core.features import validate_sparse_prefix


_AUDIT_TENSORS: tuple[str, ...] = (
    "v2_tile_feats",
    "v2_events",
    "v2_event_len",
    "v2_global_feats",
    "v2_player_feats",
    "v2_cand_ids",
    "v2_cand_mask",
    "v2_cand_mode",
    "v2_cand_dyn",
)

_INT_EXACT_TENSORS: frozenset[str] = frozenset(
    {"v2_events", "v2_event_len", "v2_cand_ids", "v2_cand_mask", "v2_cand_mode"}
)

REQUIRED_CAND_MODES: tuple[str, ...] = tuple(str(m) for m in range(7))

REQUIRED_RULE_CATEGORIES: tuple[str, ...] = (
    "red5",
    "chi",
    "pon",
    "dmk",
    "ankan",
    "kakan",
    "dora2p",
    "riichi",
    "ippatsu",
    "furiten",
    "pending_resp",
    "kyushu",
    "win_tsumo",
    "win_ron",
    "win",
    "west_or_all_last",
    "left0",
)

_SCORE_BOUND = 10_000_000


def _open_tsv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _extract_best_effort_ctx(raw_line: str, line_no: int | None) -> str:
    prefix = f"line {line_no}" if line_no is not None else "sample"
    parts = raw_line.rstrip("\r\n").split("\t")
    uuid_s = parts[0].strip() if parts and parts[0].strip() else "<missing>"
    round_s = "<unknown>"
    try:
        if len(parts) >= 3:
            sp_s = parts[1].split(",")
            nu_s = parts[2].split(",")
            if len(sp_s) >= 9 and len(nu_s) >= 1:
                round_s = f"({int(sp_s[7]) - 75}, {int(sp_s[8]) - 78}, {int(nu_s[0])})"
    except Exception:
        pass
    return f"{prefix} (uuid={uuid_s}, round={round_s})"


def _parse_int_csv(raw: str, field_name: str, ctx: str) -> list[int]:
    if not raw or not raw.strip():
        raise ValueError(f"{ctx}: Empty {field_name} field.")
    out: list[int] = []
    for tok in raw.split(","):
        s = tok.strip()
        if not s:
            raise ValueError(f"{ctx}: Empty element in {field_name}.")
        try:
            out.append(int(s))
        except ValueError as exc:
            raise ValueError(f"{ctx}: Non-integer token {tok!r} in {field_name}.") from exc
    return out


def _parse_sample_line(
    line: str,
    *,
    line_no: int | None = None,
) -> tuple[str, list[int], list[int], list[int], list[int], list[int], list[int]]:
    """Strictly parse and validate one TSV line without silent skipping or truncation."""
    stripped = line.rstrip("\r\n")
    ctx = _extract_best_effort_ctx(line, line_no)
    if not stripped:
        raise ValueError(f"{ctx}: Blank line in audit dataset.")
    parts = stripped.split("\t")
    if len(parts) not in (5, 6, 8):
        raise ValueError(
            f"{ctx}: Expected 5, 6, or 8 tab-separated columns, got {len(parts)}."
        )
    uuid = parts[0].strip()
    if not uuid:
        raise ValueError(f"{ctx}: Empty game UUID.")

    sp_raw = _parse_int_csv(parts[1], "sparse", ctx)
    if len(sp_raw) > MAX_NUM_ACTIVE_SPARSE_FEATURES:
        raise ValueError(
            f"{ctx}: Overlong sparse field ({len(sp_raw)} > {MAX_NUM_ACTIVE_SPARSE_FEATURES}); "
            "truncation is forbidden."
        )
    if any(x < 0 or x >= NUM_TYPES_OF_SPARSE_FEATURES for x in sp_raw):
        raise ValueError(
            f"{ctx}: Sparse token out of [0, {NUM_TYPES_OF_SPARSE_FEATURES})."
        )
    try:
        validate_sparse_prefix(sp_raw, end_of_round=False)
    except RuntimeError as exc:
        raise ValueError(f"{ctx}: {exc}") from exc

    nu = _parse_int_csv(parts[2], "numeric", ctx)
    if len(nu) != NUM_NUMERIC_FEATURES:
        raise ValueError(
            f"{ctx}: Expected {NUM_NUMERIC_FEATURES} numeric features, got {len(nu)}."
        )
    if nu[0] < 0 or nu[1] < 0 or any(abs(x) > _SCORE_BOUND for x in nu):
        raise ValueError(f"{ctx}: Numeric feature out of valid range: {nu}.")

    chang = sp_raw[7] - 75
    ju = sp_raw[8] - 78
    seat = sp_raw[SEAT_INDEX] - SEAT_OFFSET
    if not (0 <= seat < 4 and 0 <= chang < 3 and 0 <= ju < 4):
        raise ValueError(f"{ctx}: Invalid seat/chang/ju ({seat}, {chang}, {ju}).")
    ctx = f"{'line ' + str(line_no) if line_no is not None else 'sample'} (uuid={uuid}, round=({chang}, {ju}, {nu[0]}))"

    pr_raw = _parse_int_csv(parts[3], "progression", ctx)
    if len(pr_raw) > MAX_LENGTH_OF_PROGRESSION_FEATURES:
        raise ValueError(
            f"{ctx}: Overlong progression field ({len(pr_raw)} > {MAX_LENGTH_OF_PROGRESSION_FEATURES}); "
            "truncation is forbidden."
        )
    if pr_raw[0] != 0 or any(x < 0 or x >= NUM_TYPES_OF_PROGRESSION_FEATURES for x in pr_raw):
        raise ValueError(f"{ctx}: Invalid progression tokens.")

    ca_raw = _parse_int_csv(parts[4], "candidates", ctx)
    if len(ca_raw) > MAX_NUM_ACTION_CANDIDATES:
        raise ValueError(
            f"{ctx}: Overlong candidates field ({len(ca_raw)} > {MAX_NUM_ACTION_CANDIDATES}); "
            "truncation is forbidden."
        )
    if any(x < 0 or x >= NUM_TYPES_OF_ACTIONS for x in ca_raw):
        raise ValueError(
            f"{ctx}: Candidate action ID out of [0, {NUM_TYPES_OF_ACTIONS})."
        )
    if any(ca_raw[i] >= ca_raw[i + 1] for i in range(len(ca_raw) - 1)):
        raise ValueError(f"{ctx}: Candidate action IDs must be strictly ascending: {ca_raw}.")

    sp = sp_raw + [NUM_TYPES_OF_SPARSE_FEATURES] * (MAX_NUM_ACTIVE_SPARSE_FEATURES - len(sp_raw))
    pr = pr_raw + [NUM_TYPES_OF_PROGRESSION_FEATURES] * (MAX_LENGTH_OF_PROGRESSION_FEATURES - len(pr_raw))
    ca = ca_raw + [PAD_ACTION_ID] * (MAX_NUM_ACTION_CANDIDATES - len(ca_raw))
    return uuid, sp, nu, pr, ca, pr_raw, ca_raw


def _classify_sample_categories(
    sp: list[int],
    pr_raw: list[int],
    ca_raw: list[int],
) -> list[str]:
    """Identify Mahjong rule categories present in one validated source-v3 row."""
    left = sp[9] - 82
    chang = sp[7] - 75
    ju = sp[8] - 78
    n_dora = sum(1 for x in sp if 152 <= x < 337)
    cats: list[str] = []
    if any(
        x % 10 == 0 and x < 30
        for x in ((t - 152) % 37 for t in sp if 152 <= t < 337)
    ) or any((t - 337) in (0, 36, 72) for t in sp if 337 <= t < 473):
        cats.append("red5")
    if any(597 <= tok < 957 for tok in pr_raw) or any(222 <= c < 312 for c in ca_raw):
        cats.append("chi")
    if any(957 <= tok < 1437 for tok in pr_raw) or any(312 <= c < 432 for c in ca_raw):
        cats.append("pon")
    if any(1437 <= tok < 1881 for tok in pr_raw) or any(432 <= c < 543 for c in ca_raw):
        cats.append("dmk")
    if any(1881 <= tok < 2017 for tok in pr_raw) or any(148 <= c < 182 for c in ca_raw):
        cats.append("ankan")
    if any(2017 <= tok < 2165 for tok in pr_raw) or any(182 <= c < 219 for c in ca_raw):
        cats.append("kakan")
    if n_dora >= 2:
        cats.append("dora2p")
    if any((5 <= tok < 597 and ((tok - 5) % 148) & 1) for tok in pr_raw) or any(
        (c < 148 and (c & 1)) for c in ca_raw
    ):
        cats.append("riichi")

    # Detect active ippatsu window matching C++/Python encoder (§2.2.3)
    riichi_last_pos = [-1, -1, -1, -1]
    has_own_act = any(c < 221 for c in ca_raw)
    lp = len(pr_raw)
    for idx, tok in enumerate(pr_raw[1:], start=1):
        if 5 <= tok < 597:
            v = tok - 5
            s = v // 148
            if ((v % 148) & 1) != 0:
                riichi_last_pos[s] = idx
    ippatsu_hit = False
    for s, r_pos in enumerate(riichi_last_pos):
        if r_pos < 0:
            continue
        if (not has_own_act) and r_pos == lp - 1:
            continue
        broken = False
        for tok in pr_raw[r_pos + 1 :]:
            if tok >= 597:
                broken = True
                break
            if 5 <= tok < 597 and ((tok - 5) // 148) == s:
                broken = True
                break
        if not broken:
            ippatsu_hit = True
            break
    if ippatsu_hit:
        cats.append("ippatsu")

    if not has_own_act:
        cats.append("pending_resp")
    if 220 in ca_raw:
        cats.append("kyushu")
    if 219 in ca_raw:
        cats.append("win_tsumo")
    if any(543 <= c < 546 for c in ca_raw):
        cats.append("win_ron")
    if 219 in ca_raw or any(543 <= c < 546 for c in ca_raw):
        cats.append("win")
    if chang >= 2 or (chang == 1 and ju == 3):
        cats.append("west_or_all_last")
    if left == 0:
        cats.append("left0")
    return cats


def load_samples(path: Path, max_samples: int):
    if not path.is_file():
        raise FileNotFoundError(f"Audit data file does not exist: {path}")
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}.")
    uuids, sp_list, nu_list, pr_list, ca_list = [], [], [], [], []
    with _open_tsv(path) as f:
        for line_idx, line in enumerate(f, start=1):
            uuid, sp, nu, pr, ca, _pr_raw, _ca_raw = _parse_sample_line(line, line_no=line_idx)
            uuids.append(uuid)
            sp_list.append(sp)
            nu_list.append(nu)
            pr_list.append(pr)
            ca_list.append(ca)
            if len(sp_list) >= max_samples:
                break
    if not sp_list:
        raise ValueError(f"Audit dataset is empty: {path}")
    if len(sp_list) < max_samples:
        raise ValueError(
            f"Insufficient samples in {path}: requested {max_samples}, found {len(sp_list)}."
        )
    return (
        uuids,
        torch.tensor(sp_list, dtype=torch.int32),
        torch.tensor(nu_list, dtype=torch.int32),
        torch.tensor(pr_list, dtype=torch.int32),
        torch.tensor(ca_list, dtype=torch.int32),
    )


def harvest_rare_categories(
    path: Path,
    *,
    scan_limit: int = 120000,
    per_cat_target: int = 64,
    required_categories: tuple[str, ...] = REQUIRED_RULE_CATEGORIES,
):
    """Harvest diverse real Mahjong records covering melds, kans, red5s, riichi, ippatsu, wins, kyushu."""
    if not path.is_file():
        raise FileNotFoundError(f"Audit data file does not exist: {path}")
    if scan_limit <= 0:
        raise ValueError(f"scan_limit must be positive, got {scan_limit}.")
    if per_cat_target <= 0:
        raise ValueError(f"per_cat_target must be positive, got {per_cat_target}.")

    targets: dict[str, int] = {}
    rows: list[tuple[str, list[int], list[int], list[int], list[int]]] = []
    selected_row_ids: set[int] = set()
    category_counts: dict[str, int] = {cat: 0 for cat in required_categories}
    scanned = 0
    furiten_buf: list[tuple[int, str, list[int], list[int], list[int], list[int]]] = []

    def _flush_furiten_buf() -> None:
        if not furiten_buf:
            return
        b_sp = torch.tensor([x[2] for x in furiten_buf], dtype=torch.int32)
        b_nu = torch.tensor([x[3] for x in furiten_buf], dtype=torch.int32)
        b_pr = torch.tensor([x[4] for x in furiten_buf], dtype=torch.int32)
        b_ca = torch.tensor([x[5] for x in furiten_buf], dtype=torch.int32)
        out = derive_structured_v2_batch(b_sp, b_nu, b_pr, b_ca, include_room_level=False)
        is_fur = (
            (out["v2_global_feats"][:, 19] > 0.5)
            | (out["v2_tile_feats"][:, :, 67].sum(dim=-1) > 0.5)
        ).tolist()
        for idx_b, hit in enumerate(is_fur):
            if not hit:
                continue
            category_counts["furiten"] = category_counts.get("furiten", 0) + 1
            if targets.get("furiten", 0) < per_cat_target:
                l_idx, u, sp_v, nu_v, pr_v, ca_v = furiten_buf[idx_b]
                targets["furiten"] = targets.get("furiten", 0) + 1
                if l_idx not in selected_row_ids:
                    selected_row_ids.add(l_idx)
                    rows.append((u, sp_v, nu_v, pr_v, ca_v))
        furiten_buf.clear()

    with _open_tsv(path) as f:
        for line_idx, line in enumerate(f):
            if line_idx >= scan_limit:
                break
            scanned += 1
            uuid, sp, nu, pr, ca, pr_raw, ca_raw = _parse_sample_line(line, line_no=line_idx + 1)
            cats = _classify_sample_categories(sp, pr_raw, ca_raw)
            keep = False
            for cat in cats:
                category_counts[cat] = category_counts.get(cat, 0) + 1
                if targets.get(cat, 0) < per_cat_target:
                    keep = True
                    targets[cat] = targets.get(cat, 0) + 1
            if keep:
                selected_row_ids.add(line_idx)
                rows.append((uuid, sp, nu, pr, ca))
            if targets.get("furiten", 0) < per_cat_target and len(pr_raw) >= 5:
                furiten_buf.append((line_idx, uuid, sp, nu, pr, ca))
                if len(furiten_buf) >= 256:
                    _flush_furiten_buf()

    _flush_furiten_buf()

    if scanned == 0:
        raise ValueError(f"Audit dataset is empty: {path}")

    if not rows:
        return (
            [],
            torch.empty((0, MAX_NUM_ACTIVE_SPARSE_FEATURES), dtype=torch.int32),
            torch.empty((0, NUM_NUMERIC_FEATURES), dtype=torch.int32),
            torch.empty((0, MAX_LENGTH_OF_PROGRESSION_FEATURES), dtype=torch.int32),
            torch.empty((0, MAX_NUM_ACTION_CANDIDATES), dtype=torch.int32),
            category_counts,
        )
    return (
        [r[0] for r in rows],
        torch.tensor([r[1] for r in rows], dtype=torch.int32),
        torch.tensor([r[2] for r in rows], dtype=torch.int32),
        torch.tensor([r[3] for r in rows], dtype=torch.int32),
        torch.tensor([r[4] for r in rows], dtype=torch.int32),
        category_counts,
    )


def _py_ref_worker(
    task: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool],
) -> dict[str, np.ndarray]:
    sp_c, nu_c, pr_c, ca_c, hint = task
    return derive_structured_v2_batch_py(
        sp_c, nu_c, pr_c, ca_c, enable_early_shape_hint=hint
    )


def _derive_py_ref_parallel(
    sp_np: np.ndarray,
    nu_np: np.ndarray,
    pr_np: np.ndarray,
    ca_np: np.ndarray,
    *,
    enable_early_shape_hint: bool,
    workers: int,
) -> dict[str, np.ndarray]:
    n = int(sp_np.shape[0])
    if workers <= 1 or n < 512:
        return derive_structured_v2_batch_py(
            sp_np, nu_np, pr_np, ca_np, enable_early_shape_hint=enable_early_shape_hint
        )

    chunk_size = max(128, (n + workers - 1) // workers)
    tasks = [
        (sp_np[s:min(n, s + chunk_size)], nu_np[s:min(n, s + chunk_size)],
         pr_np[s:min(n, s + chunk_size)], ca_np[s:min(n, s + chunk_size)],
         enable_early_shape_hint)
        for s in range(0, n, chunk_size)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_py_ref_worker, tasks))

    lmax = max(int(p["v2_events"].shape[1]) for p in parts)
    padded_events: list[np.ndarray] = []
    for p in parts:
        ev = p["v2_events"]
        if ev.shape[1] < lmax:
            pad = np.zeros((ev.shape[0], lmax - ev.shape[1], EVENT_BYTES), dtype=ev.dtype)
            ev = np.concatenate([ev, pad], axis=1)
        padded_events.append(ev)

    out: dict[str, np.ndarray] = {}
    for k in _AUDIT_TENSORS:
        if k == "v2_events":
            out[k] = np.concatenate(padded_events, axis=0)
        else:
            out[k] = np.concatenate([p[k] for p in parts], axis=0)
    return out


def _compare_cpp_and_py(
    uuids: list[str],
    sp: torch.Tensor,
    nu: torch.Tensor,
    pr: torch.Tensor,
    ca: torch.Tensor,
    *,
    enable_early_shape_hint: bool,
    tag: str,
    workers: int = 1,
) -> tuple[dict[str, float], list[str]]:
    ref_cpp = derive_structured_v2_batch(
        sp, nu, pr, ca, enable_early_shape_hint=enable_early_shape_hint, include_room_level=False
    )
    ref_py = _derive_py_ref_parallel(
        sp.numpy(),
        nu.numpy(),
        pr.numpy(),
        ca.numpy(),
        enable_early_shape_hint=enable_early_shape_hint,
        workers=workers,
    )
    max_diffs: dict[str, float] = {}
    mismatches: list[str] = []
    for k in _AUDIT_TENSORS:
        if k not in ref_cpp or k not in ref_py:
            mismatches.append(f"[{tag} hint={enable_early_shape_hint}] missing tensor key={k}")
            max_diffs[k] = float("inf")
            continue
        cpp_arr = ref_cpp[k].numpy()
        py_arr = ref_py[k]
        if cpp_arr.shape != py_arr.shape:
            mismatches.append(
                f"[{tag} hint={enable_early_shape_hint}] tensor={k} shape mismatch: "
                f"cpp={cpp_arr.shape} vs py={py_arr.shape}"
            )
            max_diffs[k] = float("inf")
            continue
        diff = np.abs(cpp_arr.astype(np.float64) - py_arr.astype(np.float64))
        max_val = float(diff.max()) if diff.size > 0 else 0.0
        max_diffs[k] = max_val
        tol = 0.0 if k in _INT_EXACT_TENSORS else 1e-5
        if max_val > tol or not np.isfinite(max_val):
            flat_idx = int(diff.argmax())
            coords = np.unravel_index(flat_idx, diff.shape)
            row_i = int(coords[0])
            row_uuid = uuids[row_i] if row_i < len(uuids) else "<unknown>"
            mismatches.append(
                f"[{tag} hint={enable_early_shape_hint}] tensor={k} sample_idx={row_i} "
                f"uuid={row_uuid} coord={tuple(int(c) for c in coords)} "
                f"cpp={cpp_arr[coords]} py={py_arr[coords]} diff={max_val} tol={tol}"
            )
    return max_diffs, mismatches


def _accumulate_slot_modes_and_categories(
    uuids: list[str],
    sp: torch.Tensor,
    nu: torch.Tensor,
    pr: torch.Tensor,
    ca: torch.Tensor,
    *,
    mode_counts: dict[str, int],
    cat_counts: dict[str, int],
    enable_early_shape_hint: bool = True,
) -> None:
    """Count all 7 candidate modes (0..6) and Mahjong rule categories on reference-checked samples."""
    del uuids
    n = int(sp.shape[0])
    if n == 0:
        return
    out = derive_structured_v2_batch(
        sp,
        nu,
        pr,
        ca,
        enable_early_shape_hint=enable_early_shape_hint,
        include_room_level=False,
    )
    cand_mode = out["v2_cand_mode"]
    for m in range(7):
        mode_counts[str(m)] = mode_counts.get(str(m), 0) + int((cand_mode == m).sum().item())

    fur_hits = (
        (out["v2_global_feats"][:, 19] > 0.5)
        | (out["v2_tile_feats"][:, :, 67].sum(dim=-1) > 0.5)
    )
    cat_counts["furiten"] = cat_counts.get("furiten", 0) + int(fur_hits.sum().item())

    sp_rows = sp.tolist()
    pr_rows = pr.tolist()
    ca_rows = ca.tolist()
    for sp_r, pr_r, ca_r in zip(sp_rows, pr_rows, ca_rows, strict=True):
        pr_unpadded = [x for x in pr_r if x < NUM_TYPES_OF_PROGRESSION_FEATURES]
        ca_unpadded = [x for x in ca_r if x < NUM_TYPES_OF_ACTIONS]
        for cat in _classify_sample_categories(sp_r, pr_unpadded, ca_unpadded):
            cat_counts[cat] = cat_counts.get(cat, 0) + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--ref-steps", type=int, default=1000)
    parser.add_argument("--perm-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ref-workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--harvest-categories", action="store_true")
    parser.add_argument("--scan-limit", type=int, default=120000)
    parser.add_argument("--allow-unverified-categories", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.steps <= 0 or args.ref_steps < 0 or args.perm_steps < 0 or args.batch_size <= 0 or args.threads <= 0:
        raise ValueError(
            f"Invalid audit counts: steps={args.steps}, ref_steps={args.ref_steps}, "
            f"perm_steps={args.perm_steps}, batch_size={args.batch_size}, threads={args.threads}."
        )
    if args.ref_steps > args.steps or args.perm_steps > args.steps:
        raise ValueError(
            f"ref_steps ({args.ref_steps}) and perm_steps ({args.perm_steps}) cannot exceed steps ({args.steps})."
        )

    uuids, sp, nu, pr, ca = load_samples(args.data, args.steps)
    n = int(sp.shape[0])
    if n < args.steps or n < args.ref_steps or n < args.perm_steps:
        raise ValueError(
            f"Insufficient samples loaded ({n}) for steps={args.steps}, "
            f"ref_steps={args.ref_steps}, perm_steps={args.perm_steps}."
        )

    # 1. Single-thread C++ full audit & invariant checks across all 7 candidate modes (0..6)
    invariants_failed = 0
    mismatches: list[str] = []
    cand_mode_counts: dict[str, int] = {m: 0 for m in REQUIRED_CAND_MODES}
    t0 = time.perf_counter()
    for start in range(0, n, args.batch_size):
        end = min(n, start + args.batch_size)
        out = derive_structured_v2_batch(sp[start:end], nu[start:end], pr[start:end], ca[start:end])
        tf = out["v2_tile_feats"]
        if (
            torch.isnan(tf).any()
            or torch.isnan(out["v2_global_feats"]).any()
            or torch.isnan(out["v2_player_feats"]).any()
            or torch.isnan(out["v2_cand_dyn"]).any()
        ):
            invariants_failed += 1
        h_cnt = tf[:, :, 0:4].sum(dim=-1)
        u_cnt = tf[:, :, 14:18].sum(dim=-1)
        if (h_cnt + u_cnt > 4.0).any():
            invariants_failed += 1
        if (out["v2_event_len"] < 1).any():
            invariants_failed += 1
        k_cnt = out["v2_cand_mask"].sum(dim=-1)
        if (k_cnt < 1).any() or (k_cnt > 30).any():
            invariants_failed += 1
        cand_mode = out["v2_cand_mode"]
        cand_mask = out["v2_cand_mask"].bool()
        if not torch.equal(cand_mode == 0, ~cand_mask):
            invariants_failed += 1
            bad_pos = ((cand_mode == 0) != (~cand_mask)).nonzero(as_tuple=False)[0]
            bad_row = start + int(bad_pos[0].item())
            mismatches.append(
                f"[invariant] cand_mode/mask mismatch at sample_idx={bad_row} uuid={uuids[bad_row]} slot={int(bad_pos[1].item())}"
            )
        if ((cand_mode < 0) | (cand_mode > 6)).any():
            invariants_failed += 1
        for m in range(7):
            cand_mode_counts[str(m)] += int((cand_mode == m).sum().item())
    t1 = time.perf_counter()
    single_thread_sps = n / max(1e-9, t1 - t0)

    # 2. Multi-thread GIL-released throughput benchmark
    chunks = [
        (
            sp[s:min(n, s + args.batch_size)],
            nu[s:min(n, s + args.batch_size)],
            pr[s:min(n, s + args.batch_size)],
            ca[s:min(n, s + args.batch_size)],
        )
        for s in range(0, n, args.batch_size)
    ]
    t2 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as ex:
        list(ex.map(lambda c: derive_structured_v2_batch(c[0], c[1], c[2], c[3]), chunks))
    t3 = time.perf_counter()
    mt_sps = n / max(1e-9, t3 - t2)

    # 3. Reference parity on first `args.ref_steps` real samples (both hint=False and hint=True)
    ref_n = args.ref_steps
    ref_verified_mode_counts: dict[str, int] = {m: 0 for m in REQUIRED_CAND_MODES}
    ref_verified_cat_counts: dict[str, int] = {c: 0 for c in REQUIRED_RULE_CATEGORIES}
    if ref_n > 0:
        max_diffs, mm_false = _compare_cpp_and_py(
            uuids[:ref_n], sp[:ref_n], nu[:ref_n], pr[:ref_n], ca[:ref_n],
            enable_early_shape_hint=False, tag="head", workers=args.ref_workers,
        )
        max_diffs_hint, mm_true = _compare_cpp_and_py(
            uuids[:ref_n], sp[:ref_n], nu[:ref_n], pr[:ref_n], ca[:ref_n],
            enable_early_shape_hint=True, tag="head", workers=args.ref_workers,
        )
        mismatches.extend(mm_false)
        mismatches.extend(mm_true)
        _accumulate_slot_modes_and_categories(
            uuids[:ref_n],
            sp[:ref_n],
            nu[:ref_n],
            pr[:ref_n],
            ca[:ref_n],
            mode_counts=ref_verified_mode_counts,
            cat_counts=ref_verified_cat_counts,
        )
    else:
        max_diffs = {k: 0.0 for k in _AUDIT_TENSORS}
        max_diffs_hint = {k: 0.0 for k in _AUDIT_TENSORS}

    # 4. S_3 suit permutation equivariance on real samples with exact UUID & coordinate preservation
    perm_n = args.perm_steps
    perm_max_diff = 0.0
    if perm_n > 0:
        base_perm = derive_structured_v2_batch(
            sp[:perm_n], nu[:perm_n], pr[:perm_n], ca[:perm_n],
            enable_early_shape_hint=True, include_room_level=False,
        )
        for k in range(6):
            p_sp, p_nu, p_pr, p_ca, _ = permute_raw_four_fields(
                sp[:perm_n], nu[:perm_n], pr[:perm_n], ca[:perm_n], perm_idx=k
            )
            raw_p = derive_structured_v2_batch(
                p_sp, p_nu, p_pr, p_ca, enable_early_shape_hint=True, include_room_level=False
            )
            ten_p = permute_structured_batch(base_perm, k)
            if not torch.equal(p_ca.long(), raw_p["v2_cand_ids"].long()):
                mismatches.append(f"[suit_perm k={k}] raw permuted candidates != v2_cand_ids")
            for tk in _AUDIT_TENSORS:
                if raw_p[tk].shape != ten_p[tk].shape:
                    perm_max_diff = float("inf")
                    mismatches.append(
                        f"[suit_perm k={k}] tensor={tk} shape mismatch: raw={tuple(raw_p[tk].shape)} "
                        f"vs tensor={tuple(ten_p[tk].shape)}"
                    )
                    continue
                diff = (raw_p[tk].double() - ten_p[tk].double()).abs()
                d = float(diff.max().item()) if diff.numel() > 0 else 0.0
                perm_max_diff = max(perm_max_diff, d)
                tol = 0.0 if tk in _INT_EXACT_TENSORS else 1e-5
                if d > tol or not np.isfinite(d):
                    flat_i = int(diff.argmax().item())
                    coords = np.unravel_index(flat_i, tuple(diff.shape))
                    row_i = int(coords[0])
                    row_u = uuids[row_i] if row_i < len(uuids) else "<unknown>"
                    mismatches.append(
                        f"[suit_perm k={k}] tensor={tk} sample_idx={row_i} uuid={row_u} "
                        f"coord={tuple(int(c) for c in coords)} raw_p={raw_p[tk][coords].item()} "
                        f"ten_p={ten_p[tk][coords].item()} diff={d} tol={tol}"
                    )

    # 5. Rare-category scan & Python-reference parity audit
    harvested_count = 0
    category_counts_in_scan: dict[str, int] = {c: 0 for c in REQUIRED_RULE_CATEGORIES}
    harvest_max_diff = 0.0
    if args.harvest_categories:
        h_uuids, h_sp, h_nu, h_pr, h_ca, category_counts_in_scan = harvest_rare_categories(
            args.data, scan_limit=args.scan_limit
        )
        harvested_count = len(h_uuids)
        if harvested_count > 0:
            hd_f, hmm_f = _compare_cpp_and_py(
                h_uuids, h_sp, h_nu, h_pr, h_ca, enable_early_shape_hint=False, tag="harvest", workers=args.ref_workers
            )
            hd_t, hmm_t = _compare_cpp_and_py(
                h_uuids, h_sp, h_nu, h_pr, h_ca, enable_early_shape_hint=True, tag="harvest", workers=args.ref_workers
            )
            harvest_max_diff = max(*hd_f.values(), *hd_t.values())
            mismatches.extend(hmm_f)
            mismatches.extend(hmm_t)
            _accumulate_slot_modes_and_categories(
                h_uuids,
                h_sp,
                h_nu,
                h_pr,
                h_ca,
                mode_counts=ref_verified_mode_counts,
                cat_counts=ref_verified_cat_counts,
            )

    unverified_modes = [
        m for m in REQUIRED_CAND_MODES if ref_verified_mode_counts.get(m, 0) == 0
    ]
    unverified_categories = [
        c for c in REQUIRED_RULE_CATEGORIES if ref_verified_cat_counts.get(c, 0) == 0
    ]

    max_ref_diff = max(
        max(max_diffs.values()),
        max(max_diffs_hint.values()),
        harvest_max_diff,
        perm_max_diff,
    )

    report = {
        "data": str(args.data),
        "audited_steps": n,
        "ref_checked_steps": ref_n,
        "perm_checked_steps": perm_n,
        "harvested_rare_steps": harvested_count,
        "invariants_failed": invariants_failed,
        "single_thread_steps_per_sec": round(single_thread_sps, 1),
        "single_thread_us_per_step": round(1e6 / single_thread_sps, 2),
        f"threads_{args.threads}_steps_per_sec": round(mt_sps, 1),
        "cand_mode_counts": cand_mode_counts,
        "ref_verified_cand_mode_counts": ref_verified_mode_counts,
        "category_counts_in_scan": category_counts_in_scan,
        "ref_verified_category_counts": ref_verified_cat_counts,
        "unverified_modes": unverified_modes,
        "unverified_categories": unverified_categories,
        "max_ref_diff": max_ref_diff,
        "max_diffs_by_tensor_hint_false": max_diffs,
        "max_diffs_by_tensor_hint_true": max_diffs_hint,
        "suit_perm_max_diff": perm_max_diff,
        "harvest_max_diff": harvest_max_diff,
        "mismatches": mismatches[:20],
    }
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    missing_required_coverage = bool(unverified_modes) or (
        bool(args.harvest_categories) and bool(unverified_categories)
    )
    if (
        invariants_failed > 0
        or not np.isfinite(max_ref_diff)
        or max_ref_diff >= 1e-5
        or mismatches
        or (missing_required_coverage and not args.allow_unverified_categories)
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
