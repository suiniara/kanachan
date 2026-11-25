"""Audit C++ structured_v2 encoder against real source-v3 data & Python reference."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from kanachan.structured_v2 import (
    derive_structured_v2_batch,
    permute_raw_four_fields,
    permute_structured_batch,
)
from kanachan.structured_v2_ref import derive_structured_v2_batch_py


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


def _open_tsv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _parse_sample_line(line: str) -> tuple[str, list[int], list[int], list[int], list[int], list[int], list[int]] | None:
    parts = line.rstrip("\r\n").split("\t")
    if len(parts) < 5:
        return None
    uuid = parts[0].strip()
    sp_raw = [int(x) for x in parts[1].split(",")]
    nu = [int(x) for x in parts[2].split(",")]
    pr_raw = [int(x) for x in parts[3].split(",")]
    ca_raw = [int(x) for x in parts[4].split(",")]
    sp = (sp_raw + [510] * 29)[:29]
    pr = (pr_raw + [2165] * 109)[:109]
    ca = (ca_raw + [546] * 32)[:32]
    return uuid, sp, nu, pr, ca, pr_raw, ca_raw


def load_samples(path: Path, max_samples: int):
    uuids, sp_list, nu_list, pr_list, ca_list = [], [], [], [], []
    with _open_tsv(path) as f:
        for line in f:
            parsed = _parse_sample_line(line)
            if parsed is None:
                continue
            uuid, sp, nu, pr, ca, _pr_raw, _ca_raw = parsed
            uuids.append(uuid)
            sp_list.append(sp)
            nu_list.append(nu)
            pr_list.append(pr)
            ca_list.append(ca)
            if len(sp_list) >= max_samples:
                break
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
):
    """Harvest diverse real Mahjong records covering melds, kans, red5s, riichi, wins, kyushu, west, left0."""
    targets: dict[str, int] = {}
    selected_idx: set[int] = set()
    rows: list[tuple[str, list[int], list[int], list[int], list[int]]] = []
    category_counts: dict[str, int] = {}

    with _open_tsv(path) as f:
        for line_idx, line in enumerate(f):
            if line_idx >= scan_limit:
                break
            parsed = _parse_sample_line(line)
            if parsed is None:
                continue
            uuid, sp, nu, pr, ca, pr_raw, ca_raw = parsed
            left = sp[9] - 82
            chang = sp[7] - 75
            ju = sp[8] - 78
            n_dora = sum(1 for x in sp if 152 <= x < 337)
            cats: list[str] = []
            if any(x % 10 == 0 and x < 30 for x in [
                *((t - 152) % 37 for t in sp if 152 <= t < 337),
            ]) or any((t - 337) in (0, 36, 72) for t in sp if 337 <= t < 473):
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
            if 220 in ca_raw:
                cats.append("kyushu")
            if 219 in ca_raw or any(543 <= c < 546 for c in ca_raw):
                cats.append("win")
            if chang >= 2 or (chang == 1 and ju == 3):
                cats.append("west_or_all_last")
            if left == 0:
                cats.append("left0")

            keep = False
            for cat in cats:
                category_counts[cat] = category_counts.get(cat, 0) + 1
                if targets.get(cat, 0) < per_cat_target:
                    keep = True
                    targets[cat] = targets.get(cat, 0) + 1
            if keep:
                selected_idx.add(len(rows))
                rows.append((uuid, sp, nu, pr, ca))

    if not rows:
        return [], torch.empty((0, 29), dtype=torch.int32), torch.empty((0, 6), dtype=torch.int32), torch.empty((0, 109), dtype=torch.int32), torch.empty((0, 32), dtype=torch.int32), category_counts
    return (
        [r[0] for r in rows],
        torch.tensor([r[1] for r in rows], dtype=torch.int32),
        torch.tensor([r[2] for r in rows], dtype=torch.int32),
        torch.tensor([r[3] for r in rows], dtype=torch.int32),
        torch.tensor([r[4] for r in rows], dtype=torch.int32),
        category_counts,
    )


def _compare_cpp_and_py(
    uuids: list[str],
    sp: torch.Tensor,
    nu: torch.Tensor,
    pr: torch.Tensor,
    ca: torch.Tensor,
    *,
    enable_early_shape_hint: bool,
    tag: str,
) -> tuple[dict[str, float], list[str]]:
    ref_cpp = derive_structured_v2_batch(
        sp, nu, pr, ca, enable_early_shape_hint=enable_early_shape_hint, include_room_level=False
    )
    ref_py = derive_structured_v2_batch_py(
        sp.numpy(),
        nu.numpy(),
        pr.numpy(),
        ca.numpy(),
        enable_early_shape_hint=enable_early_shape_hint,
    )
    max_diffs: dict[str, float] = {}
    mismatches: list[str] = []
    for k in _AUDIT_TENSORS:
        diff = np.abs(ref_cpp[k].numpy().astype(np.float64) - ref_py[k].astype(np.float64))
        max_val = float(diff.max()) if diff.size > 0 else 0.0
        max_diffs[k] = max_val
        if max_val >= 1e-5:
            flat_idx = int(diff.argmax())
            coords = np.unravel_index(flat_idx, diff.shape)
            row_i = int(coords[0])
            row_uuid = uuids[row_i] if row_i < len(uuids) else "<unknown>"
            mismatches.append(
                f"[{tag} hint={enable_early_shape_hint}] tensor={k} sample_idx={row_i} "
                f"uuid={row_uuid} coord={tuple(int(c) for c in coords)} "
                f"cpp={ref_cpp[k].numpy()[coords]} py={ref_py[k][coords]} diff={max_val}"
            )
    return max_diffs, mismatches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--ref-steps", type=int, default=1000)
    parser.add_argument("--perm-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--harvest-categories", action="store_true")
    parser.add_argument("--scan-limit", type=int, default=120000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    uuids, sp, nu, pr, ca = load_samples(args.data, args.steps)
    n = int(sp.shape[0])
    assert n >= args.steps or n > 0, "No samples loaded"

    # 1. Single-thread C++ full audit & invariant checks
    invariants_failed = 0
    cand_mode_counts: dict[str, int] = {str(m): 0 for m in range(7)}
    t0 = time.perf_counter()
    for start in range(0, n, args.batch_size):
        end = min(n, start + args.batch_size)
        out = derive_structured_v2_batch(sp[start:end], nu[start:end], pr[start:end], ca[start:end])
        tf = out["v2_tile_feats"]
        if torch.isnan(tf).any() or torch.isnan(out["v2_global_feats"]).any() or \
           torch.isnan(out["v2_player_feats"]).any() or torch.isnan(out["v2_cand_dyn"]).any():
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
        modes_masked = out["v2_cand_mode"][out["v2_cand_mask"]]
        for m in range(7):
            cand_mode_counts[str(m)] += int((modes_masked == m).sum().item())
    t1 = time.perf_counter()
    single_thread_sps = n / max(1e-9, t1 - t0)

    # 2. Multi-thread GIL-released throughput benchmark
    chunks = [
        (sp[s:min(n, s + args.batch_size)],
         nu[s:min(n, s + args.batch_size)],
         pr[s:min(n, s + args.batch_size)],
         ca[s:min(n, s + args.batch_size)])
        for s in range(0, n, args.batch_size)
    ]
    t2 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as ex:
        list(ex.map(lambda c: derive_structured_v2_batch(c[0], c[1], c[2], c[3]), chunks))
    t3 = time.perf_counter()
    mt_sps = n / max(1e-9, t3 - t2)

    # 3. Reference parity on first `args.ref_steps` real samples (both hint=False and hint=True)
    ref_n = min(n, args.ref_steps)
    mismatches: list[str] = []
    max_diffs, mm_false = _compare_cpp_and_py(
        uuids[:ref_n], sp[:ref_n], nu[:ref_n], pr[:ref_n], ca[:ref_n],
        enable_early_shape_hint=False, tag="head",
    )
    max_diffs_hint, mm_true = _compare_cpp_and_py(
        uuids[:ref_n], sp[:ref_n], nu[:ref_n], pr[:ref_n], ca[:ref_n],
        enable_early_shape_hint=True, tag="head",
    )
    mismatches.extend(mm_false)
    mismatches.extend(mm_true)

    # 4. S_3 suit permutation equivariance on real samples
    perm_n = min(n, args.perm_steps)
    base_perm = derive_structured_v2_batch(
        sp[:perm_n], nu[:perm_n], pr[:perm_n], ca[:perm_n],
        enable_early_shape_hint=True, include_room_level=False,
    )
    perm_max_diff = 0.0
    for k in range(6):
        p_sp, p_nu, p_pr, p_ca, _ = permute_raw_four_fields(
            sp[:perm_n], nu[:perm_n], pr[:perm_n], ca[:perm_n], perm_idx=k
        )
        raw_p = derive_structured_v2_batch(
            p_sp, p_nu, p_pr, p_ca, enable_early_shape_hint=True, include_room_level=False
        )
        ten_p = permute_structured_batch(base_perm, k)
        for tk in _AUDIT_TENSORS:
            d = float((raw_p[tk].double() - ten_p[tk].double()).abs().max().item())
            perm_max_diff = max(perm_max_diff, d)
            if d >= 1e-5:
                mismatches.append(f"[suit_perm k={k}] tensor={tk} max_diff={d}")

    # 5. Optional rare-category scan & parity audit
    harvested_count = 0
    category_counts: dict[str, int] = {}
    harvest_max_diff = 0.0
    if args.harvest_categories:
        h_uuids, h_sp, h_nu, h_pr, h_ca, category_counts = harvest_rare_categories(
            args.data, scan_limit=args.scan_limit
        )
        harvested_count = len(h_uuids)
        if harvested_count > 0:
            hd_f, hmm_f = _compare_cpp_and_py(
                h_uuids, h_sp, h_nu, h_pr, h_ca, enable_early_shape_hint=False, tag="harvest"
            )
            hd_t, hmm_t = _compare_cpp_and_py(
                h_uuids, h_sp, h_nu, h_pr, h_ca, enable_early_shape_hint=True, tag="harvest"
            )
            harvest_max_diff = max(*hd_f.values(), *hd_t.values())
            mismatches.extend(hmm_f)
            mismatches.extend(hmm_t)

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
        "category_counts_in_scan": category_counts,
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

    if invariants_failed > 0 or max_ref_diff >= 1e-5 or mismatches:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
