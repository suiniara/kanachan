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

from kanachan.structured_v2 import derive_structured_v2_batch
from kanachan.structured_v2_ref import derive_structured_v2_batch_py


def _open_tsv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def load_samples(path: Path, max_samples: int):
    sp_list, nu_list, pr_list, ca_list = [], [], [], []
    with _open_tsv(path) as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 5:
                continue
            sp = [int(x) for x in parts[1].split(",")]
            nu = [int(x) for x in parts[2].split(",")]
            pr = [int(x) for x in parts[3].split(",")]
            ca = [int(x) for x in parts[4].split(",")]
            sp = (sp + [510] * 29)[:29]
            pr = (pr + [2165] * 109)[:109]
            ca = (ca + [546] * 32)[:32]
            sp_list.append(sp)
            nu_list.append(nu)
            pr_list.append(pr)
            ca_list.append(ca)
            if len(sp_list) >= max_samples:
                break
    return (
        torch.tensor(sp_list, dtype=torch.int32),
        torch.tensor(nu_list, dtype=torch.int32),
        torch.tensor(pr_list, dtype=torch.int32),
        torch.tensor(ca_list, dtype=torch.int32),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--ref-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sp, nu, pr, ca = load_samples(args.data, args.steps)
    n = int(sp.shape[0])
    assert n >= args.steps or n > 0, "No samples loaded"

    # 1. Single-thread C++ full audit & invariant checks
    invariants_failed = 0
    max_abs_sum = 0.0
    t0 = time.perf_counter()
    cpp_chunks = []
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
        max_abs_sum = max(max_abs_sum, float(out["v2_cand_dyn"].abs().max().item()))
        cpp_chunks.append(out)
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

    # 3. Reference parity on first `args.ref_steps` real samples
    ref_n = min(n, args.ref_steps)
    ref_cpp = derive_structured_v2_batch(sp[:ref_n], nu[:ref_n], pr[:ref_n], ca[:ref_n])
    ref_py = derive_structured_v2_batch_py(
        sp[:ref_n].numpy(), nu[:ref_n].numpy(), pr[:ref_n].numpy(), ca[:ref_n].numpy()
    )
    max_diffs = {}
    for k in (
        "v2_tile_feats", "v2_events", "v2_event_len",
        "v2_global_feats", "v2_player_feats",
        "v2_cand_mode", "v2_cand_dyn",
    ):
        max_diffs[k] = float(
            np.abs(ref_cpp[k].numpy().astype(np.float64) - ref_py[k].astype(np.float64)).max()
        )
    max_ref_diff = max(max_diffs.values())

    report = {
        "data": str(args.data),
        "audited_steps": n,
        "ref_checked_steps": ref_n,
        "invariants_failed": invariants_failed,
        "single_thread_steps_per_sec": round(single_thread_sps, 1),
        "single_thread_us_per_step": round(1e6 / single_thread_sps, 2),
        f"threads_{args.threads}_steps_per_sec": round(mt_sps, 1),
        "max_ref_diff": max_ref_diff,
        "max_diffs_by_tensor": max_diffs,
    }
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if invariants_failed > 0 or max_ref_diff >= 1e-5:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
