"""Read-only, bounded afterstate audit on BC/RL rows, including compressed data."""

import argparse
import bz2
import gzip
import hashlib
import json
from pathlib import Path
import time

import torch

from kanachan.candidate_features import (
    FEATURE_WIDTH,
    _cached_row,
    candidate_feature_schema,
    derive_candidate_features,
)


def audit(path, limit, batch_size, *, min_rows=1):
    if limit <= 0 or batch_size <= 0:
        raise ValueError("Audit row limit and batch size must be positive.")
    if min_rows <= 0 or min_rows > limit:
        raise ValueError(
            "Minimum audit rows must be positive and no greater than the row limit."
        )
    opener = (
        gzip.open
        if path.suffix == ".gz"
        else bz2.open
        if path.suffix == ".bz2"
        else open
    )
    digest = hashlib.sha256()
    report = {
        "rows": 0,
        "min_rows": min_rows,
        "row_limit": limit,
        "discard_rows": 0,
        "discard_candidates": 0,
        "unsupported_candidates": 0,
        "cold_seconds": 0.0,
        "warm_seconds": 0.0,
        "schema": candidate_feature_schema(),
    }
    pending = []

    def check(rows):
        columns = list(zip(*rows))
        sparse, numeric, history, candidates = (
            torch.tensor(values, dtype=torch.int32) for values in columns
        )
        started = time.perf_counter()
        features = derive_candidate_features(sparse, history, candidates)
        report["cold_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        repeated = derive_candidate_features(sparse, history, candidates)
        report["warm_seconds"] += time.perf_counter() - started
        if not torch.equal(features, repeated):
            raise ValueError("Warm-cache features differ.")
        mask = candidates < 148
        if features.shape != (
            *candidates.shape,
            FEATURE_WIDTH,
        ) or not torch.equal(features[..., 0].bool(), mask):
            raise ValueError("Discard feature mask or dimensions differ.")
        if (
            not torch.isfinite(features).all()
            or (features < 0).any()
            or (features > 1).any()
        ):
            raise ValueError("Nonfinite or out-of-range normalized features.")
        report["rows"] += len(rows)
        report["discard_rows"] += int(mask.any(-1).sum())
        report["discard_candidates"] += int(mask.sum())
        report["unsupported_candidates"] += int(
            ((candidates >= 148) & (candidates < 546)).sum()
        )

    _cached_row.cache_clear()
    signature = (path.stat().st_size, path.stat().st_mtime_ns)
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream):
            if index >= limit:
                break
            digest.update(line.encode("utf-8"))
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) not in (8, 10, 12):
                raise ValueError(f"Row {index}: expected v2 BC/RL fields.")
            values = [
                list(map(int, field.split(","))) for field in fields[1:5]
            ]
            for value, width, padding in (
                (values[0], 29, 510),
                (values[1], 6, 0),
                (values[2], 109, 2165),
                (values[3], 32, 546),
            ):
                if len(value) > width:
                    raise ValueError(f"Row {index}: invalid feature width.")
                value.extend([padding] * (width - len(value)))
            pending.append(values)
            if len(pending) == batch_size:
                check(pending)
                pending = []
        if pending:
            check(pending)
    if signature != (path.stat().st_size, path.stat().st_mtime_ns):
        raise ValueError("Audit input changed during the read.")
    if report["rows"] < min_rows:
        raise ValueError(
            f"Insufficient audit coverage: checked {report['rows']} rows; require at least {min_rows}."
        )
    report.update(
        status="passed",
        source_name=path.name,
        source_size=signature[0],
        uncompressed_prefix_sha256=digest.hexdigest(),
        cache=_cached_row.cache_info()._asdict(),
        rows_per_second=report["rows"] / max(report["cold_seconds"], 1e-12),
        timing_scope="CPU feature derivation and canonical cache only; no training or playing-strength benchmark",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Required coverage; set equal to --limit for a release gate.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = audit(
        args.input, args.limit, args.batch_size, min_rows=args.min_rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "rows",
                    "discard_rows",
                    "discard_candidates",
                    "rows_per_second",
                )
            }
        )
    )
