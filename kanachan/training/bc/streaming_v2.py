"""Bounded-memory external hash-bucket disk spooling, radix/fixed-fan-in sampler & prefetch (§2)."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
import contextlib
import gzip
import hashlib
import heapq
import json
import math
import operator
import queue
import shutil
import struct
import tempfile
import threading
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from kanachan.constants import (
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_SPARSE_FEATURES,
)
from kanachan.structured_v2 import PAD_ACTION_ID, derive_structured_v2_batch
from kanachan.training.core.dataset_fingerprint import (
    SOURCE_V3_BC_FORMAT_VERSION,
    combine_dataset_identity,
    compute_bc_preprocess_fingerprint,
    compute_bc_spool_recipe_fingerprint,
    compute_source_data_fingerprint,
    sha256_file_bytes,
)

_INDEX_STRUCT = struct.Struct("<QIIIIbB6x")  # 32 bytes/round
assert _INDEX_STRUCT.size == 32
_INDEX_DTYPE = np.dtype(
    [
        ("offset", "<u8"),
        ("rec_len", "<u4"),
        ("ev_len", "<u4"),
        ("obs_total", "<u4"),
        ("num_steps", "<u4"),
        ("final_rank", "i1"),
        ("obs_seat", "u1"),
        ("pad", "V6"),
    ]
)
assert _INDEX_DTYPE.itemsize == _INDEX_STRUCT.size

_REC_HDR_STRUCT = struct.Struct("<HHHHbB")  # 10 bytes header per round in rounds.bin
assert _REC_HDR_STRUCT.size == 10

_UNSORTED_INDEX_DTYPE = np.dtype(
    [
        ("first_line", "<u8"),
        ("idx_bytes", "V32"),
        ("rec_sha256", "V32"),
    ]
)
assert _UNSORTED_INDEX_DTYPE.itemsize == 72

_PART_REC_DTYPE = np.dtype(
    [
        ("key", "<u8"),
        ("round_id", "<u4"),
        ("ev_len", "<u2"),
        ("pad", "<u2"),
    ]
)
assert _PART_REC_DTYPE.itemsize == 16

_UNSORTED_BATCH_DTYPE = np.dtype(
    [
        ("bkey", "<u8"),
        ("id_offset", "<u8"),
        ("count", "<u4"),
        ("pad", "<u4"),
    ]
)
assert _UNSORTED_BATCH_DTYPE.itemsize == 24

_SCHED_STRUCT = struct.Struct("<QI")  # 12 bytes/batch: (id_offset, count)
assert _SCHED_STRUCT.size == 12
_SCHED_DTYPE = np.dtype([("id_offset", "<u8"), ("count", "<u4")])
assert _SCHED_DTYPE.itemsize == _SCHED_STRUCT.size
_ID_SIZE = 4  # <u4 round index in ids.bin
_SENTINEL = object()

_SM_C0 = np.uint64(0x9E3779B97F4A7C15)
_SM_C1 = np.uint64(0xBF58476D1CE4E5B9)
_SM_C2 = np.uint64(0x94D049BB133111EB)
_MASK64 = 0xFFFF_FFFF_FFFF_FFFF


def _splitmix64_scalar(x: int) -> int:
    z = (int(x) + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def _splitmix64_u64_vec(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x.astype(np.uint64, copy=True) + _SM_C0
        z = (z ^ (z >> np.uint64(30))) * _SM_C1
        z = (z ^ (z >> np.uint64(27))) * _SM_C2
        return z ^ (z >> np.uint64(31))


def _open_bc_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _resolve_bc_source_files(source: Path | Sequence[Path]) -> list[Path]:
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_file():
            return [p]
        if p.is_dir():
            files = sorted(
                item
                for item in p.iterdir()
                if item.is_file() and (item.name.endswith(".tsv") or item.name.endswith(".tsv.gz"))
            )
            if not files:
                raise FileNotFoundError(f"No .tsv/.tsv.gz files found in BC source directory {p}.")
            return files
        raise FileNotFoundError(f"BC source path does not exist: {p}")
    files = [Path(p) for p in source]
    if not files:
        raise ValueError("Empty BC source file sequence.")
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f"BC source file does not exist: {f}")
    return files


def _uuid_bucket(uuid: str, num_buckets: int, *, depth: int = 0) -> int:
    key = uuid.encode("utf-8") if depth == 0 else f"{depth}:{uuid}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "little") % num_buckets


def _flush_bucket_buffers(
    bucket_paths: list[Path], buffers: list[list[str]], buf_bytes: list[int]
) -> None:
    for idx, lines in enumerate(buffers):
        if not lines:
            continue
        with bucket_paths[idx].open("a", encoding="utf-8") as fp:
            fp.writelines(lines)
        lines.clear()
        buf_bytes[idx] = 0


def _spool_lines_to_buckets(
    lines_with_ids: Iterable[tuple[int, str]],
    bucket_dir: Path,
    *,
    num_buckets: int,
    prefix: str,
    depth: int = 0,
    flush_bytes: int = 128 * 1024,
) -> list[Path]:
    bucket_paths = [bucket_dir / f"{prefix}_d{depth}_b{i:04d}.tmp" for i in range(num_buckets)]
    buffers: list[list[str]] = [[] for _ in range(num_buckets)]
    buf_bytes: list[int] = [0] * num_buckets

    for line_no, raw_line in lines_with_ids:
        stripped = raw_line.rstrip("\r\n")
        if not stripped:
            raise ValueError(f"line {line_no}: Blank line in BC source.")
        tab_pos = stripped.find("\t")
        uuid = stripped[:tab_pos].strip() if tab_pos != -1 else ""
        if not uuid:
            raise ValueError(f"line {line_no}: Missing or empty game UUID before first tab.")
        b_idx = _uuid_bucket(uuid, num_buckets, depth=depth)
        entry = f"{line_no}\t{stripped}\n"
        buffers[b_idx].append(entry)
        buf_bytes[b_idx] += len(entry)
        if buf_bytes[b_idx] >= flush_bytes:
            with bucket_paths[b_idx].open("a", encoding="utf-8") as fp:
                fp.writelines(buffers[b_idx])
            buffers[b_idx].clear()
            buf_bytes[b_idx] = 0

    _flush_bucket_buffers(bucket_paths, buffers, buf_bytes)
    return [p for p in bucket_paths if p.exists() and p.stat().st_size > 0]


def _iter_leaf_buckets(
    initial_buckets: list[Path],
    bucket_dir: Path,
    *,
    max_bucket_bytes: int,
    depth: int = 0,
    max_depth: int = 4,
    sub_fanout: int = 16,
) -> Iterator[Path]:
    for idx, bpath in enumerate(initial_buckets):
        if not bpath.exists():
            continue
        sz = bpath.stat().st_size
        if sz == 0:
            bpath.unlink(missing_ok=True)
            continue
        if sz > max_bucket_bytes and depth < max_depth:
            def _sub_iter(path: Path) -> Iterable[tuple[int, str]]:
                with path.open("r", encoding="utf-8") as fp:
                    for raw in fp:
                        t1 = raw.find("\t")
                        if t1 <= 0:
                            raise ValueError(f"Corrupt spool bucket record in {path}.")
                        yield int(raw[:t1]), raw[t1 + 1 :]

            children = _spool_lines_to_buckets(
                _sub_iter(bpath),
                bucket_dir,
                num_buckets=sub_fanout,
                prefix=f"{bpath.stem}_sub{idx}",
                depth=depth + 1,
            )
            bpath.unlink(missing_ok=True)
            yield from _iter_leaf_buckets(
                children,
                bucket_dir,
                max_bucket_bytes=max_bucket_bytes,
                depth=depth + 1,
                max_depth=max_depth,
                sub_fanout=sub_fanout,
            )
        else:
            yield bpath


def _pack_round_record(rec: dict[str, Any]) -> bytes:
    """Serialize one validated ordered round into compact variable-length raw bytes."""
    sp_t: Tensor = rec["sp_t"]
    nu_t: Tensor = rec["nu_t"]
    pr_t: Tensor = rec["pr_t"]
    ca_t: Tensor = rec["ca_t"]
    act_t: Tensor = rec["act_t"]
    pr_lens: list[int] = rec["pr_lens"]
    t_steps = int(sp_t.shape[0])
    full_pr_len = int(pr_lens[-1])
    full_pr = pr_t[-1, :full_pr_len].numpy().astype(np.uint16, copy=False)

    sp_lens: list[int] = []
    sp_vals: list[np.ndarray] = []
    ca_lens: list[int] = []
    ca_vals: list[np.ndarray] = []
    for t in range(t_steps):
        sp_row = sp_t[t].numpy()
        valid_sp = sp_row[sp_row != NUM_TYPES_OF_SPARSE_FEATURES]
        sp_lens.append(int(valid_sp.shape[0]))
        sp_vals.append(valid_sp.astype(np.uint16, copy=False))

        ca_row = ca_t[t].numpy()
        valid_ca = ca_row[ca_row != PAD_ACTION_ID]
        ca_lens.append(int(valid_ca.shape[0]))
        ca_vals.append(valid_ca.astype(np.uint16, copy=False))

    sp_concat = np.concatenate(sp_vals) if sp_vals else np.empty((0,), dtype=np.uint16)
    ca_concat = np.concatenate(ca_vals) if ca_vals else np.empty((0,), dtype=np.uint16)
    uuid_bytes = str(rec["uuid"]).encode("utf-8")

    hdr = _REC_HDR_STRUCT.pack(
        t_steps,
        full_pr_len,
        int(rec["round_event_len"]),
        len(uuid_bytes),
        int(rec["final_rank"]) if rec["final_rank"] is not None else -1,
        int(rec["observer_seat"]),
    )
    parts = [
        hdr,
        np.asarray(sp_lens, dtype=np.uint8).tobytes(),
        sp_concat.tobytes(),
        nu_t.numpy().astype(np.int32, copy=False).tobytes(),
        np.asarray(pr_lens, dtype=np.uint16).tobytes(),
        full_pr.tobytes(),
        np.asarray(ca_lens, dtype=np.uint8).tobytes(),
        ca_concat.tobytes(),
        act_t.numpy().astype(np.uint8, copy=False).tobytes(),
        uuid_bytes,
    ]
    return b"".join(parts)


def _unpack_round_record(
    raw_bytes: bytes, observer_total_steps: int, round_event_len: int
) -> dict[str, Any]:
    """Reconstruct bit-exact 4-field padded tensors for one stored round."""
    buf = memoryview(raw_bytes)
    t_steps, full_pr_len, ev_len_hdr, uuid_len, final_rank, obs_seat = _REC_HDR_STRUCT.unpack_from(
        buf, 0
    )
    assert ev_len_hdr == round_event_len
    pos = _REC_HDR_STRUCT.size

    sp_lens = np.frombuffer(buf, dtype=np.uint8, count=t_steps, offset=pos)
    pos += t_steps
    total_sp = int(sp_lens.sum())
    sp_concat = np.frombuffer(buf, dtype=np.uint16, count=total_sp, offset=pos)
    pos += total_sp * 2

    nu_arr = np.frombuffer(buf, dtype=np.int32, count=t_steps * NUM_NUMERIC_FEATURES, offset=pos).reshape(
        t_steps, NUM_NUMERIC_FEATURES
    )
    pos += t_steps * NUM_NUMERIC_FEATURES * 4

    pr_lens = np.frombuffer(buf, dtype=np.uint16, count=t_steps, offset=pos)
    pos += t_steps * 2
    full_pr = np.frombuffer(buf, dtype=np.uint16, count=full_pr_len, offset=pos)
    pos += full_pr_len * 2

    ca_lens = np.frombuffer(buf, dtype=np.uint8, count=t_steps, offset=pos)
    pos += t_steps
    total_ca = int(ca_lens.sum())
    ca_concat = np.frombuffer(buf, dtype=np.uint16, count=total_ca, offset=pos)
    pos += total_ca * 2

    acts = np.frombuffer(buf, dtype=np.uint8, count=t_steps, offset=pos)
    pos += t_steps
    uuid_str = bytes(buf[pos : pos + uuid_len]).decode("utf-8")

    sp_t = torch.full(
        (t_steps, MAX_NUM_ACTIVE_SPARSE_FEATURES),
        NUM_TYPES_OF_SPARSE_FEATURES,
        dtype=torch.int32,
    )
    pr_t = torch.full(
        (t_steps, MAX_LENGTH_OF_PROGRESSION_FEATURES),
        NUM_TYPES_OF_PROGRESSION_FEATURES,
        dtype=torch.int32,
    )
    ca_t = torch.full(
        (t_steps, MAX_NUM_ACTION_CANDIDATES),
        PAD_ACTION_ID,
        dtype=torch.int32,
    )

    sp_off = 0
    ca_off = 0
    full_pr_t = torch.from_numpy(full_pr.astype(np.int32, copy=True))
    sp_concat_t = torch.from_numpy(sp_concat.astype(np.int32, copy=True))
    ca_concat_t = torch.from_numpy(ca_concat.astype(np.int32, copy=True))
    for t in range(t_steps):
        sl = int(sp_lens[t])
        if sl > 0:
            sp_t[t, :sl] = sp_concat_t[sp_off : sp_off + sl]
            sp_off += sl
        pl = int(pr_lens[t])
        if pl > 0:
            pr_t[t, :pl] = full_pr_t[:pl]
        cl = int(ca_lens[t])
        if cl > 0:
            ca_t[t, :cl] = ca_concat_t[ca_off : ca_off + cl]
            ca_off += cl

    return {
        "sp_t": sp_t,
        "nu_t": torch.from_numpy(nu_arr.copy()),
        "pr_t": pr_t,
        "ca_t": ca_t,
        "act_t": torch.from_numpy(acts.astype(np.int64, copy=True)),
        "final_rank": int(final_rank),
        "observer_total_steps": int(observer_total_steps),
        "observer_seat": int(obs_seat),
        "round_event_len": int(round_event_len),
        "uuid": uuid_str,
        "num_steps": int(t_steps),
    }


def _iter_run_records(
    fp: Any,
    dtype: np.dtype[Any],
    sort_fields: tuple[str, ...],
    chunk_records: int,
) -> Iterator[tuple[Any, bytes]]:
    rec_size = dtype.itemsize
    chunk_bytes = max(1, int(chunk_records)) * rec_size
    if len(sort_fields) == 1:
        f0 = sort_fields[0]
        while True:
            raw = fp.read(chunk_bytes)
            if not raw:
                break
            arr = np.frombuffer(raw, dtype=dtype)
            keys = arr[f0].tolist()
            mv = memoryview(raw)
            for idx, k0 in enumerate(keys):
                yield k0, bytes(mv[idx * rec_size : (idx + 1) * rec_size])
    elif len(sort_fields) == 2:
        f0, f1 = sort_fields
        while True:
            raw = fp.read(chunk_bytes)
            if not raw:
                break
            arr = np.frombuffer(raw, dtype=dtype)
            k0s = arr[f0].tolist()
            k1s = arr[f1].tolist()
            mv = memoryview(raw)
            for idx, (k0, k1) in enumerate(zip(k0s, k1s, strict=True)):
                yield (k0, k1), bytes(mv[idx * rec_size : (idx + 1) * rec_size])
    else:
        raise ValueError(f"Unsupported sort_fields arity {len(sort_fields)}: {sort_fields}.")


def _external_merge_sorted_runs(
    unsorted_path: Path,
    work_dir: Path,
    *,
    dtype: np.dtype[Any],
    sort_fields: tuple[str, ...],
    run_records: int,
    max_fan_in: int = 16,
    run_prefix: str = "run",
    emit_fn: Callable[[Iterator[bytes]], None],
) -> None:
    """External fixed-fan-in multi-pass merge sort keeping open FDs <= max_fan_in + 1."""
    if max_fan_in < 2:
        raise ValueError(f"max_fan_in must be >= 2, got {max_fan_in}.")
    rec_size = dtype.itemsize
    run_chunk_bytes = max(1, int(run_records)) * rec_size
    runs: list[Path] = []
    work_dir.mkdir(parents=True, exist_ok=True)
    key_get = operator.itemgetter(0)
    try:
        with unsorted_path.open("rb") as ufp:
            run_idx = 0
            while True:
                raw = ufp.read(run_chunk_bytes)
                if not raw:
                    break
                rarr = np.frombuffer(raw, dtype=dtype).copy()
                rarr.sort(order=list(sort_fields))
                if run_idx == 0 and (len(raw) < run_chunk_bytes or ufp.peek(1) == b""):
                    emit_fn(iter((rarr.tobytes(),)))
                    return
                rpath = work_dir / f"{run_prefix}_p0_{run_idx:06d}.bin"
                rarr.tofile(rpath)
                del rarr, raw
                runs.append(rpath)
                run_idx += 1
        unsorted_path.unlink(missing_ok=True)
        if not runs:
            emit_fn(iter(()))
            return

        read_recs = max(16, min(512, (256 * 1024) // max(1, max_fan_in * rec_size)))
        flush_bytes = 64 * 1024
        pass_idx = 1
        while len(runs) > max_fan_in:
            next_runs: list[Path] = []
            for g_idx in range(0, len(runs), max_fan_in):
                group = runs[g_idx : g_idx + max_fan_in]
                if len(group) == 1:
                    next_runs.append(group[0])
                    continue
                out_path = work_dir / f"{run_prefix}_p{pass_idx}_{len(next_runs):06d}.bin"
                with contextlib.ExitStack() as stack:
                    fps = [stack.enter_context(rp.open("rb")) for rp in group]
                    out_fp = stack.enter_context(out_path.open("wb"))
                    iters = [_iter_run_records(fp, dtype, sort_fields, read_recs) for fp in fps]
                    out_buf: list[bytes] = []
                    out_bytes = 0
                    for _k, rec_bytes in heapq.merge(*iters, key=key_get):
                        out_buf.append(rec_bytes)
                        out_bytes += rec_size
                        if out_bytes >= flush_bytes:
                            out_fp.writelines(out_buf)
                            out_buf.clear()
                            out_bytes = 0
                    if out_buf:
                        out_fp.writelines(out_buf)
                for rp in group:
                    rp.unlink(missing_ok=True)
                next_runs.append(out_path)
            runs = next_runs
            pass_idx += 1

        with contextlib.ExitStack() as stack:
            fps = [stack.enter_context(rp.open("rb")) for rp in runs]
            iters = [_iter_run_records(fp, dtype, sort_fields, read_recs) for fp in fps]

            def _chunked_merged() -> Iterator[bytes]:
                buf: list[bytes] = []
                buf_bytes = 0
                for _k, rec_bytes in heapq.merge(*iters, key=key_get):
                    buf.append(rec_bytes)
                    buf_bytes += rec_size
                    if buf_bytes >= flush_bytes:
                        yield b"".join(buf)
                        buf.clear()
                        buf_bytes = 0
                if buf:
                    yield b"".join(buf)

            emit_fn(_chunked_merged())
    finally:
        unsorted_path.unlink(missing_ok=True)
        for rp in runs:
            rp.unlink(missing_ok=True)


class DiskLengthBucketBatchSampler:
    """Bounded-memory, bounded-FD external disk length-bucketed batch sampler (§2).

    Both epoch schedule construction and steady-state rank iteration keep memory
    and file descriptors strictly independent of total dataset size ``N``:
    1. Streams ``index.bin`` in 4,096-record (~128 KB) chunks through recursive
       MSB SplitMix64 radix partitioning (at most ``radix <= max_merge_fan_in <= 16``
       concurrent open files, bounded partition size ``<= window_items``).
    2. Sorts one leaf partition at a time by random key, carries partial-pool tails across
       partitions (at most one partial batch globally when ``drop_last=False``),
       length-buckets each ``batch_size * bucket_mult`` pool, and streams batch round IDs
       to ``epoch_<e>_ids.bin`` + batch descriptors to disk.
    3. External multi-pass fixed fan-in merge sort (at most ``max_merge_fan_in + 1``
       concurrent open FDs and ``<= 320 KB`` merge buffers, ``O(log_{fan_in}(N))`` passes)
       writes fixed-width 12-byte ``epoch_<e>_schedule.bin``.
    4. Iteration and distributed rank seeking stream batches lazily via thread-local
       read handles (2 FDs per active reader thread) without full index arrays in RAM.
    """

    def __init__(
        self,
        index_path: Path,
        num_items: int,
        *,
        batch_size: int,
        seed: int = 0,
        bucket_mult: int = 8,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
        schedule_dir: Path | None = None,
        index_sha256: str = "",
        ordered_sample_sha256: str = "",
        window_items: int = 16384,
        max_merge_fan_in: int = 16,
        merge_run_batches: int | None = None,
        shared_schedule_across_ranks: bool | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if bucket_mult < 1:
            raise ValueError("bucket_mult must be >= 1.")
        if window_items < 1:
            raise ValueError("window_items must be >= 1.")
        if max_merge_fan_in < 2:
            raise ValueError(f"max_merge_fan_in must be >= 2, got {max_merge_fan_in}.")
        if merge_run_batches is not None and merge_run_batches < 1:
            raise ValueError(f"merge_run_batches must be >= 1, got {merge_run_batches}.")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"Invalid distributed sampler coordinates rank={rank}, world_size={world_size}.")

        self.index_path = Path(index_path)
        self.num_items = int(num_items)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.bucket_mult = int(bucket_mult)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.drop_last = bool(drop_last)
        self.window_items = int(window_items)
        self.max_merge_fan_in = int(max_merge_fan_in)
        self.merge_run_batches = int(merge_run_batches) if merge_run_batches is not None else None
        self.index_sha256 = index_sha256 or sha256_file_bytes(self.index_path)[1]
        if ordered_sample_sha256:
            self.ordered_sample_sha256 = str(ordered_sample_sha256)
        else:
            meta_sib = self.index_path.parent / "meta.json"
            resolved_ord = ""
            if meta_sib.is_file():
                try:
                    m_obj = json.loads(meta_sib.read_text(encoding="utf-8"))
                    if m_obj.get("index_sha256") == self.index_sha256 and m_obj.get("ordered_sample_sha256"):
                        resolved_ord = str(m_obj["ordered_sample_sha256"])
                except Exception:
                    pass
            self.ordered_sample_sha256 = resolved_ord

        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if schedule_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="kanachan_bc_sched_")
            self.schedule_dir = Path(self._temp_dir.name)
            self._shared_ddp = False
        else:
            self.schedule_dir = Path(schedule_dir)
            self.schedule_dir.mkdir(parents=True, exist_ok=True)
            if shared_schedule_across_ranks is None:
                self._shared_ddp = bool(
                    self.world_size > 1 and dist.is_available() and dist.is_initialized()
                )
            else:
                self._shared_ddp = bool(shared_schedule_across_ranks)

        self._tls = threading.local()
        self._handles_lock = threading.Lock()
        self._all_handles: list[tuple[Any, Any]] = []
        self._finalizer = weakref.finalize(self, self._cleanup_resources, self._all_handles, self._temp_dir)

        self.epoch: int = 0
        self.cursor: int = 0
        self.total_batches: int = 0
        self.num_rank_batches: int = 0
        self._sched_path: Path | None = None
        self._ids_path: Path | None = None
        self._meta_path: Path | None = None

        self._ensure_epoch_schedule(self.epoch)

    @staticmethod
    def _cleanup_resources(
        handles: list[tuple[Any, Any]],
        temp_dir: tempfile.TemporaryDirectory[str] | None,
    ) -> None:
        for sfp, ifp in handles:
            for fp in (sfp, ifp):
                try:
                    fp.close()
                except Exception:
                    pass
        handles.clear()
        if temp_dir is not None:
            temp_dir.cleanup()

    def close_readers(self) -> None:
        with self._handles_lock:
            for sfp, ifp in self._all_handles:
                for fp in (sfp, ifp):
                    try:
                        fp.close()
                    except Exception:
                        pass
            self._all_handles.clear()
        self._tls = threading.local()

    def close(self) -> None:
        self.close_readers()
        self._finalizer()

    def __enter__(self) -> DiskLengthBucketBatchSampler:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _epoch_paths(self, epoch: int) -> tuple[Path, Path, Path]:
        tag = f"s{self.seed}_b{self.batch_size}_m{self.bucket_mult}_d{int(self.drop_last)}_e{epoch}"
        return (
            self.schedule_dir / f"{tag}_schedule.bin",
            self.schedule_dir / f"{tag}_ids.bin",
            self.schedule_dir / f"{tag}_meta.json",
        )

    def _expected_schedule_meta(self, epoch: int) -> dict[str, Any]:
        return {
            "format": "kanachan_bc_disk_schedule_v2",
            "index_sha256": self.index_sha256,
            "ordered_sample_sha256": self.ordered_sample_sha256,
            "num_items": self.num_items,
            "batch_size": self.batch_size,
            "bucket_mult": self.bucket_mult,
            "drop_last": self.drop_last,
            "seed": self.seed,
            "epoch": int(epoch),
        }

    def _try_load_epoch_meta(self, epoch: int) -> bool:
        sched_p, ids_p, meta_p = self._epoch_paths(epoch)
        if not (meta_p.is_file() and sched_p.is_file() and ids_p.is_file()):
            return False
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:
            return False
        exp = self._expected_schedule_meta(epoch)
        for k, v in exp.items():
            if meta.get(k) != v:
                return False
        tot_batches = int(meta["total_batches"])
        if sched_p.stat().st_size != tot_batches * _SCHED_STRUCT.size:
            return False
        if ids_p.stat().st_size != int(meta["total_scheduled_ids"]) * _ID_SIZE:
            return False
        self._set_active_epoch_files(sched_p, ids_p, meta_p, tot_batches)
        return True

    def _set_active_epoch_files(
        self, sched_p: Path, ids_p: Path, meta_p: Path, total_batches: int
    ) -> None:
        self.close_readers()
        self._sched_path = sched_p
        self._ids_path = ids_p
        self._meta_path = meta_p
        self.total_batches = int(total_batches)
        if self.total_batches == 0:
            self.num_rank_batches = 0
        elif self.world_size == 1:
            self.num_rank_batches = self.total_batches
        elif self.drop_last:
            self.num_rank_batches = self.total_batches // self.world_size
        else:
            self.num_rank_batches = math.ceil(self.total_batches / self.world_size)

    def _build_epoch_schedule_disk(self, epoch: int) -> None:
        sched_p, ids_p, meta_p = self._epoch_paths(epoch)
        work_dir = self.schedule_dir / f".build_e{epoch}_{threading.get_ident()}"
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        tmp_sched = self.schedule_dir / f"{sched_p.name}.tmp"
        tmp_ids = self.schedule_dir / f"{ids_p.name}.tmp"
        tmp_meta = self.schedule_dir / f"{meta_p.name}.tmp"
        for p in (tmp_sched, tmp_ids, tmp_meta):
            p.unlink(missing_ok=True)

        try:
            if self.num_items == 0:
                tmp_sched.write_bytes(b"")
                tmp_ids.write_bytes(b"")
                meta_obj = {
                    **self._expected_schedule_meta(epoch),
                    "total_batches": 0,
                    "total_scheduled_ids": 0,
                }
                tmp_meta.write_text(json.dumps(meta_obj, sort_keys=True), encoding="utf-8")
                tmp_ids.replace(ids_p)
                tmp_sched.replace(sched_p)
                tmp_meta.replace(meta_p)
                self._set_active_epoch_files(sched_p, ids_p, meta_p, 0)
                return

            pool_sz = max(self.batch_size, self.batch_size * self.bucket_mult)
            part_target = max(self.window_items, 1)
            seed_mix = _splitmix64_scalar(self.seed + int(epoch) * 1009)
            radix_shift = max(1, min(4, self.max_merge_fan_in.bit_length() - 1))
            radix_fanout = 1 << radix_shift
            max_radix_depth = 64 // radix_shift
            node_counter = 0

            def _scatter_index_depth0() -> list[tuple[Path, int]]:
                read_chunk = 4096 * _INDEX_DTYPE.itemsize
                part_paths = [work_dir / f"radix_d0_b{b:02d}.bin" for b in range(radix_fanout)]
                part_bufs: list[list[bytes]] = [[] for _ in range(radix_fanout)]
                part_bytes = [0] * radix_fanout
                part_counts = [0] * radix_fanout
                pending_bytes = 0
                max_pending = 256 * 1024
                r_cursor = 0
                with contextlib.ExitStack() as stack:
                    open_fps: dict[int, Any] = {}

                    def _flush_b(bid: int) -> None:
                        nonlocal pending_bytes
                        chunks = part_bufs[bid]
                        if not chunks:
                            return
                        fp = open_fps.get(bid)
                        if fp is None:
                            fp = stack.enter_context(part_paths[bid].open("wb"))
                            open_fps[bid] = fp
                        fp.writelines(chunks)
                        chunks.clear()
                        pending_bytes -= part_bytes[bid]
                        part_bytes[bid] = 0

                    with self.index_path.open("rb") as idx_fp:
                        while True:
                            raw = idx_fp.read(read_chunk)
                            if not raw:
                                break
                            arr = np.frombuffer(raw, dtype=_INDEX_DTYPE)
                            cnt = int(arr.shape[0])
                            rids = np.arange(r_cursor, r_cursor + cnt, dtype=np.uint64)
                            r_cursor += cnt
                            keys = _splitmix64_u64_vec(np.uint64(seed_mix) ^ rids)
                            ev_lens = np.minimum(arr["ev_len"], np.uint32(65535)).astype(
                                np.uint16, copy=False
                            )
                            rec_arr = np.empty((cnt,), dtype=_PART_REC_DTYPE)
                            rec_arr["key"] = keys
                            rec_arr["round_id"] = rids.astype(np.uint32, copy=False)
                            rec_arr["ev_len"] = ev_lens
                            rec_arr["pad"] = np.uint16(0)

                            pids = (keys >> np.uint64(64 - radix_shift)) & np.uint64(radix_fanout - 1)
                            order = np.argsort(pids, kind="mergesort")
                            pids_s = pids[order]
                            rec_s = rec_arr[order]
                            uniq_pids, starts, counts = np.unique(
                                pids_s, return_index=True, return_counts=True
                            )
                            for p_val, st, ln in zip(
                                uniq_pids.tolist(), starts.tolist(), counts.tolist(), strict=True
                            ):
                                blob = rec_s[st : st + ln].tobytes()
                                part_bufs[p_val].append(blob)
                                part_bytes[p_val] += len(blob)
                                part_counts[p_val] += int(ln)
                                pending_bytes += len(blob)
                            if pending_bytes >= max_pending:
                                for bid in range(radix_fanout):
                                    _flush_b(bid)
                    for bid in range(radix_fanout):
                        _flush_b(bid)

                if r_cursor != self.num_items:
                    raise RuntimeError(
                        f"Index item count mismatch during schedule build: {r_cursor} != {self.num_items}."
                    )
                return [(part_paths[b], part_counts[b]) for b in range(radix_fanout) if part_counts[b] > 0]

            def _iter_sorted_partitions_dfs(bpath: Path, cnt: int, depth: int) -> Iterator[Path]:
                nonlocal node_counter
                if cnt <= part_target or depth >= max_radix_depth:
                    yield bpath
                    return
                nid = node_counter
                node_counter += 1
                sub_paths = [work_dir / f"radix_d{depth}_{nid}_b{b:02d}.bin" for b in range(radix_fanout)]
                sub_bufs: list[list[bytes]] = [[] for _ in range(radix_fanout)]
                sub_bytes = [0] * radix_fanout
                sub_counts = [0] * radix_fanout
                pending_bytes = 0
                max_pending = 256 * 1024
                shift_amt = np.uint64(64 - (depth + 1) * radix_shift)
                mask_u64 = np.uint64(radix_fanout - 1)
                read_chunk = 4096 * _PART_REC_DTYPE.itemsize
                with contextlib.ExitStack() as stack:
                    open_fps: dict[int, Any] = {}

                    def _flush_sub(bid: int) -> None:
                        nonlocal pending_bytes
                        chunks = sub_bufs[bid]
                        if not chunks:
                            return
                        fp = open_fps.get(bid)
                        if fp is None:
                            fp = stack.enter_context(sub_paths[bid].open("wb"))
                            open_fps[bid] = fp
                        fp.writelines(chunks)
                        chunks.clear()
                        pending_bytes -= sub_bytes[bid]
                        sub_bytes[bid] = 0

                    with bpath.open("rb") as bfp:
                        while True:
                            raw = bfp.read(read_chunk)
                            if not raw:
                                break
                            rarr = np.frombuffer(raw, dtype=_PART_REC_DTYPE)
                            pids = (rarr["key"] >> shift_amt) & mask_u64
                            order = np.argsort(pids, kind="mergesort")
                            pids_s = pids[order]
                            rec_s = rarr[order]
                            uniq_pids, starts, counts = np.unique(
                                pids_s, return_index=True, return_counts=True
                            )
                            for p_val, st, ln in zip(
                                uniq_pids.tolist(), starts.tolist(), counts.tolist(), strict=True
                            ):
                                blob = rec_s[st : st + ln].tobytes()
                                sub_bufs[p_val].append(blob)
                                sub_bytes[p_val] += len(blob)
                                sub_counts[p_val] += int(ln)
                                pending_bytes += len(blob)
                            if pending_bytes >= max_pending:
                                for bid in range(radix_fanout):
                                    _flush_sub(bid)
                    for bid in range(radix_fanout):
                        _flush_sub(bid)
                bpath.unlink(missing_ok=True)
                for bid in range(radix_fanout):
                    if sub_counts[bid] > 0:
                        yield from _iter_sorted_partitions_dfs(sub_paths[bid], sub_counts[bid], depth + 1)

            def _iter_leaf_partitions() -> Iterator[Path]:
                for top_path, top_cnt in _scatter_index_depth0():
                    yield from _iter_sorted_partitions_dfs(top_path, top_cnt, 1)

            # 2. Sort each bounded leaf partition by random key, length-bucket in pool_sz windows,
            #    and emit batch ID slices + unsorted batch descriptors on disk.
            unsorted_batches_p = work_dir / "unsorted_batches.bin"
            batch_seed = _splitmix64_scalar(seed_mix ^ 0xA5A5_5A5A_F0F0_0F0F)
            carry = np.empty((0,), dtype=_PART_REC_DTYPE)
            id_offset = 0
            total_batches = 0
            ids_buf: list[bytes] = []
            ids_buf_bytes = 0
            ub_buf: list[bytes] = []

            def _emit_batch_slice(sl: np.ndarray, ids_fp: Any, ub_fp: Any) -> None:
                nonlocal id_offset, total_batches, ids_buf_bytes
                cnt_b = int(sl.shape[0])
                id_bytes = sl["round_id"].astype("<u4", copy=False).tobytes()
                ids_buf.append(id_bytes)
                ids_buf_bytes += len(id_bytes)
                if ids_buf_bytes >= 262144:
                    ids_fp.writelines(ids_buf)
                    ids_buf.clear()
                    ids_buf_bytes = 0
                bkey = _splitmix64_scalar(batch_seed ^ total_batches)
                ub_buf.append(struct.pack("<QQII", bkey, id_offset, cnt_b, 0))
                if len(ub_buf) >= 4096:
                    ub_fp.writelines(ub_buf)
                    ub_buf.clear()
                id_offset += cnt_b
                total_batches += 1

            with tmp_ids.open("wb") as ids_fp, unsorted_batches_p.open("wb") as ub_fp:
                for ppath in _iter_leaf_partitions():
                    if not ppath.is_file() or ppath.stat().st_size == 0:
                        ppath.unlink(missing_ok=True)
                        continue
                    part_arr = np.fromfile(ppath, dtype=_PART_REC_DTYPE)
                    ppath.unlink(missing_ok=True)
                    part_arr.sort(order=["key", "round_id"])
                    if carry.shape[0] > 0:
                        work = np.concatenate([carry, part_arr])
                    else:
                        work = part_arr

                    full_n = (int(work.shape[0]) // pool_sz) * pool_sz
                    for s in range(0, full_n, pool_sz):
                        pool = work[s : s + pool_sz].copy()
                        pool.sort(order=["ev_len", "key", "round_id"])
                        for b in range(0, pool_sz, self.batch_size):
                            _emit_batch_slice(pool[b : b + self.batch_size], ids_fp, ub_fp)
                    carry = work[full_n:].copy()

                if carry.shape[0] > 0:
                    carry.sort(order=["ev_len", "key", "round_id"])
                    for b in range(0, int(carry.shape[0]), self.batch_size):
                        sl = carry[b : b + self.batch_size]
                        if self.drop_last and int(sl.shape[0]) < self.batch_size:
                            continue
                        _emit_batch_slice(sl, ids_fp, ub_fp)

                if ids_buf:
                    ids_fp.writelines(ids_buf)
                    ids_buf.clear()
                if ub_buf:
                    ub_fp.writelines(ub_buf)
                    ub_buf.clear()

            # 3. External multi-pass fixed-fan-in merge sort into fixed-record schedule.bin.
            run_batches = self.merge_run_batches or max(256, part_target // 2)
            if total_batches == 0:
                unsorted_batches_p.unlink(missing_ok=True)
                tmp_sched.write_bytes(b"")
            else:
                def _write_sched_chunks(chunks: Iterator[bytes]) -> None:
                    with tmp_sched.open("wb") as sfp:
                        for chunk in chunks:
                            if not chunk:
                                continue
                            arr = np.frombuffer(chunk, dtype=_UNSORTED_BATCH_DTYPE)
                            out_arr = np.empty((int(arr.shape[0]),), dtype=_SCHED_DTYPE)
                            out_arr["id_offset"] = arr["id_offset"]
                            out_arr["count"] = arr["count"]
                            sfp.write(out_arr.tobytes())

                _external_merge_sorted_runs(
                    unsorted_batches_p,
                    work_dir,
                    dtype=_UNSORTED_BATCH_DTYPE,
                    sort_fields=("bkey", "id_offset"),
                    run_records=run_batches,
                    max_fan_in=self.max_merge_fan_in,
                    run_prefix="batch_run",
                    emit_fn=_write_sched_chunks,
                )

            meta_obj = {
                **self._expected_schedule_meta(epoch),
                "total_batches": int(total_batches),
                "total_scheduled_ids": int(id_offset),
            }
            tmp_meta.write_text(json.dumps(meta_obj, sort_keys=True), encoding="utf-8")
            tmp_ids.replace(ids_p)
            tmp_sched.replace(sched_p)
            tmp_meta.replace(meta_p)
            self._set_active_epoch_files(sched_p, ids_p, meta_p, total_batches)
        finally:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            for p in (tmp_sched, tmp_ids, tmp_meta):
                p.unlink(missing_ok=True)

    def _ensure_epoch_schedule(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.cursor = 0
        if (
            self._shared_ddp
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            rank = dist.get_rank()
            sync_box: list[Any] = [None]
            if rank == 0:
                try:
                    if not self._try_load_epoch_meta(self.epoch):
                        self._build_epoch_schedule_disk(self.epoch)
                    sync_box[0] = {"ok": True, "total_batches": self.total_batches}
                except BaseException as exc:  # noqa: BLE001
                    sync_box[0] = {"ok": False, "err": f"{type(exc).__name__}: {exc}"}
            dist.broadcast_object_list(sync_box, src=0)
            msg = sync_box[0]
            if not isinstance(msg, dict) or not msg.get("ok"):
                err_txt = msg.get("err") if isinstance(msg, dict) else "unknown error"
                raise RuntimeError(f"Rank-0 DiskLengthBucketBatchSampler build failed: {err_txt}")
            if rank != 0:
                if not self._try_load_epoch_meta(self.epoch):
                    raise RuntimeError(
                        f"Shared disk batch schedule unavailable on rank {rank}: {self.schedule_dir}"
                    )
                if self.total_batches != int(msg["total_batches"]):
                    raise RuntimeError(
                        f"Shared disk batch schedule on rank {rank} disagrees with rank 0 batch count."
                    )
            return
        if self._try_load_epoch_meta(self.epoch):
            return
        self._build_epoch_schedule_disk(self.epoch)

    def _get_thread_handles(self) -> tuple[Any, Any]:
        pair = getattr(self._tls, "pair", None)
        if pair is None:
            assert self._sched_path is not None and self._ids_path is not None
            sfp = self._sched_path.open("rb")
            ifp = self._ids_path.open("rb")
            pair = (sfp, ifp)
            self._tls.pair = pair
            with self._handles_lock:
                self._all_handles.append(pair)
        return pair

    def get_rank_batch(self, pos: int) -> tuple[list[int], float]:
        """Seek a single batch by local rank cursor position in O(1) RAM."""
        if not 0 <= pos < self.num_rank_batches:
            raise IndexError(f"Rank batch pos={pos} out of range [0, {self.num_rank_batches}).")
        g_idx = pos * self.world_size + self.rank
        if g_idx < self.total_batches:
            sched_idx = g_idx
            weight = 1.0
        else:
            sched_idx = self.total_batches - 1
            weight = 0.0
        sfp, ifp = self._get_thread_handles()
        sfp.seek(sched_idx * _SCHED_STRUCT.size)
        id_off, count = _SCHED_STRUCT.unpack(sfp.read(_SCHED_STRUCT.size))
        ifp.seek(id_off * _ID_SIZE)
        idxs = list(np.frombuffer(ifp.read(count * _ID_SIZE), dtype="<u4").tolist())
        return idxs, weight

    def reset_epoch(self, epoch: int) -> None:
        self._ensure_epoch_schedule(int(epoch))

    def iter_pending_indexed_batches(self) -> Iterator[tuple[int, list[int], float]]:
        """Lazily stream ``(cursor_pos, round_indices, batch_weight)`` in O(1) RAM."""
        for pos in range(self.cursor, self.num_rank_batches):
            idxs, w = self.get_rank_batch(pos)
            yield pos, idxs, w

    def assigned_sample_indices(self) -> list[int]:
        """Return unique dataset indices assigned to this rank (for diagnostics/unit tests)."""
        out: list[int] = []
        for pos in range(self.num_rank_batches):
            idxs, w = self.get_rank_batch(pos)
            if w > 0.0:
                out.extend(idxs)
        return out

    def __len__(self) -> int:
        return self.num_rank_batches

    def iter_weighted_batches(self) -> Iterator[tuple[list[int], float]]:
        while self.cursor < self.num_rank_batches:
            idxs, weight = self.get_rank_batch(self.cursor)
            self.cursor += 1
            yield idxs, weight

    def __iter__(self) -> Iterator[list[int]]:
        while self.cursor < self.num_rank_batches:
            idxs, _ = self.get_rank_batch(self.cursor)
            self.cursor += 1
            yield idxs

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "kanachan_disk_sampler_v2",
            "index_sha256": self.index_sha256,
            "ordered_sample_sha256": self.ordered_sample_sha256,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "bucket_mult": self.bucket_mult,
            "num_items": self.num_items,
            "rank": self.rank,
            "world_size": self.world_size,
            "drop_last": self.drop_last,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format") != "kanachan_disk_sampler_v2":
            raise ValueError(
                f"Incompatible DiskLengthBucketBatchSampler state format: {state.get('format')!r}."
            )
        saved_ord = state.get("ordered_sample_sha256") or ""
        if self.ordered_sample_sha256 or saved_ord:
            if not self.ordered_sample_sha256 or saved_ord != self.ordered_sample_sha256:
                raise ValueError("DiskLengthBucketBatchSampler ordered_sample_sha256 mismatch on resume.")
        else:
            saved_sha = state.get("index_sha256")
            if saved_sha is not None and saved_sha != self.index_sha256:
                raise ValueError("DiskLengthBucketBatchSampler index_sha256 mismatch on resume.")
        if (
            int(state["seed"]) != self.seed
            or int(state["batch_size"]) != self.batch_size
            or int(state.get("bucket_mult", self.bucket_mult)) != self.bucket_mult
            or int(state["num_items"]) != self.num_items
            or int(state.get("rank", 0)) != self.rank
            or int(state.get("world_size", 1)) != self.world_size
            or bool(state.get("drop_last", False)) != self.drop_last
        ):
            raise ValueError("DiskLengthBucketBatchSampler config mismatch on resume.")
        target_epoch = int(state["epoch"])
        if target_epoch != self.epoch:
            self._ensure_epoch_schedule(target_epoch)
        self.cursor = int(state["cursor"])


class SpoolingRoundBCDataSource:
    """Out-of-core canonical-order hash-bucketed & disk-indexed round BC source (§2)."""

    def __init__(
        self,
        source: Path | Sequence[Path],
        *,
        cache_dir: Path | None = None,
        num_hash_buckets: int = 64,
        max_bucket_bytes: int = 64 * 1024 * 1024,
        max_rounds: int | None = None,
        allow_unlabeled: bool = False,
        allow_incomplete: bool = False,
        enable_early_shape_hint: bool = False,
        reuse_cache: bool = True,
        suit_augmentation: bool = False,
        aug_seed: int = 1337,
        sampler_seed: int = 42,
        batch_size: int | None = None,
        bucket_mult: int = 8,
        drop_last: bool = False,
        schema_version: str | None = None,
    ) -> None:
        if num_hash_buckets < 1:
            raise ValueError("num_hash_buckets must be >= 1.")
        if max_bucket_bytes < 512:
            raise ValueError("max_bucket_bytes must be >= 512.")

        self.source_files = _resolve_bc_source_files(source)
        self.num_hash_buckets = int(num_hash_buckets)
        self.max_bucket_bytes = int(max_bucket_bytes)
        self.max_rounds = max_rounds
        self.allow_unlabeled = bool(allow_unlabeled)
        self.allow_incomplete = bool(allow_incomplete)
        self.enable_early_shape_hint = bool(enable_early_shape_hint)
        self.schema_version = str(schema_version or SOURCE_V3_BC_FORMAT_VERSION)

        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if cache_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="kanachan_bc_spool_")
            self.cache_dir = Path(self._temp_dir.name)
            self._persistent_cache = False
        else:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._persistent_cache = True

        self.rounds_path = self.cache_dir / "rounds.bin"
        self.index_path = self.cache_dir / "index.bin"
        self.meta_path = self.cache_dir / "meta.json"

        is_ddp = bool(self._persistent_cache and dist.is_available() and dist.is_initialized())
        self.source_data_sha256, self.source_file_fingerprints = compute_source_data_fingerprint(
            self.source_files,
            source_format=self.schema_version,
            max_items=self.max_rounds,
            shared_across_ranks=is_ddp,
        )
        self.spool_recipe_sha256 = compute_bc_spool_recipe_fingerprint(
            allow_unlabeled=self.allow_unlabeled,
            allow_incomplete=self.allow_incomplete,
            enable_early_shape_hint=self.enable_early_shape_hint,
            max_rounds=self.max_rounds,
            schema_version=self.schema_version,
        )
        self.preprocess_recipe_sha256 = compute_bc_preprocess_fingerprint(
            allow_unlabeled=self.allow_unlabeled,
            allow_incomplete=self.allow_incomplete,
            enable_early_shape_hint=self.enable_early_shape_hint,
            max_rounds=self.max_rounds,
            suit_augmentation=suit_augmentation,
            aug_seed=aug_seed,
            sampler_seed=sampler_seed,
            batch_size=batch_size,
            bucket_mult=bucket_mult,
            drop_last=drop_last,
            schema_version=self.schema_version,
        )
        self.data_sha256 = combine_dataset_identity(
            self.source_data_sha256, self.preprocess_recipe_sha256
        )

        self.num_games: int = 0
        self.num_rounds: int = 0
        self.index_sha256: str = ""
        self.rounds_sha256: str = ""
        self.ordered_sample_sha256: str = ""
        self.index_bytes: int = 0
        self.rounds_bytes: int = 0

        self._tls = threading.local()
        self._handles_lock = threading.Lock()
        self._all_handles: list[tuple[Any, Any]] = []
        self._finalizer = weakref.finalize(self, self._cleanup_resources, self._all_handles, self._temp_dir)

        if is_ddp and dist.get_world_size() > 1:
            rank = dist.get_rank()
            sync_box: list[Any] = [None]
            if rank == 0:
                try:
                    if not (reuse_cache and self._try_load_cached_meta()):
                        self._build_disk_cache()
                    sync_box[0] = {
                        "ok": True,
                        "num_rounds": self.num_rounds,
                        "index_sha256": self.index_sha256,
                        "rounds_sha256": self.rounds_sha256,
                        "ordered_sample_sha256": self.ordered_sample_sha256,
                    }
                except BaseException as exc:  # noqa: BLE001
                    sync_box[0] = {"ok": False, "err": f"{type(exc).__name__}: {exc}"}
            dist.broadcast_object_list(sync_box, src=0)
            msg = sync_box[0]
            if not isinstance(msg, dict) or not msg.get("ok"):
                err_txt = msg.get("err") if isinstance(msg, dict) else "unknown error"
                raise RuntimeError(f"Rank-0 BC disk spool build failed: {err_txt}")
            if rank != 0:
                if not self._try_load_cached_meta():
                    raise RuntimeError(
                        f"Shared BC spool cache unavailable on rank {rank}: {self.cache_dir}"
                    )
                if (
                    self.num_rounds != int(msg["num_rounds"])
                    or self.index_sha256 != msg["index_sha256"]
                    or self.rounds_sha256 != msg["rounds_sha256"]
                    or self.ordered_sample_sha256 != msg["ordered_sample_sha256"]
                ):
                    raise RuntimeError(
                        f"Shared BC spool cache on rank {rank} disagrees with rank 0 metadata: {self.cache_dir}"
                    )
        elif reuse_cache and self._try_load_cached_meta():
            pass
        else:
            self._build_disk_cache()

    def bind_preprocess_config(
        self,
        *,
        suit_augmentation: bool,
        aug_seed: int,
        sampler_seed: int,
        batch_size: int | None,
        bucket_mult: int = 8,
        drop_last: bool = False,
        schema_version: str | None = None,
    ) -> None:
        """Bind full deterministic replay/augmentation parameters into preprocess fingerprint."""
        sv = str(schema_version or self.schema_version)
        self.preprocess_recipe_sha256 = compute_bc_preprocess_fingerprint(
            allow_unlabeled=self.allow_unlabeled,
            allow_incomplete=self.allow_incomplete,
            enable_early_shape_hint=self.enable_early_shape_hint,
            max_rounds=self.max_rounds,
            suit_augmentation=suit_augmentation,
            aug_seed=aug_seed,
            sampler_seed=sampler_seed,
            batch_size=batch_size,
            bucket_mult=bucket_mult,
            drop_last=drop_last,
            schema_version=sv,
        )
        self.data_sha256 = combine_dataset_identity(
            self.source_data_sha256, self.preprocess_recipe_sha256
        )

    @staticmethod
    def _cleanup_resources(
        handles: list[tuple[Any, Any]],
        temp_dir: tempfile.TemporaryDirectory[str] | None,
    ) -> None:
        for idx_fp, rds_fp in handles:
            for fp in (idx_fp, rds_fp):
                try:
                    fp.close()
                except Exception:
                    pass
        handles.clear()
        if temp_dir is not None:
            temp_dir.cleanup()

    def _try_load_cached_meta(self) -> bool:
        if not (self.meta_path.is_file() and self.index_path.is_file() and self.rounds_path.is_file()):
            return False
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if meta.get("format") != "kanachan_bc_round_spool_v3":
            return False
        if not meta.get("ordered_sample_sha256"):
            return False
        if meta.get("source_data_sha256") != self.source_data_sha256:
            return False
        if meta.get("spool_recipe_sha256") != self.spool_recipe_sha256:
            return False
        if meta.get("source_files") != self.source_file_fingerprints:
            return False
        for key, cur_val in (
            ("allow_unlabeled", self.allow_unlabeled),
            ("allow_incomplete", self.allow_incomplete),
            ("enable_early_shape_hint", self.enable_early_shape_hint),
            ("max_rounds", self.max_rounds),
            ("schema_version", self.schema_version),
        ):
            if meta.get(key) != cur_val:
                return False

        n_rounds = int(meta["num_rounds"])
        exp_idx_bytes = int(meta["index_bytes"])
        exp_rds_bytes = int(meta["rounds_bytes"])
        if exp_idx_bytes != n_rounds * _INDEX_STRUCT.size:
            return False
        if self.index_path.stat().st_size != exp_idx_bytes or self.rounds_path.stat().st_size != exp_rds_bytes:
            return False
        idx_sz, idx_sha = sha256_file_bytes(self.index_path)
        rds_sz, rds_sha = sha256_file_bytes(self.rounds_path)
        if idx_sz != exp_idx_bytes or idx_sha != meta.get("index_sha256"):
            return False
        if rds_sz != exp_rds_bytes or rds_sha != meta.get("rounds_sha256"):
            return False

        self.num_games = int(meta["num_games"])
        self.num_rounds = n_rounds
        self.index_sha256 = idx_sha
        self.rounds_sha256 = rds_sha
        self.ordered_sample_sha256 = str(meta["ordered_sample_sha256"])
        self.index_bytes = idx_sz
        self.rounds_bytes = rds_sz
        return True

    def _build_disk_cache(self) -> None:
        from kanachan.training.bc.trajectory_v2 import _validate_and_stage_bc_lines

        bucket_dir = self.cache_dir / "buckets_tmp"
        if bucket_dir.exists():
            shutil.rmtree(bucket_dir)
        bucket_dir.mkdir(parents=True, exist_ok=True)

        tmp_rounds = self.cache_dir / "rounds.bin.tmp"
        tmp_index = self.cache_dir / "index.bin.tmp"
        tmp_meta = self.cache_dir / "meta.json.tmp"
        tmp_unsorted_idx = self.cache_dir / "index.unsorted.tmp"
        for p in (tmp_rounds, tmp_index, tmp_meta, tmp_unsorted_idx):
            p.unlink(missing_ok=True)

        def _raw_source_iter() -> Iterable[tuple[int, str]]:
            global_line = 0
            for fpath in self.source_files:
                with _open_bc_text(fpath) as fp:
                    for raw in fp:
                        global_line += 1
                        yield global_line, raw

        rds_hasher = hashlib.sha256()
        idx_hasher = hashlib.sha256()
        ord_hasher = hashlib.sha256()
        num_rounds_total = 0
        num_games_total = 0
        seen_labeled_global: bool | None = None
        offset = 0

        try:
            top_buckets = _spool_lines_to_buckets(
                _raw_source_iter(),
                bucket_dir,
                num_buckets=self.num_hash_buckets,
                prefix="root",
                depth=0,
            )

            unsorted_buf: list[bytes] = []
            unsorted_bytes = 0
            with tmp_rounds.open("wb") as r_fp, tmp_unsorted_idx.open("wb") as uidx_fp:
                for leaf_path in _iter_leaf_buckets(
                    top_buckets,
                    bucket_dir,
                    max_bucket_bytes=self.max_bucket_bytes,
                ):
                    def _leaf_lines() -> Iterable[tuple[int, str]]:
                        with leaf_path.open("r", encoding="utf-8") as lfp:
                            for line in lfp:
                                tab = line.find("\t")
                                yield int(line[:tab]), line[tab + 1 :]

                    records, bucket_labeled, bucket_games = _validate_and_stage_bc_lines(
                        _leaf_lines(),
                        allow_unlabeled=self.allow_unlabeled,
                        allow_incomplete=self.allow_incomplete,
                        enable_early_shape_hint=self.enable_early_shape_hint,
                        derive_full_v2=False,
                    )
                    leaf_path.unlink(missing_ok=True)
                    if bucket_labeled is not None:
                        if seen_labeled_global is None:
                            seen_labeled_global = bucket_labeled
                        elif seen_labeled_global != bucket_labeled:
                            raise ValueError(
                                "BC source mixes 6-column unlabeled and 8-column labeled games across buckets."
                            )
                    num_games_total += int(bucket_games)

                    for rec in records:
                        raw_rec = _pack_round_record(rec)
                        rec_len = len(raw_rec)
                        r_fp.write(raw_rec)
                        rds_hasher.update(raw_rec)
                        rec_sha = hashlib.sha256(raw_rec).digest()
                        idx_entry = _INDEX_STRUCT.pack(
                            offset,
                            rec_len,
                            int(rec["round_event_len"]),
                            int(rec["observer_total_steps"]),
                            int(rec.get("num_steps", rec["sp_t"].shape[0])),
                            int(rec["final_rank"]) if rec["final_rank"] is not None else -1,
                            int(rec["observer_seat"]),
                        )
                        row_bytes = struct.pack("<Q", int(rec["first_line_no"])) + idx_entry + rec_sha
                        unsorted_buf.append(row_bytes)
                        unsorted_bytes += len(row_bytes)
                        if unsorted_bytes >= 128 * 1024:
                            uidx_fp.writelines(unsorted_buf)
                            unsorted_buf.clear()
                            unsorted_bytes = 0
                        offset += rec_len

                if unsorted_buf:
                    uidx_fp.writelines(unsorted_buf)
                    unsorted_buf.clear()

            # Canonicalize global sample order by source-file appearance (`first_line`) regardless of hash buckets.
            def _write_canonical_index(chunks: Iterator[bytes]) -> None:
                nonlocal num_rounds_total
                limit = int(self.max_rounds) if self.max_rounds is not None else None
                with tmp_index.open("wb") as idx_fp:
                    for chunk in chunks:
                        if not chunk:
                            continue
                        if limit is not None and num_rounds_total >= limit:
                            break
                        arr = np.frombuffer(chunk, dtype=_UNSORTED_INDEX_DTYPE)
                        for rec in arr:
                            if limit is not None and num_rounds_total >= limit:
                                break
                            idx_b = rec["idx_bytes"].tobytes()
                            rec_sha_b = rec["rec_sha256"].tobytes()
                            idx_fp.write(idx_b)
                            idx_hasher.update(idx_b)
                            ord_hasher.update(struct.pack("<Q", num_rounds_total))
                            ord_hasher.update(rec_sha_b)
                            num_rounds_total += 1

            _external_merge_sorted_runs(
                tmp_unsorted_idx,
                bucket_dir,
                dtype=_UNSORTED_INDEX_DTYPE,
                sort_fields=("first_line",),
                run_records=8192,
                max_fan_in=16,
                run_prefix="canon_idx",
                emit_fn=_write_canonical_index,
            )

            self.num_games = num_games_total
            self.num_rounds = num_rounds_total
            self.rounds_bytes = offset
            self.index_bytes = num_rounds_total * _INDEX_STRUCT.size
            self.rounds_sha256 = rds_hasher.hexdigest()
            self.index_sha256 = idx_hasher.hexdigest()
            self.ordered_sample_sha256 = ord_hasher.hexdigest()

            tmp_rounds.replace(self.rounds_path)
            tmp_index.replace(self.index_path)
            meta_obj = {
                "format": "kanachan_bc_round_spool_v3",
                "source_files": self.source_file_fingerprints,
                "source_data_sha256": self.source_data_sha256,
                "spool_recipe_sha256": self.spool_recipe_sha256,
                "schema_version": self.schema_version,
                "allow_unlabeled": self.allow_unlabeled,
                "allow_incomplete": self.allow_incomplete,
                "enable_early_shape_hint": self.enable_early_shape_hint,
                "max_rounds": self.max_rounds,
                "num_hash_buckets": self.num_hash_buckets,
                "max_bucket_bytes": self.max_bucket_bytes,
                "num_games": self.num_games,
                "num_rounds": self.num_rounds,
                "index_bytes": self.index_bytes,
                "rounds_bytes": self.rounds_bytes,
                "index_sha256": self.index_sha256,
                "rounds_sha256": self.rounds_sha256,
                "ordered_sample_sha256": self.ordered_sample_sha256,
                "data_sha256": self.data_sha256,
            }
            tmp_meta.write_text(json.dumps(meta_obj, indent=2, sort_keys=True), encoding="utf-8")
            tmp_meta.replace(self.meta_path)
        finally:
            if bucket_dir.exists():
                shutil.rmtree(bucket_dir, ignore_errors=True)
            for p in (tmp_rounds, tmp_index, tmp_meta, tmp_unsorted_idx):
                p.unlink(missing_ok=True)

    def create_sampler(
        self,
        *,
        batch_size: int,
        seed: int = 0,
        bucket_mult: int = 8,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
        window_items: int = 16384,
        max_merge_fan_in: int = 16,
        merge_run_batches: int | None = None,
        schedule_dir: Path | None = None,
    ) -> DiskLengthBucketBatchSampler:
        """Create a bounded-memory external disk-backed batch sampler over ``index.bin``."""
        sdir = Path(schedule_dir) if schedule_dir is not None else (self.cache_dir / "schedules")
        return DiskLengthBucketBatchSampler(
            self.index_path,
            self.num_rounds,
            batch_size=batch_size,
            seed=seed,
            bucket_mult=bucket_mult,
            rank=rank,
            world_size=world_size,
            drop_last=drop_last,
            schedule_dir=sdir,
            index_sha256=self.index_sha256,
            ordered_sample_sha256=self.ordered_sample_sha256,
            window_items=window_items,
            max_merge_fan_in=max_merge_fan_in,
            merge_run_batches=merge_run_batches,
            shared_schedule_across_ranks=bool(
                self._persistent_cache and world_size > 1 and dist.is_available() and dist.is_initialized()
            ),
        )

    def iter_index_lengths(self, chunk_items: int = 4096) -> Iterator[int]:
        """Stream round_event_len ints directly from disk in bounded chunks."""
        chunk_bytes = max(1, int(chunk_items)) * _INDEX_DTYPE.itemsize
        with self.index_path.open("rb") as fp:
            while True:
                raw = fp.read(chunk_bytes)
                if not raw:
                    break
                arr = np.frombuffer(raw, dtype=_INDEX_DTYPE)
                yield from arr["ev_len"].tolist()

    def _get_thread_readers(self) -> tuple[Any, Any]:
        pair = getattr(self._tls, "pair", None)
        if pair is None:
            idx_fp = self.index_path.open("rb")
            rds_fp = self.rounds_path.open("rb")
            pair = (idx_fp, rds_fp)
            self._tls.pair = pair
            with self._handles_lock:
                self._all_handles.append(pair)
        return pair

    def close_readers(self) -> None:
        with self._handles_lock:
            for idx_fp, rds_fp in self._all_handles:
                for fp in (idx_fp, rds_fp):
                    try:
                        fp.close()
                    except Exception:
                        pass
            self._all_handles.clear()
        self._tls = threading.local()

    def close(self) -> None:
        self.close_readers()
        self._finalizer()

    def __enter__(self) -> SpoolingRoundBCDataSource:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.num_rounds

    def get_rounds(self, indices: Sequence[int]) -> list[Any]:
        """Thread-safely seek requested rounds on disk and derive v2_* tensors in one batched CPU call."""
        from kanachan.training.bc.trajectory_v2 import RoundTrajectory

        if not indices:
            return []
        idx_fp, rds_fp = self._get_thread_readers()

        # Lookup index records ordered by round index for sequential seek locality.
        enumerated = []
        for pos, raw_idx in enumerate(indices):
            idx = int(raw_idx)
            if not 0 <= idx < self.num_rounds:
                raise IndexError(f"Round index {idx} out of range [0, {self.num_rounds}).")
            enumerated.append((pos, idx))

        metas: list[tuple[int, int, int, int, int]] = []
        for pos, idx in sorted(enumerated, key=lambda x: x[1]):
            idx_fp.seek(idx * _INDEX_STRUCT.size)
            off, blen, ev_len, obs_total, _, _, _ = _INDEX_STRUCT.unpack(
                idx_fp.read(_INDEX_STRUCT.size)
            )
            metas.append((pos, off, blen, ev_len, obs_total))

        raw_by_pos: list[dict[str, Any] | None] = [None] * len(indices)
        for pos, off, blen, ev_len, obs_total in sorted(metas, key=lambda x: x[1]):
            rds_fp.seek(off)
            payload = rds_fp.read(blen)
            raw_by_pos[pos] = _unpack_round_record(payload, obs_total, ev_len)
        raw_recs: list[dict[str, Any]] = [r for r in raw_by_pos if r is not None]

        sp_cat = torch.cat([r["sp_t"] for r in raw_recs], dim=0)
        nu_cat = torch.cat([r["nu_t"] for r in raw_recs], dim=0)
        pr_cat = torch.cat([r["pr_t"] for r in raw_recs], dim=0)
        ca_cat = torch.cat([r["ca_t"] for r in raw_recs], dim=0)
        feats = derive_structured_v2_batch(
            sp_cat,
            nu_cat,
            pr_cat,
            ca_cat,
            enable_early_shape_hint=self.enable_early_shape_hint,
            include_room_level=False,
        )

        out: list[RoundTrajectory] = []
        step_cursor = 0
        for rec in raw_recs:
            t_steps = rec["num_steps"]
            sl = slice(step_cursor, step_cursor + t_steps)
            step_cursor += t_steps
            ev_len = int(rec["round_event_len"])
            r_events = feats["v2_events"][step_cursor - 1, :ev_len].clone()
            rk_val = int(rec["final_rank"])
            rk_t = (
                torch.full((t_steps,), rk_val, dtype=torch.long) if rk_val >= 0 else None
            )
            out.append(
                RoundTrajectory(
                    round_events=r_events,
                    round_event_len=ev_len,
                    decision_event_len=feats["v2_event_len"][sl].to(torch.int32),
                    v2_tile_feats=feats["v2_tile_feats"][sl],
                    v2_global_feats=feats["v2_global_feats"][sl],
                    v2_player_feats=feats["v2_player_feats"][sl],
                    v2_cand_ids=feats["v2_cand_ids"][sl],
                    v2_cand_mask=feats["v2_cand_mask"][sl],
                    v2_cand_mode=feats["v2_cand_mode"][sl],
                    v2_cand_dyn=feats["v2_cand_dyn"][sl],
                    candidates=rec["ca_t"],
                    action_idx=rec["act_t"],
                    final_rank=rk_t,
                    observer_total_steps=int(rec["observer_total_steps"]),
                    uuid=str(rec["uuid"]),
                    observer_seat=int(rec["observer_seat"]),
                )
            )
        return out

    def get_round(self, idx: int) -> Any:
        return self.get_rounds([idx])[0]

    def __getitem__(self, idx: int) -> Any:
        return self.get_round(idx)


class BoundedRoundPrefetcher:
    """Bounded CPU background prefetch/collation queue streaming lazily in strict cursor order."""

    def __init__(
        self,
        source: SpoolingRoundBCDataSource | Sequence[Any],
        indexed_batches: Iterable[tuple[int, list[int], float]],
        *,
        max_prefetch: int = 4,
        suit_perm_seed: int | None = None,
        epoch: int = 0,
        num_workers: int = 1,
    ) -> None:
        from kanachan.training.bc.trajectory_v2 import collate_round_batch

        if max_prefetch < 1:
            raise ValueError("max_prefetch must be >= 1.")
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1.")
        self._source = source
        self._indexed_batches = indexed_batches
        self._collate = collate_round_batch
        self._suit_perm_seed = suit_perm_seed
        self._epoch = int(epoch)
        self._num_workers = int(num_workers)
        self._max_prefetch = int(max_prefetch)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self._max_prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bc-v2-prefetch", daemon=True)
        self._thread.start()

    def _put_respecting_stop(self, item: Any) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _fetch_and_collate(self, cursor_pos: int, idxs: list[int], weight: float) -> tuple[int, dict[str, Tensor]]:
        use_batched = hasattr(self._source, "get_rounds")
        batch_rounds = (
            self._source.get_rounds(idxs)  # type: ignore[union-attr]
            if use_batched
            else [self._source[i] for i in idxs]
        )
        step_gen: torch.Generator | None = None
        if self._suit_perm_seed is not None:
            seed_i = (
                int(self._suit_perm_seed)
                + int(self._epoch) * 1_000_003
                + int(cursor_pos) * 10_007
            ) & 0x7FFF_FFFF_FFFF_FFFF
            step_gen = torch.Generator(device="cpu").manual_seed(seed_i)
        batch_cpu = self._collate(
            batch_rounds,
            suit_perm_generator=step_gen,
            batch_weight=weight,
        )
        return int(cursor_pos), batch_cpu

    def _run(self) -> None:
        try:
            if self._num_workers == 1:
                for cursor_pos, idxs, weight in self._indexed_batches:
                    if self._stop.is_set():
                        break
                    pos, batch_cpu = self._fetch_and_collate(cursor_pos, idxs, weight)
                    if not self._put_respecting_stop(("ok", pos, batch_cpu)):
                        break
            else:
                from concurrent.futures import ThreadPoolExecutor

                max_in_flight = max(self._num_workers, min(self._max_prefetch, self._num_workers * 2))
                it = iter(self._indexed_batches)
                in_flight: deque_t[Any]
                from collections import deque

                in_flight = deque()
                with ThreadPoolExecutor(max_workers=self._num_workers, thread_name_prefix="bc-v2-w") as pool:
                    exhausted = False
                    while not self._stop.is_set():
                        while not exhausted and len(in_flight) < max_in_flight:
                            item = next(it, None)
                            if item is None:
                                exhausted = True
                                break
                            in_flight.append(pool.submit(self._fetch_and_collate, *item))
                        if not in_flight:
                            break
                        fut = in_flight.popleft()
                        pos, batch_cpu = fut.result()
                        if not self._put_respecting_stop(("ok", pos, batch_cpu)):
                            break
            self._put_respecting_stop(_SENTINEL)
        except BaseException as exc:  # noqa: BLE001
            self._put_respecting_stop(("err", exc))

    def __iter__(self) -> Iterator[tuple[int, dict[str, Tensor]]]:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            tag = item[0]
            if tag == "err":
                raise item[1]
            yield int(item[1]), item[2]

    def close(self) -> None:
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def __enter__(self) -> BoundedRoundPrefetcher:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "BoundedRoundPrefetcher",
    "DiskLengthBucketBatchSampler",
    "SpoolingRoundBCDataSource",
]
