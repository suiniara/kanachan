"""Bounded-memory external hash-bucket disk spooling & prefetch for Structured V2 BC (§2)."""

from __future__ import annotations

import array
import gzip
import hashlib
import json
import queue
import shutil
import struct
import tempfile
import threading
import weakref
from collections.abc import Iterable, Iterator, Sequence
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

_INDEX_STRUCT = struct.Struct("<QIIIIbB6x")  # 32 bytes/round
assert _INDEX_STRUCT.size == 32
_REC_HDR_STRUCT = struct.Struct("<HHHHbB")  # 10 bytes header per round in rounds.bin
assert _REC_HDR_STRUCT.size == 10
_SENTINEL = object()


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


class SpoolingRoundBCDataSource:
    """Out-of-core hash-bucketed & disk-indexed round BC source with zero full-dataset RAM retention."""

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

        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if cache_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="kanachan_bc_spool_")
            self.cache_dir = Path(self._temp_dir.name)
        else:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._finalizer = weakref.finalize(self, self._cleanup_tempdir, self._temp_dir)

        self.rounds_path = self.cache_dir / "rounds.bin"
        self.index_path = self.cache_dir / "index.bin"
        self.meta_path = self.cache_dir / "meta.json"

        self.num_games: int = 0
        self.num_rounds: int = 0
        self.data_sha256: str = ""
        self.round_event_lens: array.array[int] = array.array("I")
        self._index_fp: Any = None
        self._rounds_fp: Any = None
        self._io_lock = threading.Lock()

        if reuse_cache and self._try_load_cached_meta():
            self._load_index_lengths()
        elif cache_dir is not None and dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            err_holder: list[str | None] = [None]
            if rank == 0:
                try:
                    self._build_disk_cache()
                except BaseException as exc:  # noqa: BLE001
                    err_holder[0] = f"{type(exc).__name__}: {exc}"
            dist.broadcast_object_list(err_holder, src=0)
            if err_holder[0] is not None:
                raise RuntimeError(f"Rank-0 BC disk spool build failed: {err_holder[0]}")
            if rank != 0:
                if not self._try_load_cached_meta():
                    raise RuntimeError(f"Shared BC spool cache unavailable on rank {rank}: {self.cache_dir}")
                self._load_index_lengths()
        else:
            self._build_disk_cache()

        self._open_readers()

    @staticmethod
    def _cleanup_tempdir(temp_dir: tempfile.TemporaryDirectory[str] | None) -> None:
        if temp_dir is not None:
            temp_dir.cleanup()

    def _source_signature(self) -> list[dict[str, Any]]:
        return [
            {
                "path": str(p.resolve()),
                "size": int(p.stat().st_size),
                "mtime_ns": int(p.stat().st_mtime_ns),
            }
            for p in self.source_files
        ]

    def _try_load_cached_meta(self) -> bool:
        if not (self.meta_path.exists() and self.index_path.exists() and self.rounds_path.exists()):
            return False
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if meta.get("format") != "kanachan_bc_round_spool_v1":
            return False
        expected_sig = self._source_signature()
        if meta.get("source_files") != expected_sig:
            return False
        for key, cur_val in (
            ("allow_unlabeled", self.allow_unlabeled),
            ("allow_incomplete", self.allow_incomplete),
            ("enable_early_shape_hint", self.enable_early_shape_hint),
            ("max_rounds", self.max_rounds),
        ):
            if meta.get(key) != cur_val:
                return False
        self.num_games = int(meta["num_games"])
        self.num_rounds = int(meta["num_rounds"])
        self.data_sha256 = str(meta["data_sha256"])
        return True

    def _load_index_lengths(self) -> None:
        self.round_event_lens = array.array("I")
        with self.index_path.open("rb") as fp:
            while True:
                chunk = fp.read(_INDEX_STRUCT.size * 4096)
                if not chunk:
                    break
                for off in range(0, len(chunk), _INDEX_STRUCT.size):
                    _, _, ev_len, _, _, _, _ = _INDEX_STRUCT.unpack_from(chunk, off)
                    self.round_event_lens.append(ev_len)
        if len(self.round_event_lens) != self.num_rounds:
            raise RuntimeError(
                f"Corrupt BC spool index: {len(self.round_event_lens)} != meta num_rounds={self.num_rounds}."
            )

    def _build_disk_cache(self) -> None:
        from kanachan.training.bc.trajectory_v2 import _validate_and_stage_bc_lines

        bucket_dir = self.cache_dir / "buckets_tmp"
        if bucket_dir.exists():
            shutil.rmtree(bucket_dir)
        bucket_dir.mkdir(parents=True, exist_ok=True)

        tmp_rounds = self.cache_dir / "rounds.bin.tmp"
        tmp_index = self.cache_dir / "index.bin.tmp"
        tmp_meta = self.cache_dir / "meta.json.tmp"
        for p in (tmp_rounds, tmp_index, tmp_meta):
            p.unlink(missing_ok=True)

        def _raw_source_iter() -> Iterable[tuple[int, str]]:
            global_line = 0
            for fpath in self.source_files:
                with _open_bc_text(fpath) as fp:
                    for raw in fp:
                        global_line += 1
                        yield global_line, raw

        data_hasher = hashlib.sha256()
        data_hasher.update(
            json.dumps(
                {
                    "allow_unlabeled": self.allow_unlabeled,
                    "allow_incomplete": self.allow_incomplete,
                    "enable_early_shape_hint": self.enable_early_shape_hint,
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        round_lengths = array.array("I")
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

            with tmp_rounds.open("wb") as r_fp, tmp_index.open("wb") as idx_fp:
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
                        if self.max_rounds is not None and len(round_lengths) >= int(self.max_rounds):
                            break
                        raw_rec = _pack_round_record(rec)
                        rec_len = len(raw_rec)
                        r_fp.write(raw_rec)
                        idx_entry = _INDEX_STRUCT.pack(
                            offset,
                            rec_len,
                            int(rec["round_event_len"]),
                            int(rec["observer_total_steps"]),
                            int(rec.get("num_steps", rec["sp_t"].shape[0])),
                            int(rec["final_rank"]) if rec["final_rank"] is not None else -1,
                            int(rec["observer_seat"]),
                        )
                        idx_fp.write(idx_entry)
                        data_hasher.update(idx_entry)
                        round_lengths.append(int(rec["round_event_len"]))
                        offset += rec_len

                    if self.max_rounds is not None and len(round_lengths) >= int(self.max_rounds):
                        break

            self.num_games = num_games_total
            self.num_rounds = len(round_lengths)
            self.data_sha256 = data_hasher.hexdigest()
            self.round_event_lens = round_lengths

            tmp_rounds.replace(self.rounds_path)
            tmp_index.replace(self.index_path)
            meta_obj = {
                "format": "kanachan_bc_round_spool_v1",
                "source_files": self._source_signature(),
                "allow_unlabeled": self.allow_unlabeled,
                "allow_incomplete": self.allow_incomplete,
                "enable_early_shape_hint": self.enable_early_shape_hint,
                "max_rounds": self.max_rounds,
                "num_games": self.num_games,
                "num_rounds": self.num_rounds,
                "data_sha256": self.data_sha256,
            }
            tmp_meta.write_text(json.dumps(meta_obj, indent=2, sort_keys=True), encoding="utf-8")
            tmp_meta.replace(self.meta_path)
        finally:
            if bucket_dir.exists():
                shutil.rmtree(bucket_dir, ignore_errors=True)
            for p in (tmp_rounds, tmp_index, tmp_meta):
                p.unlink(missing_ok=True)

    def _open_readers(self) -> None:
        self.close_readers()
        self._index_fp = self.index_path.open("rb")
        self._rounds_fp = self.rounds_path.open("rb")

    def close_readers(self) -> None:
        if self._index_fp is not None:
            self._index_fp.close()
            self._index_fp = None
        if self._rounds_fp is not None:
            self._rounds_fp.close()
            self._rounds_fp = None

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
        """Seek only requested rounds on disk and derive v2_* tensors in one batched CPU call."""
        from kanachan.training.bc.trajectory_v2 import RoundTrajectory

        if not indices:
            return []
        raw_recs: list[dict[str, Any]] = []
        with self._io_lock:
            if self._index_fp is None or self._rounds_fp is None:
                self._open_readers()
            assert self._index_fp is not None and self._rounds_fp is not None
            for raw_idx in indices:
                idx = int(raw_idx)
                if not 0 <= idx < self.num_rounds:
                    raise IndexError(f"Round index {idx} out of range [0, {self.num_rounds}).")
                self._index_fp.seek(idx * _INDEX_STRUCT.size)
                off, blen, ev_len, obs_total, _, _, _ = _INDEX_STRUCT.unpack(
                    self._index_fp.read(_INDEX_STRUCT.size)
                )
                self._rounds_fp.seek(off)
                payload = self._rounds_fp.read(blen)
                raw_recs.append(_unpack_round_record(payload, obs_total, ev_len))

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
    """Bounded CPU background queue fetching and collating upcoming round batches."""

    def __init__(
        self,
        source: SpoolingRoundBCDataSource | Sequence[Any],
        indexed_batches: Sequence[tuple[int, list[int], float]],
        *,
        max_prefetch: int = 4,
        suit_perm_seed: int | None = None,
        epoch: int = 0,
    ) -> None:
        from kanachan.training.bc.trajectory_v2 import collate_round_batch

        if max_prefetch < 1:
            raise ValueError("max_prefetch must be >= 1.")
        self._source = source
        self._indexed_batches = indexed_batches
        self._collate = collate_round_batch
        self._suit_perm_seed = suit_perm_seed
        self._epoch = int(epoch)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=int(max_prefetch))
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

    def _run(self) -> None:
        try:
            use_batched = hasattr(self._source, "get_rounds")
            step_gen = torch.Generator(device="cpu") if self._suit_perm_seed is not None else None
            for cursor_pos, idxs, weight in self._indexed_batches:
                if self._stop.is_set():
                    break
                if use_batched:
                    batch_rounds = self._source.get_rounds(idxs)  # type: ignore[union-attr]
                else:
                    batch_rounds = [self._source[i] for i in idxs]
                if step_gen is not None:
                    assert self._suit_perm_seed is not None
                    seed_i = (
                        int(self._suit_perm_seed)
                        + int(self._epoch) * 1_000_003
                        + int(cursor_pos) * 10_007
                    ) & 0x7FFF_FFFF_FFFF_FFFF
                    step_gen.manual_seed(seed_i)
                batch_cpu = self._collate(
                    batch_rounds,
                    suit_perm_generator=step_gen,
                    batch_weight=weight,
                )
                if not self._put_respecting_stop(("ok", cursor_pos, batch_cpu)):
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
    "SpoolingRoundBCDataSource",
]
