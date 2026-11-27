"""Bounded-memory streaming Offline IQL pipeline for contiguous source-v3 RL streams (§2, §3)."""

from __future__ import annotations

import gzip
import math
import queue
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor

from kanachan.constants import (
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_SPARSE_FEATURES,
    SEAT_INDEX,
    SEAT_OFFSET,
)
from kanachan.structured_v2 import OFFSETS_136, PAD_ACTION_ID, permute_structured_batch
from kanachan.training.core.dataset_fingerprint import (
    SOURCE_V3_RL_FORMAT_VERSION,
    combine_dataset_identity,
    compute_iql_preprocess_fingerprint,
    compute_reward_plugin_sha256,
    compute_source_data_fingerprint,
)
from kanachan.training.core.offline_rl.data_iterator import DataIterator
from kanachan.training.core.ranking import scores_to_ranks
from kanachan.training.core.rl.rewards import configured_reward
from kanachan.training.grp import FrozenGRP, FrozenGRPPBRSReward, load_frozen_grp
from kanachan.training.iql.joint_v2 import attach_structured_v2_transition_features

_SENTINEL = object()


def _splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & 0xFFFF_FFFF_FFFF_FFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFF_FFFF_FFFF_FFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFF_FFFF_FFFF_FFFF
    return z ^ (z >> 31)


def _block_owner_perm(seed: int, block_idx: int, world_size: int) -> tuple[int, ...]:
    """Deterministic per-block permutation across ranks eliminating seat/class aliasing."""
    if world_size <= 1:
        return (0,)
    arr = list(range(world_size))
    base = (int(seed) * 0xD1342543DE82EF95 + int(block_idx) * 0x9E3779B97F4A7C15) & 0xFFFF_FFFF_FFFF_FFFF
    for i in range(world_size - 1, 0, -1):
        j = _splitmix64(base + i) % (i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return tuple(arr)


def _resolve_rl_source_files(source: Path | Sequence[Path]) -> list[Path]:
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
                raise FileNotFoundError(f"No .tsv/.tsv.gz files found in RL source directory {p}.")
            return files
        raise FileNotFoundError(f"RL source path does not exist: {p}")
    files = [Path(p) for p in source]
    if not files:
        raise ValueError("Empty RL source file sequence.")
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f"RL source file does not exist: {f}")
    return files


def _open_rl_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _apply_suit_perm_transition(
    td: TensorDict, perm_idx: int
) -> TensorDict:
    """Apply consistent suit permutation `perm_idx` to current state, action, and non-terminal next state."""
    if perm_idx == 0:
        return td
    cur_dict = {
        "v2_tile_feats": td["v2_tile_feats"],
        "v2_events": td["v2_events"],
        "v2_global_feats": td["v2_global_feats"],
        "v2_player_feats": td["v2_player_feats"],
        "v2_cand_ids": td["v2_cand_ids"],
        "v2_cand_mask": td["v2_cand_mask"],
        "v2_cand_mode": td["v2_cand_mode"],
        "v2_cand_dyn": td["v2_cand_dyn"],
        "action": td["action"],
    }
    p_cur = permute_structured_batch(cur_dict, perm_idx)

    nxt = td["next"]
    dummy_act = torch.zeros_like(td["action"])
    nxt_dict = {
        "v2_tile_feats": nxt["v2_tile_feats"],
        "v2_events": nxt["v2_events"],
        "v2_global_feats": nxt["v2_global_feats"],
        "v2_player_feats": nxt["v2_player_feats"],
        "v2_cand_ids": nxt["v2_cand_ids"],
        "v2_cand_mask": nxt["v2_cand_mask"],
        "v2_cand_mode": nxt["v2_cand_mode"],
        "v2_cand_dyn": nxt["v2_cand_dyn"],
        "action": dummy_act,
    }
    p_nxt = permute_structured_batch(nxt_dict, perm_idx)

    done_col = nxt["done"].bool()
    nxt_out: dict[str, Tensor] = {
        "sparse": nxt["sparse"],
        "numeric": nxt["numeric"],
        "progression": nxt["progression"],
        "candidates": p_nxt["v2_cand_ids"],
        "reward": nxt["reward"],
        "discount": nxt["discount"],
        "done": nxt["done"],
        "end_of_game": nxt["end_of_game"],
        "final_rank": nxt["final_rank"],
    }
    for k in (
        "v2_tile_feats",
        "v2_events",
        "v2_event_len",
        "v2_global_feats",
        "v2_player_feats",
        "v2_cand_ids",
        "v2_cand_mask",
        "v2_cand_mode",
        "v2_cand_dyn",
    ):
        if k in p_nxt:
            # Keep terminal rows zero-safe.
            val = p_nxt[k]
            if k != "v2_event_len":
                view_shape = (done_col.shape[0],) + (1,) * (val.ndim - 1)
                val = torch.where(done_col.view(view_shape), nxt[k], val)
            nxt_out[k] = val
        elif k in nxt.keys():
            nxt_out[k] = nxt[k]

    out_map: dict[str, Any] = {
        "sparse": td["sparse"],
        "numeric": td["numeric"],
        "progression": td["progression"],
        "candidates": p_cur["v2_cand_ids"],
        "action": p_cur["action"],
        "next": nxt_out,
    }
    for k in (
        "v2_tile_feats",
        "v2_events",
        "v2_event_len",
        "v2_global_feats",
        "v2_player_feats",
        "v2_cand_ids",
        "v2_cand_mask",
        "v2_cand_mode",
        "v2_cand_dyn",
    ):
        out_map[k] = p_cur[k] if k in p_cur else td[k]
    if "batch_weight" in td.keys():
        out_map["batch_weight"] = td["batch_weight"]
    return TensorDict(out_map, batch_size=td.batch_size, device=td.device)


class _CompactTransitionReservoir:
    """Fixed-capacity raw transition store using Vitter Algorithm R with LIFO rollback support."""

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("reservoir_size must be >= 1.")
        self.max_size = int(max_size)
        self.cap = 0
        self.length = 0
        self.total_seen = 0
        self._bufs: dict[str, Tensor] = {}

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self.cap:
            return
        new_cap = max(256, self.cap * 2 if self.cap > 0 else 256, needed)
        new_cap = min(self.max_size, new_cap)
        specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
            "sp": ((MAX_NUM_ACTIVE_SPARSE_FEATURES,), torch.int16),
            "nu": ((NUM_NUMERIC_FEATURES,), torch.int32),
            "pr": ((MAX_LENGTH_OF_PROGRESSION_FEATURES,), torch.int16),
            "ca": ((MAX_NUM_ACTION_CANDIDATES,), torch.int16),
            "act": ((), torch.uint8),
            "nsp": ((MAX_NUM_ACTIVE_SPARSE_FEATURES,), torch.int16),
            "nnu": ((NUM_NUMERIC_FEATURES,), torch.int32),
            "npr": ((MAX_LENGTH_OF_PROGRESSION_FEATURES,), torch.int16),
            "nca": ((MAX_NUM_ACTION_CANDIDATES,), torch.int16),
            "reward": ((), torch.float32),
            "discount": ((), torch.float32),
            "done": ((), torch.bool),
            "eog": ((), torch.bool),
            "final_rank": ((), torch.int8),
        }
        new_bufs: dict[str, Tensor] = {}
        for name, (shape_tail, dtype) in specs.items():
            t = torch.empty((new_cap, *shape_tail), dtype=dtype, device="cpu")
            if self.length > 0 and name in self._bufs:
                t[: self.length].copy_(self._bufs[name][: self.length])
            new_bufs[name] = t
        self._bufs = new_bufs
        self.cap = new_cap

    def ingest_game(
        self, game: dict[str, Tensor], rng: torch.Generator
    ) -> tuple[list[tuple[int, dict[str, Tensor] | None]], int, int]:
        """Insert one complete game via Algorithm R and return ``(undo_log, prev_len, prev_total_seen)``."""
        prev_len = self.length
        prev_seen = self.total_seen
        n_rows = int(game["act"].shape[0])
        if n_rows == 0:
            return [], prev_len, prev_seen
        self._ensure_capacity(min(self.max_size, self.length + n_rows))
        undo: list[tuple[int, dict[str, Tensor] | None]] = []

        for r in range(n_rows):
            self.total_seen += 1
            if self.length < self.max_size:
                slot = self.length
                self.length += 1
                undo.append((slot, None))
            else:
                # Vitter Algorithm R: slot inclusion probability M / total_seen per transition.
                j = int(torch.randint(0, self.total_seen, (1,), generator=rng).item())
                if j >= self.max_size:
                    continue
                slot = j
                prev = {k: v[slot].clone() for k, v in self._bufs.items()}
                undo.append((slot, prev))

            for k, buf in self._bufs.items():
                buf[slot].copy_(game[k][r].to(dtype=buf.dtype))
        return undo, prev_len, prev_seen

    def rollback_ingest(
        self,
        undo: list[tuple[int, dict[str, Tensor] | None]],
        prev_length: int,
        prev_total_seen: int,
    ) -> None:
        for slot, prev in reversed(undo):
            if prev is not None:
                for k, t in prev.items():
                    self._bufs[k][slot].copy_(t)
        self.length = int(prev_length)
        self.total_seen = int(prev_total_seen)

    def sample_batch(
        self,
        batch_size: int,
        rng: torch.Generator,
    ) -> dict[str, Tensor]:
        if self.length <= 0:
            raise RuntimeError("Cannot sample from empty RL transition reservoir.")
        if self.length >= batch_size:
            idx = torch.randperm(self.length, generator=rng)[:batch_size]
        else:
            idx = torch.randint(0, self.length, (batch_size,), generator=rng)
        return {k: v[idx].clone() for k, v in self._bufs.items()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "max_size": self.max_size,
            "length": self.length,
            "total_seen": self.total_seen,
            "buffers": {k: v[: self.length].clone() for k, v in self._bufs.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["max_size"]) != self.max_size:
            raise ValueError(
                f"Reservoir max_size mismatch: {state['max_size']} != {self.max_size}."
            )
        length = int(state["length"])
        self.length = 0
        self.total_seen = int(state["total_seen"])
        if length > 0:
            self._ensure_capacity(length)
            saved_bufs = state["buffers"]
            for k, buf in self._bufs.items():
                buf[:length].copy_(saved_bufs[k][:length].to(dtype=buf.dtype))
            self.length = length


class StreamingIQLDataset:
    """Bounded-memory streaming RL dataset with transition-uniform Algorithm R & multi-pass accounting.

    Multi-pass policy:
    - ``_reservoir`` and ``_reservoir.total_seen`` remain **cumulative** across ``reset_pass()`` calls
      so after ``p`` complete passes over ``N`` owned transitions, every transition ``i`` has been
      presented ``p`` times and retains equal marginal inclusion probability ``min(1, M / N)`` without
      biasing later passes toward pass start or re-running warmup filtering.
    - ``pass_transitions_seen``, ``pass_target_batches``, and ``pass_batches_emitted`` measure ONLY the
      current source traversal and are reset to ``(0, None, 0)`` in ``reset_pass()`` while
      ``pass_index`` increments by 1.
    - ``_ingest_rng``, ``_sample_rng``, and ``_aug_rng`` advance monotonically across passes so repeated
      passes remain bit-reproducible from checkpoints yet explore distinct sampling/augmentation draws.
    - Pre-EOF ingestion for each batch advances until the reservoir has reached ``warmup_transitions``
      AND ``pass_transitions_seen >= (pass_batches_emitted + 1) * batch_size`` (or EOF), guaranteeing no
      double-counted warmup transitions and no spurious extra batches on long games (``L > batch_size``).
    """

    def __init__(
        self,
        source: Path | Sequence[Path],
        *,
        batch_size: int = 32,
        reservoir_size: int = 4096,
        warmup_transitions: int | None = None,
        reward_scheme: str = "placement",
        reward_fn: Any | None = None,
        reward_plugin_path: Path | None = None,
        grp: FrozenGRP | Path | None = None,
        score_unit: float = 10000.0,
        round_weight: float = 0.2,
        discount_factor: float = 1.0,
        rewrite_rooms: int | None = None,
        rewrite_grades: int | None = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
        suit_augmentation: bool = False,
        aug_seed: int = 1337,
        enable_early_shape_hint: bool = False,
        max_prefetch: int = 2,
        max_games: int | None = None,
        max_batches: int | None = None,
        schema_version: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"Invalid distributed coordinates rank={rank}, world_size={world_size}.")
        if max_prefetch < 0:
            raise ValueError("max_prefetch must be >= 0.")

        self.source_files = _resolve_rl_source_files(source)
        self.batch_size = int(batch_size)
        self.reservoir_size = int(reservoir_size)
        self.warmup_transitions = (
            min(self.reservoir_size, max(0, int(warmup_transitions)))
            if warmup_transitions is not None
            else min(self.reservoir_size, max(self.batch_size, 256))
        )
        self.reward_scheme = str(reward_scheme)
        self.score_unit = float(score_unit)
        self.round_weight = float(round_weight)
        self.discount_factor = float(discount_factor)
        self.rewrite_rooms = rewrite_rooms
        self.rewrite_grades = rewrite_grades
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.drop_last = bool(drop_last)
        self.suit_augmentation = bool(suit_augmentation)
        self.aug_seed = int(aug_seed)
        self.enable_early_shape_hint = bool(enable_early_shape_hint)
        self.max_prefetch = int(max_prefetch)
        self.max_games = max_games
        self.max_batches = max_batches
        self.schema_version = str(schema_version or SOURCE_V3_RL_FORMAT_VERSION)

        self.grp: FrozenGRP | None = None
        if reward_fn is not None:
            self.reward_fn = reward_fn
        elif self.reward_scheme == "grp_pbrs":
            grp_obj = load_frozen_grp(Path(grp), require_gate=True) if isinstance(grp, (str, Path)) else grp
            if not isinstance(grp_obj, FrozenGRP):
                raise ValueError("StreamingIQLDataset reward_scheme='grp_pbrs' requires a verified FrozenGRP or Path.")
            self.grp = grp_obj
            self.reward_fn = FrozenGRPPBRSReward(grp_obj)
        else:
            if grp is not None and isinstance(grp, FrozenGRP):
                self.grp = grp
            self.reward_fn = configured_reward(
                self.reward_scheme,
                score_unit=self.score_unit,
                round_weight=self.round_weight,
                grp=self.grp,
            )

        reward_plugin_sha = (
            compute_reward_plugin_sha256(reward_fn, reward_plugin_path)
            if (reward_fn is not None or reward_plugin_path is not None)
            else None
        )
        self.source_data_sha256, self.source_file_fingerprints = compute_source_data_fingerprint(
            self.source_files,
            source_format=self.schema_version,
            max_items=self.max_games,
        )
        self.preprocess_recipe_sha256 = compute_iql_preprocess_fingerprint(
            enable_early_shape_hint=self.enable_early_shape_hint,
            rewrite_rooms=self.rewrite_rooms,
            rewrite_grades=self.rewrite_grades,
            reward_scheme=self.reward_scheme,
            reward_plugin_sha256=reward_plugin_sha,
            score_unit=self.score_unit,
            round_weight=self.round_weight,
            discount_factor=self.discount_factor,
            grp=self.grp,
            reservoir_size=self.reservoir_size,
            warmup_transitions=self.warmup_transitions,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            seed=self.seed,
            suit_augmentation=self.suit_augmentation,
            aug_seed=self.aug_seed,
            schema_version=self.schema_version,
        )
        self.data_sha256 = combine_dataset_identity(
            self.source_data_sha256, self.preprocess_recipe_sha256
        )

        self._parser_helper = DataIterator(
            path=self.source_files[0],
            num_skip_samples=0,
            rewrite_rooms=self.rewrite_rooms,
            rewrite_grades=self.rewrite_grades,
            local_rank=0,
        )
        # Close the helper's file handle immediately; we stream files explicitly per rank.
        helper_fp = getattr(self._parser_helper, "_DataIterator__fp", None)
        if helper_fp is not None:
            helper_fp.close()
            setattr(self._parser_helper, "_DataIterator__fp", None)
        self._parse_line = self._parser_helper._DataIterator__parse_line

        self._reservoir = _CompactTransitionReservoir(self.reservoir_size)
        self._ingest_rng = torch.Generator(device="cpu").manual_seed(self.seed + self.rank * 1_000_003 + 17)
        self._sample_rng = torch.Generator(device="cpu").manual_seed(self.seed + self.rank * 1_000_003 + 31)
        self._aug_rng = torch.Generator(device="cpu").manual_seed(self.aug_seed + self.rank * 1_000_003)

        self._file_idx: int = 0
        self._line_in_file: int = 0
        self._fp: Any = None
        self._game_idx: int = 0
        self._eof: bool = False
        self.pass_index: int = 0
        self.pass_transitions_seen: int = 0
        self.pass_target_batches: int | None = None
        self.pass_batches_emitted: int = 0
        self.total_batches_emitted: int = 0
        self._perm_cache: tuple[int, tuple[int, ...]] | None = None

        self._lock = threading.Lock()
        self._prefetch_queue: queue.Queue[Any] | None = None
        self._stop_event: threading.Event | None = None
        self._worker: threading.Thread | None = None
        self._last_batch: TensorDict | None = None
        self._cached_zero_ghost: TensorDict | None = None

    def _game_owner_rank(self, game_idx: int) -> int:
        if self.world_size == 1:
            return 0
        blk = game_idx // self.world_size
        off = game_idx % self.world_size
        if self._perm_cache is None or self._perm_cache[0] != blk:
            self._perm_cache = (blk, _block_owner_perm(self.seed, blk, self.world_size))
        return self._perm_cache[1][off]

    def _open_current_file(self) -> bool:
        while self._file_idx < len(self.source_files):
            if self._fp is None:
                self._fp = _open_rl_text(self.source_files[self._file_idx])
                for _ in range(self._line_in_file):
                    skipped = self._fp.readline()
                    if not skipped:
                        raise RuntimeError(
                            f"Resume cursor line={self._line_in_file} past EOF of {self.source_files[self._file_idx]}."
                        )
            return True
        return False

    def _close_fp(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _read_next_owned_game(self) -> dict[str, Tensor] | None:
        """Stream lines cleanly across files until rank's next complete observer game ends."""
        cur_rows: list[tuple[Tensor, ...]] = []
        while not self._eof:
            if self.max_games is not None and self._game_idx >= int(self.max_games):
                self._eof = True
                break
            if not self._open_current_file():
                self._eof = True
                break

            line = self._fp.readline()
            if not line:
                self._close_fp()
                if cur_rows:
                    raise RuntimeError("Incomplete episode at the end of contiguous RL replay data.")
                self._file_idx += 1
                self._line_in_file = 0
                continue

            self._line_in_file += 1
            stripped = line.rstrip("\r\n")
            if not stripped:
                raise RuntimeError(f"Blank line at {self.source_files[self._file_idx]}:{self._line_in_file}.")
            n_tabs = stripped.count("\t")
            if n_tabs not in (7, 9, 11):
                raise RuntimeError(
                    f"Expected 8, 10, or 12 columns at {self.source_files[self._file_idx]}:{self._line_in_file}, got {n_tabs + 1}."
                )
            is_eog = n_tabs == 7
            owned = self._game_owner_rank(self._game_idx) == self.rank

            if owned:
                row = self._parse_line(stripped)
                cur_rows.append(row)
                if is_eog:
                    self._game_idx += 1
                    return self._finalize_game(cur_rows)
            elif is_eog:
                self._game_idx += 1

        if cur_rows:
            raise RuntimeError("Incomplete episode at EOF of RL replay stream.")
        return None

    def _finalize_game(self, rows: list[tuple[Tensor, ...]]) -> dict[str, Tensor]:
        n = len(rows)
        sp = torch.stack([r[0] for r in rows], dim=0)
        nu = torch.stack([r[1] for r in rows], dim=0)
        pr = torch.stack([r[2] for r in rows], dim=0)
        ca = torch.stack([r[3] for r in rows], dim=0)
        act = torch.stack([r[4] for r in rows], dim=0)
        nsp = torch.stack([r[5] for r in rows], dim=0)
        nnu = torch.stack([r[6] for r in rows], dim=0)
        npr = torch.stack([r[7] for r in rows], dim=0)
        nca = torch.stack([r[8] for r in rows], dim=0)
        rs = torch.stack([r[9] for r in rows], dim=0)
        res = torch.stack([r[10] for r in rows], dim=0)
        eor = torch.stack([r[11] for r in rows], dim=0)
        eog = torch.stack([r[12] for r in rows], dim=0)

        episode = TensorDict(
            {
                "sparse": sp,
                "numeric": nu,
                "progression": pr,
                "candidates": ca,
                "action": act,
                "next": {
                    "sparse": nsp,
                    "numeric": nnu,
                    "progression": npr,
                    "candidates": nca,
                    "round_summary": rs,
                    "results": res,
                    "end_of_round": eor,
                    "end_of_game": eog,
                    "done": eog.clone(),
                },
            },
            batch_size=[n],
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            self.reward_fn(episode, True)
        reward = episode.get(("next", "reward"), None)
        if not isinstance(reward, Tensor) or reward.shape[0] != n:
            raise RuntimeError("Reward plugin did not produce valid next.reward tensor.")
        done = episode["next", "done"].bool()
        discount = episode.get(("next", "discount"), None)
        if discount is None:
            discount = torch.full((n,), self.discount_factor, dtype=torch.float32)

        final_scores = res[-1:, 10:14].long()
        own_seat = sp[-1, SEAT_INDEX].long() - SEAT_OFFSET
        ranks = scores_to_ranks(final_scores)
        own_rank = int(ranks[0, own_seat].item())
        final_rank = torch.full((n,), own_rank, dtype=torch.int8)

        return {
            "sp": sp.to(torch.int16),
            "nu": nu.to(torch.int32),
            "pr": pr.to(torch.int16),
            "ca": ca.to(torch.int16),
            "act": act.to(torch.uint8),
            "nsp": nsp.to(torch.int16),
            "nnu": nnu.to(torch.int32),
            "npr": npr.to(torch.int16),
            "nca": nca.to(torch.int16),
            "reward": reward.float(),
            "discount": discount.float(),
            "done": done,
            "eog": eog.bool(),
            "final_rank": final_rank,
        }

    def _produce_next_raw_with_rollback(self) -> tuple[dict[str, Tensor] | None, dict[str, Any] | None]:
        """Advance current-pass stream and cumulative reservoir while recording exact LIFO rollback state."""
        if self.max_batches is not None and self.total_batches_emitted >= int(self.max_batches):
            return None, None
        if self._eof and self.pass_target_batches is not None:
            if self.pass_batches_emitted >= self.pass_target_batches:
                return None, None

        snap: dict[str, Any] = {
            "file_idx": self._file_idx,
            "line_in_file": self._line_in_file,
            "game_idx": self._game_idx,
            "eof": self._eof,
            "pass_index": self.pass_index,
            "pass_transitions_seen": self.pass_transitions_seen,
            "pass_target_batches": self.pass_target_batches,
            "pass_batches_emitted": self.pass_batches_emitted,
            "total_batches_emitted": self.total_batches_emitted,
            "res_len": self._reservoir.length,
            "res_seen": self._reservoir.total_seen,
            "ingest_rng": self._ingest_rng.get_state().clone(),
            "sample_rng": self._sample_rng.get_state().clone(),
            "aug_rng": self._aug_rng.get_state().clone(),
            "undos": [],
        }

        needed_pass_transitions = (self.pass_batches_emitted + 1) * self.batch_size
        while not self._eof and (
            self._reservoir.length < self.warmup_transitions
            or self.pass_transitions_seen < needed_pass_transitions
        ):
            game = self._read_next_owned_game()
            if game is None:
                break
            n_rows = int(game["act"].shape[0])
            snap["undos"].append(self._reservoir.ingest_game(game, self._ingest_rng))
            self.pass_transitions_seen += n_rows

        if self._eof and self.pass_target_batches is None:
            if self.pass_transitions_seen == 0:
                self.pass_target_batches = 0
            elif self.drop_last:
                self.pass_target_batches = self.pass_transitions_seen // self.batch_size
            else:
                self.pass_target_batches = math.ceil(self.pass_transitions_seen / self.batch_size)

        if self._eof and self.pass_target_batches is not None:
            if self.pass_batches_emitted >= self.pass_target_batches or self._reservoir.length == 0:
                return None, snap

        raw = self._reservoir.sample_batch(self.batch_size, self._sample_rng)
        perm_idx = (
            int(torch.randint(0, 6, (1,), generator=self._aug_rng).item())
            if self.suit_augmentation
            else 0
        )
        raw["perm_idx"] = torch.tensor(perm_idx, dtype=torch.int64)
        self.pass_batches_emitted += 1
        self.total_batches_emitted += 1
        return raw, snap

    def _build_tensordict_batch(self, raw: dict[str, Tensor], *, batch_weight: float = 1.0) -> TensorDict:
        bsz = int(raw["act"].shape[0])
        td = TensorDict(
            {
                "sparse": raw["sp"].to(torch.int32),
                "numeric": raw["nu"].to(torch.int32),
                "progression": raw["pr"].to(torch.int32),
                "candidates": raw["ca"].to(torch.int32),
                "action": raw["act"].to(torch.int64),
                "batch_weight": torch.full((bsz,), float(batch_weight), dtype=torch.float32),
                "next": {
                    "sparse": raw["nsp"].to(torch.int32),
                    "numeric": raw["nnu"].to(torch.int32),
                    "progression": raw["npr"].to(torch.int32),
                    "candidates": raw["nca"].to(torch.int32),
                    "reward": raw["reward"].float(),
                    "discount": raw["discount"].float(),
                    "done": raw["done"].bool(),
                    "end_of_game": raw["eog"].bool(),
                    "final_rank": raw["final_rank"].to(torch.int64),
                },
            },
            batch_size=[bsz],
            device=torch.device("cpu"),
        )
        td = attach_structured_v2_transition_features(
            td, enable_early_shape_hint=self.enable_early_shape_hint
        )
        perm_idx = int(raw["perm_idx"].item()) if "perm_idx" in raw else 0
        if perm_idx != 0:
            td = _apply_suit_perm_transition(td, perm_idx)
        return td

    def _rollback_snapshot(self, snap: dict[str, Any]) -> None:
        self._close_fp()
        for undo, prev_len, prev_seen in reversed(snap["undos"]):
            self._reservoir.rollback_ingest(undo, prev_len, prev_seen)
        self._reservoir.length = int(snap["res_len"])
        self._reservoir.total_seen = int(snap["res_seen"])
        self._file_idx = int(snap["file_idx"])
        self._line_in_file = int(snap["line_in_file"])
        self._game_idx = int(snap["game_idx"])
        self._eof = bool(snap["eof"])
        self.pass_index = int(snap["pass_index"])
        self.pass_transitions_seen = int(snap["pass_transitions_seen"])
        self.pass_target_batches = snap["pass_target_batches"]
        self.pass_batches_emitted = int(snap["pass_batches_emitted"])
        self.total_batches_emitted = int(snap["total_batches_emitted"])
        self._ingest_rng.set_state(snap["ingest_rng"])
        self._sample_rng.set_state(snap["sample_rng"])
        self._aug_rng.set_state(snap["aug_rng"])

    def _stop_prefetcher(self, rollback_queued: bool = False) -> None:
        if self._worker is None:
            return
        assert self._stop_event is not None and self._prefetch_queue is not None
        self._stop_event.set()
        drained: list[dict[str, Any] | None] = []
        while True:
            try:
                item = self._prefetch_queue.get_nowait()
            except queue.Empty:
                if not self._worker.is_alive():
                    break
                self._worker.join(timeout=0.05)
                continue
            if item is _SENTINEL or (isinstance(item, tuple) and item[0] == "err"):
                continue
            if isinstance(item, tuple) and item[0] == "ok":
                drained.append(item[2])
            elif isinstance(item, tuple) and item[0] == "eof":
                drained.append(item[1])
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if rollback_queued and drained:
            # Undo unconsumed queued batches AND pre-fetched EOF transitions in reverse order.
            with self._lock:
                for snap in reversed(drained):
                    if snap is not None:
                        self._rollback_snapshot(snap)
        self._worker = None
        self._prefetch_queue = None
        self._stop_event = None

    def _ensure_prefetcher_started(self) -> None:
        if self.max_prefetch <= 0 or self._worker is not None:
            return
        q: queue.Queue[Any] = queue.Queue(maxsize=self.max_prefetch)
        stop = threading.Event()
        self._prefetch_queue = q
        self._stop_event = stop

        def _put(obj: Any) -> bool:
            while not stop.is_set():
                try:
                    q.put(obj, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        def _worker_fn() -> None:
            try:
                while not stop.is_set():
                    with self._lock:
                        if stop.is_set():
                            break
                        raw, snap = self._produce_next_raw_with_rollback()
                    if raw is None:
                        if not _put(("eof", snap)):
                            with self._lock:
                                if snap is not None:
                                    self._rollback_snapshot(snap)
                        return
                    td = self._build_tensordict_batch(raw, batch_weight=1.0)
                    if not _put(("ok", td, snap)):
                        with self._lock:
                            if snap is not None:
                                self._rollback_snapshot(snap)
                        return
            except BaseException as exc:  # noqa: BLE001
                _put(("err", exc))

        t = threading.Thread(target=_worker_fn, name="iql-v2-prefetch", daemon=True)
        self._worker = t
        t.start()

    def next_batch(self) -> TensorDict | None:
        if self.max_prefetch > 0:
            self._ensure_prefetcher_started()
            assert self._prefetch_queue is not None
            while True:
                item = self._prefetch_queue.get()
                if item is _SENTINEL or (isinstance(item, tuple) and item[0] == "eof"):
                    self._stop_prefetcher(rollback_queued=False)
                    return None
                if item[0] == "err":
                    self._stop_prefetcher(rollback_queued=False)
                    raise item[1]
                td = item[1]
                self._last_batch = td
                return td
        else:
            with self._lock:
                raw, _ = self._produce_next_raw_with_rollback()
            if raw is None:
                return None
            td = self._build_tensordict_batch(raw, batch_weight=1.0)
            self._last_batch = td
            return td

    def _empty_ghost_batch(self) -> TensorDict:
        if self._cached_zero_ghost is not None:
            return self._cached_zero_ghost
        bsz = self.batch_size
        sp_toks = [2, 5, 7, 23, 39, 55, 71, 75, 78, 82 + 69, 152 + 1]
        closed_hand = {1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1, 21: 1, 22: 1, 23: 1, 31: 3, 32: 1}
        for t37, cnt in sorted(closed_hand.items()):
            for k in range(cnt):
                sp_toks.append(337 + OFFSETS_136[t37] + k)
        sp_toks.append(473 + 32)
        sp_row = torch.full((MAX_NUM_ACTIVE_SPARSE_FEATURES,), NUM_TYPES_OF_SPARSE_FEATURES, dtype=torch.int32)
        sp_row[: len(sp_toks)] = torch.tensor(sp_toks, dtype=torch.int32)
        sp = sp_row.unsqueeze(0).expand(bsz, -1).clone()
        nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]], dtype=torch.int32).expand(bsz, -1).clone()
        pr = torch.full((bsz, MAX_LENGTH_OF_PROGRESSION_FEATURES), NUM_TYPES_OF_PROGRESSION_FEATURES, dtype=torch.int32)
        pr[:, 0] = 0
        ca = torch.full((bsz, MAX_NUM_ACTION_CANDIDATES), PAD_ACTION_ID, dtype=torch.int32)
        ca[:, 0] = 4
        ca[:, 1] = 44
        ca[:, 2] = 84
        ca[:, 3] = 130
        raw = {
            "sp": sp.to(torch.int16),
            "nu": nu,
            "pr": pr.to(torch.int16),
            "ca": ca.to(torch.int16),
            "act": torch.zeros((bsz,), dtype=torch.uint8),
            "nsp": sp.to(torch.int16),
            "nnu": nu,
            "npr": pr.to(torch.int16),
            "nca": ca.to(torch.int16),
            "reward": torch.zeros((bsz,), dtype=torch.float32),
            "discount": torch.full((bsz,), self.discount_factor, dtype=torch.float32),
            "done": torch.ones((bsz,), dtype=torch.bool),
            "eog": torch.ones((bsz,), dtype=torch.bool),
            "final_rank": torch.zeros((bsz,), dtype=torch.int8),
        }
        td = self._build_tensordict_batch(raw, batch_weight=0.0)
        self._cached_zero_ghost = td
        return td

    def ghost_batch(self) -> TensorDict:
        """Return a valid zero-weight batch when another DDP rank still has remaining work."""
        if self._last_batch is not None:
            cloned = self._last_batch.clone(recurse=False)
            bsz = int(cloned.batch_size[0])
            cloned["batch_weight"] = torch.zeros((bsz,), dtype=torch.float32)
            return cloned
        return self._empty_ghost_batch().clone(recurse=False)

    def __iter__(self) -> Iterator[TensorDict]:
        while True:
            b = self.next_batch()
            if b is None:
                return
            yield b

    def state_dict(self) -> dict[str, Any]:
        """Pause background prefetch, roll back unconsumed items cleanly, and return exact stream state."""
        self._stop_prefetcher(rollback_queued=True)
        with self._lock:
            return {
                "format": "kanachan_iql_stream_v2",
                "data_sha256": self.data_sha256,
                "source_data_sha256": self.source_data_sha256,
                "preprocess_recipe_sha256": self.preprocess_recipe_sha256,
                "rank": self.rank,
                "world_size": self.world_size,
                "batch_size": self.batch_size,
                "reservoir_size": self.reservoir_size,
                "file_idx": self._file_idx,
                "line_in_file": self._line_in_file,
                "game_idx": self._game_idx,
                "eof": self._eof,
                "pass_index": self.pass_index,
                "pass_transitions_seen": self.pass_transitions_seen,
                "pass_target_batches": self.pass_target_batches,
                "pass_batches_emitted": self.pass_batches_emitted,
                "total_batches_emitted": self.total_batches_emitted,
                "ingest_rng": self._ingest_rng.get_state().clone(),
                "sample_rng": self._sample_rng.get_state().clone(),
                "aug_rng": self._aug_rng.get_state().clone(),
                "reservoir": self._reservoir.state_dict(),
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._stop_prefetcher(rollback_queued=False)
        with self._lock:
            if state.get("format") != "kanachan_iql_stream_v2":
                raise ValueError("Invalid StreamingIQLDataset checkpoint state format (expected v2).")
            if state.get("source_data_sha256") != self.source_data_sha256:
                raise ValueError(
                    "StreamingIQLDataset checkpoint source_data_sha256 mismatch (Fail-Closed)."
                )
            if state.get("preprocess_recipe_sha256") != self.preprocess_recipe_sha256:
                raise ValueError(
                    "StreamingIQLDataset checkpoint preprocess_recipe_sha256 mismatch (Fail-Closed)."
                )
            if state.get("data_sha256") != self.data_sha256:
                raise ValueError("StreamingIQLDataset checkpoint data_sha256 mismatch (Fail-Closed).")
            if int(state["rank"]) != self.rank or int(state["world_size"]) != self.world_size:
                raise ValueError("StreamingIQLDataset checkpoint rank/world_size mismatch.")
            if int(state["batch_size"]) != self.batch_size or int(state["reservoir_size"]) != self.reservoir_size:
                raise ValueError("StreamingIQLDataset checkpoint batch_size/reservoir_size mismatch.")
            self._close_fp()
            self._file_idx = int(state["file_idx"])
            self._line_in_file = int(state["line_in_file"])
            self._game_idx = int(state["game_idx"])
            self._eof = bool(state["eof"])
            self.pass_index = int(state["pass_index"])
            self.pass_transitions_seen = int(state["pass_transitions_seen"])
            self.pass_target_batches = state["pass_target_batches"]
            self.pass_batches_emitted = int(state["pass_batches_emitted"])
            self.total_batches_emitted = int(state["total_batches_emitted"])
            self._ingest_rng.set_state(state["ingest_rng"].cpu())
            self._sample_rng.set_state(state["sample_rng"].cpu())
            self._aug_rng.set_state(state["aug_rng"].cpu())
            self._reservoir.load_state_dict(state["reservoir"])

    def reset_pass(self) -> None:
        """Reset current-pass cursor/counters for another epoch while keeping cumulative reservoir & RNGs."""
        self._stop_prefetcher(rollback_queued=False)
        with self._lock:
            self._close_fp()
            self._file_idx = 0
            self._line_in_file = 0
            self._game_idx = 0
            self._eof = False
            self.pass_index += 1
            self.pass_transitions_seen = 0
            self.pass_target_batches = None
            self.pass_batches_emitted = 0

    def close(self) -> None:
        self._stop_prefetcher(rollback_queued=False)
        with self._lock:
            self._close_fp()
        if hasattr(self._parser_helper, "_DataIterator__fp"):
            fp = getattr(self._parser_helper, "_DataIterator__fp", None)
            if fp is not None:
                fp.close()

    def __enter__(self) -> StreamingIQLDataset:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = ["StreamingIQLDataset"]
