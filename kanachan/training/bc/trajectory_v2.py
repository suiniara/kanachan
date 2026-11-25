"""Stage-1 Parallel Round-Trajectory BC, Suit Aug, Auxiliary Heads & 9-Family Eval (§4.2)."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.distributed as dist
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from kanachan.constants import (
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    MAX_NUM_ROUND_SUMMARY,
    NUM_NUMERIC_FEATURES,
    NUM_RESULTS,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ROUND_SUMMARY,
    NUM_TYPES_OF_SPARSE_FEATURES,
    SEAT_INDEX,
    SEAT_OFFSET,
)
from kanachan.model_loader import dump_model, dump_object, load_model
from kanachan.nn.decode_converter import DecodeConverter
from kanachan.nn.decision_decoder import require_tensor
from kanachan.nn.structured_dual_decoder import (
    ARCHITECTURE_ID,
    NextEventHead,
    StructuredActionHead,
    StructuredDualDecoder,
    StructuredStateHead,
    structured_v2_schema,
)
from kanachan.structured_v2 import (
    ACTION_FAMILY_NAMES,
    KANACHAN_ACTION_MANIFEST,
    KANACHAN_ACTION_MANIFEST_SHA256,
    NUM_TYPES_OF_ACTIONS,
    PAD_ACTION_ID,
    PERM_TILE37,
    derive_structured_v2_batch,
    permute_structured_batch,
)
from kanachan.training.core.features import validate_sparse_prefix
from kanachan.training.core.optimizers import adamw_parameter_groups, move_optimizer_state
from kanachan.training.core.ranking import scores_to_ranks
from kanachan.training.online.state import atomic_save, capture_rng, cpu_state, restore_rng


def classify_action_id(global_id: int) -> str:
    gid = int(global_id)
    if not 0 <= gid < NUM_TYPES_OF_ACTIONS:
        raise ValueError(f"Invalid global action ID: {gid}")
    row = KANACHAN_ACTION_MANIFEST[gid]
    atype = int(row[0])
    is_riichi = int(row[8])
    if atype == 0:
        return "DISCARD_RIICHI" if is_riichi else "DISCARD_DAMA"
    if atype == 6:
        return "CHI"
    if atype == 7:
        return "PON"
    if atype in (1, 2, 8):
        return "KAN"
    if atype == 3:
        return "TSUMO"
    if atype == 9:
        return "RON"
    if atype == 5:
        return "SKIP"
    if atype == 4:
        return "KYUSHU"
    raise ValueError(f"Unexpected action type {atype} for global action ID {gid}")


@dataclass
class RoundTrajectory:
    """One complete observer round grouped from decision steps."""

    round_events: Tensor  # [L_r, 10] uint8
    round_event_len: int
    decision_event_len: Tensor  # [T] int32
    v2_tile_feats: Tensor  # [T, 34, 70] float32
    v2_global_feats: Tensor  # [T, 48] float32
    v2_player_feats: Tensor  # [T, 4, 32] float32
    v2_cand_ids: Tensor  # [T, 32] int32
    v2_cand_mask: Tensor  # [T, 32] bool
    v2_cand_mode: Tensor  # [T, 32] uint8
    v2_cand_dyn: Tensor  # [T, 32, 8] float32
    candidates: Tensor  # [T, 32] int32
    action_idx: Tensor  # [T] int64
    final_rank: Tensor | None = None  # [T] int64 (-1 if unlabeled)


def _open_tsv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _parse_int_csv(raw: str, field_name: str, ctx: str) -> list[int]:
    if not raw or not raw.strip():
        raise ValueError(f"{ctx}: Empty {field_name} field.")
    tokens = raw.split(",")
    out: list[int] = []
    for tok in tokens:
        t = tok.strip()
        if not t:
            raise ValueError(f"{ctx}: Empty element in {field_name}.")
        try:
            out.append(int(t))
        except ValueError as exc:
            raise ValueError(f"{ctx}: Non-integer token {tok!r} in {field_name}.") from exc
    return out


_SCORE_BOUND = 10_000_000


def _parse_bc_line_full(
    line: str,
    *,
    allow_unlabeled: bool = True,
    line_no: int | None = None,
) -> tuple[
    str,
    list[int],
    list[int],
    list[int],
    list[int],
    int,
    tuple[int, int, int, int] | None,
    tuple[int, ...] | None,
    tuple[int, ...] | None,
    int,
]:
    """Parse one source-v3 BC line (8 cols) or 6-col unlabeled row, returning summary metadata."""
    stripped = line.rstrip("\r\n")
    ctx = f"line {line_no}" if line_no is not None else "BC line"
    if not stripped:
        raise ValueError(f"{ctx}: Blank line in BC source.")
    parts = stripped.split("\t")
    if len(parts) not in (6, 8):
        raise ValueError(
            f"{ctx}: Expected 8 columns (source-v3) or 6 columns (unlabeled action-only), got {len(parts)}."
        )
    if len(parts) == 6 and not allow_unlabeled:
        raise ValueError(f"{ctx}: Missing round_summary and results columns (expected 8-column source-v3).")

    uuid = parts[0].strip()
    if not uuid:
        raise ValueError(f"{ctx}: Empty game UUID.")
    ctx = f"{ctx} ({uuid})"

    sp_raw = _parse_int_csv(parts[1], "sparse", ctx)
    if len(sp_raw) > MAX_NUM_ACTIVE_SPARSE_FEATURES:
        raise ValueError(f"{ctx}: Too many sparse features ({len(sp_raw)}).")
    if any(x < 0 or x >= NUM_TYPES_OF_SPARSE_FEATURES for x in sp_raw):
        raise ValueError(f"{ctx}: Sparse feature out of [0, {NUM_TYPES_OF_SPARSE_FEATURES}).")
    try:
        validate_sparse_prefix(sp_raw, end_of_round=False)
    except RuntimeError as exc:
        raise ValueError(f"{ctx}: {exc}") from exc
    sp = sp_raw + [NUM_TYPES_OF_SPARSE_FEATURES] * (MAX_NUM_ACTIVE_SPARSE_FEATURES - len(sp_raw))

    nu = _parse_int_csv(parts[2], "numeric", ctx)
    if len(nu) != NUM_NUMERIC_FEATURES:
        raise ValueError(f"{ctx}: Expected {NUM_NUMERIC_FEATURES} numeric features, got {len(nu)}.")
    if nu[0] < 0 or nu[1] < 0 or any(abs(x) > _SCORE_BOUND for x in nu):
        raise ValueError(f"{ctx}: Numeric feature out of valid range: {nu}.")

    pr_raw = _parse_int_csv(parts[3], "progression", ctx)
    if len(pr_raw) > MAX_LENGTH_OF_PROGRESSION_FEATURES:
        raise ValueError(f"{ctx}: Too many progression features ({len(pr_raw)}).")
    if pr_raw[0] != 0 or any(x < 0 or x >= NUM_TYPES_OF_PROGRESSION_FEATURES for x in pr_raw):
        raise ValueError(f"{ctx}: Invalid progression features.")
    pr = pr_raw + [NUM_TYPES_OF_PROGRESSION_FEATURES] * (MAX_LENGTH_OF_PROGRESSION_FEATURES - len(pr_raw))

    ca_raw = _parse_int_csv(parts[4], "candidates", ctx)
    if len(ca_raw) > MAX_NUM_ACTION_CANDIDATES:
        raise ValueError(f"{ctx}: Too many candidate actions ({len(ca_raw)}).")
    if any(x < 0 or x >= NUM_TYPES_OF_ACTIONS for x in ca_raw):
        raise ValueError(f"{ctx}: Candidate action ID out of [0, {NUM_TYPES_OF_ACTIONS}).")
    if any(ca_raw[i] >= ca_raw[i + 1] for i in range(len(ca_raw) - 1)):
        raise ValueError(f"{ctx}: Candidate action IDs must be strictly ascending: {ca_raw}.")
    ca = ca_raw + [PAD_ACTION_ID] * (MAX_NUM_ACTION_CANDIDATES - len(ca_raw))

    act_raw = parts[5].strip()
    try:
        act = int(act_raw)
    except ValueError as exc:
        raise ValueError(f"{ctx}: Non-integer action index {parts[5]!r}.") from exc
    if not 0 <= act < len(ca_raw):
        raise ValueError(f"{ctx}: Action index {act} out of legal range [0, {len(ca_raw)}).")

    if len(parts) == 6:
        return uuid, sp, nu, pr, ca, act, None, None, None, len(pr_raw)

    # Validate 7th column (round_summary) and 8th column (14-int results)
    rs_raw = _parse_int_csv(parts[6], "round_summary", ctx)
    if len(rs_raw) > MAX_NUM_ROUND_SUMMARY:
        raise ValueError(f"{ctx}: Too many round_summary tokens ({len(rs_raw)}).")
    if any(x < 0 or x >= NUM_TYPES_OF_ROUND_SUMMARY for x in rs_raw):
        raise ValueError(f"{ctx}: round_summary token out of [0, {NUM_TYPES_OF_ROUND_SUMMARY}).")

    res_raw = _parse_int_csv(parts[7], "results", ctx)
    if len(res_raw) != NUM_RESULTS:
        raise ValueError(f"{ctx}: Expected {NUM_RESULTS} fields in results, got {len(res_raw)}.")
    # results layout: [round_delta_scores(4), benchang(1), num_liqibangs(1), post_round_scores(4), final_game_scores(4)]
    if not 0 <= res_raw[4] < 255 or not 0 <= res_raw[5] < 255:
        raise ValueError(f"{ctx}: Invalid benchang/num_liqibangs in results: {res_raw[4:6]}.")
    if any(abs(s) > _SCORE_BOUND for s in res_raw[0:4] + res_raw[6:14]):
        raise ValueError(f"{ctx}: Score out of valid bounds in results: {res_raw}.")

    final_scores = (res_raw[10], res_raw[11], res_raw[12], res_raw[13])
    return uuid, sp, nu, pr, ca, act, final_scores, tuple(rs_raw), tuple(res_raw), len(pr_raw)


def parse_bc_line(
    line: str,
    *,
    allow_unlabeled: bool = True,
    line_no: int | None = None,
) -> tuple[str, list[int], list[int], list[int], list[int], int, tuple[int, int, int, int] | None]:
    """Parse one source-v3 BC line (8 cols with 14-int results) or explicit 6-col unlabeled row."""
    return _parse_bc_line_full(line, allow_unlabeled=allow_unlabeled, line_no=line_no)[:7]


def _decode_prog_token(tok: int) -> tuple[str, int, bool]:
    """Decode one progression action token into (kind, actor_abs, is_riichi_discard)."""
    if 5 <= tok < 597:
        v = tok - 5
        return "dahai", v // 148, bool((v % 148) & 1)
    if 597 <= tok < 957:
        return "chi", (tok - 597) // 90, False
    if 957 <= tok < 1437:
        return "pon", (tok - 957) // 120, False
    if 1437 <= tok < 1881:
        return "dmk", (tok - 1437) // 111, False
    if 1881 <= tok < 2017:
        return "ankan", (tok - 1881) // 34, False
    if 2017 <= tok < 2165:
        return "kakan", (tok - 2017) // 37, False
    raise ValueError(f"Unsupported or invalid progression token: {tok}.")


def _order_and_validate_prefix_steps(
    sp_t: Tensor,
    nu_t: Tensor,
    pr_t: Tensor,
    ca_t: Tensor,
    feats: dict[str, Tensor],
    *,
    ctx: Any,
    check_v2_events: bool = True,
) -> list[int]:
    """Order steps chronologically and reject duplicates or divergent/non-prefix histories."""
    n_steps = int(sp_t.shape[0])
    pr_lens_arr = (pr_t != NUM_TYPES_OF_PROGRESSION_FEATURES).sum(dim=-1).tolist()
    left_arr = (sp_t[:, 9] - 82).tolist()
    is_own_arr = (
        ((ca_t < 221) & (ca_t < NUM_TYPES_OF_ACTIONS)).any(dim=-1).to(torch.int64).tolist()
    )
    order = sorted(
        range(n_steps),
        key=lambda u: (int(pr_lens_arr[u]), -int(left_arr[u]), int(is_own_arr[u]), u),
    )

    ev_lens = feats["v2_event_len"]
    v2_events = feats["v2_events"]
    for pos in range(1, len(order)):
        prev_u = order[pos - 1]
        u = order[pos]
        prev_plen = int(pr_lens_arr[prev_u])
        u_plen = int(pr_lens_arr[u])
        prev_pr = pr_t[prev_u, :prev_plen]
        u_pr_prefix = pr_t[u, :prev_plen]

        if (
            prev_plen == u_plen
            and int(left_arr[prev_u]) == int(left_arr[u])
            and int(is_own_arr[prev_u]) == int(is_own_arr[u])
            and torch.equal(prev_pr, pr_t[u, :u_plen])
        ):
            raise ValueError(f"Duplicate decision step in round {ctx} at progression length {u_plen}.")

        if u_plen < prev_plen or not torch.equal(prev_pr, u_pr_prefix):
            raise ValueError(
                f"Divergent or non-prefix progression history in round {ctx} "
                f"(lengths {prev_plen} -> {u_plen})."
            )

        if check_v2_events:
            prev_ev_len = int(ev_lens[prev_u].item())
            u_ev_len = int(ev_lens[u].item())
            if u_ev_len < prev_ev_len or not torch.equal(
                v2_events[prev_u, :prev_ev_len], v2_events[u, :prev_ev_len]
            ):
                raise ValueError(f"Divergent v2_events stream in round {ctx}.")

    return order


def _validate_games_completeness(
    game_meta: dict[str, tuple[int, int]],
    phys_rounds: dict[tuple[str, int, int, int], dict[str, Any]],
) -> None:
    """Enforce complete-game & complete-round structural integrity across source-v3 games."""
    rounds_by_uuid: dict[str, list[tuple[int, int, int, dict[str, Any]]]] = {}
    for (uuid, chang, ju, ben), info in phys_rounds.items():
        # Verify cross-seat consistency within physical round (same max_pr prefix & round outcome)
        max_pr: tuple[int, ...] = max(info["all_prs"], key=len)
        if not max_pr or max_pr[0] != 0:
            raise ValueError(f"Physical round ({uuid}, {chang}, {ju}, {ben}) has invalid progression start.")
        for pr_seq in info["all_prs"]:
            if pr_seq != max_pr[: len(pr_seq)]:
                raise ValueError(
                    f"Observer seats disagree on progression inside round ({uuid}, {chang}, {ju}, {ben})."
                )

        # Dealer must open the round at progression == (0,)
        seat_prs: dict[int, set[tuple[int, ...]]] = info["seat_prs"]
        if ju not in seat_prs or (0,) not in seat_prs[ju]:
            raise ValueError(
                f"Truncated round start in ({uuid}, {chang}, {ju}, {ben}): missing dealer {ju} opening step."
            )

        # Every non-riichi action (or non-dahai action even during riichi) must have a recorded step
        riichi_active = [False, False, False, False]
        for i in range(1, len(max_pr)):
            kind, actor, is_riichi = _decode_prog_token(max_pr[i])
            if (not riichi_active[actor]) or (kind != "dahai"):
                prefix_i = max_pr[:i]
                if prefix_i not in seat_prs.get(actor, ()):
                    raise ValueError(
                        f"Incomplete round ({uuid}, {chang}, {ju}, {ben}): missing seat {actor} "
                        f"decision at progression length {i} for {kind}."
                    )
            if is_riichi:
                riichi_active[actor] = True

        # Validate trailing decision steps against round_summary
        rs_raw: tuple[int, ...] = info["rs_raw"]
        seat_actions_at_max: dict[int, set[int]] = info["seat_actions_at_max"]
        if rs_raw[0] <= 15:
            # Every winner encoded in round_summary must have declared Tsumo (219) or Ron (543..545) at max_pr
            for code in rs_raw:
                if code <= 3:
                    winner = code
                    expected_acts = {219}
                elif code <= 15:
                    winner = (code - 4) // 3
                    expected_acts = {543, 544, 545}
                else:
                    continue
                if not (seat_actions_at_max.get(winner, set()) & expected_acts):
                    raise ValueError(
                        f"Incomplete win round ({uuid}, {chang}, {ju}, {ben}): missing winning decision "
                        f"for seat {winner} (round_summary={rs_raw})."
                    )
        elif 16 <= rs_raw[0] <= 27:
            # Exhaustive wall draw must reach late wall (accounting for auto-tsumogiri / 4-kan remainder <= 4)
            if info["min_left"] > 4:
                raise ValueError(
                    f"Truncated exhaustive-draw round ({uuid}, {chang}, {ju}, {ben}): "
                    f"min left_tiles={info['min_left']} > 4."
                )

        rounds_by_uuid.setdefault(uuid, []).append((chang, ju, ben, info))

    for uuid, rlist in rounds_by_uuid.items():
        rlist.sort(key=lambda item: (item[0], item[1], item[2]))
        if (rlist[0][0], rlist[0][1], rlist[0][2]) != (0, 0, 0):
            raise ValueError(
                f"Truncated game {uuid}: first observed round is {(rlist[0][0], rlist[0][1], rlist[0][2])}, expected (0, 0, 0)."
            )
        for pos in range(1, len(rlist)):
            pc, pj, pb, _ = rlist[pos - 1]
            cc, cj, cb, _ = rlist[pos]
            p_idx = pc * 4 + pj
            c_idx = cc * 4 + cj
            same_dealer_cont = (c_idx == p_idx) and (cb == pb + 1)
            next_dealer_step = (c_idx == p_idx + 1) and (cb in (0, pb + 1))
            if not (same_dealer_cont or next_dealer_step):
                raise ValueError(
                    f"Truncated or non-contiguous game {uuid}: round {(pc, pj, pb)} -> {(cc, cj, cb)}."
                )

        lc, lj, _lb, last_info = rlist[-1]
        game_mode = game_meta[uuid][1]
        min_end_idx = 3 if game_mode == 5 else 7  # Tonpuusen East-4 (3) vs Hanchan South-4 (7)
        post_scores = tuple(int(x) for x in last_info["results"][6:10])
        final_scores = tuple(int(x) for x in last_info["results"][10:14])
        if not any(s < 0 for s in post_scores) and (lc * 4 + lj) < min_end_idx:
            raise ValueError(
                f"Truncated game {uuid}: ended early at (chang={lc}, ju={lj}) with post_scores={post_scores}."
            )
        if post_scores != final_scores:
            kyot = int(last_info["results"][5])
            draw_kyotaku_match = (
                last_info["rs_raw"][0] >= 16
                and kyot > 0
                and sorted(final_scores[i] - post_scores[i] for i in range(4)) == [0, 0, 0, 1000 * kyot]
            )
            if not draw_kyotaku_match:
                raise ValueError(
                    f"Truncated game {uuid}: final round post_scores={post_scores} "
                    f"do not match final_game_scores={final_scores}."
                )


def parse_bc_rounds(
    path: Path,
    *,
    max_rounds: int | None = None,
    max_lines: int | None = None,
    enable_early_shape_hint: bool = False,
    allow_unlabeled: bool = False,
    allow_incomplete: bool = False,
) -> list[RoundTrajectory]:
    """Read source-v3 BC TSV/TSV.GZ lines, validate isolation/completeness, and group rounds."""
    raw_groups: list[
        tuple[
            tuple[Any, ...],
            list[tuple[list[int], list[int], list[int], list[int], int, int, tuple[int, ...], int]],
        ]
    ] = []
    index_by_key: dict[tuple[Any, ...], int] = {}
    scores_by_game: dict[str, tuple[int, int, int, int] | None] = {}
    game_meta: dict[str, tuple[int, int]] = {}
    seen_labeled: bool | None = None

    phys_rounds: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    parsed_rows: list[
        tuple[
            tuple[Any, ...],
            list[int],
            list[int],
            list[int],
            list[int],
            int,
            tuple[int, int, int, int] | None,
            tuple[int, ...] | None,
            tuple[int, ...] | None,
            int,
        ]
    ] = []

    with _open_tsv(path) as fp:
        for line_no, line in enumerate(fp):
            if max_lines is not None and line_no >= max_lines:
                break
            uuid, sp, nu, pr, ca, act, final_scores, rs_raw, res_raw, pr_len = _parse_bc_line_full(
                line, allow_unlabeled=allow_unlabeled, line_no=line_no + 1
            )
            is_labeled = final_scores is not None
            if seen_labeled is None:
                seen_labeled = is_labeled
            elif seen_labeled != is_labeled:
                raise ValueError(
                    f"line {line_no + 1} ({uuid}): Mixed 6-column unlabeled and 8-column labeled rows."
                )

            seat = sp[SEAT_INDEX] - SEAT_OFFSET
            chang = sp[7] - 75
            ju = sp[8] - 78
            if not 0 <= seat < 4 or not 0 <= chang < 3 or not 0 <= ju < 4:
                raise ValueError(f"line {line_no + 1} ({uuid}): Invalid observer seat/chang/ju.")

            meta = (sp[0], sp[1])
            if uuid in game_meta:
                if game_meta[uuid] != meta:
                    raise ValueError(
                        f"line {line_no + 1} ({uuid}): Inconsistent room/mode {game_meta[uuid]} vs {meta}."
                    )
            else:
                game_meta[uuid] = meta

            if uuid in scores_by_game:
                if scores_by_game[uuid] != final_scores:
                    raise ValueError(
                        f"line {line_no + 1} ({uuid}): Inconsistent final_game_scores for game {uuid} "
                        f"(seat {seat}): {scores_by_game[uuid]} vs {final_scores}."
                    )
            else:
                scores_by_game[uuid] = final_scores

            if rs_raw is not None and res_raw is not None:
                phys_key = (uuid, chang, ju, nu[0])
                pinfo = phys_rounds.get(phys_key)
                if pinfo is None:
                    pinfo = {
                        "rs_raw": rs_raw,
                        "results": res_raw,
                        "all_prs": [],
                        "seat_prs": {},
                        "seat_actions_at_max": {},
                        "min_left": sp[9] - 82,
                    }
                    phys_rounds[phys_key] = pinfo
                else:
                    if (not allow_incomplete) and (
                        pinfo["rs_raw"] != rs_raw or pinfo["results"] != res_raw
                    ):
                        raise ValueError(
                            f"line {line_no + 1} ({uuid}): Conflicting round_summary/results within "
                            f"physical round {(chang, ju, nu[0])}."
                        )
                pr_tuple = tuple(pr[:pr_len])
                pinfo["all_prs"].append(pr_tuple)
                pinfo["seat_prs"].setdefault(seat, set()).add(pr_tuple)
                pinfo["min_left"] = min(int(pinfo["min_left"]), int(sp[9] - 82))

            key = (uuid, sp[SEAT_INDEX], sp[7], sp[8], nu[0])
            idx = index_by_key.get(key)
            if idx is None:
                idx = len(index_by_key)
                index_by_key[key] = idx
                raw_groups.append((key, []))
            parsed_rows.append((key, sp, nu, pr, ca, act, final_scores, rs_raw, res_raw, pr_len))

    if not parsed_rows:
        return []
    if not allow_incomplete and seen_labeled is False:
        raise ValueError(
            "Complete game/round reconstruction requires 8-column source-v3 rows with round_summary and results."
        )

    # Record each seat's chosen action at that physical round's longest progression for trailing win checks
    if phys_rounds:
        max_len_by_phys = {
            pk: max(len(x) for x in pinfo["all_prs"]) for pk, pinfo in phys_rounds.items()
        }
        for key, sp, nu, pr, ca, act, _fs, rs_raw, _rr, pr_len in parsed_rows:
            if rs_raw is None:
                continue
            pk = (key[0], sp[7] - 75, sp[8] - 78, nu[0])
            if pr_len == max_len_by_phys[pk]:
                seat = sp[SEAT_INDEX] - SEAT_OFFSET
                phys_rounds[pk]["seat_actions_at_max"].setdefault(seat, set()).add(ca[act])

    if not allow_incomplete:
        _validate_games_completeness(game_meta, phys_rounds)

    # Compute native tie-broken final_rank via scores_to_ranks for all labeled rows
    labeled_idx = [i for i, r in enumerate(parsed_rows) if r[6] is not None]
    computed_ranks: dict[int, int] = {}
    if labeled_idx:
        scores_t = torch.tensor([parsed_rows[i][6] for i in labeled_idx], dtype=torch.int64)
        ranks_t = scores_to_ranks(scores_t)
        seats_t = torch.tensor(
            [parsed_rows[i][1][SEAT_INDEX] - SEAT_OFFSET for i in labeled_idx],
            dtype=torch.long,
        )
        own_ranks = ranks_t.gather(1, seats_t.unsqueeze(1)).squeeze(1).tolist()
        for row_idx, rnk in zip(labeled_idx, own_ranks, strict=True):
            computed_ranks[row_idx] = int(rnk)

    for row_idx, (key, sp, nu, pr, ca, act, _fs, rs_raw, _rr, pr_len) in enumerate(parsed_rows):
        g_idx = index_by_key[key]
        final_rank = computed_ranks.get(row_idx, -1)
        raw_groups[g_idx][1].append((sp, nu, pr, ca, act, final_rank, rs_raw or (), pr_len))

    rounds: list[RoundTrajectory] = []
    for key, steps in raw_groups:
        sp_t = torch.tensor([s[0] for s in steps], dtype=torch.int32)
        nu_t = torch.tensor([s[1] for s in steps], dtype=torch.int32)
        pr_t = torch.tensor([s[2] for s in steps], dtype=torch.int32)
        ca_t = torch.tensor([s[3] for s in steps], dtype=torch.int32)
        act_t = torch.tensor([s[4] for s in steps], dtype=torch.long)
        rk_vals = [s[5] for s in steps]

        feats = derive_structured_v2_batch(
            sp_t, nu_t, pr_t, ca_t,
            enable_early_shape_hint=enable_early_shape_hint,
            include_room_level=False,
        )
        order = _order_and_validate_prefix_steps(sp_t, nu_t, pr_t, ca_t, feats, ctx=key)
        sub_idx = torch.tensor(order, dtype=torch.long)
        ev_lens = feats["v2_event_len"]
        last_i = order[-1]
        l_r = int(ev_lens[last_i].item())
        r_events = feats["v2_events"][last_i, :l_r].clone()
        rk_sub = [rk_vals[i] for i in order]
        has_lab = [r >= 0 for r in rk_sub]
        if any(has_lab) and not all(has_lab):
            raise ValueError(f"Round group {key} mixes labeled and unlabeled steps.")
        rk_t = torch.tensor(rk_sub, dtype=torch.long) if all(has_lab) else None
        rounds.append(
            RoundTrajectory(
                round_events=r_events,
                round_event_len=l_r,
                decision_event_len=ev_lens[sub_idx].to(torch.int32),
                v2_tile_feats=feats["v2_tile_feats"][sub_idx],
                v2_global_feats=feats["v2_global_feats"][sub_idx],
                v2_player_feats=feats["v2_player_feats"][sub_idx],
                v2_cand_ids=feats["v2_cand_ids"][sub_idx],
                v2_cand_mask=feats["v2_cand_mask"][sub_idx],
                v2_cand_mode=feats["v2_cand_mode"][sub_idx],
                v2_cand_dyn=feats["v2_cand_dyn"][sub_idx],
                candidates=ca_t[sub_idx],
                action_idx=act_t[sub_idx],
                final_rank=rk_t,
            )
        )
    if max_rounds is not None:
        return rounds[:max_rounds]
    return rounds


def build_round_trajectories_from_raw(
    sp: Tensor,
    nu: Tensor,
    pr: Tensor,
    ca: Tensor,
    actions: Tensor,
    round_group_ids: Sequence[Any],
    *,
    final_ranks: Tensor | None = None,
    enable_early_shape_hint: bool = False,
) -> list[RoundTrajectory]:
    """Group pre-loaded 4-field tensors into validated single-prefix RoundTrajectory objects."""
    feats = derive_structured_v2_batch(
        sp, nu, pr, ca,
        enable_early_shape_hint=enable_early_shape_hint,
        include_room_level=False,
    )
    grouped: dict[Any, list[int]] = {}
    for idx, gid in enumerate(round_group_ids):
        grouped.setdefault(gid, []).append(idx)

    if final_ranks is not None:
        rk_all = final_ranks.long()
        if not ((rk_all == -1) | ((rk_all >= 0) & (rk_all < 4))).all():
            raise ValueError("final_ranks values must be -1 or in [0, 3].")
        any_lab = bool((rk_all >= 0).any().item())
        all_lab = bool((rk_all >= 0).all().item())
        if any_lab and not all_lab:
            raise ValueError(
                "build_round_trajectories_from_raw: cannot mix labeled and unlabeled final_ranks across rounds."
            )

    rounds: list[RoundTrajectory] = []
    for gid, idxs in grouped.items():
        sub = torch.tensor(idxs, dtype=torch.long)
        sub_feats = {k: v[sub] for k, v in feats.items()}
        local_order = _order_and_validate_prefix_steps(
            sp[sub], nu[sub], pr[sub], ca[sub], sub_feats, ctx=gid, check_v2_events=False
        )
        order = sub[torch.tensor(local_order, dtype=torch.long)]
        last_i = int(order[-1].item())
        l_r = int(feats["v2_event_len"][last_i].item())
        rk_round: Tensor | None = None
        if final_ranks is not None:
            rk_vals = final_ranks[order].long()
            if bool((rk_vals >= 0).all().item()):
                rk_round = rk_vals
        rounds.append(
            RoundTrajectory(
                round_events=feats["v2_events"][last_i, :l_r].clone(),
                round_event_len=l_r,
                decision_event_len=feats["v2_event_len"][order].to(torch.int32),
                v2_tile_feats=feats["v2_tile_feats"][order],
                v2_global_feats=feats["v2_global_feats"][order],
                v2_player_feats=feats["v2_player_feats"][order],
                v2_cand_ids=feats["v2_cand_ids"][order],
                v2_cand_mask=feats["v2_cand_mask"][order],
                v2_cand_mode=feats["v2_cand_mode"][order],
                v2_cand_dyn=feats["v2_cand_dyn"][order],
                candidates=ca[order],
                action_idx=actions[order].long(),
                final_rank=rk_round,
            )
        )
    return rounds


class LengthBucketBatchSampler:
    """Bucketed round sampler minimizing round_events padding with exact checkpoint resume."""

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        seed: int = 0,
        bucket_mult: int = 8,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.bucket_mult = int(bucket_mult)
        self.epoch = 0
        self.cursor = 0
        self.batches: list[list[int]] = []
        self._build_epoch_batches()

    def _build_epoch_batches(self) -> None:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * 1009)
        n = len(self.lengths)
        if n == 0:
            self.batches = []
            self.cursor = 0
            return
        perm = torch.randperm(n, generator=g).tolist()
        pool_sz = max(self.batch_size, self.batch_size * self.bucket_mult)
        all_batches: list[list[int]] = []
        for s in range(0, n, pool_sz):
            chunk = sorted(perm[s : s + pool_sz], key=lambda i: self.lengths[i])
            for b in range(0, len(chunk), self.batch_size):
                all_batches.append(chunk[b : b + self.batch_size])
        batch_perm = torch.randperm(len(all_batches), generator=g).tolist()
        self.batches = [all_batches[i] for i in batch_perm]
        self.cursor = 0

    def reset_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._build_epoch_batches()

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[list[int]]:
        while self.cursor < len(self.batches):
            batch = self.batches[self.cursor]
            self.cursor += 1
            yield batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_items": len(self.lengths),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            state["seed"] != self.seed
            or state["batch_size"] != self.batch_size
            or state["num_items"] != len(self.lengths)
        ):
            raise ValueError("LengthBucketBatchSampler config mismatch on resume.")
        self.epoch = int(state["epoch"])
        self._build_epoch_batches()
        self.cursor = int(state["cursor"])


def _permute_events_only(events: Tensor, perm_idx: int) -> Tensor:
    """Apply suit permutation 0..5 to a [..., 10] events tensor."""
    if perm_idx == 0:
        return events
    ev = events.clone()
    p37 = torch.as_tensor(PERM_TILE37[perm_idx].copy(), dtype=ev.dtype, device=ev.device)
    for col in (3, 4, 5):
        valid = (ev[..., 0] > 0) & (ev[..., col] < 37)
        idx = ev[..., col].clamp(0, 36).long()
        ev[..., col] = torch.where(valid, p37[idx], ev[..., col])
    return ev


def collate_round_batch(
    rounds: Sequence[RoundTrajectory],
    *,
    suit_perm_generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Pad variable-length rounds to [B_r, L_max, ...] and [B_r, T_max, ...]."""
    bsz_r = len(rounds)
    l_max = max(r.round_event_len for r in rounds)
    t_max = max(int(r.decision_event_len.shape[0]) for r in rounds)

    round_events = torch.zeros((bsz_r, l_max, 10), dtype=torch.uint8)
    round_events[:, :, 3:6] = 37
    round_event_len = torch.zeros((bsz_r,), dtype=torch.int32)
    dec_event_len = torch.ones((bsz_r, t_max), dtype=torch.int32)
    dec_mask = torch.zeros((bsz_r, t_max), dtype=torch.bool)

    tile_feats = torch.zeros((bsz_r, t_max, 34, 70), dtype=torch.float32)
    global_feats = torch.zeros((bsz_r, t_max, 48), dtype=torch.float32)
    player_feats = torch.zeros((bsz_r, t_max, 4, 32), dtype=torch.float32)
    cand_ids = torch.full((bsz_r, t_max, 32), PAD_ACTION_ID, dtype=torch.int32)
    cand_mask = torch.zeros((bsz_r, t_max, 32), dtype=torch.bool)
    cand_mode = torch.zeros((bsz_r, t_max, 32), dtype=torch.uint8)
    cand_dyn = torch.zeros((bsz_r, t_max, 32, 8), dtype=torch.float32)
    candidates = torch.full((bsz_r, t_max, 32), PAD_ACTION_ID, dtype=torch.int32)
    action_idx = torch.zeros((bsz_r, t_max), dtype=torch.int64)
    final_rank = torch.full((bsz_r, t_max), -1, dtype=torch.int64)

    for b, r in enumerate(rounds):
        lr = r.round_event_len
        t = int(r.decision_event_len.shape[0])
        perm_idx = (
            int(torch.randint(0, 6, (1,), generator=suit_perm_generator).item())
            if suit_perm_generator is not None
            else 0
        )

        r_ev = _permute_events_only(r.round_events, perm_idx)
        round_events[b, :lr] = r_ev
        round_event_len[b] = lr
        dec_event_len[b, :t] = r.decision_event_len
        dec_mask[b, :t] = True

        if perm_idx != 0:
            dummy_ev = torch.zeros((t, 1, 10), dtype=torch.uint8)
            step_batch = {
                "v2_tile_feats": r.v2_tile_feats,
                "v2_events": dummy_ev,
                "v2_global_feats": r.v2_global_feats,
                "v2_player_feats": r.v2_player_feats,
                "v2_cand_ids": r.v2_cand_ids,
                "v2_cand_mask": r.v2_cand_mask,
                "v2_cand_mode": r.v2_cand_mode,
                "v2_cand_dyn": r.v2_cand_dyn,
                "action": r.action_idx,
            }
            p_batch = permute_structured_batch(step_batch, perm_idx)
            tf_s = p_batch["v2_tile_feats"]
            gf_s = p_batch["v2_global_feats"]
            pf_s = p_batch["v2_player_feats"]
            cids_s = p_batch["v2_cand_ids"]
            cmask_s = p_batch["v2_cand_mask"]
            cmode_s = p_batch["v2_cand_mode"]
            cdyn_s = p_batch["v2_cand_dyn"]
            cands_s = cids_s
            act_s = p_batch["action"]
        else:
            tf_s = r.v2_tile_feats
            gf_s = r.v2_global_feats
            pf_s = r.v2_player_feats
            cids_s = r.v2_cand_ids
            cmask_s = r.v2_cand_mask
            cmode_s = r.v2_cand_mode
            cdyn_s = r.v2_cand_dyn
            cands_s = r.candidates
            act_s = r.action_idx

        tile_feats[b, :t] = tf_s
        global_feats[b, :t] = gf_s
        player_feats[b, :t] = pf_s
        cand_ids[b, :t] = cids_s
        cand_mask[b, :t] = cmask_s
        cand_mode[b, :t] = cmode_s
        cand_dyn[b, :t] = cdyn_s
        candidates[b, :t] = cands_s
        action_idx[b, :t] = act_s
        if r.final_rank is not None:
            final_rank[b, :t] = r.final_rank

    return {
        "round_events": round_events,
        "round_event_len": round_event_len,
        "decision_event_len": dec_event_len,
        "decision_mask": dec_mask,
        "v2_tile_feats": tile_feats,
        "v2_global_feats": global_feats,
        "v2_player_feats": player_feats,
        "v2_cand_ids": cand_ids,
        "v2_cand_mask": cand_mask,
        "v2_cand_mode": cand_mode,
        "v2_cand_dyn": cand_dyn,
        "candidates": candidates,
        "action": action_idx,
        "final_rank": final_rank,
    }


@torch.no_grad()
def evaluate_9_action_families(
    backbone: StructuredDualDecoder,
    policy_head: StructuredActionHead,
    batches: Sequence[dict[str, Tensor]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Compute per-family NLL, Top-1/Top-2 accuracy (K >= 2) and Riichi F1 (§4.2.3)."""
    backbone.eval()
    policy_head.eval()
    stats: dict[str, dict[str, float]] = {
        fam: {"count": 0.0, "nll_sum": 0.0, "top1": 0.0, "top2": 0.0}
        for fam in ACTION_FAMILY_NAMES
    }
    riichi_tp = riichi_fp = riichi_fn = 0.0

    for batch in batches:
        b_dev = {k: v.to(device=device) for k, v in batch.items()}
        out = backbone.forward_round_trajectory(
            b_dev["round_events"],
            b_dev["round_event_len"],
            b_dev["decision_event_len"],
            b_dev["decision_mask"],
            b_dev["v2_tile_feats"],
            b_dev["v2_global_feats"],
            b_dev["v2_player_feats"],
            b_dev["v2_cand_ids"],
            b_dev["v2_cand_mask"],
            b_dev["v2_cand_mode"],
            b_dev["v2_cand_dyn"],
        )
        dmask = b_dev["decision_mask"].bool()
        cands = b_dev["candidates"][dmask]
        actions = b_dev["action"][dmask].long()
        encode = out["encode"]
        logits = policy_head(cands, encode).float()
        cand_mask = cands < NUM_TYPES_OF_ACTIONS
        has_choice = cand_mask.sum(dim=-1) >= 2

        masked_logits = logits.masked_fill(~cand_mask, -torch.inf)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        preds_top2 = torch.topk(masked_logits, k=min(2, masked_logits.shape[1]), dim=-1).indices

        cands_cpu = cands.cpu()
        acts_cpu = actions.cpu()
        has_choice_cpu = has_choice.cpu()
        lp_cpu = log_probs.cpu()
        top2_cpu = preds_top2.cpu()

        for i in range(int(acts_cpu.shape[0])):
            if not bool(has_choice_cpu[i].item()):
                continue
            gt_idx = int(acts_cpu[i].item())
            gt_gid = int(cands_cpu[i, gt_idx].item())
            fam = classify_action_id(gt_gid)
            nll = -float(lp_cpu[i, gt_idx].item())
            p1 = int(top2_cpu[i, 0].item())
            p2 = int(top2_cpu[i, 1].item()) if top2_cpu.shape[1] > 1 else p1
            st = stats[fam]
            st["count"] += 1.0
            st["nll_sum"] += nll
            st["top1"] += 1.0 if p1 == gt_idx else 0.0
            st["top2"] += 1.0 if (gt_idx == p1 or gt_idx == p2) else 0.0

            # Track Riichi F1 over states where at least one legal riichi candidate exists
            legal_gids = cands_cpu[i][cands_cpu[i] < NUM_TYPES_OF_ACTIONS].tolist()
            has_riichi_opt = any(
                int(KANACHAN_ACTION_MANIFEST[g, 0]) == 0
                and int(KANACHAN_ACTION_MANIFEST[g, 8]) == 1
                for g in legal_gids
            )
            if has_riichi_opt:
                pred_gid = int(cands_cpu[i, p1].item())
                pred_riichi = (
                    int(KANACHAN_ACTION_MANIFEST[pred_gid, 0]) == 0
                    and int(KANACHAN_ACTION_MANIFEST[pred_gid, 8]) == 1
                )
                gt_riichi = fam == "DISCARD_RIICHI"
                if pred_riichi and gt_riichi:
                    riichi_tp += 1.0
                elif pred_riichi and not gt_riichi:
                    riichi_fp += 1.0
                elif (not pred_riichi) and gt_riichi:
                    riichi_fn += 1.0

    result: dict[str, Any] = {}
    tot_c = tot_nll = tot_t1 = tot_t2 = 0.0
    for fam, st in stats.items():
        c = st["count"]
        tot_c += c
        tot_nll += st["nll_sum"]
        tot_t1 += st["top1"]
        tot_t2 += st["top2"]
        result[fam] = {
            "count": int(c),
            "nll": (st["nll_sum"] / c) if c > 0 else 0.0,
            "top1_acc": (st["top1"] / c) if c > 0 else 0.0,
            "top2_acc": (st["top2"] / c) if c > 0 else 0.0,
        }
    prec = riichi_tp / max(1.0, riichi_tp + riichi_fp)
    rec = riichi_tp / max(1.0, riichi_tp + riichi_fn)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    result["overall"] = {
        "count": int(tot_c),
        "nll": (tot_nll / tot_c) if tot_c > 0 else 0.0,
        "top1_acc": (tot_t1 / tot_c) if tot_c > 0 else 0.0,
        "top2_acc": (tot_t2 / tot_c) if tot_c > 0 else 0.0,
        "riichi_f1": f1,
        "riichi_precision": prec,
        "riichi_recall": rec,
    }
    return result


class TrajectoryBCModel(nn.Module):
    """Combined module wrapping backbone + policy_head + optional NextEvent & FinalRank heads."""

    def __init__(
        self,
        *,
        preset: str | None = "v2_base",
        dimension: int | None = None,
        num_heads: int | None = None,
        dim_feedforward: int | None = None,
        n_d1_layers: int | None = None,
        n_d2_layers: int | None = None,
        use_rel_bias: bool | None = None,
        cand_self_attn: bool = True,
        head_type: str | None = None,
        enable_early_shape_hint: bool = False,
        checkpointing: bool = False,
        use_next_event_head: bool = True,
        next_event_weight: float = 0.20,
        use_final_rank_head: bool = False,
        final_rank_weight: float = 0.10,
        label_smoothing: float = 0.0,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(label_smoothing) < 1.0:
            raise ValueError(f"label_smoothing must be in [0.0, 1.0), got {label_smoothing}.")
        self.label_smoothing = float(label_smoothing)
        self.backbone = StructuredDualDecoder(
            preset=preset,
            dimension=dimension,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            n_d1_layers=n_d1_layers,
            n_d2_layers=n_d2_layers,
            use_rel_bias=use_rel_bias,
            cand_self_attn=cand_self_attn,
            head_type=head_type,
            enable_early_shape_hint=enable_early_shape_hint,
            checkpointing=checkpointing,
            device=device,
            dtype=dtype,
        )
        d = self.backbone.dimension
        self.policy_head = StructuredActionHead(
            input_dimension=d,
            head_type=self.backbone.head_type,
            device=device,
            dtype=dtype,
        )
        self.next_event_head: NextEventHead | None = (
            NextEventHead(d, device=device, dtype=dtype)
            if use_next_event_head
            else None
        )
        self.state_head: StructuredStateHead | None = (
            StructuredStateHead(input_dimension=d, final_rank=True, device=device, dtype=dtype)
            if use_final_rank_head
            else None
        )
        self.next_event_weight = float(next_event_weight)
        self.final_rank_weight = float(final_rank_weight)
        if self.next_event_head is not None and self.next_event_weight <= 0.0:
            self.next_event_head.requires_grad_(False)
        if self.state_head is not None:
            # Train only final_rank when state value isn't supervised in BC.
            for p in self.state_head.parameters():
                p.requires_grad_(False)
            if self.final_rank_weight > 0.0 and self.state_head.final_rank is not None:
                self.state_head.final_rank.requires_grad_(True)
                # FinalRank uses RMSNorm + first linear layers as well!
                self.state_head.norm.requires_grad_(True)
                self.state_head.layers[0].requires_grad_(True)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        out = self.backbone.forward_round_trajectory(
            batch["round_events"],
            batch["round_event_len"],
            batch["decision_event_len"],
            batch["decision_mask"],
            batch["v2_tile_feats"],
            batch["v2_global_feats"],
            batch["v2_player_feats"],
            batch["v2_cand_ids"],
            batch["v2_cand_mask"],
            batch["v2_cand_mode"],
            batch["v2_cand_dyn"],
            flatten_active=False,
        )
        bsz_r, tmax = batch["decision_mask"].shape[:2]
        ntot = bsz_r * tmax
        dmask = batch["decision_mask"].bool().reshape(ntot)
        cands = batch["candidates"].reshape(ntot, MAX_NUM_ACTION_CANDIDATES)
        actions = batch["action"].reshape(ntot).long()
        encode = out["encode"]

        with torch.autocast(device_type=encode.device.type, enabled=False):
            logits = self.policy_head(cands, encode).float()
            cand_mask = cands < NUM_TYPES_OF_ACTIONS
            has_choice = dmask & (cand_mask.sum(dim=-1) >= 2)
            has_f = has_choice.float()

            # Padded decision steps may have all-False cand_mask; avoid all-(-inf) log_softmax NaN on masked slots.
            safe_cand_mask = torch.where(
                dmask.unsqueeze(-1),
                cand_mask,
                F.one_hot(torch.zeros_like(actions), MAX_NUM_ACTION_CANDIDATES).bool(),
            )
            masked_logits = logits.masked_fill(~safe_cand_mask, -torch.inf)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            log_probs_safe = torch.where(safe_cand_mask, log_probs, torch.zeros_like(log_probs))
            nll_per_step = F.nll_loss(log_probs_safe, actions, reduction="none")
            if self.label_smoothing > 0.0:
                eps = self.label_smoothing
                k_cnt = safe_cand_mask.float().sum(dim=-1).clamp_min(1.0)
                uniform_nll = -(log_probs_safe.sum(dim=-1) / k_cnt)
                step_policy_loss = (1.0 - eps) * nll_per_step + eps * uniform_nll
            else:
                step_policy_loss = nll_per_step
            policy_loss = (step_policy_loss * has_f).sum() / has_f.sum().clamp_min(1.0)

            loss_total = policy_loss
            metrics: dict[str, Tensor] = {
                "loss_policy": policy_loss.detach(),
                "num_choice_steps": has_f.sum().detach(),
            }

            if self.next_event_head is not None and self.next_event_weight > 0.0:
                nev_loss, nev_parts = self.next_event_head(
                    out["h_ev"], batch["round_events"], batch["round_event_len"]
                )
                loss_total = loss_total + self.next_event_weight * nev_loss
                metrics["loss_next_event"] = nev_loss.detach()
                for k, v in nev_parts.items():
                    metrics[f"next_ev_{k}"] = v

            if self.state_head is not None and self.final_rank_weight > 0.0:
                val_out, rank_logits = self.state_head(encode)
                rk_tgt = batch["final_rank"].reshape(ntot).long()
                require_tensor(
                    (~dmask | ((rk_tgt >= 0) & (rk_tgt < 4))).all(),
                    "Invalid final_rank target; expected [0, 3] when final_rank_weight > 0.0.",
                )
                lab_f = dmask.float()
                rk_ce = F.cross_entropy(
                    rank_logits.float(), rk_tgt.clamp(0, 3), reduction="none"
                )
                rk_loss = (rk_ce * lab_f).sum() / lab_f.sum().clamp_min(1.0)
                if val_out.requires_grad:
                    rk_loss = rk_loss + val_out.float().sum() * 0.0
                loss_total = loss_total + self.final_rank_weight * rk_loss
                metrics["loss_final_rank"] = rk_loss.detach()

        metrics["loss"] = loss_total
        return metrics

    def recipe(self, *, data_sha256: str = "", lr: float = 0.0, weight_decay: float = 0.0) -> dict[str, Any]:
        return {
            "format_version": 1,
            "architecture": ARCHITECTURE_ID,
            "manifest_sha256": KANACHAN_ACTION_MANIFEST_SHA256,
            "schema": structured_v2_schema(),
            "backbone": {
                "preset": self.backbone.preset,
                "dimension": self.backbone.dimension,
                "num_heads": self.backbone.num_heads,
                "dim_feedforward": self.backbone.dim_feedforward,
                "n_d1_layers": self.backbone.n_d1_layers,
                "n_d2_layers": self.backbone.n_d2_layers,
                "use_rel_bias": self.backbone.use_rel_bias,
                "cand_self_attn": self.backbone.cand_self_attn,
                "head_type": self.backbone.head_type,
                "enable_early_shape_hint": self.backbone.enable_early_shape_hint,
                "include_room_level": False,
                "rope_base": self.backbone.rope_base,
            },
            "use_next_event_head": self.next_event_head is not None,
            "next_event_weight": self.next_event_weight,
            "use_final_rank_head": self.state_head is not None,
            "final_rank_weight": self.final_rank_weight,
            "label_smoothing": self.label_smoothing,
            "data_sha256": data_sha256,
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        }

    def dump_actor(self, output_path: Path, *, dtype: torch.dtype | None = None) -> None:
        save_dtype = dtype or next(self.backbone.parameters()).dtype
        enc_tdm = TensorDictModule(
            self.backbone,
            in_keys=["sparse", "numeric", "progression", "candidates"],
            out_keys=["encode"],
        )
        head_tdm = TensorDictModule(
            self.policy_head,
            in_keys=["candidates", "encode"],
            out_keys=["decode"],
        )
        lp_mod = DecodeConverter("log_probs")
        lp_tdm = TensorDictModule(
            lp_mod, in_keys=["candidates", "decode"], out_keys=["log_probs"]
        )
        arg_mod = DecodeConverter("argmax", tie_break_by_action_id=True)
        arg_tdm = TensorDictModule(
            arg_mod, in_keys=["candidates", "log_probs"], out_keys=["action"]
        )
        seq = TensorDictSequential(enc_tdm, head_tdm, lp_tdm, arg_tdm)
        state = dump_model(
            seq,
            [
                dump_object(
                    enc_tdm,
                    [
                        dump_object(
                            self.backbone,
                            [],
                            {
                                "preset": self.backbone.preset,
                                "dimension": self.backbone.dimension,
                                "num_heads": self.backbone.num_heads,
                                "dim_feedforward": self.backbone.dim_feedforward,
                                "n_d1_layers": self.backbone.n_d1_layers,
                                "n_d2_layers": self.backbone.n_d2_layers,
                                "use_rel_bias": self.backbone.use_rel_bias,
                                "cand_self_attn": self.backbone.cand_self_attn,
                                "head_type": self.backbone.head_type,
                                "enable_early_shape_hint": self.backbone.enable_early_shape_hint,
                                "include_room_level": False,
                                "rope_base": self.backbone.rope_base,
                                "checkpointing": False,
                                "device": torch.device("cpu"),
                                "dtype": save_dtype,
                            },
                        )
                    ],
                    {
                        "in_keys": ["sparse", "numeric", "progression", "candidates"],
                        "out_keys": ["encode"],
                    },
                ),
                dump_object(
                    head_tdm,
                    [
                        dump_object(
                            self.policy_head,
                            [],
                            {
                                "input_dimension": self.policy_head.input_dimension,
                                "hidden_dimension": self.policy_head.hidden_dimension,
                                "head_type": self.policy_head.head_type,
                                "device": torch.device("cpu"),
                                "dtype": save_dtype,
                            },
                        )
                    ],
                    {"in_keys": ["candidates", "encode"], "out_keys": ["decode"]},
                ),
                dump_object(
                    lp_tdm,
                    [dump_object(lp_mod, ["log_probs"], {})],
                    {"in_keys": ["candidates", "decode"], "out_keys": ["log_probs"]},
                ),
                dump_object(
                    arg_tdm,
                    [dump_object(arg_mod, ["argmax"], {"tie_break_by_action_id": True})],
                    {"in_keys": ["candidates", "log_probs"], "out_keys": ["action"]},
                ),
            ],
            {},
        )
        cpu_sd: dict[str, Any] = {}
        for k, v in state["state_dict"].items():
            if isinstance(v, Tensor):
                cpu_sd[k] = (
                    v.detach().to(device="cpu", dtype=save_dtype)
                    if v.is_floating_point()
                    else v.detach().to(device="cpu")
                )
            else:
                cpu_sd[k] = v
        state["state_dict"] = cpu_sd
        atomic_save(Path(output_path), state)


def save_trajectory_bc_checkpoint(
    ckpt_dir: Path,
    model: TrajectoryBCModel,
    optimizer: torch.optim.Optimizer,
    *,
    sampler: LengthBucketBatchSampler,
    step: int,
    recipe: dict[str, Any],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_scaler: GradScaler | None = None,
    aug_generator: torch.Generator | None = None,
    numerical_stats: dict[str, int] | None = None,
) -> Path:
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    assert isinstance(raw, TrajectoryBCModel)
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1

    local_rng = capture_rng()
    if is_dist:
        rng_by_rank: list[Any] = [None] * world_size
        dist.all_gather_object(rng_by_rank, local_rng)
    else:
        rng_by_rank = [local_rng]

    ckpt_path = ckpt_dir / f"traj-bc-{step}.pt"
    actor_path = ckpt_dir / f"model.{step}.kanachan"
    err_holder: list[str | None] = [None]
    if rank == 0:
        try:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            stats_clean = {str(k): int(v) for k, v in (numerical_stats or {}).items()}
            payload = {
                "step": int(step),
                "world_size": int(world_size),
                "recipe": recipe,
                "numerical_stats": stats_clean,
                "model": cpu_state(raw.state_dict()),
                "optimizer": cpu_state(optimizer.state_dict()),
                "scheduler": cpu_state(scheduler.state_dict()) if scheduler is not None else None,
                "grad_scaler": cpu_state(grad_scaler.state_dict())
                if grad_scaler is not None and grad_scaler.is_enabled()
                else None,
                "sampler": sampler.state_dict(),
                "rng": rng_by_rank[0],
                "rng_by_rank": rng_by_rank,
                "aug_rng": aug_generator.get_state().cpu().clone() if aug_generator is not None else None,
            }
            atomic_save(ckpt_path, payload)
            raw.dump_actor(actor_path)
        except BaseException as exc:  # noqa: BLE001
            err_holder[0] = f"{type(exc).__name__}: {exc}"

    if is_dist:
        dist.broadcast_object_list(err_holder, src=0)
    if err_holder[0] is not None:
        raise RuntimeError(f"Rank-0 Trajectory BC checkpoint save failed: {err_holder[0]}")
    return ckpt_path


def load_trajectory_bc_checkpoint(
    ckpt_path: Path,
    model: TrajectoryBCModel,
    optimizer: torch.optim.Optimizer,
    *,
    expected_recipe: dict[str, Any],
    sampler: LengthBucketBatchSampler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_scaler: GradScaler | None = None,
    aug_generator: torch.Generator | None = None,
    map_location: torch.device | str = "cpu",
    stats_out: dict[str, int] | None = None,
) -> int:
    del map_location  # Load on CPU first to preserve rank/device isolation and avoid VRAM spikes.
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if payload.get("recipe") != expected_recipe:
        raise ValueError("Trajectory BC resume recipe/manifest/hyperparameters differ (Fail-Closed).")
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    assert isinstance(raw, TrajectoryBCModel)
    raw.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    move_optimizer_state(optimizer)
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if grad_scaler is not None and grad_scaler.is_enabled() and payload.get("grad_scaler") is not None:
        grad_scaler.load_state_dict(payload["grad_scaler"])
    if sampler is not None:
        sampler.load_state_dict(payload["sampler"])

    if stats_out is not None:
        stats_out.clear()
        for k, v in (payload.get("numerical_stats") or {}).items():
            stats_out[str(k)] = int(v)

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1
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

    if aug_generator is not None and payload.get("aug_rng") is not None:
        aug_generator.set_state(payload["aug_rng"].cpu())
    return int(payload["step"])


def _compute_bc_grad_norm_and_nonfinite(
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


def train_trajectory_bc_steps(
    model: TrajectoryBCModel,
    rounds: Sequence[RoundTrajectory],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int = 8,
    max_steps: int | None = None,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    suit_augmentation: bool = False,
    aug_seed: int = 1337,
    sampler_seed: int = 42,
    grad_clip: float = 1.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_scaler: GradScaler | None = None,
    ckpt_dir: Path | None = None,
    checkpoint_every: int = 0,
    resume_from: Path | None = None,
    data_sha256: str = "",
    log_interval: int = 50,
    check_finite_interval: int = 50,
    max_nonfinite_steps: int | None = None,
    loss_transform: Callable[[Tensor, int], Tensor] | None = None,
    post_backward_hook: Callable[[nn.Module, torch.optim.Optimizer, int], None] | None = None,
) -> dict[str, Any]:
    """Deterministic single/multi-step round-trajectory BC training loop with pre-update safety."""
    sampler = LengthBucketBatchSampler(
        [r.round_event_len for r in rounds], batch_size=batch_size, seed=sampler_seed
    )
    aug_gen: torch.Generator | None = None
    if suit_augmentation:
        aug_gen = torch.Generator(device="cpu")
        aug_gen.manual_seed(aug_seed)

    base_lr = float(optimizer.param_groups[0]["lr"])
    base_wd = float(optimizer.param_groups[0].get("weight_decay", 0.0))
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    assert isinstance(raw, TrajectoryBCModel)
    recipe = raw.recipe(data_sha256=data_sha256, lr=base_lr, weight_decay=base_wd)

    is_amp = amp_dtype is not None and device.type == "cuda" and amp_dtype != torch.float32
    scaler = grad_scaler or GradScaler("cuda", enabled=(is_amp and amp_dtype == torch.float16))

    start_step = 0
    stats_restored: dict[str, int] = {}
    if resume_from is not None:
        start_step = load_trajectory_bc_checkpoint(
            resume_from,
            raw,
            optimizer,
            expected_recipe=recipe,
            sampler=sampler,
            scheduler=scheduler,
            grad_scaler=scaler,
            aug_generator=aug_gen,
            map_location=device,
            stats_out=stats_restored,
        )

    is_dist = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_dist else 1
    can_gpu_fused_guard = (
        device.type == "cuda"
        and bool(optimizer.param_groups)
        and all(bool(g.get("fused", False)) for g in optimizer.param_groups)
        and isinstance(optimizer, torch.optim.AdamW)
        and scheduler is None
    )
    one_f32 = torch.ones((), device=device, dtype=torch.float32)
    zero_f32 = torch.zeros((), device=device, dtype=torch.float32)
    clip_f32 = torch.full((), float(grad_clip), device=device, dtype=torch.float32)

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

    model.train()
    step = start_step
    last_metrics: dict[str, float] = {}
    last_stats: dict[str, int] = {
        "optimizer_steps": int(stats_restored.get("optimizer_steps", 0)),
        "skipped_steps": int(stats_restored.get("skipped_steps", 0)),
        "nonfinite_loss_steps": int(stats_restored.get("nonfinite_loss_steps", 0)),
        "nonfinite_grad_steps": int(stats_restored.get("nonfinite_grad_steps", 0)),
    }
    last_out: dict[str, Tensor] | None = None

    def _flush_metrics(out_map: dict[str, Tensor]) -> tuple[dict[str, float], dict[str, int]]:
        keys = list(out_map.keys())
        packed = torch.stack(
            [
                optimizer_steps_t.double(),
                skipped_steps_t.double(),
                nonfinite_loss_steps_t.double(),
                nonfinite_grad_steps_t.double(),
            ]
            + [out_map[k].detach().double() for k in keys]
        ).cpu().tolist()
        stats = {
            "optimizer_steps": int(round(packed[0])),
            "skipped_steps": int(round(packed[1])),
            "nonfinite_loss_steps": int(round(packed[2])),
            "nonfinite_grad_steps": int(round(packed[3])),
        }
        metrics_f = {k: float(v) for k, v in zip(keys, packed[4:], strict=True)}
        if max_nonfinite_steps is not None and stats["skipped_steps"] > int(max_nonfinite_steps):
            raise RuntimeError(
                f"Exceeded max_nonfinite_steps={max_nonfinite_steps}: skipped={stats['skipped_steps']} "
                f"(nonfinite_loss={stats['nonfinite_loss_steps']}, "
                f"nonfinite_grad={stats['nonfinite_grad_steps']})."
            )
        return metrics_f, stats

    while max_steps is None or step < max_steps:
        for idxs in sampler:
            if max_steps is not None and step >= max_steps:
                break
            batch_rounds = [rounds[i] for i in idxs]
            batch_cpu = collate_round_batch(batch_rounds, suit_perm_generator=aug_gen)
            batch_dev = {k: v.to(device=device, non_blocking=True) for k, v in batch_cpu.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=bool(is_amp),
                cache_enabled=False,
            ):
                out = model(batch_dev)
                loss = out["loss"]
                if loss_transform is not None:
                    loss = loss_transform(loss, step)

            loss_bad = (~torch.isfinite(loss.detach())).reshape(())
            if loss.requires_grad:
                scaler.scale(loss).backward()
            if post_backward_hook is not None:
                post_backward_hook(model, optimizer, step)
            if scaler.is_enabled():
                scaler.unscale_(optimizer)

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        grads.append(p.grad)

            grad_norm, grad_bad = _compute_bc_grad_norm_and_nonfinite(grads, device=device)
            if scaler.is_enabled():
                opt_state = scaler._per_optimizer_states[id(optimizer)]
                found_map = opt_state["found_inf_per_device"]
                scaler_bad = zero_f32 > 1.0
                for v in found_map.values():
                    scaler_bad = scaler_bad | (v.to(device=device, dtype=torch.float32) > 0.0)
                grad_bad = grad_bad | scaler_bad
            step_bad = loss_bad | grad_bad

            if is_dist and world_size > 1:
                sync_dev = torch.device("cpu") if dist.get_backend() == "gloo" else device
                flags_sync = torch.stack([loss_bad, grad_bad]).to(device=sync_dev, dtype=torch.int32)
                dist.all_reduce(flags_sync, op=dist.ReduceOp.MAX)
                flags_dev = (flags_sync > 0).to(device=device)
                loss_bad = flags_dev[0]
                grad_bad = flags_dev[1]
                step_bad = loss_bad | grad_bad

            if params_with_grad and grad_clip > 0.0:
                safe_norm = torch.where(step_bad, clip_f32, grad_norm)
                torch.nn.utils.clip_grads_with_norm_(
                    params_with_grad, float(grad_clip), safe_norm
                )

            found_inf_f32 = step_bad.to(dtype=torch.float32)
            if scaler.is_enabled():
                opt_state = scaler._per_optimizer_states[id(optimizer)]
                found_map = opt_state["found_inf_per_device"]
                if found_map:
                    for d_key in list(found_map.keys()):
                        cur = found_map[d_key]
                        found_map[d_key] = torch.maximum(
                            cur, found_inf_f32.to(device=cur.device, dtype=cur.dtype)
                        )
                else:
                    found_map[device] = found_inf_f32

            if can_gpu_fused_guard:
                if scaler.is_enabled():
                    optimizer.grad_scale = one_f32  # type: ignore[attr-defined]
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.grad_scale = one_f32  # type: ignore[attr-defined]
                    optimizer.found_inf = found_inf_f32  # type: ignore[attr-defined]
                    try:
                        optimizer.step()
                    finally:
                        if hasattr(optimizer, "grad_scale"):
                            del optimizer.grad_scale  # type: ignore[attr-defined]
                        if hasattr(optimizer, "found_inf"):
                            del optimizer.found_inf  # type: ignore[attr-defined]
            else:
                bad_host = bool(step_bad.item())
                if not bad_host:
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                elif scaler.is_enabled():
                    scaler.update()

            optimizer.zero_grad(set_to_none=True)

            step += 1
            last_out = out
            step_bad_i = step_bad.to(dtype=torch.int64)
            skipped_steps_t.add_(step_bad_i)
            optimizer_steps_t.add_(1 - step_bad_i)
            nonfinite_loss_steps_t.add_(loss_bad.to(dtype=torch.int64))
            nonfinite_grad_steps_t.add_(grad_bad.to(dtype=torch.int64))

            at_end = (max_steps is not None and step >= max_steps) or (
                max_steps is None and sampler.cursor >= len(sampler.batches)
            )
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
                save_trajectory_bc_checkpoint(
                    ckpt_dir,
                    raw,
                    optimizer,
                    sampler=sampler,
                    step=step,
                    recipe=recipe,
                    scheduler=scheduler,
                    grad_scaler=scaler,
                    aug_generator=aug_gen,
                    numerical_stats=last_stats,
                )
        if max_steps is None:
            break
        if step < max_steps:
            sampler.reset_epoch(sampler.epoch + 1)

    if last_out is not None and not last_metrics:
        last_metrics, last_stats = _flush_metrics(last_out)

    return {
        "step": step,
        "optimizer_steps": last_stats["optimizer_steps"],
        "skipped_steps": last_stats["skipped_steps"],
        "nonfinite_loss_steps": last_stats["nonfinite_loss_steps"],
        "nonfinite_grad_steps": last_stats["nonfinite_grad_steps"],
        "metrics": last_metrics,
    }


__all__ = [
    "ACTION_FAMILY_NAMES",
    "LengthBucketBatchSampler",
    "RoundTrajectory",
    "TrajectoryBCModel",
    "build_round_trajectories_from_raw",
    "classify_action_id",
    "collate_round_batch",
    "evaluate_9_action_families",
    "load_trajectory_bc_checkpoint",
    "parse_bc_line",
    "parse_bc_rounds",
    "save_trajectory_bc_checkpoint",
    "train_trajectory_bc_steps",
]
