"""Independent pure-Python reference implementation of structured_v2 (§2.1-2.6)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld
from mahjong.shanten import Shanten

from kanachan.structured_v2 import (
    CAND_DYN_DIM,
    CHI_TABLE,
    EVENT_BYTES,
    GLOBAL_FEATS_DIM,
    KANACHAN_ACTION_MANIFEST,
    MAX_CANDIDATES,
    MAX_PROGRESSION_LEN,
    NUM_TYPES_OF_ACTIONS,
    OFFSETS_136,
    PAD_ACTION_ID,
    PLAYER_FEATS_DIM,
    PON_TABLE,
    TILE37_NONE,
    TILE_FEATS_DIM,
    base34_to_tile37,
    idx136_to_tile37,
    is_red5_tile37,
    tile34,
)
def _scores_to_ranks_1d(scores: Sequence[int]) -> list[int]:
    ranks = [0] * 4
    for i in range(4):
        r = 0
        for j in range(4):
            if i != j and (scores[j] > scores[i] or (scores[j] == scores[i] and j < i)):
                r += 1
        ranks[i] = r
    return ranks


_SHANTEN = Shanten()
_HAND_CALC = HandCalculator()
_OPTIONAL_RULES = OptionalRules(has_open_tanyao=True, has_aka_dora=True)
_LN61 = math.log(61.0)
_LN64001 = math.log(64001.0)

# Meld representation matching kanachan.simulation.Tool
_MeldItem = tuple[str, list[int], bool, int]  # (type_str, tiles136, opened, b34)


def _next_dora_b34(ind_b34: int) -> int:
    if ind_b34 < 27:
        s = ind_b34 // 9
        r = ind_b34 % 9
        return s * 9 + (0 if r == 8 else r + 1)
    if ind_b34 < 31:
        return 27 + ((ind_b34 - 27 + 1) % 4)
    return 31 + ((ind_b34 - 31 + 1) % 3)


def _ceil100(v: int) -> int:
    return ((v + 99) // 100) * 100


def _convert_tile37_to_136(t37: int) -> int:
    if t37 == 0:
        return 16
    if 1 <= t37 <= 4:
        return (t37 - 1) * 4
    if t37 == 5:
        return 17
    if 6 <= t37 <= 9:
        return (t37 - 1) * 4
    if t37 == 10:
        return 52
    if 11 <= t37 <= 14:
        return (t37 - 2) * 4
    if t37 == 15:
        return 53
    if 16 <= t37 <= 19:
        return (t37 - 2) * 4
    if t37 == 20:
        return 88
    if 21 <= t37 <= 24:
        return (t37 - 3) * 4
    if t37 == 25:
        return 89
    return (t37 - 3) * 4


def _build_136_from_counts(
    closed37: Sequence[int], melds: Sequence[tuple[str, tuple[int, ...], bool]]
) -> tuple[list[int], list[Meld]]:
    """Build valid distinct 136-tile arrays matching Shoupai.getShoupai136_."""
    used = [False] * 136
    tiles136: list[int] = []
    mj_melds: list[Meld] = []

    def take_t37(t37: int) -> int:
        if t37 in (0, 10, 20):
            idx = _convert_tile37_to_136(t37)
            assert not used[idx]
            used[idx] = True
            return idx
        b34 = tile34(t37)
        start = b34 * 4 + (1 if b34 in (4, 13, 22) else 0)
        for k in range(start, b34 * 4 + 4):
            if not used[k]:
                used[k] = True
                return k
        raise ValueError(f"Ran out of physical copies for tile37={t37}")

    for mtype, mtiles37, opened in melds:
        if mtype == "kan":
            b34 = tile34(mtiles37[0])
            m136 = [b34 * 4 + j for j in range(4)]
            for idx in m136:
                assert not used[idx]
                used[idx] = True
            tiles136.extend(m136[:3])
            mj_melds.append(Meld(meld_type=Meld.KAN, tiles=m136, opened=opened))
        elif mtype == "pon":
            m136 = sorted(take_t37(t) for t in mtiles37[:3])
            tiles136.extend(m136)
            mj_melds.append(Meld(meld_type=Meld.PON, tiles=m136, opened=True))
        elif mtype == "chi":
            m136 = sorted(take_t37(t) for t in mtiles37[:3])
            tiles136.extend(m136)
            mj_melds.append(Meld(meld_type=Meld.CHI, tiles=m136, opened=True))

    for t37 in range(37):
        for _ in range(closed37[t37]):
            tiles136.append(take_t37(t37))
    tiles136.sort()
    return tiles136, mj_melds


def _eval_hand_with_tool(
    closed37: Sequence[int],
    melds: Sequence[tuple[str, tuple[int, ...], bool]],
    win_t37: int,
    dora_indicators_37: Sequence[int],
    *,
    is_tsumo: bool,
    is_riichi: bool,
    is_ippatsu: bool,
    is_rinshan: bool,
    is_chankan: bool,
    is_haitei: bool,
    is_houtei: bool,
    is_daburu_riichi: bool,
    is_tenhou: bool,
    is_chiihou: bool,
    player_wind: int,
    round_wind: int,
) -> tuple[bool, int, int, bool]:
    """Return (valid, han, fu, is_yakuman) matching Tool.py exactly."""
    tiles136, mj_melds = _build_136_from_counts(closed37, melds)
    # Pick matching 136 copy for win_tile already inside tiles136 (matching win_t37 red5 status)
    wb34 = tile34(win_t37)
    win_136 = None
    if is_red5_tile37(win_t37):
        cand = wb34 * 4
        if cand in tiles136:
            win_136 = cand
    else:
        start = wb34 * 4 + (1 if wb34 in (4, 13, 22) else 0)
        for k in range(start, wb34 * 4 + 4):
            if k in tiles136:
                win_136 = k
                break
    if win_136 is None:
        win_136 = tiles136[0]

    # Convert dora_indicators_37 cleanly to unused/matching 136 ids
    dora136: list[int] = []
    for ind37 in dora_indicators_37:
        ib34 = tile34(ind37)
        if is_red5_tile37(ind37):
            dora136.append(ib34 * 4)
        else:
            start = ib34 * 4 + (1 if ib34 in (4, 13, 22) else 0)
            for k in range(start, ib34 * 4 + 4):
                if k not in dora136:
                    dora136.append(k)
                    break

    cfg = HandConfig(
        is_tsumo=is_tsumo,
        is_riichi=is_riichi,
        is_ippatsu=is_ippatsu,
        is_rinshan=is_rinshan,
        is_chankan=is_chankan,
        is_haitei=is_haitei,
        is_houtei=is_houtei,
        is_daburu_riichi=is_daburu_riichi,
        is_tenhou=is_tenhou,
        is_chiihou=is_chiihou,
        player_wind=player_wind,
        round_wind=round_wind,
        options=_OPTIONAL_RULES,
    )
    resp = _HAND_CALC.estimate_hand_value(
        tiles=tiles136,
        win_tile=win_136,
        melds=mj_melds,
        dora_indicators=dora136,
        config=cfg,
    )
    if resp.error is not None:
        return False, 0, 0, False
    is_yakuman = any(y.is_yakuman for y in (resp.yaku or []))
    han = resp.han
    if not is_yakuman and 13 <= han < 26:
        han = 13
    return True, han, resp.fu, is_yakuman


def _shanten_raw(counts34: Sequence[int]) -> int:
    # mahjong.shanten expects list of len 34 and returns -1 for winning, 0 for tenpai
    return _SHANTEN.calculate_shanten(list(counts34))


def _waits_34(h_closed34: Sequence[int], m_self34: Sequence[int]) -> list[int]:
    waits: list[int] = []
    tmp = list(h_closed34)
    for w in range(34):
        if tmp[w] + m_self34[w] >= 4:
            continue
        tmp[w] += 1
        if _SHANTEN.calculate_shanten(tmp) == -1:
            waits.append(w)
        tmp[w] -= 1
    return waits


def _effective_3k1_tiles(h_closed34: Sequence[int], m_self34: Sequence[int]) -> tuple[int, int, list[int]]:
    """Return (corrected_shanten, waits_list, effective_tiles_list)."""
    tmp = list(h_closed34)
    raw_s = _SHANTEN.calculate_shanten(tmp)
    waits = _waits_34(tmp, m_self34)
    if raw_s == 0 and len(waits) == 0:
        # Pure fake 5th-tile tenpai -> corrected s = 1, u = 0
        return 1, waits, []
    if raw_s == 0:
        return 0, waits, waits
    eff: list[int] = []
    for t in range(34):
        if tmp[t] + m_self34[t] >= 4:
            continue
        tmp[t] += 1
        if _SHANTEN.calculate_shanten(tmp) < raw_s:
            eff.append(t)
        tmp[t] -= 1
    return raw_s, waits, eff


def _settle_single_win(
    han: int,
    fu: int,
    is_yakuman: bool,
    is_tsumo: bool,
    winner_abs: int,
    loser_abs: int,
    ju: int,
    chang: int,
    ben: int,
    liqibang: int,
    is_tonpuusen: bool,
    pre_scores: list[int],
    pao_daisangen_payer: int,
    pao_daisuushii_payer: int,
) -> tuple[list[int], int, int, bool]:
    if not is_yakuman and 13 <= han < 26:
        base = 8000
    elif han >= 5:
        if han >= 78:
            base = 48000
        elif han >= 65:
            base = 40000
        elif han >= 52:
            base = 32000
        elif han >= 39:
            base = 24000
        elif han >= 26:
            base = 16000
        elif han >= 13:
            base = 8000
        elif han >= 11:
            base = 6000
        elif han >= 8:
            base = 4000
        elif han >= 6:
            base = 3000
        else:
            base = 2000
    else:
        base = min(2000, fu * (1 << (2 + han)))

    post = list(pre_scores)
    is_dealer = winner_abs == ju
    gain = 0

    if is_tsumo:
        if is_dealer:
            pay = base * 2 if han >= 5 else _ceil100(base * 2)
            gain = pay * 3 + 300 * ben + 1000 * liqibang
            post[winner_abs] += gain
            for i in range(4):
                if i != winner_abs:
                    post[i] -= pay + 100 * ben
            if 0 <= pao_daisangen_payer != winner_abs:
                for i in range(4):
                    if i == pao_daisangen_payer:
                        post[i] -= 32000 + 200 * ben
                    elif i != winner_abs:
                        post[i] += 16000 + 100 * ben
            if 0 <= pao_daisuushii_payer != winner_abs:
                for i in range(4):
                    if i == pao_daisuushii_payer:
                        post[i] -= 64000 + 200 * ben
                    elif i != winner_abs:
                        post[i] += 32000 + 100 * ben
        else:
            pay_d = base * 2 if han >= 5 else _ceil100(base * 2)
            pay_c = base if han >= 5 else _ceil100(base)
            gain = pay_d + 2 * pay_c + 300 * ben + 1000 * liqibang
            post[winner_abs] += gain
            for i in range(4):
                if i == winner_abs:
                    continue
                if i == ju:
                    post[i] -= pay_d + 100 * ben
                else:
                    post[i] -= pay_c + 100 * ben
            if 0 <= pao_daisangen_payer != winner_abs:
                for i in range(4):
                    if i == pao_daisangen_payer:
                        post[i] -= (16000 if i == ju else 24000) + 200 * ben
                    elif i != winner_abs:
                        post[i] += (16000 if i == ju else 8000) + 100 * ben
            if 0 <= pao_daisuushii_payer != winner_abs:
                for i in range(4):
                    if i == pao_daisuushii_payer:
                        post[i] -= (32000 if i == ju else 48000) + 200 * ben
                    elif i != winner_abs:
                        post[i] += (32000 if i == ju else 16000) + 100 * ben
    else:
        if han >= 5:
            pay = base * 6 if is_dealer else base * 4
        else:
            pay = _ceil100(base * 6) if is_dealer else _ceil100(base * 4)
        gain = pay + 300 * ben + 1000 * liqibang
        post[winner_abs] += gain
        post[loser_abs] -= pay + 300 * ben
        if 0 <= pao_daisangen_payer != winner_abs:
            sb = 24000 if is_dealer else 16000
            post[pao_daisangen_payer] -= sb + 300 * ben
            post[loser_abs] += sb + 300 * ben
        if 0 <= pao_daisuushii_payer != winner_abs:
            sb = 48000 if is_dealer else 32000
            post[pao_daisuushii_payer] -= sb + 300 * ben
            post[loser_abs] += sb + 300 * ben

    ranks = _scores_to_ranks_1d(post)
    post_rank = ranks[winner_abs]
    dealer_rank = ranks[ju]

    if any(s < 0 for s in post):
        return post, gain, post_rank, True

    suddendeath = any(s >= 30000 for s in post)
    if is_tonpuusen:
        if chang == 1 and ju == 3:
            end_game = ((not is_dealer) or (dealer_rank == 0)) if suddendeath else (not is_dealer)
        elif (chang == 0 and ju == 3) or chang == 1:
            end_game = ((not is_dealer) or (dealer_rank == 0)) if suddendeath else False
        else:
            end_game = False
    else:
        if chang == 2 and ju == 3:
            end_game = ((not is_dealer) or (dealer_rank == 0)) if suddendeath else (not is_dealer)
        elif (chang == 1 and ju == 3) or chang == 2:
            end_game = ((not is_dealer) or (dealer_rank == 0)) if suddendeath else False
        else:
            end_game = False

    return post, gain, post_rank, end_game


def encode_structured_v2_sample_py(
    sparse: Sequence[int],
    numeric: Sequence[int],
    progression: Sequence[int],
    candidates: Sequence[int],
    *,
    enable_early_shape_hint: bool = False,
    reward_weights: tuple[float, float, float, float] = (1.5, 0.5, -0.5, -1.5),
) -> dict[str, np.ndarray]:
    """Pure-Python single-sample reference implementation of §2.1-2.6."""
    assert len(sparse) == 29
    assert len(numeric) == 6
    assert len(candidates) == MAX_CANDIDATES

    is_tonpuusen = int(sparse[1]) == 5
    seat = int(sparse[6]) - 71
    chang = int(sparse[7]) - 75
    ju = int(sparse[8]) - 78
    left_tile_count = int(sparse[9]) - 82
    assert 0 <= seat < 4 and 0 <= chang < 3 and 0 <= ju < 4

    dora_indicators_37: list[int] = []
    hand37 = [0] * 37
    zimo_tile37 = TILE37_NONE
    public37 = [0] * 37
    dora34 = [0] * 34

    for idx in range(10, 29):
        v = int(sparse[idx])
        if v == 510:
            break
        if 152 <= v < 337:
            t37 = (v - 152) % 37
            dora_indicators_37.append(t37)
            public37[t37] += 1
            dora34[_next_dora_b34(tile34(t37))] += 1
        elif 337 <= v < 473:
            t37, _ = idx136_to_tile37(v - 337)
            hand37[t37] += 1
        elif 473 <= v < 510:
            zimo_tile37 = v - 473
            hand37[zimo_tile37] += 1

    # Extract actual progression length
    lp = 0
    for v in progression:
        if int(v) == 2165:
            break
        lp += 1
    assert 1 <= lp <= MAX_PROGRESSION_LEN
    assert int(progression[0]) == 0

    # Extract candidates and check strictly ascending invariant
    valid_cands: list[int] = []
    seen_pad = False
    for v in candidates:
        c = int(v)
        if c == PAD_ACTION_ID:
            seen_pad = True
        else:
            assert not seen_pad and 0 <= c < NUM_TYPES_OF_ACTIONS
            if valid_cands:
                assert c > valid_cands[-1]
            valid_cands.append(c)
    assert 1 <= len(valid_cands) <= 30

    # Parse progression events once
    parsed_evs: list[dict] = []
    any_meld_or_kan_all = False
    for step in range(1, lp):
        tok = int(progression[step])
        if 5 <= tok < 597:
            v = tok - 5
            ev_seat = v // 148
            rem = v % 148
            t37 = rem // 4
            moqie = (rem // 2) & 1
            liqi = rem & 1
            if step == 1 and ev_seat == ju and not any_meld_or_kan_all:
                moqie = 0  # §2.2 rule 3: dealer first discard normalized
            parsed_evs.append({
                "type": 3, "seat": ev_seat, "src_rel": 4,
                "t37": t37, "c0": TILE37_NONE, "c1": TILE37_NONE,
                "moqie": moqie, "liqi": liqi, "b34": tile34(t37),
            })
        elif 597 <= tok < 957:
            v = tok - 597
            ev_seat = v // 90
            code = v % 90
            c0, c1, called, _ = CHI_TABLE[code]
            from_abs = (ev_seat + 3) % 4
            src_rel = (from_abs + 4 - seat) % 4
            parsed_evs.append({
                "type": 5, "seat": ev_seat, "src_rel": src_rel,
                "t37": called, "c0": c0, "c1": c1, "code": code,
            })
            any_meld_or_kan_all = True
        elif 957 <= tok < 1437:
            v = tok - 957
            ev_seat = v // 120
            rem = v % 120
            rel_from = rem // 40
            code = rem % 40
            c0, c1, called = PON_TABLE[code]
            from_abs = (ev_seat + 1 + rel_from) % 4
            src_rel = (from_abs + 4 - seat) % 4
            parsed_evs.append({
                "type": 6, "seat": ev_seat, "src_rel": src_rel,
                "t37": called, "c0": c0, "c1": c1, "code": code,
            })
            any_meld_or_kan_all = True
        elif 1437 <= tok < 1881:
            v = tok - 1437
            ev_seat = v // 111
            rem = v % 111
            rel_from = rem // 37
            called = rem % 37
            from_abs = (ev_seat + 1 + rel_from) % 4
            src_rel = (from_abs + 4 - seat) % 4
            parsed_evs.append({
                "type": 7, "seat": ev_seat, "src_rel": src_rel,
                "t37": called, "c0": TILE37_NONE, "c1": TILE37_NONE,
            })
            any_meld_or_kan_all = True
        elif 1881 <= tok < 2017:
            v = tok - 1881
            ev_seat = v // 34
            b34 = v % 34
            parsed_evs.append({
                "type": 8, "seat": ev_seat, "src_rel": 4,
                "t37": base34_to_tile37(b34), "c0": TILE37_NONE, "c1": TILE37_NONE,
                "b34": b34,
            })
            any_meld_or_kan_all = True
        elif 2017 <= tok < 2165:
            v = tok - 2017
            ev_seat = v // 37
            t37 = v % 37
            parsed_evs.append({
                "type": 9, "seat": ev_seat, "src_rel": 4,
                "t37": t37, "c0": TILE37_NONE, "c1": TILE37_NONE,
            })
            any_meld_or_kan_all = True
        else:
            raise ValueError(f"Unknown progression token: {tok}")

    # Determine phase (§2.2.2)
    has_own_act = any(c < 221 for c in valid_cands)
    if has_own_act:
        phase = 0
        trigger_actor_rel = -1
        trigger_tile37 = TILE37_NONE
        l_landed = lp
    else:
        assert parsed_evs, "Response phase cannot have empty progression"
        last_ev = parsed_evs[-1]
        l_landed = lp - 1
        trigger_actor_rel = (last_ev["seat"] + 4 - seat) % 4
        assert trigger_actor_rel in (1, 2, 3)
        trigger_tile37 = last_ev["t37"]
        if last_ev["type"] == 3:
            phase = 1
        elif last_ev["type"] == 9:
            phase = 2
        elif last_ev["type"] == 8:
            phase = 3
        else:
            raise ValueError("Unexpected last event type in response phase")

    # Build events[lp, 10] and track melds / visible tiles / riichi / pao
    events = np.zeros((lp, EVENT_BYTES), dtype=np.uint8)
    events[0, 0] = 1  # DORA
    events[0, 1] = 4
    events[0, 2] = 4
    events[0, 3] = dora_indicators_37[0]
    events[0, 4] = TILE37_NONE
    events[0, 5] = TILE37_NONE
    events[0, 6] = 16 if is_red5_tile37(dora_indicators_37[0]) else 0

    disc_cnt = [0] * 4
    meld_cnt = [0] * 4
    open_meld_cnt = [0] * 4
    melds_all: list[list[tuple[str, tuple[int, ...], bool]]] = [[] for _ in range(4)]
    meld34_counts = [[0] * 34 for _ in range(4)]
    open_meld34_counts = [[0] * 34 for _ in range(4)]

    had_call_before = False
    had_discard_by_seat = [False] * 4
    riichi_declared = [False] * 4
    double_riichi = [False] * 4
    riichi_turn_idx = [0] * 4
    riichi_decl_tile34 = [-1] * 4
    riichi_ev_idx = [-1] * 4

    # Pao trackers (strictly per round_state.cpp)
    sangen_melds = [0] * 4
    wind_melds = [0] * 4
    pao_daisangen_payer = -1
    pao_daisangen_winner = -1
    pao_daisuushii_payer = -1
    pao_daisuushii_winner = -1
    total_kans = 0

    for idx, pev in enumerate(parsed_evs):
        step = idx + 1
        s = pev["seat"]
        rel = (s + 4 - seat) % 4
        et = pev["type"]
        events[step, 0] = et
        events[step, 1] = rel
        events[step, 2] = pev["src_rel"]
        events[step, 3] = pev["t37"]
        events[step, 4] = pev["c0"]
        events[step, 5] = pev["c1"]
        events[step, 8] = step

        flags = 0
        if et == 3:  # DAHAI
            t37 = pev["t37"]
            public37[t37] += 1
            seq_idx = disc_cnt[s]
            disc_cnt[s] += 1
            events[step, 9] = seq_idx
            is_dbl = False
            if pev["liqi"]:
                flags |= 2
                if (not had_discard_by_seat[s]) and (not had_call_before):
                    is_dbl = True
                    flags |= 4
                riichi_declared[s] = True
                double_riichi[s] = is_dbl
                riichi_turn_idx[s] = seq_idx
                riichi_decl_tile34[s] = tile34(t37)
                riichi_ev_idx[s] = step
            if pev["moqie"]:
                flags |= 1
            if idx > 0 and parsed_evs[idx - 1]["seat"] == s and parsed_evs[idx - 1]["type"] in (7, 8, 9):
                flags |= 8  # is_rinshan
            if is_red5_tile37(t37):
                flags |= 16
            had_discard_by_seat[s] = True
        else:
            had_call_before = True
            if et == 5:  # CHI
                c0, c1, called = pev["c0"], pev["c1"], pev["t37"]
                public37[c0] += 1
                public37[c1] += 1
                events[step, 9] = meld_cnt[s]
                meld_cnt[s] += 1
                open_meld_cnt[s] += 1
                melds_all[s].append(("chi", (c0, c1, called), True))
                for t in (c0, c1, called):
                    b = tile34(t)
                    meld34_counts[s][b] += 1
                    open_meld34_counts[s][b] += 1
                if any(is_red5_tile37(t) for t in (c0, c1, called)):
                    flags |= 16
            elif et == 6:  # PON
                c0, c1, called = pev["c0"], pev["c1"], pev["t37"]
                public37[c0] += 1
                public37[c1] += 1
                events[step, 9] = meld_cnt[s]
                meld_cnt[s] += 1
                open_meld_cnt[s] += 1
                melds_all[s].append(("pon", (c0, c1, called), True))
                for t in (c0, c1, called):
                    b = tile34(t)
                    meld34_counts[s][b] += 1
                    open_meld34_counts[s][b] += 1
                if any(is_red5_tile37(t) for t in (c0, c1, called)):
                    flags |= 16
                tb = tile34(called)
                from_abs = (seat + pev["src_rel"]) % 4
                if tb in (31, 32, 33):
                    sangen_melds[s] += 1
                    if sangen_melds[s] == 3:
                        pao_daisangen_payer = from_abs
                        pao_daisangen_winner = s
                elif 27 <= tb <= 30:
                    wind_melds[s] += 1
                    if wind_melds[s] == 4:
                        pao_daisuushii_payer = from_abs
                        pao_daisuushii_winner = s
            elif et == 7:  # DAIMINKAN
                called = pev["t37"]
                tb = tile34(called)
                b_nonred = base34_to_tile37(tb)
                if tb in (4, 13, 22):
                    red = b_nonred - 5
                    if called == red:
                        public37[b_nonred] += 3
                    else:
                        public37[red] += 1
                        public37[b_nonred] += 2
                    flags |= 16
                else:
                    public37[b_nonred] += 3
                events[step, 9] = meld_cnt[s]
                meld_cnt[s] += 1
                open_meld_cnt[s] += 1
                total_kans += 1
                melds_all[s].append(("kan", (called,), True))
                meld34_counts[s][tb] += 4
                open_meld34_counts[s][tb] += 4
                from_abs = (seat + pev["src_rel"]) % 4
                if tb in (31, 32, 33):
                    sangen_melds[s] += 1
                    if sangen_melds[s] == 3:
                        pao_daisangen_payer = from_abs
                        pao_daisangen_winner = s
                elif 27 <= tb <= 30:
                    wind_melds[s] += 1
                    if wind_melds[s] == 4:
                        pao_daisuushii_payer = from_abs
                        pao_daisuushii_winner = s
            elif et == 8:  # ANKAN
                tb = pev["b34"]
                b_nonred = base34_to_tile37(tb)
                if tb in (4, 13, 22):
                    red = b_nonred - 5
                    public37[red] += 1
                    public37[b_nonred] += 3
                    flags |= 16
                else:
                    public37[b_nonred] += 4
                events[step, 9] = meld_cnt[s]
                meld_cnt[s] += 1
                total_kans += 1
                melds_all[s].append(("kan", (b_nonred,), False))
                meld34_counts[s][tb] += 4
                if tb in (31, 32, 33):
                    sangen_melds[s] += 1
                elif 27 <= tb <= 30:
                    wind_melds[s] += 1
            elif et == 9:  # KAKAN
                t37 = pev["t37"]
                tb = tile34(t37)
                public37[t37] += 1
                if is_red5_tile37(t37):
                    flags |= 16
                # Upgrade pon slot to kan
                k_slot = 0
                for mi, m in enumerate(melds_all[s]):
                    if m[0] == "pon" and tile34(m[1][0]) == tb:
                        melds_all[s][mi] = ("kan", (base34_to_tile37(tb),), True)
                        k_slot = mi
                        break
                events[step, 9] = k_slot
                meld34_counts[s][tb] += 1
                open_meld34_counts[s][tb] += 1
                total_kans += 1

        events[step, 6] = flags

    # Physical invariants check (§2.2.1)
    h_closed34 = [0] * 34
    v_public34 = [0] * 34
    for t37 in range(37):
        lim = 1 if (t37 < 30 and t37 % 10 == 0) else (3 if (t37 < 30 and t37 % 10 == 5) else 4)
        tot = hand37[t37] + public37[t37]
        if tot > lim:
            raise ValueError(f"Physical tile overflow at tile37={t37}: {tot} > {lim}")
        b = tile34(t37)
        h_closed34[b] += hand37[t37]
        v_public34[b] += public37[t37]
    for b in range(34):
        if not (0 <= h_closed34[b] + v_public34[b] <= 4):
            raise ValueError(f"Base34 conservation failed at {b}")
    u34 = [4 - h_closed34[b] - v_public34[b] for b in range(34)]
    red5_unseen = [hand37[s * 10] + public37[s * 10] == 0 for s in range(3)]

    # Landed river analysis (strictly excluding pending trigger event when phase != OWN_ACT)
    landed_disc_tiles34: list[list[int]] = [[] for _ in range(4)]
    landed_disc_moqie: list[list[int]] = [[] for _ in range(4)]
    landed_disc_ev_step: list[list[int]] = [[] for _ in range(4)]
    last_actor_ev_idx = [0] * 4
    last_disc_ev_idx = [-1] * 4
    for step in range(1, l_landed):
        pev = parsed_evs[step - 1]
        s = pev["seat"]
        last_actor_ev_idx[s] = step
        if pev["type"] == 3:
            landed_disc_tiles34[s].append(pev["b34"])
            landed_disc_moqie[s].append(pev["moqie"])
            landed_disc_ev_step[s].append(step)
            last_disc_ev_idx[s] = step

    # Ippatsu check (§2.2.3)
    ippatsu_active = [False] * 4
    for s in range(4):
        re_idx = riichi_ev_idx[s]
        if re_idx < 0:
            continue
        if phase != 0 and re_idx == lp - 1:
            continue  # Ev_last itself is this player's riichi declaration
        broken = False
        for step in range(re_idx + 1, lp):
            pev = parsed_evs[step - 1]
            if pev["type"] in (5, 6, 7, 8, 9):
                broken = True
                break
            if pev["type"] == 3 and pev["seat"] == s:
                broken = True
                break
        ippatsu_active[s] = not broken

    # Post-riichi and same-turn landed table genbutsu per player
    post_riichi_table = [set() for _ in range(4)]
    same_turn_table = [set() for _ in range(4)]
    for s in range(4):
        re_idx = riichi_ev_idx[s]
        if re_idx >= 0 and not (phase != 0 and re_idx == lp - 1):
            for step in range(re_idx + 1, l_landed):
                pev = parsed_evs[step - 1]
                if pev["type"] == 3:
                    post_riichi_table[s].add(pev["b34"])
                elif pev["type"] in (8, 9):
                    post_riichi_table[s].add(tile34(pev["t37"]))
        ld_idx = last_disc_ev_idx[s]
        if ld_idx >= 0:
            for step in range(ld_idx + 1, l_landed):
                pev = parsed_evs[step - 1]
                if pev["type"] == 3:
                    same_turn_table[s].add(pev["b34"])
                elif pev["type"] in (8, 9):
                    same_turn_table[s].add(tile34(pev["t37"]))

    # Furiten state for self
    self_river_set = set(landed_disc_tiles34[seat])
    if phase == 0:
        cur_shanten_raw = 0
        # For cand_mode != POST_DISCARD we may need cur_shanten before discarding (3k+2 or 3k+1 after chi/pon)
        cur_shanten, cur_waits, cur_eff = 0, [], []
        if zimo_tile37 != TILE37_NONE:
            cur_shanten = _SHANTEN.calculate_shanten(h_closed34)
        else:
            # Post-chi/pon 3k+2 without zimo_tile; calculate_shanten works directly
            cur_shanten = _SHANTEN.calculate_shanten(h_closed34)
        cur_waits = []
        riichi_furiten = False
        if riichi_declared[seat]:
            # Check landed table discards/kans after self riichi against waits after current zimo or fixed hand?
            # If riichi and zimo_tile present (auto tsumogiri), closed 3k+1 waits are h_closed - zimo:
            if zimo_tile37 != TILE37_NONE:
                base_h = list(h_closed34)
                base_h[tile34(zimo_tile37)] -= 1
                cur_waits = _waits_34(base_h, meld34_counts[seat])
                for step in range(riichi_ev_idx[seat] + 1, l_landed):
                    pev = parsed_evs[step - 1]
                    if pev["type"] == 3 and pev["b34"] in cur_waits:
                        riichi_furiten = True
                    elif pev["type"] in (8, 9) and tile34(pev["t37"]) in cur_waits:
                        riichi_furiten = True
        temp_furiten = False
    else:
        cur_shanten, cur_waits, cur_eff = _effective_3k1_tiles(h_closed34, meld34_counts[seat])
        riichi_furiten = False
        temp_furiten = False
        if cur_waits:
            wait_set = set(cur_waits)
            if riichi_declared[seat]:
                for step in range(riichi_ev_idx[seat] + 1, l_landed):
                    pev = parsed_evs[step - 1]
                    t_b = pev["b34"] if pev["type"] == 3 else (tile34(pev["t37"]) if pev["type"] in (8, 9) else -1)
                    if t_b in wait_set:
                        riichi_furiten = True
                        break
            else:
                idx_self = last_actor_ev_idx[seat]
                for step in range(idx_self + 1, l_landed):
                    pev = parsed_evs[step - 1]
                    t_b = pev["b34"] if pev["type"] == 3 else (tile34(pev["t37"]) if pev["type"] in (8, 9) else -1)
                    if t_b in wait_set:
                        temp_furiten = True
                        break
    perm_furiten = bool(set(cur_waits) & self_river_set) if phase != 0 else False
    any_cur_furiten = perm_furiten or riichi_furiten or temp_furiten

    # Self dora total (including melds)
    def count_dora_for_hand(c37: Sequence[int], m_all: Sequence[tuple[str, tuple[int, ...], bool]]) -> int:
        d = 0
        for t37 in range(37):
            cnt = c37[t37]
            if cnt > 0:
                d += cnt * dora34[tile34(t37)]
                if is_red5_tile37(t37):
                    d += cnt
        for mtype, mtiles, _ in m_all:
            if mtype == "kan":
                b = tile34(mtiles[0])
                d += 4 * dora34[b]
                if b in (4, 13, 22):
                    d += 1
            else:
                for t in mtiles:
                    d += dora34[tile34(t)]
                    if is_red5_tile37(t):
                        d += 1
        return d

    cur_dora = count_dora_for_hand(hand37, melds_all[seat])

    # Kuikae forbidden set on OWN_ACT
    kuikae_forbid_b34: set[int] = set()
    if phase == 0 and zimo_tile37 == TILE37_NONE and parsed_evs:
        last_ev = parsed_evs[-1]
        if last_ev["seat"] == seat:
            if last_ev["type"] == 5:
                called, k_suji = CHI_TABLE[last_ev["code"]][2], CHI_TABLE[last_ev["code"]][3]
                kuikae_forbid_b34.add(tile34(called))
                if k_suji != 255:
                    kuikae_forbid_b34.add(tile34(k_suji))
            elif last_ev["type"] == 6:
                kuikae_forbid_b34.add(tile34(last_ev["t37"]))

    # Legal discards in OWN_ACT (needed for tile_feats[60..67] & cand_dyn POST_DISCARD)
    legal_disc_b34s = {c // 4 for c in valid_cands if c < 148}
    legal_disc_b34s = {tile34(t37) for t37 in legal_disc_b34s}
    discard_eval: dict[int, tuple[int, int, bool, bool]] = {}  # b34 -> (s_post, u_post, post_tenpai, post_furiten)
    best_disc_s = 999
    best_disc_u = 0

    if phase == 0:
        for b in sorted(legal_disc_b34s):
            after = list(h_closed34)
            after[b] -= 1
            s_post, waits_post, eff_post = _effective_3k1_tiles(after, meld34_counts[seat])
            u_post = sum(u34[t] for t in eff_post)
            post_tenpai = len(waits_post) > 0
            post_fur = riichi_furiten or bool(set(waits_post) & (self_river_set | {b})) if post_tenpai else False
            discard_eval[b] = (s_post, u_post, post_tenpai, post_fur)
            if s_post < best_disc_s:
                best_disc_s = s_post
        for b, (sp, up, _, _) in discard_eval.items():
            if up > best_disc_u:
                best_disc_u = up

    # ---------------- Build tile_feats[34, 70] ----------------
    tile_feats = np.zeros((34, TILE_FEATS_DIM), dtype=np.float32)
    for b in range(34):
        hc = h_closed34[b]
        mc = meld34_counts[seat][b]
        for k in range(4):
            tile_feats[b, k] = 1.0 if hc >= k + 1 else 0.0
            tile_feats[b, 5 + k] = 1.0 if mc >= k + 1 else 0.0
            tile_feats[b, 14 + k] = 1.0 if u34[b] >= k + 1 else 0.0
        red_t = {4: 0, 13: 10, 22: 20}.get(b, -1)
        if red_t >= 0:
            tile_feats[b, 4] = 1.0 if hand37[red_t] > 0 else 0.0
            tile_feats[b, 13] = 1.0 if red5_unseen[red_t // 10] else 0.0
        if phase == 0 and zimo_tile37 != TILE37_NONE and not (seat == ju and left_tile_count == 69):
            if tile34(zimo_tile37) == b:
                tile_feats[b, 9] = 1.0
                if is_red5_tile37(zimo_tile37):
                    tile_feats[b, 10] = 1.0
        if b in kuikae_forbid_b34:
            tile_feats[b, 11] = 1.0
        tile_feats[b, 12] = min(1.25, dora34[b] / 4.0)
        tile_feats[b, 18] = 1.0 if b in self_river_set else 0.0

    # Group B1: 3 opponents
    for p_rel in (1, 2, 3):
        off = 19 + (p_rel - 1) * 11
        opp_abs = (seat + p_rel) % 4
        river = landed_disc_tiles34[opp_abs]
        river_m = landed_disc_moqie[opp_abs]
        opp_river_set = set(river)

        last_idx_of: dict[int, tuple[int, int]] = {}
        for di, (tb, mq) in enumerate(zip(river, river_m)):
            last_idx_of[tb] = (di, mq)

        for b in opp_river_set:
            tile_feats[b, off + 0] = 1.0
            di, mq = last_idx_of[b]
            delta_t = (len(river) - 1) - di
            tile_feats[b, off + 6] = math.exp(-delta_t / 6.0)
            tile_feats[b, off + 7] = 1.0 if mq == 0 else 0.0

        if riichi_declared[opp_abs] and riichi_decl_tile34[opp_abs] >= 0:
            tile_feats[riichi_decl_tile34[opp_abs], off + 1] = 1.0
        if river:
            tile_feats[river[-1], off + 2] = 1.0
        for b in post_riichi_table[opp_abs]:
            tile_feats[b, off + 9] = 1.0
        for b in same_turn_table[opp_abs]:
            tile_feats[b, off + 10] = 1.0

        genbutsu_union = opp_river_set | post_riichi_table[opp_abs]
        for s in range(3):
            for r in range(9):
                b = s * 9 + r
                has_lo = r >= 3
                has_hi = r <= 5
                lo_safe = ((s * 9 + r - 3) in genbutsu_union) if has_lo else False
                hi_safe = ((s * 9 + r + 3) in genbutsu_union) if has_hi else False
                tile_feats[b, off + 3] = 1.0 if (lo_safe or hi_safe) else 0.0
                full_safe = (not has_lo or lo_safe) and (not has_hi or hi_safe)
                tile_feats[b, off + 4] = 1.0 if full_safe else 0.0

        if enable_early_shape_hint:
            early = river[:6]
            for db in early:
                if db < 27:
                    ds = db // 9
                    dr = db % 9
                    if dr in (1, 2, 3):  # 2, 3, 4 -> 1..(dr)
                        for r2 in range(0, dr):
                            tile_feats[ds * 9 + r2, off + 5] = 1.0
                    elif dr in (5, 6, 7):  # 6, 7, 8 -> (dr+2)..9
                        for r2 in range(dr + 1, 9):
                            tile_feats[ds * 9 + r2, off + 5] = 1.0

        for b in range(34):
            tile_feats[b, off + 8] = meld34_counts[opp_abs][b] / 4.0

    # Group B2: kabe, dora adj, yakuhai
    k_tot = [h_closed34[b] + v_public34[b] for b in range(34)]
    for s in range(3):
        for r in range(9):
            b = s * 9 + r
            has_l = r >= 3
            has_r = r <= 5
            l_no = has_l and (k_tot[s * 9 + r - 2] == 4 or k_tot[s * 9 + r - 1] == 4)
            l_one = has_l and (k_tot[s * 9 + r - 2] >= 3 or k_tot[s * 9 + r - 1] >= 3)
            r_no = has_r and (k_tot[s * 9 + r + 1] == 4 or k_tot[s * 9 + r + 2] == 4)
            r_one = has_r and (k_tot[s * 9 + r + 1] >= 3 or k_tot[s * 9 + r + 2] >= 3)
            nc = (not has_l or l_no) and (not has_r or r_no)
            oc = (not nc) and (not has_l or l_one) and (not has_r or r_one)
            tile_feats[b, 52] = 1.0 if nc else 0.0
            tile_feats[b, 53] = 1.0 if oc else 0.0

    dora_target_b34s = [b for b, cnt in enumerate(dora34) if cnt > 0]
    for s in range(3):
        for r in range(9):
            b = s * 9 + r
            for db in dora_target_b34s:
                if db < 27 and db // 9 == s:
                    dist = abs(r - (db % 9))
                    if dist == 1:
                        tile_feats[b, 54] = 1.0
                    elif dist == 2:
                        tile_feats[b, 55] = 1.0

    for rel in range(4):
        abs_p = (seat + rel) % 4
        sw = 27 + (abs_p + 4 - ju) % 4
        rw = 27 + chang
        for b in (31, 32, 33):
            tile_feats[b, 56 + rel] = 0.5
        for w in range(27, 31):
            val = 0.0
            if w == sw:
                val += 0.5
            if w == rw:
                val += 0.5
            tile_feats[w, 56 + rel] = val

    # Group C
    if phase == 0:
        for b, (sp, up, tenp, fur) in discard_eval.items():
            tile_feats[b, 60] = 1.0
            tile_feats[b, 61] = 1.0 if sp == best_disc_s else 0.0
            tile_feats[b, 62] = 1.0 if sp == best_disc_s + 1 else 0.0
            tile_feats[b, 63] = min(1.0, max(0, sp) / 8.0)
            tile_feats[b, 64] = min(1.0, math.log1p(up) / _LN61)
            tile_feats[b, 65] = up / max(1, best_disc_u)
            tile_feats[b, 66] = 1.0 if tenp else 0.0
            tile_feats[b, 67] = 1.0 if fur else 0.0
    else:
        for w in cur_waits:
            tile_feats[w, 68] = 1.0
        for e in cur_eff:
            tile_feats[e, 69] = 1.0

    # ---------------- global_feats[48] ----------------
    global_feats = np.zeros((GLOBAL_FEATS_DIM,), dtype=np.float32)
    global_feats[chang] = 1.0
    global_feats[3 + ju] = 1.0
    ben = int(numeric[0])
    liqibang = int(numeric[1])
    scores = [int(numeric[2 + i]) for i in range(4)]
    global_feats[7] = min(2.0, ben / 8.0)
    global_feats[8] = min(2.0, liqibang / 8.0)
    global_feats[9] = max(0.0, min(1.0, left_tile_count / 70.0))
    is_all_last = (chang == 0 and ju == 3) if is_tonpuusen else (chang == 1 and ju == 3)
    is_ext = (chang >= 1) if is_tonpuusen else (chang >= 2)
    global_feats[10] = 1.0 if is_all_last else 0.0
    global_feats[11] = 1.0 if is_ext else 0.0
    global_feats[12] = min(1.0, total_kans / 4.0)
    global_feats[13] = min(1.0, len(dora_indicators_37) / 5.0)
    global_feats[14] = 1.0 if (not had_call_before and disc_cnt[seat] == 0) else 0.0
    global_feats[15 + phase] = 1.0
    global_feats[19] = 1.0 if (riichi_furiten or temp_furiten) else 0.0
    if phase != 0 and trigger_actor_rel in (1, 2, 3):
        global_feats[20 + trigger_actor_rel - 1] = 1.0
    global_feats[23] = min(1.0, max(0, cur_shanten) / 8.0)
    global_feats[24] = meld_cnt[seat] / 4.0
    global_feats[25] = 1.0 if open_meld_cnt[seat] == 0 else 0.0
    # 26..36 rule constants
    global_feats[26:37] = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32
    )
    # 37..43 game flags
    global_feats[37:44] = np.array(
        [1.0 if is_tonpuusen else 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
    )
    global_feats[44:48] = np.array(reward_weights, dtype=np.float32)

    # ---------------- player_feats[4, 32] ----------------
    player_feats = np.zeros((4, PLAYER_FEATS_DIM), dtype=np.float32)
    ranks = _scores_to_ranks_1d(scores)
    sorted_idx = sorted(range(4), key=lambda i: ranks[i])
    for rel in range(4):
        abs_p = (seat + rel) % 4
        player_feats[rel, rel] = 1.0
        sw = (abs_p + 4 - ju) % 4
        player_feats[rel, 4 + sw] = 1.0
        player_feats[rel, 8] = 1.0 if abs_p == ju else 0.0
        player_feats[rel, 9] = max(-3.0, min(3.0, (scores[abs_p] - 25000) / 25000.0))
        player_feats[rel, 10] = max(-3.0, min(3.0, (scores[abs_p] - scores[seat]) / 25000.0))
        rk = ranks[abs_p]
        if rk > 0:
            higher_abs = sorted_idx[rk - 1]
            player_feats[rel, 11] = max(0.0, min(3.0, (scores[higher_abs] - scores[abs_p]) / 25000.0))
        if rk < 3:
            lower_abs = sorted_idx[rk + 1]
            player_feats[rel, 12] = max(0.0, min(3.0, (scores[abs_p] - scores[lower_abs]) / 25000.0))
        player_feats[rel, 13] = rk / 3.0
        player_feats[rel, 14] = 1.0 if riichi_declared[abs_p] else 0.0
        player_feats[rel, 15] = 1.0 if double_riichi[abs_p] else 0.0
        player_feats[rel, 16] = 1.0 if ippatsu_active[abs_p] else 0.0
        if riichi_declared[abs_p]:
            player_feats[rel, 17] = min(1.0, riichi_turn_idx[abs_p] / 24.0)
        player_feats[rel, 18] = open_meld_cnt[abs_p] / 4.0
        recent_m = landed_disc_moqie[abs_p][-6:]
        if recent_m:
            player_feats[rel, 19] = sum(1 for m in recent_m if m == 0) / len(recent_m)

    # ---------------- cand_mode & cand_dyn ----------------
    cand_mode = np.zeros((MAX_CANDIDATES,), dtype=np.uint8)
    cand_dyn = np.zeros((MAX_CANDIDATES, CAND_DYN_DIM), dtype=np.float32)

    self_player_wind = 27 + (seat + 4 - ju) % 4
    self_round_wind = 27 + chang
    cur_u = sum(u34[t] for t in cur_eff) if phase != 0 else 0
    is_cur_rinshan = (
        phase == 0 and zimo_tile37 != TILE37_NONE and bool(parsed_evs)
        and parsed_evs[-1]["seat"] == seat and parsed_evs[-1]["type"] in (7, 8, 9)
    )
    distinct_yaochuu = sum(
        1 for b in (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33) if h_closed34[b] > 0
    )

    def _add_wait_tile37(c37: Sequence[int], m_all: Sequence[tuple[str, tuple[int, ...], bool]], w34: int) -> tuple[list[int], int]:
        out37 = list(c37)
        # Count black copies already used across closed hand and melds
        black_t37 = base34_to_tile37(w34)
        used_black = out37[black_t37]
        for mtype, mtiles, _ in m_all:
            if mtype == "kan":
                if tile34(mtiles[0]) == w34:
                    used_black += 3 if w34 in (4, 13, 22) else 4
            else:
                for mt in mtiles:
                    if mt == black_t37:
                        used_black += 1
        max_black = 3 if w34 in (4, 13, 22) else 4
        if used_black < max_black:
            out37[black_t37] += 1
            return out37, black_t37
        red_t37 = (w34 // 9) * 10
        out37[red_t37] += 1
        return out37, red_t37

    def has_dama_yaku_after_discard(c37: Sequence[int], m_all: Sequence[tuple[str, tuple[int, ...], bool]], waits: Sequence[int]) -> bool:
        for w in waits:
            test37, wt37 = _add_wait_tile37(c37, m_all, w)
            valid, _, _, _ = _eval_hand_with_tool(
                test37, m_all, wt37, [],
                is_tsumo=False, is_riichi=False, is_ippatsu=False,
                is_rinshan=False, is_chankan=False, is_haitei=False,
                is_houtei=False, is_daburu_riichi=False,
                is_tenhou=False, is_chiihou=False,
                player_wind=self_player_wind, round_wind=self_round_wind,
            )
            if valid:
                return True
        return False

    for ci, aid in enumerate(valid_cands):
        cm = int(KANACHAN_ACTION_MANIFEST[aid, 9])
        cand_mode[ci] = cm
        if cm == 1:  # POST_DISCARD_EXACT
            d37 = aid // 4
            b = tile34(d37)
            sp, up, post_tenpai, post_fur = discard_eval[b]
            after37 = list(hand37)
            after37[d37] -= 1
            after_dora = cur_dora - dora34[b] - (1 if is_red5_tile37(d37) else 0)
            cand_dyn[ci, 0] = min(1.0, max(0, sp) / 8.0)
            cand_dyn[ci, 1] = max(-1.0, min(2.0, (sp - best_disc_s) / 4.0))
            cand_dyn[ci, 2] = min(1.0, math.log1p(up) / _LN61)
            cand_dyn[ci, 3] = min(1.0, up / max(1, best_disc_u))
            cand_dyn[ci, 4] = 1.0 if post_tenpai else 0.0
            cand_dyn[ci, 5] = 1.0 if post_fur else 0.0
            cand_dyn[ci, 6] = min(2.0, after_dora / 8.0)
            if post_tenpai:
                after34 = [0] * 34
                for t37 in range(37):
                    after34[tile34(t37)] += after37[t37]
                waits_post = _waits_34(after34, meld34_counts[seat])
                cand_dyn[ci, 7] = 1.0 if has_dama_yaku_after_discard(after37, melds_all[seat], waits_post) else 0.0
        elif cm == 2:  # PRE_DISCARD_MELD_ENVELOPE
            after37 = list(hand37)
            m_all = list(melds_all[seat])
            m34_s = list(meld34_counts[seat])
            forbid = set()
            if aid < 312:
                code = aid - 222
                c0, c1, called, k_suji = CHI_TABLE[code]
                after37[c0] -= 1
                after37[c1] -= 1
                m_all.append(("chi", (c0, c1, called), True))
                for t in (c0, c1, called):
                    m34_s[tile34(t)] += 1
                forbid.add(tile34(called))
                if k_suji != 255:
                    forbid.add(tile34(k_suji))
                post_meld_dora = cur_dora + dora34[tile34(called)] + (1 if is_red5_tile37(called) else 0)
            else:
                code = (aid - 312) % 40
                c0, c1, called = PON_TABLE[code]
                after37[c0] -= 1
                after37[c1] -= 1
                m_all.append(("pon", (c0, c1, called), True))
                for t in (c0, c1, called):
                    m34_s[tile34(t)] += 1
                forbid.add(tile34(called))
                post_meld_dora = cur_dora + dora34[tile34(called)] + (1 if is_red5_tile37(called) else 0)

            after34 = [0] * 34
            for t37 in range(37):
                after34[tile34(t37)] += after37[t37]

            s_opt = 999
            cand_bs: list[tuple[int, int, int, list[int]]] = []
            for db in range(34):
                if after34[db] == 0 or db in forbid:
                    continue
                rem34 = list(after34)
                rem34[db] -= 1
                sb, wb, eb = _effective_3k1_tiles(rem34, m34_s)
                ub = sum(u34[t] for t in eb)
                cand_bs.append((db, sb, ub, wb))
                if sb < s_opt:
                    s_opt = sb
            opt_list = [x for x in cand_bs if x[1] == s_opt]
            max_u = max((x[2] for x in opt_list), default=0)
            can_tenpai = s_opt == 0
            all_fur = False
            has_yaku_exit = False
            if can_tenpai and opt_list:
                all_fur = all(
                    riichi_furiten or bool(set(wb) & (self_river_set | {db}))
                    for db, _, _, wb in opt_list
                )
                for db, _, _, wb in opt_list:
                    tmp37 = list(after37)
                    # Remove one tile matching db (prefer non-red if both, though yaku without aka_dora identical!)
                    d_nonred = base34_to_tile37(db)
                    if tmp37[d_nonred] > 0:
                        tmp37[d_nonred] -= 1
                    else:
                        tmp37[db // 9 * 10] -= 1
                    if has_dama_yaku_after_discard(tmp37, m_all, wb):
                        has_yaku_exit = True
                        break
            cand_dyn[ci, 0] = min(1.0, max(0, s_opt) / 8.0)
            cand_dyn[ci, 1] = max(-2.0, min(2.0, (s_opt - cur_shanten) / 4.0))
            cand_dyn[ci, 2] = min(1.0, math.log1p(max_u) / _LN61)
            cand_dyn[ci, 3] = len(opt_list) / 14.0
            cand_dyn[ci, 4] = 1.0 if can_tenpai else 0.0
            cand_dyn[ci, 5] = 1.0 if (can_tenpai and all_fur) else 0.0
            cand_dyn[ci, 6] = min(2.0, post_meld_dora / 8.0)
            cand_dyn[ci, 7] = 1.0 if (can_tenpai and has_yaku_exit) else 0.0
        elif cm == 3:  # PENDING_RINSHAN_KAN
            post_d = cur_dora
            if aid >= 432:  # DAIMINKAN
                called = (aid - 432) % 37
                post_d += dora34[tile34(called)] + (1 if is_red5_tile37(called) else 0)
            # note ANKAN & KAKAN tiles are already all in self hand+melds so cur_dora already includes all 4!
            if phase == 0:
                s_for_kan = _SHANTEN.calculate_shanten(h_closed34)
                tenp = s_for_kan == 0
            else:
                s_for_kan = cur_shanten
                tenp = len(cur_waits) > 0
            cand_dyn[ci, 0] = min(1.0, max(0, s_for_kan) / 8.0)
            cand_dyn[ci, 4] = 1.0 if tenp else 0.0
            cand_dyn[ci, 5] = 0.0 if aid >= 432 else 1.0
            cand_dyn[ci, 6] = min(2.0, post_d / 8.0)
            cand_dyn[ci, 7] = min(1.0, (total_kans + 1) / 4.0)
        elif cm == 4:  # PASS_CURRENT_STATE
            cand_dyn[ci, 0] = min(1.0, max(0, cur_shanten) / 8.0)
            cand_dyn[ci, 2] = min(1.0, math.log1p(cur_u) / _LN61)
            cand_dyn[ci, 3] = 1.0
            cand_dyn[ci, 4] = 1.0 if len(cur_waits) > 0 else 0.0
            cand_dyn[ci, 5] = 1.0 if any_cur_furiten else 0.0
            cand_dyn[ci, 6] = min(2.0, cur_dora / 8.0)
        elif cm == 5:  # WIN_EVAL
            is_tsumo_win = aid == 219
            eval_c37 = list(hand37)
            if is_tsumo_win:
                win_t37 = zimo_tile37
                loser_abs = -1
            else:
                win_t37 = trigger_tile37
                eval_c37[win_t37] += 1
                loser_abs = (seat + 1 + (aid - 543)) % 4

            is_tenhou = is_tsumo_win and (not any_meld_or_kan_all) and (seat == ju) and lp == 1
            is_chiihou = (
                is_tsumo_win and (not any_meld_or_kan_all) and (seat != ju)
                and (disc_cnt[seat] == 0) and (lp - 1 == (seat + 4 - ju) % 4)
            )
            valid, han, fu, is_yk = _eval_hand_with_tool(
                eval_c37, melds_all[seat], win_t37, dora_indicators_37,
                is_tsumo=is_tsumo_win,
                is_riichi=riichi_declared[seat],
                is_ippatsu=ippatsu_active[seat],
                is_rinshan=is_cur_rinshan,
                is_chankan=(not is_tsumo_win and phase in (2, 3)),
                is_haitei=(is_tsumo_win and left_tile_count == 0 and not is_cur_rinshan),
                is_houtei=(not is_tsumo_win and phase == 1 and left_tile_count == 0),
                is_daburu_riichi=double_riichi[seat],
                is_tenhou=is_tenhou,
                is_chiihou=is_chiihou,
                player_wind=self_player_wind,
                round_wind=self_round_wind,
            )
            assert valid, f"WIN_EVAL hand unexpectedly invalid for aid={aid}"
            ds_p = pao_daisangen_payer if pao_daisangen_winner == seat else -1
            dw_p = pao_daisuushii_payer if pao_daisuushii_winner == seat else -1
            _, gain, post_rank, end_g = _settle_single_win(
                han, fu, is_yk, is_tsumo_win, seat, loser_abs,
                ju, chang, ben, liqibang, is_tonpuusen, scores,
                ds_p, dw_p,
            )
            cand_dyn[ci, 0] = min(2.0, han / 13.0)
            cand_dyn[ci, 1] = min(1.5, fu / 110.0)
            cand_dyn[ci, 2] = min(1.5, math.log1p(max(0, gain)) / _LN64001)
            cand_dyn[ci, 3] = 1.0 if is_yk else 0.0
            cand_dyn[ci, 4] = post_rank / 3.0
            cand_dyn[ci, 5] = 1.0 if end_g else 0.0
            cand_dyn[ci, 6] = 1.0 if riichi_declared[seat] else 0.0
        elif cm == 6:  # KYUSHU_DRAW_EVAL
            s_cur_k = _SHANTEN.calculate_shanten(h_closed34)
            cand_dyn[ci, 0] = min(1.0, max(0, s_cur_k) / 8.0)
            cand_dyn[ci, 1] = distinct_yaochuu / 13.0
            cand_dyn[ci, 2] = min(2.0, cur_dora / 8.0)
            cand_dyn[ci, 3] = 1.0 if seat == ju else 0.0
            cand_dyn[ci, 4] = ranks[seat] / 3.0

    cand_ids = np.array(candidates, dtype=np.int32)
    cand_mask = cand_ids != PAD_ACTION_ID

    return {
        "v2_tile_feats": tile_feats,
        "v2_events": events,
        "v2_event_len": np.int32(lp),
        "v2_global_feats": global_feats,
        "v2_player_feats": player_feats,
        "v2_cand_ids": cand_ids,
        "v2_cand_mask": cand_mask,
        "v2_cand_mode": cand_mode,
        "v2_cand_dyn": cand_dyn,
    }


def derive_structured_v2_batch_py(
    sparse,
    numeric,
    progression,
    candidates,
    *,
    enable_early_shape_hint: bool = False,
    reward_weights: tuple[float, float, float, float] = (1.5, 0.5, -0.5, -1.5),
):
    """Batched pure-Python reference implementation returning padded arrays."""
    sp = np.asarray(sparse, dtype=np.int32)
    nu = np.asarray(numeric, dtype=np.int32)
    pr = np.asarray(progression, dtype=np.int32)
    ca = np.asarray(candidates, dtype=np.int32)
    bsz = sp.shape[0]

    samples = [
        encode_structured_v2_sample_py(
            sp[i], nu[i], pr[i], ca[i],
            enable_early_shape_hint=enable_early_shape_hint,
            reward_weights=reward_weights,
        )
        for i in range(bsz)
    ]
    lmax = max(int(s["v2_event_len"]) for s in samples)
    events = np.zeros((bsz, lmax, EVENT_BYTES), dtype=np.uint8)
    for i, s in enumerate(samples):
        le = int(s["v2_event_len"])
        events[i, :le] = s["v2_events"]

    return {
        "v2_tile_feats": np.stack([s["v2_tile_feats"] for s in samples], axis=0),
        "v2_events": events,
        "v2_event_len": np.array([int(s["v2_event_len"]) for s in samples], dtype=np.int32),
        "v2_global_feats": np.stack([s["v2_global_feats"] for s in samples], axis=0),
        "v2_player_feats": np.stack([s["v2_player_feats"] for s in samples], axis=0),
        "v2_cand_ids": np.stack([s["v2_cand_ids"] for s in samples], axis=0),
        "v2_cand_mask": np.stack([s["v2_cand_mask"] for s in samples], axis=0),
        "v2_cand_mode": np.stack([s["v2_cand_mode"] for s in samples], axis=0),
        "v2_cand_dyn": np.stack([s["v2_cand_dyn"] for s in samples], axis=0),
    }


__all__ = [
    "derive_structured_v2_batch_py",
    "encode_structured_v2_sample_py",
]
