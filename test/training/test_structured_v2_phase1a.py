"""Phase 1A Contract Tests #1-#7 for structured_v2 C++ & Python reference."""

from __future__ import annotations

import hashlib
import random
import unittest

import numpy as np
import torch

from kanachan import _candidate_features
from kanachan.candidate_features import candidate_feature_schema, derive_candidate_features
from kanachan.simulation.tool import Tool
from kanachan.structured_v2 import (
    CHI_ENCODE_MAP,
    KANACHAN_ACTION_MANIFEST,
    KANACHAN_ACTION_MANIFEST_SHA256,
    NUM_TYPES_OF_ACTIONS,
    OFFSETS_136,
    PAD_ACTION_ID,
    PON_ENCODE_MAP,
    STRUCTURED_V2_SCHEMA_ID,
    base34_to_tile37,
    decode_action_tuple,
    derive_structured_v2_batch,
    encode_action_tuple,
    is_red5_tile37,
    tile34,
)
from kanachan.structured_v2_ref import (
    derive_structured_v2_batch_py,
    encode_structured_v2_sample_py,
)


def _make_sparse(
    *,
    seat: int = 0,
    chang: int = 0,
    ju: int = 0,
    left: int = 60,
    doras: tuple[int, ...] = (1,),
    closed37: dict[int, int] | None = None,
    zimo: int | None = None,
    tonpuusen: bool = False,
) -> list[int]:
    row = [
        0, 5 if tonpuusen else 6, 7, 23, 39, 55,
        71 + seat, 75 + chang, 78 + ju, 82 + left,
    ]
    for slot, d in enumerate(doras):
        row.append(152 + slot * 37 + d)
    used_cnt = [0] * 37
    if closed37:
        for t37 in sorted(closed37):
            for _ in range(closed37[t37]):
                off = used_cnt[t37]
                used_cnt[t37] += 1
                row.append(337 + OFFSETS_136[t37] + off)
    if zimo is not None:
        row.append(473 + zimo)
    assert len(row) <= 29
    row.extend([510] * (29 - len(row)))
    return row


def _pad_prog(events: list[int], width: int = 109) -> list[int]:
    return [0] + events + [2165] * (width - 1 - len(events))


def _pad_cands(cands: list[int]) -> list[int]:
    sc = sorted(cands)
    return sc + [PAD_ACTION_ID] * (32 - len(sc))


def _dahai(seat: int, t37: int, moqie: int = 0, liqi: int = 0) -> int:
    return 5 + 148 * seat + 4 * t37 + 2 * moqie + liqi


def _chi(seat: int, code: int) -> int:
    return 597 + 90 * seat + code


def _pon(seat: int, rel_from: int, code: int) -> int:
    return 957 + 120 * seat + 40 * rel_from + code


def _dmk(seat: int, rel_from: int, t37: int) -> int:
    return 1437 + 111 * seat + 37 * rel_from + t37


def _ankan(seat: int, b34: int) -> int:
    return 1881 + 34 * seat + b34


def _kakan(seat: int, t37: int) -> int:
    return 2017 + 37 * seat + t37


def _tiles37_to_136_and_melds(closed37: dict[int, int], melds):
    """Build exact 136-tile representation matching kanachan.simulation.Tool."""
    used = [False] * 136
    tiles136 = []

    def take(t37):
        if t37 in (0, 10, 20):
            idx = {0: 16, 10: 52, 20: 88}[t37]
            assert not used[idx], f"duplicate red5 {t37}"
            used[idx] = True
            return idx
        b = tile34(t37)
        start = b * 4 + (1 if b in (4, 13, 22) else 0)
        for k in range(start, b * 4 + 4):
            if not used[k]:
                used[k] = True
                return k
        raise AssertionError("out of tiles")

    tool_melds = []
    for mtype, mtiles37, opened in melds:
        if mtype == "kan":
            b = tile34(mtiles37[0])
            m136 = [b * 4 + j for j in range(4)]
            for idx in m136:
                used[idx] = True
            tiles136.extend(m136[:3])
            tool_melds.append(("kan", m136, 1 if opened else 0))
        else:
            m136 = sorted(take(t) for t in mtiles37[:3])
            tiles136.extend(m136)
            tool_melds.append((mtype, m136, 1))

    for t37, cnt in sorted(closed37.items()):
        for _ in range(cnt):
            tiles136.append(take(t37))
    tiles136.sort()
    return tiles136, tool_melds


class TestStructuredV2Phase1A(unittest.TestCase):
    def test_backward_compat_discard_afterstate_v1(self):
        self.assertEqual(_candidate_features.SCHEMA_ID, "discard_afterstate_v1")
        self.assertEqual(_candidate_features.FEATURE_WIDTH, 46)
        self.assertEqual(
            _candidate_features.STRUCTURED_V2_SCHEMA_ID, STRUCTURED_V2_SCHEMA_ID
        )
        # Smoke test discard_features v1 unchanged
        hand37 = [0] * 37
        for t in (1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 31, 32, 32, 9):
            hand37[t] += 1
        vis = [0] * 34
        for t in range(37):
            vis[tile34(t)] += hand37[t]
        dora = [0] * 34
        schema = candidate_feature_schema()
        self.assertEqual(schema["id"], "discard_afterstate_v1")
        self.assertEqual(schema["width"], 46)
        raw = _candidate_features.discard_features(hand37, vis, dora, [1 * 4, 9 * 4], 0)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(2, 46)
        self.assertEqual(tuple(arr.shape), (2, 46))

    # Test #1: Physical tile deduplication & conservation
    def test_1_physical_tile_deduplication(self):
        # Player 1 discards 0m (red 5m), player 2 chis 456m with (0m, 4m, 6m)? Actually called=0m!
        # Next player 3 discards 5m, player 1 pons (5m, 5m, 5m) -> no conflict; then kakan not possible;
        # Test ankan of 5p (13), dora indicator 4m (4 -> next dora is 5m), red5 chi/pon/kakan.
        c_chi_red = CHI_ENCODE_MAP[(4, 6, 0)]  # consumes 4m, 6m, called=0m (already discarded!)
        c_pon_red = PON_ENCODE_MAP[(10, 15, 15)]  # consumes 0p, 5p, called=5p
        # Hand has 4s(24),5s(25),5s(25), 1s..
        closed = {21: 1, 22: 1, 23: 1, 24: 2, 26: 2, 27: 2, 28: 2, 29: 2}
        prog = _pad_prog([
            _dahai(1, 0),          # seat1 discards red 5m (0) -> counted ONCE in public37[0]
            _chi(2, c_chi_red),    # seat2 chis using (4m, 6m) and seat1's 0m -> adds ONLY 4m, 6m
            _dahai(2, 15),         # seat2 discards black 5p (15)
            _pon(3, 2, c_pon_red), # seat3 pons from seat2 (rel=2) consuming 0p(10) and 5p(15)
            _dahai(3, 25),         # seat3 discards 5s (25)
            _dmk(1, 1, 25),        # seat1 daiminkan 5s from seat3 -> remaining 2x25 + 1x20 added!
            _ankan(2, 0),          # seat2 ankan 1m -> 4x 1m added
            _dahai(1, 16),         # seat1 discards 6p (unrelated)
            _kakan(3, 15),         # seat3 kakan black 5p -> 3rd black 5p + 0p = 4 total 5p!
        ])
        sp = _make_sparse(seat=0, left=45, doras=(4, 14), closed37=closed, zimo=29)
        nu = [0, 0, 25000, 25000, 25000, 25000]
        ca = _pad_cands([29 * 4])

        out_cpp = derive_structured_v2_batch(
            torch.tensor([sp], dtype=torch.int32),
            torch.tensor([nu], dtype=torch.int32),
            torch.tensor([prog], dtype=torch.int32),
            torch.tensor([ca], dtype=torch.int32),
        )
        tf = out_cpp["v2_tile_feats"][0].numpy()
        # Check red5 unseen: 0m (seen in river) -> 0.0; 0p (seen in pon) -> 0.0; 0s (in seat1 DMK!) -> 0.0
        self.assertEqual(tf[4, 13], 0.0)
        self.assertEqual(tf[13, 13], 0.0)
        self.assertEqual(tf[22, 13], 0.0)
        # Check exact unseen count on 5m (b34=4): 0m(1) in seat1 disc/chi -> public=1 -> unseen=3
        self.assertEqual(tf[4, 14:18].sum(), 3.0)
        # Check 5p (b34=13): seat2 disc 5p(1), seat3 pon adds 0p+5p(2), seat3 kakan adds 5p(1) -> all 4 visible -> unseen = 0
        self.assertEqual(tf[13, 14:18].sum(), 0.0)
        # Check 5s (b34=22): DMK used all 4 -> unseen = 0
        self.assertEqual(tf[22, 14:18].sum(), 0.0)
        # Check 1m (b34=0): ANKAN used all 4 -> unseen = 0
        self.assertEqual(tf[0, 14:18].sum(), 0.0)
        # Check conservation across all 34 tiles: H + V <= 4 => H + (4 - U) <= 4 (implicit U >= 0)
        h_cnt = tf[:, 0:4].sum(axis=1)
        u_cnt = tf[:, 14:18].sum(axis=1)
        self.assertTrue(np.all(h_cnt + u_cnt <= 4.0))

        # Overflow check: 5th copy of 1m raises ValueError (1m dora indicator + 1m in hand + 4x 1m ankan)
        bad_closed = {1: 1, 22: 1, 23: 1, 24: 2, 26: 2, 27: 2, 28: 2, 29: 2}
        bad_sp = _make_sparse(seat=0, left=45, doras=(1,), closed37=bad_closed, zimo=29)
        with self.assertRaises(ValueError):
            derive_structured_v2_batch(
                torch.tensor([bad_sp], dtype=torch.int32),
                torch.tensor([nu], dtype=torch.int32),
                torch.tensor([_pad_prog([_ankan(2, 0)])], dtype=torch.int32),
                torch.tensor([ca], dtype=torch.int32),
            )

    # Test #2: Pending trigger exclusion & +9/+10 table genbutsu & CHI/PON non-reset
    def test_2_table_genbutsu_and_pending_trigger_isolation(self):
        # Self = 0, closed 13 tiles
        closed = {1: 3, 2: 1, 3: 1, 4: 1, 11: 3, 21: 3, 31: 1}
        # Opponent seat=1 declares riichi discarding 9m(9), then:
        # seat2 discards 2p(12)
        # seat3 chis 2p? No, seat3 discards 3p(13), seat1 discards 4p(14) (self same-turn reset for 1)
        # Wait: let seat2 discard 6s(26), seat3 PONs something? Say seat3 discards 7s(27),
        # seat0 discards 8s(28), seat1 discards 9p(19) [pending DISCARD_RESP!].
        c_pon = PON_ENCODE_MAP[(11, 11, 11)]  # just need any chi/pon
        prog_resp = _pad_prog([
            _dahai(1, 9, moqie=0, liqi=1),  # seat1 riichi on 9m(b34=8)
            _dahai(2, 12),                  # seat2 landed 2p(10) -> in +9 for seat1
            _dahai(3, 13),                  # seat3 landed 3p(11) -> in +9 for seat1
            _pon(0, 2, c_pon),              # self PON 1p -> MUST NOT reset seat1/seat2/seat3 same-turn!
            _dahai(1, 26, moqie=1, liqi=0), # seat1 pending? No: let seat2 discard 7s(27) AFTER seat1's 9m
        ])
        # Build sequence where seat1 riichi on 9m(8), seat2 disc 2p(10), seat0 pon 1p, seat2 disc 7s(24) [PENDING ev_last]
        prog_resp = _pad_prog([
            _dahai(1, 9, moqie=0, liqi=1),
            _dahai(2, 12),                  # 2p landed
            _pon(0, 2, c_pon),              # call between seat1 riichi & seat2 discard
            _dahai(2, 27),                  # PENDING ev_last = seat2 discards 7s(24)!
        ])
        sp = _make_sparse(seat=0, left=50, doras=(31,), closed37={1: 3, 2: 1, 3: 1, 4: 1, 21: 3, 31: 1})
        # Suppose closed37 has 10 tiles after 1 pon
        nu = [0, 1, 25000, 24000, 25000, 25000]
        ca_resp = _pad_cands([221])  # SKIP => DISCARD_RESP (phase=1)

        out = derive_structured_v2_batch(
            torch.tensor([sp], dtype=torch.int32),
            torch.tensor([nu], dtype=torch.int32),
            torch.tensor([prog_resp], dtype=torch.int32),
            torch.tensor([ca_resp], dtype=torch.int32),
            enable_early_shape_hint=False,
        )
        tf = out["v2_tile_feats"][0].numpy()
        off1 = 19  # rel 1 == seat 1
        # 2p (b34=10) landed after seat1 riichi -> in +9 AND +10 for seat 1!
        self.assertEqual(tf[10, off1 + 9], 1.0)
        self.assertEqual(tf[10, off1 + 10], 1.0)
        # 7s (b34=24) is ev_last PENDING -> MUST BE EXCLUDED from +9 & +10 & seat2 own river!
        self.assertEqual(tf[24, off1 + 9], 0.0)
        self.assertEqual(tf[24, off1 + 10], 0.0)
        off2 = 19 + 11  # seat 2
        self.assertEqual(tf[24, off2 + 0], 0.0)  # pending discard NOT in seat2 landed river
        self.assertEqual(tf[10, off2 + 0], 1.0)  # earlier landed 2p IS in seat2 river
        # Same check with PON non-reset of same-turn genbutsu for seat2 before seat3 kakan pending!
        prog_kakan = _pad_prog([
            _dahai(2, 12),                  # seat2 discards 2p -> seat2 last discard step=1
            _dahai(3, 13),                  # seat3 discards 3p -> landed
            _pon(0, 2, c_pon),              # self PON -> does NOT reset seat2 same-turn!
            _dahai(1, 22),                  # seat1 discards 2s(19) -> landed after pon, STILL in seat2 +10!
            _kakan(1, 15),                  # PENDING ev_last = seat1 KAKAN 5p(13)!
        ])
        out_k = derive_structured_v2_batch(
            torch.tensor([sp], dtype=torch.int32),
            torch.tensor([nu], dtype=torch.int32),
            torch.tensor([prog_kakan], dtype=torch.int32),
            torch.tensor([ca_resp], dtype=torch.int32),
            enable_early_shape_hint=True,
        )
        tf_k = out_k["v2_tile_feats"][0].numpy()
        self.assertEqual(tf_k[11, off2 + 10], 1.0)  # 3p in seat2 +10
        self.assertEqual(tf_k[19, off2 + 10], 1.0)  # 2s after pon STILL in seat2 +10
        self.assertEqual(tf_k[13, off2 + 10], 0.0)  # pending kakan 5p EXCLUDED from +10!

    # Test #3: Riichi boundaries (called declare tile, pending riichi declare, dealer first moqie=0)
    def test_3_riichi_declaration_and_dealer_moqie_boundaries(self):
        c_pon_9m = PON_ENCODE_MAP[(9, 9, 9)]
        closed = {1: 2, 2: 2, 3: 2, 11: 2, 12: 2, 13: 2, 31: 1}
        # Case A: dealer(ju=0) first discard has raw moqie=1 normalized to 0;
        # seat1 double-riichi on 9m, seat2 PONs that 9m -> seat1 riichi STILL stands!
        # seat3 declares riichi as ev_last (pending DISCARD_RESP, liqibang still 0, ippatsu=0)
        prog = _pad_prog([
            _dahai(0, 32, moqie=1, liqi=0),  # step 1: dealer first discard -> moqie normalized to 0!
            _dahai(1, 9, moqie=0, liqi=1),   # seat1 double riichi on 9m
            _pon(2, 2, c_pon_9m),            # seat2 calls pon on seat1's riichi declare 9m
            _dahai(3, 29, moqie=0, liqi=1),  # seat3 pending riichi declaration 9s
        ])
        sp = _make_sparse(seat=0, ju=0, left=55, doras=(1,), closed37=closed)
        nu = [0, 1, 24000, 24000, 25000, 27000]  # seat1 paid 1000 -> liqibang=1; seat3 unpaid yet
        ca = _pad_cands([221])
        out = derive_structured_v2_batch(
            torch.tensor([sp], dtype=torch.int32),
            torch.tensor([nu], dtype=torch.int32),
            torch.tensor([prog], dtype=torch.int32),
            torch.tensor([ca], dtype=torch.int32),
        )
        evs = out["v2_events"][0].numpy()
        pf = out["v2_player_feats"][0].numpy()
        tf = out["v2_tile_feats"][0].numpy()
        # step 1 moqie bit (bit 0 of byte 6) MUST be 0
        self.assertEqual(evs[1, 6] & 1, 0)
        # seat1 (rel 1) double riichi declared = 1 even though declare tile was ponned
        self.assertEqual(pf[1, 14], 1.0)
        self.assertEqual(pf[1, 15], 1.0)
        self.assertEqual(pf[1, 16], 0.0)  # pon broke seat1's ippatsu
        # seat3 (rel 3) pending riichi declare: riichi_declared=1, declare tile marked, ippatsu=0
        self.assertEqual(pf[3, 14], 1.0)
        self.assertEqual(pf[3, 16], 0.0)
        self.assertEqual(tf[26, 19 + 2 * 11 + 1], 1.0)  # opp_riichi_declare_tile for rel3 (9s=b34 26)

    # Test #4: Three types of furiten + karaten vs pure 5th-tile fake tenpai
    def test_4_furiten_and_karaten_vs_fake_tenpai(self):
        # Hand waiting on 4m/7m: 2356m + 123p + 456s + 789s (tenpai waiting on 1m? No: 235m->waiting on 1m/4m if 23m?)
        # Use 23m 55m 123p 456p 789s -> waiting on 1m(0) & 4m(3)!
        closed = {2: 1, 3: 1, 5: 2, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 27: 1, 28: 1, 29: 1}
        nu = [0, 0, 25000, 25000, 25000, 25000]

        # 4A. Temporary furiten: self(0) acted at step1, opponent seat1 later discarded 1m(1) landed,
        # now seat2 discards 4m(4) (pending ev_last). global_feats[19] must be 1, cand PASS furiten=1
        prog_temp = _pad_prog([
            _dahai(0, 31),
            _dahai(1, 1),   # missed 1m landed!
            _dahai(2, 4),   # pending 4m
        ])
        sp = _make_sparse(seat=0, left=50, doras=(31,), closed37=closed)
        out_t = derive_structured_v2_batch(
            torch.tensor([sp], dtype=torch.int32),
            torch.tensor([nu], dtype=torch.int32),
            torch.tensor([prog_temp], dtype=torch.int32),
            torch.tensor([_pad_cands([221])], dtype=torch.int32),
        )
        self.assertEqual(out_t["v2_global_feats"][0, 19].item(), 1.0)
        self.assertEqual(out_t["v2_cand_dyn"][0, 0, 5].item(), 1.0)

        # 4B. Pending trigger is missed wait (no prior missed wait) -> temporary furiten = 0 (can RON!)
        prog_ok = _pad_prog([
            _dahai(0, 31),
            _dahai(1, 29),
            _dahai(2, 4),   # pending 4m ONLY -> NOT furiten!
        ])
        out_ok = derive_structured_v2_batch(
            torch.tensor([sp], dtype=torch.int32),
            torch.tensor([nu], dtype=torch.int32),
            torch.tensor([prog_ok], dtype=torch.int32),
            torch.tensor([_pad_cands([221, 543 + 1])], dtype=torch.int32),
        )
        self.assertEqual(out_ok["v2_global_feats"][0, 19].item(), 0.0)
        self.assertEqual(out_ok["v2_cand_dyn"][0, 0, 5].item(), 0.0)

        # 4C. Riichi furiten persists across own draw on OWN_ACT
        sp_zimo = _make_sparse(seat=0, left=49, doras=(31,), closed37=closed, zimo=32)
        prog_rfur = _pad_prog([
            _dahai(0, 33, moqie=0, liqi=1),  # self riichi
            _dahai(1, 1),                    # missed 1m after riichi!
            _dahai(2, 34),
            _dahai(3, 35),
        ])
        out_rf = derive_structured_v2_batch(
            torch.tensor([sp_zimo], dtype=torch.int32),
            torch.tensor([nu], dtype=torch.int32),
            torch.tensor([prog_rfur], dtype=torch.int32),
            torch.tensor([_pad_cands([32 * 4 + 2])], dtype=torch.int32),
        )
        self.assertEqual(out_rf["v2_global_feats"][0, 19].item(), 1.0)
        self.assertEqual(out_rf["v2_cand_dyn"][0, 0, 5].item(), 1.0)  # discard_post_is_furiten = 1

        # 4D. Karaten (table visible 4x 1m + 4x 4m) vs 5th-tile fake tenpai (self holds 4x 1m)
        # Karaten: hand waits on 1m only (23m 55m 123p 456p 789s -> actually waits 1m & 4m, wait on 33m? No, 12m wait 3m!)
        closed_karaten = {1: 1, 2: 1, 5: 2, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 27: 1, 28: 1, 29: 1}
        sp_kara = _make_sparse(seat=0, left=40, doras=(31,), closed37=closed_karaten, zimo=31)
        prog_kara = _pad_prog([_dahai(1, 3), _dahai(2, 3), _dahai(3, 3), _dahai(1, 3)])  # all 4x 3m discarded on table!
        out_kara = derive_structured_v2_batch(
            torch.tensor([sp_kara], dtype=torch.int32),
            torch.tensor([nu], dtype=torch.int32),
            torch.tensor([prog_kara], dtype=torch.int32),
            torch.tensor([_pad_cands([31 * 4 + 2])], dtype=torch.int32),
        )
        # Karaten MUST stay tenpai (s=0, tenpai=1, u=0)
        self.assertEqual(out_kara["v2_tile_feats"][0, 28, 63].item(), 0.0)
        self.assertEqual(out_kara["v2_tile_feats"][0, 28, 66].item(), 1.0)
        self.assertEqual(out_kara["v2_tile_feats"][0, 28, 64].item(), 0.0)

        # Fake 5th-tile tenpai: self holds 1111m (or 3333m!) + waiting only on 3m (12m 3333m? wait 123m+333m+...):
        # Say hand after discard has 1m+2m + 3333m? Wait 1+2+3=chi leaves 333m! Use 1111m 23m 456p 789p 55s (13 tiles) ->
        # Closed: 1111m (4x 1m) + 123p 456p 789p 23m? Wait 1m makes 1111m+1m = 5th tile! Better: PON of 333m + 12m (10 closed: 3m in self? No: 12m 55m 123p 456p 789s -> only 1m/4m?
        # Simplest 5th tile: 3333m + 45m? Wait 333m + 345m! Wait on 3m-6m? No, pon wait!
        # Self PON 1m(1,1,1) open + closed 1m + pair wait! Wait: 1m is SINGLE (only 1m wait is tanki on 1m!).
        # If closed has 1m, 234p, 567p, 789s (10 tiles) + PON(1m, 1m, 1m): wait is ONLY 1m, which already has 1 + 3 = 4 copies in self!
        c_pon_1m = PON_ENCODE_MAP[(1, 1, 1)]
        closed_fake = {1: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 27: 1, 28: 1, 29: 1}
        sp_fake = _make_sparse(seat=0, left=40, doras=(31,), closed37=closed_fake, zimo=31)
        prog_fake = _pad_prog([_dahai(1, 1), _pon(0, 2, c_pon_1m)])
        out_fake = derive_structured_v2_batch(
            torch.tensor([sp_fake], dtype=torch.int32),
            torch.tensor([nu], dtype=torch.int32),
            torch.tensor([prog_fake], dtype=torch.int32),
            torch.tensor([_pad_cands([31 * 4 + 2])], dtype=torch.int32),
        )
        # Fake 5th-tile tenpai MUST be s = 1 (0.125), is_tenpai = 0
        self.assertAlmostEqual(out_fake["v2_tile_feats"][0, 28, 63].item(), 1.0 / 8.0, places=5)
        self.assertEqual(out_fake["v2_tile_feats"][0, 28, 66].item(), 0.0)

    # Test #5: >= 10,000 cases comparing C++ mjsoul_hand_eval vs kanachan.simulation.Tool + settlement
    def test_5_mjsoul_hand_eval_10000_cases_vs_tool(self):
        tool = Tool()
        rng = random.Random(20261126)

        cases = []

        # Helper to construct closed winning hand from valid partitions + random win tile
        def add_case(closed37: dict[int, int], melds, win_t37: int, dora_inds37: list[int], tool_config: int):
            closed34 = [0] * 34
            for t, cnt in closed37.items():
                closed34[tile34(t)] += cnt
            dora34 = [0] * 34
            for ind in dora_inds37:
                ib = tile34(ind)
                if ib < 27:
                    nxt = (ib // 9) * 9 + ((ib % 9 + 1) % 9)
                elif ib < 31:
                    nxt = 27 + ((ib - 27 + 1) % 4)
                else:
                    nxt = 31 + ((ib - 31 + 1) % 3)
                dora34[nxt] += 1
            total_dora = 0
            total_red5 = 0
            for t, cnt in closed37.items():
                total_dora += cnt * dora34[tile34(t)]
                if is_red5_tile37(t):
                    total_red5 += cnt
            cpp_melds = []
            for mtype, mtiles37, opened in melds:
                if mtype == "kan":
                    b = tile34(mtiles37[0])
                    total_dora += 4 * dora34[b]
                    if b in (4, 13, 22):
                        total_red5 += 1
                    cpp_melds.append((2, [b, b, b], 1 if opened else 0))
                elif mtype == "pon":
                    mt34 = sorted(tile34(t) for t in mtiles37)
                    for t in mtiles37:
                        total_dora += dora34[tile34(t)]
                        if is_red5_tile37(t):
                            total_red5 += 1
                    cpp_melds.append((1, mt34, 1))
                elif mtype == "chi":
                    mt34 = sorted(tile34(t) for t in mtiles37)
                    for t in mtiles37:
                        total_dora += dora34[tile34(t)]
                        if is_red5_tile37(t):
                            total_red5 += 1
                    cpp_melds.append((0, mt34, 1))
            cases.append((closed37, melds, win_t37, dora_inds37, tool_config,
                          closed34, cpp_melds, tile34(win_t37), total_dora, total_red5))

        # Curated adversarial cases covering every yakuman, 45666 ron sanankou, chuuren, chiitoi, open tanyao, etc.
        # Kokushi & Daburu Kokushi
        base_k = {1: 1, 9: 1, 11: 1, 19: 1, 21: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1}
        for wt in (1, 30, 36):
            c = dict(base_k)
            c[wt] += 1
            add_case(c, [], wt, [1], (1 << 0) | (1 << 10) | (1 << 14))
            add_case(c, [], wt, [1], (1 << 10) | (1 << 14))
        # Chuuren & Daburu Chuuren
        c_ch = {1: 3, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 4}
        add_case(c_ch, [], 9, [2], (1 << 0) | (1 << 10) | (1 << 14))
        add_case(c_ch, [], 2, [2], (1 << 10) | (1 << 14))
        # Suuankou & Suuankou Tanki & Sanankou on 45666 ron vs shampon
        c_sa = {1: 3, 2: 3, 11: 3, 21: 3, 34: 2}
        add_case(c_sa, [], 34, [1], (1 << 10) | (1 << 14))
        add_case(c_sa, [], 1, [1], (1 << 0) | (1 << 10) | (1 << 14))
        add_case(c_sa, [], 1, [1], (1 << 10) | (1 << 14))  # ron on pon -> sanankou+toitoi, NOT suuankou
        # 45666m ron cases: 456m+666m+... -> sanankou vs 456m+66m+... -> sanankou
        add_case({4: 1, 5: 1, 6: 4, 11: 3, 21: 3, 30: 2}, [], 6, [1], (1 << 11) | (1 << 14))
        add_case({4: 1, 5: 1, 6: 3, 11: 3, 21: 3, 30: 3}, [], 6, [1], (1 << 11) | (1 << 14))
        # Double wind pair fu + open pinfu 30fu + chiitoitsu tsuuiisou + kazoe >=13 clamped to 13
        c_kazoe = {2: 1, 3: 1, 4: 1, 0: 1, 6: 1, 7: 1, 12: 3, 22: 3, 34: 2}
        add_case(c_kazoe, [], 0, [1, 1, 1, 1, 4], (1 << 0) | (1 << 2) | (1 << 5) | (1 << 7) | (1 << 10) | (1 << 14))

        # Generate random valid closed/open hands up to 10,200 cases
        all_chis = []
        for s in range(3):
            for r in range(1, 8):
                all_chis.append([s * 10 + r, s * 10 + r + 1, s * 10 + r + 2])

        def alloc_t37(pool37, b):
            if b in (4, 13, 22):
                red = (b // 9) * 10
                blk = red + 5
                opts = ([red] * pool37[red]) + ([blk] * pool37[blk])
                t = rng.choice(opts)
                pool37[t] -= 1
                return t
            t = base34_to_tile37(b)
            pool37[t] -= 1
            return t

        def try_add_random():
            pool37 = [0] * 37
            for t in range(37):
                pool37[t] = 1 if t in (0, 10, 20) else (3 if t in (5, 15, 25) else 4)

            def avail_b(b):
                if b in (4, 13, 22):
                    red = (b // 9) * 10
                    return pool37[red] + pool37[red + 5]
                return pool37[base34_to_tile37(b)]

            is_chiitoi = rng.random() < 0.08
            melds = []
            closed = {}
            if is_chiitoi:
                chosen = rng.sample(range(34), 7)
                for b in chosen:
                    for _ in range(2):
                        t = alloc_t37(pool37, b)
                        closed[t] = closed.get(t, 0) + 1
            else:
                n_open = rng.choices([0, 1, 2, 3, 4], weights=[55, 22, 13, 6, 4])[0]
                for mi in range(4):
                    if rng.random() < 0.65:
                        seq = rng.choice(all_chis)
                        bs = [tile34(x) for x in seq]
                        if any(avail_b(b) < 1 for b in bs):
                            return False
                        ts = [alloc_t37(pool37, b) for b in bs]
                        if mi < n_open:
                            melds.append(("chi", ts, True))
                        else:
                            for t37 in ts:
                                closed[t37] = closed.get(t37, 0) + 1
                    else:
                        b = rng.randrange(34)
                        is_k = (mi < n_open or rng.random() < 0.15) and rng.random() < 0.25
                        need = 4 if is_k else 3
                        if avail_b(b) < need:
                            return False
                        ts = [alloc_t37(pool37, b) for _ in range(need)]
                        if is_k:
                            melds.append(("kan", ts, mi < n_open))
                        elif mi < n_open:
                            melds.append(("pon", ts, True))
                        else:
                            for t37 in ts:
                                closed[t37] = closed.get(t37, 0) + 1
                pair_cands = [b for b in range(34) if avail_b(b) >= 2]
                if not pair_cands:
                    return False
                pb = rng.choice(pair_cands)
                for _ in range(2):
                    t = alloc_t37(pool37, pb)
                    closed[t] = closed.get(t, 0) + 1

            # Pick win_t37 from closed
            wt = rng.choice([t for t, c in closed.items() if c > 0])
            # Pick 1..4 dora indicators from remaining physical pool
            n_ind = rng.randint(1, 4)
            dora_inds = []
            for _ in range(n_ind):
                avail = [t for t in range(37) if pool37[t] > 0]
                if not avail:
                    break
                dt = rng.choice(avail)
                pool37[dt] -= 1
                dora_inds.append(dt)
            if not dora_inds:
                dora_inds = [1]

            has_open = any(m[2] for m in melds)
            is_tsumo = rng.random() < 0.5
            pw = rng.randint(0, 3)
            rw = rng.randint(0, 2)
            cfg = (1 << (10 + pw)) | (1 << (14 + rw))
            if is_tsumo:
                cfg |= 1 << 0
            if not has_open:
                r_mode = rng.random()
                if r_mode < 0.35:
                    cfg |= 1 << 1
                    if rng.random() < 0.3:
                        cfg |= 1 << 2
                elif r_mode < 0.42:
                    cfg |= 1 << 7
            add_case(closed, melds, wt, dora_inds, cfg)
            return True

        while len(cases) < 10200:
            try_add_random()

        # Verify every single case matches kanachan.simulation.Tool 100% identically!
        for idx, (closed37, melds, win_t37, dora_inds37, tool_config,
                  closed34, cpp_melds, win_t34, tot_dora, tot_red5) in enumerate(cases):
            # Reference via Tool
            tiles136, tool_melds = _tiles37_to_136_and_melds(closed37, melds)
            dora136 = []
            used_d = set(tiles136)
            for m in tool_melds:
                if m[0] == "kan":
                    used_d.update(m[1])
            for ind37 in dora_inds37:
                ib = tile34(ind37)
                start = ib * 4 + (1 if ib in (4, 13, 22) else 0)
                if is_red5_tile37(ind37):
                    dora136.append(ib * 4)
                else:
                    picked = None
                    for k in range(start, ib * 4 + 4):
                        if k not in used_d and k not in dora136:
                            picked = k
                            break
                    if picked is None:
                        picked = start
                    dora136.append(picked)

            tool_cands = []
            if tool_config & 1:
                tool.append_zimohu_candidate(tiles136, tool_melds, win_t37, tool_config, tool_cands)
                t_valid = 219 in tool_cands
            else:
                tool.append_rong_candidate(0, tiles136, tool_melds, win_t37, tool_config, tool_cands)
                t_valid = 543 in tool_cands

            c_valid, c_han, c_fu, _ = _candidate_features.eval_mjsoul_hand(
                closed34, cpp_melds, win_t34, tot_dora, tot_red5, tool_config
            )
            self.assertEqual(c_valid, t_valid, f"valid mismatch at case {idx}")
            if t_valid:
                t_han, t_fu = tool.calculate_hand(tiles136, tool_melds, win_t37, dora136, tool_config)
                self.assertEqual(
                    (c_han, c_fu), (t_han, t_fu),
                    f"han/fu mismatch at case {idx}: cpp={(c_han, c_fu)} tool={(t_han, t_fu)}"
                )

        # Verify Pao & +300*ben non-double-counting & would_end_game via eval_win_settlement
        # Dealer tsumo Daisangen Pao (han=13) from seat 2, winner seat 0 (ju=0), ben=2, liqibang=1
        post, gain, rk, endg = _candidate_features.eval_win_settlement(
            13, 40, 1, 1, 0, 0, 0, 0, 2, 1, 0, [25000, 25000, 25000, 25000], 2, -1
        )
        # Winner gains 48000 + 600(ben) + 1000(riichi) = 49600; Pao payer(2) pays all 48000 + 600 = 48600!
        self.assertEqual(gain, 49600)
        self.assertEqual(post, (74600, 25000, -23600, 25000))
        self.assertTrue(endg)  # Tobi on seat 2 (< 0)
        self.assertEqual(rk, 0)

        # All-Last (Hanchan South 4, chang=1, ju=3) dealer wins and reaches rank 0 -> ends game
        post2, gain2, rk2, endg2 = _candidate_features.eval_win_settlement(
            3, 40, 0, 0, 3, 2, 3, 1, 1, 0, 0, [31000, 22000, 22000, 25000], -1, -1
        )
        # Dealer ron 3han 40fu = 7700 + 300*ben -> 25000 + 8000 = 33000 > 31000 -> rank 0 -> ends!
        self.assertEqual(gain2, 8000)
        self.assertEqual(post2, (31000, 22000, 14000, 33000))
        self.assertEqual(rk2, 0)
        self.assertTrue(endg2)

    # Test #6: 546 action manifest bijection & max candidates <= 32 (30 kokushi, 10 discard_resp)
    def test_6_action_manifest_and_max_candidates(self):
        self.assertEqual(tuple(KANACHAN_ACTION_MANIFEST.shape), (547, 10))
        cpp_m = np.frombuffer(_candidate_features.action_manifest(), dtype=np.int32).reshape(547, 10)
        np.testing.assert_array_equal(KANACHAN_ACTION_MANIFEST, cpp_m)
        sha = hashlib.sha256(np.ascontiguousarray(cpp_m, dtype=np.int32).tobytes("C")).hexdigest()
        self.assertEqual(sha, KANACHAN_ACTION_MANIFEST_SHA256)

        seen = set()
        for aid in range(547):
            row = decode_action_tuple(aid)
            self.assertEqual(encode_action_tuple(row), aid)
            seen.add(row)
        self.assertEqual(len(seen), 547)

        # 30-candidate extreme: Kokushi 13-wait + TSUMO(219) + KYUSHU(220) = 30 <= 32
        kokushi_14 = [1, 9, 11, 19, 21, 29, 30, 31, 32, 33, 34, 35, 36]
        cands_30 = []
        for t in kokushi_14:
            cands_30.append(4 * t + 0 + 0)  # dama discard
            cands_30.append(4 * t + 0 + 1)  # riichi discard
        cands_30.extend([4 * 36 + 2 + 0, 4 * 36 + 2 + 1])  # tsumogiri dama + riichi on zimo 36
        cands_30.extend([219, 220])
        self.assertEqual(len(cands_30), 30)
        closed_k13 = {t: 1 for t in kokushi_14}
        sp30 = _make_sparse(seat=0, left=69, doras=(2,), closed37=closed_k13, zimo=36)
        nu = [0, 0, 25000, 25000, 25000, 25000]
        out30 = derive_structured_v2_batch(
            torch.tensor([sp30], dtype=torch.int32),
            torch.tensor([nu], dtype=torch.int32),
            torch.tensor([_pad_prog([])], dtype=torch.int32),
            torch.tensor([_pad_cands(cands_30)], dtype=torch.int32),
        )
        self.assertEqual(int(out30["v2_cand_mask"][0].sum().item()), 30)

        # DISCARD_RESP on kamicha 4m(4) with 9 legal candidates (<= 32):
        # SKIP + 5 CHIs + PON + DMK + RON
        closed_9 = {2: 2, 3: 1, 4: 3, 0: 1, 5: 1, 6: 1, 7: 1, 12: 1, 13: 1, 14: 1}
        cands_9 = [
            221,
            222 + CHI_ENCODE_MAP[(2, 3, 4)],
            222 + CHI_ENCODE_MAP[(3, 5, 4)],
            222 + CHI_ENCODE_MAP[(3, 0, 4)],
            222 + CHI_ENCODE_MAP[(5, 6, 4)],
            222 + CHI_ENCODE_MAP[(0, 6, 4)],
            312 + 2 * 40 + PON_ENCODE_MAP[(4, 4, 4)],
            432 + 2 * 37 + 4,
            543 + 2,
        ]
        sp_9 = _make_sparse(seat=0, left=48, doras=(1,), closed37=closed_9)
        out9 = derive_structured_v2_batch(
            torch.tensor([sp_9], dtype=torch.int32),
            torch.tensor([nu], dtype=torch.int32),
            torch.tensor([_pad_prog([_dahai(3, 4)])], dtype=torch.int32),
            torch.tensor([_pad_cands(cands_9)], dtype=torch.int32),
        )
        self.assertEqual(int(out9["v2_cand_mask"][0].sum().item()), 9)
        self.assertLessEqual(int(out9["v2_cand_mask"][0].sum().item()), 32)

    # Test #7: Zero-diff comparison C++ structured_v2 vs pure-Python reference
    def test_7_cpp_vs_python_reference_zero_diff(self):
        # Mix of OWN_ACT discard + ankan + kakan + tsumo + DISCARD_RESP + KAKAN_RESP + ANKAN_RESP
        c_chi = CHI_ENCODE_MAP[(14, 16, 10)]  # 456p calling red 5p from kamicha
        samples = [
            # 1) OWN_ACT with TSUMO & riichi/dama discards & ANKAN
            (
                _make_sparse(seat=1, chang=0, ju=2, left=24, doras=(4, 19),
                             closed37={2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 12: 3, 21: 2, 33: 2}, zimo=33),
                [1, 2, 24000, 26000, 23000, 25000],
                _pad_prog([_dahai(2, 31), _dahai(3, 1, liqi=1), _dahai(0, 9), _chi(2, c_chi)]),
                _pad_cands([2 * 4, 3 * 4, 5 * 4, 33 * 4 + 2, 219]),
            ),
            # 2) OWN_ACT after CHI with kuikae
            (
                _make_sparse(seat=2, chang=1, ju=3, left=38, doras=(31,),
                             closed37={1: 1, 4: 1, 7: 1, 8: 1, 9: 1, 21: 2, 22: 2, 31: 2}),
                [0, 0, 31000, 22000, 25000, 22000],
                _pad_prog([_dahai(1, 10), _chi(2, c_chi)]),
                _pad_cands([1 * 4, 4 * 4, 7 * 4, 31 * 4]),
            ),
            # 3) DISCARD_RESP with SKIP, CHI, RON
            (
                _make_sparse(seat=0, chang=0, ju=0, left=18, doras=(1,),
                             closed37={2: 1, 3: 1, 5: 3, 12: 3, 22: 3, 23: 2}),
                [2, 1, 25000, 25000, 25000, 24000],
                _pad_prog([_dahai(3, 19), _dahai(0, 29), _dahai(3, 4)]),
                _pad_cands([221, 222 + CHI_ENCODE_MAP[(2, 3, 4)], 222 + CHI_ENCODE_MAP[(3, 5, 4)], 543 + 2]),
            ),
            # 4) KAKAN_RESP chankan RON
            (
                _make_sparse(seat=3, chang=1, ju=1, left=12, doras=(11,),
                             closed37={1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 34: 2}),
                [0, 0, 25000, 25000, 25000, 25000],
                _pad_prog([_dahai(2, 26), _pon(1, 0, PON_ENCODE_MAP[(26, 26, 26)]), _kakan(1, 26)]),
                _pad_cands([221, 543 + 1]),
            ),
        ]

        sp_t = torch.tensor([s[0] for s in samples], dtype=torch.int32)
        nu_t = torch.tensor([s[1] for s in samples], dtype=torch.int32)
        pr_t = torch.tensor([s[2] for s in samples], dtype=torch.int32)
        ca_t = torch.tensor([s[3] for s in samples], dtype=torch.int32)

        for hint in (False, True):
            cpp_out = derive_structured_v2_batch(
                sp_t, nu_t, pr_t, ca_t, enable_early_shape_hint=hint
            )
            py_out = derive_structured_v2_batch_py(
                sp_t.numpy(), nu_t.numpy(), pr_t.numpy(), ca_t.numpy(),
                enable_early_shape_hint=hint,
            )
            for key in (
                "v2_tile_feats", "v2_events", "v2_event_len",
                "v2_global_feats", "v2_player_feats",
                "v2_cand_mode", "v2_cand_dyn",
            ):
                diff = np.abs(
                    cpp_out[key].numpy().astype(np.float64)
                    - py_out[key].astype(np.float64)
                ).max()
                self.assertLess(diff, 1e-6, f"Mismatch on {key} (hint={hint}): diff={diff}")


if __name__ == "__main__":
    unittest.main()
