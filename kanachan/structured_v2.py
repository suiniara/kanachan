"""Static action manifest, 6x suit permutation tables, and structured_v2 helpers."""

from __future__ import annotations

import hashlib
from itertools import permutations
from typing import Any

import numpy as np
import torch


STRUCTURED_V2_SCHEMA_ID = "kanachan_structured_v2_70_10_48_32_8"
NUM_TYPES_OF_ACTIONS = 546
PAD_ACTION_ID = 546
NULL_TILE_IDX = 34
TILE37_NONE = 37
NULL_SOURCE_IDX = 4
MAX_CANDIDATES = 32
MAX_PROGRESSION_LEN = 109
MAX_EVENT_LEN = 214
MAX_ROPE_LEN = 2048

TILE_FEATS_DIM = 70
EVENT_BYTES = 10
GLOBAL_FEATS_DIM = 48
PLAYER_FEATS_DIM = 32
CAND_DYN_DIM = 8
NUM_CAND_MODES = 7
REL_BIAS_NUM_CLASSES = 17

SUIT_PERMS: tuple[tuple[int, int, int], ...] = tuple(permutations((0, 1, 2)))

ACTION_TYPE_NAMES: tuple[str, ...] = (
    "DISCARD",
    "ANKAN",
    "KAKAN",
    "TSUMO",
    "KYUSHU",
    "SKIP",
    "CHI",
    "PON",
    "DAIMINKAN",
    "RON",
    "PAD",
)

CAND_MODE_NAMES: tuple[str, ...] = (
    "PAD",
    "POST_DISCARD_EXACT",
    "PRE_DISCARD_MELD_ENVELOPE",
    "PENDING_RINSHAN_KAN",
    "PASS_CURRENT_STATE",
    "WIN_EVAL",
    "KYUSHU_DRAW_EVAL",
)

# 9-family evaluation buckets matching §4.2.4
ACTION_FAMILY_NAMES: tuple[str, ...] = (
    "DISCARD_DAMA",
    "DISCARD_RIICHI",
    "CHI",
    "PON",
    "KAN",
    "TSUMO",
    "RON",
    "SKIP",
    "KYUSHU",
)

OFFSETS_136: tuple[int, ...] = (
    0, 1, 5, 9, 13, 17, 20, 24, 28, 32,
    36, 37, 41, 45, 49, 53, 56, 60, 64, 68,
    72, 73, 77, 81, 85, 89, 92, 96, 100, 104,
    108, 112, 116, 120, 124, 128, 132,
)


def tile34(tile37: int) -> int:
    """Map kanachan tile37 [0..36] (or 37->34) to base34 [0..33] (or 34)."""
    if tile37 == TILE37_NONE:
        return NULL_TILE_IDX
    if not 0 <= tile37 < 37:
        raise ValueError(f"Invalid tile37 ID: {tile37}")
    if tile37 < 30:
        return (tile37 // 10) * 9 + (4 if tile37 % 10 == 0 else tile37 % 10 - 1)
    return tile37 - 3


def base34_to_tile37(base34: int) -> int:
    """Map canonical base34 [0..33] (or 34->37) to non-red tile37."""
    if base34 == NULL_TILE_IDX:
        return TILE37_NONE
    if not 0 <= base34 < 34:
        raise ValueError(f"Invalid base34 ID: {base34}")
    if base34 < 27:
        return (base34 // 9) * 10 + (base34 % 9) + 1
    return base34 + 3


def is_red5_tile37(tile37: int) -> bool:
    return tile37 in (0, 10, 20)


def idx136_to_tile37(idx136: int) -> tuple[int, int]:
    """Return (tile37, copy_offset) for a sparse physical index 0..135."""
    if not 0 <= idx136 < 136:
        raise ValueError(f"Invalid idx136: {idx136}")
    for t37 in range(36, -1, -1):
        base = OFFSETS_136[t37]
        if idx136 >= base:
            return t37, idx136 - base
    raise AssertionError("Unreachable")


def _build_chi_table() -> tuple[tuple[int, int, int, int], ...]:
    rows: list[tuple[int, int, int, int]] = []
    for suit in range(3):
        for called in range(1, 10):
            for start in range(max(1, called - 2), min(7, called) + 1):
                consumed = [t for t in range(start, start + 3) if t != called]
                kuikae = TILE37_NONE
                if called == start and start + 3 <= 9:
                    kuikae = suit * 10 + start + 3
                elif called == start + 2 and start - 1 >= 1:
                    kuikae = suit * 10 + start - 1
                row = (consumed[0], consumed[1], called)
                rows.append((*tuple(suit * 10 + t for t in row), kuikae))
                if start <= 5 < start + 3:
                    rows.append((
                        *(suit * 10 + (0 if t == 5 else t) for t in row),
                        kuikae,
                    ))
    assert len(rows) == 90
    return tuple(rows)


def _build_pon_table() -> tuple[tuple[int, int, int], ...]:
    rows: list[tuple[int, int, int]] = []
    for t34 in range(34):
        t37 = base34_to_tile37(t34)
        rows.append((t37, t37, t37))
        if t34 in (4, 13, 22):
            red = t37 - 5
            rows.append((red, t37, t37))
            rows.append((t37, t37, red))
    assert len(rows) == 40
    return tuple(rows)


CHI_TABLE: tuple[tuple[int, int, int, int], ...] = _build_chi_table()
PON_TABLE: tuple[tuple[int, int, int], ...] = _build_pon_table()
CHI_ENCODE_MAP: dict[tuple[int, int, int], int] = {
    row[:3]: idx for idx, row in enumerate(CHI_TABLE)
}
PON_ENCODE_MAP: dict[tuple[int, int, int], int] = {
    row: idx for idx, row in enumerate(PON_TABLE)
}


def action_cand_mode(action_type: int) -> int:
    if action_type == 10:
        return 0  # PAD
    if action_type == 0:
        return 1  # POST_DISCARD_EXACT
    if action_type in (6, 7):
        return 2  # PRE_DISCARD_MELD_ENVELOPE
    if action_type in (1, 2, 8):
        return 3  # PENDING_RINSHAN_KAN
    if action_type == 5:
        return 4  # PASS_CURRENT_STATE
    if action_type in (3, 9):
        return 5  # WIN_EVAL
    if action_type == 4:
        return 6  # KYUSHU_DRAW_EVAL
    raise ValueError(f"Unknown action_type: {action_type}")


def action_family_id(global_id: int) -> int:
    """Return 0..8 index into ACTION_FAMILY_NAMES, or -1 for PAD."""
    if not 0 <= global_id < NUM_TYPES_OF_ACTIONS:
        return -1
    if global_id < 148:
        return 1 if (global_id & 1) == 1 else 0  # DISCARD_RIICHI vs DAMA
    if global_id < 219:
        return 4  # KAN (ANKAN + KAKAN)
    if global_id == 219:
        return 5  # TSUMO
    if global_id == 220:
        return 8  # KYUSHU
    if global_id == 221:
        return 7  # SKIP
    if global_id < 312:
        return 2  # CHI
    if global_id < 432:
        return 3  # PON
    if global_id < 543:
        return 4  # KAN (DAIMINKAN)
    return 6  # RON


def _build_action_manifest() -> np.ndarray:
    # Columns:
    # 0: action_type (0..10)
    # 1: target_base34 (0..34)
    # 2: c0_base34 (0..34)
    # 3: c1_base34 (0..34)
    # 4: source_rel (1..4)
    # 5: target_is_red5 (0/1)
    # 6: consumed_has_red5 (0/1)
    # 7: is_moqie (0/1)
    # 8: is_riichi (0/1)
    # 9: cand_mode (0..6)
    manifest = np.zeros((547, 10), dtype=np.int32)
    manifest[:, 1:4] = NULL_TILE_IDX
    manifest[:, 4] = NULL_SOURCE_IDX

    for aid in range(547):
        if aid < 148:
            t37 = aid // 4
            moqie = (aid // 2) & 1
            riichi = aid & 1
            manifest[aid, 0] = 0
            manifest[aid, 1] = tile34(t37)
            manifest[aid, 5] = int(is_red5_tile37(t37))
            manifest[aid, 7] = moqie
            manifest[aid, 8] = riichi
        elif aid < 182:
            b34 = aid - 148
            manifest[aid, 0] = 1
            manifest[aid, 1] = b34
            manifest[aid, 6] = int(b34 in (4, 13, 22))
        elif aid < 219:
            t37 = aid - 182
            manifest[aid, 0] = 2
            manifest[aid, 1] = tile34(t37)
            manifest[aid, 5] = int(is_red5_tile37(t37))
        elif aid == 219:
            manifest[aid, 0] = 3
        elif aid == 220:
            manifest[aid, 0] = 4
        elif aid == 221:
            manifest[aid, 0] = 5
        elif aid < 312:
            c0_37, c1_37, called_37, _ = CHI_TABLE[aid - 222]
            manifest[aid, 0] = 6
            manifest[aid, 1] = tile34(called_37)
            manifest[aid, 2] = tile34(c0_37)
            manifest[aid, 3] = tile34(c1_37)
            manifest[aid, 4] = 3
            manifest[aid, 5] = int(is_red5_tile37(called_37))
            manifest[aid, 6] = int(
                is_red5_tile37(c0_37) or is_red5_tile37(c1_37)
            )
        elif aid < 432:
            rel_from = (aid - 312) // 40
            code = (aid - 312) % 40
            c0_37, c1_37, called_37 = PON_TABLE[code]
            manifest[aid, 0] = 7
            manifest[aid, 1] = tile34(called_37)
            manifest[aid, 2] = tile34(c0_37)
            manifest[aid, 3] = tile34(c1_37)
            manifest[aid, 4] = rel_from + 1
            manifest[aid, 5] = int(is_red5_tile37(called_37))
            manifest[aid, 6] = int(
                is_red5_tile37(c0_37) or is_red5_tile37(c1_37)
            )
        elif aid < 543:
            rel_from = (aid - 432) // 37
            called_37 = (aid - 432) % 37
            b34 = tile34(called_37)
            manifest[aid, 0] = 8
            manifest[aid, 1] = b34
            manifest[aid, 4] = rel_from + 1
            manifest[aid, 5] = int(is_red5_tile37(called_37))
            manifest[aid, 6] = int(
                b34 in (4, 13, 22) and not is_red5_tile37(called_37)
            )
        elif aid < 546:
            rel_from = aid - 543
            manifest[aid, 0] = 9
            manifest[aid, 4] = rel_from + 1
        else:
            manifest[aid, 0] = 10

        manifest[aid, 9] = action_cand_mode(int(manifest[aid, 0]))

    # Assert strict 1:1 bijection over 0..546.
    rows = {tuple(int(x) for x in manifest[i]) for i in range(547)}
    if len(rows) != 547:
        raise AssertionError("KANACHAN_ACTION_MANIFEST is not bijective!")
    return manifest


KANACHAN_ACTION_MANIFEST: np.ndarray = _build_action_manifest()
KANACHAN_ACTION_MANIFEST.setflags(write=False)

HAS_SLOT_FLAGS: np.ndarray = np.stack(
    [
        KANACHAN_ACTION_MANIFEST[:, 1] != NULL_TILE_IDX,
        KANACHAN_ACTION_MANIFEST[:, 2] != NULL_TILE_IDX,
        KANACHAN_ACTION_MANIFEST[:, 3] != NULL_TILE_IDX,
        KANACHAN_ACTION_MANIFEST[:, 4] != NULL_SOURCE_IDX,
    ],
    axis=-1,
).astype(np.float32)
HAS_SLOT_FLAGS.setflags(write=False)

STATIC_FLAGS: np.ndarray = KANACHAN_ACTION_MANIFEST[:, 5:9].astype(np.float32)
STATIC_FLAGS.setflags(write=False)

KANACHAN_ACTION_MANIFEST_SHA256: str = hashlib.sha256(
    np.ascontiguousarray(KANACHAN_ACTION_MANIFEST, dtype=np.int32).tobytes(
        order="C"
    )
).hexdigest()

ACTION_MANIFEST_REVERSE: dict[tuple[int, ...], int] = {
    tuple(int(x) for x in KANACHAN_ACTION_MANIFEST[i]): i for i in range(547)
}


def decode_action_tuple(action_id: int) -> tuple[int, ...]:
    """Forward lookup into static manifest row for action_id [0..546]."""
    return tuple(int(x) for x in KANACHAN_ACTION_MANIFEST[action_id])


def encode_action_tuple(row: tuple[int, ...] | list[int]) -> int:
    """Inverse lookup from 10-int manifest row back to global action_id."""
    key = tuple(int(x) for x in row)
    if key not in ACTION_MANIFEST_REVERSE:
        raise KeyError(f"Unknown manifest row: {key}")
    return ACTION_MANIFEST_REVERSE[key]


def _build_suit_permutation_tables() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    perm_b34 = np.zeros((6, 35), dtype=np.int32)
    inv_b34 = np.zeros((6, 35), dtype=np.int32)
    perm_t37 = np.zeros((6, 38), dtype=np.int32)
    inv_t37 = np.zeros((6, 38), dtype=np.int32)
    perm_act = np.zeros((6, 547), dtype=np.int32)
    perm_chi = np.zeros((6, 90), dtype=np.int32)
    perm_pon = np.zeros((6, 40), dtype=np.int32)

    for p_idx, perm in enumerate(SUIT_PERMS):
        for b in range(35):
            if b < 27:
                mapped = perm[b // 9] * 9 + (b % 9)
            else:
                mapped = b
            perm_b34[p_idx, b] = mapped
            inv_b34[p_idx, mapped] = b

        for t in range(38):
            if t < 30:
                mapped = perm[t // 10] * 10 + (t % 10)
            else:
                mapped = t
            perm_t37[p_idx, t] = mapped
            inv_t37[p_idx, mapped] = t

        for c in range(90):
            c0, c1, called, _ = CHI_TABLE[c]
            perm_chi[p_idx, c] = CHI_ENCODE_MAP[
                (perm_t37[p_idx, c0], perm_t37[p_idx, c1], perm_t37[p_idx, called])
            ]

        for pc in range(40):
            c0, c1, called = PON_TABLE[pc]
            perm_pon[p_idx, pc] = PON_ENCODE_MAP[
                (perm_t37[p_idx, c0], perm_t37[p_idx, c1], perm_t37[p_idx, called])
            ]

        for aid in range(547):
            row = list(int(x) for x in KANACHAN_ACTION_MANIFEST[aid])
            row[1] = int(perm_b34[p_idx, row[1]])
            row[2] = int(perm_b34[p_idx, row[2]])
            row[3] = int(perm_b34[p_idx, row[3]])
            perm_act[p_idx, aid] = ACTION_MANIFEST_REVERSE[tuple(row)]

    for arr in (perm_b34, inv_b34, perm_t37, inv_t37, perm_act, perm_chi, perm_pon):
        arr.setflags(write=False)
    return perm_b34, inv_b34, perm_t37, inv_t37, perm_act, perm_chi, perm_pon


(
    PERM_BASE34,
    INV_PERM_BASE34,
    PERM_TILE37,
    INV_PERM_TILE37,
    PERM_ACTION_ID,
    PERM_CHI_ENCODE,
    PERM_PENG_ENCODE,
) = _build_suit_permutation_tables()


def permute_raw_four_fields(
    sparse: torch.Tensor,
    numeric: torch.Tensor,
    progression: torch.Tensor,
    candidates: torch.Tensor,
    perm_idx: int | torch.Tensor,
    chosen_cand_idx: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
]:
    """Apply S_3 suit permutation(s) to raw 4-field tensors (and action index).

    Supported shapes: unbatched [29]/[6]/[L_p]/[32] or batched [B, ...].
    """
    was_1d = sparse.dim() == 1
    if was_1d:
        sparse = sparse.unsqueeze(0)
        numeric = numeric.unsqueeze(0)
        progression = progression.unsqueeze(0)
        candidates = candidates.unsqueeze(0)
        if chosen_cand_idx is not None and chosen_cand_idx.dim() == 0:
            chosen_cand_idx = chosen_cand_idx.unsqueeze(0)

    bsz = sparse.shape[0]
    device = sparse.device
    if isinstance(perm_idx, int):
        p_vec = torch.full((bsz,), perm_idx, dtype=torch.long, device=device)
    else:
        p_vec = perm_idx.to(device=device, dtype=torch.long)

    p_t37 = torch.from_numpy(PERM_TILE37.copy()).to(device=device, dtype=torch.long)[p_vec]
    p_b34 = torch.from_numpy(PERM_BASE34.copy()).to(device=device, dtype=torch.long)[p_vec]
    p_chi = torch.from_numpy(PERM_CHI_ENCODE.copy()).to(device=device, dtype=torch.long)[p_vec]
    p_pon = torch.from_numpy(PERM_PENG_ENCODE.copy()).to(device=device, dtype=torch.long)[p_vec]
    p_act = torch.from_numpy(PERM_ACTION_ID.copy()).to(device=device, dtype=torch.long)[p_vec]
    offsets_t = torch.tensor(OFFSETS_136, device=device, dtype=torch.long)

    # Precompute 136 -> t37 and copy offset
    idx136_to_t37 = torch.empty(136, dtype=torch.long, device=device)
    idx136_to_off = torch.empty(136, dtype=torch.long, device=device)
    for idx in range(136):
        t37, off = idx136_to_tile37(idx)
        idx136_to_t37[idx] = t37
        idx136_to_off[idx] = off

    out_sparse = sparse.clone()
    tail = sparse[:, 10:].long()
    out_tail = tail.clone()

    m_dora = (tail >= 152) & (tail < 337)
    if m_dora.any():
        rel_v = torch.where(m_dora, tail - 152, 0)
        slot = rel_v // 37
        t37 = rel_v % 37
        mapped_t37 = torch.gather(p_t37, 1, t37)
        out_tail = torch.where(m_dora, 152 + slot * 37 + mapped_t37, out_tail)

    m_hand = (tail >= 337) & (tail < 473)
    if m_hand.any():
        idx136 = torch.where(m_hand, tail - 337, 0)
        t37 = idx136_to_t37[idx136]
        off = idx136_to_off[idx136]
        mapped_t37 = torch.gather(p_t37, 1, t37)
        mapped_136 = offsets_t[mapped_t37] + off
        out_tail = torch.where(m_hand, 337 + mapped_136, out_tail)

    m_zimo = (tail >= 473) & (tail < 510)
    if m_zimo.any():
        t37 = torch.where(m_zimo, tail - 473, 0)
        mapped_t37 = torch.gather(p_t37, 1, t37)
        out_tail = torch.where(m_zimo, 473 + mapped_t37, out_tail)

    out_sparse[:, 10:] = out_tail.to(out_sparse.dtype)
    # Keep physical hand slots sorted ascending inside each row so sparse order
    # stays canonical cleanly (hand occupies contiguous slots after doras).
    for b in range(bsz):
        row = out_sparse[b]
        hand_positions = (row[10:] >= 337) & (row[10:] < 473)
        if hand_positions.any():
            vals, _ = torch.sort(row[10:][hand_positions])
            row_tail = row[10:].clone()
            row_tail[hand_positions] = vals
            out_sparse[b, 10:] = row_tail

    # Progression permutation
    prog = progression.long()
    out_prog = prog.clone()
    m_dahai = (prog >= 5) & (prog < 597)
    if m_dahai.any():
        v = torch.where(m_dahai, prog - 5, 0)
        seat = v // 148
        rem = v % 148
        t37 = rem // 4
        ml = rem % 4
        out_prog = torch.where(
            m_dahai,
            5 + seat * 148 + torch.gather(p_t37, 1, t37) * 4 + ml,
            out_prog,
        )
    m_chi = (prog >= 597) & (prog < 957)
    if m_chi.any():
        v = torch.where(m_chi, prog - 597, 0)
        seat = v // 90
        code = v % 90
        out_prog = torch.where(
            m_chi,
            597 + seat * 90 + torch.gather(p_chi, 1, code),
            out_prog,
        )
    m_pon = (prog >= 957) & (prog < 1437)
    if m_pon.any():
        v = torch.where(m_pon, prog - 957, 0)
        seat = v // 120
        rem = v % 120
        rel_from = rem // 40
        code = rem % 40
        out_prog = torch.where(
            m_pon,
            957 + seat * 120 + rel_from * 40 + torch.gather(p_pon, 1, code),
            out_prog,
        )
    m_dmk = (prog >= 1437) & (prog < 1881)
    if m_dmk.any():
        v = torch.where(m_dmk, prog - 1437, 0)
        seat = v // 111
        rem = v % 111
        rel_from = rem // 37
        t37 = rem % 37
        out_prog = torch.where(
            m_dmk,
            1437 + seat * 111 + rel_from * 37 + torch.gather(p_t37, 1, t37),
            out_prog,
        )
    m_ankan = (prog >= 1881) & (prog < 2017)
    if m_ankan.any():
        v = torch.where(m_ankan, prog - 1881, 0)
        seat = v // 34
        b34 = v % 34
        out_prog = torch.where(
            m_ankan,
            1881 + seat * 34 + torch.gather(p_b34, 1, b34),
            out_prog,
        )
    m_kakan = (prog >= 2017) & (prog < 2165)
    if m_kakan.any():
        v = torch.where(m_kakan, prog - 2017, 0)
        seat = v // 37
        t37 = v % 37
        out_prog = torch.where(
            m_kakan,
            2017 + seat * 37 + torch.gather(p_t37, 1, t37),
            out_prog,
        )
    out_prog = out_prog.to(progression.dtype)

    # Candidates mapping + ascending re-sort + chosen_cand_idx realignment
    mapped_cands = torch.gather(p_act, 1, candidates.long())
    sorted_cands, sort_order = torch.sort(mapped_cands, dim=-1, stable=True)
    out_cands = sorted_cands.to(candidates.dtype)

    out_action = None
    if chosen_cand_idx is not None:
        inv_order = torch.empty_like(sort_order)
        inv_order.scatter_(
            1,
            sort_order,
            torch.arange(
                sort_order.shape[1], device=device, dtype=torch.long
            ).expand_as(sort_order),
        )
        out_action = torch.gather(
            inv_order, 1, chosen_cand_idx.long().unsqueeze(1)
        ).squeeze(1)
        out_action = out_action.to(chosen_cand_idx.dtype)

    if was_1d:
        out_sparse = out_sparse.squeeze(0)
        numeric = numeric.squeeze(0)
        out_prog = out_prog.squeeze(0)
        out_cands = out_cands.squeeze(0)
        if out_action is not None:
            out_action = out_action.squeeze(0)
    return out_sparse, numeric, out_prog, out_cands, out_action


def _gather_tile34_dim(t: torch.Tensor, inv_idx: torch.Tensor, dim: int) -> torch.Tensor:
    shape = [1] * t.dim()
    shape[0] = t.shape[0]
    shape[dim] = 34
    expand_idx = inv_idx.view(*shape).expand_as(t)
    return torch.gather(t, dim, expand_idx)


def permute_structured_batch(
    batch: Any,
    perm_idx: int | torch.Tensor,
) -> Any:
    """Apply S_3 suit permutation directly on precomputed structured_v2 tensors (§4.1.4)."""
    ref = batch["v2_tile_feats"] if "v2_tile_feats" in batch else batch["tile_feats"]
    bsz = ref.shape[0]
    device = ref.device

    if isinstance(perm_idx, int):
        p_vec = torch.full((bsz,), perm_idx, dtype=torch.long, device=device)
    else:
        p_vec = perm_idx.to(device=device, dtype=torch.long)

    inv_b34 = torch.from_numpy(INV_PERM_BASE34.copy()).to(device=device, dtype=torch.long)[p_vec, :34]
    p_t37 = torch.from_numpy(PERM_TILE37.copy()).to(device=device, dtype=torch.long)[p_vec]
    p_act = torch.from_numpy(PERM_ACTION_ID.copy()).to(device=device, dtype=torch.long)[p_vec]

    out = batch.clone(recurse=False) if not isinstance(batch, dict) and hasattr(batch, "clone") else dict(batch)
    if "next" in batch and batch["next"] is not None:
        out["next"] = permute_structured_batch(batch["next"], p_vec)

    for key in ("v2_tile_feats", "tile_feats", "oracle_tile", "v2_oracle_tile"):
        if key in out:
            out[key] = _gather_tile34_dim(out[key], inv_b34, dim=1)

    for key in ("opp_waits", "opp_hand_counts"):
        if key in out:
            out[key] = _gather_tile34_dim(out[key], inv_b34, dim=2)

    if "exact_dealin_now" in out:
        out["exact_dealin_now"] = _gather_tile34_dim(
            out["exact_dealin_now"], inv_b34, dim=1
        )

    for key in ("v2_events", "events", "v2_round_events", "round_events"):
        if key in out:
            ev = out[key].clone()
            ev_t = ev[..., 3:6].long()
            mapped = torch.gather(
                p_t37.unsqueeze(1).expand(-1, ev.shape[1], -1),
                2,
                ev_t,
            )
            ev[..., 3:6] = mapped.to(ev.dtype)
            out[key] = ev

    def _remap_cand_group(prefix: str, action_key: str | None = None) -> None:
        id_key = f"{prefix}cand_ids"
        if id_key not in out:
            if action_key and "candidates" in out:
                pass
            return
        cids = out[id_key]
        mapped = torch.gather(p_act, 1, cids.long())
        sorted_cids, order = torch.sort(mapped, dim=-1, stable=True)
        out[id_key] = sorted_cids.to(cids.dtype)
        for suffix in ("cand_mask", "cand_mode"):
            k = f"{prefix}{suffix}"
            if k in out:
                out[k] = torch.gather(out[k], 1, order)
        dyn_k = f"{prefix}cand_dyn"
        if dyn_k in out:
            order_exp = order.unsqueeze(-1).expand(-1, -1, out[dyn_k].shape[-1])
            out[dyn_k] = torch.gather(out[dyn_k], 1, order_exp)
        if action_key and action_key in out:
            act = out[action_key].long()
            inv_order = torch.empty_like(order)
            inv_order.scatter_(
                1,
                order,
                torch.arange(order.shape[1], device=device, dtype=torch.long).expand_as(order),
            )
            out[action_key] = torch.gather(inv_order, 1, act.unsqueeze(1)).squeeze(1).to(
                out[action_key].dtype
            )

    _remap_cand_group("v2_", action_key="action")
    if "cand_ids" in out and "v2_cand_ids" not in out:
        _remap_cand_group("", action_key="action")
    _remap_cand_group("next_v2_", action_key="next_action")
    if "next_cand_ids" in out and "next_v2_cand_ids" not in out:
        _remap_cand_group("next_", action_key="next_action")

    if "candidates" in out:
        cids = out["candidates"]
        mapped = torch.gather(p_act, 1, cids.long())
        sorted_cids, order = torch.sort(mapped, dim=-1, stable=True)
        out["candidates"] = sorted_cids.to(cids.dtype)
        if "action" in out and "v2_cand_ids" not in out and "cand_ids" not in out:
            act = out["action"].long()
            inv_order = torch.empty_like(order)
            inv_order.scatter_(
                1,
                order,
                torch.arange(order.shape[1], device=device, dtype=torch.long).expand_as(order),
            )
            out["action"] = torch.gather(inv_order, 1, act.unsqueeze(1)).squeeze(1).to(
                out["action"].dtype
            )

    return out


def derive_structured_v2_batch(
    sparse: torch.Tensor,
    numeric: torch.Tensor,
    progression: torch.Tensor,
    candidates: torch.Tensor,
    *,
    enable_early_shape_hint: bool = False,
    reward_weights: tuple[float, float, float, float] = (1.5, 0.5, -0.5, -1.5),
    include_room_level: bool = False,
) -> dict[str, torch.Tensor]:
    """Encode raw 4-field tensors into structured_v2 tensors via native C++ backend."""
    if include_room_level:
        raise ValueError("structured_v2 Phase 1 requires include_room_level=False")
    from kanachan import _candidate_features

    target_device = sparse.device
    sp_cpu = np.ascontiguousarray(sparse.detach().cpu().numpy(), dtype=np.int32)
    nu_cpu = np.ascontiguousarray(numeric.detach().cpu().numpy(), dtype=np.int32)
    pr_cpu = np.ascontiguousarray(progression.detach().cpu().numpy(), dtype=np.int32)
    ca_cpu = np.ascontiguousarray(candidates.detach().cpu().numpy(), dtype=np.int32)
    rw = tuple(float(x) for x in reward_weights)

    tile_b, ev_b, ev_len_b, glob_b, play_b, cmode_b, cdyn_b, lmax = (
        _candidate_features.encode_structured_v2_batch(
            sp_cpu,
            nu_cpu,
            pr_cpu,
            ca_cpu,
            bool(enable_early_shape_hint),
            rw,
        )
    )
    bsz = sp_cpu.shape[0]

    def _from_ba(buf: bytearray | bytes, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        if bsz == 0 or len(buf) == 0:
            return torch.empty(shape, dtype=dtype)
        return torch.frombuffer(buf, dtype=dtype).reshape(shape)

    tile_feats = _from_ba(tile_b, torch.float32, (bsz, 34, TILE_FEATS_DIM))
    events = _from_ba(ev_b, torch.uint8, (bsz, lmax, EVENT_BYTES))
    event_len = _from_ba(ev_len_b, torch.int32, (bsz,))
    global_feats = _from_ba(glob_b, torch.float32, (bsz, GLOBAL_FEATS_DIM))
    player_feats = _from_ba(play_b, torch.float32, (bsz, 4, PLAYER_FEATS_DIM))
    cand_mode = _from_ba(cmode_b, torch.uint8, (bsz, MAX_CANDIDATES))
    cand_dyn = _from_ba(cdyn_b, torch.float32, (bsz, MAX_CANDIDATES, CAND_DYN_DIM))
    cand_ids = torch.from_numpy(ca_cpu).clone()
    cand_mask = cand_ids != PAD_ACTION_ID

    out = {
        "v2_tile_feats": tile_feats,
        "v2_events": events,
        "v2_event_len": event_len,
        "v2_global_feats": global_feats,
        "v2_player_feats": player_feats,
        "v2_cand_ids": cand_ids,
        "v2_cand_mask": cand_mask,
        "v2_cand_mode": cand_mode,
        "v2_cand_dyn": cand_dyn,
    }
    if target_device.type != "cpu":
        out = {
            k: v.to(device=target_device, non_blocking=True) for k, v in out.items()
        }
    return out


__all__ = [
    "ACTION_FAMILY_NAMES",
    "ACTION_TYPE_NAMES",
    "CAND_DYN_DIM",
    "CAND_MODE_NAMES",
    "CHI_ENCODE_MAP",
    "CHI_TABLE",
    "EVENT_BYTES",
    "GLOBAL_FEATS_DIM",
    "HAS_SLOT_FLAGS",
    "INV_PERM_BASE34",
    "INV_PERM_TILE37",
    "KANACHAN_ACTION_MANIFEST",
    "KANACHAN_ACTION_MANIFEST_SHA256",
    "MAX_CANDIDATES",
    "MAX_EVENT_LEN",
    "MAX_PROGRESSION_LEN",
    "MAX_ROPE_LEN",
    "NULL_SOURCE_IDX",
    "NULL_TILE_IDX",
    "NUM_CAND_MODES",
    "NUM_TYPES_OF_ACTIONS",
    "OFFSETS_136",
    "PAD_ACTION_ID",
    "PERM_ACTION_ID",
    "PERM_BASE34",
    "PERM_CHI_ENCODE",
    "PERM_PENG_ENCODE",
    "PERM_TILE37",
    "PLAYER_FEATS_DIM",
    "PON_ENCODE_MAP",
    "PON_TABLE",
    "REL_BIAS_NUM_CLASSES",
    "STATIC_FLAGS",
    "STRUCTURED_V2_SCHEMA_ID",
    "SUIT_PERMS",
    "TILE37_NONE",
    "TILE_FEATS_DIM",
    "action_cand_mode",
    "action_family_id",
    "base34_to_tile37",
    "decode_action_tuple",
    "derive_structured_v2_batch",
    "encode_action_tuple",
    "idx136_to_tile37",
    "is_red5_tile37",
    "permute_raw_four_fields",
    "permute_structured_batch",
    "tile34",
]
