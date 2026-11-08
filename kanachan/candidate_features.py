"""One public-token derivation for offline data and native-model inference."""

from bisect import bisect_right
from functools import lru_cache

import torch

from kanachan.constants import NUM_TYPES_OF_ACTIONS
from kanachan.training.core.features import validate_sparse_prefix


SCHEMA_ID = "discard_afterstate_v1"
FEATURE_WIDTH = 46
OFFSETS_136 = (
    0,
    1,
    5,
    9,
    13,
    17,
    20,
    24,
    28,
    32,
    36,
    37,
    41,
    45,
    49,
    53,
    56,
    60,
    64,
    68,
    72,
    73,
    77,
    81,
    85,
    89,
    92,
    96,
    100,
    104,
    108,
    112,
    116,
    120,
    124,
    128,
    132,
)


def tile34(tile):
    if not 0 <= tile < 37:
        raise ValueError("Invalid 37-tile ID.")
    return (
        tile // 10 * 9 + (4 if tile % 10 == 0 else tile % 10 - 1)
        if tile < 30
        else tile - 3
    )


def _tile37(tile):
    return tile // 9 * 10 + tile % 9 + 1 if tile < 27 else tile + 3


def _chi_table():
    rows = []
    for suit in range(3):
        for called in range(1, 10):
            for start in range(max(1, called - 2), min(7, called) + 1):
                consumed = [
                    tile for tile in range(start, start + 3) if tile != called
                ]
                row = (*consumed, called)
                rows.append(tuple(suit * 10 + tile for tile in row))
                if start <= 5 < start + 3:
                    rows.append(
                        tuple(
                            suit * 10 + (0 if tile == 5 else tile)
                            for tile in row
                        )
                    )
    return tuple(rows)


CHI = _chi_table()
PON = tuple(
    row
    for tile in range(34)
    for row in (
        (
            (_tile37(tile),) * 3,
            (_tile37(tile) - 5, _tile37(tile), _tile37(tile)),
            (_tile37(tile), _tile37(tile), _tile37(tile) - 5),
        )
        if tile in (4, 13, 22)
        else ((_tile37(tile),) * 3,)
    )
)


def candidate_feature_schema():
    return {
        "id": SCHEMA_ID,
        "width": FEATURE_WIDTH,
        "source": "v2_observer_sparse_and_public_round_progression",
        "columns": [
            "discard_valid",
            "shanten_div_8",
            "structural_tenpai",
            "live_ukeire_div_136",
            "live_types_div_34",
        ]
        + [f"live_effective_{i}" for i in range(34)]
        + [
            "concealed_red_man",
            "concealed_red_pin",
            "concealed_red_sou",
            "concealed_dora_div_70",
            "discarded_red",
            "discarded_dora_factor_div_5",
            "concealed_count_div_13",
        ],
        "unsupported": "zero_features_with_discard_valid_0",
        "visibility": "own_concealed_draw_indicators_discards_and_meld_consumed_tiles_no_double_counting_called_tiles",
    }


@lru_cache(maxsize=1)
def native_backend():
    try:
        from kanachan import _candidate_features
    except ImportError as error:
        raise RuntimeError(
            "discard_v1 needs the native candidate feature module; build tools/features with the project's Nyanten headers."
        ) from error
    if (
        _candidate_features.SCHEMA_ID != SCHEMA_ID
        or _candidate_features.FEATURE_WIDTH != FEATURE_WIDTH
    ):
        raise ValueError(
            "Native candidate-feature schema differs from the Python runtime."
        )
    return _candidate_features


def observed_tiles(sparse, progression):
    """Decode only the current observer's hand and public tile revelations."""
    try:
        validate_sparse_prefix(sparse)
    except RuntimeError as error:
        raise ValueError(
            "Incomplete or invalid observer sparse prefix."
        ) from error
    seat = sparse[6] - 71
    hand = [0] * 37
    visible = [0] * 37
    dora = [0] * 34
    seen_hand, seen_slots = set(), set()
    draw = None
    for token in sparse[10:]:
        if 152 <= token < 337:
            slot, indicator = divmod(token - 152, 37)
            if slot in seen_slots:
                raise ValueError("Duplicate dora-indicator slot.")
            seen_slots.add(slot)
            visible[indicator] += 1
            indicator = tile34(indicator)
            target = (
                (indicator // 9 * 9 + (indicator % 9 + 1) % 9)
                if indicator < 27
                else (
                    27 + (indicator - 27 + 1) % 4
                    if indicator < 31
                    else 31 + (indicator - 31 + 1) % 3
                )
            )
            dora[target] += 1
        elif 337 <= token < 473:
            if token in seen_hand:
                raise ValueError("Duplicate concealed tile-count token.")
            seen_hand.add(token)
            hand[bisect_right(OFFSETS_136, token - 337) - 1] += 1
        elif 473 <= token < 510:
            if draw is not None:
                raise ValueError("More than one drawn tile token.")
            draw = token - 473
            hand[draw] += 1
        elif token != 510:
            raise ValueError("Unknown observer sparse token.")
    if not seen_slots or seen_slots != set(range(len(seen_slots))):
        raise ValueError("Missing or non-contiguous dora-indicator slots.")
    for tile, count in enumerate(hand):
        visible[tile] += count
    melds = 0
    if not progression or progression[0] != 0:
        raise ValueError(
            "Candidate features require complete public round history."
        )
    for event in progression:
        if event < 5:
            continue
        if event < 597:
            visible[(event - 5) % 148 // 4] += 1
        elif event < 957:
            owner, code = divmod(event - 597, 90)
            for tile in CHI[code][:2]:
                visible[tile] += 1
            melds += owner == seat
        elif event < 1437:
            owner, code = divmod(event - 957, 120)
            for tile in PON[code % 40][:2]:
                visible[tile] += 1
            melds += owner == seat
        elif event < 1881:
            owner, code = divmod(event - 1437, 111)
            called = code % 37
            for tile in range(37):
                if tile34(tile) == tile34(called):
                    count = (
                        1
                        if tile < 30 and tile % 10 == 0
                        else 3
                        if tile < 30 and tile % 10 == 5
                        else 4
                    )
                    visible[tile] += count - (tile == called)
            melds += owner == seat
        elif event < 2017:
            owner, tile = divmod(event - 1881, 34)
            normal = _tile37(tile)
            if tile in (4, 13, 22):
                visible[normal - 5] += 1
                visible[normal] += 3
            else:
                visible[normal] += 4
            melds += owner == seat
        elif event < 2165:
            visible[(event - 2017) % 37] += 1
        elif event != 2165:
            raise ValueError("Unknown public progression event.")
    counts = [0] * 34
    for tile, count in enumerate(visible):
        limit = (
            1
            if tile < 30 and tile % 10 == 0
            else 3
            if tile < 30 and tile % 10 == 5
            else 4
        )
        if not 0 <= count <= limit:
            raise ValueError(
                "Public observations contain an impossible tile multiplicity."
            )
        counts[tile34(tile)] += count
    return hand, counts, dora, melds


@lru_cache(maxsize=4096)
def _cached_row(schema, sparse, progression, actions):
    if schema != SCHEMA_ID:
        raise ValueError("Unknown candidate-feature cache schema.")
    if not any(action < 148 for action in actions):
        return torch.zeros((len(actions), FEATURE_WIDTH), dtype=torch.float32)
    hand, visible, dora, melds = observed_tiles(sparse, progression)
    packed = native_backend().discard_features(
        hand, visible, dora, actions, melds
    )
    return torch.frombuffer(bytearray(packed), dtype=torch.float32).reshape(
        len(actions), FEATURE_WIDTH
    )


@torch.compiler.disable
def derive_candidate_features(sparse, progression, candidates):
    """Bounded canonical CPU cache; return independent rows in native order.

    Raw parsed-batch caches stay valid: no derived feature is written into
    them. The versioned derivation and head schema are bound in checkpoints.
    """
    device = candidates.device
    sparse_rows = sparse.detach().cpu().tolist()
    histories = progression.detach().cpu().tolist()
    action_rows = candidates.detach().cpu().tolist()
    result = []
    for hand, history, original in zip(sparse_rows, histories, action_rows):
        actions = tuple(
            sorted(
                action for action in original if action < NUM_TYPES_OF_ACTIONS
            )
        )
        if len(actions) != len(set(actions)) or any(
            action < 0 or action > NUM_TYPES_OF_ACTIONS for action in original
        ):
            raise ValueError("Invalid or duplicate candidate IDs.")
        cached = _cached_row(
            SCHEMA_ID,
            tuple(sorted(token for token in hand if token != 510)),
            tuple(event for event in history if event != 2165),
            actions,
        )
        mapping = {action: i for i, action in enumerate(actions)}
        row = torch.zeros((len(original), FEATURE_WIDTH), dtype=torch.float32)
        for i, action in enumerate(original):
            if action != NUM_TYPES_OF_ACTIONS:
                row[i] = cached[mapping[action]]
        result.append(row)
    return torch.stack(result).to(device=device)
