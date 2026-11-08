"""Validate complete annotation games without installing Torch."""

from collections import Counter

from kanachan import constants as c
from kanachan.training.core.features import validate_sparse_prefix


def integers(text):
    return [int(value) for value in text.split(",")]


def state(fields):
    sparse, numeric, progression, candidates = map(integers, fields)
    validate_sparse_prefix(sparse)
    for values, limit, maximum in (
        (
            sparse,
            c.MAX_NUM_ACTIVE_SPARSE_FEATURES,
            c.NUM_TYPES_OF_SPARSE_FEATURES,
        ),
        (
            progression,
            c.MAX_LENGTH_OF_PROGRESSION_FEATURES,
            c.NUM_TYPES_OF_PROGRESSION_FEATURES,
        ),
        (candidates, c.MAX_NUM_ACTION_CANDIDATES, c.NUM_TYPES_OF_ACTIONS),
    ):
        if (
            not values
            or len(values) > limit
            or any(value < 0 or value >= maximum for value in values)
        ):
            raise ValueError("Invalid feature lengths or values.")
    if len(numeric) != c.NUM_NUMERIC_FEATURES or min(numeric[:2]) < 0:
        raise ValueError("Invalid numeric features.")
    return sparse, numeric, progression, candidates


def grades(level_ids):
    mapping = {
        10000 + tier * 100 + star: (tier - 1) * 3 + star - 1
        for tier in range(1, 6)
        for star in range(1, 4)
    }
    mapping.update(
        {10601: 15, **{10700 + level: 15 for level in range(1, 21)}}
    )
    return [mapping[level] for level in level_ids]


def header(sparse, metadata, *, end_of_round=False):
    expected = [3 if metadata["mode_id"] == 12 else 4, 6]
    expected += [
        7 + 16 * seat + grade
        for seat, grade in enumerate(grades(metadata["level_ids"]))
    ]
    if sparse[:6] != expected:
        raise ValueError(
            "Annotation room/style/seat grades differ from raw metadata."
        )
    if end_of_round:
        validate_sparse_prefix(sparse, end_of_round=True)


def validate_bc(lines, metadata):
    if not lines:
        raise ValueError("No decision annotations.")
    seats = set()
    for line in lines:
        columns = line.rstrip("\r\n").split("\t")
        if len(columns) != 8 or columns[0] != metadata["uuid"]:
            raise ValueError("Invalid BC columns or UUID.")
        sparse, _, _, candidates = state(columns[1:5])
        header(sparse, metadata)
        if not 0 <= int(columns[5]) < len(candidates):
            raise ValueError("Illegal demonstrated action index.")
        summary, results = integers(columns[6]), integers(columns[7])
        if not 1 <= len(summary) <= c.MAX_NUM_ROUND_SUMMARY or any(
            not 0 <= value < c.NUM_TYPES_OF_ROUND_SUMMARY for value in summary
        ):
            raise ValueError("Invalid round summary.")
        if len(results) != c.NUM_RESULTS or results[-4:] != metadata["scores"]:
            raise ValueError("Final scores differ from the original game.")
        seats.add(sparse[6] - c.SEAT_OFFSET)
    # A genuine early game end can leave a seat with no decision at all.
    return seats


def validate_rl(lines, metadata, expected_rows, expected_seats):
    terminal = Counter()
    rows = 0
    expected_next = None
    for line in lines:
        columns = line.rstrip("\r\n").split("\t")
        if len(columns) not in (8, 10, 12) or columns[0] != metadata["uuid"]:
            raise ValueError("Invalid RL columns or UUID.")
        sparse, _, _, candidates = state(columns[1:5])
        if expected_next is not None and tuple(columns[1:5]) != expected_next:
            raise ValueError(
                "RL next-state features do not match the next decision."
            )
        header(sparse, metadata)
        if not 0 <= int(columns[5]) < len(candidates):
            raise ValueError("Illegal RL action index.")
        seat = sparse[6] - c.SEAT_OFFSET
        if len(columns) == 8:
            result = integers(columns[-1])
            if (
                len(result) != c.NUM_RESULTS
                or result[-4:] != metadata["scores"]
            ):
                raise ValueError(
                    "RL terminal scores differ from raw metadata."
                )
            terminal[seat] += 1
            expected_next = None
        else:
            next_sparse, _, _, _ = state(columns[6:10])
            header(next_sparse, metadata)
            if sparse[6] != next_sparse[6]:
                raise ValueError("RL transition crossed player seats.")
            if len(columns) == 12 and len(integers(columns[-1])) != 10:
                raise ValueError("Invalid RL round-result shape.")
            expected_next = tuple(columns[6:10])
        rows += 1
    if (
        expected_next is not None
        or rows != expected_rows
        or dict(terminal) != {seat: 1 for seat in expected_seats}
    ):
        raise ValueError(
            "RL conversion lost rows or terminal player trajectories."
        )
    return {"rows": rows, "terminal": sum(terminal.values())}


def validate_eor(lines, metadata):
    rows = 0
    for line in lines:
        columns = line.rstrip("\r\n").split("\t")
        if len(columns) != 4 or columns[0] != metadata["uuid"]:
            raise ValueError("Invalid EOR columns or UUID.")
        sparse, numeric, results = map(integers, columns[1:])
        header(sparse, metadata, end_of_round=True)
        if (
            len(sparse) != c.EOR_NUM_SPARSE_FEATURES
            or len(numeric) != c.EOR_NUM_NUMERIC_FEATURES
            or results != metadata["scores"]
        ):
            raise ValueError("Invalid EOR shape or final scores.")
        rows += 1
    return rows
