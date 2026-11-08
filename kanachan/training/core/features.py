"""Validate fixed v2 feature positions before embedding or reward lookup."""


def validate_sparse_prefix(features, *, end_of_round: bool = False) -> None:
    ranges = [(0, 5), (5, 7), (7, 23), (23, 39), (39, 55), (55, 71)]
    ranges += (
        [(71, 74), (74, 78)]
        if end_of_round
        else [(71, 75), (75, 78), (78, 82), (82, 152)]
    )
    if len(features) < len(ranges):
        raise RuntimeError(
            f"Expected at least {len(ranges)} fixed v2 sparse features."
        )
    for position, (value, (lower, upper)) in enumerate(
        zip(features, ranges, strict=False)
    ):
        if not lower <= value < upper:
            raise RuntimeError(
                f"Invalid v2 sparse prefix at position {position}: {value}; "
                f"expected [{lower}, {upper}). Use the matching annotation version."
            )
