import torch
from torch import Tensor


def scores_to_ranks(scores: Tensor) -> Tensor:
    """Return zero-based ranks, resolving tied scores by starting seat."""
    if scores.ndim != 2 or scores.size(1) != 4:
        raise ValueError("Expected four final scores per game.")
    seats = torch.arange(4, device=scores.device)
    own_score = scores.unsqueeze(2)
    other_score = scores.unsqueeze(1)
    earlier_seat = seats.unsqueeze(0) < seats.unsqueeze(1)
    ahead = (other_score > own_score) | (
        (other_score == own_score) & earlier_seat
    )
    return ahead.sum(2)
