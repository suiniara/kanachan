"""Outcome supervision; future results are targets and never model inputs."""

from torch.nn import functional as F

from kanachan.constants import SEAT_INDEX, SEAT_OFFSET
from kanachan.nn.decision_decoder import require_tensor
from kanachan.training.core.ranking import scores_to_ranks


def final_rank_loss(logits, data):
    """Train only where the RL format actually carries final game scores.

    Normalize over decisions, including unlabeled rows, so accumulation and
    distributed gradient averaging use the same objective as a single batch.
    In particular, zero-filled intermediate results never mean first place.
    """
    terminal = data["next", "end_of_game"]
    scores = data["next", "results"][:, 10:14]
    seats = data["sparse"][:, SEAT_INDEX].long() - SEAT_OFFSET
    require_tensor(
        ((seats >= 0) & (seats < 4)).all(), "Invalid starting seat."
    )
    ranks = scores_to_ranks(scores)
    target = ranks.gather(1, seats[:, None]).squeeze(1)
    losses = F.cross_entropy(logits, target, reduction="none")
    return losses.masked_fill(~terminal, 0.0).mean()
