"""Observed round/placement rewards and legacy optional potential shaping."""

from pathlib import Path
import math

import torch
from torch import nn
from tensordict import TensorDict

from kanachan.constants import SEAT_INDEX, SEAT_OFFSET
from kanachan.model_loader import load_model
from kanachan.training.core.eor.feature_adaptor import FeatureAdaptor
from kanachan.training.core.ranking import scores_to_ranks


class PlacementReward:
    """Reward = 2.5 - final rank; intermediate rewards telescope exactly.

    An optional EOR model supplies a public-state potential at round starts.
    It must export ranking_probs or decode logits with shape (N, 4, 4).
    This is a shaping heuristic, not access to opponents' private hands.
    Terminal potential is zero. The first-state potential is subtracted
    from the episode return, leaving the policy's objective unchanged.
    """

    def __init__(self, shaping_model: Path | nn.Module | None = None):
        if isinstance(shaping_model, (str, Path)):
            shaping_model = load_model(
                Path(shaping_model), map_location=torch.device("cpu")
            )
        self.model = shaping_model
        if self.model is not None:
            self.model.cpu().float().eval().requires_grad_(False)
        self.adaptor = FeatureAdaptor()

    @torch.no_grad()
    def __call__(self, episode: TensorDict, contiguous: bool) -> None:
        terminal = episode["next", "end_of_game"].bool()
        count = int(episode.size(0))
        if count == 0:
            raise ValueError("An empty episode has no placement reward.")
        seats = episode["sparse"][:, SEAT_INDEX].long() - SEAT_OFFSET
        if ((seats < 0) | (seats >= 4)).any():
            raise ValueError("Invalid starting seat.")
        reward = torch.zeros(
            count, dtype=torch.float32, device=terminal.device
        )
        if terminal.any():
            final_scores = episode["next", "results"][terminal, 10:14]
            ranks = scores_to_ranks(final_scores)
            own_rank = ranks.gather(1, seats[terminal, None]).squeeze(1)
            reward[terminal] = 1.5 - own_rank.float()
        if self.model is not None:
            if not contiguous or not terminal[-1]:
                raise ValueError(
                    "Potential shaping needs complete, contiguous games."
                )
            starts = episode["next", "end_of_round"].roll(
                1
            ).bool() | terminal.roll(1)
            starts[0] = True
            indices = starts.nonzero().flatten()
            sparse, numeric = self.adaptor(
                episode["sparse"][indices].cpu(),
                episode["numeric"][indices].cpu(),
            )
            td = TensorDict(
                {"sparse": sparse, "numeric": numeric}, [len(indices)]
            )
            self.model(td)
            probabilities = td.get("ranking_probs", None)
            if probabilities is None:
                logits = td.get("decode", None)
                if logits is None:
                    raise ValueError(
                        "EOR model must output ranking_probs or decode."
                    )
                probabilities = logits.softmax(-1)
            if (
                probabilities.shape != (len(indices), 4, 4)
                or not torch.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not torch.allclose(
                    probabilities.sum(-1),
                    torch.ones(len(indices), 4),
                    atol=1e-5,
                )
            ):
                raise ValueError("Invalid EOR rank probabilities.")
            utilities = torch.tensor([1.5, 0.5, -0.5, -1.5])
            predicted = (probabilities @ utilities)[
                torch.arange(len(indices)), seats[indices].cpu()
            ]
            segment = starts.long().cumsum(0) - 1
            potential = predicted.to(reward.device)[segment]
            next_potential = potential.roll(-1).masked_fill(terminal, 0.0)
            reward += next_potential - potential
        episode["next", "done"] = terminal.clone()
        episode["next", "reward"] = reward


class RoundAndPlacementReward:
    """Settled own-score change per round plus final half-game placement.

    Score changes are divided by ``score_unit`` and scaled by ``round_weight``.
    This deliberately optimizes a
    combination of score and placement; it is not policy-invariant shaping.
    Complete observer trajectories are required before transitions are shuffled.

    Differences between consecutive settlement snapshots include riichi costs,
    deposits, draw payments and intervening rounds without an observer decision.
    On the last transition, actual final scores replace the last observed round
    settlement, including games ending before this seat acts in the final round.
    """

    def __init__(self, *, score_unit=10000.0, round_weight=0.2):
        if not math.isfinite(score_unit) or score_unit <= 0:
            raise ValueError("round score_unit must be finite and positive.")
        if not math.isfinite(round_weight) or round_weight < 0:
            raise ValueError("round_weight must be finite and nonnegative.")
        self.score_unit = score_unit
        self.round_weight = round_weight
        self.placement = PlacementReward()

    @torch.no_grad()
    def __call__(self, episode: TensorDict, contiguous: bool) -> None:
        terminal = episode["next", "end_of_game"].bool()
        if not contiguous or not len(episode) or not bool(terminal[-1]):
            raise ValueError("Round rewards require complete, contiguous observer games.")
        boundary = episode["next", "end_of_round"].bool()
        if bool((terminal & ~boundary).any()):
            raise ValueError("A terminal game must also end a round.")
        indices = boundary.nonzero().flatten()
        results = episode["next", "results"][indices]
        seats = episode["sparse"][indices, SEAT_INDEX].long() - SEAT_OFFSET
        if bool(((seats < 0) | (seats >= 4)).any()):
            raise ValueError("Invalid starting seat.")
        terminal_boundaries = terminal[indices]
        first = terminal_boundaries.roll(1)
        first[0] = True
        if bool(((seats != seats.roll(1)) & ~first).any()):
            raise ValueError("An observer seat changed within a game.")
        settled = torch.where(
            terminal_boundaries[:, None], results[:, 10:14], results[:, 6:10]
        ).to(torch.float32)
        previous = settled.roll(1, 0)
        # The authoritative annotation contains settlement minus round-start
        # delta, not a possibly post-riichi current-decision score snapshot.
        previous[first] = (results[first, 6:10] - results[first, :4]).float()
        own_delta = (settled - previous).gather(1, seats[:, None]).squeeze(1)
        self.placement(episode, contiguous)
        episode["next", "reward"][indices] += own_delta * (self.round_weight / self.score_unit)


def configured_reward(scheme="placement", *, shaping_model=None,
                      score_unit=10000.0, round_weight=0.2):
    if scheme == "placement":
        return PlacementReward(shaping_model)
    if scheme == "round_and_placement":
        if shaping_model is not None:
            raise ValueError("Round rewards and EOR potential shaping are separate recipes.")
        return RoundAndPlacementReward(score_unit=score_unit, round_weight=round_weight)
    raise ValueError(f"Unknown reward scheme: {scheme}")
