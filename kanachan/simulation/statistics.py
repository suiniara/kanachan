"""Descriptive simulation statistics; undefined sample variances stay null."""

from statistics import mean, variance
import math


def _statistic(values):
    return {
        "average": float(mean(values)),
        "variance": float(variance(values)) if len(values) > 1 else None,
    }


def _cluster_values(game_results, group_ids):
    if len(group_ids) != len(game_results):
        raise ValueError("One group ID is required per game.")
    groups = {}
    for game, group_id in zip(game_results, group_ids, strict=True):
        values = groups.setdefault(group_id, [[], []])
        for player in game:
            values[bool(player["proposed"])].append(player["ranking"] + 1)
    counts = {tuple(map(len, values)) for values in groups.values()}
    if len(counts) != 1 or any(n == 0 for count in counts for n in count):
        raise ValueError(
            "Each group must have the same nonzero policy sample counts."
        )
    return {
        key: [
            mean(values[1]),
            mean(x == 4 for x in values[1]),
            mean(values[0]) - mean(values[1]),
            mean(x == 4 for x in values[0]) - mean(x == 4 for x in values[1]),
        ]
        for key, values in groups.items()
    }


def _intervals(values, *, bootstrap_samples, seed, confidence):
    """Percentile bootstrap of independent groups, never individual seats."""
    if not 0 < confidence < 1 or bootstrap_samples < 100:
        raise ValueError(
            "Use confidence in (0, 1) and at least 100 bootstrap samples."
        )
    width = len(values[0])
    if len(values) < 2:
        return [None] * width
    import torch

    data = torch.tensor(values, dtype=torch.float64)
    if not torch.isfinite(data).all():
        raise ValueError("Nonfinite statistics.")
    generator = torch.Generator().manual_seed(seed)
    samples = []
    chunk_size = max(1, min(64, 2_000_000 // data.numel()))
    for start in range(0, bootstrap_samples, chunk_size):
        indices = torch.randint(
            len(values),
            (min(chunk_size, bootstrap_samples - start), len(values)),
            generator=generator,
        )
        samples.append(data[indices].mean(1))
    means = torch.cat(samples)
    tail = (1 - confidence) / 2
    bounds = torch.quantile(
        means, torch.tensor([tail, 1 - tail], dtype=torch.float64), dim=0
    )
    return bounds.T.tolist()


def summarize(
    game_results,
    *,
    group_ids=None,
    bootstrap_samples=2000,
    seed=0,
    confidence=0.95,
):
    if not game_results:
        raise ValueError("No simulation results.")
    grouped = [[], []]
    differences = []
    for game in game_results:
        if len(game) != 4 or sorted(player["ranking"] for player in game) != [
            0,
            1,
            2,
            3,
        ]:
            raise ValueError("Each game must have four distinct ranks.")
        groups = [
            [player for player in game if bool(player["proposed"]) == proposed]
            for proposed in (False, True)
        ]
        if not all(groups):
            raise ValueError("Each game must contain both policies.")
        for players, all_players in zip(groups, grouped, strict=True):
            all_players.extend(players)
        differences.append(
            mean(player["ranking"] for player in groups[0])
            - mean(player["ranking"] for player in groups[1])
        )

    def policy_statistics(players):
        ranks = [player["ranking"] for player in players]
        return {
            "num_samples": len(players),
            "ranking": _statistic([rank + 1 for rank in ranks]),
            "grading_point": _statistic(
                [
                    [125, 60, -5, -255][player["ranking"]]
                    + (int(player["score"]) - 25000) // 1000
                    for player in players
                ]
            ),
            "soul_point": _statistic(
                [[0.5, 0.2, -0.2, -0.5][rank] for rank in ranks]
            ),
            "top_rate": _statistic([float(rank == 0) for rank in ranks]),
            "quinella_rate": _statistic([float(rank <= 1) for rank in ranks]),
            "fourth_rate": _statistic([float(rank == 3) for rank in ranks]),
        }

    result = {
        "num_games": len(game_results),
        "baseline": policy_statistics(grouped[0]),
        "proposed": policy_statistics(grouped[1]),
        "difference": {"ranking": _statistic(differences)},
    }
    if group_ids is not None:
        values = _cluster_values(game_results, group_ids)
        intervals = _intervals(
            list(values.values()),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            confidence=confidence,
        )
        result["uncertainty"] = {
            "method": "cluster_percentile_bootstrap",
            "num_groups": len(values),
            "confidence": confidence,
            **dict(
                zip(
                    (
                        "proposed_ranking",
                        "proposed_fourth_rate",
                        "ranking_difference",
                        "fourth_rate_difference",
                    ),
                    intervals,
                    strict=True,
                )
            ),
        }
    return result


def compare_policies(
    candidate,
    incumbent,
    *,
    group_ids,
    bootstrap_samples=2000,
    seed=0,
    confidence=0.95,
):
    """Paired candidate-minus-incumbent differences against one fixed baseline.

    Both evaluations must use identical wall seeds and seat arrangements.
    Negative differences favor the candidate. Groups are independent seeds.
    """
    # Reuse rank validation before calculating the paired statistics.
    summarize(candidate)
    summarize(incumbent)
    left = _cluster_values(candidate, group_ids)
    right = _cluster_values(incumbent, group_ids)
    values = [[left[key][i] - right[key][i] for i in (0, 1)] for key in left]
    intervals = _intervals(
        values,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        confidence=confidence,
    )
    return {
        "num_groups": len(values),
        "confidence": confidence,
        "method": "paired_cluster_percentile_bootstrap",
        "ranking": {
            "average": mean(x[0] for x in values),
            "interval": intervals[0],
        },
        "fourth_rate": {
            "average": mean(x[1] for x in values),
            "interval": intervals[1],
        },
    }


def promotion_decision(
    comparison,
    *,
    min_groups=32,
    min_improvement=0.0,
    max_fourth_rate_increase=0.0,
):
    if (
        min_groups < 2
        or not math.isfinite(min_improvement)
        or min_improvement < 0
        or not math.isfinite(max_fourth_rate_increase)
        or not 0 <= max_fourth_rate_increase <= 1
    ):
        raise ValueError("Invalid promotion thresholds.")
    rank = comparison["ranking"]["interval"]
    fourth = comparison["fourth_rate"]["interval"]
    return bool(
        comparison["num_groups"] >= min_groups
        and rank is not None
        and fourth is not None
        and rank[1] < -min_improvement
        and fourth[1] <= max_fourth_rate_increase
    )
