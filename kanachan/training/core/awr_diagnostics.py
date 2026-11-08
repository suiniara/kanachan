"""Observe AWR sample concentration without changing its policy objective."""

import json
import math
from pathlib import Path

import torch
from torch import distributed

from kanachan.constants import SEAT_INDEX, SEAT_OFFSET
from kanachan.training.core.ranking import scores_to_ranks


ACTION_GROUPS = (
    "discard",
    "riichi_discard",
    "closed_kan",
    "added_kan",
    "tsumo",
    "abortive_draw",
    "pass",
    "chi",
    "pon",
    "open_kan",
    "ron",
)


def _gather(rows):
    if not distributed.is_initialized() or distributed.get_world_size() == 1:
        return rows.cpu()
    size = torch.tensor([len(rows)], dtype=torch.int64, device=rows.device)
    sizes = [
        torch.empty_like(size) for _ in range(distributed.get_world_size())
    ]
    distributed.all_gather(sizes, size)
    sizes = [int(item.item()) for item in sizes]
    padded = torch.zeros(
        (max(sizes), rows.shape[1]), dtype=rows.dtype, device=rows.device
    )
    padded[: len(rows)] = rows
    gathered = [torch.empty_like(padded) for _ in sizes]
    distributed.all_gather(gathered, padded)
    return torch.cat(
        [value[:size] for value, size in zip(gathered, sizes)]
    ).cpu()


def _action_groups(ids):
    groups = ids.remainder(2)
    for group, (lower, upper) in enumerate(
        (
            (148, 182),
            (182, 219),
            (219, 220),
            (220, 221),
            (221, 222),
            (222, 312),
            (312, 432),
            (432, 543),
            (543, 546),
        ),
        start=2,
    ):
        groups = torch.where((ids >= lower) & (ids < upper), group, groups)
    return groups


@torch.no_grad()
def weight_statistics(advantage, weight, data, *, beta, max_weight):
    """Gather observations first, then compute true global distribution metrics.

    Groups use only public state: actual logged action ID, round stage,
    current scores/rank and opponents' publicly declared riichi.
    """
    if (
        not math.isfinite(beta)
        or beta <= 0
        or not math.isfinite(max_weight)
        or max_weight <= 0
    ):
        raise ValueError(
            "AWR temperature and clipping weight must be finite and positive."
        )
    batch = len(weight)
    device = weight.device
    ids = (
        data["candidates"].gather(1, data["action"].long()[:, None]).squeeze(1)
    )
    seats = data["sparse"][:, SEAT_INDEX].long() - SEAT_OFFSET
    scores = data["numeric"][:, 2:6]
    own_score = scores.gather(1, seats[:, None]).squeeze(1)
    current_rank = scores_to_ranks(scores).gather(1, seats[:, None]).squeeze(1)
    round_id = (data["sparse"][:, 7] - 75) * 4 + data["sparse"][:, 8] - 78
    phase = torch.where(round_id < 4, 0, torch.where(round_id < 7, 1, 2))
    score_band = torch.where(
        own_score < 20000, 0, torch.where(own_score < 30000, 1, 2)
    )
    history = data["progression"]
    discard = (history >= 5) & (history < 597)
    owner = (history - 5) // 148
    declared = discard & ((history - 5).remainder(2) == 1)
    opponents = torch.zeros(batch, dtype=torch.long, device=device)
    for seat in range(4):
        opponents += (
            (declared & (owner == seat)).any(-1) & (seats != seat)
        ).long()
    rows = torch.stack(
        (
            advantage.detach().double(),
            weight.detach().double(),
            _action_groups(ids).double(),
            phase.double(),
            score_band.double(),
            current_rank.double(),
            opponents.double(),
        ),
        -1,
    )
    rows = _gather(rows)
    if not len(rows):
        raise ValueError("AWR diagnostics require at least one global sample.")
    advantage, weight = rows[:, 0], rows[:, 1]
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("AWR weights must be finite and nonnegative.")
    finite = torch.isfinite(advantage)
    clipped = advantage / beta > math.log(max_weight)
    total, squared = weight.sum(), weight.square().sum()
    ess = (
        (total.square() / squared).clamp(max=float(len(rows)))
        if squared > 0
        else torch.tensor(0.0)
    )
    metrics = {
        "Samples": float(len(rows)),
        "Weight/Mean": weight.mean().item(),
        "Weight/Sum": total.item(),
        "Weight/ESS": ess.item(),
        "Weight/ESS fraction": (ess / len(rows)).item(),
        "Weight/Clipped fraction": clipped.double().mean().item(),
        "Weight/Zero fraction": (weight == 0).double().mean().item(),
        "Advantage/Nonfinite fraction": (~finite).double().mean().item(),
        "Weight/Largest sample share": (weight.max() / total).item()
        if total > 0
        else 0.0,
        "Weight/Top one percent share": (
            weight.topk(max(1, math.ceil(len(rows) * 0.01))).values.sum()
            / total
        ).item()
        if total > 0
        else 0.0,
    }
    for name, values in (("Advantage", advantage[finite]), ("Weight", weight)):
        for label, quantile in (
            ("Min", 0.0),
            ("P05", 0.05),
            ("P50", 0.5),
            ("P95", 0.95),
            ("P99", 0.99),
            ("Max", 1.0),
        ):
            metrics[f"{name}/{label}"] = (
                torch.quantile(values, quantile).item() if len(values) else 0.0
            )
    for column, namespace, names in (
        (2, "Action", ACTION_GROUPS),
        (3, "Phase", ("east", "south_1_3", "south_4_or_later")),
        (4, "Score", ("below_20000", "20000_to_29999", "30000_or_more")),
        (5, "Current rank", ("first", "second", "third", "fourth")),
        (6, "Opponent riichi", ("0", "1", "2", "3")),
    ):
        for group, name in enumerate(names):
            mask = rows[:, column] == group
            selected = weight[mask]
            prefix = f"{namespace}/{name}"
            metrics[f"{prefix}/Sample fraction"] = mask.double().mean().item()
            metrics[f"{prefix}/Weight share"] = (
                (selected.sum() / total).item() if total > 0 else 0.0
            )
            metrics[f"{prefix}/Clipped fraction"] = (
                clipped[mask].double().mean().item() if mask.any() else 0.0
            )
    return metrics


def record_statistics(metrics, writer, samples, path=None):
    for name, value in metrics.items():
        writer.add_scalar(f"AWR/{name}", value, samples)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "samples": samples,
                        "group_schema": "public_mahjong_v1",
                        "metrics": metrics,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
