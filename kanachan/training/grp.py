"""20-dim seat-equivariant Pure-State GRP potential & PBRS hanchan shaping (§4.3)."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_RANK_PAYOFFS: tuple[float, float, float, float] = (1.5, 0.5, -0.5, -1.5)
GRP_INPUT_DIM = 20


def canonicalize_grp_state(
    sparse: Tensor,
    numeric: Tensor,
    reward_weights: Tensor | Sequence[float] = DEFAULT_RANK_PAYOFFS,
) -> Tensor:
    """Build 20-dim seat-equivariant state vector from round-start 4 fields (§4.3.1).

    Features per sample [20]:
      [0..3]:   relative scores (score[(i + r) % 4] - 25000) / 25000 for r=0..3
      [4..7]:   relative dealer one-hot (oya_rel = (ju + 4 - i) % 4)
      [8]:      clamp(chang, 0, 2) / 2.0
      [9..12]:  round index `ju % 4` one-hot (distinguishes East/South 1..4!)
      [13..14]: min(ben, 8) / 8.0, min(liqibang, 8) / 8.0
      [15]:     is_tonpuusen (sparse[:, 1] == 5)
      [16..19]: target rank payoffs w(R) = (w1, w2, w3, w4)
    Never reads round deltas or post-round results!
    """
    squeeze = sparse.ndim == 1
    if squeeze:
        sparse = sparse.unsqueeze(0)
        numeric = numeric.unsqueeze(0)

    bsz = sparse.shape[0]
    device = sparse.device
    seat = sparse[:, 6].long() - 71  # 0..3
    chang = sparse[:, 7].long() - 75  # 0..2
    ju = sparse[:, 8].long() - 78  # 0..3
    is_tonpuusen = (sparse[:, 1] == 5).float()

    ben = numeric[:, 0].float()
    liqibang = numeric[:, 1].float()
    scores = numeric[:, 2:6].float()

    out = torch.zeros((bsz, GRP_INPUT_DIM), dtype=torch.float32, device=device)

    rel_offsets = torch.arange(4, device=device, dtype=torch.long).unsqueeze(0)
    abs_seats = (seat.unsqueeze(1) + rel_offsets) % 4
    rel_scores = torch.gather(scores, 1, abs_seats)
    out[:, 0:4] = (rel_scores - 25000.0) / 25000.0

    oya_rel = (ju + 4 - seat) % 4
    out[:, 4:8] = F.one_hot(oya_rel, num_classes=4).float()
    out[:, 8] = torch.clamp(chang.float(), min=0.0, max=2.0) / 2.0
    out[:, 9:13] = F.one_hot(ju % 4, num_classes=4).float()
    out[:, 13] = torch.clamp(ben, min=0.0, max=8.0) / 8.0
    out[:, 14] = torch.clamp(liqibang, min=0.0, max=8.0) / 8.0
    out[:, 15] = is_tonpuusen

    rw = torch.as_tensor(reward_weights, dtype=torch.float32, device=device)
    if rw.ndim == 1:
        rw = rw.unsqueeze(0).expand(bsz, 4)
    out[:, 16:20] = rw

    return out.squeeze(0) if squeeze else out


class PureStateGRP(nn.Module):
    """MLP Linear(20,128) -> SiLU -> Linear(128,128) -> SiLU -> Linear(128,4)."""

    def __init__(self, *, rank_payoffs: Sequence[float] = DEFAULT_RANK_PAYOFFS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GRP_INPUT_DIM, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
        )
        self.register_buffer(
            "rank_payoffs",
            torch.tensor(rank_payoffs, dtype=torch.float32),
        )
        self.ruleset_sha256: str = hashlib.sha256(
            json.dumps([float(x) for x in rank_payoffs]).encode("utf-8")
        ).hexdigest()

    def forward(self, features: Tensor) -> Tensor:
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.net(features.float())

    @torch.no_grad()
    def predict_probs_and_potential(
        self,
        sparse: Tensor,
        numeric: Tensor,
        reward_weights: Tensor | Sequence[float] | None = None,
    ) -> tuple[Tensor, Tensor]:
        rw = self.rank_payoffs if reward_weights is None else reward_weights
        feats = canonicalize_grp_state(sparse, numeric, rw)
        logits = self.forward(feats)
        probs = torch.softmax(logits, dim=-1)
        payoffs = torch.as_tensor(rw, dtype=torch.float32, device=probs.device)
        if payoffs.ndim == 1:
            potential = (probs * payoffs.unsqueeze(0)).sum(dim=-1)
        else:
            potential = (probs * payoffs).sum(dim=-1)
        return probs, potential


@torch.no_grad()
def compute_grp_calibration_metrics(
    logits: Tensor, targets: Tensor, *, n_bins: int = 15
) -> dict[str, float]:
    """Compute top-1 ECE, 4-class marginal classwise_ece, Brier score, and NLL."""
    logits = logits.float().cpu()
    targets = targets.long().cpu()
    probs = torch.softmax(logits, dim=-1)
    n = max(1, int(targets.numel()))

    # Top-1 ECE
    confidences, preds = probs.max(dim=-1)
    accuracies = (preds == targets).float()
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    top1_ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (confidences >= lo) & (confidences <= hi if i == n_bins - 1 else confidences < hi)
        cnt = int(in_bin.sum().item())
        if cnt > 0:
            avg_conf = float(confidences[in_bin].mean().item())
            avg_acc = float(accuracies[in_bin].mean().item())
            top1_ece += (cnt / n) * abs(avg_acc - avg_conf)

    # 4-class marginal classwise ECE
    class_eces = []
    for c in range(4):
        c_probs = probs[:, c]
        c_true = (targets == c).float()
        c_ece = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            in_bin = (c_probs >= lo) & (c_probs <= hi if i == n_bins - 1 else c_probs < hi)
            cnt = int(in_bin.sum().item())
            if cnt > 0:
                avg_p = float(c_probs[in_bin].mean().item())
                avg_y = float(c_true[in_bin].mean().item())
                c_ece += (cnt / n) * abs(avg_y - avg_p)
        class_eces.append(c_ece)
    classwise_ece = sum(class_eces) / 4.0

    one_hot = F.one_hot(targets, num_classes=4).float()
    brier = float(((probs - one_hot).square().sum(dim=-1)).mean().item())
    nll = float(F.cross_entropy(logits, targets).item())
    acc = float(accuracies.mean().item())

    return {
        "top1_ece": top1_ece,
        "classwise_ece": classwise_ece,
        "brier": brier,
        "nll": nll,
        "accuracy": acc,
    }


def passes_gate(metrics: dict[str, float]) -> bool:
    """Calibration gate §4.3.2: ECE <= 0.03, classwise_ece <= 0.03, Brier <= 0.68."""
    return (
        metrics["top1_ece"] <= 0.03
        and metrics["classwise_ece"] <= 0.03
        and metrics["brier"] <= 0.68
    )


class FrozenGRP(nn.Module):
    """Frozen, checksum-bound GRP wrapper for Stage-2/3 PBRS shaping (§4.3.2)."""

    def __init__(
        self,
        model: PureStateGRP,
        calibration_metrics: dict[str, float] | None = None,
        *,
        require_gate: bool = True,
    ) -> None:
        super().__init__()
        if require_gate and calibration_metrics is not None and not passes_gate(calibration_metrics):
            raise ValueError(f"GRP failed calibration gate: {calibration_metrics}")
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self.rank_payoffs: tuple[float, float, float, float] = tuple(
            float(x) for x in model.rank_payoffs.tolist()
        )  # type: ignore[assignment]
        self.ruleset_sha256: str = model.ruleset_sha256
        h = hashlib.sha256()
        for k, v in sorted(model.state_dict().items()):
            h.update(k.encode("utf-8"))
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        h.update(self.ruleset_sha256.encode("utf-8"))
        self.grp_sha256: str = h.hexdigest()

    @torch.no_grad()
    def potential(self, sparse: Tensor, numeric: Tensor) -> Tensor:
        _, phi = self.model.predict_probs_and_potential(sparse, numeric)
        return phi


@torch.no_grad()
def shape_hanchan_pbrs_rewards(
    round_potentials: Sequence[float],
    round_ids: Sequence[tuple[int, int, int] | int],
    terminal_return: float,
) -> tuple[list[float], list[bool]]:
    """Compute exact full-hanchan PBRS shaped rewards (gamma = 1.0, §4.3.3).

    Args:
      round_potentials: Phi_t for each active decision step t = 0..T-1 of the observer.
      round_ids:        round key (e.g. (chang, ju, ben)) at each active step t.
      terminal_return:  R_terminal(i) at end of hanchan (Phi_terminal === 0).

    Note: Because `round_potentials` and `round_ids` are aligned to active decision steps,
    any skipped 0-decision intermediate rounds between step `t` (in round k) and `t+1`
    (in round k_next != k) automaticallytelescope into `Phi_{k_next} - Phi_k`, so
    sum(rewards) == terminal_return - round_potentials[0] ALWAYS!
    """
    n = len(round_potentials)
    if n == 0:
        return [], []
    assert len(round_ids) == n
    rewards = [0.0] * n
    dones = [False] * n
    for t in range(n - 1):
        if round_ids[t + 1] != round_ids[t]:
            rewards[t] = float(round_potentials[t + 1]) - float(round_potentials[t])
    rewards[n - 1] = float(terminal_return) - float(round_potentials[n - 1])
    dones[n - 1] = True
    return rewards, dones


__all__ = [
    "DEFAULT_RANK_PAYOFFS",
    "FrozenGRP",
    "GRP_INPUT_DIM",
    "PureStateGRP",
    "canonicalize_grp_state",
    "compute_grp_calibration_metrics",
    "passes_gate",
    "shape_hanchan_pbrs_rewards",
]
