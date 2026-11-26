"""20-dim seat-equivariant Pure-State GRP potential & PBRS hanchan shaping (§4.3)."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import math
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn import functional as F

from kanachan.constants import (
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    NUM_RESULTS,
    NUM_TYPES_OF_SPARSE_FEATURES,
    SEAT_INDEX,
    SEAT_OFFSET,
)
from kanachan.training.core.ranking import scores_to_ranks
from kanachan.training.online.state import atomic_save


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


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float, np.number)):
        return False
    return math.isfinite(float(value))


def compute_grp_checksum(model: PureStateGRP) -> str:
    """Compute SHA-256 checksum over PureStateGRP parameters/buffers and ruleset hash."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode("utf-8"))
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    h.update(model.ruleset_sha256.encode("utf-8"))
    return h.hexdigest()


def passes_gate(metrics: Mapping[str, Any]) -> bool:
    """Calibration gate §4.3.2: ECE <= 0.03, classwise_ece <= 0.03, Brier <= 0.68."""
    required = ("top1_ece", "classwise_ece", "brier")
    if not isinstance(metrics, Mapping):
        return False
    vals: dict[str, float] = {}
    for key in required:
        if key not in metrics or not _is_finite_number(metrics[key]):
            return False
        v = float(metrics[key])
        if v < 0.0:
            return False
        vals[key] = v
    return (
        vals["top1_ece"] <= 0.03
        and vals["classwise_ece"] <= 0.03
        and vals["brier"] <= 0.68
    )


class FrozenGRP(nn.Module):
    """Frozen, checksum-bound GRP wrapper for Stage-2/3 PBRS shaping (§4.3.2)."""

    def __init__(
        self,
        model: PureStateGRP,
        calibration_metrics: Mapping[str, Any] | None = None,
        *,
        require_gate: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self.rank_payoffs: tuple[float, float, float, float] = tuple(
            float(x) for x in model.rank_payoffs.tolist()
        )  # type: ignore[assignment]
        self.ruleset_sha256: str = model.ruleset_sha256
        self.grp_sha256: str = compute_grp_checksum(model)

        report: dict[str, Any]
        if calibration_metrics is None:
            if require_gate:
                raise ValueError(
                    "FrozenGRP requires verified calibration_metrics when require_gate=True (got None)."
                )
            report = {
                "gated": False,
                "passed_gate": False,
                "grp_sha256": self.grp_sha256,
                "ruleset_sha256": self.ruleset_sha256,
                "rank_payoffs": list(self.rank_payoffs),
            }
        else:
            if not isinstance(calibration_metrics, Mapping):
                raise ValueError(
                    f"calibration_metrics must be a mapping, got {type(calibration_metrics)}."
                )
            for req in ("top1_ece", "classwise_ece", "brier"):
                if req not in calibration_metrics:
                    raise ValueError(f"calibration_metrics missing required key {req!r}.")
            for k, v in calibration_metrics.items():
                if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                    if not math.isfinite(float(v)):
                        raise ValueError(
                            f"Non-finite scalar metric {k}={v!r} in GRP calibration_metrics."
                        )

            if calibration_metrics.get("grp_sha256") is not None:
                if str(calibration_metrics["grp_sha256"]) != self.grp_sha256:
                    raise ValueError(
                        "calibration_metrics['grp_sha256'] disagrees with PureStateGRP checksum: "
                        f"{calibration_metrics['grp_sha256']!r} != {self.grp_sha256!r}."
                    )
            if calibration_metrics.get("ruleset_sha256") is not None:
                if str(calibration_metrics["ruleset_sha256"]) != self.ruleset_sha256:
                    raise ValueError(
                        "calibration_metrics['ruleset_sha256'] disagrees with PureStateGRP ruleset: "
                        f"{calibration_metrics['ruleset_sha256']!r} != {self.ruleset_sha256!r}."
                    )
            if calibration_metrics.get("rank_payoffs") is not None:
                rp_raw = list(calibration_metrics["rank_payoffs"])
                if len(rp_raw) != 4 or any(not _is_finite_number(x) for x in rp_raw):
                    raise ValueError(
                        f"Invalid rank_payoffs in calibration_metrics: {rp_raw!r}."
                    )
                rp_vals = tuple(float(x) for x in rp_raw)
                if any(abs(a - b) > 1e-6 for a, b in zip(rp_vals, self.rank_payoffs, strict=True)):
                    raise ValueError(
                        "calibration_metrics['rank_payoffs'] disagrees with PureStateGRP payoffs: "
                        f"{rp_vals} != {self.rank_payoffs}."
                    )

            gate_ok = passes_gate(calibration_metrics)
            if "passes_gate" in calibration_metrics and bool(calibration_metrics["passes_gate"]) != gate_ok:
                raise ValueError(
                    "Inconsistent calibration_metrics['passes_gate']="
                    f"{calibration_metrics['passes_gate']!r} vs computed passes_gate={gate_ok}."
                )
            if require_gate and not gate_ok:
                raise ValueError(f"GRP failed calibration gate: {dict(calibration_metrics)}")

            report = {}
            for k, v in calibration_metrics.items():
                if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                    report[str(k)] = int(v) if isinstance(v, (int, np.integer)) else float(v)
                elif isinstance(v, (list, tuple)):
                    report[str(k)] = list(v)
                else:
                    report[str(k)] = v
            report["gated"] = bool(require_gate)
            report["passed_gate"] = bool(gate_ok)
            report["grp_sha256"] = self.grp_sha256
            report["ruleset_sha256"] = self.ruleset_sha256
            report["rank_payoffs"] = list(self.rank_payoffs)

        self.calibration_report: dict[str, Any] = report

    @torch.no_grad()
    def potential(self, sparse: Tensor, numeric: Tensor) -> Tensor:
        _, phi = self.model.predict_probs_and_potential(sparse, numeric)
        return phi


class FrozenGRPPBRSReward:
    """Offline RL PBRS reward using verified 20-dim FrozenGRP + terminal payoffs (§4.3.3)."""

    def __init__(self, frozen_grp: FrozenGRP) -> None:
        if not isinstance(frozen_grp, FrozenGRP):
            raise TypeError(f"Expected FrozenGRP instance, got {type(frozen_grp)}.")
        self.frozen_grp = frozen_grp
        self.payoffs = torch.tensor(frozen_grp.rank_payoffs, dtype=torch.float32)

    def grp_metadata(self) -> dict[str, Any]:
        return {
            "grp_sha256": self.frozen_grp.grp_sha256,
            "ruleset_sha256": self.frozen_grp.ruleset_sha256,
            "rank_payoffs": list(self.frozen_grp.rank_payoffs),
            "calibration_report": dict(self.frozen_grp.calibration_report),
        }

    @torch.no_grad()
    def __call__(self, episode: TensorDict, contiguous: bool) -> None:
        terminal = episode["next", "end_of_game"].bool()
        count = int(episode.size(0))
        if count == 0:
            raise ValueError("An empty episode has no GRP PBRS reward.")
        if not contiguous or not bool(terminal[-1]):
            raise ValueError("FrozenGRPPBRSReward requires complete, contiguous observer games.")
        seats = episode["sparse"][:, SEAT_INDEX].long() - SEAT_OFFSET
        if bool(((seats < 0) | (seats >= 4)).any()):
            raise ValueError("Invalid starting seat in episode['sparse'].")
        first_game = terminal.roll(1).clone()
        first_game[0] = True
        if bool(((seats != seats.roll(1)) & ~first_game).any()):
            raise ValueError("An observer seat changed within a game.")

        device = terminal.device
        payoffs_dev = self.payoffs.to(device=device)
        reward = torch.zeros(count, dtype=torch.float32, device=device)

        if terminal.any():
            final_scores = episode["next", "results"][terminal, 10:14]
            ranks = scores_to_ranks(final_scores)
            own_rank = ranks.gather(1, seats[terminal, None]).squeeze(1)
            reward[terminal] = payoffs_dev[own_rank]

        starts = episode["next", "end_of_round"].roll(1).bool() | terminal.roll(1)
        starts[0] = True
        indices = starts.nonzero().flatten()
        grp_dev = next(self.frozen_grp.parameters()).device
        phi_rounds = self.frozen_grp.potential(
            episode["sparse"][indices].to(device=grp_dev),
            episode["numeric"][indices].to(device=grp_dev),
        )
        segment = starts.long().cumsum(0) - 1
        potential = phi_rounds.to(device=device, dtype=torch.float32)[segment]
        next_potential = potential.roll(-1).masked_fill(terminal, 0.0)
        reward += next_potential - potential

        episode["next", "done"] = terminal.clone()
        episode["next", "reward"] = reward


@dataclass(frozen=True)
class GRPRoundDataset:
    """Compact round-start dataset (sparse[:, :9], numeric[:, :6], final_rank)."""

    sparse: Tensor
    numeric: Tensor
    final_rank: Tensor
    uuids: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.final_rank.shape[0])

    def select(self, indices: Sequence[int] | Tensor) -> GRPRoundDataset:
        idx_t = torch.as_tensor(indices, dtype=torch.long)
        idx_list = [int(i) for i in idx_t.tolist()]
        return GRPRoundDataset(
            sparse=self.sparse[idx_t],
            numeric=self.numeric[idx_t],
            final_rank=self.final_rank[idx_t],
            uuids=tuple(self.uuids[i] for i in idx_list),
        )

    def split_by_uuid(
        self, val_fraction: float = 0.1, *, seed: int = 42
    ) -> tuple[GRPRoundDataset, GRPRoundDataset]:
        """Partition rounds into (train, val) without leaking any game UUID across splits."""
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")
        unique_uuids = sorted(set(self.uuids))
        if len(unique_uuids) < 2:
            raise ValueError("At least 2 distinct game UUIDs are required for train/val split.")
        scored: list[tuple[bytes, str]] = []
        for uid in unique_uuids:
            digest = hashlib.blake2b(f"{seed}:{uid}".encode("utf-8"), digest_size=8).digest()
            scored.append((digest, uid))
        scored.sort(key=lambda item: (item[0], item[1]))
        n_val_games = max(1, min(len(unique_uuids) - 1, round(len(unique_uuids) * val_fraction)))
        val_uuids = {uid for _, uid in scored[:n_val_games]}
        train_idx = [i for i, uid in enumerate(self.uuids) if uid not in val_uuids]
        val_idx = [i for i, uid in enumerate(self.uuids) if uid in val_uuids]
        return self.select(train_idx), self.select(val_idx)


def grp_samples_from_rounds(rounds: Sequence[Any]) -> GRPRoundDataset:
    """Build a GRPRoundDataset from parsed RoundTrajectory objects."""
    if not rounds:
        raise ValueError("Cannot build GRPRoundDataset from empty rounds sequence.")
    sp_list: list[Tensor] = []
    nu_list: list[Tensor] = []
    rk_list: list[int] = []
    uid_list: list[str] = []
    for r in rounds:
        if r.final_rank is None:
            raise ValueError(
                f"Round uuid={r.uuid!r} seat={r.observer_seat} has final_rank=None (unlabeled)."
            )
        if r.sparse.shape[0] == 0 or r.sparse.shape[1] < 9 or r.numeric.shape[1] < 6:
            raise ValueError(f"Invalid round feature shape for uuid={r.uuid!r}.")
        sp_list.append(r.sparse[0, :9].to(dtype=torch.int32, device="cpu"))
        nu_list.append(r.numeric[0, :6].to(dtype=torch.float32, device="cpu"))
        rk_list.append(int(r.final_rank))
        uid_list.append(str(r.uuid))
    return GRPRoundDataset(
        sparse=torch.stack(sp_list, dim=0),
        numeric=torch.stack(nu_list, dim=0),
        final_rank=torch.tensor(rk_list, dtype=torch.long),
        uuids=tuple(uid_list),
    )


def _open_grp_source(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _iter_grp_source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        files = sorted(
            p for p in source.iterdir()
            if p.is_file() and (p.name.endswith(".tsv") or p.name.endswith(".tsv.gz"))
        )
        if not files:
            raise FileNotFoundError(f"No .tsv/.tsv.gz files found in {source}.")
        return files
    raise FileNotFoundError(f"GRP source path does not exist: {source}")


def extract_grp_samples_from_source(
    source: Path,
    *,
    max_games: int | None = None,
    allow_unlabeled: bool = False,
) -> GRPRoundDataset:
    """Extract unique round-start 4 fields -> final_rank samples directly from source-v3 files."""
    files = _iter_grp_source_files(Path(source))
    # Map (uuid, seat, chang, ju, ben) -> (prog_len, sp9, nu6, seat)
    rounds_map: dict[tuple[str, int, int, int, int], tuple[int, list[int], list[int], int]] = {}
    final_scores_by_uuid: dict[str, tuple[int, int, int, int]] = {}
    selected_uuids: set[str] | None = set() if max_games is not None else None

    for fpath in files:
        with _open_grp_source(fpath) as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) not in (6, 8):
                    raise ValueError(
                        f"{fpath}:{line_no}: Expected 6 or 8 columns in source-v3, got {len(parts)}."
                    )
                uuid = parts[0].strip()
                if not uuid:
                    raise ValueError(f"{fpath}:{line_no}: Empty uuid.")
                if selected_uuids is not None and uuid not in selected_uuids:
                    if max_games is not None and len(selected_uuids) >= int(max_games):
                        continue
                    selected_uuids.add(uuid)

                if len(parts) == 6:
                    if allow_unlabeled:
                        continue
                    raise ValueError(
                        f"{fpath}:{line_no}: Missing results column for uuid={uuid!r} (allow_unlabeled=False)."
                    )

                sp_raw = [int(tok) for tok in parts[1].split(",") if tok != ""]
                if len(sp_raw) < 9 or len(sp_raw) > MAX_NUM_ACTIVE_SPARSE_FEATURES:
                    raise ValueError(
                        f"{fpath}:{line_no}: Invalid sparse length {len(sp_raw)} for uuid={uuid!r}."
                    )
                if any(x < 0 or x >= NUM_TYPES_OF_SPARSE_FEATURES for x in sp_raw):
                    raise ValueError(f"{fpath}:{line_no}: Sparse token out of range.")
                seat = sp_raw[SEAT_INDEX] - SEAT_OFFSET
                chang = sp_raw[7] - 75
                ju = sp_raw[8] - 78
                if not (0 <= seat < 4 and 0 <= chang <= 2 and 0 <= ju < 4):
                    raise ValueError(
                        f"{fpath}:{line_no}: Invalid seat/chang/ju ({seat}, {chang}, {ju})."
                    )

                nu_raw = [int(tok) for tok in parts[2].split(",") if tok != ""]
                if len(nu_raw) != NUM_NUMERIC_FEATURES:
                    raise ValueError(
                        f"{fpath}:{line_no}: Expected {NUM_NUMERIC_FEATURES} numeric features, got {len(nu_raw)}."
                    )
                ben = nu_raw[0]
                if not (0 <= ben < 255 and 0 <= nu_raw[1] < 255):
                    raise ValueError(f"{fpath}:{line_no}: Invalid benchang/liqibang in numeric: {nu_raw[:2]}.")

                res_raw = [int(tok) for tok in parts[7].split(",") if tok != ""]
                if len(res_raw) != NUM_RESULTS:
                    raise ValueError(
                        f"{fpath}:{line_no}: Expected {NUM_RESULTS} results fields, got {len(res_raw)}."
                    )
                if any(abs(s) > 1_000_000 for s in res_raw[0:4] + res_raw[6:14]):
                    raise ValueError(f"{fpath}:{line_no}: Score bound exceeded in results: {res_raw}.")
                fscores = (res_raw[10], res_raw[11], res_raw[12], res_raw[13])
                prev_scores = final_scores_by_uuid.get(uuid)
                if prev_scores is None:
                    final_scores_by_uuid[uuid] = fscores
                elif prev_scores != fscores:
                    raise ValueError(
                        f"{fpath}:{line_no}: Conflicting final_game_scores for uuid={uuid!r}: "
                        f"{prev_scores} != {fscores}."
                    )

                prog_len = 0 if not parts[3].strip() else (parts[3].count(",") + 1)
                rkey = (uuid, seat, chang, ju, ben)
                existing = rounds_map.get(rkey)
                if existing is None or prog_len < existing[0]:
                    rounds_map[rkey] = (prog_len, sp_raw[:9], nu_raw, seat)

    if not rounds_map:
        raise ValueError(f"No labeled round samples extracted from {source}.")

    sorted_keys = sorted(rounds_map.keys())
    uuids_unique = sorted({k[0] for k in sorted_keys})
    score_tensor = torch.tensor(
        [final_scores_by_uuid[u] for u in uuids_unique], dtype=torch.int64
    )
    ranks_tensor = scores_to_ranks(score_tensor)
    ranks_by_uuid = {
        u: tuple(int(r) for r in ranks_tensor[i].tolist())
        for i, u in enumerate(uuids_unique)
    }

    sp_rows: list[list[int]] = []
    nu_rows: list[list[int]] = []
    rk_rows: list[int] = []
    uid_rows: list[str] = []
    for k in sorted_keys:
        uid = k[0]
        _, sp9, nu6, seat = rounds_map[k]
        sp_rows.append(sp9)
        nu_rows.append(nu6)
        rk_rows.append(ranks_by_uuid[uid][seat])
        uid_rows.append(uid)

    return GRPRoundDataset(
        sparse=torch.tensor(sp_rows, dtype=torch.int32),
        numeric=torch.tensor(nu_rows, dtype=torch.float32),
        final_rank=torch.tensor(rk_rows, dtype=torch.long),
        uuids=tuple(uid_rows),
    )


@torch.no_grad()
def evaluate_grp_calibration_report(
    model: PureStateGRP,
    sparse: Tensor,
    numeric: Tensor,
    final_ranks: Tensor,
    *,
    temperature: float = 1.0,
    batch_size: int = 4096,
) -> dict[str, Any]:
    """Compute calibrated GRP evaluation metrics + checksum/ruleset identity report."""
    model.eval()
    dev = next(model.parameters()).device
    rw = model.rank_payoffs.to(device=dev)
    logits_chunks: list[Tensor] = []
    n = int(final_ranks.shape[0])
    if n == 0:
        raise ValueError("Cannot evaluate GRP calibration report on 0 samples.")
    for start in range(0, n, max(1, int(batch_size))):
        sl = slice(start, min(n, start + max(1, int(batch_size))))
        feats = canonicalize_grp_state(sparse[sl].to(device=dev), numeric[sl].to(device=dev), rw)
        logits_chunks.append(model(feats).cpu())
    logits = torch.cat(logits_chunks, dim=0)
    metrics = compute_grp_calibration_metrics(logits, final_ranks.long().cpu())
    gate_ok = passes_gate(metrics)
    return {
        **metrics,
        "temperature": float(temperature),
        "num_samples": n,
        "passes_gate": gate_ok,
        "grp_sha256": compute_grp_checksum(model),
        "ruleset_sha256": model.ruleset_sha256,
        "rank_payoffs": [float(x) for x in model.rank_payoffs.tolist()],
    }


def calibrate_grp_temperature_(
    model: PureStateGRP,
    val_sparse: Tensor,
    val_numeric: Tensor,
    val_ranks: Tensor,
    *,
    batch_size: int = 4096,
    max_iter: int = 200,
) -> tuple[float, dict[str, Any]]:
    """Post-hoc fit scalar temperature T > 0 on validation logits and fold 1/T into head weights."""
    model.eval()
    dev = next(model.parameters()).device
    rw = model.rank_payoffs.to(device=dev)
    with torch.no_grad():
        logits_list: list[Tensor] = []
        n = int(val_ranks.shape[0])
        for start in range(0, n, max(1, int(batch_size))):
            sl = slice(start, min(n, start + max(1, int(batch_size))))
            feats = canonicalize_grp_state(
                val_sparse[sl].to(device=dev), val_numeric[sl].to(device=dev), rw
            )
            logits_list.append(model(feats).detach().float().cpu())
        logits = torch.cat(logits_list, dim=0)
    targets = val_ranks.long().cpu()
    nll_before = float(F.cross_entropy(logits, targets).item())

    log_t = nn.Parameter(torch.zeros((), dtype=torch.float32))
    opt = torch.optim.LBFGS([log_t], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe")

    def _closure() -> Tensor:
        opt.zero_grad()
        temp = torch.exp(log_t.clamp(math.log(0.05), math.log(20.0)))
        loss = F.cross_entropy(logits / temp, targets)
        loss.backward()
        return loss

    opt.step(_closure)
    best_t = float(torch.exp(log_t.detach().clamp(math.log(0.05), math.log(20.0))).item())
    nll_after = float(F.cross_entropy(logits / best_t, targets).item())
    if not math.isfinite(best_t) or nll_after > nll_before:
        best_t = 1.0
    if best_t != 1.0:
        inv_t = 1.0 / best_t
        last_linear = model.net[-1]
        assert isinstance(last_linear, nn.Linear)
        with torch.no_grad():
            last_linear.weight.mul_(inv_t)
            if last_linear.bias is not None:
                last_linear.bias.mul_(inv_t)
    report = evaluate_grp_calibration_report(
        model, val_sparse, val_numeric, val_ranks, temperature=best_t, batch_size=batch_size
    )
    return best_t, report


def train_pure_state_grp(
    train_data: GRPRoundDataset,
    val_data: GRPRoundDataset,
    *,
    rank_payoffs: Sequence[float] = DEFAULT_RANK_PAYOFFS,
    epochs: int = 15,
    batch_size: int = 2048,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    calibrate_temperature: bool = True,
    require_gate: bool = True,
    device: torch.device | str | None = None,
    seed: int = 42,
) -> tuple[FrozenGRP, dict[str, Any]]:
    """Train 20-dim PureStateGRP, calibrate on validation set, enforce gate, and freeze."""
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("Both train_data and val_data must be non-empty.")
    overlap = set(train_data.uuids) & set(val_data.uuids)
    if overlap:
        raise ValueError(
            f"GRP train/val UUID leakage detected ({len(overlap)} overlapping game UUIDs)."
        )

    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    torch.manual_seed(int(seed))

    model = PureStateGRP(rank_payoffs=rank_payoffs).to(device=dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rw = model.rank_payoffs.to(device=dev)

    # Pre-canonicalize once on CPU to keep epoch shuffling cheap and deterministic.
    train_feats = canonicalize_grp_state(train_data.sparse, train_data.numeric, model.rank_payoffs.cpu())
    train_targets = train_data.final_rank.long()
    n_train = int(train_targets.shape[0])
    bsz = max(1, int(batch_size))

    for _epoch in range(max(1, int(epochs))):
        model.train()
        perm = torch.randperm(n_train, generator=gen)
        for start in range(0, n_train, bsz):
            idx = perm[start:start + bsz]
            xb = train_feats[idx].to(device=dev)
            yb = train_targets[idx].to(device=dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    if calibrate_temperature:
        _, report = calibrate_grp_temperature_(
            model, val_data.sparse, val_data.numeric, val_data.final_rank
        )
    else:
        report = evaluate_grp_calibration_report(
            model, val_data.sparse, val_data.numeric, val_data.final_rank, temperature=1.0
        )

    frozen = FrozenGRP(model, report, require_gate=require_gate)
    return frozen, report


def save_frozen_grp(output_path: Path, frozen_grp: FrozenGRP) -> Path:
    """Save checksum-bound FrozenGRP artifact atomically."""
    if not isinstance(frozen_grp, FrozenGRP):
        raise TypeError(f"Expected FrozenGRP, got {type(frozen_grp)}.")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cpu_sd = {
        k: v.detach().cpu().clone()
        for k, v in frozen_grp.model.state_dict().items()
    }
    payload = {
        "format": "kanachan_frozen_grp_v1",
        "model_state_dict": cpu_sd,
        "rank_payoffs": list(frozen_grp.rank_payoffs),
        "ruleset_sha256": frozen_grp.ruleset_sha256,
        "grp_sha256": frozen_grp.grp_sha256,
        "calibration_report": dict(frozen_grp.calibration_report),
    }
    atomic_save(out, payload)
    return out


def load_frozen_grp(
    ckpt_path: Path,
    *,
    require_gate: bool = True,
    map_location: torch.device | str = "cpu",
) -> FrozenGRP:
    """Load FrozenGRP artifact with strict checksum, ruleset, and calibration gate verification."""
    payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "kanachan_frozen_grp_v1":
        raise ValueError(f"Invalid FrozenGRP checkpoint format in {ckpt_path}.")
    model = PureStateGRP(rank_payoffs=payload["rank_payoffs"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    actual_sha = compute_grp_checksum(model)
    expected_sha = payload.get("grp_sha256")
    if actual_sha != expected_sha:
        raise ValueError(
            f"FrozenGRP checksum mismatch in {ckpt_path}: expected {expected_sha!r}, got {actual_sha!r}."
        )
    if model.ruleset_sha256 != payload.get("ruleset_sha256"):
        raise ValueError(
            f"FrozenGRP ruleset mismatch in {ckpt_path}: {model.ruleset_sha256!r} != {payload.get('ruleset_sha256')!r}."
        )
    frozen = FrozenGRP(model, payload.get("calibration_report"), require_gate=require_gate)
    return frozen.to(device=torch.device(map_location))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint: train PureStateGRP on source-v3, calibrate/gate, and save FrozenGRP."""
    parser = argparse.ArgumentParser(description="Train & calibrate 20-dim PureStateGRP")
    parser.add_argument("--source", type=Path, required=True, help="Source-v3 .bc.tsv(.gz) file or dir.")
    parser.add_argument("--val-source", type=Path, default=None, help="Optional separate validation source-v3.")
    parser.add_argument("--output", type=Path, required=True, help="Destination FrozenGRP .pt file.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation UUID fraction when --val-source is omitted.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--allow-ungated", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.val_source is not None:
        train_ds = extract_grp_samples_from_source(args.source, max_games=args.max_games)
        val_ds = extract_grp_samples_from_source(args.val_source, max_games=args.max_games)
    else:
        full_ds = extract_grp_samples_from_source(args.source, max_games=args.max_games)
        train_ds, val_ds = full_ds.split_by_uuid(val_fraction=args.val_fraction, seed=args.seed)

    frozen, report = train_pure_state_grp(
        train_ds,
        val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        calibrate_temperature=not args.no_calibrate,
        require_gate=not args.allow_ungated,
        device=args.device,
        seed=args.seed,
    )
    save_frozen_grp(args.output, frozen)
    print(json.dumps(report, sort_keys=True))
    return 0


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
    "FrozenGRPPBRSReward",
    "GRP_INPUT_DIM",
    "GRPRoundDataset",
    "PureStateGRP",
    "calibrate_grp_temperature_",
    "canonicalize_grp_state",
    "compute_grp_calibration_metrics",
    "compute_grp_checksum",
    "evaluate_grp_calibration_report",
    "extract_grp_samples_from_source",
    "grp_samples_from_rounds",
    "load_frozen_grp",
    "main",
    "passes_gate",
    "save_frozen_grp",
    "shape_hanchan_pbrs_rewards",
    "train_pure_state_grp",
]
