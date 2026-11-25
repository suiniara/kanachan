"""Stage-1 Parallel Round-Trajectory BC, Suit Aug, Auxiliary Heads & 9-Family Eval (§4.2)."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from kanachan.model_loader import dump_model, dump_object, load_model
from kanachan.nn.decode_converter import DecodeConverter
from kanachan.nn.structured_dual_decoder import (
    ARCHITECTURE_ID,
    NextEventHead,
    StructuredActionHead,
    StructuredDualDecoder,
    StructuredStateHead,
    structured_v2_schema,
)
from kanachan.structured_v2 import (
    KANACHAN_ACTION_MANIFEST,
    KANACHAN_ACTION_MANIFEST_SHA256,
    NUM_TYPES_OF_ACTIONS,
    PAD_ACTION_ID,
    PERM_TILE37,
    derive_structured_v2_batch,
    permute_structured_batch,
)
from kanachan.training.core.optimizers import adamw_parameter_groups
from kanachan.training.online.state import atomic_save, capture_rng, restore_rng


ACTION_FAMILY_NAMES: tuple[str, ...] = (
    "discard",
    "riichi",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "win",
    "pass_kyuushu",
)


def classify_action_id(global_id: int) -> str:
    row = KANACHAN_ACTION_MANIFEST[int(global_id)]
    atype = int(row[0])
    is_riichi = int(row[8])
    if atype == 0:
        return "riichi" if is_riichi else "discard"
    if atype == 6:
        return "chi"
    if atype == 7:
        return "pon"
    if atype == 8:
        return "daiminkan"
    if atype == 1:
        return "ankan"
    if atype == 2:
        return "kakan"
    if atype in (3, 9):
        return "win"
    return "pass_kyuushu"


@dataclass
class RoundTrajectory:
    """One complete observer round grouped from decision steps."""

    round_events: Tensor  # [L_r, 10] uint8
    round_event_len: int
    decision_event_len: Tensor  # [T] int32
    v2_tile_feats: Tensor  # [T, 34, 70] float32
    v2_global_feats: Tensor  # [T, 48] float32
    v2_player_feats: Tensor  # [T, 4, 32] float32
    v2_cand_ids: Tensor  # [T, 32] int32
    v2_cand_mask: Tensor  # [T, 32] bool
    v2_cand_mode: Tensor  # [T, 32] uint8
    v2_cand_dyn: Tensor  # [T, 32, 8] float32
    candidates: Tensor  # [T, 32] int32
    action_idx: Tensor  # [T] int64
    final_rank: Tensor | None = None  # [T] int64 (-1 if unlabeled)


def _open_tsv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def parse_bc_rounds(
    path: Path,
    *,
    max_rounds: int | None = None,
    max_lines: int | None = None,
    enable_early_shape_hint: bool = False,
) -> list[RoundTrajectory]:
    """Read BC TSV/TSV.GZ lines and group consecutive steps of each round."""
    raw_groups: list[tuple[tuple[Any, ...], list[tuple[list[int], list[int], list[int], list[int], int, int]]]] = []
    index_by_key: dict[tuple[Any, ...], int] = {}

    with _open_tsv(path) as fp:
        for line_no, line in enumerate(fp):
            if max_lines is not None and line_no >= max_lines:
                break
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 6:
                continue
            uuid = parts[0]
            sp = ([int(x) for x in parts[1].split(",")] + [510] * 29)[:29]
            nu = [int(x) for x in parts[2].split(",")]
            pr = ([int(x) for x in parts[3].split(",")] + [2165] * 109)[:109]
            ca = ([int(x) for x in parts[4].split(",")] + [PAD_ACTION_ID] * 32)[:32]
            act = int(parts[5])
            final_rank = int(parts[6]) if len(parts) >= 7 and parts[6].lstrip("-").isdigit() else -1

            # Group by (uuid, seat, chang, ju, ben)
            key = (uuid, sp[6], sp[7], sp[8], nu[0])
            idx = index_by_key.get(key)
            if idx is None:
                if max_rounds is not None and len(raw_groups) >= max_rounds:
                    continue
                idx = len(raw_groups)
                index_by_key[key] = idx
                raw_groups.append((key, []))
            raw_groups[idx][1].append((sp, nu, pr, ca, act, final_rank))

    rounds: list[RoundTrajectory] = []
    for _, steps in raw_groups:
        sp_t = torch.tensor([s[0] for s in steps], dtype=torch.int32)
        nu_t = torch.tensor([s[1] for s in steps], dtype=torch.int32)
        pr_t = torch.tensor([s[2] for s in steps], dtype=torch.int32)
        ca_t = torch.tensor([s[3] for s in steps], dtype=torch.int32)
        act_t = torch.tensor([s[4] for s in steps], dtype=torch.long)
        rk_vals = [s[5] for s in steps]

        feats = derive_structured_v2_batch(
            sp_t, nu_t, pr_t, ca_t,
            enable_early_shape_hint=enable_early_shape_hint,
            include_room_level=False,
        )
        ev_lens = feats["v2_event_len"]
        # Sort by progression length ascending to reconstruct chronological order.
        order = torch.argsort(ev_lens, stable=True)
        ev_lens_s = ev_lens[order]
        events_s = feats["v2_events"][order]

        # Split only if events aren't a prefix chain (safety against reused uuids/rounds).
        start = 0
        t_total = int(order.numel())
        for t in range(t_total):
            is_split = False
            if t > start:
                prev_len = int(ev_lens_s[t - 1].item())
                cur_len = int(ev_lens_s[t].item())
                if cur_len < prev_len or not torch.equal(
                    events_s[t - 1, :prev_len], events_s[t, :prev_len]
                ):
                    is_split = True
            if is_split or t == t_total - 1:
                end = t if is_split else t_total
                sub_idx = order[start:end]
                last_i = int(sub_idx[-1].item())
                l_r = int(ev_lens[last_i].item())
                r_events = feats["v2_events"][last_i, :l_r].clone()
                rk_sub = [rk_vals[int(i.item())] for i in sub_idx]
                rk_t = (
                    torch.tensor(rk_sub, dtype=torch.long)
                    if any(r >= 0 for r in rk_sub)
                    else None
                )
                rounds.append(
                    RoundTrajectory(
                        round_events=r_events,
                        round_event_len=l_r,
                        decision_event_len=ev_lens[sub_idx].to(torch.int32),
                        v2_tile_feats=feats["v2_tile_feats"][sub_idx],
                        v2_global_feats=feats["v2_global_feats"][sub_idx],
                        v2_player_feats=feats["v2_player_feats"][sub_idx],
                        v2_cand_ids=feats["v2_cand_ids"][sub_idx],
                        v2_cand_mask=feats["v2_cand_mask"][sub_idx],
                        v2_cand_mode=feats["v2_cand_mode"][sub_idx],
                        v2_cand_dyn=feats["v2_cand_dyn"][sub_idx],
                        candidates=ca_t[sub_idx],
                        action_idx=act_t[sub_idx],
                        final_rank=rk_t,
                    )
                )
                if is_split:
                    start = t
                    if t == t_total - 1:
                        # Single trailing round step
                        last_i2 = int(order[t].item())
                        l_r2 = int(ev_lens[last_i2].item())
                        sub_idx2 = order[t : t + 1]
                        rk_sub2 = [rk_vals[last_i2]]
                        rounds.append(
                            RoundTrajectory(
                                round_events=feats["v2_events"][last_i2, :l_r2].clone(),
                                round_event_len=l_r2,
                                decision_event_len=ev_lens[sub_idx2].to(torch.int32),
                                v2_tile_feats=feats["v2_tile_feats"][sub_idx2],
                                v2_global_feats=feats["v2_global_feats"][sub_idx2],
                                v2_player_feats=feats["v2_player_feats"][sub_idx2],
                                v2_cand_ids=feats["v2_cand_ids"][sub_idx2],
                                v2_cand_mask=feats["v2_cand_mask"][sub_idx2],
                                v2_cand_mode=feats["v2_cand_mode"][sub_idx2],
                                v2_cand_dyn=feats["v2_cand_dyn"][sub_idx2],
                                candidates=ca_t[sub_idx2],
                                action_idx=act_t[sub_idx2],
                                final_rank=(
                                    torch.tensor(rk_sub2, dtype=torch.long)
                                    if rk_sub2[0] >= 0
                                    else None
                                ),
                            )
                        )
    return rounds


def build_round_trajectories_from_raw(
    sp: Tensor,
    nu: Tensor,
    pr: Tensor,
    ca: Tensor,
    actions: Tensor,
    round_group_ids: Sequence[Any],
    *,
    final_ranks: Tensor | None = None,
    enable_early_shape_hint: bool = False,
) -> list[RoundTrajectory]:
    """Group pre-loaded 4-field tensors into RoundTrajectory objects (in-memory helper)."""
    feats = derive_structured_v2_batch(
        sp, nu, pr, ca,
        enable_early_shape_hint=enable_early_shape_hint,
        include_room_level=False,
    )
    grouped: dict[Any, list[int]] = {}
    for idx, gid in enumerate(round_group_ids):
        grouped.setdefault(gid, []).append(idx)

    rounds: list[RoundTrajectory] = []
    for idxs in grouped.values():
        sub = torch.tensor(idxs, dtype=torch.long)
        ev_lens = feats["v2_event_len"][sub]
        order = sub[torch.argsort(ev_lens, stable=True)]
        last_i = int(order[-1].item())
        l_r = int(feats["v2_event_len"][last_i].item())
        rounds.append(
            RoundTrajectory(
                round_events=feats["v2_events"][last_i, :l_r].clone(),
                round_event_len=l_r,
                decision_event_len=feats["v2_event_len"][order].to(torch.int32),
                v2_tile_feats=feats["v2_tile_feats"][order],
                v2_global_feats=feats["v2_global_feats"][order],
                v2_player_feats=feats["v2_player_feats"][order],
                v2_cand_ids=feats["v2_cand_ids"][order],
                v2_cand_mask=feats["v2_cand_mask"][order],
                v2_cand_mode=feats["v2_cand_mode"][order],
                v2_cand_dyn=feats["v2_cand_dyn"][order],
                candidates=ca[order],
                action_idx=actions[order].long(),
                final_rank=final_ranks[order].long() if final_ranks is not None else None,
            )
        )
    return rounds


class LengthBucketBatchSampler:
    """Bucketed round sampler minimizing round_events padding with exact checkpoint resume."""

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        seed: int = 0,
        bucket_mult: int = 8,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.bucket_mult = int(bucket_mult)
        self.epoch = 0
        self.cursor = 0
        self.batches: list[list[int]] = []
        self._build_epoch_batches()

    def _build_epoch_batches(self) -> None:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * 1009)
        n = len(self.lengths)
        if n == 0:
            self.batches = []
            self.cursor = 0
            return
        perm = torch.randperm(n, generator=g).tolist()
        pool_sz = max(self.batch_size, self.batch_size * self.bucket_mult)
        all_batches: list[list[int]] = []
        for s in range(0, n, pool_sz):
            chunk = sorted(perm[s : s + pool_sz], key=lambda i: self.lengths[i])
            for b in range(0, len(chunk), self.batch_size):
                all_batches.append(chunk[b : b + self.batch_size])
        batch_perm = torch.randperm(len(all_batches), generator=g).tolist()
        self.batches = [all_batches[i] for i in batch_perm]
        self.cursor = 0

    def reset_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._build_epoch_batches()

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[list[int]]:
        while self.cursor < len(self.batches):
            batch = self.batches[self.cursor]
            self.cursor += 1
            yield batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_items": len(self.lengths),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            state["seed"] != self.seed
            or state["batch_size"] != self.batch_size
            or state["num_items"] != len(self.lengths)
        ):
            raise ValueError("LengthBucketBatchSampler config mismatch on resume.")
        self.epoch = int(state["epoch"])
        self._build_epoch_batches()
        self.cursor = int(state["cursor"])


def _permute_events_only(events: Tensor, perm_idx: int) -> Tensor:
    """Apply suit permutation 0..5 to a [..., 10] events tensor."""
    if perm_idx == 0:
        return events
    ev = events.clone()
    p37 = torch.as_tensor(PERM_TILE37[perm_idx].copy(), dtype=ev.dtype, device=ev.device)
    for col in (3, 4, 5):
        valid = ev[..., col] < 37
        idx = ev[..., col].clamp(0, 36).long()
        ev[..., col] = torch.where(valid, p37[idx], ev[..., col])
    return ev


def collate_round_batch(
    rounds: Sequence[RoundTrajectory],
    *,
    suit_perm_generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Pad variable-length rounds to [B_r, L_max, ...] and [B_r, T_max, ...]."""
    bsz_r = len(rounds)
    l_max = max(r.round_event_len for r in rounds)
    t_max = max(int(r.decision_event_len.shape[0]) for r in rounds)

    round_events = torch.zeros((bsz_r, l_max, 10), dtype=torch.uint8)
    round_events[:, :, 3:6] = 37
    round_event_len = torch.zeros((bsz_r,), dtype=torch.int32)
    dec_event_len = torch.ones((bsz_r, t_max), dtype=torch.int32)
    dec_mask = torch.zeros((bsz_r, t_max), dtype=torch.bool)

    tile_feats = torch.zeros((bsz_r, t_max, 34, 70), dtype=torch.float32)
    global_feats = torch.zeros((bsz_r, t_max, 48), dtype=torch.float32)
    player_feats = torch.zeros((bsz_r, t_max, 4, 32), dtype=torch.float32)
    cand_ids = torch.full((bsz_r, t_max, 32), PAD_ACTION_ID, dtype=torch.int32)
    cand_mask = torch.zeros((bsz_r, t_max, 32), dtype=torch.bool)
    cand_mode = torch.zeros((bsz_r, t_max, 32), dtype=torch.uint8)
    cand_dyn = torch.zeros((bsz_r, t_max, 32, 8), dtype=torch.float32)
    candidates = torch.full((bsz_r, t_max, 32), PAD_ACTION_ID, dtype=torch.int32)
    action_idx = torch.zeros((bsz_r, t_max), dtype=torch.int64)
    final_rank = torch.full((bsz_r, t_max), -1, dtype=torch.int64)

    for b, r in enumerate(rounds):
        lr = r.round_event_len
        t = int(r.decision_event_len.shape[0])
        perm_idx = (
            int(torch.randint(0, 6, (1,), generator=suit_perm_generator).item())
            if suit_perm_generator is not None
            else 0
        )

        r_ev = _permute_events_only(r.round_events, perm_idx)
        round_events[b, :lr] = r_ev
        round_event_len[b] = lr
        dec_event_len[b, :t] = r.decision_event_len
        dec_mask[b, :t] = True

        if perm_idx != 0:
            dummy_ev = torch.zeros((t, 1, 10), dtype=torch.uint8)
            step_batch = {
                "v2_tile_feats": r.v2_tile_feats,
                "v2_events": dummy_ev,
                "v2_global_feats": r.v2_global_feats,
                "v2_player_feats": r.v2_player_feats,
                "v2_cand_ids": r.v2_cand_ids,
                "v2_cand_mask": r.v2_cand_mask,
                "v2_cand_mode": r.v2_cand_mode,
                "v2_cand_dyn": r.v2_cand_dyn,
                "action": r.action_idx,
            }
            p_batch = permute_structured_batch(step_batch, perm_idx)
            tf_s = p_batch["v2_tile_feats"]
            gf_s = p_batch["v2_global_feats"]
            pf_s = p_batch["v2_player_feats"]
            cids_s = p_batch["v2_cand_ids"]
            cmask_s = p_batch["v2_cand_mask"]
            cmode_s = p_batch["v2_cand_mode"]
            cdyn_s = p_batch["v2_cand_dyn"]
            cands_s = cids_s
            act_s = p_batch["action"]
        else:
            tf_s = r.v2_tile_feats
            gf_s = r.v2_global_feats
            pf_s = r.v2_player_feats
            cids_s = r.v2_cand_ids
            cmask_s = r.v2_cand_mask
            cmode_s = r.v2_cand_mode
            cdyn_s = r.v2_cand_dyn
            cands_s = r.candidates
            act_s = r.action_idx

        tile_feats[b, :t] = tf_s
        global_feats[b, :t] = gf_s
        player_feats[b, :t] = pf_s
        cand_ids[b, :t] = cids_s
        cand_mask[b, :t] = cmask_s
        cand_mode[b, :t] = cmode_s
        cand_dyn[b, :t] = cdyn_s
        candidates[b, :t] = cands_s
        action_idx[b, :t] = act_s
        if r.final_rank is not None:
            final_rank[b, :t] = r.final_rank

    return {
        "round_events": round_events,
        "round_event_len": round_event_len,
        "decision_event_len": dec_event_len,
        "decision_mask": dec_mask,
        "v2_tile_feats": tile_feats,
        "v2_global_feats": global_feats,
        "v2_player_feats": player_feats,
        "v2_cand_ids": cand_ids,
        "v2_cand_mask": cand_mask,
        "v2_cand_mode": cand_mode,
        "v2_cand_dyn": cand_dyn,
        "candidates": candidates,
        "action": action_idx,
        "final_rank": final_rank,
    }


@torch.no_grad()
def evaluate_9_action_families(
    backbone: StructuredDualDecoder,
    policy_head: StructuredActionHead,
    batches: Sequence[dict[str, Tensor]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Compute per-family NLL, Top-1/Top-2 accuracy (K >= 2) and Riichi F1 (§4.2.3)."""
    backbone.eval()
    policy_head.eval()
    stats: dict[str, dict[str, float]] = {
        fam: {"count": 0.0, "nll_sum": 0.0, "top1": 0.0, "top2": 0.0}
        for fam in ACTION_FAMILY_NAMES
    }
    riichi_tp = riichi_fp = riichi_fn = 0.0

    for batch in batches:
        b_dev = {k: v.to(device=device) for k, v in batch.items()}
        out = backbone.forward_round_trajectory(
            b_dev["round_events"],
            b_dev["round_event_len"],
            b_dev["decision_event_len"],
            b_dev["decision_mask"],
            b_dev["v2_tile_feats"],
            b_dev["v2_global_feats"],
            b_dev["v2_player_feats"],
            b_dev["v2_cand_ids"],
            b_dev["v2_cand_mask"],
            b_dev["v2_cand_mode"],
            b_dev["v2_cand_dyn"],
        )
        dmask = b_dev["decision_mask"].bool()
        cands = b_dev["candidates"][dmask]
        actions = b_dev["action"][dmask].long()
        encode = out["encode"]
        logits = policy_head(cands, encode).float()
        cand_mask = cands < NUM_TYPES_OF_ACTIONS
        has_choice = cand_mask.sum(dim=-1) >= 2

        masked_logits = logits.masked_fill(~cand_mask, -torch.inf)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        preds_top2 = torch.topk(masked_logits, k=min(2, masked_logits.shape[1]), dim=-1).indices

        cands_cpu = cands.cpu()
        acts_cpu = actions.cpu()
        has_choice_cpu = has_choice.cpu()
        lp_cpu = log_probs.cpu()
        top2_cpu = preds_top2.cpu()

        for i in range(int(acts_cpu.shape[0])):
            if not bool(has_choice_cpu[i].item()):
                continue
            gt_idx = int(acts_cpu[i].item())
            gt_gid = int(cands_cpu[i, gt_idx].item())
            fam = classify_action_id(gt_gid)
            nll = -float(lp_cpu[i, gt_idx].item())
            p1 = int(top2_cpu[i, 0].item())
            p2 = int(top2_cpu[i, 1].item()) if top2_cpu.shape[1] > 1 else p1
            st = stats[fam]
            st["count"] += 1.0
            st["nll_sum"] += nll
            st["top1"] += 1.0 if p1 == gt_idx else 0.0
            st["top2"] += 1.0 if (gt_idx == p1 or gt_idx == p2) else 0.0

            # Track Riichi F1 over states where at least one legal riichi candidate exists
            legal_gids = cands_cpu[i][cands_cpu[i] < NUM_TYPES_OF_ACTIONS].tolist()
            has_riichi_opt = any(
                int(KANACHAN_ACTION_MANIFEST[g, 0]) == 0
                and int(KANACHAN_ACTION_MANIFEST[g, 8]) == 1
                for g in legal_gids
            )
            if has_riichi_opt:
                pred_gid = int(cands_cpu[i, p1].item())
                pred_riichi = (
                    int(KANACHAN_ACTION_MANIFEST[pred_gid, 0]) == 0
                    and int(KANACHAN_ACTION_MANIFEST[pred_gid, 8]) == 1
                )
                gt_riichi = fam == "riichi"
                if pred_riichi and gt_riichi:
                    riichi_tp += 1.0
                elif pred_riichi and not gt_riichi:
                    riichi_fp += 1.0
                elif (not pred_riichi) and gt_riichi:
                    riichi_fn += 1.0

    result: dict[str, Any] = {}
    tot_c = tot_nll = tot_t1 = tot_t2 = 0.0
    for fam, st in stats.items():
        c = st["count"]
        tot_c += c
        tot_nll += st["nll_sum"]
        tot_t1 += st["top1"]
        tot_t2 += st["top2"]
        result[fam] = {
            "count": int(c),
            "nll": (st["nll_sum"] / c) if c > 0 else 0.0,
            "top1_acc": (st["top1"] / c) if c > 0 else 0.0,
            "top2_acc": (st["top2"] / c) if c > 0 else 0.0,
        }
    prec = riichi_tp / max(1.0, riichi_tp + riichi_fp)
    rec = riichi_tp / max(1.0, riichi_tp + riichi_fn)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    result["overall"] = {
        "count": int(tot_c),
        "nll": (tot_nll / tot_c) if tot_c > 0 else 0.0,
        "top1_acc": (tot_t1 / tot_c) if tot_c > 0 else 0.0,
        "top2_acc": (tot_t2 / tot_c) if tot_c > 0 else 0.0,
        "riichi_f1": f1,
        "riichi_precision": prec,
        "riichi_recall": rec,
    }
    return result


class TrajectoryBCModel(nn.Module):
    """Combined module wrapping backbone + policy_head + optional NextEvent & FinalRank heads."""

    def __init__(
        self,
        *,
        preset: str | None = "v2_base",
        dimension: int | None = None,
        num_heads: int | None = None,
        dim_feedforward: int | None = None,
        n_d1_layers: int | None = None,
        n_d2_layers: int | None = None,
        use_rel_bias: bool | None = None,
        cand_self_attn: bool = True,
        head_type: str | None = None,
        enable_early_shape_hint: bool = False,
        checkpointing: bool = False,
        use_next_event_head: bool = True,
        next_event_weight: float = 0.20,
        use_final_rank_head: bool = False,
        final_rank_weight: float = 0.10,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.backbone = StructuredDualDecoder(
            preset=preset,
            dimension=dimension,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            n_d1_layers=n_d1_layers,
            n_d2_layers=n_d2_layers,
            use_rel_bias=use_rel_bias,
            cand_self_attn=cand_self_attn,
            head_type=head_type,
            enable_early_shape_hint=enable_early_shape_hint,
            checkpointing=checkpointing,
            device=device,
            dtype=dtype,
        )
        d = self.backbone.dimension
        self.policy_head = StructuredActionHead(
            input_dimension=d,
            head_type=self.backbone.head_type,
            device=device,
            dtype=dtype,
        )
        self.next_event_head: NextEventHead | None = (
            NextEventHead(d, device=device, dtype=dtype)
            if use_next_event_head
            else None
        )
        self.state_head: StructuredStateHead | None = (
            StructuredStateHead(input_dimension=d, final_rank=True, device=device, dtype=dtype)
            if use_final_rank_head
            else None
        )
        if self.state_head is not None:
            # Train only final_rank when state value isn't supervised in BC.
            for p in self.state_head.layers[2].parameters():
                p.requires_grad_(False)
        self.next_event_weight = float(next_event_weight)
        self.final_rank_weight = float(final_rank_weight)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        out = self.backbone.forward_round_trajectory(
            batch["round_events"],
            batch["round_event_len"],
            batch["decision_event_len"],
            batch["decision_mask"],
            batch["v2_tile_feats"],
            batch["v2_global_feats"],
            batch["v2_player_feats"],
            batch["v2_cand_ids"],
            batch["v2_cand_mask"],
            batch["v2_cand_mode"],
            batch["v2_cand_dyn"],
        )
        dmask = batch["decision_mask"].bool()
        cands = batch["candidates"][dmask]
        actions = batch["action"][dmask].long()
        encode = out["encode"]

        with torch.autocast(device_type=encode.device.type, enabled=False):
            logits = self.policy_head(cands, encode).float()
            cand_mask = cands < NUM_TYPES_OF_ACTIONS
            has_choice = cand_mask.sum(dim=-1) >= 2
            has_f = has_choice.float()

            masked_logits = logits.masked_fill(~cand_mask, -torch.inf)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            log_probs_safe = torch.where(cand_mask, log_probs, torch.zeros_like(log_probs))
            nll_per_step = F.nll_loss(log_probs_safe, actions, reduction="none")
            policy_loss = (nll_per_step * has_f).sum() / has_f.sum().clamp_min(1.0)

            loss_total = policy_loss
            metrics: dict[str, Tensor] = {
                "loss_policy": policy_loss.detach(),
                "num_choice_steps": has_f.sum().detach(),
            }

            if self.next_event_head is not None and self.next_event_weight > 0.0:
                nev_loss, nev_parts = self.next_event_head(
                    out["h_ev"], batch["round_events"], batch["round_event_len"]
                )
                loss_total = loss_total + self.next_event_weight * nev_loss
                metrics["loss_next_event"] = nev_loss.detach()
                for k, v in nev_parts.items():
                    metrics[f"next_ev_{k}"] = v

            if self.state_head is not None and self.final_rank_weight > 0.0:
                _, rank_logits = self.state_head(encode)
                rk_tgt = batch["final_rank"][dmask].long()
                labeled = rk_tgt >= 0
                lab_f = labeled.float()
                rk_ce = F.cross_entropy(
                    rank_logits.float(), rk_tgt.clamp(0, 3), reduction="none"
                )
                rk_loss = (rk_ce * lab_f).sum() / lab_f.sum().clamp_min(1.0)
                loss_total = loss_total + self.final_rank_weight * rk_loss
                metrics["loss_final_rank"] = rk_loss.detach()

        metrics["loss"] = loss_total
        return metrics

    def recipe(self, *, data_sha256: str = "", lr: float = 0.0, weight_decay: float = 0.0) -> dict[str, Any]:
        return {
            "format_version": 1,
            "architecture": ARCHITECTURE_ID,
            "manifest_sha256": KANACHAN_ACTION_MANIFEST_SHA256,
            "schema": structured_v2_schema(),
            "backbone": {
                "preset": self.backbone.preset,
                "dimension": self.backbone.dimension,
                "num_heads": self.backbone.num_heads,
                "dim_feedforward": self.backbone.dim_feedforward,
                "n_d1_layers": self.backbone.n_d1_layers,
                "n_d2_layers": self.backbone.n_d2_layers,
                "use_rel_bias": self.backbone.use_rel_bias,
                "cand_self_attn": self.backbone.cand_self_attn,
                "head_type": self.backbone.head_type,
                "enable_early_shape_hint": self.backbone.enable_early_shape_hint,
                "include_room_level": False,
                "rope_base": self.backbone.rope_base,
            },
            "use_next_event_head": self.next_event_head is not None,
            "next_event_weight": self.next_event_weight,
            "use_final_rank_head": self.state_head is not None,
            "final_rank_weight": self.final_rank_weight,
            "data_sha256": data_sha256,
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        }

    def dump_actor(self, output_path: Path, *, dtype: torch.dtype | None = None) -> None:
        save_dtype = dtype or next(self.backbone.parameters()).dtype
        enc_tdm = TensorDictModule(
            self.backbone,
            in_keys=["sparse", "numeric", "progression", "candidates"],
            out_keys=["encode"],
        )
        head_tdm = TensorDictModule(
            self.policy_head,
            in_keys=["candidates", "encode"],
            out_keys=["decode"],
        )
        lp_mod = DecodeConverter("log_probs")
        lp_tdm = TensorDictModule(
            lp_mod, in_keys=["candidates", "decode"], out_keys=["log_probs"]
        )
        arg_mod = DecodeConverter("argmax", tie_break_by_action_id=True)
        arg_tdm = TensorDictModule(
            arg_mod, in_keys=["candidates", "log_probs"], out_keys=["action"]
        )
        seq = TensorDictSequential(enc_tdm, head_tdm, lp_tdm, arg_tdm)
        state = dump_model(
            seq,
            [
                dump_object(
                    enc_tdm,
                    [
                        dump_object(
                            self.backbone,
                            [],
                            {
                                "preset": self.backbone.preset,
                                "dimension": self.backbone.dimension,
                                "num_heads": self.backbone.num_heads,
                                "dim_feedforward": self.backbone.dim_feedforward,
                                "n_d1_layers": self.backbone.n_d1_layers,
                                "n_d2_layers": self.backbone.n_d2_layers,
                                "use_rel_bias": self.backbone.use_rel_bias,
                                "cand_self_attn": self.backbone.cand_self_attn,
                                "head_type": self.backbone.head_type,
                                "enable_early_shape_hint": self.backbone.enable_early_shape_hint,
                                "include_room_level": False,
                                "rope_base": self.backbone.rope_base,
                                "checkpointing": False,
                                "device": torch.device("cpu"),
                                "dtype": save_dtype,
                            },
                        )
                    ],
                    {
                        "in_keys": ["sparse", "numeric", "progression", "candidates"],
                        "out_keys": ["encode"],
                    },
                ),
                dump_object(
                    head_tdm,
                    [
                        dump_object(
                            self.policy_head,
                            [],
                            {
                                "input_dimension": self.policy_head.input_dimension,
                                "hidden_dimension": self.policy_head.hidden_dimension,
                                "head_type": self.policy_head.head_type,
                                "device": torch.device("cpu"),
                                "dtype": save_dtype,
                            },
                        )
                    ],
                    {"in_keys": ["candidates", "encode"], "out_keys": ["decode"]},
                ),
                dump_object(
                    lp_tdm,
                    [dump_object(lp_mod, ["log_probs"], {})],
                    {"in_keys": ["candidates", "decode"], "out_keys": ["log_probs"]},
                ),
                dump_object(
                    arg_tdm,
                    [dump_object(arg_mod, ["argmax"], {"tie_break_by_action_id": True})],
                    {"in_keys": ["candidates", "log_probs"], "out_keys": ["action"]},
                ),
            ],
            {},
        )
        torch.save(state, output_path)


def save_trajectory_bc_checkpoint(
    ckpt_dir: Path,
    model: TrajectoryBCModel,
    optimizer: torch.optim.Optimizer,
    *,
    sampler: LengthBucketBatchSampler,
    step: int,
    recipe: dict[str, Any],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_scaler: GradScaler | None = None,
    aug_generator: torch.Generator | None = None,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "recipe": recipe,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler": grad_scaler.state_dict() if grad_scaler is not None and grad_scaler.is_enabled() else None,
        "sampler": sampler.state_dict(),
        "rng": capture_rng(),
        "aug_rng": aug_generator.get_state() if aug_generator is not None else None,
    }
    ckpt_path = ckpt_dir / f"traj-bc-{step}.pt"
    atomic_save(ckpt_path, payload)
    model.dump_actor(ckpt_dir / f"model.{step}.kanachan")
    return ckpt_path


def load_trajectory_bc_checkpoint(
    ckpt_path: Path,
    model: TrajectoryBCModel,
    optimizer: torch.optim.Optimizer,
    *,
    expected_recipe: dict[str, Any],
    sampler: LengthBucketBatchSampler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_scaler: GradScaler | None = None,
    aug_generator: torch.Generator | None = None,
    map_location: torch.device | str = "cpu",
) -> int:
    payload = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    if payload.get("recipe") != expected_recipe:
        raise ValueError("Trajectory BC resume recipe/manifest/hyperparameters differ (Fail-Closed).")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if grad_scaler is not None and grad_scaler.is_enabled() and payload.get("grad_scaler") is not None:
        grad_scaler.load_state_dict(payload["grad_scaler"])
    if sampler is not None:
        sampler.load_state_dict(payload["sampler"])
    if payload.get("rng") is not None:
        restore_rng(payload["rng"])
    if aug_generator is not None and payload.get("aug_rng") is not None:
        aug_generator.set_state(payload["aug_rng"])
    return int(payload["step"])


def train_trajectory_bc_steps(
    model: TrajectoryBCModel,
    rounds: Sequence[RoundTrajectory],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int = 8,
    max_steps: int | None = None,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    suit_augmentation: bool = False,
    aug_seed: int = 1337,
    sampler_seed: int = 42,
    grad_clip: float = 1.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_scaler: GradScaler | None = None,
    ckpt_dir: Path | None = None,
    checkpoint_every: int = 0,
    resume_from: Path | None = None,
    data_sha256: str = "",
) -> dict[str, Any]:
    """Deterministic single/multi-step round-trajectory BC training loop with exact resume."""
    sampler = LengthBucketBatchSampler(
        [r.round_event_len for r in rounds], batch_size=batch_size, seed=sampler_seed
    )
    aug_gen: torch.Generator | None = None
    if suit_augmentation:
        aug_gen = torch.Generator(device="cpu")
        aug_gen.manual_seed(aug_seed)

    base_lr = float(optimizer.param_groups[0]["lr"])
    base_wd = float(optimizer.param_groups[0].get("weight_decay", 0.0))
    recipe = model.recipe(data_sha256=data_sha256, lr=base_lr, weight_decay=base_wd)

    start_step = 0
    if resume_from is not None:
        start_step = load_trajectory_bc_checkpoint(
            resume_from,
            model,
            optimizer,
            expected_recipe=recipe,
            sampler=sampler,
            scheduler=scheduler,
            grad_scaler=grad_scaler,
            aug_generator=aug_gen,
            map_location=device,
        )

    is_amp = amp_dtype is not None and device.type == "cuda" and amp_dtype != torch.float32
    scaler = grad_scaler or GradScaler("cuda", enabled=(is_amp and amp_dtype == torch.float16))

    model.train()
    step = start_step
    last_metrics: dict[str, float] = {}
    while max_steps is None or step < max_steps:
        for idxs in sampler:
            if max_steps is not None and step >= max_steps:
                break
            batch_rounds = [rounds[i] for i in idxs]
            batch_cpu = collate_round_batch(batch_rounds, suit_perm_generator=aug_gen)
            batch_dev = {k: v.to(device=device, non_blocking=True) for k, v in batch_cpu.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=bool(is_amp),
            ):
                out = model(batch_dev)
                loss = out["loss"]

            if not math.isfinite(float(loss.item())):
                raise RuntimeError("Trajectory BC training loss became non-finite.")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            old_scale = scaler.get_scale() if scaler.is_enabled() else None
            scaler.step(optimizer)
            scaler.update()
            step_skipped = old_scale is not None and scaler.get_scale() < old_scale
            if not step_skipped and scheduler is not None:
                scheduler.step()

            step += 1
            last_metrics = {k: float(v.item()) for k, v in out.items()}
            if ckpt_dir is not None and checkpoint_every > 0 and step % checkpoint_every == 0:
                save_trajectory_bc_checkpoint(
                    ckpt_dir, model, optimizer, sampler=sampler, step=step,
                    recipe=recipe, scheduler=scheduler, grad_scaler=scaler,
                    aug_generator=aug_gen,
                )
        if max_steps is None:
            break
        if step < max_steps:
            sampler.reset_epoch(sampler.epoch + 1)

    return {"step": step, "metrics": last_metrics}


__all__ = [
    "ACTION_FAMILY_NAMES",
    "LengthBucketBatchSampler",
    "RoundTrajectory",
    "TrajectoryBCModel",
    "build_round_trajectories_from_raw",
    "classify_action_id",
    "collate_round_batch",
    "evaluate_9_action_families",
    "load_trajectory_bc_checkpoint",
    "parse_bc_rounds",
    "save_trajectory_bc_checkpoint",
    "train_trajectory_bc_steps",
]
