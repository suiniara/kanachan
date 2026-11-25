"""Joint Shared-Backbone Offline IQL + AWR and Optional Quantile CQL (§4.4)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import Tensor, nn
from torch.nn import functional as F

from kanachan.constants import MAX_NUM_ACTION_CANDIDATES, NUM_TYPES_OF_ACTIONS
from kanachan.model_loader import dump_model, dump_object
from kanachan.nn.decode_converter import DecodeConverter
from kanachan.nn.decision_decoder import require_tensor
from kanachan.nn.modern_encoder import RMSNorm
from kanachan.nn.structured_dual_decoder import (
    StructuredActionHead,
    StructuredDualDecoder,
    StructuredStateHead,
    _init_linear,
)
from kanachan.structured_v2 import PAD_ACTION_ID
from kanachan.training.core.final_rank import final_rank_loss
from kanachan.training.online.state import atomic_save


V2_OBS_KEYS: tuple[str, ...] = (
    "v2_tile_feats",
    "v2_events",
    "v2_event_len",
    "v2_global_feats",
    "v2_player_feats",
    "v2_cand_ids",
    "v2_cand_mask",
    "v2_cand_mode",
    "v2_cand_dyn",
)


def _fp32_rms_norm(norm: RMSNorm, x: Tensor, out_dtype: torch.dtype) -> Tensor:
    w = norm.weight.to(out_dtype)
    var = x.square().mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + norm.epsilon) * w


def _fp32_linear(lin: nn.Linear, x: Tensor, out_dtype: torch.dtype) -> Tensor:
    return F.linear(
        x,
        lin.weight.to(out_dtype),
        lin.bias.to(out_dtype) if lin.bias is not None else None,
    )


class DuelingQHead(nn.Module):
    """Mean-subtracted dueling Q head over (h_val, h_cand) in FP32 (§4.4 Track B)."""

    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        hdim = int(hidden_dimension or input_dimension)
        self.input_dimension = input_dimension
        self.hidden_dimension = hdim

        self.v_norm = RMSNorm(input_dimension, device=device, dtype=dtype, epsilon=1e-6)
        self.v_fc1 = nn.Linear(input_dimension, hdim, device=device, dtype=dtype)
        self.v_fc2 = nn.Linear(hdim, 1, device=device, dtype=dtype)

        self.a_norm = RMSNorm(input_dimension, device=device, dtype=dtype, epsilon=1e-6)
        self.a_fc1 = nn.Linear(input_dimension, hdim, device=device, dtype=dtype)
        self.a_fc2 = nn.Linear(hdim, 1, device=device, dtype=dtype)

        for lin in (self.v_fc1, self.v_fc2, self.a_fc1, self.a_fc2):
            _init_linear(lin)

    @torch.no_grad()
    def zero_output(self) -> None:
        for lin in (self.v_fc2, self.a_fc2):
            lin.weight.zero_()
            if lin.bias is not None:
                lin.bias.zero_()

    def forward(self, candidates: Tensor, encode: Tensor) -> Tensor:
        cand_mask = candidates < NUM_TYPES_OF_ACTIONS
        out_dtype = torch.float64 if encode.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=encode.device.type, enabled=False):
            h_val = encode[:, 0].to(out_dtype)
            h_cand = encode[:, 1:].to(out_dtype)

            hv = _fp32_rms_norm(self.v_norm, h_val, out_dtype)
            hv = F.silu(_fp32_linear(self.v_fc1, hv, out_dtype))
            v = _fp32_linear(self.v_fc2, hv, out_dtype)  # [B, 1]

            ha = _fp32_rms_norm(self.a_norm, h_cand, out_dtype)
            ha = F.silu(_fp32_linear(self.a_fc1, ha, out_dtype))
            adv = _fp32_linear(self.a_fc2, ha, out_dtype).squeeze(-1)  # [B, 32]

            mask_f = cand_mask.to(out_dtype)
            adv_masked = adv * mask_f
            adv_mean = adv_masked.sum(dim=-1, keepdim=True) / mask_f.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
            q = v + (adv_masked - adv_mean)
            return torch.where(cand_mask, q, torch.zeros_like(q))


class QuantileQHead(nn.Module):
    """Distributional M-quantile Q head (M=32) for QR-DQN + CQL (§4.4.4)."""

    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int | None = None,
        num_quantiles: int = 32,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        hdim = int(hidden_dimension or input_dimension)
        self.input_dimension = input_dimension
        self.hidden_dimension = hdim
        self.num_quantiles = int(num_quantiles)

        self.norm = RMSNorm(2 * input_dimension, device=device, dtype=dtype, epsilon=1e-6)
        self.fc1 = nn.Linear(2 * input_dimension, hdim, device=device, dtype=dtype)
        self.fc2 = nn.Linear(hdim, self.num_quantiles, device=device, dtype=dtype)
        _init_linear(self.fc1)
        _init_linear(self.fc2)

    @torch.no_grad()
    def zero_output(self) -> None:
        self.fc2.weight.zero_()
        if self.fc2.bias is not None:
            self.fc2.bias.zero_()

    def forward(self, candidates: Tensor, encode: Tensor) -> Tensor:
        cand_mask = candidates < NUM_TYPES_OF_ACTIONS
        out_dtype = torch.float64 if encode.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=encode.device.type, enabled=False):
            h_val = encode[:, 0:1].to(out_dtype).expand(-1, MAX_NUM_ACTION_CANDIDATES, -1)
            h_cand = encode[:, 1:].to(out_dtype)
            cat = torch.cat([h_val, h_cand], dim=-1)
            h = _fp32_rms_norm(self.norm, cat, out_dtype)
            h = F.silu(_fp32_linear(self.fc1, h, out_dtype))
            quantiles = _fp32_linear(self.fc2, h, out_dtype)  # [B, 32, M]
            return torch.where(cand_mask.unsqueeze(-1), quantiles, torch.zeros_like(quantiles))


def compute_qr_dqn_target_quantiles(
    reward: Tensor,
    done: Tensor,
    discount: Tensor | float,
    next_candidates: Tensor,
    next_target_quantiles: Tensor,
    next_selector_quantiles: Tensor | None = None,
) -> Tensor:
    """Construct [B, M] distributional Bellman target quantiles in FP32.

    Computes ``reward + discount * (~done) * target_quantiles(s_{t+1}, a*)``
    where ``a*`` maximizes mean predicted Q over legal ``next_candidates < 546``
    with deterministic minimum-``global_id`` tie-breaking (matching
    ``DecodeConverter("argmax", tie_break_by_action_id=True)``).
    When ``next_selector_quantiles`` is supplied (online QuantileQHead), performs
    Double QR-DQN selection while gathering quantiles from ``next_target_quantiles``.
    """
    dev = next_target_quantiles.device
    with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
        r_f = reward.float()
        done_b = done.bool()
        disc_f = (
            discount.float()
            if isinstance(discount, Tensor)
            else torch.full_like(r_f, float(discount))
        )
        non_term = (~done_b) & (disc_f != 0.0)
        legal_next = next_candidates < NUM_TYPES_OF_ACTIONS
        require_tensor(
            (~non_term | legal_next.any(dim=-1)).all(),
            "Non-terminal next state has zero legal candidates.",
        )

        # Avoid all-False masks on terminal rows before argmax; terminal rows are masked out below.
        fallback_slot0 = F.one_hot(
            torch.zeros_like(done_b, dtype=torch.long), MAX_NUM_ACTION_CANDIDATES
        ).bool()
        safe_legal = torch.where(legal_next.any(dim=-1, keepdim=True), legal_next, fallback_slot0)

        sel_q = (
            next_selector_quantiles
            if next_selector_quantiles is not None
            else next_target_quantiles
        ).float().mean(dim=-1)  # [B, 32]
        masked_sel = sel_q.masked_fill(~safe_legal, -torch.inf)
        row_max = masked_sel.max(dim=-1, keepdim=True).values
        tied = safe_legal & (sel_q == row_max)
        tie_ids = torch.where(
            tied,
            next_candidates.long(),
            torch.full_like(next_candidates.long(), PAD_ACTION_ID + 1),
        )
        best_slot = tie_ids.argmin(dim=-1)  # [B]

        tgt_q_all = next_target_quantiles.float()  # [B, 32, M]
        num_q = tgt_q_all.shape[-1]
        next_best_q = tgt_q_all.gather(
            1, best_slot.view(-1, 1, 1).expand(-1, 1, num_q)
        ).squeeze(1)  # [B, M]

        safe_next_q = torch.where(non_term.unsqueeze(1), next_best_q, torch.zeros_like(next_best_q))
        boot = torch.where(non_term.unsqueeze(1), disc_f.unsqueeze(1) * safe_next_q, torch.zeros_like(next_best_q))
        return (r_f.unsqueeze(1) + boot).detach()


def quantile_cql_loss(
    q_quantiles: Tensor,
    candidates: Tensor,
    chosen_idx: Tensor,
    target_returns: Tensor,
    *,
    alpha: float = 0.1,
    temperature: float = 1.0,
    kappa: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Pairwise Quantile Huber loss + mean-Q CQL regularizer strictly over K >= 2.

    Supports both full distributional Bellman target distributions ``[B, M]``
    and scalar 1D Monte-Carlo returns ``[B]``. Includes ``K = 1`` forced actions
    in distributional value regression while masking CQL gap strictly to ``K >= 2``.
    """
    with torch.autocast(device_type=q_quantiles.device.type, enabled=False):
        q_f = q_quantiles.float()  # [B, 32, M]
        bsz, _, num_q = q_f.shape
        b_idx = torch.arange(bsz, device=q_f.device)
        act_l = chosen_idx.long()
        cand_mask = candidates < NUM_TYPES_OF_ACTIONS
        require_tensor(
            ((act_l >= 0) & (act_l < MAX_NUM_ACTION_CANDIDATES)).all(),
            "Action slot index out of range in quantile_cql_loss.",
        )
        require_tensor(
            cand_mask[b_idx, act_l].all(),
            "Chosen action slot points to PAD candidate in quantile_cql_loss.",
        )
        q_chosen = q_f[b_idx, act_l]  # [B, M]

        tgt = target_returns.float()
        if tgt.ndim == 1:
            tgt = tgt.unsqueeze(1)  # [B, 1]
        elif tgt.ndim != 2:
            raise ValueError(f"Expected 1D or 2D target_returns, got shape {tuple(tgt.shape)}.")

        # Pairwise TD error delta[b, m_pred, m_tgt] = target[b, m_tgt] - q_chosen[b, m_pred]
        delta = tgt.unsqueeze(1) - q_chosen.unsqueeze(2)
        abs_d = delta.abs()
        kappa_f = float(kappa)
        if kappa_f < 0.0:
            raise ValueError(f"kappa must be non-negative, got {kappa_f}.")
        if kappa_f > 0.0:
            huber_norm = torch.where(
                abs_d <= kappa_f,
                0.5 * delta.square() / kappa_f,
                abs_d - 0.5 * kappa_f,
            )
        else:
            huber_norm = abs_d

        taus = (
            (torch.arange(num_q, device=q_f.device, dtype=torch.float32) + 0.5)
            / float(num_q)
        ).view(1, num_q, 1)
        weight = torch.abs(taus - (delta.detach() < 0.0).float())
        # Average over target quantiles (dim=2), sum over predicted quantiles (dim=1), mean over batch.
        qr_loss = (weight * huber_norm).mean(dim=2).sum(dim=1).mean()

        q_mean = q_f.mean(dim=-1)  # [B, 32]
        masked_q = q_mean.masked_fill(~cand_mask, -torch.inf)
        q_sa = q_mean[b_idx, act_l]
        cql_gap = (
            temperature * torch.logsumexp(masked_q / temperature, dim=-1) - q_sa
        )
        has_choice = cand_mask.sum(dim=-1) >= 2
        has_f = has_choice.float()
        cql_loss = (cql_gap * has_f).sum() / has_f.sum().clamp_min(1.0)
        total = qr_loss + alpha * cql_loss
        return total, {
            "qr_loss": qr_loss.detach(),
            "cql_loss": cql_loss.detach(),
            "q_mean_chosen": q_sa.detach().mean(),
        }


def _encode_current_from_batch(
    backbone: StructuredDualDecoder,
    batch: TensorDict | dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Run StructuredDualDecoder on current state without GPU->CPU sync when v2_* exist."""
    cands = batch["candidates"]
    if "v2_tile_feats" in batch:
        encode = backbone.forward_structured(
            batch["v2_tile_feats"],
            batch["v2_events"],
            batch["v2_event_len"],
            batch["v2_global_feats"],
            batch["v2_player_feats"],
            batch["v2_cand_ids"],
            batch["v2_cand_mask"],
            batch["v2_cand_mode"],
            batch["v2_cand_dyn"],
        )
    else:
        encode = backbone(
            batch["sparse"],
            batch["numeric"],
            batch["progression"],
            cands,
        )
    assert isinstance(encode, Tensor)
    return encode, cands


def _encode_next_states_zero_sync(
    backbone: StructuredDualDecoder,
    batch: TensorDict | dict[str, Tensor],
    done: Tensor,
) -> tuple[Tensor, Tensor]:
    """Encode next states across the full batch with pure GPU masking and zero CUDA syncs.

    On terminal rows (``done=True``), index tensors in ``batch["next"]`` may contain
    all-padding sentinel values (e.g. ``sparse=510``, ``candidates=546``). We replace
    terminal rows tensorially using ``torch.where`` before entering ``backbone`` so
    all kernels run at static shape ``[B, ...]`` with zero ``.item()`` / boolean-index
    synchronizations, and callers mask terminal outputs by ``~done``.
    """
    nxt = batch["next"]
    done_b = done.bool()
    done_2 = done_b.unsqueeze(1)
    done_3 = done_b.view(-1, 1, 1)

    if "v2_tile_feats" in nxt:
        raw_cands = nxt["v2_cand_ids"]
        dummy_row = F.pad(
            torch.zeros((1, 1), dtype=raw_cands.dtype, device=raw_cands.device),
            (0, MAX_NUM_ACTION_CANDIDATES - 1),
            value=PAD_ACTION_ID,
        )
        cand_ids = torch.where(done_2, dummy_row, raw_cands)
        cand_mask = cand_ids < NUM_TYPES_OF_ACTIONS
        raw_ev = nxt["v2_events"]
        events = torch.where(done_3, torch.zeros_like(raw_ev), raw_ev)
        raw_ev_len = nxt["v2_event_len"]
        ev_len = torch.where(done_b, torch.ones_like(raw_ev_len), raw_ev_len.clamp_min(1))
        tile_feats = torch.where(done_3, torch.zeros_like(nxt["v2_tile_feats"]), nxt["v2_tile_feats"])
        global_feats = torch.where(done_2, torch.zeros_like(nxt["v2_global_feats"]), nxt["v2_global_feats"])
        player_feats = torch.where(done_3, torch.zeros_like(nxt["v2_player_feats"]), nxt["v2_player_feats"])
        cand_mode = torch.where(done_2, torch.zeros_like(nxt["v2_cand_mode"]), nxt["v2_cand_mode"])
        cand_dyn = torch.where(done_3, torch.zeros_like(nxt["v2_cand_dyn"]), nxt["v2_cand_dyn"])
        encode = backbone.forward_structured(
            tile_feats, events, ev_len, global_feats, player_feats,
            cand_ids, cand_mask, cand_mode, cand_dyn,
        )
        assert isinstance(encode, Tensor)
        return encode, cand_ids

    safe_sp = torch.where(done_2, batch["sparse"], nxt["sparse"])
    safe_nu = torch.where(done_2, batch["numeric"], nxt["numeric"])
    safe_pr = torch.where(done_2, batch["progression"], nxt["progression"])
    safe_ca = torch.where(done_2, batch["candidates"], nxt["candidates"])
    encode = backbone(safe_sp, safe_nu, safe_pr, safe_ca)
    assert isinstance(encode, Tensor)
    return encode, safe_ca


class JointStructuredIQLModule(nn.Module):
    """Single shared-backbone StructuredDualDecoder + Policy + V + Twin Dueling Q (§4.4B)."""

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
        final_rank: bool = False,
        final_rank_weight: float = 0.0,
        freeze_decoder1: bool = False,
        freeze_shared_tile_emb: bool | None = None,
        use_cql_quantile_head: bool = False,
        num_quantiles: int = 32,
        qr_double_q: bool = True,
        expectile: float = 0.70,
        discount_factor: float = 1.0,
        awr_inv_temp_beta: float = 3.0,
        awr_max_weight: float = 20.0,
        kl_coef: float = 0.0,
        keep_bc_anchor: bool = False,
        quantile_cql_alpha: float = 0.05,
        cql_temperature: float = 1.0,
        target_ema_tau: float = 0.005,
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
        ht = self.backbone.head_type

        self.policy_head = StructuredActionHead(
            input_dimension=d, head_type=ht, device=device, dtype=dtype
        )
        self.value_head = StructuredStateHead(
            input_dimension=d, final_rank=final_rank, device=device, dtype=dtype
        )
        self.value_head.zero_output()

        self.q1_head = DuelingQHead(input_dimension=d, device=device, dtype=dtype)
        self.q2_head = DuelingQHead(input_dimension=d, device=device, dtype=dtype)

        self.q1_target_head = copy.deepcopy(self.q1_head)
        self.q2_target_head = copy.deepcopy(self.q2_head)
        self.q1_target_head.requires_grad_(False)
        self.q2_target_head.requires_grad_(False)

        self.quantile_q_head: QuantileQHead | None = None
        self.quantile_q_target_head: QuantileQHead | None = None
        if use_cql_quantile_head:
            self.quantile_q_head = QuantileQHead(
                input_dimension=d,
                num_quantiles=num_quantiles,
                device=device,
                dtype=dtype,
            )
            self.quantile_q_target_head = copy.deepcopy(self.quantile_q_head)
            self.quantile_q_target_head.requires_grad_(False)

        self.qr_double_q = bool(qr_double_q)
        self.expectile = float(expectile)
        self.discount_factor = float(discount_factor)
        self.awr_inv_temp_beta = float(awr_inv_temp_beta)
        self.awr_max_weight = float(awr_max_weight)
        self.kl_coef = float(kl_coef)
        self.quantile_cql_alpha = float(quantile_cql_alpha)
        self.cql_temperature = float(cql_temperature)
        self.target_ema_tau = float(target_ema_tau)
        self.final_rank_weight: float = 0.0
        self.freeze_decoder1: bool = False
        self.freeze_shared_tile_emb: bool = False

        self.bc_anchor_backbone: StructuredDualDecoder | None = None
        self.bc_anchor_head: StructuredActionHead | None = None
        if keep_bc_anchor or self.kl_coef > 0.0:
            self._instantiate_bc_anchor(device=device, dtype=dtype)

        self.configure_freeze(
            freeze_decoder1=freeze_decoder1,
            freeze_shared_tile_emb=freeze_shared_tile_emb,
            final_rank_weight=final_rank_weight,
        )

    def _instantiate_bc_anchor(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        ref = next(self.backbone.parameters())
        dev = device if device is not None else ref.device
        dt = dtype if dtype is not None else ref.dtype
        self.bc_anchor_backbone = StructuredDualDecoder(
            preset=self.backbone.preset,
            dimension=self.backbone.dimension,
            num_heads=self.backbone.num_heads,
            dim_feedforward=self.backbone.dim_feedforward,
            n_d1_layers=self.backbone.n_d1_layers,
            n_d2_layers=self.backbone.n_d2_layers,
            use_rel_bias=self.backbone.use_rel_bias,
            cand_self_attn=self.backbone.cand_self_attn,
            head_type=self.backbone.head_type,
            enable_early_shape_hint=self.backbone.enable_early_shape_hint,
            rope_base=self.backbone.rope_base,
            checkpointing=False,
            device=dev,
            dtype=dt,
        ).eval()
        self.bc_anchor_head = StructuredActionHead(
            input_dimension=self.policy_head.input_dimension,
            hidden_dimension=self.policy_head.hidden_dimension,
            head_type=self.policy_head.head_type,
            device=dev,
            dtype=dt,
        ).eval()
        self.bc_anchor_backbone.requires_grad_(False)
        self.bc_anchor_head.requires_grad_(False)

    @staticmethod
    def _backbone_sig(bb: StructuredDualDecoder) -> tuple[Any, ...]:
        return (
            bb.dimension,
            bb.num_heads,
            bb.dim_feedforward,
            bb.n_d1_layers,
            bb.n_d2_layers,
            bb.use_rel_bias,
            bb.cand_self_attn,
            bb.head_type,
            bb.enable_early_shape_hint,
            bb.rope_base,
        )

    @torch.no_grad()
    def init_from_bc(
        self,
        bc_backbone: StructuredDualDecoder,
        bc_policy_head: StructuredActionHead,
        *,
        keep_bc_anchor: bool = False,
    ) -> None:
        if self._backbone_sig(self.backbone) != self._backbone_sig(bc_backbone):
            raise ValueError("Structured V2 BC and IQL backbone architectures differ.")
        if (
            self.policy_head.head_type != bc_policy_head.head_type
            or self.policy_head.hidden_dimension != bc_policy_head.hidden_dimension
        ):
            raise ValueError("Structured V2 BC and IQL policy head configurations differ.")
        ref = next(self.backbone.parameters())
        self.backbone.load_state_dict(
            {k: v.to(device=ref.device, dtype=ref.dtype) for k, v in bc_backbone.state_dict().items()},
            strict=True,
        )
        p_ref = next(self.policy_head.parameters())
        self.policy_head.load_state_dict(
            {k: v.to(device=p_ref.device, dtype=p_ref.dtype) for k, v in bc_policy_head.state_dict().items()},
            strict=True,
        )
        self.value_head.zero_output()
        if keep_bc_anchor or self.kl_coef > 0.0:
            if self.bc_anchor_backbone is None or self.bc_anchor_head is None:
                self._instantiate_bc_anchor(device=ref.device, dtype=ref.dtype)
            assert self.bc_anchor_backbone is not None and self.bc_anchor_head is not None
            self.bc_anchor_backbone.load_state_dict(self.backbone.state_dict(), strict=True)
            self.bc_anchor_head.load_state_dict(self.policy_head.state_dict(), strict=True)
            self.bc_anchor_backbone.eval().requires_grad_(False)
            self.bc_anchor_head.eval().requires_grad_(False)

    def configure_freeze(
        self,
        *,
        freeze_decoder1: bool = False,
        freeze_shared_tile_emb: bool | None = None,
        final_rank_weight: float = 0.0,
    ) -> None:
        """Apply Stage-2 Decoder 1 freeze & shared tile_emb guard (§4.4.3)."""
        if freeze_decoder1 and freeze_shared_tile_emb is None:
            freeze_shared_tile_emb = True
        resolved_tile_freeze = bool(
            freeze_shared_tile_emb if freeze_shared_tile_emb is not None else False
        )
        self.freeze_decoder1 = bool(freeze_decoder1)
        self.freeze_shared_tile_emb = resolved_tile_freeze
        self.final_rank_weight = float(final_rank_weight)

        tile_ids = {id(p) for p in self.backbone.tile_emb.parameters()}
        for p in self.backbone.decoder1.parameters():
            if id(p) in tile_ids:
                continue
            p.requires_grad_(not self.freeze_decoder1)
        for p in self.backbone.tile_emb.parameters():
            p.requires_grad_(not self.freeze_shared_tile_emb)

        if self.value_head.final_rank is not None:
            self.value_head.final_rank.requires_grad_(self.final_rank_weight > 0.0)

    def parameter_groups(
        self,
        *,
        base_lr: float,
        weight_decay: float,
        decoder1_lr_scale: float = 0.25,
    ) -> list[dict[str, Any]]:
        """Split AdamW groups with _no_weight_decay_param_ids and decoder1_lr_scale."""
        no_decay: set[int] = {
            id(p)
            for layer in self.modules()
            if isinstance(layer, nn.Embedding)
            for p in layer.parameters(recurse=False)
        }
        for layer in self.modules():
            hook = getattr(layer, "_no_weight_decay_param_ids", None)
            if callable(hook):
                no_decay.update(hook())

        tile_ids = {id(p) for p in self.backbone.tile_emb.parameters()}
        d1_non_shared = {
            id(p)
            for p in self.backbone.decoder1.parameters()
            if id(p) not in tile_ids
        }

        main_decay, main_no_decay, d1_decay, d1_no_decay = [], [], [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            is_decay = p.ndim >= 2 and id(p) not in no_decay
            is_d1 = id(p) in d1_non_shared
            if is_d1:
                (d1_decay if is_decay else d1_no_decay).append(p)
            else:
                (main_decay if is_decay else main_no_decay).append(p)

        groups: list[dict[str, Any]] = []
        if main_decay:
            groups.append({"params": main_decay, "lr": base_lr, "weight_decay": weight_decay})
        if main_no_decay:
            groups.append({"params": main_no_decay, "lr": base_lr, "weight_decay": 0.0})
        d1_lr = float(base_lr) * float(decoder1_lr_scale)
        if d1_decay:
            groups.append({"params": d1_decay, "lr": d1_lr, "weight_decay": weight_decay})
        if d1_no_decay:
            groups.append({"params": d1_no_decay, "lr": d1_lr, "weight_decay": 0.0})
        return groups

    def _target_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        pairs: list[tuple[nn.Module, nn.Module]] = [
            (self.q1_head, self.q1_target_head),
            (self.q2_head, self.q2_target_head),
        ]
        if self.quantile_q_head is not None and self.quantile_q_target_head is not None:
            pairs.append((self.quantile_q_head, self.quantile_q_target_head))
        return tuple(pairs)

    @torch.no_grad()
    def update_targets(self, tau: float | None = None) -> None:
        t = float(self.target_ema_tau if tau is None else tau)
        for src, tgt in self._target_pairs():
            for ps, pt in zip(src.parameters(), tgt.parameters(), strict=True):
                pt.data.mul_(1.0 - t).add_(ps.data, alpha=t)

    @torch.no_grad()
    def sync_targets_ddp(self) -> None:
        """Broadcast Rank-0 target heads across DDP ranks, preserving Polyak lag."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        for _, tgt in self._target_pairs():
            for p in tgt.parameters():
                torch.distributed.broadcast(p.data, src=0)
            for b in tgt.buffers():
                torch.distributed.broadcast(b.data, src=0)

    def forward(
        self,
        batch: TensorDict | dict[str, Tensor],
        *,
        amp_dtype: torch.dtype | None = None,
    ) -> dict[str, Tensor]:
        device = next(self.parameters()).device
        is_amp = (
            amp_dtype is not None
            and device.type == "cuda"
            and amp_dtype != self.policy_head.mlp[0].weight.dtype
            if self.policy_head.mlp is not None
            else (amp_dtype is not None and device.type == "cuda")
        )
        encode, cands = _encode_current_from_batch(self.backbone, batch)
        action = batch["action"].long()
        nxt = batch["next"]
        done = nxt["done"].bool()
        reward = nxt["reward"].float()
        if "discount" in nxt:
            gamma_t = nxt["discount"].float()
        else:
            gamma_t = torch.full_like(reward, self.discount_factor)

        bsz = int(action.shape[0])
        b_idx = torch.arange(bsz, device=encode.device)

        logits = self.policy_head(cands, encode)  # FP32 [B, 32]
        v_out = self.value_head(encode)
        if isinstance(v_out, tuple):
            v_pred, rank_logits = v_out
        else:
            v_pred, rank_logits = v_out, None
        q1_all = self.q1_head(cands, encode)
        q2_all = self.q2_head(cands, encode)

        # No-grad target Q, next-state V(s_{t+1}), and QR-DQN target distribution (zero GPU sync).
        with torch.no_grad(), torch.autocast(
            device_type=encode.device.type,
            dtype=amp_dtype or torch.float32,
            enabled=False,
            cache_enabled=False,
        ):
            enc_det = encode.detach()
            q1_tgt = self.q1_target_head(cands, enc_det)
            q2_tgt = self.q2_target_head(cands, enc_det)
            q_hat = torch.minimum(q1_tgt, q2_tgt)[b_idx, action].float().detach()

            non_term = (~done) & (gamma_t != 0.0)
            need_next = (
                (self.discount_factor != 0.0)
                or ("discount" in nxt)
                or (self.quantile_q_head is not None)
            )
            next_v = torch.zeros_like(reward)
            qr_target_quantiles: Tensor | None = None
            if need_next:
                with torch.autocast(
                    device_type=encode.device.type,
                    dtype=amp_dtype or torch.float32,
                    enabled=bool(is_amp),
                    cache_enabled=False,
                ):
                    next_enc, next_cands = _encode_next_states_zero_sync(
                        self.backbone, batch, done
                    )
                nv_out = self.value_head(next_enc)
                nv = nv_out[0] if isinstance(nv_out, tuple) else nv_out
                nv_f = nv.float()
                next_v = torch.where(non_term, nv_f, torch.zeros_like(nv_f)).detach()

                if self.quantile_q_head is not None and self.quantile_q_target_head is not None:
                    next_tgt_q = self.quantile_q_target_head(next_cands, next_enc)
                    next_sel_q = (
                        self.quantile_q_head(next_cands, next_enc)
                        if self.qr_double_q
                        else None
                    )
                    qr_target_quantiles = compute_qr_dqn_target_quantiles(
                        reward,
                        done,
                        gamma_t,
                        next_cands,
                        next_tgt_q,
                        next_sel_q,
                    )

            td_target = torch.where(
                non_term, reward + gamma_t * next_v, reward
            ).detach()

            bc_log_pi: Tensor | None = None
            if (
                self.kl_coef > 0.0
                and self.bc_anchor_backbone is not None
                and self.bc_anchor_head is not None
            ):
                with torch.autocast(
                    device_type=encode.device.type,
                    dtype=amp_dtype or torch.float32,
                    enabled=bool(is_amp),
                    cache_enabled=False,
                ):
                    bc_enc, _ = _encode_current_from_batch(self.bc_anchor_backbone, batch)
                bc_logits = self.bc_anchor_head(cands, bc_enc).float()
                cm_bc = cands < NUM_TYPES_OF_ACTIONS
                bc_lp = F.log_softmax(bc_logits.masked_fill(~cm_bc, -torch.inf), dim=-1)
                bc_log_pi = torch.where(cm_bc, bc_lp, torch.zeros_like(bc_lp)).detach()

        with torch.autocast(device_type=encode.device.type, enabled=False):
            # 1. Expectile Value loss
            v_f = v_pred.float()
            u = q_hat - v_f
            loss_v = (torch.abs(self.expectile - (u < 0).float()) * u.square()).mean()
            if self.final_rank_weight > 0.0 and rank_logits is not None:
                if "results" in nxt and "sparse" in batch:
                    loss_rank = final_rank_loss(rank_logits.float(), batch)
                elif "final_rank" in nxt:
                    eog = nxt["end_of_game"].bool() if "end_of_game" in nxt else done
                    rk_tgt = nxt["final_rank"].long()
                    require_tensor(
                        (~eog | ((rk_tgt >= 0) & (rk_tgt < 4))).all(),
                        "Invalid next.final_rank; expected [0, 3] on end_of_game rows.",
                    )
                    rk_ce = F.cross_entropy(rank_logits.float(), rk_tgt.clamp(0, 3), reduction="none")
                    loss_rank = rk_ce.masked_fill(~eog, 0.0).mean()
                else:
                    loss_rank = rank_logits.float().sum() * 0.0
                loss_v = loss_v + self.final_rank_weight * loss_rank

            # 2. TD Action Value loss
            q1_sa = q1_all.float()[b_idx, action]
            q2_sa = q2_all.float()[b_idx, action]
            loss_q = 0.5 * (F.mse_loss(q1_sa, td_target) + F.mse_loss(q2_sa, td_target))

            # 3. AWR Policy extraction (+ optional BC KL) strictly masked on K >= 2
            cand_mask = cands < NUM_TYPES_OF_ACTIONS
            masked_logits = logits.float().masked_fill(~cand_mask, -torch.inf)
            log_pi = F.log_softmax(masked_logits, dim=-1)
            log_pi_safe = torch.where(cand_mask, log_pi, torch.zeros_like(log_pi))
            chosen_lp = log_pi_safe[b_idx, action]

            adv = q_hat - v_f.detach()
            weights = torch.clamp(
                torch.exp(torch.clamp(self.awr_inv_temp_beta * adv, max=10.0)),
                max=self.awr_max_weight,
            ).detach()
            per_step_pi = -weights * chosen_lp
            if bc_log_pi is not None and self.kl_coef > 0.0:
                probs_safe = torch.where(cand_mask, torch.exp(log_pi_safe), torch.zeros_like(log_pi_safe))
                kl_step = (probs_safe * (log_pi_safe - bc_log_pi)).sum(dim=-1)
                per_step_pi = per_step_pi + self.kl_coef * kl_step

            has_choice = cand_mask.sum(dim=-1) >= 2
            has_f = has_choice.float()
            loss_pi = (per_step_pi * has_f).sum() / has_f.sum().clamp_min(1.0)

            loss_total = loss_v + loss_q + loss_pi
            qr_metrics: dict[str, Tensor] = {}
            if self.quantile_q_head is not None:
                qq = self.quantile_q_head(cands, encode)
                tgt_dist = qr_target_quantiles if qr_target_quantiles is not None else td_target
                loss_qr, qr_metrics = quantile_cql_loss(
                    qq,
                    cands,
                    action,
                    tgt_dist,
                    alpha=self.quantile_cql_alpha,
                    temperature=self.cql_temperature,
                    kappa=1.0,
                )
                loss_total = loss_total + loss_qr
                qr_metrics["loss_qr_cql"] = loss_qr.detach()

            # AWR diagnostics (pure tensorized, zero CUDA sync)
            w_masked = torch.where(has_choice, weights, torch.zeros_like(weights))
            n_choice = has_f.sum()
            ess_num = w_masked.sum().square()
            ess_den = (n_choice * w_masked.square().sum()).clamp_min(1e-12)
            ess_ratio = torch.where(n_choice > 0.0, ess_num / ess_den, torch.ones_like(ess_num))
            clip_hits = ((weights >= self.awr_max_weight - 1e-6).float() * has_f).sum()
            clip_frac = torch.where(
                n_choice > 0.0, clip_hits / n_choice.clamp_min(1.0), torch.zeros_like(clip_hits)
            )

        out: dict[str, Tensor] = {
            "loss": loss_total,
            "loss_v": loss_v.detach(),
            "loss_q": loss_q.detach(),
            "loss_pi": loss_pi.detach(),
            "v_mean": v_f.detach().mean(),
            "q_hat_mean": q_hat.mean(),
            "awr_ess_ratio": ess_ratio.detach(),
            "awr_clip_frac": clip_frac.detach(),
            "num_choice_steps": has_f.sum().detach(),
        }
        out.update(qr_metrics)
        return out

    def export_actor_module(self) -> TensorDictSequential:
        """Build 4-field stateless TensorDictSequential actor compatible with DecisionMaker."""
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
        lp_tdm = TensorDictModule(
            DecodeConverter("log_probs"),
            in_keys=["candidates", "decode"],
            out_keys=["log_probs"],
        )
        arg_tdm = TensorDictModule(
            DecodeConverter("argmax", tie_break_by_action_id=True),
            in_keys=["candidates", "log_probs"],
            out_keys=["action"],
        )
        return TensorDictSequential(enc_tdm, head_tdm, lp_tdm, arg_tdm)

    def dump_actor(self, output_path: Any, *, dtype: torch.dtype | None = None) -> None:
        """Serialize portable model.kanachan readable by load_model(..., weights_only=True)."""
        actor = self.export_actor_module()
        save_dtype = dtype or next(self.backbone.parameters()).dtype
        state = dump_model(
            actor,
            [
                dump_object(
                    actor[0],
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
                    actor[1],
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
                    actor[2],
                    [dump_object(actor[2].module, ["log_probs"], {})],
                    {"in_keys": ["candidates", "decode"], "out_keys": ["log_probs"]},
                ),
                dump_object(
                    actor[3],
                    [
                        dump_object(
                            actor[3].module,
                            ["argmax"],
                            {"tie_break_by_action_id": True},
                        )
                    ],
                    {"in_keys": ["candidates", "log_probs"], "out_keys": ["action"]},
                ),
            ],
            {},
        )
        atomic_save(Path(output_path), state)


__all__ = [
    "DuelingQHead",
    "JointStructuredIQLModule",
    "QuantileQHead",
    "V2_OBS_KEYS",
    "compute_qr_dqn_target_quantiles",
    "quantile_cql_loss",
]
