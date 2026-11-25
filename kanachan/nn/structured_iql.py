"""Joint Shared-Backbone Offline IQL + AWR and Optional Quantile CQL (§4.4)."""

from __future__ import annotations

import copy
from typing import Any

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import Tensor, nn
from torch.nn import functional as F

from kanachan.constants import MAX_NUM_ACTION_CANDIDATES, NUM_TYPES_OF_ACTIONS
from kanachan.model_loader import dump_model, dump_object
from kanachan.nn.decode_converter import DecodeConverter
from kanachan.nn.modern_encoder import RMSNorm
from kanachan.nn.structured_dual_decoder import (
    StructuredActionHead,
    StructuredDualDecoder,
    StructuredStateHead,
    _init_linear,
)


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
    """Quantile Huber loss (kappa=1.0) + mean-Q CQL loss strictly over K >= 2."""
    with torch.autocast(device_type=q_quantiles.device.type, enabled=False):
        q_f = q_quantiles.float()  # [B, 32, M]
        bsz, _, num_q = q_f.shape
        b_idx = torch.arange(bsz, device=q_f.device)
        q_chosen = q_f[b_idx, chosen_idx.long()]  # [B, M]

        tgt = target_returns.float()
        if tgt.ndim == 1:
            tgt = tgt.unsqueeze(1)  # [B, 1]
        # delta[b, m, m'] = target[b, m'] - q_chosen[b, m]
        delta = tgt.unsqueeze(1) - q_chosen.unsqueeze(2)
        abs_d = delta.abs()
        huber = torch.where(
            abs_d <= kappa,
            0.5 * delta.square(),
            kappa * (abs_d - 0.5 * kappa),
        )
        taus = (
            (torch.arange(num_q, device=q_f.device, dtype=torch.float32) + 0.5)
            / float(num_q)
        ).view(1, num_q, 1)
        weight = torch.abs(taus - (delta < 0).float())
        qr_loss = (weight * huber / max(kappa, 1e-6)).sum(dim=1).mean()

        q_mean = q_f.mean(dim=-1)  # [B, 32]
        cand_mask = candidates < NUM_TYPES_OF_ACTIONS
        masked_q = q_mean.masked_fill(~cand_mask, -torch.inf)
        q_sa = q_mean[b_idx, chosen_idx.long()]
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


def _backbone_encode_from_batch(
    backbone: StructuredDualDecoder,
    batch: TensorDict | dict[str, Tensor],
    *,
    prefix: tuple[str, ...] | None = None,
    subset_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run StructuredDualDecoder without GPU->CPU sync when v2_* keys exist."""
    def _get(k: str) -> Tensor:
        val = batch[prefix + (k,)] if prefix else batch[k]
        return val[subset_mask] if subset_mask is not None else val

    cands = _get("candidates")
    if (
        (prefix is None and "v2_tile_feats" in batch)
        or (prefix is not None and (prefix + ("v2_tile_feats",)) in batch)
    ):
        encode = backbone.forward_structured(
            _get("v2_tile_feats"),
            _get("v2_events"),
            _get("v2_event_len"),
            _get("v2_global_feats"),
            _get("v2_player_feats"),
            _get("v2_cand_ids"),
            _get("v2_cand_mask"),
            _get("v2_cand_mode"),
            _get("v2_cand_dyn"),
        )
    else:
        encode = backbone(
            _get("sparse"),
            _get("numeric"),
            _get("progression"),
            cands,
        )
    assert isinstance(encode, Tensor)
    return encode, cands


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
        use_cql_quantile_head: bool = False,
        num_quantiles: int = 32,
        expectile: float = 0.70,
        discount_factor: float = 1.0,
        awr_inv_temp_beta: float = 3.0,
        awr_max_weight: float = 20.0,
        kl_coef: float = 0.0,
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
        if use_cql_quantile_head:
            self.quantile_q_head = QuantileQHead(
                input_dimension=d,
                num_quantiles=num_quantiles,
                device=device,
                dtype=dtype,
            )

        self.bc_anchor_backbone: StructuredDualDecoder | None = None
        self.bc_anchor_head: StructuredActionHead | None = None

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

        # Ensure optional auxiliary heads start unused-safe if inactive.
        if self.value_head.final_rank is not None:
            self.value_head.final_rank.requires_grad_(False)

    @torch.no_grad()
    def init_from_bc(
        self,
        bc_backbone: StructuredDualDecoder,
        bc_policy_head: StructuredActionHead,
        *,
        keep_bc_anchor: bool = False,
    ) -> None:
        self.backbone.load_state_dict(bc_backbone.state_dict())
        self.policy_head.load_state_dict(bc_policy_head.state_dict())
        self.value_head.zero_output()
        if keep_bc_anchor or self.kl_coef > 0.0:
            self.bc_anchor_backbone = copy.deepcopy(bc_backbone).eval()
            self.bc_anchor_head = copy.deepcopy(bc_policy_head).eval()
            self.bc_anchor_backbone.requires_grad_(False)
            self.bc_anchor_head.requires_grad_(False)

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

    @torch.no_grad()
    def update_targets(self, tau: float | None = None) -> None:
        t = float(self.target_ema_tau if tau is None else tau)
        for src, tgt in ((self.q1_head, self.q1_target_head), (self.q2_head, self.q2_target_head)):
            for ps, pt in zip(src.parameters(), tgt.parameters(), strict=True):
                pt.data.mul_(1.0 - t).add_(ps.data, alpha=t)

    @torch.no_grad()
    def sync_targets_ddp(self) -> None:
        """Broadcast Rank-0 target heads across DDP ranks, preserving Polyak lag."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        for head in (self.q1_target_head, self.q2_target_head):
            for p in head.parameters():
                torch.distributed.broadcast(p.data, src=0)
            for b in head.buffers():
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
        encode, cands = _backbone_encode_from_batch(self.backbone, batch)
        action = batch["action"].long()
        done = batch["next", "done"].bool()
        reward = batch["next", "reward"].float()
        if ("next", "discount") in batch:
            gamma_t = batch["next", "discount"].float()
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

        # No-grad target Q on logged chosen action
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

            # Non-terminal filtered next-state V(s_{t+1}) under cache_enabled=False
            next_v = torch.zeros_like(reward)
            non_term = (~done) & (gamma_t != 0)
            if bool(non_term.any().item()):
                with torch.autocast(
                    device_type=encode.device.type,
                    dtype=amp_dtype or torch.float32,
                    enabled=bool(is_amp),
                    cache_enabled=False,
                ):
                    next_enc, _ = _backbone_encode_from_batch(
                        self.backbone, batch, prefix=("next",), subset_mask=non_term
                    )
                nv_out = self.value_head(next_enc)
                nv = nv_out[0] if isinstance(nv_out, tuple) else nv_out
                next_v[non_term] = nv.float().detach()
            td_target = (reward + gamma_t * (~done).float() * next_v).detach()

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
                    bc_enc, _ = _backbone_encode_from_batch(self.bc_anchor_backbone, batch)
                bc_logits = self.bc_anchor_head(cands, bc_enc).float()
                cm_bc = cands < NUM_TYPES_OF_ACTIONS
                bc_lp = F.log_softmax(bc_logits.masked_fill(~cm_bc, -torch.inf), dim=-1)
                bc_log_pi = torch.where(cm_bc, bc_lp, torch.zeros_like(bc_lp)).detach()

        with torch.autocast(device_type=encode.device.type, enabled=False):
            # 1. Expectile Value loss
            v_f = v_pred.float()
            u = q_hat - v_f
            loss_v = (torch.abs(self.expectile - (u < 0).float()) * u.square()).mean()
            if (
                self.final_rank_weight > 0.0
                and rank_logits is not None
                and ("next", "end_of_game") in batch
                and ("next", "final_rank") in batch
            ):
                eog = batch["next", "end_of_game"].bool()
                eog_f = eog.float()
                rk_tgt = batch["next", "final_rank"].long().clamp(0, 3)
                rk_ce = F.cross_entropy(rank_logits.float(), rk_tgt, reduction="none")
                loss_rank = (rk_ce * eog_f).sum() / eog_f.sum().clamp_min(1.0)
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
                loss_qr, qr_metrics = quantile_cql_loss(
                    qq,
                    cands,
                    action,
                    td_target,
                    alpha=self.quantile_cql_alpha,
                    temperature=self.cql_temperature,
                    kappa=1.0,
                )
                loss_total = loss_total + loss_qr
                qr_metrics["loss_qr_cql"] = loss_qr.detach()

            # AWR diagnostics
            w_choice = weights[has_choice] if bool(has_choice.any().item()) else weights[:1]
            ess_ratio = (
                w_choice.sum().square() / (float(w_choice.numel()) * w_choice.square().sum().clamp_min(1e-12))
            )
            clip_frac = (w_choice >= self.awr_max_weight - 1e-6).float().mean()

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
        arg_tdm = TensorDictModule(
            DecodeConverter("argmax", tie_break_by_action_id=True),
            in_keys=["candidates", "decode"],
            out_keys=["action"],
        )
        return TensorDictSequential(enc_tdm, head_tdm, arg_tdm)

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
                    [
                        dump_object(
                            actor[2].module,
                            ["argmax"],
                            {"tie_break_by_action_id": True},
                        )
                    ],
                    {"in_keys": ["candidates", "decode"], "out_keys": ["action"]},
                ),
            ],
            {},
        )
        torch.save(state, output_path)


__all__ = [
    "DuelingQHead",
    "JointStructuredIQLModule",
    "QuantileQHead",
    "V2_OBS_KEYS",
    "quantile_cql_loss",
]
