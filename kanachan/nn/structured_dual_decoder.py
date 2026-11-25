"""Structured Dual-Decoder V2 (Scheme A) architecture, readout heads, and KVCache (§3)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Hashable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from kanachan.nn.decision_decoder import DecisionContext, legal_actions, require_tensor
from kanachan.nn.modern_encoder import RMSNorm
from kanachan.structured_v2 import (
    CAND_DYN_DIM,
    EVENT_BYTES,
    GLOBAL_FEATS_DIM,
    KANACHAN_ACTION_MANIFEST,
    KANACHAN_ACTION_MANIFEST_SHA256,
    MAX_CANDIDATES,
    MAX_ROPE_LEN,
    NULL_SOURCE_IDX,
    NULL_TILE_IDX,
    NUM_TYPES_OF_ACTIONS,
    PAD_ACTION_ID,
    PLAYER_FEATS_DIM,
    REL_BIAS_NUM_CLASSES,
    STRUCTURED_V2_SCHEMA_ID,
    TILE37_NONE,
    TILE_FEATS_DIM,
    derive_structured_v2_batch,
    is_red5_tile37,
    tile34,
)


ARCHITECTURE_ID = "mahjong_structured_dual_decoder_v2"


def swiglu_ffn_dim(d_model: int) -> int:
    """Unified SwiGLU hidden width (§3.4): ceil((8 * d_model / 3) / 16) * 16."""
    return int(math.ceil((8.0 * d_model / 3.0) / 16.0) * 16)


STRUCTURED_V2_PRESETS: dict[str, dict[str, Any]] = {
    "v2_tiny": {
        "dimension": 192,
        "num_heads": 6,
        "n_d1_layers": 4,
        "n_d2_layers": 2,
        "dim_feedforward": 512,
        "use_rel_bias": True,
        "head_type": "factorized",
    },
    "v2_small": {
        "dimension": 256,
        "num_heads": 8,
        "n_d1_layers": 8,
        "n_d2_layers": 4,
        "dim_feedforward": 688,
        "use_rel_bias": True,
        "head_type": "factorized",
    },
    "v2_small_no_bias": {
        "dimension": 256,
        "num_heads": 8,
        "n_d1_layers": 8,
        "n_d2_layers": 4,
        "dim_feedforward": 688,
        "use_rel_bias": False,
        "head_type": "factorized",
    },
    "spatial_only": {
        "dimension": 256,
        "num_heads": 8,
        "n_d1_layers": 0,
        "n_d2_layers": 6,
        "dim_feedforward": 688,
        "use_rel_bias": True,
        "head_type": "factorized",
    },
    "v2_base": {
        "dimension": 384,
        "num_heads": 8,
        "n_d1_layers": 8,
        "n_d2_layers": 4,
        "dim_feedforward": 1024,
        "use_rel_bias": True,
        "head_type": "factorized",
    },
    "v2_base_flat": {
        "dimension": 384,
        "num_heads": 8,
        "n_d1_layers": 8,
        "n_d2_layers": 4,
        "dim_feedforward": 1024,
        "use_rel_bias": True,
        "head_type": "flat546",
    },
    "v2_large": {
        "dimension": 512,
        "num_heads": 8,
        "n_d1_layers": 10,
        "n_d2_layers": 6,
        "dim_feedforward": 1376,
        "use_rel_bias": True,
        "head_type": "factorized",
    },
}


def structured_v2_schema() -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE_ID,
        "structured_schema": STRUCTURED_V2_SCHEMA_ID,
        "manifest_sha256": KANACHAN_ACTION_MANIFEST_SHA256,
        "tile_feats_dim": TILE_FEATS_DIM,
        "event_bytes": EVENT_BYTES,
        "global_feats_dim": GLOBAL_FEATS_DIM,
        "player_feats_dim": PLAYER_FEATS_DIM,
        "cand_dyn_dim": CAND_DYN_DIM,
        "max_candidates": MAX_CANDIDATES,
        "max_rope_len": MAX_ROPE_LEN,
    }


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embedding on [B, H, L, Dh]."""
    x_fp32 = x.float()
    half = x_fp32.shape[-1] // 2
    x1 = x_fp32[..., :half]
    x2 = x_fp32[..., half:]
    # Expand cos/sin [L, Dh//2] or [1, 1, L, Dh//2]
    while cos.ndim < x1.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1).to(x.dtype)


def _compute_rope_tables(
    max_len: int, head_dim: int, base: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    pos = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = float(base) ** (
        -torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
        / float(head_dim)
    )
    angles = pos[:, None] * freqs[None, :]
    return angles.cos(), angles.sin()


def _init_linear(linear: nn.Linear, std: float = 0.02) -> None:
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _build_tile_lookup_tables() -> tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor
]:
    t37_suit = torch.zeros(38, dtype=torch.long)
    t37_rank = torch.zeros(38, dtype=torch.long)
    t37_red5 = torch.zeros(38, dtype=torch.long)
    for t in range(38):
        if t == TILE37_NONE:
            t37_suit[t] = 4
            t37_rank[t] = 16
        elif t < 30:
            t37_suit[t] = t // 10
            r = 4 if t % 10 == 0 else (t % 10 - 1)
            t37_rank[t] = r
            t37_red5[t] = 1 if is_red5_tile37(t) else 0
        else:
            t37_suit[t] = 3
            t37_rank[t] = 9 + (t - 30)

    b34_suit = torch.zeros(35, dtype=torch.long)
    b34_rank = torch.zeros(35, dtype=torch.long)
    for b in range(35):
        if b == NULL_TILE_IDX:
            b34_suit[b] = 4
            b34_rank[b] = 16
        elif b < 27:
            b34_suit[b] = b // 9
            b34_rank[b] = b % 9
        else:
            b34_suit[b] = 3
            b34_rank[b] = 9 + (b - 27)
    return t37_suit, t37_rank, t37_red5, b34_suit, b34_rank


def _build_static_prefix_rel_classes() -> Tensor:
    """Build [45, 45] mahjong graph relation IDs (0..16) for Decoder 2 prefix."""
    rel = torch.full((45, 45), 16, dtype=torch.long)
    for i in range(34):
        qi = 10 + i
        for j in range(34):
            kj = 10 + j
            if i == j:
                cls = 0
            elif i < 27 and j < 27:
                si, sj = i // 9, j // 9
                ri, rj = i % 9, j % 9
                if si == sj:
                    dr = abs(ri - rj)
                    cls = dr if dr <= 3 else 4
                else:
                    cls = 5 if ri == rj else 9
            elif 27 <= i <= 30 and 27 <= j <= 30:
                cls = 6
            elif 31 <= i <= 33 and 31 <= j <= 33:
                cls = 7
            elif i >= 27 and j >= 27:
                cls = 8
            else:
                cls = 9
            rel[qi, kj] = cls
    for r in range(4):
        rel[1 + r, 6 + r] = 10
        rel[6 + r, 1 + r] = 10
    return rel


class DecomposedTileEmbedding(nn.Module):
    """Shared suit + rank + red5 tile embedding (§3.1)."""

    def __init__(self, d_model: int, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.suit_emb = nn.Embedding(5, d_model, device=device, dtype=dtype)
        self.rank_emb = nn.Embedding(17, d_model, device=device, dtype=dtype)
        self.red5_emb = nn.Embedding(2, d_model, device=device, dtype=dtype)
        for emb in (self.suit_emb, self.rank_emb, self.red5_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        t37_s, t37_r, t37_red, b34_s, b34_r = _build_tile_lookup_tables()
        self.register_buffer("t37_suit", t37_s.to(device=device))
        self.register_buffer("t37_rank", t37_r.to(device=device))
        self.register_buffer("t37_red5", t37_red.to(device=device))
        self.register_buffer("b34_suit", b34_s.to(device=device))
        self.register_buffer("b34_rank", b34_r.to(device=device))

    def embed_base34(self, b34: Tensor) -> Tensor:
        return self.suit_emb(self.b34_suit[b34]) + self.rank_emb(self.b34_rank[b34])

    def embed_tile37(self, t37: Tensor) -> Tensor:
        return (
            self.suit_emb(self.t37_suit[t37])
            + self.rank_emb(self.t37_rank[t37])
            + self.red5_emb(self.t37_red5[t37])
        )


class CausalTransformerBlock(nn.Module):
    """Pre-RMSNorm causal MHA + 1D-RoPE + SwiGLU block for Decoder 1."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_norm = RMSNorm(d_model, device=device, dtype=dtype, epsilon=1e-6)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(d_model, device=device, dtype=dtype, epsilon=1e-6)
        self.gate_up = nn.Linear(d_model, 2 * dim_feedforward, bias=False, device=device, dtype=dtype)
        self.down = nn.Linear(dim_feedforward, d_model, bias=False, device=device, dtype=dtype)

        _init_linear(self.qkv)
        _init_linear(self.out_proj, std=0.02 / math.sqrt(2.0))
        _init_linear(self.gate_up)
        _init_linear(self.down, std=0.02 / math.sqrt(2.0))

    def forward(
        self,
        x: Tensor,
        allow_mask: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
        *,
        kv_cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        bsz, length, d_model = x.shape
        h = self.attn_norm(x)
        qkv = self.qkv(h).view(bsz, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)

        if kv_cache is not None:
            k_full = torch.cat([kv_cache[0], k], dim=-2)
            v_full = torch.cat([kv_cache[1], v], dim=-2)
        else:
            k_full, v_full = k, v

        attn = F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=allow_mask, dropout_p=0.0, is_causal=False
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, length, d_model)
        x = x + self.out_proj(attn)

        gate, up = self.gate_up(self.ffn_norm(x)).chunk(2, dim=-1)
        x = x + self.down(F.silu(gate) * up)
        return x, (k_full, v_full)


@dataclass
class StructuredKVCache:
    """Explicit Decoder-1 KVCache keyed to a DecisionContext (§3.2.4, §5.4)."""

    context_key: tuple[Hashable, ...] | None = None
    consumed_len: int = 0
    prefix_events: Tensor | None = None  # [1, consumed_len, 10] int64/uint8 on cache device
    layer_kv: list[tuple[Tensor, Tensor]] = field(default_factory=list)
    last_hidden: Tensor | None = None  # [1, D]
    last_seat_hidden: Tensor | None = None  # [1, 4, D]
    has_seat_event: Tensor | None = None  # [1, 4] bool

    def reset(self) -> None:
        self.context_key = None
        self.consumed_len = 0
        self.prefix_events = None
        self.layer_kv.clear()
        self.last_hidden = None
        self.last_seat_hidden = None
        self.has_seat_event = None


class TrajectoryCausalDecoder(nn.Module):
    """Decoder 1: 10B causal event stream encoder + 5-slot D1_MEM (§3.2)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
        tile_emb: DecomposedTileEmbedding,
        *,
        rope_base: float = 10000.0,
        checkpointing: bool = False,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_layers = num_layers
        self.rope_base = float(rope_base)
        self.checkpointing = checkpointing
        self.tile_emb = tile_emb

        self.kind_emb = nn.Embedding(11, d_model, device=device, dtype=dtype)
        self.actor_emb = nn.Embedding(5, d_model, device=device, dtype=dtype)
        self.src_emb = nn.Embedding(5, d_model, device=device, dtype=dtype)
        self.w_t = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.w_c0 = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.w_c1 = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.proj_bits = nn.Linear(13, d_model, bias=False, device=device, dtype=dtype)
        self.ev_norm = RMSNorm(d_model, device=device, dtype=dtype, epsilon=1e-6)

        for emb in (self.kind_emb, self.actor_emb, self.src_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        for lin in (self.w_t, self.w_c0, self.w_c1, self.proj_bits):
            _init_linear(lin)

        self.no_event_seat_emb = nn.Embedding(4, d_model, device=device, dtype=dtype)
        nn.init.normal_(self.no_event_seat_emb.weight, mean=0.0, std=0.02)

        self.disabled_d1_mem: nn.Embedding | None = None
        if num_layers == 0:
            self.disabled_d1_mem = nn.Embedding(5, d_model, device=device, dtype=dtype)
            nn.init.normal_(self.disabled_d1_mem.weight, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList([
            CausalTransformerBlock(
                d_model, num_heads, dim_feedforward, device=device, dtype=dtype
            )
            for _ in range(num_layers)
        ])

        # Preallocated FP32 1D-RoPE table up to MAX_ROPE_LEN=2048 (kept outside
        # state_dict and safe against .to(half) precision downcast).
        cos_t, sin_t = _compute_rope_tables(MAX_ROPE_LEN, self.head_dim, self.rope_base, device)
        self._rope_cos: Tensor = cos_t
        self._rope_sin: Tensor = sin_t

    def _apply(self, fn):
        old_dev = self._rope_cos.device
        ret = super()._apply(fn)
        new_dev = self.kind_emb.weight.device
        if new_dev != old_dev or self._rope_cos.dtype != torch.float32:
            cos_t, sin_t = _compute_rope_tables(
                MAX_ROPE_LEN, self.head_dim, self.rope_base, new_dev
            )
            self._rope_cos = cos_t
            self._rope_sin = sin_t
        return ret

    def _get_rope_slice(self, start: int, end: int, device: torch.device) -> tuple[Tensor, Tensor]:
        if end > MAX_ROPE_LEN:
            raise ValueError(f"Sequence position {end} exceeds MAX_ROPE_LEN={MAX_ROPE_LEN}")
        if self._rope_cos.device != device or self._rope_cos.dtype != torch.float32:
            cos_t, sin_t = _compute_rope_tables(MAX_ROPE_LEN, self.head_dim, self.rope_base, device)
            self._rope_cos = cos_t
            self._rope_sin = sin_t
        return self._rope_cos[start:end], self._rope_sin[start:end]

    def embed_events(self, events: Tensor) -> Tensor:
        ev = events.long()
        h_cat = (
            self.kind_emb(ev[..., 0])
            + self.actor_emb(ev[..., 1])
            + self.src_emb(ev[..., 2])
            + self.w_t(self.tile_emb.embed_tile37(ev[..., 3]))
            + self.w_c0(self.tile_emb.embed_tile37(ev[..., 4]))
            + self.w_c1(self.tile_emb.embed_tile37(ev[..., 5]))
        )
        flags = ev[..., 6]
        sub_mask = ev[..., 7]
        turn_norm = ev[..., 8].float() / 100.0
        seq_norm = ev[..., 9].float() / 24.0
        wt_dtype = self.proj_bits.weight.dtype
        bits = [
            ((flags >> b) & 1).to(wt_dtype) for b in range(5)
        ] + [
            ((sub_mask >> b) & 1).to(wt_dtype) for b in range(4)
        ] + [
            turn_norm.to(wt_dtype),
            seq_norm.to(wt_dtype),
            torch.sin(turn_norm * math.pi).to(wt_dtype),
            torch.cos(turn_norm * math.pi).to(wt_dtype),
        ]
        bit_feats = torch.stack(bits, dim=-1)
        return self.ev_norm(h_cat + self.proj_bits(bit_feats))

    def _run_blocks(self, h: Tensor, event_len: Tensor) -> Tensor:
        bsz, seq_len, _ = h.shape
        pos = torch.arange(seq_len, device=h.device, dtype=torch.long)
        valid_k = pos.unsqueeze(0) < event_len.long().unsqueeze(1)  # [B, L]
        tril = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=h.device))
        allow = valid_k.unsqueeze(1) & tril.unsqueeze(0)  # [B, L, L]
        # Ensure padding query rows still see index 0 (BOS/DORA) so SDPA never outputs NaN
        has_any = allow.any(dim=-1, keepdim=True)
        fallback = torch.zeros_like(allow)
        fallback[:, :, 0] = True
        allow = torch.where(has_any, allow, fallback).unsqueeze(1)  # [B, 1, L, L]

        cos, sin = self._get_rope_slice(0, seq_len, h.device)
        for block in self.blocks:
            if self.checkpointing and self.training and torch.is_grad_enabled():
                def _call(x_in: Tensor, blk: CausalTransformerBlock = block) -> Tensor:
                    y, _ = blk(x_in, allow, cos, sin)
                    return y
                h = checkpoint(_call, h, use_reentrant=False)
            else:
                h, _ = block(h, allow, cos, sin)
        return h

    def extract_d1_mem_from_seq(
        self, h_ev: Tensor, events: Tensor, event_len: Tensor
    ) -> Tensor:
        """Extract D1_MEM [B, 5, D] = [h_last, h_seat_0..3] from causal outputs."""
        bsz, seq_len, d = h_ev.shape
        ev_len_l = event_len.long().clamp(min=1, max=seq_len)
        last_idx = ev_len_l - 1
        b_idx = torch.arange(bsz, device=h_ev.device)
        h_last = h_ev[b_idx, last_idx].unsqueeze(1)  # [B, 1, D]

        pos = torch.arange(seq_len, device=h_ev.device, dtype=torch.long).unsqueeze(0)
        valid = pos < ev_len_l.unsqueeze(1)
        ev_type = events[..., 0].long()
        ev_actor = events[..., 1].long()

        seat_mems = []
        for r in range(4):
            match = valid & (ev_type != 0) & (ev_actor == r)
            scores = torch.where(match, pos, torch.full_like(pos, -1))
            max_pos, _ = scores.max(dim=-1)  # [B]
            has_ev = max_pos >= 0
            safe_pos = max_pos.clamp_min(0)
            gathered = h_ev[b_idx, safe_pos]
            fallback = self.no_event_seat_emb.weight[r].unsqueeze(0).expand(bsz, -1)
            seat_mems.append(torch.where(has_ev.unsqueeze(-1), gathered, fallback))
        return torch.cat([h_last, torch.stack(seat_mems, dim=1)], dim=1)  # [B, 5, D]

    def forward_events(
        self,
        events: Tensor,
        event_len: Tensor,
        *,
        return_seq: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        bsz = events.shape[0]
        if self.num_layers == 0:
            assert self.disabled_d1_mem is not None
            d1_mem = self.disabled_d1_mem.weight.unsqueeze(0).expand(bsz, 5, -1)
            if return_seq:
                return d1_mem, self.embed_events(events)
            return d1_mem

        h = self.embed_events(events)
        h_ev = self._run_blocks(h, event_len)
        d1_mem = self.extract_d1_mem_from_seq(h_ev, events, event_len)
        return (d1_mem, h_ev) if return_seq else d1_mem

    def forward_round_events(
        self,
        round_events: Tensor,
        round_event_len: Tensor,
        decision_event_len: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run Decoder 1 ONCE per round and extract [B_r, T, 5, D] via cummax (§3.2.4)."""
        bsz_r, l_r, _ = round_events.shape
        tmax = decision_event_len.shape[1]
        if self.num_layers == 0:
            assert self.disabled_d1_mem is not None
            d1_mem = self.disabled_d1_mem.weight.view(1, 1, 5, self.d_model).expand(
                bsz_r, tmax, 5, self.d_model
            )
            h_ev = self.embed_events(round_events)
            return d1_mem, h_ev

        h = self.embed_events(round_events)
        h_ev = self._run_blocks(h, round_event_len)  # [B_r, L_r, D]

        dec_last = decision_event_len.long().clamp(min=1, max=l_r) - 1  # [B_r, T]
        idx_last = dec_last.unsqueeze(-1).expand(-1, -1, self.d_model)
        h_last = torch.gather(h_ev, 1, idx_last).unsqueeze(2)  # [B_r, T, 1, D]

        pos = torch.arange(l_r, device=h_ev.device, dtype=torch.long).unsqueeze(0)
        valid = pos < round_event_len.long().unsqueeze(1)
        ev_type = round_events[..., 0].long()
        ev_actor = round_events[..., 1].long()

        seat_cols = []
        for r in range(4):
            match = valid & (ev_type != 0) & (ev_actor == r)
            scores = torch.where(match, pos, torch.full_like(pos, -1))
            running_pos, _ = scores.cummax(dim=-1)  # [B_r, L_r]
            pos_at_dec = torch.gather(running_pos, 1, dec_last)  # [B_r, T]
            has_ev = pos_at_dec >= 0
            safe_pos = pos_at_dec.clamp_min(0).unsqueeze(-1).expand(-1, -1, self.d_model)
            gathered = torch.gather(h_ev, 1, safe_pos)
            fallback = self.no_event_seat_emb.weight[r].view(1, 1, self.d_model).expand_as(gathered)
            seat_cols.append(torch.where(has_ev.unsqueeze(-1), gathered, fallback))

        d1_mem = torch.cat([h_last, torch.stack(seat_cols, dim=2)], dim=2)  # [B_r, T, 5, D]
        return d1_mem, h_ev

    @torch.no_grad()
    def forward_step_cached(
        self,
        events: Tensor,
        event_len: Tensor,
        cache: StructuredKVCache,
        context_key: tuple[Hashable, ...],
    ) -> Tensor:
        """Incremental single-sample (B=1) causal Decoder 1 with prefix check (§3.2.4)."""
        if events.shape[0] != 1:
            raise ValueError("forward_step_cached supports B=1 explicit context.")
        bsz = 1
        cur_len = int(event_len[0].item())
        cur_events = events[:, :cur_len]

        if self.num_layers == 0:
            return self.forward_events(cur_events, event_len)

        must_reset = (
            cache.context_key != context_key
            or cache.prefix_events is None
            or cur_len < cache.consumed_len
            or (
                cache.consumed_len > 0
                and not torch.equal(
                    cur_events[:, : cache.consumed_len], cache.prefix_events
                )
            )
            or len(cache.layer_kv) != self.num_layers
        )
        if must_reset:
            cache.reset()
            cache.context_key = context_key
            start = 0
        else:
            start = cache.consumed_len

        if start == cur_len and cache.last_hidden is not None and cache.last_seat_hidden is not None:
            fallback = self.no_event_seat_emb.weight.unsqueeze(0)
            seats = torch.where(
                cache.has_seat_event.unsqueeze(-1),
                cache.last_seat_hidden,
                fallback,
            )
            return torch.cat([cache.last_hidden.unsqueeze(1), seats], dim=1)

        new_ev = cur_events[:, start:cur_len]
        new_len = cur_len - start
        h_new = self.embed_events(new_ev)
        cos, sin = self._get_rope_slice(start, cur_len, h_new.device)

        # Incremental causal mask: query at [start..cur_len-1], key at [0..cur_len-1]
        q_pos = torch.arange(start, cur_len, device=h_new.device)[:, None]
        k_pos = torch.arange(0, cur_len, device=h_new.device)[None, :]
        allow = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)  # [1, 1, new_len, cur_len]

        new_kv: list[tuple[Tensor, Tensor]] = []
        h = h_new
        for li, block in enumerate(self.blocks):
            prev = cache.layer_kv[li] if li < len(cache.layer_kv) else None
            h, kv_out = block(h, allow, cos, sin, kv_cache=prev)
            new_kv.append(kv_out)

        cache.layer_kv = new_kv
        cache.consumed_len = cur_len
        cache.prefix_events = cur_events.detach().clone()
        cache.last_hidden = h[:, -1]

        if cache.last_seat_hidden is None or cache.has_seat_event is None:
            cache.last_seat_hidden = torch.zeros(
                (bsz, 4, self.d_model), device=h.device, dtype=h.dtype
            )
            cache.has_seat_event = torch.zeros((bsz, 4), device=h.device, dtype=torch.bool)

        ev_types = new_ev[0, :, 0].long()
        ev_actors = new_ev[0, :, 1].long()
        for t in range(new_len):
            et = int(ev_types[t].item())
            ar = int(ev_actors[t].item())
            if et != 0 and 0 <= ar < 4:
                cache.last_seat_hidden[0, ar] = h[0, t]
                cache.has_seat_event[0, ar] = True

        fallback = self.no_event_seat_emb.weight.unsqueeze(0)
        seats = torch.where(
            cache.has_seat_event.unsqueeze(-1),
            cache.last_seat_hidden,
            fallback,
        )
        return torch.cat([cache.last_hidden.unsqueeze(1), seats], dim=1)


class ModeRoutedCandDyn(nn.Module):
    """DDP-safe branchless 7-mode router for cand_dyn [B, K, 8] -> [B, K, 64] (§3.3.1)."""

    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.empty((7, 8, 64), device=device, dtype=dtype))
        self.biases = nn.Parameter(torch.zeros((7, 64), device=device, dtype=dtype))
        nn.init.normal_(self.weights, mean=0.0, std=0.02)
        self.norm = RMSNorm(64, device=device, dtype=dtype, epsilon=1e-6)

    def _no_weight_decay_param_ids(self) -> set[int]:
        return {id(self.biases)}

    def forward(self, cand_dyn: Tensor, cand_mode: Tensor) -> Tensor:
        dyn = cand_dyn.to(self.weights.dtype)
        # [B, K, 7, 64]
        per_mode = torch.einsum("bki,mio->bkmo", dyn, self.weights) + self.biases[None, None, :, :]
        route = F.one_hot(cand_mode.long(), num_classes=7).to(self.weights.dtype)
        routed = (per_mode * route.unsqueeze(-1)).sum(dim=2)
        return self.norm(F.silu(routed))


class StructuralAttentionBlock(nn.Module):
    """Pre-RMSNorm MHA with optional 17-class graph rel_bias + SwiGLU (§3.3)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        *,
        use_rel_bias: bool = True,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rel_bias = bool(use_rel_bias)

        self.attn_norm = RMSNorm(d_model, device=device, dtype=dtype, epsilon=1e-6)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.rel_bias: nn.Embedding | None = None
        if self.use_rel_bias:
            self.rel_bias = nn.Embedding(
                REL_BIAS_NUM_CLASSES, num_heads, device=device, dtype=dtype
            )
            nn.init.zeros_(self.rel_bias.weight)

        self.ffn_norm = RMSNorm(d_model, device=device, dtype=dtype, epsilon=1e-6)
        self.gate_up = nn.Linear(d_model, 2 * dim_feedforward, bias=False, device=device, dtype=dtype)
        self.down = nn.Linear(dim_feedforward, d_model, bias=False, device=device, dtype=dtype)

        _init_linear(self.qkv)
        _init_linear(self.out_proj, std=0.02 / math.sqrt(2.0))
        _init_linear(self.gate_up)
        _init_linear(self.down, std=0.02 / math.sqrt(2.0))

    def forward(
        self,
        x: Tensor,
        allow_mask: Tensor,
        rel_idx: Tensor | None = None,
    ) -> Tensor:
        bsz, seq_len, d_model = x.shape
        h = self.attn_norm(x)
        qkv = self.qkv(h).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        if self.use_rel_bias and self.rel_bias is not None and rel_idx is not None:
            bias = self.rel_bias(rel_idx).permute(0, 3, 1, 2).to(q.dtype)  # [B, H, S, S]
            attn_mask = bias.masked_fill(~allow_mask, float("-inf"))
        else:
            attn_mask = allow_mask

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        x = x + self.out_proj(attn)

        gate, up = self.gate_up(self.ffn_norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


class DecisionStitchDecoder(nn.Module):
    """Decoder 2: [D1_MEM(5), GLOBAL(1), PLAYER(4), TILE(34), VAL(1), CAND(K)] (§3.3)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
        tile_emb: DecomposedTileEmbedding,
        *,
        use_rel_bias: bool = True,
        cand_self_attn: bool = True,
        head_type: str = "factorized",
        checkpointing: bool = False,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if head_type not in ("factorized", "flat546"):
            raise ValueError(f"Unsupported head_type: {head_type}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.tile_emb = tile_emb
        self.use_rel_bias = bool(use_rel_bias)
        self.cand_self_attn = bool(cand_self_attn)
        self.head_type = head_type
        self.checkpointing = checkpointing

        self.token_type_emb = nn.Embedding(6, d_model, device=device, dtype=dtype)
        self.d1_slot_emb = nn.Embedding(5, d_model, device=device, dtype=dtype)
        self.seat_rel_emb = nn.Embedding(4, d_model, device=device, dtype=dtype)
        self.action_type_emb = nn.Embedding(11, 32, device=device, dtype=dtype)
        for emb in (self.token_type_emb, self.d1_slot_emb, self.seat_rel_emb, self.action_type_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        # Single-slot vectors MUST be 1D nn.Parameter so AdamW does NOT weight-decay them!
        self.val_token_emb = nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))
        self.null_tile_emb = nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))
        self.null_source_emb = nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))
        for p in (self.val_token_emb, self.null_tile_emb, self.null_source_emb):
            nn.init.normal_(p, mean=0.0, std=0.02)

        self.global_proj = nn.Linear(GLOBAL_FEATS_DIM, d_model, device=device, dtype=dtype)
        self.player_proj = nn.Linear(PLAYER_FEATS_DIM, d_model, device=device, dtype=dtype)
        self.tile_proj = nn.Linear(TILE_FEATS_DIM, d_model, device=device, dtype=dtype)
        for lin in (self.global_proj, self.player_proj, self.tile_proj):
            _init_linear(lin)

        self.cand_dyn_router: ModeRoutedCandDyn | None = None
        self.cand_proj: nn.Linear | None = None
        if head_type == "factorized":
            self.cand_dyn_router = ModeRoutedCandDyn(device=device, dtype=dtype)
            self.cand_proj = nn.Linear(
                4 * d_model + 32 + 64 + 4 + 4, d_model, device=device, dtype=dtype
            )
            _init_linear(self.cand_proj)

        self.blocks = nn.ModuleList([
            StructuralAttentionBlock(
                d_model, num_heads, dim_feedforward,
                use_rel_bias=use_rel_bias, device=device, dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.out_norm = RMSNorm(d_model, device=device, dtype=dtype, epsilon=1e-6)

        # Static lookup buffers (integers / floats kept in buffer registry)
        manifest_t = torch.from_numpy(KANACHAN_ACTION_MANIFEST.copy()).to(device=device, dtype=torch.long)
        self.register_buffer("manifest", manifest_t)
        has_slot = torch.stack(
            [
                manifest_t[:, 1] != NULL_TILE_IDX,
                manifest_t[:, 2] != NULL_TILE_IDX,
                manifest_t[:, 3] != NULL_TILE_IDX,
                manifest_t[:, 4] != NULL_SOURCE_IDX,
            ],
            dim=-1,
        ).float()
        self.register_buffer("has_slot_flags", has_slot)
        self.register_buffer("static_flags", manifest_t[:, 5:9].float())
        self.register_buffer("base34_range", torch.arange(34, device=device, dtype=torch.long))
        self.register_buffer("d1_slot_range", torch.arange(5, device=device, dtype=torch.long))
        self.register_buffer("seat_rel_range", torch.arange(4, device=device, dtype=torch.long))
        self.register_buffer("static_rel_45", _build_static_prefix_rel_classes().to(device=device))

    def _build_masks_and_rel(
        self,
        bsz: int,
        kmax: int,
        cand_mask: Tensor,
        target_b34: Tensor | None,
        c0_b34: Tensor | None,
        c1_b34: Tensor | None,
        src_rel: Tensor | None,
        device: torch.device,
    ) -> tuple[Tensor, Tensor | None]:
        seq_len = 45 + (kmax if self.head_type == "factorized" else 0)
        allow = torch.zeros((bsz, 1, seq_len, seq_len), dtype=torch.bool, device=device)
        # Block 0: D1_MEM(0..4) -> 0..4
        allow[:, 0, 0:5, 0:5] = True
        # Block 1: BOARD(5..43) -> 0..43
        allow[:, 0, 5:44, 0:44] = True
        # Block 2: VAL(44) -> 0..44 (+ CAND)
        allow[:, 0, 44, 0:45] = True

        if self.head_type == "factorized":
            cm = cand_mask.bool()  # [B, K]
            # VAL(44) attends all valid CAND keys
            allow[:, 0, 44, 45:] = cm
            # CAND(45..) attend 0..44
            allow[:, 0, 45:, 0:45] = True
            if self.cand_self_attn:
                allow[:, 0, 45:, 45:] = cm.unsqueeze(1)
            else:
                eye = torch.eye(kmax, dtype=torch.bool, device=device).unsqueeze(0)
                allow[:, 0, 45:, 45:] = eye & cm.unsqueeze(1)

        if not self.use_rel_bias:
            return allow, None

        rel = torch.full((bsz, seq_len, seq_len), 16, dtype=torch.long, device=device)
        rel[:, :45, :45] = self.static_rel_45.unsqueeze(0)

        if self.head_type == "factorized" and target_b34 is not None:
            assert c0_b34 is not None and c1_b34 is not None and src_rel is not None
            # Tile edges: c0/c1 -> 12, target -> 11 (higher priority)
            tile_rel = torch.full((bsz, kmax, 35), 16, dtype=torch.long, device=device)
            src12 = torch.full((bsz, kmax, 1), 12, dtype=torch.long, device=device)
            src11 = torch.full((bsz, kmax, 1), 11, dtype=torch.long, device=device)
            tile_rel.scatter_(2, c0_b34.unsqueeze(-1), src12)
            tile_rel.scatter_(2, c1_b34.unsqueeze(-1), src12)
            tile_rel.scatter_(2, target_b34.unsqueeze(-1), src11)
            tile_34 = tile_rel[:, :, :34]  # drops NULL_TILE=34 cleanly!
            rel[:, 45:, 10:44] = tile_34
            rel[:, 10:44, 45:] = tile_34.transpose(1, 2)

            # Player source edges -> 13
            pl_rel = torch.full((bsz, kmax, 5), 16, dtype=torch.long, device=device)
            src13 = torch.full((bsz, kmax, 1), 13, dtype=torch.long, device=device)
            pl_rel.scatter_(2, src_rel.unsqueeze(-1), src13)
            pl_4 = pl_rel[:, :, :4]  # drops NULL_SOURCE=4 cleanly!
            rel[:, 45:, 6:10] = pl_4
            rel[:, 6:10, 45:] = pl_4.transpose(1, 2)

            # CAND <-> CAND: self=14, cross=15
            cand_cc = torch.full((kmax, kmax), 15, dtype=torch.long, device=device)
            cand_cc.fill_diagonal_(14)
            rel[:, 45:, 45:] = cand_cc.unsqueeze(0)

        return allow, rel

    def forward(
        self,
        d1_mem: Tensor,
        global_feats: Tensor,
        player_feats: Tensor,
        tile_feats: Tensor,
        cand_ids: Tensor,
        cand_mask: Tensor,
        cand_mode: Tensor,
        cand_dyn: Tensor,
    ) -> dict[str, Tensor]:
        bsz = d1_mem.shape[0]
        device = d1_mem.device
        wt_dtype = self.global_proj.weight.dtype

        e_d1 = d1_mem.to(wt_dtype) + self.d1_slot_emb(self.d1_slot_range) + self.token_type_emb.weight[0]
        e_global = (self.global_proj(global_feats.to(wt_dtype)) + self.token_type_emb.weight[1]).unsqueeze(1)
        e_player = self.player_proj(player_feats.to(wt_dtype)) + self.seat_rel_emb(self.seat_rel_range) + self.token_type_emb.weight[2]
        e_tile = (
            self.tile_proj(tile_feats.to(wt_dtype))
            + self.tile_emb.embed_base34(self.base34_range)
            + self.token_type_emb.weight[3]
        )
        e_val = (self.val_token_emb + self.token_type_emb.weight[4]).view(1, 1, self.d_model).expand(bsz, 1, -1)

        kmax = cand_ids.shape[1]
        if self.head_type == "factorized":
            assert self.cand_dyn_router is not None and self.cand_proj is not None
            m = self.manifest[cand_ids.long()]  # [B, K, 10]
            atype = m[..., 0]
            target_b34 = m[..., 1]
            c0_b34 = m[..., 2]
            c1_b34 = m[..., 3]
            src_rel = m[..., 4]
            has_slot = self.has_slot_flags[cand_ids.long()].to(wt_dtype)
            s_flags = self.static_flags[cand_ids.long()].to(wt_dtype)

            e_tile_ext = torch.cat(
                [e_tile, self.null_tile_emb.view(1, 1, self.d_model).expand(bsz, 1, -1)], dim=1
            )
            e_player_ext = torch.cat(
                [e_player, self.null_source_emb.view(1, 1, self.d_model).expand(bsz, 1, -1)], dim=1
            )

            def _gather_node(ext: Tensor, idx: Tensor) -> Tensor:
                return torch.gather(ext, 1, idx.unsqueeze(-1).expand(-1, -1, self.d_model))

            et = _gather_node(e_tile_ext, target_b34)
            ec0 = _gather_node(e_tile_ext, c0_b34)
            ec1 = _gather_node(e_tile_ext, c1_b34)
            esrc = _gather_node(e_player_ext, src_rel)
            e_atype = self.action_type_emb(atype)
            e_dyn = self.cand_dyn_router(cand_dyn, cand_mode)

            cand_in = torch.cat([et, ec0, ec1, esrc, e_atype, e_dyn, has_slot, s_flags], dim=-1)
            e_cand = self.cand_proj(cand_in) + self.token_type_emb.weight[5]
            e_cand = e_cand.masked_fill(~cand_mask.unsqueeze(-1), 0.0)
            x = torch.cat([e_d1, e_global, e_player, e_tile, e_val, e_cand], dim=1)
        else:
            target_b34 = c0_b34 = c1_b34 = src_rel = None
            x = torch.cat([e_d1, e_global, e_player, e_tile, e_val], dim=1)

        allow_mask, rel_idx = self._build_masks_and_rel(
            bsz, kmax, cand_mask, target_b34, c0_b34, c1_b34, src_rel, device
        )

        for block in self.blocks:
            if self.checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(
                    lambda h_in, b=block: b(h_in, allow_mask, rel_idx),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x, allow_mask, rel_idx)

        x = self.out_norm(x)
        h_global = x[:, 5]
        h_player = x[:, 6:10]
        h_tile = x[:, 10:44]
        h_val = x[:, 44]
        if self.head_type == "factorized":
            h_cand = x[:, 45:].masked_fill(~cand_mask.unsqueeze(-1), 0.0)
        else:
            h_cand = h_val.unsqueeze(1).expand(bsz, MAX_CANDIDATES, self.d_model)

        return {
            "h_global": h_global,
            "h_player": h_player,
            "h_tile": h_tile,
            "h_val": h_val,
            "h_cand": h_cand,
        }


class StructuredDualDecoder(nn.Module):
    """Unified V2 backbone packing [h_val(1), h_cand(32)] -> encode [B, 33, D] (§3, §5.3)."""

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
        include_room_level: bool = False,
        rope_base: float = 10000.0,
        checkpointing: bool = False,
        schema: dict[str, Any] | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if include_room_level:
            raise ValueError("structured_dual_v2 requires include_room_level=False.")
        if schema is not None and schema.get("manifest_sha256") not in (
            None,
            KANACHAN_ACTION_MANIFEST_SHA256,
        ):
            raise ValueError("Structured V2 action manifest SHA-256 mismatch.")

        cfg = dict(STRUCTURED_V2_PRESETS.get(preset or "v2_base", {}))
        d_model = int(dimension if dimension is not None else cfg.get("dimension", 384))
        heads = int(num_heads if num_heads is not None else cfg.get("num_heads", 8))
        ffn_dim = int(
            dim_feedforward
            if dim_feedforward is not None
            else cfg.get("dim_feedforward", swiglu_ffn_dim(d_model))
        )
        d1_l = int(n_d1_layers if n_d1_layers is not None else cfg.get("n_d1_layers", 8))
        d2_l = int(n_d2_layers if n_d2_layers is not None else cfg.get("n_d2_layers", 4))
        rel_bias = bool(use_rel_bias if use_rel_bias is not None else cfg.get("use_rel_bias", True))
        ht = str(head_type if head_type is not None else cfg.get("head_type", "factorized"))

        self.preset = preset
        self.dimension = d_model
        self.num_heads = heads
        self.dim_feedforward = ffn_dim
        self.n_d1_layers = d1_l
        self.n_d2_layers = d2_l
        self.use_rel_bias = rel_bias
        self.cand_self_attn = bool(cand_self_attn)
        self.head_type = ht
        self.enable_early_shape_hint = bool(enable_early_shape_hint)
        self.include_room_level = False
        self.rope_base = float(rope_base)

        self.tile_emb = DecomposedTileEmbedding(d_model, device=device, dtype=dtype)
        self.decoder1 = TrajectoryCausalDecoder(
            d_model, heads, ffn_dim, d1_l, self.tile_emb,
            rope_base=rope_base, checkpointing=checkpointing, device=device, dtype=dtype,
        )
        self.decoder2 = DecisionStitchDecoder(
            d_model, heads, ffn_dim, d2_l, self.tile_emb,
            use_rel_bias=rel_bias, cand_self_attn=cand_self_attn, head_type=ht,
            checkpointing=checkpointing, device=device, dtype=dtype,
        )

    def _pack_encode(self, h_val: Tensor, h_cand: Tensor) -> Tensor:
        return torch.cat([h_val.unsqueeze(1), h_cand], dim=1)  # [B, 33, D]

    def forward_structured(
        self,
        tile_feats: Tensor,
        events: Tensor,
        event_len: Tensor,
        global_feats: Tensor,
        player_feats: Tensor,
        cand_ids: Tensor,
        cand_mask: Tensor,
        cand_mode: Tensor,
        cand_dyn: Tensor,
        *,
        return_dict: bool = False,
        return_d1_seq: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        d1_out = self.decoder1.forward_events(
            events, event_len, return_seq=return_d1_seq
        )
        if return_d1_seq:
            d1_mem, h_ev = d1_out
        else:
            d1_mem, h_ev = d1_out, None
        d2_out = self.decoder2(
            d1_mem, global_feats, player_feats, tile_feats,
            cand_ids, cand_mask, cand_mode, cand_dyn,
        )
        encode = self._pack_encode(d2_out["h_val"], d2_out["h_cand"])
        if not return_dict:
            return encode
        d2_out["encode"] = encode
        if h_ev is not None:
            d2_out["h_ev"] = h_ev
        return d2_out

    def forward_round_trajectory(
        self,
        round_events: Tensor,
        round_event_len: Tensor,
        decision_event_len: Tensor,
        decision_mask: Tensor,
        tile_feats: Tensor,
        global_feats: Tensor,
        player_feats: Tensor,
        cand_ids: Tensor,
        cand_mask: Tensor,
        cand_mode: Tensor,
        cand_dyn: Tensor,
    ) -> dict[str, Tensor]:
        """Whole-round parallel Teacher Forcing forward (§3.2.4, §4.2).

        round_events:       [B_r, L_r, 10]
        round_event_len:    [B_r]
        decision_event_len: [B_r, T]
        decision_mask:      [B_r, T] bool
        Step tensors can be [B_r, T, ...] (filtered via decision_mask) OR already
        flattened [N_active, ...]; we accept [B_r, T, ...] uniformly.
        """
        d1_mem_round, h_ev = self.decoder1.forward_round_events(
            round_events, round_event_len, decision_event_len
        )
        dmask = decision_mask.bool()
        d1_flat = d1_mem_round[dmask]
        tf_flat = tile_feats[dmask] if tile_feats.ndim == 4 else tile_feats
        gf_flat = global_feats[dmask] if global_feats.ndim == 3 else global_feats
        pf_flat = player_feats[dmask] if player_feats.ndim == 4 else player_feats
        cids_flat = cand_ids[dmask] if cand_ids.ndim == 3 else cand_ids
        cmask_flat = cand_mask[dmask] if cand_mask.ndim == 3 else cand_mask
        cmode_flat = cand_mode[dmask] if cand_mode.ndim == 3 else cand_mode
        cdyn_flat = cand_dyn[dmask] if cand_dyn.ndim == 4 else cand_dyn

        d2_out = self.decoder2(
            d1_flat, gf_flat, pf_flat, tf_flat,
            cids_flat, cmask_flat, cmode_flat, cdyn_flat,
        )
        d2_out["encode"] = self._pack_encode(d2_out["h_val"], d2_out["h_cand"])
        d2_out["h_ev"] = h_ev
        d2_out["cand_mask"] = cmask_flat
        return d2_out

    def forward_step_cached(
        self,
        *,
        context: DecisionContext,
        cache: StructuredKVCache,
        tile_feats: Tensor,
        events: Tensor,
        event_len: Tensor,
        global_feats: Tensor,
        player_feats: Tensor,
        cand_ids: Tensor,
        cand_mask: Tensor,
        cand_mode: Tensor,
        cand_dyn: Tensor,
    ) -> Tensor:
        key = context.cache_key() if hasattr(context, "cache_key") else (
            id(context.environment), context.game, context.seat,
            context.round_instance, context.model_version,
        )
        d1_mem = self.decoder1.forward_step_cached(events, event_len, cache, key)
        d2_out = self.decoder2(
            d1_mem, global_feats, player_feats, tile_feats,
            cand_ids, cand_mask, cand_mode, cand_dyn,
        )
        return self._pack_encode(d2_out["h_val"], d2_out["h_cand"])

    def forward(self, *args: Tensor, return_dict: bool = False) -> Tensor | dict[str, Tensor]:
        """Accept either precomputed 9 v2_* tensors OR 4 raw fields (portable export path)."""
        if len(args) == 4:
            sparse, numeric, progression, candidates = args
            legal_actions(candidates)
            feats = derive_structured_v2_batch(
                sparse, numeric, progression, candidates,
                enable_early_shape_hint=self.enable_early_shape_hint,
                include_room_level=False,
            )
            return self.forward_structured(
                feats["v2_tile_feats"],
                feats["v2_events"],
                feats["v2_event_len"],
                feats["v2_global_feats"],
                feats["v2_player_feats"],
                feats["v2_cand_ids"],
                feats["v2_cand_mask"],
                feats["v2_cand_mode"],
                feats["v2_cand_dyn"],
                return_dict=return_dict,
            )
        if len(args) == 9:
            return self.forward_structured(*args, return_dict=return_dict)
        if len(args) == 13:
            # 4 raw fields followed by 9 precomputed v2_* tensors
            # (allows TensorDictModule in_keys to bind both without .cpu() sync!)
            candidates = args[3]
            legal_actions(candidates)
            return self.forward_structured(*args[4:], return_dict=return_dict)
        raise TypeError(
            f"StructuredDualDecoder expected 4, 9, or 13 tensor arguments, got {len(args)}."
        )


def cached_structured_decision(
    encoder: StructuredDualDecoder,
    head: nn.Module,
    context: DecisionContext,
    cache: StructuredKVCache,
    sparse: Tensor,
    numeric: Tensor,
    progression: Tensor,
    candidates: Tensor,
) -> Tensor:
    """Explicit incremental KVCache decision path mirroring stateless forward (§5.4)."""
    legal_actions(candidates)
    feats = derive_structured_v2_batch(
        sparse, numeric, progression, candidates,
        enable_early_shape_hint=encoder.enable_early_shape_hint,
        include_room_level=False,
    )
    encode = encoder.forward_step_cached(
        context=context,
        cache=cache,
        tile_feats=feats["v2_tile_feats"],
        events=feats["v2_events"],
        event_len=feats["v2_event_len"],
        global_feats=feats["v2_global_feats"],
        player_feats=feats["v2_player_feats"],
        cand_ids=feats["v2_cand_ids"],
        cand_mask=feats["v2_cand_mask"],
        cand_mode=feats["v2_cand_mode"],
        cand_dyn=feats["v2_cand_dyn"],
    )
    return head(candidates, encode)


class StructuredActionHead(nn.Module):
    """FP32 policy/value readout from packed encode: [B, 33, D] (§3.3.4, §3.4)."""

    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int | None = None,
        head_type: str = "factorized",
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if head_type not in ("factorized", "flat546"):
            raise ValueError(f"Unsupported head_type: {head_type}")
        hdim = int(hidden_dimension or input_dimension)
        self.input_dimension = input_dimension
        self.hidden_dimension = hdim
        self.head_type = head_type

        if head_type == "factorized":
            self.mlp = nn.Sequential(
                nn.Linear(input_dimension, hdim, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hdim, 1, device=device, dtype=dtype),
            )
            _init_linear(self.mlp[0])
            _init_linear(self.mlp[2])
            self.flat_proj = None
        else:
            self.mlp = None
            self.flat_proj = nn.Linear(input_dimension, 547, device=device, dtype=dtype)
            _init_linear(self.flat_proj)

    @torch.no_grad()
    def zero_output(self) -> None:
        if self.mlp is not None:
            self.mlp[-1].weight.zero_()
            if self.mlp[-1].bias is not None:
                self.mlp[-1].bias.zero_()
        if self.flat_proj is not None:
            self.flat_proj.weight.zero_()
            if self.flat_proj.bias is not None:
                self.flat_proj.bias.zero_()

    def forward(self, candidates: Tensor, encode: Tensor) -> Tensor:
        cand_mask = candidates != PAD_ACTION_ID
        out_dtype = torch.float64 if encode.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=encode.device.type, enabled=False):
            if self.head_type == "factorized":
                assert self.mlp is not None
                h_cand = encode[:, 1:].to(out_dtype)  # [B, 32, D]
                h = F.silu(
                    F.linear(
                        h_cand,
                        self.mlp[0].weight.to(out_dtype),
                        self.mlp[0].bias.to(out_dtype) if self.mlp[0].bias is not None else None,
                    )
                )
                logits = F.linear(
                    h,
                    self.mlp[2].weight.to(out_dtype),
                    self.mlp[2].bias.to(out_dtype) if self.mlp[2].bias is not None else None,
                ).squeeze(-1)
            else:
                assert self.flat_proj is not None
                h_val = encode[:, 0].to(out_dtype)
                all_logits = F.linear(
                    h_val,
                    self.flat_proj.weight.to(out_dtype),
                    self.flat_proj.bias.to(out_dtype) if self.flat_proj.bias is not None else None,
                )
                logits = torch.gather(all_logits, 1, candidates.long())
            return torch.where(cand_mask, logits, torch.zeros_like(logits))


class StructuredStateHead(nn.Module):
    """FP32 state value V_psi(h_val) + optional FinalRankHead (§3.3.4, §3.4, §4.4)."""

    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int | None = None,
        final_rank: bool = False,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        hdim = int(hidden_dimension or input_dimension)
        self.norm = RMSNorm(input_dimension, device=device, dtype=dtype, epsilon=1e-6)
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, hdim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hdim, 1, device=device, dtype=dtype),
        )
        _init_linear(self.layers[0])
        _init_linear(self.layers[2])
        self.final_rank: nn.Linear | None = None
        if final_rank:
            self.final_rank = nn.Linear(hdim, 4, device=device, dtype=dtype)
            _init_linear(self.final_rank)

    @torch.no_grad()
    def zero_output(self) -> None:
        self.layers[-1].weight.zero_()
        if self.layers[-1].bias is not None:
            self.layers[-1].bias.zero_()

    def forward(self, encode: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        h_val = encode[:, 0] if encode.ndim == 3 else encode
        out_dtype = torch.float64 if encode.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=encode.device.type, enabled=False):
            hn = self.norm(h_val.to(out_dtype))
            hidden = F.silu(
                F.linear(
                    hn,
                    self.layers[0].weight.to(out_dtype),
                    self.layers[0].bias.to(out_dtype) if self.layers[0].bias is not None else None,
                )
            )
            value = F.linear(
                hidden,
                self.layers[2].weight.to(out_dtype),
                self.layers[2].bias.to(out_dtype) if self.layers[2].bias is not None else None,
            ).squeeze(-1)
            if self.final_rank is None:
                return value
            rank_logits = F.linear(
                hidden,
                self.final_rank.weight.to(out_dtype),
                self.final_rank.bias.to(out_dtype) if self.final_rank.bias is not None else None,
            )
            return value, rank_logits


class NextEventHead(nn.Module):
    """Training-only self-supervised next-event predictor on h_ev[:, :-1] (§4.2.2)."""

    def __init__(
        self,
        d_model: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model, device=device, dtype=dtype, epsilon=1e-6)
        self.kind_head = nn.Linear(d_model, 11, device=device, dtype=dtype)
        self.actor_head = nn.Linear(d_model, 5, device=device, dtype=dtype)
        self.tile34_head = nn.Linear(d_model, 35, device=device, dtype=dtype)
        self.tedashi_head = nn.Linear(d_model, 2, device=device, dtype=dtype)
        for lin in (self.kind_head, self.actor_head, self.tile34_head, self.tedashi_head):
            _init_linear(lin)

        t37_to_b34 = torch.full((38,), NULL_TILE_IDX, dtype=torch.long, device=device)
        for t in range(37):
            t37_to_b34[t] = tile34(t)
        self.register_buffer("t37_to_b34", t37_to_b34)

    def forward(
        self, h_ev: Tensor, events: Tensor, event_len: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        bsz, lmax, _ = h_ev.shape
        device = h_ev.device
        with torch.autocast(device_type=device.type, enabled=False):
            if lmax < 2:
                # Connect zero loss to every head parameter so DDP never sees unused params.
                zero = (
                    sum(p.float().sum() * 0.0 for p in self.parameters())
                )
                return zero, {
                    "kind": zero.detach(),
                    "actor": zero.detach(),
                    "tile34": zero.detach(),
                    "tedashi": zero.detach(),
                }

            x = self.norm(h_ev[:, :-1]).float()  # [B, L-1, D]
            next_ev = events[:, 1:].long()  # [B, L-1, 10]
            pos_next = torch.arange(1, lmax, device=device, dtype=torch.long).unsqueeze(0)
            valid = pos_next < event_len.long().unsqueeze(1)  # [B, L-1]

            def _lin(mod: nn.Linear) -> Tensor:
                return F.linear(
                    x, mod.weight.float(),
                    mod.bias.float() if mod.bias is not None else None,
                )

            k_logits = _lin(self.kind_head)
            a_logits = _lin(self.actor_head)
            t_logits = _lin(self.tile34_head)
            td_logits = _lin(self.tedashi_head)

            denom = valid.float().sum().clamp_min(1.0)
            valid_f = valid.float()

            k_loss = (
                F.cross_entropy(k_logits.reshape(-1, 11), next_ev[..., 0].reshape(-1), reduction="none")
                .view(bsz, lmax - 1)
                * valid_f
            ).sum() / denom
            a_loss = (
                F.cross_entropy(a_logits.reshape(-1, 5), next_ev[..., 1].reshape(-1), reduction="none")
                .view(bsz, lmax - 1)
                * valid_f
            ).sum() / denom
            target_b34 = self.t37_to_b34[next_ev[..., 3]]
            t_loss = (
                F.cross_entropy(t_logits.reshape(-1, 35), target_b34.reshape(-1), reduction="none")
                .view(bsz, lmax - 1)
                * valid_f
            ).sum() / denom

            is_dahai = valid & (next_ev[..., 0] == 3)
            td_target = (next_ev[..., 6] & 1).clamp(0, 1)
            td_raw = F.cross_entropy(
                td_logits.reshape(-1, 2), td_target.reshape(-1), reduction="none"
            ).view(bsz, lmax - 1)
            td_mask = is_dahai.float()
            td_loss = (td_raw * td_mask).sum() / td_mask.sum().clamp_min(1.0)
            if not bool(is_dahai.any().item()):
                td_loss = td_logits.sum() * 0.0

            total = 1.0 * k_loss + 0.5 * a_loss + 1.0 * t_loss + 0.5 * td_loss
            return total, {
                "kind": k_loss.detach(),
                "actor": a_loss.detach(),
                "tile34": t_loss.detach(),
                "tedashi": td_loss.detach(),
            }


__all__ = [
    "ARCHITECTURE_ID",
    "CausalTransformerBlock",
    "DecisionStitchDecoder",
    "DecomposedTileEmbedding",
    "ModeRoutedCandDyn",
    "NextEventHead",
    "STRUCTURED_V2_PRESETS",
    "StructuralAttentionBlock",
    "StructuredActionHead",
    "StructuredDualDecoder",
    "StructuredKVCache",
    "StructuredStateHead",
    "TrajectoryCausalDecoder",
    "cached_structured_decision",
    "structured_v2_schema",
    "swiglu_ffn_dim",
]
