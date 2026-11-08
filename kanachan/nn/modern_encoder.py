"""Bidirectional Pre-Norm blocks using RMSNorm, SwiGLU and PyTorch SDPA."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(
        self, dimension: int, *, device: torch.device, dtype: torch.dtype,
        epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension, device=device, dtype=dtype))
        self.epsilon = epsilon

    def forward(self, value: Tensor) -> Tensor:
        working = (
            value.float()
            if value.dtype in (torch.float16, torch.bfloat16) else value
        )
        normalized = working * torch.rsqrt(
            working.square().mean(-1, keepdim=True) + self.epsilon
        )
        return (normalized * self.weight.to(working.dtype)).to(value.dtype)


class RMSNormSwiGLULayer(nn.Module):
    def __init__(
        self, dimension: int, num_heads: int, dim_feedforward: int,
        dropout: float, *, device: torch.device, dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if dimension <= 0 or num_heads <= 0 or dimension % num_heads:
            raise ValueError(
                "The encoder dimension must be divisible by its head count."
            )
        if dim_feedforward <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("Invalid feedforward dimension or dropout.")
        self.num_heads = num_heads
        self.head_dimension = dimension // num_heads
        self.dropout = dropout
        # Three gated projections replace two dense FFN projections. Align to
        # eight while keeping the parameter budget close to the existing FFN.
        self.hidden_dimension = max(8, round(dim_feedforward / 12) * 8)
        arguments = {"device": device, "dtype": dtype, "bias": False}
        self.attention_norm = RMSNorm(dimension, device=device, dtype=dtype)
        self.qkv = nn.Linear(dimension, 3 * dimension, **arguments)
        self.attention_output = nn.Linear(dimension, dimension, **arguments)
        self.feedforward_norm = RMSNorm(dimension, device=device, dtype=dtype)
        self.gate_up = nn.Linear(dimension, 2 * self.hidden_dimension, **arguments)
        self.down = nn.Linear(self.hidden_dimension, dimension, **arguments)
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        batch, length, dimension = value.shape
        normalized = self.attention_norm(value)
        qkv = self.qkv(normalized).reshape(
            batch, length, 3, self.num_heads, self.head_dimension
        )
        query, key, projected = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        allowed = None
        if padding_mask is not None:
            if (
                padding_mask.dtype != torch.bool
                or padding_mask.shape != (batch, length)
            ):
                raise ValueError(
                    "Expected a boolean batch-by-token padding mask."
                )
            # SDPA uses True for an allowed key, unlike key-padding masks.
            allowed = (~padding_mask)[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query, key, projected, attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0, is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch, length, dimension
        )
        value = value + self.attention_dropout(self.attention_output(attended))
        gate, up = self.gate_up(self.feedforward_norm(value)).chunk(2, dim=-1)
        return value + self.feedforward_dropout(self.down(F.silu(gate) * up))


class RMSNormSwiGLUEncoder(nn.Module):
    def __init__(
        self, dimension: int, num_heads: int, dim_feedforward: int,
        dropout: float, num_layers: int, *, device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(num_layers)
        self.layers = nn.ModuleList([
            RMSNormSwiGLULayer(
                dimension, num_heads, dim_feedforward, dropout,
                device=device, dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dimension, device=device, dtype=dtype)

    def forward(
        self, value: Tensor, padding_mask: Tensor | None = None, *,
        checkpointing: bool = False,
    ) -> Tensor:
        for layer in self.layers:
            if checkpointing:
                value = checkpoint(layer, value, padding_mask, use_reentrant=False)
            else:
                value = layer(value, padding_mask)
        return self.norm(value)
