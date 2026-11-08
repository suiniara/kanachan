"""Attention kernels for the decoder's causal-history/current-state mask.

SDPA remains the portable default. The explicit ``flex`` backend compiles
PyTorch FlexAttention on CUDA; CPU inference continues to use SDPA. Kernel
failures on CUDA are surfaced instead of silently changing the backend.
"""

from functools import lru_cache

import torch
from torch.nn import functional as F


ATTENTION_BACKENDS = ("sdpa", "flex")


def flex_attention(query, key, value, allowed):
    """Compiler entry point, also usable unfused for CPU equivalence tests.

    Keep only tensor indices in score_mod. Mixing a literal head index with
    the batched scalar indices breaks tracing in supported PyTorch versions.
    The mask is an input, so changing batch contents does not bake in padding.
    At 147 tokens, avoid rebuilding a BlockMask for each decoder layer.
    """
    from torch.nn.attention.flex_attention import flex_attention as kernel

    mask = allowed[:, 0]

    def score_mod(score, batch, head, query_index, key_index):
        return torch.where(
            mask[batch, query_index, key_index], score, -float("inf")
        )

    return kernel(query, key, value, score_mod=score_mod)


@lru_cache(maxsize=1)
def _compiled_flex_attention():
    # Lazy compilation leaves CPU loading and old model imports inexpensive.
    # Native stateless inference and tail batches vary in size. Support those
    # dimensions instead of compiling a separate graph for every batch size.
    return torch.compile(flex_attention, fullgraph=True, dynamic=True)


def decision_attention(query, key, value, allowed, backend="sdpa"):
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(f"Unknown decision attention backend: {backend}")
    if (allowed.ndim != 4 or allowed.dtype != torch.bool
            or allowed.shape != (query.shape[0], 1, query.shape[-2], key.shape[-2])
            or allowed.device != query.device):
        raise ValueError("Decision attention requires a boolean [B,1,L,S] mask.")
    if backend == "flex" and query.device.type == "cuda":
        from torch._dynamo import config as dynamo_config

        if dynamo_config.disable:
            raise RuntimeError(
                "FlexAttention requires torch.compile; unset TORCHDYNAMO_DISABLE."
            )
        return _compiled_flex_attention()(query, key, value, allowed)
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=allowed, dropout_p=0.0, is_causal=False
    )
