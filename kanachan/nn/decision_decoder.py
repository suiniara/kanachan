"""Dense Mahjong decision decoder, with immutable causal history caches.

History is causal. The simultaneously observed state is an unordered block,
and a final readout token sees both. Action identifiers never acquire a
position from the order of the native candidate list.
"""

from dataclasses import dataclass, field
import math
from typing import Hashable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from kanachan.constants import (
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_ACTIONS,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_SPARSE_FEATURES,
)
from kanachan.nn.modern_encoder import RMSNorm
from kanachan.nn.numeric_encoding import encode_numeric_features
from kanachan.nn.decision_attention import ATTENTION_BACKENDS, decision_attention


ARCHITECTURE_ID = "mahjong_dense_block_decoder_v1"


def feature_schema(candidate_features="none") -> dict:
    schema = {
        "version": 1,
        "sparse": [MAX_NUM_ACTIVE_SPARSE_FEATURES, NUM_TYPES_OF_SPARSE_FEATURES],
        "numeric": NUM_NUMERIC_FEATURES,
        "progression": [
            MAX_LENGTH_OF_PROGRESSION_FEATURES,
            NUM_TYPES_OF_PROGRESSION_FEATURES,
        ],
        "candidates": [MAX_NUM_ACTION_CANDIDATES, NUM_TYPES_OF_ACTIONS],
        "action": "candidate_index",
        "numeric_encoding": "legacy_piecewise_linear_v1",
    }
    if candidate_features != "none":
        if candidate_features != "discard_v1":
            raise ValueError("Unknown derived-candidate schema.")
        from kanachan.candidate_features import candidate_feature_schema
        schema["derived_candidates"] = candidate_feature_schema()
    return schema


def require_tensor(condition: Tensor, message: str) -> None:
    """Keep data-dependent validation on device under CUDA and compilation."""
    if condition.device.type == "cuda" or torch.compiler.is_compiling():
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def legal_actions(candidates: Tensor) -> tuple[Tensor, Tensor]:
    if candidates.dtype not in (torch.int32, torch.int64):
        raise ValueError("Candidate IDs must be integers.")
    if candidates.ndim != 2 or candidates.shape[1] != MAX_NUM_ACTION_CANDIDATES:
        raise ValueError("Invalid candidate tensor shape.")
    require_tensor(
        ((candidates >= 0) & (candidates <= NUM_TYPES_OF_ACTIONS)).all(),
        "Candidate ID out of range.",
    )
    valid = candidates != NUM_TYPES_OF_ACTIONS
    require_tensor(valid.any(-1).all(), "Current observation has no legal action.")
    # Integer accumulation is independent of candidate ordering.
    counts = torch.zeros(
        (candidates.shape[0], NUM_TYPES_OF_ACTIONS + 1),
        device=candidates.device, dtype=torch.int32,
    )
    counts.scatter_add_(1, candidates.long(), valid.to(torch.int32))
    require_tensor(
        (counts[:, :NUM_TYPES_OF_ACTIONS] <= 1).all(),
        "Duplicate legal action IDs.",
    )
    return valid, counts[:, :NUM_TYPES_OF_ACTIONS].bool()


def legal_argmax(candidates: Tensor, values: Tensor) -> Tensor:
    """Return original candidate indices; break exact ties by global ID."""
    valid, _ = legal_actions(candidates)
    return _legal_argmax(candidates, values, valid)


def _legal_argmax(candidates: Tensor, values: Tensor, valid: Tensor) -> Tensor:
    """Internal path for candidate IDs already validated in this forward."""
    if values.shape != candidates.shape:
        raise ValueError("Candidate Q values have the wrong shape.")
    require_tensor(
        torch.isfinite(values.masked_fill(~valid, 0)).all(),
        "Nonfinite legal action values.",
    )
    masked = values.masked_fill(~valid, -torch.inf)
    maxima = masked.amax(-1, keepdim=True)
    ids = candidates.masked_fill(
        ~valid | (masked != maxima), NUM_TYPES_OF_ACTIONS
    )
    selected = ids.amin(-1, keepdim=True)
    return (candidates == selected).to(torch.int32).argmax(-1).to(torch.int32)


def rotary_embeddings(positions, head_dimension, base):
    # Compute once per forward, shared by Q/K and all layers. Reconstruct in
    # FP32 so module.half()/bfloat16() cannot round a frequency buffer.
    frequencies = base ** (
        -torch.arange(0, head_dimension, 2, device=positions.device,
                      dtype=torch.float32) / head_dimension
    )
    angles = positions.float()[..., None] * frequencies
    return angles.cos()[:, None], angles.sin()[:, None]


@dataclass(frozen=True)
class DecisionContext:
    environment: Hashable
    game: Hashable
    seat: int
    round_instance: Hashable
    model_version: Hashable

    def __post_init__(self):
        if type(self.seat) is not int or not 0 <= self.seat < 4:
            raise ValueError("Cache context requires an observer seat in [0,4).")
        for value in (
            self.environment, self.game, self.round_instance, self.model_version
        ):
            if value is None:
                raise ValueError("Cache identity fields cannot be None.")
            hash(value)


@dataclass
class HistoryCache:
    identity: tuple
    ids: tuple[int, ...] = ()
    keys: list[Tensor] = field(default_factory=list)
    values: list[Tensor] = field(default_factory=list)


class DecisionBlock(nn.Module):
    def __init__(
        self, dimension, num_heads, hidden_dimension, rope_base, *,
        device, dtype, attention_backend="sdpa",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dimension = dimension // num_heads
        self.rope_base = rope_base
        self.attention_backend = attention_backend
        self.attention_norm = RMSNorm(
            dimension, device=device, dtype=dtype, epsilon=1e-6
        )
        self.feedforward_norm = RMSNorm(
            dimension, device=device, dtype=dtype, epsilon=1e-6
        )
        args = {"device": device, "dtype": dtype, "bias": False}
        self.qkv = nn.Linear(dimension, 3 * dimension, **args)
        self.attention_output = nn.Linear(dimension, dimension, **args)
        self.gate_up = nn.Linear(dimension, 2 * hidden_dimension, **args)
        self.down = nn.Linear(hidden_dimension, dimension, **args)

    def rotate(self, value: Tensor, positions: Tensor, rotary=None) -> Tensor:
        if rotary is None:
            rotary = rotary_embeddings(positions, self.head_dimension, self.rope_base)
        cosine, sine = (item.to(value.dtype) for item in rotary)
        even, odd = value[..., 0::2], value[..., 1::2]
        return torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), -1
        ).flatten(-2)

    def project(self, value: Tensor, positions: Tensor, rotary=None):
        if rotary is None:
            rotary = rotary_embeddings(positions, self.head_dimension, self.rope_base)
        batch, length, dimension = value.shape
        packed = self.qkv(self.attention_norm(value)).reshape(
            batch, length, 3, self.num_heads, self.head_dimension
        )
        query, key, projected = packed.permute(2, 0, 3, 1, 4).unbind(0)
        rotary = tuple(item.to(query.dtype) for item in rotary)
        return (
            self.rotate(query, positions, rotary),
            self.rotate(key, positions, rotary), projected,
        )

    def attend(self, value, query, key, projected, allowed, *, backend=None):
        batch, length, dimension = value.shape
        attended = decision_attention(
            query, key, projected, allowed,
            self.attention_backend if backend is None else backend,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, dimension)
        value = value + self.attention_output(attended)
        gate, up = self.gate_up(self.feedforward_norm(value)).chunk(2, -1)
        return value + self.down(F.silu(gate) * up)

    def forward(self, value, positions, allowed, rotary=None):
        return self.attend(value, *self.project(value, positions, rotary), allowed)


class DenseDecisionDecoder(nn.Module):
    architecture_id = ARCHITECTURE_ID

    def __init__(
        self, *, dimension: int, num_heads: int, dim_feedforward: int,
        num_layers: int, device: torch.device, dtype: torch.dtype,
        checkpointing: bool = False, numeric_dimension: int | None = None,
        rope_base: float = 10000.0, history_attention: str = "causal",
        schema: dict | None = None,
        attention_backend: str = "sdpa",
    ):
        super().__init__()
        if (
            dimension <= 0 or num_heads <= 0 or dimension % num_heads
            or (dimension // num_heads) % 2 or dimension % 2
            or num_layers <= 0 or dim_feedforward <= 0
        ):
            raise ValueError("Invalid decoder dimensions or even RoPE head width.")
        if not math.isfinite(rope_base) or rope_base <= 0:
            raise ValueError("Invalid RoPE base.")
        if history_attention not in ("causal", "bidirectional"):
            raise ValueError("Unknown history attention mode.")
        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError("Unknown decision attention backend.")
        if schema is not None and schema != feature_schema():
            raise ValueError("Decoder feature schema differs from this runtime.")
        self.dimension = dimension
        self.numeric_dimension = (
            dimension // 2 if numeric_dimension is None else numeric_dimension
        )
        if self.numeric_dimension <= 0:
            raise ValueError("Numeric bin count must be positive.")
        self.checkpointing = checkpointing
        self.history_attention = history_attention
        self.attention_backend = attention_backend
        self.history_width = 1 + MAX_LENGTH_OF_PROGRESSION_FEATURES
        self.state_width = MAX_NUM_ACTIVE_SPARSE_FEATURES + NUM_NUMERIC_FEATURES + 1
        self.width = self.history_width + self.state_width + 1
        args = {"device": device, "dtype": dtype}
        self.sparse_embedding = nn.Embedding(
            NUM_TYPES_OF_SPARSE_FEATURES + 1, dimension,
            padding_idx=NUM_TYPES_OF_SPARSE_FEATURES, **args,
        )
        self.progression_embedding = nn.Embedding(
            NUM_TYPES_OF_PROGRESSION_FEATURES + 1, dimension,
            padding_idx=NUM_TYPES_OF_PROGRESSION_FEATURES, **args,
        )
        self.action_embedding = nn.Embedding(
            NUM_TYPES_OF_ACTIONS + 1, dimension,
            padding_idx=NUM_TYPES_OF_ACTIONS, **args,
        )
        self.numeric_projection = nn.Parameter(torch.empty(
            NUM_NUMERIC_FEATURES, self.numeric_dimension, dimension, **args
        ))
        self.numeric_identity = nn.Embedding(NUM_NUMERIC_FEATURES, dimension, **args)
        self.type_embedding = nn.Embedding(5, dimension, **args)
        self.bos = nn.Parameter(torch.empty(dimension, **args))
        self.decision = nn.Parameter(torch.empty(dimension, **args))
        self.action_count = nn.Parameter(torch.empty(dimension, **args))
        self.blocks = nn.ModuleList([
            DecisionBlock(
                dimension, num_heads, dim_feedforward, rope_base,
                attention_backend=attention_backend, **args
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dimension, epsilon=1e-6, **args)
        mask = torch.zeros(self.width, self.width, device=device, dtype=torch.bool)
        history = torch.ones(
            self.history_width, self.history_width, device=device, dtype=torch.bool
        )
        mask[:self.history_width, :self.history_width] = (
            history.tril() if history_attention == "causal" else history
        )
        mask[self.history_width:-1, :-1] = True
        mask[-1, :] = True
        self.register_buffer("base_mask", mask, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        for parameter in (
            self.numeric_projection, self.bos, self.decision, self.action_count
        ):
            nn.init.normal_(parameter, std=0.02)
        for block in self.blocks:
            for output in (block.attention_output, block.down):
                nn.init.normal_(
                    output.weight, std=0.02 / math.sqrt(2 * len(self.blocks))
                )
        with torch.no_grad():
            for table in (
                self.sparse_embedding, self.progression_embedding, self.action_embedding
            ):
                table.weight[table.padding_idx].zero_()

    def validate(self, sparse, numeric, progression, candidates):
        batch = sparse.shape[0]
        for name, value, width in (
            ("sparse", sparse, MAX_NUM_ACTIVE_SPARSE_FEATURES),
            ("numeric", numeric, NUM_NUMERIC_FEATURES),
            ("progression", progression, MAX_LENGTH_OF_PROGRESSION_FEATURES),
            ("candidates", candidates, MAX_NUM_ACTION_CANDIDATES),
        ):
            if (
                value.ndim != 2 or value.shape != (batch, width)
                or value.dtype not in (torch.int32, torch.int64)
                or value.device != self.bos.device
            ):
                raise ValueError(f"Invalid {name} shape, integer dtype or device.")
        require_tensor(
            ((sparse >= 0) & (sparse <= NUM_TYPES_OF_SPARSE_FEATURES)).all(),
            "Sparse feature ID out of range.",
        )
        require_tensor(
            ((progression >= 0) & (
                progression <= NUM_TYPES_OF_PROGRESSION_FEATURES
            )).all(), "History event ID out of range.",
        )
        valid_history = progression != NUM_TYPES_OF_PROGRESSION_FEATURES
        require_tensor(
            ~((~valid_history[:, :-1]) & valid_history[:, 1:]).any(),
            "History padding must be a contiguous right suffix.",
        )
        _, legal = legal_actions(candidates)
        return valid_history, legal

    def embed_state(self, sparse, numeric, legal):
        sparse_tokens = self.sparse_embedding(sparse.long()) + self.type_embedding.weight[1]
        numeric_features = encode_numeric_features(
            numeric, self.numeric_dimension,
            device=numeric.device, dtype=self.numeric_projection.dtype,
        )
        numeric_tokens = torch.einsum(
            "bik,ikd->bid", numeric_features, self.numeric_projection
        )
        numeric_tokens = (
            numeric_tokens + self.numeric_identity.weight + self.type_embedding.weight[2]
        )
        # Force FP32 pooling even when an outer autocast context is active.
        with torch.autocast(device_type=sparse.device.type, enabled=False):
            count = legal.sum(-1, keepdim=True).float()
            pooled = (
                legal.float() @ self.action_embedding.weight[:NUM_TYPES_OF_ACTIONS].float()
            ) / count
            pooled = pooled + count / MAX_NUM_ACTION_CANDIDATES * self.action_count.float()
            pooled = pooled + self.type_embedding.weight[3].float()
        pooled = pooled.to(sparse_tokens.dtype).unsqueeze(1)
        decision = (
            self.decision + self.type_embedding.weight[4]
        ).expand(sparse.shape[0], 1, -1)
        return torch.cat((sparse_tokens, numeric_tokens, pooled, decision), 1)

    def embed_history(self, progression, include_bos=True):
        events = self.progression_embedding(progression.long())
        events = events + self.type_embedding.weight[0]
        if not include_bos:
            return events
        bos = (self.bos + self.type_embedding.weight[0]).expand(
            progression.shape[0], 1, -1
        )
        return torch.cat((bos, events), 1)

    @staticmethod
    def mask_padding(base, valid):
        allowed = base[None, None] & valid[:, None, None, :]
        allowed = allowed & valid[:, None, :, None]
        # Invalid queries read BOS, then are zeroed after each block.
        allowed[..., 0] |= ~valid[:, None, :]
        return allowed

    def tokens(self, sparse, numeric, progression, candidates):
        valid_history, legal = self.validate(sparse, numeric, progression, candidates)
        return self._tokens(sparse, numeric, progression, valid_history, legal)

    def _tokens(self, sparse, numeric, progression, valid_history, legal):
        length = valid_history.sum(-1)
        value = torch.cat((
            self.embed_history(progression),
            self.embed_state(sparse, numeric, legal),
        ), 1)
        valid = torch.cat((
            torch.ones_like(valid_history[:, :1]), valid_history,
            sparse != NUM_TYPES_OF_SPARSE_FEATURES,
            torch.ones(
                sparse.shape[0], NUM_NUMERIC_FEATURES + 2,
                device=sparse.device, dtype=torch.bool,
            ),
        ), 1)
        positions = torch.cat((
            torch.arange(self.history_width, device=sparse.device).expand(
                sparse.shape[0], -1
            ),
            (length + 1)[:, None].expand(-1, self.state_width + 1),
        ), 1)
        return (
            value.masked_fill(~valid[..., None], 0),
            positions, self.mask_padding(self.base_mask, valid), valid,
        )

    def forward(self, sparse, numeric, progression, candidates):
        representation, _ = self.forward_with_validity(
            sparse, numeric, progression, candidates
        )
        return representation

    def forward_with_validity(self, sparse, numeric, progression, candidates):
        """Share strict candidate validation with a scalar head in Q training."""
        valid_history, legal = self.validate(sparse, numeric, progression, candidates)
        value, positions, allowed, valid = self._tokens(
            sparse, numeric, progression, valid_history, legal
        )
        rotary = rotary_embeddings(
            positions, self.blocks[0].head_dimension, self.blocks[0].rope_base
        )
        for block in self.blocks:
            if self.checkpointing and self.training and torch.is_grad_enabled():
                value = checkpoint(
                    block, value, positions, allowed, rotary, use_reentrant=False
                )
            else:
                value = block(value, positions, allowed, rotary)
            value = value.masked_fill(~valid[..., None], 0)
        return self.norm(value[:, -1]), candidates != NUM_TYPES_OF_ACTIONS

    @torch.no_grad()
    def incremental(
        self, sparse, numeric, progression, candidates, *,
        context: DecisionContext, cache: HistoryCache | None = None,
    ) -> tuple[Tensor, HistoryCache]:
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise ValueError("History caches require an eval, frozen inference model.")
        if self.history_attention != "causal":
            raise ValueError("Bidirectional history cannot use an append-only KV cache.")
        if sparse.ndim != 2 or sparse.shape[0] != 1:
            raise ValueError("Each history cache belongs to exactly one observer stream.")
        valid_history, legal = self.validate(sparse, numeric, progression, candidates)
        ids = tuple(progression[0][valid_history[0]].tolist())
        identity = (
            id(self), context, str(self.bos.device), self.bos.dtype,
            torch.is_autocast_enabled(self.bos.device.type),
            torch.get_autocast_dtype(self.bos.device.type),
            tuple(p._version for p in self.parameters()),
        )
        if (
            cache is None or cache.identity != identity
            or ids[:len(cache.ids)] != cache.ids
            or len(cache.keys) != len(self.blocks)
            or len(cache.values) != len(self.blocks)
        ):
            cache = HistoryCache(identity)
        old_length = cache.keys[0].shape[-2] if cache.keys else 0
        if cache.keys and any(
            k.shape[-2] != old_length or v.shape[-2] != old_length
            for k, v in zip(cache.keys, cache.values)
        ):
            raise ValueError("Inconsistent layer cache lengths.")
        # Reuse the input device tensor rather than uploading the same IDs again.
        new_ids = progression[:, len(cache.ids):len(ids)]
        new_value = self.embed_history(new_ids, include_bos=not cache.keys)
        new_length = new_value.shape[1]
        if new_length:
            positions = torch.arange(
                old_length, old_length + new_length, device=progression.device
            )[None]
            allowed = (
                torch.arange(old_length + new_length, device=progression.device)[None, :]
                <= positions[0, :, None]
            )[None, None]
            rotary = rotary_embeddings(
                positions, self.blocks[0].head_dimension, self.blocks[0].rope_base
            )
            for i, block in enumerate(self.blocks):
                query, key, projected = block.project(new_value, positions, rotary)
                if old_length:
                    key = torch.cat((cache.keys[i], key), -2)
                    projected = torch.cat((cache.values[i], projected), -2)
                    cache.keys[i], cache.values[i] = key, projected
                else:
                    cache.keys.append(key)
                    cache.values.append(projected)
                # Appended history has tiny, changing shapes. Keep its native
                # SDPA path instead of compiling a new Flex kernel per append.
                new_value = block.attend(
                    new_value, query, key, projected, allowed, backend="sdpa"
                )
        cache.ids = ids
        history_length = 1 + len(ids)
        value = self.embed_state(sparse, numeric, legal)
        valid = torch.cat((
            torch.ones(1, history_length, device=sparse.device, dtype=torch.bool),
            sparse != NUM_TYPES_OF_SPARSE_FEATURES,
            torch.ones(
                1, NUM_NUMERIC_FEATURES + 2, device=sparse.device, dtype=torch.bool
            ),
        ), 1)
        suffix_width = self.state_width + 1
        base = torch.ones(
            suffix_width, history_length + suffix_width,
            device=sparse.device, dtype=torch.bool,
        )
        base[:-1, -1] = False
        allowed = base[None, None] & valid[:, None, None, :]
        suffix_valid = valid[:, history_length:]
        allowed &= suffix_valid[:, None, :, None]
        allowed[..., 0] |= ~suffix_valid[:, None, :]
        value = value.masked_fill(~suffix_valid[..., None], 0)
        positions = torch.full(
            (1, suffix_width), history_length,
            device=sparse.device, dtype=torch.int64,
        )
        rotary = rotary_embeddings(
            positions, self.blocks[0].head_dimension, self.blocks[0].rope_base
        )
        for i, block in enumerate(self.blocks):
            query, key, projected = block.project(value, positions, rotary)
            key = torch.cat((cache.keys[i], key), -2)
            projected = torch.cat((cache.values[i], projected), -2)
            value = block.attend(
                value, query, key, projected, allowed, backend="sdpa"
            )
            value = value.masked_fill(~suffix_valid[..., None], 0)
        return self.norm(value[:, -1]), cache


class GlobalActionHead(nn.Module):
    """Finite scalar logits/Q, gathered to the original candidate order."""

    def __init__(
        self, *, input_dimension: int, hidden_dimension: int,
        device: torch.device, dtype: torch.dtype,
        candidate_features: str = "none", candidate_schema: dict | None = None,
    ):
        super().__init__()
        if input_dimension <= 0 or hidden_dimension <= 0:
            raise ValueError("Invalid action-head dimensions.")
        if candidate_features not in ("none", "discard_v1"):
            raise ValueError("Unknown candidate-feature mode.")
        self.candidate_features = candidate_features
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dimension, NUM_TYPES_OF_ACTIONS, device=device, dtype=dtype),
        )
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        self.candidate_embedding = None
        self.candidate_layers = None
        if candidate_features != "none":
            from kanachan.candidate_features import FEATURE_WIDTH, candidate_feature_schema, native_backend
            if candidate_schema is not None and candidate_schema != candidate_feature_schema():
                raise ValueError("Candidate head schema differs from this runtime.")
            native_backend()
            self.candidate_embedding = nn.Embedding(NUM_TYPES_OF_ACTIONS + 1, hidden_dimension, device=device, dtype=dtype)
            self.candidate_layers = nn.Sequential(
                nn.Linear(2 * hidden_dimension + FEATURE_WIDTH, hidden_dimension, device=device, dtype=dtype),
                nn.SiLU(), nn.Linear(hidden_dimension, 1, device=device, dtype=dtype),
            )
            nn.init.normal_(self.candidate_embedding.weight, std=0.02)
            for module in self.candidate_layers:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    nn.init.zeros_(module.bias)
            # Start as the existing global scorer; learn a shared correction.
            nn.init.zeros_(self.candidate_layers[-1].weight)
            nn.init.zeros_(self.candidate_layers[-1].bias)

    @torch.no_grad()
    def zero_output(self):
        self.layers[-1].weight.zero_()
        self.layers[-1].bias.zero_()
        if self.candidate_layers is not None:
            self.candidate_layers[-1].weight.zero_()
            self.candidate_layers[-1].bias.zero_()

    def global_values(self, representation):
        if representation.ndim != 2:
            raise ValueError("Expected one decision representation per observation.")
        # Q and BC logits are evaluated in FP32 even inside AMP.
        dtype = torch.float64 if representation.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=representation.device.type, enabled=False):
            value = F.linear(
                representation.to(dtype), self.layers[0].weight.to(dtype),
                self.layers[0].bias.to(dtype),
            )
            return F.linear(
                F.silu(value), self.layers[-1].weight.to(dtype),
                self.layers[-1].bias.to(dtype),
            )

    def forward(self, candidates, representation, sparse=None, progression=None):
        valid, _ = legal_actions(candidates)
        return self._from_validated(candidates, representation, valid, sparse, progression)

    def _from_validated(self, candidates, representation, valid, sparse=None, progression=None):
        values = self.global_values(representation)
        safe_ids = candidates.masked_fill(~valid, 0).long()
        result = values.gather(1, safe_ids)
        if self.candidate_layers is not None:
            if sparse is None or progression is None:
                raise ValueError("Candidate afterstates require observer sparse and progression inputs.")
            from kanachan.candidate_features import derive_candidate_features
            features = derive_candidate_features(sparse, progression, candidates)
            dtype = values.dtype
            with torch.autocast(device_type=representation.device.type, enabled=False):
                state = F.silu(F.linear(representation.to(dtype), self.layers[0].weight.to(dtype), self.layers[0].bias.to(dtype)))
                identity = F.embedding(safe_ids, self.candidate_embedding.weight.to(dtype))
                inputs = torch.cat((state[:, None].expand(-1, candidates.shape[1], -1), identity, features.to(dtype)), -1)
                hidden = F.silu(F.linear(inputs, self.candidate_layers[0].weight.to(dtype), self.candidate_layers[0].bias.to(dtype)))
                result = result + F.linear(hidden, self.candidate_layers[-1].weight.to(dtype), self.candidate_layers[-1].bias.to(dtype)).squeeze(-1)
        return result.masked_fill(~valid, 0)


@torch.no_grad()
def cached_decision(
    backbone: DenseDecisionDecoder, head: GlobalActionHead,
    observation, *, context: DecisionContext, cache: HistoryCache | None = None,
):
    """Explicit observer-scoped inference; return native candidate indices."""
    representation, cache = backbone.incremental(
        *(observation[key] for key in (
            "sparse", "numeric", "progression", "candidates"
        )), context=context, cache=cache,
    )
    values = head(observation["candidates"], representation, observation["sparse"], observation["progression"])
    return legal_argmax(observation["candidates"], values), cache
