import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint_sequential

from kanachan.constants import (
    ENCODER_WIDTH,
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_ACTIONS,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_SPARSE_FEATURES,
)
from kanachan.nn.modern_encoder import RMSNormSwiGLUEncoder
from kanachan.nn.numeric_encoding import encode_numeric_features
from kanachan.nn.position_embedding import PositionEmbedding
from kanachan.nn.positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        position_encoder: str,
        dimension: int,
        num_heads: int,
        dim_feedforward: int,
        activation_function: str,
        dropout: float,
        layer_normalization: bool = False,
        architecture: str = "transformer",
        num_layers: int,
        checkpointing: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if position_encoder not in (
            "positional_encoding",
            "position_embedding",
        ):
            raise ValueError(position_encoder)
        if dimension <= 0:
            raise ValueError(dimension)
        if num_heads <= 0:
            raise ValueError(num_heads)
        if dim_feedforward <= 0:
            raise ValueError(dim_feedforward)
        if activation_function not in ("relu", "gelu"):
            raise ValueError(activation_function)
        if dropout < 0.0 or 1.0 <= dropout:
            raise ValueError(dropout)
        if num_layers <= 0:
            raise ValueError(num_layers)
        if architecture not in ("transformer", "rmsnorm_swiglu"):
            raise ValueError(architecture)
        if architecture == "rmsnorm_swiglu" and layer_normalization:
            raise ValueError(
                "rmsnorm_swiglu supplies its own per-token normalization."
            )
        if dtype not in (torch.float64, torch.float32, torch.float16):
            raise ValueError(dtype)

        super().__init__()

        self.__dimension = dimension
        self.__architecture = architecture

        self.sparse_embedding = nn.Embedding(
            NUM_TYPES_OF_SPARSE_FEATURES + 1,
            self.__dimension,
            padding_idx=NUM_TYPES_OF_SPARSE_FEATURES,
            device=device,
            dtype=dtype,
        )

        self.numeric_embedding = nn.Embedding(
            NUM_NUMERIC_FEATURES,
            self.__dimension // 2,
            device=device,
            dtype=dtype,
        )

        self.progression_embedding = nn.Embedding(
            NUM_TYPES_OF_PROGRESSION_FEATURES + 1,
            self.__dimension,
            padding_idx=NUM_TYPES_OF_PROGRESSION_FEATURES,
            device=device,
            dtype=dtype,
        )
        self.position_encoder: nn.Module
        if position_encoder == "positional_encoding":
            self.position_encoder = PositionalEncoding(
                max_length=MAX_LENGTH_OF_PROGRESSION_FEATURES,
                dimension=self.__dimension,
                dropout=dropout,
                device=device,
                dtype=dtype,
            )
        elif position_encoder == "position_embedding":
            self.position_encoder = PositionEmbedding(
                max_length=MAX_LENGTH_OF_PROGRESSION_FEATURES,
                dimension=self.__dimension,
                dropout=dropout,
                device=device,
                dtype=dtype,
            )
        else:
            raise NotImplementedError(position_encoder)

        self.candidates_embedding = nn.Embedding(
            NUM_TYPES_OF_ACTIONS + 1,
            self.__dimension,
            padding_idx=NUM_TYPES_OF_ACTIONS,
            device=device,
            dtype=dtype,
        )

        _layer_normalization: nn.LayerNorm | None = None
        if layer_normalization:
            _layer_normalization = nn.LayerNorm(
                [ENCODER_WIDTH, self.__dimension], device=device, dtype=dtype
            )

        if architecture == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                self.__dimension,
                num_heads,
                dim_feedforward=dim_feedforward,
                activation=activation_function,
                dropout=dropout,
                batch_first=True,
                device=device,
                dtype=dtype,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers, norm=_layer_normalization
            )
        else:
            self.encoder = RMSNormSwiGLUEncoder(
                dimension, num_heads, dim_feedforward, dropout, num_layers,
                device=device, dtype=dtype,
            )

        self.checkpointing = checkpointing

    def __create_numeric(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        numeric: Tensor,
    ) -> Tensor:
        _numeric = encode_numeric_features(
            numeric, self.__dimension // 2, device=device, dtype=dtype
        )
        numeric_embedding: Tensor = self.numeric_embedding(
            torch.arange(
                NUM_NUMERIC_FEATURES, device=device, dtype=torch.int32
            )
        )
        numeric_embedding = numeric_embedding.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        return torch.cat((_numeric, numeric_embedding), 2)

    @torch.compile
    def __encode(
        self, embedding: Tensor, padding_mask: Tensor | None = None
    ) -> Tensor:
        if self.__architecture == "rmsnorm_swiglu":
            return self.encoder(
                embedding, padding_mask, checkpointing=self.checkpointing
            )
        if self.checkpointing:
            encoder_layers = self.encoder.layers
            result = checkpoint_sequential(
                encoder_layers,
                len(encoder_layers),
                embedding,
                use_reentrant=False,
            )  # type: ignore
            if self.encoder.norm is not None:
                result = self.encoder.norm(result)
            return result
        return self.encoder(embedding)

    def forward(
        self,
        sparse: Tensor,
        numeric: Tensor,
        progression: Tensor,
        candidates: Tensor,
    ) -> Tensor:
        device = sparse.device
        batch_size = int(sparse.size(0))

        assert isinstance(sparse, Tensor)
        assert sparse.device == device
        assert sparse.dtype == torch.int32
        assert sparse.dim() == 2
        assert sparse.size(0) == batch_size
        assert sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES

        assert isinstance(numeric, Tensor)
        assert numeric.device == device
        assert numeric.dtype == torch.int32
        assert numeric.dim() == 2
        assert numeric.size(0) == batch_size
        assert numeric.size(1) == NUM_NUMERIC_FEATURES

        assert isinstance(progression, Tensor)
        assert progression.device == device
        assert progression.dtype == torch.int32
        assert progression.dim() == 2
        assert progression.size(0) == batch_size
        assert progression.size(1) == MAX_LENGTH_OF_PROGRESSION_FEATURES

        assert isinstance(candidates, Tensor)
        assert candidates.device == device
        assert candidates.dtype == torch.int32
        assert candidates.dim() == 2
        assert candidates.size(0) == batch_size
        assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES

        padding_mask = None
        if self.__architecture == "rmsnorm_swiglu":
            padding_mask = torch.cat(
                (
                    sparse == NUM_TYPES_OF_SPARSE_FEATURES,
                    torch.zeros(
                        batch_size, NUM_NUMERIC_FEATURES,
                        device=device, dtype=torch.bool,
                    ),
                    progression == NUM_TYPES_OF_PROGRESSION_FEATURES,
                    candidates == NUM_TYPES_OF_ACTIONS,
                ),
                dim=1,
            )
        sparse = self.sparse_embedding(sparse)

        dtype = sparse.dtype
        numeric = self.__create_numeric(device, dtype, batch_size, numeric)

        progression = self.progression_embedding(progression)
        progression = self.position_encoder(progression)

        candidates = self.candidates_embedding(candidates)

        embedding = torch.cat((sparse, numeric, progression, candidates), 1)

        return self.__encode(embedding, padding_mask)
