from pathlib import Path
import logging
import math
from typing import Any
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


from kanachan.nn.structured_dual_decoder import STRUCTURED_V2_PRESETS, swiglu_ffn_dim


@dataclass
class EncoderConfig:
    position_encoder: str
    dimension: int
    num_heads: int
    dim_feedforward: None | int = None
    activation_function: str = "gelu"
    dropout: float = 0.1
    num_layers: int = 4
    layer_normalization: bool = False
    load_from: None | Path = None
    architecture: str = "transformer"
    numeric_dimension: int | None = None
    rope_base: float = 10000.0
    history_attention: str = "causal"
    attention_backend: str = "sdpa"
    preset: str | None = None
    n_d1_layers: int | None = None
    n_d2_layers: int | None = None
    use_rel_bias: bool = True
    cand_self_attn: bool = True
    head_type: str = "factorized"
    enable_early_shape_hint: bool = False
    include_room_level: bool = False


@dataclass
class DecisionTinyConfig(EncoderConfig):
    position_encoder: str = "rope"
    dimension: int = 128
    num_heads: int = 2
    dim_feedforward: int | None = 192
    activation_function: str = "silu"
    dropout: float = 0.0
    num_layers: int = 2
    architecture: str = "dense_decoder"


@dataclass
class DecisionSmallConfig(DecisionTinyConfig):
    dimension: int = 256
    num_heads: int = 4
    dim_feedforward: int | None = 704
    num_layers: int = 6


@dataclass
class StructuredV2BaseEncoderConfig(EncoderConfig):
    position_encoder: str = "rope"
    dimension: int = 384
    num_heads: int = 8
    dim_feedforward: int | None = 1024
    activation_function: str = "silu"
    dropout: float = 0.0
    num_layers: int = 12
    layer_normalization: bool = False
    architecture: str = "structured_dual_v2"
    preset: str | None = "v2_base"
    n_d1_layers: int | None = 8
    n_d2_layers: int | None = 4
    use_rel_bias: bool = True
    cand_self_attn: bool = True
    head_type: str = "factorized"


@dataclass
class StructuredV2TinyEncoderConfig(StructuredV2BaseEncoderConfig):
    preset: str | None = "v2_tiny"
    dimension: int = 192
    num_heads: int = 6
    dim_feedforward: int | None = 512
    n_d1_layers: int | None = 4
    n_d2_layers: int | None = 2
    num_layers: int = 6


@dataclass
class StructuredV2SmallEncoderConfig(StructuredV2BaseEncoderConfig):
    preset: str | None = "v2_small"
    dimension: int = 256
    num_heads: int = 8
    dim_feedforward: int | None = 688
    n_d1_layers: int | None = 8
    n_d2_layers: int | None = 4
    num_layers: int = 12


@dataclass
class StructuredV2SmallNoBiasEncoderConfig(StructuredV2SmallEncoderConfig):
    preset: str | None = "v2_small_no_bias"
    use_rel_bias: bool = False


@dataclass
class StructuredV2SpatialOnlyEncoderConfig(StructuredV2SmallEncoderConfig):
    preset: str | None = "spatial_only"
    n_d1_layers: int | None = 0
    n_d2_layers: int | None = 6
    num_layers: int = 6


@dataclass
class StructuredV2BaseFlatEncoderConfig(StructuredV2BaseEncoderConfig):
    preset: str | None = "v2_base_flat"
    head_type: str = "flat546"


@dataclass
class StructuredV2LargeEncoderConfig(StructuredV2BaseEncoderConfig):
    preset: str | None = "v2_large"
    dimension: int = 512
    num_heads: int = 8
    dim_feedforward: int | None = 1376
    n_d1_layers: int | None = 10
    n_d2_layers: int | None = 6
    num_layers: int = 16


@dataclass
class BertTinyEncoderConfig(EncoderConfig):
    position_encoder: str = "position_embedding"
    dimension: int = 128
    num_heads: int = 2
    dim_feedforward: None | int = None
    activation_function: str = "gelu"
    dropout: float = 0.1
    num_layers: int = 2
    layer_normalization: bool = False
    load_from: None | Path = None


@dataclass
class BertMiniEncoderConfig(EncoderConfig):
    position_encoder: str = "position_embedding"
    dimension: int = 256
    num_heads: int = 4
    dim_feedforward: None | int = None
    activation_function: str = "gelu"
    dropout: float = 0.1
    num_layers: int = 4
    layer_normalization: bool = False
    load_from: None | Path = None


@dataclass
class BertSmallEncoderConfig(EncoderConfig):
    position_encoder: str = "position_embedding"
    dimension: int = 512
    num_heads: int = 8
    dim_feedforward: None | int = None
    activation_function: str = "gelu"
    dropout: float = 0.1
    num_layers: int = 4
    layer_normalization: bool = False
    load_from: None | Path = None


@dataclass
class BertMediumEncoderConfig(EncoderConfig):
    position_encoder: str = "position_embedding"
    dimension: int = 512
    num_heads: int = 8
    dim_feedforward: None | int = None
    activation_function: str = "gelu"
    dropout: float = 0.1
    num_layers: int = 8
    layer_normalization: bool = False
    load_from: None | Path = None


@dataclass
class BertBaseEncoderConfig(EncoderConfig):
    position_encoder: str = "position_embedding"
    dimension: int = 768
    num_heads: int = 12
    dim_feedforward: None | int = None
    activation_function: str = "gelu"
    dropout: float = 0.1
    num_layers: int = 12
    layer_normalization: bool = False
    load_from: None | Path = None


@dataclass
class BertLargeEncoderConfig(EncoderConfig):
    position_encoder: str = "position_embedding"
    dimension: int = 1024
    num_heads: int = 16
    dim_feedforward: None | int = None
    activation_function: str = "gelu"
    dropout: float = 0.1
    num_layers: int = 24
    layer_normalization: bool = False
    load_from: None | Path = None


config_store = ConfigStore.instance()
config_store.store(name="decision_tiny", node=DecisionTinyConfig, group="encoder")
config_store.store(name="decision_small", node=DecisionSmallConfig, group="encoder")
config_store.store(name="v2_tiny", node=StructuredV2TinyEncoderConfig, group="encoder")
config_store.store(name="v2_small", node=StructuredV2SmallEncoderConfig, group="encoder")
config_store.store(name="v2_small_no_bias", node=StructuredV2SmallNoBiasEncoderConfig, group="encoder")
config_store.store(name="spatial_only", node=StructuredV2SpatialOnlyEncoderConfig, group="encoder")
config_store.store(name="v2_base", node=StructuredV2BaseEncoderConfig, group="encoder")
config_store.store(name="v2_base_flat", node=StructuredV2BaseFlatEncoderConfig, group="encoder")
config_store.store(name="v2_large", node=StructuredV2LargeEncoderConfig, group="encoder")
config_store.store(
    name="bert_tiny", node=BertTinyEncoderConfig, group="encoder"
)
config_store.store(
    name="bert_mini", node=BertMiniEncoderConfig, group="encoder"
)
config_store.store(
    name="bert_small", node=BertSmallEncoderConfig, group="encoder"
)
config_store.store(
    name="bert_medium", node=BertMediumEncoderConfig, group="encoder"
)
config_store.store(
    name="bert_base", node=BertBaseEncoderConfig, group="encoder"
)
config_store.store(
    name="bert_large", node=BertLargeEncoderConfig, group="encoder"
)


def validate(encoder_config: Any) -> None:
    backend = getattr(encoder_config, "attention_backend", "sdpa")
    if backend not in ("sdpa", "flex"):
        raise ValueError("encoder.attention_backend must be sdpa or flex.")
    if (encoder_config.architecture != "dense_decoder"
            and backend != "sdpa"):
        raise ValueError("FlexAttention is implemented for the dense decoder.")
    if encoder_config.architecture not in (
        "transformer", "rmsnorm_swiglu", "dense_decoder", "structured_dual_v2"
    ):
        raise RuntimeError(
            f"Unknown encoder architecture: {encoder_config.architecture}"
        )
    if (
        encoder_config.architecture in ("rmsnorm_swiglu", "dense_decoder", "structured_dual_v2")
        and encoder_config.layer_normalization
    ):
        raise RuntimeError(
            "This architecture supplies its own normalization; "
            "leave encoder.layer_normalization=false."
        )
    if encoder_config.position_encoder not in (
        "positional_encoding",
        "position_embedding",
        "rope",
    ):
        errmsg = (
            f"{encoder_config.position_encoder}: "
            "An invalid position encoder."
        )
        raise RuntimeError(errmsg)

    if encoder_config.architecture == "dense_decoder":
        if (
            encoder_config.position_encoder != "rope"
            or encoder_config.activation_function != "silu"
            or encoder_config.dropout != 0.0
            or encoder_config.layer_normalization
        ):
            raise ValueError(
                "Dense decision decoder requires RoPE, SiLU, dropout=0 "
                "and layer_normalization=false."
            )
        if encoder_config.history_attention not in ("causal", "bidirectional"):
            raise ValueError("Unknown decoder history attention mode.")
        if encoder_config.numeric_dimension is not None and (
            encoder_config.numeric_dimension <= 0
        ):
            raise ValueError("Numeric bin count must be positive.")
        if not math.isfinite(encoder_config.rope_base) or encoder_config.rope_base <= 0:
            raise ValueError("RoPE base must be finite and positive.")
    elif encoder_config.architecture == "structured_dual_v2":
        preset_name = getattr(encoder_config, "preset", None)
        if preset_name is not None:
            if preset_name not in STRUCTURED_V2_PRESETS:
                raise ValueError(f"Unknown structured_v2 preset: {preset_name}")
            p = STRUCTURED_V2_PRESETS[preset_name]
            if getattr(encoder_config, "n_d1_layers", None) is None:
                encoder_config.n_d1_layers = p["n_d1_layers"]
            if getattr(encoder_config, "n_d2_layers", None) is None:
                encoder_config.n_d2_layers = p["n_d2_layers"]
        if getattr(encoder_config, "n_d1_layers", None) is None:
            encoder_config.n_d1_layers = 8
        if getattr(encoder_config, "n_d2_layers", None) is None:
            encoder_config.n_d2_layers = 4
        if (
            encoder_config.position_encoder != "rope"
            or encoder_config.activation_function != "silu"
            or encoder_config.dropout != 0.0
            or encoder_config.layer_normalization
        ):
            raise ValueError(
                "structured_dual_v2 requires RoPE, SiLU, dropout=0 and layer_normalization=false."
            )
        if getattr(encoder_config, "include_room_level", False):
            raise ValueError("structured_dual_v2 requires include_room_level=false.")
        if getattr(encoder_config, "head_type", "factorized") not in ("factorized", "flat546"):
            raise ValueError("structured_dual_v2 head_type must be factorized or flat546.")
        if encoder_config.n_d1_layers < 0 or encoder_config.n_d2_layers < 1:
            raise ValueError("structured_dual_v2 requires n_d1_layers >= 0 and n_d2_layers >= 1.")
        encoder_config.num_layers = int(encoder_config.n_d1_layers) + int(encoder_config.n_d2_layers)
        if not math.isfinite(encoder_config.rope_base) or encoder_config.rope_base <= 0:
            raise ValueError("RoPE base must be finite and positive.")
    elif encoder_config.position_encoder == "rope":
        raise ValueError("RoPE is supported by encoder=decision_tiny/decision_small.")

    if encoder_config.dimension <= 0:
        errmsg = (
            f"{encoder_config.dimension}: "
            "`encoder.dimension` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if encoder_config.num_heads <= 0:
        errmsg = (
            f"{encoder_config.num_heads}: "
            "`encoder.num_heads` must be a positive integer."
        )
        raise RuntimeError(errmsg)
    if encoder_config.architecture == "structured_dual_v2" and (
        encoder_config.dimension % encoder_config.num_heads != 0
    ):
        raise ValueError("structured_dual_v2 requires encoder.dimension divisible by num_heads.")

    if encoder_config.dim_feedforward is None:
        if encoder_config.architecture == "structured_dual_v2":
            encoder_config.dim_feedforward = swiglu_ffn_dim(encoder_config.dimension)
        elif encoder_config.architecture == "dense_decoder":
            encoder_config.dim_feedforward = max(8, round((8 * encoder_config.dimension / 3) / 8) * 8)
        else:
            encoder_config.dim_feedforward = 4 * encoder_config.dimension
    if encoder_config.architecture == "structured_dual_v2" and encoder_config.dim_feedforward % 16 != 0:
        raise ValueError("structured_dual_v2 requires encoder.dim_feedforward to be a multiple of 16.")
    if encoder_config.dim_feedforward <= 0:
        errmsg = (
            f"{encoder_config.dim_feedforward}: "
            "`encoder.dim_feedforward` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if encoder_config.activation_function not in ("relu", "gelu", "silu"):
        errmsg = (
            f"{encoder_config.activation_function}: "
            "An invalid activation function for the encoder."
        )
        raise RuntimeError(errmsg)

    if encoder_config.dropout < 0.0 or 1.0 <= encoder_config.dropout:
        errmsg = (
            f"{encoder_config.dropout}: `encoder.dropout` must be a real "
            "number within the range [0.0, 1.0)."
        )
        raise RuntimeError(errmsg)

    if encoder_config.num_layers <= 0:
        errmsg = (
            f"{encoder_config.num_layers}: "
            "`encoder.num_layers` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if encoder_config.load_from is not None:
        if not encoder_config.load_from.exists():
            errmsg = f"{encoder_config.load_from}: Does not exist."
            raise RuntimeError(errmsg)
        if not encoder_config.load_from.is_file():
            errmsg = f"{encoder_config.load_from}: Not a file."
            raise RuntimeError(errmsg)


def dump(encoder_config: Any, prefix: str = "") -> None:
    logging.info(
        "%sEncoder architecture: %s", prefix, encoder_config.architecture
    )
    logging.info(
        "%sPosition encoder: %s", prefix, encoder_config.position_encoder
    )
    logging.info("%sEncoder dimension: %d", prefix, encoder_config.dimension)
    logging.info(
        "%s# of heads for encoder: %d", prefix, encoder_config.num_heads
    )
    logging.info(
        "%sDimension of feedforward networks for encoder: %d",
        prefix,
        encoder_config.dim_feedforward,
    )
    logging.info(
        "%sActivation function for encoder: %s",
        prefix,
        (
            "SwiGLU" if encoder_config.architecture in ("rmsnorm_swiglu", "dense_decoder", "structured_dual_v2")
            else encoder_config.activation_function
        ),
    )
    logging.info("%sDropout for encoder: %f", prefix, encoder_config.dropout)
    if encoder_config.architecture == "dense_decoder":
        logging.info(
            "%sNumeric bins per field: %d; history: %s; RoPE base: %g",
            prefix, encoder_config.numeric_dimension or encoder_config.dimension // 2,
            encoder_config.history_attention, encoder_config.rope_base,
        )
        logging.info("%sAttention backend: %s (CPU uses SDPA)",
                     prefix, getattr(encoder_config, "attention_backend", "sdpa"))
    elif encoder_config.architecture == "structured_dual_v2":
        logging.info(
            "%sStructured V2 preset=%s d1=%s d2=%s rel_bias=%s cand_self_attn=%s head=%s early_hint=%s",
            prefix,
            getattr(encoder_config, "preset", None),
            getattr(encoder_config, "n_d1_layers", None),
            getattr(encoder_config, "n_d2_layers", None),
            getattr(encoder_config, "use_rel_bias", True),
            getattr(encoder_config, "cand_self_attn", True),
            getattr(encoder_config, "head_type", "factorized"),
            getattr(encoder_config, "enable_early_shape_hint", False),
        )
    logging.info(
        "%sLayer normalization for encoder: %s",
        prefix,
        encoder_config.layer_normalization,
    )
    logging.info(
        "%s# of encoder layers: %d", prefix, encoder_config.num_layers
    )
    if encoder_config.load_from is not None:
        logging.info(
            "%sLoad encoder from: %s", prefix, encoder_config.load_from
        )
