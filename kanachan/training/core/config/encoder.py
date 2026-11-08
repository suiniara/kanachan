from pathlib import Path
import logging
import math
from typing import Any
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


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
        "transformer", "rmsnorm_swiglu", "dense_decoder"
    ):
        raise RuntimeError(
            f"Unknown encoder architecture: {encoder_config.architecture}"
        )
    if (
        encoder_config.architecture in ("rmsnorm_swiglu", "dense_decoder")
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

    if encoder_config.dim_feedforward is None:
        encoder_config.dim_feedforward = (
            max(8, round((8 * encoder_config.dimension / 3) / 8) * 8)
            if encoder_config.architecture == "dense_decoder"
            else 4 * encoder_config.dimension
        )
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
            "SwiGLU" if encoder_config.architecture in ("rmsnorm_swiglu", "dense_decoder")
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
