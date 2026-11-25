"""Shared training integration without changing legacy encoder constructors."""

import torch

from kanachan.nn.encoder import Encoder
from kanachan.nn.decoder import Decoder
from kanachan.nn.decision_decoder import (
    DenseDecisionDecoder, GlobalActionHead, feature_schema,
)
from kanachan.nn.decision_state import DecisionStateHead
from kanachan.nn.structured_dual_decoder import (
    StructuredActionHead,
    StructuredDualDecoder,
    StructuredStateHead,
    structured_v2_schema,
)
from kanachan.nn.structured_iql import V2_OBS_KEYS


OBSERVATION_KEYS = ["sparse", "numeric", "progression", "candidates"]


def backbone_in_keys(config, *, precomputed: bool = False) -> list[str]:
    if getattr(config, "architecture", None) == "structured_dual_v2" and precomputed:
        return [*OBSERVATION_KEYS, *V2_OBS_KEYS]
    return list(OBSERVATION_KEYS)


def backbone_kwargs(config, *, checkpointing, dtype):
    if config.architecture == "structured_dual_v2":
        return {
            "preset": getattr(config, "preset", None),
            "dimension": config.dimension,
            "num_heads": config.num_heads,
            "dim_feedforward": config.dim_feedforward,
            "n_d1_layers": getattr(config, "n_d1_layers", 8),
            "n_d2_layers": getattr(config, "n_d2_layers", 4),
            "use_rel_bias": getattr(config, "use_rel_bias", True),
            "cand_self_attn": getattr(config, "cand_self_attn", True),
            "head_type": getattr(config, "head_type", "factorized"),
            "enable_early_shape_hint": getattr(config, "enable_early_shape_hint", False),
            "include_room_level": bool(getattr(config, "include_room_level", False)),
            "rope_base": config.rope_base,
            "checkpointing": checkpointing,
            "schema": structured_v2_schema(),
            "device": torch.device("cpu"),
            "dtype": dtype,
        }
    arguments = {
        "dimension": config.dimension,
        "num_heads": config.num_heads,
        "dim_feedforward": config.dim_feedforward,
        "num_layers": config.num_layers,
        "checkpointing": checkpointing,
        "device": torch.device("cpu"),
        "dtype": dtype,
    }
    if config.architecture == "dense_decoder":
        arguments.update(
            numeric_dimension=config.numeric_dimension,
            rope_base=config.rope_base,
            history_attention=config.history_attention,
            schema=feature_schema(),
        )
        backend = getattr(config, "attention_backend", "sdpa")
        # Preserve existing SDPA exports and exact-resume recipe dictionaries.
        if backend != "sdpa":
            arguments["attention_backend"] = backend
    else:
        arguments.update(
            position_encoder=config.position_encoder,
            activation_function=config.activation_function,
            dropout=config.dropout,
            layer_normalization=config.layer_normalization,
            architecture=config.architecture,
        )
    return arguments


def make_backbone(config, *, checkpointing, dtype):
    if config.architecture == "structured_dual_v2":
        implementation = StructuredDualDecoder
    elif config.architecture == "dense_decoder":
        implementation = DenseDecisionDecoder
    else:
        implementation = Encoder
    return implementation(**backbone_kwargs(
        config, checkpointing=checkpointing, dtype=dtype
    ))


def candidate_head_kwargs(
    config, *, input_dimension, architecture, dtype, head_type: str | None = None
):
    if architecture == "structured_dual_v2":
        if (
            config.num_layers != 2 or config.activation_function != "silu"
            or config.dropout != 0.0 or config.layer_normalization
        ):
            raise ValueError(
                "Structured V2 decoder needs decoder=global_actions "
                "(two layers, SiLU, zero dropout and no head LayerNorm)."
            )
        mode = getattr(config, "candidate_features", "none")
        if mode not in ("none", "structured_v2"):
            raise ValueError(
                f"Structured V2 decoder cannot use legacy candidate_features={mode!r}."
            )
        return {
            "input_dimension": input_dimension,
            "hidden_dimension": config.dimension or input_dimension,
            "head_type": head_type or "factorized",
            "device": torch.device("cpu"),
            "dtype": dtype,
        }
    if architecture == "dense_decoder":
        if (
            config.num_layers != 2 or config.activation_function != "silu"
            or config.dropout != 0.0 or config.layer_normalization
        ):
            raise ValueError(
                "Dense decision decoder needs decoder=global_actions "
                "(two layers, SiLU, zero dropout and no head LayerNorm)."
            )
        arguments = {
            "input_dimension": input_dimension,
            "hidden_dimension": config.dimension or input_dimension,
            "device": torch.device("cpu"), "dtype": dtype,
        }
        mode = getattr(config, "candidate_features", "none")
        if mode != "none":
            from kanachan.candidate_features import candidate_feature_schema
            arguments.update(candidate_features=mode, candidate_schema=candidate_feature_schema())
        return arguments
    if getattr(config, "candidate_features", "none") != "none":
        raise ValueError("Candidate afterstates require the dense decision decoder.")
    return {
        "input_dimension": input_dimension,
        "dimension": config.dimension,
        "activation_function": config.activation_function,
        "dropout": config.dropout,
        "layer_normalization": config.layer_normalization,
        "num_layers": config.num_layers,
        "output_mode": "candidates",
        "noise_init_std": None,
        "device": torch.device("cpu"), "dtype": dtype,
    }


def make_candidate_head(
    config, *, input_dimension, architecture, dtype, head_type: str | None = None
):
    if architecture == "structured_dual_v2":
        return StructuredActionHead(**candidate_head_kwargs(
            config, input_dimension=input_dimension, architecture=architecture,
            dtype=dtype, head_type=head_type,
        ))
    implementation = GlobalActionHead if architecture == "dense_decoder" else Decoder
    return implementation(**candidate_head_kwargs(
        config, input_dimension=input_dimension, architecture=architecture, dtype=dtype
    ))


def candidate_head_keys(architecture, config=None, *, precomputed=False):
    if architecture == "structured_dual_v2":
        return ["candidates", "encode"]
    keys = ["candidates", "encode"] if architecture == "dense_decoder" else ["encode"]
    if getattr(config, "candidate_features", "none") != "none":
        keys += ["sparse", "progression"]
        if precomputed:
            keys.append("candidate_afterstate")
    return keys


def load_candidate_head_state(head, state):
    """Allow an old global BC policy to initialize a zero-correction branch.

    Full resumes still load strictly and bind the exact feature schema.
    This migration is only for explicit BC/AWR decoder.load_from weights.
    """
    global_keys = {f"layers.{layer}.{name}" for layer in (0, 2) for name in ("weight", "bias")}
    if type(head) is GlobalActionHead and head.candidate_features == "discard_v1" and set(state) == global_keys:
        result = head.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not key.startswith("candidate_") for key in result.missing_keys):
            raise ValueError("Unexpected legacy action-head state.")
        with torch.no_grad():
            head.candidate_layers[-1].weight.zero_()
            head.candidate_layers[-1].bias.zero_()
    else:
        head.load_state_dict(state, strict=True)


def state_head_kwargs(
    config, *, input_dimension, architecture, dtype, final_rank=False,
):
    if architecture in ("dense_decoder", "structured_dual_v2"):
        arguments = candidate_head_kwargs(
            config, input_dimension=input_dimension,
            architecture=architecture, dtype=dtype,
        )
        arguments.pop("candidate_features", None)
        arguments.pop("candidate_schema", None)
        arguments.pop("head_type", None)
        return dict(arguments, final_rank=final_rank)
    if final_rank:
        raise ValueError(
            "The final-rank auxiliary head requires a dense decision decoder."
        )
    arguments = candidate_head_kwargs(
        config, input_dimension=input_dimension,
        architecture=architecture, dtype=dtype,
    )
    return dict(arguments, output_mode="state")


def make_state_head(
    config, *, input_dimension, architecture, dtype, final_rank=False,
):
    if architecture == "structured_dual_v2":
        implementation = StructuredStateHead
    elif architecture == "dense_decoder":
        implementation = DecisionStateHead
    else:
        implementation = Decoder
    return implementation(**state_head_kwargs(
        config, input_dimension=input_dimension, architecture=architecture,
        dtype=dtype, final_rank=final_rank,
    ))


def network_description(network, config, *, dtype, state=False, final_rank=False):
    """Describe the shared two-module feature/readout topology for export."""
    from kanachan.model_loader import dump_model, dump_object

    encoder_tdm, head_tdm = list(network.module)
    if state:
        head_args = state_head_kwargs(
            config.decoder, input_dimension=config.encoder.dimension,
            architecture=config.encoder.architecture, dtype=dtype, final_rank=final_rank,
        )
    else:
        extra = (
            {"head_type": getattr(config.encoder, "head_type", "factorized")}
            if config.encoder.architecture == "structured_dual_v2"
            else {}
        )
        head_args = candidate_head_kwargs(
            config.decoder, input_dimension=config.encoder.dimension,
            architecture=config.encoder.architecture, dtype=dtype, **extra,
        )
    enc_in_keys = (
        list(OBSERVATION_KEYS)
        if config.encoder.architecture == "structured_dual_v2"
        else list(encoder_tdm.in_keys)
    )
    modules = [
        (encoder_tdm, backbone_kwargs(
            config.encoder, checkpointing=config.checkpointing, dtype=dtype
        ), enc_in_keys),
        (head_tdm, head_args, list(head_tdm.in_keys)),
    ]
    descriptions = [
        dump_object(tdm, [dump_model(tdm.module, [], arguments)], {
            "in_keys": in_keys, "out_keys": list(tdm.out_keys),
        })
        for tdm, arguments, in_keys in modules
    ]
    return dump_object(network, descriptions, {})
