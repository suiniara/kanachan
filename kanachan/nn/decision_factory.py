"""Shared training integration without changing legacy encoder constructors."""

import torch

from kanachan.nn.encoder import Encoder
from kanachan.nn.decoder import Decoder
from kanachan.nn.decision_decoder import (
    DenseDecisionDecoder, GlobalActionHead, feature_schema,
)
from kanachan.nn.decision_state import DecisionStateHead


def backbone_kwargs(config, *, checkpointing, dtype):
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
    implementation = (
        DenseDecisionDecoder if config.architecture == "dense_decoder" else Encoder
    )
    return implementation(**backbone_kwargs(
        config, checkpointing=checkpointing, dtype=dtype
    ))


def candidate_head_kwargs(config, *, input_dimension, architecture, dtype):
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


def make_candidate_head(config, *, input_dimension, architecture, dtype):
    implementation = GlobalActionHead if architecture == "dense_decoder" else Decoder
    return implementation(**candidate_head_kwargs(
        config, input_dimension=input_dimension, architecture=architecture, dtype=dtype
    ))


def candidate_head_keys(architecture, config=None, *, precomputed=False):
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
    if architecture == "dense_decoder":
        arguments = candidate_head_kwargs(
            config, input_dimension=input_dimension,
            architecture=architecture, dtype=dtype,
        )
        arguments.pop("candidate_features", None)
        arguments.pop("candidate_schema", None)
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
    implementation = (
        DecisionStateHead if architecture == "dense_decoder" else Decoder
    )
    return implementation(**state_head_kwargs(
        config, input_dimension=input_dimension, architecture=architecture,
        dtype=dtype, final_rank=final_rank,
    ))


def network_description(network, config, *, dtype, state=False, final_rank=False):
    """Describe the shared two-module feature/readout topology for export."""
    from kanachan.model_loader import dump_model, dump_object

    encoder_tdm, head_tdm = list(network.module)
    head_kwargs = state_head_kwargs if state else candidate_head_kwargs
    options = {"final_rank": final_rank} if state else {}
    modules = [
        (encoder_tdm, backbone_kwargs(
                config.encoder, checkpointing=config.checkpointing, dtype=dtype
        )),
        (head_tdm, head_kwargs(
            config.decoder, input_dimension=config.encoder.dimension,
            architecture=config.encoder.architecture, dtype=dtype, **options,
        )),
    ]
    descriptions = [
        dump_object(tdm, [dump_model(tdm.module, [], arguments)], {
            "in_keys": list(tdm.in_keys), "out_keys": list(tdm.out_keys),
        })
        for tdm, arguments in modules
    ]
    return dump_object(network, descriptions, {})
