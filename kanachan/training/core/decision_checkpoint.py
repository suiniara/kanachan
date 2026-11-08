"""Bind new decoder resumes to their actual schema, recipe and saved bytes."""

import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from kanachan.nn.decision_decoder import ARCHITECTURE_ID, feature_schema
from kanachan.nn.decision_factory import backbone_kwargs, candidate_head_kwargs
from kanachan.training.online.state import (
    atomic_save, capture_rng, restore_rng, file_digest,
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def recipe(config, algorithm, *, dtype, amp_dtype):
    if config.encoder.architecture != "dense_decoder":
        return None
    data = Path(config.training_data)
    optimizer = OmegaConf.to_container(config.optimizer, resolve=True)
    optimizer.pop("initialize", None)
    keys = [
        "batch_size", "gradient_accumulation_steps", "max_gradient_norm",
        "rewrite_rooms", "rewrite_grades", "num_workers",
    ]
    if algorithm == "cql":
        keys += [
            "contiguous_training_data", "replay_buffer_size", "discount_factor",
            "double_q_learning", "target_kind", "td_loss", "alpha", "cql_temperature",
            "target_update_interval", "target_update_rate", "export_network",
        ]
    elif algorithm == "iql":
        keys.remove("max_gradient_norm")
        keys += [
            "contiguous_training_data", "replay_buffer_size", "discount_factor",
            "double_q_learning", "expectile", "q_max_gradient_norm", "v_max_gradient_norm",
            "target_update_interval", "target_update_rate", "final_rank_weight",
        ]
    elif algorithm == "awr":
        keys += ["beta", "weight_clipping"]
    value = {
        "format_version": 1, "architecture": ARCHITECTURE_ID,
        "schema": feature_schema(getattr(config.decoder, "candidate_features", "none")), "algorithm": algorithm,
        "backbone": backbone_kwargs(
            config.encoder, checkpointing=config.checkpointing, dtype=dtype
        ),
        "head": candidate_head_kwargs(
            config.decoder, input_dimension=config.encoder.dimension,
            architecture=config.encoder.architecture, dtype=dtype,
        ),
        "optimizer": optimizer,
        "training": {key: config[key] for key in keys},
        "amp_dtype": str(amp_dtype),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "data_sha256": file_digest(data),
    }
    if algorithm in ("cql", "iql"):
        if algorithm == "cql" and config.get("candidate_feature_workers", 0):
            value["prepared_mc"] = {
                "version": "prepared_mc_afterstate_v1",
                "seed": config.prepared_mc_seed,
                "epochs": config.prepared_mc_epochs,
            }
        if config.get("bc_initial_model") is not None:
            value["bc_feature_transfer"] = {
                "sha256": file_digest(config.bc_initial_model),
                "policy": (
                    "backbone_and_hidden_head_independent_fresh_q_zero_v_v1"
                    if algorithm == "iql" else "backbone_and_hidden_head_fresh_zero_scalar_output_v1"
                ),
            }
        if algorithm == "cql" and config.get("reward_scheme", "placement") != "placement":
            value["training"].update(
                reward_scheme=config.reward_scheme,
                round_score_unit=config.round_score_unit,
                round_reward_weight=config.round_reward_weight,
            )
        value["reward"] = {
            key: file_digest(config[key]) if config.get(key) is not None else None
            for key in ("reward_plugin", "reward_model")
        }
    if algorithm == "iql":
        value["state_head"] = {
            "final_rank": config.final_rank_weight > 0,
            "supervision": "terminal_final_scores_starting_seat_tiebreak_v1",
        }
        value["update_schedule"] = "accumulated_joint_q_v_pre_step_bootstrap_v1"
    if algorithm == "awr":
        value["teachers"] = {
            key: file_digest(config[key]) if config[key] is not None else None
            for key in ("q_model", "value_model")
        }
    # Feature transfer is initial provenance, not a repeated operation on resume.
    if algorithm in ("cql", "iql") and config.get("bc_initial_model") is None and config.initial_model_prefix is not None:
        infix = f".{config.initial_model_index}" if config.initial_model_index else ""
        path = Path(config.initial_model_prefix) / f"decision-checkpoint{infix}.json"
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))["recipe"]
            if saved.get("algorithm") == algorithm and "bc_feature_transfer" in saved:
                value["bc_feature_transfer"] = saved["bc_feature_transfer"]
    return _plain(value)


def validate_resume(config, expected):
    if expected is None or config.initial_model_prefix is None:
        return None
    if config.optimizer.initialize:
        return None
    infix = f".{config.initial_model_index}" if config.initial_model_index else ""
    prefix = Path(config.initial_model_prefix)
    path = prefix / f"decision-checkpoint{infix}.json"
    if not path.is_file():
        raise ValueError("Decoder checkpoint has no bound recipe and byte manifest.")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["recipe"] != expected:
        raise ValueError("Decoder resume schema, recipe or training data differs.")
    for name, digest in saved["files"].items():
        if Path(name).name != name or file_digest(prefix / name) != digest:
            raise ValueError(f"Decoder checkpoint bytes differ: {name}")
    if config.start_new_epoch:
        return None
    return torch.load(
        prefix / f"decision-rng{infix}.pth", weights_only=True, map_location="cpu"
    )


def save(prefix, infix, expected, *, algorithm):
    if expected is None:
        return
    prefix = Path(prefix)
    atomic_save(prefix / f"decision-rng{infix}.pth", capture_rng())
    if algorithm == "iql":
        names = [f"{network}-{part}" for network in (
            "value", "q1-source", "q2-source", "q1-target", "q2-target"
        ) for part in ("encoder", "decoder")]
        names += ["value-optimizer", "q1-optimizer", "q2-optimizer"]
    else:
        names = (
            ["encoder", "decoder"] if algorithm in ("bc", "awr") or not expected["training"].get("double_q_learning", True)
            else ["source-encoder", "source-decoder", "target-encoder", "target-decoder"]
        )
        names.append("optimizer")
    names = [f"{name}{infix}.pth" for name in names]
    names += [
        f"model{infix}.kanachan",
        f"training-state{infix}.json", f"decision-rng{infix}.pth",
    ]
    if algorithm == "iql":
        names.append(f"value-model{infix}.kanachan")
    for optional in ("scheduler", "lr-scheduler", "replay-state", "value-lr-scheduler", "q1-lr-scheduler", "q2-lr-scheduler"):
        name = f"{optional}{infix}.pth"
        if (prefix / name).is_file():
            names.append(name)
    contents = {
        "recipe": expected, "files": {name: file_digest(prefix / name) for name in names},
    }
    path = prefix / f"decision-checkpoint{infix}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(contents, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def restore(state):
    if state is not None:
        restore_rng(state)
