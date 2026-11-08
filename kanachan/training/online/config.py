"""Explicit configuration for the placement-first online training recipe."""

from dataclasses import dataclass, asdict, fields
import math
from pathlib import Path

import yaml


@dataclass
class Config:
    initial_model: str
    output_dir: str
    opponents: list[str] | None = None
    reward_model: str | None = None
    device: str = "cuda:0"
    amp_dtype: str = "float16"
    torch_num_threads: int = 2
    enable_compile: bool = False
    seed: int = 20261029
    room: int = 3
    grade: int = 12
    iterations: int = 1000
    games_per_iteration: int = 16
    simulation_batch_size: int = 4
    simulation_concurrency: int = 8
    replay_capacity: int = 65536
    learning_starts: int = 1024
    batch_size: int = 128
    updates_per_iteration: int = 64
    td_steps: int = 1
    learning_rate: float = 0.0001
    td_loss: str = "mse"
    cql_temperature: float = 1.0
    offline_data: str | None = None
    offline_pool_cache: str | None = None
    offline_capacity: int = 65536
    offline_fraction: float = 0.5
    offline_cql_alpha: float = 0.1
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    target_update_interval: int = 1
    target_update_rate: float = 0.005
    epsilon_first: float = 0.1
    epsilon_last: float = 0.01
    epsilon_steps: int = 1000000
    cql_alpha_first: float = 0.1
    cql_alpha_last: float = 0.0
    cql_annealing_steps: int = 1000000
    evaluate_every: int = 10
    evaluation_sets: int = 128
    evaluation_seed: int = 1649032026
    bootstrap_samples: int = 2000
    confidence: float = 0.95
    min_evaluation_groups: int = 32
    min_rank_improvement: float = 0.0
    max_fourth_rate_increase: float = 0.0
    max_history_opponents: int = 4

    def validate(self):
        if type(self.enable_compile) is not bool:
            raise ValueError("enable_compile must be a boolean.")
        integer_fields = (
            "iterations",
            "torch_num_threads",
            "games_per_iteration",
            "simulation_batch_size",
            "simulation_concurrency",
            "replay_capacity",
            "learning_starts",
            "batch_size",
            "updates_per_iteration",
            "td_steps",
            "target_update_interval",
            "offline_capacity",
            "evaluation_sets",
            "bootstrap_samples",
            "min_evaluation_groups",
        )
        for key in integer_fields:
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer.")
        for key in (
            "epsilon_steps",
            "cql_annealing_steps",
            "evaluate_every",
            "max_history_opponents",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer.")
        for key in ("seed", "evaluation_seed"):
            if (
                type(getattr(self, key)) is not int
                or not 0 <= getattr(self, key) < 2**63
            ):
                raise ValueError(f"{key} must be an integer in [0, 2**63).")
        for key in (
            "learning_rate",
            "max_gradient_norm",
            "target_update_rate",
            "cql_temperature",
        ):
            if (
                not math.isfinite(getattr(self, key))
                or getattr(self, key) <= 0
            ):
                raise ValueError(f"{key} must be finite and positive.")
        for key in (
            "weight_decay",
            "cql_alpha_first",
            "cql_alpha_last",
            "min_rank_improvement",
            "offline_cql_alpha",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative.")
        if (
            not 0 <= self.epsilon_last <= self.epsilon_first <= 1
            or self.cql_alpha_last > self.cql_alpha_first
            or self.target_update_rate > 1
            or not 0 < self.confidence < 1
            or not 0 <= self.max_fourth_rate_increase <= 1
        ):
            raise ValueError(
                "Invalid exploration, CQL, target or acceptance bounds."
            )
        if (
            self.epsilon_steps == 0 and self.epsilon_first != self.epsilon_last
        ) or (
            self.cql_annealing_steps == 0
            and self.cql_alpha_first != self.cql_alpha_last
        ):
            raise ValueError(
                "A changing schedule requires positive annealing steps."
            )
        if self.simulation_concurrency < 2 * self.simulation_batch_size - 1:
            raise ValueError(
                "simulation_concurrency must be at least 2 * batch_size - 1."
            )
        if self.learning_starts > self.replay_capacity:
            raise ValueError("learning_starts exceeds replay capacity.")
        if self.bootstrap_samples < 100 or self.min_evaluation_groups < 2:
            raise ValueError(
                "Insufficient bootstrap samples or evaluation groups."
            )
        if (
            self.evaluate_every
            and self.evaluation_sets < self.min_evaluation_groups
        ):
            raise ValueError(
                "evaluation_sets is smaller than min_evaluation_groups."
            )
        if self.room not in (3, 4) or not 0 <= self.grade <= 15:
            raise ValueError(
                "This recipe requires Jade/Throne room and a valid grade."
            )
        if self.amp_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("amp_dtype must be float32, float16 or bfloat16.")
        if self.td_loss not in ("mse", "huber"):
            raise ValueError("td_loss must be mse or huber.")
        if not math.isfinite(self.offline_fraction) or not 0 < self.offline_fraction < 1:
            raise ValueError("offline_fraction must be strictly between zero and one.")
        if self.offline_data is not None:
            if self.batch_size < 2:
                raise ValueError("Mixed offline/online batches require batch_size >= 2.")
            if self.reward_model is not None:
                raise ValueError("The immutable offline pool uses unshaped placement rewards.")
        elif self.offline_pool_cache is not None:
            raise ValueError("offline_pool_cache requires offline_data.")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict) or set(values) - {
        field.name for field in fields(Config)
    }:
        raise ValueError(
            "Unknown online configuration fields or invalid YAML mapping."
        )
    for key in ("initial_model", "output_dir", "reward_model", "offline_data", "offline_pool_cache"):
        if values.get(key) is not None:
            values[key] = str((path.parent / values[key]).resolve())
    if values.get("opponents") is not None:
        values["opponents"] = [
            str((path.parent / value).resolve())
            for value in values["opponents"]
        ]
    return Config(**values).validate()
