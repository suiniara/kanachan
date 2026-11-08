from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from kanachan.training.core.config.device import DeviceConfig
from kanachan.training.core.config.encoder import EncoderConfig
from kanachan.training.core.config.decoder import DecoderConfig
from kanachan.training.core.config.optimizer import OptimizerConfig


@dataclass
class Config:
    device: DeviceConfig
    encoder: EncoderConfig
    decoder: DecoderConfig
    optimizer: OptimizerConfig
    training_data: Path = MISSING
    contiguous_training_data: bool = False
    rewrite_rooms: str | int | None = None
    rewrite_grades: str | int | None = None
    num_workers: int | None = None
    replay_buffer_size: int = 0
    num_qr_intervals: int | None = 200
    dueling_network: bool = True
    initial_model_prefix: Path | None = None
    initial_model_index: int | None = None
    start_new_epoch: bool = False
    reward_plugin: Path | None = None
    reward_model: Path | None = None
    bc_initial_model: Path | None = None
    reward_scheme: str = "placement"
    round_score_unit: float = 10000.0
    round_reward_weight: float = 0.2
    double_q_learning: bool = True
    discount_factor: float = 0.99
    kappa: float = 1.0
    alpha: float = 1.0
    checkpointing: bool = False
    enable_compile: bool = True
    batch_size: int = MISSING
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    target_update_interval: int = 1
    target_update_rate: float = 0.005
    cql_temperature: float = 1.0
    td_loss: str = "mse"
    target_kind: str = "td"
    q_monitor_interval: int = 100
    data_cache: Path | None = None
    data_cache_bytes: int = 256 * 1024 * 1024
    export_network: str = "target"
    snapshot_interval: int = 0
    candidate_feature_workers: int = 0
    prepared_mc_bytes: int = 3 * 1024**3
    prepared_mc_seed: int = 20261109
    prepared_mc_epochs: int = 1
    defaults: list[Any] = field(default_factory=lambda: [
        {"device": "cuda"},
        {"encoder": "bert_base"},
        {"decoder": "single"},
        {"optimizer": "adam"},
        "_self_",
    ])


config_store = ConfigStore.instance()
config_store.store(name="config", node=Config)


@dataclass
class PlacementConfig(Config):
    """Scalar Double Q, complete half-games and final-placement utility."""

    num_qr_intervals: int | None = None
    discount_factor: float = 1.0
    contiguous_training_data: bool = True
    replay_buffer_size: int = 65536
    num_workers: int | None = 0
    enable_compile: bool = False


config_store.store(name="placement", node=PlacementConfig)


@dataclass
class DecisionConfig(PlacementConfig):
    target_kind: str = "mc"
    double_q_learning: bool = False
    alpha: float = 0.5
    td_loss: str = "mse"
    reward_scheme: str = "round_and_placement"
    export_network: str = "online"
    defaults: list[Any] = field(default_factory=lambda: [
        {"device": "cuda"}, {"encoder": "decision_tiny"},
        {"decoder": "global_actions"}, {"optimizer": "adamw"}, "_self_",
    ])


config_store.store(name="decision", node=DecisionConfig)
