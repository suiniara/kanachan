from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
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
    num_workers: int | None = None
    rewrite_rooms: str | int | None = None
    rewrite_grades: str | int | None = None
    initial_model_prefix: Path | None = None
    initial_model_index: int | None = None
    start_new_epoch: bool = False
    checkpointing: bool = False
    batch_size: int = MISSING
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    snapshot_interval: int = 0
    candidate_feature_workers: int = 0
    candidate_prefetch_batches: int = 8
    # Structured V2 trajectory training options (used when training_mode="v2_trajectory")
    training_mode: str = "single_step"
    use_next_event_head: bool = False
    next_event_weight: float = 0.20
    use_final_rank_head: bool = False
    final_rank_weight: float = 0.0
    label_smoothing: float = 0.0
    suit_augmentation: bool = False
    aug_seed: int = 1337
    sampler_seed: int = 42
    spool_cache_dir: Path | None = None
    spool_num_buckets: int = 64
    spool_max_bucket_bytes: int = 67108864
    max_steps: int | None = None
    max_rounds: int | None = None
    allow_unlabeled: bool = False
    allow_incomplete: bool = False
    prefetch_batches: int = 2
    drop_last: bool = False
    resume_from: Path | None = None
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
class DecisionConfig(Config):
    defaults: list[Any] = field(default_factory=lambda: [
        {"device": "cuda"}, {"encoder": "decision_tiny"},
        {"decoder": "global_actions"}, {"optimizer": "adamw"}, "_self_",
    ])


config_store.store(name="decision", node=DecisionConfig)


@dataclass
class TrajectoryV2Config(Config):
    training_mode: str = "v2_trajectory"
    batch_size: int = 16
    defaults: list[Any] = field(default_factory=lambda: [
        {"device": "cuda"},
        {"encoder": "v2_base"},
        {"decoder": "global_actions"},
        {"optimizer": "adamw"},
        "_self_",
    ])


config_store.store(name="v2_trajectory", node=TrajectoryV2Config)
