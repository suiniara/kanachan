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
    initial_model_prefix: Path | None = None
    initial_model_index: int | None = None
    start_new_epoch: bool = False
    reward_plugin: Path | None = MISSING
    bc_initial_model: Path | None = None
    double_q_learning: bool = True
    discount_factor: float = 0.99
    expectile: float = MISSING
    final_rank_weight: float = 0.0
    checkpointing: bool = False
    batch_size: int = MISSING
    gradient_accumulation_steps: int = 1
    q_max_gradient_norm: float = 1.0
    v_max_gradient_norm: float = 1.0
    target_update_interval: int = 1
    target_update_rate: float = 0.005
    snapshot_interval: int = 0
    # Structured V2 Joint IQL options (used when training_mode="v2_joint")
    training_mode: str = "legacy_five_net"
    reward_scheme: str = "placement"
    grp_checkpoint: Path | None = None
    score_unit: float = 10000.0
    round_weight: float = 0.2
    awr_inv_temp_beta: float = 3.0
    awr_max_weight: float = 20.0
    kl_coef: float = 0.0
    keep_bc_anchor: bool = False
    decoder1_lr_scale: float = 0.25
    freeze_decoder1: bool = False
    freeze_shared_tile_emb: bool | None = None
    use_cql_quantile_head: bool = False
    num_quantiles: int = 51
    qr_double_q: bool = True
    quantile_cql_alpha: float = 0.0
    cql_temperature: float = 1.0
    warmup_transitions: int | None = None
    suit_augmentation: bool = False
    aug_seed: int = 1337
    sampler_seed: int = 42
    max_steps: int | None = None
    max_games: int | None = None
    max_batches: int | None = None
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
    expectile: float = 0.7
    discount_factor: float = 1.0
    defaults: list[Any] = field(default_factory=lambda: [
        {"device": "cuda"}, {"encoder": "decision_tiny"},
        {"decoder": "global_actions"}, {"optimizer": "adamw"}, "_self_",
    ])


config_store.store(name="decision", node=DecisionConfig)


@dataclass
class JointV2Config(Config):
    training_mode: str = "v2_joint"
    expectile: float = 0.7
    discount_factor: float = 1.0
    contiguous_training_data: bool = True
    replay_buffer_size: int = 4096
    reward_plugin: Path | None = None
    batch_size: int = 32
    defaults: list[Any] = field(default_factory=lambda: [
        {"device": "cuda"},
        {"encoder": "v2_base"},
        {"decoder": "global_actions"},
        {"optimizer": "adamw"},
        "_self_",
    ])


config_store.store(name="v2_joint", node=JointV2Config)
