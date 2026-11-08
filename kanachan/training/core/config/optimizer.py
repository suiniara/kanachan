import logging
import math
from typing import Type, Any
from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from torch.optim.radam import RAdam
from torch.optim import AdamW
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed import is_initialized, get_world_size
import torch.optim.lr_scheduler as lr_scheduler
from kanachan.training.core.optimizers import (
    CompatibleAdam, CompatibleZeroRedundancyOptimizer, Lamb,
    adamw_parameter_groups,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    type: str
    momentum: float | None
    epsilon: float | None
    learning_rate: float
    warmup_start_lr: float
    warmup_steps: int
    annealing_steps: int
    annealing_steps_factor: int
    use_zero: bool
    initialize: bool


@dataclass
class SgdOptimizerConfig(OptimizerConfig):
    type: str = "sgd"
    momentum: float = 0.0
    epsilon: float | None = None
    learning_rate: float = MISSING
    warmup_start_lr: float = 1.0e-8
    warmup_steps: int = 0
    annealing_steps: int = 0
    annealing_steps_factor: int = 1
    use_zero: bool = False
    initialize: bool = False


@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    type: str = "adam"
    momentum: float | None = None
    epsilon: float = 1.0e-8
    learning_rate: float = 1.0e-4
    warmup_start_lr: float = 1.0e-8
    warmup_steps: int = 0
    annealing_steps: int = 0
    annealing_steps_factor: int = 1
    use_zero: bool = False
    initialize: bool = False


@dataclass
class RAdamOptimizerConfig(OptimizerConfig):
    type: str = "radam"
    momentum: float | None = None
    epsilon: float = 1.0e-8
    learning_rate: float = 1.0e-4
    warmup_start_lr: float = 1.0e-8
    warmup_steps: int = 0
    annealing_steps: int = 0
    annealing_steps_factor: int = 1
    use_zero: bool = False
    initialize: bool = False


@dataclass
class AdamWOptimizerConfig(AdamOptimizerConfig):
    type: str = "adamw"
    weight_decay: float = 0.01


@dataclass
class LambOptimizerConfig(OptimizerConfig):
    type: str = "lamb"
    momentum: float | None = None
    epsilon: float = 1.0e-6
    learning_rate: float = 1.0e-4
    warmup_start_lr: float = 1.0e-8
    warmup_steps: int = 0
    annealing_steps: int = 0
    annealing_steps_factor: int = 1
    use_zero: bool = False
    initialize: bool = False


config_store = ConfigStore.instance()
config_store.store(name="sgd", node=SgdOptimizerConfig, group="optimizer")
config_store.store(name="adam", node=AdamOptimizerConfig, group="optimizer")
config_store.store(name="adamw", node=AdamWOptimizerConfig, group="optimizer")
config_store.store(name="radam", node=RAdamOptimizerConfig, group="optimizer")
config_store.store(name="lamb", node=LambOptimizerConfig, group="optimizer")


def validate(optimizer_config: Any):
    if optimizer_config.type == "adamw" and (
        not math.isfinite(optimizer_config.weight_decay)
        or optimizer_config.weight_decay < 0
    ):
        raise ValueError("AdamW weight_decay must be finite and nonnegative.")
    if optimizer_config.type in ("sgd",):
        if optimizer_config.momentum < 0.0 or 1.0 <= optimizer_config.momentum:
            errmsg = (
                f"{optimizer_config.momentum}: `optimizer.momentum`"
                " must be a real value within the range [0.0, 1.0)."
            )
            raise RuntimeError(errmsg)
    else:
        if optimizer_config.momentum is not None:
            errmsg = (
                "`optimizer.momentum` is useless for"
                f" `{optimizer_config.type}`."
            )
            raise RuntimeError(errmsg)

    if optimizer_config.type != "sgd" and (
        optimizer_config.epsilon is None or optimizer_config.epsilon <= 0.0
    ):
        errmsg = (
            f"{optimizer_config.epsilon}: `optimizer.epsilon` must be a"
            " non-negative real value."
        )
        raise RuntimeError(errmsg)

    if optimizer_config.learning_rate <= 0.0:
        errmsg = (
            f"{optimizer_config.learning_rate}: "
            "`optimizer.learning_rate` must be a positive real value."
        )
        raise RuntimeError(errmsg)

    if optimizer_config.warmup_start_lr <= 0.0:
        errmsg = (
            f"{optimizer_config.warmup_start_lr}: "
            "`optimizer.warmup_start_lr` must be a positive real value."
        )
        raise RuntimeError(errmsg)
    if optimizer_config.warmup_start_lr > optimizer_config.learning_rate:
        errmsg = (
            f"{optimizer_config.warmup_start_lr}:"
            " `optimizer_config.warmup_start_lr` must be less than"
            " `optimizer_config.learning_rate`."
        )
        raise RuntimeError(errmsg)

    if optimizer_config.warmup_steps < 0:
        errmsg = (
            f"{optimizer_config.warmup_steps}: `optimizer.warmup_steps`"
            " must be a non-negative integer."
        )
        raise RuntimeError(errmsg)

    if optimizer_config.annealing_steps < 0:
        errmsg = (
            f"{optimizer_config.annealing_steps}:"
            " `optimizer.annealing_steps` must be a non-negative"
            " integer."
        )
        raise RuntimeError(errmsg)

    if optimizer_config.annealing_steps_factor <= 0:
        errmsg = (
            f"{optimizer_config.annealing_steps_factor}: "
            "`optimizer.annealing_steps_factor` must be a positive"
            " integer."
        )
        raise RuntimeError(errmsg)


def dump(optimizer_config: Any, prefix: str = "") -> None:
    logging.info("%sOptimizer: %s", prefix, optimizer_config.type)
    if optimizer_config.type in ("sgd",):
        logging.info(
            "%sMomentum factor: %f", prefix, optimizer_config.momentum
        )
    if optimizer_config.type in ("adam", "radam", "mtadam", "lamb"):
        logging.info(
            "%sEpsilon parameter: %E", prefix, optimizer_config.epsilon
        )
    logging.info("%sLearning rate: %E", prefix, optimizer_config.learning_rate)
    if optimizer_config.warmup_steps == 0:
        logging.info("%sLR warm-up: (disabled)", prefix)
    else:
        logging.info(
            "%sLR warm-up start LR: %E",
            prefix,
            optimizer_config.warmup_start_lr,
        )
        logging.info(
            "%sLR warm-up steps: %d", prefix, optimizer_config.warmup_steps
        )
    if optimizer_config.annealing_steps == 0:
        logging.info("%sLR annealing: (disabled)", prefix)
    else:
        logging.info(
            "%sLR annealing steps: %d",
            prefix,
            optimizer_config.annealing_steps,
        )
        logging.info(
            "%sLR annealing steps factor: %d",
            prefix,
            optimizer_config.annealing_steps_factor,
        )
    logging.info("%sUse ZeRO: %s", prefix, optimizer_config.use_zero)


def create(
    device_type: str, optimizer_config: Any, module: nn.Module
) -> tuple[Optimizer, lr_scheduler.LRScheduler | None]:
    classes = {
        "sgd": SGD, "adam": CompatibleAdam, "adamw": AdamW,
        "radam": RAdam, "lamb": Lamb,
    }
    if optimizer_config.type not in classes:
        raise NotImplementedError(optimizer_config.type)
    optimizer_class: Type[Optimizer] = classes[optimizer_config.type]
    if device_type == "cuda" and optimizer_config.type in ("sgd", "adam", "lamb"):
        try:
            from apex.multi_tensor_apply import multi_tensor_applier
            from apex.optimizers import FusedAdam, FusedLAMB, FusedSGD
            if multi_tensor_applier.available:
                optimizer_class = {"sgd": FusedSGD, "adam": FusedAdam, "lamb": FusedLAMB}[optimizer_config.type]
        except ImportError:
            pass
    logger.info("Using optimizer: %s.%s", optimizer_class.__module__, optimizer_class.__name__)
    optimizer_kwargs = {"lr": optimizer_config.learning_rate}
    if optimizer_config.type == "sgd":
        optimizer_kwargs["momentum"] = optimizer_config.momentum
    else:
        optimizer_kwargs["eps"] = optimizer_config.epsilon
    if optimizer_class is CompatibleAdam and device_type == "cuda":
        optimizer_kwargs["fused"] = True
    parameters = module.parameters()
    if optimizer_config.type == "adamw":
        parameters = adamw_parameter_groups(module, optimizer_config.weight_decay)
        if device_type == "cuda":
            optimizer_kwargs["fused"] = True

    optimizer: Optimizer
    if optimizer_config.use_zero and is_initialized() and get_world_size() > 1:
        optimizer = CompatibleZeroRedundancyOptimizer(
            parameters, optimizer_class, **optimizer_kwargs
        )
    else:
        optimizer = optimizer_class(parameters, **optimizer_kwargs)

    if optimizer_config.warmup_steps == 0:
        warmup_scheduler = None
    else:
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=optimizer_config.warmup_start_lr
            / optimizer_config.learning_rate,
            total_iters=optimizer_config.warmup_steps,
        )
    if optimizer_config.annealing_steps == 0:
        annealing_scheduler = None
    else:
        annealing_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            optimizer_config.annealing_steps,
            optimizer_config.annealing_steps_factor,
        )
    scheduler: lr_scheduler.LRScheduler | None
    if warmup_scheduler is None and annealing_scheduler is None:
        scheduler = None
    elif warmup_scheduler is not None and annealing_scheduler is not None:
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            [warmup_scheduler, annealing_scheduler],
            [warmup_scheduler.total_iters],
        )
    elif warmup_scheduler is not None:
        assert annealing_scheduler is None
        scheduler = warmup_scheduler
    else:
        assert warmup_scheduler is None
        assert annealing_scheduler is not None
        scheduler = annealing_scheduler

    return optimizer, scheduler
