import logging
import os
from dataclasses import dataclass
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore
import torch
from torch import backends


@dataclass
class DeviceConfig:
    type: str
    dtype: str
    amp_dtype: str | None


@dataclass
class CpuConfig(DeviceConfig):
    type: str = "cpu"
    dtype: str = "float32"
    amp_dtype: str | None = None


@dataclass
class CudaConfig(DeviceConfig):
    type: str = "cuda"
    dtype: str = "float32"
    amp_dtype: str = "float16"


config_store = ConfigStore.instance()
config_store.store(name="cpu", node=CpuConfig, group="device")
config_store.store(name="cuda", node=CudaConfig, group="device")


def validate(
    config: DictConfig,
) -> tuple[int, int, int, torch.device, torch.dtype, torch.dtype]:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as error:
        raise RuntimeError("Invalid distributed process environment.") from error
    if world_size <= 0 or not 0 <= rank < world_size or local_rank < 0:
        raise RuntimeError(f"Invalid process ranks: world_size={world_size}, rank={rank}, local_rank={local_rank}.")
    if world_size > 1 and ("RANK" not in os.environ or "LOCAL_RANK" not in os.environ):
        raise RuntimeError("Distributed training requires RANK and LOCAL_RANK from a launcher.")

    if config.device.type not in ("cpu", "cuda"):
        errmsg = f"{config.device.type}: An invalid value for `device.type`."
        raise RuntimeError(errmsg)
    if config.device.type == "cuda":
        if local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            raise RuntimeError(
                f"CUDA device {local_rank} was requested but only {torch.cuda.device_count()} devices are available."
            )
    elif config.device.type == "cpu":
        device = torch.device("cpu")
    else:
        errmsg = f"{config.device.type}: An invalid value for `device.type`."
        raise RuntimeError(errmsg)

    parameter_dtypes = {
        "float64": torch.float64, "double": torch.float64,
        "float32": torch.float32, "float": torch.float32,
        "float16": torch.float16, "half": torch.float16,
    }
    if config.device.dtype not in parameter_dtypes:
        errmsg = (
            f"{config.device.dtype}: An invalid value for" " `device.dtype`."
        )
        raise RuntimeError(errmsg)
    dtype = parameter_dtypes[config.device.dtype]
    encoder = getattr(config, "encoder", None)
    if getattr(encoder, "architecture", None) == "dense_decoder" and dtype != torch.float32:
        raise RuntimeError(
            "Dense decoder training requires FP32 master parameters "
            "(device.dtype=float32). Select device.amp_dtype=float16 or "
            "bfloat16 for mixed precision."
        )

    if config.device.amp_dtype is None:
        amp_dtype = dtype
    else:
        amp_dtypes = {
            **parameter_dtypes, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        }
        if config.device.amp_dtype not in amp_dtypes:
            errmsg = (
                f"{config.device.amp_dtype}: An invalid value for"
                " `device.amp_dtype`."
            )
            raise RuntimeError(errmsg)
        amp_dtype = amp_dtypes[config.device.amp_dtype]
    if device.type == "cpu":
        amp_dtype = dtype
    if amp_dtype == torch.float64 and dtype in (torch.float32, torch.float16):
        errmsg = (
            "An invalid combination of `device.dtype`"
            f" (`{config.device.dtype}`) and `device.amp_dtype`"
            f" (`{config.device.amp_dtype}`)."
        )
        raise RuntimeError(errmsg)
    if amp_dtype == torch.float32 and dtype == torch.float16:
        errmsg = (
            "An invalid combination of `device.dtype`"
            f" (`{config.device.dtype}`) and `device.amp_dtype`"
            f" (`{config.device.amp_dtype}`)."
        )
        raise RuntimeError(errmsg)
    if amp_dtype == torch.bfloat16:
        if dtype != torch.float32:
            raise RuntimeError("BF16 AMP requires FP32 master parameters.")
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support BF16 AMP.")

    if backends.cudnn.is_available():
        backends.cudnn.benchmark = True

    return world_size, rank, local_rank, device, dtype, amp_dtype


def dump(
    *,
    world_size: int,
    rank: int,
    local_rank: int,
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype,
) -> None:
    logging.info("World size: %d", world_size)
    logging.info("Process rank: %d", rank)
    logging.info("Local process rank: %d", local_rank)

    logging.info("Device: %s", device)
    logging.info("dtype: %s", dtype)
    if dtype == amp_dtype:
        logging.info("AMP dtype: (AMP is disabled)")
    else:
        logging.info("AMP dtype: %s", amp_dtype)

    if backends.cudnn.is_available():
        logging.info("cuDNN: available")
    else:
        logging.info("cuDNN: N/A")
