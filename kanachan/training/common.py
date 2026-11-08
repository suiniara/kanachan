import re
import pickle
import os
from typing import Any, TypeVar
from collections.abc import Iterable, Iterator
import torch
from torch import Tensor, nn
from torch.amp import GradScaler
from torch.distributed import (
    is_initialized,
    get_world_size,
    get_rank,
    ProcessGroup,
    send,
    recv,
    all_reduce,
)


def get_distributed_environment() -> tuple[int, int, int]:
    if not is_initialized():
        return 1, 0, 0

    if "LOCAL_RANK" not in os.environ:
        errmsg = "The `LOCAL_RANK` environment variable must be defined."
        raise RuntimeError(errmsg)

    world_size = get_world_size()
    rank = get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    return world_size, rank, local_rank


def average_tensor(tensor: Tensor) -> None:
    """Average a detached statistic on either Gloo or NCCL."""
    world_size, _, _ = get_distributed_environment()
    if world_size >= 2:
        all_reduce(tensor)
        tensor.div_(world_size)


_Batch = TypeVar("_Batch")


def with_last(batches: Iterable[_Batch]) -> Iterator[tuple[_Batch, bool]]:
    iterator = iter(batches)
    try:
        current = next(iterator)
    except StopIteration:
        return
    for following in iterator:
        yield current, False
        current = following
    yield current, True


@torch.no_grad()
def rescale_gradients(model: nn.Module, factor: float) -> None:
    if factor != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)


def make_noisy(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert decoder linear means while retaining normalization and buffers."""
    result = {}
    for key, value in state_dict.items():
        if re.search(r"(?:^|\.)layers\.linear\d*\.(weight|bias)$", key):
            key += "_mu"
        result[key] = value
    return result


def get_gradient(model: nn.Module) -> Tensor:
    gradient = [
        param.grad.view(-1)
        for param in model.parameters()
        if param.grad is not None
    ]
    return torch.cat(gradient)


def is_gradient_nan(model: nn.Module) -> Tensor:
    gradient = get_gradient(model)
    return gradient.isnan().any()


def is_gradient_nonfinite(model: nn.Module) -> Tensor:
    return ~get_gradient(model).isfinite().all()


@torch.no_grad()
def checked_clip_gradients(model: nn.Module, max_norm: float) -> Tensor | None:
    """Compute the clipping norm once; skip nonfinite gradients on all ranks.

    The uncommon failure path distinguishes invalid gradient values from
    overflow of an otherwise finite norm, preserving the existing behavior.
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        raise RuntimeError("No parameter gradients to clip.")
    norm = nn.utils.get_total_norm(
        [p.grad for p in parameters], error_if_nonfinite=False
    )
    invalid = (~torch.isfinite(norm)).to(torch.int32)
    world_size, _, _ = get_distributed_environment()
    if world_size >= 2:
        all_reduce(invalid)
    if invalid.item():
        nonfinite = is_gradient_nonfinite(model).to(torch.int32)
        if world_size >= 2:
            all_reduce(nonfinite)
        if nonfinite.item():
            return None
        raise RuntimeError("The total gradient norm is nonfinite despite finite gradients.")
    nn.utils.clip_grads_with_norm_(parameters, max_norm, norm)
    return norm


def backoff_gradient_scale(grad_scaler: GradScaler) -> None:
    """Reduce the AMP scale when skipping a non-finite optimization step."""
    if grad_scaler.is_enabled():
        grad_scaler.update(
            new_scale=(
                grad_scaler.get_scale() * grad_scaler.get_backoff_factor()
            )
        )


def send_object(
    obj, *, device: torch.device, dst: int, group: ProcessGroup | None = None
) -> None:
    data = pickle.dumps(obj)
    buf = torch.frombuffer(bytearray(data), dtype=torch.int8).to(device=device)
    length = torch.tensor(buf.size(0), device=device, dtype=torch.int64)

    send(length, dst, group=group)
    send(buf, dst, group=group)


def recv_object(
    *,
    device: torch.device,
    src: int | None = None,
    group: ProcessGroup | None = None,
) -> tuple[Any, int]:
    length = torch.tensor(-1, device=device, dtype=torch.int64)
    src = recv(length, src, group=group)

    buf = torch.empty((int(length.item()),), device=device, dtype=torch.int8)
    recv(buf, src, group=group)
    obj = pickle.loads(buf.cpu().numpy().tobytes())

    return obj, src
