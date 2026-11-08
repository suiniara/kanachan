"""Batch the legacy piecewise-linear numeric features on the model device."""

import torch
from torch import Tensor

from kanachan.constants import NUM_NUMERIC_FEATURES


@torch.no_grad()
def encode_numeric_features(
    numeric: Tensor,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Encode honba, deposits and four scores with the existing bin semantics.

    Use float64 for the intermediate bin arithmetic, as the scalar reference
    does, and cast only the final features. No scalar extraction or host copy
    is needed when the input already resides on the model device.
    """
    if numeric.dim() != 2 or numeric.size(1) != NUM_NUMERIC_FEATURES:
        raise ValueError("Expected honba, deposits and four seat scores.")
    if dimension <= 0:
        raise ValueError(dimension)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(dtype)

    values = numeric.detach().to(device=device, dtype=torch.float64)
    if numeric.is_floating_point():
        values = values.trunc()  # Preserve the previous int(value.item()).
    widths = torch.tensor(
        [1.0, 1.0] + [100000.0 / dimension] * 4,
        device=device,
        dtype=torch.float64,
    )
    boundaries = widths[:, None] * torch.arange(
        1, dimension + 1, device=device, dtype=torch.float64
    )
    result = (values[..., None] >= boundaries).to(torch.float64)
    maximum = torch.tensor(
        [float(dimension)] * 2 + [100000.0] * 4,
        device=device,
        dtype=torch.float64,
    )
    clipped = torch.minimum(values.clamp_min(0.0), maximum)
    position = clipped / widths
    integral = position.to(torch.int64)
    fractional = position - integral
    index = integral.clamp(max=dimension - 1).unsqueeze(-1)
    previous = result.gather(-1, index)
    partial = torch.where(
        fractional.unsqueeze(-1) > 0.0,
        fractional.unsqueeze(-1),
        previous,
    )
    return result.scatter(-1, index, partial).to(dtype=dtype)
