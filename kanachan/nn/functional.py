import math

import torch
from torch import Tensor


def expectile_loss(target: Tensor, value: Tensor, expectile: float) -> Tensor:
    """Regress V to a fixed data-action Q target using asymmetric squares."""
    if not math.isfinite(expectile) or not 0.0 < expectile < 1.0:
        raise ValueError("expectile must be finite and within (0, 1).")
    dtype = (
        torch.float64
        if torch.float64 in (target.dtype, value.dtype) else torch.float32
    )
    residual = target.detach().to(dtype) - value.to(dtype)
    weight = torch.where(residual > 0, expectile, 1.0 - expectile)
    return (weight * residual.square()).mean()


@torch.no_grad()
def compute_td_target(
    reward: Tensor,
    next_value: Tensor,
    done: Tensor,
    discount_factor: float | Tensor,
) -> Tensor:
    """Build a fixed TD target, masking terminal values before arithmetic."""
    while reward.dim() < next_value.dim():
        reward = reward.unsqueeze(-1)
    while done.dim() < next_value.dim():
        done = done.unsqueeze(-1)
    discount = torch.as_tensor(
        discount_factor, device=next_value.device, dtype=next_value.dtype
    )
    while discount.dim() < next_value.dim():
        discount = discount.unsqueeze(-1)
    bootstrap = next_value.masked_fill(done | (discount == 0.0), 0.0)
    return reward + discount * bootstrap


@torch.no_grad()
def advantage_weights(
    action_value: Tensor,
    state_value: Tensor,
    beta: float,
    max_weight: float,
) -> Tensor:
    """Compute detached AWR weights with clipping before exponentiation."""
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive.")
    if not math.isfinite(max_weight) or max_weight <= 0.0:
        raise ValueError("max_weight must be finite and positive.")
    dtype = (
        torch.float64
        if torch.float64 in (action_value.dtype, state_value.dtype)
        else torch.float32
    )
    advantage = action_value.to(dtype) - state_value.to(dtype)
    log_weight = (advantage / beta).clamp(max=math.log(max_weight))
    return log_weight.exp()
