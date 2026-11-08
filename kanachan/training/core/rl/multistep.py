import math

import torch
from torch import Tensor
from tensordict import TensorDict  # type: ignore
from kanachan.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ACTIONS,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ROUND_SUMMARY,
    NUM_RESULTS,
)


@torch.no_grad()
def make_multistep(
    episodes: TensorDict, discount_factor: float, td_steps: int
) -> None:
    """Collapse contiguous one-step transitions without crossing a terminal.

    Only bootstrap observations advance. Round summaries and result labels
    remain attached to the original transition. ``next.discount`` contains
    gamma raised to the actual horizon, including a shortened final window.
    """
    length = int(episodes.size(0))  # type: ignore
    sparse: Tensor = episodes["sparse"]
    assert sparse.dim() == 2
    assert sparse.size(0) == length
    assert sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES
    assert sparse.dtype == torch.int32
    numeric: Tensor = episodes["numeric"]
    assert numeric.dim() == 2
    assert numeric.size(0) == length
    assert numeric.size(1) == NUM_NUMERIC_FEATURES
    assert numeric.dtype == torch.int32
    progression: Tensor = episodes["progression"]
    assert progression.dim() == 2
    assert progression.size(0) == length
    assert progression.size(1) == MAX_LENGTH_OF_PROGRESSION_FEATURES
    assert progression.dtype == torch.int32
    candidates: Tensor = episodes["candidates"]
    assert candidates.dim() == 2
    assert candidates.size(0) == length
    assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES
    assert candidates.dtype == torch.int32
    round_summary: Tensor = episodes["next", "round_summary"]
    assert round_summary.dim() == 2
    assert round_summary.size(0) == length
    assert round_summary.size(1) == MAX_NUM_ROUND_SUMMARY
    assert round_summary.dtype == torch.int32
    results: Tensor = episodes["next", "results"]
    assert results.dim() == 2
    assert results.size(0) == length
    assert results.size(1) == NUM_RESULTS
    assert results.dtype == torch.int32
    end_of_round: Tensor = episodes["next", "end_of_round"]
    assert end_of_round.dim() == 1
    assert end_of_round.size(0) == length
    assert end_of_round.dtype == torch.bool
    end_of_game: Tensor = episodes["next", "end_of_game"]
    assert end_of_game.dim() == 1
    assert end_of_game.size(0) == length
    assert end_of_game.dtype == torch.bool
    reward: Tensor = episodes["next", "reward"]
    assert reward.dim() == 1
    assert reward.size(0) == length
    assert reward.dtype in (torch.float64, torch.float32, torch.float16)
    done: Tensor = episodes["next", "done"]
    assert done.dim() == 1
    assert done.size(0) == length
    assert done.dtype == torch.bool

    if not math.isfinite(discount_factor) or not 0.0 <= discount_factor <= 1.0:
        errmsg = (
            f"{discount_factor}: `discount_factor` must be a real"
            " number within the range [0.0, 1.0]."
        )
        raise ValueError(errmsg)
    if td_steps <= 0:
        errmsg = f"{td_steps}: `td_steps` must be a positive integer."
        raise ValueError(errmsg)

    source = episodes["next"].detach().clone()
    reward = source["reward"]
    done = source["done"]
    positions = torch.arange(length, device=reward.device)
    endpoints = positions.clone()
    active = torch.ones_like(done)
    accumulated = torch.zeros_like(reward)
    steps = torch.zeros_like(positions)
    coefficient = 1.0
    for offset in range(min(td_steps, length)):
        indices = positions + offset
        valid = active & (indices < length)
        indices = indices.clamp(max=length - 1)
        if coefficient != 0.0:
            accumulated += coefficient * reward[indices].masked_fill(~valid, 0.0)
        endpoints = torch.where(valid, indices, endpoints)
        steps += valid.to(steps.dtype)
        active = valid & ~done[indices]
        coefficient *= discount_factor

    terminal = done[endpoints]
    padding = {
        "sparse": NUM_TYPES_OF_SPARSE_FEATURES,
        "numeric": 0,
        "progression": NUM_TYPES_OF_PROGRESSION_FEATURES,
        "candidates": NUM_TYPES_OF_ACTIONS,
    }
    for key, pad_value in padding.items():
        observation = source[key][endpoints].clone()
        episodes["next", key] = observation.masked_fill(terminal.unsqueeze(1), pad_value)
    episodes["next", "reward"] = accumulated
    episodes["next", "done"] = terminal
    episodes["next", "steps_to_next_obs"] = steps
    episodes["next", "discount"] = torch.full_like(reward, discount_factor).pow(steps)
