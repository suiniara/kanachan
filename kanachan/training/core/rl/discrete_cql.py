"""Legal-action scalar CQL with explicit observed-return or TD targets."""

from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F

from kanachan.constants import NUM_TYPES_OF_ACTIONS
from kanachan.nn.decision_decoder import legal_argmax, _legal_argmax, require_tensor
from kanachan.nn.decision_value import DecisionQNetwork


OBSERVATIONS = ("sparse", "numeric", "progression", "candidates")


@contextmanager
def evaluating(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield model
    finally:
        for module, training in modes:
            module.training = training


def scalar_field(value, batch_size, *, name, kind):
    if value.shape == (batch_size, 1):
        value = value.squeeze(-1)
    if value.shape != (batch_size,):
        raise ValueError(f"{name} must have shape [B] or [B,1].")
    if kind == "integer":
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must already contain integer indices.")
        return value.long()
    if kind == "boolean":
        require_tensor(
            ((value == 0) | (value == 1)).all(), f"{name} must be boolean or 0/1."
        )
        return value.bool()
    value = value.float()
    require_tensor(torch.isfinite(value).all(), f"Nonfinite {name}.")
    return value


def action_values(model, data):
    keys = OBSERVATIONS
    if data.get("candidate_afterstate", None) is not None:
        keys = (*keys, "candidate_afterstate")
    observation = data.select(*keys).clone(False)
    model(observation)
    values = observation.get("action_value", None)
    if values is None or values.shape != data["candidates"].shape:
        raise ValueError("Scalar CQL needs one Q value per original candidate.")
    mask = data["candidates"] >= NUM_TYPES_OF_ACTIONS
    if not isinstance(model, DecisionQNetwork):
        require_tensor((~mask).any(-1).all(), "Current decision has no legal actions.")
    require_tensor(
        torch.isfinite(values.masked_fill(mask, 0)).all(),
        "Nonfinite Q values for legal actions.",
    )
    return values.float().masked_fill(mask, -torch.inf)


def _observed_action_values(source, data):
    batch_size = int(data.batch_size[0])
    values = action_values(source, data)
    action = scalar_field(data["action"], batch_size, name="action", kind="integer")
    require_tensor(
        ((action >= 0) & (action < values.shape[1])).all(),
        "Action index is outside the candidate list.",
    )
    chosen = values.gather(1, action[:, None]).squeeze(-1)
    require_tensor(torch.isfinite(chosen).all(), "Replay action is not legal.")
    return values, chosen


def mc_components(source, data):
    """Regress complete observed returns; never query a next-state network.

    EpisodeReplayBuffer supplies ``mc_return`` only after complete contiguous
    trajectories have been rewarded and accumulated, before replay shuffling.
    It is separate from ``action_value``, which the network may overwrite.
    These are behavior-trajectory returns, not current-policy Q labels.
    """
    observed = data.get("mc_return", None)
    if observed is None:
        raise ValueError("MC CQL requires complete-trajectory mc_return labels.")
    expected = scalar_field(
        observed, int(data.batch_size[0]), name="MC return", kind="float"
    ).detach().clone()
    values, chosen = _observed_action_values(source, data)
    return values, chosen, expected


def td_components(source, target, data, *, default_discount=1.0):
    """Detached one-step Double-DQN target, bootstrapping only nonterminals.

    Complete logged returns may be present for diagnostics. They never enter
    the target or loss: online selects a legal next action and target evaluates
    it. A whole-game terminal uses only its observed immediate reward.
    """
    batch_size = int(data.batch_size[0])
    values, chosen = _observed_action_values(source, data)
    reward = scalar_field(
        data["next", "reward"], batch_size, name="reward", kind="float"
    )
    done = scalar_field(data["next", "done"], batch_size, name="done", kind="boolean")
    discount = data.get(("next", "discount"), None)
    if discount is None:
        discount = torch.full_like(reward, default_discount)
    discount = scalar_field(discount, batch_size, name="discount", kind="float")
    require_tensor(
        ((discount >= 0) & (discount <= 1)).all(), "Invalid transition discount."
    )
    with torch.no_grad(), evaluating(source):
        expected = reward.clone()
        active = ~done & (discount != 0)
        if active.any():
            next_data = data["next"][active]
            next_online = action_values(source, next_data)
            candidates = next_data["candidates"]
            indices = (
                _legal_argmax(candidates, next_online, candidates < NUM_TYPES_OF_ACTIONS)
                if isinstance(source, DecisionQNetwork)
                else legal_argmax(candidates, next_online)
            ).long()
            next_target = (
                action_values(target, next_data) if target is not None
                else next_online
            )
            bootstrap = next_target.gather(1, indices[:, None]).squeeze(-1)
            expected[active] += discount[active] * bootstrap
    return values, chosen, expected


def td_penalty(chosen, expected, kind="mse"):
    if kind == "mse":
        return 0.5 * (chosen - expected).square()
    if kind == "huber":
        return F.huber_loss(chosen, expected, reduction="none", delta=1.0)
    raise ValueError("TD loss must be mse or huber.")


def conservative_penalty(values, chosen, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("CQL temperature must be finite and positive.")
    return temperature * torch.logsumexp(values.float() / temperature, -1) - chosen.float()


@torch.no_grad()
def q_statistics(values, chosen, candidates, *, temperature=1.0, gap=None,
                 logged_return=None, target=None, done=None, target_kind="td"):
    """Diagnostics only; logged behavior returns are not current-policy Q labels."""
    chosen = chosen.detach().float()
    values = values.detach().float().masked_fill(candidates >= NUM_TYPES_OF_ACTIONS, -torch.inf)
    count = (candidates < NUM_TYPES_OF_ACTIONS).sum(-1).float()
    percentiles = torch.quantile(
        chosen, torch.tensor([0.05, 0.5, 0.95], device=chosen.device)
    )
    if gap is None:
        gap = conservative_penalty(values, chosen, temperature)
    metrics = {
        "Q/P05": percentiles[0], "Q/P50": percentiles[1], "Q/P95": percentiles[2],
        "Q/Std": chosen.std(unbiased=False),
        "Q/Legal action count": count.mean(),
        "Q/CQL gap centered": (gap.detach().float() - temperature * count.log()).mean(),
    }
    if logged_return is not None:
        if logged_return.shape != chosen.shape:
            raise ValueError("Logged returns must have shape [B].")
        difference = chosen - logged_return.detach().float()
        metrics["Q/Logged return MAE"] = difference.abs().mean()
        metrics["Q/Logged return bias"] = difference.mean()
    if target is not None:
        metrics["Q/Target mean"] = target.detach().float().mean()
        residual = chosen - target.detach().float()
        prefix = "MC" if target_kind == "mc" else "TD"
        metrics[f"{prefix}/Residual mean"] = residual.mean()
        metrics[f"{prefix}/Absolute residual"] = residual.abs().mean()
        if done is not None:
            terminal = scalar_field(
                done, len(chosen), name="done", kind="boolean"
            )
            for label, mask in (("Terminal", terminal), ("Nonterminal", ~terminal)):
                count = mask.sum()
                metrics[f"{prefix}/{label} count"] = count.float()
                # Empty groups report zero with an explicit zero count.
                metrics[f"{prefix}/{label} residual"] = residual.masked_fill(~mask, 0).sum() / count.clamp_min(1)
    return metrics


def scalar_cql_loss(
    source, target, data, alpha, *, td_loss="mse", temperature=1.0,
    tensor_metrics=False, target_kind="td",
):
    if target_kind == "mc":
        if target is not None:
            raise ValueError("MC CQL has no target network.")
        values, chosen, expected = mc_components(source, data)
    elif target_kind == "td":
        values, chosen, expected = td_components(source, target, data)
    else:
        raise ValueError("CQL target_kind must be mc or td.")
    td = td_penalty(chosen, expected, td_loss)
    conservative = conservative_penalty(values, chosen, temperature)
    coefficient = torch.as_tensor(alpha, dtype=torch.float32, device=chosen.device)
    if coefficient.ndim not in (0, 1) or (
        coefficient.ndim == 1 and coefficient.shape != chosen.shape
    ):
        raise ValueError("CQL alpha must be scalar or [B].")
    require_tensor(
        (torch.isfinite(coefficient) & (coefficient >= 0)).all(), "Invalid CQL alpha."
    )
    metrics = {
        ("return_loss" if target_kind == "mc" else "td_loss"): td.detach().mean(),
        "cql_loss": conservative.detach().mean(),
        "mean_q": chosen.detach().mean(),
        "mean_target": expected.detach().mean(),
    }
    if not tensor_metrics:
        metrics = {key: float(value) for key, value in metrics.items()}
    return (td + coefficient * conservative).mean(), metrics
