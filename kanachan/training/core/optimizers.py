import math

import torch
from torch import Tensor
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim import Adam, AdamW, Optimizer, RAdam


ADAMW_POLICY = "adamw_matrix_decay_v1"
LEGACY_ADAMW_POLICY = "adamw_all_parameters_v1"


def adamw_parameter_groups(module, weight_decay):
    """Decay matrices, excluding embeddings; retain deterministic group order."""
    no_decay = {
        id(parameter)
        for layer in module.modules() if isinstance(layer, torch.nn.Embedding)
        for parameter in layer.parameters(recurse=False)
    }
    decayed, unchanged = [], []
    for parameter in module.parameters():
        if parameter.requires_grad:
            (decayed if parameter.ndim >= 2 and id(parameter) not in no_decay
             else unchanged).append(parameter)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": unchanged, "weight_decay": 0.0},
    ]


def online_adamw(module, *, lr, weight_decay, device_type, resume_state=None):
    """New runs share offline AdamW; old flat-group runs keep their recipe.

    Regrouping an old optimizer silently changes weight decay and parameter
    IDs. Restore its original layout and implementation instead of migrating
    moments by positional coincidence.
    """
    policy = ADAMW_POLICY
    if resume_state is not None:
        policy = resume_state.get("optimizer_policy", LEGACY_ADAMW_POLICY)
        expected = 1 if policy == LEGACY_ADAMW_POLICY else 2
        if policy not in (ADAMW_POLICY, LEGACY_ADAMW_POLICY) or len(
            resume_state["optimizer"]["param_groups"]
        ) != expected:
            raise ValueError("Unsupported online AdamW policy or parameter groups.")
    if policy == LEGACY_ADAMW_POLICY:
        optimizer = AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = AdamW(
            adamw_parameter_groups(module, weight_decay), lr=lr,
            fused=True if device_type == "cuda" else None,
        )
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        move_optimizer_state(optimizer)
    return optimizer, policy


def _prepare_adam_state(state_dict, current_groups):
    states = {
        key: dict(value) if value is not None else None
        for key, value in state_dict["state"].items()
    }
    groups = []
    for group in state_dict["param_groups"]:
        if not group.get("bias_correction", True):
            raise ValueError(
                "Native Adam requires bias-corrected Apex checkpoints."
            )
        for parameter in group["params"]:
            state = states.get(parameter)
            if state and "step" not in state:
                if "step" not in group:
                    raise ValueError(
                        "Adam checkpoint has moments but no step counter."
                    )
                state["step"] = torch.tensor(float(group["step"]))
    for current, saved in zip(
        current_groups, state_dict["param_groups"], strict=True
    ):
        groups.append({**current, **saved})
    return {**state_dict, "state": states, "param_groups": groups}


class CompatibleAdam(Adam):
    """Adam that also accepts Apex checkpoints with a group-level step."""

    def load_state_dict(self, state_dict):
        super().load_state_dict(
            _prepare_adam_state(state_dict, self.param_groups)
        )


class CompatibleZeroRedundancyOptimizer(ZeroRedundancyOptimizer):
    """Adapt legacy Adam state before ZeRO distributes it to local optimizers."""

    def load_state_dict(self, state_dict):
        if isinstance(self.optim, CompatibleAdam):
            state_dict = _prepare_adam_state(
                state_dict, self.optim.param_groups
            )
        super().load_state_dict(state_dict)


def move_optimizer_state(optimizer: Optimizer) -> None:
    """Place restored moments beside parameters after moving their model."""
    local_optimizer = getattr(optimizer, "optim", optimizer)
    for group in local_optimizer.param_groups:
        for parameter in group["params"]:
            for key, value in local_optimizer.state.get(parameter, {}).items():
                if not isinstance(value, Tensor):
                    continue
                device = parameter.device
                if (
                    key == "step"
                    and isinstance(local_optimizer, (Adam, AdamW, RAdam))
                    and not group.get("capturable", False)
                    and not group.get("fused", False)
                ):
                    device = torch.device("cpu")
                local_optimizer.state[parameter][key] = value.to(device=device)


def consolidate_optimizer_state(*optimizers: Optimizer) -> None:
    """Collect ZeRO state on global rank zero before writing a checkpoint."""
    from torch.distributed.optim import ZeroRedundancyOptimizer

    for optimizer in optimizers:
        if isinstance(optimizer, ZeroRedundancyOptimizer):
            optimizer.consolidate_state_dict(to=0)


class Lamb(Optimizer):
    """Portable LAMB with Apex-compatible group counters and defaults.

    Keeps weight decay, bias correction, gradient averaging and global
    clipping explicit so an Apex LAMB state can be resumed without resetting
    its moments. Half precision parameters use float32 moment calculations.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        *,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
        bias_correction=True,
        grad_averaging=True,
        max_grad_norm=1.0,
        use_nvlamb=False,
    ):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError(lr)
        if any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in betas):
            raise ValueError(betas)
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(eps)
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError(weight_decay)
        if max_grad_norm <= 0 or math.isnan(max_grad_norm):
            raise ValueError(max_grad_norm)
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "bias_correction": bias_correction,
                "grad_averaging": grad_averaging,
                "max_grad_norm": max_grad_norm,
                "use_nvlamb": use_nvlamb,
            },
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        gradients = [
            p.grad
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not gradients:
            return loss
        if any(gradient.is_sparse for gradient in gradients):
            raise RuntimeError("LAMB does not support sparse gradients.")
        norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(
                        gradient.to(
                            torch.float64
                            if gradient.dtype == torch.float64
                            else torch.float32
                        )
                    )
                    for gradient in gradients
                ]
            )
        )
        for group in self.param_groups:
            group["step"] = int(group.get("step", 0)) + 1
            step = group["step"]
            beta1, beta2 = group["betas"]
            clip = (norm / group["max_grad_norm"]).clamp_min(1.0)
            correction1 = 1 - beta1**step if group["bias_correction"] else 1
            correction2 = 1 - beta2**step if group["bias_correction"] else 1
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                dtype = (
                    torch.float64
                    if parameter.dtype == torch.float64
                    else torch.float32
                )
                gradient = parameter.grad.to(dtype) / clip
                state = self.state[parameter]
                for key in ("exp_avg", "exp_avg_sq"):
                    if key not in state:
                        state[key] = torch.zeros_like(parameter, dtype=dtype)
                    else:
                        state[key] = state[key].to(
                            device=parameter.device, dtype=dtype
                        )
                momentum, variance = state["exp_avg"], state["exp_avg_sq"]
                momentum.mul_(beta1).add_(
                    gradient, alpha=1 - beta1 if group["grad_averaging"] else 1
                )
                variance.mul_(beta2).addcmul_(
                    gradient, gradient, value=1 - beta2
                )
                update = (momentum / correction1) / (
                    (variance / correction2).sqrt() + group["eps"]
                )
                values = parameter.to(dtype)
                if group["weight_decay"]:
                    update = update.add(values, alpha=group["weight_decay"])
                if group.get("use_nvlamb", False) or group["weight_decay"]:
                    parameter_norm = torch.linalg.vector_norm(values)
                    update_norm = torch.linalg.vector_norm(update)
                    ratio = torch.where(
                        (parameter_norm > 0) & (update_norm > 0),
                        parameter_norm
                        / update_norm.clamp_min(torch.finfo(dtype).tiny),
                        torch.ones_like(update_norm),
                    )
                    update = update * ratio
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
        return loss
