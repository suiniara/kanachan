"""State readouts for the decision representation, independent of action order."""

import torch
from torch import nn
from torch.nn import functional as F


class DecisionStateHead(nn.Module):
    """An FP32 V readout with an optional final-placement auxiliary head."""

    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
        device: torch.device,
        dtype: torch.dtype,
        final_rank: bool = False,
    ):
        super().__init__()
        if input_dimension <= 0 or hidden_dimension <= 0:
            raise ValueError("Invalid state-head dimensions.")
        self.layers = nn.Sequential(
            nn.Linear(
                input_dimension, hidden_dimension, device=device, dtype=dtype
            ),
            nn.SiLU(),
            nn.Linear(hidden_dimension, 1, device=device, dtype=dtype),
        )
        self.final_rank = (
            nn.Linear(hidden_dimension, 4, device=device, dtype=dtype)
            if final_rank
            else None
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    @torch.no_grad()
    def zero_output(self):
        self.layers[-1].weight.zero_()
        self.layers[-1].bias.zero_()

    def forward(self, representation):
        if representation.ndim != 2:
            raise ValueError(
                "Expected one decision representation per observation."
            )
        dtype = (
            torch.float64
            if representation.dtype == torch.float64
            else torch.float32
        )
        with torch.autocast(
            device_type=representation.device.type, enabled=False
        ):
            hidden = F.silu(
                F.linear(
                    representation.to(dtype),
                    self.layers[0].weight.to(dtype),
                    self.layers[0].bias.to(dtype),
                )
            )
            value = F.linear(
                hidden,
                self.layers[-1].weight.to(dtype),
                self.layers[-1].bias.to(dtype),
            ).squeeze(-1)
            if self.final_rank is None:
                return value
            logits = F.linear(
                hidden,
                self.final_rank.weight.to(dtype),
                self.final_rank.bias.to(dtype),
            )
            return value, logits
