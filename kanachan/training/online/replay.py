"""Bounded CPU replay with replacement sampling and complete persistence."""

import torch
from tensordict import TensorDict


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.columns: dict[str, torch.Tensor] = {}
        self.metadata: dict = {}
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.size

    def append(self, transitions: TensorDict):
        if transitions.ndim != 1 or not len(transitions):
            raise ValueError("Replay needs a nonempty, flat transition batch.")
        flat = transitions.detach().cpu().flatten_keys(".")
        if self.columns and set(flat.keys()) != set(self.columns):
            raise ValueError("Replay schema changed during training.")
        if len(flat) > self.capacity:
            flat = flat[-self.capacity :]
        for key, value in flat.items():
            if key not in self.columns:
                self.columns[key] = torch.empty(
                    (self.capacity, *value.shape[1:]), dtype=value.dtype
                )
            if (
                self.columns[key].dtype != value.dtype
                or self.columns[key].shape[1:] != value.shape[1:]
            ):
                raise ValueError(
                    f"Replay column {key} changed shape or dtype."
                )
            first = min(len(flat), self.capacity - self.position)
            self.columns[key][self.position : self.position + first].copy_(
                value[:first]
            )
            if first < len(flat):
                self.columns[key][: len(flat) - first].copy_(value[first:])
        self.position = (self.position + len(flat)) % self.capacity
        self.size = min(self.capacity, self.size + len(flat))

    def sample(self, batch_size: int):
        if not self.size or batch_size < 1:
            raise ValueError(
                "Cannot sample empty replay or a nonpositive batch."
            )
        indices = torch.randint(
            self.size, (batch_size,), generator=self.generator
        )
        return TensorDict(
            {
                tuple(key.split(".")): value[indices]
                for key, value in self.columns.items()
            },
            [batch_size],
        )

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "rng": self.generator.get_state(),
            "metadata": self.metadata,
            "columns": {
                key: value[: self.size].clone()
                for key, value in self.columns.items()
            },
        }

    def load_state_dict(self, state):
        if (
            state["capacity"] != self.capacity
            or not 0 <= state["size"] <= self.capacity
            or not 0 <= state["position"] < self.capacity
        ):
            raise ValueError(
                "Invalid replay checkpoint or different capacity."
            )
        if (
            state["size"] < self.capacity
            and state["position"] != state["size"]
        ):
            raise ValueError("Invalid replay cursor.")
        self.size, self.position = state["size"], state["position"]
        self.metadata = state.get("metadata", {})
        self.columns = {}
        for key, value in state["columns"].items():
            if len(value) != self.size:
                raise ValueError("Incomplete replay checkpoint.")
            self.columns[key] = torch.empty(
                (self.capacity, *value.shape[1:]), dtype=value.dtype
            )
            self.columns[key][: self.size].copy_(value)
        self.generator.set_state(state["rng"].cpu())
