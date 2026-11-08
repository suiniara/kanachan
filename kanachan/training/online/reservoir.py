"""Exact Algorithm R over transitions, with batch writes and private RNGs."""

import random

import torch

from .replay import ReplayBuffer


class TransitionReservoir:
    def __init__(self, capacity, selection_seed, replay_seed):
        self.replay = ReplayBuffer(capacity, replay_seed)
        self.selection = random.Random(selection_seed)
        self.seen = 0
        self.indices = torch.full((capacity,), -1, dtype=torch.int64)

    def consume(self, data):
        if data.ndim != 1 or not len(data):
            raise ValueError("Reservoir needs a nonempty transition batch.")
        capacity = self.replay.capacity
        fill = min(len(data), capacity - len(self.replay))
        if fill:
            self.replay.append(data[:fill])
            self.indices[self.seen:self.seen + fill] = torch.arange(
                self.seen, self.seen + fill
            )
        # Only the last replacement of a slot in this batch survives. Sampling
        # remains sequential Algorithm R; expensive tensor copies are batched.
        replacements = {}
        for row in range(fill, len(data)):
            slot = self.selection.randrange(self.seen + row + 1)
            if slot < capacity:
                replacements[slot] = row
        if replacements:
            slots = torch.tensor(list(replacements), dtype=torch.int64)
            rows = torch.tensor(list(replacements.values()), dtype=torch.int64)
            flat = data[rows].detach().cpu().flatten_keys(".")
            if set(flat.keys()) != set(self.replay.columns):
                raise ValueError("Reservoir transition schema changed.")
            for key, value in flat.items():
                column = self.replay.columns[key]
                if column.dtype != value.dtype or column.shape[1:] != value.shape[1:]:
                    raise ValueError("Reservoir column shape or dtype changed.")
                column[slots] = value
            self.indices[slots] = self.seen + rows
        self.seen += len(data)

    def state_dict(self):
        return {
            "seen": self.seen,
            "indices": self.indices.clone(),
            "selection_rng": self.selection.getstate(),
            "replay": self.replay.state_dict(),
        }

    def load_state_dict(self, state):
        self.replay.load_state_dict(state["replay"])
        self.seen = state["seen"]
        self.indices = state["indices"].cpu().clone()
        size = len(self.replay)
        if (
            type(self.seen) is not int or self.seen < size
            or size != min(self.seen, self.replay.capacity)
            or self.indices.shape != (self.replay.capacity,)
            or self.indices.dtype != torch.int64
            or not ((self.indices[:size] >= 0) & (self.indices[:size] < self.seen)).all()
            or len(self.indices[:size].unique()) != size
            or not (self.indices[size:] == -1).all()
        ):
            raise ValueError("Invalid reservoir membership or source cursor.")
        self.selection.setstate(state["selection_rng"])
