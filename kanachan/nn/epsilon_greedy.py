import math

import torch
from torch import Tensor
from tensordict import TensorDict  # type: ignore
from tensordict.nn import TensorDictModuleBase  # type: ignore
from kanachan.constants import NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES


class EpsilonGreedy(TensorDictModuleBase):
    def __init__(
        self,
        *,
        epsilon_first: float,
        epsilon_last: float,
        annealing_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not math.isfinite(epsilon_first) or not 0.0 <= epsilon_first <= 1.0:
            msg = (
                "`epsilon_first` must be a real number within the range"
                " [0.0, 1.0]."
            )
            raise ValueError(msg)
        if not math.isfinite(epsilon_last) or epsilon_last > epsilon_first:
            msg = (
                "`epsilon_last` must be less than or equal to"
                " `epsilon_first`."
            )
            raise ValueError(msg)
        if epsilon_last < 0.0:
            msg = "`epsilon_last` must be a non-negative real number."
            raise ValueError(msg)
        if annealing_steps < 0:
            msg = "`annealing_steps` must be a non-negative integer."
            raise ValueError(msg)
        if annealing_steps == 0 and epsilon_first != epsilon_last:
            msg = (
                "`annealing_steps == 0` is possible only if"
                " `epsilon_first == epsilon_last`."
            )
            raise ValueError(msg)

        super().__init__()

        self.register_buffer(
            "epsilon_first",
            torch.tensor(epsilon_first, device=device, dtype=dtype),
        )
        self.register_buffer(
            "epsilon_last",
            torch.tensor(epsilon_last, device=device, dtype=dtype),
        )
        self.register_buffer(
            "epsilon", torch.tensor(epsilon_first, device=device, dtype=dtype)
        )
        self.annealing_steps = annealing_steps
        self.register_buffer("_step", torch.tensor(0, device=device, dtype=torch.int64))

        self.in_keys = ["candidates", "action"]
        self.out_keys = ["action"]

    def forward(self, td: TensorDict) -> TensorDict:
        batch_size = td.size(0)
        assert isinstance(batch_size, int)
        candidates: Tensor = td["candidates"]
        assert candidates.dim() == 2
        assert candidates.size(0) == batch_size
        assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES
        action: Tensor = td["action"]
        assert action.dim() == 1
        assert action.size(0) == batch_size

        if not self.training:
            return td

        stable_ids = getattr(self, "stable_action_ids", False)
        if stable_ids:
            sorted_candidates, original_indices = candidates.sort(-1)
            action_mask = sorted_candidates < NUM_TYPES_OF_ACTIONS
        else:
            action_mask = candidates < NUM_TYPES_OF_ACTIONS
        random_action = (
            action_mask.float().multinomial(1).squeeze(1).to(dtype=torch.int32)
        )
        if stable_ids:
            random_action = original_indices.gather(
                1, random_action.long()[:, None]
            ).squeeze(-1).to(torch.int32)
        assert random_action.dim() == 1
        assert random_action.size(0) == batch_size
        assert (
            (
                candidates[torch.arange(batch_size), random_action]
                < NUM_TYPES_OF_ACTIONS
            )
            .all()
            .item()
        )

        random = torch.rand_like(action, dtype=torch.float32)
        epsilon_mask = random < self.epsilon

        td["action"] = torch.where(epsilon_mask, random_action, action)

        return td

    def step(self, steps: int = 1) -> None:
        if steps < 0:
            msg = "`steps` must be a non-negative integer."
            raise ValueError(msg)

        self._step.fill_(min(int(self._step.item()) + steps, self.annealing_steps))

        epsilon_first: Tensor = self.epsilon_first
        epsilon_last: Tensor = self.epsilon_last
        if self.annealing_steps == 0:
            assert epsilon_first.item() == epsilon_last.item()
            return

        self.epsilon.copy_(epsilon_first * (
            (self.annealing_steps - self._step) / self.annealing_steps
        ) + epsilon_last * (self._step / self.annealing_steps))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        if prefix + "_step" not in state_dict and all(
            prefix + key in state_dict for key in ("epsilon_first", "epsilon_last", "epsilon")
        ):
            # Legacy snapshots saved epsilon but omitted the schedule position.
            first = float(state_dict[prefix + "epsilon_first"].item())
            last = float(state_dict[prefix + "epsilon_last"].item())
            current = float(state_dict[prefix + "epsilon"].item())
            step = 0 if first == last else round(
                self.annealing_steps * (first - current) / (first - last)
            )
            state_dict[prefix + "_step"] = torch.tensor(
                max(0, min(step, self.annealing_steps)), dtype=torch.int64,
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )
