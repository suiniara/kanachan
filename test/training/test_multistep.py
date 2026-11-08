import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import unittest

import torch
from tensordict import TensorDict

from kanachan.constants import (
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    MAX_NUM_ROUND_SUMMARY,
    NUM_NUMERIC_FEATURES,
    NUM_RESULTS,
    NUM_TYPES_OF_ACTIONS,
)
from kanachan.nn.functional import compute_td_target
from kanachan.training.core.rl.multistep import make_multistep


def episode_batch(rewards, dones):
    length = len(rewards)
    observations = {
        "sparse": torch.zeros(
            length, MAX_NUM_ACTIVE_SPARSE_FEATURES, dtype=torch.int32
        ),
        "numeric": torch.zeros(
            length, NUM_NUMERIC_FEATURES, dtype=torch.int32
        ),
        "progression": torch.zeros(
            length, MAX_LENGTH_OF_PROGRESSION_FEATURES, dtype=torch.int32
        ),
        "candidates": torch.zeros(
            length, MAX_NUM_ACTION_CANDIDATES, dtype=torch.int32
        ),
    }
    next_observations = {
        key: value.clone() for key, value in observations.items()
    }
    next_observations["numeric"][:, 0] = torch.arange(1, length + 1)
    next_observations.update(
        {
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "done": torch.tensor(dones, dtype=torch.bool),
            "end_of_game": torch.tensor(dones, dtype=torch.bool),
            "end_of_round": torch.tensor(dones, dtype=torch.bool),
            "round_summary": torch.zeros(
                length, MAX_NUM_ROUND_SUMMARY, dtype=torch.int32
            ),
            "results": torch.zeros(length, NUM_RESULTS, dtype=torch.int32),
        }
    )
    return TensorDict({**observations, "next": next_observations}, [length])


class MultistepTests(unittest.TestCase):
    def test_nonterminal_returns_use_shortened_horizon_at_file_end(self):
        data = episode_batch([1, 2, 3, 4], [False] * 4)
        make_multistep(data, 0.5, 2)
        torch.testing.assert_close(
            data["next", "reward"], torch.tensor([2.0, 3.5, 5.0, 4.0])
        )
        torch.testing.assert_close(
            data["next", "numeric"][:, 0],
            torch.tensor([2, 3, 4, 4], dtype=torch.int32),
        )
        torch.testing.assert_close(
            data["next", "discount"], torch.tensor([0.25, 0.25, 0.25, 0.5])
        )
        self.assertEqual(
            data["next", "steps_to_next_obs"].tolist(), [2, 2, 2, 1]
        )
        self.assertFalse(data["next", "done"].any().item())

    def test_multistep_targets_do_not_cross_episode_boundaries(self):
        data = episode_batch([1, 2, 100, 200], [False, True, False, True])
        original_metadata = data["next", "end_of_game"].clone()
        make_multistep(data, 0.5, 3)
        torch.testing.assert_close(
            data["next", "reward"], torch.tensor([2.0, 2.0, 200.0, 200.0])
        )
        self.assertEqual(
            data["next", "steps_to_next_obs"].tolist(), [2, 1, 2, 1]
        )
        self.assertTrue(data["next", "done"].all().item())
        self.assertTrue(
            (data["next", "candidates"] == NUM_TYPES_OF_ACTIONS).all().item()
        )
        torch.testing.assert_close(
            data["next", "end_of_game"], original_metadata
        )

    def test_td_targets_accept_per_transition_discounts(self):
        reward = torch.tensor([1.0, 2.0], requires_grad=True)
        values = torch.tensor(
            [[4.0, 8.0], [-torch.inf, torch.inf]], requires_grad=True
        )
        discount = torch.tensor([0.25, 0.0], requires_grad=True)
        target = compute_td_target(
            reward, values, torch.tensor([False, False]), discount
        )
        torch.testing.assert_close(
            target, torch.tensor([[2.0, 3.0], [2.0, 2.0]])
        )
        self.assertFalse(target.requires_grad)

    def test_empty_and_invalid_multistep_input(self):
        data = episode_batch([], [])
        make_multistep(data, 0.5, 3)
        self.assertEqual(data["next", "discount"].numel(), 0)
        for gamma, steps in ((float("nan"), 1), (0.5, 0)):
            with self.assertRaises(ValueError):
                make_multistep(data, gamma, steps)
