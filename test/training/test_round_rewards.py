import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

from kanachan.training.core.rl.rewards import RoundAndPlacementReward, configured_reward


def fixture(seat=0):
    size = 6
    sparse = torch.zeros(size, 29, dtype=torch.int32)
    sparse[:, 6] = 71 + seat
    results = torch.zeros(size, 14, dtype=torch.int32)
    # First round: seat 0's riichi cost is already included in the net +7000.
    results[1, :4] = torch.tensor([7000, -4000, -2000, -2000])
    results[1, 6:10] = torch.tensor([32000, 21000, 23000, 23000])
    # Draw / later settlement. The +1000 deposit remains in the total table.
    results[3, :4] = torch.tensor([-3000, 1000, 1000, 1000])
    results[3, 6:10] = torch.tensor([29000, 22000, 24000, 24000])
    results[5, :4] = torch.tensor([-5000, 5000, 0, 0])
    results[5, 6:10] = torch.tensor([24000, 27000, 24000, 24000])
    # Actual final scores include a round in which this observer never acted.
    results[5, 10:14] = torch.tensor([20000, 30000, 24000, 26000])
    return TensorDict({
        "sparse": sparse,
        "next": {
            "results": results,
            "end_of_round": torch.tensor([False, True, False, True, False, True]),
            "end_of_game": torch.tensor([False, False, False, False, False, True]),
        },
    }, [size])


class RoundRewardTests(unittest.TestCase):
    def test_observed_round_rewards_do_not_load_a_predictor(self):
        with patch("kanachan.training.core.rl.rewards.load_model", side_effect=AssertionError("Unexpected reward predictor")):
            reward = configured_reward("round_and_placement")
            data = fixture()
            reward(data, True)
        self.assertIsNone(reward.placement.model)
        self.assertTrue(torch.isfinite(data["next", "reward"]).all())

    def test_round_payments_and_terminal_addition(self):
        data = fixture()
        RoundAndPlacementReward(round_weight=1.)(data, True)
        # Net points +7000, -3000, -9000 (including missed final round), then 4th.
        torch.testing.assert_close(data["next", "reward"],
                                   torch.tensor([0., .7, 0., -.3, 0., -2.4]))
        self.assertEqual(data["next", "done"].tolist(), [False]*5 + [True])
        self.assertAlmostEqual(float(data["next", "reward"].sum()), -2., places=6)

    def test_all_seats_and_no_cross_game_leakage(self):
        pieces = [fixture(seat) for seat in range(4)]
        together = torch.cat(pieces)
        RoundAndPlacementReward(round_weight=1.)(together, True)
        for seat, data in enumerate(pieces):
            RoundAndPlacementReward(round_weight=1.)(data, True)
            torch.testing.assert_close(together["next", "reward"][seat*6:seat*6+6],
                                       data["next", "reward"])
            final = data["next", "results"][-1, 10+seat]
            rank = [3, 0, 2, 1][seat]
            expected = (float(final) - 25000)/10000 + 1.5-rank
            self.assertAlmostEqual(float(data["next", "reward"].sum()), expected, places=6)

    def test_missing_intermediate_round_is_not_lost(self):
        data = fixture()
        # The next observed round starts 2000 lower than the previous settlement.
        # Endpoint differences include that gap, rather than trusting only delta.
        data["next", "results"][3, 6] -= 2000
        RoundAndPlacementReward(round_weight=1.)(data, True)
        self.assertAlmostEqual(float(data["next", "reward"][3]), -.5, places=6)
        self.assertAlmostEqual(float(data["next", "reward"].sum()), -2., places=6)

    def test_zero_point_draw_ties_and_config_validation(self):
        data = fixture()
        data["next", "results"][:, :4] = 0
        data["next", "results"][:, 6:14] = 25000
        RoundAndPlacementReward()(data, True)
        torch.testing.assert_close(data["next", "reward"],
                                   torch.tensor([0., 0., 0., 0., 0., 1.5]))
        for kwargs in ({"score_unit": 0}, {"score_unit": float("nan")}, {"round_weight": -1}):
            with self.assertRaises(ValueError):
                RoundAndPlacementReward(**kwargs)
        with self.assertRaisesRegex(ValueError, "complete"):
            RoundAndPlacementReward()(fixture(), False)
        with self.assertRaisesRegex(ValueError, "separate recipes"):
            configured_reward("round_and_placement", shaping_model=torch.nn.Identity())

    def test_mixed_observer_rejection(self):
        data = fixture()
        data["sparse"][2:, 6] = 72
        with self.assertRaisesRegex(ValueError, "seat changed"):
            RoundAndPlacementReward()(data, True)

    def test_default_round_weight_preserves_terminal_scale(self):
        data = fixture()
        RoundAndPlacementReward()(data, True)
        torch.testing.assert_close(data["next", "reward"],
                                   torch.tensor([0., .14, 0., -.06, 0., -1.68]))
        self.assertAlmostEqual(float(data["next", "reward"].sum()), -1.6, places=6)


if __name__ == "__main__":
    unittest.main()
