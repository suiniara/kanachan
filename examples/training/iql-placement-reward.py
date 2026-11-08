"""Use the existing placement reward through the IQL reward-plugin interface."""

from kanachan.training.core.rl.rewards import PlacementReward

get_reward = PlacementReward()
