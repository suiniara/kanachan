"""Scalar Double DQN targets and optional conservative Q regularization."""

from kanachan.training.core.rl.discrete_cql import (
    action_values, evaluating, scalar_cql_loss,
)


def q_loss(
    source, target, data, alpha, *, td_loss="mse", temperature=1.0,
    tensor_metrics=False,
):
    return scalar_cql_loss(
        source, target, data, alpha, td_loss=td_loss, temperature=temperature,
        tensor_metrics=tensor_metrics,
    )
