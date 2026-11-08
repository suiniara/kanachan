import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import unittest

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from kanachan.constants import ENCODER_WIDTH, NUM_TYPES_OF_ACTIONS
from kanachan.nn import Decoder, QRDecoder
from kanachan.nn.epsilon_greedy import EpsilonGreedy
from kanachan.nn.noisy_linear import NoisyLinear
from kanachan.nn.qr_decoder import _get_a_star, compute_td_error
from kanachan.nn.twin_q_actor import TwinQActor


class ConstantQ(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values))

    def forward(self, data):
        data["action_value"] = self.values.unsqueeze(0).expand(
            data.batch_size[0], -1
        )
        return data


class PolicyUtilityTests(unittest.TestCase):
    def test_zero_kappa_quantile_regression_keeps_td_gradients(self):
        class Quantiles(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.values = torch.nn.Parameter(torch.zeros(2))

            def forward(self, data):
                data["qr_action_value"] = self.values.expand(1, 32, 2)
                return data

        model = Quantiles()
        fields = {
            key: torch.zeros(1, 32, dtype=torch.int32)
            for key in ("sparse", "numeric", "progression", "candidates")
        }
        data = TensorDict(
            {
                **fields,
                "action": torch.zeros(1, dtype=torch.int32),
                "next": {
                    **fields,
                    "reward": torch.ones(1),
                    "done": torch.ones(1, dtype=torch.bool),
                },
            },
            [1],
        )
        compute_td_error(
            source_network=model,
            target_network=None,
            data=data,
            discount_factor=0.99,
            kappa=0,
        )
        torch.testing.assert_close(data["td_error"], torch.ones(1))
        data["td_error"].sum().backward()
        torch.testing.assert_close(
            model.values.grad, torch.tensor([-0.25, -0.75])
        )
        for invalid in (-1.0, float("nan"), float("inf")):
            with self.subTest(kappa=invalid), self.assertRaises(ValueError):
                compute_td_error(
                    source_network=model,
                    target_network=None,
                    data=data,
                    discount_factor=0.99,
                    kappa=invalid,
                )

    def test_noisy_conversion_keeps_normalization_and_unrelated_state(self):
        from kanachan.training.common import make_noisy

        source = {
            name: torch.tensor(float(index))
            for index, name in enumerate(
                [
                    "layers.linear0.weight",
                    "layers.linear0.bias",
                    "layers.layer_normalization0.weight",
                    "encoder.layers.0.linear1.weight",
                    "buffer",
                ]
            )
        }
        converted = make_noisy(source)
        self.assertEqual(len(converted), len(source))
        for name, value in source.items():
            expected = (
                name + "_mu" if name.startswith("layers.linear0.") else name
            )
            self.assertIs(converted[expected], value)

    def test_decoder_normalization_honors_parameter_dtype(self):
        decoder = Decoder(
            input_dimension=8,
            dimension=8,
            activation_function="relu",
            dropout=0.0,
            layer_normalization=True,
            num_layers=3,
            output_mode="candidates",
            noise_init_std=None,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        self.assertTrue(
            all(
                parameter.dtype == torch.float64
                for parameter in decoder.parameters()
            )
        )
        decoder(
            torch.randn(2, ENCODER_WIDTH, 8, dtype=torch.float64)
        ).sum().backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in decoder.parameters()
            )
        )

    def test_bias_free_noisy_linear_accepts_different_output_width(self):
        layer = NoisyLinear(3, 2, bias=False)
        inputs = torch.randn(4, 3)
        self.assertEqual(layer(inputs).shape, (4, 2))
        layer.eval()
        torch.testing.assert_close(layer(inputs), inputs @ layer.weight_mu.T)

    def make_policy(self):
        return EpsilonGreedy(
            epsilon_first=1.0,
            epsilon_last=0.0,
            annealing_steps=10,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def test_epsilon_schedule_advances_and_clamps(self):
        policy = self.make_policy()
        policy.step()
        self.assertAlmostEqual(policy.epsilon.item(), 0.9, places=6)
        policy.step(2)
        self.assertAlmostEqual(policy.epsilon.item(), 0.7, places=6)
        policy.step(0)
        self.assertAlmostEqual(policy.epsilon.item(), 0.7, places=6)
        policy.step(100)
        self.assertEqual(policy.epsilon.item(), 0.0)

    def test_epsilon_schedule_resumes_from_saved_state(self):
        policy = self.make_policy()
        policy.step(3)
        restored = self.make_policy()
        restored.load_state_dict(policy.state_dict())
        restored.step()
        self.assertAlmostEqual(restored.epsilon.item(), 0.6, places=6)

    def test_epsilon_schedule_loads_legacy_epsilon_only_state(self):
        policy = self.make_policy()
        state = policy.state_dict()
        state.pop("_step", None)
        state["epsilon"] = torch.tensor(0.7)
        policy.load_state_dict(state)
        policy.step()
        self.assertAlmostEqual(policy.epsilon.item(), 0.6, places=6)

    def test_exploration_only_selects_legal_actions(self):
        policy = self.make_policy()
        candidates = torch.full(
            (128, 32), NUM_TYPES_OF_ACTIONS, dtype=torch.int32
        )
        candidates[:, :2] = torch.tensor([3, 5])
        data = TensorDict(
            {
                "candidates": candidates,
                "action": torch.zeros(128, dtype=torch.int32),
            },
            [128],
        )
        policy(data)
        self.assertEqual(set(data["action"].tolist()), {0, 1})
        policy.eval()
        data["action"].zero_()
        policy(data)
        self.assertTrue((data["action"] == 0).all().item())

    def test_twin_actor_uses_the_same_pessimistic_values_as_training(self):
        candidates = torch.full(
            (2, 32), NUM_TYPES_OF_ACTIONS, dtype=torch.int32
        )
        candidates[:, :2] = torch.tensor([0, 1])
        q0 = ConstantQ([10.0, 2.0] + [1000.0] * 30)
        q1 = ConstantQ([-10.0, 3.0] + [1000.0] * 30)
        inputs = [torch.zeros(2, 1, dtype=torch.int32)] * 3 + [candidates]
        actor = TwinQActor(q0, q1, return_values=True)
        action, values = actor(*inputs)
        self.assertEqual(action.tolist(), [1, 1])
        torch.testing.assert_close(
            values[:, :2], torch.tensor([[-10.0, 2.0]]).expand(2, -1)
        )
        self.assertTrue(values[:, 2:].isneginf().all().item())
        self.assertFalse(values.requires_grad)
        actor.return_values = False
        self.assertEqual(actor(*inputs).tolist(), [1, 1])

    def test_awr_can_read_values_from_legacy_action_only_twin_export(self):
        from kanachan.training.awr.train import _expose_action_values

        candidates = torch.zeros(2, 32, dtype=torch.int32)
        actor = TwinQActor(ConstantQ([1.0] * 32), ConstantQ([2.0] * 32))
        keys = ["sparse", "numeric", "progression", "candidates"]
        legacy = TensorDictModule(actor, in_keys=keys, out_keys=["action"])
        prepared = _expose_action_values(legacy)
        data = TensorDict({key: candidates.clone() for key in keys}, [2])
        prepared(data)
        torch.testing.assert_close(data["action_value"], torch.ones(2, 32))

    def test_quantile_action_selection_preserves_frozen_parameters_and_modes(
        self,
    ):
        class QuantileModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.frozen = torch.nn.Parameter(
                    torch.tensor(1.0), requires_grad=False
                )
                self.trainable = torch.nn.Parameter(torch.tensor(2.0))
                self.dropout = torch.nn.Dropout()

            def forward(self, data):
                self.asserted_no_grad = not torch.is_grad_enabled()
                data["qr_action_value"] = torch.zeros(
                    data.batch_size[0], 32, 2
                )
                return data

        model = QuantileModel()
        model.train()
        model.dropout.eval()
        data = TensorDict(
            {
                "next": {
                    key: torch.zeros(2, 32, dtype=torch.int32)
                    for key in (
                        "sparse",
                        "numeric",
                        "progression",
                        "candidates",
                    )
                }
            },
            [2],
        )
        _get_a_star(model, data)
        self.assertFalse(model.frozen.requires_grad)
        self.assertTrue(model.trainable.requires_grad)
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        self.assertTrue(model.asserted_no_grad)

    def test_dueling_quantiles_center_only_legal_advantages(self):
        class FixedHead(torch.nn.Module):
            def __init__(self, values):
                super().__init__()
                self.register_buffer(
                    "values", torch.tensor(values, dtype=torch.float32)
                )

            def forward(self, encode):
                return self.values.unsqueeze(0).expand(
                    encode.size(0), *self.values.shape
                )

        decoder = QRDecoder(
            input_dimension=8,
            dimension=None,
            activation_function=None,
            dropout=None,
            layer_normalization=False,
            num_layers=1,
            num_qr_intervals=2,
            dueling_network=True,
            noise_init_std=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        decoder.state_value_decoder_list = torch.nn.ModuleList(
            [FixedHead(5.0), FixedHead(7.0)]
        )
        decoder.advantage_decoder_list = torch.nn.ModuleList(
            [FixedHead(list(range(32)))] * 2
        )
        candidates = torch.full(
            (3, 32), NUM_TYPES_OF_ACTIONS, dtype=torch.int32
        )
        for row, count in enumerate((1, 2, 4)):
            candidates[row, :count] = torch.arange(count)
        values = decoder(candidates, torch.randn(3, ENCODER_WIDTH, 8))
        for row, count in enumerate((1, 2, 4)):
            torch.testing.assert_close(
                values[row, :count].mean(0), torch.tensor([5.0, 7.0])
            )
        self.assertTrue(values[0, 1:].isneginf().all().item())
