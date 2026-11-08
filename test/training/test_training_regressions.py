import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import math
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch import nn

from kanachan.constants import ENCODER_WIDTH, NUM_TYPES_OF_ACTIONS
from kanachan.nn import DecodeConverter, Decoder, QRDecoder
from kanachan.nn.functional import advantage_weights, compute_td_target
from kanachan.training.bc.evaluate import evaluate
from kanachan.training.common import (
    backoff_gradient_scale,
    is_gradient_nonfinite,
)
from kanachan.training.core.bc.data_iterator import DataIterator
from kanachan.training.core.config import optimizer as optimizer_config
from kanachan.training.ilql.train import _backward


def bc_line(action: int = 0) -> str:
    return (
        "\t".join(
            [
                "game-0",
                "3,6,7,23,39,55,71,75,78,82,152",
                "0,0,25000,25000,25000,25000",
                "0,5",
                "0,1,2,3",
                str(action),
                "0",
                ",".join(["0"] * 14),
            ]
        )
        + "\n"
    )


class ToyQV(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Parameter(torch.tensor(0.2))
        self.v = nn.Parameter(torch.tensor(0.3))

    def forward(self, data):
        data["action_value"] = (self.q * data["x"]).unsqueeze(1).expand(-1, 32)
        data["state_value"] = self.v * data["x"]
        return data


class TrainingRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_nonlinear_decoders_learn_after_zero_output(self):
        torch.manual_seed(29)
        inputs = torch.randn(16, ENCODER_WIDTH, 8)
        targets = inputs[:, -32:, 0][:, :4].argmax(1)
        candidates = torch.full(
            (16, 32), NUM_TYPES_OF_ACTIONS, dtype=torch.int32
        )
        candidates[:, :4] = torch.arange(4, dtype=torch.int32)
        for layers in (1, 2, 3):
            with self.subTest(layers=layers):
                features = nn.Parameter(inputs.clone())
                decoder = Decoder(
                    input_dimension=8,
                    dimension=8 if layers > 1 else None,
                    activation_function="relu" if layers > 1 else None,
                    dropout=0.0 if layers > 1 else None,
                    layer_normalization=False,
                    num_layers=layers,
                    output_mode="candidates",
                    noise_init_std=None,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                decoder.zero_output()
                optimizer = torch.optim.Adam(
                    [features, *decoder.parameters()], lr=0.03
                )
                converter = DecodeConverter("log_probs")
                for step in range(50):
                    optimizer.zero_grad()
                    log_probs = converter(candidates, decoder(features))
                    loss = nn.functional.nll_loss(log_probs, targets)
                    loss.backward()
                    optimizer.step()
                self.assertLess(loss.item(), 0.1)
                self.assertGreater(features.grad.norm().item(), 0.0)

    def test_zero_output_preserves_hidden_layer_normalization(self):
        decoder = Decoder(
            input_dimension=8,
            dimension=8,
            activation_function="gelu",
            dropout=0.0,
            layer_normalization=True,
            num_layers=2,
            output_mode="candidates",
            noise_init_std=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        decoder.zero_output()
        self.assertTrue(
            torch.equal(
                decoder.layers.layer_normalization0.weight, torch.ones(8)
            )
        )

    def test_quantile_decoder_zero_output_keeps_hidden_layers_trainable(self):
        decoder = QRDecoder(
            input_dimension=8,
            dimension=8,
            activation_function="relu",
            dropout=0.0,
            layer_normalization=False,
            num_layers=2,
            num_qr_intervals=2,
            dueling_network=True,
            noise_init_std=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        decoder.zero_output()
        candidates = torch.zeros(2, 32, dtype=torch.int32)
        encode = torch.randn(2, ENCODER_WIDTH, 8)
        optimizer = torch.optim.SGD(decoder.parameters(), lr=0.01)
        for _ in range(2):
            optimizer.zero_grad()
            loss = (decoder(candidates, encode)[:, 0] - 1.0).square().mean()
            loss.backward()
            optimizer.step()
        gradient = decoder.advantage_decoder_list[0].layers.linear0.weight.grad
        self.assertGreater(gradient.norm().item(), 0.0)

    def test_td_targets_mask_terminal_infinities_and_are_detached(self):
        for quantiles in (False, True):
            with self.subTest(quantiles=quantiles):
                shape = (2, 3) if quantiles else (2,)
                next_value = torch.full(shape, 4.0, requires_grad=True)
                with torch.no_grad():
                    next_value[0] = -math.inf
                target = compute_td_target(
                    torch.tensor([1.0, 2.0]),
                    next_value,
                    torch.tensor([True, False]),
                    0.5,
                )
                expected = torch.tensor([1.0, 4.0])
                if quantiles:
                    expected = expected.unsqueeze(1).expand(-1, 3)
                torch.testing.assert_close(target, expected)
                self.assertFalse(target.requires_grad)

    def test_zero_discount_ignores_bootstrap_infinities(self):
        target = compute_td_target(
            torch.tensor([1.0]),
            torch.tensor([-math.inf]),
            torch.tensor([False]),
            0.0,
        )
        torch.testing.assert_close(target, torch.tensor([1.0]))

    def test_ilql_td_loss_does_not_update_bootstrap_value_parameters(self):
        for scaling in (0.0, 1.0):
            with self.subTest(value_loss_scaling=scaling):
                model = ToyQV()
                data = TensorDict(
                    {
                        "x": torch.tensor([1.0]),
                        "action": torch.tensor([0], dtype=torch.int32),
                        "next": {
                            "x": torch.tensor([2.0]),
                            "done": torch.tensor([False]),
                            "reward": torch.tensor([1.0]),
                        },
                    },
                    batch_size=[1],
                )
                _backward(
                    world_size=1,
                    autocast_kwargs={"device_type": "cpu", "enabled": False},
                    data=data,
                    source_model=model,
                    discount_factor=0.9,
                    expectile=0.9,
                    v_loss_scaling=scaling,
                    gradient_accumulation_steps=1,
                    grad_scaler=torch.amp.GradScaler("cpu", enabled=False),
                    q_target=torch.tensor([1.0]),
                )
                self.assertGreater(abs(model.q.grad.item()), 0.0)
                if scaling == 0.0:
                    self.assertEqual(model.v.grad.item(), 0.0)
                else:
                    self.assertGreater(abs(model.v.grad.item()), 0.0)

    def test_awr_weights_are_stable_clipped_and_detached(self):
        q = torch.tensor([100.0, 1000.0, -1000.0], requires_grad=True)
        value = torch.tensor([100.0, 0.0, 0.0], requires_grad=True)
        weights = advantage_weights(q, value, 1.0, 20.0)
        torch.testing.assert_close(weights, torch.tensor([1.0, 20.0, 0.0]))
        self.assertFalse(weights.requires_grad)
        shifted = advantage_weights(q + 10000, value + 10000, 1.0, 20.0)
        torch.testing.assert_close(weights, shifted)

    def test_awr_weights_use_float32_for_half_precision_values(self):
        weights = advantage_weights(
            torch.tensor([100.0], dtype=torch.float16),
            torch.tensor([100.0], dtype=torch.float16),
            1.0,
            20.0,
        )
        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(weights.item(), 1.0)

    def test_cpu_optimizer_presets_validate_without_apex(self):
        for config_type in (
            optimizer_config.AdamOptimizerConfig,
            optimizer_config.RAdamOptimizerConfig,
            optimizer_config.SgdOptimizerConfig,
        ):
            with self.subTest(config=config_type.__name__):
                config = OmegaConf.structured(config_type(learning_rate=0.01))
                optimizer_config.validate(config)
                model = nn.Linear(1, 1)
                optimizer, _ = optimizer_config.create("cpu", config, model)
                model(torch.ones(1, 1)).sum().backward()
                optimizer.step()

    def test_nonfinite_gradient_backoff_allows_next_step_to_recover(self):
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu")
        scaler.scale(model(torch.ones(1, 1)).sum() * math.inf).backward()
        self.assertTrue(is_gradient_nonfinite(model).item())
        old_scale = scaler.get_scale()
        backoff_gradient_scale(scaler)
        self.assertLess(scaler.get_scale(), old_scale)
        optimizer.zero_grad()
        scaler.scale(model(torch.ones(1, 1)).sum()).backward()
        self.assertFalse(is_gradient_nonfinite(model).item())
        scaler.step(optimizer)
        scaler.update()
        for parameter in model.parameters():
            self.assertTrue(parameter.isfinite().all().item())

    def test_validation_keeps_targets_separate_from_predicted_actions(self):
        class Policy(nn.Module):
            def forward(self, data):
                self.assert_no_labels(data)
                logits = torch.full((data.size(0), 32), -math.inf)
                logits[:, :2] = torch.tensor([1.0, 0.0])
                data["log_probs"] = logits.log_softmax(1)
                data["action"] = torch.zeros(data.size(0), dtype=torch.int32)
                return data

            def assert_no_labels(self, data):
                if "action" in data or "results" in data:
                    raise AssertionError("Labels must not be model inputs.")

        candidates = torch.full(
            (3, 32), NUM_TYPES_OF_ACTIONS, dtype=torch.int32
        )
        candidates[:, :2] = torch.tensor([0, 1], dtype=torch.int32)
        data = TensorDict(
            {
                "sparse": torch.zeros(3, 29, dtype=torch.int32),
                "numeric": torch.zeros(3, 6, dtype=torch.int32),
                "progression": torch.zeros(3, 109, dtype=torch.int32),
                "candidates": candidates,
                "action": torch.tensor([0, 1, 0], dtype=torch.int32),
                "results": torch.zeros(3, 14),
            },
            batch_size=[3],
        )
        policy = Policy()
        result = evaluate(policy, [data[:2], data[2:]], torch.device("cpu"))
        self.assertEqual(result["num_samples"], 3)
        self.assertAlmostEqual(result["action_accuracy"], 2 / 3)
        self.assertEqual(result["random_action_accuracy"], 0.5)
        self.assertTrue(policy.training)
        torch.testing.assert_close(
            data["action"], torch.tensor([0, 1, 0], dtype=torch.int32)
        )
        result = evaluate(policy, [data], torch.device("cpu"), max_samples=2)
        self.assertEqual(result["num_samples"], 2)
        self.assertEqual(result["action_accuracy"], 0.5)

    def test_bc_reader_rejects_legacy_columns_and_invalid_targets(self):
        for content, message in [
            ("\t".join(bc_line().rstrip().split("\t")[:7]) + "\n", "v2 BC"),
            (bc_line(4), "not legal"),
            (bc_line(-1), "not legal"),
            (bc_line().replace("3,6,7", "-1,6,7"), "-1"),
        ]:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "data.tsv"
                path.write_text(content, encoding="utf-8")
                iterator = DataIterator(
                    path=path,
                    num_skip_samples=0,
                    rewrite_rooms=None,
                    rewrite_grades=None,
                    local_rank=0,
                )
                try:
                    with self.assertRaisesRegex(RuntimeError, message):
                        next(iterator)
                finally:
                    del iterator


if __name__ == "__main__":
    unittest.main()
