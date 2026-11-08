import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from kanachan.constants import (
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_ACTIONS,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_SPARSE_FEATURES,
)
from kanachan.model_loader import dump_model, load_model
from kanachan.nn import Encoder, RoundEncoder
from kanachan.nn.modern_encoder import RMSNorm, RMSNormSwiGLULayer
from kanachan.training.core.config.encoder import (
    BertTinyEncoderConfig, validate,
)


def encoder_arguments(**extra):
    return dict(
        position_encoder="position_embedding", dimension=8, num_heads=2,
        dim_feedforward=32, num_layers=2, activation_function="gelu",
        dropout=0.0, checkpointing=False, device=torch.device("cpu"),
        dtype=torch.float64, **extra,
    )


def decision_inputs():
    sparse = torch.full(
        (2, MAX_NUM_ACTIVE_SPARSE_FEATURES), NUM_TYPES_OF_SPARSE_FEATURES,
        dtype=torch.int32,
    )
    sparse[:, :8] = torch.tensor([3, 6, 7, 23, 39, 55, 71, 75])
    numeric = torch.tensor(
        [[1, 2, 24900, 25100, -1200, 101200],
         [3, 0, 100, 200, 300, 400]], dtype=torch.int32,
    )
    progression = torch.full(
        (2, MAX_LENGTH_OF_PROGRESSION_FEATURES),
        NUM_TYPES_OF_PROGRESSION_FEATURES, dtype=torch.int32,
    )
    progression[:, :3] = torch.tensor([0, 5, 10])
    candidates = torch.full(
        (2, MAX_NUM_ACTION_CANDIDATES), NUM_TYPES_OF_ACTIONS,
        dtype=torch.int32,
    )
    candidates[:, :4] = torch.tensor([1, 2, 3, 4])
    return sparse, numeric, progression, candidates


def round_inputs():
    numeric = decision_inputs()[1]
    sparse = torch.tensor(
        [[3, 6, 7, 23, 39, 55, 71, 74]] * 2, dtype=torch.int32,
    )
    return sparse, numeric


class ModernEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_rmsnorm_uses_stable_accumulation_and_retains_dtype(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                norm = RMSNorm(4, device=torch.device("cpu"), dtype=dtype)
                values = torch.tensor(
                    [[0.0, 0.0, 0.0, 0.0], [1e4, -1e4, 3.0, 4.0]],
                    dtype=dtype, requires_grad=True,
                )
                actual = norm(values)
                working = values.double() if dtype == torch.float64 else values.float()
                expected = working / (working.square().mean(-1, keepdim=True) + 1e-5).sqrt()
                self.assertEqual(actual.dtype, dtype)
                self.assertTrue(actual.isfinite().all())
                torch.testing.assert_close(actual, expected.to(dtype))
                actual.square().sum().backward()
                self.assertTrue(values.grad.isfinite().all())
                self.assertTrue(norm.weight.grad.isfinite().all())

    def test_sdpa_matches_explicit_bidirectional_attention_and_gradients(self):
        torch.manual_seed(41)
        layer = RMSNormSwiGLULayer(
            8, 2, 32, 0.0, device=torch.device("cpu"), dtype=torch.float64,
        )
        reference = copy.deepcopy(layer)
        values = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
        original = values.detach().clone().requires_grad_(True)
        padding = torch.tensor(
            [[False, False, True, False, True],
             [False, True, False, False, False]],
        )

        def norm(value, weight):
            return value / (value.square().mean(-1, keepdim=True) + 1e-5).sqrt() * weight

        normalized = norm(original, reference.attention_norm.weight)
        qkv = F.linear(normalized, reference.qkv.weight).reshape(2, 5, 3, 2, 4)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = query @ key.transpose(-2, -1) / math.sqrt(4)
        scores = scores.masked_fill(padding[:, None, None, :], -torch.inf)
        attended = (scores.softmax(-1) @ value).transpose(1, 2).reshape(2, 5, 8)
        residual = original + F.linear(attended, reference.attention_output.weight)
        normalized = norm(residual, reference.feedforward_norm.weight)
        gate, up = F.linear(normalized, reference.gate_up.weight).chunk(2, dim=-1)
        expected = residual + F.linear(gate * gate.sigmoid() * up, reference.down.weight)
        actual = layer(values, padding)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(values.grad, original.grad, rtol=1e-11, atol=1e-12)
        for left, right in zip(layer.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-11, atol=1e-12)

    def test_later_tokens_can_influence_earlier_tokens(self):
        layer = RMSNormSwiGLULayer(
            8, 2, 32, 0.0, device=torch.device("cpu"), dtype=torch.float64,
        ).eval()
        values = torch.randn(1, 3, 8, dtype=torch.float64)
        changed = values.clone()
        changed[:, -1] = torch.arange(8, dtype=torch.float64)
        self.assertFalse(torch.equal(layer(values)[:, 0], layer(changed)[:, 0]))
        mask = torch.tensor([[False, False, True]])
        self.assertTrue(torch.equal(layer(values, mask)[:, 0], layer(changed, mask)[:, 0]))

    def test_decision_padding_cannot_influence_valid_tokens(self):
        model = Encoder(**encoder_arguments(architecture="rmsnorm_swiglu")).eval()
        inputs = decision_inputs()
        padding = torch.cat((
            inputs[0] == NUM_TYPES_OF_SPARSE_FEATURES,
            torch.zeros(2, NUM_NUMERIC_FEATURES, dtype=torch.bool),
            inputs[2] == NUM_TYPES_OF_PROGRESSION_FEATURES,
            inputs[3] == NUM_TYPES_OF_ACTIONS,
        ), dim=1)
        with torch.no_grad():
            expected = model(*inputs)
            for embedding in (model.sparse_embedding, model.progression_embedding,
                              model.candidates_embedding):
                embedding.weight[-1].normal_(mean=5.0, std=4.0)
            actual = model(*inputs)
        self.assertTrue(torch.equal(expected[~padding], actual[~padding]))

    def test_evaluation_disables_attention_and_residual_dropout(self):
        layer = RMSNormSwiGLULayer(
            8, 2, 32, 0.4, device=torch.device("cpu"), dtype=torch.float32,
        ).eval()
        values = torch.randn(2, 5, 8)
        rng = torch.get_rng_state().clone()
        first, second = layer(values), layer(values)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        layer.train()
        self.assertFalse(torch.equal(layer(values), layer(values)))

    def test_checkpointing_preserves_dropout_outputs_and_gradients(self):
        for cls in (Encoder, RoundEncoder):
            with self.subTest(encoder=cls.__name__):
                args = encoder_arguments(architecture="rmsnorm_swiglu")
                args["dropout"] = 0.2
                ordinary = cls(**args).train()
                checkpointed = copy.deepcopy(ordinary)
                checkpointed.checkpointing = True
                inputs = decision_inputs()
                if cls is RoundEncoder:
                    inputs = round_inputs()
                torch.manual_seed(101)
                expected = ordinary(*inputs)
                weights = torch.linspace(
                    -1.0, 1.0, expected.numel(), dtype=expected.dtype
                ).reshape_as(expected)
                (expected * weights).mean().backward()
                torch.manual_seed(101)
                actual = checkpointed(*inputs)
                (actual * weights).mean().backward()
                self.assertTrue(torch.equal(actual, expected))
                for left, right in zip(ordinary.parameters(), checkpointed.parameters(), strict=True):
                    self.assertTrue(torch.equal(left.grad, right.grad))

    def test_modern_exports_reload_and_legacy_weights_are_not_silently_converted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.kanachan"
            for cls in (Encoder, RoundEncoder):
                with self.subTest(encoder=cls.__name__):
                    args = encoder_arguments(architecture="rmsnorm_swiglu")
                    model = cls(**args).eval()
                    torch.save(dump_model(model, [], args), path)
                    restored = load_model(path, map_location=torch.device("cpu")).eval()
                    inputs = decision_inputs()
                    if cls is RoundEncoder:
                        inputs = round_inputs()
                    self.assertTrue(torch.equal(model(*inputs), restored(*inputs)))
                    legacy = cls(**encoder_arguments())
                    with self.assertRaises(RuntimeError):
                        restored.load_state_dict(legacy.state_dict(), strict=True)

    def test_legacy_default_keeps_weights_and_forward_values(self):
        for cls in (Encoder, RoundEncoder):
            with self.subTest(encoder=cls.__name__):
                torch.manual_seed(73)
                legacy = cls(**encoder_arguments()).eval()
                torch.manual_seed(73)
                explicit = cls(**encoder_arguments(architecture="transformer")).eval()
                self.assertEqual(legacy.state_dict().keys(), explicit.state_dict().keys())
                inputs = decision_inputs()
                if cls is RoundEncoder:
                    inputs = round_inputs()
                self.assertTrue(torch.equal(legacy(*inputs), explicit(*inputs)))

    def test_architecture_configuration_and_parameter_budget(self):
        cfg = BertTinyEncoderConfig(architecture="rmsnorm_swiglu")
        validate(cfg)
        self.assertEqual(cfg.dim_feedforward, 512)
        cfg.layer_normalization = True
        with self.assertRaisesRegex(RuntimeError, "own normalization"):
            validate(cfg)
        args = encoder_arguments()
        for architecture in ("unknown", "rmsnorm_swiglu"):
            with self.assertRaises(ValueError):
                Encoder(**args, architecture=architecture, layer_normalization=True)
        args.update(dimension=128, num_heads=2, dim_feedforward=512)
        legacy = Encoder(**args)
        modern = Encoder(**args, architecture="rmsnorm_swiglu")
        legacy_count = sum(p.numel() for p in legacy.parameters())
        modern_count = sum(p.numel() for p in modern.parameters())
        self.assertLess(abs(modern_count / legacy_count - 1), 0.01)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_cuda_autocast_checkpointed_backward_is_finite(self):
        args = encoder_arguments(architecture="rmsnorm_swiglu")
        args.update(device=torch.device("cuda"), dtype=torch.float32, checkpointing=True)
        model = Encoder(**args).train()
        inputs = tuple(value.cuda() for value in decision_inputs())
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            output = model(*inputs)
            output.square().mean().backward()
        self.assertTrue(output.isfinite().all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(parameter.grad.isfinite().all())


if __name__ == "__main__":
    unittest.main()
