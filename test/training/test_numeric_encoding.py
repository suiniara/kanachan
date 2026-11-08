import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import unittest
from unittest.mock import patch

import torch

from kanachan import piecewise_linear_encoding
from kanachan.constants import (
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_TYPES_OF_ACTIONS,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_SPARSE_FEATURES,
)
from kanachan.nn import Encoder, RoundEncoder
from kanachan.nn.numeric_encoding import encode_numeric_features


def scalar_reference(numeric, dimension, *, device, dtype):
    rows = []
    for row in numeric.cpu():
        rows.append(torch.stack([
            piecewise_linear_encoding(
                int(value.item()), 0.0,
                float(dimension) if column < 2 else 100000.0,
                dimension, torch.device("cpu"), dtype,
            )
            for column, value in enumerate(row)
        ]))
    if not rows:
        return torch.empty(0, 6, dimension, device=device, dtype=dtype)
    return torch.stack(rows).to(device=device)


class NumericEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_legacy_bins_are_bit_exact_at_boundaries_and_outside_range(self):
        generator = torch.Generator().manual_seed(20261104)
        for dimension in (1, 3, 4, 64, 128, 256, 384, 512):
            random = torch.randint(-20000, 140000, (17, 6), generator=generator, dtype=torch.int32)
            boundary = torch.tensor([
                [0, 0, 0, -1, 100000, 100001],
                [dimension, dimension + 1, 25000, 50000, 75000, 99999],
                [-1, 1, 99, 100, 101, 1000],
            ], dtype=torch.int32)
            values = torch.cat((boundary, random))
            for dtype in (torch.float16, torch.float32, torch.float64):
                with self.subTest(dimension=dimension, dtype=dtype):
                    expected = scalar_reference(values, dimension, device=torch.device("cpu"), dtype=dtype)
                    actual = encode_numeric_features(values, dimension, device=torch.device("cpu"), dtype=dtype)
                    self.assertTrue(torch.equal(actual, expected))

    def test_fractional_inputs_keep_legacy_integer_conversion(self):
        values = torch.tensor([[2.9, -0.9, 24999.9, -100.5, 100001.2, 0.5]], dtype=torch.float64, requires_grad=True)
        actual = encode_numeric_features(values, 64, device=torch.device("cpu"), dtype=torch.float64)
        expected = scalar_reference(values, 64, device=torch.device("cpu"), dtype=torch.float64)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(actual.requires_grad)

    def test_empty_batch_and_noncontiguous_numeric_views(self):
        base = torch.arange(48, dtype=torch.int32).reshape(4, 12)
        values = base[:, ::2]
        self.assertFalse(values.is_contiguous())
        expected = scalar_reference(values, 8, device=torch.device("cpu"), dtype=torch.float32)
        actual = encode_numeric_features(values, 8, device=torch.device("cpu"), dtype=torch.float32)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(encode_numeric_features(values[:0], 8, device=torch.device("cpu"), dtype=torch.float32).shape, (0, 6, 8))

    def test_decision_and_round_outputs_gradients_and_state_keys_match(self):
        arguments = dict(position_encoder="position_embedding", dimension=8, num_heads=2,
                         dim_feedforward=16, num_layers=2, activation_function="gelu", dropout=0.0,
                         checkpointing=False, device=torch.device("cpu"), dtype=torch.float64)
        numeric = torch.tensor([[1, 2, 24900, 25100, -1200, 101200], [3, 0, 100, 200, 300, 400]], dtype=torch.int32)
        for encoder_class, target in [(Encoder, "kanachan.nn.encoder.encode_numeric_features"),
                                      (RoundEncoder, "kanachan.nn.round_encoder.encode_numeric_features")]:
            with self.subTest(encoder=encoder_class.__name__):
                torch.manual_seed(41)
                actual_model = encoder_class(**arguments).train()
                expected_model = encoder_class(**arguments).train()
                expected_model.load_state_dict(actual_model.state_dict(), strict=True)
                sparse = torch.full((2, MAX_NUM_ACTIVE_SPARSE_FEATURES), NUM_TYPES_OF_SPARSE_FEATURES, dtype=torch.int32)
                if encoder_class is Encoder:
                    sparse[:, :8] = torch.tensor([3, 6, 7, 23, 39, 55, 71, 75])
                    progression = torch.full((2, MAX_LENGTH_OF_PROGRESSION_FEATURES), NUM_TYPES_OF_PROGRESSION_FEATURES, dtype=torch.int32)
                    candidates = torch.full((2, MAX_NUM_ACTION_CANDIDATES), NUM_TYPES_OF_ACTIONS, dtype=torch.int32)
                    candidates[:, :2] = torch.tensor([1, 2])
                    inputs = (sparse, numeric, progression, candidates)
                else:
                    inputs = (torch.tensor([[3, 6, 7, 23, 39, 55, 71, 74]] * 2, dtype=torch.int32), numeric)
                with patch(target, side_effect=scalar_reference):
                    expected = expected_model(*inputs)
                    expected.square().mean().backward()
                actual = actual_model(*inputs)
                actual.square().mean().backward()
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(actual_model.state_dict().keys(), expected_model.state_dict().keys())
                for left, right in zip(actual_model.parameters(), expected_model.parameters(), strict=True):
                    if left.grad is not None or right.grad is not None:
                        self.assertTrue(torch.equal(left.grad, right.grad))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_cuda_matches_scalar_reference_without_host_scalar_extraction(self):
        device = torch.device("cuda")
        values = torch.tensor([[1, 2, 25000, 33300, -500, 101000]] * 32, dtype=torch.int32, device=device)
        for dtype in (torch.float16, torch.float32, torch.float64):
            expected = scalar_reference(values, 384, device=device, dtype=dtype)
            with patch.object(torch.Tensor, "item", side_effect=AssertionError("Host scalar extraction")):
                actual = encode_numeric_features(values, 384, device=device, dtype=dtype)
            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
