import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import tempfile
import unittest
from pathlib import Path

import torch
from tensordict import TensorDict
from test_training_regressions import bc_line

from kanachan.constants import (
    ENCODER_WIDTH,
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_TYPES_OF_ACTIONS,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_SPARSE_FEATURES,
)
from kanachan.model_loader import dump_model, load_model
from kanachan.nn import DecodeConverter, Decoder, Encoder
from kanachan.simulation.test_model import TestModel


class ModelCompatibilityTests(unittest.TestCase):
    def test_legacy_regular_decoder_metadata_loads_without_noisy_parameters(
        self,
    ):
        kwargs = {
            "input_dimension": 8,
            "dimension": 8,
            "activation_function": "relu",
            "dropout": 0.0,
            "num_layers": 2,
            "output_mode": "candidates",
            "noise_init_std": None,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        }
        original = Decoder(**kwargs)
        kwargs["noise_init_std"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.kanachan"
            torch.save(dump_model(original, [], kwargs), path)
            restored = load_model(path)
            inputs = torch.randn(2, ENCODER_WIDTH, 8)
            torch.testing.assert_close(restored(inputs), original(inputs))

    def test_legacy_encoder_and_default_argmax_converter_remain_loadable(self):
        kwargs = {
            "position_encoder": "position_embedding",
            "dimension": 8,
            "num_heads": 2,
            "dim_feedforward": 16,
            "activation_function": "gelu",
            "dropout": 0.0,
            "num_layers": 1,
            "checkpointing": False,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        }
        original = Encoder(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.kanachan"
            torch.save(dump_model(original, [], kwargs), path)
            self.assertEqual(
                load_model(path).state_dict().keys(),
                original.state_dict().keys(),
            )
            torch.save(dump_model(DecodeConverter("argmax"), [], {}), path)
            restored = load_model(path)
            candidates = torch.full(
                (1, 32), NUM_TYPES_OF_ACTIONS, dtype=torch.int32
            )
            candidates[:, :2] = torch.tensor([1, 2])
            scores = torch.arange(32).float().unsqueeze(0)
            self.assertEqual(restored(candidates, scores).item(), 1)

    def test_replay_model_consumes_raw_integer_tensordict_and_all_decisions(
        self,
    ):
        fields = bc_line(1).split("\t")
        keys = ["sparse", "numeric", "progression", "candidates"]
        decision = {
            key: [int(value) for value in field.split(",")]
            for key, field in zip(keys, fields[1:5], strict=True)
        }
        decision["index"] = 1
        data = {}
        for key, width, padding in [
            (
                "sparse",
                MAX_NUM_ACTIVE_SPARSE_FEATURES,
                NUM_TYPES_OF_SPARSE_FEATURES,
            ),
            ("numeric", 6, 0),
            (
                "progression",
                MAX_LENGTH_OF_PROGRESSION_FEATURES,
                NUM_TYPES_OF_PROGRESSION_FEATURES,
            ),
            ("candidates", MAX_NUM_ACTION_CANDIDATES, NUM_TYPES_OF_ACTIONS),
        ]:
            values = decision[key]
            data[key] = torch.tensor(
                [values + [padding] * (width - len(values))], dtype=torch.int32
            )
        model = TestModel([{"decisions": [decision]}])
        with self.assertRaisesRegex(RuntimeError, "not replayed"):
            model.assert_complete()
        self.assertEqual(model(TensorDict(data, [1]))["action"].item(), 1)
        model.assert_complete()
