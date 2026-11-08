import os
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from kanachan.training.core.config.device import validate


class DeviceConfigurationTests(unittest.TestCase):
    def config(self, kind="cpu"):
        return OmegaConf.create(
            {"device": {"type": kind, "dtype": "float32", "amp_dtype": None}}
        )

    def test_direct_single_process_run_needs_no_launcher_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            result = validate(self.config())
        self.assertEqual(result[:3], (1, 0, 0))
        self.assertEqual(
            result[3:], (torch.device("cpu"), torch.float32, torch.float32)
        )

    def test_requested_cuda_does_not_silently_fall_back_to_cpu(self):
        with (
            patch.dict(
                os.environ, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}
            ),
            patch("torch.cuda.device_count", return_value=0),
            self.assertRaisesRegex(RuntimeError, "CUDA|cuda"),
        ):
            validate(self.config("cuda"))

    def test_invalid_distributed_ranks_are_rejected(self):
        for world, rank, local in ((0, 0, 0), (2, 2, 0), (1, 0, -1)):
            with (
                self.subTest(world=world, rank=rank, local=local),
                patch.dict(
                    os.environ,
                    {
                        "WORLD_SIZE": str(world),
                        "RANK": str(rank),
                        "LOCAL_RANK": str(local),
                    },
                ),
                self.assertRaises(RuntimeError),
            ):
                validate(self.config())

    def test_incomplete_multi_process_environment_is_rejected(self):
        with (
            patch.dict(os.environ, {"WORLD_SIZE": "2"}, clear=True),
            self.assertRaises(RuntimeError),
        ):
            validate(self.config())
