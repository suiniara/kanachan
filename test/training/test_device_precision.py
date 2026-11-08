import os
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from kanachan.training.core.config.device import validate


def configuration(*, device="cpu", dtype="float32", amp=None, architecture="dense_decoder"):
    return OmegaConf.create({
        "device": {"type": device, "dtype": dtype, "amp_dtype": amp},
        "encoder": {"architecture": architecture},
        "enable_compile": False,
    })


class DevicePrecisionTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, WORLD_SIZE="1", RANK="0", LOCAL_RANK="0")
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_cuda_fp32_master_and_supported_amp_modes(self):
        for name, expected in (
            (None, torch.float32), ("float32", torch.float32),
            ("float16", torch.float16), ("half", torch.float16),
            ("bfloat16", torch.bfloat16), ("bf16", torch.bfloat16),
        ):
            with self.subTest(amp=name), patch("torch.cuda.device_count", return_value=1), patch(
                "torch.cuda.set_device"
            ) as select, patch("torch.cuda.is_bf16_supported", return_value=True) as supported:
                _, _, _, device, dtype, amp = validate(configuration(device="cuda", amp=name))
                self.assertEqual(device, torch.device("cuda", 0))
                self.assertEqual(dtype, torch.float32)
                self.assertEqual(amp, expected)
                select.assert_called_once_with(0)
                self.assertEqual(supported.call_count, int(expected == torch.bfloat16))

    def test_unsupported_gpu_rejects_bf16_without_silent_fallback(self):
        with patch("torch.cuda.device_count", return_value=1), patch("torch.cuda.set_device"), patch(
            "torch.cuda.is_bf16_supported", return_value=False
        ), self.assertRaisesRegex(RuntimeError, "does not support BF16"):
            validate(configuration(device="cuda", amp="bfloat16"))

    def test_cpu_retains_fp32_without_cuda_amp(self):
        for name in (None, "float16", "bfloat16", "bf16"):
            with self.subTest(amp=name), patch("torch.cuda.is_bf16_supported") as supported:
                *_, dtype, amp = validate(configuration(amp=name))
                self.assertEqual(dtype, torch.float32)
                self.assertEqual(amp, torch.float32)
                supported.assert_not_called()

    def test_dense_master_precision_is_rejected_in_both_actual_entrypoints(self):
        from kanachan.training.bc.train import _main as bc_main
        from kanachan.training.cql.train import _main as cql_main

        # No dataset, model or output is needed: rejection must happen before
        # data I/O instead of failing later at FP32 head/loss dtype assertions.
        for entry in (bc_main, cql_main):
            for dtype in ("float64", "double", "float16", "half"):
                with self.subTest(entry=entry.__module__, dtype=dtype), self.assertRaisesRegex(
                    RuntimeError, "requires FP32 master parameters"
                ):
                    entry.__wrapped__(configuration(dtype=dtype))

    def test_legacy_double_and_half_cpu_parameters_remain_supported(self):
        for architecture in ("transformer", "rmsnorm_swiglu"):
            for name, expected in (("double", torch.float64), ("half", torch.float16)):
                with self.subTest(architecture=architecture, dtype=name):
                    *_, dtype, amp = validate(configuration(dtype=name, architecture=architecture))
                    self.assertEqual(dtype, expected)
                    self.assertEqual(amp, expected)

    def test_bf16_cuda_requires_fp32_master_even_for_legacy_models(self):
        with patch("torch.cuda.device_count", return_value=1), patch("torch.cuda.set_device"), self.assertRaisesRegex(
            RuntimeError, "BF16 AMP requires FP32"
        ):
            validate(configuration(device="cuda", dtype="double", amp="bf16", architecture="transformer"))

    def test_cpu_fp32_alias_and_missing_encoder_config(self):
        config = configuration(dtype="float", amp="bf16")
        del config.encoder
        *_, dtype, amp = validate(config)
        self.assertEqual(dtype, torch.float32)
        self.assertEqual(amp, torch.float32)


if __name__ == "__main__":
    unittest.main()
