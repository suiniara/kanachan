import copy
import os
import unittest
import warnings
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from kanachan.nn.decision_attention import decision_attention, flex_attention
from kanachan.nn.decision_factory import backbone_kwargs
from kanachan.training.core.config.encoder import DecisionTinyConfig, validate
from test_decision_decoder import apply, backbone, observation


class DecisionAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(73)

    def test_flex_mask_outputs_and_qkv_gradients_match_math(self):
        from torch._dynamo import config as dynamo_config

        for mode in ("causal", "bidirectional"):
            with self.subTest(mode=mode), patch.dict(
                os.environ, TORCHDYNAMO_DISABLE="0"
            ), dynamo_config.patch(disable=False), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="flex_attention called without")
                body = backbone(history_attention=mode)
                data = observation()
                # Empty and maximum histories, mixed sparse holes, plus a
                # 147-token non-tile-aligned tail exercise padding and masks.
                data["progression"][2] = torch.arange(109)
                data["sparse"][1, 1::2] = 510
                _, _, mask, _ = body.tokens(*(data[k] for k in (
                    "sparse", "numeric", "progression", "candidates"
                )))
                packed = torch.randn(3, 147, 3, 2, 8, requires_grad=True)
                q, k, v = packed.permute(2, 0, 3, 1, 4).unbind(0)
                weight = torch.randn_like(q)
                with sdpa_kernel(SDPBackend.MATH):
                    expected = decision_attention(q, k, v, mask)
                expected_grad, = torch.autograd.grad((expected * weight).sum(), packed)
                actual = flex_attention(q, k, v, mask)
                actual_grad, = torch.autograd.grad((actual * weight).sum(), packed)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=2e-4)

    def test_compiled_frontend_reads_new_mask_contents(self):
        from torch._dynamo import config as dynamo_config

        with patch.dict(os.environ, TORCHDYNAMO_DISABLE="0"), dynamo_config.patch(disable=False):
            compiled = torch.compile(flex_attention, backend="eager", fullgraph=True, dynamic=True)
            for batch, length, offset in ((2, 9, 0), (3, 11, 3), (1, 7, 0), (4, 9, 2)):
                q, k, v = (torch.randn(batch, 2, length, 8) for _ in range(3))
                mask = torch.ones(batch, 1, length, length, dtype=torch.bool).tril()
                mask[-1, :, :, 1:1 + offset] = False
                with sdpa_kernel(SDPBackend.MATH):
                    expected = decision_attention(q, k, v, mask)
                torch.testing.assert_close(compiled(q, k, v, mask), expected,
                                           atol=1e-6, rtol=1e-5)

    def test_full_decoder_and_gradients_match_with_flex_frontend(self):
        from torch._dynamo import config as dynamo_config

        reference = backbone()
        candidate = copy.deepcopy(reference)
        data = observation()
        expected = apply(reference, data)
        expected.square().sum().backward()
        with patch.dict(os.environ, TORCHDYNAMO_DISABLE="0"), dynamo_config.patch(disable=False):
            with patch("kanachan.nn.decision_decoder.decision_attention",
                       side_effect=lambda q, k, v, mask, backend: flex_attention(q, k, v, mask)):
                actual = apply(candidate, data)
                actual.square().sum().backward()
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
        for (name, a), (_, b) in zip(reference.named_parameters(), candidate.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=5e-4, msg=name)

    def test_aot_autograd_frontend_compiles_forward_and_backward(self):
        from torch._dynamo import config as dynamo_config

        with patch.dict(os.environ, TORCHDYNAMO_DISABLE="0"), dynamo_config.patch(disable=False):
            compiled = torch.compile(flex_attention, backend="aot_eager", fullgraph=True, dynamic=True)
            tensors = tuple(torch.randn(2, 2, 9, 8, requires_grad=True) for _ in range(3))
            mask = torch.ones(2, 1, 9, 9, dtype=torch.bool).tril()
            mask[1, :, :, 2:4] = False
            with sdpa_kernel(SDPBackend.MATH):
                expected = decision_attention(*tensors, mask)
            expected_grad = torch.autograd.grad(expected.square().sum(), tensors)
            actual = compiled(*tensors, mask)
            actual_grad = torch.autograd.grad(actual.square().sum(), tensors)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5)
            for a, b in zip(actual_grad, expected_grad):
                torch.testing.assert_close(a, b, atol=3e-6, rtol=2e-4)

    def test_backend_export_and_legacy_recipe_compatibility(self):
        config = DecisionTinyConfig()
        old = backbone_kwargs(config, checkpointing=False, dtype=torch.float32)
        self.assertNotIn("attention_backend", old)
        config.attention_backend = "flex"
        validate(config)
        new = backbone_kwargs(config, checkpointing=False, dtype=torch.float32)
        self.assertEqual(new.pop("attention_backend"), "flex")
        self.assertEqual(old, new)
        reference = backbone()
        candidate = backbone(attention_backend="flex")
        candidate.load_state_dict(reference.state_dict(), strict=True)
        # CPU is an explicit portable SDPA path, including for Flex exports.
        torch.testing.assert_close(apply(reference, observation()),
                                   apply(candidate, observation()), atol=0, rtol=0)
        config.attention_backend = "flashy"
        with self.assertRaises(ValueError):
            validate(config)
        with self.assertRaises(ValueError):
            backbone(attention_backend="flashy")

    def test_reject_invalid_mask_and_backend(self):
        q = torch.randn(1, 2, 3, 8)
        for mask in (torch.ones(3, 3, dtype=torch.bool),
                     torch.ones(1, 1, 3, 3),
                     torch.ones(1, 2, 3, 3, dtype=torch.bool)):
            with self.assertRaisesRegex(ValueError, "boolean"):
                decision_attention(q, q, q, mask)
        with self.assertRaisesRegex(ValueError, "backend"):
            decision_attention(q, q, q, torch.ones(1, 1, 3, 3, dtype=torch.bool), "bad")
        with self.assertRaisesRegex(ValueError, "boolean"):
            decision_attention(q.expand(2, -1, -1, -1), q, q,
                               torch.ones(1, 1, 3, 3, dtype=torch.bool))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not mounted")
    def test_real_cuda_fused_forward_backward(self):
        from torch._dynamo import config as dynamo_config

        with patch.dict(os.environ, TORCHDYNAMO_DISABLE="0"), dynamo_config.patch(disable=False):
            self._cuda_forward_backward()

    def _cuda_forward_backward(self):
        data = observation(device="cuda")
        reference = backbone(device=torch.device("cuda"), dimension=128,
                             num_heads=2, dim_feedforward=192)
        candidate = backbone(device=torch.device("cuda"), dimension=128,
                             num_heads=2, dim_feedforward=192, attention_backend="flex")
        candidate.load_state_dict(reference.state_dict())
        with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(SDPBackend.MATH):
            expected = apply(reference, data)
        expected.square().sum().backward()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = apply(candidate, data)
        actual.square().sum().backward()
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.05)
        for (name, a), (_, b) in zip(reference.named_parameters(), candidate.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=5e-4, rtol=0.05, msg=name)


if __name__ == "__main__":
    unittest.main()
