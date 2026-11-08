"""Ordered features, identical updates/resume and raw-input policy exports."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from test_candidate_features import observation, training_line

from kanachan.candidate_features import derive_candidate_features
from kanachan.model_loader import load_model
from kanachan.nn.decision_decoder import GlobalActionHead
from kanachan.training.core.bc import DataLoader
from kanachan.training.core.bc.prefetch import CandidateFeaturePrefetch

ROOT = Path(__file__).resolve().parents[2]


def assert_state_equal(test, first, second):
    if isinstance(first, torch.Tensor):
        test.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_state_equal(test, first[key], second[key])
    elif isinstance(first, (tuple, list)):
        test.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            assert_state_equal(test, a, b)
    else:
        test.assertEqual(first, second)


@unittest.skipUnless(
    importlib.util.find_spec("kanachan._candidate_features"),
    "Build native features for prefetch checks",
)
class BCPrefetchTests(unittest.TestCase):
    def test_order_ragged_batches_features_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.tsv"
            path.write_text("".join(training_line(i) for i in range(17)))
            arguments = {
                "path": path,
                "num_skip_samples": 0,
                "rewrite_rooms": None,
                "rewrite_grades": None,
                "batch_size": 4,
                "num_workers": 0,
            }
            expected = list(DataLoader(**arguments))
            source = DataLoader(**arguments)
            rng = torch.get_rng_state()
            with CandidateFeaturePrefetch(
                source, workers=2, batches=2
            ) as pipeline:
                result = list(pipeline)
                with self.assertRaises(StopIteration):
                    next(pipeline)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(len(expected), len(result))
            raw_source = DataLoader(**arguments, raw_batches=True)
            with CandidateFeaturePrefetch(
                raw_source, workers=2, batches=2, raw_batches=True
            ) as pipeline:
                raw_result = list(pipeline)
            for reference, actual in zip(result, raw_result):
                for key in reference.keys():  # noqa: SIM118 - TensorDict keys
                    self.assertTrue(
                        torch.equal(reference[key], actual[key]), key
                    )
            for reference, actual in zip(expected, result):
                for key in reference.keys():  # noqa: SIM118 - TensorDict keys
                    self.assertTrue(
                        torch.equal(reference[key], actual[key]), key
                    )
                derived = derive_candidate_features(
                    reference["sparse"],
                    reference["progression"],
                    reference["candidates"],
                )
                self.assertTrue(
                    torch.equal(derived, actual["candidate_afterstate"])
                )

    def test_prefetched_head_values_and_gradients_match(self):
        data = observation()
        head = GlobalActionHead(
            input_dimension=8,
            hidden_dimension=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            candidate_features="discard_v1",
        )
        torch.nn.init.normal_(head.candidate_layers[-1].weight, std=0.03)
        representation = torch.randn(1, 8)
        values = head(
            data["candidates"],
            representation,
            data["sparse"],
            data["progression"],
        )
        values.sum().backward()
        gradients = {
            name: p.grad.clone() for name, p in head.named_parameters()
        }
        head.zero_grad(set_to_none=True)
        features = derive_candidate_features(
            data["sparse"], data["progression"], data["candidates"]
        )
        cached = head(
            data["candidates"],
            representation,
            data["sparse"],
            data["progression"],
            features,
        )
        cached.sum().backward()
        self.assertTrue(torch.equal(values, cached))
        for name, parameter in head.named_parameters():
            self.assertTrue(torch.equal(parameter.grad, gradients[name]), name)
        with self.assertRaises(ValueError):
            head(
                data["candidates"],
                representation,
                data["sparse"],
                data["progression"],
                features.double(),
            )

    def test_worker_error_is_propagated_and_pool_closed(self):
        data = observation()
        data["sparse"][0, 10] = 510
        with CandidateFeaturePrefetch(
            [data], workers=1, batches=1
        ) as pipeline:
            with self.assertRaises(ValueError):
                next(pipeline)
            self.assertTrue(pipeline.closed)

    def test_real_bc_updates_resume_and_export_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "data.tsv"
            path.write_text("".join(training_line(i) for i in range(64)))
            seed = "import random,numpy,torch,runpy; random.seed(71); numpy.random.seed(71); torch.set_num_threads(2); torch.manual_seed(71); runpy.run_module('kanachan.training.bc.train',run_name='__main__')"
            env = dict(
                os.environ,
                PYTHONPATH=str(ROOT),
                TORCHDYNAMO_DISABLE="1",
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
            )

            def entry(name, workers, resume=False):
                arguments = [
                    sys.executable,
                    "-B",
                    "-c",
                    seed,
                    "--config-name",
                    "decision",
                    "device=cpu",
                    "device.dtype=float32",
                    "device.amp_dtype=float32",
                    "encoder.dimension=16",
                    "encoder.num_heads=2",
                    "encoder.dim_feedforward=24",
                    "encoder.num_layers=1",
                    "decoder.candidate_features=discard_v1",
                    "batch_size=8",
                    "num_workers=0",
                    "snapshot_interval=32",
                    f"candidate_feature_workers={workers}",
                    "candidate_prefetch_batches=2",
                    f"training_data={path}",
                    f"hydra.run.dir={root / name}",
                ]
                if resume:
                    arguments += [
                        f"initial_model_prefix={root}/baseline/snapshots",
                        "initial_model_index=32",
                    ]
                result = subprocess.run(
                    arguments,
                    check=False,
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )

            entry("baseline", 0)
            entry("prefetched", 2)
            entry("resumed", 2, True)
            for name in ("prefetched", "resumed"):
                for file in (
                    "encoder.pth",
                    "decoder.pth",
                    "optimizer.pth",
                    "decision-rng.pth",
                ):
                    assert_state_equal(
                        self,
                        torch.load(
                            root / "baseline/snapshots" / file,
                            weights_only=True,
                        ),
                        torch.load(
                            root / name / "snapshots" / file, weights_only=True
                        ),
                    )
                policy = load_model(
                    root / name / "snapshots/model.kanachan",
                    map_location="cpu",
                )
                batch = observation()
                policy(batch)
                self.assertTrue(
                    torch.isfinite(batch["log_probs"][0, :3]).all()
                )


if __name__ == "__main__":
    unittest.main()
