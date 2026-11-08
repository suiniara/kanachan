"""Canonical MC labels, prepared-cache updates and small recovery state."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_bc_feature_prefetch import assert_state_equal
from test_candidate_features import observation, training_line

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    importlib.util.find_spec("kanachan._candidate_features"),
    "Build native features",
)
class PreparedMCTests(unittest.TestCase):
    def test_ram_budget_and_recovery_identity_are_enforced(self):
        from kanachan.training.core.offline_rl.prepared_mc import (
            PreparedMCCache,
        )
        from kanachan.training.online.state import capture_rng

        class Source:
            def __iter__(self):
                data = observation()
                data["mc_return"] = torch.zeros(1)
                yield data

            def state_dict(self):
                return {
                    "source": "test-observation",
                    "reward_identity": "test-reward",
                    "discount_factor": 1.0,
                    "batch_size": 1,
                    "max_size": 1,
                }

        before = capture_rng()
        with self.assertRaisesRegex(MemoryError, "RAM budget"):
            PreparedMCCache(Source, workers=1, max_bytes=1, seed=71)
        assert_state_equal(self, before, capture_rng())
        with PreparedMCCache(
            Source, workers=1, max_bytes=100000, seed=71, epochs=2
        ) as cache:
            self.assertLessEqual(cache.bytes, 100000)
            state = cache.state_dict()
            changed = dict(state, prepared_mc={"cache_sha256": "changed"})
            with self.assertRaisesRegex(ValueError, "regenerated cache"):
                cache.load_state_dict(changed)
            for cursor in (-1, 3, 0.5, True):
                with self.assertRaisesRegex(ValueError, "batch position"):
                    cache.load_state_dict(dict(state, batch_position=cursor))
            cache.load_state_dict(dict(state, batch_position=1))
            self.assertEqual(cache.position, 1)
        assert_state_equal(self, before, capture_rng())

    def test_q_only_uses_prepared_features_without_gpu_cpu_recomputation(self):
        from kanachan.candidate_features import derive_candidate_features
        from kanachan.nn.decision_decoder import GlobalActionHead
        from kanachan.nn.decision_value import DecisionQNetwork
        from kanachan.training.core.rl.discrete_cql import action_values

        class Body(torch.nn.Module):
            def forward_with_validity(
                self, sparse, numeric, progression, candidates
            ):
                return torch.ones((1, 8)), candidates < 546

        data = observation()
        data["candidate_afterstate"] = derive_candidate_features(
            data["sparse"], data["progression"], data["candidates"]
        )
        head = GlobalActionHead(
            input_dimension=8,
            hidden_dimension=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            candidate_features="discard_v1",
        )
        with patch(
            "kanachan.candidate_features.derive_candidate_features",
            side_effect=AssertionError("Prepared features were discarded"),
        ):
            values = action_values(DecisionQNetwork(Body(), head), data)
        self.assertTrue(torch.isfinite(values[0, :3]).all())

    def test_canonical_updates_multi_epoch_resume_and_small_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data.tsv"
            data.write_text("".join(training_line(i) for i in range(64)))
            seed = "import random,numpy,torch,runpy; random.seed(71); numpy.random.seed(71); torch.set_num_threads(2); torch.manual_seed(71); runpy.run_module('kanachan.training.cql.train',run_name='__main__')"
            env = dict(
                os.environ,
                PYTHONPATH=str(ROOT),
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                TORCHDYNAMO_DISABLE="1",
            )

            def run(name, workers, epochs=1, resume=False):
                args = [
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
                    "replay_buffer_size=16",
                    "reward_scheme=placement",
                    "snapshot_interval=32",
                    f"candidate_feature_workers={workers}",
                    f"prepared_mc_epochs={epochs}",
                    "prepared_mc_seed=71",
                    f"training_data={data}",
                    f"hydra.run.dir={root / name}",
                ]
                if resume:
                    args.extend(
                        [
                            f"initial_model_prefix={root}/double/snapshots",
                            "initial_model_index=64",
                        ]
                    )
                result = subprocess.run(
                    args,
                    check=False,
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )

            run("baseline", 0)
            run("prepared", 2)
            for filename in (
                "encoder.pth",
                "decoder.pth",
                "optimizer.pth",
            ):
                assert_state_equal(
                    self,
                    torch.load(
                        root / "baseline/snapshots" / filename,
                        weights_only=True,
                    ),
                    torch.load(
                        root / "prepared/snapshots" / filename,
                        weights_only=True,
                    ),
                )
            run("double", 2, 2)
            run("resumed", 2, 2, True)
            for filename in (
                "encoder.pth",
                "decoder.pth",
                "optimizer.pth",
                "decision-rng.pth",
            ):
                assert_state_equal(
                    self,
                    torch.load(
                        root / "double/snapshots" / filename, weights_only=True
                    ),
                    torch.load(
                        root / "resumed/snapshots" / filename,
                        weights_only=True,
                    ),
                )
            state = torch.load(
                root / "resumed/snapshots/replay-state.pth", weights_only=True
            )
            self.assertEqual(state["format_version"], 3)
            self.assertNotIn("batches", state)
            self.assertLess(
                (root / "resumed/snapshots/replay-state.pth").stat().st_size,
                10000,
            )


if __name__ == "__main__":
    unittest.main()
