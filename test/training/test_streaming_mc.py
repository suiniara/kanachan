"""Canonical streaming MC order, bounded queues and exact recovery of prefetch."""

import importlib.util
import os
import random
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch
from test_bc_feature_prefetch import assert_state_equal
from test_candidate_features import training_line

from kanachan.candidate_features import derive_candidate_features
from kanachan.training.core.offline_rl import EpisodeReplayBuffer
from kanachan.training.core.offline_rl.streaming_mc import StreamingMCPrefetch
from kanachan.training.core.rl.rewards import configured_reward
from kanachan.training.online.state import capture_rng

ROOT = Path(__file__).resolve().parents[2]


def arguments(path, cursor=0):
    return {
        "training_data": path,
        "contiguous_training_data": True,
        "num_skip_samples": cursor,
        "rewrite_rooms": None,
        "rewrite_grades": None,
        "get_reward": configured_reward("placement"),
        "discount_factor": 1.0,
        "dtype": torch.float32,
        "max_size": 16,
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": False,
        "ranks": [0],
        "main_rank": 0,
        "reward_identity": "placement-test-v1",
    }


@unittest.skipUnless(
    importlib.util.find_spec("kanachan._candidate_features"),
    "Build native features",
)
class StreamingMCTests(unittest.TestCase):
    def assert_batch(self, first, second):
        for key in first.keys(include_nested=True, leaves_only=True):
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_canonical_order_tail_and_isolated_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.tsv"
            path.write_text("".join(training_line(i) for i in range(67)))
            random.seed(71)
            expected = list(EpisodeReplayBuffer(**arguments(path)))
            rng = capture_rng()
            with StreamingMCPrefetch(
                arguments(path), seed=71, batches=2
            ) as stream:
                actual = list(stream.iter_with_last())
                self.assertLessEqual(len(stream.ready), 2)
            assert_state_equal(self, rng, capture_rng())
            self.assertEqual(sum(len(batch) for batch, _ in actual), 67)
            self.assertEqual(
                [last for _, last in actual], [False] * 8 + [True]
            )
            for reference, (batch, _) in zip(expected, actual, strict=True):
                self.assert_batch(reference, batch)
                self.assertTrue(
                    torch.equal(
                        batch["candidate_afterstate"],
                        derive_candidate_features(
                            batch["sparse"],
                            batch["progression"],
                            batch["candidates"],
                        ),
                    )
                )

    def test_full_queue_checkpoint_keeps_replay_ready_and_lookahead(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.tsv"
            path.write_text("".join(training_line(i) for i in range(131)))
            with StreamingMCPrefetch(
                arguments(path), seed=71, batches=2
            ) as stream:
                iterator = stream.iter_with_last()
                next(iterator)
                deadline = time.monotonic() + 10
                while len(stream.ready) < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(len(stream.ready), 2)
                state = stream.state_dict()
                self.assertIsNotNone(state["pending"])
                self.assertNotIn("replay", state["worker"])
                torch.save(state, Path(temporary) / "state.pth")
                expected = list(iterator)
            state = torch.load(
                Path(temporary) / "state.pth", weights_only=True
            )
            with StreamingMCPrefetch(
                arguments(path, state["input_samples"]), seed=71, batches=4
            ) as stream:
                stream.load_state_dict(state)
                actual = list(stream.iter_with_last())
            self.assertEqual(len(expected), len(actual))
            for (first, last), (second, resumed_last) in zip(
                expected, actual, strict=True
            ):
                self.assert_batch(first, second)
                self.assertEqual(last, resumed_last)

    def test_worker_failure_is_reported_and_reader_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.tsv"
            path.write_text("invalid row\n")
            stream = StreamingMCPrefetch(arguments(path), seed=71, batches=1)
            with (
                stream,
                self.assertRaisesRegex(RuntimeError, "MC reader failed"),
            ):
                next(stream)
            self.assertFalse(stream.process.is_alive())

    def test_real_cql_updates_and_resume_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "data.tsv"
            path.write_text("".join(training_line(i) for i in range(67)))
            seed = "import random,numpy,torch,runpy; random.seed(71); numpy.random.seed(71); torch.set_num_threads(2); torch.manual_seed(71); runpy.run_module('kanachan.training.cql.train',run_name='__main__')"
            env = dict(
                os.environ,
                PYTHONPATH=str(ROOT),
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                TORCHDYNAMO_DISABLE="1",
                CUDA_VISIBLE_DEVICES="",
            )

            def run(name, streaming=False, resume=False):
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
                    f"training_data={path}",
                    f"hydra.run.dir={root / name}",
                ]
                if streaming:
                    args += [
                        "streaming_mc=true",
                        "streaming_mc_seed=71",
                        "streaming_mc_batches=2",
                    ]
                if resume:
                    args += [
                        f"initial_model_prefix={root}/stream/snapshots",
                        "initial_model_index=32",
                    ]
                result = subprocess.run(
                    args,
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=180,
                )
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )

            run("baseline")
            run("stream", True)
            run("resumed", True, True)
            for filename in ("encoder.pth", "decoder.pth", "optimizer.pth"):
                baseline = torch.load(
                    root / "baseline/snapshots" / filename, weights_only=True
                )
                full = torch.load(
                    root / "stream/snapshots" / filename, weights_only=True
                )
                resumed = torch.load(
                    root / "resumed/snapshots" / filename, weights_only=True
                )
                assert_state_equal(self, baseline, full)
                assert_state_equal(self, full, resumed)
            assert_state_equal(
                self,
                torch.load(
                    root / "stream/snapshots/decision-rng.pth",
                    weights_only=True,
                ),
                torch.load(
                    root / "resumed/snapshots/decision-rng.pth",
                    weights_only=True,
                ),
            )
            state = torch.load(
                root / "resumed/snapshots/replay-state.pth", weights_only=True
            )
            self.assertEqual(state["format_version"], 4)
            self.assertLess(
                (root / "resumed/snapshots/replay-state.pth").stat().st_size,
                200000,
            )
            with (
                StreamingMCPrefetch(
                    arguments(path, state["input_samples"]), seed=72
                ) as changed,
                self.assertRaisesRegex(ValueError, "source or recipe"),
            ):
                changed.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
