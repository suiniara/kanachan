import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch import distributed
from tensordict import TensorDict

from kanachan.training.core.awr_diagnostics import weight_statistics


def data(count):
    sparse = torch.full((count, 29), 510, dtype=torch.int32)
    sparse[:, :10] = torch.tensor([3, 6, 7, 23, 39, 55, 71, 75, 78, 142])
    candidates = torch.full((count, 32), 546, dtype=torch.int32)
    candidates[:, 0] = 4
    progression = torch.full((count, 109), 2165, dtype=torch.int32)
    progression[:, :2] = torch.tensor([0, 5 + 148 + 4 + 1])
    return TensorDict(
        {
            "sparse": sparse,
            "numeric": torch.tensor(
                [[0, 0, 35000, 25000, 20000, 20000]] * count, dtype=torch.int32
            ),
            "progression": progression,
            "candidates": candidates,
            "action": torch.zeros(count, dtype=torch.int32),
        },
        [count],
    )


def _worker(root, rank):
    torch.set_num_threads(1)
    distributed.init_process_group(
        "gloo",
        init_method=(root / "store").as_uri(),
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        # Unequal local sizes also rule out averaging rank-local fractions.
        values = (
            torch.tensor([1.0, 1.0]) if rank == 0 else torch.tensor([100.0])
        )
        advantage = (
            torch.tensor([0.0, 0.0])
            if rank == 0
            else torch.tensor([math.log(200.0)])
        )
        metrics = weight_statistics(
            advantage, values, data(len(values)), beta=1.0, max_weight=100.0
        )
        (root / f"rank-{rank}.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
    finally:
        distributed.destroy_process_group()


class AWRDiagnosticTests(unittest.TestCase):
    def test_concentration_and_clipping_match_direct_global_formula(self):
        weights = torch.tensor([1.0, 1.0, 100.0], requires_grad=True)
        advantage = torch.tensor(
            [0.0, 0.0, math.log(200.0)], requires_grad=True
        )
        rng = torch.random.get_rng_state().clone()
        metrics = weight_statistics(
            advantage, weights, data(3), beta=1.0, max_weight=100.0
        )
        self.assertAlmostEqual(metrics["Weight/ESS"], 102**2 / 10002)
        self.assertAlmostEqual(metrics["Weight/Clipped fraction"], 1 / 3)
        self.assertAlmostEqual(
            metrics["Weight/Largest sample share"], 100 / 102
        )
        self.assertEqual(metrics["Action/discard/Weight share"], 1)
        self.assertEqual(metrics["Current rank/first/Weight share"], 1)
        self.assertEqual(metrics["Opponent riichi/1/Weight share"], 1)
        self.assertIsNone(weights.grad)
        self.assertIsNone(advantage.grad)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_zero_weights_and_nonfinite_advantages_are_reported(self):
        metrics = weight_statistics(
            torch.tensor([float("inf"), -float("inf"), float("nan")]),
            torch.zeros(3),
            data(3),
            beta=1.0,
            max_weight=20.0,
        )
        self.assertEqual(metrics["Weight/ESS"], 0)
        self.assertEqual(metrics["Weight/Zero fraction"], 1)
        self.assertEqual(metrics["Advantage/Nonfinite fraction"], 1)
        self.assertTrue(
            all(math.isfinite(value) for value in metrics.values())
        )
        with self.assertRaises(ValueError):
            weight_statistics(
                torch.ones(3),
                torch.tensor([1.0, -1.0, 1.0]),
                data(3),
                beta=1.0,
                max_weight=20.0,
            )

    def test_action_groups_use_logged_global_ids(self):
        observations = data(11)
        observations["candidates"][:, 0] = torch.tensor(
            [4, 5, 148, 182, 219, 220, 221, 222, 312, 432, 543]
        )
        metrics = weight_statistics(
            torch.zeros(11),
            torch.arange(1, 12).float(),
            observations,
            beta=1.0,
            max_weight=20.0,
        )
        from kanachan.training.core.awr_diagnostics import ACTION_GROUPS

        for i, name in enumerate(ACTION_GROUPS):
            self.assertAlmostEqual(
                metrics[f"Action/{name}/Weight share"], (i + 1) / 66
            )
            self.assertAlmostEqual(
                metrics[f"Action/{name}/Sample fraction"], 1 / 11
            )

    @unittest.skipUnless(
        distributed.is_gloo_available()
        and os.environ.get("KANACHAN_TEST_DISTRIBUTED") == "1",
        "Enable KANACHAN_TEST_DISTRIBUTED=1 for Gloo",
    )
    def test_two_rank_global_ess_is_not_average_local_ess(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processes = []
            try:
                for rank in range(2):
                    environment = dict(
                        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
                    )
                    environment["PYTHONPATH"] = os.pathsep.join(
                        (
                            str(Path(__file__).resolve().parents[2]),
                            str(Path(__file__).resolve().parent),
                            environment.get("PYTHONPATH", ""),
                        )
                    )
                    processes.append(
                        subprocess.Popen(
                            [
                                sys.executable,
                                "-B",
                                str(Path(__file__).resolve()),
                                "--worker",
                                str(root),
                                str(rank),
                            ],
                            cwd=Path(__file__).resolve().parents[2],
                            env=environment,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                        )
                    )
                for process in processes:
                    stdout, stderr = process.communicate(timeout=75)
                    self.assertEqual(process.returncode, 0, stdout + stderr)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                        process.communicate(timeout=10)
            values = [
                json.loads((root / f"rank-{rank}.json").read_text())
                for rank in range(2)
            ]
            self.assertEqual(values[0], values[1])
            self.assertEqual(values[0]["Samples"], 3)
            self.assertAlmostEqual(values[0]["Weight/ESS"], 102**2 / 10002)
            self.assertNotAlmostEqual(values[0]["Weight/ESS"], 1.5)
            self.assertAlmostEqual(values[0]["Weight/Clipped fraction"], 1 / 3)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        unittest.main()
