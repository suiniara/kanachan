import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from test_training_regressions import bc_line
from torch import distributed

from kanachan.training.core.checkpoint import load_training_state


@unittest.skipUnless(
    distributed.is_gloo_available()
    and os.environ.get("KANACHAN_TEST_DISTRIBUTED") == "1",
    "Set KANACHAN_TEST_DISTRIBUTED=1 for real distributed training",
)
class DistributedTrainingTests(unittest.TestCase):
    def run_distributed(self, module, data, output, *extra):
        environment = dict(
            os.environ,
            TORCHDYNAMO_DISABLE="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            HYDRA_FULL_ERROR="1",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc-per-node=2",
                "--module",
                f"kanachan.training.{module}.train",
                "device=cpu",
                "encoder=bert_tiny",
                "encoder.dimension=8",
                "encoder.num_heads=2",
                "encoder.num_layers=1",
                "encoder.dropout=0.0",
                "decoder=double",
                "optimizer=adam",
                "batch_size=4",
                "num_workers=0",
                f"training_data={data.as_posix()}",
                f"hydra.run.dir={output.as_posix()}",
                *extra,
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_two_rank_training_and_zero_checkpoint_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.tsv"
            data.write_text(bc_line(0) * 16)
            eor_data = root / "eor.tsv"
            eor_data.write_text(
                "game\t3,6,7,23,39,55,71,74\t0,0,25000,25000,25000,25000\t100,100,200,50\n"
                * 16
            )
            reward = root / "reward.py"
            reward.write_text(
                "def get_reward(data, contiguous):\n    data['next', 'reward'] = data['next', 'end_of_game'].float()\n"
            )
            cases = [
                (
                    "bc",
                    "bc",
                    ["optimizer.use_zero=true", "snapshot_interval=8"],
                ),
                ("iql", "iql", ["expectile=0.9", "optimizer.use_zero=true"]),
                ("ilql", "ilql", ["expectile=0.9", "optimizer.use_zero=true"]),
                (
                    "cql_scalar",
                    "cql",
                    ["num_qr_intervals=null", "optimizer.use_zero=true"],
                ),
                (
                    "cql_quantile",
                    "cql",
                    [
                        "num_qr_intervals=2",
                        "encoder.num_layers=2",
                        "checkpointing=true",
                    ],
                ),
                (
                    "awr",
                    "awr",
                    [
                        f"q_model={(root / 'cql_scalar/snapshots/model.kanachan').as_posix()}",
                        "optimizer.use_zero=true",
                    ],
                ),
                ("ranking", "ranking", []),
                (
                    "eor2ranking",
                    "eor2ranking",
                    [
                        "encoder.layer_normalization=true",
                        "encoder.num_layers=2",
                        "checkpointing=true",
                    ],
                ),
            ]
            for name, module, arguments in cases:
                with self.subTest(module=name):
                    if module in ("iql", "ilql", "cql"):
                        arguments = [
                            *arguments,
                            f"reward_plugin={reward.as_posix()}",
                        ]
                    output = root / name
                    logs = self.run_distributed(
                        module,
                        eor_data if module == "eor2ranking" else data,
                        output,
                        *arguments,
                    )
                    self.assertIn(
                        "Completed in this run: 2 optimizer steps", logs
                    )
                    state = load_training_state(output / "snapshots", "", 0)
                    self.assertEqual(state.num_samples, 16)
                    self.assertEqual(state.optimizer_steps, 2)
                    if name == "bc":
                        optimizer_state = torch.load(
                            output / "snapshots/optimizer.pth",
                            map_location="cpu",
                            weights_only=True,
                        )
                        self.assertEqual(
                            len(optimizer_state["state"]),
                            len(optimizer_state["param_groups"][0]["params"]),
                        )
                        self.assertTrue(
                            (output / "snapshots/encoder.8.pth").is_file()
                        )
            data.write_text(bc_line(0) * 24)
            legacy_path = root / "bc/snapshots/optimizer.pth"
            legacy = torch.load(
                legacy_path, map_location="cpu", weights_only=True
            )
            for group in legacy["param_groups"]:
                group["step"] = 2
                group["bias_correction"] = True
                for parameter in group["params"]:
                    legacy["state"][parameter].pop("step")
            torch.save(legacy, legacy_path)
            eor_data.write_text(
                "game\t3,6,7,23,39,55,71,74\t0,0,25000,25000,25000,25000\t100,100,200,50\n"
                * 24
            )
            for name, module, arguments in cases:
                with self.subTest(resume=name):
                    if module in ("iql", "ilql", "cql"):
                        arguments = [
                            *arguments,
                            f"reward_plugin={reward.as_posix()}",
                        ]
                    output = root / f"{name}-resumed"
                    self.run_distributed(
                        module,
                        eor_data if module == "eor2ranking" else data,
                        output,
                        *arguments,
                        f"initial_model_prefix={(root / name / 'snapshots').as_posix()}",
                    )
                    state = load_training_state(output / "snapshots", "", 0)
                    self.assertEqual(
                        (state.num_samples, state.optimizer_steps), (24, 3)
                    )
