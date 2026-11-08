import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from test_training_regressions import bc_line

from kanachan.training.core.checkpoint import load_training_state


@unittest.skipUnless(
    os.environ.get("KANACHAN_TEST_COMPILE") == "1",
    "Set KANACHAN_TEST_COMPILE=1 for TorchInductor training",
)
class CompiledTrainingTests(unittest.TestCase):
    def test_compiled_bc_and_quantile_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.tsv"
            data.write_text(bc_line(0) * 8)
            reward = root / "reward.py"
            reward.write_text(
                "def get_reward(data, contiguous):\n    data['next', 'reward'] = data['next', 'end_of_game'].float()\n"
            )
            environment = dict(
                os.environ,
                TORCHDYNAMO_DISABLE="0",
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                HYDRA_FULL_ERROR="1",
            )
            for module in ("bc", "cql"):
                with self.subTest(module=module):
                    output = root / module
                    extra = (
                        ["checkpointing=true", "encoder.num_layers=2"]
                        if module == "bc"
                        else [
                            "num_qr_intervals=2",
                            "dueling_network=true",
                            "checkpointing=true",
                            "encoder.num_layers=2",
                            f"reward_plugin={reward.as_posix()}",
                        ]
                    )
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            "-m",
                            f"kanachan.training.{module}.train",
                            f"device={'cuda' if torch.cuda.is_available() else 'cpu'}",
                            "num_workers=0",
                            "encoder=bert_tiny",
                            "encoder.dimension=8",
                            "encoder.num_heads=2",
                            "encoder.num_layers=1",
                            "encoder.dropout=0.0",
                            "encoder.layer_normalization=true",
                            "decoder=double",
                            "decoder.layer_normalization=true",
                            "optimizer=adam",
                            "batch_size=4",
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
                        timeout=360,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    state = load_training_state(output / "snapshots", "", 0)
                    self.assertEqual(state.num_samples, 8)
                    self.assertGreater(state.optimizer_steps, 0)
                    self.assertLessEqual(state.optimizer_steps, 2)
                    if state.optimizer_steps < 2:
                        self.assertIn(
                            "Skip an optimization step",
                            result.stdout + result.stderr,
                        )
