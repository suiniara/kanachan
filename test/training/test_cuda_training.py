import math
import tempfile
import unittest
from pathlib import Path

import test_training_recovery as recovery
import torch
from test_training_regressions import bc_line

from kanachan.model_loader import load_model
from kanachan.training.core.bc import DataLoader
from kanachan.training.core.checkpoint import load_training_state


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class CudaTrainingTests(unittest.TestCase):
    def run_training(self, module, data, output, *extra):
        return recovery.TrainingRecoveryTests.run_training(
            self, module, data, output, "device=cuda", "num_workers=0", *extra
        )

    def test_cuda_amp_training_exports_and_resumption(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.tsv"
            data.write_text(bc_line(0) * 128)
            eor_data = root / "eor.tsv"
            eor_data.write_text(
                "game\t3,6,7,23,39,55,71,74\t0,0,25000,25000,25000,25000\t100,100,200,50\n"
                * 128
            )
            reward = root / "reward.py"
            reward.write_text(
                "def get_reward(data, contiguous):\n    data['next', 'reward'] = data['next', 'end_of_game'].float()\n"
            )
            cases = [
                (
                    "bc",
                    "bc",
                    [
                        "encoder.num_layers=2",
                        "encoder.layer_normalization=true",
                        "checkpointing=true",
                        "optimizer.learning_rate=0.003",
                        "snapshot_interval=64",
                    ],
                ),
                ("iql", "iql", ["expectile=0.9", "target_update_interval=2"]),
                (
                    "ilql",
                    "ilql",
                    ["expectile=0.9", "target_update_interval=2"],
                ),
                ("cql", "cql", ["num_qr_intervals=null"]),
                (
                    "cql_quantile",
                    "cql",
                    [
                        "num_qr_intervals=2",
                        "dueling_network=true",
                        "encoder.num_layers=2",
                        "encoder.layer_normalization=true",
                        "decoder.layer_normalization=true",
                        "checkpointing=true",
                    ],
                ),
                (
                    "awr",
                    "awr",
                    [
                        f"q_model={(root / 'cql/snapshots/model.kanachan').as_posix()}"
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
                (
                    "bc_sgd",
                    "bc",
                    ["optimizer=sgd", "optimizer.learning_rate=0.01"],
                ),
                ("bc_radam", "bc", ["optimizer=radam"]),
                ("bc_lamb", "bc", ["optimizer=lamb"]),
            ]
            for name, module, arguments in cases:
                with self.subTest(name=name):
                    training_data = (
                        eor_data if module == "eor2ranking" else data
                    )
                    if module in ("iql", "ilql", "cql"):
                        arguments = [
                            *arguments,
                            f"reward_plugin={reward.as_posix()}",
                        ]
                    output = root / name
                    logs = self.run_training(
                        module, training_data, output, *arguments
                    )
                    self.assertIn("Device: cuda:0", logs)
                    state = load_training_state(output / "snapshots", "", 0)
                    self.assertEqual(state.num_samples, 128)
                    self.assertGreater(state.optimizer_steps, 0)
                    self.assertIsNotNone(state.grad_scaler_state)
                    if name == "bc":
                        periodic = load_training_state(
                            output / "snapshots", ".64", 0
                        )
                        self.assertIsNotNone(periodic.grad_scaler_state)
                        self.assertEqual(periodic.num_samples, 64)
                    model = load_model(
                        output / "snapshots/model.kanachan",
                        map_location=torch.device("cuda"),
                    )
                    model.eval()
                    if module != "eor2ranking":
                        batch = next(
                            DataLoader(
                                path=data,
                                num_skip_samples=0,
                                rewrite_rooms=None,
                                rewrite_grades=None,
                                batch_size=4,
                            )
                        ).to("cuda")
                        with torch.no_grad():
                            model(batch)
                        if module == "bc" and name == "bc":
                            self.assertLess(
                                -batch["log_probs"][:, 0].mean().item(),
                                math.log(4),
                            )
                        if "ranking_probs" in batch:
                            self.assertTrue(
                                batch["ranking_probs"].isfinite().all().item()
                            )
                        elif "action_value" in batch:
                            self.assertTrue(
                                batch["action_value"][:, :4]
                                .isfinite()
                                .all()
                                .item()
                            )

            data.write_text(bc_line(0) * 160)
            for module in ("iql", "ilql"):
                with self.subTest(resume=module):
                    previous = load_training_state(
                        root / module / "snapshots", "", 0
                    )
                    output = root / f"{module}-resumed"
                    logs = self.run_training(
                        module,
                        data,
                        output,
                        "expectile=0.9",
                        "target_update_interval=2",
                        f"reward_plugin={reward.as_posix()}",
                        f"initial_model_prefix={(root / module / 'snapshots').as_posix()}",
                    )
                    self.assertIn(
                        f"Resuming at sample 128 after {previous.optimizer_steps} optimizer steps.",
                        logs,
                    )
                    state = load_training_state(output / "snapshots", "", 0)
                    self.assertEqual(state.num_samples, 160)
                    self.assertGreater(
                        state.optimizer_steps, previous.optimizer_steps
                    )
                    self.assertIsNotNone(state.grad_scaler_state)
