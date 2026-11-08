import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from test_training_pipeline import rl_line
from test_training_regressions import bc_line

from kanachan.model_loader import load_model
from kanachan.training.core.bc import DataLoader as BCDataLoader
from kanachan.training.core.checkpoint import (
    TrainingState,
    load_training_state,
    save_training_state,
)
from kanachan.training.core.eor import DataLoader as EORDataLoader


class TrainingRecoveryTests(unittest.TestCase):
    def run_training(self, module, data, output, *extra):
        environment = dict(
            os.environ,
            WORLD_SIZE="1",
            RANK="0",
            LOCAL_RANK="0",
            TORCHDYNAMO_DISABLE="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            HYDRA_FULL_ERROR="1",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
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
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_training_state_validation_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            expected = TrainingState(128, 3)
            save_training_state(prefix, "", expected)
            self.assertEqual(load_training_state(prefix, "", 0), expected)
            self.assertEqual(
                load_training_state(prefix, ".64", 64), TrainingState(64)
            )
            (prefix / "training-state.json").write_text(
                '{"version": 1, "num_samples": -1}'
            )
            with self.assertRaisesRegex(
                RuntimeError, "Invalid training state"
            ):
                load_training_state(prefix, "", 0)

            save_training_state(prefix, "", TrainingState(12, 3, False))
            with self.assertRaisesRegex(
                RuntimeError, "no exact input position"
            ):
                load_training_state(prefix, "", 0)
            self.assertEqual(
                load_training_state(prefix, "", 0, restart_data=True),
                TrainingState(0),
            )

            self.assertEqual(
                load_training_state(prefix, "", 0, start_new_epoch=True),
                TrainingState(0, 3),
            )
            scaler = torch.amp.GradScaler("cpu", init_scale=512).state_dict()
            state = TrainingState(12, 3, True, scaler)
            save_training_state(prefix, "", state)
            self.assertEqual(load_training_state(prefix, "", 0), state)
            self.assertEqual(
                load_training_state(prefix, "", 0, restart_data=True),
                TrainingState(0),
            )
            self.assertEqual(
                load_training_state(prefix, ".64", 64, restart_data=True),
                TrainingState(0),
            )

    def test_accumulation_checkpoints_flush_tail_and_resume_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "bc.tsv"
            data.write_text("".join(bc_line(i % 4) for i in range(17)))
            first = root / "first"
            self.run_training(
                "bc",
                data,
                first,
                "gradient_accumulation_steps=2",
                "snapshot_interval=1",
            )
            snapshots = first / "snapshots"
            self.assertEqual(
                load_training_state(snapshots, "", 0), TrainingState(17, 3)
            )
            self.assertEqual(
                sorted(path.name for path in snapshots.glob("encoder.*.pth")),
                ["encoder.16.pth", "encoder.17.pth", "encoder.8.pth"],
            )
            data.write_text("".join(bc_line(i % 4) for i in range(25)))
            second = root / "second"
            logs = self.run_training(
                "bc",
                data,
                second,
                "gradient_accumulation_steps=2",
                f"initial_model_prefix={snapshots.as_posix()}",
            )
            self.assertIn(
                "Resuming at sample 17 after 3 optimizer steps.", logs
            )
            self.assertIn("Completed in this run: 1 optimizer steps", logs)
            self.assertEqual(
                load_training_state(second / "snapshots", "", 0),
                TrainingState(25, 4),
            )
            third = root / "third"
            logs = self.run_training(
                "bc",
                data,
                third,
                "gradient_accumulation_steps=2",
                "start_new_epoch=true",
                f"initial_model_prefix={(second / 'snapshots').as_posix()}",
            )
            self.assertIn(
                "Starting a new data epoch after 4 optimizer steps.", logs
            )
            self.assertIn("Completed in this run: 4 optimizer steps", logs)
            self.assertEqual(
                load_training_state(third / "snapshots", "", 0),
                TrainingState(25, 8),
            )

    def test_rl_resume_with_and_without_lr_schedulers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reward = root / "reward.py"
            reward.write_text(
                "def get_reward(data, contiguous):\n    data['next', 'reward'] = data['next', 'end_of_game'].float()\n"
            )
            for module in ("iql", "ilql"):
                for warmup in (0, 2):
                    with self.subTest(module=module, warmup=warmup):
                        data = root / f"{module}-{warmup}.tsv"
                        data.write_text(
                            "".join(
                                rl_line(i, terminal=True) for i in range(17)
                            )
                        )
                        first = root / f"{module}-{warmup}-first"
                        arguments = [
                            "expectile=0.9",
                            "target_update_interval=2",
                            f"reward_plugin={reward.as_posix()}",
                            f"optimizer.warmup_steps={warmup}",
                        ]
                        self.run_training(module, data, first, *arguments)
                        snapshots = first / "snapshots"
                        if module == "iql" and warmup:
                            # Backward compatibility with snapshots written before the filename fix.
                            for prefix in ("q1", "q2"):
                                (
                                    snapshots / f"{prefix}-lr-scheduler.pth"
                                ).rename(
                                    snapshots / f"{prefix}_lr_scheduler.pth"
                                )
                        self.assertEqual(
                            load_training_state(snapshots, "", 0),
                            TrainingState(17, 5),
                        )
                        data.write_text(
                            "".join(
                                rl_line(i, terminal=True) for i in range(21)
                            )
                        )
                        second = root / f"{module}-{warmup}-second"
                        logs = self.run_training(
                            module,
                            data,
                            second,
                            *arguments,
                            f"initial_model_prefix={snapshots.as_posix()}",
                        )
                        self.assertIn(
                            "Resuming at sample 17 after 5 optimizer steps.",
                            logs,
                        )
                        self.assertIn(
                            "Completed in this run: 1 optimizer steps, 1 target updates.",
                            logs,
                        )
                        self.assertEqual(
                            load_training_state(second / "snapshots", "", 0),
                            TrainingState(21, 6),
                        )

    def test_ranking_entry_points_export_usable_models(self):
        self.check_ranking_entry_points_export_usable_models("transformer")

    def test_modern_ranking_entry_points_export_usable_models(self):
        self.check_ranking_entry_points_export_usable_models("rmsnorm_swiglu")

    def check_ranking_entry_points_export_usable_models(self, architecture):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for module, loader in (
                ("ranking", BCDataLoader),
                ("eor2ranking", EORDataLoader),
            ):
                with self.subTest(module=module):
                    data = root / f"{module}.tsv"
                    if module == "ranking":
                        text = "".join(bc_line(i % 4) for i in range(17))
                    else:
                        text = (
                            "game\t3,6,7,23,39,55,71,74\t0,0,25000,25000,25000,25000\t100,100,200,50\n"
                            * 17
                        )
                    data.write_text(text)
                    output = root / module
                    logs = self.run_training(
                        module, data, output,
                        f"encoder.architecture={architecture}",
                    )
                    self.assertIn(
                        "Completed in this run: 5 optimizer steps", logs
                    )
                    model = load_model(
                        output / "snapshots/model.kanachan",
                        map_location=torch.device("cpu"),
                    )
                    model.eval()
                    batch = next(
                        loader(
                            path=data,
                            num_skip_samples=0,
                            rewrite_rooms=None,
                            rewrite_grades=None,
                            batch_size=4,
                        )
                    )
                    with torch.no_grad():
                        model(batch)
                    probabilities = batch["ranking_probs"]
                    self.assertEqual(tuple(probabilities.shape), (4, 4, 4))
                    self.assertTrue(probabilities.isfinite().all().item())
                    torch.testing.assert_close(
                        probabilities.sum(2), torch.ones(4, 4)
                    )
                    transferred = root / f"{module}-transferred"
                    self.run_training(
                        module,
                        data,
                        transferred,
                        "optimizer.learning_rate=1e-8",
                        f"encoder.architecture={architecture}",
                        f"encoder.load_from={(output / 'snapshots/encoder.pth').as_posix()}",
                        f"decoder.load_from={(output / 'snapshots/decoder.pth').as_posix()}",
                    )
                    initial_decoder = torch.load(
                        output / "snapshots/decoder.pth", weights_only=True
                    )
                    transferred_decoder = torch.load(
                        transferred / "snapshots/decoder.pth",
                        weights_only=True,
                    )
                    for key in initial_decoder:
                        torch.testing.assert_close(
                            transferred_decoder[key],
                            initial_decoder[key],
                            rtol=0,
                            atol=1e-6,
                        )


if __name__ == "__main__":
    unittest.main()
