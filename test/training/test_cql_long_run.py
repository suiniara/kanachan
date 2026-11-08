import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from kanachan.model_loader import load_model
from kanachan.nn.decision_value import q_only
from kanachan.training.cql import train as cql
from kanachan.training.core.offline_rl import EpisodeReplayBuffer
from kanachan.training.core.rl.rewards import configured_reward
import test_decision_training as decision_training
from test_training_pipeline import rl_line
from test_training_regressions import bc_line


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from train_decision_bc import sha256
from train_decision_cql import equal, run, training_counters


class CqlLongRunTests(unittest.TestCase):
    cli = decision_training.DecisionTrainingTests.cli

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_epoch_counters_include_each_short_final_batch(self):
        self.assertEqual(training_counters(2, 5, 5, 4), dict(
            epoch=2, cursor=5, total_samples=15, completed_epochs=3,
            attempted_updates=6,
        ))
        self.assertEqual(training_counters(2, 4, 5, 4)["attempted_updates"], 5)
        for arguments in ((-1, 0, 5, 4), (0, 6, 5, 4), (0, 0, 0, 4)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                training_counters(*arguments)

    def fixture(self, root):
        bc_data, rl_data = root / "bc.tsv", root / "rl.tsv"
        bc_data.write_text("".join(bc_line(i % 4) for i in range(12)), encoding="utf-8")
        rl_data.write_text("".join(rl_line(i, terminal=i % 4 == 3) for i in range(12)), encoding="utf-8")
        self.cli("bc", bc_data, root / "bc")
        teacher = root / "bc/snapshots/model.kanachan"
        self.cli("cql", rl_data, root / "template", "replay_buffer_size=8",
                 f"bc_initial_model={teacher.as_posix()}")
        template = root / "template/snapshots/model.kanachan"
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"complete_candidate": {
            "kanachan/training/cql/train.py": sha256(ROOT / "kanachan/training/cql/train.py"),
        }}), encoding="utf-8")
        functional = root / "functional-fixture.json"
        functional.write_text(json.dumps(dict(
            status="complete", scope="unit_test_fixture_only",
            source_manifest_sha256=sha256(manifest), target_kind="monte_carlo_return",
            feature_freezing=False, logged_return_mixing=False, learned_reward_predictor=False,
            cuda_cql=dict(exact_resume_all_state=True),
        )), encoding="utf-8")
        identity = lambda path: dict(path=str(path), sha256=sha256(path))
        return dict(
            source=str(ROOT), manifest=str(manifest), manifest_sha256=sha256(manifest),
            controller_sha256=sha256(ROOT / "tools/train_decision_cql.py"),
            utility_sha256=sha256(ROOT / "tools/train_decision_bc.py"), device="cpu", seed=1234,
            parameters=sum(p.numel() for p in load_model(template).parameters()),
            bc=identity(teacher), opponent=identity(teacher), template=identity(template),
            pilot_data=identity(rl_data),
            dataset={split: identity(rl_data) | dict(rows=12, terminal_trajectories=3)
                     for split in ("train", "validation", "test")},
            target_kind="mc", td_loss="mse", reward_scheme="round_and_placement",
            round_score_unit=10000., round_reward_weight=.2, alpha=.5,
            learning_rate=1e-4, warmup_steps=2, weight_decay=.01,
            batch_size=5, replay_size=8, target_update_rate=.005,
            pilot_checkpoint_interval=5, checkpoint_interval=5,
            training_budget_seconds=180, maximum_seconds=300, max_epochs=3,
            native_sets=0, functional_receipt=identity(functional),
        )

    def test_normal_workflow_multiple_epochs_and_exact_mid_epoch_resume(self):
        class Interrupted(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.fixture(root)
            plans = {}
            for name in ("continuous", "interrupted"):
                path = root / f"{name}-plan.json"
                path.write_text(json.dumps(plan | dict(output_prefix=str(root / name))), encoding="utf-8")
                plans[name] = path
            with patch.object(torch, "set_num_interop_threads"):
                run(plans["continuous"])
                original = cql._training
                epoch_calls = []

                def interrupt_second_epoch(**arguments):
                    epoch_calls.append(None)
                    snapshot = arguments["snapshot_writer"]

                    def save_and_interrupt(index, **metadata):
                        snapshot(index, **metadata)
                        if len(epoch_calls) == 2 and index == 5:
                            raise Interrupted("Simulated interruption after a saved checkpoint.")

                    arguments["snapshot_writer"] = save_and_interrupt
                    return original(**arguments)

                with patch.object(cql, "_training", side_effect=interrupt_second_epoch), self.assertRaises(Interrupted):
                    run(plans["interrupted"])
                partial = torch.load(root / "interrupted-checkpoint.pt", weights_only=True)
                self.assertEqual((partial["epoch"], partial["cursor"], partial["optimizer_steps"]), (1, 5, 4))
                run(plans["interrupted"], resume=True)
            full = torch.load(root / "continuous-checkpoint.pt", weights_only=True)
            resumed = torch.load(root / "interrupted-checkpoint.pt", weights_only=True)
            for key in ("model", "target", "optimizer", "scheduler", "scaler", "rng", "replay",
                        "cursor", "epoch", "total_samples", "completed_epochs", "attempted_updates", "optimizer_steps"):
                equal(full[key], resumed[key])
            self.assertEqual((full["total_samples"], full["completed_epochs"], full["optimizer_steps"]), (36, 3, 9))
            summary = json.loads((root / "interrupted-summary.json").read_text())
            self.assertEqual(summary["attempted_updates"], 9)
            self.assertEqual(summary["last_metrics"]["Optimizer steps (this run)"], 3.)
            self.assertIn("Loss (MC Return)", summary["last_metrics"])
            self.assertNotIn("Loss (TD Error)", summary["last_metrics"])
            model = load_model(root / "interrupted.kanachan").eval()
            reader = EpisodeReplayBuffer(
                training_data=root / "rl.tsv", contiguous_training_data=True,
                num_skip_samples=0, rewrite_rooms=None, rewrite_grades=None,
                get_reward=configured_reward("round_and_placement"), discount_factor=1.,
                dtype=torch.float32, max_size=128, batch_size=128, num_workers=0, pin_memory=False,
            )
            batches = list(reader)
            self.assertEqual(len(batches), 1)
            batch = batches[0]
            with torch.no_grad():
                q_only(model)(batch)
                chosen = batch["action_value"].gather(1, batch["action"].long()[:, None]).flatten()
                mse = float((chosen - batch["mc_return"]).square().mean())
            self.assertAlmostEqual(summary["validation"]["mse_mc"], mse, places=5)
            self.assertFalse((root / "interrupted-error.json").exists())
            with patch.object(torch, "set_num_interop_threads"), self.assertRaisesRegex(ValueError, "unfinished training"):
                run(plans["interrupted"], resume=True)


if __name__ == "__main__":
    unittest.main()
