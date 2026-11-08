import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from kanachan.model_loader import load_model
from kanachan.nn.decision_decoder import DenseDecisionDecoder, GlobalActionHead
from kanachan.training.core.bc import DataLoader
from kanachan.training.online.config import Config as OnlineConfig
from kanachan.training.online.train import run
from test_training_regressions import bc_line
from test_training_pipeline import rl_line
from test_online_training import collect_fixture


class DecisionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def cli(self, module, data, output, *extra):
        environment = dict(
            os.environ, WORLD_SIZE="1", RANK="0", LOCAL_RANK="0",
            TORCHDYNAMO_DISABLE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
            HYDRA_FULL_ERROR="1", PYTHONDONTWRITEBYTECODE="1",
        )
        result = subprocess.run([
            sys.executable, "-B", "-m", f"kanachan.training.{module}.train",
            "--config-name", "decision", "device=cpu",
            "encoder.dimension=16", "encoder.num_heads=2",
            "encoder.dim_feedforward=24", "encoder.num_layers=2",
            "num_workers=0", "batch_size=2",
            f"training_data={data.as_posix()}",
            f"hydra.run.dir={output.as_posix()}", *extra,
        ], cwd=Path(__file__).resolve().parents[2], env=environment,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_flex_backend_export_and_recipe_through_bc_and_cql_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc_data, rl_data = root / "bc.tsv", root / "rl.tsv"
            bc_data.write_text("".join(bc_line(i) for i in range(4)), encoding="utf-8")
            rl_data.write_text("".join(rl_line(i, terminal=i == 3) for i in range(4)),
                               encoding="utf-8")
            for algorithm, data, extra in (
                ("bc", bc_data, []),
                ("cql", rl_data, ["replay_buffer_size=4",
                                 f"bc_initial_model={(root / 'bc/snapshots/model.kanachan').as_posix()}"]),
            ):
                self.cli(algorithm, data, root / algorithm,
                         "encoder.attention_backend=flex", *extra)
                prefix = root / algorithm / "snapshots"
                actor = load_model(prefix / "model.kanachan")
                body = next(module for module in actor.modules()
                            if isinstance(module, DenseDecisionDecoder))
                self.assertEqual(body.attention_backend, "flex")
                self.assertTrue(all(block.attention_backend == "flex" for block in body.blocks))
                manifest = json.loads((prefix / "decision-checkpoint.json").read_text())
                self.assertEqual(manifest["recipe"]["backbone"]["attention_backend"], "flex")

    def test_bc_cql_exports_replay_resume_and_online_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc_data, rl_data = root / "bc.tsv", root / "rl.tsv"
            bc_data.write_text("".join(bc_line(i % 4) for i in range(12)), encoding="utf-8")
            rl_data.write_text("".join(
                rl_line(i, terminal=i % 4 == 3) for i in range(12)
            ), encoding="utf-8")
            self.cli("bc", bc_data, root / "bc", "snapshot_interval=4")
            bc = load_model(root / "bc/snapshots/model.kanachan").eval()
            self.assertTrue(any(isinstance(m, DenseDecisionDecoder) for m in bc.modules()))
            self.assertTrue(any(isinstance(m, GlobalActionHead) for m in bc.modules()))
            batch = next(DataLoader(
                path=bc_data, num_skip_samples=0, rewrite_rooms=None,
                rewrite_grades=None, batch_size=4, num_workers=0,
            ))
            bc(batch)
            self.assertTrue(torch.isfinite(batch["log_probs"][:, :4]).all())
            self.assertEqual(batch["action"].shape, (4,))
            self.cli(
                "bc", bc_data, root / "bc-resumed", "snapshot_interval=4",
                f"initial_model_prefix={(root / 'bc/snapshots').as_posix()}",
                "initial_model_index=4",
            )
            for name in ("encoder.pth", "decoder.pth"):
                left = torch.load(root / "bc/snapshots" / name, weights_only=True)
                right = torch.load(root / "bc-resumed/snapshots" / name, weights_only=True)
                for key in left:
                    torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
            options = [
                "replay_buffer_size=4", "snapshot_interval=2",
                f"bc_initial_model={(root / 'bc/snapshots/model.kanachan').as_posix()}",
            ]
            self.cli("cql", rl_data, root / "cql", *options)
            prefix = root / "cql/snapshots"
            exported = load_model(prefix / "model.kanachan")
            source = torch.load(prefix / "encoder.pth", weights_only=True)
            actual_backbone = next(m for m in exported.modules() if isinstance(m, DenseDecisionDecoder))
            for key, value in source.items():
                torch.testing.assert_close(value, actual_backbone.state_dict()[key], atol=0, rtol=0)
            self.assertFalse((prefix / "target-encoder.pth").exists())
            self.cli(
                "cql", rl_data, root / "resumed",
                "replay_buffer_size=4", "snapshot_interval=2",
                f"bc_initial_model={(root / 'bc/snapshots/model.kanachan').as_posix()}",
                f"initial_model_prefix={prefix.as_posix()}",
                "initial_model_index=2",
            )
            for name in ("encoder.pth", "decoder.pth", "optimizer.pth", "replay-state.pth"):
                left = torch.load(prefix / name, weights_only=True)
                right = torch.load(root / "resumed/snapshots" / name, weights_only=True)
                from test_decision_optimization import compare_state
                compare_state(self, left, right)
            training_state = json.loads((prefix / "training-state.json").read_text())
            manifest = json.loads((prefix / "decision-checkpoint.json").read_text())
            self.assertEqual(manifest["recipe"]["algorithm"], "cql")
            self.assertEqual(manifest["recipe"]["training"]["td_loss"], "mse")
            self.assertEqual(manifest["recipe"]["training"]["target_kind"], "mc")
            self.assertFalse(manifest["recipe"]["training"]["double_q_learning"])
            self.assertEqual(manifest["recipe"]["training"]["reward_scheme"], "round_and_placement")
            self.assertEqual(manifest["recipe"]["training"]["round_reward_weight"], .2)
            self.assertNotIn("mc_weight", manifest["recipe"]["training"])
            self.assertEqual(manifest["recipe"]["training"]["alpha"], .5)
            self.assertNotIn("critic_warmup_steps", manifest["recipe"]["training"])
            self.assertEqual(manifest["recipe"]["bc_feature_transfer"]["policy"],
                             "backbone_and_hidden_head_fresh_zero_scalar_output_v1")
            self.assertEqual(training_state["optimizer_steps"], 6)
            self.assertEqual(training_state["num_samples"], 12)
            # The transferred body and hidden head must learn, not merely
            # preserve their BC bytes through a feature-freezing phase.
            for name, actual in (("encoder.pth", source),
                                 ("decoder.pth", torch.load(prefix / "decoder.pth", weights_only=True))):
                teacher = torch.load(root / "bc/snapshots" / name, weights_only=True)
                keys = [key for key in actual if key in teacher and (name == "encoder.pth" or key.startswith("layers.0."))]
                self.assertTrue(any(not torch.equal(actual[key], teacher[key]) for key in keys))
            optimizer = torch.load(prefix / "optimizer.pth", weights_only=True)
            self.assertEqual({int(state["step"]) for state in optimizer["state"].values()}, {6})
            options = OnlineConfig(
                initial_model=str(prefix / "model.kanachan"),
                output_dir=str(root / "online-full"), device="cpu",
                amp_dtype="float32", iterations=2, games_per_iteration=1,
                simulation_batch_size=1, simulation_concurrency=1,
                replay_capacity=16, learning_starts=1, batch_size=2,
                updates_per_iteration=2, evaluate_every=0, td_loss="huber",
                cql_alpha_first=0.05, cql_alpha_last=0.05, cql_annealing_steps=0,
                offline_data=str(rl_data), offline_capacity=8,
            )
            full = run(options, collect_fn=collect_fixture)
            partial_options = replace(
                options, output_dir=str(root / "online-partial"), iterations=1
            )
            run(partial_options, collect_fn=collect_fixture)
            state = torch.load(root / "online-partial/latest.pt", weights_only=True)
            partial = run(
                replace(partial_options, iterations=2), resume_state=state,
                collect_fn=collect_fixture,
            )
            expected = torch.load(root / "online-full/latest.pt", weights_only=True)
            actual = torch.load(root / "online-partial/latest.pt", weights_only=True)
            self.assertEqual(actual["optimizer_steps"], 4)
            self.assertEqual(actual["offline_replay"]["size"], 8)
            self.assertTrue(torch.equal(
                expected["offline_replay"]["rng"], actual["offline_replay"]["rng"]
            ))
            for key, value in expected["offline_replay"]["columns"].items():
                torch.testing.assert_close(
                    value, actual["offline_replay"]["columns"][key], atol=0, rtol=0
                )
            for key, value in expected["model"]["state_dict"].items():
                torch.testing.assert_close(value, actual["model"]["state_dict"][key], atol=0, rtol=0)
            for key, value in expected["target"].items():
                torch.testing.assert_close(value, actual["target"][key], atol=0, rtol=0)
            self.assertTrue(torch.equal(expected["rng"]["torch"], actual["rng"]["torch"]))

    def test_resume_rejects_changed_mask_or_bytes(self):
        from types import SimpleNamespace
        from kanachan.training.core.decision_checkpoint import validate_resume
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SimpleNamespace(
                initial_model_prefix=root, initial_model_index=4,
                optimizer=SimpleNamespace(initialize=False), start_new_epoch=False,
            )
            path = root / "decision-checkpoint.4.json"
            path.write_text(json.dumps({
                "recipe": {"history_attention": "bidirectional"}, "files": {},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recipe"):
                validate_resume(config, {"history_attention": "causal"})
            for field, value in (("mc_weight", .5), ("critic_warmup_steps", 2048)):
                path.write_text(json.dumps({
                    "recipe": {"history_attention": "causal", "training": {field: value}},
                    "files": {},
                }), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "recipe"):
                    validate_resume(config, {"history_attention": "causal"})
            path.write_text(json.dumps({
                "recipe": {"history_attention": "causal"},
                "files": {"encoder.4.pth": "wrong_digest"},
            }), encoding="utf-8")
            (root / "encoder.4.pth").write_bytes(b"incomplete snapshot")
            with self.assertRaisesRegex(ValueError, "bytes"):
                validate_resume(config, {"history_attention": "causal"})


if __name__ == "__main__":
    unittest.main()
