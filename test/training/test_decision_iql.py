import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from kanachan.constants import NUM_TYPES_OF_ACTIONS
from kanachan.model_loader import load_model
from kanachan.nn import TwinQActor
from kanachan.nn.decision_state import DecisionStateHead
from kanachan.nn.decision_value import transfer_bc_features
from kanachan.nn.decision_decoder import DenseDecisionDecoder, GlobalActionHead
from kanachan.nn.functional import advantage_weights, expectile_loss
from kanachan.training.core.bc import DataLoader
from kanachan.training.core.final_rank import final_rank_loss
from test_training_pipeline import rl_line
from test_training_regressions import bc_line


class DecisionIQLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def cli(self, module, data, output, *extra, device="cpu", error=None):
        command = [
            sys.executable,
            "-B",
            "-m",
            f"kanachan.training.{module}.train",
            "--config-name",
            "decision",
            f"device={device}",
            "encoder.dimension=16",
            "encoder.num_heads=2",
            "encoder.dim_feedforward=24",
            "encoder.num_layers=1",
            "num_workers=0",
            "batch_size=2",
            f"training_data={data.as_posix()}",
            f"hydra.run.dir={output.as_posix()}",
            *extra,
        ]
        environment = dict(
            os.environ,
            WORLD_SIZE="1",
            RANK="0",
            LOCAL_RANK="0",
            TORCHDYNAMO_DISABLE="1",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            HYDRA_FULL_ERROR="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        result = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            cwd=Path(__file__).resolve().parents[2],
        )
        messages = result.stdout + result.stderr
        if error is None:
            self.assertEqual(result.returncode, 0, messages)
        else:
            self.assertNotEqual(result.returncode, 0, messages)
            self.assertIn(error, messages)

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_expectile_and_author_inverse_temperature(self):
        target = torch.tensor([2.0, -2.0], requires_grad=True)
        value = torch.zeros(2, requires_grad=True)
        loss = expectile_loss(target, value, 0.75)
        self.assertEqual(loss.item(), 2.0)
        loss.backward()
        self.assertIsNone(target.grad)
        torch.testing.assert_close(value.grad, torch.tensor([-1.5, 0.5]))
        weights = advantage_weights(target, value, beta=0.5, max_weight=100.0)
        torch.testing.assert_close(weights, torch.exp(2 * target.detach()))
        self.assertFalse(weights.requires_grad)
        self.assertAlmostEqual(
            advantage_weights(
                torch.tensor([1e20]), value[:1], 0.1, 100
            ).item(),
            100.0,
            places=4,
        )
        for invalid in (math.nan, math.inf, 0.0, 1.0):
            with self.assertRaises(ValueError):
                expectile_loss(target, value, invalid)

    def test_rank_labels_mask_unobserved_outcomes_and_use_starting_seat(self):
        # Equal scores use the starting-seat order, not relative feature order.
        sparse = torch.tensor(
            [[3, 6, 7, 23, 39, 55, 74], [3, 6, 7, 23, 39, 55, 71]],
            dtype=torch.int32,
        )
        data = TensorDict(
            {
                "sparse": sparse,
                "next": {
                    "end_of_game": torch.tensor([True, False]),
                    "results": torch.tensor(
                        [[0] * 10 + [25000] * 4, [0] * 14]
                    ),
                },
            },
            [2],
        )
        logits = torch.tensor(
            [[0.0, 0.0, 0.0, 2.0], [4.0, 3.0, 2.0, 1.0]], requires_grad=True
        )
        expected = -logits[0].log_softmax(-1)[3] / 2
        torch.testing.assert_close(final_rank_loss(logits, data), expected)
        final_rank_loss(logits, data).backward()
        torch.testing.assert_close(logits.grad[1], torch.zeros(4))
        data["next", "end_of_game"].zero_()
        self.assertEqual(final_rank_loss(logits, data).item(), 0)

    def test_bc_transfer_to_value_keeps_fresh_scalar_output(self):
        kwargs = dict(
            dimension=16,
            num_heads=2,
            dim_feedforward=24,
            num_layers=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        teacher = DenseDecisionDecoder(**kwargs)
        teacher_head = GlobalActionHead(
            input_dimension=16,
            hidden_dimension=16,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        bc = TensorDictSequential(
            TensorDictModule(
                teacher,
                ["sparse", "numeric", "progression", "candidates"],
                ["encode"],
            ),
            TensorDictModule(
                teacher_head, ["candidates", "encode"], ["decode"]
            ),
        )
        body = DenseDecisionDecoder(**kwargs)
        head = DecisionStateHead(
            input_dimension=16,
            hidden_dimension=16,
            device=torch.device("cpu"),
            dtype=torch.float32,
            final_rank=True,
        )
        transfer_bc_features(body, head, bc)
        self.assert_state_equal(body.state_dict(), teacher.state_dict())
        self.assert_state_equal(
            head.layers[0].state_dict(), teacher_head.layers[0].state_dict()
        )
        self.assertEqual(head.layers[-1].weight.count_nonzero().item(), 0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            value, logits = head(torch.randn(3, 16))
        self.assertEqual(value.dtype, torch.float32)
        self.assertEqual(logits.dtype, torch.float32)
        (value.sum() + logits.sum()).backward()
        self.assertTrue(all(p.grad is not None for p in head.parameters()))

    def run_pipeline(self, device):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc_data, rl_data, reward = (
                root / "bc.tsv",
                root / "rl.tsv",
                root / "reward.py",
            )
            bc_data.write_text(
                "".join(bc_line(i % 4) for i in range(12)), encoding="utf-8"
            )
            rl_data.write_text(
                "".join(rl_line(i, terminal=i % 4 == 3) for i in range(12)),
                encoding="utf-8",
            )
            reward.write_text(
                "from kanachan.training.core.rl.rewards import PlacementReward\n"
                "get_reward = PlacementReward()\n",
                encoding="utf-8",
            )
            amp = ["device.amp_dtype=bfloat16"] if device == "cuda" else []
            self.cli("bc", bc_data, root / "bc", *amp, device=device)
            arguments = [
                f"reward_plugin={reward.as_posix()}",
                "final_rank_weight=0.1",
                "gradient_accumulation_steps=2",
                "snapshot_interval=4",
                "optimizer.warmup_steps=2",
                "optimizer.annealing_steps=8",
                *amp,
            ]
            self.cli(
                "iql",
                rl_data,
                root / "iql",
                *arguments,
                f"bc_initial_model={(root / 'bc/snapshots/model.kanachan').as_posix()}",
                device=device,
            )
            prefix = root / "iql/snapshots"
            self.cli(
                "iql",
                rl_data,
                root / "resume",
                *arguments,
                f"initial_model_prefix={prefix.as_posix()}",
                "initial_model_index=4",
                device=device,
            )
            for path in prefix.glob("*.pth"):
                if "." in path.stem or path.stem == "decision-rng":
                    continue
                self.assert_state_equal(
                    torch.load(path, map_location="cpu", weights_only=True),
                    torch.load(
                        root / "resume/snapshots" / path.name,
                        map_location="cpu",
                        weights_only=True,
                    ),
                )
            manifest = json.loads(
                (prefix / "decision-checkpoint.json").read_text()
            )
            self.assertIn("value-model.kanachan", manifest["files"])
            self.assertIn("value-lr-scheduler.pth", manifest["files"])
            self.assertIn("q1-lr-scheduler.pth", manifest["files"])
            self.assertIn("q2-lr-scheduler.pth", manifest["files"])
            self.assertIn("bc_feature_transfer", manifest["recipe"])
            self.assertEqual(
                json.loads((prefix / "training-state.json").read_text())[
                    "optimizer_steps"
                ],
                3,
            )
            q_model = load_model(prefix / "model.kanachan").eval()
            value_model = load_model(prefix / "value-model.kanachan").eval()
            batch = next(
                DataLoader(
                    path=bc_data,
                    num_skip_samples=0,
                    rewrite_rooms=None,
                    rewrite_grades=None,
                    batch_size=2,
                )
            )
            with torch.no_grad():
                q_model(batch)
                value_model(batch)
            self.assertEqual(tuple(batch["state_value"].shape), (2,))
            self.assertEqual(tuple(batch["final_rank_logits"].shape), (2, 4))
            # A zeroed twin Q must resolve ties by global action ID.
            actor = next(
                m for m in q_model.modules() if isinstance(m, TwinQActor)
            )
            valid = batch["candidates"] < NUM_TYPES_OF_ACTIONS
            first, second = batch.clone(), batch.clone()
            with torch.no_grad():
                actor.model0(first)
                actor.model1(second)
            torch.testing.assert_close(
                batch["action_value"][valid],
                torch.minimum(first["action_value"], second["action_value"])[
                    valid
                ],
            )
            first_head = next(
                m
                for m in actor.model0.modules()
                if isinstance(m, GlobalActionHead)
            )
            second_head = next(
                m
                for m in actor.model1.modules()
                if isinstance(m, GlobalActionHead)
            )
            self.assertFalse(
                torch.equal(
                    first_head.layers[-1].weight, second_head.layers[-1].weight
                )
            )
            for head in q_model.modules():
                if isinstance(head, GlobalActionHead):
                    head.zero_output()
            batch["candidates"][:, :4] = torch.tensor(
                [3, 2, 0, 1], dtype=torch.int32
            )
            q_model(batch)
            self.assertTrue(actor.tie_break_by_action_id)
            self.assertTrue(
                torch.equal(
                    batch["action"], torch.full((2,), 2, dtype=torch.int32)
                )
            )
            awr_args = [
                f"q_model={(prefix / 'model.kanachan').as_posix()}",
                f"value_model={(prefix / 'value-model.kanachan').as_posix()}",
                "snapshot_interval=4",
                "gradient_accumulation_steps=2",
                "optimizer.warmup_steps=2",
                "optimizer.annealing_steps=8",
                *amp,
            ]
            self.cli(
                "awr",
                bc_data,
                root / "awr",
                *awr_args,
                f"encoder.load_from={(root / 'bc/snapshots/encoder.pth').as_posix()}",
                f"decoder.load_from={(root / 'bc/snapshots/decoder.pth').as_posix()}",
                device=device,
            )
            self.cli(
                "awr",
                bc_data,
                root / "awr-resume",
                *awr_args,
                f"initial_model_prefix={(root / 'awr/snapshots').as_posix()}",
                "initial_model_index=4",
                device=device,
            )
            for name in ("encoder", "decoder", "optimizer", "scheduler"):
                self.assert_state_equal(
                    torch.load(
                        root / f"awr/snapshots/{name}.pth", weights_only=True
                    ),
                    torch.load(
                        root / f"awr-resume/snapshots/{name}.pth",
                        weights_only=True,
                    ),
                )
            policy = load_model(root / "awr/snapshots/model.kanachan").eval()
            policy(batch)
            self.assertTrue(torch.isfinite(batch["log_probs"][:, :4]).all())
            self.cli(
                "iql",
                rl_data,
                root / "wrong-recipe",
                *arguments,
                f"initial_model_prefix={prefix.as_posix()}",
                "expectile=0.8",
                device=device,
                error="recipe or training data differs",
            )
            payload = (prefix / "value-model.kanachan").read_bytes()
            (prefix / "value-model.kanachan").write_bytes(payload + b"tamper")
            self.cli(
                "iql",
                rl_data,
                root / "wrong-bytes",
                *arguments,
                f"initial_model_prefix={prefix.as_posix()}",
                device=device,
                error="checkpoint bytes differ",
            )

    def test_cpu_exports_bc_transfer_awr_and_exact_resume(self):
        self.run_pipeline("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_bfloat16_exports_awr_and_exact_resume(self):
        self.run_pipeline("cuda")

    def test_terminal_only_and_replay_without_auxiliary_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, reward = root / "terminal.tsv", root / "reward.py"
            data.write_text(
                "".join(bc_line(i % 4) for i in range(4)), encoding="utf-8"
            )
            reward.write_text(
                "def get_reward(data, contiguous):\n    data['next', 'reward'] = data['next', 'end_of_game'].float()\n",
                encoding="utf-8",
            )
            self.cli(
                "iql",
                data,
                root / "terminal",
                f"reward_plugin={reward.as_posix()}",
            )
            self.cli(
                "iql",
                data,
                root / "replay",
                f"reward_plugin={reward.as_posix()}",
                "contiguous_training_data=true",
                "replay_buffer_size=4",
            )
            value = load_model(
                root / "terminal/snapshots/value-model.kanachan"
            )
            self.assertTrue(
                any(
                    isinstance(m, DecisionStateHead) and m.final_rank is None
                    for m in value.modules()
                )
            )

    def test_legacy_value_export_can_supply_existing_awr(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "terminal.tsv"
            data.write_text(bc_line(0) * 4, encoding="utf-8")
            legacy = ["encoder=bert_tiny", "decoder=double"]
            reward = (
                Path(__file__).resolve().parents[2]
                / "examples/training/iql-placement-reward.py"
            )
            self.cli(
                "iql",
                data,
                root / "iql",
                *legacy,
                f"reward_plugin={reward.as_posix()}",
            )
            prefix = root / "iql/snapshots"
            self.cli(
                "awr",
                data,
                root / "awr",
                *legacy,
                f"q_model={(prefix / 'model.kanachan').as_posix()}",
                f"value_model={(prefix / 'value-model.kanachan').as_posix()}",
            )
            policy = load_model(root / "awr/snapshots/model.kanachan").eval()
            batch = next(
                DataLoader(
                    path=data,
                    num_skip_samples=0,
                    rewrite_rooms=None,
                    rewrite_grades=None,
                    batch_size=2,
                )
            )
            policy(batch)
            self.assertTrue(torch.isfinite(batch["log_probs"][:, :4]).all())


if __name__ == "__main__":
    unittest.main()
