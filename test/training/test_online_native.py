"""Opt-in real simulator, CUDA/AMP and online recovery acceptance test."""

import os
import tempfile
from dataclasses import replace
from pathlib import Path
import json
import unittest

import torch

from test_online_training import write_q_model
from kanachan.model_loader import load_model
from kanachan.simulation import simulate
from kanachan.training.online.config import Config
from kanachan.training.online.train import run


@unittest.skipUnless(
    os.environ.get("KANACHAN_TEST_ONLINE_NATIVE"),
    "Set KANACHAN_TEST_ONLINE_NATIVE=1 for real online simulator acceptance",
)
class OnlineNativeTests(unittest.TestCase):
    def test_seeded_games_and_online_amp_recovery(self):
        torch.set_num_threads(2)
        torch.manual_seed(20261029)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        keep = os.environ.get("KANACHAN_ONLINE_SMOKE_OUTPUT")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(keep) if keep else Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            initial = root / "initial.kanachan"
            write_q_model(initial)
            actor = (
                load_model(initial, torch.device("cpu"))
                .float()
                .to(device)
                .eval()
            )

            def games(concurrency, seed):
                return simulate(
                    device,
                    torch.float32,
                    3,
                    12,
                    actor,
                    12,
                    actor,
                    4,
                    1,
                    1,
                    concurrency,
                    lambda: None,
                    seed=seed,
                )

            with torch.inference_mode():
                left = games(1, 20261029)
                right = games(2, 20261029)
                changed = games(1, 20261030)
            self.assertEqual(
                [item.get_result() for item in left],
                [item.get_result() for item in right],
            )
            self.assertEqual(
                [item.get_metadata() for item in left],
                [{"set_id": 0, "game_id": i} for i in range(4)],
            )
            self.assertFalse(
                torch.equal(
                    left[0].get_episode(0)["episode"]["sparse"][0],
                    changed[0].get_episode(0)["episode"]["sparse"][0],
                )
            )
            del left, right, changed
            options = Config(
                str(initial),
                str(root / "run"),
                device=str(device),
                amp_dtype="float16" if device.type == "cuda" else "float32",
                iterations=1,
                games_per_iteration=1,
                simulation_batch_size=1,
                simulation_concurrency=1,
                learning_starts=1,
                batch_size=4,
                updates_per_iteration=4,
                replay_capacity=1024,
                evaluate_every=0,
            )
            first = run(options)
            state = torch.load(
                root / "run/latest.pt", map_location="cpu", weights_only=True
            )
            second = run(replace(options, iterations=2), resume_state=state)
            resumed = torch.load(
                root / "run/latest.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(second["policy_version"], 2)
            self.assertGreater(
                second["optimizer_steps"], first["optimizer_steps"]
            )
            self.assertLessEqual(second["optimizer_steps"], 8)
            self.assertGreater(second["samples"], first["samples"])
            self.assertIn(
                1, resumed["replay"]["columns"]["policy_version"].tolist()
            )
            self.assertTrue(
                any(
                    not torch.equal(value, resumed["model"]["state_dict"][key])
                    for key, value in state["model"]["state_dict"].items()
                )
            )
            if device.type == "cuda":
                self.assertTrue(resumed["scaler"])
            receipt = {
                "device": str(device),
                "seed_repeatable_across_concurrency": True,
                "different_seed_changes_hand": True,
                "first": first,
                "resumed": second,
                "amp_enabled": device.type == "cuda",
            }
            (root / "receipt.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(receipt))
