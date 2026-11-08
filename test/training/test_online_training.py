import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import copy
from dataclasses import replace
import tempfile
import random
import subprocess
import sys
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from kanachan.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ACTIONS,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ROUND_SUMMARY,
)
from kanachan.model_loader import dump_model, dump_object, load_model
from kanachan.nn import Encoder, Decoder, DecodeConverter, RoundEncoder
from kanachan.training.core.eor.feature_adaptor import FeatureAdaptor
from kanachan.training.core.rl.simulation import simulate as collect_native
from kanachan.simulation.statistics import (
    compare_policies,
    promotion_decision,
    summarize,
)
from kanachan.training.core.rl.rewards import PlacementReward
from kanachan.training.core.offline_rl import EpisodeReplayBuffer
from test_training_pipeline import rl_line, unit_reward
from kanachan.training.online.config import Config
from kanachan.training.online.q_learning import q_loss
from kanachan.training.online.replay import ReplayBuffer
from kanachan.training.online.train import run


def transitions():
    size = 4
    sparse = torch.full(
        (size, MAX_NUM_ACTIVE_SPARSE_FEATURES),
        NUM_TYPES_OF_SPARSE_FEATURES,
        dtype=torch.int32,
    )
    sparse[:, :11] = torch.tensor([3, 6, 19, 35, 51, 67, 71, 75, 78, 82, 152])
    sparse[2:, 8] = 79
    numeric = torch.tensor(
        [[0, 0, 25000, 25000, 25000, 25000]] * size, dtype=torch.int32
    )
    numeric[2:, 2] = 30000
    progression = torch.full(
        (size, MAX_LENGTH_OF_PROGRESSION_FEATURES),
        NUM_TYPES_OF_PROGRESSION_FEATURES,
        dtype=torch.int32,
    )
    progression[:, :2] = torch.tensor([0, 5])
    candidates = torch.full(
        (size, MAX_NUM_ACTION_CANDIDATES),
        NUM_TYPES_OF_ACTIONS,
        dtype=torch.int32,
    )
    candidates[:, :4] = torch.arange(4, dtype=torch.int32)
    data = TensorDict(
        {
            "sparse": sparse,
            "numeric": numeric,
            "progression": progression,
            "candidates": candidates,
            "action": torch.zeros(size, dtype=torch.int32),
        },
        [size],
    )
    for key, padding in (
        ("sparse", NUM_TYPES_OF_SPARSE_FEATURES),
        ("numeric", 0),
        ("progression", NUM_TYPES_OF_PROGRESSION_FEATURES),
        ("candidates", NUM_TYPES_OF_ACTIONS),
    ):
        data["next", key] = data[key].roll(-1, 0).clone()
        data["next", key][-1].fill_(padding)
    data["next", "results"] = torch.zeros(size, 14, dtype=torch.int32)
    data["next", "results"][-1, 10:] = torch.tensor(
        [40000, 30000, 20000, 10000]
    )
    data["next", "round_summary"] = torch.full(
        (size, MAX_NUM_ROUND_SUMMARY), 29, dtype=torch.int32
    )
    data["next", "end_of_round"] = torch.tensor([False, True, False, True])
    data["next", "end_of_game"] = torch.tensor([False, False, False, True])
    data["next", "discount"] = torch.ones(size)
    PlacementReward()(data, True)
    return data


def write_q_model(path, *, ranking=False, architecture="transformer"):
    encoder_options = dict(
        architecture=architecture,
        position_encoder="position_embedding",
        dimension=8,
        num_heads=2,
        dim_feedforward=16,
        num_layers=1,
        activation_function="gelu",
        dropout=0.1,
        layer_normalization=False,
        checkpointing=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    decoder_options = dict(
        input_dimension=8,
        dimension=None,
        activation_function=None,
        dropout=None,
        num_layers=1,
        output_mode="ranking" if ranking else "candidates",
        noise_init_std=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    encoder = (RoundEncoder if ranking else Encoder)(**encoder_options)
    decoder = Decoder(**decoder_options)
    converter = nn.Softmax(2) if ranking else DecodeConverter("argmax")
    descriptions = []
    modules = []
    for model, args, kwargs, inputs, outputs in (
        (
            encoder,
            [],
            encoder_options,
            ["sparse", "numeric"]
            if ranking
            else ["sparse", "numeric", "progression", "candidates"],
            ["encode"],
        ),
        (
            decoder,
            [],
            decoder_options,
            ["encode"],
            ["decode"] if ranking else ["action_value"],
        ),
        (
            converter,
            [2] if ranking else ["argmax"],
            {},
            ["decode"] if ranking else ["candidates", "action_value"],
            ["ranking_probs"] if ranking else ["action"],
        ),
    ):
        module = TensorDictModule(model, in_keys=inputs, out_keys=outputs)
        modules.append(module)
        descriptions.append(
            dump_object(
                module,
                [dump_object(model, args, kwargs)],
                {"in_keys": inputs, "out_keys": outputs},
            )
        )
    actor = TensorDictSequential(*modules)
    torch.save(dump_model(actor, descriptions, {}), path)
    return actor


def collect_fixture(**kwargs):
    data = transitions()
    kwargs["policy_model"](data)
    for key, value in (
        ("policy_version", kwargs["policy_version"]),
        ("opponent_version", kwargs["opponent_version"]),
        ("collection_seed", kwargs["seed"]),
        ("game_id", 0),
        ("set_id", 0),
    ):
        data[key] = torch.full((len(data),), value, dtype=torch.int64)
    kwargs["get_reward"](data, True)
    return data


def ranked_game(rank):
    return [
        {"ranking": i, "score": 40000 - i * 10000, "proposed": i == rank}
        for i in range(4)
    ]


class PublicPotential(nn.Module):
    def forward(self, data):
        probability = (data["numeric"][:, 2].float() / 10000).sigmoid()
        probabilities = torch.zeros(len(data), 4, 4)
        probabilities[:, :, 0] = probability[:, None]
        probabilities[:, :, 1] = 1 - probability[:, None]
        data["ranking_probs"] = probabilities
        return data


class OnlineTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_terminal_reward_and_tie_break_follow_starting_seat(self):
        data = transitions()
        self.assertEqual(data["next", "reward"].tolist(), [0, 0, 0, 1.5])
        for seat in range(4):
            data["sparse"][:, 6] = 71 + seat
            data["next", "results"][-1, 10:] = 25000
            PlacementReward()(data, True)
            self.assertEqual(float(data["next", "reward"][-1]), 1.5 - seat)
        self.assertEqual(
            data["next", "done"].tolist(), [False, False, False, True]
        )

    def test_shaping_telescopes_and_does_not_cross_games(self):
        first = transitions()
        data = torch.cat([first, first.clone()])
        PlacementReward(PublicPotential())(data, True)
        initial = 0.5 + torch.tensor(2.5).sigmoid().item()
        for start in (0, 4):
            self.assertAlmostEqual(
                float(data["next", "reward"][start : start + 4].sum()),
                1.5 - initial,
                places=6,
            )
            self.assertGreater(float(data["next", "reward"][start + 1]), 0)
        with self.assertRaises(ValueError):
            PlacementReward(PublicPotential())(transitions(), False)

    def test_replay_wraparound_restores_contents_and_sampling_rng(self):
        replay = ReplayBuffer(6, 11)
        replay.append(transitions())
        later = transitions()
        later["numeric"][:, 0] = torch.arange(4) + 10
        replay.append(later)
        state = replay.state_dict()
        restored = ReplayBuffer(6, 999)
        restored.load_state_dict(state)
        self.assertEqual(len(restored), 6)
        for key in state["columns"]:
            self.assertTrue(
                torch.equal(replay.columns[key], restored.columns[key])
            )
        left, right = replay.sample(32), restored.sample(32)
        for key in left.keys(True, True):
            self.assertTrue(torch.equal(left[key], right[key]))
        self.assertTrue((left["numeric"][:, 0] < 4).any())
        self.assertTrue((left["numeric"][:, 0] >= 10).any())

    def test_real_eor_export_contract_supports_potential_shaping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eor.kanachan"
            write_q_model(path, ranking=True)
            data = transitions()
            reference = load_model(path, torch.device("cpu")).eval()
            sparse, numeric = FeatureAdaptor()(
                data["sparse"][:1], data["numeric"][:1]
            )
            td = TensorDict({"sparse": sparse, "numeric": numeric}, [1])
            reference(td)
            initial = float(
                td["ranking_probs"][0, 0]
                @ torch.tensor([1.5, 0.5, -0.5, -1.5])
            )
            PlacementReward(path)(data, True)
            self.assertAlmostEqual(
                float(data["next", "reward"].sum()), 1.5 - initial, places=6
            )

    def test_games_without_any_proposed_decision_do_not_break_collection(self):
        class EmptyGame:
            def get_result(self):
                return [
                    {"proposed": i == 3, "round_results": []} for i in range(4)
                ]

            def get_episode(self, seat):
                if seat != 3:
                    raise AssertionError(
                        "Do not request baseline trajectories."
                    )
                return {
                    "scores": [40000, 30000, 20000, 10000],
                    "episode": TensorDict({}, [0]),
                }

        with patch("kanachan.simulation.simulate", return_value=[EmptyGame()]):
            data = collect_native(
                device=torch.device("cpu"),
                dtype=torch.float32,
                room=3,
                dongfengzhan=False,
                grade=12,
                baseline_model=nn.Identity(),
                policy_model=nn.Identity(),
                num_simulations=1,
                simulation_batch_size=1,
                simulation_concurrency=1,
                progress=lambda: None,
                get_reward=PlacementReward(),
                seed=29,
            )
        self.assertEqual(len(data), 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.kanachan"
            write_q_model(initial)
            config = Config(
                str(initial),
                str(root / "run"),
                device="cpu",
                amp_dtype="float32",
                iterations=1,
                evaluate_every=0,
            )
            result = run(config, collect_fn=lambda **_: data)
            self.assertEqual(result["iteration"], 1)
            self.assertEqual(result["samples"], 0)
            self.assertEqual(result["optimizer_steps"], 0)

    def test_offline_replay_restores_source_cursor_and_prefetched_batch(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "kanachan.training.core.offline_rl.episode_replay_buffer._INTERNAL_BATCH_SIZE",
                3,
            ),
        ):
            path = Path(directory) / "data.tsv"
            path.write_text(
                "".join(rl_line(i, terminal=i % 4 == 3) for i in range(12)),
                encoding="utf-8",
            )
            arguments = dict(
                training_data=path,
                contiguous_training_data=True,
                rewrite_rooms=None,
                rewrite_grades=None,
                get_reward=unit_reward,
                discount_factor=1.0,
                dtype=torch.float32,
                max_size=4,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
            )
            random.seed(41)
            replay = EpisodeReplayBuffer(num_skip_samples=0, **arguments)
            iterator = replay.iter_with_last()
            next(iterator)
            state = replay.state_dict()
            expected = list(iterator)
            restored = EpisodeReplayBuffer(
                num_skip_samples=state["input_samples"], **arguments
            )
            restored.load_state_dict(state)
            actual = list(restored.iter_with_last())
            self.assertEqual(len(expected), len(actual))
            self.assertGreater(len(actual), 1)
            for (left, left_last), (right, right_last) in zip(
                expected, actual, strict=True
            ):
                self.assertEqual(left_last, right_last)
                for key in left.keys(True, True):
                    self.assertTrue(
                        torch.equal(left[key], right[key]), str(key)
                    )

    def test_double_q_selects_with_source_and_evaluates_with_target(self):
        class Q(nn.Module):
            def __init__(self, values):
                super().__init__()
                self.values = nn.Parameter(
                    torch.tensor(values, dtype=torch.float32)
                )

            def forward(self, data):
                data["action_value"] = self.values.expand(len(data), -1)
                return data

        source = Q([1, 4, 3, 2] + [0] * 28)
        target = Q([100, 20, 300, 400] + [0] * 28).requires_grad_(False)
        data = transitions()[[0, 3]]
        loss, _ = q_loss(source, target, data, 0.0)
        expected = 0.5 * ((1 - 20) ** 2 + (1 - 1.5) ** 2) / 2
        self.assertAlmostEqual(float(loss), expected)
        loss.backward()
        self.assertTrue(torch.isfinite(source.values.grad).all())
        self.assertIsNone(target.values.grad)
        self.assertTrue(source.training)

    def test_cluster_intervals_do_not_count_duplicate_games_as_independent(
        self,
    ):
        games = [ranked_game(0), ranked_game(3)]
        original = summarize(games, group_ids=[0, 1], bootstrap_samples=100)
        duplicated = summarize(
            [game for game in games for _ in range(4)],
            group_ids=[0] * 4 + [1] * 4,
            bootstrap_samples=100,
        )
        self.assertEqual(original["uncertainty"], duplicated["uncertainty"])
        comparison = compare_policies(
            [ranked_game(0)] * 4,
            [ranked_game(3)] * 4,
            group_ids=list(range(4)),
            bootstrap_samples=100,
        )
        self.assertTrue(promotion_decision(comparison, min_groups=4))
        self.assertFalse(promotion_decision(comparison, min_groups=32))
        one = summarize([ranked_game(0)], group_ids=[0], bootstrap_samples=100)
        self.assertIsNone(one["uncertainty"]["proposed_ranking"])

    def test_online_updates_and_resume_match_uninterrupted_training(self):
        self.check_online_updates_and_resume("transformer")

    def test_modern_online_updates_and_resume_match_uninterrupted_training(self):
        self.check_online_updates_and_resume("rmsnorm_swiglu")

    def check_online_updates_and_resume(self, architecture):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.kanachan"
            write_q_model(initial, architecture=architecture)
            options = Config(
                str(initial),
                str(root / "full"),
                device="cpu",
                amp_dtype="float32",
                iterations=2,
                games_per_iteration=1,
                simulation_batch_size=1,
                simulation_concurrency=1,
                learning_starts=1,
                batch_size=2,
                updates_per_iteration=2,
                replay_capacity=12,
                evaluate_every=0,
            )
            run(options, collect_fn=collect_fixture)
            partial = replace(
                options, output_dir=str(root / "partial"), iterations=1
            )
            run(partial, collect_fn=collect_fixture)
            saved = torch.load(root / "partial/latest.pt", weights_only=True)
            run(
                replace(partial, iterations=2),
                resume_state=saved,
                collect_fn=collect_fixture,
            )
            full = torch.load(root / "full/latest.pt", weights_only=True)
            resumed = torch.load(root / "partial/latest.pt", weights_only=True)
            self.assertEqual(full["policy_version"], 2)
            self.assertEqual(resumed["optimizer_steps"], 4)
            for key in full["model"]["state_dict"]:
                self.assertTrue(
                    torch.equal(
                        full["model"]["state_dict"][key],
                        resumed["model"]["state_dict"][key],
                    ),
                    key,
                )
            versions = resumed["replay"]["columns"]["policy_version"]
            self.assertEqual(set(versions.tolist()), {0, 1})
            original = load_model(initial, torch.device("cpu"))
            updated = load_model(
                root / "partial/model.kanachan", torch.device("cpu")
            )
            self.assertTrue(
                any(
                    not torch.equal(a, b)
                    for a, b in zip(
                        original.parameters(), updated.parameters()
                    )
                )
            )
            self.assertTrue(
                torch.equal(full["rng"]["torch"], resumed["rng"]["torch"])
            )

    def test_promotion_is_persisted_and_resume_rejects_algorithm_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.kanachan"
            write_q_model(initial)
            config = Config(
                str(initial),
                str(root / "out"),
                device="cpu",
                amp_dtype="float32",
                iterations=1,
                learning_starts=1,
                batch_size=2,
                updates_per_iteration=1,
                evaluate_every=1,
                evaluation_sets=4,
                min_evaluation_groups=4,
                bootstrap_samples=100,
            )
            calls = []

            def evaluation(*_):
                calls.append(True)
                return [ranked_game(3 if len(calls) == 1 else 0)] * 4, list(
                    range(4)
                )

            run(config, collect_fn=collect_fixture, evaluate_fn=evaluation)
            state = torch.load(root / "out/latest.pt", weights_only=True)
            self.assertEqual(len(state["opponents"]["history"]), 1)
            self.assertTrue((root / "out/best.kanachan").is_file())
            with self.assertRaises(ValueError):
                run(
                    replace(config, td_steps=2),
                    resume_state=state,
                    collect_fn=collect_fixture,
                )

    def test_placement_cql_recipe_resumes_an_intermediate_replay_checkpoint(
        self,
    ):
        self.check_placement_cql_replay_checkpoint("transformer")

    def test_modern_placement_cql_resumes_an_intermediate_replay_checkpoint(self):
        self.check_placement_cql_replay_checkpoint("rmsnorm_swiglu")

    def check_placement_cql_replay_checkpoint(self, architecture):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.tsv"
            data.write_text(
                "".join(rl_line(i, terminal=i % 4 == 3) for i in range(12)),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                "-B",
                "-m",
                "kanachan.training.cql.train",
                "--config-name",
                "placement",
                "device=cpu",
                "encoder=bert_tiny",
                f"encoder.architecture={architecture}",
                "encoder.dimension=8",
                "encoder.num_heads=2",
                "encoder.num_layers=1",
                "encoder.dim_feedforward=16",
                "encoder.dropout=0.1",
                "batch_size=2",
                "replay_buffer_size=4",
                "snapshot_interval=2",
                f"training_data={data.as_posix()}",
            ]
            environment = dict(
                os.environ, OMP_NUM_THREADS="1", TORCHDYNAMO_DISABLE="1"
            )
            first = subprocess.run(
                command + [f"hydra.run.dir={(root / 'first').as_posix()}"],
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            prefix = root / "first/snapshots"
            cursor = json.loads((prefix / "training-state.2.json").read_text())
            self.assertTrue(cursor["data_position_known"])
            self.assertTrue((prefix / "replay-state.2.pth").is_file())
            second = subprocess.run(
                command
                + [
                    f"hydra.run.dir={(root / 'second').as_posix()}",
                    f"initial_model_prefix={prefix.as_posix()}",
                    "initial_model_index=2",
                ],
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            self.assertEqual(
                second.returncode, 0, second.stdout + second.stderr
            )
            for name in (
                "source-encoder.pth",
                "source-decoder.pth",
                "target-encoder.pth",
                "target-decoder.pth",
            ):
                expected = torch.load(prefix / name, weights_only=True)
                actual = torch.load(
                    root / "second/snapshots" / name, weights_only=True
                )
                for key in expected:
                    self.assertTrue(
                        torch.equal(expected[key], actual[key]),
                        name + ":" + key,
                    )


if __name__ == "__main__":
    unittest.main()
