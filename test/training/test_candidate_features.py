import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch
from tensordict import TensorDict

from kanachan.candidate_features import (
    CHI,
    PON,
    OFFSETS_136,
    FEATURE_WIDTH,
    _cached_row,
    derive_candidate_features,
    observed_tiles,
    tile34,
)
from kanachan.model_loader import load_model
from kanachan.nn.decision_decoder import GlobalActionHead, legal_argmax
from kanachan.training.core.bc import DataLoader


HAND = [1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 31, 31, 32]


def sparse_tokens(hand=HAND, draw=32, indicator=9):
    result = [3, 6, 7, 23, 39, 55, 71, 75, 78, 142, 152 + indicator]
    for tile in range(37):
        result.extend(
            337 + OFFSETS_136[tile] + i for i in range(hand.count(tile))
        )
    if draw is not None:
        result.append(473 + draw)
    return result


def observation(
    hand=HAND, draw=32, history=(0,), actions=(4, 128, 219), indicator=9
):
    sparse = sparse_tokens(hand, draw, indicator)
    return TensorDict(
        {
            "sparse": torch.tensor(
                [sparse + [510] * (29 - len(sparse))], dtype=torch.int32
            ),
            "numeric": torch.tensor(
                [[0, 0, 25000, 25000, 25000, 25000]], dtype=torch.int32
            ),
            "progression": torch.tensor(
                [list(history) + [2165] * (109 - len(history))],
                dtype=torch.int32,
            ),
            "candidates": torch.tensor(
                [list(actions) + [546] * (32 - len(actions))],
                dtype=torch.int32,
            ),
        },
        [1],
    )


def training_line(index=0):
    return (
        "\t".join(
            (
                f"game-{index}",
                ",".join(map(str, sparse_tokens())),
                "0,0,25000,25000,25000,25000",
                "0",
                "4,128,219",
                str(index % 2),
                "0",
                ",".join(map(str, [0] * 10 + [25000] * 4)),
            )
        )
        + "\n"
    )


@unittest.skipUnless(
    importlib.util.find_spec("kanachan._candidate_features") is not None,
    "Build tools/features to enable native afterstate checks",
)
class CandidateFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def features(self, data):
        return derive_candidate_features(
            data["sparse"], data["progression"], data["candidates"]
        )

    def test_known_tenpai_live_tiles_dora_and_non_discard_mask(self):
        values = self.features(observation())[0]
        self.assertEqual(values.shape, (32, FEATURE_WIDTH))
        self.assertEqual(values[1, 0].item(), 1)
        self.assertEqual(values[1, 1].item(), 0)
        self.assertEqual(values[1, 2].item(), 1)
        self.assertAlmostEqual(values[1, 3].item() * 136, 2, places=5)
        self.assertEqual(values[1, 5:39].nonzero().flatten().tolist(), [29])
        self.assertAlmostEqual(values[0, 3].item() * 136, 7, places=5)
        self.assertAlmostEqual(values[1, 42].item() * 70, 1, places=5)
        self.assertEqual(values[0, 42].item(), 0)
        self.assertAlmostEqual(values[0, 44].item() * 5, 1, places=5)
        self.assertEqual(values[2:].count_nonzero().item(), 0)
        changed = observation(history=(0, 5 + 148 + 4))
        self.assertAlmostEqual(
            self.features(changed)[0, 0, 3].item() * 136, 6, places=5
        )

    def test_called_tiles_count_once_and_kan_upgrade_adds_only_one_tile(self):
        after_chi = [11, 12, 13, 21, 22, 23, 31, 31, 31, 32, 32]
        data = observation(
            after_chi, None, (0, 5 + 3 * 148 + 3 * 4, 597 + 3), (128,)
        )
        hand, visible, dora, melds = observed_tiles(
            sparse_tokens(after_chi, None), [0, 461, 600]
        )
        self.assertEqual(melds, 1)
        self.assertEqual(visible[:3], [1, 1, 1])
        self.assertEqual(self.features(data)[0, 0, 2].item(), 1)
        history = (0, 5 + 2 * 148 + 5 * 4, 957 + 120 + 5, 2017 + 37 + 5)
        self.assertEqual(observed_tiles(sparse_tokens(), history)[1][4], 4)
        open_kan = (0, 5 + 2 * 148 + 5 * 4, 1437 + 111 + 5)
        self.assertEqual(observed_tiles(sparse_tokens(), open_kan)[1][4], 4)
        closed_kan = (0, 1881 + 34 + 27)
        self.assertEqual(observed_tiles(sparse_tokens(), closed_kan)[1][27], 4)

    def test_red_retention_and_invalid_observations(self):
        hand = [0] + HAND[1:]
        data = observation(hand, actions=(0, 128))
        values = self.features(data)[0]
        self.assertEqual(values[0, 39].item(), 0)
        self.assertEqual(values[1, 39].item(), 1)
        self.assertEqual(values[0, 43].item(), 1)
        missing_indicator = observation()
        missing_indicator["sparse"][0, 10] = 510
        missing_wind = observation()
        missing_wind["sparse"][0, 7] = 510
        skipped_indicator_slot = observation()
        skipped_indicator_slot["sparse"][0, 10] += 37
        for data in (
            observation(HAND[:-2]),
            observation(actions=(36,)),
            observation(history=(0, 5 + 148, 5 + 296)),
            missing_indicator,
            missing_wind,
            skipped_indicator_slot,
        ):
            with self.assertRaises(ValueError):
                self.features(data)

    def test_encoding_tables_match_native_simulator_tables(self):
        import re

        path = (
            Path(__file__).resolve().parents[2] / "src/simulation/shoupai.cpp"
        )
        # Full source check is optional in the lean Python training bundle.
        if not path.is_file():
            self.skipTest(
                "Full simulator source is not in this training bundle"
            )
        text = path.read_text(encoding="utf-8")
        for name, rows, width in (
            ("chi_encode_", CHI, 4),
            ("peng_encode_", PON, 3),
        ):
            block = text.split(name + " = {", 1)[1].split("};", 1)[0]
            parsed = [
                tuple(map(int, match))
                for match in re.findall(
                    r"\{\s*(\d+)u,\s*(\d+)u,\s*(\d+)u", block
                )
            ]
            self.assertEqual(list(rows), parsed)

    def test_random_discard_shapes_and_ukeire_match_independent_python_shanten(
        self,
    ):
        from mahjong.shanten import Shanten

        rng = random.Random(73)
        deck = [
            tile
            for tile in range(37)
            for _ in range(
                1
                if tile < 30 and tile % 10 == 0
                else 3
                if tile < 30 and tile % 10 == 5
                else 4
            )
        ]
        for _ in range(12):
            tiles = rng.sample(range(136), 15)
            hand = [deck[i] for i in tiles[:13]]
            draw, indicator = deck[tiles[13]], deck[tiles[14]]
            actions = tuple(
                tile * 4 for tile in sorted(set(hand + [draw]))[:5]
            )
            data = observation(
                hand, draw, actions=actions, indicator=indicator
            )
            values = self.features(data)[0]
            counts, visible, _, _ = observed_tiles(
                sparse_tokens(hand, draw, indicator), [0]
            )
            for i, action in enumerate(actions):
                after = [0] * 34
                for tile, count in enumerate(counts):
                    after[tile34(tile)] += count
                after[tile34(action // 4)] -= 1
                expected = Shanten().calculate_shanten(after)
                self.assertAlmostEqual(values[i, 1].item() * 8, expected)
                effective = []
                for tile in range(34):
                    if visible[tile] == 4:
                        continue
                    after[tile] += 1
                    if Shanten().calculate_shanten(after) < expected:
                        effective.append(tile)
                    after[tile] -= 1
                self.assertEqual(
                    values[i, 5:39].nonzero().flatten().tolist(), effective
                )
                self.assertAlmostEqual(
                    values[i, 3].item() * 136,
                    sum(4 - visible[tile] for tile in effective),
                    places=4,
                )

    def test_shared_scorer_permutation_gradients_and_immutable_cache(self):
        data = observation()
        original = self.features(data)
        original.zero_()
        self.assertGreater(self.features(data).count_nonzero().item(), 0)
        head = GlobalActionHead(
            input_dimension=16,
            hidden_dimension=16,
            device=torch.device("cpu"),
            dtype=torch.float32,
            candidate_features="discard_v1",
        )
        torch.nn.init.normal_(head.candidate_layers[-1].weight, std=0.02)
        representation = torch.randn(1, 16, requires_grad=True)
        expected = head(
            data["candidates"],
            representation,
            data["sparse"],
            data["progression"],
        )
        order = torch.tensor([2, 0, 1] + list(range(3, 32)))
        permuted = head(
            data["candidates"][:, order],
            representation,
            data["sparse"][:, torch.arange(28, -1, -1)],
            data["progression"],
        )
        torch.testing.assert_close(permuted, expected[:, order])
        expected[:, :3].sum().backward()
        self.assertGreater(
            head.candidate_embedding.weight.grad.abs().sum().item(), 0
        )
        self.assertGreater(
            head.candidate_layers[0].weight.grad.abs().sum().item(), 0
        )
        self.assertTrue(
            torch.equal(
                legal_argmax(data["candidates"], expected),
                order[
                    legal_argmax(data["candidates"][:, order], permuted).long()
                ].int(),
            )
        )
        self.assertGreater(_cached_row.cache_info().hits, 0)

    def test_old_global_policy_initializes_exactly_before_correction_training(
        self,
    ):
        from kanachan.nn.decision_factory import load_candidate_head_state

        kwargs = dict(
            input_dimension=16,
            hidden_dimension=16,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        teacher = GlobalActionHead(**kwargs)
        candidate = GlobalActionHead(**kwargs, candidate_features="discard_v1")
        load_candidate_head_state(candidate, teacher.state_dict())
        values = observation()
        representation = torch.randn(1, 16)
        torch.testing.assert_close(
            teacher(values["candidates"], representation),
            candidate(
                values["candidates"],
                representation,
                values["sparse"],
                values["progression"],
            ),
            atol=0,
            rtol=0,
        )
        with self.assertRaises(RuntimeError):
            load_candidate_head_state(teacher, candidate.state_dict())

    def test_bc_candidate_features_transfer_without_policy_outputs_as_q(self):
        from tensordict.nn import TensorDictModule, TensorDictSequential
        from kanachan.nn.decision_decoder import DenseDecisionDecoder
        from kanachan.nn.decision_value import transfer_bc_features

        body_args = dict(
            dimension=16,
            num_heads=2,
            dim_feedforward=24,
            num_layers=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        head_args = dict(
            input_dimension=16,
            hidden_dimension=16,
            device=torch.device("cpu"),
            dtype=torch.float32,
            candidate_features="discard_v1",
        )
        teacher_body, teacher_head = (
            DenseDecisionDecoder(**body_args),
            GlobalActionHead(**head_args),
        )
        torch.nn.init.normal_(
            teacher_head.candidate_layers[-1].weight, std=0.1
        )
        teacher = TensorDictSequential(
            TensorDictModule(
                teacher_body,
                ["sparse", "numeric", "progression", "candidates"],
                ["encode"],
            ),
            TensorDictModule(
                teacher_head,
                ["candidates", "encode", "sparse", "progression"],
                ["decode"],
            ),
        )
        body, head = (
            DenseDecisionDecoder(**body_args),
            GlobalActionHead(**head_args),
        )
        transfer_bc_features(body, head, teacher)
        torch.testing.assert_close(
            head.candidate_layers[0].weight,
            teacher_head.candidate_layers[0].weight,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            head.candidate_embedding.weight,
            teacher_head.candidate_embedding.weight,
            rtol=0,
            atol=0,
        )
        batch = observation()
        values = head(
            batch["candidates"],
            body(
                *(
                    batch[key]
                    for key in (
                        "sparse",
                        "numeric",
                        "progression",
                        "candidates",
                    )
                )
            ),
            batch["sparse"],
            batch["progression"],
        )
        self.assertEqual(values.count_nonzero().item(), 0)


@unittest.skipUnless(
    importlib.util.find_spec("kanachan._candidate_features") is not None,
    "Build tools/features for training integration",
)
class CandidateTrainingTests(unittest.TestCase):
    def pipeline(self, device):
        import test_decision_iql as fixtures

        helper = fixtures.DecisionIQLTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "decisions.tsv"
            data.write_text(
                "".join(training_line(i) for i in range(6)), encoding="utf-8"
            )
            rl_data = root / "transitions.tsv"
            transitions = []
            for i in range(6):
                fields = training_line(i).rstrip().split("\t")
                fields[0] = f"episode-{i // 3}"
                if i % 3 != 2:
                    fields = fields[:6] + fields[1:5]
                transitions.append("\t".join(fields) + "\n")
            rl_data.write_text("".join(transitions), encoding="utf-8")
            extra = ["decoder.candidate_features=discard_v1"]
            if device == "cuda":
                extra += ["device.amp_dtype=bfloat16"]
            helper.cli(
                "bc",
                data,
                root / "old-bc",
                *[
                    item
                    for item in extra
                    if not item.startswith("decoder.candidate_features=")
                ],
                device=device,
            )
            helper.cli(
                "bc",
                data,
                root / "bc",
                *extra,
                "snapshot_interval=2",
                f"encoder.load_from={(root / 'old-bc/snapshots/encoder.pth').as_posix()}",
                f"decoder.load_from={(root / 'old-bc/snapshots/decoder.pth').as_posix()}",
                device=device,
            )
            helper.cli(
                "bc",
                data,
                root / "bc-resume",
                *extra,
                "snapshot_interval=2",
                f"initial_model_prefix={(root / 'bc/snapshots').as_posix()}",
                "initial_model_index=2",
                device=device,
            )
            for name in ("encoder", "decoder", "optimizer"):
                helper.assert_state_equal(
                    torch.load(
                        root / f"bc/snapshots/{name}.pth",
                        map_location="cpu",
                        weights_only=True,
                    ),
                    torch.load(
                        root / f"bc-resume/snapshots/{name}.pth",
                        map_location="cpu",
                        weights_only=True,
                    ),
                )
            reward = (
                Path(__file__).resolve().parents[2]
                / "examples/training/iql-placement-reward.py"
            )
            helper.cli(
                "iql",
                rl_data,
                root / "iql",
                *extra,
                f"reward_plugin={reward.as_posix()}",
                f"bc_initial_model={(root / 'bc/snapshots/model.kanachan').as_posix()}",
                device=device,
            )
            helper.cli(
                "awr",
                data,
                root / "awr",
                *extra,
                "weight_diagnostics_interval=1",
                f"q_model={(root / 'iql/snapshots/model.kanachan').as_posix()}",
                f"value_model={(root / 'iql/snapshots/value-model.kanachan').as_posix()}",
                f"encoder.load_from={(root / 'bc/snapshots/encoder.pth').as_posix()}",
                f"decoder.load_from={(root / 'bc/snapshots/decoder.pth').as_posix()}",
                device=device,
            )
            rows = [
                json.loads(line)
                for line in (root / "awr/awr-weight-diagnostics.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertTrue(
                all(
                    0 < row["metrics"]["Weight/ESS fraction"] <= 1
                    for row in rows
                )
            )
            for name in ("bc", "iql", "awr"):
                model = load_model(
                    root / name / "snapshots/model.kanachan"
                ).eval()
                batch = observation()
                model(batch)
                self.assertEqual(batch["action"].shape, (1,))
            if device == "cpu":
                for mode in ("mc", "td"):
                    args = [*extra, f"target_kind={mode}"]
                    if mode == "td":
                        args += [
                            "double_q_learning=true",
                            "export_network=target",
                        ]
                    helper.cli(
                        "cql",
                        rl_data,
                        root / mode,
                        *args,
                        f"bc_initial_model={(root / 'bc/snapshots/model.kanachan').as_posix()}",
                        device=device,
                    )
                    model = load_model(
                        root / mode / "snapshots/model.kanachan"
                    ).eval()
                    model(observation())
            manifest = json.loads(
                (root / "bc/snapshots/decision-checkpoint.json").read_text()
            )
            self.assertIn("derived_candidates", manifest["recipe"]["schema"])

    def test_cpu_bc_iql_awr_mc_cql_td_cql_and_resume(self):
        self.pipeline("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_bfloat16_bc_iql_awr_and_resume(self):
        self.pipeline("cuda")


if __name__ == "__main__":
    unittest.main()
