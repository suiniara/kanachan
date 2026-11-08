import copy
from pathlib import Path
import tempfile
import unittest

import torch

from kanachan.training.core.offline_rl import EpisodeReplayBuffer
from kanachan.training.core.rl.discrete_cql import (
    mc_components, scalar_cql_loss, q_statistics,
)
from test_decision_decoder import FixedQ
import test_decision_decoder as fixtures
from test_training_pipeline import rl_line


class MonteCarloCQLTests(unittest.TestCase):
    def data(self):
        data = fixtures.DiscreteCQLTests().data()
        data["mc_return"] = torch.tensor([.5, -1.5])
        return data

    def test_fixed_return_target_never_queries_next_states_or_target_network(self):
        source = FixedQ([20.] * 32)
        data = self.data()
        del data["next"]
        data["action_value"] = torch.full((2,), float("nan"))
        data["mc_return"].requires_grad_(True)
        _, _, first = mc_components(source, data)
        self.assertEqual(source.calls, [2])
        self.assertFalse(first.requires_grad)
        with torch.no_grad():
            source.values.add_(1000.)
        _, _, second = mc_components(source, data)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        torch.testing.assert_close(second, torch.tensor([.5, -1.5]))
        with self.assertRaisesRegex(ValueError, "no target network"):
            scalar_cql_loss(source, copy.deepcopy(source), data, .5, target_kind="mc")

    def test_mc_objective_and_gradients_match_fixed_label_formula(self):
        source = FixedQ([3., 1.] + [100.] * 30)
        data = self.data()
        loss, metrics = scalar_cql_loss(source, None, data, .5, target_kind="mc")
        loss.backward()
        q = torch.tensor([3., 1.], requires_grad=True)
        expected = .5 * (q - torch.tensor([.5, -1.5])).square().mean()
        expected += .5 * (q.logsumexp(0) - q.mean())
        expected.backward()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(source.values.grad[:2], q.grad)
        self.assertEqual(float(source.values.grad[2:].abs().sum()), 0.)
        self.assertIn("return_loss", metrics)
        self.assertNotIn("td_loss", metrics)

    def test_large_common_offset_now_has_correction_and_recovers(self):
        data = self.data()
        source = FixedQ([20.] * 32)
        optimizer = torch.optim.SGD(source.parameters(), lr=.2)
        initial, _ = scalar_cql_loss(source, None, data, .5, target_kind="mc")
        for _ in range(128):
            optimizer.zero_grad()
            loss, _ = scalar_cql_loss(source, None, data, .5, target_kind="mc")
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss.detach()), float(initial.detach()) / 100)
        self.assertAlmostEqual(float(source.values[:2].mean().detach()), -.5, places=3)
        _, _, target = mc_components(source, data)
        torch.testing.assert_close(target, data["mc_return"], atol=0, rtol=0)

    def test_missing_or_nonfinite_labels_and_unknown_mode_rejected(self):
        source = FixedQ([0.] * 32)
        data = self.data()
        del data["mc_return"]
        with self.assertRaisesRegex(ValueError, "complete-trajectory"):
            mc_components(source, data)
        data["mc_return"] = torch.tensor([float("nan"), 1.])
        with self.assertRaisesRegex(ValueError, "Nonfinite MC return"):
            mc_components(source, data)
        with self.assertRaisesRegex(ValueError, "target_kind"):
            scalar_cql_loss(source, None, self.data(), .5, target_kind="mixed")

    def test_mc_metric_names_describe_return_regression(self):
        data = self.data()
        source = FixedQ([2., 9.] + [0.] * 30)
        values, chosen, target = mc_components(source, data)
        stats = q_statistics(values, chosen, data["candidates"], target=target,
                             done=data["next", "done"], target_kind="mc")
        self.assertIn("MC/Residual mean", stats)
        self.assertNotIn("TD/Residual mean", stats)

    def test_raw_settlement_reference_matches_known_complete_returns(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        from verify_mc_cql import observed_return_reference, row_key
        from test_round_rewards import fixture
        rows, results = [], fixture()["next", "results"]
        for i in range(6):
            columns = rl_line(i, terminal=i == 5).rstrip().split("\t")
            if i in (1, 3):
                columns += ["0", ",".join(map(str, results[i, :10].tolist()))]
            elif i == 5:
                columns[6:] = ["0", ",".join(map(str, results[i].tolist()))]
            rows.append(columns)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "complete.tsv"
            path.write_text("".join("\t".join(row)+"\n" for row in rows))
            reference = observed_return_reference(path)
            for row, expected in zip(rows, (-1.6, -1.6, -1.74, -1.74, -1.68, -1.68), strict=True):
                key = row_key(*(list(map(int, value.split(","))) for value in row[1:5]), row[5])
                self.assertAlmostEqual(reference[key][0], expected, places=6)

    def test_checkpoint_rejects_mc_td_target_change(self):
        import json
        from types import SimpleNamespace
        from kanachan.training.core.decision_checkpoint import validate_resume
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SimpleNamespace(initial_model_prefix=root, initial_model_index=None,
                                     optimizer=SimpleNamespace(initialize=False), start_new_epoch=False)
            (root / "decision-checkpoint.json").write_text(json.dumps({
                "recipe": {"training": {"target_kind": "td"}}, "files": {}}))
            with self.assertRaisesRegex(ValueError, "recipe"):
                validate_resume(config, {"training": {"target_kind": "mc"}})

    def test_complete_returns_survive_shuffling_batches_and_output_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "games.tsv"
            # The first complete observer game crosses the raw 1024-row batch.
            path.write_text("".join(rl_line(i, terminal=i in (1029, 1032))
                                    for i in range(1033)), encoding="utf-8")

            def observed_reward(data, contiguous):
                self.assertTrue(contiguous)
                ids = data["numeric"][:, 0].long()
                data["next", "reward"] = torch.where(ids < 1030, 1., -2.)

            reader = EpisodeReplayBuffer(
                training_data=path, contiguous_training_data=True, num_skip_samples=0,
                rewrite_rooms=None, rewrite_grades=None, get_reward=observed_reward,
                discount_factor=1., dtype=torch.float32, max_size=64, batch_size=17,
                num_workers=0, pin_memory=False,
            )
            actual = {}
            for batch in reader:
                ids = batch["numeric"][:, 0].long().tolist()
                labels = batch["mc_return"].clone()
                batch["action_value"].fill_(float("nan"))
                torch.testing.assert_close(batch["mc_return"], labels, atol=0, rtol=0)
                actual.update(zip(ids, labels.tolist(), strict=True))
            self.assertEqual(len(actual), 1033)
            for index, value in actual.items():
                self.assertEqual(value, 1030-index if index < 1030 else -2*(1033-index))

    def test_training_loop_rejects_bootstrap_and_incomplete_input(self):
        from kanachan.training.cql.train import _training
        from unittest.mock import Mock
        arguments = dict(
            training_data=Path("does-not-exist"), contiguous_training_data=True,
            rewrite_rooms=None, rewrite_grades=None, num_workers=0, replay_buffer_size=4,
            device=torch.device("cpu"), dtype=torch.float32, amp_dtype=torch.float32,
            source_network=FixedQ([0.] * 32), target_network=None, reward_plugin=None,
            discount_factor=1., td_error_computation=None, alpha=.5, batch_size=2,
            gradient_accumulation_steps=1, max_gradient_norm=1., optimizer=Mock(),
            scheduler=None, target_update_interval=1, target_update_rate=.005,
            snapshot_interval=0, num_samples=0, summary_writer=Mock(), snapshot_writer=Mock(),
            target_kind="mc",
        )
        for changes in (dict(target_network=FixedQ([0.]*32)),
                        dict(contiguous_training_data=False),
                        dict(reward_model=Path("predictor")), dict(replay_buffer_size=0)):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "complete episode replay"):
                _training(**(arguments | changes))


if __name__ == "__main__":
    unittest.main()
