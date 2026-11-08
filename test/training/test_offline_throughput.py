import copy
from contextlib import closing
from pathlib import Path
import random
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from kanachan.training.common import checked_clip_gradients
from kanachan.training.core.offline_rl import EpisodeReplayBuffer
from kanachan.training.core.offline_rl.data_iterator import DataIterator
from kanachan.training.core.rl.discrete_cql import td_components, q_statistics, td_penalty
from kanachan.training.core.rl.rewards import PlacementReward
from kanachan.training.online.offline_pool import prepare_offline_pool
from kanachan.training.online.state import capture_rng
from test_decision_decoder import FixedQ
from test_decision_optimization import compare_state
from test_training_pipeline import rl_line


def replay_arguments(path, **extra):
    return dict(
        training_data=path, contiguous_training_data=True, rewrite_rooms=None,
        rewrite_grades=None, get_reward=PlacementReward(), discount_factor=1.0,
        dtype=torch.float32, max_size=4, batch_size=2, num_workers=0,
        pin_memory=False, **extra,
    )


def source(path, count=12):
    path.write_text("".join(rl_line(i, terminal=i % 4 == 3) for i in range(count)))


class OfflineThroughputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gradient_clipping_matches_native_without_mutating_failures(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        other = copy.deepcopy(model)
        for left, right in zip(model.parameters(), other.parameters(), strict=True):
            left.grad = torch.randn_like(left) * 3
            right.grad = left.grad.clone()
        expected = torch.nn.utils.clip_grad_norm_(other.parameters(), 1.0, error_if_nonfinite=True)
        actual = checked_clip_gradients(model, 1.0)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for left, right in zip(model.parameters(), other.parameters(), strict=True):
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
        parameter = next(model.parameters())
        parameter.grad.fill_(float("inf"))
        before = parameter.detach().clone()
        self.assertIsNone(checked_clip_gradients(model, 1.0))
        torch.testing.assert_close(parameter, before, atol=0, rtol=0)
        for p in model.parameters():
            p.grad = torch.full_like(p, 1e30)
        with self.assertRaisesRegex(RuntimeError, "despite finite gradients"):
            checked_clip_gradients(model, 1.0)

    def test_both_replay_versions_restore_exact_order_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "kanachan.training.core.offline_rl.episode_replay_buffer._INTERNAL_BATCH_SIZE", 3
        ):
            path = Path(temporary) / "rl.tsv"
            source(path)
            for policy, version in (("ordered_v1", 1), ("swap_v2", 2)):
                random.seed(82)
                replay = EpisodeReplayBuffer(num_skip_samples=0, **replay_arguments(path))
                replay._EpisodeReplayBuffer__sampling_policy = policy
                iterator = replay.iter_with_last()
                next(iterator)
                state = replay.state_dict()
                self.assertEqual(state["format_version"], version)
                expected = list(iterator)
                restored = EpisodeReplayBuffer(
                    num_skip_samples=state["input_samples"], **replay_arguments(path)
                )
                restored.load_state_dict(state)
                self.assertEqual(restored.state_dict()["format_version"], version)
                actual = list(restored.iter_with_last())
                self.assertEqual(len(actual), len(expected))
                for (a, last_a), (b, last_b) in zip(actual, expected, strict=True):
                    self.assertEqual(last_a, last_b)
                    compare_state(self, a.to_dict(), b.to_dict())

    def test_swap_sampling_is_uniform_and_without_replacement(self):
        counts = [0] * 20
        replay = object.__new__(EpisodeReplayBuffer)
        replay._EpisodeReplayBuffer__sampling_policy = "swap_v2"
        for seed in range(1000):
            random.seed(seed)
            replay._EpisodeReplayBuffer__replay_buffer = list(range(20))
            selected = [replay._sample() for _ in range(20)]
            self.assertEqual(sorted(selected), list(range(20)))
            counts[selected[0]] += 1
        self.assertTrue(all(25 < count < 80 for count in counts), counts)

    def test_cache_cold_and_warm_preserve_samples_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "kanachan.training.core.offline_rl.episode_replay_buffer._INTERNAL_BATCH_SIZE", 3
        ):
            path, cache = Path(temporary) / "rl.tsv", Path(temporary) / "cache.sqlite3"
            source(path)

            def consume(cached):
                random.seed(82)
                torch.manual_seed(991)
                reader = EpisodeReplayBuffer(
                    num_skip_samples=0, **replay_arguments(path),
                    data_cache=cache if cached else None, data_cache_bytes=256 * 1024,
                )
                batches = [(batch.to_dict(), last) for batch, last in reader.iter_with_last()]
                return batches, capture_rng(), reader.cache_statistics

            plain, rng, _ = consume(False)
            cold, cold_rng, stats = consume(True)
            compare_state(self, plain, cold)
            compare_state(self, rng, cold_rng)
            self.assertGreater(stats["writes"], 0)
            with patch.object(DataIterator, "__next__", side_effect=AssertionError("unexpected raw parsing")):
                warm, warm_rng, stats = consume(True)
            compare_state(self, plain, warm)
            compare_state(self, rng, warm_rng)
            self.assertGreater(stats["hits"], 0)
            self.assertEqual(stats["misses"], 0)
            self.assertLessEqual(cache.stat().st_size, 256 * 1024)
            # A cache is never accepted after its source or reader identity changes.
            source(path, 16)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                consume(True)

    def test_cache_budget_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "kanachan.training.core.offline_rl.episode_replay_buffer._INTERNAL_BATCH_SIZE", 3
        ):
            path, cache = Path(temporary) / "rl.tsv", Path(temporary) / "cache.sqlite3"
            source(path, 100)
            args = replay_arguments(path, data_cache=cache, data_cache_bytes=128 * 1024)
            first = EpisodeReplayBuffer(num_skip_samples=0, **args)
            expected = [batch.to_dict() for batch in first]
            self.assertLessEqual(cache.stat().st_size, 128 * 1024)
            self.assertGreater(first.cache_statistics["misses"], first.cache_statistics["writes"])
            # Fill/eviction policy cannot silently truncate the training source.
            self.assertEqual(sum(len(batch["action"]) for batch in expected), 100)
            with closing(sqlite3.connect(cache)) as connection:
                with connection:
                    connection.execute("UPDATE blocks SET payload=? WHERE start=0", (b"corrupted",))
            restored = EpisodeReplayBuffer(num_skip_samples=0, **args)
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                next(restored)

    def test_completed_pool_reuse_skips_parsing_and_preserves_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "rl.tsv"
            source(path)
            config = SimpleNamespace(offline_data=str(path), offline_capacity=5,
                                     seed=28, output_dir=str(root / "first"))
            original, identity = prepare_offline_pool(config)
            config.offline_pool_cache = str(root / "first/offline-pool.pt")
            config.output_dir = str(root / "second")
            with patch("kanachan.training.online.offline_pool.DataLoader", side_effect=AssertionError("unexpected parsing")):
                restored, restored_identity = prepare_offline_pool(config)
            self.assertEqual(identity, restored_identity)
            compare_state(self, original.state_dict(), restored.state_dict())
            self.assertFalse((root / "second").exists())
            config.seed += 1
            with self.assertRaisesRegex(ValueError, "identity changed"):
                prepare_offline_pool(config)

    def test_q_monitor_masks_padding_and_does_not_change_gradients(self):
        values = torch.tensor([[0., 0., 999.], [0., 999., 999.]], requires_grad=True)
        chosen = values[:, 0]
        candidates = torch.tensor([[1, 2, 546], [2, 546, 546]])
        observed = torch.tensor([1., -1.])
        metrics = q_statistics(values, chosen, candidates, logged_return=observed)
        self.assertEqual(float(metrics["Q/CQL gap centered"]), 0.)
        self.assertEqual(float(metrics["Q/Logged return MAE"]), 1.)
        self.assertEqual(float(metrics["Q/Logged return bias"]), 0.)
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))
        chosen.sum().backward()
        torch.testing.assert_close(values.grad, torch.tensor([[1., 0., 0.], [1., 0., 0.]]))

    def test_actual_cql_cache_and_monitor_are_training_transparent(self):
        from kanachan.training.cql.train import _training

        with tempfile.TemporaryDirectory() as temporary, patch(
            "kanachan.training.core.offline_rl.episode_replay_buffer._INTERNAL_BATCH_SIZE", 3
        ):
            path, cache = Path(temporary) / "rl.tsv", Path(temporary) / "cache.sqlite3"
            source(path)

            def train(cached, monitor):
                random.seed(791)
                torch.manual_seed(512)
                network = FixedQ([0.1, 0.2, 0.3, 0.4] + [0.] * 28)
                target = copy.deepcopy(network).requires_grad_(False).eval()
                optim = torch.optim.SGD(network.parameters(), lr=0.01)
                writer, snapshots = Mock(), []

                def td(data):
                    values, chosen, expected = td_components(network, target, data)
                    data["action_value"], data["td_target"] = values, expected
                    data["td_error"] = 2 * td_penalty(chosen, expected, "huber")

                def snapshot(index, **state):
                    if index is None:
                        snapshots.append(state)

                _training(
                    training_data=path, contiguous_training_data=True,
                    rewrite_rooms=None, rewrite_grades=None, num_workers=0,
                    replay_buffer_size=4, device=torch.device("cpu"),
                    dtype=torch.float32, amp_dtype=torch.float32,
                    source_network=network, target_network=target,
                    reward_plugin=None, discount_factor=1., td_error_computation=td,
                    alpha=0.1, batch_size=2, gradient_accumulation_steps=1,
                    max_gradient_norm=1., optimizer=optim, scheduler=None,
                    target_update_interval=1, target_update_rate=0.005,
                    snapshot_interval=0, num_samples=0, summary_writer=writer,
                    snapshot_writer=snapshot, q_monitor_interval=monitor,
                    data_cache=cache if cached else None, data_cache_bytes=256 * 1024,
                )
                return network.state_dict(), target.state_dict(), optim.state_dict(), snapshots, capture_rng(), writer

            ordinary = train(False, 0)
            for candidate in (train(True, 1), train(True, 1)):
                compare_state(self, ordinary[:-1], candidate[:-1])
                tags = {call.args[0] for call in candidate[-1].add_scalar.call_args_list}
                self.assertTrue({"Q/P05", "Q/CQL gap centered", "Q/Logged return MAE"} <= tags)


if __name__ == "__main__":
    unittest.main()
