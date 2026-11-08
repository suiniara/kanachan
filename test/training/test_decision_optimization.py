import copy
from dataclasses import replace
import io
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential

from kanachan.nn.decode_converter import DecodeConverter
from kanachan.nn.decision_decoder import legal_actions, rotary_embeddings
from kanachan.nn.decision_value import DecisionQNetwork, q_only
from kanachan.training.core.config.optimizer import AdamWOptimizerConfig, create
from kanachan.training.core.optimizers import (
    ADAMW_POLICY, LEGACY_ADAMW_POLICY, online_adamw,
)
from kanachan.training.online import offline_pool
from kanachan.training.online.reservoir import TransitionReservoir
from kanachan.training.core.rl.discrete_cql import scalar_cql_loss
from test_decision_decoder import backbone, build_q, observation
from test_decision_decoder import FixedQ
import test_decision_decoder as decoder_fixtures
from test_online_training import transitions


def rows(count):
    data = torch.cat([transitions()] * ((count + 3) // 4), 0)[:count]
    data["numeric"][:, 0] = torch.arange(count)
    data["next", "discount"] = torch.ones(count)
    return offline_pool.training_columns(data)


def compare_state(test, left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            compare_state(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(type(left), type(right))
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right, strict=True):
            compare_state(test, a, b)
    else:
        test.assertEqual(left, right)


class OptimizedOptimizerTests(unittest.TestCase):
    def test_offline_online_groups_and_decay_are_identical(self):
        model = build_q()
        config = AdamWOptimizerConfig()
        offline, _ = create("cpu", config, model)
        online, policy = online_adamw(
            model, lr=config.learning_rate, weight_decay=config.weight_decay,
            device_type="cpu",
        )
        self.assertEqual(policy, ADAMW_POLICY)
        for a, b in zip(offline.param_groups, online.param_groups, strict=True):
            self.assertEqual(a["weight_decay"], b["weight_decay"])
            self.assertEqual([id(p) for p in a["params"]], [id(p) for p in b["params"]])
        embedding_ids = {
            id(p) for layer in model.modules() if isinstance(layer, torch.nn.Embedding)
            for p in layer.parameters(recurse=False)
        }
        for group in online.param_groups:
            for p in group["params"]:
                self.assertEqual(group["weight_decay"] != 0,
                                 p.ndim >= 2 and id(p) not in embedding_ids)
        # Zero gradients expose decoupled decay, without mixing in Adam moments.
        before = {id(p): p.detach().clone() for p in model.parameters()}
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        online.step()
        for group in online.param_groups:
            for p in group["params"]:
                torch.testing.assert_close(
                    p, before[id(p)] * (1 - config.learning_rate * group["weight_decay"]),
                    atol=0, rtol=0,
                )

    def test_new_and_legacy_adamw_moments_resume_exactly(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.LayerNorm(3))
                if legacy:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
                else:
                    optimizer, _ = online_adamw(model, lr=0.01, weight_decay=0.1, device_type="cpu")
                model(torch.tensor([[0.2, 0.4]])).square().sum().backward()
                optimizer.step()
                checkpoint = {"optimizer": copy.deepcopy(optimizer.state_dict())}
                if not legacy:
                    checkpoint["optimizer_policy"] = ADAMW_POLICY
                restored = copy.deepcopy(model)
                second, policy = online_adamw(
                    restored, lr=0.01, weight_decay=0.1, device_type="cpu",
                    resume_state=checkpoint,
                )
                self.assertEqual(policy, LEGACY_ADAMW_POLICY if legacy else ADAMW_POLICY)
                for network, optim in ((model, optimizer), (restored, second)):
                    optim.zero_grad(set_to_none=True)
                    network(torch.tensor([[0.3, -0.2]])).square().sum().backward()
                    optim.step()
                compare_state(self, model.state_dict(), restored.state_dict())
                compare_state(self, optimizer.state_dict(), second.state_dict())


class ReservoirTests(unittest.TestCase):
    def test_matches_sequential_algorithm_r_across_batch_sizes_and_resume(self):
        data, capacity, seed = rows(101), 9, 7841
        rng = random.Random(seed)
        expected = list(range(capacity))
        for row in range(capacity, len(data)):
            slot = rng.randrange(row + 1)
            if slot < capacity:
                expected[slot] = row
        for width in (1, 4, 17, 101):
            reservoir = TransitionReservoir(capacity, seed, 98)
            for offset in range(0, len(data), width):
                reservoir.consume(data[offset:offset + width])
            self.assertEqual(reservoir.indices.tolist(), expected)
            self.assertEqual(reservoir.replay.columns["numeric"][:, 0].tolist(), expected)
        partial = TransitionReservoir(capacity, seed, 98)
        partial.consume(data[:38])
        serialized = io.BytesIO()
        torch.save(partial.state_dict(), serialized)
        serialized.seek(0)
        resumed = TransitionReservoir(capacity, 0, 0)
        resumed.load_state_dict(torch.load(serialized, weights_only=True))
        resumed.consume(data[38:])
        compare_state(self, reservoir.state_dict(), resumed.state_dict())
        compare_state(self, reservoir.replay.sample(12).to_dict(), resumed.replay.sample(12).to_dict())

    def test_selection_covers_tail_and_has_no_prefix_bias(self):
        data = rows(40)
        counts = torch.zeros(40, dtype=torch.int64)
        for seed in range(800):
            reservoir = TransitionReservoir(5, seed, 42)
            reservoir.consume(data)
            counts[reservoir.indices] += 1
        # Expected inclusion 100 per row. This deterministic population check
        # would immediately reject prefix-only or suffix-only implementations.
        self.assertTrue(((counts > 65) & (counts < 140)).all(), counts.tolist())

    def test_initial_build_interruption_resumes_cursor_and_rng(self):
        data = rows(47)
        starts = []
        interrupt = False

        def reader(**kwargs):
            nonlocal interrupt
            start = kwargs["num_skip_samples"]
            starts.append(start)
            for offset in range(start, len(data), 7):
                if interrupt and offset == 14:
                    interrupt = False
                    raise InterruptedError("simulated process interruption")
                yield data[offset:offset + 7].clone()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "train.tsv"
            source.write_text("source identity fixture")
            config = SimpleNamespace(offline_data=str(source), offline_capacity=9,
                                     seed=28, output_dir=str(root / "full"))
            with patch.object(offline_pool, "DataLoader", side_effect=reader), patch.object(
                offline_pool, "_CHECKPOINT_SECONDS", 0
            ):
                rng = torch.get_rng_state().clone()
                full, sha = offline_pool.prepare_offline_pool(config)
                torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
                config.output_dir = str(root / "partial")
                interrupt = True
                with self.assertRaises(InterruptedError):
                    offline_pool.prepare_offline_pool(config)
                self.assertTrue(offline_pool.is_pool_preparation(config.output_dir))
                resumed, restored_sha = offline_pool.prepare_offline_pool(config)
                self.assertEqual(starts, [0, 0, 14])
                self.assertEqual(sha, restored_sha)
                compare_state(self, full.state_dict(), resumed.state_dict())
                self.assertEqual(resumed.metadata["source_rows"], 47)
                self.assertGreater(resumed.metadata["source_indices"].max().item(), 30)
                config.seed += 1
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    offline_pool.prepare_offline_pool(config)

    def test_online_entry_can_continue_an_interrupted_initial_pool(self):
        from kanachan.training.online.config import Config
        from kanachan.training.online.train import run
        from test_online_training import write_q_model, collect_fixture
        data = rows(47)
        interrupt = False

        def reader(**kwargs):
            for offset in range(kwargs["num_skip_samples"], len(data), 7):
                if interrupt and offset == 14:
                    raise InterruptedError("initial preparation interrupted")
                yield data[offset:offset + 7].clone()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial, source = root / "initial.kanachan", root / "train.tsv"
            write_q_model(initial)
            source.write_text("training identity")
            config = Config(
                str(initial), str(root / "full"), device="cpu", amp_dtype="float32",
                iterations=1, games_per_iteration=1, simulation_batch_size=1,
                simulation_concurrency=1, replay_capacity=16, learning_starts=1,
                batch_size=2, updates_per_iteration=1, evaluate_every=0,
                offline_data=str(source), offline_capacity=9,
            )
            with patch.object(offline_pool, "DataLoader", side_effect=reader), patch.object(
                offline_pool, "_CHECKPOINT_SECONDS", 0
            ):
                run(config, collect_fn=collect_fixture)
                partial = replace(config, output_dir=str(root / "partial"))
                interrupt = True
                with self.assertRaises(InterruptedError):
                    run(partial, collect_fn=collect_fixture)
                self.assertFalse((root / "partial/latest.pt").exists())
                interrupt = False
                run(partial, collect_fn=collect_fixture)
                full = torch.load(root / "full/latest.pt", weights_only=True)
                resumed = torch.load(root / "partial/latest.pt", weights_only=True)
                for key in ("model", "target", "optimizer", "optimizer_policy", "replay",
                            "offline_replay", "rng", "samples", "optimizer_steps"):
                    compare_state(self, full[key], resumed[key])


class QOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_single_validation_and_no_unused_action_selection(self):
        model = build_q()
        converter = DecodeConverter("argmax", tie_break_by_action_id=True)
        actor = TensorDictSequential(*list(model.module), TensorDictModule(
            converter, in_keys=["candidates", "action_value"], out_keys=["action"]
        ))
        optimized = q_only(actor)
        self.assertIsInstance(optimized, DecisionQNetwork)
        self.assertEqual({id(p) for p in actor.parameters()}, {id(p) for p in optimized.parameters()})
        reference = observation()
        actor(reference)
        with patch("kanachan.nn.decision_decoder.legal_actions", wraps=legal_actions) as validation:
            with patch.object(converter, "forward", side_effect=AssertionError("unused argmax")):
                actual = observation()
                optimized(actual)
            self.assertEqual(validation.call_count, 1)
        torch.testing.assert_close(actual["action_value"], reference["action_value"], atol=0, rtol=0)
        for path in (actor, optimized):
            path.zero_grad(set_to_none=True)
            data = observation()
            path(data)
            data["action_value"].square().mean().backward()
            gradients = [p.grad.clone() for p in path.parameters()]
            if path is actor:
                expected = gradients
            else:
                for a, b in zip(expected, gradients, strict=True):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
        invalid = observation()
        invalid["candidates"][:, 1] = invalid["candidates"][:, 0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            optimized(invalid)

    def test_rope_shared_once_without_changing_forward_or_gradients(self):
        for checkpointing in (False, True):
            model = backbone(checkpointing=checkpointing)
            obs = observation()
            with patch("kanachan.nn.decision_decoder.rotary_embeddings", wraps=rotary_embeddings) as rope:
                actual = model(*(obs[k] for k in ("sparse", "numeric", "progression", "candidates")))
                actual.square().sum().backward()
                self.assertEqual(rope.call_count, 1)
            gradients = [p.grad.clone() for p in model.parameters()]
            model.zero_grad(set_to_none=True)
            value, positions, allowed, valid = model.tokens(
                *(obs[k] for k in ("sparse", "numeric", "progression", "candidates"))
            )
            for block in model.blocks:
                value = block(value, positions, allowed).masked_fill(~valid[..., None], 0)
            reference = model.norm(value[:, -1])
            reference.square().sum().backward()
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            for expected, p in zip(gradients, model.parameters(), strict=True):
                torch.testing.assert_close(expected, p.grad, atol=0, rtol=0)

    def test_tensor_metrics_and_all_terminal_batch(self):
        source = FixedQ([0.2, 0.3] + [0.] * 30)
        target = copy.deepcopy(source).requires_grad_(False)
        data = decoder_fixtures.DiscreteCQLTests().data()
        data["next", "done"].fill_(True)
        data["next", "candidates"].fill_(546)
        first, floats = scalar_cql_loss(source, target, data, 0.1)
        second, tensors = scalar_cql_loss(source, target, data, 0.1, tensor_metrics=True)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertEqual(target.calls, [])
        self.assertEqual(source.calls, [2, 2])
        for key, value in tensors.items():
            self.assertIsInstance(value, torch.Tensor)
            self.assertFalse(value.requires_grad)
            self.assertEqual(floats[key], float(value))


if __name__ == "__main__":
    unittest.main()
