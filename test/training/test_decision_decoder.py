import copy
import unittest

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from kanachan.constants import (
    MAX_NUM_ACTIVE_SPARSE_FEATURES, NUM_NUMERIC_FEATURES,
    MAX_LENGTH_OF_PROGRESSION_FEATURES, MAX_NUM_ACTION_CANDIDATES,
    NUM_TYPES_OF_SPARSE_FEATURES, NUM_TYPES_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ACTIONS,
)
from kanachan.nn.decision_decoder import (
    DenseDecisionDecoder, GlobalActionHead, DecisionContext, legal_argmax,
    feature_schema,
    cached_decision,
)
from kanachan.nn.decode_converter import DecodeConverter
from kanachan.training.core.rl.discrete_cql import (
    td_components, td_penalty, conservative_penalty, scalar_cql_loss,
)
from kanachan.model_loader import dump_model, dump_object, load_model_state


def observation(batch=3, device="cpu"):
    sparse = torch.full(
        (batch, MAX_NUM_ACTIVE_SPARSE_FEATURES), NUM_TYPES_OF_SPARSE_FEATURES,
        dtype=torch.int32, device=device,
    )
    sparse[:, :4] = torch.tensor([1, 6, 77, 95], device=device)
    numeric = torch.tensor(
        [[1, 2, 25000, 30000, 20000, 25000]], dtype=torch.int32, device=device
    ).expand(batch, -1).clone()
    progression = torch.full(
        (batch, MAX_LENGTH_OF_PROGRESSION_FEATURES),
        NUM_TYPES_OF_PROGRESSION_FEATURES, dtype=torch.int32, device=device,
    )
    for i in range(batch):
        progression[i, :i * 2] = torch.arange(i * 2, device=device) + 5
    candidates = torch.full(
        (batch, MAX_NUM_ACTION_CANDIDATES), NUM_TYPES_OF_ACTIONS,
        dtype=torch.int32, device=device,
    )
    candidates[:, :3] = torch.tensor([2, 100, 545], device=device)
    return TensorDict({
        "sparse": sparse, "numeric": numeric,
        "progression": progression, "candidates": candidates,
    }, [batch])


def backbone(**extra):
    arguments = dict(
        dimension=16, num_heads=2, dim_feedforward=24, num_layers=2,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    arguments.update(extra)
    return DenseDecisionDecoder(**arguments)


def build_q(model=None):
    model = backbone() if model is None else model
    head = GlobalActionHead(
        input_dimension=model.dimension, hidden_dimension=model.dimension,
        device=model.bos.device, dtype=model.bos.dtype,
    )
    return TensorDictSequential(
        TensorDictModule(
            model, in_keys=["sparse", "numeric", "progression", "candidates"],
            out_keys=["encode"],
        ),
        TensorDictModule(
            head, in_keys=["candidates", "encode"], out_keys=["action_value"]
        ),
    )


def apply(model, obs):
    return model(*(obs[key] for key in (
        "sparse", "numeric", "progression", "candidates"
    )))


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(4)

    def test_actual_parameter_budget_and_shapes(self):
        model = backbone(dimension=128, num_heads=2, dim_feedforward=192)
        q = build_q(model)
        self.assertEqual(sum(p.numel() for p in q.parameters()), 829730)
        obs = observation()
        q(obs)
        self.assertEqual(obs["encode"].shape, (3, 128))
        self.assertEqual(obs["action_value"].shape, (3, MAX_NUM_ACTION_CANDIDATES))
        self.assertEqual(model.width, 147)
        self.assertEqual(model.numeric_dimension, 64)
        self.assertTrue(torch.isfinite(obs["action_value"]).all())
        expanded = build_q(backbone(
            dimension=256, num_heads=4, dim_feedforward=704, num_layers=6
        ))
        self.assertEqual(sum(p.numel() for p in expanded.parameters()), 6051874)

    def test_permutation_and_padding_invariance(self):
        model = backbone().eval()
        q = build_q(model).eval()
        a = observation()
        b = a.clone()
        sparse_order = torch.randperm(MAX_NUM_ACTIVE_SPARSE_FEATURES)
        candidate_order = torch.randperm(MAX_NUM_ACTION_CANDIDATES)
        b["sparse"] = b["sparse"][:, sparse_order]
        b["candidates"] = b["candidates"][:, candidate_order]
        q(a)
        q(b)
        torch.testing.assert_close(a["encode"], b["encode"], atol=3e-6, rtol=1e-5)
        torch.testing.assert_close(
            a["action_value"][:, candidate_order], b["action_value"],
            atol=3e-6, rtol=1e-5,
        )
        with torch.no_grad():
            model.sparse_embedding.weight[NUM_TYPES_OF_SPARSE_FEATURES].fill_(100)
            model.progression_embedding.weight[NUM_TYPES_OF_PROGRESSION_FEATURES].fill_(-100)
        torch.testing.assert_close(apply(model, a), a["encode"], atol=3e-6, rtol=1e-5)

    def test_history_causality_and_state_independence(self):
        model = backbone().eval()
        a, b = observation(), observation()
        b["progression"][2, 3] = 90
        va, pa, ma, valid = model.tokens(*(a[k] for k in (
            "sparse", "numeric", "progression", "candidates"
        )))
        vb, pb, mb, _ = model.tokens(*(b[k] for k in (
            "sparse", "numeric", "progression", "candidates"
        )))
        for layer in model.blocks:
            va = layer(va, pa, ma).masked_fill(~valid[..., None], 0)
            vb = layer(vb, pb, mb).masked_fill(~valid[..., None], 0)
            torch.testing.assert_close(va[2, :4], vb[2, :4], atol=0, rtol=0)
        frozen = model.eval().requires_grad_(False)
        context = DecisionContext("env", "game", 0, 0, "v1")
        args = tuple(a[k][2:3] for k in (
            "sparse", "numeric", "progression", "candidates"
        ))
        _, cache = frozen.incremental(*args, context=context)
        old_keys = [key.clone() for key in cache.keys]
        args[1][0, 2] += 10000
        frozen.incremental(*args, context=context, cache=cache)
        for old, new in zip(old_keys, cache.keys):
            torch.testing.assert_close(old, new, atol=0, rtol=0)

    def test_cache_full_equivalence_and_lifecycles(self):
        model = backbone().eval().requires_grad_(False)
        obs = observation(1)
        context = DecisionContext("env1", "game1", 0, 0, "v1")
        cache = None
        for n in (0, 1, 4, 4, 6, 2, 3):
            obs["progression"].fill_(NUM_TYPES_OF_PROGRESSION_FEATURES)
            obs["progression"][0, :n] = torch.arange(n) + 10
            obs["numeric"][0, 2] += 100
            obs["candidates"][0, 0] = n
            result, cache = model.incremental(
                *(obs[k] for k in ("sparse", "numeric", "progression", "candidates")),
                context=context, cache=cache,
            )
            torch.testing.assert_close(
                result, apply(model, obs), atol=5e-6, rtol=1e-5
            )
            self.assertEqual(cache.keys[0].shape[-2], n + 1)
        for context in (
            DecisionContext("env2", "game1", 0, 0, "v1"),
            DecisionContext("env2", "game2", 0, 0, "v1"),
            DecisionContext("env2", "game2", 1, 0, "v1"),
            DecisionContext("env2", "game2", 1, 1, "v1"),
            DecisionContext("env2", "game2", 1, 1, "v2"),
        ):
            old = cache
            result, cache = model.incremental(
                *(obs[k] for k in ("sparse", "numeric", "progression", "candidates")),
                context=context, cache=cache,
            )
            self.assertIsNot(cache, old)
            torch.testing.assert_close(result, apply(model, obs), atol=5e-6, rtol=1e-5)
        old = cache
        with torch.no_grad():
            model.blocks[0].qkv.weight.add_(0.001)
        _, cache = model.incremental(
            *(obs[k] for k in ("sparse", "numeric", "progression", "candidates")),
            context=context, cache=cache,
        )
        self.assertIsNot(old, cache)
        head = GlobalActionHead(
            input_dimension=16, hidden_dimension=16,
            device=torch.device("cpu"), dtype=torch.float32,
        ).eval().requires_grad_(False)
        index, cache = cached_decision(
            model, head, obs, context=context, cache=cache
        )
        expected = legal_argmax(obs["candidates"], head(obs["candidates"], apply(model, obs)))
        torch.testing.assert_close(index, expected, atol=0, rtol=0)

    def test_invalid_inputs_and_cache_preconditions(self):
        model = backbone()
        obs = observation()
        for key, value in (
            ("candidates", -1), ("candidates", 547),
            ("sparse", 511), ("progression", 2166),
        ):
            broken = obs.clone()
            broken[key][0, 0] = value
            with self.assertRaises(ValueError):
                apply(model, broken)
        broken = obs.clone()
        broken["candidates"].fill_(546)
        with self.assertRaisesRegex(ValueError, "no legal"):
            apply(model, broken)
        broken = obs.clone()
        broken["candidates"][0, 1] = broken["candidates"][0, 0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            apply(model, broken)
        broken = obs.clone()
        broken["progression"][0, 1] = 8
        with self.assertRaisesRegex(ValueError, "padding"):
            apply(model, broken)
        with self.assertRaisesRegex(ValueError, "frozen"):
            model.incremental(
                *(obs[k][:1] for k in ("sparse", "numeric", "progression", "candidates")),
                context=DecisionContext("e", "g", 0, 0, 1),
            )
        with self.assertRaisesRegex(ValueError, "schema"):
            backbone(schema={"version": 9})

    def test_candidate_ties_and_single_action(self):
        obs = observation(1)
        candidates = obs["candidates"]
        values = torch.zeros_like(candidates).float()
        order = torch.randperm(MAX_NUM_ACTION_CANDIDATES)
        first = legal_argmax(candidates, values)
        second = legal_argmax(candidates[:, order], values[:, order])
        self.assertEqual(candidates[0, first], candidates[:, order][0, second])
        self.assertEqual(candidates[0, first], 2)
        candidates.fill_(546)
        candidates[0, 17] = 545
        values[0, 0] = 100
        self.assertEqual(legal_argmax(candidates, values).item(), 17)

    def test_epsilon_sampling_is_invariant_to_candidate_order(self):
        from kanachan.nn.epsilon_greedy import EpsilonGreedy
        exploration = EpsilonGreedy(
            epsilon_first=1.0, epsilon_last=1.0, annealing_steps=0,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        exploration.stable_action_ids = True
        first = observation(24)
        second = first.clone()
        order = torch.randperm(MAX_NUM_ACTION_CANDIDATES)
        second["candidates"] = second["candidates"][:, order]
        first["action"] = torch.zeros(24, dtype=torch.int32)
        second["action"] = first["action"].clone()
        torch.manual_seed(61)
        exploration(first)
        torch.manual_seed(61)
        exploration(second)
        a = first["candidates"].gather(1, first["action"].long()[:, None])
        b = second["candidates"].gather(1, second["action"].long()[:, None])
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_gradient_checkpointing_and_pad_gradients(self):
        first, obs = backbone(), observation()
        second = copy.deepcopy(first)
        second.checkpointing = True
        for model in (first, second):
            apply(model, obs).square().mean().backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertEqual(
                model.progression_embedding.weight.grad[2165].abs().sum(), 0
            )
            self.assertEqual(model.sparse_embedding.weight.grad[510].abs().sum(), 0)
        for p, q in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)

    def test_bfloat16_full_and_cached(self):
        model = backbone().eval().requires_grad_(False)
        obs = observation(1)
        context = DecisionContext("e", "g", 0, 0, 0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = apply(model, obs)
            actual, cache = model.incremental(
                *(obs[k] for k in ("sparse", "numeric", "progression", "candidates")),
                context=context,
            )
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        old = cache
        _, cache = model.incremental(
            *(obs[k] for k in ("sparse", "numeric", "progression", "candidates")),
            context=context, cache=cache,
        )
        self.assertIsNot(cache, old)


class FixedQ(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.calls = []

    def forward(self, data):
        self.calls.append(len(data))
        data["action_value"] = self.values.expand(len(data), -1)
        return data


class DiscreteCQLTests(unittest.TestCase):
    def data(self):
        obs = observation(2)
        obs["candidates"][:, 2:] = 546
        nxt = obs.clone()
        nxt["candidates"][1].fill_(546)
        nxt["reward"] = torch.tensor([0.5, -1.5])
        nxt["done"] = torch.tensor([False, True])
        nxt["discount"] = torch.ones(2)
        obs["next"] = nxt
        obs["action"] = torch.tensor([0, 1], dtype=torch.int32)
        return obs

    def test_ddqn_legal_target_and_terminal_subbatch(self):
        source = FixedQ([3., 1.] + [100.] * 30)
        target = FixedQ([0.2, 0.8] + [100.] * 30).requires_grad_(False).eval()
        data = self.data()
        values, chosen, expected = td_components(source, target, data)
        torch.testing.assert_close(expected, torch.tensor([0.7, -1.5]))
        self.assertEqual(source.calls, [2, 1])
        self.assertEqual(target.calls, [1])
        loss, _ = scalar_cql_loss(source, target, data, 0.1, td_loss="huber")
        loss.backward()
        self.assertIsNotNone(source.values.grad)
        self.assertIsNone(target.values.grad)
        self.assertEqual(source.values.grad[2:].abs().sum(), 0)

    def test_cql_math_singleton_and_count_centering(self):
        values = torch.zeros(1, 2, requires_grad=True)
        chosen = values[:, 0]
        gap = conservative_penalty(values, chosen)
        torch.testing.assert_close(gap, torch.log(torch.tensor([2.])))
        gap.sum().backward()
        torch.testing.assert_close(values.grad, torch.tensor([[-0.5, 0.5]]))
        single = torch.tensor([[1.2]], requires_grad=True)
        gap = conservative_penalty(single, single[:, 0])
        self.assertEqual(gap.item(), 0.)
        gap.sum().backward()
        self.assertEqual(single.grad.item(), 0.)
        torch.testing.assert_close(
            td_penalty(torch.tensor([3.]), torch.tensor([0.]), "huber"),
            torch.tensor([2.5]),
        )

    def test_shapes_and_float_action_rejection(self):
        source = FixedQ([0.] * 32)
        target = copy.deepcopy(source).requires_grad_(False)
        data = self.data()
        reference, _ = scalar_cql_loss(source, target, data, 0.1)
        for key in ("action", ("next", "reward"), ("next", "done"), ("next", "discount")):
            data[key] = data[key][:, None]
        actual, _ = scalar_cql_loss(source, target, data, 0.1)
        torch.testing.assert_close(reference, actual)
        data["action"] = data["action"].float()
        with self.assertRaisesRegex(ValueError, "integer"):
            scalar_cql_loss(source, target, data, 0.1)

    def test_real_decoder_cql_backward(self):
        source = build_q()
        target = copy.deepcopy(source).requires_grad_(False).eval()
        data = self.data()
        loss, metrics = scalar_cql_loss(source, target, data, 0.1, td_loss="huber")
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in source.parameters()))
        self.assertTrue(all(p.grad is None for p in target.parameters()))
        self.assertTrue(torch.isfinite(torch.tensor(list(metrics.values()))).all())

    def test_td_target_ignores_logged_returns_and_is_detached(self):
        source = FixedQ([3., 1.] + [100.] * 30)
        target = FixedQ([0.2, 0.8] + [100.] * 30).requires_grad_(False).eval()
        data = self.data()
        observed = torch.tensor([float("nan"), 1.e6], requires_grad=True)
        data["action_value"] = observed
        _, _, expected = td_components(source, target, data)
        torch.testing.assert_close(expected, torch.tensor([.7, -1.5]))
        self.assertFalse(expected.requires_grad)
        loss, _ = scalar_cql_loss(source, target, data, .1)
        loss.backward()
        self.assertIsNone(observed.grad)
        self.assertIsNone(target.values.grad)
        self.assertGreater(float(source.values.grad.abs().sum()), 0.)

    def test_zero_discount_and_terminal_skip_next_networks(self):
        source = FixedQ([3., 1.] + [100.] * 30)
        target = FixedQ([0.2, 0.8] + [100.] * 30)
        data = self.data()
        data["next", "discount"] = torch.zeros(2)
        data["next", "candidates"].fill_(546)
        _, _, expected = td_components(source, target, data)
        torch.testing.assert_close(expected, data["next", "reward"])
        self.assertEqual(source.calls, [2])
        self.assertEqual(target.calls, [])

    def test_td_cql_loss_and_gradient_match_direct_formula(self):
        source = FixedQ([3., 1.] + [100.] * 30)
        target = FixedQ([.2, .8] + [100.] * 30)
        actual, _ = scalar_cql_loss(source, target, self.data(), .5)
        actual.backward()
        values = torch.tensor([3., 1.], requires_grad=True)
        expected = (.5 * (values-torch.tensor([.7, -1.5])).square()).mean()
        expected += .5 * (values.logsumexp(0)-values.mean())
        expected.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(source.values.grad[:2], values.grad)
        self.assertEqual(float(source.values.grad[2:].abs().sum()), 0.)

    def test_td_joint_offset_property_is_not_hidden_by_return_mixing(self):
        data = self.data()
        source = FixedQ([3., 1.] + [100.] * 30)
        target = FixedQ([.2, .8] + [100.] * 30)
        shifted_source = FixedQ([7., 5.] + [104.] * 30)
        shifted_target = FixedQ([4.2, 4.8] + [104.] * 30)
        values, chosen, expected = td_components(source, target, data)
        changed_values, changed_chosen, changed_expected = td_components(
            shifted_source, shifted_target, data)
        torch.testing.assert_close(
            (changed_chosen-changed_expected) - (chosen-expected),
            torch.tensor([0., 4.]),
        )
        torch.testing.assert_close(conservative_penalty(values, chosen),
                                   conservative_penalty(changed_values, changed_chosen))

    def test_td_diagnostics_distinguish_terminal_correction(self):
        from kanachan.training.core.rl.discrete_cql import q_statistics
        data = self.data()
        values = torch.zeros(2, 32)
        stats = q_statistics(values, torch.tensor([2., 9.]), data["candidates"],
                             target=torch.tensor([2.1, -1.5]), done=data["next", "done"])
        self.assertEqual(float(stats["TD/Terminal count"]), 1.)
        self.assertAlmostEqual(float(stats["TD/Terminal residual"]), 10.5)
        self.assertAlmostEqual(float(stats["TD/Nonterminal residual"]), -.1, places=6)

    def test_bc_feature_transfer_does_not_transfer_logits_as_q(self):
        from kanachan.nn.decision_value import transfer_bc_features
        teacher, current = build_q(), build_q()
        body, head = current.module[0].module, current.module[1].module
        transfer_bc_features(body, head, teacher)
        for key, value in teacher.module[0].module.state_dict().items():
            torch.testing.assert_close(value, body.state_dict()[key], atol=0, rtol=0)
        for key, value in teacher.module[1].module.layers[0].state_dict().items():
            torch.testing.assert_close(value, head.layers[0].state_dict()[key], atol=0, rtol=0)
        observed = observation()
        current(observed)
        self.assertEqual(float(observed["action_value"].abs().max()), 0.)
        self.assertTrue(all(p.requires_grad for p in current.parameters()))

    def test_bc_feature_transfer_rejects_attention_semantics_mismatch(self):
        from kanachan.nn.decision_value import transfer_bc_features
        teacher = build_q(backbone(history_attention="bidirectional"))
        current = build_q()
        with self.assertRaisesRegex(ValueError, "architectures differ"):
            transfer_bc_features(current.module[0].module, current.module[1].module, teacher)


if __name__ == "__main__":
    unittest.main()
