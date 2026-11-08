import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from test_training_regressions import bc_line

from kanachan.constants import ENCODER_WIDTH, MAX_NUM_ACTIVE_SPARSE_FEATURES
from kanachan.nn import Encoder, RoundEncoder
from kanachan.training.core.eor.data_iterator import (
    DataIterator as EORDataIterator,
)
from kanachan.training.core.eor.feature_adaptor import FeatureAdaptor
from kanachan.training.core.offline_rl import EpisodeReplayBuffer
from kanachan.training.core.offline_rl.data_iterator import (
    DataIterator as RLDataIterator,
)
from kanachan.training.core.ranking import scores_to_ranks


def rl_line(index: int, *, terminal: bool) -> str:
    columns = bc_line(index % 4).rstrip().split("\t")
    columns[2] = f"{index},0,25000,25000,25000,25000"
    if not terminal:
        columns = columns[:6] + columns[1:5]
    return "\t".join(columns) + "\n"


def unit_reward(data, contiguous):
    data["next", "reward"] = torch.ones(data.batch_size)


class TrainingPipelineTests(unittest.TestCase):
    def test_round_encoder_checkpointing_preserves_final_normalization_and_gradients(
        self,
    ):
        arguments = {
            "position_encoder": "position_embedding",
            "dimension": 8,
            "num_heads": 2,
            "dim_feedforward": 16,
            "num_layers": 2,
            "activation_function": "gelu",
            "dropout": 0.0,
            "layer_normalization": True,
            "device": torch.device("cpu"),
            "dtype": torch.float64,
        }
        ordinary = RoundEncoder(**arguments, checkpointing=False)
        checkpointed = RoundEncoder(**arguments, checkpointing=True)
        with torch.no_grad():
            ordinary.encoder.norm.weight.fill_(1.7)
            ordinary.encoder.norm.bias.fill_(0.3)
        checkpointed.load_state_dict(ordinary.state_dict())
        sparse = torch.tensor(
            [[3, 6, 7, 23, 39, 55, 71, 74]] * 2, dtype=torch.int32
        )
        numeric = torch.tensor(
            [[0, 0, 25000, 25000, 25000, 25000]] * 2, dtype=torch.int32
        )
        expected = ordinary(sparse, numeric)
        actual = checkpointed(sparse, numeric)
        torch.testing.assert_close(actual, expected)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for source, restored in zip(
            ordinary.parameters(), checkpointed.parameters(), strict=True
        ):
            self.assertEqual(source.dtype, torch.float64)
            self.assertIsNotNone(restored.grad)
            torch.testing.assert_close(source.grad, restored.grad)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_round_features_use_wind_and_round_instead_of_seat(self):
        sparse = torch.zeros(
            4, MAX_NUM_ACTIVE_SPARSE_FEATURES, dtype=torch.int32
        )
        sparse[:, :9] = torch.tensor([3, 6, 7, 23, 39, 55, 71, 76, 81])
        sparse[:, 6] = torch.arange(71, 75)
        numeric = torch.zeros(4, 6, dtype=torch.int32)
        actual, actual_numeric = FeatureAdaptor()(sparse, numeric)
        expected = torch.tensor([3, 6, 7, 23, 39, 55, 72, 77])
        self.assertTrue(torch.equal(actual, expected.expand(4, -1)))
        self.assertIs(actual_numeric, numeric)

    def make_encoder(self, *, checkpointing, dtype=torch.float32):
        return Encoder(
            position_encoder="position_embedding",
            dimension=8,
            num_heads=2,
            dim_feedforward=16,
            activation_function="gelu",
            dropout=0.0,
            layer_normalization=True,
            num_layers=2,
            checkpointing=checkpointing,
            device=torch.device("cpu"),
            dtype=dtype,
        )

    def test_checkpointing_preserves_final_normalization_and_gradients(self):
        torch.manual_seed(29)
        ordinary = self.make_encoder(checkpointing=False)
        checkpointed = self.make_encoder(checkpointing=True)
        with torch.no_grad():
            ordinary.encoder.norm.weight.fill_(1.7)
            ordinary.encoder.norm.bias.fill_(0.3)
        checkpointed.load_state_dict(ordinary.state_dict())
        inputs = torch.randn(2, ENCODER_WIDTH, 8, requires_grad=True)
        copy = inputs.detach().clone().requires_grad_()
        expected = ordinary._Encoder__encode(inputs)
        actual = checkpointed._Encoder__encode(copy)
        torch.testing.assert_close(actual, expected)
        weights = torch.randn_like(actual)
        (expected * weights).sum().backward()
        (actual * weights).sum().backward()
        torch.testing.assert_close(copy.grad, inputs.grad)
        for (_, expected_parameter), (_, actual_parameter) in zip(
            ordinary.named_parameters(),
            checkpointed.named_parameters(),
            strict=True,
        ):
            if expected_parameter.grad is not None:
                torch.testing.assert_close(
                    actual_parameter.grad, expected_parameter.grad
                )

    def test_encoder_normalization_uses_requested_dtype(self):
        encoder = self.make_encoder(checkpointing=False, dtype=torch.float64)
        self.assertEqual(encoder.encoder.norm.weight.dtype, torch.float64)
        result = encoder._Encoder__encode(
            torch.randn(1, ENCODER_WIDTH, 8, dtype=torch.float64)
        )
        self.assertEqual(result.dtype, torch.float64)

    def make_replay(self, path, *, contiguous, **kwargs):
        return EpisodeReplayBuffer(
            training_data=path,
            contiguous_training_data=contiguous,
            num_skip_samples=0,
            rewrite_rooms=None,
            rewrite_grades=None,
            get_reward=kwargs.pop("get_reward", unit_reward),
            discount_factor=0.5,
            dtype=torch.float32,
            max_size=4,
            batch_size=3,
            num_workers=kwargs.pop("num_workers", 0),
            pin_memory=False,
            **kwargs,
        )

    def test_replay_defaults_and_partial_batches_preserve_all_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rl.tsv"
            path.write_text(
                "".join(rl_line(i, terminal=i in (2, 4)) for i in range(5))
            )
            with patch(
                "kanachan.training.core.offline_rl.episode_replay_buffer._INTERNAL_BATCH_SIZE",
                2,
            ):
                replay = self.make_replay(path, contiguous=True)
                batches = list(replay)
                with self.assertRaises(StopIteration):
                    next(replay)
            self.assertEqual(
                [batch.batch_size[0] for batch in batches], [3, 2]
            )
            values = {}
            for batch in batches:
                values.update(
                    zip(
                        batch["numeric"][:, 0].tolist(),
                        batch["action_value"].tolist(),
                        strict=True,
                    )
                )
            self.assertEqual(values, {0: 1.75, 1: 1.5, 2: 1.0, 3: 1.5, 4: 1.0})

    def test_shuffled_replay_does_not_require_a_terminal_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rl.tsv"
            path.write_text(
                "".join(rl_line(i, terminal=False) for i in range(3))
            )
            replay = self.make_replay(path, contiguous=False)
            batches = list(replay)
            self.assertEqual(len(batches), 1)
            self.assertEqual(
                sorted(batches[0]["numeric"][:, 0].tolist()), [0, 1, 2]
            )
            torch.testing.assert_close(
                batches[0]["action_value"], torch.ones(3)
            )

    def test_replay_preserves_multistep_discounts_and_full_returns(self):
        from kanachan.training.core.rl.multistep import make_multistep

        def reward(data, contiguous):
            unit_reward(data, contiguous)
            make_multistep(data, 0.5, 2)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rl.tsv"
            path.write_text(
                "".join(rl_line(i, terminal=i in (2, 4)) for i in range(5))
            )
            replay = self.make_replay(path, contiguous=True, get_reward=reward)
            values, discounts, horizons = {}, {}, {}
            for batch in replay:
                ids = batch["numeric"][:, 0].tolist()
                values.update(
                    zip(ids, batch["action_value"].tolist(), strict=True)
                )
                discounts.update(
                    zip(ids, batch["next", "discount"].tolist(), strict=True)
                )
                horizons.update(
                    zip(
                        ids,
                        batch["next", "steps_to_next_obs"].tolist(),
                        strict=True,
                    )
                )
            self.assertEqual(values, {0: 1.75, 1: 1.5, 2: 1.0, 3: 1.5, 4: 1.0})
            self.assertEqual(
                discounts, {0: 0.25, 1: 0.25, 2: 0.5, 3: 0.25, 4: 0.5}
            )
            self.assertEqual(horizons, {0: 2, 1: 2, 2: 1, 3: 2, 4: 1})

    def test_contiguous_replay_preserves_order_with_multiple_workers_configured(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rl.tsv"
            path.write_text(
                "".join(rl_line(i, terminal=i == 2) for i in range(3))
            )
            with self.assertLogs(
                "kanachan.training.core.offline_rl.episode_replay_buffer",
                level="WARNING",
            ):
                replay = self.make_replay(path, contiguous=True, num_workers=2)
            batch = next(replay)
            expected = {0: 1.75, 1: 1.5, 2: 1.0}
            self.assertEqual(
                dict(
                    zip(
                        batch["numeric"][:, 0].tolist(),
                        batch["action_value"].tolist(),
                        strict=True,
                    )
                ),
                expected,
            )
            with self.assertRaises(StopIteration):
                next(replay)

    def test_contiguous_replay_rejects_truncated_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rl.tsv"
            path.write_text(rl_line(0, terminal=False))
            replay = self.make_replay(path, contiguous=True)
            with self.assertRaisesRegex(RuntimeError, "[Ii]ncomplete episode"):
                list(replay)

    def test_replay_shards_and_finishes_on_every_rank(self):
        module = "kanachan.training.core.offline_rl.episode_replay_buffer"
        for bad_input in (False, True):
            with (
                self.subTest(bad_input=bad_input),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "rl.tsv"
                text = "".join(rl_line(i, terminal=True) for i in range(13))
                path.write_text(text + ("invalid\n" if bad_input else ""))
                context = threading.local()
                barrier = threading.Barrier(2, timeout=5)
                shared = {}

                def broadcast(
                    tensor,
                    *,
                    src,
                    group,
                    context=context,
                    shared=shared,
                    barrier=barrier,
                ):
                    if context.rank == src:
                        shared["tensor"] = tensor.clone()
                    barrier.wait()
                    if context.rank != src:
                        tensor.copy_(shared["tensor"])
                    barrier.wait()

                def run_rank(rank, context=context, path=path):
                    context.rank = rank
                    replay = EpisodeReplayBuffer(
                        training_data=path,
                        contiguous_training_data=True,
                        num_skip_samples=0,
                        rewrite_rooms=None,
                        rewrite_grades=None,
                        get_reward=unit_reward,
                        discount_factor=0.5,
                        dtype=torch.float32,
                        max_size=8,
                        batch_size=4,
                        num_workers=0,
                        pin_memory=False,
                        ranks=[1, 2],
                        main_rank=2,
                    )
                    batches = []
                    error = None
                    try:
                        batches = list(replay)
                    except RuntimeError as exception:
                        error = str(exception)
                    with self.assertRaises(StopIteration):
                        next(replay)
                    return batches, error

                with (
                    patch(
                        f"{module}.get_distributed_environment",
                        side_effect=lambda context=context: (
                            3,
                            context.rank,
                            context.rank,
                        ),
                    ),
                    patch(f"{module}.new_group", return_value=object()),
                    patch(f"{module}.broadcast", side_effect=broadcast),
                    ThreadPoolExecutor(max_workers=2) as executor,
                ):
                    futures = [
                        executor.submit(run_rank, rank) for rank in (1, 2)
                    ]
                    results = [future.result(timeout=15) for future in futures]
                if bad_input:
                    self.assertTrue(all(error for _, error in results))
                else:
                    self.assertTrue(all(error is None for _, error in results))
                    self.assertTrue(
                        all(
                            [batch.batch_size[0] for batch in batches]
                            == [4, 2]
                            for batches, _ in results
                        )
                    )
                    ids = [
                        index
                        for batches, _ in results
                        for batch in batches
                        for index in batch["numeric"][:, 0].tolist()
                    ]
                    self.assertEqual(len(ids), 12)
                    self.assertEqual(len(set(ids)), 12)

    def test_rank_labels_resolve_ties_for_all_seats(self):
        scores = torch.tensor([[100, 100, 200, 50], [0, 0, 0, 0]])
        expected = torch.tensor([[1, 2, 0, 3], [0, 1, 2, 3]])
        torch.testing.assert_close(scores_to_ranks(scores), expected)

    def test_round_ranking_training_supervises_every_seat(self):
        from kanachan.training.eor2ranking.train import _train

        class RankingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(4, 4))

            def forward(self, data):
                data["decode"] = self.logits.unsqueeze(0).expand(
                    data.batch_size[0], -1, -1
                )
                return data

        model = RankingModel()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eor.tsv"
            path.write_text(
                "game\t3,6,7,23,39,55,71,74\t0,0,25000,25000,25000,25000\t100,100,200,50\n"
            )
            _train(
                device=torch.device("cpu"),
                dtype=torch.float32,
                amp_dtype=torch.float32,
                training_data=path,
                rewrite_rooms=None,
                rewrite_grades=None,
                num_workers=0,
                num_samples=0,
                network_tdm=model,
                batch_size=4,
                gradient_accumulation_steps=1,
                max_gradient_norm=1.0,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                scheduler=None,
                snapshot_interval=0,
                summary_writer=Mock(),
                snapshot_writer=Mock(),
            )
        self.assertTrue((model.logits.detach().abs().sum(1) > 0).all().item())
        self.assertEqual(model.logits.argmax(1).tolist(), [1, 2, 0, 3])

    def test_next_state_rejects_negative_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rl.tsv"
            for column in (6, 8, 9):
                with self.subTest(column=column):
                    columns = rl_line(0, terminal=False).rstrip().split("\t")
                    columns[column] = "-1"
                    path.write_text("\t".join(columns) + "\n")
                    iterator = RLDataIterator(
                        path=path,
                        num_skip_samples=0,
                        rewrite_rooms=None,
                        rewrite_grades=None,
                        local_rank=0,
                    )
                    with self.assertRaises(RuntimeError):
                        next(iterator)
                    del iterator

    def test_round_reader_accepts_celestial_grade(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eor.tsv"
            path.write_text(
                "game\t3,6,7,23,39,55,71,74\t0,0,25000,25000,25000,25000\t100,100,200,50\n"
            )
            iterator = EORDataIterator(path, 0, None, 15, 0)
            sparse, _, _ = next(iterator)
            self.assertEqual(sparse[2:6].tolist(), [22, 38, 54, 70])
            del iterator
