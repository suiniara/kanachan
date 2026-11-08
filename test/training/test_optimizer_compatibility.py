import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from kanachan.training.core.config.optimizer import (
    AdamOptimizerConfig,
    LambOptimizerConfig,
    RAdamOptimizerConfig,
    SgdOptimizerConfig,
    create,
)
from kanachan.training.core.optimizers import CompatibleAdam, Lamb


class OptimizerCompatibilityTests(unittest.TestCase):
    def test_adam_restores_apex_group_step_without_mutating_input(self):
        parameter = torch.nn.Parameter(
            torch.tensor([1.0, -2.0], dtype=torch.float64)
        )
        reference = torch.optim.Adam([parameter], lr=0.1)
        for _ in range(3):
            parameter.grad = torch.tensor([0.2, 0.3], dtype=torch.float64)
            reference.step()
        legacy = copy.deepcopy(reference.state_dict())
        legacy["param_groups"][0] = {
            "params": legacy["param_groups"][0]["params"],
            "lr": 0.1,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "bias_correction": True,
            "step": 3,
        }
        for state in legacy["state"].values():
            state.pop("step")
        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = CompatibleAdam([restored_parameter], lr=0.1)
        restored.load_state_dict(legacy)
        self.assertNotIn("step", next(iter(legacy["state"].values())))
        parameter.grad = restored_parameter.grad = torch.tensor(
            [0.4, -0.1], dtype=torch.float64
        )
        reference.step()
        restored.step()
        torch.testing.assert_close(restored_parameter, parameter)
        self.assertEqual(restored.state[restored_parameter]["step"].item(), 4)

    def test_lamb_first_update_matches_closed_form_trust_ratio(self):
        initial = np.array([1.0, -2.0])
        gradient = np.array([0.3, 0.4])
        direction = gradient / (np.abs(gradient) + 1e-6) + 0.01 * initial
        expected = initial - 0.1 * np.linalg.norm(
            initial
        ) * direction / np.linalg.norm(direction)
        parameter = torch.nn.Parameter(torch.tensor(initial))
        optimizer = Lamb([parameter], lr=0.1)
        parameter.grad = torch.tensor(gradient)
        optimizer.step()
        torch.testing.assert_close(parameter, torch.tensor(expected))

    def test_lamb_zero_norm_and_checkpoint_resume(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        optimizer = Lamb([parameter], lr=0.01)
        for _ in range(3):
            parameter.grad = torch.tensor([0.3, 0.4])
            optimizer.step()
        self.assertTrue(parameter.isfinite().all().item())
        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = Lamb([restored_parameter], lr=0.01)
        state = copy.deepcopy(optimizer.state_dict())
        state["param_groups"][0].pop("use_nvlamb")
        restored.load_state_dict(state)
        parameter.grad = restored_parameter.grad = torch.tensor([-0.1, 0.5])
        optimizer.step()
        restored.step()
        torch.testing.assert_close(restored_parameter, parameter)
        self.assertEqual(restored.param_groups[0]["step"], 4)

    def test_all_cpu_presets_work_without_apex(self):
        for config_type in (
            AdamOptimizerConfig,
            RAdamOptimizerConfig,
            LambOptimizerConfig,
            SgdOptimizerConfig,
        ):
            with (
                self.subTest(optimizer=config_type.__name__),
                patch.dict("sys.modules", {"apex": None}),
            ):
                config = OmegaConf.structured(config_type())
                config.learning_rate = 0.01
                model = torch.nn.Linear(2, 1)
                optimizer, _ = create("cpu", config, model)
                model(torch.ones(1, 2)).sum().backward()
                optimizer.step()
                self.assertTrue(
                    all(
                        parameter.isfinite().all().item()
                        for parameter in model.parameters()
                    )
                )

    def test_unscale_overflow_does_not_count_as_an_optimizer_update(self):
        from test_training_regressions import bc_line

        from kanachan.training.bc.train import _training

        class LargeDerivative(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, derivative):
                return (derivative.double() * 1e40).float()

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, data):
                data["log_probs"] = LargeDerivative.apply(self.value).expand(
                    data.batch_size[0], 32
                )
                return data

        model = Policy()
        scaler = torch.amp.GradScaler("cpu", init_scale=1e-4)
        snapshot_writer = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bc.tsv"
            path.write_text(bc_line())
            with (
                patch(
                    "kanachan.training.bc.train.GradScaler",
                    return_value=scaler,
                ),
                self.assertRaisesRegex(RuntimeError, "No optimizer updates"),
            ):
                _training(
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    amp_dtype=torch.float32,
                    training_data=path,
                    rewrite_rooms=None,
                    rewrite_grades=None,
                    num_workers=0,
                    num_samples=0,
                    network_tdm=model,
                    batch_size=1,
                    gradient_accumulation_steps=1,
                    max_gradient_norm=1.0,
                    optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                    scheduler=None,
                    snapshot_interval=0,
                    summary_writer=Mock(),
                    snapshot_writer=snapshot_writer,
                )
        snapshot_writer.assert_not_called()
        self.assertEqual(model.value.item(), 0.0)
        self.assertLess(scaler.get_scale(), 1e-4)

    def test_accumulation_flush_uses_samples_instead_of_batch_means(self):
        from test_training_regressions import bc_line

        from kanachan.training.bc.train import _training

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, data):
                target = data["numeric"][:, 0].float()
                data["log_probs"] = (
                    (-0.5 * (self.value - target).square())
                    .unsqueeze(1)
                    .expand(-1, 32)
                )
                return data

        model = Policy()
        snapshot_writer = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bc.tsv"
            rows = []
            for target in range(1, 6):
                columns = bc_line().rstrip().split("\t")
                columns[2] = f"{target},0,25000,25000,25000,25000"
                rows.append("\t".join(columns) + "\n")
            path.write_text("".join(rows))
            _training(
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
                gradient_accumulation_steps=100,
                max_gradient_norm=10.0,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                scheduler=None,
                snapshot_interval=0,
                summary_writer=Mock(),
                snapshot_writer=snapshot_writer,
            )
        self.assertAlmostEqual(model.value.item(), 0.3, places=6)
        self.assertEqual(
            snapshot_writer.call_args.kwargs["optimizer_steps"], 1
        )
