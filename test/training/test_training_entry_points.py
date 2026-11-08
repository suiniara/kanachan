import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from test_training_regressions import bc_line

from kanachan.model_loader import load_model
from kanachan.training.core.bc import DataLoader


class TrainingEntryPointTests(unittest.TestCase):
    def test_cpu_training_and_model_exports(self):
        self.check_cpu_training_and_model_exports("transformer")

    def test_modern_cpu_training_and_model_exports(self):
        self.check_cpu_training_and_model_exports("rmsnorm_swiglu")

    def check_cpu_training_and_model_exports(self, architecture):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            bc_data = directory / "bc.tsv"
            bc_data.write_text(
                "".join(bc_line(i % 4) for i in range(17)), encoding="utf-8"
            )
            rl_data = directory / "rl.tsv"
            rl_data.write_text(
                bc_data.read_text(encoding="utf-8"), encoding="utf-8"
            )
            reward = directory / "reward.py"
            reward.write_text(
                "def get_reward(data, simulation):\n"
                "    data['next', 'reward'] = "
                "data['next', 'end_of_game'].float()\n",
                encoding="utf-8",
            )
            environment = dict(
                os.environ,
                WORLD_SIZE="1",
                RANK="0",
                LOCAL_RANK="0",
                TORCHDYNAMO_DISABLE="1",
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                HYDRA_FULL_ERROR="1",
            )
            common = [
                "device=cpu",
                "encoder=bert_tiny",
                f"encoder.architecture={architecture}",
                "encoder.dimension=8",
                "encoder.num_heads=2",
                "encoder.num_layers=1",
                "encoder.dropout=0.0",
                "optimizer=adam",
                "num_workers=0",
                "batch_size=4",
            ]
            cases = [
                ("bc", "bc", ["decoder=double"], 5, 0),
                (
                    "bc_accum",
                    "bc",
                    [
                        "decoder=double",
                        "gradient_accumulation_steps=2",
                    ],
                    3,
                    0,
                ),
                (
                    "bc_large_accumulation",
                    "bc",
                    [
                        "decoder=single",
                        "gradient_accumulation_steps=1000",
                    ],
                    1,
                    0,
                ),
                ("iql", "iql", ["decoder=double", "expectile=0.9"], 5, 5),
                ("bc_lamb", "bc", ["decoder=double", "optimizer=lamb"], 5, 0),
                (
                    "bc_single_process_zero",
                    "bc",
                    ["decoder=double", "optimizer.use_zero=true"],
                    5,
                    0,
                ),
                (
                    "ilql",
                    "ilql",
                    [
                        "decoder=double",
                        "expectile=0.9",
                        "target_update_interval=2",
                    ],
                    5,
                    2,
                ),
                (
                    "cql_scalar",
                    "cql",
                    [
                        "decoder=double",
                        "num_qr_intervals=null",
                        "gradient_accumulation_steps=2",
                        "target_update_interval=2",
                    ],
                    3,
                    1,
                ),
                (
                    "cql_quantile",
                    "cql",
                    [
                        "decoder=double",
                        "num_qr_intervals=2",
                    ],
                    5,
                    5,
                ),
                (
                    "cql_bc_transfer",
                    "cql",
                    [
                        "decoder=double",
                        "num_qr_intervals=null",
                        "optimizer.learning_rate=1e-8",
                    ],
                    5,
                    5,
                ),
                ("awr", "awr", ["decoder=double"], 5, 0),
                ("awr_iql", "awr", ["decoder=double"], 5, 0),
                ("awr_ilql", "awr", ["decoder=double"], 5, 0),
                ("awr_iql_legacy", "awr", ["decoder=double"], 5, 0),
                (
                    "awr_bc_decoder",
                    "awr",
                    ["decoder=double", "optimizer.learning_rate=1e-8"],
                    5,
                    0,
                ),
                (
                    "iql_replay",
                    "iql",
                    [
                        "decoder=double",
                        "expectile=0.9",
                        "replay_buffer_size=8",
                        "contiguous_training_data=true",
                    ],
                    5,
                    5,
                ),
                (
                    "ilql_replay",
                    "ilql",
                    [
                        "decoder=double",
                        "expectile=0.9",
                        "replay_buffer_size=8",
                        "contiguous_training_data=true",
                    ],
                    5,
                    5,
                ),
                (
                    "cql_replay",
                    "cql",
                    [
                        "decoder=double",
                        "num_qr_intervals=null",
                        "replay_buffer_size=8",
                        "contiguous_training_data=true",
                    ],
                    5,
                    5,
                ),
                (
                    "cql_without_target",
                    "cql",
                    [
                        "decoder=double",
                        "num_qr_intervals=null",
                        "double_q_learning=false",
                    ],
                    5,
                    0,
                ),
                (
                    "cql_quantile_without_target",
                    "cql",
                    [
                        "decoder=double",
                        "num_qr_intervals=2",
                        "double_q_learning=false",
                    ],
                    5,
                    0,
                ),
            ]
            for (
                name,
                module,
                arguments,
                optimizer_steps,
                target_updates,
            ) in cases:
                with self.subTest(training=name):
                    data = bc_data if module in ("bc", "awr") else rl_data
                    output = directory / name
                    extra = [
                        f"training_data={data.as_posix()}",
                        f"hydra.run.dir={output.as_posix()}",
                    ]
                    if module in ("iql", "ilql", "cql"):
                        extra.append(f"reward_plugin={reward.as_posix()}")
                    if name == "cql_bc_transfer":
                        extra.append(
                            f"encoder.load_from={(directory / 'bc/snapshots/encoder.pth').as_posix()}"
                        )
                    if module == "awr":
                        critic = {
                            "awr": "cql_scalar",
                            "awr_iql": "iql",
                            "awr_ilql": "ilql",
                            "awr_iql_legacy": "iql",
                            "awr_bc_decoder": "iql",
                        }[name]
                        q_model = (
                            directory / critic / "snapshots/model.kanachan"
                        )
                        if name == "awr_iql_legacy":
                            legacy = torch.load(
                                q_model, map_location="cpu", weights_only=True
                            )
                            legacy["args"][0]["kwargs"].pop("return_values")
                            legacy["kwargs"]["out_keys"] = ["action"]
                            q_model = directory / "legacy-iql.kanachan"
                            torch.save(legacy, q_model)
                        extra.append(f"q_model={q_model.as_posix()}")
                        if name == "awr_bc_decoder":
                            extra += [
                                f"encoder.load_from={(directory / 'bc/snapshots/encoder.pth').as_posix()}",
                                f"decoder.load_from={(directory / 'bc/snapshots/decoder.pth').as_posix()}",
                            ]
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            "-m",
                            f"kanachan.training.{module}.train",
                            *common,
                            *arguments,
                            *extra,
                        ],
                        cwd=root,
                        check=False,
                        env=environment,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=90,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertIn(
                        f"Completed in this run: {optimizer_steps} optimizer "
                        f"steps, {target_updates} target updates.",
                        result.stdout + result.stderr,
                    )
                    if optimizer_steps == 0:
                        self.assertIn(
                            "No optimizer updates completed.",
                            result.stdout + result.stderr,
                        )
                    model_path = output / "snapshots/model.kanachan"
                    self.assertTrue(
                        model_path.is_file(), result.stdout + result.stderr
                    )
                    state = torch.load(
                        model_path, map_location="cpu", weights_only=True
                    )
                    self.assertIn("__kanachan__", state)
                    if name == "cql_bc_transfer":
                        initial = torch.load(
                            directory / "bc/snapshots/encoder.pth",
                            weights_only=True,
                        )
                        trained = torch.load(
                            output / "snapshots/source-encoder.pth",
                            weights_only=True,
                        )
                        for key in initial:
                            torch.testing.assert_close(
                                trained[key], initial[key], rtol=0, atol=1e-6
                            )
                    if name == "awr_bc_decoder":
                        initial = torch.load(
                            directory / "bc/snapshots/decoder.pth",
                            weights_only=True,
                        )
                        trained = torch.load(
                            output / "snapshots/decoder.pth", weights_only=True
                        )
                        for key in initial:
                            torch.testing.assert_close(
                                trained[key], initial[key], rtol=0, atol=1e-6
                            )
                    model = load_model(
                        model_path, map_location=torch.device("cpu")
                    )
                    model.eval()
                    data = next(
                        DataLoader(
                            path=bc_data,
                            num_skip_samples=0,
                            rewrite_rooms=None,
                            rewrite_grades=None,
                            batch_size=4,
                        )
                    )
                    with torch.no_grad():
                        model(data)
                    self.assertTrue((data["action"] >= 0).all().item())
                    self.assertTrue((data["action"] < 4).all().item())

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "kanachan.training.bc.evaluate",
                    "--model",
                    str(directory / "bc/snapshots/model.kanachan"),
                    "--data",
                    str(bc_data),
                    "--batch-size",
                    "7",
                ],
                cwd=root,
                check=False,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            self.assertEqual(
                result.returncode, 0, result.stdout + result.stderr
            )
            metrics = json.loads(result.stdout)
            self.assertEqual(metrics["num_samples"], 17)
            self.assertTrue(0.0 <= metrics["action_accuracy"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
