"""Real two-process checks of IQL target initialization and continuation."""

import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import distributed


def _flat_state(network):
    return torch.cat(
        [
            tensor.detach().cpu().float().reshape(-1)
            for tensor in (*network.parameters(), *network.buffers())
        ]
    )


def _equal_across_ranks(value):
    gathered = [torch.empty_like(value) for _ in range(2)]
    distributed.all_gather(gathered, value)
    return torch.equal(gathered[0], gathered[1])


def _worker(root, mode, rank):
    from hydra import compose, initialize
    from hydra.core.hydra_config import HydraConfig
    import kanachan.training.iql.train as train
    from kanachan.training.core.bc import DataLoader

    torch.set_num_threads(1)
    torch.manual_seed(20261108 + rank)
    result = {"rank": rank, "mode": mode}
    overrides = [
        "device=cpu",
        "encoder.dimension=16",
        "encoder.num_heads=2",
        "encoder.dim_feedforward=24",
        "encoder.num_layers=1",
        "num_workers=0",
        "batch_size=4",
        "final_rank_weight=0.1",
        f"training_data={(root / 'data.tsv').as_posix()}",
        f"reward_plugin={(Path(__file__).resolve().parents[2] / 'examples/training/iql-placement-reward.py').as_posix()}",
    ]
    if mode == "warmstart":
        overrides.append(
            f"bc_initial_model={(root / 'bc/snapshots/model.kanachan').as_posix()}"
        )
    else:
        overrides += [
            f"initial_model_prefix={(root / 'warmstart/snapshots').as_posix()}",
            "start_new_epoch=true",
        ]
    with initialize(version_base=None, config_path=None):
        config = compose(config_name="decision", overrides=overrides)

    def initialize_group(*, backend):
        distributed.init_process_group(
            backend=backend,
            init_method=(root / f"store-{mode}").as_uri(),
            rank=rank,
            world_size=2,
            timeout=datetime.timedelta(seconds=60),
        )

    def observe(arguments):
        state = {
            "sources_synchronized": [],
            "targets_synchronized": [],
            "targets_match_sources": [],
        }
        batch = next(
            DataLoader(
                path=root / "data.tsv",
                num_skip_samples=0,
                rewrite_rooms=None,
                rewrite_grades=None,
                batch_size=2,
            )
        )
        minimum = None
        for number in (1, 2):
            source = arguments[f"q{number}_source_network"]
            target = arguments[f"q{number}_target_network"]
            source_state, target_state = (
                _flat_state(source),
                _flat_state(target),
            )
            state["sources_synchronized"].append(
                _equal_across_ranks(source_state)
            )
            state["targets_synchronized"].append(
                _equal_across_ranks(target_state)
            )
            state["targets_match_sources"].append(
                torch.equal(target_state, source_state)
            )
            with torch.no_grad():
                values = target(batch.clone())["action_value"]
                logged = values.gather(
                    1, batch["action"].long()[:, None]
                ).squeeze(1)
                minimum = (
                    logged
                    if minimum is None
                    else torch.minimum(minimum, logged)
                )
        state["v_supervision_synchronized"] = _equal_across_ranks(minimum)
        return state

    original_training = train._training

    def inspect_and_train(**arguments):
        result["before"] = observe(arguments)
        if mode == "resume":
            matches = []
            for number in (1, 2):
                target = arguments[f"q{number}_target_network"]
                for part, tdm in zip(("encoder", "decoder"), target.module):
                    expected = torch.load(
                        root
                        / f"warmstart/snapshots/q{number}-target-{part}.pth",
                        map_location="cpu",
                        weights_only=True,
                    )
                    actual = tdm.module.state_dict()
                    matches.append(
                        actual.keys() == expected.keys()
                        and all(
                            torch.equal(actual[key].cpu(), value)
                            for key, value in expected.items()
                        )
                    )
            result["loaded_targets_preserved"] = all(matches)
        original_training(**arguments)
        result["after"] = observe(arguments)

    try:
        with (
            patch.object(
                train, "init_process_group", side_effect=initialize_group
            ),
            patch.object(train, "_training", side_effect=inspect_and_train),
            patch.object(
                HydraConfig,
                "get",
                return_value=SimpleNamespace(
                    runtime=SimpleNamespace(output_dir=str(root / mode))
                ),
            ),
        ):
            train._main.__wrapped__(config)
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()
    (root / f"{mode}-rank-{rank}.json").write_text(
        json.dumps(result), encoding="utf-8"
    )


@unittest.skipUnless(
    distributed.is_gloo_available()
    and os.environ.get("KANACHAN_TEST_DISTRIBUTED") == "1",
    "Set KANACHAN_TEST_DISTRIBUTED=1 for the real two-process Gloo test",
)
class DecisionIQLDistributedTests(unittest.TestCase):
    def test_bc_warmstart_target_sync_and_preserved_resume_lag(self):
        import test_decision_iql as fixtures
        from test_training_regressions import bc_line

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # The loader consumes four samples per rank in each update.
            (root / "data.tsv").write_text(bc_line(0) * 16, encoding="utf-8")
            fixtures.DecisionIQLTests().cli(
                "bc", root / "data.tsv", root / "bc"
            )
            evidence = []
            for mode in ("warmstart", "resume"):
                processes = []
                try:
                    for rank in range(2):
                        environment = dict(
                            os.environ,
                            WORLD_SIZE="2",
                            RANK=str(rank),
                            LOCAL_RANK=str(rank),
                            OMP_NUM_THREADS="1",
                            MKL_NUM_THREADS="1",
                            TORCHDYNAMO_DISABLE="1",
                            PYTHONDONTWRITEBYTECODE="1",
                        )
                        environment["PYTHONPATH"] = os.pathsep.join(
                            (
                                str(Path(__file__).resolve().parents[2]),
                                str(Path(__file__).resolve().parent),
                                environment.get("PYTHONPATH", ""),
                            )
                        )
                        processes.append(
                            subprocess.Popen(
                                [
                                    sys.executable,
                                    "-B",
                                    str(Path(__file__).resolve()),
                                    "--worker",
                                    str(root),
                                    mode,
                                    str(rank),
                                ],
                                cwd=Path(__file__).resolve().parents[2],
                                env=environment,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                encoding="utf-8",
                                errors="replace",
                            )
                        )
                    for process in processes:
                        stdout, stderr = process.communicate(timeout=150)
                        self.assertEqual(
                            process.returncode, 0, stdout + stderr
                        )
                finally:
                    for process in processes:
                        if process.poll() is None:
                            process.terminate()
                            process.communicate(timeout=15)
                results = [
                    json.loads((root / f"{mode}-rank-{rank}.json").read_text())
                    for rank in range(2)
                ]
                evidence.extend(results)
                for result in results:
                    for stage in ("before", "after"):
                        state = result[stage]
                        self.assertTrue(
                            all(state["sources_synchronized"]), result
                        )
                        self.assertTrue(
                            all(state["targets_synchronized"]), result
                        )
                        self.assertTrue(
                            state["v_supervision_synchronized"], result
                        )
                    if mode == "warmstart":
                        self.assertTrue(
                            all(result["before"]["targets_match_sources"]),
                            result,
                        )
                    else:
                        self.assertTrue(
                            result["loaded_targets_preserved"], result
                        )
                        self.assertFalse(
                            any(result["before"]["targets_match_sources"]),
                            result,
                        )
                training = json.loads(
                    (root / mode / "snapshots/training-state.json").read_text()
                )
                self.assertEqual(
                    training["optimizer_steps"],
                    2 if mode == "warmstart" else 4,
                )
            print(
                "IQL_TARGET_SYNC_RESULT: " + json.dumps(evidence), flush=True
            )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    else:
        unittest.main()
