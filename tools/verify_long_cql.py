"""Exercise the normal long CQL workflow and exact cross-epoch CUDA recovery."""

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

from train_decision_bc import sha256, write_json
from train_decision_cql import equal, run


def verify(plan_path, output):
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    if output.exists() or not output.parent.is_dir():
        raise ValueError("The cross-epoch verification output already exists or has no parent.")
    sys.path.insert(0, plan["source"])
    import torch
    from kanachan.training.cql import train as cql

    torch.set_num_interop_threads(1)
    fixture = plan["pilot_data"] | dict(
        rows=627, terminal_trajectories=4,
        bytes=Path(plan["pilot_data"]["path"]).stat().st_size,
    )
    variants = {}
    for name in ("continuous", "resumed"):
        candidate = copy.deepcopy(plan)
        prefix = output.with_name("long-mc-epochs-" + name)
        for suffix in ("-plan.json", "-status.json", "-checkpoint.pt", ".kanachan", "-summary.json"):
            if prefix.with_name(prefix.name + suffix).exists():
                raise ValueError("A previous cross-epoch verification artifact exists.")
        candidate.update(
            scope="cross_epoch_workflow_functional_fixture_only",
            output_prefix=str(prefix), dataset={split: dict(fixture) for split in ("train", "validation", "test")},
            max_epochs=3, training_budget_seconds=120, maximum_seconds=300,
            native_sets=0, checkpoint_interval=64,
        )
        candidate.pop("original_uuid_split_preserved", None)
        candidate.pop("uuid_disjoint", None)
        path = prefix.with_name(prefix.name + "-plan.json")
        write_json(path, candidate)
        variants[name] = (path, prefix)

    class Interrupted(RuntimeError):
        pass

    original = cql._training
    calls = []

    def interrupt_second_epoch(**arguments):
        calls.append(None)
        snapshot = arguments["snapshot_writer"]

        def save_and_interrupt(index, **metadata):
            snapshot(index, **metadata)
            if len(calls) == 2 and index == 64:
                raise Interrupted("Intentional CUDA verification interruption after a saved checkpoint.")

        arguments["snapshot_writer"] = save_and_interrupt
        return original(**arguments)

    # The runtime is already configured for one interop thread. Repeated calls
    # to the same normal workflow must retain that setting in this process.
    with patch.object(torch, "set_num_interop_threads"), contextlib.redirect_stdout(io.StringIO()):
        run(variants["continuous"][0])
        try:
            with patch.object(cql, "_training", side_effect=interrupt_second_epoch):
                run(variants["resumed"][0])
        except Interrupted:
            pass
        else:
            raise ValueError("The intended cross-epoch interruption was not reached.")
        prefix = variants["resumed"][1]
        partial = torch.load(prefix.with_name(prefix.name + "-checkpoint.pt"), map_location="cpu", weights_only=True)
        if (partial["epoch"], partial["cursor"], partial["attempted_updates"]) != (1, 64, 22):
            raise ValueError("The CUDA interruption checkpoint has another exact cursor.")
        run(variants["resumed"][0], resume=True)

    results = {}
    for name, (_, prefix) in variants.items():
        results[name] = torch.load(prefix.with_name(prefix.name + "-checkpoint.pt"), map_location="cpu", weights_only=True)
    for key in ("model", "target", "optimizer", "scheduler", "scaler", "rng", "replay", "cursor",
                "epoch", "total_samples", "completed_epochs", "attempted_updates", "optimizer_steps"):
        equal(results["continuous"][key], results["resumed"][key])
    final = results["resumed"]
    if (final["total_samples"], final["completed_epochs"], final["attempted_updates"]) != (1881, 3, 60):
        raise ValueError("CUDA epoch totals do not include all three short final batches.")
    if final["optimizer_steps"] + final["run_state"]["skipped_updates"] != 60 or final["target"] is not None:
        raise ValueError("CUDA optimizer/AMP counts or MC target network differ.")
    receipt = dict(
        status="complete", scope="real_cuda_multi_epoch_workflow_functional_only",
        plan_sha256=sha256(plan_path), controller_sha256=plan["controller_sha256"],
        source_manifest_sha256=plan["manifest_sha256"], helper_sha256=sha256(__file__),
        real_decisions_per_epoch=627, completed_epochs=3, total_samples=1881,
        attempted_updates=60, successful_updates=final["optimizer_steps"],
        amp_skipped_updates=final["run_state"]["skipped_updates"],
        interrupted_epoch=1, interrupted_cursor=64, exact_resume_all_state=True,
        target_network=None, training_validation_test_share_fixture=True,
        model_sha256={name: sha256(prefix.with_name(prefix.name + ".kanachan")) for name, (_, prefix) in variants.items()},
    )
    write_json(output, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    verify(arguments.plan, arguments.output)
