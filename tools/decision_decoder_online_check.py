"""Bounded real native/CUDA online entry-point and interrupted-run check."""

import json
import os
from pathlib import Path
import sys
from dataclasses import replace

os.environ.update(
    OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TORCHDYNAMO_DISABLE="1",
    CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1",
)
ROOT = Path(r"C:\Users\Suini\kanachan-data-work")
DIAGNOSTIC = ROOT / "diagnostics/decision-decoder-20261104-v1"
sys.path.insert(0, str(DIAGNOSTIC / "source"))

import torch

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.45)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.use_deterministic_algorithms(True)

from kanachan.training.online.config import Config
from kanachan.training.online.train import run


def compare(left, right, name):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0, msg=name)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f"State keys differ: {name}")
        for key in left:
            compare(left[key], right[key], f"{name}.{key}")
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError(f"State sequence differs: {name}")
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            compare(a, b, f"{name}.{i}")
    elif left != right:
        raise ValueError(f"State scalar differs: {name}")


def main():
    if (DIAGNOSTIC / "online-entry-full").exists() or (DIAGNOSTIC / "online-entry-partial").exists():
        raise ValueError("Online verification outputs exist; inspect before rerunning.")
    config = Config(
        initial_model=str(DIAGNOSTIC / "cql/snapshots/model.kanachan"),
        output_dir=str(DIAGNOSTIC / "online-entry-full"),
        opponents=[str(ROOT / "experiments/bc-tiny-20261103-v2/model.kanachan")],
        offline_data=str(DIAGNOSTIC / "data/train.rl.tsv"),
        offline_capacity=128, offline_fraction=0.5, offline_cql_alpha=0.1,
        device="cuda:0", amp_dtype="float16", seed=2026110403,
        evaluation_seed=2026110404, iterations=2, games_per_iteration=1,
        simulation_batch_size=1, simulation_concurrency=1,
        replay_capacity=256, learning_starts=1, batch_size=32,
        updates_per_iteration=2, evaluate_every=0, td_loss="huber",
        cql_alpha_first=0.05, cql_alpha_last=0.05, cql_annealing_steps=0,
        epsilon_first=0.05, epsilon_last=0.05, epsilon_steps=0,
    )
    run(config)
    partial = replace(
        config, output_dir=str(DIAGNOSTIC / "online-entry-partial"), iterations=1
    )
    run(partial)
    saved = torch.load(
        DIAGNOSTIC / "online-entry-partial/latest.pt", weights_only=True,
        map_location="cpu",
    )
    run(replace(partial, iterations=2), resume_state=saved)
    full = torch.load(
        DIAGNOSTIC / "online-entry-full/latest.pt", weights_only=True,
        map_location="cpu",
    )
    resumed = torch.load(
        DIAGNOSTIC / "online-entry-partial/latest.pt", weights_only=True,
        map_location="cpu",
    )
    fields = [
        "target", "optimizer", "scaler", "exploration", "replay",
        "offline_replay", "offline_data_sha256", "rng", "samples",
        "optimizer_steps", "policy_version",
    ]
    compare(full["model"]["state_dict"], resumed["model"]["state_dict"], "model")
    for field in fields:
        compare(full[field], resumed[field], field)
    if full["optimizer_steps"] != 4:
        raise ValueError("Expected four successful actual CUDA online updates.")
    records = [
        json.loads(line) for line in (
            DIAGNOSTIC / "online-entry-full/metrics.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    if any(record["offline_batch_size"] != 16 for record in records):
        raise ValueError("Online learner did not use the declared 50/50 replay mix.")
    proof = {
        "status": "complete",
        "scope": "real_native_cuda_online_entry_and_exact_resume_only",
        "complete_native_collection_runs": 4,
        "uninterrupted_iterations": 2,
        "successful_cuda_updates_per_final_run": full["optimizer_steps"],
        "collected_decisions_per_final_run": full["samples"],
        "offline_pool_size": full["offline_replay"]["size"],
        "offline_samples_per_batch": 16,
        "online_samples_per_batch": 16,
        "amp_dtype": "float16",
        "identical_resume_components": ["model", *fields],
        "strength_improvement_established": False,
    }
    (DIAGNOSTIC / "online-entry-acceptance.json").write_text(
        json.dumps(proof, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
