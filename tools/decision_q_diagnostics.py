"""Read terminal TD targets and residuals from a completed decoder CQL trial."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(plan_path, maximum_seconds):
    if not 0 < maximum_seconds <= 600:
        raise ValueError("Terminal readback needs a positive, bounded time budget.")
    started = time.monotonic()
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8-sig"))
    plan = read(plan_path)
    source, prefix = Path(plan["source"]), Path(plan["output_prefix"])
    artifact = lambda suffix: prefix.with_name(prefix.name + suffix)
    summary, proof = read(artifact("-summary.json")), read(artifact("-independent.json"))
    if (summary["status"] != "complete" or proof["status"] != "complete"
            or summary["plan_sha256"] != digest(plan_path)
            or proof["model_sha256"] != summary["model_sha256"]
            or digest(artifact(".kanachan")) != summary["model_sha256"]):
        raise ValueError("Terminal calibration requires the independently checked final model.")
    destination = artifact("-terminal-calibration.json")
    if destination.exists() or artifact(".lock").exists():
        raise ValueError("Terminal calibration exists or the original trial is still active.")
    if digest(plan["manifest"]) != plan["manifest_sha256"]:
        raise ValueError("Terminal calibration source manifest differs.")
    manifest = read(plan["manifest"])["complete_candidate"]
    for name, checksum in manifest.items():
        if digest(source / name) != checksum:
            raise ValueError(f"Terminal calibration source differs: {name}")
    identity = plan["dataset"]["test"]
    if digest(identity["path"]) != identity["sha256"]:
        raise ValueError("Terminal calibration test data differs.")
    os.environ.update(TORCHDYNAMO_DISABLE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1")
    sys.path.insert(0, str(source))
    import torch
    from kanachan.model_loader import load_model
    from kanachan.nn.decision_value import q_only
    from kanachan.training.core.offline_rl import EpisodeReplayBuffer
    from kanachan.training.core.rl.discrete_cql import td_components
    from kanachan.training.core.rl.rewards import configured_reward

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    random.seed(plan["seed"])
    torch.manual_seed(plan["seed"])
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    actor = load_model(artifact(".kanachan"), map_location=torch.device("cpu")).float().cuda().eval().requires_grad_(False)
    network = q_only(actor)

    class ForbiddenBootstrap(torch.nn.Module):
        def forward(self, data):
            raise AssertionError("A terminal target attempted to bootstrap.")

    loader = EpisodeReplayBuffer(
        training_data=Path(identity["path"]), contiguous_training_data=True,
        num_skip_samples=0, rewrite_rooms=None, rewrite_grades=None,
        get_reward=configured_reward(plan["reward_scheme"], score_unit=plan["round_score_unit"],
                                     round_weight=plan["round_reward_weight"]),
        discount_factor=1., dtype=torch.float32, max_size=8192,
        batch_size=128, num_workers=0, pin_memory=False,
    )
    count, predictions, rewards = 0, [], []
    with torch.no_grad():
        for batch in loader:
            if time.monotonic() - started > maximum_seconds:
                raise TimeoutError("Terminal readback exceeded its active time budget.")
            count += len(batch)
            terminal = batch["next", "done"].bool().reshape(-1)
            if not terminal.any():
                continue
            data = batch[terminal].cuda()
            _, chosen, target = td_components(network, ForbiddenBootstrap(), data)
            reward = data["next", "reward"].float().reshape(-1)
            if not torch.equal(target, reward):
                raise ValueError("Terminal TD target differs from observed immediate reward.")
            predictions.append(chosen.cpu())
            rewards.append(reward.cpu())
    q, reward = torch.cat(predictions).double(), torch.cat(rewards).double()
    if count != identity["rows"] or len(q) != identity["terminal_trajectories"]:
        raise ValueError("Terminal readback row or terminal coverage differs.")
    error = q - reward
    result = dict(
        status="complete", scope="heldout_terminal_td_targets_and_residuals_only",
        plan_sha256=digest(plan_path), diagnostic_sha256=digest(__file__),
        model_sha256=summary["model_sha256"], test_sha256=identity["sha256"],
        test_decisions=count, terminal_transitions=len(q),
        terminal_q_mean=q.mean().item(), terminal_reward_mean=reward.mean().item(),
        terminal_td_residual_mean=error.mean().item(), terminal_td_mae=error.abs().mean().item(),
        terminal_td_rmse=error.square().mean().sqrt().item(),
        terminal_q_percentiles=torch.quantile(q, torch.tensor([.05, .5, .95], dtype=torch.float64)).tolist(),
        terminal_reward_range=[reward.min().item(), reward.max().item()],
        all_terminal_targets_equal_observed_reward=True, terminal_bootstrap_calls=0,
        new_training_started=False, seconds=time.monotonic() - started,
        limitation="Residuals condition on observed termination; they are not unbiased current-policy Q errors or playing-strength estimates.",
    )
    for name, checksum in manifest.items():
        if digest(source / name) != checksum:
            raise ValueError("Source changed during terminal readback.")
    if digest(artifact(".kanachan")) != result["model_sha256"]:
        raise ValueError("Model changed during terminal readback.")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--maximum-seconds", type=int, required=True)
    args = parser.parse_args()
    run(args.plan, args.maximum_seconds)
