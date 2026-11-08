"""Placement-first single-host online Double DQN training.

Run with --config examples/training/online.yaml. Resume using --resume
OUTPUT/latest.pt; its self-contained model, replay, RNG and optimizer state
are authoritative. Iterations may be increased with --iterations.
"""

import argparse
import copy
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import random
import shutil
import time

import torch
from torch import nn
from tensordict import TensorDict

from kanachan.model_loader import load_model, load_model_state
from kanachan.nn import EpsilonGreedy, QRDecoder, TwinQActor
from kanachan.nn.decision_decoder import DenseDecisionDecoder, legal_argmax
from kanachan.nn.decision_value import q_only
from kanachan.simulation.statistics import (
    compare_policies,
    promotion_decision,
    summarize,
)
from kanachan.training.core.optimizers import online_adamw
from kanachan.training.common import backoff_gradient_scale
from kanachan.training.core.rl.multistep import make_multistep
from kanachan.training.core.rl.rewards import PlacementReward
from kanachan.training.core.rl.simulation import simulate as collect

from .config import Config, load_config
from .opponents import OpponentPool
from .q_learning import action_values, evaluating, q_loss
from .replay import ReplayBuffer
from .offline_pool import prepare_offline_pool, mixed_batch, is_pool_preparation
from .state import (
    atomic_save,
    capture_rng,
    cpu_state,
    export_model,
    file_digest,
    restore_rng,
)


class SamplingPolicy(nn.Module):
    def __init__(self, model, exploration):
        super().__init__()
        self.model = q_only(model)
        self.exploration = exploration
        self.stable_ties = any(
            isinstance(module, DenseDecisionDecoder) for module in model.modules()
        )

    @torch.no_grad()
    def forward(self, data: TensorDict):
        values = action_values(self.model, data)
        data["action"] = (
            legal_argmax(data["candidates"], values)
            if self.stable_ties else values.argmax(-1).to(torch.int32)
        )
        return self.exploration(data)


def _frozen_model(path, device):
    # Module constructors must not consume the training/exploration RNG.
    with torch.random.fork_rng(devices=[]):
        model = load_model(path, map_location=torch.device("cpu"))
    return model.float().to(device).eval().requires_grad_(False)


@torch.no_grad()
def evaluate(model, baseline, config, device):
    from kanachan.simulation import simulate

    with evaluating(model):
        logs = simulate(
            device,
            torch.float32,
            config.room,
            config.grade,
            baseline,
            config.grade,
            model,
            4,
            config.evaluation_sets,
            config.simulation_batch_size,
            config.simulation_concurrency,
            lambda: None,
            seed=config.evaluation_seed,
        )
    results = [log.get_result() for log in logs]
    group_ids = [log.get_metadata()["set_id"] for log in logs]
    expected = [
        index for index in range(config.evaluation_sets) for _ in range(4)
    ]
    if group_ids != expected:
        raise ValueError(
            "Evaluation results lost their seed/seat grouping; rebuild the native module."
        )
    return results, group_ids


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(
    config: Config,
    *,
    resume_state=None,
    collect_fn=collect,
    evaluate_fn=evaluate,
):
    config.validate()
    from torch._dynamo import config as dynamo_config

    dynamo_config.disable = (
        not config.enable_compile
        or os.environ.get("TORCHDYNAMO_DISABLE") == "1"
    )
    if resume_state is not None:
        permitted = {"output_dir", "iterations", "device"}
        original_config = Config(**resume_state["config"]).to_dict()
        if any(
            value != original_config[key]
            for key, value in config.to_dict().items()
            if key not in permitted
        ):
            raise ValueError(
                "Resume must retain the original algorithm and evaluation configuration."
            )
        if config.iterations < resume_state["iteration"]:
            raise ValueError("iterations is smaller than the saved iteration.")
    if (
        torch.distributed.is_initialized()
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
    ):
        raise ValueError(
            "The online entry point is single-host, single-learner. Do not launch with torchrun."
        )
    device = torch.device(config.device)
    torch.set_num_threads(config.torch_num_threads)
    if device.type not in ("cpu", "cuda") or (
        device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError(f"Unavailable training device: {device}.")
    if (
        config.amp_dtype == "bfloat16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("This GPU does not support bfloat16.")
    output = Path(config.output_dir).resolve()
    if (
        resume_state is None and output.exists() and any(output.iterdir())
        and not (config.offline_data is not None and is_pool_preparation(output))
    ):
        raise ValueError(
            f"{output}: output is not empty. Use --resume to continue."
        )
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if resume_state is None:
        template = torch.load(
            config.initial_model, map_location="cpu", weights_only=True
        )
        model = load_model_state(template, torch.device("cpu")).float()
    else:
        if resume_state.get("format_version") != 1:
            raise ValueError("Unsupported online checkpoint version.")
        template = resume_state["model"]
        model = load_model_state(template, torch.device("cpu")).float()
    if any(
        isinstance(module, (QRDecoder, TwinQActor))
        for module in model.modules()
    ) or "action_value" not in getattr(model, "out_keys", []):
        raise ValueError(
            "initial_model must be a scalar CQL .kanachan export, not BC, QR or IQL."
        )
    model.to(device).requires_grad_(True)
    target = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer, optimizer_policy = online_adamw(
        model, lr=config.learning_rate, weight_decay=config.weight_decay,
        device_type=device.type, resume_state=resume_state,
    )
    source_values, target_values = q_only(model), q_only(target)
    amp_enabled = device.type == "cuda" and config.amp_dtype != "float32"
    amp_dtype = getattr(torch, config.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    exploration = EpsilonGreedy(
        epsilon_first=config.epsilon_first,
        epsilon_last=config.epsilon_last,
        annealing_steps=config.epsilon_steps,
        device=device,
        dtype=torch.float32,
    )
    exploration.stable_action_ids = any(
        isinstance(module, DenseDecisionDecoder) for module in model.modules()
    )
    replay = ReplayBuffer(config.replay_capacity, config.seed)
    offline_replay, offline_identity = prepare_offline_pool(config, resume_state)
    pool = OpponentPool(
        output,
        config.max_history_opponents,
        None if resume_state is None else resume_state["opponents"],
    )
    iteration = samples = optimizer_steps = policy_version = 0
    best = export_model(template, model)
    best_results = None
    reward_identity = None
    if resume_state is None:
        for path in config.opponents or [config.initial_model]:
            pool.add_fixed(path)
        if config.reward_model is not None:
            reward_identity = file_digest(config.reward_model)
            reward_path = output / "reward.kanachan"
            with (
                Path(config.reward_model).open("rb") as source,
                reward_path.open("wb") as destination,
            ):
                shutil.copyfileobj(source, destination)
        _write_json(
            output / "run-info.json",
            {
                "config": config.to_dict(),
                "objective": "minimum expected final half-game rank",
                "utility": [1.5, 0.5, -0.5, -1.5],
                "discount_factor": 1.0,
                "initial_model_sha256": file_digest(config.initial_model),
                "offline_data_sha256": offline_identity,
                "optimizer_policy": optimizer_policy,
                "offline_pool_policy": (
                    None if offline_replay is None else offline_replay.metadata["policy"]
                ),
                "offline_pool_size": 0 if offline_replay is None else len(offline_replay),
                "offline_pool_source_rows": (
                    None if offline_replay is None
                    else offline_replay.metadata.get("source_rows")
                ),
                "reward_model_sha256": reward_identity,
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "fixed_opponents": pool.state_dict()["fixed"],
            },
        )
    else:
        target.load_state_dict(resume_state["target"])
        scaler.load_state_dict(resume_state["scaler"])
        exploration.load_state_dict(resume_state["exploration"])
        replay.load_state_dict(resume_state["replay"])
        iteration, samples = resume_state["iteration"], resume_state["samples"]
        optimizer_steps, policy_version = (
            resume_state["optimizer_steps"],
            resume_state["policy_version"],
        )
        best, best_results = resume_state["best"], resume_state["best_results"]
        reward_identity = resume_state["reward_model_sha256"]
    reward_path = output / "reward.kanachan"
    if reward_identity is not None and (
        not reward_path.is_file()
        or file_digest(reward_path) != reward_identity
    ):
        raise ValueError("Missing or modified shaping model.")
    reward = PlacementReward(
        reward_path if reward_identity is not None else None
    )
    if resume_state is not None:
        restore_rng(resume_state["rng"])

    def save():
        state = {
            "format_version": 1,
            "config": config.to_dict(),
            "iteration": iteration,
            "samples": samples,
            "optimizer_steps": optimizer_steps,
            "policy_version": policy_version,
            "model": export_model(template, model),
            "target": cpu_state(target.state_dict()),
            "optimizer": cpu_state(optimizer.state_dict()),
            "optimizer_policy": optimizer_policy,
            "scaler": scaler.state_dict(),
            "exploration": cpu_state(exploration.state_dict()),
            "replay": replay.state_dict(),
            "offline_replay": None if offline_replay is None else offline_replay.state_dict(),
            "offline_data_sha256": offline_identity,
            "opponents": pool.state_dict(),
            "best": best,
            "best_results": best_results,
            "reward_model_sha256": reward_identity,
            "rng": cpu_state(capture_rng()),
        }
        atomic_save(output / "latest.pt", state)
        atomic_save(output / "model.kanachan", state["model"])
        return state

    # A valid initial snapshot also makes interruption before the first rollout resumable.
    if resume_state is None:
        save()
        atomic_save(output / "best.kanachan", best)
    for iteration_index in range(iteration, config.iterations):
        start = time.monotonic()
        opponent = pool.choose()
        baseline = _frozen_model(pool.path(opponent), device)
        rollout_seed = random.randrange(2**63)
        while rollout_seed == config.evaluation_seed:
            rollout_seed = random.randrange(2**63)
        epsilon = float(exploration.epsilon.item())
        with evaluating(model):
            exploration.train()
            episodes = collect_fn(
                device=device,
                dtype=torch.float32,
                room=config.room,
                dongfengzhan=False,
                grade=config.grade,
                baseline_model=baseline,
                policy_model=SamplingPolicy(model, exploration),
                num_simulations=config.games_per_iteration,
                simulation_batch_size=config.simulation_batch_size,
                simulation_concurrency=config.simulation_concurrency,
                progress=lambda: None,
                get_reward=reward,
                seed=rollout_seed,
                policy_version=policy_version,
                opponent_version=opponent["version"],
            )
        del baseline
        collected = len(episodes)
        if collected:
            episodes["opponent_id"] = torch.full(
                (collected,), int(opponent["id"][:15], 16), dtype=torch.int64
            )
            make_multistep(episodes, 1.0, config.td_steps)
            replay.append(episodes)
        samples += collected
        exploration.step(collected)
        fraction = (
            min(1.0, samples / config.cql_annealing_steps)
            if config.cql_annealing_steps
            else 1.0
        )
        alpha = config.cql_alpha_first + fraction * (
            config.cql_alpha_last - config.cql_alpha_first
        )
        losses = []
        skipped_updates = 0
        model.train()
        offline_batch_size = 0
        if len(replay) >= config.learning_starts or offline_replay is not None:
            for _ in range(config.updates_per_iteration):
                data, coefficient, offline_batch_size = mixed_batch(
                    replay, offline_replay, config, alpha
                )
                data = data.to(device)
                if isinstance(coefficient, torch.Tensor):
                    coefficient = coefficient.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                    cache_enabled=False,
                ):
                    loss, metrics = q_loss(
                        source_values, target_values, data, coefficient,
                        td_loss=config.td_loss, temperature=config.cql_temperature,
                        tensor_metrics=True,
                    )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite online Q loss.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_gradient_norm
                )
                if not torch.isfinite(norm):
                    if not amp_enabled:
                        raise ValueError("Nonfinite online Q gradient.")
                    backoff_gradient_scale(scaler)
                    skipped_updates += 1
                    continue
                scaler.step(optimizer)
                scaler.update()
                optimizer_steps += 1
                if optimizer_steps % config.target_update_interval == 0:
                    with torch.no_grad():
                        for source_value, target_value in zip(
                            model.parameters(),
                            target.parameters(),
                            strict=True,
                        ):
                            target_value.lerp_(
                                source_value, config.target_update_rate
                            )
                        for source_value, target_value in zip(
                            model.buffers(), target.buffers(), strict=True
                        ):
                            target_value.copy_(source_value)
                metrics["loss"] = loss.detach()
                metrics["gradient_norm"] = norm.detach()
                losses.append(metrics)
        if losses:
            policy_version += 1
            keys = list(losses[0])
            # One host transfer per iteration, rather than one per metric and
            # update. Finite-loss/gradient safety checks above remain in place.
            means = torch.stack([
                torch.stack([item[key] for key in keys]) for item in losses
            ]).double().mean(0).cpu().tolist()
            training_metrics = dict(zip(keys, means, strict=True))
        else:
            training_metrics = None
        iteration = iteration_index + 1
        record = {
            "iteration": iteration,
            "policy_version": policy_version,
            "samples": samples,
            "collected": collected,
            "replay_size": len(replay),
            "offline_pool_size": 0 if offline_replay is None else len(offline_replay),
            "offline_batch_size": offline_batch_size,
            "optimizer_steps": optimizer_steps,
            "epsilon": epsilon,
            "cql_alpha": alpha,
            "opponent": opponent,
            "rollout_seed": rollout_seed,
            "skipped_updates": skipped_updates,
            "training": training_metrics,
        }
        if config.evaluate_every and iteration % config.evaluate_every == 0:
            rng = capture_rng()
            try:
                evaluation_baseline = _frozen_model(
                    pool.path(pool.fixed[0]), device
                )
                if best_results is None:
                    incumbent = (
                        load_model_state(best, torch.device("cpu"))
                        .float()
                        .to(device)
                        .eval()
                    )
                    best_results, _ = evaluate_fn(
                        incumbent, evaluation_baseline, config, device
                    )
                    del incumbent
                candidate_results, groups = evaluate_fn(
                    model, evaluation_baseline, config, device
                )
                del evaluation_baseline
                comparison = compare_policies(
                    candidate_results,
                    best_results,
                    group_ids=groups,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=config.evaluation_seed,
                    confidence=config.confidence,
                )
                promoted = promotion_decision(
                    comparison,
                    min_groups=config.min_evaluation_groups,
                    min_improvement=config.min_rank_improvement,
                    max_fourth_rate_increase=config.max_fourth_rate_increase,
                )
                record["evaluation"] = {
                    "comparison": comparison,
                    "promoted": promoted,
                    "summary": summarize(
                        candidate_results,
                        group_ids=groups,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.evaluation_seed,
                        confidence=config.confidence,
                    ),
                }
                if promoted:
                    best = export_model(template, model)
                    best_results = candidate_results
                    record["evaluation"]["opponent"] = pool.promote(
                        best, policy_version
                    )
                    atomic_save(output / "best.kanachan", best)
                _write_json(
                    output / f"evaluation.{iteration}.json",
                    {
                        **record["evaluation"],
                        "seed": config.evaluation_seed,
                        "group_ids": groups,
                        "candidate": candidate_results,
                        "incumbent_comparison": comparison,
                    },
                )
            finally:
                restore_rng(rng)
        record["elapsed_seconds"] = time.monotonic() - start
        save()
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
        logging.info(
            "iteration=%d version=%d decisions=%d updates=%d loss=%s",
            iteration,
            policy_version,
            samples,
            optimizer_steps,
            record["training"],
        )
    return {
        "iteration": iteration,
        "samples": samples,
        "optimizer_steps": optimizer_steps,
        "policy_version": policy_version,
        "checkpoint": str(output / "latest.pt"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    state = None
    if arguments.resume:
        state = torch.load(
            arguments.resume, map_location="cpu", weights_only=True
        )
        config = Config(**state["config"])
        config = replace(
            config, output_dir=str(arguments.resume.resolve().parent)
        )
    else:
        config = load_config(arguments.config)
    changes = {
        key: getattr(arguments, key)
        for key in ("iterations", "device", "output_dir")
        if getattr(arguments, key) is not None
    }
    if "output_dir" in changes:
        changes["output_dir"] = str(changes["output_dir"].resolve())
    config = replace(config, **changes)
    result = run(config, resume_state=state)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
