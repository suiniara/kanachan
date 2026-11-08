"""Actual-data round reward, CUDA CQL exact-resume and native functional checks."""

import argparse
import copy
from datetime import datetime, timezone
import io
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

from train_decision_bc import sha256, write_json, fresh_descriptor
from train_decision_cql import equal


def run(plan_path, manifest_path, output):
    if not output.parent.is_dir() or output.exists() or output.with_name(output.stem + "-status.json").exists():
        raise ValueError("The round-reward probe already has a receipt; inspect it.")
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    source = Path(plan["source"])
    expected = json.loads(manifest_path.read_text(encoding="utf-8-sig"))["complete_candidate"]
    for name, checksum in expected.items():
        if sha256(source / name) != checksum:
            raise ValueError(f"Candidate source changed: {name}")
    for name in ("bc", "opponent", "template", "pilot_data"):
        if sha256(plan[name]["path"]) != plan[name]["sha256"]:
            raise ValueError(f"Preserved {name} input differs.")
    os.environ.update(TORCHDYNAMO_DISABLE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1")
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from kanachan.model_loader import load_model, load_model_state
    from kanachan.nn.decision_decoder import DenseDecisionDecoder, GlobalActionHead
    from kanachan.nn.decision_value import q_only, transfer_bc_features
    from kanachan.training.cql import train as cql
    from kanachan.training.core.offline_rl import DataLoader
    from kanachan.training.core.config.optimizer import AdamWOptimizerConfig, create
    from kanachan.training.core.rl.discrete_cql import (
        td_components, td_penalty,
    )
    from kanachan.training.core.rl.rewards import PlacementReward, RoundAndPlacementReward
    from kanachan.training.online.state import cpu_state, capture_rng, export_model, atomic_save
    from kanachan.simulation import simulate
    from kanachan.simulation.statistics import summarize

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    started = time.monotonic()
    reward = RoundAndPlacementReward(round_weight=.2)
    proof = dict(status="running", scope="standard_td_cql_cuda_and_native_functional_only",
                 source_manifest_sha256=sha256(manifest_path), helper_sha256=sha256(__file__),
                 old_models_preserved=True, production_sources_modified=False,
                 objective="0.2 * round net own points / 10000 + final placement [1.5,.5,-.5,-1.5]",
                 target_kind="one_step_double_dqn", feature_freezing=False, logged_return_mixing=False,
                 learned_reward_predictor=False, alpha=.5, td_loss="mse")
    status_path = output.with_name(output.stem + "-status.json")

    def status(phase):
        if phase != "complete" and time.monotonic()-started > 900:
            raise TimeoutError("The bounded TD-CQL functional check exceeded 15 minutes.")
        write_json(status_path, dict(phase=phase, seconds=time.monotonic()-started,
                                    pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(),
                                    status="complete" if phase == "complete" else "running"))
        print(json.dumps(dict(phase=phase)), flush=True)

    status("real_reward_reference")
    reader = DataLoader(path=Path(plan["dataset"]["train"]["path"]), num_skip_samples=0,
                        rewrite_rooms=None, rewrite_grades=None, get_reward=PlacementReward(),
                        dtype=torch.float32, batch_size=4096, num_workers=0)
    chunks, games, old_nonzero, new_nonzero, rounds, rows = [], 0, 0, 0, 0, 0
    max_error = 0.
    return_range = [float("inf"), -float("inf")]
    for batch in reader:
        start = 0
        for end in batch["next", "end_of_game"].nonzero().flatten().tolist():
            chunks.append(batch[start:end+1])
            episode = torch.cat(chunks)
            chunks = []
            reference = torch.zeros(len(episode))
            previous = None
            seat = int(episode["sparse"][0, 6])-71
            for index in episode["next", "end_of_round"].nonzero().flatten().tolist():
                result = episode["next", "results"][index].tolist()
                if previous is None:
                    previous = result[6+seat]-result[seat]
                    first_score = previous
                terminal = bool(episode["next", "end_of_game"][index])
                settled = result[10+seat] if terminal else result[6+seat]
                reference[index] = .2*(settled-previous)/10000
                previous = settled
                rounds += 1
                if terminal:
                    scores = result[10:14]
                    rank = sorted(range(4), key=lambda position: (-scores[position], position)).index(seat)
                    reference[index] += 1.5-rank
            old_nonzero += int((episode["next", "reward"] != 0).sum())
            reward(episode, True)
            actual = episode["next", "reward"]
            error = float((actual-reference).abs().max())
            if error > 2e-6 or abs(float(actual.sum())-(.2*(previous-first_score)/10000+1.5-rank)) > 3e-6:
                raise ValueError("Real reward differs from independent endpoint accounting.")
            max_error = max(max_error, error)
            new_nonzero += int((actual != 0).sum())
            rows += len(episode)
            games += 1
            return_range[0] = min(return_range[0], float(actual.sum()))
            return_range[1] = max(return_range[1], float(actual.sum()))
            start = end+1
            if games == 32:
                break
        if games == 32:
            break
        chunks.append(batch[start:])
    if games != 32 or new_nonzero <= old_nonzero:
        raise ValueError("Real reward coverage or density check failed.")
    proof["real_reward"] = dict(observer_games=games, rows=rows, round_boundaries=rounds,
                                old_nonzero_rewards=old_nonzero, new_nonzero_rewards=new_nonzero,
                                max_absolute_error=max_error, observed_episode_return_range=return_range)
    template = torch.load(plan["template"]["path"], weights_only=True, map_location="cpu")

    def initialize():
        seed = 2026110411
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        bc = load_model(plan["bc"]["path"], map_location=torch.device("cpu")).float()
        actor = load_model_state(fresh_descriptor(template), map_location=torch.device("cpu")).float()
        body = next(m for m in actor.modules() if isinstance(m, DenseDecisionDecoder))
        head = next(m for m in actor.modules() if isinstance(m, GlobalActionHead))
        transfer_bc_features(body, head, bc)
        actor.cuda().train().requires_grad_(True)
        if sum(p.numel() for p in actor.parameters()) != plan["parameters"]:
            raise ValueError("Controlled model parameter count differs.")
        net = actor.select_subsequence(out_keys=["action_value"])
        target = copy.deepcopy(net).eval().requires_grad_(False)
        optimizer, scheduler = create("cuda", AdamWOptimizerConfig(
            learning_rate=1e-4, warmup_steps=2, weight_decay=.01), net)
        return actor, net, target, optimizer, scheduler

    class Reached(Exception):
        pass

    class NullWriter:
        def add_scalar(self, *args):
            pass

    def optimize(saved=None, interrupt=False):
        actor, net, target, optimizer, scheduler = initialize()
        if saved is not None:
            net.load_state_dict(saved["model"])
            target.load_state_dict(saved["target"])
            optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
            scheduler.load_state_dict(saved["scheduler"])
        first = q_only(net)
        second = q_only(target)
        result = {}
        initial_rng = capture_rng() if saved is None else saved["rng"]
        def feature_hash():
            checksum = hashlib.sha256()
            body = next(m for m in actor.modules() if isinstance(m, DenseDecisionDecoder))
            hidden = next(m for m in actor.modules() if isinstance(m, GlobalActionHead)).layers[0]
            for module in (body, hidden):
                for name, value in module.state_dict().items():
                    checksum.update(name.encode())
                    checksum.update(value.detach().cpu().contiguous().numpy().tobytes())
            return checksum.hexdigest()
        initial_features = feature_hash()
        def td(data):
            if not all(p.requires_grad for p in net.parameters()):
                raise ValueError("An online-network parameter was frozen.")
            values, chosen, expected_return = td_components(first, second, data)
            data["action_value"] = values
            data["td_target"] = expected_return
            data["td_error"] = 2 * td_penalty(chosen, expected_return, "mse")

        def snapshot(index, **metadata):
            if index is not None and not interrupt:
                return
            result.clear()
            result.update(cursor=index if index is not None else metadata["resume_num_samples"],
                          model=cpu_state(net.state_dict()), target=cpu_state(target.state_dict()),
                          optimizer=cpu_state(optimizer.state_dict()), scheduler=scheduler.state_dict(),
                          scaler=metadata["grad_scaler_state"], rng=capture_rng(), replay=metadata["replay_state"],
                          optimizer_steps=metadata["optimizer_steps"])
            if index is not None:
                raise Reached()

        try:
            with patch("kanachan.training.core.rl.rewards.load_model", side_effect=AssertionError("Reward predictor used")):
                cql._training(
                training_data=Path(plan["pilot_data"]["path"]),
                contiguous_training_data=True, rewrite_rooms=None, rewrite_grades=None, num_workers=0,
                replay_buffer_size=128,
                device=device, dtype=torch.float32, amp_dtype=torch.float16,
                source_network=net, target_network=target, reward_plugin=None,
                reward_scheme="round_and_placement", round_score_unit=10000., round_reward_weight=.2,
                discount_factor=1., td_error_computation=td, alpha=.5, batch_size=32,
                gradient_accumulation_steps=1, max_gradient_norm=1., optimizer=optimizer,
                scheduler=scheduler, target_update_interval=1, target_update_rate=.005,
                snapshot_interval=64, num_samples=0 if saved is None else saved["cursor"],
                summary_writer=NullWriter(), snapshot_writer=snapshot,
                initial_optimizer_steps=0 if saved is None else saved["optimizer_steps"],
                initial_grad_scaler_state=None if saved is None else saved["scaler"],
                replay_state=None if saved is None else saved["replay"],
                initial_rng_state=initial_rng, q_monitor_interval=1,
                )
        except Reached:
            if not interrupt:
                raise
        result["trainable_parameters"] = sum(p.numel() for p in actor.parameters() if p.requires_grad)
        result["optimizer_parameter_tensors"] = len(optimizer.state)
        result["feature_parameters_changed"] = initial_features != feature_hash()
        if not interrupt:
            counts = {int(v["step"]) for v in optimizer.state.values()}
            if counts != {result["optimizer_steps"]} or len(optimizer.state) != len(list(net.parameters())):
                raise ValueError("Not every online parameter was optimized at every successful update.")
            if not result["feature_parameters_changed"] or result["trainable_parameters"] != plan["parameters"]:
                raise ValueError("The full transferred feature network did not learn.")
        return actor, result

    status("cuda_exact_resume")
    actor, continuous = optimize()
    _, partial = optimize(interrupt=True)
    serialized = io.BytesIO()
    torch.save(partial, serialized)
    serialized.seek(0)
    _, resumed = optimize(saved=torch.load(serialized, map_location="cpu", weights_only=True))
    equal(continuous, resumed)
    if continuous["optimizer_steps"] <= 0 or continuous["cursor"] != 627:
        raise ValueError("Real CQL pilot cursor or updates differ.")
    if continuous["replay"]["reward_identity"] != f"round-and-placement-v1:unit={10000.:.17g}:weight={.2:.17g}":
        raise ValueError("Replay is not bound to the new reward objective.")
    model_path = output.with_suffix(".kanachan")
    atomic_save(model_path, export_model(template, actor))
    proof["cuda_cql"] = dict(samples=continuous["cursor"], optimizer_steps=continuous["optimizer_steps"],
                             exact_resume_all_state=True, model_sha256=sha256(model_path),
                             replay_reward_identity=continuous["replay"]["reward_identity"])
    proof["cuda_cql"].update(
        trainable_parameters=continuous["trainable_parameters"],
        optimizer_parameter_tensors=continuous["optimizer_parameter_tensors"],
        feature_parameters_changed=continuous["feature_parameters_changed"],
        actual_optimizer_counters=sorted({int(v["step"]) for v in continuous["optimizer"]["state"].values()}))
    checkpoint_path = output.with_name("td-cql-checkpoint.pt")
    continuous.update(target_kind="one_step_double_dqn", source_manifest_sha256=sha256(manifest_path))
    atomic_save(checkpoint_path, continuous)
    proof.update(plan=str(plan_path), plan_sha256=sha256(plan_path), source=str(source), manifest=str(manifest_path),
                 model=str(model_path), checkpoint=str(checkpoint_path), checkpoint_sha256=sha256(checkpoint_path))
    status("native_functional")
    opponent = load_model(plan["opponent"]["path"], map_location=torch.device("cpu")).float().cuda().eval().requires_grad_(False)
    actor.eval().requires_grad_(False)
    with torch.inference_mode():
        logs = simulate(device, torch.float32, 3, 12, opponent, 12, actor,
                        4, 1, 1, 1, lambda: None, seed=2026110411)
    results = [value.get_result() for value in logs]
    proof["native_functional"] = dict(results=results, statistics=summarize(results), seed=2026110411)
    if len(results) != 4:
        raise ValueError("Native probe did not complete all four seat rotations.")
    proof.update(status="complete", seconds=time.monotonic()-started,
                 limitation="Functional checks only; four games and a 627-row pilot do not prove stronger play.")
    for name in ("bc", "opponent", "template", "pilot_data"):
        if sha256(plan[name]["path"]) != plan[name]["sha256"]:
            raise ValueError(f"Preserved {name} changed during the probe.")
    for name, checksum in expected.items():
        if sha256(source / name) != checksum:
            raise ValueError(f"Source changed during the functional check: {name}")
    write_json(output, proof)
    status("complete")
    print(json.dumps({key: proof[key] for key in ("status", "scope", "cuda_cql", "seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.plan, args.manifest, args.output)
    except BaseException as error:
        import traceback
        if args.output.parent.is_dir():
            write_json(args.output.with_name(args.output.stem+"-error.json"),
                       dict(status="failed", error=str(error), traceback=traceback.format_exc(), pid=os.getpid()))
        raise
