"""Bounded real-data MC-CQL verification in the existing development folder."""

import argparse
import copy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

from train_decision_bc import sha256, write_json, fresh_descriptor
from train_decision_cql import equal


def row_key(sparse, numeric, progression, candidates, action):
    return (tuple(x for x in sparse if x != 510), tuple(numeric),
            tuple(x for x in progression if x != 2165),
            tuple(x for x in candidates if x != 546), int(action))


def observed_return_reference(path):
    """Independent scalar settlement accounting and backward sums on raw rows."""
    reference, episode = {}, []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        columns = line.split("\t")
        if len(columns) not in (8, 10, 12):
            raise ValueError("Unexpected raw RL row.")
        episode.append(columns)
        if len(columns) != 8:
            continue
        seat = int(episode[0][1].split(",")[6]) - 71
        rewards, previous = [0.] * len(episode), None
        for index, row in enumerate(episode):
            if int(row[1].split(",")[6]) - 71 != seat:
                raise ValueError("Observer changed inside a complete game.")
            if len(row) == 10:
                continue
            result = list(map(int, row[-1].split(",")))
            if previous is None:
                previous = result[6+seat] - result[seat]
            settled = result[10+seat] if len(row) == 8 else result[6+seat]
            rewards[index] = .2 * (settled-previous) / 10000
            previous = settled
            if len(row) == 8:
                scores = result[10:14]
                rank = sorted(range(4), key=lambda p: (-scores[p], p)).index(seat)
                rewards[index] += 1.5 - rank
        total = 0.
        for row, reward in zip(reversed(episode), reversed(rewards), strict=True):
            total += reward
            arrays = [list(map(int, value.split(","))) for value in row[1:5]]
            key = row_key(*arrays, int(row[5]))
            reference.setdefault(key, []).append(total)
        episode = []
    if episode:
        raise ValueError("Incomplete observer game in the raw MC reference.")
    return reference


def run(plan_path, manifest_path, output, independent=False):
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    expected = json.loads(manifest_path.read_text(encoding="utf-8-sig"))["complete_candidate"]
    source = Path(plan["source"])
    for name, checksum in expected.items():
        if sha256(source / name) != checksum:
            raise ValueError(f"Development source differs: {name}")
    for name in ("bc", "opponent", "template", "pilot_data"):
        if sha256(plan[name]["path"]) != plan[name]["sha256"]:
            raise ValueError(f"Retained {name} differs.")
    if plan.get("target_kind") != "mc":
        raise ValueError("MC verification requires an explicit MC plan.")
    prefix = output.with_name(output.stem.removesuffix("-acceptance"))
    artifact = lambda suffix: prefix.with_name(prefix.name + suffix)
    destination = artifact("-independent-acceptance.json") if independent else output
    if destination.exists() or not destination.parent.is_dir():
        raise ValueError("Verification output exists or its parent is missing.")
    os.environ.update(TORCHDYNAMO_DISABLE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1")
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from kanachan.model_loader import load_model, load_model_state
    from kanachan.nn.decision_decoder import DenseDecisionDecoder, GlobalActionHead
    from kanachan.nn.decision_value import q_only, transfer_bc_features
    from kanachan.training.cql import train as cql
    from kanachan.training.core.offline_rl import EpisodeReplayBuffer
    from kanachan.training.core.rl.rewards import configured_reward
    from kanachan.training.core.rl.discrete_cql import mc_components, scalar_cql_loss
    from kanachan.training.core.config.optimizer import AdamWOptimizerConfig, create
    from kanachan.training.online.state import cpu_state, capture_rng, export_model, atomic_save

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    model_path, checkpoint_path = output.with_suffix(".kanachan"), artifact("-checkpoint.pt")

    def status(phase):
        if time.monotonic() - started > 900:
            raise TimeoutError("The MC functional check exceeded 15 minutes.")
        write_json(artifact("-status.json"), dict(
            status="running", phase=phase, pid=os.getpid(),
            seconds=time.monotonic()-started, checked_utc=datetime.now(timezone.utc).isoformat()))

    def reader():
        return EpisodeReplayBuffer(
            training_data=Path(plan["pilot_data"]["path"]), contiguous_training_data=True,
            num_skip_samples=0, rewrite_rooms=None, rewrite_grades=None,
            get_reward=configured_reward("round_and_placement", score_unit=10000., round_weight=.2),
            discount_factor=1., dtype=torch.float32, max_size=128, batch_size=32,
            num_workers=0, pin_memory=False,
        )

    status("independent_raw_return_reference" if independent else "raw_return_reference")
    reference, actual = observed_return_reference(plan["pilot_data"]["path"]), {}
    batches = list(reader())
    for batch in batches:
        for i in range(len(batch)):
            key = row_key(*(batch[name][i].tolist() for name in ("sparse", "numeric", "progression", "candidates")), batch["action"][i])
            actual.setdefault(key, []).append(float(batch["mc_return"][i]))
    if actual.keys() != reference.keys() or sum(map(len, actual.values())) != 627:
        raise ValueError("MC labels do not cover the real input rows.")
    maximum_error = 0.
    for key in reference:
        if len(actual[key]) != len(reference[key]):
            raise ValueError("MC label multiplicities differ.")
        for a, b in zip(sorted(actual[key]), sorted(reference[key]), strict=True):
            maximum_error = max(maximum_error, abs(a-b))
    if maximum_error > 3e-6:
        raise ValueError("MC labels differ from independent raw settlement returns.")
    template = torch.load(plan["template"]["path"], map_location="cpu", weights_only=True)

    def initialize():
        random.seed(2026110503); np.random.seed(2026110503); torch.manual_seed(2026110503)
        actor = load_model_state(fresh_descriptor(template), map_location=torch.device("cpu")).float()
        teacher = load_model(plan["bc"]["path"], map_location=torch.device("cpu")).float()
        transfer_bc_features(next(m for m in actor.modules() if isinstance(m, DenseDecisionDecoder)),
                             next(m for m in actor.modules() if isinstance(m, GlobalActionHead)), teacher)
        actor.cuda().train().requires_grad_(True)
        if sum(p.numel() for p in actor.parameters()) != plan["parameters"]:
            raise ValueError("Actual MC model dimensions differ from the declared plan.")
        net = actor.select_subsequence(out_keys=["action_value"])
        optimizer, scheduler = create("cuda", AdamWOptimizerConfig(
            learning_rate=1e-4, warmup_steps=2, weight_decay=.01), net)
        return actor, net, optimizer, scheduler

    def measure(actor):
        with torch.no_grad():
            values, labels = [], []
            for batch in batches:
                data = batch.cuda()
                observation = data.select("sparse", "numeric", "progression", "candidates").clone(False)
                q_only(actor)(observation)
                values.append(observation["action_value"].gather(1, data["action"].long()[:, None]).flatten().cpu())
                labels.append(batch["mc_return"])
            prediction, label = torch.cat(values).double(), torch.cat(labels).double()
            residual = prediction-label
            return dict(q_mean=float(prediction.mean()), return_mean=float(label.mean()),
                        mse=float(residual.square().mean()), mae=float(residual.abs().mean()))

    if independent:
        receipt = json.loads(output.read_text(encoding="utf-8-sig"))
        if (receipt["status"] != "complete" or receipt["target_kind"] != "monte_carlo_return"
                or receipt["source_manifest_sha256"] != sha256(manifest_path)):
            raise ValueError("The MC functional receipt binds another recipe.")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        actor = load_model(model_path, map_location=torch.device("cpu")).float()
        equal(actor.state_dict(), state["model"])
        if (state["target"] is not None or state["target_kind"] != "mc"
                or sha256(model_path) != receipt["cuda_cql"]["model_sha256"]
                or sha256(checkpoint_path) != receipt["checkpoint_sha256"]):
            raise ValueError("MC checkpoint/export identity differs.")
        counters = sorted({int(v["step"]) for v in state["optimizer"]["state"].values()})
        if counters != [state["optimizer_steps"]] or len(state["optimizer"]["state"]) != len(list(actor.parameters())):
            raise ValueError("MC optimizer counters differ.")
        if (state["cursor"] != 627 or state["forward_calls"] != 20
                or state["scheduler"]["last_epoch"] != counters[0]
                or sum(p.numel() for p in actor.parameters()) != plan["parameters"]):
            raise ValueError("MC forward/cursor/scheduler/dimension counters differ.")
        teacher = load_model(plan["bc"]["path"], map_location=torch.device("cpu")).float()
        for get in (lambda m: next(x for x in m.modules() if isinstance(x, DenseDecisionDecoder)),
                    lambda m: next(x for x in m.modules() if isinstance(x, GlobalActionHead)).layers[0]):
            old, new = get(teacher).state_dict(), get(actor).state_dict()
            if not any(not torch.equal(value, old[key]) for key, value in new.items()):
                raise ValueError("MC feature parameters did not learn.")
        actor.cuda().eval().requires_grad_(False)
        before = copy.deepcopy(actor)
        next(m for m in before.modules() if isinstance(m, GlobalActionHead)).layers[-1].bias.add_(20.)
        initial = measure(before)
        actor.load_state_dict(state["offset_recovery"]["model"])
        final = measure(actor)
        for name, measured in (("initial", initial), ("final", final)):
            for key, value in measured.items():
                if abs(value-receipt["offset_recovery"][name][key]) > 2e-5:
                    raise ValueError("Independent offset-recovery measurement differs.")
        from kanachan.simulation.statistics import summarize
        equal(summarize(receipt["native_functional"]["results"]), receipt["native_functional"]["statistics"])
        result = dict(status="complete", scope="independent_mc_cql_functional_readback_only",
                      source_files_verified=len(expected), real_decisions=627,
                      maximum_independent_return_error=maximum_error, target_network=None,
                      actual_optimizer_steps=counters, offset_recovery_remeasured=True,
                      new_training_started=False, native_statistics_recomputed=True,
                      model_sha256=sha256(model_path))
    else:
        class Reached(Exception): pass
        class Writer:
            def add_scalar(self, *args): pass

        def optimize(saved=None, interrupt=False):
            actor, net, optimizer, scheduler = initialize()
            if saved is not None:
                net.load_state_dict(saved["model"])
                optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
                scheduler.load_state_dict(saved["scheduler"])
            result, precisions, calls = {}, set(), [0]
            def record_projection(module, inputs, value):
                precisions.add(str(value.dtype))
                calls[0] += 1
            hook = next(m for m in actor.modules() if isinstance(m, DenseDecisionDecoder)).blocks[0].qkv.register_forward_hook(
                record_projection)
            def snapshot(index, **metadata):
                if index is not None and not interrupt: return
                result.update(model=cpu_state(net.state_dict()), target=None, target_kind="mc",
                              optimizer=cpu_state(optimizer.state_dict()), scheduler=scheduler.state_dict(),
                              scaler=metadata["grad_scaler_state"], rng=capture_rng(), replay=metadata["replay_state"],
                              cursor=index if index is not None else metadata["resume_num_samples"],
                              forward_calls=calls[0] + (0 if saved is None else saved["forward_calls"]),
                              optimizer_steps=metadata["optimizer_steps"])
                if index is not None: raise Reached()
            try:
                with patch.object(cql, "td_components", side_effect=AssertionError("Bootstrap called")), patch(
                        "kanachan.training.core.rl.rewards.load_model", side_effect=AssertionError("Reward predictor called")):
                    cql._training(
                        training_data=Path(plan["pilot_data"]["path"]), contiguous_training_data=True,
                        rewrite_rooms=None, rewrite_grades=None, num_workers=0, replay_buffer_size=128,
                        device=torch.device("cuda:0"), dtype=torch.float32, amp_dtype=torch.float16,
                        source_network=net, target_network=None, reward_plugin=None, discount_factor=1.,
                        td_error_computation=None, target_kind="mc", regression_loss="mse",
                        reward_scheme="round_and_placement", round_score_unit=10000., round_reward_weight=.2,
                        alpha=.5, batch_size=32, gradient_accumulation_steps=1, max_gradient_norm=1.,
                        optimizer=optimizer, scheduler=scheduler, target_update_interval=1, target_update_rate=.005,
                        snapshot_interval=64, num_samples=0 if saved is None else saved["cursor"],
                        summary_writer=Writer(), snapshot_writer=snapshot,
                        initial_optimizer_steps=0 if saved is None else saved["optimizer_steps"],
                        initial_grad_scaler_state=None if saved is None else saved["scaler"],
                        replay_state=None if saved is None else saved["replay"],
                        initial_rng_state=capture_rng() if saved is None else saved["rng"], q_monitor_interval=1)
            except Reached:
                if not interrupt: raise
            finally: hook.remove()
            if precisions != {"torch.float16"}: raise ValueError("Actual CUDA AMP precision differs.")
            return actor, result

        status("cuda_exact_resume")
        actor, continuous = optimize()
        _, partial = optimize(interrupt=True)
        serialized = io.BytesIO(); torch.save(partial, serialized); serialized.seek(0)
        _, resumed = optimize(torch.load(serialized, map_location="cpu", weights_only=True))
        equal(continuous, resumed)
        if continuous["cursor"] != 627 or continuous["optimizer_steps"] <= 0:
            raise ValueError("Actual MC training counters differ.")
        atomic_save(model_path, export_model(template, actor))
        status("real_offset_recovery")
        corrected = copy.deepcopy(actor).train().requires_grad_(True)
        next(m for m in corrected.modules() if isinstance(m, GlobalActionHead)).layers[-1].bias.data.add_(20.)
        before = measure(corrected)
        optimizer, _ = create("cuda", AdamWOptimizerConfig(learning_rate=1e-4, warmup_steps=0, weight_decay=.01), corrected)
        first_batch = batches[0].cuda()
        _, _, target_before = mc_components(q_only(corrected), first_batch)
        for step in range(300):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = scalar_cql_loss(q_only(corrected), None, batches[step % len(batches)].cuda(), .5, target_kind="mc")
            loss.backward(); torch.nn.utils.clip_grad_norm_(corrected.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            if step % 50 == 0: status("real_offset_recovery")
        after = measure(corrected)
        _, _, target_after = mc_components(q_only(corrected), first_batch)
        equal(target_before, target_after)
        if not after["mse"] < before["mse"]: raise ValueError("Real common-offset error was not reduced.")
        continuous["offset_recovery"] = dict(model=cpu_state(corrected.state_dict()), updates=300)
        atomic_save(checkpoint_path, continuous)
        status("native_functional")
        from kanachan.simulation import simulate
        from kanachan.simulation.statistics import summarize
        opponent = load_model(plan["opponent"]["path"], map_location=torch.device("cpu")).float().cuda().eval().requires_grad_(False)
        actor.eval().requires_grad_(False)
        with torch.inference_mode():
            games = simulate(torch.device("cuda:0"), torch.float32, 3, 12, opponent, 12, actor, 4, 1, 1, 1, lambda: None, seed=2026110503)
        results = [game.get_result() for game in games]
        if len(results) != 4: raise ValueError("Native four-seat coverage differs.")
        result = dict(status="complete", scope="mc_cql_cuda_and_native_functional_only",
                      source=str(source), manifest=str(manifest_path), source_manifest_sha256=sha256(manifest_path),
                      helper_sha256=sha256(__file__), target_kind="monte_carlo_return", feature_freezing=False,
                      logged_return_mixing=False, learned_reward_predictor=False, source_files_verified=len(expected),
                      maximum_independent_return_error=maximum_error, bootstrap_calls=0,
                      cuda_cql=dict(samples=627, optimizer_steps=continuous["optimizer_steps"], exact_resume_all_state=True,
                                    model_sha256=sha256(model_path), trainable_parameters=sum(p.numel() for p in actor.parameters()),
                                    target_network=None), checkpoint_sha256=sha256(checkpoint_path),
                      offset_recovery=dict(initial=before, final=after, updates=300, targets_bitwise_unchanged=True),
                      native_functional=dict(games=4, results=results, statistics=summarize(results)),
                      limitation="627-row functional and training-set calibration check; playing-strength improvement not established.")
    result.update(plan=str(plan_path), plan_sha256=sha256(plan_path),
                  manifest=str(manifest_path), source_manifest_sha256=sha256(manifest_path),
                  input_sha256={name: plan[name]["sha256"] for name in ("bc", "opponent", "template", "pilot_data")})
    for name, checksum in expected.items():
        if sha256(source / name) != checksum: raise ValueError("Source changed during MC verification.")
    for name in ("bc", "opponent", "template", "pilot_data"):
        if sha256(plan[name]["path"]) != plan[name]["sha256"]: raise ValueError("Retained input changed.")
    result["seconds"] = time.monotonic()-started
    write_json(destination, result)
    write_json(artifact("-status.json"), dict(status="complete", phase="independent_complete" if independent else "complete", pid=os.getpid()))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--independent", action="store_true")
    args = parser.parse_args()
    run(args.plan, args.manifest, args.output, args.independent)
