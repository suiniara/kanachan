"""Update the existing development source and verify it without new directories.

Production sources and retained models are never written. The existing v1
source archive is the rollback reference; all training checks stay in memory.
"""

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import types
import zipfile


ROOT = Path(r"C:\Users\Suini\kanachan-data-work")
BASE = ROOT / "diagnostics/decision-decoder-20261104-v1"
OUTPUT = BASE
SOURCE = OUTPUT / "source"
ARCHIVE = ROOT / "tools/decision-optimization-delta-v2.zip"
V1_HELPER_SHA = "9e7dcff1e25c8a1fa953babc3700551f8d1303bb09144a94545c75d108a852b9"
NATIVE_SHA = "c74c250bcb127ef413dad749dc2f143edb009ef9a590d86ce9294b11d303f072"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_python(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def helper():
    path = ROOT / "tools/decision-decoder-verify-v1.py"
    if digest(path) != V1_HELPER_SHA:
        raise ValueError("The read-only v1 verification helper differs.")
    module = load_python("optimization_v1_helpers", path)
    module.OUTPUT, module.SOURCE = OUTPUT, SOURCE
    original_write = module.write
    module.write = lambda name, value: original_write(f"optimization-{name}", value)
    return module


def compare(left, right, path="state"):
    import torch
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0, msg=path)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f"Different keys: {path}")
        for key in left:
            compare(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError(f"Different sequence: {path}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            compare(a, b, f"{path}.{index}")
    elif left != right:
        raise ValueError(f"Different scalar: {path}")


def prepare(proof, archive_sha, continue_after_probe_fix=False):
    if not SOURCE.is_dir():
        raise ValueError("The existing development source is missing.")
    if (OUTPUT / "optimization-status.json").exists():
        if not continue_after_probe_fix:
            raise ValueError("Optimization status exists: inspect before another execution.")
        status = proof.read(OUTPUT / "optimization-status.json")
        previous_error = proof.read(OUTPUT / "optimization-error.json")
        receipt = proof.read(OUTPUT / "optimization-source-manifest.json")
        if (
            status.get("status") != "failed"
            or previous_error.get("error") != "legacy_cuda_next_update.module.0.module.numeric_projection"
            or receipt["archive_sha256"] != archive_sha
        ):
            raise ValueError("Probe-fix continuation does not match the inspected failure.")
        for name, identity in receipt["complete_candidate"].items():
            path = (SOURCE / name).resolve()
            if not path.is_relative_to(SOURCE.resolve()) or digest(path) != identity:
                raise ValueError(f"Already updated source differs: {name}")
        receipt["previous_probe_issue"] = previous_error
        proof.write("source-manifest", receipt)
        return receipt["complete_candidate"]
    if digest(ARCHIVE) != archive_sha:
        raise ValueError("Candidate archive hash differs.")
    original_archive = ROOT / "tools/decision-decoder-candidate-v1.zip"
    if digest(original_archive) != "ca20d5ac0f1499ed7c1620ec8525201efa5616aafd5da40b76d5b1dd705247ad":
        raise ValueError("The existing rollback archive differs.")
    with zipfile.ZipFile(original_archive) as archive:
        baseline_manifest = json.loads(archive.read("candidate-manifest.json"))["files"]
    for name, identity in baseline_manifest.items():
        if digest(BASE / "source" / name) != identity:
            raise ValueError(f"Validated v1 source differs: {name}")
    if digest(BASE / "source/kanachan/simulation/_simulation.pyd") != NATIVE_SHA:
        raise ValueError("Previously verified native extension differs.")
    with zipfile.ZipFile(ARCHIVE) as archive:
        candidate = proof.read_archive_manifest(archive)
        complete = json.loads(archive.read("candidate-manifest.json"))["expected_full_candidate"]
        # Check every member before changing the first file. New modules use
        # existing package directories, and all writes are atomic replacements.
        for name, identity in candidate.items():
            path = (SOURCE / name).resolve()
            if not path.is_relative_to(SOURCE.resolve()) or not path.parent.is_dir():
                raise ValueError("Candidate path escaped its own source.")
            if name not in baseline_manifest and path.exists():
                raise ValueError(f"A new candidate path already exists: {name}")
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != identity:
                raise ValueError(f"Candidate bytes differ: {name}")
        proof.phase("update_existing_development_source", changed_files=list(candidate))
        for name in candidate:
            path = SOURCE / name
            temporary = path.with_name(path.name + ".optimization.tmp")
            temporary.write_bytes(archive.read(name))
            os.replace(temporary, path)
    for name, identity in complete.items():
        path = (SOURCE / name).resolve()
        if not path.is_relative_to(SOURCE.resolve()) or digest(path) != identity:
            raise ValueError(f"Complete candidate identity differs: {name}")
    proof.write("source-manifest", {
        "archive_sha256": archive_sha, "candidate": candidate,
        "complete_candidate": complete,
        "native_extension_sha256": NATIVE_SHA,
    })
    data_receipt = proof.read(BASE / "pilot-data.json")
    for name, key in (("train.bc.tsv", "bc_sha256"), ("train.rl.tsv", "rl_sha256")):
        if digest(OUTPUT / "data" / name) != data_receipt[key]:
            raise ValueError("Existing real pilot data identity differs.")
    return complete


def semantic_and_speed_checks(proof):
    torch = proof.setup_runtime()
    from tensordict.nn import TensorDictModule, TensorDictSequential
    from kanachan.model_loader import load_model
    from kanachan.nn.decode_converter import DecodeConverter
    from kanachan.nn.decision_decoder import DenseDecisionDecoder, GlobalActionHead
    from kanachan.nn.decision_value import q_only
    from kanachan.training.core.bc import DataLoader
    from kanachan.training.core.offline_rl import DataLoader as RLDataLoader
    from kanachan.training.core.rl.rewards import PlacementReward
    from kanachan.training.core.rl.discrete_cql import scalar_cql_loss
    from kanachan.training.core.optimizers import online_adamw, LEGACY_ADAMW_POLICY
    # The previous implementation is loaded in memory from its existing
    # archive, without creating a second source tree or restoring old files.
    old = types.ModuleType("immutable_decoder_v1")
    sys.modules[old.__name__] = old
    with zipfile.ZipFile(ROOT / "tools/decision-decoder-candidate-v1.zip") as archive:
        exec(compile(archive.read("kanachan/nn/decision_decoder.py"),
                     "archived_decision_decoder_v1.py", "exec"), old.__dict__)
    current = load_model(BASE / "cql/snapshots/model.kanachan", map_location=torch.device("cpu")).cuda()
    body = next(m for m in current.modules() if isinstance(m, DenseDecisionDecoder))
    head = next(m for m in current.modules() if isinstance(m, GlobalActionHead))
    previous_body = old.DenseDecisionDecoder(
        dimension=128, num_heads=2, dim_feedforward=192, num_layers=2,
        device=torch.device("cuda"), dtype=torch.float32,
    )
    previous_head = old.GlobalActionHead(
        input_dimension=128, hidden_dimension=128,
        device=torch.device("cuda"), dtype=torch.float32,
    )
    previous_body.load_state_dict(body.state_dict())
    previous_head.load_state_dict(head.state_dict())
    previous = TensorDictSequential(
        TensorDictModule(previous_body, in_keys=["sparse", "numeric", "progression", "candidates"], out_keys=["encode"]),
        TensorDictModule(previous_head, in_keys=["candidates", "encode"], out_keys=["action_value"]),
        TensorDictModule(DecodeConverter("argmax", tie_break_by_action_id=True),
                         in_keys=["candidates", "action_value"], out_keys=["action"]),
    )
    optimized = q_only(current)
    data = next(DataLoader(
        path=proof.BASELINE / "data/validation.bc.tsv.gz", num_skip_samples=0,
        rewrite_rooms=None, rewrite_grades=None, batch_size=32, num_workers=0,
    )).cuda()
    precisions = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        outputs, gradients = [], []
        for model in (previous, optimized):
            model.train().requires_grad_(True).zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False):
                observation = data.clone(False)
                model(observation)
                value = observation["action_value"]
                (value * torch.linspace(-1, 1, value.numel(), device="cuda").reshape_as(value)).mean().backward()
            outputs.append(value.detach())
            gradients.append([p.grad.detach().clone() for p in model.parameters()])
        compare(outputs[0], outputs[1], "candidate_values")
        compare(gradients[0], gradients[1], "candidate_gradients")
        precisions.append({"dtype": str(dtype), "forward_and_gradients_bitwise_identical": True})
    for model in (previous, optimized):
        model.eval()
    results = {"forward": {}, "cql_forward_backward": {}}

    def timed_pair(functions, repeats):
        for function in functions:
            for _ in range(4):
                function()
        timings = [[], []]
        for iteration in range(repeats):
            for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                torch.cuda.synchronize()
                start = time.perf_counter()
                functions[index]()
                torch.cuda.synchronize()
                timings[index].append(1000 * (time.perf_counter() - start))
        return {name: {"median_ms": statistics.median(values), "repeats": repeats}
                for name, values in zip(("v1", "v2"), timings, strict=True)}

    for batch in (1, 32):
        sample = data[:batch]

        @torch.no_grad()
        def forward(model):
            with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
                model(sample.clone(False))

        results["forward"][str(batch)] = timed_pair(
            [lambda: forward(previous), lambda: forward(optimized)], 24
        )
    replay = next(RLDataLoader(
        path=OUTPUT / "data/train.rl.tsv", num_skip_samples=0,
        rewrite_rooms=None, rewrite_grades=None, get_reward=PlacementReward(),
        dtype=torch.float32, batch_size=1024, num_workers=0,
    ))
    selection = torch.cat((torch.arange(30), (replay["next", "done"] != 0).nonzero().flatten()[-2:]))
    replay = replay[selection].cuda()
    targets = [copy.deepcopy(model).eval().requires_grad_(False) for model in (previous, optimized)]

    def step(model, target, tensor_metrics):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            loss, _ = scalar_cql_loss(model, target, replay, 0.1, td_loss="huber", tensor_metrics=tensor_metrics)
        loss.backward()

    results["cql_forward_backward"] = timed_pair([
        lambda: step(previous, targets[0], False),
        lambda: step(optimized, targets[1], True),
    ], 12)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            optimized(data.clone(False))
    kernels = [event.key for event in trace.key_averages() if "scaled_dot_product" in event.key]
    # Restore an actual pre-optimization flat-group optimizer on CUDA and
    # perform the same next update, without editing the original checkpoint.
    legacy = torch.load(BASE / "online-entry-partial/latest.pt", weights_only=True, map_location="cpu")
    legacy_model = load_model(BASE / "online-entry-partial/model.kanachan", map_location=torch.device("cpu")).cuda()
    reference_model = copy.deepcopy(legacy_model)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters())
    # Each reference needs independent CPU step tensors. Native load_state_dict
    # retains those tensor objects, so loading the same dict twice aliases the
    # counters and makes the second optimizer advance an extra step.
    reference_optimizer.load_state_dict(copy.deepcopy(legacy["optimizer"]))
    restored_optimizer, policy = online_adamw(
        legacy_model, lr=1e-4, weight_decay=0.01, device_type="cuda", resume_state=copy.deepcopy(legacy),
    )
    if policy != LEGACY_ADAMW_POLICY:
        raise ValueError("A v1 checkpoint was silently regrouped.")
    for left, right in zip(reference_model.parameters(), legacy_model.parameters(), strict=True):
        left.grad = torch.full_like(left, 0.001)
        right.grad = left.grad.clone()
    reference_optimizer.step()
    restored_optimizer.step()
    compare(reference_model.state_dict(), legacy_model.state_dict(), "legacy_cuda_next_update")
    compare(reference_optimizer.state_dict(), restored_optimizer.state_dict(), "legacy_cuda_optimizer")
    proof.write("performance", {
        "status": "complete", "precision": precisions,
        "legacy_flat_adamw_cuda_update_identical": True,
        "benchmark": results, "sdpa_events": kernels,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
        "backend": "math_only", "amp_dtype": "float16", "alternating_pair_order": True,
        "benchmark_scope": "fixed_weights_real_observation_microbenchmarks_only",
        "strength_improvement_established": False,
    })


def online_checks(proof):
    import torch
    from kanachan.model_loader import load_model
    from kanachan.nn import EpsilonGreedy
    from kanachan.nn.decision_value import q_only
    from kanachan.training.common import backoff_gradient_scale
    from kanachan.training.core.optimizers import ADAMW_POLICY, online_adamw
    from kanachan.training.core.offline_rl import DataLoader
    from kanachan.training.core.rl.simulation import simulate as collect
    from kanachan.training.core.rl.multistep import make_multistep
    from kanachan.training.core.rl.rewards import PlacementReward
    from kanachan.training.core.rl.discrete_cql import scalar_cql_loss
    from kanachan.training.online.train import SamplingPolicy
    from kanachan.training.online.offline_pool import training_columns, POOL_POLICY
    from kanachan.training.online.reservoir import TransitionReservoir
    from kanachan.training.online.replay import ReplayBuffer
    from kanachan.training.online.state import cpu_state, capture_rng, restore_rng
    model_path = OUTPUT / "cql/snapshots/model.kanachan"

    def load(path):
        return load_model(path, map_location=torch.device("cpu")).float().cuda()

    torch.manual_seed(2026110402)
    torch.cuda.manual_seed_all(2026110402)
    baseline = load(proof.BASELINE / "model.kanachan").eval().requires_grad_(False)
    actor = load(model_path).eval().requires_grad_(False)
    exploration = EpsilonGreedy(
        epsilon_first=0.05, epsilon_last=0.05, annealing_steps=0,
        device=torch.device("cuda"), dtype=torch.float32,
    )
    exploration.stable_action_ids = True
    exploration.train()
    episodes = collect(
        device=torch.device("cuda"), dtype=torch.float32, room=3, dongfengzhan=False,
        grade=12, baseline_model=baseline, policy_model=SamplingPolicy(actor, exploration),
        num_simulations=1, simulation_batch_size=1, simulation_concurrency=1,
        progress=lambda: None, get_reward=PlacementReward(), seed=2026110402,
        policy_version=0, opponent_version=0,
    )
    if not len(episodes) or not episodes["next", "done"].any():
        raise ValueError("Native collection did not finish a real half-game.")
    make_multistep(episodes, 1.0, 1)
    reservoir = TransitionReservoir(128, 2026110403 ^ 0x5A17, 2026110403 ^ 0x05C01)
    for batch in DataLoader(
        path=OUTPUT / "data/train.rl.tsv", num_skip_samples=0,
        rewrite_rooms=None, rewrite_grades=None, get_reward=PlacementReward(),
        dtype=torch.float32, batch_size=4096, num_workers=0,
    ):
        batch["next", "discount"] = torch.ones(len(batch))
        reservoir.consume(training_columns(batch))
    reservoir.replay.metadata = {
        "policy": POOL_POLICY, "source_rows": reservoir.seen,
        "source_indices": reservoir.indices.clone(),
    }
    if reservoir.seen != 627 or reservoir.indices.max() <= 128:
        raise ValueError("The real reservoir did not cover the entire pilot source.")

    def learner(state=None):
        net = load(model_path).train().requires_grad_(True)
        target = copy.deepcopy(net).eval().requires_grad_(False)
        if state is not None:
            net.load_state_dict(state["model"])
            target.load_state_dict(state["target"])
        optim, policy = online_adamw(
            net, lr=1e-4, weight_decay=0.01, device_type="cuda", resume_state=state,
        )
        scaler = torch.amp.GradScaler("cuda")
        online, offline = ReplayBuffer(256, 2026110403), ReplayBuffer(128, 0)
        if state is None:
            online.append(training_columns(episodes))
            offline.load_state_dict(reservoir.replay.state_dict())
            updates = 0
        else:
            scaler.load_state_dict(state["scaler"])
            online.load_state_dict(state["replay"])
            offline.load_state_dict(state["offline_replay"])
            updates = state["updates"]
            restore_rng(state["rng"])
        return {"net": net, "target": target, "optim": optim, "scaler": scaler,
                "online": online, "offline": offline, "updates": updates, "policy": policy}

    def steps(state, count):
        source, target = q_only(state["net"]), q_only(state["target"])
        metrics = []
        for _ in range(count):
            batch = torch.cat((state["offline"].sample(16), state["online"].sample(16)), 0).cuda()
            alpha = torch.tensor([0.1] * 16 + [0.05] * 16, device="cuda")
            optim, scaler = state["optim"], state["scaler"]
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
                loss, record = scalar_cql_loss(source, target, batch, alpha,
                                               td_loss="huber", tensor_metrics=True)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite CUDA CQL loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            norm = torch.nn.utils.clip_grad_norm_(state["net"].parameters(), 1.0)
            if not torch.isfinite(norm):
                backoff_gradient_scale(scaler)
                continue
            scaler.step(optim)
            scaler.update()
            state["updates"] += 1
            with torch.no_grad():
                for a, b in zip(state["net"].parameters(), state["target"].parameters(), strict=True):
                    b.lerp_(a, 0.005)
                for a, b in zip(state["net"].buffers(), state["target"].buffers(), strict=True):
                    b.copy_(a)
            metrics.append({"loss": float(loss.detach()), "gradient_norm": float(norm),
                            **{key: float(value) for key, value in record.items()}})
        return metrics

    def snapshot(state):
        return cpu_state({
            "model": state["net"].state_dict(), "target": state["target"].state_dict(),
            "optimizer": state["optim"].state_dict(), "optimizer_policy": state["policy"],
            "scaler": state["scaler"].state_dict(), "replay": state["online"].state_dict(),
            "offline_replay": state["offline"].state_dict(), "rng": capture_rng(),
            "updates": state["updates"],
        })

    initial_rng = capture_rng()
    full = learner()
    restore_rng(initial_rng)
    full_metrics = steps(full, 4)
    expected = snapshot(full)
    del full
    partial = learner()
    restore_rng(initial_rng)
    steps(partial, 2)
    saved = io.BytesIO()
    torch.save(snapshot(partial), saved)
    saved.seek(0)
    del partial
    restored = learner(torch.load(saved, map_location="cpu", weights_only=True))
    steps(restored, 2)
    actual = snapshot(restored)
    compare(expected, actual, "in_memory_cuda_resume")
    if expected["updates"] == 0 or expected["optimizer_policy"] != ADAMW_POLICY:
        raise ValueError("No successful grouped AdamW CUDA updates.")
    proof.write("training-checks", {
        "status": "complete", "scope": "in_memory_cuda_mixed_cql_update_and_resume",
        "attempted_updates_per_run": 4, "successful_updates_per_run": expected["updates"],
        "optimizer_policy": ADAMW_POLICY, "offline_batch_size": 16, "online_batch_size": 16,
        "all_resume_components_bitwise_identical": True,
        "identical_resume_components": list(expected), "metrics": full_metrics,
        "persisted_model_or_replay_files": False,
    })
    proof.write("online-checks", {
        "status": "complete", "complete_native_collection_runs": 1,
        "collected_decisions": len(episodes),
        "pool": {"policy": POOL_POLICY, "source_rows": reservoir.seen, "retained": 128,
                 "lowest_source_index": int(reservoir.indices.min()),
                 "highest_source_index": int(reservoir.indices.max())},
        "scope": "real_native_collection_and_full_stream_reservoir_only",
    })


def precision_checks(device_sha, test_sha):
    """Actual offline training loops and exact AMP recovery, entirely in RAM."""
    proof = helper()
    status_path = OUTPUT / "optimization-precision-status.json"
    if status_path.exists():
        raise ValueError("Precision check already has status; inspect before another execution.")
    expected = dict(proof.read(OUTPUT / "optimization-source-manifest.json")["complete_candidate"])
    expected.update({
        "kanachan/training/core/config/device.py": device_sha,
        "test/training/test_device_precision.py": test_sha,
    })
    for name, identity in expected.items():
        path = (SOURCE / name).resolve()
        if not path.is_relative_to(SOURCE.resolve()) or digest(path) != identity:
            raise ValueError(f"Precision candidate identity differs: {name}")
    model_paths = {
        algorithm: OUTPUT / algorithm / "snapshots/model.kanachan"
        for algorithm in ("bc", "cql")
    }
    model_hashes = {algorithm: digest(path) for algorithm, path in model_paths.items()}
    proof.write("precision-source-manifest", {
        "complete_candidate": expected,
        "parent_manifest_sha256": digest(OUTPUT / "optimization-source-manifest.json"),
        "retained_model_hashes": model_hashes,
    })
    torch = proof.setup_runtime()
    from unittest.mock import Mock
    from omegaconf import OmegaConf
    from kanachan.model_loader import load_model
    from kanachan.nn.decision_decoder import DenseDecisionDecoder
    from kanachan.nn.decision_value import q_only
    from kanachan.training.core.config.device import validate
    from kanachan.training.core.config.optimizer import create, AdamWOptimizerConfig
    from kanachan.training.core.rl.discrete_cql import td_components, td_penalty
    from kanachan.training.bc.train import _training as train_bc
    from kanachan.training.cql.train import _training as train_cql
    from kanachan.training.online.state import cpu_state, capture_rng

    class CheckpointReached(Exception):
        pass

    cases = []
    for amp_name in ("float16", "bfloat16"):
        config = OmegaConf.create({
            "device": {"type": "cuda", "dtype": "float32", "amp_dtype": amp_name},
            "encoder": {"architecture": "dense_decoder"},
        })
        *_, device, dtype, amp = validate(config)
        for algorithm in ("bc", "cql"):
            proof.write("precision-status", {
                "status": "running", "phase": f"{algorithm}_{amp_name}", "pid": os.getpid(),
            })
            initial_rng = capture_rng()

            def run(saved=None, stop_at=None):
                actor = load_model(model_paths[algorithm], map_location=torch.device("cpu")).float().cuda()
                output_key = "log_probs" if algorithm == "bc" else "action_value"
                network = actor.select_subsequence(out_keys=[output_key]).train().requires_grad_(True)
                target = None if algorithm == "bc" else copy.deepcopy(network).eval().requires_grad_(False)
                optim, _ = create("cuda", AdamWOptimizerConfig(), network)
                if saved is not None:
                    network.load_state_dict(saved["model"])
                    if target is not None:
                        target.load_state_dict(saved["target"])
                    optim.load_state_dict(saved["optimizer"])
                actual_dtypes = set()
                body = next(module for module in network.modules() if isinstance(module, DenseDecisionDecoder))
                handle = body.blocks[0].qkv.register_forward_hook(
                    lambda module, inputs, output: actual_dtypes.add(str(output.dtype))
                )
                snapshots = {}

                def snapshot(index, **metadata):
                    if index is not None and index != stop_at:
                        return
                    value = cpu_state({
                        "model": network.state_dict(),
                        "target": None if target is None else target.state_dict(),
                        "optimizer": optim.state_dict(), "rng": capture_rng(),
                        "cursor": index if index is not None else metadata["resume_num_samples"],
                        "optimizer_steps": metadata["optimizer_steps"],
                        "scaler": metadata["grad_scaler_state"],
                        "replay": metadata.get("replay_state"),
                    })
                    snapshots[index] = value
                    if index is not None:
                        raise CheckpointReached()

                common = dict(
                    training_data=OUTPUT / "data" / f"train.{algorithm if algorithm == 'bc' else 'rl'}.tsv",
                    rewrite_rooms=None, rewrite_grades=None, num_workers=0,
                    device=device, dtype=dtype, amp_dtype=amp,
                    num_samples=0 if saved is None else saved["cursor"],
                    batch_size=32, gradient_accumulation_steps=1, max_gradient_norm=1.0,
                    optimizer=optim, scheduler=None, snapshot_interval=64,
                    summary_writer=Mock(), snapshot_writer=snapshot,
                    initial_optimizer_steps=0 if saved is None else saved["optimizer_steps"],
                    initial_grad_scaler_state=None if saved is None else saved["scaler"],
                    initial_rng_state=initial_rng if saved is None else saved["rng"],
                )
                try:
                    if algorithm == "bc":
                        train_bc(network_tdm=network, **common)
                    else:
                        source_values, target_values = q_only(network), q_only(target)

                        def td_error(data):
                            values, chosen, reward = td_components(source_values, target_values, data)
                            data["action_value"] = values
                            data["td_error"] = 2 * td_penalty(chosen, reward, "huber")

                        train_cql(
                            source_network=network, target_network=target,
                            td_error_computation=td_error, contiguous_training_data=True,
                            replay_buffer_size=128, reward_plugin=None, discount_factor=1.0,
                            alpha=0.1, cql_temperature=1.0, target_update_interval=1,
                            target_update_rate=0.005,
                            replay_state=None if saved is None else saved["replay"], **common,
                        )
                except CheckpointReached:
                    if stop_at is None:
                        raise
                finally:
                    handle.remove()
                if actual_dtypes != {str(amp)}:
                    raise ValueError(f"The actual attention projection did not use {amp}: {actual_dtypes}")
                result = snapshots[stop_at]
                if result["optimizer_steps"] <= 0:
                    raise ValueError("No successful actual AMP updates.")
                return result

            full = run()
            checkpoint = run(stop_at=64)
            stream = io.BytesIO()
            torch.save(checkpoint, stream)
            stream.seek(0)
            restored = run(torch.load(stream, map_location="cpu", weights_only=True))
            compare(full, restored, f"{algorithm}_{amp_name}_resume")
            cases.append({
                "algorithm": algorithm, "master_dtype": str(dtype), "amp_dtype": str(amp),
                "samples": full["cursor"], "successful_updates": full["optimizer_steps"],
                "checkpoint_cursor": 64, "actual_attention_projection_dtype_verified": True,
                "all_resume_components_bitwise_identical": True,
                "components": list(full), "grad_scaler_state": full["scaler"],
            })
    for name, identity in expected.items():
        if digest(SOURCE / name) != identity:
            raise ValueError(f"Precision source changed during training: {name}")
    for algorithm, path in model_paths.items():
        if digest(path) != model_hashes[algorithm]:
            raise ValueError("A retained pilot model changed.")
    if digest(proof.BASELINE / "model.kanachan") != proof.BC_SHA:
        raise ValueError("The preserved BC model changed.")
    proof.write("precision-acceptance", {
        "status": "complete", "scope": "offline_amp_precision_functional_and_exact_resume_only",
        "source": str(SOURCE), "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
        "cases": cases, "all_source_files_verified": len(expected),
        "existing_development_directory_updated_in_place": True,
        "new_experiment_directories_created": False, "persisted_model_or_replay_files": False,
        "retained_models_unchanged": True, "formal_training_started": False,
    })
    proof.write("precision-status", {"status": "complete", "phase": "complete", "pid": os.getpid()})
    print(json.dumps({"status": "complete", "precision_cases": cases}), flush=True)


def throughput_backup(manifest_sha):
    proof = helper()
    path = ROOT / "tools/decision-throughput-source-manifest.json"
    if digest(path) != manifest_sha:
        raise ValueError("Throughput manifest differs before backup.")
    manifest = proof.read(path)
    previous = proof.read(OUTPUT / manifest["parent_manifest"])["complete_candidate"]
    for name, checksum in previous.items():
        if digest(SOURCE / name) != checksum:
            raise ValueError(f"Previous development source differs: {name}")
    for name, checksum in manifest["previous_changed_files"].items():
        if previous.get(name) != checksum:
            raise ValueError(f"Conflicting previous identity: {name}")
    archive_path = OUTPUT / "throughput-before.zip"
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in manifest["previous_changed_files"]:
            archive.write(SOURCE / name, name)
    print(json.dumps({"status": "backup_complete", "files": len(previous),
                      "sha256": digest(archive_path)}), flush=True)


def throughput_checks(manifest_sha, continue_after_probe_fix=False):
    """Compare real BC/CQL loops and cache/resume paths, using existing files."""
    proof = helper()
    status = OUTPUT / "optimization-throughput-status.json"
    previous_issue = None
    if status.exists():
        failure = proof.read(OUTPUT / "optimization-throughput-error.json")
        if not (continue_after_probe_fix and proof.read(status)["status"] == "failed"
                and 'line 139, in _training' in failure["traceback"]
                and failure["traceback"].rstrip().endswith("AssertionError")):
            raise ValueError("Throughput verification already has status; inspect before rerunning.")
        previous_issue = {"status": proof.read(status), "failure": failure,
                          "correction": "Probe uses indexed CUDA device cuda:0 for existing tensor-device assertions."}
    manifest_path = ROOT / "tools/decision-throughput-source-manifest.json"
    if digest(manifest_path) != manifest_sha:
        raise ValueError("Throughput manifest identity differs.")
    manifest = proof.read(manifest_path)
    expected = manifest["complete_candidate"]

    def verify_sources():
        for name, checksum in expected.items():
            if digest(SOURCE / name) != checksum:
                raise ValueError(f"Throughput source identity differs: {name}")

    verify_sources()
    baseline = OUTPUT / "throughput-before.zip"
    with zipfile.ZipFile(baseline) as archive:
        before = {name: archive.read(name) for name in manifest["previous_changed_files"]}
    for name, content in before.items():
        if hashlib.sha256(content).hexdigest() != manifest["previous_changed_files"][name]:
            raise ValueError(f"Previous source identity differs: {name}")
    proof.write("throughput-source-manifest", manifest | {"baseline_archive_sha256": digest(baseline)})
    torch = proof.setup_runtime()
    from unittest.mock import Mock
    from kanachan.model_loader import load_model
    from kanachan.nn.decision_value import q_only
    from kanachan.training.core.config.optimizer import create, AdamWOptimizerConfig
    from kanachan.training.core.rl.discrete_cql import td_components, td_penalty
    from kanachan.training.online.state import capture_rng, cpu_state
    from kanachan.training.bc.train import _training as train_bc
    from kanachan.training.cql.train import _training as train_cql

    previous = {}
    for algorithm in ("bc", "cql"):
        name = f"kanachan.training.{algorithm}.throughput_previous"
        module = types.ModuleType(name)
        module.__package__ = f"kanachan.training.{algorithm}"
        sys.modules[name] = module
        exec(compile(before[f"kanachan/training/{algorithm}/train.py"], name, "exec"), module.__dict__)
        previous[algorithm] = module._training
    model_paths = {key: OUTPUT / key / "snapshots/model.kanachan" for key in previous}
    model_hashes = {key: digest(path) for key, path in model_paths.items()}
    initial_rng = capture_rng()
    cache_path = OUTPUT / "rl-cache.sqlite3"
    if cache_path.exists():
        raise ValueError("The probe cache already exists; inspect its ownership first.")

    class CheckpointReached(Exception):
        pass

    def run(algorithm, amp, *, old=False, saved=None, stop=None, cached=False, monitor=0):
        actor = load_model(model_paths[algorithm], map_location=torch.device("cpu")).float().cuda()
        key = "log_probs" if algorithm == "bc" else "action_value"
        network = actor.select_subsequence(out_keys=[key]).train().requires_grad_(True)
        target = None if algorithm == "bc" else copy.deepcopy(network).eval().requires_grad_(False)
        optimizer, _ = create("cuda", AdamWOptimizerConfig(), network)
        if saved:
            network.load_state_dict(saved["model"])
            if target is not None:
                target.load_state_dict(saved["target"])
            optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        result = {}

        def snapshot(index, **metadata):
            if index is not None and index != stop:
                return
            result.update(cpu_state({
                "model": network.state_dict(),
                "target": None if target is None else target.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": capture_rng(),
                "cursor": index if index is not None else metadata["resume_num_samples"],
                "optimizer_steps": metadata["optimizer_steps"],
                "scaler": metadata["grad_scaler_state"],
                "replay": metadata.get("replay_state"),
            }))
            if index is not None:
                raise CheckpointReached()

        arguments = dict(
            training_data=OUTPUT / "data" / ("train.bc.tsv" if algorithm == "bc" else "train.rl.tsv"),
            rewrite_rooms=None, rewrite_grades=None, num_workers=0,
            device=torch.device("cuda:0"), dtype=torch.float32, amp_dtype=amp,
            num_samples=0 if saved is None else saved["cursor"],
            batch_size=32, gradient_accumulation_steps=1, max_gradient_norm=1.,
            optimizer=optimizer, scheduler=None, snapshot_interval=64,
            summary_writer=Mock(), snapshot_writer=snapshot,
            initial_optimizer_steps=0 if saved is None else saved["optimizer_steps"],
            initial_grad_scaler_state=None if saved is None else saved["scaler"],
            initial_rng_state=initial_rng if saved is None else saved["rng"],
        )
        function = previous[algorithm] if old else (train_bc if algorithm == "bc" else train_cql)
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            if algorithm == "bc":
                function(network_tdm=network, **arguments)
            else:
                source_values, target_values = q_only(network), q_only(target)

                def td(data):
                    values, chosen, expected_return = td_components(source_values, target_values, data)
                    data["action_value"] = values
                    data["td_error"] = 2 * td_penalty(chosen, expected_return, "huber")
                    data["td_target"] = expected_return

                extra = {} if old else dict(
                    q_monitor_interval=monitor, data_cache=cache_path if cached else None,
                    data_cache_bytes=4 * 1024 * 1024,
                )
                function(
                    source_network=network, target_network=target, td_error_computation=td,
                    contiguous_training_data=True, replay_buffer_size=128, reward_plugin=None,
                    discount_factor=1., alpha=.1, cql_temperature=1.,
                    target_update_interval=1, target_update_rate=.005,
                    replay_state=None if saved is None else saved["replay"], **extra, **arguments,
                )
        except CheckpointReached:
            if stop is None:
                raise
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if not result or result["optimizer_steps"] <= 0:
            raise ValueError("No actual successful updates in throughput verification.")
        return result, elapsed

    cases = []
    for amp in (torch.float16, torch.bfloat16):
        for algorithm in ("bc", "cql"):
            proof.write("throughput-status", {
                "status": "running", "phase": f"{algorithm}_{amp}", "pid": os.getpid(),
            })
            old, _ = run(algorithm, amp, old=True)
            current, _ = run(algorithm, amp)
            compare(old, current, f"{algorithm}_{amp}_previous_loop")
            partial, _ = run(algorithm, amp, stop=64, cached=algorithm == "cql", monitor=1)
            stream = io.BytesIO()
            torch.save(partial, stream)
            stream.seek(0)
            restored, _ = run(
                algorithm, amp, saved=torch.load(stream, weights_only=True),
                cached=algorithm == "cql", monitor=1,
            )
            compare(current, restored, f"{algorithm}_{amp}_resume")
            if algorithm == "cql":
                warm, _ = run(algorithm, amp, cached=True, monitor=1)
                compare(current, warm, f"{amp}_warm_cache_and_monitor")
            cases.append({
                "algorithm": algorithm, "amp": str(amp), "samples": current["cursor"],
                "successful_updates": current["optimizer_steps"],
                "previous_loop_bitwise_identical": True, "resume_cursor": 64,
                "resume_components_bitwise_identical": list(current),
                "cache_and_monitor_bitwise_identical": algorithm == "cql",
            })
    times = {algorithm: {"previous": [], "current": []} for algorithm in ("bc", "cql")}
    for repeat in range(4):
        for algorithm in times:
            for variant in (("previous", "current") if repeat % 2 == 0 else ("current", "previous")):
                _, elapsed = run(algorithm, torch.float16, old=variant == "previous")
                times[algorithm][variant].append(elapsed)
    benchmark = {
        name: {variant: {"seconds": data, "median_seconds": statistics.median(data)}
               for variant, data in runs.items()} for name, runs in times.items()
    }
    verify_sources()
    for key, path in model_paths.items():
        if digest(path) != model_hashes[key]:
            raise ValueError("A retained model changed during throughput verification.")
    receipt = {
        "status": "complete", "scope": "real_offline_cuda_loops_cache_and_exact_resume_only",
        "cases": cases, "benchmark": benchmark,
        "benchmark_repeats": 4, "benchmark_order": "alternating",
        "benchmark_includes": "real input, forward/backward, optimizer, scalar metrics, RAM snapshots",
        "benchmark_excludes": "full corpus, simulation, persistent checkpoint I/O",
        "source_manifest_sha256": digest(OUTPUT / "optimization-throughput-source-manifest.json"),
        "retained_models": model_hashes, "new_experiment_directories": False,
        "playing_strength_established": False,
        "previous_probe_issue": previous_issue,
    }
    proof.write("throughput-acceptance", receipt)
    proof.write("throughput-status", {"status": "complete", "phase": "complete", "pid": os.getpid()})
    if previous_issue:
        (OUTPUT / "optimization-throughput-error.json").unlink()
    print(json.dumps(receipt), flush=True)


def bc_result_checks(plan_path):
    """Independently read the finished BC artifacts and repeat held-out metrics."""
    proof = helper()
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8-sig"))
    prefix = Path(plan["output_prefix"])
    artifact = lambda suffix: prefix.with_name(prefix.name + suffix)
    summary = proof.read(artifact("-summary.json"))
    if summary["status"] != "complete" or summary["plan_sha256"] != digest(plan_path):
        raise ValueError("BC trial is incomplete or belongs to another plan.")
    if digest(artifact(".kanachan")) != summary["model_sha256"]:
        raise ValueError("Selected BC model checksum differs.")
    if digest(artifact("-checkpoint.pt")) != summary["checkpoint_sha256"]:
        raise ValueError("BC checkpoint checksum differs.")
    if digest(plan["baseline"]["path"]) != plan["baseline"]["sha256"]:
        raise ValueError("The original BC model differs.")
    if (Path(plan["source"]).resolve() != SOURCE.resolve()
            or digest(plan["manifest"]) != plan["manifest_sha256"]):
        raise ValueError("The BC source location or manifest checksum differs.")
    expected = proof.read(Path(plan["manifest"]))["complete_candidate"]
    for name, checksum in expected.items():
        if digest(SOURCE / name) != checksum:
            raise ValueError(f"BC source differs: {name}")
    native_sha = digest(SOURCE / "kanachan/simulation/_simulation.pyd")
    if native_sha != NATIVE_SHA:
        raise ValueError("The verified native rules extension differs.")
    for split, identity in plan["dataset"].items():
        if digest(identity["path"]) != identity["sha256"]:
            raise ValueError(f"The BC {split} data differs.")
    torch = proof.setup_runtime()
    from kanachan.model_loader import load_model
    from kanachan.training.core.bc import DataLoader
    from kanachan.training.bc.evaluate import evaluate
    from kanachan.simulation.statistics import summarize
    saved = torch.load(artifact("-checkpoint.pt"), map_location="cpu", weights_only=True)
    if not saved["cursor_is_exact"] or saved["cursor"] != plan["dataset"]["train"]["rows"]:
        raise ValueError("BC did not complete the planned exact single-epoch cursor.")
    if saved["cursor"] != summary["cursor"] or saved["optimizer_steps"] != summary["optimizer_steps"]:
        raise ValueError("BC checkpoint and summary counters differ.")
    attempts = (saved["cursor"] + plan["batch_size"] - 1) // plan["batch_size"]
    if attempts - summary["skipped_updates"] != saved["optimizer_steps"]:
        raise ValueError("Successful and skipped BC updates do not reconcile.")
    optimizer_counts = sorted({int(value["step"].item()) for value in saved["optimizer"]["state"].values()})
    if optimizer_counts != [saved["optimizer_steps"]]:
        raise ValueError("Actual AdamW update counters differ from the BC summary.")
    if saved["scheduler"]["last_epoch"] != saved["optimizer_steps"]:
        raise ValueError("Actual BC scheduler cursor differs.")
    best = min(summary["validation_curve"], key=lambda value: value["negative_log_likelihood"])
    if best != summary["selected_validation"] or best["cursor"] != summary["selected_cursor"]:
        raise ValueError("BC selected checkpoint violates the declared validation rule.")
    actor = load_model(artifact(".kanachan"), map_location=torch.device("cpu")).float()
    if summary["selected_cursor"] == saved["cursor"]:
        compare(actor.state_dict(), saved["model"], "selected_portable_model_and_last_checkpoint")
    actor = actor.cuda().eval()
    actor.requires_grad_(False)
    if sum(p.numel() for p in actor.parameters()) != plan["parameters"]:
        raise ValueError("The exported BC parameter count differs.")
    if any(p.dtype != torch.float32 for p in actor.parameters()):
        raise ValueError("The exported BC master parameter precision differs.")
    results = {}
    for split in ("validation", "test"):
        measured = evaluate(actor, DataLoader(
            path=Path(plan["dataset"][split]["path"]), num_skip_samples=0,
            rewrite_rooms=None, rewrite_grades=None, batch_size=128, num_workers=0,
        ), torch.device("cuda:0"))
        reference = summary["selected_validation"] if split == "validation" else summary["test"]
        for key, value in measured.items():
            if value != reference[key]:
                raise ValueError(f"The independently remeasured {split}.{key} differs: {value} != {reference[key]}")
        results[split] = measured
    native = proof.read(artifact("-native.json"))
    if (native["proposed_sha256"] != summary["model_sha256"]
            or native["baseline_sha256"] != plan["baseline"]["sha256"]
            or native["seed"] != plan["native_seed"]
            or len(native["results"]) != plan["native_sets"] * 4):
        raise ValueError("The BC native evaluation identity or game count differs.")
    metadata = native["metadata"]
    counts = {}
    seats = {}
    for entry, game in zip(metadata, native["results"], strict=True):
        counts[entry["set_id"]] = counts.get(entry["set_id"], 0) + 1
        proposed = [index for index, player in enumerate(game) if player["proposed"]]
        if len(proposed) != 1:
            raise ValueError("BC native evaluation did not use one proposed seat.")
        seats.setdefault(entry["set_id"], set()).add(proposed[0])
    if len(counts) != plan["native_sets"] or set(counts.values()) != {4}:
        raise ValueError("BC native evaluation is missing complete four-seat groups.")
    if any(value != set(range(4)) for value in seats.values()):
        raise ValueError("BC native evaluation omitted a proposed-player seat.")
    statistics = summarize(
        native["results"], group_ids=[entry["set_id"] for entry in metadata],
        bootstrap_samples=2000, seed=plan["native_seed"], confidence=.95,
    )
    compare(native["statistics"], statistics, "independent_native_statistics")
    for name, checksum in expected.items():
        if digest(SOURCE / name) != checksum:
            raise ValueError(f"BC source changed during verification: {name}")
    receipt = {
        "status": "complete", "scope": "finished_small_decoder_bc_trial_independent_readback",
        "model_sha256": summary["model_sha256"], "parameters": plan["parameters"],
        "cursor": saved["cursor"], "optimizer_steps": saved["optimizer_steps"],
        "skipped_updates": summary["skipped_updates"], "optimizer_actual_step_values": optimizer_counts,
        "scheduler_cursor_verified": True, "exact_cursor_verified": True,
        "selected_cursor": summary["selected_cursor"], "selection_rule_verified": True,
        "heldout_metrics_exactly_reproduced": results,
        "native_games": len(native["results"]), "independent_seed_groups": len(counts),
        "native_statistics_recomputed": statistics, "native_sha256": native_sha,
        "source_files_verified": len(expected), "old_bc_model_unchanged": True,
        "new_training_started": False,
    }
    proof.write("bc-result-acceptance", receipt)
    print(json.dumps(receipt), flush=True)


def cql_result_checks(plan_path):
    """Read real counters and independently repeat Q diagnostics and paired games."""
    proof = helper()
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8-sig"))
    target_kind = plan.get("target_kind", "td")
    if target_kind not in ("td", "mc"):
        raise ValueError("Unknown CQL target kind.")
    regression_key = "mse_mc" if target_kind == "mc" else "mse_td"
    prefix = Path(plan["output_prefix"])
    artifact = lambda suffix: prefix.with_name(prefix.name + suffix)
    summary = proof.read(artifact("-summary.json"))
    if summary["status"] != "complete" or summary["plan_sha256"] != digest(plan_path):
        raise ValueError("CQL trial identity or completion differs.")
    if digest(plan["manifest"]) != plan["manifest_sha256"] or Path(plan["source"]).resolve() != SOURCE.resolve():
        raise ValueError("CQL source location or manifest identity differs.")
    expected = proof.read(Path(plan["manifest"]))["complete_candidate"]
    for name, checksum in expected.items():
        if digest(SOURCE / name) != checksum:
            raise ValueError(f"CQL source differs: {name}")
    for split, identity in plan["dataset"].items():
        if digest(identity["path"]) != identity["sha256"]:
            raise ValueError(f"CQL {split} source differs.")
    for name in ("bc", "opponent", "template", "pilot_data"):
        if digest(plan[name]["path"]) != plan[name]["sha256"]:
            raise ValueError(f"Retained CQL {name} input differs.")
    if digest(artifact(".kanachan")) != summary["model_sha256"] or digest(artifact("-checkpoint.pt")) != summary["checkpoint_sha256"]:
        raise ValueError("CQL model or checkpoint bytes differ.")
    if digest(SOURCE / "kanachan/simulation/_simulation.pyd") != plan["native_sha256"]:
        raise ValueError("CQL native extension differs.")
    torch = proof.setup_runtime()
    from kanachan.model_loader import load_model
    from kanachan.constants import NUM_TYPES_OF_ACTIONS
    from kanachan.nn.decision_decoder import legal_argmax
    from kanachan.nn.decision_value import q_only
    from kanachan.training.core.offline_rl import EpisodeReplayBuffer
    from kanachan.training.core.rl.rewards import configured_reward
    from kanachan.training.core.rl.discrete_cql import td_components, mc_components, td_penalty, conservative_penalty
    from kanachan.training.online.state import restore_rng
    from kanachan.simulation.statistics import summarize, compare_policies
    checkpoint = torch.load(artifact("-checkpoint.pt"), map_location="cpu", weights_only=True)
    if checkpoint.get("target_kind", "td") != target_kind:
        raise ValueError("Checkpoint CQL target kind differs.")
    if (not checkpoint["cursor_is_exact"] or checkpoint["plan_sha256"] != digest(plan_path)
            or checkpoint["cursor"] != summary["cursor"]
            or checkpoint["optimizer_steps"] != summary["optimizer_steps"]):
        raise ValueError("CQL checkpoint identity or exact counters differ.")
    if not 0 < checkpoint["cursor"] <= plan["dataset"]["train"]["rows"]:
        raise ValueError("CQL consumed cursor exceeds its training data.")
    epoch = checkpoint.get("epoch", 0)
    rows = plan["dataset"]["train"]["rows"]
    batch_size = plan["batch_size"]
    if type(epoch) is not int or not 0 <= epoch < plan.get("max_epochs", 1):
        raise ValueError("CQL checkpoint epoch differs from its training plan.")
    attempts = epoch * ((rows + batch_size - 1) // batch_size) + (checkpoint["cursor"] + batch_size - 1) // batch_size
    total_samples = epoch * rows + checkpoint["cursor"]
    if (checkpoint.get("total_samples", checkpoint["cursor"]) != total_samples
            or summary.get("total_samples", summary["cursor"]) != total_samples
            or summary.get("completed_epochs", int(checkpoint["cursor"] == rows)) != epoch + int(checkpoint["cursor"] == rows)):
        raise ValueError("CQL global sample or epoch counters differ.")
    if attempts - summary["skipped_updates"] != checkpoint["optimizer_steps"]:
        raise ValueError("CQL actual/AMP-skipped updates do not reconcile.")
    steps = sorted({int(value["step"].item()) for value in checkpoint["optimizer"]["state"].values()})
    if steps != [checkpoint["optimizer_steps"]] or checkpoint["scheduler"]["last_epoch"] != checkpoint["optimizer_steps"]:
        raise ValueError("CQL actual optimizer or scheduler steps differ.")
    replay = checkpoint["replay"]
    if (replay["format_version"] != 2 or replay["sampling_policy"] != "swap_v2"
            or replay["max_size"] != plan["replay_size"] or replay["batch_size"] != plan["batch_size"]
            or replay["discount_factor"] != 1. or not replay["contiguous"]
            or replay["input_samples"] != summary["input_samples"]):
        raise ValueError("CQL serialized replay identity differs.")
    restore_rng(checkpoint["rng"])
    actor = load_model(artifact(".kanachan"), map_location=torch.device("cpu")).float()
    compare(actor.state_dict(), checkpoint["model"], "cql_online_export")
    if sum(parameter.numel() for parameter in actor.parameters()) != plan["parameters"]:
        raise ValueError("CQL exported parameter count differs.")
    actor = actor.cuda().eval().requires_grad_(False)
    if target_kind == "mc":
        if checkpoint["target"] is not None:
            raise ValueError("MC CQL checkpoint unexpectedly has a target network.")
        target = target_values = None
    else:
        target = copy.deepcopy(actor.select_subsequence(out_keys=["action_value"]))
        target.load_state_dict(checkpoint["target"])
        target.eval().requires_grad_(False)
        target_values = q_only(target)
    source_values = q_only(actor)
    del checkpoint
    heldout = {}
    for split in ("validation", "test"):
        reader = EpisodeReplayBuffer(
            training_data=Path(plan["dataset"][split]["path"]), contiguous_training_data=True,
            num_skip_samples=0, rewrite_rooms=None, rewrite_grades=None,
            get_reward=configured_reward(plan["reward_scheme"], score_unit=plan["round_score_unit"], round_weight=plan["round_reward_weight"]),
            discount_factor=1., dtype=torch.float32,
            max_size=8192, batch_size=128, num_workers=0, pin_memory=False,
        )
        totals = {key: 0. for key in (regression_key, "cql_gap", "centered_cql_gap",
                                       "logged_return_mae", "logged_return_bias", "target", "q")}
        count = terminals = agreement = 0
        samples = []
        with torch.no_grad():
            for batch in reader:
                data = batch.cuda()
                observed = data["action_value"].clone()
                values, chosen, expected_return = (
                    mc_components(source_values, data) if target_kind == "mc"
                    else td_components(source_values, target_values, data)
                )
                gap = conservative_penalty(values, chosen)
                legal_count = (data["candidates"] < NUM_TYPES_OF_ACTIONS).sum(-1).float()
                measured = {
                    regression_key: (chosen - expected_return).square().sum(),
                    "cql_gap": gap.sum(), "centered_cql_gap": (gap - legal_count.log()).sum(),
                    "logged_return_mae": (chosen - observed).abs().sum(),
                    "logged_return_bias": (chosen - observed).sum(),
                    "target": expected_return.sum(), "q": chosen.sum(),
                    "agreement": (legal_argmax(data["candidates"], values) == data["action"]).sum(),
                    "terminals": data["next", "done"].sum(),
                }
                numbers = torch.stack([value.float() for value in measured.values()]).cpu().tolist()
                measured = dict(zip(measured, numbers, strict=True))
                for key in totals:
                    totals[key] += measured[key]
                count += len(batch)
                terminals += int(measured["terminals"])
                agreement += int(measured["agreement"])
                samples.append(chosen.cpu())
        if count != plan["dataset"][split]["rows"] or terminals != plan["dataset"][split]["terminal_trajectories"]:
            raise ValueError("Independent CQL held-out row/terminal coverage differs.")
        measured = {key: value / count for key, value in totals.items()}
        measured.update(
            behavior_action_agreement=agreement / count,
            samples=count, terminal_trajectories=terminals,
            q_percentiles=torch.quantile(torch.cat(samples), torch.tensor([.05, .5, .95])).tolist(),
        )
        reference = summary[split]
        maximum_error = 0.
        for key in measured:
            if key in ("samples", "terminal_trajectories", "behavior_action_agreement"):
                if measured[key] != reference[key]:
                    raise ValueError(f"Independent CQL {split}.{key} differs.")
            else:
                first, second = measured[key], reference[key]
                if not isinstance(first, list):
                    first, second = [first], [second]
                for left, right in zip(first, second, strict=True):
                    error = abs(left - right)
                    maximum_error = max(maximum_error, error)
                    if not math.isfinite(left) or error > 2e-6 * max(1., abs(right)):
                        raise ValueError(f"Independent CQL {split}.{key} differs: {left} != {right}")
        heldout[split] = dict(metrics=measured, maximum_absolute_reduction_error=maximum_error)
    records, compact = {}, {}
    for arm in ("bc", "cql"):
        record = proof.read(artifact(f"-{arm}-native.json"))
        checksum = plan["bc"]["sha256"] if arm == "bc" else summary["model_sha256"]
        if (record["model_sha256"] != checksum or record["opponent_sha256"] != plan["opponent"]["sha256"]
                or record["seed"] != plan["native_seed"] or len(record["results"]) != plan["native_sets"] * 4):
            raise ValueError("CQL/BC paired game identity or count differs.")
        seats = {}
        for metadata, game in zip(record["metadata"], record["results"], strict=True):
            proposed = [index for index, player in enumerate(game) if player["proposed"]]
            if len(proposed) != 1:
                raise ValueError("CQL evaluation did not use one proposed player.")
            seats.setdefault(metadata["set_id"], []).append(proposed[0])
        if len(seats) != plan["native_sets"] or any(sorted(value) != [0, 1, 2, 3] for value in seats.values()):
            raise ValueError("CQL paired evaluation is missing four-seat groups.")
        stats = summarize(
            record["results"], group_ids=[value["set_id"] for value in record["metadata"]],
            bootstrap_samples=2000, seed=plan["native_seed"], confidence=.95,
        )
        compare(stats, record["statistics"], f"independent_{arm}_statistics")
        records[arm] = record
        compact[arm] = dict(
            metadata=record["metadata"],
            results=[[{key: player[key] for key in ("proposed", "score", "ranking")} for player in game]
                     for game in record["results"]],
            statistics=stats,
        )
    if records["bc"]["metadata"] != records["cql"]["metadata"]:
        raise ValueError("CQL and BC did not use matching paired game metadata.")
    paired = compare_policies(
        records["cql"]["results"], records["bc"]["results"],
        group_ids=[value["set_id"] for value in records["bc"]["metadata"]],
        bootstrap_samples=4000, seed=plan["native_seed"], confidence=.95,
    )
    compare(paired, summary["paired_comparison"], "independent_paired_cql_vs_bc")
    for name, checksum in expected.items():
        if digest(SOURCE / name) != checksum:
            raise ValueError("CQL source changed during independent verification.")
    receipt = {
        "status": "complete", "scope": "small_decoder_cql_independent_readback_and_heldout_recheck",
        "model_sha256": summary["model_sha256"], "parameters": plan["parameters"], "target_kind": target_kind,
        "cursor": summary["cursor"], "optimizer_steps": summary["optimizer_steps"],
        "epoch": epoch, "total_samples": total_samples,
        "skipped_updates": summary["skipped_updates"], "actual_optimizer_step_values": steps,
        "exact_replay_and_scheduler_counters_verified": True,
        "heldout_diagnostics_independently_reproduced": heldout,
        "native_compact": compact, "paired_comparison": paired,
        "source_files_verified": len(expected), "retained_bc_models_unchanged": True,
        "new_training_started": False,
    }
    destination = prefix.with_name(prefix.name + "-independent.json")
    destination.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "native_compact"}), flush=True)


def stability_result_checks(receipt_path, policy=False):
    """Independently re-read a bounded value-anchor ablation without training."""
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8-sig"))
    receipt = read(receipt_path)
    protocol = receipt["protocol"]
    if receipt["status"] != "complete" or protocol["status"] != "declared_before_training":
        raise ValueError("The stability ablation is incomplete.")
    if Path(protocol["source"]).resolve() != SOURCE.resolve():
        raise ValueError("Controlled source location differs.")
    if digest(protocol["manifest"]) != protocol["manifest_sha256"]:
        raise ValueError("Controlled manifest differs.")
    if digest(protocol["original_plan"]) != protocol["original_plan_sha256"]:
        raise ValueError("Original input plan differs.")
    plan = read(protocol["original_plan"])
    expected = read(protocol["manifest"])["complete_candidate"]
    for name, checksum in expected.items():
        if digest(SOURCE/name) != checksum:
            raise ValueError(f"Source differs: {name}")
    for name in ("bc", "opponent", "template", "pilot_data"):
        if digest(plan[name]["path"]) != plan[name]["sha256"]:
            raise ValueError(f"Preserved {name} differs.")
    for split in ("train", "validation"):
        if digest(plan["dataset"][split]["path"]) != plan["dataset"][split]["sha256"]:
            raise ValueError(f"Controlled {split} data differs.")
    os.environ.update(TORCHDYNAMO_DISABLE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1")
    sys.path.insert(0, str(SOURCE))
    import random
    import torch
    from kanachan.constants import NUM_TYPES_OF_ACTIONS
    from kanachan.model_loader import load_model
    from kanachan.nn.decision_decoder import legal_argmax
    from kanachan.nn.decision_value import q_only
    from kanachan.training.core.offline_rl import EpisodeReplayBuffer
    from kanachan.training.core.rl.discrete_cql import OBSERVATIONS, td_components, td_penalty, conservative_penalty
    from kanachan.training.core.rl.rewards import RoundAndPlacementReward
    from kanachan.training.online.state import restore_rng
    from kanachan.simulation.statistics import summarize, compare_policies

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if digest(SOURCE/"kanachan/simulation/_simulation.pyd") != plan["native_sha256"]:
        raise ValueError("Native extension differs.")
    checkpoint = torch.load(BASE/("cql-policy-checkpoint.pt" if policy else "cql-stability-checkpoint.pt"), map_location="cpu", weights_only=True)
    protocol_path = Path(receipt_path).with_name(Path(receipt_path).stem+"-protocol.json")
    if checkpoint["protocol_sha256"] != digest(protocol_path) or read(protocol_path) != protocol:
        raise ValueError("Candidate checkpoint belongs to another protocol.")
    candidate = receipt["arms"]["candidate"]
    if checkpoint["recipe"] != candidate["recipe"] or checkpoint["cursor"] != protocol["samples_per_arm"]:
        raise ValueError("Candidate objective or actual sample count differs.")
    optimizer_steps = sorted({int(state["step"].item()) for state in checkpoint["optimizer"]["state"].values()})
    if optimizer_steps != [candidate["actual_optimizer_steps"]] or checkpoint["scheduler"]["last_epoch"] != optimizer_steps[0]:
        raise ValueError("Actual candidate optimizer/scheduler counts differ.")
    if checkpoint["initial_model_sha256"] != candidate["initial_model_sha256"]:
        raise ValueError("Candidate initialization identity differs.")
    replay_identity = f"round-and-placement-v1:unit={10000.:.17g}:weight={.2:.17g}"
    if checkpoint["replay"]["reward_identity"] != replay_identity:
        raise ValueError("Candidate replay uses another reward objective.")
    if not policy and len({arm["initial_model_sha256"] for arm in receipt["arms"].values()}) != 1:
        raise ValueError("Controlled initial weights differ.")
    if policy:
        if candidate["initial_feature_sha256"] != candidate["final_feature_sha256"]:
            raise ValueError("Candidate BC features changed during the protected warmup.")
        if candidate["trainable_parameters_at_end"] != 70434:
            raise ValueError("Candidate warmup trained an unexpected set of parameters.")
    # Reproduce the full sequence of sampled raw transitions, without updates.
    restore_rng(checkpoint["initial_rng"])
    reader = EpisodeReplayBuffer(
        training_data=Path(plan["dataset"]["train"]["path"]), contiguous_training_data=True,
        num_skip_samples=0, rewrite_rooms=None, rewrite_grades=None,
        get_reward=RoundAndPlacementReward(round_weight=.2), discount_factor=1., dtype=torch.float32,
        max_size=8192, batch_size=32, num_workers=0, pin_memory=False)
    restore_rng(checkpoint["initial_rng"])
    schedule = hashlib.sha256()
    count = 0
    for batch, _ in reader.iter_with_last():
        batch = batch[:protocol["samples_per_arm"]-count]
        for key in (*OBSERVATIONS, "action", *(("next", key) for key in OBSERVATIONS), ("next", "done")):
            schedule.update(batch[key].contiguous().numpy().tobytes())
        count += len(batch)
        if count == protocol["samples_per_arm"]:
            break
    schedule_digest = schedule.hexdigest()
    if any(arm["sample_schedule_sha256"] != schedule_digest for arm in receipt["arms"].values()):
        raise ValueError("Independent training-transition schedule differs.")
    del reader
    checked = {}
    for name, arm in receipt["arms"].items():
        if digest(arm["model"]) != arm["model_sha256"]:
            raise ValueError(f"Controlled {name} model differs.")
        actor = load_model(Path(arm["model"]), map_location=torch.device("cpu")).float()
        if name == "candidate":
            compare(actor.state_dict(), checkpoint["model"], "portable_candidate")
            if policy:
                from kanachan.nn.decision_decoder import DenseDecisionDecoder, GlobalActionHead
                teacher = load_model(Path(plan["bc"]["path"]), map_location=torch.device("cpu")).float()
                teacher_body = next(m for m in teacher.modules() if isinstance(m, DenseDecisionDecoder))
                teacher_hidden = next(m for m in teacher.modules() if isinstance(m, GlobalActionHead)).layers[0]
                body = next(m for m in actor.modules() if isinstance(m, DenseDecisionDecoder))
                hidden = next(m for m in actor.modules() if isinstance(m, GlobalActionHead)).layers[0]
                compare(body.state_dict(), teacher_body.state_dict(), "preserved_bc_backbone")
                compare(hidden.state_dict(), teacher_hidden.state_dict(), "preserved_bc_hidden_head")
                del teacher
        actor = actor.cuda().eval().requires_grad_(False)
        if name == "candidate":
            target = copy.deepcopy(actor.select_subsequence(out_keys=["action_value"]))
            target.load_state_dict(checkpoint["target"])
            target.eval().requires_grad_(False)
        else:
            target = None
        qnet = q_only(actor)
        random.seed(protocol["validation_seed"])
        torch.manual_seed(protocol["validation_seed"])
        loader = EpisodeReplayBuffer(
            training_data=Path(plan["dataset"]["validation"]["path"]), contiguous_training_data=True,
            num_skip_samples=0, rewrite_rooms=None, rewrite_grades=None,
            get_reward=RoundAndPlacementReward(round_weight=arm["recipe"]["round_weight"]),
            discount_factor=1., dtype=torch.float32, max_size=8192, batch_size=128,
            num_workers=0, pin_memory=False)
        totals = {key: 0. for key in ("q", "mc_mae", "mc_bias", "td_mse", "target", "cql_gap",
                                       "terminal_residual_sum", "nonterminal_residual_sum")}
        rows = terminals = agreements = 0
        qs, observed_values, targets = [], [], []
        with torch.no_grad():
            for batch in loader:
                data = batch.cuda()
                observed = data["action_value"].clone()
                if target is not None:
                    values, chosen, expected_return = td_components(qnet, q_only(target), data, mc_weight=.5)
                else:
                    observation = data.select(*OBSERVATIONS).clone(False)
                    qnet(observation)
                    values = observation["action_value"].masked_fill(data["candidates"] >= NUM_TYPES_OF_ACTIONS, -torch.inf)
                    chosen = values.gather(1, data["action"].long()[:,None]).squeeze(1)
                    expected_return = None
                quantities = dict(q=chosen.sum(), mc_mae=(chosen-observed).abs().sum(),
                                  mc_bias=(chosen-observed).sum(), cql_gap=conservative_penalty(values, chosen).sum())
                if expected_return is not None:
                    residual = chosen-expected_return
                    done = data["next", "done"].bool()
                    quantities.update(td_mse=td_penalty(chosen, expected_return).sum(), target=expected_return.sum(),
                                      terminal_residual_sum=residual.masked_fill(~done,0).sum(),
                                      nonterminal_residual_sum=residual.masked_fill(done,0).sum())
                    targets.append(expected_return.cpu())
                for key, value in quantities.items():
                    totals[key] += float(value)
                rows += len(data)
                terminals += int(data["next", "done"].sum())
                agreements += int((legal_argmax(data["candidates"], values) == data["action"]).sum())
                qs.append(chosen.cpu())
                observed_values.append(observed.cpu())
        quantile = lambda values: torch.quantile(torch.cat(values), torch.tensor([.05,.5,.95])).tolist()
        measured = dict(samples=rows, terminals=terminals, behavior_agreement=agreements/rows,
                        q_percentiles=quantile(qs), observed_return_percentiles=quantile(observed_values))
        for key, value in totals.items():
            if name == "candidate" or key not in ("target", "td_mse", "terminal_residual_sum", "nonterminal_residual_sum"):
                measured[key] = value/rows
        if targets:
            measured["target_percentiles"] = quantile(targets)
        maximum = 0.
        for key, value in measured.items():
            reference = arm["validation"][key]
            left = value if isinstance(value, list) else [value]
            right = reference if isinstance(reference, list) else [reference]
            for first, second in zip(left, right, strict=True):
                error = abs(first-second)
                if not math.isfinite(first) or error > 2e-6*max(1.,abs(second)):
                    raise ValueError(f"Independent {name}.{key} differs: {first} != {second}")
                maximum = max(maximum,error)
        checked[name] = dict(metrics=measured, maximum_absolute_reduction_error=maximum)
        del actor, target, qnet, loader
    for name, record in receipt["native"].items():
        metadata, results = record["metadata"], record["results"]
        if len(results) != protocol["native_games_per_arm"]:
            raise ValueError("Controlled native game count differs.")
        seats = {}
        for item, game in zip(metadata, results, strict=True):
            proposed = [index for index, player in enumerate(game) if player["proposed"]]
            if len(proposed) != 1:
                raise ValueError("Controlled native game has invalid proposed seats.")
            seats.setdefault(item["set_id"], []).append(proposed[0])
        if len(seats) != protocol["native_groups"] or any(sorted(value) != [0,1,2,3] for value in seats.values()):
            raise ValueError("Controlled native seat coverage differs.")
        stats = summarize(results, group_ids=[item["set_id"] for item in metadata], seed=protocol["native_seed"])
        compare(stats, record["statistics"], "independent_native_"+name)
    for key, reference in receipt["paired_comparisons"].items():
        left, right = key.split("_minus_")
        if receipt["native"][left]["metadata"] != receipt["native"][right]["metadata"]:
            raise ValueError("Controlled paired metadata differs.")
        actual = compare_policies(receipt["native"][left]["results"], receipt["native"][right]["results"],
                                  group_ids=[item["set_id"] for item in receipt["native"][right]["metadata"]],
                                  bootstrap_samples=4000, seed=protocol["native_seed"], confidence=.95)
        compare(actual, reference, "independent_paired_"+key)
    if policy:
        for name, checksum in expected.items():
            if digest(SOURCE/name) != checksum:
                raise ValueError("Policy source changed during independent verification.")
        result = dict(status="complete", scope="independent_cql_feature_warmup_readback_only",
                      receipt_sha256=digest(receipt_path), source_files_verified=len(expected),
                      sampled_training_rows=count, sample_schedule_sha256=schedule_digest,
                      same_training_rows_independently_reproduced=True,
                      bc_backbone_and_hidden_features_bitwise_preserved=True,
                      candidate_checkpoint_and_export_identical=True,
                      actual_optimizer_steps=optimizer_steps, heldout=checked,
                      native_statistics_and_paired_intervals_recomputed=True, new_training_started=False)
        destination = Path(receipt_path).with_name("optimization-cql-policy-result-acceptance.json")
        destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",encoding="utf-8")
        print(json.dumps(result), flush=True)
        return
    # Read-only gradient diagnosis: no optimizer and no weight writes. Compare
    # the actual value/CQL objectives on identical training transitions.
    from kanachan.nn.decision_decoder import DenseDecisionDecoder, GlobalActionHead
    actor = load_model(Path(candidate["model"]), map_location=torch.device("cpu")).float().cuda().eval().requires_grad_(True)
    target = copy.deepcopy(actor.select_subsequence(out_keys=["action_value"]))
    target.load_state_dict(checkpoint["target"])
    target.eval().requires_grad_(False)
    teacher = load_model(Path(plan["bc"]["path"]), map_location=torch.device("cpu")).float().cuda().eval().requires_grad_(False)
    body = next(m for m in actor.modules() if isinstance(m, DenseDecisionDecoder))
    head = next(m for m in actor.modules() if isinstance(m, GlobalActionHead))
    teacher_body = next(m for m in teacher.modules() if isinstance(m, DenseDecisionDecoder))
    teacher_head = next(m for m in teacher.modules() if isinstance(m, GlobalActionHead))
    network, target_network = q_only(actor), q_only(target)
    parameters = tuple(actor.parameters())
    averages = {key: torch.zeros(sum(p.numel() for p in parameters), device="cuda")
                for key in ("td", "anchored", "cql")}
    norms = {key: [] for key in averages}
    hybrid_agreements = {key: 0 for key in ("bc", "bc_head_current_body", "current_head_bc_body", "current")}
    random.seed(2026110411)
    reader = EpisodeReplayBuffer(
        training_data=Path(plan["dataset"]["train"]["path"]), contiguous_training_data=True,
        num_skip_samples=0, rewrite_rooms=None, rewrite_grades=None,
        get_reward=RoundAndPlacementReward(round_weight=.2), discount_factor=1., dtype=torch.float32,
        max_size=8192, batch_size=32, num_workers=0, pin_memory=False)
    gradient_rows = 0
    for batch in reader:
        data = batch.cuda()
        observed = data["action_value"].detach()
        values, chosen, anchored_target = td_components(network, target_network, data, mc_weight=.5)
        plain_target = 2*anchored_target-observed
        losses = dict(td=td_penalty(chosen, plain_target).mean(),
                      anchored=td_penalty(chosen, anchored_target).mean(),
                      cql=conservative_penalty(values, chosen).mean())
        for key, loss in losses.items():
            gradients = torch.autograd.grad(loss, parameters, retain_graph=key != "cql", allow_unused=True)
            vector = torch.cat([(torch.zeros_like(p) if g is None else g.detach()).reshape(-1)
                                for p,g in zip(parameters,gradients,strict=True)])
            averages[key] += vector * len(data)
            norms[key].append(float(vector.norm()))
        with torch.no_grad():
            args = [data[name] for name in OBSERVATIONS]
            teacher_representation = teacher_body(*args)
            current_representation = body(*args)
            variants = dict(bc=teacher_head(data["candidates"], teacher_representation),
                            bc_head_current_body=teacher_head(data["candidates"], current_representation),
                            current_head_bc_body=head(data["candidates"], teacher_representation), current=values)
            for name, logits in variants.items():
                hybrid_agreements[name] += int((legal_argmax(data["candidates"], logits) == data["action"]).sum())
        gradient_rows += len(data)
        if gradient_rows >= 2048:
            break
    average_norms = {name: float((value/gradient_rows).norm()) for name,value in averages.items()}
    gradient_diagnosis = dict(
        scope="fp32_reference_gradients_and_hybrid_feature_controls_only", training_rows=gradient_rows,
        mean_batch_gradient_norm={name: statistics.mean(values) for name,values in norms.items()},
        accumulated_mean_gradient_norm=average_norms,
        alpha=protocol["alpha"],
        weighted_cql_to_anchored_gradient_ratio=protocol["alpha"]*average_norms["cql"]/max(average_norms["anchored"],1e-12),
        hybrid_behavior_agreement={name: value/gradient_rows for name,value in hybrid_agreements.items()},
        optimizer_steps=0,
    )
    result = dict(status="complete", scope="independent_cql_stability_readback_only",
                  receipt_sha256=digest(receipt_path), source_files_verified=len(expected),
                  sampled_training_rows=count, sample_schedule_sha256=schedule_digest,
                  same_initial_weights_and_training_rows=True, actual_optimizer_steps=optimizer_steps,
                  candidate_checkpoint_and_export_identical=True, heldout=checked,
                  gradient_diagnosis=gradient_diagnosis,
                  native_statistics_and_paired_intervals_recomputed=True, new_training_started=False)
    destination = Path(receipt_path).with_name("optimization-cql-stability-result-acceptance.json")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result), flush=True)


def td_result_checks(receipt_path):
    """Independent readback of the current standard TD-CQL functional check."""
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8-sig"))
    if (receipt["status"] != "complete" or receipt["target_kind"] != "one_step_double_dqn"
            or receipt["feature_freezing"] or receipt["logged_return_mixing"]
            or receipt["learned_reward_predictor"]):
        raise ValueError("Unexpected TD-CQL functional recipe.")
    if Path(receipt["source"]).resolve() != SOURCE.resolve():
        raise ValueError("Development source differs.")
    if digest(receipt["manifest"]) != receipt["source_manifest_sha256"] or digest(receipt["plan"]) != receipt["plan_sha256"]:
        raise ValueError("TD-CQL source/input identity differs.")
    expected = json.loads(Path(receipt["manifest"]).read_text(encoding="utf-8-sig"))["complete_candidate"]
    for name, checksum in expected.items():
        if digest(SOURCE/name) != checksum:
            raise ValueError(f"Current source differs: {name}")
    plan = json.loads(Path(receipt["plan"]).read_text(encoding="utf-8-sig"))
    for name in ("bc", "opponent", "template", "pilot_data"):
        if digest(plan[name]["path"]) != plan[name]["sha256"]:
            raise ValueError(f"Retained input differs: {name}")
    if digest(receipt["model"]) != receipt["cuda_cql"]["model_sha256"] or digest(receipt["checkpoint"]) != receipt["checkpoint_sha256"]:
        raise ValueError("Functional artifacts differ.")
    os.environ.update(TORCHDYNAMO_DISABLE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1")
    sys.path.insert(0, str(SOURCE))
    import torch
    from kanachan.model_loader import load_model
    from kanachan.nn.decision_decoder import DenseDecisionDecoder, GlobalActionHead
    from kanachan.nn.decision_value import q_only
    from kanachan.training.core.offline_rl import DataLoader
    from kanachan.training.core.rl.rewards import PlacementReward, RoundAndPlacementReward
    from kanachan.training.core.rl.discrete_cql import td_components, OBSERVATIONS
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
    state = torch.load(receipt["checkpoint"], weights_only=True, map_location="cpu")
    if state["target_kind"] != "one_step_double_dqn" or state["source_manifest_sha256"] != receipt["source_manifest_sha256"]:
        raise ValueError("The checkpoint binds another target or source.")
    actor = load_model(Path(receipt["model"]), map_location=torch.device("cpu")).float()
    compare(actor.state_dict(), state["model"], "td_cql_portable_state")
    counters = sorted({int(item["step"]) for item in state["optimizer"]["state"].values()})
    if (counters != [state["optimizer_steps"]] or state["scheduler"]["last_epoch"] != counters[0]
            or len(state["optimizer"]["state"]) != len(list(actor.parameters()))
            or sum(p.numel() for p in actor.parameters()) != plan["parameters"]
            or state["cursor"] != 627):
        raise ValueError("Actual full-parameter optimizer/scheduler/cursor differs.")
    teacher = load_model(Path(plan["bc"]["path"]), map_location=torch.device("cpu")).float()
    changed = {}
    for name, get in (("backbone", lambda m: next(x for x in m.modules() if isinstance(x, DenseDecisionDecoder))),
                      ("hidden_head", lambda m: next(x for x in m.modules() if isinstance(x, GlobalActionHead)).layers[0])):
        new, old = get(actor).state_dict(), get(teacher).state_dict()
        changed[name] = any(not torch.equal(tensor, old[key]) for key, tensor in new.items())
        if not changed[name]:
            raise ValueError(f"The transferred {name} did not learn.")
    replay_identity = "round-and-placement-v1:unit=10000:weight=0.20000000000000001"
    if state["replay"]["reward_identity"] != replay_identity:
        raise ValueError("Replay uses a different reward scheme.")
    loader = DataLoader(path=Path(plan["pilot_data"]["path"]), num_skip_samples=0,
                        rewrite_rooms=None, rewrite_grades=None, get_reward=PlacementReward(),
                        dtype=torch.float32, batch_size=1024, num_workers=0)
    data = next(loader)
    if len(data) != 627 or next(loader, None) is not None or int(data["next", "end_of_game"].sum()) != 4:
        raise ValueError("Actual pilot coverage differs.")
    RoundAndPlacementReward()(data, True)
    actor = actor.cuda().eval().requires_grad_(False)
    target = copy.deepcopy(actor.select_subsequence(out_keys=["action_value"]))
    target.load_state_dict(state["target"])
    target.eval().requires_grad_(False)
    online, frozen = q_only(actor), q_only(target)
    maximum = 0.
    with torch.no_grad():
        for start in range(0, len(data), 32):
            batch = data[start:start+32].cuda()
            batch["action_value"] = torch.full((len(batch),), float("nan"), device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                _, _, actual = td_components(online, frozen, batch)
                reference = batch["next", "reward"].float().clone()
                active = ~batch["next", "done"].bool()
                if active.any():
                    obs = batch["next"][active]
                    first, second = (obs.select(*OBSERVATIONS).clone(False) for _ in range(2))
                    online(first)
                    frozen(second)
                    ids = obs["candidates"]
                    legal_q = first["action_value"].masked_fill(ids >= 546, -torch.inf)
                    best_id = ids.masked_fill(legal_q != legal_q.max(-1, keepdim=True).values, 546).min(-1).values
                    selected = (ids == best_id[:, None]).int().argmax(-1)
                    reference[active] += second["action_value"].gather(1, selected[:, None]).squeeze(1)
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                maximum = max(maximum, float((actual-reference).abs().max()))
    native = receipt["native_functional"]
    compare(summarize(native["results"]), native["statistics"], "td_native_statistics")
    if len(native["results"]) != 4:
        raise ValueError("Native functional coverage differs.")
    for name, checksum in expected.items():
        if digest(SOURCE/name) != checksum:
            raise ValueError("Source changed during independent readback.")
    result = dict(status="complete", scope="independent_standard_td_cql_functional_readback_only",
                  receipt_sha256=digest(receipt_path), source_files_verified=len(expected),
                  actual_optimizer_steps=counters, optimized_parameter_tensors=len(state["optimizer"]["state"]),
                  parameters=plan["parameters"], changed_bc_features=changed,
                  candidate_checkpoint_and_export_identical=True, real_decisions=627,
                  maximum_td_formula_error=maximum, poisoned_logged_returns_ignored=True,
                  native_statistics_recomputed=True, new_training_started=False)
    destination = Path(receipt_path).with_name("td-cql-independent-acceptance.json")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


def main(archive_sha, continue_after_probe_fix=False):
    proof = helper()
    started = time.monotonic()
    baseline_manifest = prepare(proof, archive_sha, continue_after_probe_fix)
    proof.phase("old_new_cuda_equivalence_and_benchmark")
    semantic_and_speed_checks(proof)
    proof.phase("cuda_amp_and_cache_checks")
    proof.gpu_checks()
    proof.phase("native_collection_and_in_memory_cuda_resume")
    online_checks(proof)
    for name, identity in baseline_manifest.items():
        if digest(BASE / "source" / name) != identity:
            raise ValueError("The updated development source changed during verification.")
    if proof.manifest(proof.BASELINE / "source") != proof.read(proof.BASELINE / "source-manifest.json")["files"]:
        raise ValueError("The preserved BC source changed.")
    if digest(proof.BASELINE / "model.kanachan") != proof.BC_SHA:
        raise ValueError("The preserved BC model changed.")
    acceptance = {
        "status": "complete", "scope": "decoder_optimization_functional_and_microbenchmark_only",
        "source": str(SOURCE), "archive_sha256": archive_sha,
        "checks": {key: proof.read(OUTPUT / f"optimization-{key}.json") for key in (
            "performance", "cuda-checks", "training-checks", "online-checks"
        )},
        "elapsed_seconds": time.monotonic() - started,
        "development_source_updated_in_place": True,
        "previous_source_archive_preserved": True,
        "old_bc_source_and_model_unchanged": True,
        "native_extension_sha256": digest(SOURCE / "kanachan/simulation/_simulation.pyd"),
        "production_deployed": False, "formal_gpu_training_started": False,
        "strength_improvement_established": False,
        "previous_probe_issue": proof.read(OUTPUT / "optimization-source-manifest.json").get("previous_probe_issue"),
    }
    proof.write("acceptance", acceptance)
    proof.write("status", {"status": "complete", "phase": "complete", "pid": os.getpid()})
    error_path = OUTPUT / "optimization-error.json"
    if error_path.is_file():
        error_path.unlink()
    print(json.dumps({"status": "complete", "receipt": str(OUTPUT / "optimization-acceptance.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-sha256")
    parser.add_argument("--continue-after-probe-fix", action="store_true")
    parser.add_argument("--precision-check", action="store_true")
    parser.add_argument("--throughput-check", action="store_true")
    parser.add_argument("--throughput-backup", action="store_true")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--bc-result-plan", type=Path)
    parser.add_argument("--cql-result-plan", type=Path)
    parser.add_argument("--stability-result", type=Path)
    parser.add_argument("--policy-result", type=Path)
    parser.add_argument("--td-result", type=Path)
    parser.add_argument("--device-sha256")
    parser.add_argument("--test-sha256")
    arguments = parser.parse_args()
    try:
        if arguments.td_result:
            td_result_checks(arguments.td_result)
        elif arguments.policy_result:
            stability_result_checks(arguments.policy_result, policy=True)
        elif arguments.stability_result:
            stability_result_checks(arguments.stability_result)
        elif arguments.cql_result_plan:
            cql_result_checks(arguments.cql_result_plan)
        elif arguments.bc_result_plan:
            bc_result_checks(arguments.bc_result_plan)
        elif arguments.throughput_backup:
            throughput_backup(arguments.manifest_sha256)
        elif arguments.throughput_check:
            throughput_checks(arguments.manifest_sha256, arguments.continue_after_probe_fix)
        elif arguments.precision_check:
            precision_checks(arguments.device_sha256, arguments.test_sha256)
        else:
            main(arguments.archive_sha256, arguments.continue_after_probe_fix)
    except BaseException as error:
        if arguments.bc_result_plan or arguments.cql_result_plan or arguments.stability_result or arguments.policy_result or arguments.td_result:
            raise
        if OUTPUT.exists():
            proof = helper()
            prefix = "throughput-" if arguments.throughput_check else ("precision-" if arguments.precision_check else "")
            proof.write(f"{prefix}error", {"error": str(error), "traceback": traceback.format_exc()})
            proof.write(f"{prefix}status", {"status": "failed", "pid": os.getpid(), "error": str(error)})
        raise
