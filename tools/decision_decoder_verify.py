"""One bounded, isolated CUDA/native verification of the dense decoder.

No production source, task, dataset, checkpoint or old model is overwritten.
The original audited training subset stays remote. This is functional
validation; the resulting tiny pilot models do not establish Mahjong strength.
"""

import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import zipfile


ROOT = Path(r"C:\Users\Suini\kanachan-data-work")
OUTPUT = ROOT / "diagnostics/decision-decoder-20261104-v1"
SOURCE = OUTPUT / "source"
BASELINE = ROOT / "experiments/bc-tiny-20261103-v2"
RL_TRIAL = ROOT / "experiments/rl-tiny-20261103-v1"
ARCHIVE = ROOT / "tools/decision-decoder-candidate-v1.zip"
BC_SHA = "1c6e4def6e0d8398273edf920b9a13fea4095c71a7952a56ef21a9187c41d3d6"
OLD_PAO = "2835baae6b65fcc7d96ea1e584bc37c3d82c4a9c080fc703a8e3022d166d21ba"
NEW_PAO = "53b825e36446ef0dfd97534b9641ec69c64312b3b053f8b24b79824e6635af50"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def manifest(root):
    return {
        path.relative_to(root).as_posix(): digest(path)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def write(name, value):
    path = OUTPUT / f"{name}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def phase(name, **extra):
    value = {
        "status": "running", "phase": name, "pid": os.getpid(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra,
    }
    write("status", value)
    print(json.dumps(value), flush=True)


def setup_runtime():
    os.environ.update(
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TORCHDYNAMO_DISABLE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1",
        WORLD_SIZE="1", RANK="0", LOCAL_RANK="0", HYDRA_FULL_ERROR="1",
    )
    sys.path.insert(0, str(SOURCE))
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.45)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261104)
    torch.cuda.manual_seed_all(20261104)
    return torch


def command(arguments, label, timeout=360):
    start = time.monotonic()
    with (OUTPUT / f"{label}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(map(str, arguments)), stdout=log, stderr=subprocess.STDOUT,
            cwd=SOURCE, env=os.environ.copy(),
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            # The tracked venv launcher and its base-Python child are one owned tree.
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, check=False,
            )
            raise
    if code:
        raise RuntimeError(f"{label} exited {code}; inspect its remote log.")
    return {"exit_code": code, "elapsed_seconds": time.monotonic() - start}


def prepare(archive_sha):
    if OUTPUT.exists():
        raise RuntimeError("Diagnostic already exists; inspect it, never silently rerun.")
    if digest(ARCHIVE) != archive_sha or digest(BASELINE / "model.kanachan") != BC_SHA:
        raise RuntimeError("Candidate archive or preserved BC model identity differs.")
    expected = read(BASELINE / "source-manifest.json")["files"]
    if manifest(BASELINE / "source") != expected:
        raise RuntimeError("The preserved BC source manifest differs.")
    OUTPUT.mkdir(parents=True)
    phase("copy_and_validate_source")
    shutil.copytree(
        BASELINE / "source", SOURCE,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
    )
    if manifest(SOURCE) != expected:
        raise RuntimeError("Isolated source copy differs.")
    original_pao = SOURCE / "src/simulation/round_state.cpp"
    if not original_pao.is_file():
        native = ROOT / "smoke-20261030/source/src"
        shutil.copytree(native, SOURCE / "src", dirs_exist_ok=True)
    if digest(original_pao) != OLD_PAO:
        raise RuntimeError("Original native pao source differs.")
    with zipfile.ZipFile(ARCHIVE) as archive:
        contents = read_archive_manifest(archive)
        for name, identity in contents.items():
            destination = (SOURCE / name).resolve()
            if not destination.is_relative_to(SOURCE.resolve()):
                raise RuntimeError("Archive path escaped isolated source.")
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != identity:
                raise RuntimeError("Candidate member identity differs.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    if digest(original_pao) != NEW_PAO:
        raise RuntimeError("Candidate native pao fix differs.")
    write("source-manifest", {
        "archive_sha256": archive_sha, "candidate": contents,
        "preserved_bc_source": expected, "candidate_source": manifest(SOURCE),
    })
    phase("native_build")
    build = command([
        "powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-File", ROOT / "tools/decision-decoder-native-build-v1.ps1",
        "-SourceRoot", SOURCE, "-BuildRoot", OUTPUT / "native-build",
    ], "native-build", timeout=600)
    write("native-build", {
        **build, "pao_source_sha256": digest(original_pao),
        "extension_sha256": digest(SOURCE / "kanachan/simulation/_simulation.pyd"),
        "assertions_enabled": True,
    })


def read_archive_manifest(archive):
    expected = json.loads(archive.read("candidate-manifest.json"))["files"]
    if set(archive.namelist()) != {"candidate-manifest.json", *expected}:
        raise RuntimeError("Unexpected archive members.")
    for name in expected:
        if not (
            name.startswith("kanachan/") and name.endswith(".py")
            or name.startswith("test/training/") and name.endswith(".py")
            or name == "src/simulation/round_state.cpp"
            or name == "tools/runtime/CMakeLists.txt"
        ):
            raise RuntimeError("Unexpected candidate source path.")
    return expected


def collect_pilot_data():
    data_dir = OUTPUT / "data"
    data_dir.mkdir()
    bc_source = BASELINE / "data/train.bc.tsv.gz"
    rl_source = RL_TRIAL / "data/train.rl.tsv.gz"
    if not rl_source.is_file():
        raise RuntimeError("Locate the audited RL subset path before running the probe.")
    rl_receipt = read(RL_TRIAL / "subset-acceptance.json")
    if (
        rl_receipt["status"] != "complete"
        or digest(rl_source) != rl_receipt["files"]["train"]["sha256"]
        or digest(bc_source) != "da586fa252d07d766b6e64deac1508929f3fb24750053726e80d0a5bf7d3fd1d"
    ):
        raise RuntimeError("Preserved audited train subset identity differs.")
    rows, terminal, identities = 0, 0, set()
    rl_destination = data_dir / "train.rl.tsv"
    with gzip.open(rl_source, "rt", encoding="utf-8") as source:
        with rl_destination.open("w", encoding="utf-8") as destination:
            for line in source:
                fields = line.rstrip("\n").split("\t")
                if len(fields) not in (8, 10, 12):
                    raise RuntimeError("Unexpected audited RL record schema.")
                destination.write(line)
                rows += 1
                identities.add(fields[0])
                terminal += len(fields) == 8
                if terminal == 4:
                    break
                if rows > 50000:
                    raise RuntimeError("Whole-episode pilot bound exceeded.")
    if terminal != 4:
        raise RuntimeError("Pilot did not retain four complete observer episodes.")
    bc_rows = 0
    with gzip.open(bc_source, "rt", encoding="utf-8") as source:
        with (data_dir / "train.bc.tsv").open("w", encoding="utf-8") as destination:
            for line in source:
                destination.write(line)
                bc_rows += 1
                if bc_rows == 512:
                    break
    proof = {
        "scope": "functional_pilot_train_subset_only",
        "bc_source": str(bc_source), "rl_source": str(rl_source),
        "bc_rows": bc_rows, "rl_rows": rows, "terminal_observer_episodes": terminal,
        "source_record_identities": sorted(identities),
        "bc_sha256": digest(data_dir / "train.bc.tsv"),
        "rl_sha256": digest(rl_destination),
        "validation_used_for_optimizer_updates": False,
    }
    write("pilot-data", proof)
    return proof


def gpu_checks():
    torch = setup_runtime()
    from kanachan.nn.decision_decoder import (
        DenseDecisionDecoder, GlobalActionHead, DecisionContext,
    )
    from kanachan.training.core.bc import DataLoader
    from kanachan.model_loader import load_model
    loader = DataLoader(
        path=BASELINE / "data/validation.bc.tsv.gz", num_skip_samples=0,
        rewrite_rooms=None, rewrite_grades=None, batch_size=32, num_workers=0,
    )
    data = next(iter(loader)).to("cuda")
    inputs = tuple(data[key] for key in ("sparse", "numeric", "progression", "candidates"))
    args = dict(
        dimension=128, num_heads=2, dim_feedforward=192, num_layers=2,
        device=torch.device("cuda"), dtype=torch.float32,
    )
    model = DenseDecisionDecoder(**args)
    head = GlobalActionHead(
        input_dimension=128, hidden_dimension=128,
        device=torch.device("cuda"), dtype=torch.float32,
    )
    parameter_count = sum(p.numel() for module in (model, head) for p in module.parameters())
    if parameter_count != 829730:
        raise RuntimeError("Actual Tiny parameter budget differs.")
    reference = model(*inputs).detach()
    checks = []
    for dtype in (torch.float16, torch.bfloat16):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Target GPU lacks the planned bf16 validation support.")
        ordinary = copy.deepcopy(model).train()
        checkpointed = copy.deepcopy(model).train()
        checkpointed.checkpointing = True
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            a, b = ordinary(*inputs), checkpointed(*inputs)
            weights = torch.linspace(-1, 1, a.numel(), device="cuda").reshape_as(a)
            (a * weights).mean().backward()
            (b * weights).mean().backward()
        torch.testing.assert_close(a, reference, atol=0.02, rtol=0.02)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        nonzero = 0
        for left, right in zip(ordinary.parameters(), checkpointed.parameters(), strict=True):
            if left.grad is None or right.grad is None or not left.grad.isfinite().all():
                raise RuntimeError("Missing or nonfinite CUDA gradient.")
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
            nonzero += bool(left.grad.abs().max() > 0)
        if not nonzero:
            raise RuntimeError("No nonzero CUDA gradients.")
        checks.append({"dtype": str(dtype), "checkpoint_gradients_identical": True,
                       "nonzero_gradient_tensors": nonzero})
        del ordinary, checkpointed, a, b
    frozen = model.eval().requires_grad_(False)
    one = [value[:1].clone() for value in inputs]
    real_history = one[2][0][one[2][0] != 2165].clone()
    from kanachan.nn.decision_decoder import legal_argmax
    cache = None
    cache_errors = []
    context = DecisionContext("cuda-probe", "real-validation-observer", 0, 0, 1)
    for count in sorted({0, min(1, len(real_history)), len(real_history)}):
        one[2].fill_(2165)
        one[2][0, :count] = real_history[:count]
        full = frozen(*one)
        cached, cache = frozen.incremental(*one, context=context, cache=cache)
        torch.testing.assert_close(full, cached, atol=5e-5, rtol=1e-4)
        full_q, cached_q = head(one[3], full), head(one[3], cached)
        torch.testing.assert_close(full_q, cached_q, atol=5e-5, rtol=1e-4)
        if not torch.equal(legal_argmax(one[3], full_q), legal_argmax(one[3], cached_q)):
            raise RuntimeError("CUDA cache changed a non-tied legal action.")
        cache_errors.append(float((full_q - cached_q).abs().max()))
    one[1][0, 2] += 100
    cached, cache = frozen.incremental(*one, context=context, cache=cache)
    torch.testing.assert_close(frozen(*one), cached, atol=5e-5, rtol=1e-4)
    # Warm and time synchronized batch-one full and already-cached inference.
    latencies = {}
    for name, function in (
        ("full", lambda: head(one[3], frozen(*one))),
        ("cached", lambda: head(one[3], frozen.incremental(
            *one, context=context, cache=cache
        )[0])),
    ):
        with torch.no_grad():
            for _ in range(5):
                function()
            timings = []
            for _ in range(20):
                torch.cuda.synchronize()
                started = time.perf_counter()
                function()
                torch.cuda.synchronize()
                timings.append(1000 * (time.perf_counter() - started))
            latencies[name] = {"median_ms": statistics.median(timings), "repeats": len(timings)}
    old = load_model(BASELINE / "model.kanachan", map_location=torch.device("cpu"))
    old.to("cuda").eval().requires_grad_(False)(data)
    if not torch.isfinite(data["log_probs"][data["candidates"] < 546]).all():
        raise RuntimeError("The preserved BC model did not load and score real CUDA data.")
    write("cuda-checks", {
        "status": "complete", "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(), "parameters": parameter_count,
        "input_shapes": [list(value.shape) for value in inputs],
        "representation_shape": [32, 128], "candidate_output_shape": [32, 32],
        "precision": checks, "cache_max_q_errors": cache_errors,
        "batch_one_microbenchmark": latencies,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "old_bc_export_loads": True,
        "full_training_or_strength_benchmark": False,
    })


def cli(module, data, output, *extra):
    return command([
        sys.executable, ROOT / "tools/decision-decoder-verify-v1.py",
        "--child-module", f"kanachan.training.{module}.train", "--",
        "--config-name", "decision", "device=cuda",
        "device.dtype=float32", "device.amp_dtype=float16",
        "batch_size=32", "num_workers=0", "snapshot_interval=64",
        f"training_data={data.as_posix()}", f"hydra.run.dir={output.as_posix()}",
        *extra,
    ], output.name, timeout=420)


def check_training():
    import torch
    from kanachan.model_loader import load_model
    from kanachan.nn.decision_decoder import DenseDecisionDecoder
    data = OUTPUT / "data"
    proof = {"bc": cli("bc", data / "train.bc.tsv", OUTPUT / "bc")}
    proof["cql"] = cli(
        "cql", data / "train.rl.tsv", OUTPUT / "cql", "replay_buffer_size=128",
        f"encoder.load_from={(OUTPUT / 'bc/snapshots/encoder.pth').as_posix()}",
    )
    prefix = OUTPUT / "cql/snapshots"
    proof["cql_resume"] = cli(
        "cql", data / "train.rl.tsv", OUTPUT / "cql-resumed",
        "replay_buffer_size=128", f"initial_model_prefix={prefix.as_posix()}",
        "initial_model_index=64",
    )
    maximum = 0.0
    for name in (
        "source-encoder.pth", "source-decoder.pth",
        "target-encoder.pth", "target-decoder.pth",
    ):
        left = torch.load(prefix / name, weights_only=True, map_location="cpu")
        right = torch.load(OUTPUT / "cql-resumed/snapshots" / name, weights_only=True, map_location="cpu")
        for key in left:
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
            maximum = max(maximum, float((left[key] - right[key]).abs().max()))
    state = read(prefix / "training-state.json")
    if state["optimizer_steps"] <= 0:
        raise RuntimeError("No successful CUDA CQL optimizer updates.")
    exported = load_model(prefix / "model.kanachan", map_location=torch.device("cpu"))
    body = next(m for m in exported.modules() if isinstance(m, DenseDecisionDecoder))
    online = torch.load(prefix / "source-encoder.pth", weights_only=True, map_location="cpu")
    for key, value in body.state_dict().items():
        torch.testing.assert_close(value, online[key], atol=0, rtol=0)
    proof.update(
        status="complete", checkpoint=state, exported_network="online",
        restored_parameters_max_error=maximum,
        bc_model_sha256=digest(OUTPUT / "bc/snapshots/model.kanachan"),
        q_model_sha256=digest(prefix / "model.kanachan"),
    )
    write("training-checks", proof)


def native_check():
    torch = setup_runtime()
    from kanachan.model_loader import load_model
    from kanachan.simulation import simulate
    from kanachan.training.core.rl.simulation import simulate as collect
    from kanachan.training.core.rl.rewards import PlacementReward
    from kanachan.training.core.rl.discrete_cql import scalar_cql_loss
    from kanachan.training.online.train import SamplingPolicy
    from kanachan.nn import EpsilonGreedy
    def load(path):
        return load_model(path, map_location=torch.device("cpu")).to("cuda").float().eval().requires_grad_(False)
    baseline = load(BASELINE / "model.kanachan")
    model_path = OUTPUT / "cql/snapshots/model.kanachan"
    model = load(model_path)
    logs = simulate(
        torch.device("cuda"), torch.float32, 3, 12, baseline, 12, model,
        4, 2, 1, 1, lambda: None, seed=2026110401,
    )
    results = [log.get_result() for log in logs]
    if len(results) != 8:
        raise RuntimeError("Native evaluation did not complete all eight half-games.")
    exploration = EpsilonGreedy(
        epsilon_first=0.05, epsilon_last=0.05, annealing_steps=0,
        device=torch.device("cuda"), dtype=torch.float32,
    )
    exploration.stable_action_ids = True
    exploration.train()
    episodes = collect(
        device=torch.device("cuda"), dtype=torch.float32, room=3, dongfengzhan=False,
        grade=12, baseline_model=baseline, policy_model=SamplingPolicy(model, exploration),
        num_simulations=1, simulation_batch_size=1, simulation_concurrency=1,
        progress=lambda: None, get_reward=PlacementReward(), seed=2026110402,
        policy_version=0, opponent_version=0,
    )
    if not len(episodes):
        raise RuntimeError("No real online transitions.")
    from kanachan.training.core.rl.multistep import make_multistep
    make_multistep(episodes, 1.0, 1)
    model.requires_grad_(True).train()
    target = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    metrics = []
    for step in range(4):
        sample = episodes[(torch.arange(32) + 32 * step) % len(episodes)].to("cuda")
        optimizer.zero_grad(set_to_none=True)
        loss, record = scalar_cql_loss(
            model, target, sample, 0.05, td_loss="huber"
        )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite real online gradient.")
        optimizer.step()
        with torch.no_grad():
            for source, destination in zip(model.parameters(), target.parameters(), strict=True):
                destination.lerp_(source, 0.005)
        metrics.append({"loss": float(loss), "gradient_norm": float(norm), **record})
    write("native-checks", {
        "status": "complete", "evaluation_half_games": len(results),
        "evaluation_seed_groups": 2, "results": results,
        "online_decisions": len(episodes), "online_successful_updates": len(metrics),
        "online_metrics": metrics,
        "scope": "functional_native_and_online_smoke_only",
        "strength_improvement_established": False,
    })


def main(archive_sha):
    start = time.monotonic()
    prepare(archive_sha)
    phase("cuda_forward_gradient_cache")
    gpu_checks()
    phase("pilot_data")
    collect_pilot_data()
    phase("bc_cql_and_cuda_resume")
    check_training()
    phase("native_half_games_and_online_updates")
    command([
        sys.executable, ROOT / "tools/decision-decoder-verify-v1.py",
        "--native-check",
    ], "native-check", timeout=480)
    if manifest(BASELINE / "source") != read(BASELINE / "source-manifest.json")["files"]:
        raise RuntimeError("Preserved BC source changed during verification.")
    if digest(BASELINE / "model.kanachan") != BC_SHA:
        raise RuntimeError("Preserved BC model changed during verification.")
    acceptance = {
        "status": "complete", "scope": "dense_decoder_functional_verification_only",
        "source": str(SOURCE), "archive_sha256": archive_sha,
        "cuda": read(OUTPUT / "cuda-checks.json"),
        "training": read(OUTPUT / "training-checks.json"),
        "native": read(OUTPUT / "native-checks.json"),
        "native_build": read(OUTPUT / "native-build.json"),
        "elapsed_seconds": time.monotonic() - start,
        "old_bc_source_and_model_unchanged": True,
        "production_deployed": False, "formal_gpu_training_started": False,
        "formal_full_dataset_acceptance": False, "strength_improvement_established": False,
    }
    write("acceptance", acceptance)
    write("status", {"status": "complete", "phase": "complete", "pid": os.getpid()})
    print(json.dumps({"status": "complete", "receipt": str(OUTPUT / "acceptance.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-sha256")
    parser.add_argument("--child-module")
    parser.add_argument("--native-check", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.child_module:
        setup_runtime()
        sys.argv = [arguments.child_module, *(
            arguments.extra[1:] if arguments.extra[:1] == ["--"] else arguments.extra
        )]
        runpy.run_module(arguments.child_module, run_name="__main__")
    elif arguments.native_check:
        native_check()
    else:
        try:
            main(arguments.archive_sha256)
        except BaseException as error:
            if OUTPUT.exists():
                write("error", {"error": str(error), "traceback": traceback.format_exc()})
                write("status", {"status": "failed", "pid": os.getpid(), "error": str(error)})
            raise
