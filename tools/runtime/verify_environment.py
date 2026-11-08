"""Check a saved training image using a complete real game and native games."""

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("Checkpoint keys differ.")
        for key in left:
            compare(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError("Checkpoint lengths differ.")
        for first, second in zip(left, right):
            compare(first, second)
    elif left != right:
        raise ValueError("Checkpoint values differ.")


def train(module, data, output, device, extra):
    command = [
        sys.executable, "-B", "-m", f"kanachan.training.{module}.train",
        "--config-name", "decision", f"device={device.type}",
        "device.dtype=float32",
        f"device.amp_dtype={'bfloat16' if device.type == 'cuda' else 'float32'}",
        "encoder.dimension=128", "encoder.num_heads=2",
        "encoder.dim_feedforward=192", "encoder.num_layers=2",
        "num_workers=0", "batch_size=32", "snapshot_interval=128",
        f"training_data={data}", f"hydra.run.dir={output}", *extra,
    ]
    result = subprocess.run(
        command, cwd=ROOT, env=dict(os.environ, TORCHDYNAMO_DISABLE="1"),
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode:
        raise RuntimeError(f"{module} entry failed.\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}")


def verify(output, require_cuda):
    import torch
    from kanachan.model_loader import load_model
    from kanachan.nn.decision_decoder import DenseDecisionDecoder, GlobalActionHead
    from kanachan.training.core.bc import DataLoader
    from kanachan.simulation import simulate

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261108)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    receipt = {
        "scope": "training_image_environment_functional_only",
        "status": "checking", "device": str(device),
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "cuda_runtime": torch.version.cuda,
        "compiled_cuda_architectures": torch._C._cuda_getArchFlags(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "tensordict", "torchrl", "numpy", "hydra-core",
        )},
        "cuda_verified": False,
    }
    if require_cuda and device.type != "cuda":
        raise RuntimeError("Mount a GPU before running --require-cuda.")
    if device.type == "cuda":
        receipt["gpu"] = torch.cuda.get_device_name(device)
        receipt["bf16_supported"] = torch.cuda.is_bf16_supported()
        if not receipt["bf16_supported"]:
            raise RuntimeError("This training profile requires CUDA BF16.")

    fixture = ROOT / "fixtures/real-game.rl.tsv"
    identity = json.loads((ROOT / "fixtures/manifest.json").read_text())
    if digest(fixture) != identity["sha256"]:
        raise ValueError("Real-game fixture checksum differs.")
    lines = [line.rstrip("\n").split("\t") for line in fixture.read_text().splitlines()]
    if len(lines) != identity["rows"] or sum(len(row) == 8 for row in lines) != 4:
        raise ValueError("Real-game fixture is incomplete.")

    with tempfile.TemporaryDirectory(prefix="kanachan-check-") as directory:
        work = Path(directory)
        raw_record = ROOT / "fixtures" / identity["uuid"]
        if digest(raw_record) != "f5e06e3c65e540e5e804914379435e11004593f3f8de3a06fb417c0a981ad3ab":
            raise ValueError("Raw-game fixture checksum differs.")
        records = work / "records.txt"
        records.write_text(str(raw_record) + "\n")
        annotated = subprocess.run(
            [str(ROOT / "bin/annotate"), "--files-from", str(records)],
            capture_output=True, text=True, timeout=120, check=True,
        )
        expected_decisions = Counter(tuple(row[:6]) for row in lines)
        actual_decisions = Counter(
            tuple(line.split("\t")[:6]) for line in annotated.stdout.splitlines()
        )
        if actual_decisions != expected_decisions:
            raise ValueError("Native annotation differs from the accepted real-game decisions.")
        receipt["native_annotation_matches_real_game"] = True
        # RL boundary records contain the matching round metadata. Propagate it
        # backward within each complete observer trajectory for the BC view.
        metadata = None
        final_scores = None
        bc_lines = []
        for row in reversed(lines):
            if len(row) == 8:
                final_scores = row[-1].split(",")[-4:]
                metadata = row[-2:]
            elif len(row) == 12:
                if final_scores is None:
                    raise ValueError("Observer trajectory has no terminal scores.")
                metadata = [row[-2], ",".join(row[-1].split(",") + final_scores)]
            if metadata is None:
                raise ValueError("Observer trajectory has no final result.")
            bc_lines.append("\t".join(row[:6] + metadata) + "\n")
        bc_data = work / "real-game.bc.tsv"
        bc_data.write_text("".join(reversed(bc_lines)))

        for module, data, extra in (
            ("bc", bc_data, []),
            ("cql", fixture, ["replay_buffer_size=1024",
                              f"bc_initial_model={work / 'bc/snapshots/model.kanachan'}"]),
        ):
            target = work / module
            train(module, data, target, device, extra)
            resumed = work / f"{module}-resumed"
            train(module, data, resumed, device, [
                *extra, f"initial_model_prefix={target / 'snapshots'}",
                "initial_model_index=128",
            ])
            names = ["encoder.pth", "decoder.pth", "optimizer.pth"]
            if module == "cql":
                names.append("replay-state.pth")
            for name in names:
                compare(
                    torch.load(target / "snapshots" / name, map_location="cpu", weights_only=True),
                    torch.load(resumed / "snapshots" / name, map_location="cpu", weights_only=True),
                )
            receipt[f"{module}_real_game_exact_resume"] = True

        batch = next(DataLoader(
            path=bc_data, num_skip_samples=0, rewrite_rooms=None,
            rewrite_grades=None, batch_size=8, num_workers=0,
        )).to(device)
        models = {
            "decoder-bc.kanachan": "cb033b04f88eafd6bdf4cd67e8efe6f8749ababe7ee98edfb61577799e73d334",
            "decoder-mc-cql.kanachan": "3360bced2d59335b8fb33241197b03e0221bed7ce40c894748493cc1675af0e1",
            "opponent-encoder-bc.kanachan": "1c6e4def6e0d8398273edf920b9a13fea4095c71a7952a56ef21a9187c41d3d6",
        }
        for name, expected in models.items():
            path = ROOT / "fixtures" / name
            if digest(path) != expected:
                raise ValueError(f"Retained model checksum differs: {name}")
            actor = load_model(path, map_location=device).float().eval()
            with torch.inference_mode():
                result = actor(batch.clone())
            indices = result["action"].long()
            selected = batch["candidates"].gather(1, indices[:, None])
            if not (selected < 546).all():
                raise ValueError("An exported model selected padding.")
        receipt["retained_models_loaded_and_legal"] = list(models)

        body = DenseDecisionDecoder(
            dimension=512, num_heads=8, dim_feedforward=1408,
            num_layers=8, numeric_dimension=256,
            device=device, dtype=torch.float32,
        )
        head = GlobalActionHead(
            input_dimension=512, hidden_dimension=512, device=device, dtype=torch.float32,
        )
        parameters = list(body.parameters()) + list(head.parameters())
        receipt["medium_parameters"] = sum(value.numel() for value in parameters)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with context:
            representation = body(*(batch[key] for key in ("sparse", "numeric", "progression", "candidates")))
            values = head(batch["candidates"], representation)
            loss = values.square().mean()
        loss.backward()
        if not torch.isfinite(loss) or any(value.grad is None or not torch.isfinite(value.grad).all() for value in parameters):
            raise ValueError("The medium model produced invalid gradients.")
        receipt["medium_forward_and_gradients"] = True
        del body, head, parameters

        baseline = load_model(ROOT / "fixtures/opponent-encoder-bc.kanachan", map_location=device).float().eval()
        candidate = load_model(ROOT / "fixtures/decoder-mc-cql.kanachan", map_location=device).float().eval()
        started = time.monotonic()
        with torch.inference_mode():
            games = simulate(
                device, torch.float32, 3, 12, baseline, 12, candidate,
                4, 1, 1, 1, lambda: None, seed=2026110801,
            )
        if len(games) != 4:
            raise ValueError("Native simulator did not complete four games.")
        results = [game.get_result() for game in games]
        for game in results:
            if sorted(player["ranking"] for player in game) != [0, 1, 2, 3]:
                raise ValueError("Native result rankings differ from four seats.")
        receipt["native_games"] = 4
        receipt["native_seconds"] = time.monotonic() - started
    receipt["cuda_verified"] = device.type == "cuda"
    if require_cuda:
        attention_output = output.with_name("attention-acceptance.json")
        subprocess.run([
            sys.executable, "-B", str(ROOT / "tools/runtime/verify_attention.py"),
            "--output", str(attention_output),
        ], cwd=ROOT, check=True, timeout=900)
        receipt["attention"] = json.loads(attention_output.read_text())
    receipt["status"] = "complete" if receipt["cuda_verified"] else "cpu_ready_gpu_verification_pending"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    output.with_name("environment-error.json").unlink(missing_ok=True)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "environment-acceptance.json")
    arguments = parser.parse_args()
    try:
        verify(arguments.output, arguments.require_cuda)
    except Exception:
        arguments.output.with_name("environment-error.json").write_text(traceback.format_exc())
        raise
