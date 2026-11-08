"""Validate attention semantics, CUDA dispatch and model-step performance.

CPU reference mode checks the Flex frontend without claiming fused execution.
CUDA mode uses the real-game fixture, BF16, actual decoder shapes and AdamW.
Data loading, checkpoint I/O and initial compilation are outside timed steps.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import warnings

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from kanachan.constants import (
    MAX_NUM_ACTIVE_SPARSE_FEATURES, MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES, NUM_TYPES_OF_SPARSE_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES, NUM_TYPES_OF_ACTIONS, NUM_NUMERIC_FEATURES,
)
from kanachan.nn.decision_attention import decision_attention, flex_attention
from kanachan.nn.decision_decoder import DenseDecisionDecoder, GlobalActionHead


ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "tiny": dict(dimension=128, num_heads=2, dim_feedforward=192, num_layers=2),
    "medium": dict(dimension=512, num_heads=8, dim_feedforward=1408, num_layers=8),
}


def real_observations(batch_size, device):
    path = ROOT / "fixtures/real-game.rl.tsv"
    identity = json.loads((ROOT / "fixtures/manifest.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != identity["sha256"]:
        raise ValueError("The accepted real-game fixture changed.")
    rows = [row.split("\t") for row in path.read_text().splitlines()]
    if batch_size > len(rows):
        raise ValueError("Batch size exceeds the complete real-game fixture.")
    result = []
    for column, width, padding in (
        (1, MAX_NUM_ACTIVE_SPARSE_FEATURES, NUM_TYPES_OF_SPARSE_FEATURES),
        (2, NUM_NUMERIC_FEATURES, 0),
        (3, MAX_LENGTH_OF_PROGRESSION_FEATURES, NUM_TYPES_OF_PROGRESSION_FEATURES),
        (4, MAX_NUM_ACTION_CANDIDATES, NUM_TYPES_OF_ACTIONS),
    ):
        tensor = torch.full((batch_size, width), padding, dtype=torch.int32, device=device)
        for i, row in enumerate(rows[:batch_size]):
            values = [int(value) for value in row[column].split(",") if value]
            if len(values) > width or (column == 2 and len(values) != width):
                raise ValueError("The real-game feature shape differs.")
            tensor[i, :len(values)] = torch.tensor(values, device=device, dtype=torch.int32)
        result.append(tensor)
    actions = torch.tensor([int(row[5]) for row in rows[:batch_size]], device=device)
    return tuple(result), actions, digest


def model(profile, backend, device):
    torch.manual_seed(20261108)
    body = DenseDecisionDecoder(
        **PROFILES[profile], attention_backend=backend,
        device=device, dtype=torch.float32,
    )
    head = GlobalActionHead(
        input_dimension=body.dimension, hidden_dimension=body.dimension,
        device=device, dtype=torch.float32,
    )
    return body, head


def reference_cpu():
    from torch._dynamo import config as dynamo_config

    features, _, digest = real_observations(8, torch.device("cpu"))
    body, _ = model("tiny", "sdpa", torch.device("cpu"))
    tokens, positions, mask, _ = body.tokens(*features)
    qkv = tuple(t.detach().requires_grad_(True)
                for t in body.blocks[0].project(tokens, positions))
    weight = torch.randn_like(qkv[0])
    with sdpa_kernel(SDPBackend.MATH):
        expected = decision_attention(*qkv, mask)
    expected_grad = torch.autograd.grad((expected * weight).sum(), qkv)
    with dynamo_config.patch(disable=False), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="flex_attention called without")
        actual = flex_attention(*qkv, mask)
        gradients = torch.autograd.grad((actual * weight).sum(), qkv)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for first, second in zip(expected_grad, gradients):
        torch.testing.assert_close(first, second, atol=3e-6, rtol=2e-4)
    return {"status": "cpu_reference_passed_cuda_pending", "fixture_sha256": digest,
            "qkv_shape": list(qkv[0].shape), "fused_cuda_verified": False,
            "max_output_absolute_error": float((actual - expected).abs().max()),
            "max_gradient_absolute_error": max(float((a - b).abs().max())
                                               for a, b in zip(gradients, expected_grad))}


def cuda_equivalence(profile, device):
    features, actions, _ = real_observations(8, device)
    body, head = model(profile, "sdpa", device)
    candidate, candidate_head = model(profile, "flex", device)
    candidate.load_state_dict(body.state_dict(), strict=True)
    candidate_head.load_state_dict(head.state_dict(), strict=True)
    with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(SDPBackend.MATH):
        expected = head(features[3], body(*features))
        loss = F.cross_entropy(expected.masked_fill(features[3] == NUM_TYPES_OF_ACTIONS, -torch.inf), actions)
    loss.backward()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = candidate_head(features[3], candidate(*features))
        actual_loss = F.cross_entropy(actual.masked_fill(features[3] == NUM_TYPES_OF_ACTIONS, -torch.inf), actions)
    actual_loss.backward()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.05)
    largest_error = 0.0
    for reference, result in ((body, candidate), (head, candidate_head)):
        for (name, a), (_, b) in zip(reference.named_parameters(), result.named_parameters()):
            if a.grad is None or b.grad is None:
                raise ValueError(f"Missing CUDA gradient: {name}")
            torch.testing.assert_close(a.grad, b.grad, atol=5e-4, rtol=0.05, msg=name)
            largest_error = max(largest_error, float((a.grad - b.grad).abs().max()))
    return {"max_output_absolute_error": float((actual - expected).abs().max()),
            "max_gradient_absolute_error": largest_error,
            "output_tolerance": {"atol": 0.02, "rtol": 0.05},
            "gradient_tolerance": {"atol": 0.0005, "rtol": 0.05}}


def benchmark(profile, backend, batch_size, steps, device):
    features, actions, digest = real_observations(batch_size, device)
    body, head = model(profile, "flex" if backend == "flex" else "sdpa", device)
    parameters = list(body.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, fused=True)

    def step():
        optimizer.zero_grad(set_to_none=True)
        context = sdpa_kernel(SDPBackend.MATH) if backend == "sdpa_math" else nullcontext()
        with context, torch.autocast("cuda", dtype=torch.bfloat16):
            values = head(features[3], body(*features))
            loss = F.cross_entropy(values.masked_fill(features[3] == NUM_TYPES_OF_ACTIONS, -torch.inf), actions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        return loss

    torch.cuda.synchronize()
    start = time.perf_counter()
    step()
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - start
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(steps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        loss = step()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite loss during the CUDA attention benchmark.")
    peak = torch.cuda.max_memory_allocated()
    kernels = []
    if backend == "flex":
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
            step()
            torch.cuda.synchronize()
        kernels = sorted({event.name for event in profiler.events()
                          if event.device_type == torch.autograd.DeviceType.CUDA
                          and "flex_attention" in event.name.lower()})
        if not kernels:
            raise ValueError("Profiler did not observe a CUDA FlexAttention kernel.")
    median = statistics.median(timings)
    result = {"profile": profile, "backend": backend, "batch_size": batch_size,
              "parameters": sum(value.numel() for value in parameters),
              "tokens": body.width, "dtype": "bfloat16_autocast_float32_parameters",
              "initial_step_seconds_including_compile": initialization_seconds,
              "measured_steps": steps, "median_step_ms": median,
              "samples_per_second": batch_size * 1000 / median,
              "peak_allocated_bytes": peak, "cuda_flex_kernels": kernels,
              "fixture_sha256": digest}
    del body, head, parameters, optimizer, features, actions, loss
    torch.cuda.empty_cache()
    return result


def verify(*, cpu_reference=False, profiles=("tiny", "medium"), batch_sizes=(128,), steps=12):
    torch.set_num_threads(2)
    result = {"scope": "attention_correctness_and_model_step_microbenchmark_only",
              "torch": torch.__version__, "checked_utc": datetime.now(timezone.utc).isoformat()}
    if cpu_reference:
        return dict(result, **reference_cpu())
    if not torch.cuda.is_available():
        raise RuntimeError("Mount the GPU before CUDA attention verification; CPU mode has no GPU evidence.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This profile requires BF16 support.")
    result.update(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  fused_cuda_verified=False, equivalence={}, benchmarks=[])
    for profile in profiles:
        result["equivalence"][profile] = cuda_equivalence(profile, torch.device("cuda"))
        for batch in batch_sizes:
            for backend in ("sdpa_math", "sdpa_auto", "flex"):
                result["benchmarks"].append(benchmark(profile, backend, batch, steps, torch.device("cuda")))
    result.update(status="complete", fused_cuda_verified=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[128])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.steps < 3 or any(batch <= 0 for batch in arguments.batch_sizes):
        parser.error("Use at least 3 steps and positive batch sizes.")
    receipt = verify(cpu_reference=arguments.cpu_reference, profiles=arguments.profiles,
                     batch_sizes=arguments.batch_sizes, steps=arguments.steps)
    text = json.dumps(receipt, indent=2) + "\n"
    if arguments.output:
        arguments.output.write_text(text)
    print(text)
