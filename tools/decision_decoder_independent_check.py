"""Read-only state comparison and additional cache precision verification."""

import json
import os
from pathlib import Path
import sys

os.environ.update(
    OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TORCHDYNAMO_DISABLE="1",
    CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1",
)
ROOT = Path(r"C:\Users\Suini\kanachan-data-work\diagnostics\decision-decoder-20261104-v1")
sys.path.insert(0, str(ROOT / "source"))

import torch

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def equal(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0, msg=path)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f"Different keys: {path}")
        for key in left:
            equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError(f"Different sequence: {path}")
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            equal(a, b, f"{path}.{i}")
    elif left != right:
        raise ValueError(f"Different scalar: {path}")


def main():
    proof = {"status": "running", "scope": "independent_decoder_resume_and_precision_check"}
    first, second = ROOT / "cql/snapshots", ROOT / "cql-resumed/snapshots"
    compared = []
    for name in (
        "optimizer.pth", "replay-state.pth", "decision-rng.pth",
        "source-encoder.pth", "source-decoder.pth",
        "target-encoder.pth", "target-decoder.pth",
    ):
        equal(
            torch.load(first / name, map_location="cpu", weights_only=True),
            torch.load(second / name, map_location="cpu", weights_only=True),
            name,
        )
        compared.append(name)
    left = json.loads((first / "training-state.json").read_text())
    right = json.loads((second / "training-state.json").read_text())
    equal(left, right, "training-state")
    proof["identical_resume_components"] = compared + ["training-state.json"]
    proof["successful_optimizer_updates"] = left["optimizer_steps"]
    from kanachan.model_loader import load_model
    from kanachan.nn.decision_decoder import (
        DenseDecisionDecoder, GlobalActionHead, DecisionContext,
    )
    from kanachan.training.core.bc import DataLoader
    torch.cuda.set_per_process_memory_fraction(0.45)
    actor = load_model(first / "model.kanachan", map_location=torch.device("cpu"))
    actor.cuda().eval().requires_grad_(False)
    body = next(m for m in actor.modules() if isinstance(m, DenseDecisionDecoder))
    head = next(m for m in actor.modules() if isinstance(m, GlobalActionHead))
    loader = DataLoader(
        path=ROOT.parents[1] / "experiments/bc-tiny-20261103-v2/data/validation.bc.tsv.gz",
        num_skip_samples=0, rewrite_rooms=None, rewrite_grades=None,
        batch_size=32, num_workers=0,
    )
    observations = next(iter(loader))
    # Choose the longest real history in this small held-out batch.
    index = int((observations["progression"] != 2165).sum(-1).argmax())
    features = [
        observations[key][index:index + 1].cuda().clone()
        for key in ("sparse", "numeric", "progression", "candidates")
    ]
    events = features[2][0][features[2][0] != 2165].clone()
    cases = []
    with torch.no_grad():
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            cache = None
            maximum = 0.0
            with torch.autocast(
                "cuda", dtype=dtype, enabled=dtype != torch.float32,
                cache_enabled=False,
            ):
                context = DecisionContext("independent", "held-out", 0, 0, str(dtype))
                for length in sorted({0, min(1, len(events)), min(3, len(events)), len(events)}):
                    features[2].fill_(2165)
                    features[2][0, :length] = events[:length]
                    expected = head(features[3], body(*features))
                    representation, cache = body.incremental(
                        *features, context=context, cache=cache
                    )
                    actual = head(features[3], representation)
                    tolerance = 1e-5 if dtype == torch.float32 else 1e-3
                    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0.01)
                    maximum = max(maximum, float((actual - expected).abs().max()))
            cases.append({
                "dtype": str(dtype), "history_events": len(events),
                "maximum_candidate_q_error": maximum,
                "full_vs_incremental_passed": True,
            })
    proof["cache_precision_cases"] = cases
    proof["status"] = "complete"
    output = ROOT / "independent-check.json"
    if output.exists():
        raise ValueError("Independent check receipt already exists.")
    output.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
