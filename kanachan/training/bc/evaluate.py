"""Evaluate a saved BC policy on a separate v2 annotation file."""

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path

import torch
from tensordict import TensorDict
from torch import nn

from kanachan.constants import NUM_TYPES_OF_ACTIONS
from kanachan.model_loader import load_model
from kanachan.nn.decode_converter import DecodeConverter
from kanachan.training.core.bc import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: Iterable[TensorDict],
    device: torch.device,
    max_samples: int = 0,
) -> dict[str, int | float]:
    """Measure NLL and action agreement without exposing labels to the model."""
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative.")
    was_training = model.training
    model.eval()
    num_samples = 0
    total_loss = 0.0
    num_correct = 0
    random_accuracy = 0.0
    try:
        for batch in data_loader:
            if max_samples:
                batch = batch[: max_samples - num_samples]
            targets = batch["action"].to(device=device, dtype=torch.int64)
            features = (
                batch.select("sparse", "numeric", "progression", "candidates")
                .clone()
                .to(device=device)
            )
            model(features)
            if "log_probs" not in features and "decode" in features:
                features["log_probs"] = DecodeConverter("log_probs")(
                    features["candidates"], features["decode"]
                )
            if "log_probs" not in features:
                raise RuntimeError("Expected a BC model producing log_probs.")
            log_probs = features["log_probs"]
            loss = nn.functional.nll_loss(log_probs, targets, reduction="sum")
            if not math.isfinite(loss.item()):
                raise RuntimeError("Validation loss is not finite.")
            total_loss += loss.item()
            num_correct += int((log_probs.argmax(1) == targets).sum().item())
            valid_actions = (
                features["candidates"] < NUM_TYPES_OF_ACTIONS
            ).sum(1)
            random_accuracy += float(
                valid_actions.double().reciprocal().sum().item()
            )
            num_samples += int(targets.numel())
            if max_samples and num_samples >= max_samples:
                break
    finally:
        model.train(was_training)
    if num_samples == 0:
        raise RuntimeError("The validation data contains no samples.")
    return {
        "num_samples": num_samples,
        "negative_log_likelihood": total_loss / num_samples,
        "action_accuracy": num_correct / num_samples,
        "random_action_accuracy": random_accuracy / num_samples,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_samples < 0:
        parser.error("Invalid batch size, worker count or sample limit.")
    device = torch.device(args.device)
    model = load_model(args.model, map_location=torch.device("cpu"))
    model.to(device=device, dtype=torch.float32)
    model.requires_grad_(False)
    data_loader = DataLoader(
        path=args.data,
        num_skip_samples=0,
        rewrite_rooms=None,
        rewrite_grades=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(evaluate(model, data_loader, device, args.max_samples)))


if __name__ == "__main__":
    _main()
