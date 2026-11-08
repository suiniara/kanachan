#!/usr/bin/env python3

import re
import datetime
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Any
import sys
import json
import secrets
import os
from tqdm import tqdm
import torch
import yaml
from kanachan.model_loader import load_model
from kanachan.simulation.statistics import summarize
from kanachan.training.online.state import file_digest


def _parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Run Mahjong Soul game simulation.")
    parser.add_argument("--device", help="device", metavar="DEV")
    parser.add_argument(
        "--dtype",
        choices=("half", "float16", "float", "float32", "double", "float64"),
        help="data type (defaults to `float16`)",
        metavar="DTYPE",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        required=True,
        help="path to config file for baseline model",
        metavar="BASELINE_MODEL",
    )
    parser.add_argument(
        "--baseline-grade",
        type=int,
        required=True,
        help="grade of baseline model",
        metavar="BASELINE_GRADE",
    )
    parser.add_argument(
        "--proposed-model",
        type=Path,
        required=True,
        help="path to config file for proposed model",
        metavar="PROPOSED_MODEL",
    )
    parser.add_argument(
        "--proposed-grade",
        type=int,
        required=True,
        help="grade of proposed model",
        metavar="PROPOSED_GRADE",
    )
    parser.add_argument(
        "--room",
        choices=(
            "bronze",
            "silver",
            "gold",
            "jade",
            "throne",
        ),
        required=True,
        help="room for simulation",
    )
    parser.add_argument(
        "--dongfengzhan",
        action="store_true",
        help="simulate dong feng zhan (東風戦)",
    )
    parser.add_argument(
        "--mode",
        default="2vs2",
        choices=(
            "2vs2",
            "1vs3",
        ),
        help="simulation mode (defaults to `2vs2`)",
    )
    parser.add_argument(
        "--non-duplicated",
        action="store_true",
        help="disable duplicated simulation",
    )
    parser.add_argument(
        "-n",
        default=1,
        type=int,
        help="# of sets to simulate (defaults to `1`)",
        metavar="N",
    )
    parser.add_argument("--batch-size", default=1, type=int, metavar="N_BATCH")
    parser.add_argument("--concurrency", type=int, metavar="N_THREADS")
    parser.add_argument("--seed", type=int, help="uint64 seed for walls and seats")
    parser.add_argument("--output", type=Path, help="save games, seed, hashes and statistics as JSON")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--compile", action="store_true", help="enable TorchInductor (requires a working compiler)")

    return parser.parse_args()


def _main():
    from kanachan.simulation import simulate

    config = _parse_arguments()
    if config.torch_threads < 1:
        raise ValueError("--torch-threads must be positive.")
    torch.set_num_threads(config.torch_threads)
    from torch._dynamo import config as dynamo_config

    dynamo_config.disable = not config.compile or os.environ.get("TORCHDYNAMO_DISABLE") == "1"
    if config.seed is None:
        config.seed = secrets.randbits(63)
    if not 0 <= config.seed < 2**64:
        raise ValueError("--seed must be in [0, 2**64).")
    if config.bootstrap_samples < 100 or not 0 < config.confidence < 1:
        raise ValueError("Invalid bootstrap sample count or confidence level.")

    if config.device is not None:
        match = re.fullmatch(r"cpu|cuda(?::[0-9]+)?", config.device)
        if match is None:
            msg = f"{config.device}: invalid device"
            raise RuntimeError(msg)
        device = torch.device(config.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if config.dtype is None:
        if device.type == "cpu":
            config.dtype = "float32"
        else:
            config.dtype = "float16"
    if config.dtype in ("half", "float16"):
        dtype = torch.half
    elif config.dtype in ("float", "float32"):
        dtype = torch.float
    elif config.dtype in ("double", "float64"):
        dtype = torch.double
    else:
        msg = f"{config.dtype}: An invalid data type."
        raise ValueError(msg)

    if not config.baseline_model.exists():
        msg = f"{config.baseline_model}: Does not exist."
        raise RuntimeError(msg)
    if not config.baseline_model.is_file():
        msg = f"{config.baseline_model}: Not a file."
        raise RuntimeError(msg)
    baseline_model = load_model(
        config.baseline_model, map_location=torch.device("cpu")
    )
    baseline_model.to(device=device, dtype=dtype)
    baseline_model.requires_grad_(False)
    baseline_model.eval()

    if config.baseline_grade < 0 or 15 < config.baseline_grade:
        msg = (
            f"{config.baseline_grade}: An invalid value for"
            " `--baseline-grade`."
        )
        raise RuntimeError(msg)

    if not config.proposed_model.exists():
        msg = f"{config.proposed_model}: Does not exist."
        raise RuntimeError(msg)
    if not config.proposed_model.is_file():
        msg = f"{config.proposed_model}: Not a file."
        raise RuntimeError(msg)
    proposed_model = load_model(
        config.proposed_model, map_location=torch.device("cpu")
    )
    proposed_model.to(device=device, dtype=dtype)
    proposed_model.requires_grad_(False)
    proposed_model.eval()

    if config.proposed_grade < 0 or 15 < config.proposed_grade:
        msg = (
            f"{config.proposed_grade}: An invalid value for"
            " `--proposed-grade`."
        )
        raise RuntimeError(msg)

    room = {"bronze": 0, "silver": 1, "gold": 2, "jade": 3, "throne": 4}[
        config.room
    ]

    mode = 0
    if config.non_duplicated:
        mode |= 1
    if config.dongfengzhan:
        mode |= 2
    if config.mode == "1vs3":
        mode |= 4

    if config.n < 1:
        msg = f"{config.n}: An invalid value for the `-n` option."
        raise RuntimeError(msg)

    if config.batch_size < 1:
        msg = (
            f"{config.batch_size}: An invalid value for the"
            " `--batch-size` option."
        )
        raise RuntimeError(msg)

    if config.concurrency is None:
        config.concurrency = config.batch_size * 2
    if config.concurrency < 1:
        msg = (
            f"{config.concurrency}: An invalid value for the"
            " `--concurrency` option."
        )
        raise RuntimeError(msg)

    if mode & 1 != 0:
        num_total_games = config.n
    else:
        if mode & 4 != 0:
            num_total_games = config.n * 4
        else:
            num_total_games = config.n * 6

    start_time = datetime.datetime.now()
    tqdm_kwargs = {
        "desc": "Simulating games...",
        "total": num_total_games,
        "maxinterval": 0.1,
        "smoothing": 0.0,
    }
    with tqdm(**tqdm_kwargs) as progress, torch.inference_mode():
        game_logs = simulate(
            device,
            dtype,
            room,
            config.baseline_grade,
            baseline_model,
            config.proposed_grade,
            proposed_model,
            mode,
            config.n,
            config.batch_size,
            config.concurrency,
            lambda: progress.update(1),
            seed=config.seed,
        )
    game_results: list[list[dict[str, Any]]] = []
    metadata = []
    for game_log in game_logs:
        game_result = game_log.get_result()
        game_results.append(game_result)
        metadata.append(game_log.get_metadata())

    elapsed_time = datetime.datetime.now() - start_time
    if config.non_duplicated:
        print(f"Elapsed time: {elapsed_time} ({elapsed_time / config.n}/game)")
    elif config.mode == "2vs2":
        print(
            f"Elapsed time: {elapsed_time} ({elapsed_time / (config.n * 6.0)}/game)"
        )
    else:
        assert config.mode == "1vs3"
        print(
            f"Elapsed time: {elapsed_time} ({elapsed_time / (config.n * 4.0)}/game)"
        )

    statistics = summarize(game_results, group_ids=[item["set_id"] for item in metadata],
                           bootstrap_samples=config.bootstrap_samples,
                           seed=config.seed, confidence=config.confidence)
    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "format_version": 1, "seed": config.seed, "room": room,
            "dongfengzhan": config.dongfengzhan, "simulation_mode": mode,
            "sets": config.n, "device": str(device), "dtype": str(dtype),
            "batch_size": config.batch_size, "concurrency": config.concurrency,
            "baseline_model": str(config.baseline_model.resolve()),
            "baseline_sha256": file_digest(config.baseline_model),
            "proposed_model": str(config.proposed_model.resolve()),
            "proposed_sha256": file_digest(config.proposed_model),
            "baseline_grade": config.baseline_grade, "proposed_grade": config.proposed_grade,
            "metadata": metadata, "results": game_results, "statistics": statistics,
        }
        temporary = config.output.with_name(config.output.name + ".tmp")
        temporary.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(config.output)
    print(yaml.safe_dump({"seed": config.seed, **statistics}, sort_keys=False, explicit_start=True))


if __name__ == "__main__":
    _main()
    sys.exit(0)
