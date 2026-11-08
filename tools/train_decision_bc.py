"""Bounded single-epoch decoder BC run, with flat artifacts and exact cursors.

The JSON plan binds existing source, dataset and model identities. A template
supplies constructors only: all weights are initialized afresh. No experiment
directories are created, and retained input models are never overwritten.
"""

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
import traceback


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fresh_descriptor(template):
    result = copy.deepcopy(template)

    def visit(value):
        if isinstance(value, dict):
            if "__kanachan__" in value:
                value.pop("state_dict", None)
                arguments = value.get("kwargs", {})
                if "device" in arguments:
                    import torch
                    arguments["device"] = torch.device("cpu")
                if "dtype" in arguments:
                    import torch
                    arguments["dtype"] = torch.float32
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
    visit(result)
    return result


def run(plan_path, resume=False):
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    plan_sha = sha256(plan_path)
    source, prefix = Path(plan["source"]), Path(plan["output_prefix"])
    if not source.is_dir() or not prefix.parent.is_dir():
        raise ValueError("Source and output parent directories must already exist.")
    if sha256(__file__) != plan["controller_sha256"]:
        raise ValueError("BC controller identity differs from the plan.")
    if sha256(plan["manifest"]) != plan["manifest_sha256"]:
        raise ValueError("BC source manifest identity differs.")
    expected = json.loads(Path(plan["manifest"]).read_text(encoding="utf-8-sig"))["complete_candidate"]

    def verify_source():
        for name, checksum in expected.items():
            if sha256(source / name) != checksum:
                raise ValueError(f"BC source identity differs: {name}")
    verify_source()
    for split, identity in plan["dataset"].items():
        if sha256(identity["path"]) != identity["sha256"]:
            raise ValueError(f"BC data identity differs: {split}")
    for name in ("template", "baseline"):
        if sha256(plan[name]["path"]) != plan[name]["sha256"]:
            raise ValueError(f"Retained {name} model identity differs.")

    os.environ.update(
        TORCHDYNAMO_DISABLE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1",
    )
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from kanachan.model_loader import load_model, load_model_state
    from kanachan.nn.decision_decoder import DenseDecisionDecoder
    from kanachan.training.bc import train as bc
    from kanachan.training.bc.evaluate import evaluate
    from kanachan.training.core.bc import DataLoader
    from kanachan.training.core.config.optimizer import AdamWOptimizerConfig, create
    from kanachan.training.online.state import (
        atomic_save, capture_rng, cpu_state, export_model, restore_rng,
    )

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(plan.get("device", "cuda:0"))
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.45)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    torch.manual_seed(plan["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(plan["seed"])
    amp = torch.float16 if device.type == "cuda" else torch.float32
    artifact = lambda suffix: prefix.with_name(prefix.name + suffix)
    status_path, checkpoint_path = artifact("-status.json"), artifact("-checkpoint.pt")
    model_path = artifact(".kanachan")
    owned = [status_path, checkpoint_path, model_path, artifact("-summary.json"),
             artifact("-metrics.jsonl"), artifact("-native.json"), artifact("-error.json")]
    if not resume and any(path.exists() for path in owned):
        raise ValueError("BC output already exists; inspect it before an explicit resume.")
    # Keep this file handle open throughout the run to reject a second launch.
    lock_path = artifact(".lock")
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    state = {
        "scope": "small_decoder_bc_trial_only", "status": "running",
        "pid": os.getpid(), "plan_sha256": plan_sha, "started_utc": datetime.now(timezone.utc).isoformat(),
        "training_seconds": 0., "validation_seconds": 0., "cursor": 0,
        "optimizer_steps": 0, "skipped_updates": 0, "validation_curve": [],
        "selected_validation": None, "selected_cursor": None, "last_training_metrics": {},
    }
    wall_start = time.monotonic()
    train_start = None
    prior_training_seconds = 0.
    validation_during_training = 0.
    try:
        template = torch.load(plan["template"]["path"], map_location="cpu", weights_only=True)
        actor = load_model_state(fresh_descriptor(template), map_location=torch.device("cpu")).float().to(device)
        if not any(isinstance(module, DenseDecisionDecoder) for module in actor.modules()):
            raise ValueError("BC template is not the dense decision decoder.")
        if sum(parameter.numel() for parameter in actor.parameters()) != plan["parameters"]:
            raise ValueError("BC parameter budget differs from the plan.")
        network = actor.select_subsequence(out_keys=["log_probs"]).train().requires_grad_(True)
        optimizer, scheduler = create("cuda" if device.type == "cuda" else "cpu", AdamWOptimizerConfig(
            learning_rate=plan["learning_rate"], warmup_steps=plan["warmup_steps"],
            weight_decay=plan["weight_decay"],
        ), network)
        saved = None
        if resume:
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if saved["plan_sha256"] != plan_sha or not saved["cursor_is_exact"]:
                raise ValueError("BC checkpoint identity or exact cursor differs.")
            state = saved["run_state"]
            state.update(status="running", pid=os.getpid())
            network.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(saved["scheduler"])
            prior_training_seconds = state["training_seconds"]

        def training_seconds():
            if train_start is None:
                return prior_training_seconds
            return prior_training_seconds + time.monotonic() - train_start - validation_during_training

        def status(phase, **values):
            state.update(phase=phase, **values, updated_utc=datetime.now(timezone.utc).isoformat())
            state["training_seconds"] = training_seconds()
            state["wall_seconds_this_process"] = time.monotonic() - wall_start
            if device.type == "cuda":
                state["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                state["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            write_json(status_path, state)

        def validate(cursor):
            nonlocal validation_during_training
            rng = capture_rng()
            started = time.monotonic()
            status("validation", cursor=cursor)
            try:
                data = DataLoader(
                    path=Path(plan["dataset"]["validation"]["path"]), num_skip_samples=0,
                    rewrite_rooms=None, rewrite_grades=None, batch_size=128, num_workers=0,
                )
                result = evaluate(network, data, device)
                if result["num_samples"] != plan["dataset"]["validation"]["rows"]:
                    raise ValueError("BC validation sample count differs.")
            finally:
                restore_rng(rng)
                elapsed = time.monotonic() - started
                state["validation_seconds"] += elapsed
                if train_start is not None:
                    validation_during_training += elapsed
            point = dict(result, cursor=cursor)
            state["validation_curve"].append(point)
            best = state["selected_validation"]
            if best is None or result["negative_log_likelihood"] < best["negative_log_likelihood"]:
                atomic_save(model_path, export_model(template, actor))
                state.update(selected_validation=point, selected_cursor=cursor)
            print(json.dumps({"event": "validation", **point}), flush=True)
            status("training" if train_start is not None else "initialized")

        class Metrics:
            def __init__(self):
                self.last_status = time.monotonic()
                self.next_log = (state["cursor"] // 4096 + 1) * 4096
                self.loss_sum = self.loss_weight = self.last_loss_cursor = 0

            def add_scalar(self, tag, value, step):
                state["cursor"] = step
                if tag == "Optimizer steps (this run)":
                    state["optimizer_steps"] = (0 if saved is None else saved["optimizer_steps"]) + int(value)
                elif tag == "Loss":
                    weight = min(plan["batch_size"], step - self.last_loss_cursor)
                    self.last_loss_cursor = step
                    self.loss_sum += float(value) * weight
                    self.loss_weight += weight
                state["last_training_metrics"][tag] = float(value)
                if step >= self.next_log and self.loss_weight:
                    point = {
                        "cursor": step, "optimizer_steps": state["optimizer_steps"],
                        "training_seconds": training_seconds(),
                        "mean_reported_batch_nll": self.loss_sum / self.loss_weight,
                        "skipped_updates": state["skipped_updates"],
                    }
                    with artifact("-metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(point, allow_nan=False) + "\n")
                    self.loss_sum = self.loss_weight = 0
                    self.next_log = (step // 4096 + 1) * 4096
                if time.monotonic() - self.last_status >= 30:
                    status("training")
                    self.last_status = time.monotonic()

        class Skipped(logging.Handler):
            def emit(self, record):
                if "non-finite values in the gradient" in record.getMessage():
                    state["skipped_updates"] += 1

        logging.basicConfig(level=logging.WARNING)
        skip_handler = Skipped()
        logging.getLogger().addHandler(skip_handler)
        metrics = Metrics()
        next_validation = (state["cursor"] // plan["validation_interval"] + 1) * plan["validation_interval"]

        def snapshot(index, **metadata):
            nonlocal next_validation
            cursor = index if index is not None else metadata["resume_num_samples"]
            state.update(cursor=cursor, optimizer_steps=metadata["optimizer_steps"])
            if cursor >= next_validation or index is None:
                if not state["validation_curve"] or state["validation_curve"][-1]["cursor"] != cursor:
                    validate(cursor)
                next_validation = (cursor // plan["validation_interval"] + 1) * plan["validation_interval"]
            state["training_seconds"] = training_seconds()
            atomic_save(checkpoint_path, {
                "format_version": 1, "plan_sha256": plan_sha,
                "cursor": cursor, "cursor_is_exact": metadata.get("cursor_is_exact", True),
                "optimizer_steps": metadata["optimizer_steps"],
                "scaler": metadata["grad_scaler_state"], "rng": capture_rng(),
                "model": cpu_state(network.state_dict()), "optimizer": cpu_state(optimizer.state_dict()),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "run_state": copy.deepcopy(state),
            })

        if saved is None:
            validate(0)
        original_loader = bc.DataLoader

        class BoundedLoader:
            def __init__(self, **arguments):
                self.inner = original_loader(**arguments)

            def __iter__(self):
                while training_seconds() < plan["training_budget_seconds"]:
                    try:
                        yield next(self.inner)
                    except StopIteration:
                        return

        train_start = time.monotonic()
        status("training")
        bc.DataLoader = BoundedLoader
        try:
            bc._training(
                device=device, dtype=torch.float32, amp_dtype=amp,
                training_data=Path(plan["dataset"]["train"]["path"]), num_workers=0,
                rewrite_rooms=None, rewrite_grades=None,
                num_samples=0 if saved is None else saved["cursor"],
                network_tdm=network, batch_size=plan["batch_size"],
                gradient_accumulation_steps=1, max_gradient_norm=1.,
                optimizer=optimizer, scheduler=scheduler,
                snapshot_interval=plan["checkpoint_interval"], summary_writer=metrics,
                snapshot_writer=snapshot,
                initial_optimizer_steps=0 if saved is None else saved["optimizer_steps"],
                initial_grad_scaler_state=None if saved is None else saved["scaler"],
                initial_rng_state=capture_rng() if saved is None else saved["rng"],
            )
        finally:
            bc.DataLoader = original_loader
            prior_training_seconds = training_seconds()
            train_start = None
            logging.getLogger().removeHandler(skip_handler)
        if state["cursor"] > plan["dataset"]["train"]["rows"]:
            raise ValueError("Training cursor exceeds the bound dataset.")
        if state["optimizer_steps"] <= 0:
            raise ValueError("BC did not perform any successful optimization updates.")
        verify_source()
        del optimizer, scheduler, network, actor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        selected = load_model(model_path, map_location=torch.device("cpu")).float().to(device).eval()
        selected.requires_grad_(False)
        status("test")
        test = evaluate(selected, DataLoader(
            path=Path(plan["dataset"]["test"]["path"]), num_skip_samples=0,
            rewrite_rooms=None, rewrite_grades=None, batch_size=128, num_workers=0,
        ), device)
        if test["num_samples"] != plan["dataset"]["test"]["rows"]:
            raise ValueError("BC test sample count differs.")
        state["test"] = test
        if plan["native_sets"] > 0:
            from kanachan.simulation import simulate
            from kanachan.simulation.statistics import summarize
            baseline = load_model(plan["baseline"]["path"], map_location=torch.device("cpu")).float().to(device).eval()
            baseline.requires_grad_(False)
            completed = 0

            def progress():
                nonlocal completed
                completed += 1
                if completed % 4 == 0:
                    status("native", completed_games=completed)
            status("native", completed_games=0)
            with torch.inference_mode():
                games = simulate(
                    device, torch.float32, 3, 12, baseline, 12, selected,
                    4, plan["native_sets"], 1, 1, progress, seed=plan["native_seed"],
                )
            results = [game.get_result() for game in games]
            metadata = [game.get_metadata() for game in games]
            if len(results) != plan["native_sets"] * 4:
                raise ValueError("Native seat-rotation count differs.")
            stats = summarize(
                results, group_ids=[entry["set_id"] for entry in metadata],
                bootstrap_samples=2000, seed=plan["native_seed"], confidence=.95,
            )
            native = {
                "seed": plan["native_seed"], "sets": plan["native_sets"],
                "games": len(results), "results": results, "metadata": metadata,
                "baseline_sha256": sha256(plan["baseline"]["path"]),
                "proposed_sha256": sha256(model_path), "statistics": stats,
            }
            write_json(artifact("-native.json"), native)
            state["native"] = {key: value for key, value in native.items() if key not in ("results", "metadata")}
        verify_source()
        if sha256(plan["baseline"]["path"]) != plan["baseline"]["sha256"]:
            raise ValueError("The original BC model changed during training.")
        state.update(
            status="complete", model=str(model_path), model_sha256=sha256(model_path),
            checkpoint_sha256=sha256(checkpoint_path), parameters=plan["parameters"],
            completed_epoch=state["cursor"] == plan["dataset"]["train"]["rows"],
            dataset=plan["dataset"], recipe={key: plan[key] for key in (
                "seed", "batch_size", "learning_rate", "warmup_steps", "weight_decay",
            )},
            optimizer="AdamW matrix-weight decay", amp_dtype=str(amp),
            source_manifest_sha256=plan["manifest_sha256"],
            old_bc_metrics=plan.get("old_bc_metrics"),
            selected_learned_checkpoint=state["selected_cursor"] > 0,
            limitation="Small subset and finite model opponents do not establish real playing rank.",
        )
        status("complete")
        write_json(artifact("-summary.json"), state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json(artifact("-error.json"), state)
        write_json(status_path, state)
        raise
    finally:
        lock_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    run(arguments.plan, arguments.resume)
