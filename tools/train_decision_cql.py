"""Bounded decoder CQL trial in existing directories, with paired BC control."""

import argparse
import copy
from datetime import datetime, timezone
import gc
import io
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

from train_decision_bc import sha256, write_json, fresh_descriptor


def equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("Checkpoint dictionary keys differ.")
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError("Checkpoint sequence differs.")
        for first, second in zip(left, right, strict=True):
            equal(first, second)
    elif left != right:
        raise ValueError(f"Checkpoint scalar differs: {left} != {right}")


def training_counters(epoch, cursor, rows, batch_size):
    """Account for the short final batch separately in every completed epoch."""
    if (any(type(value) is not int for value in (epoch, cursor, rows, batch_size))
            or epoch < 0 or rows < 1 or batch_size < 1 or not 0 <= cursor <= rows):
        raise ValueError("Invalid CQL epoch or sample counters.")
    batches_per_epoch = (rows + batch_size - 1) // batch_size
    return dict(
        epoch=epoch, cursor=cursor, total_samples=epoch * rows + cursor,
        completed_epochs=epoch + int(cursor == rows),
        attempted_updates=epoch * batches_per_epoch + (cursor + batch_size - 1) // batch_size,
    )


def evaluation_only_state(plan, plan_digest, artifact, request):
    """Validate an interrupted evaluation without allowing another optimizer step."""
    if request["plan_sha256"] != plan_digest:
        raise ValueError("Evaluation recovery belongs to another training plan.")
    for suffix, checksum in request["artifacts"].items():
        if suffix not in ("-status.json", "-error.json", "-checkpoint.pt", ".kanachan", "-pilot.json"):
            raise ValueError("Unexpected evaluation recovery artifact.")
        if sha256(artifact(suffix)) != checksum:
            raise ValueError(f"Saved evaluation artifact differs: {suffix}")
    if set(request["artifacts"]) != {"-status.json", "-error.json", "-checkpoint.pt", ".kanachan", "-pilot.json"}:
        raise ValueError("Evaluation recovery is missing a required identity.")
    previous = json.loads(artifact("-status.json").read_text(encoding="utf-8-sig"))
    error = json.loads(artifact("-error.json").read_text(encoding="utf-8-sig"))
    if (previous != error or previous["status"] != "failed"
            or previous["phase"] != "validation" or previous["plan_sha256"] != plan_digest
            or previous["error"] != "[Errno 22] Invalid argument"
            or "tqdm" not in previous["traceback"] or "flush" not in previous["traceback"]
            or previous["optimizer_steps"] <= 0):
        raise ValueError("Evaluation recovery does not match the inspected console failure.")
    remaining = plan["maximum_seconds"] - previous["wall_seconds"]
    if not 0 < request["maximum_evaluation_seconds"] <= remaining:
        raise ValueError("Evaluation recovery would exceed the remaining experiment budget.")
    if any(artifact(suffix).exists() for suffix in (
            "-summary.json", "-bc-native.json", "-cql-native.json", "-evaluation-original-error.json")):
        raise ValueError("Evaluation output already exists; inspect it before continuing.")
    return previous


def run(plan_path, evaluation_recovery=None, resume=False):
    entered = time.monotonic()
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    plan_digest = sha256(plan_path)
    target_kind = plan.get("target_kind", "td")
    if target_kind not in ("td", "mc"):
        raise ValueError("CQL target_kind must be mc or td.")
    max_epochs = plan.get("max_epochs", 1)
    if type(max_epochs) is not int or max_epochs < 1:
        raise ValueError("CQL max_epochs must be a positive integer.")
    if not 0 < plan["training_budget_seconds"] < plan["maximum_seconds"]:
        raise ValueError("CQL training must leave time for evaluation.")
    if resume and evaluation_recovery is not None:
        raise ValueError("Training and evaluation recovery are separate operations.")
    regression_key = "mse_mc" if target_kind == "mc" else "mse_td"
    recovery = None if evaluation_recovery is None else json.loads(evaluation_recovery.read_text(encoding="utf-8-sig"))
    source, prefix = Path(plan["source"]), Path(plan["output_prefix"])
    if not source.is_dir() or not prefix.parent.is_dir():
        raise ValueError("CQL source and output directory must already exist.")
    if sha256(__file__) != (plan["controller_sha256"] if recovery is None else recovery["controller_sha256"]):
        raise ValueError("CQL controller differs from the declared plan.")
    if sha256(Path(__file__).with_name("train_decision_bc.py")) != plan["utility_sha256"]:
        raise ValueError("Shared BC workflow utilities differ.")
    if sha256(plan["manifest"]) != plan["manifest_sha256"]:
        raise ValueError("CQL source manifest differs.")
    expected = json.loads(Path(plan["manifest"]).read_text(encoding="utf-8-sig"))["complete_candidate"]

    def verify_inputs():
        for name, checksum in expected.items():
            if sha256(source / name) != checksum:
                raise ValueError(f"CQL source differs: {name}")
        for split, identity in plan["dataset"].items():
            if sha256(identity["path"]) != identity["sha256"]:
                raise ValueError(f"CQL {split} data differs.")
        for name in ("bc", "opponent", "template", "pilot_data"):
            if sha256(plan[name]["path"]) != plan[name]["sha256"]:
                raise ValueError(f"CQL {name} input differs.")
        if plan.get("native_sha256") and sha256(source / "kanachan/simulation/_simulation.pyd") != plan["native_sha256"]:
            raise ValueError("Verified native extension differs.")
    verify_inputs()
    artifact = lambda suffix: prefix.with_name(prefix.name + suffix)
    if resume:
        previous = json.loads(artifact("-status.json").read_text(encoding="utf-8-sig"))
        if (previous["plan_sha256"] != plan_digest or previous["phase"] != "training"
                or previous["status"] not in ("running", "failed")
                or any(artifact(suffix).exists() for suffix in (
                    "-summary.json", "-bc-native.json", "-cql-native.json"))):
            raise ValueError("Training recovery requires an unfinished training checkpoint of this plan.")
    elif recovery is None:
        for suffix in ("-status.json", "-checkpoint.pt", ".kanachan", "-pilot.json",
                       "-summary.json", "-metrics.jsonl", "-bc-native.json", "-cql-native.json", "-error.json"):
            if artifact(suffix).exists():
                raise ValueError("CQL output already exists; inspect it instead of starting another run.")
        previous = None
    else:
        previous = evaluation_only_state(plan, plan_digest, artifact, recovery)
    wall_start = entered if previous is None else entered - previous["wall_seconds"]
    maximum_seconds = (
        plan["maximum_seconds"] if recovery is None
        else previous["wall_seconds"] + recovery["maximum_evaluation_seconds"]
    )
    lock_path = artifact(".lock")
    with lock_path.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    state = {
        "status": "running", "scope": "small_decoder_offline_cql_trial_only",
        "target_kind": target_kind,
        "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan_digest, "cursor": 0, "optimizer_steps": 0,
        "skipped_updates": 0, "last_metrics": {},
    }
    if previous is not None:
        state = copy.deepcopy(previous)
        for key in ("error", "traceback"):
            state.pop(key, None)
        state.update(status="running", pid=os.getpid())
        if recovery is not None:
            state["evaluation_recovery"] = {
                "request_sha256": sha256(evaluation_recovery),
                "controller_sha256": sha256(__file__),
                "original_training_controller_sha256": plan["controller_sha256"],
                "original_failure": previous,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "new_training_started": False,
                "maximum_seconds": recovery["maximum_evaluation_seconds"],
                "budget_clock": "accumulated active attempts; idle time before evaluation recovery excluded",
            }
            artifact("-error.json").replace(artifact("-evaluation-original-error.json"))
    sleep_guard = None
    try:
        if sys.platform == "win32" and plan["maximum_seconds"] >= 7200:
            import ctypes
            sleep_guard = ctypes.windll.kernel32
            if not sleep_guard.SetThreadExecutionState(0x80000001):
                raise OSError("Unable to keep the system awake during long CQL training.")
        os.environ.update(
            TORCHDYNAMO_DISABLE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
            CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1",
        )
        sys.path.insert(0, str(source))
        import numpy as np
        import torch
        from kanachan.constants import NUM_TYPES_OF_ACTIONS
        from kanachan.model_loader import load_model, load_model_state
        from kanachan.nn.decision_decoder import DenseDecisionDecoder, GlobalActionHead, legal_argmax
        from kanachan.nn.decision_value import q_only, transfer_bc_features
        from kanachan.training.cql import train as cql
        from kanachan.training.core.offline_rl import EpisodeReplayBuffer
        from kanachan.training.core.config.optimizer import AdamWOptimizerConfig, create
        from kanachan.training.core.rl.discrete_cql import td_components, mc_components, td_penalty, conservative_penalty
        from kanachan.training.core.rl.rewards import configured_reward
        from kanachan.training.online.state import atomic_save, capture_rng, cpu_state, export_model, restore_rng

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
        amp = torch.float16 if device.type == "cuda" else torch.float32
        template = torch.load(plan["template"]["path"], map_location="cpu", weights_only=True)

        def status(phase, **values):
            if time.monotonic() - wall_start >= maximum_seconds:
                raise TimeoutError("The bounded CQL training/evaluation deadline was reached.")
            state.update(
                phase=phase, **values,
                updated_utc=datetime.now(timezone.utc).isoformat(),
                wall_seconds=time.monotonic() - wall_start,
            )
            if device.type == "cuda":
                state["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                state["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            write_json(artifact("-status.json"), state)

        def seed():
            random.seed(plan["seed"])
            np.random.seed(plan["seed"])
            torch.manual_seed(plan["seed"])
            if device.type == "cuda":
                torch.cuda.manual_seed_all(plan["seed"])

        def initialize():
            seed()
            bc_actor = load_model(plan["bc"]["path"], map_location=torch.device("cpu")).float()
            actor = load_model_state(fresh_descriptor(template), map_location=torch.device("cpu")).float()
            backbone = next(module for module in actor.modules() if isinstance(module, DenseDecisionDecoder))
            head = next(module for module in actor.modules() if isinstance(module, GlobalActionHead))
            transfer_bc_features(backbone, head, bc_actor)
            if sum(parameter.numel() for parameter in actor.parameters()) != plan["parameters"]:
                raise ValueError("CQL actor parameter count differs.")
            actor = actor.to(device)
            network = actor.select_subsequence(out_keys=["action_value"]).train().requires_grad_(True)
            target = None if target_kind == "mc" else copy.deepcopy(network).eval().requires_grad_(False)
            optimizer, scheduler = create(
                device.type, AdamWOptimizerConfig(
                    learning_rate=plan["learning_rate"], warmup_steps=plan["warmup_steps"],
                    weight_decay=plan["weight_decay"],
                ), network,
            )
            return actor, network, target, optimizer, scheduler

        class CheckpointReached(Exception):
            pass

        class Metrics:
            def __init__(self, active, epoch=0, cursor=0, optimizer_offset=0):
                self.active = active
                self.epoch = epoch
                self.optimizer_offset = optimizer_offset
                self.last_status = time.monotonic()
                self.next_log = (cursor // 4096 + 1) * 4096
                self.latest = {}

            def add_scalar(self, tag, value, step):
                if not self.active:
                    return
                self.latest[tag] = float(value)
                state.update(training_counters(
                    self.epoch, step, plan["dataset"]["train"]["rows"], plan["batch_size"]))
                if tag == "Optimizer steps (this run)":
                    state["optimizer_steps"] = self.optimizer_offset + int(value)
                if tag.startswith("Q/"):
                    state["q_monitor_cursor"] = step
                state["last_metrics"] = dict(self.latest)
                if step >= self.next_log and tag == "Loss":
                    point = dict(epoch=self.epoch, cursor=step, total_samples=state["total_samples"],
                                 optimizer_steps=state["optimizer_steps"],
                                 seconds=time.monotonic() - training_start,
                                 skipped_updates=state["skipped_updates"], metrics=self.latest)
                    with artifact("-metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(point, allow_nan=False) + "\n")
                    self.next_log = (step // 4096 + 1) * 4096
                if time.monotonic() - self.last_status >= 30:
                    status("training", training_seconds=time.monotonic() - training_start)
                    self.last_status = time.monotonic()

        class Skips(logging.Handler):
            def emit(self, record):
                if "non-finite values in the gradient" in record.getMessage():
                    state["skipped_updates"] += 1

        logging.basicConfig(level=logging.WARNING)
        original_replay = cql.EpisodeReplayBuffer
        training_start = None

        def optimize(*, pilot=False, saved=None, interrupt=False, epoch=0, new_epoch=False):
            actor, network, target, optimizer, scheduler = initialize()
            if saved is not None:
                network.load_state_dict(saved["model"])
                if saved.get("target_kind", "td") != target_kind or (saved["target"] is None) != (target is None):
                    raise ValueError("Saved CQL target kind differs.")
                if target is not None:
                    target.load_state_dict(saved["target"])
                optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
                scheduler.load_state_dict(saved["scheduler"])
            source_values = q_only(network)
            target_values = q_only(target) if target is not None else None
            result = {}
            actual_dtypes = set()
            body = next(module for module in network.modules() if isinstance(module, DenseDecisionDecoder))
            handle = body.blocks[0].qkv.register_forward_hook(
                lambda module, inputs, output: actual_dtypes.add(str(output.dtype))
            ) if pilot else None
            checked_initial = False

            class BoundedReplay(original_replay):
                def iter_with_last(self):
                    for batch, last in super().iter_with_last():
                        boundary = not pilot and time.monotonic() - training_start >= plan["training_budget_seconds"]
                        yield batch, last or boundary
                        if last or boundary:
                            return

            def snapshot(index, **metadata):
                if pilot and index is not None and not interrupt:
                    return
                cursor = index if index is not None else metadata["resume_num_samples"]
                result.clear()
                result.update({
                    "format_version": 1, "plan_sha256": plan_digest, "target_kind": target_kind, "cursor": cursor,
                    "cursor_is_exact": metadata.get("cursor_is_exact", True),
                    "optimizer_steps": metadata["optimizer_steps"],
                    "scaler": metadata["grad_scaler_state"], "rng": capture_rng(),
                    "model": cpu_state(network.state_dict()), "target": None if target is None else cpu_state(target.state_dict()),
                    "optimizer": cpu_state(optimizer.state_dict()), "scheduler": scheduler.state_dict(),
                    "replay": metadata["replay_state"],
                })
                if pilot:
                    if index is not None:
                        raise CheckpointReached()
                    return
                result.update(training_counters(
                    epoch, cursor, plan["dataset"]["train"]["rows"], plan["batch_size"]))
                state.update(training_counters(
                    epoch, cursor, plan["dataset"]["train"]["rows"], plan["batch_size"]))
                state.update(optimizer_steps=metadata["optimizer_steps"],
                             training_seconds=time.monotonic() - training_start,
                             input_samples=result["replay"]["input_samples"],
                             total_input_samples=epoch * plan["dataset"]["train"]["rows"] + result["replay"]["input_samples"])
                result["run_state"] = copy.deepcopy(state)
                atomic_save(artifact("-checkpoint.pt"), result)
                atomic_save(artifact(".kanachan"), export_model(template, actor))
                status("training")

            def td(data):
                nonlocal checked_initial
                if not all(parameter.requires_grad for parameter in network.parameters()):
                    raise ValueError("An online-network parameter was frozen.")
                values, chosen, expected_return = td_components(source_values, target_values, data)
                if not checked_initial and saved is None:
                    legal_values = values.masked_fill(data["candidates"] >= NUM_TYPES_OF_ACTIONS, 0)
                    if not bool((legal_values == 0).all()):
                        raise ValueError("The fresh Q head did not start with zero Q values.")
                    checked_initial = True
                data["action_value"] = values
                data["td_target"] = expected_return
                data["td_error"] = 2 * td_penalty(chosen, expected_return, plan["td_loss"])

            cql.EpisodeReplayBuffer = BoundedReplay
            try:
                cql._training(
                    training_data=Path(plan["pilot_data"]["path"] if pilot else plan["dataset"]["train"]["path"]),
                    contiguous_training_data=True, rewrite_rooms=None, rewrite_grades=None, num_workers=0,
                    replay_buffer_size=min(plan["replay_size"], 128) if pilot else plan["replay_size"],
                    device=device, dtype=torch.float32, amp_dtype=amp,
                    source_network=network, target_network=target, reward_plugin=None, reward_model=None,
                    reward_scheme=plan["reward_scheme"], round_score_unit=plan["round_score_unit"],
                    round_reward_weight=plan["round_reward_weight"], discount_factor=1.,
                    td_error_computation=None if target_kind == "mc" else td,
                    target_kind=target_kind, regression_loss=plan["td_loss"],
                    alpha=plan["alpha"], cql_temperature=1.,
                    batch_size=plan["batch_size"], gradient_accumulation_steps=1, max_gradient_norm=1.,
                    optimizer=optimizer, scheduler=scheduler, target_update_interval=1,
                    target_update_rate=plan["target_update_rate"],
                    snapshot_interval=plan["pilot_checkpoint_interval"] if pilot else plan["checkpoint_interval"],
                    num_samples=0 if saved is None or new_epoch else saved["cursor"],
                    summary_writer=Metrics(
                        not pilot, epoch, 0 if saved is None or new_epoch else saved["cursor"],
                        0 if saved is None else saved["optimizer_steps"]),
                    snapshot_writer=snapshot,
                    initial_optimizer_steps=0 if saved is None else saved["optimizer_steps"],
                    initial_grad_scaler_state=None if saved is None else saved["scaler"],
                    replay_state=None if saved is None or new_epoch else saved["replay"],
                    initial_rng_state=capture_rng() if saved is None else saved["rng"],
                    q_monitor_interval=100 if not pilot else 1,
                )
            except CheckpointReached:
                if not interrupt:
                    raise
            finally:
                cql.EpisodeReplayBuffer = original_replay
                if handle is not None:
                    handle.remove()
            if result["optimizer_steps"] <= 0:
                raise ValueError("CQL completed no actual optimization updates.")
            if pilot and actual_dtypes != {str(amp)}:
                raise ValueError("CQL pilot attention projection precision differs.")
            return result

        status("verified_functional_receipt")
        functional = plan["functional_receipt"]
        if sha256(functional["path"]) != functional["sha256"]:
            raise ValueError("The completed functional verification receipt differs.")
        pilot_receipt = json.loads(Path(functional["path"]).read_text(encoding="utf-8-sig"))
        if (pilot_receipt["status"] != "complete"
                or pilot_receipt["source_manifest_sha256"] != plan["manifest_sha256"]
                or pilot_receipt["target_kind"] != ("monte_carlo_return" if target_kind == "mc" else "one_step_double_dqn")
                or pilot_receipt["feature_freezing"] or pilot_receipt["logged_return_mixing"]
                or pilot_receipt["learned_reward_predictor"]
                or not pilot_receipt["cuda_cql"]["exact_resume_all_state"]):
            raise ValueError("The completed functional check binds another CQL recipe.")
        if recovery is None:
            write_json(artifact("-pilot.json"), pilot_receipt)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if recovery is None:
            trained = None
            elapsed_training = 0.
            epoch = 0
            if resume:
                trained = torch.load(artifact("-checkpoint.pt"), map_location="cpu", weights_only=True)
                if trained["plan_sha256"] != plan_digest or not trained["cursor_is_exact"]:
                    raise ValueError("Training recovery checkpoint identity or cursor differs.")
                elapsed_training = previous["training_seconds"]
                epoch = trained.get("epoch", 0)
                state = copy.deepcopy(trained["run_state"])
                state.update(status="running", pid=os.getpid())
                with artifact("-recovery.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(previous=previous, checkpoint_sha256=sha256(artifact("-checkpoint.pt"))), allow_nan=False) + "\n")
                artifact("-error.json").unlink(missing_ok=True)
            else:
                state.update(skipped_updates=0, optimizer_steps=0,
                             **training_counters(0, 0, plan["dataset"]["train"]["rows"], plan["batch_size"]))
            training_start = time.monotonic() - elapsed_training
            status("training")
            skip_handler = Skips()
            logging.getLogger().addHandler(skip_handler)
            try:
                while epoch < max_epochs:
                    new_epoch = trained is not None and trained["cursor"] == plan["dataset"]["train"]["rows"]
                    if new_epoch:
                        epoch += 1
                    if epoch >= max_epochs or time.monotonic() - training_start >= plan["training_budget_seconds"]:
                        break
                    trained = optimize(saved=trained, epoch=epoch, new_epoch=new_epoch)
                    if trained["cursor"] < plan["dataset"]["train"]["rows"]:
                        break
            finally:
                logging.getLogger().removeHandler(skip_handler)
            state["training_seconds"] = time.monotonic() - training_start
        else:
            trained = torch.load(artifact("-checkpoint.pt"), map_location="cpu", weights_only=True)
            if (trained["plan_sha256"] != plan_digest or not trained["cursor_is_exact"]
                    or trained["cursor"] != state["cursor"]
                    or trained["optimizer_steps"] != state["optimizer_steps"]
                    or trained["replay"]["input_samples"] != state["input_samples"]):
                raise ValueError("Saved evaluation checkpoint counters or identity differ.")
            restore_rng(trained["rng"])
        state.update(training_counters(
            trained.get("epoch", 0), trained["cursor"], plan["dataset"]["train"]["rows"], plan["batch_size"]))
        state.update(optimizer_steps=trained["optimizer_steps"],
                     completed_epoch=trained["cursor"] == plan["dataset"]["train"]["rows"])
        if state["attempted_updates"] - state["skipped_updates"] != trained["optimizer_steps"]:
            raise ValueError("CQL consumed batches and successful/skipped updates do not reconcile.")
        optimizer_counts = {int(value["step"].item()) for value in trained["optimizer"]["state"].values()}
        if optimizer_counts != {trained["optimizer_steps"]}:
            raise ValueError("Actual AdamW counters differ.")
        actor = load_model(artifact(".kanachan"), map_location=torch.device("cpu")).float().to(device).eval()
        actor.requires_grad_(False)
        if target_kind == "mc":
            if trained["target"] is not None:
                raise ValueError("MC CQL checkpoint unexpectedly has a target network.")
            target = target_values = None
        else:
            target = copy.deepcopy(actor.select_subsequence(out_keys=["action_value"]))
            target.load_state_dict(trained["target"])
            target.eval().requires_grad_(False)
            target_values = q_only(target)
        source_values = q_only(actor)
        # Discard the large replay copy; the exact restart state remains on SSD.
        del trained
        gc.collect()

        @torch.no_grad()
        def heldout(split):
            status(split)
            totals = {key: 0. for key in (regression_key, "cql_gap", "centered_cql_gap",
                                           "logged_return_mae", "logged_return_bias", "target", "q")}
            samples = terminals = agreement = 0
            quantiles = []
            loader = original_replay(
                training_data=Path(plan["dataset"][split]["path"]), contiguous_training_data=True,
                num_skip_samples=0, rewrite_rooms=None, rewrite_grades=None,
                get_reward=configured_reward(plan["reward_scheme"], score_unit=plan["round_score_unit"], round_weight=plan["round_reward_weight"]),
                discount_factor=1., dtype=torch.float32, max_size=8192, batch_size=128,
                num_workers=0, pin_memory=False,
            )
            for batch in loader:
                data = batch.to(device)
                observed = data["action_value"].clone()
                values, chosen, expected_return = (
                    mc_components(source_values, data) if target_kind == "mc"
                    else td_components(source_values, target_values, data)
                )
                gap = conservative_penalty(values, chosen)
                count = (data["candidates"] < NUM_TYPES_OF_ACTIONS).sum(-1).float()
                predictions = legal_argmax(data["candidates"], values).long()
                quantity = {
                    regression_key: (chosen - expected_return).square().sum(),
                    "cql_gap": gap.sum(), "centered_cql_gap": (gap - count.log()).sum(),
                    "logged_return_mae": (chosen - observed).abs().sum(),
                    "logged_return_bias": (chosen - observed).sum(),
                    "target": expected_return.sum(), "q": chosen.sum(),
                    "agreement": (predictions == data["action"]).sum(),
                    "terminals": data["next", "done"].sum(),
                }
                numbers = torch.stack([value.float() for value in quantity.values()]).cpu().tolist()
                measured = dict(zip(quantity, numbers, strict=True))
                if not all(math.isfinite(value) for value in numbers):
                    raise ValueError("CQL held-out diagnostics became nonfinite.")
                for key in totals:
                    totals[key] += measured[key]
                samples += len(batch)
                agreement += int(measured["agreement"])
                terminals += int(measured["terminals"])
                quantiles.append(chosen.cpu())
                if time.monotonic() - wall_start > maximum_seconds:
                    raise TimeoutError("CQL experiment deadline reached during held-out read.")
            if samples != plan["dataset"][split]["rows"] or terminals != plan["dataset"][split]["terminal_trajectories"]:
                raise ValueError(f"CQL {split} row or terminal-trajectory counts differ.")
            percentiles = torch.quantile(torch.cat(quantiles), torch.tensor([.05, .5, .95])).tolist()
            return dict(
                samples=samples, terminal_trajectories=terminals,
                **{key: value / samples for key, value in totals.items()},
                q_percentiles=percentiles, behavior_action_agreement=agreement / samples,
                limitation="Logged-return error and behavior agreement are diagnostics, not current-policy value accuracy.",
            )

        state["validation"] = heldout("validation")
        state["test"] = heldout("test")
        del target, source_values, target_values, actor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if plan["native_sets"]:
            from kanachan.simulation import simulate
            from kanachan.simulation.statistics import summarize, compare_policies
            opponent = load_model(plan["opponent"]["path"], map_location=torch.device("cpu")).float().to(device).eval()
            opponent.requires_grad_(False)
            records = {}
            for arm, path in (("bc", plan["bc"]["path"]), ("cql", artifact(".kanachan"))):
                proposed = load_model(path, map_location=torch.device("cpu")).float().to(device).eval()
                proposed.requires_grad_(False)
                completed = 0

                def progress():
                    nonlocal completed
                    completed += 1
                    if completed % 4 == 0:
                        status(f"native_{arm}", completed_games=completed, target_games=plan["native_sets"] * 4)
                    if time.monotonic() - wall_start > maximum_seconds:
                        raise TimeoutError("CQL experiment deadline reached during native evaluation.")
                status(f"native_{arm}", completed_games=0, target_games=plan["native_sets"] * 4)
                with torch.inference_mode():
                    games = simulate(
                        device, torch.float32, 3, 12, opponent, 12, proposed,
                        4, plan["native_sets"], 1, 1, progress, seed=plan["native_seed"],
                    )
                results = [game.get_result() for game in games]
                metadata = [game.get_metadata() for game in games]
                if len(results) != plan["native_sets"] * 4:
                    raise ValueError("CQL native evaluation game count differs.")
                stats = summarize(results, group_ids=[value["set_id"] for value in metadata],
                                  bootstrap_samples=2000, seed=plan["native_seed"], confidence=.95)
                records[arm] = dict(
                    seed=plan["native_seed"], model_sha256=sha256(path),
                    opponent_sha256=plan["opponent"]["sha256"],
                    results=results, metadata=metadata, statistics=stats,
                )
                write_json(artifact(f"-{arm}-native.json"), records[arm])
                state[f"native_{arm}"] = stats
                del proposed, games
            if records["bc"]["metadata"] != records["cql"]["metadata"]:
                raise ValueError("BC and CQL paired native metadata differs.")
            state["paired_comparison"] = compare_policies(
                records["cql"]["results"], records["bc"]["results"],
                group_ids=[value["set_id"] for value in records["bc"]["metadata"]],
                bootstrap_samples=4000, seed=plan["native_seed"], confidence=.95,
            )
        verify_inputs()
        if recovery is not None:
            for suffix in (".kanachan", "-checkpoint.pt", "-pilot.json"):
                if sha256(artifact(suffix)) != recovery["artifacts"][suffix]:
                    raise ValueError("Evaluation recovery modified a retained training artifact.")
        state.update(
            status="complete", model=str(artifact(".kanachan")),
            model_sha256=sha256(artifact(".kanachan")),
            checkpoint_sha256=sha256(artifact("-checkpoint.pt")),
            parameters=plan["parameters"], target_kind=target_kind, recipe={key: plan[key] for key in (
                "seed", "alpha", "learning_rate", "warmup_steps", "weight_decay",
                "batch_size", "replay_size", "target_update_rate", "training_budget_seconds",
                "td_loss", "reward_scheme", "round_score_unit", "round_reward_weight",
            )} | dict(target_kind=target_kind, max_epochs=max_epochs), dataset=plan["dataset"], pilot=pilot_receipt,
            selection="Fixed end-of-budget online-Q checkpoint; held-out diagnostics and final games never choose a model.",
            limitation="Small fixed-opponent paired evaluation does not establish real playing rank.",
        )
        status("complete")
        write_json(artifact("-summary.json"), state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json(artifact("-error.json"), state)
        write_json(artifact("-status.json"), state)
        raise
    finally:
        if sleep_guard is not None:
            sleep_guard.SetThreadExecutionState(0x80000000)
        lock_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evaluation-recovery", type=Path)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    run(arguments.plan, arguments.evaluation_recovery, arguments.resume)
