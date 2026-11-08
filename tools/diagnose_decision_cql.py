"""Read-only value-gradient and BC representation diagnosis of a saved CQL run."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

from train_decision_bc import sha256, write_json


def run(plan_path, output, limit):
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    source = Path(plan["source"])
    prefix = Path(plan["output_prefix"])
    if output.exists():
        raise ValueError("Diagnosis already exists; inspect its receipt.")
    expected = json.loads(Path(plan["manifest"]).read_text(encoding="utf-8-sig"))["complete_candidate"]
    for name, checksum in expected.items():
        if sha256(source / name) != checksum:
            raise ValueError(f"Source differs from the saved trial: {name}")
    os.environ.update(TORCHDYNAMO_DISABLE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1")
    sys.path.insert(0, str(source))
    import torch
    from kanachan.constants import NUM_TYPES_OF_ACTIONS
    from kanachan.model_loader import load_model
    from kanachan.nn.decision_decoder import DenseDecisionDecoder, GlobalActionHead, legal_argmax
    from kanachan.nn.decision_value import q_only
    from kanachan.training.core.offline_rl import EpisodeReplayBuffer
    from kanachan.training.core.rl.discrete_cql import td_components, conservative_penalty
    from kanachan.training.core.rl.rewards import PlacementReward

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2026110403)
    started = time.monotonic()
    checkpoint_path = prefix.with_name(prefix.name + "-checkpoint.pt")
    model_path = prefix.with_suffix(".kanachan")
    bc_path = Path(plan["bc"]["path"])
    identity = dict(plan_sha256=sha256(plan_path), model_sha256=sha256(model_path),
                    checkpoint_sha256=sha256(checkpoint_path), bc_sha256=sha256(bc_path),
                    source_manifest_sha256=sha256(plan["manifest"]), helper_sha256=sha256(__file__))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    actor = load_model(model_path, map_location=torch.device("cpu")).float().cuda().eval().requires_grad_(False)
    target = copy.deepcopy(actor.select_subsequence(out_keys=["action_value"]))
    target.load_state_dict(checkpoint["target"])
    target.eval().requires_grad_(False)
    bc = load_model(bc_path, map_location=torch.device("cpu")).float().cuda().eval().requires_grad_(False)
    q_network, target_network = q_only(actor), q_only(target)
    q_body = next(m for m in actor.modules() if isinstance(m, DenseDecisionDecoder))
    bc_body = next(m for m in bc.modules() if isinstance(m, DenseDecisionDecoder))
    q_head = next(m for m in actor.modules() if isinstance(m, GlobalActionHead))
    bc_head = next(m for m in bc.modules() if isinstance(m, GlobalActionHead))
    reader = EpisodeReplayBuffer(
        training_data=Path(plan["dataset"]["validation"]["path"]), contiguous_training_data=True,
        num_skip_samples=0, rewrite_rooms=None, rewrite_grades=None,
        get_reward=PlacementReward(), discount_factor=1., dtype=torch.float32,
        max_size=8192, batch_size=128, num_workers=0, pin_memory=False,
    )
    groups = {name: [] for name in ("terminal", "nonterminal")}
    totals = {name: [0., 0, 0] for name in ("bc", "bc_head_cql_body", "cql_head_bc_body", "cql")}
    count = 0
    cql_shift_gradient = []
    with torch.no_grad():
        for batch in reader:
            if count >= limit:
                break
            data = batch[:limit-count].cuda()
            observed = data["action_value"].clone()
            values, chosen, expected_return = td_components(q_network, target_network, data)
            done = data["next", "done"].reshape(-1).bool()
            reward = data["next", "reward"].reshape(-1)
            residual = chosen - expected_return
            for name, mask in (("terminal", done), ("nonterminal", ~done)):
                groups[name].append(torch.stack((chosen[mask], expected_return[mask], residual[mask],
                                                 observed[mask], reward[mask]), -1).cpu())
            cql_shift_gradient.append((values.softmax(-1).sum(-1) - 1).cpu())
            args = [data[key] for key in ("sparse", "numeric", "progression", "candidates")]
            bc_rep = bc_body(*args)
            q_rep = q_body(*args)
            variants = {
                "bc": bc_head(data["candidates"], bc_rep),
                "bc_head_cql_body": bc_head(data["candidates"], q_rep),
                "cql_head_bc_body": q_head(data["candidates"], bc_rep),
                "cql": values,
            }
            bc_action = legal_argmax(data["candidates"], variants["bc"])
            for name, logits in variants.items():
                logits = logits.masked_fill(data["candidates"] >= NUM_TYPES_OF_ACTIONS, -torch.inf)
                predicted = legal_argmax(data["candidates"], logits)
                loss = torch.nn.functional.cross_entropy(logits, data["action"].long(), reduction="sum")
                totals[name][0] += loss.item()
                totals[name][1] += int((predicted == data["action"]).sum())
                totals[name][2] += int((predicted == bc_action).sum())
            count += len(data)
            if count % 8192 == 0:
                print(json.dumps(dict(phase="diagnosis", rows=count, seconds=time.monotonic()-started)), flush=True)
    result = dict(status="complete", scope="saved_cql_readonly_causal_diagnostics",
                  identity=identity, validation_rows=count, changed_model_or_source=False)
    stats = {}
    for name, parts in groups.items():
        values = torch.cat(parts)
        q, target_q, delta, mc, reward = values.unbind(-1)
        stats[name] = dict(
            rows=len(values), fraction=len(values)/count,
            q_mean=float(q.mean()), target_mean=float(target_q.mean()),
            q_range=[float(q.min()), float(q.max())],
            reward_values=sorted(set(reward.tolist())),
            residual_mean=float(delta.mean()),
            residual_percentiles=torch.quantile(delta, torch.tensor([.05, .5, .95])).tolist(),
            huber_saturated_fraction=float((delta.abs() > 1).float().mean()),
            mse_offset_gradient_contribution=float(delta.sum()/count),
            huber_offset_gradient_contribution=float(delta.clamp(-1, 1).sum()/count),
            mc_offset_gradient_contribution=float((q-mc).sum()/count),
        )
    result["td_groups"] = stats
    result["common_q_shift_gradient"] = {
        "mse": sum(v["mse_offset_gradient_contribution"] for v in stats.values()),
        "huber": sum(v["huber_offset_gradient_contribution"] for v in stats.values()),
        "mc_mse": sum(v["mc_offset_gradient_contribution"] for v in stats.values()),
        "cql_max_abs": float(torch.cat(cql_shift_gradient).abs().max()),
        "interpretation": "Negative gradient raises the shared Q offset; positive lowers it. Target is detached."
    }
    result["representation_controls"] = {
        name: dict(cross_entropy=values[0]/count, behavior_agreement=values[1]/count,
                   original_bc_policy_agreement=values[2]/count)
        for name, values in totals.items()
    }
    result["seconds"] = time.monotonic()-started
    if identity["model_sha256"] != sha256(model_path) or identity["bc_sha256"] != sha256(bc_path):
        raise ValueError("A preserved model changed during the diagnosis.")
    write_json(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32768)
    arguments = parser.parse_args()
    run(arguments.plan, arguments.output, arguments.limit)
