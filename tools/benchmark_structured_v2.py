"""RTX 3090 / CPU parameter, throughput, SDPA, and 64-hanchan simulate benchmark (§6.1B)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from kanachan.constants import NUM_TYPES_OF_ACTIONS
from kanachan.model_loader import dump_model, dump_object, load_model
from kanachan.nn.decode_converter import DecodeConverter
from kanachan.nn.structured_dual_decoder import (
    STRUCTURED_V2_PRESETS,
    StructuredActionHead,
    StructuredDualDecoder,
)
from kanachan.nn.structured_iql import JointStructuredIQLModule
from kanachan.simulation import simulate
from kanachan.structured_v2 import derive_structured_v2_batch
from kanachan.training.bc.trajectory_v2 import (
    TrajectoryBCModel,
    collate_round_batch,
    parse_bc_rounds,
)
from kanachan.training.core.test_env import is_cuda_skipped, resolve_val_bc_path
from tools.audit_structured_v2 import load_samples


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _export_v2_actor(
    backbone: StructuredDualDecoder, head: StructuredActionHead, path: Path, dtype: torch.dtype
) -> None:
    enc_tdm = TensorDictModule(
        backbone,
        in_keys=["sparse", "numeric", "progression", "candidates"],
        out_keys=["encode"],
    )
    head_tdm = TensorDictModule(
        head, in_keys=["candidates", "encode"], out_keys=["decode"]
    )
    lp_mod = DecodeConverter("log_probs")
    lp_tdm = TensorDictModule(lp_mod, in_keys=["candidates", "decode"], out_keys=["log_probs"])
    arg_mod = DecodeConverter("argmax", tie_break_by_action_id=True)
    arg_tdm = TensorDictModule(arg_mod, in_keys=["candidates", "log_probs"], out_keys=["action"])
    seq = TensorDictSequential(enc_tdm, head_tdm, lp_tdm, arg_tdm)
    state = dump_model(
        seq,
        [
            dump_object(
                enc_tdm,
                [
                    dump_object(
                        backbone,
                        [],
                        {
                            "preset": backbone.preset,
                            "dimension": backbone.dimension,
                            "num_heads": backbone.num_heads,
                            "dim_feedforward": backbone.dim_feedforward,
                            "n_d1_layers": backbone.n_d1_layers,
                            "n_d2_layers": backbone.n_d2_layers,
                            "use_rel_bias": backbone.use_rel_bias,
                            "cand_self_attn": backbone.cand_self_attn,
                            "head_type": backbone.head_type,
                            "enable_early_shape_hint": backbone.enable_early_shape_hint,
                            "include_room_level": False,
                            "rope_base": backbone.rope_base,
                            "checkpointing": False,
                            "device": torch.device("cpu"),
                            "dtype": dtype,
                        },
                    )
                ],
                {
                    "in_keys": ["sparse", "numeric", "progression", "candidates"],
                    "out_keys": ["encode"],
                },
            ),
            dump_object(
                head_tdm,
                [
                    dump_object(
                        head,
                        [],
                        {
                            "input_dimension": head.input_dimension,
                            "hidden_dimension": head.hidden_dimension,
                            "head_type": head.head_type,
                            "device": torch.device("cpu"),
                            "dtype": dtype,
                        },
                    )
                ],
                {"in_keys": ["candidates", "encode"], "out_keys": ["decode"]},
            ),
            dump_object(
                lp_tdm,
                [dump_object(lp_mod, ["log_probs"], {})],
                {"in_keys": ["candidates", "decode"], "out_keys": ["log_probs"]},
            ),
            dump_object(
                arg_tdm,
                [dump_object(arg_mod, ["argmax"], {"tie_break_by_action_id": True})],
                {"in_keys": ["candidates", "log_probs"], "out_keys": ["action"]},
            ),
        ],
        {},
    )
    torch.save(state, path)


def benchmark_presets(
    data_path: Path,
    *,
    num_hanchans: int = 64,
    out_path: Path | None = None,
) -> dict[str, Any]:
    _uuids, sp, nu, pr, ca = load_samples(data_path, 512)
    feats_cpu = derive_structured_v2_batch(sp, nu, pr, ca)
    rounds = parse_bc_rounds(data_path, max_rounds=48, max_lines=4096, allow_incomplete=True)
    round_batch_cpu = collate_round_batch(rounds[:16])

    preset_stats: dict[str, Any] = {}
    has_cuda = (not is_cuda_skipped()) and torch.cuda.is_available()
    cuda_dev = torch.device("cuda") if has_cuda else torch.device("cpu")
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "cpu-only"

    for name in STRUCTURED_V2_PRESETS:
        bb_cpu = StructuredDualDecoder(preset=name, device=torch.device("cpu"), dtype=torch.float32)
        hd_cpu = StructuredActionHead(
            input_dimension=bb_cpu.dimension,
            head_type=bb_cpu.head_type,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        n_params_bb = _count_params(bb_cpu)
        n_params_tot = n_params_bb + _count_params(hd_cpu)

        entry: dict[str, Any] = {
            "backbone_params": n_params_bb,
            "policy_total_params": n_params_tot,
        }

        # CPU single-step inference (bs=64)
        bs_cpu = 64
        x_cpu = (
            feats_cpu["v2_tile_feats"][:bs_cpu],
            feats_cpu["v2_events"][:bs_cpu],
            feats_cpu["v2_event_len"][:bs_cpu],
            feats_cpu["v2_global_feats"][:bs_cpu],
            feats_cpu["v2_player_feats"][:bs_cpu],
            feats_cpu["v2_cand_ids"][:bs_cpu],
            feats_cpu["v2_cand_mask"][:bs_cpu],
            feats_cpu["v2_cand_mode"][:bs_cpu],
            feats_cpu["v2_cand_dyn"][:bs_cpu],
        )
        ca_cpu = ca[:bs_cpu]
        bb_cpu.eval()
        hd_cpu.eval()
        with torch.inference_mode():
            enc = bb_cpu(*x_cpu)
            _ = hd_cpu(ca_cpu, enc)
            t0 = time.perf_counter()
            iters_cpu = 6
            for _ in range(iters_cpu):
                enc = bb_cpu(*x_cpu)
                _ = hd_cpu(ca_cpu, enc)
            t1 = time.perf_counter()
        entry["cpu_infer_steps_per_sec"] = round((bs_cpu * iters_cpu) / max(1e-9, t1 - t0), 1)

        if has_cuda:
            bs_gpu = 128
            model_bc = TrajectoryBCModel(
                preset=name,
                use_next_event_head=True,
                device=cuda_dev,
                dtype=torch.float32,
            )
            x_gpu = tuple(feats_cpu[k][:bs_gpu].to(cuda_dev) for k in (
                "v2_tile_feats", "v2_events", "v2_event_len", "v2_global_feats",
                "v2_player_feats", "v2_cand_ids", "v2_cand_mask", "v2_cand_mode", "v2_cand_dyn",
            ))
            ca_gpu = ca[:bs_gpu].to(cuda_dev)
            act_gpu = torch.zeros((bs_gpu,), dtype=torch.long, device=cuda_dev)

            # GPU BF16 inference
            model_bc.eval()
            torch.cuda.reset_peak_memory_stats(cuda_dev)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                enc = model_bc.backbone(*x_gpu)
                _ = model_bc.policy_head(ca_gpu, enc)
                _sync(cuda_dev)
                t0 = time.perf_counter()
                iters = 20
                for _ in range(iters):
                    enc = model_bc.backbone(*x_gpu)
                    _ = model_bc.policy_head(ca_gpu, enc)
                _sync(cuda_dev)
                t1 = time.perf_counter()
            entry["cuda_bf16_infer_steps_per_sec"] = round((bs_gpu * iters) / max(1e-9, t1 - t0), 1)

            # GPU BF16 single-step BC train
            opt = torch.optim.AdamW(model_bc.parameters(), lr=1e-4)
            model_bc.train()
            torch.cuda.reset_peak_memory_stats(cuda_dev)
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    enc = model_bc.backbone(*x_gpu)
                    lg = model_bc.policy_head(ca_gpu, enc)
                    loss = torch.nn.functional.cross_entropy(lg, act_gpu)
                loss.backward()
                opt.step()
            _sync(cuda_dev)
            torch.cuda.reset_peak_memory_stats(cuda_dev)
            t0 = time.perf_counter()
            for _ in range(15):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    enc = model_bc.backbone(*x_gpu)
                    lg = model_bc.policy_head(ca_gpu, enc)
                    loss = torch.nn.functional.cross_entropy(lg, act_gpu)
                loss.backward()
                opt.step()
            _sync(cuda_dev)
            t1 = time.perf_counter()
            entry["cuda_bf16_single_step_train_steps_per_sec"] = round(
                (bs_gpu * 15) / max(1e-9, t1 - t0), 1
            )
            entry["cuda_bf16_single_step_peak_mb"] = round(
                torch.cuda.max_memory_allocated(cuda_dev) / (1024 * 1024), 1
            )

            # GPU BF16 whole-round parallel Teacher Forcing BC train
            rb_gpu = {k: v.to(cuda_dev) for k, v in round_batch_cpu.items()}
            n_active_dec = int(rb_gpu["decision_mask"].sum().item())
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model_bc(rb_gpu)
                out["loss"].backward()
                opt.step()
            _sync(cuda_dev)
            torch.cuda.reset_peak_memory_stats(cuda_dev)
            t0 = time.perf_counter()
            for _ in range(15):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model_bc(rb_gpu)
                out["loss"].backward()
                opt.step()
            _sync(cuda_dev)
            t1 = time.perf_counter()
            entry["cuda_bf16_round_tf_train_steps_per_sec"] = round(
                (n_active_dec * 15) / max(1e-9, t1 - t0), 1
            )
            entry["cuda_bf16_round_tf_peak_mb"] = round(
                torch.cuda.max_memory_allocated(cuda_dev) / (1024 * 1024), 1
            )

            # Joint IQL stage-2 throughput for v2_base & v2_tiny
            if name in ("v2_tiny", "v2_base"):
                del model_bc, opt
                torch.cuda.empty_cache()
                joint = JointStructuredIQLModule(preset=name, device=cuda_dev, dtype=torch.float32)
                j_opt = torch.optim.AdamW(
                    joint.parameter_groups(base_lr=1e-4, weight_decay=0.01)
                )
                bs_iql = 64
                next_obs = TensorDict(
                    {
                        "sparse": sp[:bs_iql].to(cuda_dev),
                        "numeric": nu[:bs_iql].to(cuda_dev),
                        "progression": pr[:bs_iql].to(cuda_dev),
                        "candidates": ca[:bs_iql].to(cuda_dev),
                        "v2_tile_feats": feats_cpu["v2_tile_feats"][:bs_iql].to(cuda_dev),
                        "v2_events": feats_cpu["v2_events"][:bs_iql].to(cuda_dev),
                        "v2_event_len": feats_cpu["v2_event_len"][:bs_iql].to(cuda_dev),
                        "v2_global_feats": feats_cpu["v2_global_feats"][:bs_iql].to(cuda_dev),
                        "v2_player_feats": feats_cpu["v2_player_feats"][:bs_iql].to(cuda_dev),
                        "v2_cand_ids": feats_cpu["v2_cand_ids"][:bs_iql].to(cuda_dev),
                        "v2_cand_mask": feats_cpu["v2_cand_mask"][:bs_iql].to(cuda_dev),
                        "v2_cand_mode": feats_cpu["v2_cand_mode"][:bs_iql].to(cuda_dev),
                        "v2_cand_dyn": feats_cpu["v2_cand_dyn"][:bs_iql].to(cuda_dev),
                        "done": torch.tensor([(i % 8 == 0) for i in range(bs_iql)], device=cuda_dev),
                        "reward": torch.zeros((bs_iql,), dtype=torch.float32, device=cuda_dev),
                    },
                    batch_size=[bs_iql],
                )
                iql_td = TensorDict(
                    {
                        "sparse": sp[:bs_iql].to(cuda_dev),
                        "numeric": nu[:bs_iql].to(cuda_dev),
                        "progression": pr[:bs_iql].to(cuda_dev),
                        "candidates": ca[:bs_iql].to(cuda_dev),
                        "v2_tile_feats": feats_cpu["v2_tile_feats"][:bs_iql].to(cuda_dev),
                        "v2_events": feats_cpu["v2_events"][:bs_iql].to(cuda_dev),
                        "v2_event_len": feats_cpu["v2_event_len"][:bs_iql].to(cuda_dev),
                        "v2_global_feats": feats_cpu["v2_global_feats"][:bs_iql].to(cuda_dev),
                        "v2_player_feats": feats_cpu["v2_player_feats"][:bs_iql].to(cuda_dev),
                        "v2_cand_ids": feats_cpu["v2_cand_ids"][:bs_iql].to(cuda_dev),
                        "v2_cand_mask": feats_cpu["v2_cand_mask"][:bs_iql].to(cuda_dev),
                        "v2_cand_mode": feats_cpu["v2_cand_mode"][:bs_iql].to(cuda_dev),
                        "v2_cand_dyn": feats_cpu["v2_cand_dyn"][:bs_iql].to(cuda_dev),
                        "action": torch.zeros((bs_iql,), dtype=torch.int32, device=cuda_dev),
                        "next": next_obs,
                    },
                    batch_size=[bs_iql],
                )
                for _ in range(3):
                    j_opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                        out_iql = joint(iql_td, amp_dtype=torch.bfloat16)
                    out_iql["loss"].backward()
                    j_opt.step()
                    joint.update_targets()
                _sync(cuda_dev)
                torch.cuda.reset_peak_memory_stats(cuda_dev)
                t0 = time.perf_counter()
                for _ in range(15):
                    j_opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                        out_iql = joint(iql_td, amp_dtype=torch.bfloat16)
                    out_iql["loss"].backward()
                    j_opt.step()
                    joint.update_targets()
                _sync(cuda_dev)
                t1 = time.perf_counter()
                entry["cuda_bf16_joint_iql_steps_per_sec"] = round(
                    (bs_iql * 15) / max(1e-9, t1 - t0), 1
                )
                entry["cuda_bf16_joint_iql_peak_mb"] = round(
                    torch.cuda.max_memory_allocated(cuda_dev) / (1024 * 1024), 1
                )
                del joint, j_opt
            else:
                del model_bc, opt
            torch.cuda.empty_cache()

        preset_stats[name] = entry

    # End-to-end 64-hanchan simulation with exported v2_base model.kanachan
    sim_report: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "v2_base.kanachan"
        bb_sim = StructuredDualDecoder(preset="v2_base", device=torch.device("cpu"), dtype=torch.float32)
        hd_sim = StructuredActionHead(
            input_dimension=bb_sim.dimension,
            head_type=bb_sim.head_type,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        _export_v2_actor(bb_sim, hd_sim, model_path, torch.float32)
        sim_dtype = torch.half if has_cuda else torch.float32
        sim_dev = cuda_dev if has_cuda else torch.device("cpu")
        loaded = load_model(model_path, map_location=torch.device("cpu"))
        loaded.to(device=sim_dev, dtype=sim_dtype)
        loaded.requires_grad_(False)
        loaded.eval()

        done_games = 0
        t0 = time.perf_counter()
        with torch.inference_mode():
            game_logs = simulate(
                sim_dev,
                sim_dtype,
                3,  # jade room
                12,
                loaded,
                12,
                loaded,
                1,  # non-duplicated full hanchan
                num_hanchans,
                min(16, num_hanchans),
                min(32, max(4, num_hanchans)),
                lambda: None,
                seed=20261126,
            )
        t1 = time.perf_counter()
        for gl in game_logs:
            res = gl.get_result()
            assert len(res) == 4
            done_games += 1
        sim_report = {
            "device": str(sim_dev),
            "dtype": str(sim_dtype),
            "hanchans_completed": done_games,
            "elapsed_sec": round(t1 - t0, 2),
            "hanchans_per_sec": round(done_games / max(1e-9, t1 - t0), 2),
        }

    report = {
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled() if has_cuda else False,
        "sdpa_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled() if has_cuda else False,
        "presets": preset_stats,
        "simulation": sim_report,
    }
    print(json.dumps(report, indent=2))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=resolve_val_bc_path(),
        help="Path to validation.bc.tsv(.gz) (defaults to KANACHAN_VAL_BC_DATA env var).",
    )
    parser.add_argument("--hanchans", type=int, default=64)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/v2-verify/phase1b_benchmark.json"),
    )
    args = parser.parse_args()
    if args.data is None or not Path(args.data).is_file():
        raise FileNotFoundError(
            "Benchmark validation data not found; pass --data <path> or set KANACHAN_VAL_BC_DATA."
        )
    benchmark_presets(args.data, num_hanchans=args.hanchans, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
