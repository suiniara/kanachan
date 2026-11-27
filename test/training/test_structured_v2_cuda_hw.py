"""CUDA hardware integration tests for StructuredDualDecoder V2 training & inference (§9)."""

from __future__ import annotations

import gzip
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from tensordict import TensorDict
from torch.amp.grad_scaler import GradScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kanachan.constants import (
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_SPARSE_FEATURES,
)
from kanachan.model_loader import load_model
from kanachan.nn.structured_iql import JointStructuredIQLModule
from kanachan.simulation import simulate
from kanachan.structured_v2 import OFFSETS_136, PAD_ACTION_ID, derive_structured_v2_batch
from kanachan.training.bc.streaming_v2 import SpoolingRoundBCDataSource
from kanachan.training.bc.trajectory_v2 import (
    TrajectoryBCModel,
    load_trajectory_bc_checkpoint,
    train_trajectory_bc_steps,
)
from kanachan.training.core.optimizers import adamw_parameter_groups
from kanachan.training.core.test_env import (
    require_or_skip_cuda,
    require_or_skip_real_data,
)
from kanachan.training.iql.joint_v2 import (
    attach_structured_v2_transition_features,
    build_joint_iql_optimizer,
    build_joint_iql_recipe,
    load_joint_iql_checkpoint,
    train_joint_iql_batches,
)
from kanachan.training.iql.streaming_v2 import StreamingIQLDataset


def _extract_complete_games_subset(src: Path, *, max_games: int, max_scan: int = 30000) -> list[str]:
    by_u: dict[str, list[str]] = {}
    last_i: dict[str, int] = {}
    with gzip.open(src, "rt", encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            if i >= max_scan:
                break
            u = line.split("\t", 1)[0]
            by_u.setdefault(u, []).append(line)
            last_i[u] = i
    out: list[str] = []
    picked = 0
    for u, lines in by_u.items():
        if last_i[u] < max_scan - 4000 and len(lines) >= 60:
            out.extend(lines)
            picked += 1
            if picked >= max_games:
                break
    return out


class TestStructuredV2CudaHardware(unittest.TestCase):
    def test_cuda_bc_and_iql_amp_fp16_bf16_scaler_checkpoint_and_simulate(self):
        dev = require_or_skip_cuda(self, context="CUDA AMP + GradScaler checkpoint + simulate")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Synthetic self-contained RL transitions on CUDA for FP16 GradScaler + BF16 verification
            def _mk_row(left: int, prog: list[int]) -> tuple[list[int], list[int]]:
                r_sp = [2, 5, 7, 23, 39, 55, 71, 75, 78, 82 + left, 152 + 1]
                closed = {1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1, 21: 1, 22: 1, 23: 1, 31: 3, 32: 1}
                for t37, cnt in sorted(closed.items()):
                    for k in range(cnt):
                        r_sp.append(337 + OFFSETS_136[t37] + k)
                r_sp.append(473 + 32)
                r_sp.extend([NUM_TYPES_OF_SPARSE_FEATURES] * (MAX_NUM_ACTIVE_SPARSE_FEATURES - len(r_sp)))
                r_pr = list(prog) + [NUM_TYPES_OF_PROGRESSION_FEATURES] * (
                    MAX_LENGTH_OF_PROGRESSION_FEATURES - len(prog)
                )
                return r_sp, r_pr

            sp_rows, pr_rows = zip(
                *[
                    _mk_row(69, [0]),
                    _mk_row(68, [0, 5 + 0 * 148 + 4 * 4]),
                    _mk_row(69, [0]),
                    _mk_row(68, [0, 5 + 0 * 148 + 4 * 5]),
                    _mk_row(69, [0]),
                    _mk_row(68, [0, 5 + 0 * 148 + 4 * 6]),
                    _mk_row(69, [0]),
                    _mk_row(68, [0, 5 + 0 * 148 + 4 * 7]),
                ],
                strict=True,
            )
            sp = torch.tensor(sp_rows, dtype=torch.int32)
            nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]] * 8, dtype=torch.int32)
            pr = torch.tensor(pr_rows, dtype=torch.int32)
            ca = torch.tensor(
                [[4, 44, 84, 130] + [PAD_ACTION_ID] * (MAX_NUM_ACTION_CANDIDATES - 4)] * 8,
                dtype=torch.int32,
            )
            from kanachan.training.bc.trajectory_v2 import build_round_trajectories_from_raw

            rounds = build_round_trajectories_from_raw(
                sp, nu, pr, ca,
                torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
                ["r0", "r0", "r1", "r1", "r2", "r2", "r3", "r3"],
                final_ranks=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
            )

            # Test FP16 GradScaler + LinearLR checkpoint recovery on TrajectoryBCModel
            torch.manual_seed(420)
            bm1 = TrajectoryBCModel(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, use_next_event_head=True, use_final_rank_head=True,
                final_rank_weight=0.1, device=dev, dtype=torch.float32,
            )
            opt1 = torch.optim.AdamW(adamw_parameter_groups(bm1, 0.01), lr=5e-4, fused=True)
            sch1 = torch.optim.lr_scheduler.LinearLR(opt1, start_factor=0.5, total_iters=2)
            gs1 = GradScaler("cuda", init_scale=256.0, enabled=True)
            ckpt_dir = root / "bc_fp16"
            res1 = train_trajectory_bc_steps(
                bm1, rounds, opt1, batch_size=2, max_steps=2, device=dev,
                amp_dtype=torch.float16, scheduler=sch1, grad_scaler=gs1,
                ckpt_dir=ckpt_dir, checkpoint_every=1,
            )
            self.assertEqual(res1["step"], 2)
            bm2 = TrajectoryBCModel(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, use_next_event_head=True, use_final_rank_head=True,
                final_rank_weight=0.1, device=dev, dtype=torch.float32,
            )
            opt2 = torch.optim.AdamW(adamw_parameter_groups(bm2, 0.01), lr=5e-4, fused=True)
            sch2 = torch.optim.lr_scheduler.LinearLR(opt2, start_factor=0.5, total_iters=2)
            gs2 = GradScaler("cuda", init_scale=1024.0, enabled=True)
            res2 = train_trajectory_bc_steps(
                bm2, rounds, opt2, batch_size=2, max_steps=2, device=dev,
                amp_dtype=torch.float16, scheduler=sch2, grad_scaler=gs2,
                resume_from=ckpt_dir / "traj-bc-1.pt",
            )
            self.assertEqual(res2["step"], 2)
            self.assertEqual(gs2.get_scale(), 256.0)
            for k in bm1.state_dict():
                torch.testing.assert_close(bm1.state_dict()[k], bm2.state_dict()[k], atol=1e-5, rtol=1e-5)

            actor = load_model(ckpt_dir / "model.2.kanachan", map_location=dev).to(dev, dtype=torch.half).eval()
            with torch.inference_mode():
                sim_logs = simulate(dev, torch.half, 3, 12, actor, 12, actor, 1, 1, 1, 2, lambda: None, seed=123)
            self.assertEqual(len(sim_logs), 1)
            self.assertEqual(len(sim_logs[0].get_result()), 4)

            # Test JointStructuredIQLModule FP16 GradScaler + LinearLR + BF16 on CUDA
            iq_td = attach_structured_v2_transition_features(
                TensorDict(
                    {
                        "sparse": sp[:4],
                        "numeric": nu[:4],
                        "progression": pr[:4],
                        "candidates": ca[:4],
                        "action": torch.tensor([0, 1, 2, 3], dtype=torch.int64),
                        "batch_weight": torch.ones((4,), dtype=torch.float32),
                        "next": {
                            "sparse": sp[:4],
                            "numeric": nu[:4],
                            "progression": pr[:4],
                            "candidates": ca[:4],
                            "reward": torch.tensor([0.1, -0.1, 0.2, 0.0], dtype=torch.float32),
                            "discount": torch.full((4,), 0.99, dtype=torch.float32),
                            "done": torch.tensor([False, False, False, True]),
                            "end_of_game": torch.tensor([False, False, False, True]),
                            "final_rank": torch.tensor([0, 0, 1, 1], dtype=torch.int64),
                        },
                    },
                    batch_size=[4],
                )
            )

            iq_batches = [iq_td[:2], iq_td[2:]]
            torch.manual_seed(421)
            im1 = JointStructuredIQLModule(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, use_cql_quantile_head=True, num_quantiles=8,
                final_rank=True, final_rank_weight=0.1, device=dev, dtype=torch.float32,
            )
            iopt1 = build_joint_iql_optimizer(im1, base_lr=3e-4, weight_decay=0.01, device=dev, fused=True)
            isch1 = torch.optim.lr_scheduler.LinearLR(iopt1, start_factor=0.5, total_iters=2)
            igs1 = GradScaler("cuda", init_scale=256.0, enabled=True)
            irec = build_joint_iql_recipe(
                im1, base_lr=3e-4, weight_decay=0.01, decoder1_lr_scale=0.25,
                source_data_sha256="cuda_synth", preprocess_recipe_sha256="cuda_synth_prep",
            )
            iq_ckpt = root / "iq_fp16"
            ires1 = train_joint_iql_batches(
                im1, iq_batches, iopt1, recipe=irec, device=dev,
                amp_dtype=torch.float16, scheduler=isch1, grad_scaler=igs1,
                ckpt_dir=iq_ckpt, checkpoint_every=1, max_steps=2,
            )
            self.assertEqual(ires1["step"], 2)
            im2 = JointStructuredIQLModule(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, use_cql_quantile_head=True, num_quantiles=8,
                final_rank=True, final_rank_weight=0.1, device=dev, dtype=torch.float32,
            )
            iopt2 = build_joint_iql_optimizer(im2, base_lr=3e-4, weight_decay=0.01, device=dev, fused=True)
            isch2 = torch.optim.lr_scheduler.LinearLR(iopt2, start_factor=0.5, total_iters=2)
            igs2 = GradScaler("cuda", init_scale=1024.0, enabled=True)
            ires2 = train_joint_iql_batches(
                im2, iq_batches, iopt2, recipe=irec, device=dev,
                amp_dtype=torch.float16, scheduler=isch2, grad_scaler=igs2,
                resume_from=iq_ckpt / "joint-iql-1.pt", max_steps=2,
            )
            self.assertEqual(ires2["step"], 2)
            self.assertEqual(igs2.get_scale(), 256.0)
            for k in im1.state_dict():
                torch.testing.assert_close(im1.state_dict()[k], im2.state_dict()[k], atol=1e-5, rtol=1e-5)

            # Also verify BF16 Joint IQL step on CUDA
            im_bf = JointStructuredIQLModule(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, use_cql_quantile_head=True, num_quantiles=8,
                final_rank=True, final_rank_weight=0.1, device=dev, dtype=torch.float32,
            )
            iopt_bf = build_joint_iql_optimizer(im_bf, base_lr=3e-4, weight_decay=0.01, device=dev, fused=True)
            irec_bf = build_joint_iql_recipe(im_bf, base_lr=3e-4, weight_decay=0.01, decoder1_lr_scale=0.25)
            res_bf = train_joint_iql_batches(
                im_bf, iq_batches, iopt_bf, recipe=irec_bf, device=dev,
                amp_dtype=torch.bfloat16, max_steps=1,
            )
            self.assertEqual(res_bf["optimizer_steps"], 1)

    def test_cuda_real_data_bc_spooling_and_iql_streaming(self):
        dev = require_or_skip_cuda(self, context="CUDA real-data BC & IQL streaming integration")
        val_bc = require_or_skip_real_data(self, "bc", context="CUDA real-data BC streaming")
        val_rl = require_or_skip_real_data(self, "rl", context="CUDA real-data IQL streaming")

        bc_lines = _extract_complete_games_subset(val_bc, max_games=3, max_scan=25000)
        self.assertGreater(len(bc_lines), 100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bc_sub = root / "real.bc.tsv.gz"
            with gzip.open(bc_sub, "wt", encoding="utf-8") as fp:
                fp.writelines(bc_lines)
            with SpoolingRoundBCDataSource(bc_sub, cache_dir=root / "bc_spool", num_hash_buckets=8) as ds:
                bm = TrajectoryBCModel(
                    preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                    n_d1_layers=1, n_d2_layers=1, use_next_event_head=True, use_final_rank_head=True,
                    final_rank_weight=0.1, device=dev, dtype=torch.float32,
                )
                opt = torch.optim.AdamW(adamw_parameter_groups(bm, 0.01), lr=3e-4, fused=True)
                res_bc = train_trajectory_bc_steps(
                    bm, ds, opt, batch_size=4, max_steps=3, device=dev,
                    amp_dtype=torch.bfloat16, prefetch_batches=2, grad_accum_steps=1,
                )
                self.assertEqual(res_bc["optimizer_steps"], 3)

            with StreamingIQLDataset(
                val_rl, batch_size=8, reservoir_size=256, warmup_transitions=32,
                reward_scheme="round_and_placement", max_games=4, max_prefetch=2,
            ) as ids:
                mod = JointStructuredIQLModule(
                    preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                    n_d1_layers=1, n_d2_layers=1, use_cql_quantile_head=True, num_quantiles=8,
                    final_rank=True, final_rank_weight=0.1, device=dev, dtype=torch.float32,
                )
                jopt = build_joint_iql_optimizer(mod, base_lr=3e-4, weight_decay=0.01, device=dev, fused=True)
                rec = build_joint_iql_recipe(
                    mod, base_lr=3e-4, weight_decay=0.01, decoder1_lr_scale=0.25,
                    data_sha256=ids.data_sha256,
                    source_data_sha256=ids.source_data_sha256,
                    preprocess_recipe_sha256=ids.preprocess_recipe_sha256,
                )
                res_iq = train_joint_iql_batches(
                    mod, ids, jopt, recipe=rec, device=dev, amp_dtype=torch.bfloat16, max_steps=3
                )
                self.assertEqual(res_iq["optimizer_steps"], 3)


if __name__ == "__main__":
    unittest.main()
