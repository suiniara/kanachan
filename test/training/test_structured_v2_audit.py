"""Comprehensive Numerical Safety, QR-DQN Math, Trajectory Completeness, C++/Py Parity & E2E Audit Tests."""

from __future__ import annotations

import copy
import datetime
import gzip
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel

from kanachan.constants import NUM_TYPES_OF_ACTIONS
from kanachan.model_loader import load_model
from kanachan.nn.decode_converter import DecodeConverter
from kanachan.nn.decision_decoder import DecisionContext
from kanachan.nn.structured_dual_decoder import (
    StructuredActionHead,
    StructuredDualDecoder,
    StructuredKVCache,
    cached_structured_decision,
)
from kanachan.nn.structured_iql import (
    JointStructuredIQLModule,
    compute_qr_dqn_target_quantiles,
    quantile_cql_loss,
)
from kanachan.simulation import simulate
from kanachan.structured_v2 import (
    PAD_ACTION_ID,
    derive_structured_v2_batch,
    permute_raw_four_fields,
    permute_structured_batch,
)
from kanachan.structured_v2_ref import derive_structured_v2_batch_py
from kanachan.training.bc.trajectory_v2 import (
    TrajectoryBCModel,
    build_round_trajectories_from_raw,
    collate_round_batch,
    parse_bc_rounds,
    train_trajectory_bc_steps,
)
from kanachan.training.core.bc.data_iterator import DataIterator as CoreBCDataIterator
from kanachan.training.core.optimizers import adamw_parameter_groups
from kanachan.training.core.ranking import scores_to_ranks
from kanachan.training.iql.joint_v2 import (
    build_joint_iql_optimizer,
    build_joint_iql_recipe,
    load_joint_iql_checkpoint,
    train_joint_iql_batches,
)
from tools.audit_structured_v2 import (
    _AUDIT_TENSORS,
    _compare_cpp_and_py,
    harvest_rare_categories,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VAL_DATA_PATH = Path(r"D:\kanachan-training-data\validation.bc.tsv.gz")


def _make_sparse(
    *,
    room: int = 2,
    seat: int = 0,
    chang: int = 0,
    ju: int = 0,
    left: int = 50,
    doras: tuple[int, ...] = (1,),
    closed37: dict[int, int] | None = None,
    zimo: int = 32,
) -> list[int]:
    from kanachan.structured_v2 import OFFSETS_136

    sp = [
        room,
        5,
        7,
        23,
        39,
        55,
        71 + seat,
        75 + chang,
        78 + ju,
        82 + min(max(left, 0), 69),
    ]
    for slot, d in enumerate(doras):
        sp.append(152 + slot * 37 + d)
    if closed37:
        for t37, cnt in sorted(closed37.items()):
            base = OFFSETS_136[t37]
            for c in range(cnt):
                sp.append(337 + base + c)
    if zimo < 37:
        sp.append(473 + zimo)
    while len(sp) < 29:
        sp.append(510)
    return sp[:29]


def _make_iql_batches(
    seed: int,
    *,
    n_batches: int = 4,
    batch_size: int = 2,
    final_rank: bool = True,
    use_results: bool = False,
) -> list[TensorDict]:
    torch.manual_seed(seed)
    closed = {
        1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1,
        21: 1, 22: 1, 23: 1, 31: 2, 32: 2,
    }
    sp = torch.tensor(
        [_make_sparse(seat=k % 4, left=44 - k, closed37=closed, zimo=32) for k in range(batch_size)],
        dtype=torch.int32,
    )
    nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]] * batch_size, dtype=torch.int32)
    pr = torch.tensor([[0, 16, 2165] + [2165] * 106] * batch_size, dtype=torch.int32)
    ca = torch.tensor([[4, 44, 84, 130] + [PAD_ACTION_ID] * 28] * batch_size, dtype=torch.int32)
    feats = derive_structured_v2_batch(sp, nu, pr, ca)
    batches: list[TensorDict] = []
    for bi in range(n_batches):
        done = torch.tensor([bi == n_batches - 1 and i == 0 for i in range(batch_size)], dtype=torch.bool)
        reward = torch.tensor([0.3 if d else 0.1 * (bi + 1) for d in done.tolist()], dtype=torch.float32)
        nxt: dict[str, torch.Tensor] = {
            "sparse": sp.clone(),
            "numeric": nu.clone(),
            "progression": pr.clone(),
            "candidates": ca.clone(),
            **{k: v.clone() for k, v in feats.items()},
            "done": done,
            "reward": reward,
            "end_of_game": done,
        }
        if final_rank:
            if use_results:
                res = torch.zeros((batch_size, 14), dtype=torch.int32)
                res[:, 10:14] = torch.tensor([41000, 29000, 19000, 11000], dtype=torch.int32)
                nxt["results"] = res
            else:
                nxt["final_rank"] = torch.tensor([k % 4 for k in range(batch_size)], dtype=torch.int64)
        batches.append(
            TensorDict(
                {
                    "sparse": sp.clone(),
                    "numeric": nu.clone(),
                    "progression": pr.clone(),
                    "candidates": ca.clone(),
                    **{k: v.clone() for k, v in feats.items()},
                    "action": torch.tensor([(bi + k) % 4 for k in range(batch_size)], dtype=torch.int32),
                    "next": TensorDict(nxt, batch_size=[batch_size]),
                },
                batch_size=[batch_size],
            )
        )
    return batches


def _snapshot_training_state(
    mod: torch.nn.Module,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, object]:
    params = {k: v.detach().clone() for k, v in mod.state_dict().items()}
    opt_state: dict[int, dict[str, object]] = {}
    for p_idx, p in enumerate(p for g in opt.param_groups for p in g["params"]):
        st = opt.state.get(p, {})
        opt_state[p_idx] = {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v))
            for k, v in st.items()
        }
    return {
        "params": params,
        "opt_state": opt_state,
        "sched_epoch": sched.last_epoch if sched is not None else None,
        "sched_lr": list(sched.get_last_lr()) if sched is not None else None,
    }


def _assert_snapshot_exact(
    tc: unittest.TestCase,
    snap: dict[str, object],
    mod: torch.nn.Module,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    cur_sd = mod.state_dict()
    for k, prev_t in snap["params"].items():  # type: ignore[union-attr]
        cur_t = cur_sd[k]
        tc.assertTrue(
            torch.equal(cur_t, prev_t),
            f"Parameter/Target {k} mutated during invalid step!",
        )
    prev_opt = snap["opt_state"]
    for p_idx, p in enumerate(p for g in opt.param_groups for p in g["params"]):
        cur_st = opt.state.get(p, {})
        prev_st = prev_opt[p_idx]  # type: ignore[index]
        tc.assertEqual(set(cur_st.keys()), set(prev_st.keys()))
        for sk, sv in prev_st.items():
            cv = cur_st[sk]
            if isinstance(sv, torch.Tensor):
                tc.assertTrue(
                    torch.equal(cv, sv),
                    f"Optimizer state[{p_idx}][{sk}] mutated during invalid step!",
                )
            else:
                tc.assertEqual(cv, sv)
    if sched is not None:
        tc.assertEqual(sched.last_epoch, snap["sched_epoch"])
        tc.assertEqual(list(sched.get_last_lr()), snap["sched_lr"])


def _resolve_windows_gloo_ifname() -> str | None:
    if sys.platform != "win32" or "GLOO_SOCKET_IFNAME" in os.environ:
        return os.environ.get("GLOO_SOCKET_IFNAME")
    try:
        host_ip = socket.gethostbyname(socket.gethostname())
        raw = subprocess.check_output(["ipconfig"], text=True, errors="replace")
    except Exception:
        return None
    cur_if: str | None = None
    for line in raw.splitlines():
        s = line.strip()
        if line and not line[0].isspace() and s.endswith(":"):
            m = re.search(r"(?:adapter|适配器)\s+(.+):$", s, flags=re.IGNORECASE)
            cur_if = m.group(1).strip() if m else s[:-1].strip()
        elif host_ip in s and cur_if:
            return cur_if
    return None


def _ddp_numerical_worker(rank: int, store_uri: str, tmp_dir: str) -> None:
    os.environ["USE_LIBUV"] = "0"
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=store_uri,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        dev = torch.device("cpu")
        torch.manual_seed(9001)
        mod = JointStructuredIQLModule(
            preset="v2_tiny",
            dimension=64,
            num_heads=4,
            dim_feedforward=128,
            n_d1_layers=1,
            n_d2_layers=1,
            use_cql_quantile_head=True,
            num_quantiles=8,
            target_ema_tau=0.25,
            device=dev,
            dtype=torch.float32,
        )
        opt = build_joint_iql_optimizer(mod, base_lr=1e-3, weight_decay=0.01, device=dev)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        recipe = build_joint_iql_recipe(mod, base_lr=1e-3, weight_decay=0.01, decoder1_lr_scale=0.25)
        ddp = DistributedDataParallel(mod)
        batches = _make_iql_batches(55 + rank, n_batches=4, batch_size=2, final_rank=False)

        # Step 0: clean warmup step on both ranks
        res0 = train_joint_iql_batches(
            mod, batches[:1], opt, recipe=recipe, device=dev, scheduler=sched, ddp_wrapped=ddp
        )
        assert res0["optimizer_steps"] == 1 and res0["skipped_steps"] == 0

        snap_before = {k: v.detach().clone() for k, v in mod.state_dict().items()}

        # Rank 1 ONLY gets NaN loss on first batch (cur_step=0) and Inf grad on second (cur_step=1); step 3 is finite on both!
        def _loss_xform(loss: Tensor, cur_step: int) -> Tensor:
            if rank == 1 and cur_step == 0:
                return loss * float("nan")
            return loss

        def _post_bwd(runner: torch.nn.Module, o: torch.optim.Optimizer, cur_step: int) -> None:
            del o
            if rank == 1 and cur_step == 1:
                raw_m = runner.module if isinstance(runner, DistributedDataParallel) else runner
                assert isinstance(raw_m, JointStructuredIQLModule)
                raw_m.q1_head.v_fc2.weight.grad.fill_(float("inf"))

        res_bad = train_joint_iql_batches(
            mod,
            batches[1:3],
            opt,
            recipe=recipe,
            device=dev,
            scheduler=sched,
            ddp_wrapped=ddp,
            loss_transform=_loss_xform,
            post_backward_hook=_post_bwd,
        )
        assert res_bad["optimizer_steps"] == 0, res_bad
        assert res_bad["skipped_steps"] == 2, res_bad
        assert res_bad["nonfinite_loss_steps"] == 1, res_bad
        assert res_bad["nonfinite_grad_steps"] == 2, res_bad  # NaN loss backprop also yields NaN grad
        assert sched.last_epoch == 1
        for k, v in mod.state_dict().items():
            assert torch.equal(v, snap_before[k]), f"Rank {rank} param {k} mutated on peer failure!"

        # Rank 0 & Rank 1 remain completely identical after peer-skipped steps, then recover cleanly on step 3
        res_rec = train_joint_iql_batches(
            mod, batches[3:4], opt, recipe=recipe, device=dev, scheduler=sched, ddp_wrapped=ddp
        )
        assert res_rec["optimizer_steps"] == 1 and res_rec["skipped_steps"] == 0
        assert sched.last_epoch == 2

        flat = torch.cat([p.detach().flatten() for p in mod.state_dict().values()])
        ref = flat.clone()
        dist.broadcast(ref, src=0)
        torch.testing.assert_close(flat, ref, atol=0.0, rtol=0.0)
        (Path(tmp_dir) / f"ddp_ok_{rank}.txt").write_text("ok", encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _extract_complete_games(max_games: int = 3, max_scan_lines: int = 20000) -> list[str]:
    """Extract complete game records from validation.bc.tsv.gz without mid-game truncation."""
    by_uuid: dict[str, list[str]] = {}
    last_seen: dict[str, int] = {}
    with gzip.open(VAL_DATA_PATH, "rt", encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            if i >= max_scan_lines:
                break
            u = line.split("\t", 1)[0]
            by_uuid.setdefault(u, []).append(line)
            last_seen[u] = i
    out: list[str] = []
    picked = 0
    cutoff = max_scan_lines - 5000
    for u, lines in by_uuid.items():
        if last_seen[u] < cutoff and len(lines) >= 80:
            out.extend(lines)
            picked += 1
            if picked >= max_games:
                break
    return out


class TestPreUpdateNumericalSafety(unittest.TestCase):
    def _build_iql(self, dev: torch.device, seed: int = 101):
        torch.manual_seed(seed)
        mod = JointStructuredIQLModule(
            preset="v2_tiny",
            dimension=64,
            num_heads=4,
            dim_feedforward=128,
            n_d1_layers=1,
            n_d2_layers=1,
            use_cql_quantile_head=True,
            num_quantiles=8,
            final_rank=True,
            kl_coef=0.05,
            target_ema_tau=0.25,
            device=dev,
            dtype=torch.float32,
        )
        bc_src = JointStructuredIQLModule(
            preset="v2_tiny",
            dimension=64,
            num_heads=4,
            dim_feedforward=128,
            n_d1_layers=1,
            n_d2_layers=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        mod.init_from_bc(bc_src.backbone, bc_src.policy_head, keep_bc_anchor=True)
        mod.configure_freeze(final_rank_weight=0.10)
        opt = build_joint_iql_optimizer(mod, base_lr=1e-3, weight_decay=0.01, device=dev)
        rec = build_joint_iql_recipe(mod, base_lr=1e-3, weight_decay=0.01, decoder1_lr_scale=0.25)
        return mod, opt, rec

    def _run_injection_suite(
        self,
        dev: torch.device,
        amp_dtype: torch.dtype | None,
        with_scheduler: bool,
    ) -> None:
        batches = _make_iql_batches(202, n_batches=5, batch_size=2, final_rank=True, use_results=True)
        modes = (
            ("nan_loss", lambda loss, _s: loss * float("nan"), None, 1, 1),
            ("inf_loss", lambda loss, _s: loss + float("inf"), None, 1, 0),
            (
                "nan_grad_and_clip",
                None,
                lambda r, _o, _s: (
                    r.q1_head.v_fc2.weight.grad.fill_(float("nan")),
                    r.policy_head.mlp[0].weight.grad.fill_(50.0),
                ),
                0,
                1,
            ),
            (
                "inf_grad_multi_head",
                None,
                lambda r, _o, _s: (
                    r.q2_head.v_fc2.weight.grad.fill_(float("inf")),
                    r.quantile_q_head.fc2.weight.grad.fill_(float("-inf")),
                ),
                0,
                1,
            ),
            (
                "pre_target_ema_backbone_nan",
                None,
                lambda r, _o, _s: r.backbone.tile_emb.suit_emb.weight.grad.fill_(float("nan")),
                0,
                1,
            ),
        )
        for name, loss_fn, grad_fn, exp_loss_nf, exp_grad_nf in modes:
            with self.subTest(device=str(dev), amp=str(amp_dtype), sched=with_scheduler, case=name):
                mod, opt, rec = self._build_iql(dev, seed=303)
                sched = (
                    torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.8)
                    if with_scheduler
                    else None
                )
                is_fp16 = amp_dtype == torch.float16 and dev.type == "cuda"
                scaler = GradScaler("cuda", init_scale=256.0, enabled=is_fp16)

                # Warmup 2 steps so moments, Target EMA lag, and LR scheduler are non-trivial.
                res_warm = train_joint_iql_batches(
                    mod,
                    batches[:2],
                    opt,
                    recipe=rec,
                    device=dev,
                    amp_dtype=amp_dtype,
                    scheduler=sched,
                    grad_scaler=scaler,
                )
                self.assertEqual(res_warm["optimizer_steps"], 2)
                self.assertEqual(res_warm["skipped_steps"], 0)
                snap = _snapshot_training_state(mod, opt, sched)
                scale_before = scaler.get_scale() if scaler.is_enabled() else None

                res_bad = train_joint_iql_batches(
                    mod,
                    batches[2:3],
                    opt,
                    recipe=rec,
                    device=dev,
                    amp_dtype=amp_dtype,
                    max_grad_norm=1.0,
                    scheduler=sched,
                    grad_scaler=scaler,
                    loss_transform=loss_fn,
                    post_backward_hook=grad_fn,
                )
                self.assertEqual(res_bad["optimizer_steps"], 0)
                self.assertEqual(res_bad["skipped_steps"], 1)
                self.assertGreaterEqual(res_bad["nonfinite_loss_steps"], exp_loss_nf)
                self.assertGreaterEqual(res_bad["nonfinite_grad_steps"], exp_grad_nf)
                _assert_snapshot_exact(self, snap, mod, opt, sched)
                if scaler.is_enabled():
                    self.assertLess(scaler.get_scale(), scale_before)  # type: ignore[arg-type]

                # Subsequent finite step recovers cleanly and updates model & target EMA!
                res_rec = train_joint_iql_batches(
                    mod,
                    batches[3:4],
                    opt,
                    recipe=rec,
                    device=dev,
                    amp_dtype=amp_dtype,
                    scheduler=sched,
                    grad_scaler=scaler,
                )
                self.assertEqual(res_rec["optimizer_steps"], 1)
                self.assertEqual(res_rec["skipped_steps"], 0)
                self.assertFalse(
                    torch.equal(mod.q1_head.v_fc2.weight.detach(), snap["params"]["q1_head.v_fc2.weight"])  # type: ignore[index]
                )
                self.assertFalse(
                    torch.equal(
                        mod.quantile_q_target_head.fc2.weight.detach(),
                        snap["params"]["quantile_q_target_head.fc2.weight"],  # type: ignore[index]
                    )
                )

    def test_cpu_numerical_safety_and_recovery(self):
        self._run_injection_suite(torch.device("cpu"), None, with_scheduler=True)
        self._run_injection_suite(torch.device("cpu"), None, with_scheduler=False)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_fp32_bf16_fp16_overflow_and_zero_sync(self):
        dev = torch.device("cuda")
        self._run_injection_suite(dev, torch.float32, with_scheduler=True)
        self._run_injection_suite(dev, torch.bfloat16, with_scheduler=False)
        self._run_injection_suite(dev, torch.bfloat16, with_scheduler=True)
        self._run_injection_suite(dev, torch.float16, with_scheduler=False)
        self._run_injection_suite(dev, torch.float16, with_scheduler=True)

        # Verify CUDA fused guard runs intermediate normal + injected bad steps with <= 1 host sync!
        prev_det = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        try:
            batches = _make_iql_batches(404, n_batches=6, batch_size=2, final_rank=True, use_results=True)
            for dtype in (torch.bfloat16, torch.float16):
                mod, opt, rec = self._build_iql(dev, seed=405)
                scaler = GradScaler("cuda", init_scale=1024.0, enabled=(dtype == torch.float16))
                # Warmup 1 step outside debug capture
                train_joint_iql_batches(
                    mod, batches[:1], opt, recipe=rec, device=dev, amp_dtype=dtype, grad_scaler=scaler
                )
                snap = _snapshot_training_state(mod, opt)

                def _bad_on_step2(r, _o, s):
                    if s == 1:
                        r.q1_head.v_fc2.weight.grad.fill_(float("inf"))

                def _nan_loss_step1(loss, s):
                    return (loss * float("nan")) if s == 0 else loss

                # First test exact integrity across steps 1 & 2 (all invalid) with fused CUDA zero-sync guard
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode(1)
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        res_bad = train_joint_iql_batches(
                            mod,
                            batches[1:3],
                            opt,
                            recipe=rec,
                            device=dev,
                            amp_dtype=dtype,
                            grad_scaler=scaler,
                            log_interval=100,
                            check_finite_interval=100,
                            loss_transform=_nan_loss_step1,
                            post_backward_hook=_bad_on_step2,
                        )
                        torch.cuda.synchronize()
                finally:
                    torch.cuda.set_sync_debug_mode(0)

                sync_warns = [w for w in caught if "synchronizing CUDA operation" in str(w.message)]
                self.assertLessEqual(len(sync_warns), 1, [str(w.message) for w in sync_warns])
                self.assertEqual(res_bad["skipped_steps"], 2)
                self.assertEqual(res_bad["optimizer_steps"], 0)
                _assert_snapshot_exact(self, snap, mod, opt)
        finally:
            torch.use_deterministic_algorithms(prev_det)

    def test_checkpoint_recovery_with_skipped_steps_and_max_nonfinite_limit(self):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        amp = torch.float16 if dev.type == "cuda" else None
        batches = _make_iql_batches(505, n_batches=5, batch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            mod1, opt1, rec1 = self._build_iql(dev, seed=506)
            sc1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=1, gamma=0.9)
            gs1 = GradScaler("cuda", init_scale=256.0, enabled=(dev.type == "cuda"))

            # Batch 0 clean, Batch 1 NaN loss -> checkpoint at step 2 with skipped_steps=1, optimizer_steps=1
            res1 = train_joint_iql_batches(
                mod1,
                batches[:2],
                opt1,
                recipe=rec1,
                device=dev,
                amp_dtype=amp,
                scheduler=sc1,
                grad_scaler=gs1,
                ckpt_dir=ckpt_dir,
                checkpoint_every=2,
                loss_transform=lambda loss, s: (loss * float("nan")) if s == 1 else loss,
            )
            self.assertEqual(res1["optimizer_steps"], 1)
            self.assertEqual(res1["skipped_steps"], 1)

            # Compare continuous vs resumed across batches[2:5]
            snap_mid = _snapshot_training_state(mod1, opt1, sc1)
            scale_mid = gs1.get_scale() if gs1.is_enabled() else None
            train_joint_iql_batches(
                mod1, batches[2:5], opt1, recipe=rec1, device=dev, amp_dtype=amp, scheduler=sc1, grad_scaler=gs1
            )

            mod2, opt2, rec2 = self._build_iql(dev, seed=999)
            sc2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.9)
            gs2 = GradScaler("cuda", init_scale=4096.0, enabled=(dev.type == "cuda"))
            restored_stats: dict[str, int] = {}
            step_r, cursor_r = load_joint_iql_checkpoint(
                ckpt_dir / "joint-iql-2.pt",
                mod2,
                opt2,
                expected_recipe=rec2,
                scheduler=sc2,
                grad_scaler=gs2,
                stats_out=restored_stats,
            )
            self.assertEqual((step_r, cursor_r), (2, 2))
            self.assertEqual(restored_stats["optimizer_steps"], 1)
            self.assertEqual(restored_stats["skipped_steps"], 1)
            _assert_snapshot_exact(self, snap_mid, mod2, opt2, sc2)
            if gs2.is_enabled():
                self.assertEqual(gs2.get_scale(), scale_mid)

            res2 = train_joint_iql_batches(
                mod2,
                batches,
                opt2,
                recipe=rec2,
                device=dev,
                amp_dtype=amp,
                scheduler=sc2,
                grad_scaler=gs2,
                resume_from=ckpt_dir / "joint-iql-2.pt",
            )
            self.assertEqual(res2["optimizer_steps"], 4)
            self.assertEqual(res2["skipped_steps"], 1)
            for k, v in mod1.state_dict().items():
                torch.testing.assert_close(v, mod2.state_dict()[k], atol=0.0, rtol=0.0)

            # max_nonfinite_steps raises AFTER preserving parameter & moment integrity
            mod3, opt3, rec3 = self._build_iql(dev, seed=607)
            train_joint_iql_batches(mod3, batches[:1], opt3, recipe=rec3, device=dev)
            snap3 = _snapshot_training_state(mod3, opt3)
            with self.assertRaises(RuntimeError):
                train_joint_iql_batches(
                    mod3,
                    batches[1:2],
                    opt3,
                    recipe=rec3,
                    device=dev,
                    max_nonfinite_steps=0,
                    loss_transform=lambda loss, _s: loss * float("nan"),
                )
            _assert_snapshot_exact(self, snap3, mod3, opt3)

    def test_trajectory_bc_pre_update_numerical_guard(self):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        closed = {1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1, 21: 1, 22: 1, 23: 1, 31: 2, 32: 2}
        sp = torch.tensor([_make_sparse(seat=0, left=44 - k, closed37=closed, zimo=32) for k in range(4)], dtype=torch.int32)
        nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]] * 4, dtype=torch.int32)
        pr = torch.tensor([[0, 16, 2165] + [2165] * 106, [0, 16, 28, 2165] + [2165] * 105] * 2, dtype=torch.int32)
        ca = torch.tensor([[4, 44, 84, 130] + [PAD_ACTION_ID] * 28] * 4, dtype=torch.int32)
        rounds = build_round_trajectories_from_raw(
            sp, nu, pr, ca, torch.tensor([0, 1, 2, 3]), ["r0", "r0", "r1", "r1"],
            final_ranks=torch.tensor([0, 0, 2, 2]),
        )
        torch.manual_seed(707)
        bm = TrajectoryBCModel(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=1, n_d2_layers=1, use_next_event_head=True, use_final_rank_head=True,
            final_rank_weight=0.1, device=dev, dtype=torch.float32,
        )
        bopt = torch.optim.AdamW(adamw_parameter_groups(bm, 0.01), lr=1e-3, fused=(dev.type == "cuda"))
        bsched = torch.optim.lr_scheduler.StepLR(bopt, step_size=1, gamma=0.5)
        train_trajectory_bc_steps(bm, rounds, bopt, batch_size=2, max_steps=1, device=dev, scheduler=bsched)
        snap = _snapshot_training_state(bm, bopt, bsched)

        res_bad = train_trajectory_bc_steps(
            bm,
            rounds,
            bopt,
            batch_size=2,
            max_steps=1,
            device=dev,
            scheduler=bsched,
            post_backward_hook=lambda m, _o, _s: m.policy_head.mlp[0].weight.grad.fill_(float("inf")),
        )
        self.assertEqual(res_bad["skipped_steps"], 1)
        self.assertEqual(res_bad["optimizer_steps"], 0)
        self.assertEqual(res_bad["nonfinite_grad_steps"], 1)
        _assert_snapshot_exact(self, snap, bm, bopt, bsched)

    def test_ddp_2rank_numerical_failure_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = dict(
                os.environ,
                PYTHONPATH=str(REPO_ROOT),
                TORCHDYNAMO_DISABLE="1",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                USE_LIBUV="0",
                PYTHONIOENCODING="utf-8",
            )
            ifname = _resolve_windows_gloo_ifname()
            if ifname:
                env["GLOO_SOCKET_IFNAME"] = ifname
            store = (Path(tmp) / f"ddp_num_{port}").resolve().as_uri()
            procs = [
                subprocess.Popen(
                    [sys.executable, "-B", str(Path(__file__).resolve()), "--ddp-audit-worker", str(r), store, tmp],
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                for r in range(2)
            ]
            for r, p in enumerate(procs):
                out, err = p.communicate(timeout=45)
                self.assertEqual(p.returncode, 0, f"Rank {r} failed:\n{out}\n{err}")
                self.assertEqual((Path(tmp) / f"ddp_ok_{r}.txt").read_text(), "ok")


class TestDistributionalQRDQNMath(unittest.TestCase):
    def test_closed_form_analytical_qr_huber_and_zero_target_grad(self):
        # Analytical M=2 pairwise Quantile Huber hand calculation for kappa=1.0 and kappa=0.0.
        # q_chosen = [0.0, 4.0], tgt = [2.0, 6.0]
        # delta[i, j] = tgt[j] - q[i] = [[+2, +6], [-2, +2]]
        # taus = [0.25, 0.75]
        # weights = |tau[i] - 1(delta<0)| = [[0.25, 0.25], [0.25, 0.75]]
        # huber(kappa=1): abs_d - 0.5 -> [[1.5, 5.5], [1.5, 1.5]]
        # weighted[0] = [0.375, 1.375] (mean over j = 0.875)
        # weighted[1] = [0.375, 1.125] (mean over j = 0.750)
        # sum over i = 1.625 exact!
        cands = torch.full((1, 32), PAD_ACTION_ID, dtype=torch.int32)
        cands[0, 0] = 15  # K = 1 forced step -> CQL term must be strictly 0.0!
        q_raw = torch.zeros((1, 32, 2), dtype=torch.float32, requires_grad=True)
        q_in = q_raw.clone()
        with torch.no_grad():
            pass
        # Inject exact q_chosen=[0, 4] via differentiable tensor
        pred_vals = torch.tensor([0.0, 4.0], dtype=torch.float32, requires_grad=True)
        q_full = torch.zeros((1, 32, 2), dtype=torch.float32)
        q_full[0, 0] = pred_vals
        tgt = torch.tensor([[2.0, 6.0]], dtype=torch.float32, requires_grad=True)

        loss_k1, m_k1 = quantile_cql_loss(q_full, cands, torch.tensor([0]), tgt, alpha=0.5, kappa=1.0)
        self.assertEqual(float(m_k1["cql_loss"].item()), 0.0)
        self.assertAlmostEqual(float(loss_k1.item()), 1.625, places=6)
        loss_k1.backward()
        # Target tensor has requires_grad=True and MUST receive ZERO gradients!
        self.assertIsNone(tgt.grad)
        self.assertIsNotNone(pred_vals.grad)
        # dL/dq_0 = mean_j(-w_0j * sign(delta_0j)) = mean(-0.25, -0.25) = -0.25
        # dL/dq_1 = mean_j(-w_1j * sign(delta_1j)) = mean(+0.25, -0.75) = -0.25
        torch.testing.assert_close(pred_vals.grad, torch.tensor([-0.25, -0.25]), atol=1e-6, rtol=1e-6)

        # Invalid hyperparams fail loudly (NaN / Inf / negative)
        for bad_kw in (
            {"kappa": float("nan")},
            {"kappa": float("inf")},
            {"kappa": -0.1},
            {"alpha": -0.1},
            {"temperature": 0.0},
            {"temperature": float("nan")},
        ):
            with self.assertRaises(ValueError):
                quantile_cql_loss(q_full.detach(), cands, torch.tensor([0]), tgt.detach(), **bad_kw)

    def test_analytical_double_q_target_zero_discount_terminal_and_pad_isolation(self):
        # Online selector picks slot 1 (bimodal target {-4, -4, +4, +4}, mean 0), whereas target Q peaks on slot 0 ({+1, +1, +1, +1})
        reward = torch.tensor([2.0, -3.0, 1.25], dtype=torch.float32)
        done = torch.tensor([False, True, False], dtype=torch.bool)
        discount = torch.tensor([0.5, 0.99, 0.0], dtype=torch.float32)
        next_cands = torch.full((3, 32), PAD_ACTION_ID, dtype=torch.int32)
        next_cands[0, :2] = torch.tensor([30, 10], dtype=torch.int32)
        next_cands[2, :2] = torch.tensor([5, 6], dtype=torch.int32)

        tgt_q = torch.zeros((3, 32, 4), dtype=torch.float32, requires_grad=True)
        sel_q = torch.zeros((3, 32, 4), dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            tgt_q[0, 0] = torch.tensor([1.0, 1.0, 1.0, 1.0])
            tgt_q[0, 1] = torch.tensor([-4.0, -4.0, 4.0, 4.0])
            sel_q[0, 0] = torch.tensor([-2.0, -2.0, -2.0, -2.0])
            sel_q[0, 1] = torch.tensor([3.0, 3.0, 3.0, 3.0])
            tgt_q[1, 31] = float("inf")  # poisoned terminal row
            tgt_q[2, 0] = torch.tensor([99.0, 99.0, 99.0, 99.0])  # discount=0 row

        out_dq = compute_qr_dqn_target_quantiles(reward, done, discount, next_cands, tgt_q, sel_q)
        self.assertFalse(out_dq.requires_grad)
        torch.testing.assert_close(out_dq[0], torch.tensor([0.0, 0.0, 4.0, 4.0]))
        torch.testing.assert_close(out_dq[1], torch.tensor([-3.0, -3.0, -3.0, -3.0]))
        torch.testing.assert_close(out_dq[2], torch.tensor([1.25, 1.25, 1.25, 1.25]))

        # Single Q selects slot 0 (mean +1.0 > 0.0) -> 2.0 + 0.5*[1,1,1,1] = 2.5
        out_sq = compute_qr_dqn_target_quantiles(reward, done, discount, next_cands, tgt_q, None)
        torch.testing.assert_close(out_sq[0], torch.tensor([2.5, 2.5, 2.5, 2.5]))

        # Module forward with discount_factor=0.0 AND sentinel next_obs must never read sentinel indices!
        cpu = torch.device("cpu")
        mod = JointStructuredIQLModule(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=1, n_d2_layers=1, use_cql_quantile_head=True, num_quantiles=8,
            discount_factor=0.0, quantile_cql_alpha=0.1, cql_temperature=2.0,
            device=cpu, dtype=torch.float32,
        )
        b = _make_iql_batches(818, n_batches=1, batch_size=2)[0]
        b["next", "sparse"].fill_(510)
        b["next", "candidates"].fill_(PAD_ACTION_ID)
        b["next", "done"].fill_(False)
        out = mod(b)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()

        # Also test explicit per-row next.discount=0.0 with done=False and sentinel next indices
        mod_nz = JointStructuredIQLModule(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=1, n_d2_layers=1, use_cql_quantile_head=True, num_quantiles=8,
            discount_factor=0.99, device=cpu, dtype=torch.float32,
        )
        b_disc0 = b.clone()
        b_disc0["next", "discount"] = torch.tensor([0.0, 0.0], dtype=torch.float32)
        out_d0 = mod_nz(b_disc0)
        self.assertTrue(torch.isfinite(out_d0["loss"]))

        # Recipe captures quantile_cql_alpha & cql_temperature fail-closed
        rec = build_joint_iql_recipe(mod, base_lr=1e-3, weight_decay=0.0, decoder1_lr_scale=0.25)
        self.assertEqual(rec["iql"]["quantile_cql_alpha"], 0.1)
        self.assertEqual(rec["iql"]["cql_temperature"], 2.0)

        # Quantile Target EMA valid mask guard
        with torch.no_grad():
            mod.quantile_q_head.fc2.weight.add_(1.0)
        tgt_before = mod.quantile_q_target_head.fc2.weight.detach().clone()
        mod.update_targets(valid=torch.tensor(False))
        self.assertTrue(torch.equal(mod.quantile_q_target_head.fc2.weight, tgt_before))
        mod.update_targets(valid=torch.tensor(True))
        self.assertFalse(torch.equal(mod.quantile_q_target_head.fc2.weight, tgt_before))


class TestCompleteRoundTrajectoryReconstruction(unittest.TestCase):
    @unittest.skipUnless(VAL_DATA_PATH.is_file(), "validation.bc.tsv.gz required")
    def test_real_complete_games_strict_and_adversarial_corruption_rejection(self):
        lines = _extract_complete_games(max_games=3, max_scan_lines=25000)
        self.assertGreater(len(lines), 150)

        # Parse individual UUID groups so we can isolate 1 complete game cleanly for adversarial edits
        by_u: dict[str, list[str]] = {}
        for ln in lines:
            by_u.setdefault(ln.split("\t", 1)[0], []).append(ln)
        one_uuid = next(iter(by_u))
        game_lines = list(by_u[one_uuid])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            all_p = root / "complete_3games.tsv"
            all_p.write_text("".join(lines), encoding="utf-8")
            rounds_all = parse_bc_rounds(all_p)  # strict default: allow_incomplete=False, allow_unlabeled=False
            self.assertGreater(len(rounds_all), 20)
            self.assertEqual(sum(int(r.action_idx.numel()) for r in rounds_all), len(lines))

            # Cross-check FinalRank at every decision step against CoreBCDataIterator + scores_to_ranks
            it = CoreBCDataIterator(
                path=all_p, num_skip_samples=0, rewrite_rooms=None, rewrite_grades=None, local_rank=0
            )
            step_count_by_uuid_seat: dict[tuple[str, int], int] = {}
            rank_by_uuid_seat: dict[tuple[str, int], int] = {}
            for ln in lines:
                uuid = ln.split("\t", 1)[0]
                sp, _nu, _pr, _ca, _act, _rs, res = it.parse_line(ln)
                seat = int(sp[6].item()) - 71
                step_count_by_uuid_seat[(uuid, seat)] = step_count_by_uuid_seat.get((uuid, seat), 0) + 1
                rank_by_uuid_seat[(uuid, seat)] = int(
                    scores_to_ranks(res[10:14].unsqueeze(0))[0, seat].item()
                )
            del it
            self.assertEqual(sum(step_count_by_uuid_seat.values()), len(lines))

            # Prepare sorted physical rounds for single game `one_uuid`
            round_keys = []
            for ln in game_lines:
                p = ln.rstrip("\r\n").split("\t")
                sp_raw = [int(x) for x in p[1].split(",")]
                nu_raw = [int(x) for x in p[2].split(",")]
                pr_raw = [int(x) for x in p[3].split(",")]
                rs_raw = [int(x) for x in p[6].split(",")]
                rk = (sp_raw[7] - 75, sp_raw[8] - 78, nu_raw[0])
                round_keys.append((rk, len(pr_raw), sp_raw[6] - 71, pr_raw, rs_raw, ln))

            # 1. Truncated game (drop last physical round of complete game) -> MUST reject!
            max_rk = max(x[0] for x in round_keys)
            trunc_game_lines = [x[5] for x in round_keys if x[0] != max_rk]
            p_tg = root / "trunc_game.tsv"
            p_tg.write_text("".join(trunc_game_lines), encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_bc_rounds(p_tg)

            # 2. Drop opening round (0, 0, 0) -> MUST reject!
            p_no_open = root / "no_open.tsv"
            p_no_open.write_text("".join(x[5] for x in round_keys if x[0] != (0, 0, 0)), encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_bc_rounds(p_no_open)

            # 3. Drop dealer opening decision step (pr_len == 1, seat == ju) inside round (0, 0, 0) -> MUST reject!
            r0_lines = [x for x in round_keys if x[0] == (0, 0, 0)]
            drop_first = [
                ln for ln in game_lines
                if not (
                    int(ln.split("\t")[1].split(",")[7]) - 75 == 0
                    and int(ln.split("\t")[1].split(",")[8]) - 78 == 0
                    and int(ln.split("\t")[2].split(",")[0]) == 0
                    and len(ln.split("\t")[3].split(",")) == 1
                )
            ]
            self.assertLess(len(drop_first), len(game_lines))
            p_df = root / "drop_first.tsv"
            p_df.write_text("".join(drop_first), encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_bc_rounds(p_df)

            # 4. Drop middle non-riichi, non-SKIP step of round (0, 0, 0) -> MUST reject!
            r0_non_skip = [
                x for x in r0_lines
                if x[1] > 1 and int(x[5].split("\t")[4].split(",")[int(x[5].split("\t")[5])]) != 221
            ]
            self.assertTrue(r0_non_skip)
            r0_mid = sorted(r0_non_skip, key=lambda x: x[1])[len(r0_non_skip) // 2]
            drop_mid = [ln for ln in game_lines if ln != r0_mid[5]]
            p_dm = root / "drop_mid.tsv"
            p_dm.write_text("".join(drop_mid), encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_bc_rounds(p_dm)

            # 5. Drop trailing winning step in a Win round -> MUST reject!
            win_rk = next(
                (
                    rk for rk in sorted({x[0] for x in round_keys if x[4][0] <= 15})
                    if max(x[1] for x in round_keys if x[0] == rk) > 1
                ),
                None,
            )
            if win_rk is not None:
                win_max_pr = max(x[1] for x in round_keys if x[0] == win_rk)
                drop_win_tail = [
                    x[5] for x in round_keys
                    if not (x[0] == win_rk and x[1] == win_max_pr)
                ]
                p_dw = root / "drop_win_tail.tsv"
                p_dw.write_text("".join(drop_win_tail), encoding="utf-8")
                with self.assertRaises(ValueError):
                    parse_bc_rounds(p_dw)

            # 6. Duplicate decision line -> MUST reject!
            dup_p = root / "dup.tsv"
            dup_p.write_text("".join(game_lines + [r0_lines[0][5]]), encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_bc_rounds(dup_p)

            # 7. Same-length divergent progression -> MUST reject!
            same_seat_steps = [x for x in r0_lines if x[1] >= 4]
            self.assertTrue(same_seat_steps)
            target_x = same_seat_steps[0]
            parts_div = target_x[5].rstrip("\r\n").split("\t")
            pr_div = [int(z) for z in parts_div[3].split(",")]
            pr_div[1] = 5 if pr_div[1] != 5 else 9  # different tile token, exact same progression length!
            parts_div[3] = ",".join(map(str, pr_div))
            div_line = "\t".join(parts_div) + "\n"
            div_p = root / "div.tsv"
            div_p.write_text("".join(game_lines + [div_line]), encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_bc_rounds(div_p)

            # 8. Cross-seat final score mismatch in same UUID -> MUST reject!
            seat1_idx = next(i for i, ln in enumerate(game_lines) if int(ln.split("\t")[1].split(",")[6]) == 72)
            bad_sc_lines = list(game_lines)
            p_sc = bad_sc_lines[seat1_idx].rstrip("\r\n").split("\t")
            r_sc = [int(z) for z in p_sc[7].split(",")]
            r_sc[10] += 100
            p_sc[7] = ",".join(map(str, r_sc))
            bad_sc_lines[seat1_idx] = "\t".join(p_sc) + "\n"
            cs_p = root / "cross_seat.tsv"
            cs_p.write_text("".join(bad_sc_lines), encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_bc_rounds(cs_p)

            # 9. Mixed labeled (8-col) & unlabeled (6-col) rows -> MUST reject even with allow_unlabeled=True!
            mix_lines = list(game_lines[:10])
            p_6 = game_lines[10].rstrip("\r\n").split("\t")[:6]
            mix_lines.append("\t".join(p_6) + "\n")
            mix_p = root / "mixed_cols.tsv"
            mix_p.write_text("".join(mix_lines), encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_bc_rounds(mix_p, allow_unlabeled=True, allow_incomplete=True)

            # 10. Real game split at round boundary & truncated by max_lines must reject incomplete!
            trunc_lines_p = root / "partial.tsv"
            trunc_lines_p.write_text("".join(game_lines[: len(game_lines) // 2]), encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_bc_rounds(trunc_lines_p)

    def test_raw_trajectory_mixed_labels_duplicate_and_final_rank_weight_guard(self):
        closed = {1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1, 21: 1, 22: 1, 23: 1, 31: 2, 32: 2}
        sp = torch.tensor([_make_sparse(seat=0, left=44 - k, closed37=closed, zimo=32) for k in range(4)], dtype=torch.int32)
        nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]] * 4, dtype=torch.int32)
        pr = torch.tensor(
            [
                [0, 16, 2165] + [2165] * 106,
                [0, 16, 28, 2165] + [2165] * 105,
                [0, 16, 2165] + [2165] * 106,
                [0, 16, 28, 2165] + [2165] * 105,
            ],
            dtype=torch.int32,
        )
        ca = torch.tensor([[4, 44, 84, 130] + [PAD_ACTION_ID] * 28] * 4, dtype=torch.int32)
        acts = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        # Mixed labeled & -1 across rounds r0 vs r1 must fail loudly
        with self.assertRaises(ValueError):
            build_round_trajectories_from_raw(
                sp, nu, pr, ca, acts, ["r0", "r0", "r1", "r1"],
                final_ranks=torch.tensor([1, 1, -1, -1]),
            )

        # Duplicate step in one round must fail loudly
        with self.assertRaises(ValueError):
            build_round_trajectories_from_raw(
                torch.stack([sp[0], sp[0]]),
                nu[:2],
                torch.stack([pr[0], pr[0]]),
                ca[:2],
                acts[:2],
                ["dup", "dup"],
            )

        # Unlabeled (-1) passed to TrajectoryBCModel with final_rank_weight > 0 must fail loudly
        rounds_unlab = build_round_trajectories_from_raw(sp[:2], nu[:2], pr[:2], ca[:2], acts[:2], ["r0", "r0"])
        bm = TrajectoryBCModel(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=1, n_d2_layers=1, use_next_event_head=False,
            use_final_rank_head=True, final_rank_weight=0.5,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        coll = collate_round_batch(rounds_unlab)
        with self.assertRaises(ValueError):
            bm(coll)


class TestParityExportInferenceAndSimulate(unittest.TestCase):
    @unittest.skipUnless(VAL_DATA_PATH.is_file(), "validation.bc.tsv.gz required")
    def test_real_rare_categories_cpp_py_parity_and_6x_suit_perm(self):
        uuids, sp, nu, pr, ca, cat_counts = harvest_rare_categories(
            VAL_DATA_PATH, scan_limit=120000, per_cat_target=32
        )
        self.assertGreater(len(uuids), 100)
        for req_cat in ("red5", "chi", "pon", "dmk", "ankan", "kakan", "dora2p", "riichi", "win", "west_or_all_last", "left0"):
            self.assertGreater(cat_counts.get(req_cat, 0), 0, f"Missing category {req_cat}")

        for hint in (False, True):
            max_diffs, mismatches = _compare_cpp_and_py(
                uuids, sp, nu, pr, ca, enable_early_shape_hint=hint, tag="audit_unit"
            )
            self.assertEqual(mismatches, [], "\n".join(mismatches))
            self.assertLess(max(max_diffs.values()), 1e-5, max_diffs)

        # Check all 6 suit permutations preserve C++ exact parity
        sub_n = min(64, len(uuids))
        base = derive_structured_v2_batch(
            sp[:sub_n], nu[:sub_n], pr[:sub_n], ca[:sub_n],
            enable_early_shape_hint=True, include_room_level=False,
        )
        for k in range(6):
            p_sp, p_nu, p_pr, p_ca, _ = permute_raw_four_fields(
                sp[:sub_n], nu[:sub_n], pr[:sub_n], ca[:sub_n], perm_idx=k
            )
            c_out = derive_structured_v2_batch(
                p_sp, p_nu, p_pr, p_ca, enable_early_shape_hint=True, include_room_level=False
            )
            t_out = permute_structured_batch(base, k)
            for tk in _AUDIT_TENSORS:
                torch.testing.assert_close(c_out[tk].float(), t_out[tk].float(), atol=1e-5, rtol=1e-5)

    def test_all_inference_paths_amp_disabled_d1_and_tie_break(self):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtypes = [torch.float32]
        if dev.type == "cuda":
            dtypes.extend([torch.bfloat16, torch.float16])

        closed = {1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1, 21: 1, 22: 1, 23: 1, 31: 2, 32: 2}
        sp = torch.tensor([_make_sparse(seat=0, left=44 - k, closed37=closed, zimo=32) for k in range(3)], dtype=torch.int32)
        nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]] * 3, dtype=torch.int32)
        pr = torch.tensor(
            [
                [0] + [2165] * 108,  # zero non-DORA events -> exercises no_event_seat_emb!
                [0, 16, 2165] + [2165] * 106,
                [0, 16, 28, 52, 2165] + [2165] * 104,
            ],
            dtype=torch.int32,
        )
        ca = torch.tensor([[4, 44, 84, 130] + [PAD_ACTION_ID] * 28] * 3, dtype=torch.int32)
        rounds = build_round_trajectories_from_raw(sp, nu, pr, ca, torch.tensor([0, 1, 2]), ["r0"] * 3)
        coll = collate_round_batch(rounds)

        for n_d1 in (0, 1):
            bb = StructuredDualDecoder(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=n_d1, n_d2_layers=1, device=dev, dtype=torch.float32,
            ).eval()
            head = StructuredActionHead(
                input_dimension=64, head_type="factorized", device=dev, dtype=torch.float32
            ).eval()
            ctx = DecisionContext(
                environment="audit", game="g0", seat=0, round_instance=(0, 0, 0), model_version=0
            )
            for dt in dtypes:
                sp_d = sp.to(dev)
                nu_d = nu.to(dev)
                pr_d = pr.to(dev)
                ca_d = ca.to(dev)
                coll_d = {k: v.to(dev) for k, v in coll.items()}
                feats_d = derive_structured_v2_batch(sp_d, nu_d, pr_d, ca_d)
                with torch.no_grad(), torch.autocast(
                    device_type=dev.type, dtype=dt, enabled=(dt != torch.float32), cache_enabled=False
                ):
                    enc_single = bb(sp_d, nu_d, pr_d, ca_d)
                    logits_single = head(ca_d, enc_single)
                    enc_round = bb.forward_round_trajectory(
                        coll_d["round_events"],
                        coll_d["round_event_len"],
                        coll_d["decision_event_len"],
                        coll_d["decision_mask"],
                        coll_d["v2_tile_feats"],
                        coll_d["v2_global_feats"],
                        coll_d["v2_player_feats"],
                        coll_d["v2_cand_ids"],
                        coll_d["v2_cand_mask"],
                        coll_d["v2_cand_mode"],
                        coll_d["v2_cand_dyn"],
                    )["encode"]
                    cache = StructuredKVCache()
                    cache2 = StructuredKVCache()
                    enc_cached_list = []
                    logits_cached_list = []
                    for t in range(3):
                        enc_t = bb.forward_step_cached(
                            context=ctx,
                            cache=cache,
                            tile_feats=feats_d["v2_tile_feats"][t : t + 1],
                            events=feats_d["v2_events"][t : t + 1],
                            event_len=feats_d["v2_event_len"][t : t + 1],
                            global_feats=feats_d["v2_global_feats"][t : t + 1],
                            player_feats=feats_d["v2_player_feats"][t : t + 1],
                            cand_ids=feats_d["v2_cand_ids"][t : t + 1],
                            cand_mask=feats_d["v2_cand_mask"][t : t + 1],
                            cand_mode=feats_d["v2_cand_mode"][t : t + 1],
                            cand_dyn=feats_d["v2_cand_dyn"][t : t + 1],
                        )
                        enc_cached_list.append(enc_t)
                        logits_cached_list.append(
                            cached_structured_decision(
                                bb,
                                head,
                                ctx,
                                cache2,
                                sp_d[t : t + 1],
                                nu_d[t : t + 1],
                                pr_d[t : t + 1],
                                ca_d[t : t + 1],
                            )
                        )
                    enc_cached = torch.cat(enc_cached_list, dim=0)
                    logits_cached = torch.cat(logits_cached_list, dim=0)

                tol = 2e-2 if dt == torch.bfloat16 else (5e-3 if dt == torch.float16 else 1e-5)
                torch.testing.assert_close(enc_single.float(), enc_round.float(), atol=tol, rtol=tol)
                torch.testing.assert_close(enc_single.float(), enc_cached.float(), atol=tol, rtol=tol)
                torch.testing.assert_close(logits_single.float(), logits_cached.float(), atol=tol, rtol=tol)

        # Tie-breaking determinism with non-ascending candidate slots: smallest global_id wins!
        argmax_dec = DecodeConverter("argmax", tie_break_by_action_id=True)
        tie_cands = torch.full((1, 32), PAD_ACTION_ID, dtype=torch.int32)
        tie_cands[0, :3] = torch.tensor([50, 12, 80], dtype=torch.int32)
        tie_scores = torch.full((1, 32), -torch.inf, dtype=torch.float32)
        tie_scores[0, :3] = torch.tensor([2.5, 2.5, 2.5], dtype=torch.float32)
        chosen_slot = int(argmax_dec(tie_cands, tie_scores)[0].item())
        self.assertEqual(chosen_slot, 1)
        self.assertEqual(int(tie_cands[0, chosen_slot].item()), 12)

    def test_cuda_actor_export_cpu_normalized_reload_and_native_simulate(self):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batches = _make_iql_batches(919, n_batches=2, batch_size=2)
        torch.manual_seed(920)
        jm = JointStructuredIQLModule(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=1, n_d2_layers=1, use_cql_quantile_head=True, num_quantiles=8,
            device=dev, dtype=torch.float32,
        )
        opt = build_joint_iql_optimizer(jm, base_lr=1e-3, weight_decay=0.01, device=dev)
        rec = build_joint_iql_recipe(jm, base_lr=1e-3, weight_decay=0.01, decoder1_lr_scale=0.25)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "export_audit"
            train_joint_iql_batches(
                jm, batches, opt, recipe=rec, device=dev,
                amp_dtype=(torch.bfloat16 if dev.type == "cuda" else None),
                ckpt_dir=ckpt_dir, checkpoint_every=2,
            )
            actor_path = ckpt_dir / "model.2.kanachan"
            raw_ckpt = torch.load(actor_path, map_location="cpu", weights_only=True)
            for k, t in raw_ckpt["state_dict"].items():
                if isinstance(t, torch.Tensor):
                    self.assertEqual(t.device.type, "cpu", f"Exported actor tensor {k} left on {t.device}!")

            # Also verify TrajectoryBCModel CPU-normalized export
            closed = {1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1, 21: 1, 22: 1, 23: 1, 31: 2, 32: 2}
            sp = torch.tensor([_make_sparse(seat=0, left=44 - k, closed37=closed, zimo=32) for k in range(2)], dtype=torch.int32)
            nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]] * 2, dtype=torch.int32)
            pr = torch.tensor([[0, 16, 2165] + [2165] * 106, [0, 16, 28, 2165] + [2165] * 105], dtype=torch.int32)
            ca = torch.tensor([[4, 44, 84, 130] + [PAD_ACTION_ID] * 28] * 2, dtype=torch.int32)
            rounds = build_round_trajectories_from_raw(sp, nu, pr, ca, torch.tensor([0, 1]), ["r0", "r0"])
            bm = TrajectoryBCModel(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, device=dev, dtype=torch.float32,
            )
            bc_actor_p = Path(tmp) / "bc_actor.kanachan"
            bm.dump_actor(bc_actor_p)
            raw_bc = torch.load(bc_actor_p, map_location="cpu", weights_only=True)
            for k, t in raw_bc["state_dict"].items():
                if isinstance(t, torch.Tensor):
                    self.assertEqual(t.device.type, "cpu", f"Exported BC actor tensor {k} left on {t.device}!")

            actor_cpu = load_model(actor_path, map_location="cpu").eval()
            with torch.inference_mode():
                sample = batches[0][:1].clone().to("cpu")
                out_td = actor_cpu(sample)
                self.assertIn("action", out_td)
                logs = simulate(
                    torch.device("cpu"), torch.float32, 3, 12, actor_cpu, 12, actor_cpu,
                    1, 1, 1, 2, lambda: None, seed=777,
                )
            self.assertEqual(len(logs), 1)
            self.assertEqual(len(logs[0].get_result()), 4)

            if torch.cuda.is_available():
                actor_cuda = load_model(actor_path, map_location=torch.device("cuda")).eval()
                with torch.inference_mode():
                    out_cuda = actor_cuda(sample.to("cuda"))
                self.assertEqual(int(out_cuda["action"].item()), int(out_td["action"].item()))


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--ddp-audit-worker":
        _ddp_numerical_worker(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        unittest.main()
