"""Comprehensive regression & analytical tests for Structured V2 streaming, O(1)-RAM BC indexing, content fingerprints, reservoir statistics & DDP semantics (§§1-9)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
import time
import tracemalloc
import unittest
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel

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
from kanachan.structured_v2 import OFFSETS_136, PAD_ACTION_ID
from kanachan.training.bc.streaming_v2 import (
    BoundedRoundPrefetcher,
    DiskLengthBucketBatchSampler,
    SpoolingRoundBCDataSource,
)
from kanachan.training.bc.trajectory_v2 import (
    LengthBucketBatchSampler,
    TrajectoryBCModel,
    _compute_microgroup_denoms,
    collate_round_batch,
    load_trajectory_bc_checkpoint,
    parse_bc_rounds,
    train_trajectory_bc_steps,
)
from kanachan.training.core.dataset_fingerprint import (
    compute_source_data_fingerprint,
    sha256_file_bytes,
)
from kanachan.training.core.optimizers import adamw_parameter_groups
from kanachan.training.core.test_env import (
    STRICT_ACCEPTANCE_ENV_VAR,
    VAL_BC_ENV_VAR,
    require_or_skip_real_data,
)
from kanachan.training.grp import (
    FrozenGRP,
    FrozenGRPPBRSReward,
    GRPRoundDataset,
    PureStateGRP,
    save_frozen_grp,
    shape_hanchan_pbrs_rewards,
    train_pure_state_grp,
)
from kanachan.training.iql.joint_v2 import (
    build_joint_iql_optimizer,
    build_joint_iql_recipe,
    load_joint_iql_checkpoint,
    train_joint_iql_batches,
)
from kanachan.training.iql.streaming_v2 import StreamingIQLDataset, _CompactTransitionReservoir


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_windows_gloo_ifname() -> str | None:
    if os.environ.get("GLOO_SOCKET_IFNAME"):
        return os.environ["GLOO_SOCKET_IFNAME"]
    if os.name != "nt":
        return None
    try:
        ip_proc = subprocess.run(
            ["ipconfig"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5
        )
    except Exception:
        return None
    cur_if: str | None = None
    host_ip = socket.gethostbyname(socket.gethostname())
    for line in ip_proc.stdout.splitlines():
        s = line.strip()
        if line and not line[0].isspace() and s.endswith(":"):
            m = re.search(r"(?:adapter|适配器)\s+(.+):$", s, flags=re.IGNORECASE)
            cur_if = m.group(1).strip() if m else s[:-1].strip()
        elif host_ip in s and cur_if:
            return cur_if
    return None


def _make_sparse_row(
    *,
    seat: int,
    chang: int = 0,
    ju: int = 0,
    left: int = 60,
    game_mode: int = 5,
    zimo: int = 32,
) -> list[int]:
    sp = [2, game_mode, 7, 23, 39, 55, 71 + seat, 75 + chang, 78 + ju, 82 + left, 152 + 1]
    closed = {1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1, 21: 1, 22: 1, 23: 1, 31: 3, 32: 1}
    for t37, cnt in sorted(closed.items()):
        for k in range(cnt):
            sp.append(337 + OFFSETS_136[t37] + k)
    if zimo < 37:
        sp.append(473 + zimo)
    return sp


def _make_rl_game_lines(
    *,
    seat: int,
    n_transitions: int,
    score_step: int = 1000,
) -> list[str]:
    """Build a contiguous observer RL game with exact ``n_transitions >= 1`` lines."""
    assert n_transitions >= 1
    lines: list[str] = []
    cur_scores = [25000, 25000, 25000, 25000]
    fin_scores = [25000, 25000, 25000, 25000]
    fin_scores[seat] += score_step
    fin_scores[(seat + 1) % 4] -= score_step

    for t in range(n_transitions):
        is_last = t == n_transitions - 1
        chang = (t // 16) % 3
        ju = (t // 4) % 4
        r_t = t % 4
        left = max(1, 69 - r_t * 4)
        sp_cur = _make_sparse_row(seat=seat, chang=chang, ju=ju, left=left, zimo=32)
        nu_cur = [0, 0, *cur_scores]
        pr_cur = [0] + [5 + seat * 148 + ((4 + k) * 4) for k in range(r_t)]
        if is_last:
            delta = [fin_scores[i] - cur_scores[i] for i in range(4)]
            res_eog = [*delta, 0, 0, *fin_scores, *fin_scores]
            lines.append(
                "\t".join(
                    [
                        "",
                        ",".join(map(str, sp_cur)),
                        ",".join(map(str, nu_cur)),
                        ",".join(map(str, pr_cur)),
                        "4,44,84,130",
                        str(t % 4),
                        "5",
                        ",".join(map(str, res_eog)),
                    ]
                )
                + "\n"
            )
        else:
            nxt_t = t + 1
            nxt_chang = (nxt_t // 16) % 3
            nxt_ju = (nxt_t // 4) % 4
            nxt_r_t = nxt_t % 4
            nxt_left = max(1, 69 - nxt_r_t * 4)
            is_eor = (nxt_ju != ju) or (nxt_chang != chang)
            sp_nxt = _make_sparse_row(seat=seat, chang=nxt_chang, ju=nxt_ju, left=nxt_left, zimo=32)
            pr_nxt = [0] + ([] if is_eor else [5 + seat * 148 + ((4 + k) * 4) for k in range(nxt_r_t)])
            if is_eor:
                nxt_scores = list(cur_scores)
                nxt_scores[seat] += 500
                nxt_scores[(seat + 1) % 4] -= 500
                delta = [nxt_scores[i] - cur_scores[i] for i in range(4)]
                cur_scores = nxt_scores
                res_eor = [*delta, 0, 0, *nxt_scores]
                lines.append(
                    "\t".join(
                        [
                            "",
                            ",".join(map(str, sp_cur)),
                            ",".join(map(str, nu_cur)),
                            ",".join(map(str, pr_cur)),
                            "4,44,84,130",
                            str(t % 4),
                            ",".join(map(str, sp_nxt)),
                            ",".join(map(str, [0, 0, *nxt_scores])),
                            ",".join(map(str, pr_nxt)),
                            "4,44,84,130",
                            "5",
                            ",".join(map(str, res_eor)),
                        ]
                    )
                    + "\n"
                )
            else:
                lines.append(
                    "\t".join(
                        [
                            "",
                            ",".join(map(str, sp_cur)),
                            ",".join(map(str, nu_cur)),
                            ",".join(map(str, pr_cur)),
                            "4,44,84,130",
                            str(t % 4),
                            ",".join(map(str, sp_nxt)),
                            ",".join(map(str, nu_cur)),
                            ",".join(map(str, pr_nxt)),
                            "4,44,84,130",
                        ]
                    )
                    + "\n"
                )
    return lines


def _make_complete_tonpuusen_bc_lines(
    uuid: str,
    *,
    final_scores: tuple[int, int, int, int] = (32000, 27000, 22000, 19000),
    steps_per_round: int = 2,
) -> list[str]:
    """Build 4 physically consistent East-1..East-4 rounds with configurable turns on East-1."""
    assert sum(final_scores) == 100000
    s0 = (25000, 25000, 25000, 25000)

    def _interp(frac: float) -> tuple[int, int, int, int]:
        pts = [25000 + int(round((final_scores[i] - 25000) * frac / 100.0)) * 100 for i in range(4)]
        pts[0] += 100000 - sum(pts)
        return (pts[0], pts[1], pts[2], pts[3])

    post_seqs = [_interp(0.35), _interp(0.65), _interp(0.85), final_scores]
    code_table = [
        [-1, 4, 5, 6],
        [7, -1, 8, 9],
        [10, 11, -1, 12],
        [13, 14, 15, -1],
    ]
    lines: list[str] = []
    cur_scores = s0
    for ju in range(4):
        loser = ju
        winner = (ju + 1) % 4
        rs_code = code_table[winner][loser]
        post_s = post_seqs[ju]
        delta = tuple(post_s[i] - cur_scores[i] for i in range(4))
        res_common = [*delta, 0, 0, *post_s, *final_scores]
        nu = [0, 0, *cur_scores]
        pr_list = [0]
        extra_turns = max(0, steps_per_round - 2) if ju == 0 else 0
        assert extra_turns % 4 == 0 and extra_turns <= 16
        free_tiles = [4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27]
        for turn in range(extra_turns):
            s_turn = (loser + turn) % 4
            sp_turn = _make_sparse_row(seat=s_turn, chang=0, ju=ju, left=69 - turn, zimo=32)
            lines.append(
                "\t".join(
                    [
                        uuid,
                        ",".join(map(str, sp_turn)),
                        ",".join(map(str, nu)),
                        ",".join(map(str, pr_list)),
                        "4,44,84,130",
                        "0",
                        str(rs_code),
                        ",".join(map(str, res_common)),
                    ]
                )
                + "\n"
            )
            pr_list.append(5 + s_turn * 148 + 4 * free_tiles[turn] + 0)
        sp_loser = _make_sparse_row(seat=loser, chang=0, ju=ju, left=69 - extra_turns, zimo=32)
        lines.append(
            "\t".join(
                [
                    uuid,
                    ",".join(map(str, sp_loser)),
                    ",".join(map(str, nu)),
                    ",".join(map(str, pr_list)),
                    "4,44,84,130",
                    "3",
                    str(rs_code),
                    ",".join(map(str, res_common)),
                ]
            )
            + "\n"
        )
        disc_tok = 5 + loser * 148 + 4 * 32 + 2
        pr_win = pr_list + [disc_tok]
        sp_win = _make_sparse_row(seat=winner, chang=0, ju=ju, left=69 - extra_turns, zimo=37)
        lines.append(
            "\t".join(
                [
                    uuid,
                    ",".join(map(str, sp_win)),
                    ",".join(map(str, nu)),
                    ",".join(map(str, pr_win)),
                    "221,543",
                    "1",
                    str(rs_code),
                    ",".join(map(str, res_common)),
                ]
            )
            + "\n"
        )
        cur_scores = post_s
    return lines


def _ddp_env(rank: int, world_size: int, port: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "TORCHDYNAMO_DISABLE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "USE_LIBUV": "0",
            "KANACHAN_TEST_DISTRIBUTED": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
        }
    )
    ifname = _resolve_windows_gloo_ifname()
    if ifname:
        env["GLOO_SOCKET_IFNAME"] = ifname
    return env


def _spawn_2rank(mode: str, *args: str) -> None:
    port = _free_port()
    procs = [
        subprocess.Popen(
            [sys.executable, "-B", __file__, mode, str(rank), "2", *args],
            cwd=REPO_ROOT,
            env=_ddp_env(rank, 2, port),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for rank in (0, 1)
    ]
    deadline = time.monotonic() + 120.0
    try:
        while True:
            rcs = [p.poll() for p in procs]
            if all(rc is not None for rc in rcs):
                break
            for bad_rank, rc in enumerate(rcs):
                if rc is not None and rc != 0:
                    for p in procs:
                        if p.poll() is None:
                            p.kill()
                    outs = [p.communicate(timeout=5) for p in procs]
                    so, se = outs[bad_rank]
                    raise AssertionError(
                        f"2-rank worker mode={mode} rank={bad_rank} failed early ({rc}):\n{so}\n{se}"
                    )
            if time.monotonic() >= deadline:
                for p in procs:
                    if p.poll() is None:
                        p.kill()
                outs = [p.communicate(timeout=5) for p in procs]
                raise AssertionError(f"2-rank worker mode={mode} timed out: {outs}")
            time.sleep(0.1)
        outs = [p.communicate(timeout=5) for p in procs]
    finally:
        for p in procs:
            for s in (p.stdout, p.stderr):
                if s is not None:
                    s.close()
    for rank, (p, (so, se)) in enumerate(zip(procs, outs, strict=True)):
        if p.returncode != 0:
            raise AssertionError(f"2-rank worker mode={mode} rank={rank} failed ({p.returncode}):\n{so}\n{se}")


# ------------------------------------------------------------------
# Subprocess DDP Workers (§§5-6)
# ------------------------------------------------------------------


def _worker_bc_all_heads(rank: int, world_size: int, bc_path: str, rounds_pt: str, out_dir: str) -> None:
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        out = Path(out_dir)
        dev = torch.device("cpu")
        all_rounds = torch.load(rounds_pt, weights_only=False)
        sampler = LengthBucketBatchSampler(
            [r.round_event_len for r in all_rounds],
            batch_size=2,
            seed=71,
            bucket_mult=2,
            rank=rank,
            world_size=world_size,
            drop_last=False,
        )
        # 1. Gradient equivalence across Policy, NextEvent, Tedashi, FinalRank with ghost batch + grad_accum_steps=2
        torch.manual_seed(501)
        m_ddp = TrajectoryBCModel(
            preset="v2_tiny",
            dimension=64,
            num_heads=4,
            dim_feedforward=128,
            n_d1_layers=1,
            n_d2_layers=1,
            use_next_event_head=True,
            next_event_weight=0.25,
            use_final_rank_head=True,
            final_rank_weight=0.15,
            device=dev,
            dtype=torch.float32,
        )
        ddp_m = DistributedDataParallel(m_ddp)
        micros = [
            collate_round_batch([all_rounds[i] for i in idxs], batch_weight=bw)
            for _pos, idxs, bw in sampler.iter_pending_indexed_batches()
        ]
        denoms = _compute_microgroup_denoms(micros, is_dist=True, world_size=world_size, device=dev)
        ddp_m.zero_grad(set_to_none=True)
        for m_i, bcpu in enumerate(micros):
            if m_i < len(micros) - 1:
                with ddp_m.no_sync():
                    ddp_m(bcpu, denom_overrides=denoms, ddp_world_size=world_size)["loss"].backward()
            else:
                ddp_m(bcpu, denom_overrides=denoms, ddp_world_size=world_size)["loss"].backward()
        if rank == 0:
            grads = {n: p.grad.detach().clone() for n, p in m_ddp.named_parameters() if p.grad is not None}
            torch.save(grads, out / "bc_ddp_grads.pt")

        # 2. Spooling + DiskLengthBucketBatchSampler + suit augmentation checkpoint recovery
        with SpoolingRoundBCDataSource(Path(bc_path), cache_dir=out / "spool", num_hash_buckets=2) as ds:
            torch.manual_seed(502)
            m_run = TrajectoryBCModel(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, use_next_event_head=True,
                use_final_rank_head=True, final_rank_weight=0.1, device=dev, dtype=torch.float32,
            )
            ddp_run = DistributedDataParallel(m_run)
            opt_run = torch.optim.AdamW(adamw_parameter_groups(m_run, 0.01), lr=3e-4)
            res = train_trajectory_bc_steps(
                ddp_run, ds, opt_run, batch_size=2, max_steps=3, device=dev,
                suit_augmentation=True, aug_seed=777, sampler_seed=19,
                ckpt_dir=out / "bc_ckpt", checkpoint_every=2, prefetch_batches=2,
            )
            torch.save({k: v.cpu() for k, v in m_run.state_dict().items()}, out / f"bc_cont_{rank}.pt")
            _ = res
    finally:
        dist.destroy_process_group()


def _worker_bc_resume_world_check(rank: int, world_size: int, bc_path: str, out_dir: str) -> None:
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        out = Path(out_dir)
        dev = torch.device("cpu")
        with SpoolingRoundBCDataSource(Path(bc_path), cache_dir=out / "spool", num_hash_buckets=2) as ds:
            torch.manual_seed(502)
            m_res = TrajectoryBCModel(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, use_next_event_head=True,
                use_final_rank_head=True, final_rank_weight=0.1, device=dev, dtype=torch.float32,
            )
            ddp_res = DistributedDataParallel(m_res)
            opt_res = torch.optim.AdamW(adamw_parameter_groups(m_res, 0.01), lr=3e-4)
            # Run resume with prefetch_batches=0 to prove prefetch=0 vs prefetch=2 bit-exact suit aug parity!
            train_trajectory_bc_steps(
                ddp_res, ds, opt_res, batch_size=2, max_steps=3, device=dev,
                suit_augmentation=True, aug_seed=777, sampler_seed=19,
                ckpt_dir=out / "bc_ckpt", resume_from=out / "bc_ckpt/traj-bc-2.pt", prefetch_batches=0,
            )
            torch.save({k: v.cpu() for k, v in m_res.state_dict().items()}, out / f"bc_res_{rank}.pt")
    finally:
        dist.destroy_process_group()


def _worker_iql_single_game_ddp(rank: int, world_size: int, rl_1game_path: str, out_dir: str) -> None:
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        out = Path(out_dir)
        dev = torch.device("cpu")
        with StreamingIQLDataset(
            Path(rl_1game_path),
            batch_size=2,
            reservoir_size=16,
            warmup_transitions=2,
            reward_scheme="placement",
            seed=101,
            rank=rank,
            world_size=world_size,
            drop_last=False,
            max_prefetch=1,
        ) as ds:
            torch.manual_seed(601)
            mod = JointStructuredIQLModule(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, device=dev, dtype=torch.float32,
            )
            ddp_m = DistributedDataParallel(mod)
            opt = build_joint_iql_optimizer(mod, base_lr=1e-4, weight_decay=0.01, device=dev, fused=False)
            rec = build_joint_iql_recipe(
                mod, base_lr=1e-4, weight_decay=0.01, decoder1_lr_scale=0.25,
                data_sha256=ds.data_sha256,
                source_data_sha256=ds.source_data_sha256,
                preprocess_recipe_sha256=ds.preprocess_recipe_sha256,
            )
            res = train_joint_iql_batches(
                mod, ds, opt, recipe=rec, device=dev, ddp_wrapped=ddp_m,
                max_steps=None, ckpt_dir=out / "iql_1g_ckpt", checkpoint_every=1,
            )
            torch.save(
                {
                    "step": res["step"],
                    "pass_seen": ds.pass_transitions_seen,
                    "state": {k: v.cpu() for k, v in mod.state_dict().items()},
                },
                out / f"iql_1g_{rank}.pt",
            )
    finally:
        dist.destroy_process_group()


# ------------------------------------------------------------------
# §1: IQL Multi-Pass Transition Accounting Tests
# ------------------------------------------------------------------


class TestIQLMultiPassAccounting(unittest.TestCase):
    def _collect_pass(self, ds: StreamingIQLDataset) -> list[TensorDict]:
        return [b for b in ds]

    def test_multipass_1_2_3_empty_long_short_mixed_and_warmup_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # 1. Empty stream
            empty_p = root / "empty.rl.tsv"
            empty_p.write_text("", encoding="utf-8")
            for drop in (False, True):
                with StreamingIQLDataset(
                    empty_p, batch_size=4, reservoir_size=16, warmup_transitions=2, drop_last=drop, max_prefetch=0
                ) as ds:
                    self.assertEqual(self._collect_pass(ds), [])
                    self.assertTrue(ds._eof)
                    self.assertEqual(ds.pass_transitions_seen, 0)
                    self.assertEqual(ds.pass_target_batches, 0)
                    self.assertEqual(ds.pass_batches_emitted, 0)
                    ds.reset_pass()
                    self.assertEqual(self._collect_pass(ds), [])

            # 2. Single long game (L = 25 > batch_size = 8) -> exact 3 (drop_last) / 4 (non-drop-last)
            long_p = root / "long.rl.tsv"
            long_p.write_text("".join(_make_rl_game_lines(seat=0, n_transitions=25)), encoding="utf-8")
            for drop, expected_cnt in ((True, 3), (False, 4)):
                with StreamingIQLDataset(
                    long_p, batch_size=8, reservoir_size=64, warmup_transitions=4, drop_last=drop, max_prefetch=0
                ) as ds:
                    for p_idx in range(3):
                        if p_idx > 0:
                            ds.reset_pass()
                        batches = self._collect_pass(ds)
                        self.assertEqual(len(batches), expected_cnt, f"drop_last={drop}, pass={p_idx}")
                        self.assertEqual(ds.pass_index, p_idx)
                        self.assertEqual(ds.pass_transitions_seen, 25)
                        self.assertEqual(ds.pass_batches_emitted, expected_cnt)
                        self.assertEqual(ds.pass_target_batches, expected_cnt)
                        self.assertEqual(ds._reservoir.total_seen, 25 * (p_idx + 1))

            # 3. Many short games (15 games x 2 transitions = 30, batch_size=8, warmup=12 -> NO double counting)
            short_p = root / "short.rl.tsv"
            short_lines: list[str] = []
            for g in range(15):
                short_lines.extend(_make_rl_game_lines(seat=g % 4, n_transitions=2))
            short_p.write_text("".join(short_lines), encoding="utf-8")
            for drop, expected_cnt in ((True, 3), (False, 4)):
                with StreamingIQLDataset(
                    short_p, batch_size=8, reservoir_size=20, warmup_transitions=12, drop_last=drop, max_prefetch=0
                ) as ds:
                    seq_signatures: list[list[list[int]]] = []
                    for p_idx in range(3):
                        if p_idx > 0:
                            ds.reset_pass()
                        bs = self._collect_pass(ds)
                        self.assertEqual(len(bs), expected_cnt)
                        self.assertEqual(ds.pass_transitions_seen, 30)
                        self.assertEqual(ds._reservoir.total_seen, 30 * (p_idx + 1))
                        seq_signatures.append([b["sparse"][:, 6].tolist() + b["action"].tolist() for b in bs])
                    # Passes 1, 2, 3 must remain deterministic yet distinct!
                    self.assertNotEqual(seq_signatures[0], seq_signatures[1])
                    self.assertNotEqual(seq_signatures[1], seq_signatures[2])

            # Determinism check across fresh instance
            with StreamingIQLDataset(
                short_p, batch_size=8, reservoir_size=20, warmup_transitions=12, drop_last=False, max_prefetch=2
            ) as ds_a, StreamingIQLDataset(
                short_p, batch_size=8, reservoir_size=20, warmup_transitions=12, drop_last=False, max_prefetch=2
            ) as ds_b:
                for p_idx in range(3):
                    if p_idx > 0:
                        ds_a.reset_pass()
                        ds_b.reset_pass()
                    ba = self._collect_pass(ds_a)
                    bb = self._collect_pass(ds_b)
                    self.assertEqual(len(ba), len(bb))
                    for xa, xb in zip(ba, bb, strict=True):
                        torch.testing.assert_close(xa["sparse"], xb["sparse"])
                        torch.testing.assert_close(xa["action"], xb["action"])

            # 4. Mixed game lengths + Reservoir smaller (M=12) & larger (M=100) than N=40, plus N < B
            mixed_p = root / "mixed.rl.tsv.gz"
            lengths = [2, 11, 3, 7, 1, 16]  # sum == 40
            mixed_lines = [
                ln for i, l in enumerate(lengths) for ln in _make_rl_game_lines(seat=i % 4, n_transitions=l)
            ]
            with gzip.open(mixed_p, "wt", encoding="utf-8") as fp:
                fp.writelines(mixed_lines)
            for res_sz in (12, 100):
                for drop, exp_cnt in ((True, 40 // 9), (False, math.ceil(40 / 9))):
                    with StreamingIQLDataset(
                        mixed_p, batch_size=9, reservoir_size=res_sz, warmup_transitions=10,
                        drop_last=drop, max_prefetch=2,
                    ) as ds:
                        for p_idx in range(3):
                            if p_idx > 0:
                                ds.reset_pass()
                            bs = self._collect_pass(ds)
                            self.assertEqual(len(bs), exp_cnt)
                            self.assertEqual(ds.pass_transitions_seen, 40)
                            self.assertEqual(ds._reservoir.total_seen, 40 * (p_idx + 1))

            tiny_p = root / "tiny3.rl.tsv"
            tiny_p.write_text("".join(_make_rl_game_lines(seat=0, n_transitions=3)), encoding="utf-8")
            with StreamingIQLDataset(tiny_p, batch_size=8, reservoir_size=16, warmup_transitions=2, drop_last=True, max_prefetch=0) as d_t:
                self.assertEqual(self._collect_pass(d_t), [])
            with StreamingIQLDataset(tiny_p, batch_size=8, reservoir_size=16, warmup_transitions=2, drop_last=False, max_prefetch=0) as d_f:
                self.assertEqual(len(self._collect_pass(d_f)), 1)

    def test_checkpoint_before_and_after_eof_plus_prefetch_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rl_p = root / "eof_ckpt.rl.tsv.gz"
            lines = [ln for g in range(7) for ln in _make_rl_game_lines(seat=g % 4, n_transitions=3)]  # 21 rows
            with gzip.open(rl_p, "wt", encoding="utf-8") as fp:
                fp.writelines(lines)

            for drop in (True, False):
                with self.subTest(drop_last=drop):
                    # Reference un-interrupted with max_prefetch=0 across 2 passes
                    with StreamingIQLDataset(
                        rl_p, batch_size=4, reservoir_size=12, warmup_transitions=4, drop_last=drop, seed=17, max_prefetch=0
                    ) as ref:
                        p1_ref = self._collect_pass(ref)
                        ref.reset_pass()
                        p2_ref = self._collect_pass(ref)
                    exp_n = 21 // 4 if drop else math.ceil(21 / 4)
                    self.assertEqual(len(p1_ref), exp_n)
                    self.assertEqual(len(p2_ref), exp_n)

                    # Checkpoint immediately BEFORE EOF (after batch exp_n - 2 -> prefetch=4 queues final batch & EOF!)
                    cut = exp_n - 1
                    with StreamingIQLDataset(
                        rl_p, batch_size=4, reservoir_size=12, warmup_transitions=4, drop_last=drop, seed=17, max_prefetch=4
                    ) as ds_pre:
                        pre_consumed = [ds_pre.next_batch() for _ in range(cut)]
                        # Allow worker time to race ahead into EOF before state_dict() rollback
                        import time
                        time.sleep(0.08)
                        sd_before_eof = ds_pre.state_dict()
                        self.assertEqual(ds_pre.pass_batches_emitted, cut)

                    with StreamingIQLDataset(
                        rl_p, batch_size=4, reservoir_size=12, warmup_transitions=4, drop_last=drop, seed=17, max_prefetch=4
                    ) as ds_res_pre:
                        ds_res_pre.load_state_dict(sd_before_eof)
                        rem = self._collect_pass(ds_res_pre)
                        ds_res_pre.reset_pass()
                        p2_res = self._collect_pass(ds_res_pre)
                    full_p1 = pre_consumed + rem
                    self.assertEqual(len(full_p1), len(p1_ref))
                    for x, y in zip(full_p1 + p2_res, p1_ref + p2_ref, strict=True):
                        torch.testing.assert_close(x["sparse"], y["sparse"])
                        torch.testing.assert_close(x["action"], y["action"])

                    # Checkpoint immediately AFTER EOF (`next_batch()` returned None)
                    with StreamingIQLDataset(
                        rl_p, batch_size=4, reservoir_size=12, warmup_transitions=4, drop_last=drop, seed=17, max_prefetch=3
                    ) as ds_eof:
                        got = self._collect_pass(ds_eof)
                        self.assertEqual(len(got), exp_n)
                        sd_at_eof = ds_eof.state_dict()
                        self.assertTrue(sd_at_eof["eof"])
                        self.assertEqual(sd_at_eof["pass_batches_emitted"], exp_n)
                        self.assertEqual(sd_at_eof["pass_target_batches"], exp_n)

                    with StreamingIQLDataset(
                        rl_p, batch_size=4, reservoir_size=12, warmup_transitions=4, drop_last=drop, seed=17, max_prefetch=3
                    ) as ds_after_eof:
                        ds_after_eof.load_state_dict(sd_at_eof)
                        self.assertIsNone(ds_after_eof.next_batch())
                        ds_after_eof.reset_pass()
                        p2_after = self._collect_pass(ds_after_eof)
                        self.assertEqual(len(p2_after), len(p2_ref))
                        for x, y in zip(p2_after, p2_ref, strict=True):
                            torch.testing.assert_close(x["sparse"], y["sparse"])
                            torch.testing.assert_close(x["action"], y["action"])


# ------------------------------------------------------------------
# §2: BC O(1)-RAM External Disk Indexing & Concurrent Readers
# ------------------------------------------------------------------


class TestBCScalableExternalDiskSampler(unittest.TestCase):
    def test_external_disk_sampler_o1_ram_partitioning_invariants_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Measure tracemalloc delta for N=60 vs N=240 rounds (forced small window_items=17 -> many disk partitions & merge runs)
            peaks_by_n: dict[int, int] = {}
            for n_games, exp_rounds in ((10, 90), (30, 270)):
                bc_p = root / f"bc_{n_games}.tsv.gz"
                lines: list[str] = []
                for g in range(n_games):
                    # Variable round length on East-1 (alternating 2 vs 18 rows -> 4 vs 2 observer rounds on East-1)
                    lines.extend(
                        _make_complete_tonpuusen_bc_lines(
                            f"g-{g}",
                            steps_per_round=(18 if g % 2 == 0 else 2),
                        )
                    )
                with gzip.open(bc_p, "wt", encoding="utf-8") as fp:
                    fp.writelines(lines)

                with SpoolingRoundBCDataSource(
                    bc_p, cache_dir=root / f"spool_{n_games}", num_hash_buckets=4, max_bucket_bytes=2048
                ) as src:
                    self.assertFalse(hasattr(src, "round_event_lens"))
                    self.assertEqual(len(src), exp_rounds)

                    tracemalloc.start()
                    _, start_mem = tracemalloc.get_traced_memory()
                    with src.create_sampler(
                        batch_size=5, seed=13, bucket_mult=4, rank=0, world_size=1,
                        drop_last=False, window_items=17,
                    ) as samp:
                        covered = list(samp.assigned_sample_indices())
                        _, peak_mem = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peaks_by_n[exp_rounds] = max(0, peak_mem - start_mem)
                    self.assertEqual(len(covered), exp_rounds)
                    self.assertEqual(set(covered), set(range(exp_rounds)))

            # 3x dataset growth must stay bounded (< 4 MB Python heap delta)
            self.assertLess(peaks_by_n[270], 4 * 1024 * 1024, peaks_by_n)

            # Check drop_last=True, 3-rank DDP equal steps + 1 ghost batch, and length-bucketing efficiency
            lens_ref: list[int] = []
            with SpoolingRoundBCDataSource(
                root / "bc_10.tsv.gz", cache_dir=root / "spool_10", reuse_cache=True
            ) as src:
                lens_ref = list(src.iter_index_lengths(chunk_items=16))
                self.assertEqual(len(lens_ref), 90)

                # drop_last=True (90 rounds, batch_size=7 -> 12 full batches, 84 items)
                with src.create_sampler(
                    batch_size=7, seed=5, bucket_mult=4, drop_last=True, window_items=19
                ) as sd_drop:
                    batches_drop = list(sd_drop.iter_pending_indexed_batches())
                    self.assertEqual(len(batches_drop), 90 // 7)
                    self.assertTrue(all(len(idxs) == 7 for _, idxs, _ in batches_drop))

                # Also verify in-memory LengthBucketBatchSampler drop_last=True parity
                s_mem = LengthBucketBatchSampler(lens_ref, batch_size=7, seed=5, bucket_mult=4, drop_last=True)
                self.assertEqual(len(s_mem), 90 // 7)

                # 3-rank sharding across non-divisible global batches (ceil(90/12)=8 global batches -> 3/rank + 1 ghost)
                s_ranks = [
                    src.create_sampler(
                        batch_size=12, seed=21, bucket_mult=2, rank=r, world_size=3, drop_last=False, window_items=15
                    )
                    for r in range(3)
                ]
                try:
                    self.assertEqual({len(s) for s in s_ranks}, {3})
                    assigned = [set(s.assigned_sample_indices()) for s in s_ranks]
                    for i in range(3):
                        for j in range(i + 1, 3):
                            self.assertEqual(assigned[i] & assigned[j], set())
                    self.assertEqual(set().union(*assigned), set(range(90)))
                    ghosts = sum(1 for s in s_ranks for _, _, w in s.iter_pending_indexed_batches() if w == 0.0)
                    self.assertEqual(ghosts, 1)
                finally:
                    for s in s_ranks:
                        s.close()

                # Length-aware bucketing cost must beat unbucketed shuffle cost
                with src.create_sampler(batch_size=9, seed=99, bucket_mult=10, window_items=90) as s_buck:
                    buck_cost = sum(len(idxs) * max(lens_ref[i] for i in idxs) for _, idxs, _ in s_buck.iter_pending_indexed_batches())
                perm_rand = torch.randperm(90, generator=torch.Generator().manual_seed(99)).tolist()
                rand_cost = sum(9 * max(lens_ref[i] for i in perm_rand[k : k + 9]) for k in range(0, 90, 9))
                self.assertLess(buck_cost, rand_cost)

    def test_concurrent_multithread_readers_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bc_p = root / "conc.bc.tsv.gz"
            lines = [ln for g in range(6) for ln in _make_complete_tonpuusen_bc_lines(f"c-{g}")]
            with gzip.open(bc_p, "wt", encoding="utf-8") as fp:
                fp.writelines(lines)

            with SpoolingRoundBCDataSource(bc_p, cache_dir=root / "spool", num_hash_buckets=4) as src:
                with src.create_sampler(batch_size=3, seed=42, rank=0, world_size=1, window_items=11) as s_seq:
                    expected = [
                        collate_round_batch(
                            src.get_rounds(idxs),
                            suit_perm_generator=torch.Generator().manual_seed(777 + pos * 10_007),
                            batch_weight=w,
                        )
                        for pos, idxs, w in s_seq.iter_pending_indexed_batches()
                    ]
                    # Check concurrent 4-worker BoundedRoundPrefetcher produces bit-identical output!
                    with BoundedRoundPrefetcher(
                        src,
                        s_seq.iter_pending_indexed_batches(),
                        max_prefetch=4,
                        suit_perm_seed=777,
                        epoch=0,
                        num_workers=4,
                    ) as pf:
                        got = list(pf)
                    self.assertEqual(len(got), len(expected))
                    for (pos, b_got), b_exp in zip(got, expected, strict=True):
                        torch.testing.assert_close(b_got["round_events"], b_exp["round_events"])
                        torch.testing.assert_close(b_got["v2_tile_feats"], b_exp["v2_tile_feats"])

                    # Direct multi-thread random seeks on get_rounds() must never corrupt file cursors
                    def _read_one(i: int):
                        r0 = src.get_round(i)
                        r1 = src.get_round((i + 7) % len(src))
                        return r0.uuid, r1.uuid

                    with ThreadPoolExecutor(max_workers=6) as pool:
                        out_pairs = list(pool.map(_read_one, range(len(src))))
                    self.assertEqual(len(out_pairs), len(src))

                    # Exact resume at cursor midpoint
                    it = s_seq.iter_weighted_batches()
                    first_half = [next(it) for _ in range(5)]
                    sd = s_seq.state_dict()
                    with src.create_sampler(batch_size=3, seed=99) as s_bad:
                        with self.assertRaises(ValueError):
                            s_bad.load_state_dict(sd)
                    with src.create_sampler(batch_size=3, seed=42, window_items=11) as s_res:
                        s_res.load_state_dict(sd)
                        second_half = list(s_res.iter_weighted_batches())
                    self.assertEqual(len(first_half) + len(second_half), len(s_seq))
                    all_ids = [i for b, _ in (first_half + second_half) for i in b]
                    self.assertEqual(set(all_ids), set(range(len(src))))


# ------------------------------------------------------------------
# §3: Content-Based Dataset & Recipe Fingerprints
# ------------------------------------------------------------------


class TestContentBasedDatasetFingerprints(unittest.TestCase):
    def test_all_seven_fingerprint_tamper_and_recipe_invalidation_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f1 = root / "part1.rl.tsv"
            f2 = root / "part2.rl.tsv"
            f3 = root / "part3.rl.tsv"
            lines1 = _make_rl_game_lines(seat=0, n_transitions=4, score_step=1000)
            lines2 = _make_rl_game_lines(seat=1, n_transitions=4, score_step=2000)
            lines3 = _make_rl_game_lines(seat=2, n_transitions=4, score_step=3000)
            f1.write_text("".join(lines1), encoding="utf-8")
            f2.write_text("".join(lines2), encoding="utf-8")
            f3.write_text("".join(lines3), encoding="utf-8")

            base_src, _ = compute_source_data_fingerprint([f1, f2, f3], source_format="source_v3_rl_v1")

            # (c) File order change
            reordered_src, _ = compute_source_data_fingerprint([f2, f1, f3], source_format="source_v3_rl_v1")
            self.assertNotEqual(base_src, reordered_src)

            # Save IQL checkpoint bound to [f1, f2, f3]
            ds0 = StreamingIQLDataset([f1, f2, f3], batch_size=2, reservoir_size=16, warmup_transitions=2, max_prefetch=0)
            mod = JointStructuredIQLModule(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, device=torch.device("cpu"), dtype=torch.float32,
            )
            opt = build_joint_iql_optimizer(mod, base_lr=1e-4, weight_decay=0.01, device=torch.device("cpu"), fused=False)
            rec0 = build_joint_iql_recipe(
                mod, base_lr=1e-4, weight_decay=0.01, decoder1_lr_scale=0.25,
                data_sha256=ds0.data_sha256,
                source_data_sha256=ds0.source_data_sha256,
                preprocess_recipe_sha256=ds0.preprocess_recipe_sha256,
            )
            ckpt_dir = root / "iql_fp_ckpt"
            train_joint_iql_batches(mod, ds0, opt, recipe=rec0, device=torch.device("cpu"), max_steps=1, ckpt_dir=ckpt_dir, checkpoint_every=1)
            ckpt_p = ckpt_dir / "joint-iql-1.pt"
            ds0.close()

            # (a) Same-size edit + (b) restored mtime_ns + (d) single file change in multi-file dataset
            st_orig = f2.stat()
            orig_bytes = f2.read_bytes()
            # Replace single ASCII byte inside score preserving exact byte length!
            tampered_bytes = orig_bytes.replace(b"2000", b"2001", 1)
            self.assertEqual(len(tampered_bytes), len(orig_bytes))
            f2.write_bytes(tampered_bytes)
            os.utime(f2, ns=(st_orig.st_atime_ns, st_orig.st_mtime_ns))
            self.assertEqual(f2.stat().st_size, st_orig.st_size)
            self.assertEqual(f2.stat().st_mtime_ns, st_orig.st_mtime_ns)

            ds_edit = StreamingIQLDataset([f1, f2, f3], batch_size=2, reservoir_size=16, warmup_transitions=2, max_prefetch=0)
            self.assertNotEqual(ds_edit.source_data_sha256, base_src)
            rec_edit = build_joint_iql_recipe(
                mod, base_lr=1e-4, weight_decay=0.01, decoder1_lr_scale=0.25,
                data_sha256=ds_edit.data_sha256,
                source_data_sha256=ds_edit.source_data_sha256,
                preprocess_recipe_sha256=ds_edit.preprocess_recipe_sha256,
            )
            with self.assertRaises(ValueError):
                load_joint_iql_checkpoint(ckpt_p, mod, opt, expected_recipe=rec_edit)
            ds_edit.close()
            f2.write_bytes(orig_bytes)
            os.utime(f2, ns=(st_orig.st_atime_ns, st_orig.st_mtime_ns))

            # (e) Reward config / plugin changes (source_data_sha256 unchanged, preprocess_recipe_sha256 changed!)
            ds_rew = StreamingIQLDataset(
                [f1, f2, f3], batch_size=2, reservoir_size=16, warmup_transitions=2,
                reward_scheme="round_and_placement", score_unit=5000.0, max_prefetch=0,
            )
            self.assertEqual(ds_rew.source_data_sha256, base_src)
            self.assertNotEqual(ds_rew.preprocess_recipe_sha256, rec0["preprocess_recipe_sha256"])
            rec_rew = build_joint_iql_recipe(
                mod, base_lr=1e-4, weight_decay=0.01, decoder1_lr_scale=0.25,
                data_sha256=ds_rew.data_sha256,
                source_data_sha256=ds_rew.source_data_sha256,
                preprocess_recipe_sha256=ds_rew.preprocess_recipe_sha256,
            )
            with self.assertRaises(ValueError):
                load_joint_iql_checkpoint(ckpt_p, mod, opt, expected_recipe=rec_rew)
            ds_rew.close()

            # (f) GRP checkpoint change -> fails closed
            gm1 = PureStateGRP()
            gm2 = PureStateGRP()
            with torch.no_grad():
                gm2.net[4].bias.add_(0.5)
            grp1 = FrozenGRP(gm1, {"top1_ece": 0.01, "classwise_ece": 0.01, "brier": 0.5}, require_gate=True)
            grp2 = FrozenGRP(gm2, {"top1_ece": 0.01, "classwise_ece": 0.01, "brier": 0.5}, require_gate=True)
            ds_g1 = StreamingIQLDataset([f1], batch_size=2, reservoir_size=16, warmup_transitions=2, reward_scheme="grp_pbrs", grp=grp1, max_prefetch=0)
            ds_g2 = StreamingIQLDataset([f1], batch_size=2, reservoir_size=16, warmup_transitions=2, reward_scheme="grp_pbrs", grp=grp2, max_prefetch=0)
            self.assertEqual(ds_g1.source_data_sha256, ds_g2.source_data_sha256)
            self.assertNotEqual(ds_g1.preprocess_recipe_sha256, ds_g2.preprocess_recipe_sha256)
            ds_g1.close()
            ds_g2.close()

            # (g) Schema version change -> fails closed
            ds_sch = StreamingIQLDataset(
                [f1, f2, f3], batch_size=2, reservoir_size=16, warmup_transitions=2,
                schema_version="source_v4_test", max_prefetch=0,
            )
            self.assertNotEqual(ds_sch.source_data_sha256, base_src)
            self.assertNotEqual(ds_sch.preprocess_recipe_sha256, rec0["preprocess_recipe_sha256"])
            ds_sch.close()

            # BC Spooling cache MUST invalidate & rebuild on same-size restored-mtime edit!
            bc_p = root / "bc_tamper.tsv"
            bc_p.write_text("".join(_make_complete_tonpuusen_bc_lines("u0", final_scores=(32000, 27000, 22000, 19000))), encoding="utf-8")
            spool_dir = root / "bc_spool"
            with SpoolingRoundBCDataSource(bc_p, cache_dir=spool_dir, num_hash_buckets=2) as bsrc1:
                old_sha = bsrc1.source_data_sha256
                old_r0_rank = int(bsrc1.get_round(0).final_rank[0].item())
                bm = TrajectoryBCModel(
                    preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                    n_d1_layers=1, n_d2_layers=1, device=torch.device("cpu"), dtype=torch.float32,
                )
                bopt = torch.optim.AdamW(adamw_parameter_groups(bm, 0.01), lr=1e-4)
                train_trajectory_bc_steps(
                    bm, bsrc1, bopt, batch_size=2, max_steps=1, device=torch.device("cpu"),
                    ckpt_dir=root / "bc_fp_ckpt", checkpoint_every=1,
                )

            st_bc = bc_p.stat()
            raw_bc = bc_p.read_text(encoding="utf-8")
            # Swap seat scores (same exact byte length & restored mtime_ns!)
            raw_bc2 = raw_bc.replace("32000,27000,22000,19000", "19000,22000,27000,32000")
            self.assertEqual(len(raw_bc2), len(raw_bc))
            bc_p.write_text(raw_bc2, encoding="utf-8")
            os.utime(bc_p, ns=(st_bc.st_atime_ns, st_bc.st_mtime_ns))

            with SpoolingRoundBCDataSource(bc_p, cache_dir=spool_dir, num_hash_buckets=2, reuse_cache=True) as bsrc2:
                self.assertNotEqual(bsrc2.source_data_sha256, old_sha)
                new_r0_rank = int(bsrc2.get_round(0).final_rank[0].item())
                self.assertNotEqual(new_r0_rank, old_r0_rank)
                with self.assertRaises(ValueError):
                    train_trajectory_bc_steps(
                        bm, bsrc2, bopt, batch_size=2, max_steps=2, device=torch.device("cpu"),
                        ckpt_dir=root / "bc_fp_ckpt", resume_from=root / "bc_fp_ckpt/traj-bc-1.pt",
                    )


# ------------------------------------------------------------------
# §4: Analytical & Statistical IQL Reservoir Tests
# ------------------------------------------------------------------


class TestIQLReservoirStatisticalAndPartitioning(unittest.TestCase):
    def test_analytical_chi_square_transition_uniformity_unequal_games_and_multipass(self):
        # 10 short games of 2 transitions each (indices 0..19) + 2 long games of 30 transitions each (indices 20..79)
        # Total N = 80 transitions; reservoir M = 16 -> true transition-uniform inclusion p = M/N = 0.20!
        game_lengths = [2] * 10 + [30] * 2
        n_total = sum(game_lengths)
        m_cap = 16
        trials = 2500
        counts_p1 = [0] * n_total
        counts_p2 = [0] * n_total

        dummy_games: list[dict[str, torch.Tensor]] = []
        cursor_tid = 0
        for gl in game_lengths:
            ids = torch.arange(cursor_tid, cursor_tid + gl, dtype=torch.int32)
            cursor_tid += gl
            dummy_games.append(
                {
                    "sp": torch.zeros((gl, MAX_NUM_ACTIVE_SPARSE_FEATURES), dtype=torch.int16),
                    "nu": torch.stack([ids] + [torch.zeros_like(ids)] * 5, dim=1),
                    "pr": torch.zeros((gl, MAX_LENGTH_OF_PROGRESSION_FEATURES), dtype=torch.int16),
                    "ca": torch.zeros((gl, MAX_NUM_ACTION_CANDIDATES), dtype=torch.int16),
                    "act": torch.zeros((gl,), dtype=torch.uint8),
                    "nsp": torch.zeros((gl, MAX_NUM_ACTIVE_SPARSE_FEATURES), dtype=torch.int16),
                    "nnu": torch.zeros((gl, 6), dtype=torch.int32),
                    "npr": torch.zeros((gl, MAX_LENGTH_OF_PROGRESSION_FEATURES), dtype=torch.int16),
                    "nca": torch.zeros((gl, MAX_NUM_ACTION_CANDIDATES), dtype=torch.int16),
                    "reward": torch.zeros((gl,), dtype=torch.float32),
                    "discount": torch.ones((gl,), dtype=torch.float32),
                    "done": torch.zeros((gl,), dtype=torch.bool),
                    "eog": torch.zeros((gl,), dtype=torch.bool),
                    "final_rank": torch.zeros((gl,), dtype=torch.int8),
                }
            )

        for seed in range(trials):
            res = _CompactTransitionReservoir(m_cap)
            rng = torch.Generator(device="cpu").manual_seed(10_000 + seed)
            for g in dummy_games:
                res.ingest_game(g, rng)
            for tid in res._bufs["nu"][: res.length, 0].tolist():
                counts_p1[int(tid)] += 1
            # Pass 2 continuing same cumulative reservoir
            for g in dummy_games:
                res.ingest_game(g, rng)
            for tid in res._bufs["nu"][: res.length, 0].tolist():
                counts_p2[int(tid)] += 1

        exp_freq = trials * (m_cap / n_total)  # 500.0
        for label, counts in (("pass1", counts_p1), ("pass2", counts_p2)):
            short_mean = sum(counts[:20]) / (20 * trials)
            long_mean = sum(counts[20:]) / (60 * trials)
            self.assertAlmostEqual(short_mean, m_cap / n_total, delta=0.015, msg=label)
            self.assertAlmostEqual(long_mean, m_cap / n_total, delta=0.015, msg=label)
            # Pearson chi-square across all 80 transitions (df = 79; under uniform H0 mean=79, std~12.5 -> threshold < 140)
            chi2 = sum(((c - exp_freq) ** 2) / (exp_freq * (1.0 - m_cap / n_total)) for c in counts)
            self.assertLess(chi2, 145.0, f"{label}: chi2={chi2:.2f} rejects transition-uniform Algorithm R")

    def test_periodic_4seat_partitioning_no_aliasing_and_independent_rng_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rl_p = root / "cyclic.rl.tsv"
            lines = [ln for g in range(40) for ln in _make_rl_game_lines(seat=g % 4, n_transitions=2)]
            rl_p.write_text("".join(lines), encoding="utf-8")

            for ws in (2, 4):
                seats_by_rank: list[list[int]] = []
                games_by_rank: list[set[int]] = []
                for r in range(ws):
                    with StreamingIQLDataset(
                        rl_p, batch_size=2, reservoir_size=100, warmup_transitions=2,
                        seed=42, rank=r, world_size=ws, max_prefetch=0,
                    ) as ds:
                        owned_games = [g for g in range(40) if ds._game_owner_rank(g) == r]
                        games_by_rank.append(set(owned_games))
                        list(ds)
                        seats_owned = ds._reservoir._bufs["sp"][: ds._reservoir.length, 6].long().sub(71).tolist()
                        seats_by_rank.append(seats_owned)
                        # Verify zero aliasing: every rank sees ALL 4 seats {0, 1, 2, 3}!
                        self.assertEqual(set(seats_owned), {0, 1, 2, 3}, f"ws={ws} rank={r} excluded seats: {set(seats_owned)}")
                for i in range(ws):
                    self.assertEqual(len(games_by_rank[i]), 40 // ws)
                    for j in range(i + 1, ws):
                        self.assertEqual(games_by_rank[i] & games_by_rank[j], set())
                self.assertEqual(set().union(*games_by_rank), set(range(40)))

            # Independent ingest_rng vs sample_rng + multi-overwrite rollback in same batch
            with StreamingIQLDataset(
                rl_p, batch_size=2, reservoir_size=6, warmup_transitions=2, seed=55, max_prefetch=0
            ) as ds_a, StreamingIQLDataset(
                rl_p, batch_size=5, reservoir_size=6, warmup_transitions=2, seed=55, max_prefetch=0
            ) as ds_b:
                list(ds_a)
                list(ds_b)
                torch.testing.assert_close(ds_a._ingest_rng.get_state(), ds_b._ingest_rng.get_state())

            res_rb = _CompactTransitionReservoir(2)
            rng_rb = torch.Generator(device="cpu").manual_seed(123)
            g1 = {
                "sp": torch.ones((2, MAX_NUM_ACTIVE_SPARSE_FEATURES), dtype=torch.int16),
                "nu": torch.full((2, 6), 11, dtype=torch.int32),
                "pr": torch.zeros((2, MAX_LENGTH_OF_PROGRESSION_FEATURES), dtype=torch.int16),
                "ca": torch.zeros((2, MAX_NUM_ACTION_CANDIDATES), dtype=torch.int16),
                "act": torch.tensor([1, 2], dtype=torch.uint8),
                "nsp": torch.zeros((2, MAX_NUM_ACTIVE_SPARSE_FEATURES), dtype=torch.int16),
                "nnu": torch.zeros((2, 6), dtype=torch.int32),
                "npr": torch.zeros((2, MAX_LENGTH_OF_PROGRESSION_FEATURES), dtype=torch.int16),
                "nca": torch.zeros((2, MAX_NUM_ACTION_CANDIDATES), dtype=torch.int16),
                "reward": torch.ones((2,), dtype=torch.float32),
                "discount": torch.ones((2,), dtype=torch.float32),
                "done": torch.zeros((2,), dtype=torch.bool),
                "eog": torch.zeros((2,), dtype=torch.bool),
                "final_rank": torch.zeros((2,), dtype=torch.int8),
            }
            res_rb.ingest_game(g1, rng_rb)
            snap_act = res_rb._bufs["act"][:2].clone()
            g2 = {k: torch.cat([v, v], dim=0) for k, v in g1.items()}
            g2["act"] = torch.tensor([9, 8, 7, 6], dtype=torch.uint8)
            undo, prev_l, prev_s = res_rb.ingest_game(g2, rng_rb)
            res_rb.rollback_ingest(undo, prev_l, prev_s)
            torch.testing.assert_close(res_rb._bufs["act"][:2], snap_act)
            self.assertEqual((res_rb.length, res_rb.total_seen), (2, 2))


# ------------------------------------------------------------------
# §§5-6: BC & IQL Distributed (2-Rank Gloo) Semantics
# ------------------------------------------------------------------


class TestDistributedBCAndIQLSemantics(unittest.TestCase):
    def test_2rank_ddp_bc_all_heads_grad_accum_ghost_and_suit_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bc_p = root / "ddp_bc.tsv.gz"
            bc_lines = [ln for g in range(3) for ln in _make_complete_tonpuusen_bc_lines(f"b-{g}")]
            with gzip.open(bc_p, "wt", encoding="utf-8") as fp:
                fp.writelines(bc_lines)
            rounds = parse_bc_rounds(bc_p)
            # 24 rounds, batch_size=2 -> 12 batches? Add 1 more game -> 32 rounds -> wait: use 15 rounds (bsz=2 -> 8 global batches -> rank 0 gets 4, rank 1 gets 3 + 1 ghost; or slice 10 rounds? 10 rounds -> 5 global batches -> rank 0 gets 3, rank 1 gets 2 + 1 ghost!)
            sub_rounds = rounds[:10]
            rds_pt = root / "rounds10.pt"
            torch.save(sub_rounds, rds_pt)

            # Single-process exact reference over all 10 rounds (5 global batches) with exact global denoms
            dev = torch.device("cpu")
            torch.manual_seed(501)
            m_ref = TrajectoryBCModel(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, use_next_event_head=True, next_event_weight=0.25,
                use_final_rank_head=True, final_rank_weight=0.15, device=dev, dtype=torch.float32,
            )
            s_ref = LengthBucketBatchSampler([r.round_event_len for r in sub_rounds], batch_size=2, seed=71, bucket_mult=2, drop_last=False)
            ref_micros = [collate_round_batch([sub_rounds[i] for i in idxs], batch_weight=w) for _, idxs, w in s_ref.iter_pending_indexed_batches()]
            den_ref = _compute_microgroup_denoms(ref_micros, is_dist=False, world_size=1, device=dev)
            m_ref.zero_grad(set_to_none=True)
            for mb in ref_micros:
                m_ref(mb, denom_overrides=den_ref, ddp_world_size=1)["loss"].backward()
            g_ref = {n: p.grad.detach().clone() for n, p in m_ref.named_parameters() if p.grad is not None}

            _spawn_2rank("--worker-bc-all-heads", str(bc_p), str(rds_pt), str(root))
            g_ddp = torch.load(root / "bc_ddp_grads.pt", weights_only=True)
            for n in g_ref:
                torch.testing.assert_close(g_ddp[n], g_ref[n], atol=1e-5, rtol=1e-5)

            _spawn_2rank("--worker-bc-resume-world", str(bc_p), str(root))
            c0 = torch.load(root / "bc_cont_0.pt", weights_only=True)
            r0 = torch.load(root / "bc_res_0.pt", weights_only=True)
            for k in c0:
                torch.testing.assert_close(c0[k], r0[k], atol=0.0, rtol=0.0)

            # World-size mismatch on resume must fail closed
            bm1 = TrajectoryBCModel(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, use_next_event_head=True,
                use_final_rank_head=True, final_rank_weight=0.1, device=dev, dtype=torch.float32,
            )
            bopt1 = torch.optim.AdamW(adamw_parameter_groups(bm1, 0.01), lr=3e-4)
            with SpoolingRoundBCDataSource(bc_p, cache_dir=root / "spool", num_hash_buckets=2) as ds_w1:
                with self.assertRaises(ValueError):
                    train_trajectory_bc_steps(
                        bm1, ds_w1, bopt1, batch_size=2, max_steps=3, device=dev,
                        suit_augmentation=True, aug_seed=777, sampler_seed=19,
                        ckpt_dir=root / "bc_ckpt", resume_from=root / "bc_ckpt/traj-bc-2.pt",
                    )

    def test_2rank_ddp_iql_single_game_zero_game_rank_ghost_fallback_and_world_size_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rl_p = root / "one_game.rl.tsv"
            rl_p.write_text("".join(_make_rl_game_lines(seat=0, n_transitions=4)), encoding="utf-8")

            # Single-rank baseline over exact same 1-game file with matching seed=101
            dev = torch.device("cpu")
            with StreamingIQLDataset(
                rl_p, batch_size=2, reservoir_size=16, warmup_transitions=2,
                reward_scheme="placement", seed=101, rank=0, world_size=1, drop_last=False, max_prefetch=0,
            ) as ds_single:
                torch.manual_seed(601)
                m_single = JointStructuredIQLModule(
                    preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                    n_d1_layers=1, n_d2_layers=1, device=dev, dtype=torch.float32,
                )
                opt_single = build_joint_iql_optimizer(m_single, base_lr=1e-4, weight_decay=0.01, device=dev, fused=False)
                rec_single = build_joint_iql_recipe(
                    m_single, base_lr=1e-4, weight_decay=0.01, decoder1_lr_scale=0.25,
                    data_sha256=ds_single.data_sha256,
                    source_data_sha256=ds_single.source_data_sha256,
                    preprocess_recipe_sha256=ds_single.preprocess_recipe_sha256,
                )
                res_single = train_joint_iql_batches(m_single, ds_single, opt_single, recipe=rec_single, device=dev, max_steps=None)

            # 2-rank DDP: only 1 game exists -> one rank receives 0 games & exercises zero-transition ghost_batch()!
            _spawn_2rank("--worker-iql-1game", str(rl_p), str(root))
            o0 = torch.load(root / "iql_1g_0.pt", weights_only=True)
            o1 = torch.load(root / "iql_1g_1.pt", weights_only=True)
            self.assertEqual(o0["step"], o1["step"])
            self.assertEqual(o0["step"], res_single["step"])
            self.assertEqual({o0["pass_seen"], o1["pass_seen"]}, {0, 4})
            for k in o0["state"]:
                torch.testing.assert_close(o0["state"][k], o1["state"][k], atol=1e-6, rtol=1e-5)

            # World-size=2 checkpoint loaded into world-size=1 must fail closed!
            with self.assertRaises(ValueError):
                load_joint_iql_checkpoint(
                    root / "iql_1g_ckpt/joint-iql-1.pt",
                    m_single,
                    opt_single,
                    expected_recipe=rec_single,
                )


# ------------------------------------------------------------------
# §7 & §8: Entrypoints, Trailing Accum, GRP/PBRS Integrity & §9 Strict Env
# ------------------------------------------------------------------


class TestProductionEntrypointsAndGRPIntegrity(unittest.TestCase):
    def test_entrypoint_modes_validation_warmup_scheduler_tensorboard_and_trailing_accum(self):
        env = dict(
            os.environ,
            PYTHONPATH=str(REPO_ROOT),
            TORCHDYNAMO_DISABLE="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            HYDRA_FULL_ERROR="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bc_file = root / "cli.bc.tsv.gz"
            rl_file = root / "cli.rl.tsv.gz"
            # 12 rounds -> with batch_size=4, 3 microbatches; with gradient_accumulation_steps=2, Step 0 gets 2 microbatches & Step 1 flushes the trailing 1!
            bc_lines = [ln for g in range(2) for ln in _make_complete_tonpuusen_bc_lines(f"cli-{g}")][:16]
            # Keep 2 complete games = 16 rounds -> batch_size=3 -> 6 batches -> grad_accum=4 -> Step 0 (4 batches), Step 1 (2 trailing batches)!
            full_bc_lines = [ln for g in range(2) for ln in _make_complete_tonpuusen_bc_lines(f"cli-{g}")]
            with gzip.open(bc_file, "wt", encoding="utf-8") as fp:
                fp.writelines(full_bc_lines)

            rl_lines = [ln for g in range(3) for ln in _make_rl_game_lines(seat=g % 4, n_transitions=3)]  # 9 rows -> bsz=2 -> 5 batches -> accum=3 -> 2 steps!
            with gzip.open(rl_file, "wt", encoding="utf-8") as fp:
                fp.writelines(rl_lines)

            common_enc = [
                "device=cpu",
                "encoder=v2_tiny",
                "encoder.dimension=64",
                "encoder.num_heads=4",
                "encoder.dim_feedforward=128",
                "encoder.n_d1_layers=1",
                "encoder.n_d2_layers=1",
                "num_workers=0",
            ]

            # 1. BC CLI with LinearLR warmup + snapshot_interval != final_step + trailing grad_accum flush + TensorBoard check
            r_bc = subprocess.run(
                [
                    sys.executable, "-B", "-m", "kanachan.training.bc.train",
                    "--config-name", "v2_trajectory", *common_enc,
                    "batch_size=3", "gradient_accumulation_steps=4", "snapshot_interval=1",
                    "optimizer.warmup_steps=2", "optimizer.warmup_start_lr=1e-5", "optimizer.learning_rate=1e-3",
                    "spool_num_buckets=2",
                    f"training_data={bc_file.as_posix()}",
                    f"hydra.run.dir={(root / 'cli_bc').as_posix()}",
                ],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            )
            self.assertEqual(r_bc.returncode, 0, r_bc.stdout + r_bc.stderr)
            ckpt_bc = torch.load(root / "cli_bc/snapshots/traj-bc-2.pt", map_location="cpu", weights_only=True)
            self.assertTrue(ckpt_bc["recipe"]["source_data_sha256"])
            self.assertTrue(ckpt_bc["recipe"]["preprocess_recipe_sha256"])
            self.assertEqual(ckpt_bc["recipe"]["training_config"]["gradient_accumulation_steps"], 4)
            actor_bc = load_model(root / "cli_bc/snapshots/model.kanachan", map_location="cpu")
            # Final model.kanachan must match Rank-0 final checkpoint weights!
            torch.testing.assert_close(
                actor_bc.state_dict()["module.0.module.tile_emb.suit_emb.weight"],
                ckpt_bc["model"]["backbone.tile_emb.suit_emb.weight"],
                atol=0.0, rtol=0.0,
            )

            # Verify TensorBoard summary contains architecture + dual SHA fingerprints
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            ea_bc = EventAccumulator(str(root / "cli_bc/tensorboard"))
            ea_bc.Reload()
            txt_evs = ea_bc.Tensors("architecture/summary/text_summary")
            summary_json = json.loads(txt_evs[0].tensor_proto.string_val[0].decode("utf-8"))
            self.assertEqual(summary_json["training_mode"], "v2_trajectory")
            self.assertEqual(summary_json["source_data_sha256"], ckpt_bc["recipe"]["source_data_sha256"])

            # 2. IQL CLI with LinearLR warmup + plugin reward + trailing grad_accum flush
            plugin_p = root / "my_reward.py"
            plugin_p.write_text(
                "import torch\ndef get_reward(td, contiguous):\n    td['next', 'reward'] = td['next', 'end_of_game'].float()\n",
                encoding="utf-8",
            )
            r_iql = subprocess.run(
                [
                    sys.executable, "-B", "-m", "kanachan.training.iql.train",
                    "--config-name", "v2_joint", *common_enc,
                    "batch_size=2", "gradient_accumulation_steps=3",
                    "replay_buffer_size=16", "warmup_transitions=2", "snapshot_interval=1",
                    "optimizer.warmup_steps=2", "optimizer.warmup_start_lr=1e-5", "optimizer.learning_rate=1e-3",
                    "reward_scheme=plugin", f"reward_plugin={plugin_p.as_posix()}",
                    f"training_data={rl_file.as_posix()}",
                    f"hydra.run.dir={(root / 'cli_iql').as_posix()}",
                ],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            )
            self.assertEqual(r_iql.returncode, 0, r_iql.stdout + r_iql.stderr)
            ckpt_iq = torch.load(root / "cli_iql/snapshots/joint-iql-2.pt", map_location="cpu", weights_only=True)
            self.assertEqual(ckpt_iq["recipe"]["training_config"]["reward_scheme"], "plugin")
            self.assertIsNotNone(ckpt_iq["scheduler"])
            actor_iq = load_model(root / "cli_iql/snapshots/model.kanachan", map_location="cpu")
            torch.testing.assert_close(
                actor_iq.state_dict()["module.0.module.tile_emb.suit_emb.weight"],
                ckpt_iq["model"]["backbone.tile_emb.suit_emb.weight"],
                atol=0.0, rtol=0.0,
            )

            # 3. Verify legacy baselines stay intact (`single_step` BC, `legacy_five_net` IQL) & incompatible options fail closed!
            legacy_bc_file = root / "legacy.bc.tsv"
            legacy_bc_file.write_text("".join(full_bc_lines[:4]), encoding="utf-8")
            r_leg_bc = subprocess.run(
                [
                    sys.executable, "-B", "-m", "kanachan.training.bc.train",
                    "--config-name", "decision", *common_enc,
                    "batch_size=2",
                    f"training_data={legacy_bc_file.as_posix()}",
                    f"hydra.run.dir={(root / 'leg_bc').as_posix()}",
                ],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            )
            self.assertEqual(r_leg_bc.returncode, 0, r_leg_bc.stdout + r_leg_bc.stderr)

            r_leg_iq = subprocess.run(
                [
                    sys.executable, "-B", "-m", "kanachan.training.iql.train",
                    "--config-name", "decision",
                    "device=cpu",
                    "encoder=decision_tiny",
                    "encoder.dimension=64",
                    "encoder.num_heads=2",
                    "encoder.dim_feedforward=128",
                    "encoder.num_layers=1",
                    "num_workers=0",
                    "batch_size=2", "replay_buffer_size=4", "contiguous_training_data=true",
                    f"reward_plugin={plugin_p.as_posix()}",
                    f"training_data={rl_file.as_posix()}",
                    f"hydra.run.dir={(root / 'leg_iq').as_posix()}",
                ],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            )
            self.assertEqual(r_leg_iq.returncode, 0, r_leg_iq.stdout + r_leg_iq.stderr)

            # Incompatible options fail closed
            for cmd in (
                [
                    sys.executable, "-B", "-m", "kanachan.training.bc.train", "--config-name", "v2_trajectory", *common_enc,
                    "batch_size=2", "final_rank_weight=0.1", "use_final_rank_head=false",
                    f"training_data={bc_file.as_posix()}", f"hydra.run.dir={(root / 'bad_bc1').as_posix()}",
                ],
                [
                    sys.executable, "-B", "-m", "kanachan.training.iql.train", "--config-name", "v2_joint", *common_enc,
                    "batch_size=2", "kl_coef=0.1", f"training_data={rl_file.as_posix()}",
                    f"hydra.run.dir={(root / 'bad_iq1').as_posix()}",
                ],
                [
                    sys.executable, "-B", "-m", "kanachan.training.iql.train", "--config-name", "v2_joint", *common_enc,
                    "batch_size=2", "quantile_cql_alpha=0.1", "use_cql_quantile_head=false",
                    f"training_data={rl_file.as_posix()}", f"hydra.run.dir={(root / 'bad_iq2').as_posix()}",
                ],
            ):
                bad_p = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60)
                self.assertNotEqual(bad_p.returncode, 0, cmd)

    def test_grp_pbrs_telescoping_boundary_guards_and_strict_env_mode(self):
        # 1. Analytical PBRS telescoping over concatenated 2-game episode including skipped rounds
        gm = PureStateGRP()
        frozen = FrozenGRP(gm, {"top1_ece": 0.01, "classwise_ece": 0.01, "brier": 0.4}, require_gate=True)
        pbrs = FrozenGRPPBRSReward(frozen)

        sp_rows = torch.tensor(
            [
                _make_sparse_row(seat=1, chang=0, ju=0) + [NUM_TYPES_OF_SPARSE_FEATURES] * 4,
                _make_sparse_row(seat=1, chang=0, ju=0) + [NUM_TYPES_OF_SPARSE_FEATURES] * 4,
                _make_sparse_row(seat=1, chang=0, ju=2) + [NUM_TYPES_OF_SPARSE_FEATURES] * 4,  # skipped ju=1 round!
                _make_sparse_row(seat=2, chang=0, ju=0) + [NUM_TYPES_OF_SPARSE_FEATURES] * 4,  # Game 2 starts (seat=2)
                _make_sparse_row(seat=2, chang=0, ju=1) + [NUM_TYPES_OF_SPARSE_FEATURES] * 4,
            ],
            dtype=torch.int32,
        )
        # Pad/truncate columns cleanly to MAX_NUM_ACTIVE_SPARSE_FEATURES
        sp = torch.full((5, MAX_NUM_ACTIVE_SPARSE_FEATURES), NUM_TYPES_OF_SPARSE_FEATURES, dtype=torch.int32)
        for idx, row in enumerate([
            _make_sparse_row(seat=1, chang=0, ju=0),
            _make_sparse_row(seat=1, chang=0, ju=0),
            _make_sparse_row(seat=1, chang=0, ju=2),
            _make_sparse_row(seat=2, chang=0, ju=0),
            _make_sparse_row(seat=2, chang=0, ju=1),
        ]):
            sp[idx, : len(row)] = torch.tensor(row, dtype=torch.int32)
        nu = torch.tensor(
            [
                [0, 0, 25000, 25000, 25000, 25000],
                [0, 0, 25000, 25000, 25000, 25000],
                [0, 0, 23000, 28000, 25000, 24000],
                [0, 0, 25000, 25000, 25000, 25000],
                [0, 0, 24000, 24000, 32000, 20000],
            ],
            dtype=torch.int32,
        )
        res = torch.zeros((5, 14), dtype=torch.int32)
        res[2, 10:14] = torch.tensor([20000, 35000, 26000, 19000])  # Game 1 seat 1 gets 1st (+1.5)
        res[4, 10:14] = torch.tensor([15000, 22000, 28000, 35000])  # Game 2 seat 2 gets 2nd (+0.5)
        eor = torch.tensor([False, True, True, True, True], dtype=torch.bool)
        eog = torch.tensor([False, False, True, False, True], dtype=torch.bool)
        ep = TensorDict(
            {
                "sparse": sp,
                "numeric": nu,
                "next": {"results": res, "end_of_round": eor, "end_of_game": eog},
            },
            batch_size=[5],
        )
        pbrs(ep, contiguous=True)
        phi0_g1 = float(frozen.potential(sp[0:1], nu[0:1]).item())
        phi0_g2 = float(frozen.potential(sp[3:4], nu[3:4]).item())
        self.assertAlmostEqual(float(ep["next", "reward"][:3].sum().item()), 1.5 - phi0_g1, places=5)
        self.assertAlmostEqual(float(ep["next", "reward"][3:].sum().item()), 0.5 - phi0_g2, places=5)

        # Mid-round (chang, ju, ben) shift WITHOUT end_of_round -> must fail closed!
        ep_bad = ep.clone()
        ep_bad["next", "end_of_round"][1] = False
        with self.assertRaises(ValueError):
            pbrs(ep_bad, contiguous=True)

        # Missing end_of_round on end_of_game -> must fail closed!
        ep_bad2 = ep.clone()
        ep_bad2["next", "end_of_round"][2] = False
        with self.assertRaises(ValueError):
            pbrs(ep_bad2, contiguous=True)

        # 2. Verify test_env fail-closed strict acceptance behavior cleanly
        res_strict = subprocess.run(
            [
                sys.executable, "-c",
                "import unittest; from kanachan.training.core.test_env import require_or_skip_real_data\n"
                "class T(unittest.TestCase):\n"
                "  def test_x(self): require_or_skip_real_data(self, 'bc')\n"
                "unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(T))\n"
                "r = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(T))\n"
                "raise SystemExit(0 if not r.wasSuccessful() else 1)",
            ],
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(REPO_ROOT),
                VAL_BC_ENV_VAR: "",
                STRICT_ACCEPTANCE_ENV_VAR: "1",
            },
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(res_strict.returncode, 0, res_strict.stdout + res_strict.stderr)

    def test_real_data_streaming_bc_and_iql_acceptance(self):
        val_bc = require_or_skip_real_data(self, "bc", context="real-data BC external sampler check")
        val_rl = require_or_skip_real_data(self, "rl", context="real-data IQL multi-pass streaming check")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Extract first complete games without truncation
            by_u: dict[str, list[str]] = {}
            last_i: dict[str, int] = {}
            with gzip.open(val_bc, "rt", encoding="utf-8") as fp:
                for i, line in enumerate(fp):
                    if i >= 20000:
                        break
                    u = line.split("\t", 1)[0]
                    by_u.setdefault(u, []).append(line)
                    last_i[u] = i
            comp_games = [lns for u, lns in by_u.items() if last_i[u] < 15000 and len(lns) >= 60][:12]
            self.assertGreaterEqual(len(comp_games), 8)
            comp = [ln for g_lns in comp_games for ln in g_lns]
            bc_sub = root / "real_bc.tsv.gz"
            with gzip.open(bc_sub, "wt", encoding="utf-8") as fp:
                fp.writelines(comp)

            with SpoolingRoundBCDataSource(bc_sub, cache_dir=root / "bc_sp", num_hash_buckets=8) as src:
                with src.create_sampler(batch_size=4, seed=42, window_items=16) as samp:
                    self.assertGreater(len(samp), 0)
                    idxs = samp.assigned_sample_indices()
                    self.assertEqual(set(idxs), set(range(len(src))))

            with StreamingIQLDataset(
                val_rl, batch_size=8, reservoir_size=256, warmup_transitions=32,
                reward_scheme="round_and_placement", max_games=6, max_prefetch=2,
            ) as ids:
                b1 = [b for b in ids]
                n_seen = ids.pass_transitions_seen
                ids.reset_pass()
                b2 = [b for b in ids]
                self.assertEqual(len(b1), len(b2))
                self.assertEqual(ids.pass_transitions_seen, n_seen)
                self.assertEqual(ids._reservoir.total_seen, 2 * n_seen)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker-bc-all-heads":
        _worker_bc_all_heads(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--worker-bc-resume-world":
        _worker_bc_resume_world_check(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--worker-iql-1game":
        _worker_iql_single_game_ddp(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        unittest.main()
