"""Final regression, determinism, DDP/ZeRO, bounded-resource & CUDA FP16 tests for Structured V2 (§§1-8)."""

from __future__ import annotations

import ast
import builtins
from dataclasses import is_dataclass
import errno
import gzip
import hashlib
import io
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
import unittest
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
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
from kanachan.nn.structured_iql import JointStructuredIQLModule
from kanachan.structured_v2 import OFFSETS_136, PAD_ACTION_ID
import kanachan.training.bc.streaming_v2 as bc_streaming_mod
from kanachan.training.bc.streaming_v2 import (
    DiskLengthBucketBatchSampler,
    SpoolingRoundBCDataSource,
)
from kanachan.training.bc.trajectory_v2 import (
    LengthBucketBatchSampler,
    TrajectoryBCModel,
    _compute_microgroup_denoms,
    collate_round_batch,
    load_trajectory_bc_checkpoint,
    save_trajectory_bc_checkpoint,
    train_trajectory_bc_steps,
)
from kanachan.training.core.dataset_fingerprint import sha256_file_bytes
from kanachan.training.core.optimizers import adamw_parameter_groups
from kanachan.training.core.test_env import (
    require_or_skip_cuda,
    require_or_skip_distributed,
)
from kanachan.training.iql.joint_v2 import (
    attach_structured_v2_transition_features,
    build_joint_iql_optimizer,
    build_joint_iql_recipe,
    save_joint_iql_checkpoint,
    train_joint_iql_batches,
)
from kanachan.training.iql.streaming_v2 import StreamingIQLDataset


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
            ["ipconfig"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
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


def _base_subproc_env() -> dict[str, str]:
    return dict(
        os.environ,
        PYTHONPATH=str(REPO_ROOT),
        TORCHDYNAMO_DISABLE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        HYDRA_FULL_ERROR="1",
        PYTHONDONTWRITEBYTECODE="1",
        USE_LIBUV="0",
    )


def _ddp_env(rank: int, world_size: int, port: int) -> dict[str, str]:
    env = _base_subproc_env()
    env.update(
        {
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
    _wait_2rank_procs(procs, label=f"mode={mode}")


def _wait_2rank_procs(
    procs: list[subprocess.Popen[str]], *, label: str, timeout_s: float = 180.0
) -> None:
    deadline = time.monotonic() + timeout_s
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
                        f"2-rank subprocess ({label}) rank={bad_rank} failed early ({rc}):\n{so}\n{se}"
                    )
            if time.monotonic() >= deadline:
                for p in procs:
                    if p.poll() is None:
                        p.kill()
                outs = [p.communicate(timeout=5) for p in procs]
                raise AssertionError(f"2-rank subprocess ({label}) timed out:\n{outs}")
            time.sleep(0.1)
        outs = [p.communicate(timeout=5) for p in procs]
    finally:
        for p in procs:
            for s in (p.stdout, p.stderr):
                if s is not None:
                    s.close()
    for rank, (p, (so, se)) in enumerate(zip(procs, outs, strict=True)):
        if p.returncode != 0:
            raise AssertionError(
                f"2-rank subprocess ({label}) rank={rank} failed ({p.returncode}):\n{so}\n{se}"
            )


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


def _make_complete_tonpuusen_bc_lines(
    uuid: str,
    *,
    final_scores: tuple[int, int, int, int] = (32000, 27000, 22000, 19000),
    steps_per_round: int = 2,
) -> list[str]:
    assert sum(final_scores) == 100000
    assert max(final_scores) >= 30000
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
            pr_list.append(5 + s_turn * 148 + 4 * free_tiles[turn % len(free_tiles)] + 0)
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


def _make_rl_game_lines(
    *,
    seat: int,
    n_transitions: int,
    score_step: int = 1000,
    corrupt_sparse: bool = False,
) -> list[str]:
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
        if corrupt_sparse and t == 0:
            sp_cur = [2, 5, 7, 23, 39, 55, 71 + seat, 75 + chang, 78 + ju, 82 + left, 152 + 1]
            closed = {1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1, 21: 1, 22: 1, 31: 4, 32: 1}
            for t37, cnt in sorted(closed.items()):
                for k in range(cnt):
                    sp_cur.append(337 + OFFSETS_136[t37] + k)
            sp_cur.append(337 + OFFSETS_136[31] + 0)
            sp_cur.append(473 + 32)
        nu_cur = [0, 0, *cur_scores]
        pr_cur = [0] + [5 + seat * 148 + ((4 + k) * 4) for k in range(r_t)]
        if corrupt_sparse and t == 0:
            pr_cur = [0, 5 + seat * 148 + (4 * 31) + 2]
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


def _make_bc_file(
    path: Path, *, num_games: int = 4, steps_per_round: int = 2, seed: int | None = None
) -> Path:
    """Create valid Tonpuusen source-v3 BC file (8 observer rounds/game, winning top score >= 30000)."""
    rounds_lines: list[list[str]] = []
    for g in range(num_games):
        s_off = 6000 + ((g * 700) % 5000)
        scores = (25000 + s_off, 26000, 24000, 25000 - s_off)
        rounds_lines.append(
            _make_complete_tonpuusen_bc_lines(
                f"g-{g:04d}", final_scores=scores, steps_per_round=steps_per_round
            )
        )
    if seed is not None:
        orig_rounds = list(rounds_lines)
        random.Random(seed).shuffle(rounds_lines)
        if rounds_lines == orig_rounds and len(rounds_lines) >= 2:
            rounds_lines.reverse()
    flat = [ln for gl in rounds_lines for ln in gl]
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fp:
            fp.writelines(flat)
    else:
        path.write_text("".join(flat), encoding="utf-8")
    return path


def _make_rl_file(path: Path, *, num_games: int = 10, n_transitions: int = 3) -> Path:
    lines = [
        ln
        for g in range(num_games)
        for ln in _make_rl_game_lines(seat=g % 4, n_transitions=n_transitions, score_step=1000 + g * 200)
    ]
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fp:
            fp.writelines(lines)
    else:
        path.write_text("".join(lines), encoding="utf-8")
    return path


def _tiny_bc_model(device: torch.device) -> TrajectoryBCModel:
    return TrajectoryBCModel(
        preset="v2_tiny",
        dimension=64,
        num_heads=4,
        dim_feedforward=128,
        n_d1_layers=1,
        n_d2_layers=1,
        use_next_event_head=True,
        next_event_weight=0.2,
        use_final_rank_head=True,
        final_rank_weight=0.05,
        device=device,
        dtype=torch.float32,
    )


def _tiny_iql_module(device: torch.device, dtype: torch.dtype = torch.float32) -> JointStructuredIQLModule:
    return JointStructuredIQLModule(
        preset="v2_tiny",
        dimension=64,
        num_heads=4,
        dim_feedforward=128,
        n_d1_layers=1,
        n_d2_layers=1,
        discount_factor=0.99,
        expectile=0.7,
        final_rank=True,
        final_rank_weight=0.05,
        device=device,
        dtype=dtype,
    )


def _assert_state_close(
    a: Any,
    b: Any,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
    path: str = "root",
) -> None:
    if is_dataclass(a) and not isinstance(a, type):
        assert type(a) is type(b), f"Dataclass type mismatch at {path}: {type(a)} vs {type(b)}"
        _assert_state_close(vars(a), vars(b), atol=atol, rtol=rtol, path=path)
    elif isinstance(a, dict) and isinstance(b, dict):
        assert set(a.keys()) == set(b.keys()), f"Key mismatch at {path}: {set(a.keys())} vs {set(b.keys())}"
        for k in sorted(a.keys(), key=repr):
            _assert_state_close(a[k], b[k], atol=atol, rtol=rtol, path=f"{path}.{k!r}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b), f"Len mismatch at {path}: {len(a)} vs {len(b)}"
        for i, (va, vb) in enumerate(zip(a, b, strict=True)):
            _assert_state_close(va, vb, atol=atol, rtol=rtol, path=f"{path}[{i}]")
    elif torch.is_tensor(a) and torch.is_tensor(b):
        torch.testing.assert_close(a.cpu(), b.cpu(), atol=atol, rtol=rtol)
    else:
        assert a == b, f"Value mismatch at {path}: {a!r} vs {b!r}"


def _make_rl_transition_batch(batch_size: int, seed: int, *, weight: float = 1.0) -> TensorDict:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    sp = torch.zeros((batch_size, MAX_NUM_ACTIVE_SPARSE_FEATURES), dtype=torch.int32)
    nsp = torch.zeros_like(sp)
    pr = torch.full(
        (batch_size, MAX_LENGTH_OF_PROGRESSION_FEATURES),
        NUM_TYPES_OF_PROGRESSION_FEATURES,
        dtype=torch.int32,
    )
    npr = pr.clone()
    nu = torch.zeros((batch_size, 6), dtype=torch.int32)
    nnu = torch.zeros_like(nu)
    ca = torch.full((batch_size, MAX_NUM_ACTION_CANDIDATES), PAD_ACTION_ID, dtype=torch.int32)
    nca = ca.clone()
    for b in range(batch_size):
        row = _make_sparse_row(seat=(b + seed) % 4, left=60 - b)
        sp[b, : len(row)] = torch.tensor(row, dtype=torch.int32)
        sp[b, len(row) :] = NUM_TYPES_OF_SPARSE_FEATURES
        nrow = _make_sparse_row(seat=(b + seed) % 4, left=56 - b)
        nsp[b, : len(nrow)] = torch.tensor(nrow, dtype=torch.int32)
        nsp[b, len(nrow) :] = NUM_TYPES_OF_SPARSE_FEATURES
        nu[b, 2:] = torch.tensor([25000, 25000, 25000, 25000], dtype=torch.int32)
        nnu[b, 2:] = torch.tensor([26000, 24000, 25000, 25000], dtype=torch.int32)
        pr[b, 0] = 0
        npr[b, 0] = 0
        npr[b, 1] = 5 + ((b + seed) % 4) * 148 + (4 * (1 + (b % 3))) * 1
        ca[b, :4] = torch.tensor([4, 44, 84, 130], dtype=torch.int32)
        nca[b, :4] = torch.tensor([4, 44, 84, 130], dtype=torch.int32)
    act = torch.randint(0, 4, (batch_size,), generator=rng, dtype=torch.long)
    reward = torch.linspace(-0.5, 0.75, steps=batch_size, dtype=torch.float32)
    discount = torch.full((batch_size,), 0.99, dtype=torch.float32)
    done = torch.zeros((batch_size,), dtype=torch.bool)
    done[-1] = seed % 2 == 0
    eog = done.clone()
    final_rank = torch.arange(batch_size, dtype=torch.long) % 4
    td = TensorDict(
        {
            "sparse": sp,
            "numeric": nu,
            "progression": pr,
            "candidates": ca,
            "action": act,
            "final_rank": final_rank,
            "batch_weight": torch.full((batch_size,), float(weight), dtype=torch.float32),
            "next": {
                "sparse": nsp,
                "numeric": nnu,
                "progression": npr,
                "candidates": nca,
                "reward": reward,
                "discount": discount,
                "done": done,
                "end_of_game": eog,
            },
        },
        batch_size=[batch_size],
    )
    return attach_structured_v2_transition_features(td)


# ------------------------------------------------------------------
# Subprocess DDP Workers
# ------------------------------------------------------------------


def _worker_01_ddp_production_accum(
    rank: int, world_size: int, bc_path: str, rl_path: str, out_dir: str
) -> None:
    """Test real train_trajectory_bc_steps & train_joint_iql_batches on 2 ranks with grad_accum=2/3 & tails."""
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        out = Path(out_dir)
        dev = torch.device("cpu")
        with SpoolingRoundBCDataSource(
            Path(bc_path), cache_dir=out / "bc_cache_shared", reuse_cache=True, num_hash_buckets=2
        ) as ds:
            # 14 rounds with batch_size=2 -> 7 global batches -> R0 gets 4 real, R1 gets 3 real + 1 ghost!
            rounds_14 = [ds[i] for i in range(14)]

        # Part A1: BC production DDP with grad_accum=2 & 3 (complete + partial tail + ghost batch) + NaN guard
        for grad_accum, inject_nan in ((2, False), (3, False), (2, True)):
            torch.manual_seed(111 + grad_accum)
            m_ddp = _tiny_bc_model(dev)
            m_wrap = DistributedDataParallel(m_ddp)
            opt_ddp = torch.optim.AdamW(adamw_parameter_groups(m_ddp, weight_decay=0.01), lr=5e-4)

            def _bc_nan_rank1(loss: Tensor, cur_step: int) -> Tensor:
                if inject_nan and rank == 1 and cur_step == 1:
                    return loss * float("nan")
                return loss

            pre_sd = {k: v.clone() for k, v in m_ddp.state_dict().items()}
            res_ddp = train_trajectory_bc_steps(
                m_wrap,
                rounds_14,
                opt_ddp,
                batch_size=2,
                device=dev,
                sampler_seed=71,
                bucket_mult=2,
                drop_last=False,
                grad_clip=1.0,
                grad_accum_steps=grad_accum,
                prefetch_batches=0,
                data_sha256="ddp-bc-accum-test",
                loss_transform=_bc_nan_rank1 if inject_nan else None,
            )
            gathered_sd = [None, None]
            dist.all_gather_object(gathered_sd, {k: v.cpu() for k, v in m_ddp.state_dict().items()})
            _assert_state_close(
                gathered_sd[0], gathered_sd[1], atol=0.0, rtol=0.0, path=f"bc_r0_r1_{grad_accum}_{inject_nan}"
            )

            if inject_nan:
                assert res_ddp["skipped_steps"] == 1
                assert res_ddp["optimizer_steps"] == 1
            else:
                assert res_ddp["skipped_steps"] == 0
                assert res_ddp["optimizer_steps"] == (4 + grad_accum - 1) // grad_accum
                assert any(
                    not torch.equal(pre_sd[k], m_ddp.state_dict()[k])
                    for k in pre_sd
                    if pre_sd[k].is_floating_point()
                )
                if rank == 0:
                    torch.manual_seed(111 + grad_accum)
                    m_ref = _tiny_bc_model(dev)
                    opt_ref = torch.optim.AdamW(adamw_parameter_groups(m_ref, weight_decay=0.01), lr=5e-4)
                    lengths_14 = [r.round_event_len for r in rounds_14]
                    s0 = LengthBucketBatchSampler(
                        lengths_14, batch_size=2, seed=71, bucket_mult=2, rank=0, world_size=2, drop_last=False
                    )
                    s1 = LengthBucketBatchSampler(
                        lengths_14, batch_size=2, seed=71, bucket_mult=2, rank=1, world_size=2, drop_last=False
                    )
                    m0 = [
                        collate_round_batch([rounds_14[i] for i in idxs], batch_weight=bw)
                        for _p, idxs, bw in s0.iter_pending_indexed_batches()
                    ]
                    m1 = [
                        collate_round_batch([rounds_14[i] for i in idxs], batch_weight=bw)
                        for _p, idxs, bw in s1.iter_pending_indexed_batches()
                    ]
                    for start in range(0, len(m0), grad_accum):
                        grp_0 = m0[start : start + grad_accum]
                        grp_1 = m1[start : start + grad_accum]
                        grp_all = grp_0 + grp_1
                        denoms = _compute_microgroup_denoms(grp_all, is_dist=False, world_size=1, device=dev)
                        opt_ref.zero_grad(set_to_none=True)
                        for mb in grp_all:
                            m_ref(mb, denom_overrides=denoms, ddp_world_size=1)["loss"].backward()
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in m_ref.parameters() if p.grad is not None], 1.0
                        )
                        opt_ref.step()
                    _assert_state_close(
                        m_ddp.state_dict(), m_ref.state_dict(), atol=1e-5, rtol=1e-5, path=f"bc_ref_{grad_accum}"
                    )

        # Part B1: Joint IQL list[TensorDict] DDP with equal microbatch count + ghost (weight=0) + grad_accum=2/3 + NaN
        r0_list = [_make_rl_transition_batch(3, 10 + i, weight=1.0) for i in range(5)]
        r1_list_5 = [
            *(_make_rl_transition_batch(2, 20 + i, weight=1.0) for i in range(3)),
            _make_rl_transition_batch(2, 99, weight=0.0),
            _make_rl_transition_batch(2, 100, weight=0.0),
        ]
        for grad_accum, inject_nan in ((2, False), (3, False), (2, True)):
            n_micro = 5 if grad_accum == 2 else 4
            local_batches = (r0_list if rank == 0 else r1_list_5)[:n_micro]
            torch.manual_seed(222 + grad_accum)
            mod_ddp = _tiny_iql_module(dev)
            wrap_iql = DistributedDataParallel(mod_ddp)
            opt_iql = build_joint_iql_optimizer(mod_ddp, base_lr=5e-4, weight_decay=0.01, device=dev)
            rec_iql = build_joint_iql_recipe(
                mod_ddp, base_lr=5e-4, weight_decay=0.01, decoder1_lr_scale=0.25
            )

            def _bad_on_rank1_last(loss: Tensor, m_step: int) -> Tensor:
                if inject_nan and rank == 1 and m_step == 2:
                    return loss * float("nan")
                return loss

            pre_sd = {k: v.clone() for k, v in mod_ddp.state_dict().items()}
            captured_ddp_grads: dict[int, dict[str, Tensor]] = {}

            def _cap_iql_grad(runner_m: Any, _opt: Any, cur_step: int) -> None:
                raw_m = runner_m.module if isinstance(runner_m, DistributedDataParallel) else runner_m
                captured_ddp_grads[cur_step] = {
                    n: p.grad.detach().cpu().clone()
                    for n, p in raw_m.named_parameters()
                    if p.grad is not None
                }

            res_iql = train_joint_iql_batches(
                wrap_iql,
                local_batches,
                opt_iql,
                recipe=rec_iql,
                device=dev,
                grad_accum_steps=grad_accum,
                ddp_wrapped=wrap_iql,
                loss_transform=_bad_on_rank1_last if inject_nan else None,
                post_backward_hook=_cap_iql_grad if not inject_nan else None,
            )
            g_iql = [None, None]
            dist.all_gather_object(g_iql, {k: v.cpu() for k, v in mod_ddp.state_dict().items()})
            _assert_state_close(
                g_iql[0], g_iql[1], atol=0.0, rtol=0.0, path=f"iql_r0_r1_{grad_accum}_{inject_nan}"
            )
            if inject_nan:
                assert res_iql["skipped_steps"] == 1
                assert res_iql["optimizer_steps"] == 2
            else:
                assert res_iql["skipped_steps"] == 0
                assert any(
                    not torch.equal(pre_sd[k], mod_ddp.state_dict()[k])
                    for k in pre_sd
                    if pre_sd[k].is_floating_point()
                )
                if rank == 0:
                    torch.manual_seed(222 + grad_accum)
                    mod_ref = _tiny_iql_module(dev)
                    opt_ref = build_joint_iql_optimizer(mod_ref, base_lr=5e-4, weight_decay=0.01, device=dev)
                    l0 = r0_list[:n_micro]
                    l1 = r1_list_5[:n_micro]
                    for step_idx, start in enumerate(range(0, n_micro, grad_accum)):
                        grp_0 = l0[start : start + grad_accum]
                        grp_1 = l1[start : start + grad_accum]
                        w_sum = sum(float(b["batch_weight"].mean().item()) for b in grp_0 + grp_1)
                        opt_ref.zero_grad(set_to_none=True)
                        for mb in grp_0 + grp_1:
                            lo = mod_ref(mb, loss_scale=1.0 / max(1e-8, w_sum))
                            lo["loss"].backward()
                        ref_grads = {
                            n: p.grad.detach().cpu().clone()
                            for n, p in mod_ref.named_parameters()
                            if p.grad is not None
                        }
                        _assert_state_close(
                            captured_ddp_grads[step_idx],
                            ref_grads,
                            atol=1e-6,
                            rtol=1e-6,
                            path=f"iql_grad_ref_{grad_accum}_step{step_idx}",
                        )
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in mod_ref.parameters() if p.grad is not None], 1.0
                        )
                        opt_ref.step()
                        mod_ref.update_targets()
                    _assert_state_close(
                        mod_ddp.state_dict(), mod_ref.state_dict(), atol=5e-5, rtol=5e-2, path=f"iql_ref_{grad_accum}"
                    )

        # Part B2: 2-rank StreamingIQLDataset with odd game count + grad_accum_steps=2 (ghost batch via stream EOF)
        with StreamingIQLDataset(
            Path(rl_path),
            batch_size=2,
            reservoir_size=16,
            warmup_transitions=2,
            seed=71,
            rank=rank,
            world_size=world_size,
            max_prefetch=0,
        ) as ds_iq:
            torch.manual_seed(333)
            mod_s = _tiny_iql_module(dev)
            wrap_s = DistributedDataParallel(mod_s)
            opt_s = build_joint_iql_optimizer(mod_s, base_lr=5e-4, weight_decay=0.01, device=dev)
            rec_s = build_joint_iql_recipe(
                mod_s,
                base_lr=5e-4,
                weight_decay=0.01,
                decoder1_lr_scale=0.25,
                data_sha256=ds_iq.data_sha256,
                source_data_sha256=ds_iq.source_data_sha256,
                preprocess_recipe_sha256=ds_iq.preprocess_recipe_sha256,
            )
            res_s = train_joint_iql_batches(
                wrap_s, ds_iq, opt_s, recipe=rec_s, device=dev, grad_accum_steps=2, ddp_wrapped=wrap_s
            )
            assert res_s["skipped_steps"] == 0 and res_s["optimizer_steps"] >= 2
            g_s = [None, None]
            dist.all_gather_object(g_s, {k: v.cpu() for k, v in mod_s.state_dict().items()})
            _assert_state_close(g_s[0], g_s[1], atol=0.0, rtol=0.0, path="iql_stream_r0_r1")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker_03_bc_ddp_shared_spool_and_schedule(
    rank: int, world_size: int, bc_path: str, work_dir: str
) -> None:
    """Rank 0 builds disk cache & schedules completely BEFORE rank 1 enters the cache/schedule broadcast."""
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        wd = Path(work_dir)
        cache_dir = wd / "shared_cache"
        sched_dir = wd / "shared_sched"
        gate_spool = wd / "spool_built.flag"
        gate_sched0 = wd / "sched0_built.flag"
        gate_sched1 = wd / "sched1_built.flag"

        orig_build_cache = SpoolingRoundBCDataSource._build_disk_cache
        orig_build_sched = DiskLengthBucketBatchSampler._build_epoch_schedule_disk
        orig_prep_fp = bc_streaming_mod.compute_bc_preprocess_fingerprint

        def _wrapped_prep_fp(*a: Any, **kw: Any) -> str:
            res = orig_prep_fp(*a, **kw)
            # Runs right AFTER collective #1 (compute_source_data_fingerprint) and BEFORE cache check/broadcast (#2).
            if rank == 1 and dist.is_initialized():
                deadline = time.monotonic() + 30.0
                while not gate_spool.exists():
                    assert time.monotonic() < deadline, "Timed out waiting for rank 0 spool build"
                    time.sleep(0.02)
            return res

        def _wrapped_build_cache(self_ds: SpoolingRoundBCDataSource, *a: Any, **kw: Any) -> Any:
            try:
                return orig_build_cache(self_ds, *a, **kw)
            finally:
                if rank == 0:
                    gate_spool.write_text("done", encoding="utf-8")

        def _wrapped_build_sched(self_s: DiskLengthBucketBatchSampler, epoch: int) -> None:
            try:
                orig_build_sched(self_s, epoch)
            finally:
                if rank == 0:
                    target = gate_sched0 if epoch == 0 else gate_sched1
                    target.write_text("done", encoding="utf-8")

        bc_streaming_mod.compute_bc_preprocess_fingerprint = _wrapped_prep_fp  # type: ignore[assignment]
        SpoolingRoundBCDataSource._build_disk_cache = _wrapped_build_cache  # type: ignore[method-assign]
        DiskLengthBucketBatchSampler._build_epoch_schedule_disk = _wrapped_build_sched  # type: ignore[method-assign]
        try:
            ds = SpoolingRoundBCDataSource(Path(bc_path), cache_dir=cache_dir, reuse_cache=True)
            # Immediate collective after __init__: would mismatch/hang if rank 1 skipped broadcast!
            s1 = torch.tensor([rank + 1], dtype=torch.int64)
            dist.all_reduce(s1)
            assert int(s1.item()) == 3

            if rank == 1:
                deadline = time.monotonic() + 30.0
                while not gate_sched0.exists():
                    assert time.monotonic() < deadline, "Timed out waiting for rank 0 epoch-0 schedule build"
                    time.sleep(0.02)

            sampler = ds.create_sampler(
                batch_size=3,
                seed=77,
                bucket_mult=2,
                rank=rank,
                world_size=world_size,
                drop_last=False,
                schedule_dir=sched_dir,
            )
            s2 = torch.tensor([sampler.num_rank_batches], dtype=torch.int64)
            dist.all_reduce(s2)
            assert int(s2.item()) == 2 * sampler.num_rank_batches

            if rank == 1:
                deadline = time.monotonic() + 30.0
                while not gate_sched1.exists():
                    assert time.monotonic() < deadline, "Timed out waiting for rank 0 epoch-1 schedule build"
                    time.sleep(0.02)

            sampler.reset_epoch(1)
            s3 = torch.tensor([sampler.num_rank_batches], dtype=torch.int64)
            dist.all_reduce(s3)
            assert int(s3.item()) > 0
            covered = sampler.assigned_sample_indices()
            all_cov: list[Any] = [None, None]
            dist.all_gather_object(all_cov, covered)
            flat = all_cov[0] + all_cov[1]
            assert sorted(flat) == list(range(len(ds))), f"Coverage mismatch: {len(flat)} vs {len(ds)}"
            sha_list = [None, None]
            dist.all_gather_object(sha_list, ds.ordered_sample_sha256)
            assert sha_list[0] == sha_list[1]
            ds.close()
        finally:
            bc_streaming_mod.compute_bc_preprocess_fingerprint = orig_prep_fp  # type: ignore[assignment]
            SpoolingRoundBCDataSource._build_disk_cache = orig_build_cache  # type: ignore[method-assign]
            DiskLengthBucketBatchSampler._build_epoch_schedule_disk = orig_build_sched  # type: ignore[method-assign]
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ------------------------------------------------------------------
# Main Test Suite
# ------------------------------------------------------------------


class TestStructuredV2FinalFixes(unittest.TestCase):
    def test_01_ddp_2rank_production_bc_and_iql_no_sync_grad_accum_and_partial_tail(self):
        require_or_skip_distributed(self, context="2-rank production BC & IQL DDP grad_accum")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bc_file = _make_bc_file(root / "bc16.tsv.gz", num_games=2, steps_per_round=2)
            rl_file = _make_rl_file(root / "rl5.tsv.gz", num_games=5, n_transitions=3)
            _spawn_2rank("--worker-01", str(bc_file), str(rl_file), str(root))

    def test_02_iql_prefetch_worker_error_not_swallowed_by_checkpoint_both_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_parser = root / "parser_err.rl.tsv"
            bad_parser.write_text(
                "".join(
                    [
                        *_make_rl_game_lines(seat=0, n_transitions=2),
                        "\n",
                        *_make_rl_game_lines(seat=1, n_transitions=2),
                    ]
                ),
                encoding="utf-8",
            )
            bad_feat = root / "feat_err.rl.tsv"
            bad_feat.write_text(
                "".join(
                    [
                        *_make_rl_game_lines(seat=0, n_transitions=2),
                        *_make_rl_game_lines(seat=1, n_transitions=2, corrupt_sparse=True),
                        *_make_rl_game_lines(seat=2, n_transitions=2),
                    ]
                ),
                encoding="utf-8",
            )

            for label, rl_file in (("parser_blank", bad_parser), ("feature_conservation", bad_feat)):
                # Timing A: worker hits error and queues it BEFORE caller invokes state_dict()
                ds_a = StreamingIQLDataset(
                    rl_file, batch_size=2, reservoir_size=8, warmup_transitions=2, max_prefetch=2
                )
                try:
                    _first = ds_a.next_batch()
                    deadline = time.monotonic() + 10.0
                    while ds_a._worker_error is None and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertIsNotNone(ds_a._worker_error, f"{label}: worker did not record error in time")
                    with self.assertRaises(Exception, msg=f"{label} timing A must raise on state_dict()"):
                        ds_a.state_dict()
                    if ds_a._worker is not None:
                        self.assertFalse(ds_a._worker.is_alive())
                finally:
                    ds_a.close()

                # Timing B: state_dict() sets _stop_event BEFORE worker hits the corrupted game!
                ds_b = StreamingIQLDataset(
                    rl_file, batch_size=2, reservoir_size=8, warmup_transitions=2, max_prefetch=1
                )
                release_game2 = threading.Event()
                try:
                    entered_game2 = threading.Event()
                    orig_read = ds_b._read_next_owned_game
                    call_n = {"n": 0}

                    def _gated_read():
                        call_n["n"] += 1
                        if call_n["n"] >= 2:
                            entered_game2.set()
                            release_game2.wait(timeout=10.0)
                        return orig_read()

                    ds_b._read_next_owned_game = _gated_read  # type: ignore[method-assign]
                    _b0 = ds_b.next_batch()
                    self.assertTrue(entered_game2.wait(timeout=10.0), f"{label}: game2 gate not reached")

                    def _release_soon():
                        time.sleep(0.1)
                        release_game2.set()

                    rel_t = threading.Thread(target=_release_soon, daemon=True)
                    rel_t.start()
                    with self.assertRaises(Exception, msg=f"{label} timing B must raise even after _stop_event"):
                        ds_b.state_dict()
                    rel_t.join(timeout=5.0)
                finally:
                    release_game2.set()
                    ds_b.close()

                # Trainer integration: both checkpoint_every=0 and checkpoint_every=1 must raise cleanly
                for ckpt_every in (0, 1):
                    dev = torch.device("cpu")
                    mod = _tiny_iql_module(dev)
                    opt = build_joint_iql_optimizer(mod, base_lr=1e-4, weight_decay=0.01, device=dev)
                    rec = build_joint_iql_recipe(
                        mod, base_lr=1e-4, weight_decay=0.01, decoder1_lr_scale=0.25
                    )
                    ds_tr = StreamingIQLDataset(
                        rl_file, batch_size=2, reservoir_size=8, warmup_transitions=2, max_prefetch=2
                    )
                    try:
                        with self.assertRaises(
                            Exception,
                            msg=f"{label} checkpoint_every={ckpt_every} swallowed worker error",
                        ):
                            train_joint_iql_batches(
                                mod,
                                ds_tr,
                                opt,
                                recipe=rec,
                                device=dev,
                                ckpt_dir=root / f"ckpt_{label}_{ckpt_every}",
                                checkpoint_every=ckpt_every,
                                max_steps=10,
                            )
                    finally:
                        ds_tr.close()

    def test_03_bc_ddp_shared_spool_and_schedule_controlled_delay_collective_order(self):
        require_or_skip_distributed(self, context="2-rank controlled delay BC spool & schedule")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bc_file = _make_bc_file(root / "shared_bc.tsv.gz", num_games=2, steps_per_round=2)
            _spawn_2rank("--worker-03", str(bc_file), str(root))

    def test_04_bc_2rank_zero_redundancy_optimizer_periodic_final_and_resume(self):
        require_or_skip_distributed(self, context="2-rank ZeRO BC periodic, final & resume CLI")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bc_file = _make_bc_file(root / "zero_bc.tsv.gz", num_games=5, steps_per_round=2)
            shared_cache = root / "shared_spool_cache"

            def _run_2rank_bc_cli(
                tag: str, extra_args: list[str], *, startup_seed: int | None = None
            ) -> tuple[Path, Path]:
                port = _free_port()
                run_dirs = [root / f"{tag}_r0", root / f"{tag}_r1"]
                common_args = [
                    "--config-name", "v2_trajectory",
                    "device=cpu",
                    "encoder=v2_tiny",
                    "encoder.dimension=64",
                    "encoder.num_heads=4",
                    "encoder.dim_feedforward=128",
                    "encoder.n_d1_layers=1",
                    "encoder.n_d2_layers=1",
                    "num_workers=0",
                    "batch_size=2",
                    "gradient_accumulation_steps=2",
                    "optimizer=adamw",
                    "optimizer.use_zero=true",
                    "optimizer.learning_rate=5e-4",
                    "optimizer.warmup_steps=2",
                    "optimizer.warmup_start_lr=1e-5",
                    "snapshot_interval=2",
                    "prefetch_batches=0",
                    "suit_augmentation=true",
                    f"spool_cache_dir={shared_cache.as_posix()}",
                    f"training_data={bc_file.as_posix()}",
                    *extra_args,
                ]
                procs = []
                for rank in (0, 1):
                    cli_args = [*common_args, f"hydra.run.dir={run_dirs[rank].as_posix()}"]
                    if startup_seed is not None:
                        bootstrap = (
                            f"import random,numpy,runpy,sys,torch; "
                            f"random.seed({startup_seed}); numpy.random.seed({startup_seed}); "
                            f"torch.manual_seed({startup_seed}); "
                            f"sys.argv = ['kanachan.training.bc.train', *sys.argv[1:]]; "
                            f"runpy.run_module('kanachan.training.bc.train', run_name='__main__')"
                        )
                        cmd = [sys.executable, "-B", "-c", bootstrap, *cli_args]
                    else:
                        cmd = [sys.executable, "-B", "-m", "kanachan.training.bc.train", *cli_args]
                    procs.append(
                        subprocess.Popen(
                            cmd,
                            cwd=REPO_ROOT,
                            env=_ddp_env(rank, 2, port),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                        )
                    )
                _wait_2rank_procs(procs, label=f"zero_bc_{tag}")
                return run_dirs[0], run_dirs[1]

            cont_r0, _ = _run_2rank_bc_cli("cont5", ["max_steps=5"], startup_seed=71)
            part_r0, _ = _run_2rank_bc_cli("part3", ["max_steps=3"], startup_seed=71)

            periodic2 = part_r0 / "snapshots/traj-bc-2.pt"
            final3 = part_r0 / "snapshots/traj-bc-3.pt"
            self.assertTrue(periodic2.is_file(), "Missing periodic ZeRO checkpoint traj-bc-2.pt")
            self.assertTrue(final3.is_file(), "Missing non-periodic final ZeRO checkpoint traj-bc-3.pt")
            ckpt2 = torch.load(periodic2, map_location="cpu", weights_only=True)
            ckpt3 = torch.load(final3, map_location="cpu", weights_only=True)
            self.assertTrue(ckpt2["recipe"]["training_config"]["use_zero"])
            self.assertTrue(ckpt3["recipe"]["training_config"]["use_zero"])
            self.assertTrue(ckpt2["optimizer"]["state"], "ZeRO periodic optimizer state must be consolidated")
            self.assertTrue(ckpt3["optimizer"]["state"], "ZeRO final optimizer state must be consolidated")

            res_r0, _ = _run_2rank_bc_cli(
                "res5", ["max_steps=5", f"resume_from={final3.as_posix()}"], startup_seed=999
            )
            ckpt_cont = torch.load(cont_r0 / "snapshots/traj-bc-5.pt", map_location="cpu", weights_only=True)
            ckpt_res = torch.load(res_r0 / "snapshots/traj-bc-5.pt", map_location="cpu", weights_only=True)
            _assert_state_close(ckpt_cont["model"], ckpt_res["model"], atol=0.0, rtol=0.0)
            _assert_state_close(ckpt_cont["scheduler"], ckpt_res["scheduler"], atol=0.0, rtol=0.0)
            _assert_state_close(ckpt_cont["numerical_stats"], ckpt_res["numerical_stats"], atol=0.0, rtol=0.0)

    def test_05_bc_ordered_sample_identity_across_hash_buckets_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bc_file = _make_bc_file(root / "order_src.tsv.gz", num_games=2, steps_per_round=2)
            bc_reordered = _make_bc_file(
                root / "order_permuted.tsv.gz", num_games=2, steps_per_round=2, seed=99
            )

            with (
                SpoolingRoundBCDataSource(
                    bc_file, cache_dir=root / "cache_1b", num_hash_buckets=1, max_bucket_bytes=16 * 1024 * 1024
                ) as ds_1,
                SpoolingRoundBCDataSource(
                    bc_file, cache_dir=root / "cache_8b", num_hash_buckets=8, max_bucket_bytes=512
                ) as ds_8,
                SpoolingRoundBCDataSource(
                    bc_reordered, cache_dir=root / "cache_perm", num_hash_buckets=8, max_bucket_bytes=512
                ) as ds_perm,
            ):
                self.assertEqual(len(ds_1), 16)
                self.assertEqual(len(ds_1), len(ds_8))
                self.assertEqual(ds_1.ordered_sample_sha256, ds_8.ordered_sample_sha256)
                # Different num_hash_buckets changes rounds.bin offsets (hence index_sha256), while
                # ordered_sample_sha256 guarantees canonical logical index identity across cache dirs.
                self.assertNotEqual(ds_1.index_sha256, ds_8.index_sha256)
                self.assertNotEqual(ds_1.ordered_sample_sha256, ds_perm.ordered_sample_sha256)
                for idx in range(len(ds_1)):
                    _assert_state_close(ds_1[idx], ds_8[idx], atol=0.0, rtol=0.0, path=f"sample[{idx}]")

                dev = torch.device("cpu")
                for dsx in (ds_1, ds_8, ds_perm):
                    dsx.bind_preprocess_config(
                        suit_augmentation=False, aug_seed=1337, sampler_seed=42,
                        batch_size=2, bucket_mult=2, drop_last=False,
                    )
                torch.manual_seed(303)
                m_cont = _tiny_bc_model(dev)
                opt_cont = torch.optim.AdamW(adamw_parameter_groups(m_cont, weight_decay=0.01), lr=5e-4)
                rec_1 = m_cont.recipe(
                    data_sha256=ds_1.data_sha256,
                    source_data_sha256=ds_1.source_data_sha256,
                    preprocess_recipe_sha256=ds_1.preprocess_recipe_sha256,
                    ordered_sample_sha256=ds_1.ordered_sample_sha256,
                    lr=5e-4,
                    weight_decay=0.01,
                )
                with ds_1.create_sampler(
                    batch_size=2, seed=42, bucket_mult=2, schedule_dir=root / "sched_cont"
                ) as samp_cont:
                    train_trajectory_bc_steps(
                        m_cont, ds_1, opt_cont, batch_size=2, max_steps=4, device=dev,
                        bucket_mult=2, sampler=samp_cont, recipe=rec_1, data_sha256=ds_1.data_sha256, prefetch_batches=0,
                    )

                torch.manual_seed(303)
                m_half = _tiny_bc_model(dev)
                opt_half = torch.optim.AdamW(adamw_parameter_groups(m_half, weight_decay=0.01), lr=5e-4)
                ckpt_dir = root / "ckpts"
                with ds_1.create_sampler(
                    batch_size=2, seed=42, bucket_mult=2, schedule_dir=root / "sched_half"
                ) as samp_half:
                    train_trajectory_bc_steps(
                        m_half, ds_1, opt_half, batch_size=2, max_steps=2, device=dev,
                        bucket_mult=2, sampler=samp_half, recipe=rec_1, data_sha256=ds_1.data_sha256,
                        ckpt_dir=ckpt_dir, checkpoint_every=2, prefetch_batches=0,
                    )
                    half_state = samp_half.state_dict()

                m_res = _tiny_bc_model(dev)
                opt_res = torch.optim.AdamW(adamw_parameter_groups(m_res, weight_decay=0.01), lr=5e-4)
                rec_8 = m_res.recipe(
                    data_sha256=ds_8.data_sha256,
                    source_data_sha256=ds_8.source_data_sha256,
                    preprocess_recipe_sha256=ds_8.preprocess_recipe_sha256,
                    ordered_sample_sha256=ds_8.ordered_sample_sha256,
                    lr=5e-4,
                    weight_decay=0.01,
                )
                with ds_8.create_sampler(
                    batch_size=2, seed=42, bucket_mult=2, schedule_dir=root / "sched_res"
                ) as samp_res:
                    train_trajectory_bc_steps(
                        m_res, ds_8, opt_res, batch_size=2, max_steps=4, device=dev,
                        bucket_mult=2, sampler=samp_res, recipe=rec_8, data_sha256=ds_8.data_sha256,
                        resume_from=ckpt_dir / "traj-bc-2.pt", prefetch_batches=0,
                    )
                _assert_state_close(m_cont.state_dict(), m_res.state_dict(), atol=0.0, rtol=0.0)

                m_bad = _tiny_bc_model(dev)
                opt_bad = torch.optim.AdamW(adamw_parameter_groups(m_bad, weight_decay=0.01), lr=5e-4)
                rec_perm = m_bad.recipe(
                    data_sha256=ds_1.data_sha256,
                    source_data_sha256=ds_1.source_data_sha256,
                    preprocess_recipe_sha256=ds_1.preprocess_recipe_sha256,
                    ordered_sample_sha256=ds_perm.ordered_sample_sha256,
                    lr=5e-4,
                    weight_decay=0.01,
                )
                with ds_perm.create_sampler(
                    batch_size=2, seed=42, bucket_mult=2, schedule_dir=root / "sched_perm"
                ) as samp_perm:
                    with self.assertRaisesRegex(ValueError, "ordered_sample_sha256"):
                        load_trajectory_bc_checkpoint(
                            ckpt_dir / "traj-bc-2.pt",
                            m_bad,
                            opt_bad,
                            expected_recipe=rec_perm,
                            sampler=samp_perm,
                        )
                    with self.assertRaisesRegex(ValueError, "ordered_sample_sha256"):
                        samp_perm.load_state_dict(half_state)

    def test_06_bc_external_merge_fixed_fan_in_fd_cap_and_bounded_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 32 games * 8 rounds = 256 rounds -> at batch_size=1, merge_run_batches=2: 32 runs (64) & 128 runs (256)
            bc_file = _make_bc_file(root / "many_runs.tsv.gz", num_games=32, steps_per_round=2)

            for label, n_sub, mr_batches in (("gt24_runs", 64, 2), ("gt96_runs", 256, 2)):
                with SpoolingRoundBCDataSource(
                    bc_file, cache_dir=root / f"spool_{label}", max_rounds=n_sub
                ) as ds_sub:
                    self.assertEqual(len(ds_sub), n_sub)
                    sched_dir = root / f"sched_{label}"
                    sched_dir_b = root / f"sched_{label}_b"
                    sched_dir_res = root / f"sched_{label}_res"
                    watched_roots = (sched_dir.resolve(), sched_dir_b.resolve(), sched_dir_res.resolve())

                    active_paths: dict[int, Path] = {}
                    peak_open = {"n": 0}
                    orig_builtins_open = builtins.open

                    class _TrackedFile:
                        def __init__(self, inner: Any, fid: int) -> None:
                            self._inner = inner
                            self._fid = fid

                        def close(self) -> None:
                            active_paths.pop(self._fid, None)
                            self._inner.close()

                        def __enter__(self) -> Any:
                            self._inner.__enter__()
                            return self

                        def __exit__(self, *exc: Any) -> Any:
                            active_paths.pop(self._fid, None)
                            return self._inner.__exit__(*exc)

                        def __iter__(self) -> Any:
                            return iter(self._inner)

                        def __getattr__(self, name: str) -> Any:
                            return getattr(self._inner, name)

                    def _capped_open(file: Any, mode: str = "r", *a: Any, **kw: Any) -> Any:
                        fobj = orig_builtins_open(file, mode, *a, **kw)
                        try:
                            p = Path(file).resolve()
                            if any(wr in p.parents or p == wr for wr in watched_roots):
                                if len(active_paths) >= 7:
                                    fobj.close()
                                    raise OSError(errno.EMFILE, "Strict test FD cap exceeded (>7)", str(file))
                                fid = id(fobj)
                                active_paths[fid] = p
                                peak_open["n"] = max(peak_open["n"], len(active_paths))
                                return _TrackedFile(fobj, fid)
                        except TypeError:
                            pass
                        return fobj

                    tracemalloc.start()
                    orig_io_open = io.open
                    builtins.open = _capped_open  # type: ignore[assignment]
                    io.open = _capped_open  # type: ignore[assignment]
                    try:
                        sampler = ds_sub.create_sampler(
                            batch_size=1,
                            seed=123,
                            bucket_mult=2,
                            window_items=16,
                            max_merge_fan_in=4,
                            merge_run_batches=mr_batches,
                            schedule_dir=sched_dir,
                        )
                        batches_a = list(sampler)
                        sampler.close()
                        sampler2 = ds_sub.create_sampler(
                            batch_size=1,
                            seed=123,
                            bucket_mult=2,
                            window_items=16,
                            max_merge_fan_in=4,
                            merge_run_batches=mr_batches,
                            schedule_dir=sched_dir_b,
                        )
                        batches_b = list(sampler2)
                        sampler2.close()
                    finally:
                        builtins.open = orig_builtins_open  # type: ignore[assignment]
                        io.open = orig_io_open  # type: ignore[assignment]
                        _, peak_bytes = tracemalloc.get_traced_memory()
                        tracemalloc.stop()

                    self.assertEqual(len(active_paths), 0, f"{label}: unclosed handles: {active_paths}")
                    self.assertGreaterEqual(peak_open["n"], 3, f"{label}: open interception inactive")
                    self.assertLessEqual(peak_open["n"], 6, f"{label}: peak FDs={peak_open['n']} > 6")
                    self.assertLess(peak_bytes, 4 * 1024 * 1024, f"{label}: peak traced RAM={peak_bytes}")
                    self.assertEqual(batches_a, batches_b)
                    flat = [x for b in batches_a for x in b]
                    self.assertEqual(sorted(flat), list(range(n_sub)))

                    sampler_mid = ds_sub.create_sampler(
                        batch_size=1,
                        seed=123,
                        bucket_mult=2,
                        window_items=16,
                        max_merge_fan_in=4,
                        merge_run_batches=mr_batches,
                        schedule_dir=sched_dir,
                    )
                    sd = None
                    first_half = []
                    for step_i, batch in enumerate(sampler_mid):
                        first_half.append(batch)
                        if step_i == len(batches_a) // 2 - 1:
                            sd = sampler_mid.state_dict()
                            break
                    sampler_mid.close()
                    assert sd is not None
                    sampler_res = ds_sub.create_sampler(
                        batch_size=1,
                        seed=123,
                        bucket_mult=2,
                        window_items=16,
                        max_merge_fan_in=4,
                        merge_run_batches=mr_batches,
                        schedule_dir=sched_dir_res,
                    )
                    sampler_res.load_state_dict(sd)
                    second_half = list(sampler_res)
                    sampler_res.close()
                    self.assertEqual(first_half + second_half, batches_a)

    def test_07_python310_compatible_chunked_sha256_file_bytes(self):
        # Static AST verification across all of kanachan/ and tools/: no Python 3.11+ hashlib.file_digest
        for sub in ("kanachan", "tools"):
            base_dir = REPO_ROOT / sub
            for py_path in sorted(base_dir.rglob("*.py")):
                tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute) and node.attr == "file_digest":
                        self.fail(f"Python 3.11+ hashlib.file_digest found in {py_path}:{node.lineno}")
                    if isinstance(node, ast.ImportFrom) and node.module == "hashlib":
                        if any(alias.name == "file_digest" for alias in node.names):
                            self.fail(f"Python 3.11+ from hashlib import file_digest in {py_path}:{node.lineno}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                ("empty.bin", b""),
                ("one.bin", b"Z"),
                ("prime.bin", os.urandom(65537)),
                ("multi_chunk.bin", os.urandom(3 * 1024 * 1024 + 17)),
            ]
            for name, payload in cases:
                p = root / name
                p.write_bytes(payload)
                nbytes, digest = sha256_file_bytes(p)
                self.assertEqual(nbytes, len(payload), name)
                self.assertEqual(digest, hashlib.sha256(payload).hexdigest(), name)

            gz_p = root / "sample.tsv.gz"
            with gzip.open(gz_p, "wb") as fp:
                fp.write(b"abc\t123\n" * 5000)
            raw_gz = gz_p.read_bytes()
            n_gz, d_gz = sha256_file_bytes(gz_p, chunk_size=1234)
            self.assertEqual(n_gz, len(raw_gz))
            self.assertEqual(d_gz, hashlib.sha256(raw_gz).hexdigest())

            big_p = root / "big.bin"
            big_payload = os.urandom(2 * 1024 * 1024 + 5)
            big_p.write_bytes(big_payload)
            tracemalloc.start()
            try:
                nb, dg = sha256_file_bytes(big_p, chunk_size=4096)
                _, peak_bytes = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertEqual(nb, len(big_payload))
            self.assertEqual(dg, hashlib.sha256(big_payload).hexdigest())
            self.assertLess(peak_bytes, 256 * 1024)

            with self.assertRaises(ValueError):
                sha256_file_bytes(big_p, chunk_size=0)

    def test_08_warmup_grad_accum2_suit_aug_prefetch0_snap2_step3_final_resume_to_step5(self):
        env = _base_subproc_env()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 3 games * 8 rounds = 24 rounds >= 20 needed for 5 steps @ bsz=2, accum=2
            bc_file = _make_bc_file(root / "combo_bc.tsv.gz", num_games=3, steps_per_round=2)
            rl_file = _make_rl_file(root / "combo_rl.tsv.gz", num_games=16, n_transitions=3)

            common_cli = [
                "device=cpu",
                "encoder=v2_tiny",
                "encoder.dimension=64",
                "encoder.num_heads=4",
                "encoder.dim_feedforward=128",
                "encoder.n_d1_layers=1",
                "encoder.n_d2_layers=1",
                "num_workers=0",
                "batch_size=2",
                "gradient_accumulation_steps=2",
                "optimizer=adamw",
                "optimizer.warmup_steps=2",
                "optimizer.warmup_start_lr=1e-5",
                "optimizer.learning_rate=1e-3",
                "suit_augmentation=true",
                "prefetch_batches=0",
                "snapshot_interval=2",
            ]

            def _run_cli(
                module: str, cfg: str, run_dir: Path, extra: list[str], *, startup_seed: int
            ) -> None:
                cli_args = ["--config-name", cfg, *common_cli, *extra, f"hydra.run.dir={run_dir.as_posix()}"]
                bootstrap = (
                    f"import random,numpy,runpy,sys,torch; "
                    f"random.seed({startup_seed}); numpy.random.seed({startup_seed}); "
                    f"torch.manual_seed({startup_seed}); "
                    f"sys.argv = [{module!r}, *sys.argv[1:]]; "
                    f"runpy.run_module({module!r}, run_name='__main__')"
                )
                proc = subprocess.run(
                    [sys.executable, "-B", "-c", bootstrap, *cli_args],
                    cwd=REPO_ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=120,
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

            # 1A. CPU BC CLI: continuous step 5 vs step 3 (final non-periodic) -> resume step 5
            _run_cli(
                "kanachan.training.bc.train",
                "v2_trajectory",
                root / "bc_c5",
                [f"training_data={bc_file.as_posix()}", "max_steps=5"],
                startup_seed=71,
            )
            _run_cli(
                "kanachan.training.bc.train",
                "v2_trajectory",
                root / "bc_p3",
                [f"training_data={bc_file.as_posix()}", "max_steps=3"],
                startup_seed=71,
            )
            bc_ckpt3 = root / "bc_p3/snapshots/traj-bc-3.pt"
            self.assertTrue(bc_ckpt3.is_file())
            _run_cli(
                "kanachan.training.bc.train",
                "v2_trajectory",
                root / "bc_r5",
                [f"training_data={bc_file.as_posix()}", "max_steps=5", f"resume_from={bc_ckpt3.as_posix()}"],
                startup_seed=999,
            )
            bc5_c = torch.load(root / "bc_c5/snapshots/traj-bc-5.pt", map_location="cpu", weights_only=True)
            bc5_r = torch.load(root / "bc_r5/snapshots/traj-bc-5.pt", map_location="cpu", weights_only=True)
            _assert_state_close(bc5_c["model"], bc5_r["model"], atol=0.0, rtol=0.0)
            _assert_state_close(bc5_c["scheduler"], bc5_r["scheduler"], atol=0.0, rtol=0.0)
            self.assertIsNotNone(bc5_r["scheduler"])
            _assert_state_close(bc5_c["sampler"], bc5_r["sampler"], atol=0.0, rtol=0.0)

            # 1B. CPU IQL CLI: continuous step 5 vs step 3 (final non-periodic) -> resume step 5
            iql_extra = [f"training_data={rl_file.as_posix()}", "replay_buffer_size=32", "warmup_transitions=4"]
            _run_cli(
                "kanachan.training.iql.train",
                "v2_joint",
                root / "iql_c5",
                [*iql_extra, "max_steps=5"],
                startup_seed=71,
            )
            _run_cli(
                "kanachan.training.iql.train",
                "v2_joint",
                root / "iql_p3",
                [*iql_extra, "max_steps=3"],
                startup_seed=71,
            )
            iq_ckpt3 = root / "iql_p3/snapshots/joint-iql-3.pt"
            self.assertTrue(iq_ckpt3.is_file())
            _run_cli(
                "kanachan.training.iql.train",
                "v2_joint",
                root / "iql_r5",
                [*iql_extra, "max_steps=5", f"resume_from={iq_ckpt3.as_posix()}"],
                startup_seed=999,
            )
            iq5_c = torch.load(root / "iql_c5/snapshots/joint-iql-5.pt", map_location="cpu", weights_only=True)
            iq5_r = torch.load(root / "iql_r5/snapshots/joint-iql-5.pt", map_location="cpu", weights_only=True)
            _assert_state_close(iq5_c["model"], iq5_r["model"], atol=0.0, rtol=0.0)
            _assert_state_close(iq5_c["scheduler"], iq5_r["scheduler"], atol=0.0, rtol=0.0)
            self.assertIsNotNone(iq5_r["scheduler"])
            _assert_state_close(iq5_c["stream_state_by_rank"], iq5_r["stream_state_by_rank"], atol=0.0, rtol=0.0)

            # 2. CUDA FP16 + GradScaler (BC + IQL): warmup=2, grad_accum=2, suit_aug=True, prefetch=0, ckpt_every=2, step 3 -> 5 resume
            dev_cuda = require_or_skip_cuda(self, context="CUDA FP16 GradScaler warmup/accum2/suit_aug/snap2 step 3->5")

            with SpoolingRoundBCDataSource(bc_file, cache_dir=root / "cu_bc_cache") as ds_cu:
                ds_cu.bind_preprocess_config(
                    suit_augmentation=True,
                    aug_seed=1337,
                    sampler_seed=42,
                    batch_size=2,
                    bucket_mult=2,
                    drop_last=False,
                )

                def _run_cuda_bc(
                    tag: str, max_s: int, resume_p: Path | None = None
                ) -> tuple[TrajectoryBCModel, GradScaler, Any]:
                    torch.manual_seed(909)
                    torch.cuda.manual_seed_all(909)
                    m = _tiny_bc_model(dev_cuda)
                    rec_cu_bc = m.recipe(
                        data_sha256=ds_cu.data_sha256,
                        source_data_sha256=ds_cu.source_data_sha256,
                        preprocess_recipe_sha256=ds_cu.preprocess_recipe_sha256,
                        ordered_sample_sha256=ds_cu.ordered_sample_sha256,
                        lr=1e-3,
                        weight_decay=0.01,
                        training_config={"gradient_accumulation_steps": 2},
                    )
                    opt = torch.optim.AdamW(adamw_parameter_groups(m, weight_decay=0.01), lr=1e-3, fused=True)
                    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=2)
                    scaler = GradScaler("cuda", init_scale=256.0, growth_interval=2)
                    with ds_cu.create_sampler(
                        batch_size=2, seed=42, bucket_mult=2, schedule_dir=root / f"cu_s_{tag}"
                    ) as samp:
                        cdir = root / f"cu_ckpt_{tag}"
                        out = train_trajectory_bc_steps(
                            m, ds_cu, opt, batch_size=2, max_steps=max_s, device=dev_cuda, amp_dtype=torch.float16,
                            suit_augmentation=True, aug_seed=1337, bucket_mult=2, sampler=samp,
                            grad_accum_steps=2, prefetch_batches=0,
                            scheduler=sched, grad_scaler=scaler, recipe=rec_cu_bc, data_sha256=ds_cu.data_sha256,
                            ckpt_dir=cdir, checkpoint_every=2, resume_from=resume_p,
                        )
                        if out["step"] % 2 != 0:
                            bc_stats = {
                                k: int(out[k])
                                for k in (
                                    "optimizer_steps",
                                    "skipped_steps",
                                    "nonfinite_loss_steps",
                                    "nonfinite_grad_steps",
                                )
                            }
                            save_trajectory_bc_checkpoint(
                                cdir, m, opt, sampler=samp, step=out["step"], recipe=rec_cu_bc,
                                scheduler=sched, grad_scaler=scaler, numerical_stats=bc_stats,
                            )
                    return m, scaler, sched

                m_cu_c5, sc_cu_c5, sh_cu_c5 = _run_cuda_bc("c5", 5)
                _run_cuda_bc("p3", 3)
                cu_bc_p3 = root / "cu_ckpt_p3/traj-bc-3.pt"
                self.assertTrue(cu_bc_p3.is_file())
                m_cu_r5, sc_cu_r5, sh_cu_r5 = _run_cuda_bc("r5", 5, cu_bc_p3)
                _assert_state_close(m_cu_c5.state_dict(), m_cu_r5.state_dict(), atol=0.0, rtol=0.0)
                _assert_state_close(sh_cu_c5.state_dict(), sh_cu_r5.state_dict(), atol=0.0, rtol=0.0)
                _assert_state_close(sc_cu_c5.state_dict(), sc_cu_r5.state_dict(), atol=0.0, rtol=0.0)

            def _run_cuda_iql(
                tag: str,
                max_s: int,
                resume_p: Path | None = None,
                *,
                inject_overflow_step: int | None = None,
            ) -> tuple[JointStructuredIQLModule, GradScaler, Any, dict[str, Any]]:
                torch.manual_seed(919)
                torch.cuda.manual_seed_all(919)
                mod = _tiny_iql_module(dev_cuda)
                opt = build_joint_iql_optimizer(mod, base_lr=1e-3, weight_decay=0.01, device=dev_cuda)
                sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=2)
                scaler = GradScaler("cuda", init_scale=256.0, growth_interval=2)
                ds_iq = StreamingIQLDataset(
                    rl_file,
                    batch_size=2,
                    reservoir_size=32,
                    warmup_transitions=4,
                    seed=42,
                    suit_augmentation=True,
                    aug_seed=1337,
                    max_prefetch=0,
                )
                rec = build_joint_iql_recipe(
                    mod,
                    base_lr=1e-3,
                    weight_decay=0.01,
                    decoder1_lr_scale=0.25,
                    data_sha256=ds_iq.data_sha256,
                    source_data_sha256=ds_iq.source_data_sha256,
                    preprocess_recipe_sha256=ds_iq.preprocess_recipe_sha256,
                    training_config={"gradient_accumulation_steps": 2},
                )
                cdir = root / f"cu_iq_{tag}"

                def _nan_at_step(loss: Tensor, cur_step: int) -> Tensor:
                    if inject_overflow_step is not None and cur_step == inject_overflow_step:
                        return loss * float("nan")
                    return loss

                try:
                    out = train_joint_iql_batches(
                        mod,
                        ds_iq,
                        opt,
                        recipe=rec,
                        device=dev_cuda,
                        amp_dtype=torch.float16,
                        grad_accum_steps=2,
                        scheduler=sched,
                        grad_scaler=scaler,
                        ckpt_dir=cdir,
                        checkpoint_every=2,
                        resume_from=resume_p,
                        max_steps=max_s,
                        loss_transform=_nan_at_step if inject_overflow_step is not None else None,
                    )
                    if out["step"] % 2 != 0:
                        iq_stats = {
                            k: int(out[k])
                            for k in (
                                "optimizer_steps",
                                "skipped_steps",
                                "nonfinite_loss_steps",
                                "nonfinite_grad_steps",
                            )
                        }
                        save_joint_iql_checkpoint(
                            cdir,
                            mod,
                            opt,
                            step=out["step"],
                            cursor=out["cursor"],
                            recipe=rec,
                            scheduler=sched,
                            grad_scaler=scaler,
                            numerical_stats=iq_stats,
                            stream_state=ds_iq.state_dict(),
                        )
                    return mod, scaler, sched, out
                finally:
                    ds_iq.close()

            iq_cu_c5, iq_sc_c5, iq_sh_c5, _ = _run_cuda_iql("c5", 5)
            _run_cuda_iql("p3", 3)
            cu_iq_p3 = root / "cu_iq_p3/joint-iql-3.pt"
            self.assertTrue(cu_iq_p3.is_file())
            iq_cu_r5, iq_sc_r5, iq_sh_r5, _ = _run_cuda_iql("r5", 5, cu_iq_p3)
            _assert_state_close(iq_cu_c5.state_dict(), iq_cu_r5.state_dict(), atol=0.0, rtol=0.0)
            _assert_state_close(iq_sh_c5.state_dict(), iq_sh_r5.state_dict(), atol=0.0, rtol=0.0)
            _assert_state_close(iq_sc_c5.state_dict(), iq_sc_r5.state_dict(), atol=0.0, rtol=0.0)

            pre_ckpt3 = torch.load(cu_iq_p3, map_location="cpu", weights_only=True)
            pre_scale = float(pre_ckpt3["grad_scaler"]["scale"])
            pre_sched_count = int(pre_ckpt3["scheduler"]["_step_count"])
            iq_ov_mod, iq_ov_sc, iq_ov_sh, iq_ov_out = _run_cuda_iql(
                "ov4", 4, cu_iq_p3, inject_overflow_step=3
            )
            self.assertEqual(iq_ov_out["skipped_steps"], 1)
            self.assertEqual(iq_ov_out["optimizer_steps"], 3)
            self.assertEqual(float(iq_ov_sc.get_scale()), pre_scale * 0.5)
            self.assertEqual(iq_ov_sh.state_dict()["_step_count"], pre_sched_count)
            _assert_state_close(iq_ov_mod.state_dict(), pre_ckpt3["model"], atol=0.0, rtol=0.0)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker-01":
        _worker_01_ddp_production_accum(
            int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
        )
    elif len(sys.argv) >= 2 and sys.argv[1] == "--worker-03":
        _worker_03_bc_ddp_shared_spool_and_schedule(
            int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
        )
    else:
        unittest.main()
