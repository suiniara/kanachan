"""Comprehensive StructuredDualDecoder V2 engineering & training readiness tests (§1-§8)."""

from __future__ import annotations

import gzip
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import unittest
import warnings
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
from kanachan.nn.structured_dual_decoder import StructuredActionHead, StructuredDualDecoder
from kanachan.nn.structured_iql import JointStructuredIQLModule
from kanachan.structured_v2 import OFFSETS_136, PAD_ACTION_ID, derive_structured_v2_batch
from kanachan.training.bc.streaming_v2 import BoundedRoundPrefetcher, SpoolingRoundBCDataSource
from kanachan.training.bc.trajectory_v2 import (
    FINAL_RANK_T_REF,
    LengthBucketBatchSampler,
    RoundTrajectory,
    TrajectoryBCModel,
    _expected_inactive_bc_param_ids,
    build_round_trajectories_from_raw,
    collate_round_batch,
    parse_bc_rounds,
    save_trajectory_bc_checkpoint,
    train_trajectory_bc_steps,
)
from kanachan.training.core.optimizers import (
    OptimizerCoverageError,
    adamw_parameter_groups,
    validate_optimizer_parameter_coverage,
)
from kanachan.training.core.rl.rewards import configured_reward
from kanachan.training.core.test_env import (
    require_or_skip_cuda,
    require_or_skip_real_data,
)
from kanachan.training.grp import (
    FrozenGRP,
    FrozenGRPPBRSReward,
    GRPRoundDataset,
    PureStateGRP,
    compute_grp_checksum,
    evaluate_grp_calibration_report,
    extract_grp_samples_from_source,
    load_frozen_grp,
    main as grp_cli_main,
    save_frozen_grp,
    shape_hanchan_pbrs_rewards,
    train_pure_state_grp,
)
import kanachan.training.iql.joint_v2 as joint_v2_mod
from kanachan.training.iql.joint_v2 import (
    build_joint_iql_optimizer,
    build_joint_iql_recipe,
    expected_inactive_iql_param_ids,
    train_joint_iql_batches,
)
from kanachan.training.iql.streaming_v2 import StreamingIQLDataset
import tools.audit_structured_v2 as audit_tool


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_sparse_row(
    *,
    seat: int,
    chang: int,
    ju: int,
    left: int = 69,
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
) -> list[str]:
    """Build 4 physically consistent complete East-1..East-4 rounds ending cleanly at East-4."""
    assert sum(final_scores) == 100000
    s0 = (25000, 25000, 25000, 25000)

    def _interp_scores(frac: float) -> tuple[int, int, int, int]:
        pts = [25000 + int(round((final_scores[i] - 25000) * frac / 100.0)) * 100 for i in range(4)]
        pts[0] += 100000 - sum(pts)
        return (pts[0], pts[1], pts[2], pts[3])

    post_seqs = [
        _interp_scores(0.35),
        _interp_scores(0.65),
        _interp_scores(0.85),
        final_scores,
    ]
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
        disc_tok = 5 + loser * 148 + 4 * 32 + 2
        post_s = post_seqs[ju]
        delta = tuple(post_s[i] - cur_scores[i] for i in range(4))
        res_common = [*delta, 0, 0, *post_s, *final_scores]
        sp_loser = _make_sparse_row(seat=loser, chang=0, ju=ju, left=69, zimo=32)
        sp_win = _make_sparse_row(seat=winner, chang=0, ju=ju, left=69, zimo=37)
        nu = [0, 0, *cur_scores]
        # Dealer / loser initial action step (index 3 -> discard tile 32 matching winner ron response)
        lines.append(
            "\t".join(
                [
                    uuid,
                    ",".join(map(str, sp_loser)),
                    ",".join(map(str, nu)),
                    "0",
                    "4,44,84,130",
                    "3",
                    str(rs_code),
                    ",".join(map(str, res_common)),
                ]
            )
            + "\n"
        )
        # Winner ron response step
        lines.append(
            "\t".join(
                [
                    uuid,
                    ",".join(map(str, sp_win)),
                    ",".join(map(str, nu)),
                    f"0,{disc_tok}",
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


def _make_complete_rl_game_lines(
    *,
    seat: int = 0,
    scores_list: tuple[tuple[int, int, int, int], ...] = (
        (25000, 25000, 25000, 25000),
        (28000, 24000, 24000, 24000),
        (32000, 26000, 22000, 20000),
    ),
) -> list[str]:
    """Build valid 2-round contiguous observer game in (10-col, 12-col, 8-col) .rl.tsv format."""
    assert len(scores_list) == 3
    lines: list[str] = []
    # Step 0: round 0 step 0 -> transitions to round 0 step 1 (10 cols: not EOR, not EOG)
    sp0 = _make_sparse_row(seat=seat, chang=0, ju=0, left=60, zimo=32)
    nu0 = [0, 0, *scores_list[0]]
    sp1 = _make_sparse_row(seat=seat, chang=0, ju=0, left=56, zimo=32)
    nu1 = [0, 0, *scores_list[0]]
    lines.append(
        "\t".join(
            [
                "",
                ",".join(map(str, sp0)),
                ",".join(map(str, nu0)),
                "0",
                "4,44,84,130",
                "0",
                ",".join(map(str, sp1)),
                ",".join(map(str, nu1)),
                f"0,{5 + seat * 148}",
                "4,44,84,130",
            ]
        )
        + "\n"
    )
    # Step 1: round 0 end-of-round -> transitions to round 1 step 0 (12 cols: EOR=True, EOG=False)
    sp2 = _make_sparse_row(seat=seat, chang=0, ju=1, left=62, zimo=32)
    nu2 = [0, 0, *scores_list[1]]
    d0 = [scores_list[1][i] - scores_list[0][i] for i in range(4)]
    res_eor = [*d0, 0, 0, *scores_list[1]]  # 10 ints in 12-col RL format
    lines.append(
        "\t".join(
            [
                "",
                ",".join(map(str, sp1)),
                ",".join(map(str, nu1)),
                f"0,{5 + seat * 148}",
                "4,44,84,130",
                "1",
                ",".join(map(str, sp2)),
                ",".join(map(str, nu2)),
                "0",
                "4,44,84,130",
                "5",
                ",".join(map(str, res_eor)),
            ]
        )
        + "\n"
    )
    # Step 2: round 1 end-of-game (8 cols: EOR=True, EOG=True, dummy next in DataIterator)
    d1 = [scores_list[2][i] - scores_list[1][i] for i in range(4)]
    res_eog = [*d1, 0, 0, *scores_list[2], *scores_list[2]]  # 14 ints
    lines.append(
        "\t".join(
            [
                "",
                ",".join(map(str, sp2)),
                ",".join(map(str, nu2)),
                "0",
                "4,44,84,130",
                "2",
                "5",
                ",".join(map(str, res_eog)),
            ]
        )
        + "\n"
    )
    return lines


def _train_passing_test_grp(path: Path, *, num_games: int = 24) -> Path:
    """Create & save a calibrated, gate-passing FrozenGRP artifact for entrypoint tests."""
    # Construct crisp separable synthetic round dataset so classifier crosses gate cleanly in seconds.
    torch.manual_seed(77)
    n_samples = num_games * 4
    sp_list, nu_list, rk_list, uid_list = [], [], [], []
    rng = random.Random(77)
    score_templates = [
        (42000, 26000, 18000, 14000),
        (14000, 42000, 26000, 18000),
        (18000, 14000, 42000, 26000),
        (26000, 18000, 14000, 42000),
    ]
    for g in range(num_games):
        uid = f"grp-g-{g}"
        tmpl = score_templates[g % 4]
        ranked_seats = sorted(range(4), key=lambda s: -tmpl[s])
        ranks = [0] * 4
        for rk, s in enumerate(ranked_seats):
            ranks[s] = rk
        for s in range(4):
            jitter = [x + rng.randint(-200, 200) for x in tmpl]
            sp_list.append(_make_sparse_row(seat=s, chang=1, ju=3)[:9])
            nu_list.append([0, 0, *jitter])
            rk_list.append(ranks[s])
            uid_list.append(uid)

    from kanachan.training.grp import GRPRoundDataset

    ds = GRPRoundDataset(
        sparse=torch.tensor(sp_list, dtype=torch.int32),
        numeric=torch.tensor(nu_list, dtype=torch.float32),
        final_rank=torch.tensor(rk_list, dtype=torch.long),
        uuids=tuple(uid_list),
    )
    tr_ds, va_ds = ds.split_by_uuid(val_fraction=0.25, seed=77)
    frozen, report = train_pure_state_grp(
        tr_ds, va_ds, epochs=30, batch_size=64, lr=1e-2, calibrate_temperature=True, require_gate=True, device="cpu", seed=77
    )
    save_frozen_grp(path, frozen)
    return path


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
            "GLOO_SOCKET_IFNAME": env.get("GLOO_SOCKET_IFNAME", "singbox_tun"),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
        }
    )
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
    outs = [p.communicate(timeout=180) for p in procs]
    for rank, (p, (so, se)) in enumerate(zip(procs, outs, strict=True)):
        if p.returncode != 0:
            raise AssertionError(f"2-rank worker mode={mode} rank={rank} failed ({p.returncode}):\n{so}\n{se}")


def _worker_ddp_bc(rank: int, world_size: int, bc_path: str, out_dir: str, action: str) -> None:
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        out = Path(out_dir)
        dev = torch.device("cpu")
        ds = SpoolingRoundBCDataSource(
            Path(bc_path), cache_dir=out / "spool", num_hash_buckets=4, max_bucket_bytes=1024
        )
        torch.manual_seed(901)
        model = TrajectoryBCModel(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=1, n_d2_layers=1, use_next_event_head=True, next_event_weight=0.2,
            use_final_rank_head=True, final_rank_weight=0.1, device=dev, dtype=torch.float32,
        )
        ddp_m = DistributedDataParallel(model)
        opt = torch.optim.AdamW(adamw_parameter_groups(model, 0.01), lr=3e-4)

        sampler = ds.create_sampler(
            batch_size=2, seed=42, rank=rank, world_size=world_size
        )
        assigned = sampler.assigned_sample_indices()
        weights = [w for _, _, w in sampler.iter_pending_indexed_batches()]
        torch.save({"assigned": assigned, "weights": weights}, out / f"samp_{rank}.pt")
        sampler.close()

        if action == "cont":
            res = train_trajectory_bc_steps(
                ddp_m, ds, opt, batch_size=2, max_steps=5, device=dev,
                suit_augmentation=True, aug_seed=555, sampler_seed=42,
                ckpt_dir=out / "ckpt", checkpoint_every=2, prefetch_batches=1,
            )
            torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "res": res}, out / f"cont_{rank}.pt")
        elif action == "resume":
            res = train_trajectory_bc_steps(
                ddp_m, ds, opt, batch_size=2, max_steps=5, device=dev,
                suit_augmentation=True, aug_seed=555, sampler_seed=42,
                ckpt_dir=out / "ckpt", checkpoint_every=2, resume_from=out / "ckpt/traj-bc-2.pt",
                prefetch_batches=1,
            )
            torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "res": res}, out / f"res_{rank}.pt")
        ds.close()
    finally:
        dist.destroy_process_group()


def _worker_ddp_iql(rank: int, world_size: int, rl_path: str, out_dir: str, action: str) -> None:
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        out = Path(out_dir)
        dev = torch.device("cpu")
        ds = StreamingIQLDataset(
            Path(rl_path), batch_size=2, reservoir_size=64, warmup_transitions=4,
            reward_scheme="round_and_placement", seed=77, rank=rank, world_size=world_size,
            suit_augmentation=True, aug_seed=888, max_prefetch=2,
        )
        torch.manual_seed(902)
        mod = JointStructuredIQLModule(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=1, n_d2_layers=1, use_cql_quantile_head=True, num_quantiles=8,
            final_rank=True, final_rank_weight=0.1, device=dev, dtype=torch.float32,
        )
        ddp_m = DistributedDataParallel(mod)
        opt = build_joint_iql_optimizer(mod, base_lr=3e-4, weight_decay=0.01, device=dev, fused=False)
        rec = build_joint_iql_recipe(
            mod, base_lr=3e-4, weight_decay=0.01, decoder1_lr_scale=0.25, data_sha256=ds.data_sha256
        )

        if action == "cont":
            res = train_joint_iql_batches(
                mod, ds, opt, recipe=rec, device=dev, max_steps=5,
                ckpt_dir=out / "ckpt", checkpoint_every=2, ddp_wrapped=ddp_m,
            )
            torch.save({"state": {k: v.cpu() for k, v in mod.state_dict().items()}, "res": res}, out / f"cont_{rank}.pt")
        elif action == "resume":
            res = train_joint_iql_batches(
                mod, ds, opt, recipe=rec, device=dev, max_steps=5,
                ckpt_dir=out / "ckpt", checkpoint_every=2, resume_from=out / "ckpt/joint-iql-2.pt",
                ddp_wrapped=ddp_m,
            )
            torch.save({"state": {k: v.cpu() for k, v in mod.state_dict().items()}, "res": res}, out / f"res_{rank}.pt")
        elif action == "finite_epoch":
            res = train_joint_iql_batches(
                mod, ds, opt, recipe=rec, device=dev, max_steps=None, ddp_wrapped=ddp_m,
            )
            torch.save({"step": res["step"], "optimizer_steps": res["optimizer_steps"]}, out / f"ep_{rank}.pt")
        ds.close()
    finally:
        dist.destroy_process_group()


def _worker_ddp_fatal_bc(rank: int, world_size: int, bc_path: str, out_dir: str) -> None:
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        out = Path(out_dir)
        dev = torch.device("cpu")
        ds = SpoolingRoundBCDataSource(Path(bc_path), cache_dir=out / "spool", num_hash_buckets=2)
        rounds = [ds[i] for i in range(min(4, len(ds)))]
        torch.manual_seed(903)
        model = TrajectoryBCModel(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=1, n_d2_layers=1, use_next_event_head=False, use_final_rank_head=False,
            device=dev, dtype=torch.float32,
        )
        ddp_m = DistributedDataParallel(model)
        opt = torch.optim.AdamW(adamw_parameter_groups(model, 0.01), lr=3e-4)
        snap0 = {k: v.detach().clone() for k, v in model.state_dict().items()}

        def _xform(loss: torch.Tensor, step: int) -> torch.Tensor:
            if rank == 1 and step == 0:
                return loss.detach()
            return loss

        err_msg = ""
        try:
            train_trajectory_bc_steps(ddp_m, rounds, opt, batch_size=2, max_steps=2, device=dev, loss_transform=_xform)
        except RuntimeError as exc:
            err_msg = str(exc)
        assert err_msg, "Expected fatal RuntimeError to propagate cleanly on all ranks!"
        for k, v in snap0.items():
            torch.testing.assert_close(model.state_dict()[k], v, atol=0.0, rtol=0.0)
        ds.close()
    finally:
        dist.destroy_process_group()


def _worker_ddp_fr_grad(rank: int, world_size: int, out_dir: str, rounds_pt: str) -> None:
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        rounds = torch.load(rounds_pt, weights_only=False)
        dev = torch.device("cpu")
        torch.manual_seed(904)
        model = TrajectoryBCModel(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=1, n_d2_layers=1, use_next_event_head=False,
            use_final_rank_head=True, final_rank_weight=1.0, device=dev, dtype=torch.float32,
        )
        ddp_m = DistributedDataParallel(model)
        # Partition uneven rounds (rank 0 gets round[0], rank 1 gets rounds[1:3]) using exact global denoms!
        local_rds = [rounds[0]] if rank == 0 else list(rounds[1:])
        bcpu = collate_round_batch(local_rds)
        from kanachan.training.bc.trajectory_v2 import _compute_microgroup_denoms

        denoms = _compute_microgroup_denoms([bcpu], is_dist=True, world_size=world_size, device=dev)
        out = ddp_m(bcpu, denom_overrides=denoms, ddp_world_size=world_size)
        out["loss"].backward()
        if rank == 0:
            grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
            torch.save(grads, Path(out_dir) / "ddp_grads.pt")
    finally:
        dist.destroy_process_group()


class TestStructuredV2Readiness(unittest.TestCase):
    # ------------------------------------------------------------------
    # §1: Production Entrypoints & Fail-Closed Routing
    # ------------------------------------------------------------------
    def test_01_production_entrypoints_bc_and_iql_cli_plus_fail_closed(self):
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
            bc_file = root / "train_v3.bc.tsv.gz"
            rl_file = root / "train_v3.rl.tsv.gz"
            grp_file = root / "frozen_grp.pt"
            _train_passing_test_grp(grp_file)

            bc_lines: list[str] = []
            for g in range(6):
                fs = (34000, 28000, 21000, 17000) if g % 2 == 0 else (18000, 24000, 27000, 31000)
                bc_lines.extend(_make_complete_tonpuusen_bc_lines(f"g-{g}", final_scores=fs))
            # Shuffle across games/rows to match real source-v3 non-contiguity!
            random.Random(101).shuffle(bc_lines)
            with gzip.open(bc_file, "wt", encoding="utf-8") as fp:
                fp.writelines(bc_lines)

            rl_lines: list[str] = []
            for g in range(4):
                rl_lines.extend(_make_complete_rl_game_lines(seat=g % 4))
            with gzip.open(rl_file, "wt", encoding="utf-8") as fp:
                fp.writelines(rl_lines)

            enc_args = [
                "device=cpu",
                "encoder=v2_tiny",
                "encoder.dimension=64",
                "encoder.num_heads=4",
                "encoder.dim_feedforward=128",
                "encoder.n_d1_layers=1",
                "encoder.n_d2_layers=1",
                "num_workers=0",
            ]

            # 1A. BC v2_trajectory continuous + resume CLI
            for tag, extra in (
                ("bc_run", []),
                ("bc_res", [f"resume_from={(root / 'bc_run/snapshots/traj-bc-2.pt').as_posix()}"]),
            ):
                r = subprocess.run(
                    [
                        sys.executable, "-B", "-m", "kanachan.training.bc.train",
                        "--config-name", "v2_trajectory", *enc_args,
                        "batch_size=2", "max_steps=4", "snapshot_interval=2",
                        "spool_num_buckets=4", "use_final_rank_head=true", "final_rank_weight=0.1",
                        "suit_augmentation=true",
                        f"training_data={bc_file.as_posix()}",
                        f"hydra.run.dir={(root / tag).as_posix()}",
                        *extra,
                    ],
                    cwd=REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
                )
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

            c_bc = torch.load(root / "bc_run/snapshots/traj-bc-4.pt", map_location="cpu", weights_only=True)
            r_bc = torch.load(root / "bc_res/snapshots/traj-bc-4.pt", map_location="cpu", weights_only=True)
            self.assertEqual(c_bc["recipe"]["architecture"], "mahjong_structured_dual_decoder_v2")
            for k in c_bc["model"]:
                torch.testing.assert_close(c_bc["model"][k], r_bc["model"][k], atol=0.0, rtol=0.0)
            self.assertTrue((root / "bc_run/snapshots/model.kanachan").is_file())
            self.assertTrue(list((root / "bc_run/tensorboard").glob("events.out.*")))

            # 1B. Fail-closed: structured_dual_v2 MUST NOT silently run on legacy 5-net IQL!
            bad_iql = subprocess.run(
                [
                    sys.executable, "-B", "-m", "kanachan.training.iql.train",
                    "--config-name", "decision", *enc_args,
                    "batch_size=2", "replay_buffer_size=4", "reward_plugin=dummy.py",
                    f"training_data={rl_file.as_posix()}",
                    f"hydra.run.dir={(root / 'iql_bad').as_posix()}",
                ],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            )
            self.assertNotEqual(bad_iql.returncode, 0)
            self.assertIn("v2_joint", bad_iql.stdout + bad_iql.stderr)

            # 1C. Joint V2 IQL CLI with BC init, FrozenGRP PBRS, QR-DQN, KL anchor & exact resume!
            for tag, extra in (
                ("iql_run", []),
                ("iql_res", [f"resume_from={(root / 'iql_run/snapshots/joint-iql-2.pt').as_posix()}"]),
            ):
                r = subprocess.run(
                    [
                        sys.executable, "-B", "-m", "kanachan.training.iql.train",
                        "--config-name", "v2_joint", *enc_args,
                        "batch_size=2", "replay_buffer_size=8", "warmup_transitions=4",
                        "max_steps=4", "snapshot_interval=2",
                        "reward_scheme=grp_pbrs",
                        f"grp_checkpoint={grp_file.as_posix()}",
                        f"bc_initial_model={(root / 'bc_run/snapshots/model.kanachan').as_posix()}",
                        "kl_coef=0.05", "final_rank_weight=0.1", "use_cql_quantile_head=true",
                        "num_quantiles=8", "quantile_cql_alpha=0.1",
                        f"training_data={rl_file.as_posix()}",
                        f"hydra.run.dir={(root / tag).as_posix()}",
                        *extra,
                    ],
                    cwd=REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
                )
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

            c_iq = torch.load(root / "iql_run/snapshots/joint-iql-4.pt", map_location="cpu", weights_only=True)
            r_iq = torch.load(root / "iql_res/snapshots/joint-iql-4.pt", map_location="cpu", weights_only=True)
            self.assertEqual(c_iq["recipe"]["architecture"], "mahjong_structured_dual_decoder_v2")
            self.assertIsNotNone(c_iq["recipe"]["grp_sha256"])
            for k in c_iq["model"]:
                torch.testing.assert_close(c_iq["model"][k], r_iq["model"][k], atol=0.0, rtol=0.0)
            self.assertTrue((root / "iql_run/snapshots/model.kanachan").is_file())
            self.assertTrue(list((root / "iql_run/tensorboard").glob("events.out.*")))

    # ------------------------------------------------------------------
    # §2: Bounded Disk-Spooling BC & Streaming Offline IQL
    # ------------------------------------------------------------------
    def test_02_spooling_bc_exactness_incomplete_rejection_and_streaming_iql_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bc_p = root / "shuffled.bc.tsv.gz"
            bc_lines: list[str] = []
            for g in range(5):
                bc_lines.extend(_make_complete_tonpuusen_bc_lines(f"uid-{g}", final_scores=(35000, 26000, 21000, 18000)))
            random.Random(202).shuffle(bc_lines)
            with gzip.open(bc_p, "wt", encoding="utf-8") as fp:
                fp.writelines(bc_lines)

            direct_rounds = parse_bc_rounds(bc_p)
            # Tiny max_bucket_bytes=1024 forces recursive multi-depth hash-bucket sub-bucketing!
            with SpoolingRoundBCDataSource(
                bc_p, cache_dir=root / "spool", num_hash_buckets=4, max_bucket_bytes=1024
            ) as src:
                self.assertFalse(hasattr(src, "round_event_lens"))
                self.assertEqual(src.num_games, 5)
                self.assertEqual(len(src), len(direct_rounds))
                by_key_direct = {(r.uuid, r.observer_seat, r.round_event_len, r.action_idx[0].item()): r for r in direct_rounds}
                spooled_all = src.get_rounds(list(range(len(src))))
                for r_sp in spooled_all:
                    k = (r_sp.uuid, r_sp.observer_seat, r_sp.round_event_len, r_sp.action_idx[0].item())
                    r_dir = by_key_direct[k]
                    torch.testing.assert_close(r_sp.round_events, r_dir.round_events, atol=0, rtol=0)
                    torch.testing.assert_close(r_sp.v2_tile_feats, r_dir.v2_tile_feats, atol=0.0, rtol=0.0)
                    torch.testing.assert_close(r_sp.final_rank, r_dir.final_rank, atol=0, rtol=0)
                    self.assertEqual(r_sp.observer_total_steps, r_dir.observer_total_steps)

                # Test DiskLengthBucketBatchSampler + BoundedRoundPrefetcher determinism
                with src.create_sampler(batch_size=3, seed=7) as samp:
                    with BoundedRoundPrefetcher(
                        src, samp.iter_pending_indexed_batches(), max_prefetch=2, suit_perm_seed=91, epoch=0
                    ) as pf1, BoundedRoundPrefetcher(
                        src, samp.iter_pending_indexed_batches(), max_prefetch=2, suit_perm_seed=91, epoch=0
                    ) as pf2:
                        b1 = [b for _, b in pf1]
                        b2 = [b for _, b in pf2]
                        self.assertEqual(len(b1), len(b2))
                        for x, y in zip(b1, b2, strict=True):
                            torch.testing.assert_close(x["v2_tile_feats"], y["v2_tile_feats"], atol=0.0, rtol=0.0)

            # Truncated / incomplete UUID MUST be rejected cleanly!
            bad_bc = root / "incomplete.bc.tsv"
            bad_bc.write_text("".join(bc_lines[:-2]), encoding="utf-8")
            with self.assertRaises(ValueError):
                SpoolingRoundBCDataSource(bad_bc, num_hash_buckets=2, max_bucket_bytes=1024)

            # StreamingIQLDataset: incomplete EOF rejection & bit-exact max_prefetch=2 rollback resume
            rl_p = root / "stream.rl.tsv.gz"
            rl_lines: list[str] = []
            for g in range(6):
                rl_lines.extend(_make_complete_rl_game_lines(seat=g % 4))
            with gzip.open(rl_p, "wt", encoding="utf-8") as fp:
                fp.writelines(rl_lines)

            bad_rl = root / "trunc.rl.tsv"
            bad_rl.write_text("".join(rl_lines[:-1]), encoding="utf-8")
            with StreamingIQLDataset(bad_rl, batch_size=2, reservoir_size=4, warmup_transitions=2, max_prefetch=0) as bds:
                with self.assertRaises(RuntimeError):
                    list(bds)

            with StreamingIQLDataset(
                rl_p, batch_size=3, reservoir_size=8, warmup_transitions=3,
                reward_scheme="round_and_placement", seed=33, suit_augmentation=True, aug_seed=99, max_prefetch=2
            ) as ds_cont:
                batches_cont = [ds_cont.next_batch() for _ in range(5)]

            with StreamingIQLDataset(
                rl_p, batch_size=3, reservoir_size=8, warmup_transitions=3,
                reward_scheme="round_and_placement", seed=33, suit_augmentation=True, aug_seed=99, max_prefetch=2
            ) as ds_p1:
                b0 = ds_p1.next_batch()
                b1 = ds_p1.next_batch()
                torch.testing.assert_close(b0["v2_tile_feats"], batches_cont[0]["v2_tile_feats"], atol=0, rtol=0)
                torch.testing.assert_close(b1["v2_tile_feats"], batches_cont[1]["v2_tile_feats"], atol=0, rtol=0)
                saved = ds_p1.state_dict()

            with StreamingIQLDataset(
                rl_p, batch_size=3, reservoir_size=8, warmup_transitions=3,
                reward_scheme="round_and_placement", seed=33, suit_augmentation=True, aug_seed=99, max_prefetch=2
            ) as ds_p2:
                ds_p2.load_state_dict(saved)
                for step_i in range(2, 5):
                    nxt = ds_p2.next_batch()
                    self.assertIsNotNone(nxt)
                    torch.testing.assert_close(nxt["v2_tile_feats"], batches_cont[step_i]["v2_tile_feats"], atol=0, rtol=0)
                    torch.testing.assert_close(nxt["action"], batches_cont[step_i]["action"], atol=0, rtol=0)
                    torch.testing.assert_close(nxt["next", "reward"], batches_cont[step_i]["next", "reward"], atol=0, rtol=0)

    # ------------------------------------------------------------------
    # §3: Deterministic Distributed Sharding & 2-Rank DDP Continuity
    # ------------------------------------------------------------------
    def test_03_distributed_sampler_partition_and_2rank_ddp_bc_iql_resume(self):
        # Unit check 3-rank LengthBucketBatchSampler with uneven partial tail (13 items, bsz=3 -> 5 batches % 3 != 0)
        lens = [10 + i for i in range(13)]
        samplers = [
            LengthBucketBatchSampler(lens, batch_size=3, seed=19, rank=r, world_size=3, drop_last=False)
            for r in range(3)
        ]
        self.assertEqual(len({len(s) for s in samplers}), 1)
        assigned = [set(s.assigned_sample_indices()) for s in samplers]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertEqual(assigned[i] & assigned[j], set())
        self.assertEqual(set().union(*assigned), set(range(13)))
        # Exactly 1 ghost batch (weight 0.0) appended on rank 2 so 2+2+1+ghost = 2+2+2
        self.assertEqual(sum(1 for s in samplers for w in s.batch_weights if w == 0.0), 1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bc_p = root / "ddp.bc.tsv.gz"
            rl_p = root / "ddp.rl.tsv.gz"
            bc_lines: list[str] = []
            for g in range(4):
                bc_lines.extend(_make_complete_tonpuusen_bc_lines(f"ddp-{g}"))
            with gzip.open(bc_p, "wt", encoding="utf-8") as fp:
                fp.writelines(bc_lines)

            rl_lines: list[str] = []
            # 5 games (odd count across 2 ranks -> exercises ghost equalization on finite epoch!)
            for g in range(5):
                rl_lines.extend(_make_complete_rl_game_lines(seat=g % 4))
            with gzip.open(rl_p, "wt", encoding="utf-8") as fp:
                fp.writelines(rl_lines)

            bc_out = root / "bc_ddp"
            bc_out.mkdir()
            _spawn_2rank("--ddp-bc-worker", str(bc_p), str(bc_out), "cont")
            _spawn_2rank("--ddp-bc-worker", str(bc_p), str(bc_out), "resume")
            s0 = torch.load(bc_out / "samp_0.pt", weights_only=True)
            s1 = torch.load(bc_out / "samp_1.pt", weights_only=True)
            self.assertEqual(set(s0["assigned"]) & set(s1["assigned"]), set())
            self.assertEqual(set(s0["assigned"]) | set(s1["assigned"]), set(range(32)))
            c0 = torch.load(bc_out / "cont_0.pt", weights_only=True)
            c1 = torch.load(bc_out / "cont_1.pt", weights_only=True)
            r0 = torch.load(bc_out / "res_0.pt", weights_only=True)
            for k in c0["state"]:
                torch.testing.assert_close(c0["state"][k], c1["state"][k], atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(c0["state"][k], r0["state"][k], atol=0.0, rtol=0.0)

            iql_out = root / "iql_ddp"
            iql_out.mkdir()
            _spawn_2rank("--ddp-iql-worker", str(rl_p), str(iql_out), "cont")
            _spawn_2rank("--ddp-iql-worker", str(rl_p), str(iql_out), "resume")
            _spawn_2rank("--ddp-iql-worker", str(rl_p), str(iql_out), "finite_epoch")
            iq_c0 = torch.load(iql_out / "cont_0.pt", weights_only=True)
            iq_c1 = torch.load(iql_out / "cont_1.pt", weights_only=True)
            iq_r0 = torch.load(iql_out / "res_0.pt", weights_only=True)
            for k in iq_c0["state"]:
                torch.testing.assert_close(iq_c0["state"][k], iq_c1["state"][k], atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(iq_c0["state"][k], iq_r0["state"][k], atol=0.0, rtol=0.0)
            ep0 = torch.load(iql_out / "ep_0.pt", weights_only=True)
            ep1 = torch.load(iql_out / "ep_1.pt", weights_only=True)
            self.assertEqual(ep0["step"], ep1["step"])

    # ------------------------------------------------------------------
    # §4: Shared Optimizer Parameter Coverage Validator
    # ------------------------------------------------------------------
    def test_04_optimizer_parameter_coverage_validator_and_static_ablations(self):
        dev = torch.device("cpu")
        for preset_kw in (
            {"preset": "v2_tiny", "n_d1_layers": 1, "head_type": "factorized"},
            {"preset": "v2_tiny", "n_d1_layers": 1, "head_type": "flat546"},
            {"preset": "spatial_only", "n_d1_layers": 0, "head_type": "factorized"},
        ):
            mod = JointStructuredIQLModule(
                dimension=64, num_heads=4, dim_feedforward=128, n_d2_layers=1,
                device=dev, dtype=torch.float32, **preset_kw,
            )
            opt = build_joint_iql_optimizer(mod, base_lr=3e-4, weight_decay=0.01, device=dev)
            cov = validate_optimizer_parameter_coverage(
                mod, opt, inactive_param_ids=expected_inactive_iql_param_ids(mod), context="valid_iql"
            )
            self.assertEqual(cov["trainable_count"], cov["optimizer_count"])

            bm = TrajectoryBCModel(
                dimension=64, num_heads=4, dim_feedforward=128, n_d2_layers=1,
                use_next_event_head=False, use_final_rank_head=False, device=dev, dtype=torch.float32,
                **preset_kw,
            )
            bopt = torch.optim.AdamW(adamw_parameter_groups(bm, 0.01), lr=3e-4)
            bcov = validate_optimizer_parameter_coverage(
                bm, bopt, inactive_param_ids=_expected_inactive_bc_param_ids(bm), context="valid_bc"
            )
            self.assertEqual(bcov["trainable_count"], bcov["optimizer_count"])

        # Now inject ALL 5 error modes!
        mod_b = JointStructuredIQLModule(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=1, n_d2_layers=1, device=dev, dtype=torch.float32,
        )
        # 1. Missing trainable
        with self.assertRaises(OptimizerCoverageError) as cm1:
            validate_optimizer_parameter_coverage(
                mod_b, torch.optim.AdamW(list(mod_b.backbone.parameters()), lr=1e-4)
            )
        self.assertIn("missing_trainable", str(cm1.exception))
        # Ensure it was NOT silently frozen!
        self.assertTrue(all(p.requires_grad for p in mod_b.policy_head.parameters()))

        # 2. Duplicate parameter
        valid_groups = mod_b.parameter_groups(base_lr=1e-4, weight_decay=0.01)
        dup_param = valid_groups[0]["params"][0]
        opt_dup = torch.optim.AdamW(mod_b.parameter_groups(base_lr=1e-4, weight_decay=0.01))
        opt_dup.param_groups.append({"params": [dup_param], "lr": 1e-4})
        with self.assertRaises(OptimizerCoverageError) as cm2:
            validate_optimizer_parameter_coverage(mod_b, opt_dup)
        self.assertIn("duplicate_params", str(cm2.exception))

        # 3. Frozen parameter in optimizer
        opt_frozen = build_joint_iql_optimizer(mod_b, base_lr=1e-4, weight_decay=0.01, device=dev)
        opt_frozen.param_groups[0]["params"][0].requires_grad_(False)
        with self.assertRaises(OptimizerCoverageError) as cm3:
            validate_optimizer_parameter_coverage(mod_b, opt_frozen)
        self.assertIn("frozen_in_optimizer", str(cm3.exception))
        opt_frozen.param_groups[0]["params"][0].requires_grad_(True)

        # 4. Foreign parameter in optimizer
        foreign_p = nn.Parameter(torch.zeros(4))
        with self.assertRaises(OptimizerCoverageError) as cm4:
            validate_optimizer_parameter_coverage(
                mod_b, torch.optim.AdamW(valid_groups + [{"params": [foreign_p], "lr": 1e-4}])
            )
        self.assertIn("foreign_params", str(cm4.exception))

        # 5. Unintended trainable static inactive ablation parameter
        mod_flat = JointStructuredIQLModule(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=1, n_d2_layers=1, head_type="flat546", device=dev, dtype=torch.float32,
        )
        mod_flat.backbone.decoder2.null_tile_emb.requires_grad_(True)
        with self.assertRaises(OptimizerCoverageError) as cm5:
            validate_optimizer_parameter_coverage(
                mod_flat,
                torch.optim.AdamW(mod_flat.parameter_groups(base_lr=1e-4, weight_decay=0.01)),
                inactive_param_ids=expected_inactive_iql_param_ids(mod_flat),
            )
        self.assertIn("unintended_trainable", str(cm5.exception))

    # ------------------------------------------------------------------
    # §5: Redesigned DDP Fatal-Error Sync & Sync-Free Fused Path
    # ------------------------------------------------------------------
    def test_05_ddp_fatal_sync_redesign_no_per_batch_sync_helper(self):
        import kanachan.training.bc.trajectory_v2 as tv2_mod

        self.assertFalse(hasattr(joint_v2_mod, "_sync_ddp_fatal_code"))
        self.assertFalse(hasattr(tv2_mod, "_sync_ddp_fatal_code"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bc_p = root / "fatal.bc.tsv"
            bc_lines = []
            for g in range(2):
                bc_lines.extend(_make_complete_tonpuusen_bc_lines(f"fat-{g}"))
            bc_p.write_text("".join(bc_lines), encoding="utf-8")
            out = root / "out"
            out.mkdir()
            _spawn_2rank("--ddp-fatal-bc-worker", str(bc_p), str(out))

    def test_05b_bc_cuda_fused_zero_sync(self):
        dev = require_or_skip_cuda(self, context="BC CUDA fused zero-sync test")
        sp_raw = _make_sparse_row(seat=0, chang=0, ju=0)
        sp = torch.tensor(
            sp_raw + [NUM_TYPES_OF_SPARSE_FEATURES] * (MAX_NUM_ACTIVE_SPARSE_FEATURES - len(sp_raw)),
            dtype=torch.int32,
        ).unsqueeze(0)
        nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]], dtype=torch.int32)
        pr = torch.tensor(
            [[0] + [NUM_TYPES_OF_PROGRESSION_FEATURES] * (MAX_LENGTH_OF_PROGRESSION_FEATURES - 1)],
            dtype=torch.int32,
        )
        ca = torch.tensor(
            [[4, 44, 84, 130] + [PAD_ACTION_ID] * (MAX_NUM_ACTION_CANDIDATES - 4)],
            dtype=torch.int32,
        )
        rounds = build_round_trajectories_from_raw(
            sp.repeat(4, 1), nu.repeat(4, 1), pr.repeat(4, 1), ca.repeat(4, 1),
            torch.zeros(4, dtype=torch.long), ["g0", "g1", "g2", "g3"],
        )
        bm = TrajectoryBCModel(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=1, n_d2_layers=1, use_next_event_head=False, use_final_rank_head=False,
            device=dev, dtype=torch.float32,
        )
        bopt = torch.optim.AdamW(adamw_parameter_groups(bm, 0.01), lr=1e-4, fused=True)
        train_trajectory_bc_steps(
            bm, rounds, bopt, batch_size=2, max_steps=1, device=dev, amp_dtype=torch.bfloat16
        )
        prev_det = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode(1)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                train_trajectory_bc_steps(
                    bm, rounds, bopt, batch_size=2, max_steps=4, device=dev,
                    amp_dtype=torch.bfloat16, log_interval=0, check_finite_interval=0,
                )
                torch.cuda.synchronize()
        finally:
            torch.cuda.set_sync_debug_mode(0)
            torch.use_deterministic_algorithms(prev_det)
        sync_warns = [w for w in caught if "synchronizing CUDA operation" in str(w.message)]
        self.assertLessEqual(len(sync_warns), 1, [str(w.message) for w in sync_warns])

    # ------------------------------------------------------------------
    # §6: Enforced FrozenGRP Gate + Full Workflow + Recipe Binding
    # ------------------------------------------------------------------
    def test_06_frozen_grp_gate_rejects_none_nan_mismatch_and_full_workflow(self):
        g_mod = PureStateGRP()
        with self.assertRaises(ValueError):
            FrozenGRP(g_mod, None, require_gate=True)
        for bad_rep in (
            {"top1_ece": float("nan"), "classwise_ece": 0.01, "brier": 0.4},
            {"top1_ece": 0.01, "classwise_ece": float("inf"), "brier": 0.4},
            {"top1_ece": 0.01, "classwise_ece": 0.01, "brier": 0.4, "grp_sha256": "deadbeef"},
            {"top1_ece": 0.01, "classwise_ece": 0.01, "brier": 0.4, "ruleset_sha256": "deadbeef"},
            {"top1_ece": 0.01, "classwise_ece": 0.01, "brier": 0.4, "rank_payoffs": [2.0, 0.0, -1.0, -1.0]},
        ):
            with self.assertRaises(ValueError):
                FrozenGRP(g_mod, bad_rep, require_gate=True)

        # Explicitly disabling require_gate works with calibration_metrics=None and still records report.
        ungated = FrozenGRP(g_mod, None, require_gate=False)
        self.assertFalse(ungated.calibration_report["gated"])
        self.assertEqual(ungated.calibration_report["grp_sha256"], compute_grp_checksum(g_mod))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_bc_p = root / "grp_full_src.bc.tsv.gz"
            full_lines: list[str] = []
            pay_templates = [
                (45000, 27000, 17000, 11000),
                (11000, 45000, 27000, 17000),
                (17000, 11000, 45000, 27000),
                (27000, 17000, 11000, 45000),
            ]
            for idx in range(4):
                full_lines.extend(_make_complete_tonpuusen_bc_lines(f"full-{idx}", final_scores=pay_templates[idx % 4]))
            with gzip.open(full_bc_p, "wt", encoding="utf-8") as fp:
                fp.writelines(full_lines)
            ds_full = extract_grp_samples_from_source(full_bc_p)
            self.assertEqual(len(ds_full), 32)
            tr, va = ds_full.split_by_uuid(val_fraction=0.25, seed=11)
            self.assertEqual(set(tr.uuids) & set(va.uuids), set())

            bc_p = root / "grp_src.bc.tsv.gz"
            lines: list[str] = []
            # Build 64 cleanly separated post-opening games across all 4 rank outcomes for GRP CLI fit + temperature calibration
            for idx in range(64):
                lines.extend(_make_complete_tonpuusen_bc_lines(f"grpu-{idx}", final_scores=pay_templates[idx % 4])[2:])
            with gzip.open(bc_p, "wt", encoding="utf-8") as fp:
                fp.writelines(lines)

            grp_p = root / "trained_grp.pt"
            rc = grp_cli_main(
                [
                    "--source", str(bc_p),
                    "--output", str(grp_p),
                    "--val-fraction", "0.25",
                    "--epochs", "35",
                    "--batch-size", "64",
                    "--lr", "0.01",
                    "--seed", "11",
                    "--device", "cpu",
                ]
            )
            self.assertEqual(rc, 0)
            reloaded = load_frozen_grp(grp_p, require_gate=True)
            self.assertTrue(reloaded.calibration_report["passed_gate"])
            self.assertEqual(reloaded.calibration_report["grp_sha256"], reloaded.grp_sha256)

            # Tampering checkpoint payload must fail closed!
            bad_p = root / "tampered_grp.pt"
            raw_ckpt = torch.load(grp_p, map_location="cpu", weights_only=True)
            raw_ckpt["model_state_dict"]["net.4.bias"].add_(1.0)
            torch.save(raw_ckpt, bad_p)
            with self.assertRaises(ValueError):
                load_frozen_grp(bad_p, require_gate=True)

            # Verify StreamingIQLDataset("grp_pbrs") telescoping + recipe checksum mismatch fail-closed!
            rl_p = root / "grp_rl.tsv"
            rl_p.write_text("".join(_make_complete_rl_game_lines(seat=0) + _make_complete_rl_game_lines(seat=1)), encoding="utf-8")
            with StreamingIQLDataset(rl_p, batch_size=2, reservoir_size=16, warmup_transitions=2, reward_scheme="grp_pbrs", grp=grp_p, max_prefetch=0) as s_grp:
                b = s_grp.next_batch()
                self.assertIsNotNone(b)
                mod = JointStructuredIQLModule(
                    preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                    n_d1_layers=1, n_d2_layers=1, device=torch.device("cpu"), dtype=torch.float32,
                )
                opt = build_joint_iql_optimizer(mod, base_lr=1e-4, weight_decay=0.01)
                wrong_rec = build_joint_iql_recipe(mod, base_lr=1e-4, weight_decay=0.01, decoder1_lr_scale=0.25, grp=None)
                with self.assertRaises(ValueError):
                    train_joint_iql_batches(mod, s_grp, opt, recipe=wrong_rec, device=torch.device("cpu"), max_steps=1)

    # ------------------------------------------------------------------
    # §7: FinalRank Per-Observer Game-Level Weighting & Equivalence
    # ------------------------------------------------------------------
    def test_07_final_rank_skewed_observer_weighting_and_accum_ddp_equivalence(self):
        dev = torch.device("cpu")
        sp_raw = _make_sparse_row(seat=0, chang=0, ju=0)
        sp_base = torch.tensor(
            sp_raw + [NUM_TYPES_OF_SPARSE_FEATURES] * (MAX_NUM_ACTIVE_SPARSE_FEATURES - len(sp_raw)),
            dtype=torch.int32,
        )
        nu_base = torch.tensor([0, 0, 25000, 25000, 25000, 25000], dtype=torch.int32)
        pr_base = torch.tensor(
            [0] + [NUM_TYPES_OF_PROGRESSION_FEATURES] * (MAX_LENGTH_OF_PROGRESSION_FEATURES - 1),
            dtype=torch.int32,
        )
        ca_base = torch.tensor(
            [4, 44, 84, 130] + [PAD_ACTION_ID] * (MAX_NUM_ACTION_CANDIDATES - 4),
            dtype=torch.int32,
        )

        # Build two observers with identical per-step inputs inside one batch,
        # Observer A: 2 steps (T_player=2 -> w_t = 5.0/step), FinalRank=0
        # Observer B: 8 steps (T_player=8 -> w_t = 1.25/step), FinalRank=3
        # Each observer spans 2 equal-weight game contributions: sum(w_A) == 10.0 == sum(w_B)!
        def _make_skewed_round(t_steps: int, t_player: int, rank_lbl: int, uid: str) -> RoundTrajectory:
            n_t = t_steps
            pr_rows = []
            for t in range(n_t):
                row = [0] + [5 + 0 * 148 + k * 4 for k in range(t)]
                pr_rows.append(
                    row + [NUM_TYPES_OF_PROGRESSION_FEATURES] * (MAX_LENGTH_OF_PROGRESSION_FEATURES - len(row))
                )
            sp_t = sp_base.unsqueeze(0).repeat(n_t, 1)
            nu_t = nu_base.unsqueeze(0).repeat(n_t, 1)
            pr_t = torch.tensor(pr_rows, dtype=torch.int32)
            ca_t = ca_base.unsqueeze(0).repeat(n_t, 1)
            feats = derive_structured_v2_batch(sp_t, nu_t, pr_t, ca_t, include_room_level=False)
            ev_len = int(feats["v2_event_len"][-1].item())
            return RoundTrajectory(
                round_events=feats["v2_events"][-1, :ev_len].clone(),
                round_event_len=ev_len,
                decision_event_len=feats["v2_event_len"].to(torch.int32),
                v2_tile_feats=feats["v2_tile_feats"],
                v2_global_feats=feats["v2_global_feats"],
                v2_player_feats=feats["v2_player_feats"],
                v2_cand_ids=feats["v2_cand_ids"],
                v2_cand_mask=feats["v2_cand_mask"],
                v2_cand_mode=feats["v2_cand_mode"],
                v2_cand_dyn=feats["v2_cand_dyn"],
                candidates=ca_t,
                action_idx=torch.zeros(n_t, dtype=torch.long),
                final_rank=torch.full((n_t,), rank_lbl, dtype=torch.long),
                observer_total_steps=t_player,
                uuid=uid,
            )

        # Analytical gradient neutrality test when T_A=2 (rank 0) vs T_B=8 (rank 3) on identical 1-step obs replicated!
        r_a = _make_skewed_round(2, 2, 0, "A")
        r_b = _make_skewed_round(8, 8, 3, "B")
        c_ab = collate_round_batch([r_a, r_b])
        torch.testing.assert_close(
            c_ab["final_rank_weight"][0, :2].sum(),
            torch.tensor(FINAL_RANK_T_REF),
            atol=1e-5, rtol=1e-5,
        )
        torch.testing.assert_close(
            c_ab["final_rank_weight"][1, :8].sum(),
            torch.tensor(FINAL_RANK_T_REF),
            atol=1e-5, rtol=1e-5,
        )

        # Also verify exact gradient equivalence across:
        # (1) Single full batch [r0, r1, r2], (2) grad_accum_steps=3 microbatches, (3) 2-rank DDP!
        r0 = _make_skewed_round(2, 6, 0, "G0")
        r1 = _make_skewed_round(4, 6, 1, "G0")
        r2 = _make_skewed_round(5, 5, 3, "G1")
        all_rds = [r0, r1, r2]

        def _single_grads() -> dict[str, torch.Tensor]:
            torch.manual_seed(904)
            m = TrajectoryBCModel(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, use_next_event_head=False,
                use_final_rank_head=True, final_rank_weight=1.0, device=dev, dtype=torch.float32,
            )
            b = collate_round_batch(all_rds)
            m(b)["loss"].backward()
            return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}

        g_single = _single_grads()

        # Gradient accumulation with uneven microbatches [r0] and [r1, r2]
        from kanachan.training.bc.trajectory_v2 import _compute_microgroup_denoms

        torch.manual_seed(904)
        m_acc = TrajectoryBCModel(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=1, n_d2_layers=1, use_next_event_head=False,
            use_final_rank_head=True, final_rank_weight=1.0, device=dev, dtype=torch.float32,
        )
        mb0 = collate_round_batch([r0])
        mb1 = collate_round_batch([r1, r2])
        den = _compute_microgroup_denoms([mb0, mb1], is_dist=False, world_size=1, device=dev)
        m_acc(mb0, denom_overrides=den, ddp_world_size=1)["loss"].backward()
        m_acc(mb1, denom_overrides=den, ddp_world_size=1)["loss"].backward()
        for n, p in m_acc.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(p.grad, g_single[n], atol=1e-6, rtol=1e-5)

        # 2-rank DDP gradient equivalence with uneven split (rank0=[r0], rank1=[r1, r2])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rds_p = root / "rounds.pt"
            torch.save(all_rds, rds_p)
            _spawn_2rank("--ddp-fr-grad-worker", str(root), str(rds_p))
            g_ddp = torch.load(root / "ddp_grads.pt", weights_only=True)
            for n in g_single:
                torch.testing.assert_close(g_ddp[n], g_single[n], atol=1e-6, rtol=1e-5)

    # ------------------------------------------------------------------
    # §8: Corrected Audit Coverage Semantics (PAD vs 1..6 + 3 Furiten + 6 Kan)
    # ------------------------------------------------------------------
    def test_08_audit_tool_pad_exclusion_and_3_furiten_6_kan_coverage(self):
        self.assertNotIn("0", audit_tool.REQUIRED_CAND_MODES)
        self.assertEqual(audit_tool.REQUIRED_CAND_MODES, ("1", "2", "3", "4", "5", "6"))
        for req in (
            "permanent_furiten",
            "temporary_furiten",
            "riichi_furiten",
            "ankan_event",
            "ankan_cand",
            "kakan_event",
            "kakan_cand",
            "daiminkan_event",
            "daiminkan_cand",
        ):
            self.assertIn(req, audit_tool.REQUIRED_RULE_CATEGORIES)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # File with ONLY discard actions (cand_mode == 1) and lots of PAD (cand_mode == 0) slots
            lines = _make_complete_tonpuusen_bc_lines("pad-check")
            # Keep only step 0 (discard line with cands 4,44,84,130 -> mode 1 + 28 PAD slots!)
            only_disc = root / "only_disc.tsv"
            only_disc.write_text(lines[0] * 3, encoding="utf-8")
            rep_json = root / "rep.json"
            rc = audit_tool.main(
                ["--data", str(only_disc), "--steps", "3", "--ref-steps", "1", "--perm-steps", "1", "--out", str(rep_json)]
            )
            self.assertEqual(rc, 1)
            rep = json.loads(rep_json.read_text(encoding="utf-8"))
            self.assertNotIn("0", rep["cand_mode_counts"])
            self.assertNotIn("0", rep["ref_verified_cand_mode_counts"])
            self.assertGreater(rep["pad_slot_count"], 0)
            self.assertGreater(rep["ref_verified_pad_slot_count"], 0)
            self.assertEqual(rep["unverified_modes"], ["2", "3", "4", "5", "6"])

    def test_08b_audit_tool_real_data_category_harvest(self):
        val_bc = require_or_skip_real_data(self, "bc", context="audit rare category harvest")
        uuids, sp, nu, pr, ca, cat_counts = audit_tool.harvest_rare_categories(
            val_bc, scan_limit=150000, per_cat_target=16
        )
        for req in audit_tool.REQUIRED_RULE_CATEGORIES:
            self.assertGreater(cat_counts.get(req, 0), 0, f"Missing {req} in 150k {val_bc.name}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--ddp-bc-worker":
        _worker_ddp_bc(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--ddp-iql-worker":
        _worker_ddp_iql(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--ddp-fatal-bc-worker":
        _worker_ddp_fatal_bc(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--ddp-fr-grad-worker":
        _worker_ddp_fr_grad(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        unittest.main()
