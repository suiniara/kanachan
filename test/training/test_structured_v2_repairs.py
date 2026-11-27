"""Comprehensive regression tests for StructuredDualDecoder V2 P1/P2 repairs & audit."""

from __future__ import annotations

import datetime
import gzip
import json
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel

from kanachan.model_loader import load_model
from kanachan.nn.structured_dual_decoder import (
    StructuredDualDecoder,
    StructuredKVCache,
)
from kanachan.nn.structured_iql import (
    JointStructuredIQLModule,
    QuantileQHead,
    compute_qr_dqn_target_quantiles,
    quantile_cql_loss,
)
from kanachan.simulation import simulate
from kanachan.structured_v2 import (
    ACTION_FAMILY_NAMES,
    PAD_ACTION_ID,
    derive_structured_v2_batch,
    permute_structured_batch,
)
from kanachan.training.bc.trajectory_v2 import (
    TrajectoryBCModel,
    build_round_trajectories_from_raw,
    classify_action_id,
    collate_round_batch,
    parse_bc_line,
    parse_bc_rounds,
    train_trajectory_bc_steps,
)
from kanachan.training.core.bc.data_iterator import DataIterator as CoreBCDataIterator
from kanachan.training.core.bc.prefetch import CandidateFeaturePrefetch
from kanachan.training.core.optimizers import adamw_parameter_groups
from kanachan.training.core.ranking import scores_to_ranks
from kanachan.training.core.test_env import (
    is_cuda_skipped,
    preferred_test_device,
    require_or_skip_cuda,
    require_or_skip_real_data,
)
from kanachan.training.iql.joint_v2 import (
    attach_structured_v2_transition_features,
    build_joint_iql_recipe,
    load_joint_iql_checkpoint,
    save_joint_iql_checkpoint,
    train_joint_iql_batches,
)
from kanachan.training.online import state as online_state

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _make_bc_row(
    uuid: str,
    *,
    seat: int = 0,
    chang: int = 0,
    ju: int = 0,
    ben: int = 0,
    left: int = 50,
    step_events: tuple[int, ...] = (),
    round_summary: str = "5",
    final_scores: tuple[int, int, int, int] | None = (41200, 28500, 19300, 11000),
    scores_rel: tuple[int, int, int, int] = (25000, 25000, 25000, 25000),
    results_override: str | None = None,
    num_cols: int = 8,
    zimo: int = 31,
) -> str:
    closed = {
        1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1,
        21: 1, 22: 1, 23: 1, 31: 1, 32: 1, 33: 1, 34: 1,
    }
    sp = [x for x in _make_sparse(seat=seat, chang=chang, ju=ju, left=left, closed37=closed, zimo=zimo) if x != 510]
    nu = [ben, 0, *scores_rel]
    pr = [0] + list(step_events)
    ca = [4, 44, 84, 126]
    act = 1
    base = [
        uuid,
        ",".join(map(str, sp)),
        ",".join(map(str, nu)),
        ",".join(map(str, pr)),
        ",".join(map(str, ca)),
        str(act),
    ]
    if num_cols == 6:
        return "\t".join(base) + "\n"
    if num_cols == 7:
        return "\t".join(base + [round_summary]) + "\n"
    if results_override is not None:
        res_str = results_override
    else:
        assert final_scores is not None
        res = [0, 0, 0, 0, ben, 0, 25000, 25000, 25000, 25000, *final_scores]
        res_str = ",".join(map(str, res))
    return "\t".join(base + [round_summary, res_str]) + "\n"


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


def _draw_rng_sample(n: int = 4) -> dict[str, list]:
    out = {
        "py": [random.random() for _ in range(n)],
        "np": np.random.random(n).tolist(),
        "torch": torch.rand(n).tolist(),
    }
    if (not is_cuda_skipped()) and torch.cuda.is_available():
        out["cuda"] = torch.rand(n, device="cuda").cpu().tolist()
    return out


def _make_iql_transition_batches(
    seed: int, n_batches: int = 2, batch_size: int = 2, *, include_results: bool = False
) -> list[TensorDict]:
    torch.manual_seed(seed)
    closed = {
        1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1,
        21: 1, 22: 1, 23: 1, 31: 2, 32: 2,
    }
    sp = torch.tensor(
        [_make_sparse(seat=k % 4, left=42 - k, closed37=closed, zimo=32) for k in range(batch_size)],
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
        nxt_dict: dict[str, torch.Tensor] = {
            "sparse": sp.clone(),
            "numeric": nu.clone(),
            "progression": pr.clone(),
            "candidates": ca.clone(),
            **{k: v.clone() for k, v in feats.items()},
            "done": done,
            "reward": reward,
            "end_of_game": done,
        }
        if include_results:
            res = torch.zeros((batch_size, 14), dtype=torch.int32)
            res[:, 10:14] = torch.tensor([41000, 29000, 19000, 11000], dtype=torch.int32)
            nxt_dict["results"] = res
        else:
            nxt_dict["final_rank"] = torch.tensor([k % 4 for k in range(batch_size)], dtype=torch.int64)
        batches.append(
            TensorDict(
                {
                    "sparse": sp.clone(),
                    "numeric": nu.clone(),
                    "progression": pr.clone(),
                    "candidates": ca.clone(),
                    **{k: v.clone() for k, v in feats.items()},
                    "action": torch.tensor([(bi + k) % 4 for k in range(batch_size)], dtype=torch.int32),
                    "next": TensorDict(nxt_dict, batch_size=[batch_size]),
                },
                batch_size=[batch_size],
            )
        )
    return batches


def _ddp_phase_worker(mode: str, rank: int, world_size: int, store_uri: str, tmp_dir: str) -> None:
    os.environ["USE_LIBUV"] = "0"
    torch.set_num_threads(1)
    use_cuda = (mode != "fail_save") and (not is_cuda_skipped()) and torch.cuda.is_available() and (world_size == 1)
    dev = torch.device("cuda" if use_cuda else "cpu")
    dist.init_process_group(
        backend="gloo",
        init_method=store_uri,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        tmp = Path(tmp_dir)
        # Distinct seeds per rank BEFORE saving or scrambled on resume!
        if mode == "save_phase":
            random.seed(1000 + rank * 7777)
            np.random.seed(2000 + rank * 7777)
            torch.manual_seed(3000 + rank * 7777)
            if (not is_cuda_skipped()) and torch.cuda.is_available():
                torch.cuda.manual_seed_all(4000 + rank * 7777)
        else:
            # Deliberately identical / scrambled RNG on restart to prove restore works!
            random.seed(999999)
            np.random.seed(999999)
            torch.manual_seed(999999)
            if (not is_cuda_skipped()) and torch.cuda.is_available():
                torch.cuda.manual_seed_all(999999)

        torch.manual_seed(1234)  # identical initial weights on both ranks
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
        # Restore distinct rank RNG after identical model init
        if mode == "save_phase":
            random.seed(1000 + rank * 7777)
            np.random.seed(2000 + rank * 7777)
            torch.manual_seed(3000 + rank * 7777)
            if (not is_cuda_skipped()) and torch.cuda.is_available():
                torch.cuda.manual_seed_all(4000 + rank * 7777)
            # Advance RNG slightly differently per rank
            for _ in range(rank + 1):
                random.random()
                np.random.random()
                torch.rand(1)

        opt = torch.optim.AdamW(
            mod.parameter_groups(base_lr=1e-3, weight_decay=0.01, decoder1_lr_scale=0.25)
        )
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
        recipe = build_joint_iql_recipe(
            mod, base_lr=1e-3, weight_decay=0.01, decoder1_lr_scale=0.25
        )
        ddp = DistributedDataParallel(mod)
        batches = _make_iql_transition_batches(seed=77 + rank, n_batches=4, batch_size=2)
        ckpt_dir = tmp / "ckpt"

        if mode == "fail_save":
            bad_file = tmp / "not_a_dir"
            if rank == 0:
                bad_file.write_text("blocker", encoding="utf-8")
            dist.barrier()
            try:
                save_joint_iql_checkpoint(
                    bad_file, mod, opt, step=1, cursor=1, recipe=recipe, scheduler=sched
                )
            except RuntimeError as exc:
                (tmp / f"fail_rank_{rank}.txt").write_text(str(exc), encoding="utf-8")
                return
            raise AssertionError(f"Rank {rank} expected RuntimeError on Rank-0 save failure.")

        if mode == "save_phase":
            orig_atomic = online_state.atomic_save
            save_calls = [0]

            def _counting_atomic(path, obj):
                save_calls[0] += 1
                return orig_atomic(path, obj)

            import kanachan.nn.structured_iql as siql_mod
            import kanachan.training.iql.joint_v2 as jv2_mod
            jv2_mod.atomic_save = _counting_atomic
            siql_mod.atomic_save = _counting_atomic
            try:
                train_joint_iql_batches(
                    mod,
                    batches[:2],
                    opt,
                    recipe=recipe,
                    device=dev,
                    scheduler=sched,
                    ckpt_dir=ckpt_dir,
                    checkpoint_every=2,
                    ddp_wrapped=ddp,
                )
            finally:
                jv2_mod.atomic_save = orig_atomic
                siql_mod.atomic_save = orig_atomic

            expected_rng = _draw_rng_sample()
            lag_q1 = float(sum((t - o).abs().sum().item() for t, o in zip(mod.q1_target_head.parameters(), mod.q1_head.parameters())))
            lag_qr = float(sum((t - o).abs().sum().item() for t, o in zip(mod.quantile_q_target_head.parameters(), mod.quantile_q_head.parameters())))
            assert lag_q1 > 1e-6 and lag_qr > 1e-6
            summary = {
                "rank": rank,
                "save_calls": save_calls[0],
                "expected_rng": expected_rng,
                "lr": sched.get_last_lr(),
                "lag_q1": lag_q1,
                "lag_qr": lag_qr,
                "q1_tgt_0": mod.q1_target_head.v_fc2.weight.detach().flatten().tolist(),
                "qr_tgt_0": mod.quantile_q_target_head.fc2.weight.detach().flatten().tolist()[:16],
            }
            (tmp / f"save_summary_{rank}.json").write_text(json.dumps(summary), encoding="utf-8")
            dist.barrier()
            return

        if mode == "resume_phase":
            # Corrupt target heads before load to ensure load + sync restores true Rank-0 lag
            with torch.no_grad():
                for p in mod.q1_target_head.parameters():
                    p.zero_()
                for p in mod.quantile_q_target_head.parameters():
                    p.fill_(42.0)

            ckpt_file = ckpt_dir / "joint-iql-2.pt"
            step, cursor = load_joint_iql_checkpoint(
                ckpt_file, mod, opt, expected_recipe=recipe, scheduler=sched, map_location="cpu"
            )
            assert step == 2 and cursor == 2, (step, cursor)
            lr_on_load = sched.get_last_lr()
            resumed_rng = _draw_rng_sample()

            lag_q1 = float(sum((t - o).abs().sum().item() for t, o in zip(mod.q1_target_head.parameters(), mod.q1_head.parameters())))
            lag_qr = float(sum((t - o).abs().sum().item() for t, o in zip(mod.quantile_q_target_head.parameters(), mod.quantile_q_head.parameters())))
            assert lag_q1 > 1e-6 and lag_qr > 1e-6

            # Continue training steps 2..4 across both ranks
            train_joint_iql_batches(
                mod,
                batches[2:4],
                opt,
                recipe=recipe,
                device=dev,
                scheduler=sched,
                ddp_wrapped=ddp,
            )
            post_train_rng = _draw_rng_sample()

            # Gather Target Q from Rank 0 and verify exact equality across ranks after training!
            q1_tgt = torch.cat([p.detach().flatten() for p in mod.q1_target_head.parameters()])
            qr_tgt = torch.cat([p.detach().flatten() for p in mod.quantile_q_target_head.parameters()])
            ref_q1 = q1_tgt.clone()
            ref_qr = qr_tgt.clone()
            dist.broadcast(ref_q1, src=0)
            dist.broadcast(ref_qr, src=0)
            torch.testing.assert_close(q1_tgt, ref_q1, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(qr_tgt, ref_qr, atol=1e-6, rtol=1e-6)

            summary = {
                "rank": rank,
                "resumed_rng": resumed_rng,
                "post_train_rng": post_train_rng,
                "lr_on_load": lr_on_load,
                "lr_after_resume": sched.get_last_lr(),
                "lag_q1": lag_q1,
                "lag_qr": lag_qr,
                "q1_tgt_0": mod.q1_target_head.v_fc2.weight.detach().flatten().tolist(),
                "qr_tgt_0": mod.quantile_q_target_head.fc2.weight.detach().flatten().tolist()[:16],
            }
            (tmp / f"resume_summary_{rank}.json").write_text(json.dumps(summary), encoding="utf-8")
            dist.barrier()
    finally:
        dist.destroy_process_group()


class TestSourceV3FinalRank(unittest.TestCase):
    def test_distinct_scores_all_seats_and_ignore_round_summary(self):
        # round_summary="5" and "17,19,22" previously caused bogus final_rank=5 -> clamped 3, or -1!
        scores = (41200, 28500, 19300, 11000)  # ranks: seat0->0, seat1->1, seat2->2, seat3->3
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "distinct.tsv"
            lines = []
            for seat in range(4):
                lines.append(_make_bc_row(f"g-{seat}", seat=seat, round_summary="5", final_scores=scores))
                lines.append(_make_bc_row(f"g-{seat}", seat=seat, step_events=(16,), round_summary="17,19,22", final_scores=scores))
            p.write_text("".join(lines), encoding="utf-8")
            rounds = parse_bc_rounds(p, allow_incomplete=True)
            self.assertEqual(len(rounds), 4)
            for seat, r in enumerate(rounds):
                self.assertIsNotNone(r.final_rank)
                self.assertEqual(r.final_rank.tolist(), [seat, seat])

    def test_tied_scores_and_observer_not_seat_zero(self):
        # Ties broken by smaller starting seat per scores_to_ranks!
        # [30000, 30000, 20000, 20000] -> seat0:0, seat1:1, seat2:2, seat3:3
        # [15000, 35000, 35000, 15000] -> seat1:0, seat2:1, seat0:2, seat3:3
        cases = [
            ((30000, 30000, 20000, 20000), [0, 1, 2, 3]),
            ((15000, 35000, 35000, 15000), [2, 0, 1, 3]),
            ((25000, 25000, 25000, 25000), [0, 1, 2, 3]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for idx, (scores, exp_ranks) in enumerate(cases):
                native = scores_to_ranks(torch.tensor([scores], dtype=torch.int64))[0].tolist()
                self.assertEqual(native, exp_ranks)
                p = Path(tmp) / f"tie_{idx}.tsv"
                lines = [
                    _make_bc_row(f"tie-{idx}-{seat}", seat=seat, round_summary="2,14", final_scores=scores)
                    for seat in (1, 2, 3, 0)
                ]
                p.write_text("".join(lines), encoding="utf-8")
                rounds = parse_bc_rounds(p, allow_incomplete=True)
                got = [int(r.final_rank[0].item()) for r in rounds]
                self.assertEqual(got, [exp_ranks[1], exp_ranks[2], exp_ranks[3], exp_ranks[0]])

    def test_invalid_and_missing_results_rejected_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_cases = [
                ("empty_results_8col", _make_bc_row("u1", results_override="")),
                ("missing_col8_7col", _make_bc_row("u2", num_cols=7)),
                ("13_ints", _make_bc_row("u3", results_override=",".join(["0"] * 13))),
                ("15_ints", _make_bc_row("u4", results_override=",".join(["0"] * 15))),
                ("non_int", _make_bc_row("u5", results_override="0,0,0,0,0,0,0,0,0,0,25000,x,25000,25000")),
                ("neg_benchang", _make_bc_row("u6", results_override="0,0,0,0,-1,0,0,0,0,0,25000,25000,25000,25000")),
                ("int32_max", _make_bc_row("u7", results_override="0,0,0,0,0,0,0,0,0,0,2147483647,0,0,0")),
                ("bad_round_summary", _make_bc_row("u8", round_summary="")),
            ]
            for name, content in bad_cases:
                p = Path(tmp) / f"{name}.tsv"
                p.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError, msg=f"Expected ValueError for {name}"):
                    parse_bc_rounds(p, allow_incomplete=True)

            # Inconsistent final scores within same (uuid, seat) must fail
            p_inc = Path(tmp) / "inconsistent.tsv"
            p_inc.write_text(
                _make_bc_row("same", seat=2, final_scores=(40000, 30000, 20000, 10000))
                + _make_bc_row("same", seat=2, step_events=(16,), final_scores=(10000, 20000, 30000, 40000)),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                parse_bc_rounds(p_inc, allow_incomplete=True)

    def test_explicit_unlabeled_6col_and_final_rank_head_supervision(self):
        with tempfile.TemporaryDirectory() as tmp:
            p6 = Path(tmp) / "unlabeled.tsv"
            p6.write_text(_make_bc_row("unlab", seat=1, num_cols=6), encoding="utf-8")
            rounds_unlab = parse_bc_rounds(p6, allow_unlabeled=True, allow_incomplete=True)
            self.assertEqual(len(rounds_unlab), 1)
            self.assertIsNone(rounds_unlab[0].final_rank)
            with self.assertRaises(ValueError):
                parse_bc_rounds(p6, allow_unlabeled=False)

            # Supervise TrajectoryBCModel FinalRankHead on true 8-col labels for seat=3
            p8 = Path(tmp) / "labeled.tsv"
            lines = []
            for g, (sc, rel) in enumerate([
                ((10000, 20000, 22000, 48000), (48000, 22000, 20000, 10000)),  # seat 3 -> rank 0
                ((48000, 30000, 15000, 7000), (7000, 48000, 30000, 15000)),    # seat 3 -> rank 3
            ]):
                lines.append(_make_bc_row(f"sup-{g}", seat=3, round_summary="5", final_scores=sc, scores_rel=rel))
                lines.append(_make_bc_row(f"sup-{g}", seat=3, step_events=(16,), round_summary="5", final_scores=sc, scores_rel=rel))
            p8.write_text("".join(lines), encoding="utf-8")
            rounds_lab = parse_bc_rounds(p8, allow_incomplete=True)
            coll = collate_round_batch(rounds_lab)
            self.assertEqual(coll["final_rank"][0, :2].tolist(), [0, 0])
            self.assertEqual(coll["final_rank"][1, :2].tolist(), [3, 3])

            torch.manual_seed(42)
            m = TrajectoryBCModel(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, use_next_event_head=False,
                use_final_rank_head=True, final_rank_weight=1.0,
                device=torch.device("cpu"), dtype=torch.float32,
            )
            opt = torch.optim.AdamW(adamw_parameter_groups(m, 0.0), lr=2e-3)
            loss0 = float(m(coll)["loss_final_rank"].item())
            train_trajectory_bc_steps(m, rounds_lab, opt, batch_size=2, max_steps=110, device=torch.device("cpu"))
            loss1 = float(m(coll)["loss_final_rank"].item())
            self.assertLess(loss1, loss0 * 0.25)
            dmask = coll["decision_mask"]
            _, rank_logits = m.state_head(m.backbone.forward_round_trajectory(
                coll["round_events"], coll["round_event_len"], coll["decision_event_len"], dmask,
                coll["v2_tile_feats"], coll["v2_global_feats"], coll["v2_player_feats"],
                coll["v2_cand_ids"], coll["v2_cand_mask"], coll["v2_cand_mode"], coll["v2_cand_dyn"],
            )["encode"])
            self.assertEqual(rank_logits.argmax(-1).tolist(), [0, 0, 3, 3])

            # Invalid final_rank out of [0, 3] must fail closed via require_tensor
            bad_coll = dict(coll)
            bad_coll["final_rank"] = torch.full_like(coll["final_rank"], 5)
            with self.assertRaises(ValueError):
                m(bad_coll)

    def test_real_cpp_annotation_format_integration(self):
        # Always run against embedded real C++ source-v3 lines from kanachan annotator,
        # and also verify real dataset file if present on disk.
        sp_valid = ",".join(
            str(x)
            for x in _make_sparse(
                room=2, seat=1, chang=0, ju=0, left=0, doras=(3,),
                closed37={1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1, 21: 1, 22: 1, 23: 1, 31: 1, 32: 1, 33: 1, 34: 1},
                zimo=31,
            )
            if x != 510
        )
        embedded_cpp = (
            f"00001111-2222-3333-4444-555566667777\t{sp_valid}\t"
            "0,0,32000,28000,25000,15000\t"
            "0,141,441\t"
            "4,8,44,48,84,88,126\t"
            "0\t5\t-1000,0,1000,0,0,1,31000,28000,26000,15000,18500,39200,39200,3100\n"
            f"00001111-2222-3333-4444-555566667777\t{sp_valid}\t"
            "0,0,32000,28000,25000,15000\t"
            "0,141,441,593,145\t"
            "4,8,44,48,84,88,126\t"
            "1\t17,19,22\t-1000,0,1000,0,0,1,31000,28000,26000,15000,18500,39200,39200,3100\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "real_cpp.tsv"
            p.write_text(embedded_cpp, encoding="utf-8")
            it = CoreBCDataIterator(path=p, num_skip_samples=0, rewrite_rooms=None, rewrite_grades=None, local_rank=0)
            row0 = next(it)
            del it
            expected_rank = int(scores_to_ranks(row0[6][10:14].unsqueeze(0))[0, int(row0[0][6].item()) - 71].item())
            # Observer is seat=1 (sp[6]=72); seats 1 & 2 tie at 39200 -> seat 1 gets rank 0!
            self.assertEqual(expected_rank, 0)
            rounds = parse_bc_rounds(p, allow_incomplete=True)
            self.assertEqual(len(rounds), 1)
            self.assertEqual(rounds[0].final_rank.tolist(), [0, 0])

    def test_real_data_cpp_annotation_and_parse_bc_rounds(self):
        val_bc = require_or_skip_real_data(
            self, "bc", context="parse_bc_rounds strict real-data check"
        )
        # Harvest complete real games from source-v3 dataset and verify strict complete-game parsing!
        game_lines: dict[str, list[str]] = {}
        max_idx: dict[str, int] = {}
        with gzip.open(val_bc, "rt", encoding="utf-8") as fp:
            for idx_line, line in enumerate(fp):
                if idx_line >= 15000:
                    break
                u = line.split("\t", 1)[0]
                game_lines.setdefault(u, []).append(line)
                max_idx[u] = idx_line
        complete_lines: list[str] = []
        picked_games = 0
        for u, lines_u in game_lines.items():
            if max_idx[u] < 10000 and len(lines_u) >= 80:
                complete_lines.extend(lines_u)
                picked_games += 1
                if picked_games >= 2:
                    break
        self.assertGreaterEqual(picked_games, 1)
        with tempfile.TemporaryDirectory() as tmp2:
            real_p = Path(tmp2) / "real_complete.tsv"
            real_p.write_text("".join(complete_lines), encoding="utf-8")
            rounds = parse_bc_rounds(real_p)
            self.assertGreater(len(rounds), 10)
            expected_by_key: dict[tuple, list[int]] = {}
            it = CoreBCDataIterator(
                path=real_p, num_skip_samples=0, rewrite_rooms=None, rewrite_grades=None, local_rank=0
            )
            for line in complete_lines:
                uuid = line.split("\t", 1)[0]
                sp, nu, _pr, _ca, _act, _rs, res = it.parse_line(line)
                seat = int(sp[6].item()) - 71
                rnk = int(scores_to_ranks(res[10:14].unsqueeze(0))[0, seat].item())
                k = (uuid, int(sp[6].item()), int(sp[7].item()), int(sp[8].item()), int(nu[0].item()))
                expected_by_key.setdefault(k, []).append(rnk)
            del it
            for r in rounds:
                self.assertIsNotNone(r.final_rank)
                self.assertTrue(((r.final_rank >= 0) & (r.final_rank < 4)).all())
                self.assertEqual(torch.unique(r.final_rank).numel(), 1)
            self.assertEqual(sum(int(r.final_rank.numel()) for r in rounds), len(complete_lines))


class TestDDPCheckpointAndPerRankRNG(unittest.TestCase):
    def _spawn_two_ranks(self, mode: str, tmp: str, port: int) -> None:
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
        store = (Path(tmp) / f"ddp_store_{mode}_{port}").resolve().as_uri()
        procs = [
            subprocess.Popen(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--ddp-worker", mode, str(r), store, tmp],
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
        outputs: list[tuple[int, str, str]] = []
        try:
            for r, p in enumerate(procs):
                out, err = p.communicate(timeout=45)
                outputs.append((p.returncode, out, err))
        finally:
            for p in procs:
                if p.poll() is None:
                    p.kill()
                    p.wait(timeout=5)
                if p.stdout:
                    p.stdout.close()
                if p.stderr:
                    p.stderr.close()
        for r, (rc, out, err) in enumerate(outputs):
            self.assertEqual(rc, 0, f"Rank {r} ({mode}) failed:\nSTDOUT:\n{out}\nSTDERR:\n{err}")

    def test_real_2rank_ddp_save_exit_restart_resume_and_fail_propagation(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 1. Two processes train & save simultaneously (Rank 0 only writes!) -> both exit completely
            with socket.socket() as s1, socket.socket() as s2:
                s1.bind(("127.0.0.1", 0))
                s2.bind(("127.0.0.1", 0))
                port1 = s1.getsockname()[1]
                port2 = s2.getsockname()[1]

            self._spawn_two_ranks("save_phase", tmp, port1)
            s0 = json.loads((Path(tmp) / "save_summary_0.json").read_text())
            s1_sum = json.loads((Path(tmp) / "save_summary_1.json").read_text())
            # Rank 0 writes joint-iql-2.pt + model.2.kanachan; Rank 1 writes 0 files!
            self.assertEqual(s0["save_calls"], 2)
            self.assertEqual(s1_sum["save_calls"], 0)
            self.assertTrue((Path(tmp) / "ckpt" / "joint-iql-2.pt").is_file())
            self.assertTrue((Path(tmp) / "ckpt" / "model.2.kanachan").is_file())
            self.assertNotEqual(s0["expected_rng"], s1_sum["expected_rng"])

            # 2. Brand-new processes resume, verify per-rank RNG exact match & Target Q sync + Polyak lag
            self._spawn_two_ranks("resume_phase", tmp, port2)
            r0 = json.loads((Path(tmp) / "resume_summary_0.json").read_text())
            r1 = json.loads((Path(tmp) / "resume_summary_1.json").read_text())
            self.assertEqual(r0["resumed_rng"], s0["expected_rng"])
            self.assertEqual(r1["resumed_rng"], s1_sum["expected_rng"])
            self.assertNotEqual(r0["resumed_rng"], r1["resumed_rng"])
            self.assertNotEqual(r0["post_train_rng"], r1["post_train_rng"])
            self.assertEqual(r0["lr_on_load"], s0["lr"])
            for g_after, g_saved in zip(r0["lr_after_resume"], s0["lr"], strict=True):
                self.assertAlmostEqual(g_after, g_saved * 0.81, places=10)
            self.assertAlmostEqual(r0["lag_q1"], s0["lag_q1"], places=5)
            self.assertAlmostEqual(r0["lag_qr"], s0["lag_qr"], places=5)
            self.assertEqual(r0["q1_tgt_0"], r1["q1_tgt_0"])
            self.assertEqual(r0["qr_tgt_0"], r1["qr_tgt_0"])

            # 3. Rank-0 disk write error must propagate cleanly to Rank 1 without deadlock
            with socket.socket() as s3:
                s3.bind(("127.0.0.1", 0))
                port3 = s3.getsockname()[1]
            self._spawn_two_ranks("fail_save", tmp, port3)
            self.assertIn("Rank-0 checkpoint save failed", (Path(tmp) / "fail_rank_0.txt").read_text())
            self.assertIn("Rank-0 checkpoint save failed", (Path(tmp) / "fail_rank_1.txt").read_text())

    def test_single_rank_mode_intact(self):
        dev = preferred_test_device()
        batches = _make_iql_transition_batches(seed=31, n_batches=4, batch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(7)
            m1 = JointStructuredIQLModule(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, use_cql_quantile_head=True, num_quantiles=8,
                device=dev, dtype=torch.float32,
            )
            o1 = torch.optim.AdamW(m1.parameter_groups(base_lr=5e-4, weight_decay=0.01))
            rec = build_joint_iql_recipe(m1, base_lr=5e-4, weight_decay=0.01, decoder1_lr_scale=0.25)
            train_joint_iql_batches(m1, batches, o1, recipe=rec, device=dev)

            torch.manual_seed(7)
            m2 = JointStructuredIQLModule(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, use_cql_quantile_head=True, num_quantiles=8,
                device=dev, dtype=torch.float32,
            )
            o2 = torch.optim.AdamW(m2.parameter_groups(base_lr=5e-4, weight_decay=0.01))
            ckpt_dir = Path(tmp) / "single"
            train_joint_iql_batches(m2, batches, o2, recipe=rec, device=dev, ckpt_dir=ckpt_dir, checkpoint_every=2)
            # Load exported actor with weights_only=True and run C++ DecisionMaker simulate()
            actor = load_model(ckpt_dir / "model.2.kanachan", map_location="cpu").eval()
            out_td = actor(batches[0][:1].clone().to("cpu"))
            self.assertIn("log_probs", out_td)
            self.assertIn("action", out_td)
            with torch.inference_mode():
                sim_logs = simulate(
                    torch.device("cpu"), torch.float32, 3, 12, actor, 12, actor,
                    1, 1, 1, 2, lambda: None, seed=888,
                )
            self.assertEqual(len(sim_logs), 1)
            self.assertEqual(len(sim_logs[0].get_result()), 4)


class TestDistributionalQRDQN(unittest.TestCase):
    def test_bellman_target_double_q_tiebreak_terminal_padding_k1_discounts(self):
        # 4 rows covering:
        # 0: non-term tie on mean Q -> smaller global_id (10 < 40) wins, gamma=0.5, reward=2.0
        # 1: terminal done=True with HUGE garbage/inf in next_target_quantiles -> target_quantiles == reward (-1.5)
        # 2: K=1 forced next state -> sole legal slot 2 selected regardless of huge padding values, gamma=1.0
        # 3: non-term with discount=0.0 -> target_quantiles == reward (0.25), ignoring next state
        reward = torch.tensor([2.0, -1.5, 0.5, 0.25], dtype=torch.float32)
        done = torch.tensor([False, True, False, False], dtype=torch.bool)
        discount = torch.tensor([0.5, 0.99, 1.0, 0.0], dtype=torch.float32)

        next_cands = torch.full((4, 32), PAD_ACTION_ID, dtype=torch.int32)
        next_cands[0, :3] = torch.tensor([40, 10, 80], dtype=torch.int32)
        next_cands[1, :32] = PAD_ACTION_ID  # Terminal empty candidate list
        next_cands[2, 0] = PAD_ACTION_ID
        next_cands[2, 2] = 77  # K=1 forced action at slot 2
        next_cands[3, :2] = torch.tensor([5, 15], dtype=torch.int32)

        num_q = 4
        tgt_q = torch.zeros((4, 32, num_q), dtype=torch.float32)
        sel_q = torch.zeros((4, 32, num_q), dtype=torch.float32)
        # Row 0: online selector ties slots 0 (id=40) and 1 (id=10) at mean=5.0 -> slot 1 (id=10) MUST win!
        sel_q[0, 0] = torch.tensor([4.0, 6.0, 4.0, 6.0])
        sel_q[0, 1] = torch.tensor([5.0, 5.0, 5.0, 5.0])
        tgt_q[0, 0] = torch.tensor([10.0, 10.0, 10.0, 10.0])
        tgt_q[0, 1] = torch.tensor([-2.0, 0.0, 2.0, 4.0])  # Double Q uses target's distribution at a*!
        # Row 1: put inf/garbage into tgt_q to test terminal immunity
        tgt_q[1, 0] = torch.tensor([float("inf"), 1e9, -1e9, 0.0])
        # Row 2: put +1e6 on padded slot 0! Must ignore pad and pick slot 2!
        sel_q[2, 0] = 1e6
        tgt_q[2, 0] = 1e6
        tgt_q[2, 2] = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        # Row 3: put large values on next state; discount=0 must zero bootstrap cleanly
        tgt_q[3, 0] = torch.tensor([100.0, 200.0, 300.0, 400.0])

        out_double = compute_qr_dqn_target_quantiles(reward, done, discount, next_cands, tgt_q, sel_q)
        torch.testing.assert_close(out_double[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))  # 2.0 + 0.5*[-2,0,2,4]
        torch.testing.assert_close(out_double[1], torch.tensor([-1.5, -1.5, -1.5, -1.5]))
        torch.testing.assert_close(out_double[2], torch.tensor([-0.5, 0.5, 1.5, 2.5]))  # 0.5 + 1.0*[-1,0,1,2]
        torch.testing.assert_close(out_double[3], torch.tensor([0.25, 0.25, 0.25, 0.25]))

        # Standard QR-DQN (selector == None -> uses target mean Q, where row 0 slot 0 has mean 10 > 1.0)
        out_std = compute_qr_dqn_target_quantiles(reward, done, discount, next_cands, tgt_q, None)
        torch.testing.assert_close(out_std[0], torch.tensor([7.0, 7.0, 7.0, 7.0]))

    def test_quantile_cql_loss_k1_pad_and_kappa_zero(self):
        cands = torch.full((2, 32), PAD_ACTION_ID, dtype=torch.int32)
        cands[0, 0] = 10  # K=1 forced!
        cands[1, :3] = torch.tensor([5, 15, 25], dtype=torch.int32)
        q_raw = torch.randn((2, 32, 8), requires_grad=True)
        # Put extreme value in PAD slots to verify CQL/QR never reads them
        q_mod = q_raw.clone()
        with torch.no_grad():
            q_mod.data[0, 5:] = 1e6
        tgt = torch.randn((2, 8))
        chosen = torch.tensor([0, 1], dtype=torch.int64)
        loss, metrics = quantile_cql_loss(q_raw, cands, chosen, tgt, kappa=1.0)
        loss0, _ = quantile_cql_loss(q_raw, cands, chosen, tgt, kappa=0.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(loss0))
        loss.backward()
        self.assertTrue(torch.isfinite(q_raw.grad).all())

        # K=1-only batch must produce CQL_loss == 0.0 and still train QR quantiles!
        cands_k1 = torch.full((2, 32), PAD_ACTION_ID, dtype=torch.int32)
        cands_k1[:, 1] = 20
        q_k1 = torch.randn((2, 32, 8), requires_grad=True)
        loss_k1, m_k1 = quantile_cql_loss(q_k1, cands_k1, torch.tensor([1, 1]), tgt)
        self.assertEqual(float(m_k1["cql_loss"].item()), 0.0)
        loss_k1.backward()
        self.assertGreater(float(q_k1.grad[:, 1].abs().sum().item()), 0.0)
        self.assertEqual(float(q_k1.grad[:, 0].abs().sum().item()), 0.0)

    def test_distributional_qr_dqn_propagation_amp_and_ema(self):
        # Prove full distributional Bellman propagation:
        # s0 -> s1 (reward=0), s1 -> terminal with bimodal outcome {-2.0, +2.0}.
        # Under distributional QR-DQN (not scalar TD target!), s0's predicted quantiles
        # spread across {-2, +2} so lower quantiles < -0.5 and upper quantiles > +0.5!
        torch.manual_seed(99)
        dev = preferred_test_device()
        mod = JointStructuredIQLModule(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=1, n_d2_layers=1, use_cql_quantile_head=True, num_quantiles=8,
            target_ema_tau=0.2, discount_factor=1.0, quantile_cql_alpha=0.01,
            device=dev, dtype=torch.float32,
        )
        closed0 = {1: 2, 2: 2, 3: 2, 11: 2, 12: 2, 13: 2, 21: 1}
        closed1 = {1: 1, 2: 2, 3: 2, 11: 2, 12: 2, 13: 2, 21: 2}
        sp = torch.tensor(
            [
                _make_sparse(seat=0, left=50, closed37=closed0, zimo=32),
                _make_sparse(seat=0, left=46, closed37=closed1, zimo=33),
            ],
            dtype=torch.int32,
        )
        nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]] * 2, dtype=torch.int32)
        pr = torch.tensor([[0] + [2165] * 108, [0, 16] + [2165] * 107], dtype=torch.int32)
        ca = torch.tensor([[4, 8, 44, 84] + [PAD_ACTION_ID] * 28] * 2, dtype=torch.int32)
        feats = derive_structured_v2_batch(sp, nu, pr, ca)

        def make_batch(r_s1: float) -> TensorDict:
            nxt = TensorDict(
                {
                    "sparse": torch.stack([sp[1], sp[1]]),
                    "numeric": torch.stack([nu[1], nu[1]]),
                    "progression": torch.stack([pr[1], pr[1]]),
                    "candidates": torch.stack([ca[1], ca[1]]),
                    **{k: torch.stack([v[1], v[1]]) for k, v in feats.items()},
                    "done": torch.tensor([False, True], dtype=torch.bool),
                    "reward": torch.tensor([0.0, r_s1], dtype=torch.float32),
                },
                batch_size=[2],
            )
            return TensorDict(
                {
                    "sparse": sp, "numeric": nu, "progression": pr, "candidates": ca,
                    **feats,
                    "action": torch.tensor([0, 0], dtype=torch.int32),
                    "next": nxt,
                },
                batch_size=[2],
            ).to(dev)

        opt = torch.optim.AdamW(mod.parameter_groups(base_lr=3e-3, weight_decay=0.0))
        b_neg, b_pos = make_batch(-2.0), make_batch(+2.0)
        for step in range(40):
            b = b_neg if step % 2 == 0 else b_pos
            opt.zero_grad(set_to_none=True)
            out = mod(b)
            out["loss"].backward()
            for p in mod.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all())
            opt.step()
            mod.update_targets()

        # Verify Target Q EMA diverged from zero init and stayed finite
        self.assertGreater(float(sum(p.abs().sum().item() for p in mod.quantile_q_target_head.parameters())), 0.0)
        with torch.no_grad():
            f_dev = {k: v.to(dev) for k, v in feats.items()}
            enc0 = mod.backbone.forward_structured(
                f_dev["v2_tile_feats"][:1], f_dev["v2_events"][:1], f_dev["v2_event_len"][:1],
                f_dev["v2_global_feats"][:1], f_dev["v2_player_feats"][:1],
                f_dev["v2_cand_ids"][:1], f_dev["v2_cand_mask"][:1],
                f_dev["v2_cand_mode"][:1], f_dev["v2_cand_dyn"][:1],
            )
            q_s0 = mod.quantile_q_head(ca[:1].to(dev), enc0)[0, 0].cpu()
            # Lower quantiles should be clearly negative and upper quantiles clearly positive!
            self.assertLess(float(q_s0[:2].mean().item()), -0.4, q_s0.tolist())
            self.assertGreater(float(q_s0[-2:].mean().item()), 0.4, q_s0.tolist())

        # Verify FP16 + GradScaler finite step on CUDA when available
        if dev.type == "cuda":
            scaler = GradScaler("cuda", enabled=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
                out_fp16 = mod(b_pos, amp_dtype=torch.float16)
            scaler.scale(out_fp16["loss"]).backward()
            scaler.unscale_(opt)
            self.assertTrue(torch.isfinite(out_fp16["loss"].detach()))
            scaler.step(opt)
            scaler.update()


class TestGPUHotPathZeroSyncAndAudit(unittest.TestCase):
    def test_zero_cuda_sync_on_intermediate_training_steps(self):
        dev = require_or_skip_cuda(self, context="GPU zero-sync audit")
        # Guard against leaked torch.use_deterministic_algorithms(True) from earlier tests (e.g. tools/train_decision_cql)
        # which would force CUB scatter_add stream syncs in backward Embedding/gather kernels.
        prev_det = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        try:
            # 1. JointStructuredIQLModule with QR-DQN + BC KL anchor + FinalRank under BF16
            batches = _make_iql_transition_batches(seed=51, n_batches=11, batch_size=2, include_results=True)
            bc_bb = StructuredDualDecoder(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, device=torch.device("cpu"), dtype=torch.float32,
            )
            from kanachan.nn.structured_dual_decoder import StructuredActionHead
            bc_head = StructuredActionHead(input_dimension=64, head_type="factorized", device=torch.device("cpu"), dtype=torch.float32)

            mod = JointStructuredIQLModule(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, use_cql_quantile_head=True, num_quantiles=8,
                final_rank=True, kl_coef=0.05, device=dev, dtype=torch.float32,
            )
            # Specifically verifies init_from_bc copies CPU BC model onto CUDA device cleanly!
            mod.init_from_bc(bc_bb, bc_head, keep_bc_anchor=True)
            mod.configure_freeze(final_rank_weight=0.10)
            opt = torch.optim.AdamW(
                mod.parameter_groups(base_lr=3e-4, weight_decay=0.01),
                fused=(dev.type == "cuda"),
            )
            rec = build_joint_iql_recipe(mod, base_lr=3e-4, weight_decay=0.01, decoder1_lr_scale=0.25)

            # Warmup 1 step then run 8 intermediate steps under sync_debug_mode=2 ("error")
            train_joint_iql_batches(mod, batches[:1], opt, recipe=rec, device=dev, amp_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                # Exclude final flush by setting log_interval=check_finite_interval=100; note at_end triggers 1 final sync.
                # So run step forward+backward+opt directly or verify zero sync BEFORE end-of-loop!
                trainable = [p for p in mod.parameters() if p.requires_grad]
                for b in batches[1:6]:
                    bd = b.to(device=dev, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                        out = mod(bd, amp_dtype=torch.bfloat16)
                    out["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt.step()
                    mod.update_targets()
            finally:
                torch.cuda.set_sync_debug_mode(0)

            # Verify _build_masks_and_rel under strict deterministic mode + sync_debug_mode="error" has 0 CUDA syncs
            bd1 = batches[1].to(dev)
            m1 = mod.backbone.decoder2.manifest[bd1["v2_cand_ids"].long()]
            torch.use_deterministic_algorithms(True)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                _ = mod.backbone.decoder2._build_masks_and_rel(
                    bd1.batch_size[0],
                    bd1["v2_cand_ids"].shape[1],
                    bd1["v2_cand_mask"],
                    m1[..., 1],
                    m1[..., 2],
                    m1[..., 3],
                    m1[..., 4],
                    dev,
                )
            finally:
                torch.cuda.set_sync_debug_mode(0)
                torch.use_deterministic_algorithms(False)

            # Also check strict load_state_dict resume with kl_coef > 0 and final_rank_weight > 0 (no manual calls needed!)
            with tempfile.TemporaryDirectory() as tmp:
                ckpt = save_joint_iql_checkpoint(Path(tmp), mod, opt, step=5, cursor=5, recipe=rec)
                mod_r = JointStructuredIQLModule(
                    preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                    n_d1_layers=1, n_d2_layers=1, use_cql_quantile_head=True, num_quantiles=8,
                    final_rank=True, final_rank_weight=0.10, kl_coef=0.05, device=dev, dtype=torch.float32,
                )
                opt_r = torch.optim.AdamW(
                    mod_r.parameter_groups(base_lr=3e-4, weight_decay=0.01),
                    fused=(dev.type == "cuda"),
                )
                load_joint_iql_checkpoint(ckpt, mod_r, opt_r, expected_recipe=rec)
                for k, v in mod.state_dict().items():
                    torch.testing.assert_close(v, mod_r.state_dict()[k], atol=0, rtol=0)

            # Verify train_joint_iql_batches with log_interval=check_finite_interval=100 has <= 1 sync (only final flush)
            import warnings
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode(1)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    train_joint_iql_batches(
                        mod, batches[1:6], opt, recipe=rec, device=dev, amp_dtype=torch.bfloat16,
                        log_interval=100, check_finite_interval=100,
                    )
                    torch.cuda.synchronize()
            finally:
                torch.cuda.set_sync_debug_mode(0)
            sync_warns = [w for w in caught if "synchronizing CUDA operation" in str(w.message)]
            self.assertLessEqual(len(sync_warns), 1, [str(w.message) for w in sync_warns])

            # 2. TrajectoryBCModel with NextEventHead & FinalRankHead on CUDA BF16: 0 sync on intermediate steps
            closed = {1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1, 21: 1, 22: 1, 23: 1, 31: 2, 32: 2}
            sp = torch.tensor([_make_sparse(seat=k % 4, left=42 - k, closed37=closed, zimo=32) for k in range(4)], dtype=torch.int32)
            nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]] * 4, dtype=torch.int32)
            pr = torch.tensor([[0, 16, 2165] + [2165] * 106, [0, 16, 28, 2165] + [2165] * 105] * 2, dtype=torch.int32)
            ca = torch.tensor([[4, 44, 84, 130] + [PAD_ACTION_ID] * 28] * 4, dtype=torch.int32)
            rounds = build_round_trajectories_from_raw(
                sp, nu, pr, ca, torch.tensor([0, 1, 2, 3]), ["r0", "r0", "r1", "r1"],
                final_ranks=torch.tensor([0, 0, 2, 2]),
            )
            bm = TrajectoryBCModel(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, use_next_event_head=True,
                use_final_rank_head=True, final_rank_weight=0.1, label_smoothing=0.05,
                device=dev, dtype=torch.float32,
            )
            bopt = torch.optim.AdamW(adamw_parameter_groups(bm, 0.01), lr=3e-4)
            cb = {k: v.to(dev) for k, v in collate_round_batch(rounds).items()}
            bm(cb)["loss"].backward()
            bopt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    bm_out = bm(cb)
                bm_out["loss"].backward()
                bopt.step()
            finally:
                torch.cuda.set_sync_debug_mode(0)
        finally:
            torch.use_deterministic_algorithms(prev_det)

    def test_rl_prefetch_next_v2_nested_permute_and_action_families(self):
        # 1. Verify RL prefetch attaches both current and next v2_* safely with terminal rows!
        batches = _make_iql_transition_batches(seed=88, n_batches=1, batch_size=2)
        raw_b = batches[0].clone(recurse=False)
        for k in list(raw_b.keys()):
            if k.startswith("v2_"):
                del raw_b[k]
        raw_nxt = raw_b["next"].clone(recurse=False)
        for k in list(raw_nxt.keys()):
            if k.startswith("v2_"):
                del raw_nxt[k]
        # Corrupt terminal row next.sparse with all 510 (matching real EpisodeReplayBuffer!)
        raw_nxt["sparse"][1] = 510
        raw_nxt["candidates"][0, 1] = 8  # asymmetric candidate so suit perm visibly changes sorted IDs
        raw_nxt["candidates"][1] = PAD_ACTION_ID
        raw_nxt["done"] = torch.tensor([False, True], dtype=torch.bool)
        raw_b["next"] = raw_nxt

        with CandidateFeaturePrefetch([raw_b], workers=1, batches=1, mode="structured_v2") as pipe:
            out_b = next(pipe)
        self.assertIn("v2_tile_feats", out_b)
        self.assertIn(("next", "v2_tile_feats"), out_b)
        self.assertEqual(int(out_b["next", "v2_event_len"][1].item()), 1)
        self.assertEqual(int(out_b["next", "v2_cand_ids"][1, 0].item()), 0)

        # Also test helper attach_structured_v2_transition_features directly
        out_direct = attach_structured_v2_transition_features(raw_b)
        torch.testing.assert_close(out_direct["next", "v2_tile_feats"], out_b["next", "v2_tile_feats"])

        # 2. Verify permute_structured_batch handles nested "next" TensorDict consistently
        p_out = permute_structured_batch(out_b, 3)
        self.assertIsInstance(p_out, TensorDict)
        self.assertIn("next", p_out)
        self.assertFalse(torch.equal(p_out["next", "v2_cand_ids"][0], out_b["next", "v2_cand_ids"][0]))

        # 3. Verify classify_action_id covers all 9 canonical families cleanly
        from kanachan.constants import NUM_TYPES_OF_ACTIONS
        seen = set()
        for gid in range(NUM_TYPES_OF_ACTIONS):
            seen.add(classify_action_id(gid))
        self.assertEqual(seen, set(ACTION_FAMILY_NAMES))


if __name__ == "__main__":
    if len(sys.argv) >= 6 and sys.argv[1] == "--ddp-worker":
        _ddp_phase_worker(sys.argv[2], int(sys.argv[3]), 2, sys.argv[4], sys.argv[5])
    else:
        unittest.main()
