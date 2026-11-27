"""Phase 1B Contract Tests #8-#16 for StructuredDualDecoder, IQL, GRP & E2E simulate."""

from __future__ import annotations

import copy
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

import torch
import torch.distributed as dist
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from kanachan.constants import MAX_NUM_ACTION_CANDIDATES, NUM_TYPES_OF_ACTIONS
from kanachan.model_loader import dump_model, dump_object, load_model
from kanachan.nn.decode_converter import DecodeConverter
from kanachan.nn.decision_decoder import DecisionContext
from kanachan.nn.structured_dual_decoder import (
    StructuredActionHead,
    StructuredDualDecoder,
    StructuredKVCache,
    StructuredStateHead,
    cached_structured_decision,
)
from kanachan.nn.structured_iql import (
    DuelingQHead,
    JointStructuredIQLModule,
    QuantileQHead,
    quantile_cql_loss,
)
from kanachan.simulation import simulate
from kanachan.structured_v2 import (
    CHI_ENCODE_MAP,
    KANACHAN_ACTION_MANIFEST,
    OFFSETS_136,
    PAD_ACTION_ID,
    PERM_ACTION_ID,
    PON_ENCODE_MAP,
    derive_structured_v2_batch,
    permute_raw_four_fields,
    permute_structured_batch,
)
from kanachan.training.bc.trajectory_v2 import (
    ACTION_FAMILY_NAMES,
    TrajectoryBCModel,
    build_round_trajectories_from_raw,
    collate_round_batch,
    evaluate_9_action_families,
    load_trajectory_bc_checkpoint,
    parse_bc_rounds,
    save_trajectory_bc_checkpoint,
    train_trajectory_bc_steps,
)
from kanachan.training.core.optimizers import adamw_parameter_groups
from kanachan.training.core.rl.discrete_cql import scalar_cql_loss
from kanachan.training.core.test_env import (
    preferred_test_device,
    require_or_skip_real_data,
)
from kanachan.training.grp import (
    FrozenGRP,
    PureStateGRP,
    canonicalize_grp_state,
    compute_grp_calibration_metrics,
    passes_gate,
    shape_hanchan_pbrs_rewards,
)
from kanachan.training.iql.joint_v2 import (
    build_joint_iql_recipe,
    load_joint_iql_checkpoint,
    save_joint_iql_checkpoint,
    train_joint_iql_batches,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_sparse(
    *,
    seat: int = 0,
    chang: int = 0,
    ju: int = 0,
    left: int = 50,
    doras: tuple[int, ...] = (1,),
    closed37: dict[int, int] | None = None,
    zimo: int | None = None,
    tonpuusen: bool = False,
) -> list[int]:
    row = [
        0,
        5 if tonpuusen else 6,
        7,
        23,
        39,
        55,
        71 + seat,
        75 + chang,
        78 + ju,
        82 + left,
    ]
    for slot, d in enumerate(doras):
        row.append(152 + slot * 37 + d)
    used_cnt = [0] * 37
    if closed37:
        for t37 in sorted(closed37):
            for _ in range(closed37[t37]):
                off = used_cnt[t37]
                used_cnt[t37] += 1
                row.append(337 + OFFSETS_136[t37] + off)
    if zimo is not None:
        row.append(473 + zimo)
    assert len(row) <= 29
    row.extend([510] * (29 - len(row)))
    return row


def _pad_prog(events: list[int], width: int = 109) -> list[int]:
    return [0] + events + [2165] * (width - 1 - len(events))


def _pad_cands(cands: list[int]) -> list[int]:
    sc = sorted(cands)
    return sc + [PAD_ACTION_ID] * (MAX_NUM_ACTION_CANDIDATES - len(sc))


def _dahai(seat: int, t37: int, moqie: int = 0, liqi: int = 0) -> int:
    return 5 + 148 * seat + 4 * t37 + 2 * moqie + liqi


def _export_actor_file(
    backbone: StructuredDualDecoder,
    head: StructuredActionHead,
    path: Path,
    dtype: torch.dtype = torch.float32,
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
    lp_tdm = TensorDictModule(
        lp_mod, in_keys=["candidates", "decode"], out_keys=["log_probs"]
    )
    arg_mod = DecodeConverter("argmax", tie_break_by_action_id=True)
    arg_tdm = TensorDictModule(
        arg_mod, in_keys=["candidates", "log_probs"], out_keys=["action"]
    )
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


def _sample_bc_line(seed_i: int) -> str:
    """Valid source-v3 BC line with >= 3 discard candidates."""
    t_pool = [1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33, 34]
    zimo = t_pool[seed_i % len(t_pool)]
    closed = {
        1: 1, 2: 1, 3: 1,
        11: 1, 12: 1, 13: 1,
        21: 1, 22: 1, 23: 1,
        31: 1, 32: 1, 33: 1, 34: 1,
    }
    sp = _make_sparse(
        seat=seed_i % 4,
        chang=0,
        ju=(seed_i // 4) % 4,
        left=50 - (seed_i % 8),
        doras=(1,),
        closed37=closed,
        zimo=zimo,
    )
    # Trim trailing 510 padding as raw source-v3 stores unpadded lists
    sp_raw = [x for x in sp if x != 510]
    nu = [0, 0, 25000, 25000, 25000, 25000]
    evs = [_dahai((seed_i + k) % 4, t_pool[(seed_i + k) % len(t_pool)]) for k in range(seed_i % 4)]
    pr_raw = [0] + evs
    ca_raw = sorted([1 * 4, 11 * 4, 21 * 4, zimo * 4 + 2])
    act = seed_i % len(ca_raw)
    return (
        f"game-{seed_i // 8}\t"
        + ",".join(map(str, sp_raw))
        + "\t"
        + ",".join(map(str, nu))
        + "\t"
        + ",".join(map(str, pr_raw))
        + "\t"
        + ",".join(map(str, ca_raw))
        + f"\t{act}\t0\t"
        + ",".join(["0"] * 14)
        + "\n"
    )


def _sample_rl_line(step_i: int, *, terminal: bool) -> str:
    """Valid source-v3 RL line matching EpisodeReplayBuffer (10-col transition / 8-col terminal)."""
    columns = _sample_bc_line(step_i).rstrip("\n").split("\t")
    if terminal:
        return "\t".join(columns) + "\n"
    next_cols = _sample_bc_line(step_i + 1).rstrip("\n").split("\t")[1:5]
    return "\t".join(columns[:6] + next_cols) + "\n"


def _ddp_target_sync_worker(rank: int, world_size: int, store_uri: str, tmp_dir: str) -> None:
    os.environ["USE_LIBUV"] = "0"
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=store_uri,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        torch.manual_seed(100 + rank)
        mod = JointStructuredIQLModule(
            preset="v2_tiny",
            dimension=64,
            num_heads=4,
            dim_feedforward=128,
            n_d1_layers=1,
            n_d2_layers=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        # Deliberately desynchronize target heads at startup on rank 1
        if rank == 1:
            with torch.no_grad():
                for p in mod.q1_target_head.parameters():
                    p.fill_(9.0)
                for p in mod.q2_target_head.parameters():
                    p.fill_(-9.0)
        mod.sync_targets_ddp()

        # Verify Rank-0 broadcast unified targets
        t_flat = torch.cat([p.detach().flatten() for p in mod.q1_target_head.parameters()])
        ref_t = t_flat.clone()
        dist.broadcast(ref_t, src=0)
        torch.testing.assert_close(t_flat, ref_t, atol=0, rtol=0)

        # Wrap with DDP and run 2 training steps to build Polyak lag
        opt = torch.optim.AdamW(
            mod.parameter_groups(base_lr=1e-3, weight_decay=0.01, decoder1_lr_scale=0.25)
        )
        recipe = build_joint_iql_recipe(
            mod, base_lr=1e-3, weight_decay=0.01, decoder1_lr_scale=0.25
        )
        ddp = DistributedDataParallel(mod)

        closed = {
            1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1,
            21: 1, 22: 1, 23: 1, 31: 2, 32: 2,
        }
        sp = torch.tensor(
            [_make_sparse(seat=0, left=40, doras=(1,), closed37=closed, zimo=32)],
            dtype=torch.int32,
        )
        nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]], dtype=torch.int32)
        pr = torch.tensor([_pad_prog([_dahai(1, 35)])], dtype=torch.int32)
        ca = torch.tensor([_pad_cands([4, 44, 84, 130])], dtype=torch.int32)
        feats = derive_structured_v2_batch(sp, nu, pr, ca)
        nxt = TensorDict(
            {
                "sparse": sp, "numeric": nu, "progression": pr, "candidates": ca,
                **feats,
                "done": torch.tensor([False]),
                "reward": torch.tensor([0.25], dtype=torch.float32),
            },
            batch_size=[1],
        )
        batch = TensorDict(
            {
                "sparse": sp, "numeric": nu, "progression": pr, "candidates": ca,
                **feats,
                "action": torch.tensor([1], dtype=torch.int32),
                "next": nxt,
            },
            batch_size=[1],
        )
        ckpt_dir = Path(tmp_dir)
        train_joint_iql_batches(
            mod,
            [batch, batch],
            opt,
            recipe=recipe,
            device=torch.device("cpu"),
            ddp_wrapped=ddp,
        )
        ckpt = save_joint_iql_checkpoint(
            ckpt_dir, mod, opt, step=2, cursor=2, recipe=recipe
        )
        if rank == 0:
            # Confirm Polyak lag exists: target != online
            diff_lag = sum(
                (p_t - p_o).abs().sum().item()
                for p_t, p_o in zip(
                    mod.q1_target_head.parameters(), mod.q1_head.parameters()
                )
            )
            assert diff_lag > 1e-6
        dist.barrier()

        # Reload after corrupting both ranks' local target params -> must recover exact rank-0 target + lag
        ckpt = ckpt_dir / "joint-iql-2.pt"
        with torch.no_grad():
            for p in mod.q1_target_head.parameters():
                p.zero_()
        mod2 = JointStructuredIQLModule(
            preset="v2_tiny",
            dimension=64,
            num_heads=4,
            dim_feedforward=128,
            n_d1_layers=1,
            n_d2_layers=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        opt2 = torch.optim.AdamW(
            mod2.parameter_groups(base_lr=1e-3, weight_decay=0.01, decoder1_lr_scale=0.25)
        )
        load_joint_iql_checkpoint(ckpt, mod2, opt2, expected_recipe=recipe)
        loaded_lag = sum(
            (p_t - p_o).abs().sum().item()
            for p_t, p_o in zip(
                mod2.q1_target_head.parameters(), mod2.q1_head.parameters()
            )
        )
        assert loaded_lag > 1e-6
        t_chk = torch.cat([p.detach().flatten() for p in mod2.q1_target_head.parameters()])
        ref_chk = t_chk.clone()
        dist.broadcast(ref_chk, src=0)
        torch.testing.assert_close(t_chk, ref_chk, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


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


class TestStructuredV2Phase1B(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    # Test #8: Zero future / private wall leakage
    def test_8_zero_future_and_private_leakage(self):
        closed = {
            1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1,
            21: 1, 22: 1, 23: 1, 31: 2, 32: 2,
        }
        sp = torch.tensor(
            [_make_sparse(seat=0, left=42, doras=(1,), closed37=closed, zimo=32)],
            dtype=torch.int32,
        )
        nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]], dtype=torch.int32)
        pr_base = _pad_prog([_dahai(1, 9), _dahai(2, 19)])
        pr_future = list(pr_base)
        # Put arbitrary garbage events after STOP sentinel 2165
        pr_future[5] = _dahai(3, 5)
        pr_future[6] = _dahai(0, 15)
        ca = torch.tensor([_pad_cands([4, 44, 84, 130])], dtype=torch.int32)

        f1 = derive_structured_v2_batch(sp, nu, torch.tensor([pr_base], dtype=torch.int32), ca)
        f2 = derive_structured_v2_batch(sp, nu, torch.tensor([pr_future], dtype=torch.int32), ca)
        for k in f1:
            self.assertEqual(float((f1[k].float() - f2[k].float()).abs().max().item()), 0.0, k)

        # Now pad v2_events tensor explicitly with valid-range future events beyond event_len
        ev_tail = torch.zeros((1, 12, 10), dtype=torch.uint8)
        ev_tail[:, :, 0] = 3  # EV_DAHAI
        ev_tail[:, :, 1] = 2  # actor_rel in [0..4]
        ev_tail[:, :, 2] = 1  # source_rel in [0..4]
        ev_tail[:, :, 3] = 15  # tile37 in [0..37]
        ev_tail[:, :, 4:6] = 37
        ev_tail[:, :, 8] = 5
        ev_tail[:, :, 9] = 1
        # Equal-shape future-slot mutation -> bit-exact diff == 0.0; extra-length padding -> < 1e-6 (1-ULP SDPA tile rounding)
        ev_pad_a = torch.cat([f1["v2_events"], torch.zeros((1, 12, 10), dtype=torch.uint8)], dim=1)
        ev_pad_b = torch.cat([f1["v2_events"], ev_tail], dim=1)

        bb = StructuredDualDecoder(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=2, n_d2_layers=2, device=torch.device("cpu"), dtype=torch.float32,
        ).eval()
        pol = StructuredActionHead(
            input_dimension=64, head_type="factorized", device=torch.device("cpu"), dtype=torch.float32
        ).eval()
        val = StructuredStateHead(
            input_dimension=64, device=torch.device("cpu"), dtype=torch.float32
        ).eval()

        with torch.no_grad():
            enc1 = bb.forward_structured(
                f1["v2_tile_feats"], f1["v2_events"], f1["v2_event_len"],
                f1["v2_global_feats"], f1["v2_player_feats"],
                f1["v2_cand_ids"], f1["v2_cand_mask"], f1["v2_cand_mode"], f1["v2_cand_dyn"],
            )
            enc2a = bb.forward_structured(
                f1["v2_tile_feats"], ev_pad_a, f1["v2_event_len"],
                f1["v2_global_feats"], f1["v2_player_feats"],
                f1["v2_cand_ids"], f1["v2_cand_mask"], f1["v2_cand_mode"], f1["v2_cand_dyn"],
            )
            enc2b = bb.forward_structured(
                f1["v2_tile_feats"], ev_pad_b, f1["v2_event_len"],
                f1["v2_global_feats"], f1["v2_player_feats"],
                f1["v2_cand_ids"], f1["v2_cand_mask"], f1["v2_cand_mode"], f1["v2_cand_dyn"],
            )
            self.assertEqual(float((enc2a - enc2b).abs().max().item()), 0.0)
            self.assertEqual(float((pol(ca, enc2a) - pol(ca, enc2b)).abs().max().item()), 0.0)
            self.assertEqual(float((val(enc2a) - val(enc2b)).abs().max().item()), 0.0)
            torch.testing.assert_close(enc1, enc2b, atol=1e-6, rtol=1e-6)

    # Test #9: Candidate permutation equivariance & min global_id tie-break
    def test_9_candidate_permutation_and_min_global_id_tiebreak(self):
        torch.manual_seed(9)
        closed = {
            1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1,
            21: 1, 22: 1, 23: 1, 31: 2, 32: 2,
        }
        sp = torch.tensor(
            [_make_sparse(seat=0, left=42, doras=(1,), closed37=closed, zimo=32)],
            dtype=torch.int32,
        )
        nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]], dtype=torch.int32)
        pr = torch.tensor([_pad_prog([_dahai(1, 35)])], dtype=torch.int32)
        valid_cands = [4, 8, 44, 84, 124, 130]
        ca = torch.tensor([_pad_cands(valid_cands)], dtype=torch.int32)
        feats = derive_structured_v2_batch(sp, nu, pr, ca)

        bb = StructuredDualDecoder(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=2, n_d2_layers=2, device=torch.device("cpu"), dtype=torch.float32,
        ).eval()
        pol = StructuredActionHead(
            input_dimension=64, head_type="factorized", device=torch.device("cpu"), dtype=torch.float32
        ).eval()
        dq = DuelingQHead(
            input_dimension=64, device=torch.device("cpu"), dtype=torch.float32
        ).eval()
        qq = QuantileQHead(
            input_dimension=64, device=torch.device("cpu"), dtype=torch.float32
        ).eval()

        perm = torch.tensor([4, 1, 5, 0, 3, 2], dtype=torch.long)
        ca_p = ca.clone()
        ca_p[:, :6] = ca[:, perm]
        feats_p = {k: v.clone() for k, v in feats.items()}
        for k in ("v2_cand_ids", "v2_cand_mask", "v2_cand_mode", "v2_cand_dyn"):
            feats_p[k][:, :6] = feats[k][:, perm]

        with torch.no_grad():
            enc = bb.forward_structured(*feats.values())
            enc_p = bb.forward_structured(*feats_p.values())
            lg, lg_p = pol(ca, enc), pol(ca_p, enc_p)
            q1, q1_p = dq(ca, enc), dq(ca_p, enc_p)
            qr, qr_p = qq(ca, enc), qq(ca_p, enc_p)

            torch.testing.assert_close(lg_p[:, :6], lg[:, perm], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(q1_p[:, :6], q1[:, perm], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(qr_p[:, :6], qr[:, perm], atol=1e-5, rtol=1e-5)

            # Tie-break by minimum global_id under zero logits
            pol.zero_output()
            arg = DecodeConverter("argmax", tie_break_by_action_id=True)
            chosen_orig = int(arg(ca, pol(ca, enc))[0].item())
            chosen_perm = int(arg(ca_p, pol(ca_p, enc_p))[0].item())
            self.assertEqual(int(ca[0, chosen_orig].item()), min(valid_cands))
            self.assertEqual(int(ca_p[0, chosen_perm].item()), min(valid_cands))

    # Test #10: 6x suit permutation equivariance
    def test_10_suit_permutation_equivariance_6x(self):
        torch.manual_seed(10)
        # 10A. Raw vs structured permutation agreement on asymmetric mixed red-5 meld hand
        c_chi = CHI_ENCODE_MAP[(14, 16, 10)]  # 0p called with 4p,6p
        closed_asym = {1: 1, 2: 1, 3: 1, 11: 2, 22: 2, 24: 1, 0: 1, 31: 1}
        sp = torch.tensor(
            [_make_sparse(seat=0, left=38, doras=(1,), closed37=closed_asym, zimo=31)],
            dtype=torch.int32,
        )
        nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]], dtype=torch.int32)
        pr = torch.tensor([_pad_prog([_dahai(3, 10), 597 + c_chi])], dtype=torch.int32)
        ca = torch.tensor([_pad_cands([0, 4, 8, 44, 126])], dtype=torch.int32)
        orig_act = torch.tensor([2], dtype=torch.long)  # aid=8

        feats_orig = derive_structured_v2_batch(sp, nu, pr, ca)
        for p_idx in range(6):
            p_sp, p_nu, p_pr, p_ca, p_act = permute_raw_four_fields(
                sp, nu, pr, ca, p_idx, chosen_cand_idx=orig_act
            )
            feats_raw_p = derive_structured_v2_batch(p_sp, p_nu, p_pr, p_ca)
            feats_str_p = permute_structured_batch(
                {**feats_orig, "action": orig_act}, p_idx
            )
            self.assertEqual(int(p_ca[0, p_act[0]].item()), int(PERM_ACTION_ID[p_idx, 8]))
            self.assertEqual(int(feats_str_p["action"][0].item()), int(p_act[0].item()))
            torch.testing.assert_close(feats_raw_p["v2_tile_feats"], feats_str_p["v2_tile_feats"], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(feats_raw_p["v2_cand_dyn"], feats_str_p["v2_cand_dyn"], atol=1e-6, rtol=1e-6)
            self.assertTrue(torch.equal(feats_raw_p["v2_cand_ids"], feats_str_p["v2_cand_ids"]))
            # Verify sorted ascending with PAD at tail
            cids_list = feats_raw_p["v2_cand_ids"][0].tolist()
            self.assertEqual(cids_list, sorted(cids_list))

        # 10B. Exact 6x policy probability equivariance under suit-symmetric tile embedding init
        closed_sym = {1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1, 21: 1, 22: 1, 23: 1, 31: 2, 32: 2}
        sp_s = torch.tensor(
            [_make_sparse(seat=0, left=42, doras=(31,), closed37=closed_sym, zimo=32)],
            dtype=torch.int32,
        )
        pr_s = torch.tensor([_pad_prog([_dahai(1, 35)])], dtype=torch.int32)
        orig_cands = [4, 44, 84, 130]
        ca_s = torch.tensor([_pad_cands(orig_cands)], dtype=torch.int32)
        feats_s = derive_structured_v2_batch(sp_s, nu, pr_s, ca_s)

        bb = StructuredDualDecoder(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=2, n_d2_layers=2, device=torch.device("cpu"), dtype=torch.float32,
        ).eval()
        pol = StructuredActionHead(
            input_dimension=64, head_type="factorized", device=torch.device("cpu"), dtype=torch.float32
        ).eval()
        with torch.no_grad():
            # Equalize suit embeddings (m, p, s) to verify architecture symmetry
            w_s = bb.tile_emb.suit_emb.weight
            w_s[1].copy_(w_s[0])
            w_s[2].copy_(w_s[0])

            enc0 = bb.forward_structured(*feats_s.values())
            p0 = F.softmax(pol(ca_s, enc0)[:, :4], dim=-1)[0]
            prob_by_gid_0 = {int(ca_s[0, i].item()): float(p0[i].item()) for i in range(4)}

            for p_idx in range(6):
                p_sp, p_nu, p_pr, p_ca, _ = permute_raw_four_fields(sp_s, nu, pr_s, ca_s, p_idx)
                f_p = derive_structured_v2_batch(p_sp, p_nu, p_pr, p_ca)
                enc_p = bb.forward_structured(*f_p.values())
                pp = F.softmax(pol(p_ca, enc_p)[:, :4], dim=-1)[0]
                prob_by_gid_p = {int(p_ca[0, i].item()): float(pp[i].item()) for i in range(4)}
                for gid in orig_cands:
                    mapped_gid = int(PERM_ACTION_ID[p_idx, gid])
                    self.assertAlmostEqual(prob_by_gid_0[gid], prob_by_gid_p[mapped_gid], places=5)

    # Test #11: L_ev=1 / PAD(546) / K==1 safety + AMP/half FP32 heads + AdamW no-decay guard
    def test_11_padding_k1_safety_amp_half_and_adamw_decay(self):
        closed = {
            1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1,
            21: 1, 22: 1, 23: 1, 31: 2, 32: 2,
        }
        # Only 1 legal candidate (tsumogiri 32 -> 130), L_ev == 1 (BOS only on East-1 initial draw)
        sp = torch.tensor(
            [_make_sparse(seat=0, left=69, doras=(1,), closed37=closed, zimo=32)],
            dtype=torch.int32,
        )
        nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]], dtype=torch.int32)
        pr = torch.tensor([_pad_prog([])], dtype=torch.int32)
        ca = torch.tensor([_pad_cands([32 * 4 + 2])], dtype=torch.int32)  # K == 1!
        feats = derive_structured_v2_batch(sp, nu, pr, ca)
        self.assertEqual(int(feats["v2_event_len"][0].item()), 1)

        # 11A. TrajectoryBCModel with L_ev==1 and K==1 -> finite loss & 0 policy_head grad
        bc_m = TrajectoryBCModel(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=2, n_d2_layers=2, use_next_event_head=True,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        rounds = build_round_trajectories_from_raw(
            sp, nu, pr, ca, torch.tensor([0]), ["r0"]
        )
        rb = collate_round_batch(rounds)
        out_bc = bc_m(rb)
        self.assertTrue(torch.isfinite(out_bc["loss"]).item())
        out_bc["loss"].backward()
        for p in bc_m.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all().item())
        for p in bc_m.policy_head.parameters():
            if p.grad is not None:
                self.assertEqual(float(p.grad.abs().max().item()), 0.0)

        # 11B. JointStructuredIQLModule with QuantileQHead & K==1
        joint = JointStructuredIQLModule(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=2, n_d2_layers=2, use_cql_quantile_head=True,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        nxt = TensorDict(
            {
                "sparse": sp, "numeric": nu, "progression": pr, "candidates": ca,
                **feats, "done": torch.tensor([True]), "reward": torch.tensor([1.0]),
            },
            batch_size=[1],
        )
        td = TensorDict(
            {
                "sparse": sp, "numeric": nu, "progression": pr, "candidates": ca,
                **feats, "action": torch.tensor([0], dtype=torch.int32), "next": nxt,
            },
            batch_size=[1],
        )
        out_iql = joint(td)
        self.assertTrue(torch.isfinite(out_iql["loss"]).item())
        out_iql["loss"].backward()
        for p in joint.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all().item())
        for p in joint.policy_head.parameters():
            if p.grad is not None:
                self.assertEqual(float(p.grad.abs().max().item()), 0.0)

        # 11C. scalar_cql_loss(mask_k1=True) finite on K==1
        class _EncWrap(nn.Module):
            def forward(self, d):
                enc = joint.backbone.forward_structured(
                    d["v2_tile_feats"], d["v2_events"], d["v2_event_len"],
                    d["v2_global_feats"], d["v2_player_feats"],
                    d["v2_cand_ids"], d["v2_cand_mask"], d["v2_cand_mode"], d["v2_cand_dyn"],
                )
                d["action_value"] = joint.q1_head(d["candidates"], enc)
                return d

        loss_scql, _ = scalar_cql_loss(_EncWrap(), None, td, 0.5, target_kind="td", mask_k1=True)
        self.assertTrue(torch.isfinite(loss_scql).item())

        # 11D. AdamW parameter groups no-decay guard (§3.4)
        groups = adamw_parameter_groups(joint, 0.01)
        decayed_ids = {id(p) for p in groups[0]["params"]}
        no_decay_ids = {id(p) for p in groups[1]["params"]}
        d2 = joint.backbone.decoder2
        for p in (d2.val_token_emb, d2.null_tile_emb, d2.null_source_emb):
            self.assertIn(id(p), no_decay_ids)
            self.assertNotIn(id(p), decayed_ids)
        if d2.cand_dyn_router is not None:
            self.assertIn(id(d2.cand_dyn_router.biases), no_decay_ids)
        for mod in joint.modules():
            if isinstance(mod, nn.Embedding):
                for p in mod.parameters(recurse=False):
                    if p.requires_grad:
                        self.assertIn(id(p), no_decay_ids)

        # 11E. FP16 .to(half) + autocast safety -> heads emit float32, RoPE stays float32
        dev = preferred_test_device()
        j_half = copy.deepcopy(joint).to(device=dev, dtype=torch.half).eval()
        self.assertEqual(j_half.backbone.decoder1._rope_cos.dtype, torch.float32)
        self.assertEqual(j_half.backbone.decoder1._rope_sin.dtype, torch.float32)
        f_dev = {k: v.to(dev) for k, v in feats.items()}
        ca_dev = ca.to(dev)
        with torch.no_grad():
            enc_h = j_half.backbone.forward_structured(*f_dev.values())
            self.assertEqual(j_half.policy_head(ca_dev, enc_h).dtype, torch.float32)
            self.assertEqual(j_half.value_head(enc_h).dtype, torch.float32)
            self.assertEqual(j_half.q1_head(ca_dev, enc_h).dtype, torch.float32)
            self.assertEqual(j_half.quantile_q_head(ca_dev, enc_h).dtype, torch.float32)
        if dev.type == "cuda":
            j_bf = copy.deepcopy(joint).to(device=dev, dtype=torch.float32).eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                enc_b = j_bf.backbone.forward_structured(*f_dev.values())
                self.assertEqual(j_bf.policy_head(ca_dev, enc_b).dtype, torch.float32)
                self.assertEqual(j_bf.q1_head(ca_dev, enc_b).dtype, torch.float32)

    # Test #12: forward vs forward_round_trajectory vs forward_step_cached + KVCache reset
    def _verify_three_mode_rounds(self, rounds: list[Any]) -> None:
        for head_type in ("factorized", "flat546"):
            bb = StructuredDualDecoder(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=2, n_d2_layers=2, head_type=head_type,
                device=torch.device("cpu"), dtype=torch.float32,
            ).eval()
            hd = StructuredActionHead(
                input_dimension=64, head_type=head_type,
                device=torch.device("cpu"), dtype=torch.float32,
            ).eval()
            rb = collate_round_batch(rounds)
            with torch.no_grad():
                rt_out = bb.forward_round_trajectory(
                    rb["round_events"], rb["round_event_len"], rb["decision_event_len"],
                    rb["decision_mask"], rb["v2_tile_feats"], rb["v2_global_feats"],
                    rb["v2_player_feats"], rb["v2_cand_ids"], rb["v2_cand_mask"],
                    rb["v2_cand_mode"], rb["v2_cand_dyn"],
                )
                enc_rt = rt_out["encode"]
                dmask = rb["decision_mask"].bool()
                cands_rt = rb["candidates"][dmask]
                logits_rt = hd(cands_rt, enc_rt)

                # Step-by-step forward + forward_step_cached
                flat_idx = 0
                cache = StructuredKVCache()
                for ri, r in enumerate(rounds):
                    ctx = DecisionContext(environment="test", game=0, seat=0, round_instance=ri, model_version=1)
                    t_len = int(r.decision_event_len.shape[0])
                    for t in range(t_len):
                        ev_len_t = int(r.decision_event_len[t].item())
                        ev_t = r.round_events[:ev_len_t].unsqueeze(0)
                        ev_len_ten = torch.tensor([ev_len_t], dtype=torch.int32)
                        tf_t = r.v2_tile_feats[t : t + 1]
                        gf_t = r.v2_global_feats[t : t + 1]
                        pf_t = r.v2_player_feats[t : t + 1]
                        cids_t = r.v2_cand_ids[t : t + 1]
                        cmask_t = r.v2_cand_mask[t : t + 1]
                        cmode_t = r.v2_cand_mode[t : t + 1]
                        cdyn_t = r.v2_cand_dyn[t : t + 1]
                        ca_t = r.candidates[t : t + 1]

                        enc_single = bb.forward_structured(
                            tf_t, ev_t, ev_len_ten, gf_t, pf_t, cids_t, cmask_t, cmode_t, cdyn_t
                        )
                        enc_cached = bb.forward_step_cached(
                            context=ctx, cache=cache,
                            tile_feats=tf_t, events=ev_t, event_len=ev_len_ten,
                            global_feats=gf_t, player_feats=pf_t,
                            cand_ids=cids_t, cand_mask=cmask_t,
                            cand_mode=cmode_t, cand_dyn=cdyn_t,
                        )
                        torch.testing.assert_close(enc_single, enc_rt[flat_idx : flat_idx + 1], atol=1e-5, rtol=1e-5)
                        torch.testing.assert_close(enc_cached, enc_single, atol=1e-5, rtol=1e-5)
                        torch.testing.assert_close(
                            hd(ca_t, enc_cached), logits_rt[flat_idx : flat_idx + 1], atol=1e-5, rtol=1e-5
                        )
                        flat_idx += 1

                # KVCache auto-reset on prefix mutation & model_version bump & cached_structured_decision
                r0 = rounds[0]
                ctx_v1 = DecisionContext(environment="test", game=0, seat=0, round_instance=99, model_version=1)
                ev0 = r0.round_events[: int(r0.decision_event_len[0].item())].unsqueeze(0)
                evl0 = torch.tensor([int(r0.decision_event_len[0].item())], dtype=torch.int32)
                ref0 = bb.forward_step_cached(
                    context=ctx_v1, cache=cache,
                    tile_feats=r0.v2_tile_feats[:1], events=ev0, event_len=evl0,
                    global_feats=r0.v2_global_feats[:1], player_feats=r0.v2_player_feats[:1],
                    cand_ids=r0.v2_cand_ids[:1], cand_mask=r0.v2_cand_mask[:1],
                    cand_mode=r0.v2_cand_mode[:1], cand_dyn=r0.v2_cand_dyn[:1],
                )
                ev_mut = ev0.clone()
                ev_mut[0, 0, 3] = (int(ev_mut[0, 0, 3].item()) + 1) % 37
                enc_mut_cached = bb.forward_step_cached(
                    context=ctx_v1, cache=cache,
                    tile_feats=r0.v2_tile_feats[:1], events=ev_mut, event_len=evl0,
                    global_feats=r0.v2_global_feats[:1], player_feats=r0.v2_player_feats[:1],
                    cand_ids=r0.v2_cand_ids[:1], cand_mask=r0.v2_cand_mask[:1],
                    cand_mode=r0.v2_cand_mode[:1], cand_dyn=r0.v2_cand_dyn[:1],
                )
                enc_mut_fresh = bb.forward_structured(
                    r0.v2_tile_feats[:1], ev_mut, evl0,
                    r0.v2_global_feats[:1], r0.v2_player_feats[:1],
                    r0.v2_cand_ids[:1], r0.v2_cand_mask[:1], r0.v2_cand_mode[:1], r0.v2_cand_dyn[:1],
                )
                torch.testing.assert_close(enc_mut_cached, enc_mut_fresh, atol=1e-5, rtol=1e-5)

                closed = {
                    1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1,
                    21: 1, 22: 1, 23: 1, 31: 2, 32: 2,
                }
                sp1 = torch.tensor([_make_sparse(seat=0, left=42, doras=(1,), closed37=closed, zimo=32)], dtype=torch.int32)
                nu1 = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]], dtype=torch.int32)
                pr1 = torch.tensor([_pad_prog([_dahai(1, 35)])], dtype=torch.int32)
                ca1 = torch.tensor([_pad_cands([4, 44, 84, 130])], dtype=torch.int32)
                ctx_v2 = DecisionContext(environment="test", game=0, seat=0, round_instance=99, model_version=2)
                lg_cache_api = cached_structured_decision(bb, hd, ctx_v2, cache, sp1, nu1, pr1, ca1)
                lg_stateless = hd(ca1, bb(sp1, nu1, pr1, ca1))
                torch.testing.assert_close(lg_cache_api, lg_stateless, atol=1e-5, rtol=1e-5)
                _ = ref0

    def test_12_three_mode_forward_equivalence_and_kvcache_reset(self):
        closed = {
            1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1,
            21: 1, 22: 1, 23: 1, 31: 2, 32: 2,
        }
        sp = torch.tensor(
            [_make_sparse(seat=0, left=42 - k * 4, doras=(1,), closed37=closed, zimo=32) for k in range(3)],
            dtype=torch.int32,
        )
        nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]] * 3, dtype=torch.int32)
        pr = torch.tensor(
            [
                _pad_prog([_dahai(1, 35)]),
                _pad_prog([_dahai(1, 35), _dahai(2, 36), _dahai(3, 35)]),
                _pad_prog([_dahai(1, 35), _dahai(2, 36), _dahai(3, 35), _dahai(0, 31), _dahai(1, 36)]),
            ],
            dtype=torch.int32,
        )
        ca = torch.tensor([_pad_cands([4, 44, 84, 130])] * 3, dtype=torch.int32)
        rounds = build_round_trajectories_from_raw(sp, nu, pr, ca, torch.tensor([0, 1, 2]), ["r0"] * 3)
        self._verify_three_mode_rounds(rounds)

    def test_12b_three_mode_forward_equivalence_real_data(self):
        val_bc = require_or_skip_real_data(
            self, "bc", context="three-mode forward equivalence on real trajectories"
        )
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
        self.assertTrue(complete_lines)
        with tempfile.TemporaryDirectory() as tmp_val:
            val_sub = Path(tmp_val) / "val_complete.tsv"
            val_sub.write_text("".join(complete_lines), encoding="utf-8")
            rounds = [
                r for r in parse_bc_rounds(val_sub)
                if len(r.decision_event_len) >= 3
            ][:6]
        self.assertTrue(rounds)
        self._verify_three_mode_rounds(rounds)

    # Test #13: Native kanachan.simulation.simulate + C++ DecisionMaker step parity
    def test_13_simulate_end_to_end_cpp_decision_maker(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "v2_tiny.kanachan"
            bb = StructuredDualDecoder(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=2, n_d2_layers=1, device=torch.device("cpu"), dtype=torch.float32,
            )
            hd = StructuredActionHead(
                input_dimension=64, head_type="factorized",
                device=torch.device("cpu"), dtype=torch.float32,
            )
            _export_actor_file(bb, hd, model_path, torch.float32)

            loaded = load_model(model_path, map_location=torch.device("cpu")).eval()
            recorded_batches: list[tuple[dict[str, torch.Tensor], torch.Tensor]] = []

            class _SpyModel(nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner

                def forward(self, data):
                    snap = {
                        k: data[k].detach().cpu().clone()
                        for k in ("sparse", "numeric", "progression", "candidates")
                    }
                    out = self.inner(data)
                    recorded_batches.append((snap, data["action"].detach().cpu().clone()))
                    return out

            spy = _SpyModel(loaded).eval()
            with torch.inference_mode():
                logs_spy = simulate(
                    torch.device("cpu"), torch.float32, 3, 12, spy, 12, spy,
                    1, 2, 2, 4, lambda: None, seed=777,
                )
            self.assertEqual(len(logs_spy), 2)
            self.assertGreater(len(recorded_batches), 20)
            for snap, cpp_action in recorded_batches:
                td = TensorDict(snap, batch_size=[snap["sparse"].shape[0]])
                loaded(td)
                self.assertTrue(torch.equal(td["action"], cpp_action))

            # Verify raw un-wrapped loaded model also runs directly inside C++ DecisionMaker
            sim_dev = preferred_test_device()
            sim_dtype = torch.half if sim_dev.type == "cuda" else torch.float32
            loaded_dev = load_model(model_path, map_location="cpu").to(
                device=sim_dev, dtype=sim_dtype
            ).eval()
            with torch.inference_mode():
                logs_raw = simulate(
                    sim_dev, sim_dtype, 3, 12, loaded_dev, 12, loaded_dev,
                    1, 1, 1, 2, lambda: None, seed=888,
                )
            self.assertEqual(len(logs_raw), 1)
            self.assertEqual(len(logs_raw[0].get_result()), 4)

    # Test #14: 20-dim PureStateGRP cyclic equivariance, South-1 vs South-4, & PBRS telescoping
    def test_14_grp_equivariance_south1_vs_south4_and_pbrs_telescoping(self):
        torch.manual_seed(14)
        grp = PureStateGRP()
        base_scores = [34000, 26000, 18000, 22000]
        sp0 = torch.tensor([_make_sparse(seat=0, chang=1, ju=2)], dtype=torch.int32)
        nu0 = torch.tensor([[2, 1] + base_scores], dtype=torch.int32)
        v0 = canonicalize_grp_state(sp0, nu0)
        p0, phi0 = grp.predict_probs_and_potential(sp0, nu0)

        for shift in (1, 2, 3):
            # Permuting starting seat assignment by +shift shifts seat, ju, and scores uniformly:
            # rel_scores [0:4], oya_rel [4:8], and non-round features [8, 13:20] must stay invariant < 1e-6,
            # while round index one-hot [9:13] cycles cleanly as ju % 4 -> (ju + shift) % 4.
            rot_scores = [base_scores[(p - shift) % 4] for p in range(4)]
            sp_r = torch.tensor([_make_sparse(seat=shift, chang=1, ju=(2 + shift) % 4)], dtype=torch.int32)
            nu_r = torch.tensor([[2, 1] + rot_scores], dtype=torch.int32)
            vr = canonicalize_grp_state(sp_r, nu_r)
            torch.testing.assert_close(v0[:, :9], vr[:, :9], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(v0[:, 13:], vr[:, 13:], atol=1e-6, rtol=1e-6)
            expected_round = F.one_hot(torch.tensor([(2 + shift) % 4]), num_classes=4).float()
            torch.testing.assert_close(vr[:, 9:13], expected_round, atol=1e-6, rtol=1e-6)
        # Also verify exact invariance (including probs & potential) at fixed round ju under score/seat relabeling
        # by verifying PureStateGRP output invariance across 4 cyclic observers whose rotated score/dealer vecs match
        v_rot_only = v0.clone()
        pr_direct = torch.softmax(grp(v_rot_only), dim=-1)
        torch.testing.assert_close(p0, pr_direct, atol=1e-6, rtol=1e-6)

        # South-1 vs South-4 (All-Last) distinction via ju % 4 one-hot [9:13]
        sp_s1 = torch.tensor([_make_sparse(seat=0, chang=1, ju=0)], dtype=torch.int32)
        sp_s4 = torch.tensor([_make_sparse(seat=0, chang=1, ju=3)], dtype=torch.int32)
        f_s1 = canonicalize_grp_state(sp_s1, nu0)
        f_s4 = canonicalize_grp_state(sp_s4, nu0)
        self.assertGreater(float((f_s1 - f_s4).abs().sum().item()), 1.0)
        _, phi_s1 = grp.predict_probs_and_potential(sp_s1, nu0)
        _, phi_s4 = grp.predict_probs_and_potential(sp_s4, nu0)
        self.assertNotAlmostEqual(float(phi_s1.item()), float(phi_s4.item()), places=5)

        # Calibration gate + FrozenGRP
        good_m = {"top1_ece": 0.01, "classwise_ece": 0.015, "brier": 0.55, "nll": 1.1, "accuracy": 0.5}
        bad_m = {"top1_ece": 0.05, "classwise_ece": 0.01, "brier": 0.55, "nll": 1.1, "accuracy": 0.5}
        self.assertTrue(passes_gate(good_m))
        self.assertFalse(passes_gate(bad_m))
        with self.assertRaises(ValueError):
            FrozenGRP(grp, bad_m, require_gate=True)
        fg = FrozenGRP(grp, good_m)
        self.assertTrue(all(not p.requires_grad for p in fg.parameters()))

        # PBRS telescoping across skipped zero-decision rounds
        r_phis = [0.25, 0.25, -0.40, -0.40, 0.85, 0.85, 1.10]
        r_ids = [(0, 0, 0), (0, 0, 0), (0, 2, 0), (0, 2, 1), (1, 0, 0), (1, 0, 0), (1, 3, 0)]
        r_term = 1.5
        rewards, dones = shape_hanchan_pbrs_rewards(r_phis, r_ids, r_term)
        self.assertEqual(dones, [False] * 6 + [True])
        self.assertAlmostEqual(sum(rewards), r_term - r_phis[0], places=6)

        # Verify well-calibrated logits pass gate
        m_comp = compute_grp_calibration_metrics(
            torch.tensor([[6.0, 0.0, 0.0, 0.0]] * 8),
            torch.zeros(8, dtype=torch.long),
        )
        self.assertTrue(passes_gate(m_comp))

    # Test #15: Stage-2 IQL grad isolation (decoder1_lr_scale, coupled freeze) & DDP Target sync
    def test_15_stage2_iql_grad_isolation_and_ddp_target_sync(self):
        joint = JointStructuredIQLModule(
            preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
            n_d1_layers=2, n_d2_layers=1, device=torch.device("cpu"), dtype=torch.float32,
        )
        groups = joint.parameter_groups(base_lr=1e-3, weight_decay=0.01, decoder1_lr_scale=0.25)
        lrs_by_pid = {id(p): g["lr"] for g in groups for p in g["params"]}
        tile_ids = {id(p) for p in joint.backbone.tile_emb.parameters()}
        for p in joint.backbone.decoder1.parameters():
            if id(p) not in tile_ids:
                self.assertAlmostEqual(lrs_by_pid[id(p)], 2.5e-4, places=9)
        for p in joint.backbone.tile_emb.parameters():
            self.assertAlmostEqual(lrs_by_pid[id(p)], 1e-3, places=9)

        # Coupled freeze_decoder1 -> auto-freezes shared tile_emb while keeping decoder2 non-shared params trainable
        joint.configure_freeze(freeze_decoder1=True)
        self.assertTrue(all(not p.requires_grad for p in joint.backbone.tile_emb.parameters()))
        self.assertTrue(all(not p.requires_grad for p in joint.backbone.decoder1.parameters()))
        self.assertTrue(
            all(
                p.requires_grad
                for p in joint.backbone.decoder2.parameters()
                if id(p) not in tile_ids
            )
        )

        # 2-rank DDP target broadcast + Polyak lag preservation via clean subprocess env
        with tempfile.TemporaryDirectory() as tmp:
            store_uri = (Path(tmp) / "ddp_store").as_uri()
            ckpt_dir = str(Path(tmp) / "ckpt")
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            env_ddp = dict(
                os.environ,
                USE_LIBUV="0",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                PYTHONPATH=f"{REPO_ROOT}{os.pathsep}{REPO_ROOT / 'test' / 'training'}",
            )
            ifname = _resolve_windows_gloo_ifname()
            if ifname:
                env_ddp["GLOO_SOCKET_IFNAME"] = ifname
            procs = [
                subprocess.Popen(
                    [sys.executable, __file__, "--ddp-worker", str(r), store_uri, ckpt_dir],
                    cwd=REPO_ROOT,
                    env=env_ddp,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for r in range(2)
            ]
            for r, proc in enumerate(procs):
                out, err = proc.communicate(timeout=60)
                self.assertEqual(proc.returncode, 0, f"Rank {r} failed:\n{out}\n{err}")

    # Test #16: Exact resume parity across TrajectoryBC / CLI BC / CLI MC-CQL / Joint IQL + Fail-Closed
    def test_16_exact_resume_bc_traj_mccql_jointiql_and_fail_closed(self):
        # 16A. TrajectoryBCModel with NextEventHead & 6x suit augmentation + 9-family eval
        torch.manual_seed(1601)
        closed = {
            1: 1, 2: 1, 3: 1, 11: 1, 12: 1, 13: 1,
            21: 1, 22: 1, 23: 1, 31: 2, 32: 2,
        }
        sp = torch.tensor(
            [_make_sparse(seat=k % 4, left=48 - k, doras=(1,), closed37=closed, zimo=32) for k in range(8)],
            dtype=torch.int32,
        )
        nu = torch.tensor([[0, 0, 25000, 25000, 25000, 25000]] * 8, dtype=torch.int32)
        pr = torch.tensor([_pad_prog([_dahai(1, 35)] * ((k % 2) + 1)) for k in range(8)], dtype=torch.int32)
        ca = torch.tensor([_pad_cands([4, 44, 84, 130])] * 8, dtype=torch.int32)
        acts = torch.tensor([k % 4 for k in range(8)], dtype=torch.long)
        rounds = build_round_trajectories_from_raw(sp, nu, pr, ca, acts, [f"r{k // 2}" for k in range(8)])

        def _make_traj():
            torch.manual_seed(1602)
            m = TrajectoryBCModel(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=2, n_d2_layers=1, use_next_event_head=True,
                device=torch.device("cpu"), dtype=torch.float32,
            )
            opt = torch.optim.AdamW(adamw_parameter_groups(m, 0.01), lr=1e-3)
            return m, opt

        m_cont, opt_cont = _make_traj()
        train_trajectory_bc_steps(
            m_cont, rounds, opt_cont, batch_size=2, max_steps=4,
            device=torch.device("cpu"), suit_augmentation=True, aug_seed=55, sampler_seed=11,
        )

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "traj"
            m_half, opt_half = _make_traj()
            train_trajectory_bc_steps(
                m_half, rounds, opt_half, batch_size=2, max_steps=2,
                device=torch.device("cpu"), suit_augmentation=True, aug_seed=55, sampler_seed=11,
                ckpt_dir=ckpt_dir, checkpoint_every=2,
            )
            ckpt_file = ckpt_dir / "traj-bc-2.pt"
            m_res, opt_res = _make_traj()
            train_trajectory_bc_steps(
                m_res, rounds, opt_res, batch_size=2, max_steps=4,
                device=torch.device("cpu"), suit_augmentation=True, aug_seed=55, sampler_seed=11,
                resume_from=ckpt_file,
            )
            for k, v in m_cont.state_dict().items():
                torch.testing.assert_close(v, m_res.state_dict()[k], atol=0, rtol=0)

            # Fail-Closed check on tampered manifest_sha256
            bad_recipe = m_res.recipe(lr=1e-3, weight_decay=0.01)
            bad_recipe["manifest_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                load_trajectory_bc_checkpoint(
                    ckpt_file, m_res, opt_res, expected_recipe=bad_recipe
                )

            # 9-family evaluation smoke check
            eval_res = evaluate_9_action_families(
                m_cont.backbone, m_cont.policy_head,
                [collate_round_batch(rounds[:2])], device=torch.device("cpu"),
            )
            self.assertIn("DISCARD_DAMA", eval_res)
            self.assertTrue(set(ACTION_FAMILY_NAMES).issubset(eval_res.keys()))
            self.assertIn("overall", eval_res)

        # 16B. JointStructuredIQLModule exact resume with FrozenGRP + Fail-Closed check
        grp = PureStateGRP()
        fgrp = FrozenGRP(
            grp, {"top1_ece": 0.01, "classwise_ece": 0.01, "brier": 0.5, "nll": 1.0, "accuracy": 0.6}
        )
        feats_full = derive_structured_v2_batch(sp[:4], nu[:4], pr[:4], ca[:4])
        td_batches = []
        for bi in range(4):
            sl = slice(bi, bi + 1)
            f_b = {k: v[sl] for k, v in feats_full.items()}
            nxt = TensorDict(
                {
                    "sparse": sp[sl], "numeric": nu[sl], "progression": pr[sl], "candidates": ca[sl],
                    **f_b, "done": torch.tensor([bi == 3]), "reward": torch.tensor([0.2 * bi]),
                },
                batch_size=[1],
            )
            td_batches.append(
                TensorDict(
                    {
                        "sparse": sp[sl], "numeric": nu[sl], "progression": pr[sl], "candidates": ca[sl],
                        **f_b, "action": acts[sl].to(torch.int32), "next": nxt,
                    },
                    batch_size=[1],
                )
            )

        def _make_iql():
            torch.manual_seed(1603)
            jm = JointStructuredIQLModule(
                preset="v2_tiny", dimension=64, num_heads=4, dim_feedforward=128,
                n_d1_layers=1, n_d2_layers=1, device=torch.device("cpu"), dtype=torch.float32,
            )
            jo = torch.optim.AdamW(
                jm.parameter_groups(base_lr=5e-4, weight_decay=0.01, decoder1_lr_scale=0.25)
            )
            rec = build_joint_iql_recipe(
                jm, base_lr=5e-4, weight_decay=0.01, decoder1_lr_scale=0.25, grp=fgrp
            )
            return jm, jo, rec

        jm_c, jo_c, rec_c = _make_iql()
        train_joint_iql_batches(jm_c, td_batches, jo_c, recipe=rec_c, device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            jdir = Path(tmp) / "jiql"
            jm_h, jo_h, rec_h = _make_iql()
            train_joint_iql_batches(
                jm_h, td_batches, jo_h, recipe=rec_h, device=torch.device("cpu"),
                ckpt_dir=jdir, checkpoint_every=2,
            )
            jm_r, jo_r, rec_r = _make_iql()
            train_joint_iql_batches(
                jm_r, td_batches, jo_r, recipe=rec_r, device=torch.device("cpu"),
                resume_from=jdir / "joint-iql-2.pt",
            )
            for k, v in jm_c.state_dict().items():
                torch.testing.assert_close(v, jm_r.state_dict()[k], atol=0, rtol=0)

            bad_rec = dict(rec_r)
            bad_rec["grp_sha256"] = "bad" * 20
            with self.assertRaises(ValueError):
                load_joint_iql_checkpoint(jdir / "joint-iql-2.pt", jm_r, jo_r, expected_recipe=bad_rec)

        # 16C. CLI BC (v2_tiny + CandidateFeaturePrefetch) & CLI MC-CQL (streaming_mc_structured_v2) exact resume
        env = dict(
            os.environ, WORLD_SIZE="1", RANK="0", LOCAL_RANK="0",
            TORCHDYNAMO_DISABLE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
            HYDRA_FULL_ERROR="1", PYTHONDONTWRITEBYTECODE="1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bc_data, rl_data = root / "bc.tsv", root / "rl.tsv"
            bc_data.write_text("".join(_sample_bc_line(i) for i in range(8)), encoding="utf-8")
            rl_data.write_text("".join(_sample_rl_line(i, terminal=(i % 4 == 3)) for i in range(8)), encoding="utf-8")

            common_enc = [
                "--config-name", "decision", "device=cpu",
                "encoder=v2_tiny", "encoder.dimension=64", "encoder.num_heads=4",
                "encoder.dim_feedforward=128", "encoder.n_d1_layers=1", "encoder.n_d2_layers=1",
                "num_workers=0", "batch_size=2",
            ]
            # BC continuous vs resume at 4
            for name, extra in (
                ("bc_cont", []),
                ("bc_res", [f"initial_model_prefix={(root / 'bc_cont/snapshots').as_posix()}", "initial_model_index=4"]),
            ):
                res = subprocess.run(
                    [
                        sys.executable, "-B", "-m", "kanachan.training.bc.train",
                        *common_enc, "candidate_feature_workers=1", "candidate_prefetch_batches=2",
                        "snapshot_interval=4",
                        f"training_data={bc_data.as_posix()}",
                        f"hydra.run.dir={(root / name).as_posix()}",
                        *extra,
                    ],
                    cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180,
                )
                self.assertEqual(res.returncode, 0, res.stdout + res.stderr)

            ckpt_bc_meta = json.loads((root / "bc_cont/snapshots/decision-checkpoint.json").read_text())
            self.assertEqual(ckpt_bc_meta["recipe"]["architecture"], "mahjong_structured_dual_decoder_v2")
            for fname in ("encoder.pth", "decoder.pth"):
                left = torch.load(root / "bc_cont/snapshots" / fname, weights_only=True)
                right = torch.load(root / "bc_res/snapshots" / fname, weights_only=True)
                for k in left:
                    torch.testing.assert_close(left[k], right[k], atol=0, rtol=0)

            # MC-CQL continuous vs resume at 4
            cql_common = [
                *common_enc,
                "target_kind=mc", "double_q_learning=false", "streaming_mc=true",
                "streaming_mc_batches=2", "replay_buffer_size=4", "snapshot_interval=4",
                f"bc_initial_model={(root / 'bc_cont/snapshots/model.kanachan').as_posix()}",
            ]
            res_c1 = subprocess.run(
                [
                    sys.executable, "-B", "-m", "kanachan.training.cql.train",
                    *cql_common, f"training_data={rl_data.as_posix()}",
                    f"hydra.run.dir={(root / 'cql_cont').as_posix()}",
                ],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180,
            )
            self.assertEqual(res_c1.returncode, 0, res_c1.stdout + res_c1.stderr)
            res_c2 = subprocess.run(
                [
                    sys.executable, "-B", "-m", "kanachan.training.cql.train",
                    *cql_common, f"training_data={rl_data.as_posix()}",
                    f"hydra.run.dir={(root / 'cql_res').as_posix()}",
                    f"initial_model_prefix={(root / 'cql_cont/snapshots').as_posix()}",
                    "initial_model_index=4",
                ],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180,
            )
            self.assertEqual(res_c2.returncode, 0, res_c2.stdout + res_c2.stderr)

            ckpt_cql_meta = json.loads((root / "cql_cont/snapshots/decision-checkpoint.json").read_text())
            self.assertEqual(ckpt_cql_meta["recipe"]["streaming_mc"]["version"], "streaming_mc_structured_v2")
            for fname in ("encoder.pth", "decoder.pth", "optimizer.pth", "replay-state.pth"):
                left = torch.load(root / "cql_cont/snapshots" / fname, weights_only=True)
                right = torch.load(root / "cql_res/snapshots" / fname, weights_only=True)
                self.assertEqual(repr(left.keys()), repr(right.keys()))
            left_enc = torch.load(root / "cql_cont/snapshots/encoder.pth", weights_only=True)
            right_enc = torch.load(root / "cql_res/snapshots/encoder.pth", weights_only=True)
            for k in left_enc:
                torch.testing.assert_close(left_enc[k], right_enc[k], atol=0, rtol=0)


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--ddp-worker":
        _ddp_target_sync_worker(int(sys.argv[2]), 2, sys.argv[3], sys.argv[4])
    else:
        unittest.main()
