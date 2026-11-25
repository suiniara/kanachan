#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "structured_v2.hpp"

#include <nyanten/replacement_number.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace Kanachan::StructuredV2 {
namespace {

constexpr float LN61 = 4.110873864173311f;
constexpr float LN64001 = 11.066653989166727f;

inline int calc_shanten_raw(std::array<int, 34> const& h34) {
  return static_cast<int>(Nyanten::calculateReplacementNumber(h34)) - 1;
}

inline std::vector<int> calc_waits_34(
    std::array<int, 34> const& h_closed34,
    std::array<int, 34> const& m_self34) {
  std::vector<int> waits;
  auto tmp = h_closed34;
  for (int w = 0; w < 34; ++w) {
    if (tmp[w] + m_self34[w] >= 4) continue;
    ++tmp[w];
    if (Nyanten::calculateReplacementNumber(tmp) == 0u) {
      waits.push_back(w);
    }
    --tmp[w];
  }
  return waits;
}

inline void calc_effective_3k1(
    std::array<int, 34> const& h_closed34,
    std::array<int, 34> const& m_self34,
    int& out_shanten,
    std::vector<int>& out_waits,
    std::vector<int>& out_eff) {
  out_waits.clear();
  out_eff.clear();
  auto tmp = h_closed34;
  std::uint_fast8_t const raw_rep = Nyanten::calculateReplacementNumber(tmp);
  int const raw_s = static_cast<int>(raw_rep) - 1;
  out_waits = calc_waits_34(tmp, m_self34);

  if (raw_s == 0 && out_waits.empty()) {
    // Pure fake 5th-tile tenpai: not tenpai in Mahjong Soul / Kanachan rules.
    out_shanten = 1;
    return;
  }
  out_shanten = raw_s;
  if (raw_s == 0) {
    out_eff = out_waits;
    return;
  }
  for (int t = 0; t < 34; ++t) {
    if (tmp[t] + m_self34[t] >= 4) continue;
    ++tmp[t];
    if (Nyanten::calculateReplacementNumber(tmp) < raw_rep) {
      out_eff.push_back(t);
    }
    --tmp[t];
  }
}

struct ParsedEvent {
  std::uint8_t type = 0; // 3=DAHAI, 5=CHI, 6=PON, 7=DMK, 8=ANKAN, 9=KAKAN
  int seat = 0;
  int src_rel = NULL_SOURCE_IDX;
  int t37 = TILE37_NONE;
  int c0 = TILE37_NONE;
  int c1 = TILE37_NONE;
  int code = 0;
  int b34 = 0;
  int moqie = 0;
  int liqi = 0;
};

struct DiscardEvalInfo {
  bool legal = false;
  int s_post = 8;
  int u_post = 0;
  bool post_tenpai = false;
  bool post_furiten = false;
};

inline bool has_dama_yaku(
    std::array<int, 34> const& after_closed34,
    std::vector<MjsoulEval::MeldSpec> const& melds,
    std::vector<int> const& waits,
    int const player_wind,
    int const round_wind) {
  MjsoulEval::EvalConfig cfg;
  cfg.player_wind = player_wind;
  cfg.round_wind = round_wind;
  auto test34 = after_closed34;
  for (int const w : waits) {
    ++test34[w];
    auto const res = MjsoulEval::eval_mjsoul_hand(test34, melds, w, 0, 0, cfg);
    --test34[w];
    if (res.valid) return true;
  }
  return false;
}

} // namespace

void encode_single_sample(
    int32_t const* sparse,
    int32_t const* numeric,
    int32_t const* progression,
    int const prog_stride,
    int32_t const* candidates,
    bool const enable_early_shape_hint,
    std::array<float, 4> const& reward_weights,
    EncodedSample& out) {
  out.tile_feats.fill(0.f);
  out.global_feats.fill(0.f);
  out.player_feats.fill(0.f);
  out.cand_mode.fill(0);
  out.cand_dyn.fill(0.f);

  bool const is_tonpuusen = (sparse[1] == 5);
  int const seat = sparse[6] - 71;
  int const chang = sparse[7] - 75;
  int const ju = sparse[8] - 78;
  int const left_tile_count = sparse[9] - 82;
  if (seat < 0 || seat >= 4 || chang < 0 || chang >= 3 || ju < 0 || ju >= 4 ||
      left_tile_count < 0 || left_tile_count > 69) {
    throw std::invalid_argument("Invalid sparse header values.");
  }

  std::array<int, 5> dora_indicators{};
  int n_dora_ind = 0;
  std::array<int, 37> hand37{};
  std::array<int, 37> public37{};
  std::array<int, 34> dora34{};
  int zimo_tile37 = TILE37_NONE;

  for (int idx = 10; idx < 29; ++idx) {
    int32_t const v = sparse[idx];
    if (v == 510) break;
    if (v >= 152 && v < 337) {
      if (n_dora_ind >= 5) throw std::invalid_argument("Too many dora indicators.");
      int const t37 = (v - 152) % 37;
      dora_indicators[n_dora_ind++] = t37;
      ++public37[t37];
      ++dora34[next_dora_base34(tile34(t37))];
    } else if (v >= 337 && v < 473) {
      int const t37 = idx136_to_tile37(v - 337);
      ++hand37[t37];
    } else if (v >= 473 && v < 510) {
      if (zimo_tile37 != TILE37_NONE) throw std::invalid_argument("Multiple zimo tiles in sparse.");
      zimo_tile37 = v - 473;
      ++hand37[zimo_tile37];
    } else {
      throw std::invalid_argument("Invalid sparse token in tail.");
    }
  }
  if (n_dora_ind == 0) {
    throw std::invalid_argument("Expected at least 1 dora indicator.");
  }

  int lp = 0;
  while (lp < prog_stride && progression[lp] != 2165) {
    ++lp;
  }
  if (lp < 1 || lp > MAX_PROGRESSION_LEN || progression[0] != 0) {
    throw std::invalid_argument("Invalid progression length or missing BOS.");
  }

  std::array<int, 30> valid_cands{};
  int n_cands = 0;
  bool seen_pad = false;
  for (int i = 0; i < MAX_CANDIDATES; ++i) {
    int32_t const c = candidates[i];
    if (c == PAD_ACTION_ID) {
      seen_pad = true;
    } else {
      if (seen_pad || c < 0 || c >= NUM_TYPES_OF_ACTIONS) {
        throw std::invalid_argument("Malformed or unsorted PAD in candidates.");
      }
      if (n_cands > 0 && c <= valid_cands[n_cands - 1]) {
        throw std::invalid_argument("Candidates must be strictly ascending.");
      }
      if (n_cands >= 30) {
        throw std::invalid_argument("Candidate count exceeds upper bound 30.");
      }
      valid_cands[n_cands++] = c;
    }
  }
  if (n_cands < 1) {
    throw std::invalid_argument("Candidates cannot be empty.");
  }

  // Parse progression
  std::array<ParsedEvent, MAX_PROGRESSION_LEN> parsed_evs{};
  int const n_evs = lp - 1;
  bool any_call_or_kan = false;
  for (int idx = 0; idx < n_evs; ++idx) {
    int const step = idx + 1;
    int32_t const tok = progression[step];
    ParsedEvent pev{};
    if (tok >= 5 && tok < 597) {
      int const v = tok - 5;
      int const ev_seat = v / 148;
      int const rem = v % 148;
      int const t37 = rem / 4;
      int moqie = (rem / 2) & 1;
      int const liqi = rem & 1;
      if (step == 1 && ev_seat == ju && !any_call_or_kan) {
        moqie = 0; // Normalize dealer first discard
      }
      pev.type = 3;
      pev.seat = ev_seat;
      pev.t37 = t37;
      pev.b34 = tile34(t37);
      pev.moqie = moqie;
      pev.liqi = liqi;
    } else if (tok >= 597 && tok < 957) {
      int const v = tok - 597;
      int const ev_seat = v / 90;
      int const code = v % 90;
      int const from_abs = (ev_seat + 3) % 4;
      pev.type = 5;
      pev.seat = ev_seat;
      pev.src_rel = (from_abs + 4 - seat) % 4;
      pev.code = code;
      pev.c0 = CHI_ENCODE[code][0];
      pev.c1 = CHI_ENCODE[code][1];
      pev.t37 = CHI_ENCODE[code][2];
      any_call_or_kan = true;
    } else if (tok >= 957 && tok < 1437) {
      int const v = tok - 957;
      int const ev_seat = v / 120;
      int const rem = v % 120;
      int const rel_from = rem / 40;
      int const code = rem % 40;
      int const from_abs = (ev_seat + 1 + rel_from) % 4;
      pev.type = 6;
      pev.seat = ev_seat;
      pev.src_rel = (from_abs + 4 - seat) % 4;
      pev.code = code;
      pev.c0 = PENG_ENCODE[code][0];
      pev.c1 = PENG_ENCODE[code][1];
      pev.t37 = PENG_ENCODE[code][2];
      any_call_or_kan = true;
    } else if (tok >= 1437 && tok < 1881) {
      int const v = tok - 1437;
      int const ev_seat = v / 111;
      int const rem = v % 111;
      int const rel_from = rem / 37;
      int const called = rem % 37;
      int const from_abs = (ev_seat + 1 + rel_from) % 4;
      pev.type = 7;
      pev.seat = ev_seat;
      pev.src_rel = (from_abs + 4 - seat) % 4;
      pev.t37 = called;
      any_call_or_kan = true;
    } else if (tok >= 1881 && tok < 2017) {
      int const v = tok - 1881;
      int const ev_seat = v / 34;
      int const b34 = v % 34;
      pev.type = 8;
      pev.seat = ev_seat;
      pev.b34 = b34;
      pev.t37 = base34_to_tile37(b34);
      any_call_or_kan = true;
    } else if (tok >= 2017 && tok < 2165) {
      int const v = tok - 2017;
      int const ev_seat = v / 37;
      int const t37 = v % 37;
      pev.type = 9;
      pev.seat = ev_seat;
      pev.t37 = t37;
      any_call_or_kan = true;
    } else {
      throw std::invalid_argument("Invalid progression token.");
    }
    parsed_evs[idx] = pev;
  }

  // Phase determination
  bool const has_own_act = (valid_cands[0] < 221);
  int phase = 0;
  int trigger_actor_rel = -1;
  int trigger_tile37 = TILE37_NONE;
  int l_landed = lp;
  if (!has_own_act) {
    if (n_evs < 1) {
      throw std::invalid_argument("Response phase requires non-empty events.");
    }
    auto const& last_ev = parsed_evs[n_evs - 1];
    l_landed = lp - 1;
    trigger_actor_rel = (last_ev.seat + 4 - seat) % 4;
    if (trigger_actor_rel < 1 || trigger_actor_rel > 3) {
      throw std::invalid_argument("Response phase trigger actor must be opponent.");
    }
    trigger_tile37 = last_ev.t37;
    if (last_ev.type == 3) phase = 1;
    else if (last_ev.type == 9) phase = 2;
    else if (last_ev.type == 8) phase = 3;
    else throw std::invalid_argument("Unexpected last event type in response phase.");
  }

  // Fill events[0..lp-1]
  out.event_len = lp;
  std::memset(out.events.data(), 0, static_cast<std::size_t>(lp) * EVENT_BYTES);
  out.events[0] = 1; // DORA
  out.events[1] = 4; // TABLE
  out.events[2] = 4; // NONE
  out.events[3] = static_cast<std::uint8_t>(dora_indicators[0]);
  out.events[4] = static_cast<std::uint8_t>(TILE37_NONE);
  out.events[5] = static_cast<std::uint8_t>(TILE37_NONE);
  out.events[6] = is_red5_tile37(dora_indicators[0]) ? 16u : 0u;

  std::array<int, 4> disc_cnt{};
  std::array<int, 4> meld_cnt{};
  std::array<int, 4> open_meld_cnt{};
  std::array<std::vector<MjsoulEval::MeldSpec>, 4> melds_all{};
  std::array<std::array<int, 34>, 4> meld34{};
  bool had_call_before = false;
  std::array<bool, 4> had_disc{};
  std::array<bool, 4> riichi_declared{};
  std::array<bool, 4> double_riichi{};
  std::array<int, 4> riichi_turn_idx{};
  std::array<int, 4> riichi_decl_tile34{-1, -1, -1, -1};
  std::array<int, 4> riichi_ev_idx{-1, -1, -1, -1};

  std::array<int, 4> sangen_melds{};
  std::array<int, 4> wind_melds{};
  int pao_daisangen_payer = -1;
  int pao_daisangen_winner = -1;
  int pao_daisuushii_payer = -1;
  int pao_daisuushii_winner = -1;
  int total_kans = 0;

  for (int idx = 0; idx < n_evs; ++idx) {
    int const step = idx + 1;
    auto const& pev = parsed_evs[idx];
    int const s = pev.seat;
    int const rel = (s + 4 - seat) % 4;
    std::uint8_t* row = &out.events[step * EVENT_BYTES];
    row[0] = pev.type;
    row[1] = static_cast<std::uint8_t>(rel);
    row[2] = static_cast<std::uint8_t>(pev.src_rel);
    row[3] = static_cast<std::uint8_t>(pev.t37);
    row[4] = static_cast<std::uint8_t>(pev.c0);
    row[5] = static_cast<std::uint8_t>(pev.c1);
    row[8] = static_cast<std::uint8_t>(step);

    std::uint8_t flags = 0u;
    if (pev.type == 3) {
      ++public37[pev.t37];
      int const seq = disc_cnt[s]++;
      row[9] = static_cast<std::uint8_t>(seq);
      if (pev.liqi) {
        flags |= 2u;
        bool const dbl = (!had_disc[s] && !had_call_before);
        if (dbl) flags |= 4u;
        riichi_declared[s] = true;
        double_riichi[s] = dbl;
        riichi_turn_idx[s] = seq;
        riichi_decl_tile34[s] = pev.b34;
        riichi_ev_idx[s] = step;
      }
      if (pev.moqie) flags |= 1u;
      if (idx > 0 && parsed_evs[idx - 1].seat == s &&
          (parsed_evs[idx - 1].type == 7 || parsed_evs[idx - 1].type == 8 ||
           parsed_evs[idx - 1].type == 9)) {
        flags |= 8u;
      }
      if (is_red5_tile37(pev.t37)) flags |= 16u;
      had_disc[s] = true;
    } else {
      had_call_before = true;
      if (pev.type == 5) {
        ++public37[pev.c0];
        ++public37[pev.c1];
        row[9] = static_cast<std::uint8_t>(meld_cnt[s]++);
        ++open_meld_cnt[s];
        std::array<std::uint8_t, 3> mt{
          static_cast<std::uint8_t>(tile34(pev.c0)),
          static_cast<std::uint8_t>(tile34(pev.c1)),
          static_cast<std::uint8_t>(tile34(pev.t37))
        };
        std::sort(mt.begin(), mt.end());
        melds_all[s].push_back(MjsoulEval::MeldSpec{MjsoulEval::MeldKind::CHI, true, mt});
        for (auto x : mt) ++meld34[s][x];
        if (is_red5_tile37(pev.c0) || is_red5_tile37(pev.c1) || is_red5_tile37(pev.t37)) {
          flags |= 16u;
        }
      } else if (pev.type == 6) {
        ++public37[pev.c0];
        ++public37[pev.c1];
        row[9] = static_cast<std::uint8_t>(meld_cnt[s]++);
        ++open_meld_cnt[s];
        std::uint8_t const tb = static_cast<std::uint8_t>(tile34(pev.t37));
        melds_all[s].push_back(MjsoulEval::MeldSpec{MjsoulEval::MeldKind::PON, true, {tb, tb, tb}});
        meld34[s][tb] += 3;
        if (is_red5_tile37(pev.c0) || is_red5_tile37(pev.c1) || is_red5_tile37(pev.t37)) {
          flags |= 16u;
        }
        int const from_abs = (seat + pev.src_rel) % 4;
        if (tb >= 31 && tb <= 33) {
          if (++sangen_melds[s] == 3) {
            pao_daisangen_payer = from_abs;
            pao_daisangen_winner = s;
          }
        } else if (tb >= 27 && tb <= 30) {
          if (++wind_melds[s] == 4) {
            pao_daisuushii_payer = from_abs;
            pao_daisuushii_winner = s;
          }
        }
      } else if (pev.type == 7) {
        std::uint8_t const tb = static_cast<std::uint8_t>(tile34(pev.t37));
        int const bn = base34_to_tile37(tb);
        if (tb == 4 || tb == 13 || tb == 22) {
          int const red = bn - 5;
          if (pev.t37 == red) public37[bn] += 3;
          else { ++public37[red]; public37[bn] += 2; }
          flags |= 16u;
        } else {
          public37[bn] += 3;
        }
        row[9] = static_cast<std::uint8_t>(meld_cnt[s]++);
        ++open_meld_cnt[s];
        ++total_kans;
        melds_all[s].push_back(MjsoulEval::MeldSpec{MjsoulEval::MeldKind::KAN, true, {tb, tb, tb}});
        meld34[s][tb] += 4;
        int const from_abs = (seat + pev.src_rel) % 4;
        if (tb >= 31 && tb <= 33) {
          if (++sangen_melds[s] == 3) {
            pao_daisangen_payer = from_abs;
            pao_daisangen_winner = s;
          }
        } else if (tb >= 27 && tb <= 30) {
          if (++wind_melds[s] == 4) {
            pao_daisuushii_payer = from_abs;
            pao_daisuushii_winner = s;
          }
        }
      } else if (pev.type == 8) {
        std::uint8_t const tb = static_cast<std::uint8_t>(pev.b34);
        int const bn = base34_to_tile37(tb);
        if (tb == 4 || tb == 13 || tb == 22) {
          ++public37[bn - 5];
          public37[bn] += 3;
          flags |= 16u;
        } else {
          public37[bn] += 4;
        }
        row[9] = static_cast<std::uint8_t>(meld_cnt[s]++);
        ++total_kans;
        melds_all[s].push_back(MjsoulEval::MeldSpec{MjsoulEval::MeldKind::KAN, false, {tb, tb, tb}});
        meld34[s][tb] += 4;
        if (tb >= 31 && tb <= 33) ++sangen_melds[s];
        else if (tb >= 27 && tb <= 30) ++wind_melds[s];
      } else if (pev.type == 9) {
        std::uint8_t const tb = static_cast<std::uint8_t>(tile34(pev.t37));
        ++public37[pev.t37];
        if (is_red5_tile37(pev.t37)) flags |= 16u;
        int kslot = 0;
        for (std::size_t mi = 0; mi < melds_all[s].size(); ++mi) {
          if (melds_all[s][mi].kind == MjsoulEval::MeldKind::PON &&
              melds_all[s][mi].tiles34[0] == tb) {
            melds_all[s][mi].kind = MjsoulEval::MeldKind::KAN;
            kslot = static_cast<int>(mi);
            break;
          }
        }
        row[9] = static_cast<std::uint8_t>(kslot);
        ++meld34[s][tb];
        ++total_kans;
      }
    }
    row[6] = flags;
  }

  // Physical deduplication invariants
  std::array<int, 34> h_closed34{};
  std::array<int, 34> v_public34{};
  for (int t37 = 0; t37 < 37; ++t37) {
    int const tot = hand37[t37] + public37[t37];
    if (tot > max_copies_tile37(t37)) {
      throw std::invalid_argument("Physical tile conservation violated.");
    }
    int const b = tile34(t37);
    h_closed34[b] += hand37[t37];
    v_public34[b] += public37[t37];
  }
  std::array<int, 34> u34{};
  for (int b = 0; b < 34; ++b) {
    if (h_closed34[b] + v_public34[b] > 4) {
      throw std::invalid_argument("Base34 tile count > 4.");
    }
    u34[b] = 4 - h_closed34[b] - v_public34[b];
  }
  std::array<bool, 3> red5_unseen{
    hand37[0] + public37[0] == 0,
    hand37[10] + public37[10] == 0,
    hand37[20] + public37[20] == 0
  };

  // Landed river (excluding pending ev_last when phase != 0)
  std::array<std::vector<int>, 4> landed_disc34;
  std::array<std::vector<int>, 4> landed_moqie;
  std::array<int, 4> last_actor_step{};
  std::array<int, 4> last_disc_step{-1, -1, -1, -1};
  for (int step = 1; step < l_landed; ++step) {
    auto const& pev = parsed_evs[step - 1];
    last_actor_step[pev.seat] = step;
    if (pev.type == 3) {
      landed_disc34[pev.seat].push_back(pev.b34);
      landed_moqie[pev.seat].push_back(pev.moqie);
      last_disc_step[pev.seat] = step;
    }
  }

  std::array<bool, 4> ippatsu_active{};
  for (int s = 0; s < 4; ++s) {
    int const re = riichi_ev_idx[s];
    if (re < 0 || (phase != 0 && re == lp - 1)) continue;
    bool broken = false;
    for (int step = re + 1; step < lp; ++step) {
      auto const& pev = parsed_evs[step - 1];
      if (pev.type != 3 || pev.seat == s) {
        broken = true;
        break;
      }
    }
    ippatsu_active[s] = !broken;
  }

  std::array<std::array<bool, 34>, 4> post_riichi_table{};
  std::array<std::array<bool, 34>, 4> same_turn_table{};
  for (int s = 0; s < 4; ++s) {
    int const re = riichi_ev_idx[s];
    if (re >= 0 && !(phase != 0 && re == lp - 1)) {
      for (int step = re + 1; step < l_landed; ++step) {
        auto const& pev = parsed_evs[step - 1];
        if (pev.type == 3) post_riichi_table[s][pev.b34] = true;
        else if (pev.type == 8 || pev.type == 9) post_riichi_table[s][tile34(pev.t37)] = true;
      }
    }
    int const ld = last_disc_step[s];
    if (ld >= 0) {
      for (int step = ld + 1; step < l_landed; ++step) {
        auto const& pev = parsed_evs[step - 1];
        if (pev.type == 3) same_turn_table[s][pev.b34] = true;
        else if (pev.type == 8 || pev.type == 9) same_turn_table[s][tile34(pev.t37)] = true;
      }
    }
  }

  std::array<bool, 34> self_river_mask{};
  for (int const b : landed_disc34[seat]) self_river_mask[b] = true;

  int cur_shanten = 0;
  std::vector<int> cur_waits;
  std::vector<int> cur_eff;
  bool riichi_furiten = false;
  bool temp_furiten = false;
  bool perm_furiten = false;
  if (phase == 0) {
    cur_shanten = calc_shanten_raw(h_closed34);
    if (riichi_declared[seat] && zimo_tile37 != TILE37_NONE) {
      auto base_h = h_closed34;
      --base_h[tile34(zimo_tile37)];
      cur_waits = calc_waits_34(base_h, meld34[seat]);
      std::array<bool, 34> wmask{};
      for (int w : cur_waits) wmask[w] = true;
      for (int step = riichi_ev_idx[seat] + 1; step < l_landed; ++step) {
        auto const& pev = parsed_evs[step - 1];
        int tb = (pev.type == 3) ? pev.b34 : ((pev.type == 8 || pev.type == 9) ? tile34(pev.t37) : -1);
        if (tb >= 0 && wmask[tb]) { riichi_furiten = true; break; }
      }
    }
  } else {
    calc_effective_3k1(h_closed34, meld34[seat], cur_shanten, cur_waits, cur_eff);
    if (!cur_waits.empty()) {
      std::array<bool, 34> wmask{};
      for (int w : cur_waits) {
        wmask[w] = true;
        if (self_river_mask[w]) perm_furiten = true;
      }
      if (riichi_declared[seat]) {
        for (int step = riichi_ev_idx[seat] + 1; step < l_landed; ++step) {
          auto const& pev = parsed_evs[step - 1];
          int tb = (pev.type == 3) ? pev.b34 : ((pev.type == 8 || pev.type == 9) ? tile34(pev.t37) : -1);
          if (tb >= 0 && wmask[tb]) { riichi_furiten = true; break; }
        }
      } else {
        for (int step = last_actor_step[seat] + 1; step < l_landed; ++step) {
          auto const& pev = parsed_evs[step - 1];
          int tb = (pev.type == 3) ? pev.b34 : ((pev.type == 8 || pev.type == 9) ? tile34(pev.t37) : -1);
          if (tb >= 0 && wmask[tb]) { temp_furiten = true; break; }
        }
      }
    }
  }
  bool const any_cur_furiten = perm_furiten || riichi_furiten || temp_furiten;

  auto count_dora_for_hand = [&](std::array<int, 37> const& c37,
                                 std::vector<MjsoulEval::MeldSpec> const& ms,
                                 int const extra_red5_in_melds) -> int {
    int d = extra_red5_in_melds;
    for (int t = 0; t < 37; ++t) {
      if (c37[t] > 0) {
        d += c37[t] * dora34[tile34(t)];
        if (is_red5_tile37(t)) d += c37[t];
      }
    }
    for (auto const& m : ms) {
      if (m.kind == MjsoulEval::MeldKind::KAN) {
        int const b = m.tiles34[0];
        d += 4 * dora34[b];
      } else {
        for (int j = 0; j < 3; ++j) d += dora34[m.tiles34[j]];
      }
    }
    return d;
  };

  // Track exact red5 count inside self melds separately for fast O(1) dora lookup
  int self_meld_red5 = 0;
  for (int idx = 0; idx < n_evs; ++idx) {
    auto const& pev = parsed_evs[idx];
    if (pev.seat != seat || pev.type == 3) continue;
    if (pev.type == 5 || pev.type == 6) {
      if (is_red5_tile37(pev.c0) || is_red5_tile37(pev.c1) || is_red5_tile37(pev.t37)) {
        ++self_meld_red5;
      }
    } else if (pev.type == 7 || pev.type == 8) {
      int const b = (pev.type == 8) ? pev.b34 : tile34(pev.t37);
      if (b == 4 || b == 13 || b == 22) ++self_meld_red5;
    } else if (pev.type == 9) {
      // If upgrade to kan on 5, red5 was in original PON unless red5 itself is kakan tile
      if (is_red5_tile37(pev.t37)) ++self_meld_red5;
    }
  }
  int const cur_dora = count_dora_for_hand(hand37, melds_all[seat], self_meld_red5);

  std::array<bool, 34> kuikae_forbid{};
  if (phase == 0 && zimo_tile37 == TILE37_NONE && n_evs > 0) {
    auto const& last_ev = parsed_evs[n_evs - 1];
    if (last_ev.seat == seat) {
      if (last_ev.type == 5) {
        kuikae_forbid[tile34(CHI_ENCODE[last_ev.code][2])] = true;
        auto const ks = CHI_ENCODE[last_ev.code][3];
        if (ks != 255) kuikae_forbid[tile34(ks)] = true;
      } else if (last_ev.type == 6) {
        kuikae_forbid[tile34(last_ev.t37)] = true;
      }
    }
  }

  std::array<DiscardEvalInfo, 34> disc_eval{};
  std::array<std::vector<int>, 34> disc_waits;
  int best_disc_s = 999;
  int best_disc_u = 0;
  if (phase == 0) {
    for (int i = 0; i < n_cands; ++i) {
      int const c = valid_cands[i];
      if (c >= 148) break;
      int const b = tile34(c / 4);
      if (disc_eval[b].legal) continue;
      disc_eval[b].legal = true;
      auto after = h_closed34;
      --after[b];
      std::vector<int> eff;
      calc_effective_3k1(after, meld34[seat], disc_eval[b].s_post, disc_waits[b], eff);
      int u_sum = 0;
      for (int t : eff) u_sum += u34[t];
      disc_eval[b].u_post = u_sum;
      disc_eval[b].post_tenpai = !disc_waits[b].empty();
      if (disc_eval[b].post_tenpai) {
        bool fur = riichi_furiten;
        if (!fur) {
          for (int w : disc_waits[b]) {
            if (w == b || self_river_mask[w]) { fur = true; break; }
          }
        }
        disc_eval[b].post_furiten = fur;
      }
      if (disc_eval[b].s_post < best_disc_s) best_disc_s = disc_eval[b].s_post;
      if (disc_eval[b].u_post > best_disc_u) best_disc_u = disc_eval[b].u_post;
    }
  }

  // ---------------- Fill tile_feats [34, 70] ----------------
  auto tf = [&](int b, int ch) -> float& { return out.tile_feats[b * TILE_FEATS_DIM + ch]; };

  for (int b = 0; b < 34; ++b) {
    int const hc = h_closed34[b];
    int const mc = meld34[seat][b];
    for (int k = 0; k < 4; ++k) {
      tf(b, k) = (hc >= k + 1) ? 1.f : 0.f;
      tf(b, 5 + k) = (mc >= k + 1) ? 1.f : 0.f;
      tf(b, 14 + k) = (u34[b] >= k + 1) ? 1.f : 0.f;
    }
    int const red_t = red5_tile37_of_base34(b);
    if (red_t >= 0) {
      tf(b, 4) = (hand37[red_t] > 0) ? 1.f : 0.f;
      tf(b, 13) = red5_unseen[red_t / 10] ? 1.f : 0.f;
    }
    if (phase == 0 && zimo_tile37 != TILE37_NONE && !(seat == ju && left_tile_count == 69)) {
      if (tile34(zimo_tile37) == b) {
        tf(b, 9) = 1.f;
        if (is_red5_tile37(zimo_tile37)) tf(b, 10) = 1.f;
      }
    }
    tf(b, 11) = kuikae_forbid[b] ? 1.f : 0.f;
    tf(b, 12) = std::min(1.25f, static_cast<float>(dora34[b]) / 4.f);
    tf(b, 18) = self_river_mask[b] ? 1.f : 0.f;
  }

  // Group B1: 3 opponents
  for (int p_rel = 1; p_rel <= 3; ++p_rel) {
    int const off = 19 + (p_rel - 1) * 11;
    int const opp = (seat + p_rel) % 4;
    auto const& river = landed_disc34[opp];
    auto const& river_m = landed_moqie[opp];
    std::array<int, 34> last_di;
    last_di.fill(-1);
    std::array<int, 34> last_mq{};
    for (int di = 0; di < static_cast<int>(river.size()); ++di) {
      int const tb = river[di];
      last_di[tb] = di;
      last_mq[tb] = river_m[di];
    }
    int const rlen = static_cast<int>(river.size());
    for (int b = 0; b < 34; ++b) {
      if (last_di[b] >= 0) {
        tf(b, off + 0) = 1.f;
        int const dt = (rlen - 1) - last_di[b];
        tf(b, off + 6) = std::exp(-static_cast<float>(dt) / 6.f);
        tf(b, off + 7) = (last_mq[b] == 0) ? 1.f : 0.f;
      }
      tf(b, off + 8) = static_cast<float>(meld34[opp][b]) / 4.f;
      tf(b, off + 9) = post_riichi_table[opp][b] ? 1.f : 0.f;
      tf(b, off + 10) = same_turn_table[opp][b] ? 1.f : 0.f;
    }
    if (riichi_declared[opp] && riichi_decl_tile34[opp] >= 0) {
      tf(riichi_decl_tile34[opp], off + 1) = 1.f;
    }
    if (!river.empty()) {
      tf(river.back(), off + 2) = 1.f;
    }
    for (int s = 0; s < 3; ++s) {
      for (int r = 0; r < 9; ++r) {
        int const b = s * 9 + r;
        bool const has_lo = (r >= 3);
        bool const has_hi = (r <= 5);
        int const lb = s * 9 + r - 3;
        int const hb = s * 9 + r + 3;
        bool const lo_safe = has_lo && (last_di[lb] >= 0 || post_riichi_table[opp][lb]);
        bool const hi_safe = has_hi && (last_di[hb] >= 0 || post_riichi_table[opp][hb]);
        tf(b, off + 3) = (lo_safe || hi_safe) ? 1.f : 0.f;
        tf(b, off + 4) = ((!has_lo || lo_safe) && (!has_hi || hi_safe)) ? 1.f : 0.f;
      }
    }
    if (enable_early_shape_hint) {
      int const lim = std::min(6, rlen);
      for (int i = 0; i < lim; ++i) {
        int const db = river[i];
        if (db < 27) {
          int const ds = db / 9;
          int const dr = db % 9;
          if (dr >= 1 && dr <= 3) {
            for (int r2 = 0; r2 < dr; ++r2) tf(ds * 9 + r2, off + 5) = 1.f;
          } else if (dr >= 5 && dr <= 7) {
            for (int r2 = dr + 1; r2 < 9; ++r2) tf(ds * 9 + r2, off + 5) = 1.f;
          }
        }
      }
    }
  }

  // Group B2: kabe, dora_adj, yakuhai
  std::array<int, 34> k_tot{};
  for (int b = 0; b < 34; ++b) k_tot[b] = h_closed34[b] + v_public34[b];
  for (int s = 0; s < 3; ++s) {
    for (int r = 0; r < 9; ++r) {
      int const b = s * 9 + r;
      bool const has_l = (r >= 3);
      bool const has_r = (r <= 5);
      bool const l_no = has_l && (k_tot[s * 9 + r - 2] == 4 || k_tot[s * 9 + r - 1] == 4);
      bool const l_one = has_l && (k_tot[s * 9 + r - 2] >= 3 || k_tot[s * 9 + r - 1] >= 3);
      bool const r_no = has_r && (k_tot[s * 9 + r + 1] == 4 || k_tot[s * 9 + r + 2] == 4);
      bool const r_one = has_r && (k_tot[s * 9 + r + 1] >= 3 || k_tot[s * 9 + r + 2] >= 3);
      bool const nc = (!has_l || l_no) && (!has_r || r_no);
      bool const oc = (!nc) && (!has_l || l_one) && (!has_r || r_one);
      tf(b, 52) = nc ? 1.f : 0.f;
      tf(b, 53) = oc ? 1.f : 0.f;

      for (int db = 0; db < 27; ++db) {
        if (dora34[db] > 0 && db / 9 == s) {
          int const dist = std::abs(r - (db % 9));
          if (dist == 1) tf(b, 54) = 1.f;
          else if (dist == 2) tf(b, 55) = 1.f;
        }
      }
    }
  }
  for (int rel = 0; rel < 4; ++rel) {
    int const abs_p = (seat + rel) % 4;
    int const sw = 27 + (abs_p + 4 - ju) % 4;
    int const rw = 27 + chang;
    tf(31, 56 + rel) = 0.5f;
    tf(32, 56 + rel) = 0.5f;
    tf(33, 56 + rel) = 0.5f;
    for (int w = 27; w < 31; ++w) {
      float val = 0.f;
      if (w == sw) val += 0.5f;
      if (w == rw) val += 0.5f;
      tf(w, 56 + rel) = val;
    }
  }

  // Group C
  if (phase == 0) {
    for (int b = 0; b < 34; ++b) {
      if (!disc_eval[b].legal) continue;
      auto const& de = disc_eval[b];
      tf(b, 60) = 1.f;
      tf(b, 61) = (de.s_post == best_disc_s) ? 1.f : 0.f;
      tf(b, 62) = (de.s_post == best_disc_s + 1) ? 1.f : 0.f;
      tf(b, 63) = std::min(1.f, static_cast<float>(std::max(0, de.s_post)) / 8.f);
      tf(b, 64) = std::min(1.f, std::log1p(static_cast<float>(de.u_post)) / LN61);
      tf(b, 65) = static_cast<float>(de.u_post) / static_cast<float>(std::max(1, best_disc_u));
      tf(b, 66) = de.post_tenpai ? 1.f : 0.f;
      tf(b, 67) = de.post_furiten ? 1.f : 0.f;
    }
  } else {
    for (int w : cur_waits) tf(w, 68) = 1.f;
    for (int e : cur_eff) tf(e, 69) = 1.f;
  }

  // ---------------- global_feats [48] ----------------
  out.global_feats[chang] = 1.f;
  out.global_feats[3 + ju] = 1.f;
  int const ben = numeric[0];
  int const liqibang = numeric[1];
  std::array<int32_t, 4> scores{numeric[2], numeric[3], numeric[4], numeric[5]};
  out.global_feats[7] = std::min(2.f, static_cast<float>(ben) / 8.f);
  out.global_feats[8] = std::min(2.f, static_cast<float>(liqibang) / 8.f);
  out.global_feats[9] = std::clamp(static_cast<float>(left_tile_count) / 70.f, 0.f, 1.f);
  bool const is_all_last = is_tonpuusen ? (chang == 0 && ju == 3) : (chang == 1 && ju == 3);
  bool const is_ext = is_tonpuusen ? (chang >= 1) : (chang >= 2);
  out.global_feats[10] = is_all_last ? 1.f : 0.f;
  out.global_feats[11] = is_ext ? 1.f : 0.f;
  out.global_feats[12] = std::min(1.f, static_cast<float>(total_kans) / 4.f);
  out.global_feats[13] = std::min(1.f, static_cast<float>(n_dora_ind) / 5.f);
  out.global_feats[14] = (!had_call_before && disc_cnt[seat] == 0) ? 1.f : 0.f;
  out.global_feats[15 + phase] = 1.f;
  out.global_feats[19] = (riichi_furiten || temp_furiten) ? 1.f : 0.f;
  if (phase != 0 && trigger_actor_rel >= 1 && trigger_actor_rel <= 3) {
    out.global_feats[20 + trigger_actor_rel - 1] = 1.f;
  }
  out.global_feats[23] = std::min(1.f, static_cast<float>(std::max(0, cur_shanten)) / 8.f);
  out.global_feats[24] = static_cast<float>(meld_cnt[seat]) / 4.f;
  out.global_feats[25] = (open_meld_cnt[seat] == 0) ? 1.f : 0.f;

  constexpr std::array<float, 11> rule_consts{1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.f, 1.f, 1.f, 1.f, 0.f};
  for (int i = 0; i < 11; ++i) out.global_feats[26 + i] = rule_consts[i];
  out.global_feats[37] = is_tonpuusen ? 1.f : 0.f;
  for (int i = 38; i <= 43; ++i) out.global_feats[i] = 1.f;
  for (int i = 0; i < 4; ++i) out.global_feats[44 + i] = reward_weights[i];

  // ---------------- player_feats [4, 32] ----------------
  auto const ranks = MjsoulEval::compute_ranks(scores);
  std::array<int, 4> sorted_by_rank{};
  for (int i = 0; i < 4; ++i) sorted_by_rank[ranks[i]] = i;

  for (int rel = 0; rel < 4; ++rel) {
    int const abs_p = (seat + rel) % 4;
    float* pf = &out.player_feats[rel * PLAYER_FEATS_DIM];
    pf[rel] = 1.f;
    pf[4 + (abs_p + 4 - ju) % 4] = 1.f;
    pf[8] = (abs_p == ju) ? 1.f : 0.f;
    pf[9] = std::clamp(static_cast<float>(scores[abs_p] - 25000) / 25000.f, -3.f, 3.f);
    pf[10] = std::clamp(static_cast<float>(scores[abs_p] - scores[seat]) / 25000.f, -3.f, 3.f);
    int const rk = ranks[abs_p];
    if (rk > 0) {
      int const hi = sorted_by_rank[rk - 1];
      pf[11] = std::clamp(static_cast<float>(scores[hi] - scores[abs_p]) / 25000.f, 0.f, 3.f);
    }
    if (rk < 3) {
      int const lo = sorted_by_rank[rk + 1];
      pf[12] = std::clamp(static_cast<float>(scores[abs_p] - scores[lo]) / 25000.f, 0.f, 3.f);
    }
    pf[13] = static_cast<float>(rk) / 3.f;
    pf[14] = riichi_declared[abs_p] ? 1.f : 0.f;
    pf[15] = double_riichi[abs_p] ? 1.f : 0.f;
    pf[16] = ippatsu_active[abs_p] ? 1.f : 0.f;
    if (riichi_declared[abs_p]) {
      pf[17] = std::min(1.f, static_cast<float>(riichi_turn_idx[abs_p]) / 24.f);
    }
    pf[18] = static_cast<float>(open_meld_cnt[abs_p]) / 4.f;
    int const rlen = static_cast<int>(landed_moqie[abs_p].size());
    if (rlen > 0) {
      int const start = std::max(0, rlen - 6);
      int tedashi = 0;
      for (int i = start; i < rlen; ++i) {
        if (landed_moqie[abs_p][i] == 0) ++tedashi;
      }
      pf[19] = static_cast<float>(tedashi) / static_cast<float>(rlen - start);
    }
  }

  // ---------------- cand_mode & cand_dyn ----------------
  int const self_pw = 27 + (seat + 4 - ju) % 4;
  int const self_rw = 27 + chang;
  int cur_u = 0;
  if (phase != 0) {
    for (int t : cur_eff) cur_u += u34[t];
  }
  bool const is_cur_rinshan = (phase == 0 && zimo_tile37 != TILE37_NONE && n_evs > 0 &&
                               parsed_evs[n_evs - 1].seat == seat &&
                               (parsed_evs[n_evs - 1].type == 7 || parsed_evs[n_evs - 1].type == 8 ||
                                parsed_evs[n_evs - 1].type == 9));
  constexpr std::array<int, 13> yaochuu{0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33};
  int distinct_yaochuu = 0;
  for (int b : yaochuu) {
    if (h_closed34[b] > 0) ++distinct_yaochuu;
  }

  for (int ci = 0; ci < n_cands; ++ci) {
    int const aid = valid_cands[ci];
    int const cm = ACTION_MANIFEST[aid].cand_mode;
    out.cand_mode[ci] = static_cast<std::uint8_t>(cm);
    float* cd = &out.cand_dyn[ci * CAND_DYN_DIM];

    if (cm == 1) { // POST_DISCARD_EXACT
      int const d37 = aid / 4;
      int const b = tile34(d37);
      auto const& de = disc_eval[b];
      int const after_dora = cur_dora - dora34[b] - (is_red5_tile37(d37) ? 1 : 0);
      cd[0] = std::min(1.f, static_cast<float>(std::max(0, de.s_post)) / 8.f);
      cd[1] = std::clamp(static_cast<float>(de.s_post - best_disc_s) / 4.f, -1.f, 2.f);
      cd[2] = std::min(1.f, std::log1p(static_cast<float>(de.u_post)) / LN61);
      cd[3] = std::min(1.f, static_cast<float>(de.u_post) / static_cast<float>(std::max(1, best_disc_u)));
      cd[4] = de.post_tenpai ? 1.f : 0.f;
      cd[5] = de.post_furiten ? 1.f : 0.f;
      cd[6] = std::min(2.f, static_cast<float>(after_dora) / 8.f);
      if (de.post_tenpai) {
        auto after = h_closed34;
        --after[b];
        cd[7] = has_dama_yaku(after, melds_all[seat], disc_waits[b], self_pw, self_rw) ? 1.f : 0.f;
      }
    } else if (cm == 2) { // PRE_DISCARD_MELD_ENVELOPE
      auto after34 = h_closed34;
      auto m_all = melds_all[seat];
      auto m34_s = meld34[seat];
      std::array<bool, 34> forbid{};
      int post_meld_dora = cur_dora;
      if (aid < 312) {
        int const code = aid - 222;
        int const c0 = CHI_ENCODE[code][0], c1 = CHI_ENCODE[code][1], called = CHI_ENCODE[code][2];
        --after34[tile34(c0)];
        --after34[tile34(c1)];
        std::array<std::uint8_t, 3> mt{
          static_cast<std::uint8_t>(tile34(c0)),
          static_cast<std::uint8_t>(tile34(c1)),
          static_cast<std::uint8_t>(tile34(called))
        };
        std::sort(mt.begin(), mt.end());
        m_all.push_back(MjsoulEval::MeldSpec{MjsoulEval::MeldKind::CHI, true, mt});
        for (auto x : mt) ++m34_s[x];
        forbid[tile34(called)] = true;
        if (CHI_ENCODE[code][3] != 255) forbid[tile34(CHI_ENCODE[code][3])] = true;
        post_meld_dora += dora34[tile34(called)] + (is_red5_tile37(called) ? 1 : 0);
      } else {
        int const code = (aid - 312) % 40;
        int const c0 = PENG_ENCODE[code][0], c1 = PENG_ENCODE[code][1], called = PENG_ENCODE[code][2];
        --after34[tile34(c0)];
        --after34[tile34(c1)];
        std::uint8_t const tb = static_cast<std::uint8_t>(tile34(called));
        m_all.push_back(MjsoulEval::MeldSpec{MjsoulEval::MeldKind::PON, true, {tb, tb, tb}});
        m34_s[tb] += 3;
        forbid[tb] = true;
        post_meld_dora += dora34[tb] + (is_red5_tile37(called) ? 1 : 0);
      }
      struct OptExit { int db; int sb; int ub; std::vector<int> wb; };
      std::vector<OptExit> exits;
      int s_opt = 999;
      for (int db = 0; db < 34; ++db) {
        if (after34[db] == 0 || forbid[db]) continue;
        --after34[db];
        int sb = 8;
        std::vector<int> wb, eb;
        calc_effective_3k1(after34, m34_s, sb, wb, eb);
        ++after34[db];
        int ub = 0;
        for (int t : eb) ub += u34[t];
        if (sb < s_opt) s_opt = sb;
        exits.push_back(OptExit{db, sb, ub, std::move(wb)});
      }
      int max_u = 0, n_opt = 0;
      bool can_tenpai = (s_opt == 0);
      bool all_fur = !exits.empty() && can_tenpai;
      bool has_yaku_exit = false;
      for (auto const& ex : exits) {
        if (ex.sb != s_opt) continue;
        ++n_opt;
        if (ex.ub > max_u) max_u = ex.ub;
        if (can_tenpai) {
          bool fur = riichi_furiten;
          if (!fur) {
            for (int w : ex.wb) {
              if (w == ex.db || self_river_mask[w]) { fur = true; break; }
            }
          }
          if (!fur) all_fur = false;
          if (!has_yaku_exit) {
            --after34[ex.db];
            if (has_dama_yaku(after34, m_all, ex.wb, self_pw, self_rw)) {
              has_yaku_exit = true;
            }
            ++after34[ex.db];
          }
        }
      }
      cd[0] = std::min(1.f, static_cast<float>(std::max(0, s_opt)) / 8.f);
      cd[1] = std::clamp(static_cast<float>(s_opt - cur_shanten) / 4.f, -2.f, 2.f);
      cd[2] = std::min(1.f, std::log1p(static_cast<float>(max_u)) / LN61);
      cd[3] = static_cast<float>(n_opt) / 14.f;
      cd[4] = can_tenpai ? 1.f : 0.f;
      cd[5] = (can_tenpai && all_fur) ? 1.f : 0.f;
      cd[6] = std::min(2.f, static_cast<float>(post_meld_dora) / 8.f);
      cd[7] = (can_tenpai && has_yaku_exit) ? 1.f : 0.f;
    } else if (cm == 3) { // PENDING_RINSHAN_KAN
      int post_d = cur_dora;
      if (aid >= 432) {
        int const called = (aid - 432) % 37;
        post_d += dora34[tile34(called)] + (is_red5_tile37(called) ? 1 : 0);
      }
      int const s_kan = (phase == 0) ? calc_shanten_raw(h_closed34) : cur_shanten;
      bool const tenp = (phase == 0) ? (s_kan == 0) : (!cur_waits.empty());
      cd[0] = std::min(1.f, static_cast<float>(std::max(0, s_kan)) / 8.f);
      cd[4] = tenp ? 1.f : 0.f;
      cd[5] = (aid >= 432) ? 0.f : 1.f;
      cd[6] = std::min(2.f, static_cast<float>(post_d) / 8.f);
      cd[7] = std::min(1.f, static_cast<float>(total_kans + 1) / 4.f);
    } else if (cm == 4) { // PASS_CURRENT_STATE
      cd[0] = std::min(1.f, static_cast<float>(std::max(0, cur_shanten)) / 8.f);
      cd[2] = std::min(1.f, std::log1p(static_cast<float>(cur_u)) / LN61);
      cd[3] = 1.f;
      cd[4] = !cur_waits.empty() ? 1.f : 0.f;
      cd[5] = any_cur_furiten ? 1.f : 0.f;
      cd[6] = std::min(2.f, static_cast<float>(cur_dora) / 8.f);
    } else if (cm == 5) { // WIN_EVAL
      bool const is_tsumo = (aid == 219);
      auto eval34 = h_closed34;
      int win_t37 = zimo_tile37;
      int loser_abs = -1;
      int win_dora = 0;
      int win_red5 = self_meld_red5;
      for (int t = 0; t < 37; ++t) {
        if (hand37[t] > 0) {
          win_dora += hand37[t] * dora34[tile34(t)];
          if (is_red5_tile37(t)) win_red5 += hand37[t];
        }
      }
      for (auto const& m : melds_all[seat]) {
        if (m.kind == MjsoulEval::MeldKind::KAN) {
          win_dora += 4 * dora34[m.tiles34[0]];
        } else {
          for (int j = 0; j < 3; ++j) win_dora += dora34[m.tiles34[j]];
        }
      }
      if (!is_tsumo) {
        win_t37 = trigger_tile37;
        ++eval34[tile34(win_t37)];
        loser_abs = (seat + 1 + (aid - 543)) % 4;
        win_dora += dora34[tile34(win_t37)];
        if (is_red5_tile37(win_t37)) ++win_red5;
      }

      MjsoulEval::EvalConfig cfg;
      cfg.is_tsumo = is_tsumo;
      cfg.is_riichi = riichi_declared[seat];
      cfg.is_ippatsu = ippatsu_active[seat];
      cfg.is_rinshan = is_cur_rinshan;
      cfg.is_chankan = (!is_tsumo && (phase == 2 || phase == 3));
      cfg.is_haitei = (is_tsumo && left_tile_count == 0 && !is_cur_rinshan);
      cfg.is_houtei = (!is_tsumo && phase == 1 && left_tile_count == 0);
      cfg.is_daburu_riichi = double_riichi[seat];
      cfg.is_tenhou = (is_tsumo && !any_call_or_kan && seat == ju && lp == 1);
      cfg.is_chiihou = (is_tsumo && !any_call_or_kan && seat != ju &&
                        disc_cnt[seat] == 0 && lp - 1 == (seat + 4 - ju) % 4);
      cfg.player_wind = self_pw;
      cfg.round_wind = self_rw;

      auto const eval_res = MjsoulEval::eval_mjsoul_hand(
          eval34, melds_all[seat], tile34(win_t37), win_dora, win_red5, cfg);
      if (!eval_res.valid) {
        throw std::runtime_error("WIN_EVAL hand unexpectedly failed validation.");
      }
      int const ds_p = (pao_daisangen_winner == seat) ? pao_daisangen_payer : -1;
      int const dw_p = (pao_daisuushii_winner == seat) ? pao_daisuushii_payer : -1;
      auto const sres = MjsoulEval::eval_win_settlement(
          eval_res, is_tsumo, seat, loser_abs, ju, chang, ben, liqibang,
          is_tonpuusen, scores, ds_p, dw_p);
      cd[0] = std::min(2.f, static_cast<float>(eval_res.han) / 13.f);
      cd[1] = std::min(1.5f, static_cast<float>(eval_res.fu) / 110.f);
      cd[2] = std::min(1.5f, std::log1p(static_cast<float>(std::max<int32_t>(0, sres.winner_gain))) / LN64001);
      cd[3] = eval_res.is_yakuman ? 1.f : 0.f;
      cd[4] = static_cast<float>(sres.post_rank) / 3.f;
      cd[5] = sres.would_end_game ? 1.f : 0.f;
      cd[6] = riichi_declared[seat] ? 1.f : 0.f;
    } else if (cm == 6) { // KYUSHU_DRAW_EVAL
      int const sk = calc_shanten_raw(h_closed34);
      cd[0] = std::min(1.f, static_cast<float>(std::max(0, sk)) / 8.f);
      cd[1] = static_cast<float>(distinct_yaochuu) / 13.f;
      cd[2] = std::min(2.f, static_cast<float>(cur_dora) / 8.f);
      cd[3] = (seat == ju) ? 1.f : 0.f;
      cd[4] = static_cast<float>(ranks[seat]) / 3.f;
    }
  }
}

namespace {

struct Int32MatrixView {
  std::vector<int32_t> owned;
  int32_t const* ptr = nullptr;
  int rows = 0;
  int cols = 0;
};

Int32MatrixView extract_int32_2d(PyObject* obj, char const* name) {
  Int32MatrixView view;
  PyBufferProcs* bp = Py_TYPE(obj)->tp_as_buffer;
  Py_buffer pb;
  if (bp != nullptr && PyObject_GetBuffer(obj, &pb, PyBUF_ND | PyBUF_STRIDES) == 0) {
    bool const fmt_ok = (pb.format == nullptr ||
                         std::strcmp(pb.format, "i") == 0 ||
                         std::strcmp(pb.format, "<i") == 0 ||
                         std::strcmp(pb.format, "=i") == 0 ||
                         std::strcmp(pb.format, "l") == 0 ||
                         std::strcmp(pb.format, "<l") == 0 ||
                         std::strcmp(pb.format, "=l") == 0);
    if (pb.ndim == 2 && pb.itemsize == 4 && fmt_ok &&
        pb.strides[1] == 4 && pb.strides[0] == pb.shape[1] * 4) {
      view.rows = static_cast<int>(pb.shape[0]);
      view.cols = static_cast<int>(pb.shape[1]);
      view.owned.resize(static_cast<std::size_t>(view.rows) * view.cols);
      std::memcpy(view.owned.data(), pb.buf, view.owned.size() * sizeof(int32_t));
      view.ptr = view.owned.data();
      PyBuffer_Release(&pb);
      return view;
    }
    PyBuffer_Release(&pb);
  }

  PyObject* outer = PySequence_Fast(obj, "Expected 2D integer sequence.");
  if (!outer) throw std::invalid_argument(std::string(name) + ": expected 2D sequence.");
  view.rows = static_cast<int>(PySequence_Fast_GET_SIZE(outer));
  if (view.rows > 0) {
    PyObject* row0 = PySequence_Fast(PySequence_Fast_GET_ITEM(outer, 0), "Expected 1D row.");
    if (!row0) { Py_DECREF(outer); throw std::invalid_argument("Expected 2D sequence rows."); }
    view.cols = static_cast<int>(PySequence_Fast_GET_SIZE(row0));
    Py_DECREF(row0);
    view.owned.resize(static_cast<std::size_t>(view.rows) * view.cols);
    for (int i = 0; i < view.rows; ++i) {
      PyObject* row = PySequence_Fast(PySequence_Fast_GET_ITEM(outer, i), "Expected row sequence.");
      if (!row || static_cast<int>(PySequence_Fast_GET_SIZE(row)) != view.cols) {
        Py_XDECREF(row);
        Py_DECREF(outer);
        throw std::invalid_argument("Jagged 2D sequence.");
      }
      for (int j = 0; j < view.cols; ++j) {
        long const val = PyLong_AsLong(PySequence_Fast_GET_ITEM(row, j));
        if (PyErr_Occurred()) {
          Py_DECREF(row);
          Py_DECREF(outer);
          throw std::invalid_argument("Expected integer entries.");
        }
        view.owned[static_cast<std::size_t>(i) * view.cols + j] = static_cast<int32_t>(val);
      }
      Py_DECREF(row);
    }
  }
  Py_DECREF(outer);
  view.ptr = view.owned.data();
  return view;
}

struct GilReleaseGuard {
  PyThreadState* state;
  GilReleaseGuard() noexcept : state(PyEval_SaveThread()) {}
  ~GilReleaseGuard() noexcept {
    if (state != nullptr) PyEval_RestoreThread(state);
  }
  GilReleaseGuard(GilReleaseGuard const&) = delete;
  GilReleaseGuard& operator=(GilReleaseGuard const&) = delete;
};

} // namespace

PyObject* py_encode_structured_v2_batch(PyObject*, PyObject* args) {
  PyObject *sp_obj, *nu_obj, *pr_obj, *ca_obj;
  int early_hint = 0;
  PyObject* rw_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OOOO|pO", &sp_obj, &nu_obj, &pr_obj, &ca_obj,
                        &early_hint, &rw_obj)) {
    return nullptr;
  }
  try {
    auto sp = extract_int32_2d(sp_obj, "sparse");
    auto nu = extract_int32_2d(nu_obj, "numeric");
    auto pr = extract_int32_2d(pr_obj, "progression");
    auto ca = extract_int32_2d(ca_obj, "candidates");
    if (sp.cols != 29 || nu.cols != 6 || ca.cols != MAX_CANDIDATES || pr.cols < 1) {
      throw std::invalid_argument("Unexpected input column widths.");
    }
    int const B = sp.rows;
    if (nu.rows != B || pr.rows != B || ca.rows != B) {
      throw std::invalid_argument("Batch size mismatch across 4-field inputs.");
    }

    std::array<float, 4> rw{1.5f, 0.5f, -0.5f, -1.5f};
    if (rw_obj != nullptr && rw_obj != Py_None) {
      PyObject* seq = PySequence_Fast(rw_obj, "Expected 4 reward_weights.");
      if (!seq || PySequence_Fast_GET_SIZE(seq) != 4) {
        Py_XDECREF(seq);
        throw std::invalid_argument("reward_weights must have length 4.");
      }
      for (int i = 0; i < 4; ++i) {
        rw[i] = static_cast<float>(PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i)));
        if (PyErr_Occurred()) { Py_DECREF(seq); throw std::invalid_argument("Invalid reward weight."); }
      }
      Py_DECREF(seq);
    }

    bool const hint = (early_hint != 0);
    std::vector<EncodedSample> samples(B);
    int l_max = 1;

    // Release Python GIL for pure C++ feature extraction + Nyanten + hand eval!
    {
      GilReleaseGuard gil_release;
      for (int b = 0; b < B; ++b) {
        encode_single_sample(
            sp.ptr + static_cast<std::size_t>(b) * 29,
            nu.ptr + static_cast<std::size_t>(b) * 6,
            pr.ptr + static_cast<std::size_t>(b) * pr.cols,
            pr.cols,
            ca.ptr + static_cast<std::size_t>(b) * MAX_CANDIDATES,
            hint,
            rw,
            samples[b]);
        if (samples[b].event_len > l_max) l_max = samples[b].event_len;
      }
    }

    Py_ssize_t const n0 = static_cast<Py_ssize_t>(
        static_cast<std::size_t>(B) * 34 * TILE_FEATS_DIM * sizeof(float));
    Py_ssize_t const n1 = static_cast<Py_ssize_t>(
        static_cast<std::size_t>(B) * l_max * EVENT_BYTES);
    Py_ssize_t const n2 = static_cast<Py_ssize_t>(
        static_cast<std::size_t>(B) * sizeof(int32_t));
    Py_ssize_t const n3 = static_cast<Py_ssize_t>(
        static_cast<std::size_t>(B) * GLOBAL_FEATS_DIM * sizeof(float));
    Py_ssize_t const n4 = static_cast<Py_ssize_t>(
        static_cast<std::size_t>(B) * 4 * PLAYER_FEATS_DIM * sizeof(float));
    Py_ssize_t const n5 = static_cast<Py_ssize_t>(
        static_cast<std::size_t>(B) * MAX_CANDIDATES);
    Py_ssize_t const n6 = static_cast<Py_ssize_t>(
        static_cast<std::size_t>(B) * MAX_CANDIDATES * CAND_DYN_DIM * sizeof(float));

    auto alloc_ba = [](Py_ssize_t n) -> PyObject* {
      return PyByteArray_FromStringAndSize(n == 0 ? "" : nullptr, n);
    };
    PyObject* t0 = alloc_ba(n0);
    PyObject* t1 = alloc_ba(n1);
    PyObject* t2 = alloc_ba(n2);
    PyObject* t3 = alloc_ba(n3);
    PyObject* t4 = alloc_ba(n4);
    PyObject* t5 = alloc_ba(n5);
    PyObject* t6 = alloc_ba(n6);
    PyObject* t7 = PyLong_FromLong(l_max);
    if (!t0 || !t1 || !t2 || !t3 || !t4 || !t5 || !t6 || !t7) {
      Py_XDECREF(t0); Py_XDECREF(t1); Py_XDECREF(t2); Py_XDECREF(t3);
      Py_XDECREF(t4); Py_XDECREF(t5); Py_XDECREF(t6); Py_XDECREF(t7);
      return nullptr;
    }

    char* p0 = n0 ? PyByteArray_AS_STRING(t0) : nullptr;
    char* p1 = n1 ? PyByteArray_AS_STRING(t1) : nullptr;
    char* p2 = n2 ? PyByteArray_AS_STRING(t2) : nullptr;
    char* p3 = n3 ? PyByteArray_AS_STRING(t3) : nullptr;
    char* p4 = n4 ? PyByteArray_AS_STRING(t4) : nullptr;
    char* p5 = n5 ? PyByteArray_AS_STRING(t5) : nullptr;
    char* p6 = n6 ? PyByteArray_AS_STRING(t6) : nullptr;

    {
      GilReleaseGuard gil_release;
      if (p1) {
        std::memset(p1, 0, static_cast<std::size_t>(n1));
      }
      for (int b = 0; b < B; ++b) {
        auto const& s = samples[b];
        std::memcpy(p0 + static_cast<std::size_t>(b) * 34 * TILE_FEATS_DIM * sizeof(float),
                    s.tile_feats.data(), s.tile_feats.size() * sizeof(float));
        if (s.event_len > 0) {
          std::memcpy(p1 + static_cast<std::size_t>(b) * l_max * EVENT_BYTES,
                      s.events.data(), static_cast<std::size_t>(s.event_len) * EVENT_BYTES);
        }
        int32_t const evl = static_cast<int32_t>(s.event_len);
        std::memcpy(p2 + static_cast<std::size_t>(b) * sizeof(int32_t),
                    &evl, sizeof(int32_t));
        std::memcpy(p3 + static_cast<std::size_t>(b) * GLOBAL_FEATS_DIM * sizeof(float),
                    s.global_feats.data(), s.global_feats.size() * sizeof(float));
        std::memcpy(p4 + static_cast<std::size_t>(b) * 4 * PLAYER_FEATS_DIM * sizeof(float),
                    s.player_feats.data(), s.player_feats.size() * sizeof(float));
        std::memcpy(p5 + static_cast<std::size_t>(b) * MAX_CANDIDATES,
                    s.cand_mode.data(), s.cand_mode.size());
        std::memcpy(p6 + static_cast<std::size_t>(b) * MAX_CANDIDATES * CAND_DYN_DIM * sizeof(float),
                    s.cand_dyn.data(), s.cand_dyn.size() * sizeof(float));
      }
    }

    PyObject* tuple = PyTuple_Pack(8, t0, t1, t2, t3, t4, t5, t6, t7);
    Py_DECREF(t0); Py_DECREF(t1); Py_DECREF(t2); Py_DECREF(t3);
    Py_DECREF(t4); Py_DECREF(t5); Py_DECREF(t6); Py_DECREF(t7);
    return tuple;
  } catch (std::exception const& err) {
    PyErr_Clear();
    PyErr_SetString(PyExc_ValueError, err.what());
    return nullptr;
  }
}

PyObject* py_eval_mjsoul_hand(PyObject*, PyObject* args) {
  PyObject *c34_obj, *melds_obj;
  int win_t34 = 0, total_dora = 0, total_red5 = 0;
  long tool_config = 0;
  if (!PyArg_ParseTuple(args, "OOiiil", &c34_obj, &melds_obj, &win_t34,
                        &total_dora, &total_red5, &tool_config)) {
    return nullptr;
  }
  try {
    PyObject* cseq = PySequence_Fast(c34_obj, "closed34 must be 34-element sequence.");
    if (!cseq || PySequence_Fast_GET_SIZE(cseq) != 34) {
      Py_XDECREF(cseq);
      throw std::invalid_argument("closed34 must have length 34.");
    }
    std::array<int, 34> closed34{};
    for (int i = 0; i < 34; ++i) {
      closed34[i] = static_cast<int>(PyLong_AsLong(PySequence_Fast_GET_ITEM(cseq, i)));
    }
    Py_DECREF(cseq);

    PyObject* mseq = PySequence_Fast(melds_obj, "melds must be a sequence.");
    if (!mseq) throw std::invalid_argument("melds must be a sequence.");
    Py_ssize_t const nm = PySequence_Fast_GET_SIZE(mseq);
    std::vector<MjsoulEval::MeldSpec> melds;
    melds.reserve(nm);
    for (Py_ssize_t i = 0; i < nm; ++i) {
      PyObject* item = PySequence_Fast(PySequence_Fast_GET_ITEM(mseq, i), "Expected (kind, tiles34, opened).");
      if (!item || PySequence_Fast_GET_SIZE(item) != 3) {
        Py_XDECREF(item);
        Py_DECREF(mseq);
        throw std::invalid_argument("Each meld must be (kind, tiles34, opened).");
      }
      int const kind_i = static_cast<int>(PyLong_AsLong(PySequence_Fast_GET_ITEM(item, 0)));
      PyObject* tseq = PySequence_Fast(PySequence_Fast_GET_ITEM(item, 1), "Expected 3 tiles34.");
      int const opened_i = static_cast<int>(PyLong_AsLong(PySequence_Fast_GET_ITEM(item, 2)));
      if (!tseq || PySequence_Fast_GET_SIZE(tseq) < 3) {
        Py_XDECREF(tseq);
        Py_DECREF(item);
        Py_DECREF(mseq);
        throw std::invalid_argument("Meld tiles34 must have >= 3 elements.");
      }
      MjsoulEval::MeldSpec ms;
      ms.kind = static_cast<MjsoulEval::MeldKind>(kind_i);
      ms.opened = (opened_i != 0);
      for (int j = 0; j < 3; ++j) {
        ms.tiles34[j] = static_cast<std::uint8_t>(PyLong_AsLong(PySequence_Fast_GET_ITEM(tseq, j)));
      }
      melds.push_back(ms);
      Py_DECREF(tseq);
      Py_DECREF(item);
    }
    Py_DECREF(mseq);

    auto const cfg = MjsoulEval::detail::decode_tool_config(tool_config);
    MjsoulEval::HandEvalResult res;
    {
      GilReleaseGuard gil_release;
      res = MjsoulEval::eval_mjsoul_hand(closed34, melds, win_t34, total_dora, total_red5, cfg);
    }

    return Py_BuildValue("(OiiO)",
                         res.valid ? Py_True : Py_False,
                         res.han,
                         res.fu,
                         res.is_yakuman ? Py_True : Py_False);
  } catch (std::exception const& err) {
    PyErr_Clear();
    PyErr_SetString(PyExc_ValueError, err.what());
    return nullptr;
  }
}

PyObject* py_eval_win_settlement(PyObject*, PyObject* args) {
  int han = 0, fu = 0, is_yk = 0, is_tsumo = 0, winner = 0, loser = 0;
  int ju = 0, chang = 0, ben = 0, liqibang = 0, is_tonpuusen = 0;
  int pao_ds = -1, pao_dw = -1;
  PyObject* scores_obj = nullptr;
  if (!PyArg_ParseTuple(args, "iiiiiiiiiiiO|ii",
                        &han, &fu, &is_yk, &is_tsumo, &winner, &loser,
                        &ju, &chang, &ben, &liqibang, &is_tonpuusen, &scores_obj,
                        &pao_ds, &pao_dw)) {
    return nullptr;
  }
  try {
    PyObject* seq = PySequence_Fast(scores_obj, "pre_scores must be length 4.");
    if (!seq || PySequence_Fast_GET_SIZE(seq) != 4) {
      Py_XDECREF(seq);
      throw std::invalid_argument("pre_scores must have length 4.");
    }
    std::array<int32_t, 4> pre_scores{};
    for (int i = 0; i < 4; ++i) {
      pre_scores[i] = static_cast<int32_t>(PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i)));
    }
    Py_DECREF(seq);

    MjsoulEval::HandEvalResult hr{true, han, fu, is_yk != 0};
    auto const res = MjsoulEval::eval_win_settlement(
        hr, is_tsumo != 0, winner, loser, ju, chang, ben, liqibang,
        is_tonpuusen != 0, pre_scores, pao_ds, pao_dw);
    return Py_BuildValue("((iiii)iiO)",
                         res.post_scores[0], res.post_scores[1],
                         res.post_scores[2], res.post_scores[3],
                         res.winner_gain, res.post_rank,
                         res.would_end_game ? Py_True : Py_False);
  } catch (std::exception const& err) {
    PyErr_Clear();
    PyErr_SetString(PyExc_ValueError, err.what());
    return nullptr;
  }
}

PyObject* py_action_manifest_bytes(PyObject*, PyObject*) {
  std::array<int32_t, 547 * 10> buf{};
  for (int i = 0; i < 547; ++i) {
    auto const& r = ACTION_MANIFEST[i];
    buf[i * 10 + 0] = r.action_type;
    buf[i * 10 + 1] = r.target_base34;
    buf[i * 10 + 2] = r.c0_base34;
    buf[i * 10 + 3] = r.c1_base34;
    buf[i * 10 + 4] = r.source_rel;
    buf[i * 10 + 5] = r.target_is_red5;
    buf[i * 10 + 6] = r.consumed_has_red5;
    buf[i * 10 + 7] = r.is_moqie;
    buf[i * 10 + 8] = r.is_riichi;
    buf[i * 10 + 9] = r.cand_mode;
  }
  return PyBytes_FromStringAndSize(reinterpret_cast<char const*>(buf.data()),
                                   static_cast<Py_ssize_t>(buf.size() * sizeof(int32_t)));
}

} // namespace Kanachan::StructuredV2
