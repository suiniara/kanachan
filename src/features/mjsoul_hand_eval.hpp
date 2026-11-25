#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace Kanachan::MjsoulEval {

inline constexpr int HAKU = 31;
inline constexpr int HATSU = 32;
inline constexpr int CHUN = 33;
inline constexpr int EAST = 27;
inline constexpr int SOUTH = 28;
inline constexpr int WEST = 29;
inline constexpr int NORTH = 30;

enum class MeldKind : std::uint8_t {
  CHI = 0,
  PON = 1,
  KAN = 2, // covers ankan (opened=false), daiminkan/kakan (opened=true)
};

struct MeldSpec {
  MeldKind kind;
  bool opened;
  std::array<std::uint8_t, 3> tiles34{};
};

struct EvalConfig {
  bool is_tsumo = false;
  bool is_riichi = false;
  bool is_ippatsu = false;
  bool is_rinshan = false;
  bool is_chankan = false;
  bool is_haitei = false;
  bool is_houtei = false;
  bool is_daburu_riichi = false;
  bool is_tenhou = false;
  bool is_chiihou = false;
  int player_wind = EAST; // 27..30
  int round_wind = EAST;  // 27..30
};

struct HandEvalResult {
  bool valid = false;
  int han = 0;
  int fu = 0;
  bool is_yakuman = false;
};

struct SettlementResult {
  std::array<int32_t, 4> post_scores{};
  int32_t winner_gain = 0;
  int post_rank = 0; // 0..3 (0 = 1st)
  bool would_end_game = false;
};

namespace detail {

inline bool is_honor(int const t) noexcept { return t >= 27; }
inline bool is_terminal(int const t) noexcept {
  return t == 0 || t == 8 || t == 9 || t == 17 || t == 18 || t == 26;
}
inline bool is_terminal_or_honor(int const t) noexcept {
  return is_honor(t) || is_terminal(t);
}
inline bool is_man(int const t) noexcept { return t <= 8; }
inline bool is_pin(int const t) noexcept { return t >= 9 && t <= 17; }
inline bool is_sou(int const t) noexcept { return t >= 18 && t <= 26; }
inline int simplify(int const t) noexcept { return t - 9 * (t / 9); }

struct Group {
  std::uint8_t len = 0; // 2 or 3
  std::array<std::uint8_t, 3> t{};
  bool is_chi() const noexcept {
    return len == 3 && t[0] + 1 == t[1] && t[1] + 1 == t[2];
  }
  bool is_pon() const noexcept {
    return len == 3 && t[0] == t[1] && t[1] == t[2];
  }
  bool is_pair() const noexcept { return len == 2; }
  bool contains(int const tile) const noexcept {
    for (std::uint8_t i = 0; i < len; ++i) {
      if (t[i] == tile) return true;
    }
    return false;
  }
  bool operator==(Group const& other) const noexcept {
    return len == other.len && t == other.t;
  }
  bool operator<(Group const& other) const noexcept {
    if (len != other.len) {
      return t[0] != other.t[0] ? t[0] < other.t[0] : len < other.len;
    }
    return t < other.t;
  }
};

struct HandDivision {
  std::uint8_t count = 0; // 5 for standard, 7 for chiitoitsu
  std::array<Group, 7> groups{};
  bool operator==(HandDivision const& o) const noexcept {
    if (count != o.count) return false;
    for (std::uint8_t i = 0; i < count; ++i) {
      if (!(groups[i] == o.groups[i])) return false;
    }
    return true;
  }
};

inline void recurse_suit(
    std::array<int, 9>& cnt,
    int const suit_offset,
    int const min_code, // >= 0: 0..8 = pon at r, 9..15 = chi starting at r
    std::vector<Group>& cur,
    std::vector<std::vector<Group>>& out) {
  int first = -1;
  for (int i = 0; i < 9; ++i) {
    if (cnt[i] > 0) {
      first = i;
      break;
    }
  }
  if (first < 0) {
    out.push_back(cur);
    return;
  }
  // Try pon of first
  if (cnt[first] >= 3) {
    int const code = first * 2;
    if (code >= min_code) {
      cnt[first] -= 3;
      std::uint8_t const b = static_cast<std::uint8_t>(suit_offset + first);
      cur.push_back(Group{3, {b, b, b}});
      recurse_suit(cnt, suit_offset, code, cur, out);
      cur.pop_back();
      cnt[first] += 3;
    }
  }
  // Try chi starting at first
  if (first <= 6 && cnt[first + 1] > 0 && cnt[first + 2] > 0) {
    int const code = first * 2 + 1;
    if (code >= min_code) {
      --cnt[first];
      --cnt[first + 1];
      --cnt[first + 2];
      std::uint8_t const b = static_cast<std::uint8_t>(suit_offset + first);
      cur.push_back(Group{3, {b, static_cast<std::uint8_t>(b + 1), static_cast<std::uint8_t>(b + 2)}});
      recurse_suit(cnt, suit_offset, code, cur, out);
      cur.pop_back();
      ++cnt[first];
      ++cnt[first + 1];
      ++cnt[first + 2];
    }
  }
}

inline int round_fu(int const fu) noexcept { return ((fu + 9) / 10) * 10; }

inline EvalConfig decode_tool_config(long const tool_config) {
  EvalConfig c;
  c.is_tsumo = (tool_config & (1L << 0)) != 0;
  c.is_riichi = (tool_config & (1L << 1)) != 0;
  c.is_ippatsu = (tool_config & (1L << 2)) != 0;
  c.is_rinshan = (tool_config & (1L << 3)) != 0;
  c.is_chankan = (tool_config & (1L << 4)) != 0;
  c.is_haitei = (tool_config & (1L << 5)) != 0;
  c.is_houtei = (tool_config & (1L << 6)) != 0;
  c.is_daburu_riichi = (tool_config & (1L << 7)) != 0;
  c.is_tenhou = (tool_config & (1L << 8)) != 0;
  c.is_chiihou = (tool_config & (1L << 9)) != 0;
  if (tool_config & (1L << 10)) c.player_wind = EAST;
  else if (tool_config & (1L << 11)) c.player_wind = SOUTH;
  else if (tool_config & (1L << 12)) c.player_wind = WEST;
  else if (tool_config & (1L << 13)) c.player_wind = NORTH;
  if (tool_config & (1L << 14)) c.round_wind = EAST;
  else if (tool_config & (1L << 15)) c.round_wind = SOUTH;
  else if (tool_config & (1L << 16)) c.round_wind = WEST;
  else if (tool_config & (1L << 17)) c.round_wind = NORTH;
  return c;
}

inline int dora_hits(int const tile34, std::array<int, 34> const& dora_ind34) noexcept {
  int hits = 0;
  for (int ind = 0; ind < 34; ++ind) {
    int const c = dora_ind34[ind];
    if (c == 0) continue;
    int target = -1;
    if (ind < 27) {
      int const s = ind / 9;
      int const r = ind % 9;
      target = s * 9 + (r == 8 ? 0 : r + 1);
    } else if (ind < 31) {
      target = 27 + ((ind - 27 + 1) % 4);
    } else {
      target = 31 + ((ind - 31 + 1) % 3);
    }
    if (tile34 == target) hits += c;
  }
  return hits;
}

} // namespace detail

/**
 * Evaluate a completed 14-tile (plus optional 4th-tile kan counts via melds)
 * Mahjong Soul hand strictly matching kanachan.simulation.Tool / mahjong==1.1.11.
 *
 * `total34`: closed tiles (including `win_tile34`) + 3 * melds.size() == 14,
 *            and wait: Note for ankan/daiminkan/kakan, `total34` uses 3 tiles
 *            per meld (just like `mahjong`'s 14-tile `tiles` array!), while
 *            `dora_count` / `red5_count` includes ALL tiles including the 4th kan tile.
 * Actually to make it foolproof from physical state: pass:
 *  - `closed34`: closed tiles ONLY (3k+2, already includes `win_tile34`)
 *  - `melds`: up to 4 melds (opened/closed chi/pon/kan)
 *  - `win_tile34`: base34 of winning tile (guaranteed in `closed34 >= 1`)
 *  - `total_dora`: precomputed sum of normal dora hits across closed + all meld tiles (incl 4th kan tile)
 *  - `total_red5`: precomputed count of red 5s across closed + all meld tiles (incl 4th kan tile)
 *  - `cfg`: `EvalConfig`
 */
inline HandEvalResult eval_mjsoul_hand(
    std::array<int, 34> const& closed34,
    std::vector<MeldSpec> const& melds,
    int const win_tile34,
    int const total_dora,
    int const total_red5,
    EvalConfig const& cfg) {
  using namespace detail;
  if (win_tile34 < 0 || win_tile34 >= 34 || closed34[win_tile34] <= 0) {
    return {};
  }

  std::array<int, 34> tiles34 = closed34;
  bool is_open_hand = false;
  int total_closed = 0;
  for (int i = 0; i < 34; ++i) total_closed += closed34[i];
  if (total_closed + static_cast<int>(melds.size()) * 3 != 14) {
    return {};
  }
  for (auto const& m : melds) {
    if (m.opened) is_open_hand = true;
    for (int j = 0; j < 3; ++j) {
      ++tiles34[m.tiles34[j]];
    }
  }

  if (cfg.is_riichi && is_open_hand) return {};
  if (cfg.is_daburu_riichi && is_open_hand) return {};
  if (cfg.is_ippatsu && (is_open_hand || (!cfg.is_riichi && !cfg.is_daburu_riichi))) return {};

  HandEvalResult best{};
  auto consider = [&](int const raw_han, int const fu, bool const is_yakuman) {
    int final_han = raw_han;
    if (!is_yakuman && final_han >= 13 && final_han < 26) {
      final_han = 13; // kanachan Tool.py clamps 13..25 non-yakuman to 13
    }
    if (!best.valid || final_han > best.han || (final_han == best.han && fu > best.fu)) {
      best.valid = true;
      best.han = final_han;
      best.fu = fu;
      best.is_yakuman = is_yakuman;
    }
  };

  // 1. Exception hand: Kokushi Musou
  if (!is_open_hand) {
    constexpr std::array<int, 13> kokushi_tiles{0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33};
    int prod = 1;
    for (int idx : kokushi_tiles) {
      prod *= tiles34[idx];
    }
    if (prod == 2) {
      int han = (tiles34[win_tile34] == 2) ? 26 : 13;
      if (cfg.is_tenhou) han += 13;
      if (cfg.is_chiihou) han += 13;
      consider(han, 0, true);
    }
  }

  // 2. Decompose into Standard hands (4 mentsu + 1 pair) and Chiitoitsu (7 pairs)
  std::vector<HandDivision> divisions;

  // Check Chiitoitsu (only possible when melds is empty)
  if (melds.empty()) {
    int n_pairs = 0;
    for (int t = 0; t < 34; ++t) {
      if (closed34[t] >= 2) ++n_pairs;
    }
    if (n_pairs == 7) {
      HandDivision div{};
      div.count = 7;
      int idx = 0;
      for (int t = 0; t < 34; ++t) {
        if (closed34[t] >= 2) {
          div.groups[idx++] = Group{2, {static_cast<std::uint8_t>(t), static_cast<std::uint8_t>(t), 0}};
        }
      }
      divisions.push_back(div);
    }
  }

  // Standard 4 sets + 1 pair over closed34
  for (int p = 0; p < 34; ++p) {
    if (is_honor(p) && closed34[p] != 2) continue;
    if (closed34[p] < 2) continue;

    std::array<int, 34> rem = closed34;
    rem[p] -= 2;

    // Honors in remaining closed must all be multiples of 3
    std::vector<Group> honor_groups;
    bool honor_ok = true;
    for (int h = 27; h < 34; ++h) {
      if (rem[h] == 3) {
        std::uint8_t const hb = static_cast<std::uint8_t>(h);
        honor_groups.push_back(Group{3, {hb, hb, hb}});
      } else if (rem[h] != 0) {
        honor_ok = false;
        break;
      }
    }
    if (!honor_ok) continue;

    std::array<int, 9> c0{}, c1{}, c2{};
    int s0 = 0, s1 = 0, s2 = 0;
    for (int i = 0; i < 9; ++i) {
      c0[i] = rem[i]; s0 += c0[i];
      c1[i] = rem[9 + i]; s1 += c1[i];
      c2[i] = rem[18 + i]; s2 += c2[i];
    }
    if (s0 % 3 != 0 || s1 % 3 != 0 || s2 % 3 != 0) continue;

    std::vector<std::vector<Group>> p0, p1, p2;
    std::vector<Group> scratch;
    if (s0 == 0) p0.emplace_back();
    else recurse_suit(c0, 0, 0, scratch, p0);
    if (p0.empty()) continue;

    if (s1 == 0) p1.emplace_back();
    else { scratch.clear(); recurse_suit(c1, 9, 0, scratch, p1); }
    if (p1.empty()) continue;

    if (s2 == 0) p2.emplace_back();
    else { scratch.clear(); recurse_suit(c2, 18, 0, scratch, p2); }
    if (p2.empty()) continue;

    for (auto const& g0 : p0) {
      for (auto const& g1 : p1) {
        for (auto const& g2 : p2) {
          HandDivision div{};
          div.count = 5;
          int idx = 0;
          std::uint8_t const pb = static_cast<std::uint8_t>(p);
          div.groups[idx++] = Group{2, {pb, pb, 0}};
          for (auto const& g : g0) div.groups[idx++] = g;
          for (auto const& g : g1) div.groups[idx++] = g;
          for (auto const& g : g2) div.groups[idx++] = g;
          for (auto const& g : honor_groups) div.groups[idx++] = g;
          for (auto const& m : melds) {
            div.groups[idx++] = Group{3, m.tiles34};
          }
          std::sort(div.groups.begin(), div.groups.begin() + 5);
          bool dup = false;
          for (auto const& prev : divisions) {
            if (prev == div) { dup = true; break; }
          }
          if (!dup) divisions.push_back(div);
        }
      }
    }
  }

  // Evaluate each division and each unique closed win_group
  for (auto const& div : divisions) {
    bool const is_chiitoitsu = (div.count == 7);

    // Closed set items: remove one matching group per opened meld
    std::vector<Group> open_meld_groups;
    std::vector<Group> chi_meld_groups;
    for (auto const& m : melds) {
      if (m.opened) {
        open_meld_groups.push_back(Group{3, m.tiles34});
      }
      if (m.kind == MeldKind::CHI) {
        chi_meld_groups.push_back(Group{3, m.tiles34});
      }
    }

    std::vector<Group> open_rem = open_meld_groups;
    std::vector<Group> closed_items;
    for (int i = 0; i < div.count; ++i) {
      Group const& g = div.groups[i];
      auto it = std::find(open_rem.begin(), open_rem.end(), g);
      if (it == open_rem.end()) {
        closed_items.push_back(g);
      } else {
        open_rem.erase(it);
      }
    }

    std::vector<Group> win_groups;
    for (auto const& g : closed_items) {
      if (g.contains(win_tile34)) {
        if (std::find(win_groups.begin(), win_groups.end(), g) == win_groups.end()) {
          win_groups.push_back(g);
        }
      }
    }

    // Closed chi sets for fu.py (excludes melds with type == CHI)
    std::vector<Group> chi_rem = chi_meld_groups;
    std::vector<Group> closed_chi_sets_fu;
    for (int i = 0; i < div.count; ++i) {
      Group const& g = div.groups[i];
      auto it = std::find(chi_rem.begin(), chi_rem.end(), g);
      if (it == chi_rem.end()) {
        closed_chi_sets_fu.push_back(g);
      } else {
        chi_rem.erase(it);
      }
    }

    std::vector<Group> pon_sets;
    std::vector<Group> chi_sets;
    Group pair_group{};
    for (int i = 0; i < div.count; ++i) {
      Group const& g = div.groups[i];
      if (g.is_pair()) pair_group = g;
      else if (g.is_pon()) pon_sets.push_back(g);
      else if (g.is_chi()) chi_sets.push_back(g);
    }

    for (auto const& win_group : win_groups) {
      if (is_chiitoitsu && is_open_hand) continue;

      // ---- Calculate Fu (strictly matching mahjong/hand_calculating/fu.py) ----
      int fu = 0;
      bool has_non_base_fu = false;
      if (is_chiitoitsu) {
        fu = 25;
      } else {
        int fu_sum = 0;
        if (win_group.is_chi() &&
            std::find(closed_chi_sets_fu.begin(), closed_chi_sets_fu.end(), win_group) !=
            closed_chi_sets_fu.end()) {
          int const rank = simplify(win_tile34);
          bool const has_term = is_terminal(win_group.t[0]) || is_terminal(win_group.t[1]) ||
                                is_terminal(win_group.t[2]);
          if (has_term) {
            if (rank == 2 && win_group.t[2] == win_tile34) {
              fu_sum += 2; // 1-2-3 penchan
            } else if (rank == 6 && win_group.t[0] == win_tile34) {
              fu_sum += 2; // 7-8-9 penchan
            }
          }
          if (win_group.t[1] == win_tile34) {
            fu_sum += 2; // kanchan
          }
        }
        // Valued pair (can count twice if double wind!)
        int valued_cnt = 0;
        int const pt = pair_group.t[0];
        if (pt == HAKU || pt == HATSU || pt == CHUN) ++valued_cnt;
        if (pt == cfg.player_wind) ++valued_cnt;
        if (pt == cfg.round_wind) ++valued_cnt;
        fu_sum += 2 * valued_cnt;

        if (win_group.is_pair()) {
          fu_sum += 2;
        }

        // Pons/Kans fu
        for (auto const& item : pon_sets) {
          MeldSpec const* match_meld = nullptr;
          for (auto const& m : melds) {
            if (m.tiles34 == item.t) {
              match_meld = &m;
              break;
            }
          }
          bool was_open = match_meld ? match_meld->opened : false;
          bool const is_kan = match_meld ? (match_meld->kind == MeldKind::KAN) : false;
          bool const is_yao = is_terminal_or_honor(item.t[0]);
          if (!cfg.is_tsumo && item == win_group) {
            was_open = true;
          }
          if (is_yao) {
            if (is_kan) fu_sum += was_open ? 16 : 32;
            else fu_sum += was_open ? 4 : 8;
          } else {
            if (is_kan) fu_sum += was_open ? 8 : 16;
            else fu_sum += was_open ? 2 : 4;
          }
        }

        has_non_base_fu = (fu_sum > 0);
        if (cfg.is_tsumo && fu_sum > 0) {
          fu_sum += 2; // fu_for_pinfu_tsumo = False in mahjong OptionalRules defaults
        }
        if (is_open_hand && fu_sum == 0) {
          fu_sum += 2; // fu_for_open_pinfu = True
        }
        fu_sum += (is_open_hand || cfg.is_tsumo) ? 20 : 30;
        fu = round_fu(fu_sum);
      }

      bool const is_pinfu = (!has_non_base_fu && !is_chiitoitsu && !is_open_hand);

      // ---- Evaluate Yakuman first ----
      int yakuman_han = 0;
      if (!is_open_hand && cfg.is_tenhou) yakuman_han += 13;
      if (!is_open_hand && cfg.is_chiihou) yakuman_han += 13;

      // Tsuuiisou (checks ALL groups incl chiitoitsu)
      {
        bool all_honor = true;
        for (int i = 0; i < div.count; ++i) {
          if (!is_honor(div.groups[i].t[0])) { all_honor = false; break; }
        }
        if (all_honor) yakuman_han += 13;
      }
      // Chinroutou
      {
        bool all_term = true;
        for (int i = 0; i < div.count; ++i) {
          for (int j = 0; j < div.groups[i].len; ++j) {
            if (!is_terminal(div.groups[i].t[j])) { all_term = false; break; }
          }
          if (!all_term) break;
        }
        if (all_term) yakuman_han += 13;
      }
      // Ryuuiisou
      {
        bool all_green = true;
        for (int i = 0; i < div.count; ++i) {
          for (int j = 0; j < div.groups[i].len; ++j) {
            int const x = div.groups[i].t[j];
            if (!(x == 19 || x == 20 || x == 21 || x == 23 || x == 25 || x == HATSU)) {
              all_green = false;
              break;
            }
          }
          if (!all_green) break;
        }
        if (all_green) yakuman_han += 13;
      }

      if (!pon_sets.empty()) {
        // Daisangen
        int dragon_pons = 0;
        int wind_pons = 0;
        for (auto const& ps : pon_sets) {
          if (ps.t[0] == HAKU || ps.t[0] == HATSU || ps.t[0] == CHUN) ++dragon_pons;
          if (ps.t[0] >= EAST && ps.t[0] <= NORTH) ++wind_pons;
        }
        if (dragon_pons == 3) yakuman_han += 13;
        if (wind_pons == 4) yakuman_han += 26;
        else if (wind_pons == 3 && pair_group.t[0] >= EAST && pair_group.t[0] <= NORTH) {
          yakuman_han += 13;
        }

        // Chuuren Poutou (no melds at all!)
        if (melds.empty()) {
          int suit_mask = 0;
          bool has_h = false;
          for (int i = 0; i < div.count; ++i) {
            int const x = div.groups[i].t[0];
            if (is_man(x)) suit_mask |= 1;
            else if (is_pin(x)) suit_mask |= 2;
            else if (is_sou(x)) suit_mask |= 4;
            else has_h = true;
          }
          if (!has_h && (suit_mask == 1 || suit_mask == 2 || suit_mask == 4)) {
            int const off = (suit_mask == 1) ? 0 : (suit_mask == 2 ? 9 : 18);
            bool chuuren = (tiles34[off] >= 3 && tiles34[off + 8] >= 3);
            for (int r = 1; r < 8 && chuuren; ++r) {
              if (tiles34[off + r] < 1) chuuren = false;
            }
            if (chuuren) {
              bool const daburu = (tiles34[win_tile34] == 2 || tiles34[win_tile34] == 4);
              yakuman_han += daburu ? 26 : 13;
            }
          }
        }

        // Suuankou / Suuankou Tanki
        if (!is_open_hand) {
          int closed_pons = 0;
          for (auto const& ps : pon_sets) {
            if (!cfg.is_tsumo && ps.contains(win_tile34)) continue;
            ++closed_pons;
          }
          if (closed_pons == 4) {
            yakuman_han += (tiles34[win_tile34] == 2) ? 26 : 13;
          }
        }

        // Suukantsu
        int kan_cnt = 0;
        for (auto const& m : melds) {
          if (m.kind == MeldKind::KAN) ++kan_cnt;
        }
        if (kan_cnt == 4) yakuman_han += 13;
      }

      if (yakuman_han > 0) {
        consider(yakuman_han, fu, true);
        continue;
      }

      // ---- Regular Yaku ----
      int han = 0;
      if (cfg.is_tsumo && !is_open_hand) ++han; // Menzen Tsumo
      if (is_pinfu) ++han;                      // Pinfu
      if (is_chiitoitsu) han += 2;              // Chiitoitsu

      // Tanyao (has_open_tanyao = True)
      {
        bool all_simples = true;
        for (int i = 0; i < div.count; ++i) {
          for (int j = 0; j < div.groups[i].len; ++j) {
            if (is_terminal_or_honor(div.groups[i].t[j])) { all_simples = false; break; }
          }
          if (!all_simples) break;
        }
        if (all_simples) ++han;
      }

      if (cfg.is_riichi && !cfg.is_daburu_riichi) ++han;
      if (cfg.is_daburu_riichi) han += 2;
      if (cfg.is_ippatsu) ++han;
      if (cfg.is_rinshan) ++han;
      if (cfg.is_chankan) ++han;
      if (cfg.is_haitei) ++han;
      if (cfg.is_houtei) ++han;

      // Honitsu / Chinitsu
      {
        int ms = 0, ps = 0, ss = 0, hs = 0;
        for (int i = 0; i < div.count; ++i) {
          int const x = div.groups[i].t[0];
          if (is_man(x)) ++ms;
          else if (is_pin(x)) ++ps;
          else if (is_sou(x)) ++ss;
          else ++hs;
        }
        int const num_suits = (ms > 0) + (ps > 0) + (ss > 0);
        if (num_suits == 1 && hs > 0) han += is_open_hand ? 2 : 3;
        if (num_suits == 1 && hs == 0) han += is_open_hand ? 5 : 6;
      }

      // Honroutou
      {
        bool all_th = true;
        for (int i = 0; i < div.count; ++i) {
          for (int j = 0; j < div.groups[i].len; ++j) {
            if (!is_terminal_or_honor(div.groups[i].t[j])) { all_th = false; break; }
          }
          if (!all_th) break;
        }
        if (all_th) han += 2;
      }

      if (!chi_sets.empty()) {
        // Chanta & Junchan
        int term_sets = 0, honor_sets = 0;
        for (int i = 0; i < div.count; ++i) {
          bool has_t = false, has_h = false;
          for (int j = 0; j < div.groups[i].len; ++j) {
            if (is_terminal(div.groups[i].t[j])) has_t = true;
            if (is_honor(div.groups[i].t[j])) has_h = true;
          }
          if (has_t) ++term_sets;
          if (has_h) ++honor_sets;
        }
        if (term_sets == 5) {
          han += is_open_hand ? 2 : 3; // Junchan
        } else if (term_sets + honor_sets == 5 && term_sets != 0 && honor_sets != 0) {
          han += is_open_hand ? 1 : 2; // Chanta
        }

        // Ittsu
        for (int s = 0; s < 3; ++s) {
          bool h1 = false, h4 = false, h7 = false;
          int const off = s * 9;
          for (auto const& cs : chi_sets) {
            if (cs.t[0] == off + 0) h1 = true;
            if (cs.t[0] == off + 3) h4 = true;
            if (cs.t[0] == off + 6) h7 = true;
          }
          if (h1 && h4 && h7) {
            han += is_open_hand ? 1 : 2;
            break;
          }
        }

        // Ryanpeikou / Iipeikou
        if (!is_open_hand) {
          int identical_ge2 = 0;
          for (std::size_t a = 0; a < chi_sets.size(); ++a) {
            int cnt = 0;
            for (std::size_t b = 0; b < chi_sets.size(); ++b) {
              if (chi_sets[a] == chi_sets[b]) ++cnt;
            }
            if (cnt >= 2) ++identical_ge2;
          }
          if (identical_ge2 == 4) han += 3;
          else if (identical_ge2 >= 2) han += 1;
        }

        // Sanshoku Doujun
        for (int r = 0; r <= 6; ++r) {
          bool hm = false, hp = false, hs_ = false;
          for (auto const& cs : chi_sets) {
            if (cs.t[0] == r) hm = true;
            else if (cs.t[0] == 9 + r) hp = true;
            else if (cs.t[0] == 18 + r) hs_ = true;
          }
          if (hm && hp && hs_) {
            han += is_open_hand ? 1 : 2;
            break;
          }
        }
      }

      if (!pon_sets.empty()) {
        // Toitoi
        if (pon_sets.size() == 4) han += 2;

        // Sanankou (matches sanankou.py verbatim)
        {
          int closed_chis_with_win = 0;
          std::vector<Group> orem = open_meld_groups;
          // wait: sanankou.py checks `x not in open_sets` (which includes chi melds too!)
          for (auto const& cs : chi_sets) {
            bool is_o = false;
            for (auto const& om : open_meld_groups) {
              if (om == cs) { is_o = true; break; }
            }
            if (!is_o && cs.contains(win_tile34)) ++closed_chis_with_win;
          }
          std::vector<Group> open_pons = open_meld_groups;
          int closed_pons = 0;
          for (auto const& ps : pon_sets) {
            auto it = std::find(open_pons.begin(), open_pons.end(), ps);
            if (it != open_pons.end()) {
              open_pons.erase(it);
              continue;
            }
            if (ps.contains(win_tile34) && !cfg.is_tsumo && closed_chis_with_win == 0) {
              continue;
            }
            ++closed_pons;
          }
          if (closed_pons == 3) han += 2;
        }

        // Sanshoku Doukou
        for (int r = 0; r < 9; ++r) {
          bool hm = false, hp = false, hs_ = false;
          for (auto const& ps : pon_sets) {
            if (ps.t[0] == r) hm = true;
            else if (ps.t[0] == 9 + r) hp = true;
            else if (ps.t[0] == 18 + r) hs_ = true;
          }
          if (hm && hp && hs_) {
            han += 2;
            break;
          }
        }

        // Shousangen
        int dragon_pons = 0;
        for (auto const& ps : pon_sets) {
          int const x = ps.t[0];
          if (x == HAKU || x == HATSU || x == CHUN) ++dragon_pons;
          if (x == HAKU) ++han;
          if (x == HATSU) ++han;
          if (x == CHUN) ++han;
          if (x == cfg.player_wind) ++han;
          if (x == cfg.round_wind) ++han;
        }
        if (dragon_pons == 2 && (pair_group.t[0] == HAKU || pair_group.t[0] == HATSU || pair_group.t[0] == CHUN)) {
          han += 2;
        }

        // Sankantsu
        int kan_cnt = 0;
        for (auto const& m : melds) {
          if (m.kind == MeldKind::KAN) ++kan_cnt;
        }
        if (kan_cnt == 3) han += 2;
      }

      if (han == 0) continue; // Dora / Aka-dora CANNOT satisfy yaku requirement!
      han += total_dora + total_red5;
      consider(han, fu, false);
    }
  }

  return best;
}

inline std::array<int, 4> compute_ranks(std::array<int32_t, 4> const& scores) noexcept {
  std::array<int, 4> ranks{};
  for (int i = 0; i < 4; ++i) {
    int r = 0;
    for (int j = 0; j < 4; ++j) {
      if (i == j) continue;
      if (scores[j] > scores[i] || (scores[j] == scores[i] && j < i)) ++r;
    }
    ranks[i] = r;
  }
  return ranks;
}

inline int compute_base_points(int const han, int const fu, bool const is_yakuman) noexcept {
  if (!is_yakuman && han >= 13 && han < 26) {
    return 8000;
  }
  if (han >= 5) {
    if (han >= 78) return 48000;
    if (han >= 65) return 40000;
    if (han >= 52) return 32000;
    if (han >= 39) return 24000;
    if (han >= 26) return 16000;
    if (han >= 13) return 8000;
    if (han >= 11) return 6000;
    if (han >= 8) return 4000;
    if (han >= 6) return 3000;
    return 2000;
  }
  int const bp = fu * (1 << (2 + han));
  return bp > 2000 ? 2000 : bp;
}

inline int ceil100(int const v) noexcept { return ((v + 99) / 100) * 100; }

/**
 * Compute single-winner settlement matching src/simulation/round_state.cpp verbatim:
 *  - Normal payments: tsumo/ron using ben (ONLY +300*ben added once in settlement!)
 *  - All liqibang * 1000 goes to winner
 *  - Daisangen / Daisuushii Pao transfer (tsumo: pao pays full base + 300*ben; ron: 50/50 split + 300*ben from pao if pao != payer)
 *  - Exact `would_end_game` matching round_state.cpp#L2391-L2552
 */
inline SettlementResult eval_win_settlement(
    HandEvalResult const& hand_eval,
    bool const is_tsumo,
    int const winner_abs,
    int const loser_abs, // ignored for tsumo
    int const ju,
    int const chang,
    int const ben,
    int const liqibang,
    bool const is_tonpuusen,
    std::array<int32_t, 4> const& pre_scores,
    int const pao_daisangen_payer = -1,
    int const pao_daisuushii_payer = -1) {
  SettlementResult out{};
  out.post_scores = pre_scores;
  if (!hand_eval.valid) return out;

  int const base = compute_base_points(hand_eval.han, hand_eval.fu, hand_eval.is_yakuman);
  bool const is_dealer = (winner_abs == ju);
  int32_t winner_delta = 0;

  if (is_tsumo) {
    if (is_dealer) {
      int const pay = (hand_eval.han >= 5) ? (base * 2) : ceil100(base * 2);
      int32_t const w = pay * 3 + 300 * ben + 1000 * liqibang;
      out.post_scores[winner_abs] += w;
      winner_delta += w;
      for (int i = 0; i < 4; ++i) {
        if (i != winner_abs) out.post_scores[i] -= (pay + 100 * ben);
      }
      if (pao_daisangen_payer >= 0 && pao_daisangen_payer != winner_abs) {
        for (int i = 0; i < 4; ++i) {
          if (i == pao_daisangen_payer) out.post_scores[i] -= (32000 + 200 * ben);
          else if (i != winner_abs) out.post_scores[i] += (16000 + 100 * ben);
        }
      }
      if (pao_daisuushii_payer >= 0 && pao_daisuushii_payer != winner_abs) {
        for (int i = 0; i < 4; ++i) {
          if (i == pao_daisuushii_payer) out.post_scores[i] -= (64000 + 200 * ben);
          else if (i != winner_abs) out.post_scores[i] += (32000 + 100 * ben);
        }
      }
    } else {
      int const pay_dealer = (hand_eval.han >= 5) ? (base * 2) : ceil100(base * 2);
      int const pay_child = (hand_eval.han >= 5) ? base : ceil100(base);
      int32_t const w = pay_dealer + 2 * pay_child + 300 * ben + 1000 * liqibang;
      out.post_scores[winner_abs] += w;
      winner_delta += w;
      for (int i = 0; i < 4; ++i) {
        if (i == winner_abs) continue;
        if (i == ju) out.post_scores[i] -= (pay_dealer + 100 * ben);
        else out.post_scores[i] -= (pay_child + 100 * ben);
      }
      if (pao_daisangen_payer >= 0 && pao_daisangen_payer != winner_abs) {
        for (int i = 0; i < 4; ++i) {
          if (i == pao_daisangen_payer) {
            out.post_scores[i] -= ((i == ju ? 16000 : 24000) + 200 * ben);
          } else if (i != winner_abs) {
            out.post_scores[i] += ((i == ju ? 16000 : 8000) + 100 * ben);
          }
        }
      }
      if (pao_daisuushii_payer >= 0 && pao_daisuushii_payer != winner_abs) {
        for (int i = 0; i < 4; ++i) {
          if (i == pao_daisuushii_payer) {
            out.post_scores[i] -= ((i == ju ? 32000 : 48000) + 200 * ben);
          } else if (i != winner_abs) {
            out.post_scores[i] += ((i == ju ? 32000 : 16000) + 100 * ben);
          }
        }
      }
    }
  } else {
    int const pay = (hand_eval.han >= 5)
        ? (is_dealer ? base * 6 : base * 4)
        : (is_dealer ? ceil100(base * 6) : ceil100(base * 4));
    int32_t const w = pay + 300 * ben + 1000 * liqibang;
    out.post_scores[winner_abs] += w;
    out.post_scores[loser_abs] -= (pay + 300 * ben);
    winner_delta += w;

    if (pao_daisangen_payer >= 0 && pao_daisangen_payer != winner_abs) {
      int32_t const sb = is_dealer ? 24000 : 16000;
      out.post_scores[pao_daisangen_payer] -= (sb + 300 * ben);
      out.post_scores[loser_abs] += (sb + 300 * ben);
    }
    if (pao_daisuushii_payer >= 0 && pao_daisuushii_payer != winner_abs) {
      int32_t const sb = is_dealer ? 48000 : 32000;
      out.post_scores[pao_daisuushii_payer] -= (sb + 300 * ben);
      out.post_scores[loser_abs] += (sb + 300 * ben);
    }
  }

  out.winner_gain = winner_delta;
  auto const ranks = compute_ranks(out.post_scores);
  out.post_rank = ranks[winner_abs];

  // Tobi end check
  for (int i = 0; i < 4; ++i) {
    if (out.post_scores[i] < 0) {
      out.would_end_game = true;
      return out;
    }
  }

  bool suddendeath = false;
  for (int i = 0; i < 4; ++i) {
    if (out.post_scores[i] >= 30000) {
      suddendeath = true;
      break;
    }
  }
  int const dealer_rank = ranks[ju];
  if (is_tonpuusen) {
    if (chang == 1 && ju == 3) {
      if (suddendeath) {
        out.would_end_game = (!is_dealer) || (dealer_rank == 0);
      } else {
        out.would_end_game = !is_dealer;
      }
    } else if ((chang == 0 && ju == 3) || chang == 1) {
      if (suddendeath) {
        out.would_end_game = (!is_dealer) || (dealer_rank == 0);
      } else {
        out.would_end_game = false;
      }
    } else {
      out.would_end_game = false;
    }
  } else {
    if (chang == 2 && ju == 3) {
      if (suddendeath) {
        out.would_end_game = (!is_dealer) || (dealer_rank == 0);
      } else {
        out.would_end_game = !is_dealer;
      }
    } else if ((chang == 1 && ju == 3) || chang == 2) {
      if (suddendeath) {
        out.would_end_game = (!is_dealer) || (dealer_rank == 0);
      } else {
        out.would_end_game = false;
      }
    } else {
      out.would_end_game = false;
    }
  }
  return out;
}

} // namespace Kanachan::MjsoulEval
