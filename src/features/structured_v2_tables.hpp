#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace Kanachan::StructuredV2 {

inline constexpr int NUM_TYPES_OF_ACTIONS = 546;
inline constexpr int PAD_ACTION_ID = 546;
inline constexpr int NULL_TILE_IDX = 34;
inline constexpr int TILE37_NONE = 37;
inline constexpr int NULL_SOURCE_IDX = 4;
inline constexpr int MAX_CANDIDATES = 32;
inline constexpr int MAX_PROGRESSION_LEN = 109;
inline constexpr int TILE_FEATS_DIM = 70;
inline constexpr int EVENT_BYTES = 10;
inline constexpr int GLOBAL_FEATS_DIM = 48;
inline constexpr int PLAYER_FEATS_DIM = 32;
inline constexpr int CAND_DYN_DIM = 8;

inline constexpr std::array<int, 37> OFFSETS_136 = {
  0, 1, 5, 9, 13, 17, 20, 24, 28, 32,
  36, 37, 41, 45, 49, 53, 56, 60, 64, 68,
  72, 73, 77, 81, 85, 89, 92, 96, 100, 104,
  108, 112, 116, 120, 124, 128, 132
};

inline constexpr std::array<std::array<std::uint_fast8_t, 4u>, 90u> CHI_ENCODE = {
  std::array<std::uint_fast8_t, 4u>{ 2u, 3u, 1u, 4u },
  std::array<std::uint_fast8_t, 4u>{ 1u, 3u, 2u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 3u, 4u, 2u, 5u },
  std::array<std::uint_fast8_t, 4u>{ 1u, 2u, 3u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 2u, 4u, 3u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 4u, 5u, 3u, 6u },
  std::array<std::uint_fast8_t, 4u>{ 4u, 0u, 3u, 6u },
  std::array<std::uint_fast8_t, 4u>{ 2u, 3u, 4u, 1u },
  std::array<std::uint_fast8_t, 4u>{ 3u, 5u, 4u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 3u, 0u, 4u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 5u, 6u, 4u, 7u },
  std::array<std::uint_fast8_t, 4u>{ 0u, 6u, 4u, 7u },
  std::array<std::uint_fast8_t, 4u>{ 3u, 4u, 5u, 2u },
  std::array<std::uint_fast8_t, 4u>{ 3u, 4u, 0u, 2u },
  std::array<std::uint_fast8_t, 4u>{ 4u, 6u, 5u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 4u, 6u, 0u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 6u, 7u, 5u, 8u },
  std::array<std::uint_fast8_t, 4u>{ 6u, 7u, 0u, 8u },
  std::array<std::uint_fast8_t, 4u>{ 4u, 5u, 6u, 3u },
  std::array<std::uint_fast8_t, 4u>{ 4u, 0u, 6u, 3u },
  std::array<std::uint_fast8_t, 4u>{ 5u, 7u, 6u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 0u, 7u, 6u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 7u, 8u, 6u, 9u },
  std::array<std::uint_fast8_t, 4u>{ 5u, 6u, 7u, 4u },
  std::array<std::uint_fast8_t, 4u>{ 0u, 6u, 7u, 4u },
  std::array<std::uint_fast8_t, 4u>{ 6u, 8u, 7u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 8u, 9u, 7u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 6u, 7u, 8u, 5u },
  std::array<std::uint_fast8_t, 4u>{ 7u, 9u, 8u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 7u, 8u, 9u, 6u },
  std::array<std::uint_fast8_t, 4u>{ 12u, 13u, 11u, 14u },
  std::array<std::uint_fast8_t, 4u>{ 11u, 13u, 12u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 13u, 14u, 12u, 15u },
  std::array<std::uint_fast8_t, 4u>{ 11u, 12u, 13u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 12u, 14u, 13u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 14u, 15u, 13u, 16u },
  std::array<std::uint_fast8_t, 4u>{ 14u, 10u, 13u, 16u },
  std::array<std::uint_fast8_t, 4u>{ 12u, 13u, 14u, 11u },
  std::array<std::uint_fast8_t, 4u>{ 13u, 15u, 14u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 13u, 10u, 14u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 15u, 16u, 14u, 17u },
  std::array<std::uint_fast8_t, 4u>{ 10u, 16u, 14u, 17u },
  std::array<std::uint_fast8_t, 4u>{ 13u, 14u, 15u, 12u },
  std::array<std::uint_fast8_t, 4u>{ 13u, 14u, 10u, 12u },
  std::array<std::uint_fast8_t, 4u>{ 14u, 16u, 15u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 14u, 16u, 10u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 16u, 17u, 15u, 18u },
  std::array<std::uint_fast8_t, 4u>{ 16u, 17u, 10u, 18u },
  std::array<std::uint_fast8_t, 4u>{ 14u, 15u, 16u, 13u },
  std::array<std::uint_fast8_t, 4u>{ 14u, 10u, 16u, 13u },
  std::array<std::uint_fast8_t, 4u>{ 15u, 17u, 16u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 10u, 17u, 16u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 17u, 18u, 16u, 19u },
  std::array<std::uint_fast8_t, 4u>{ 15u, 16u, 17u, 14u },
  std::array<std::uint_fast8_t, 4u>{ 10u, 16u, 17u, 14u },
  std::array<std::uint_fast8_t, 4u>{ 16u, 18u, 17u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 18u, 19u, 17u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 16u, 17u, 18u, 15u },
  std::array<std::uint_fast8_t, 4u>{ 17u, 19u, 18u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 17u, 18u, 19u, 16u },
  std::array<std::uint_fast8_t, 4u>{ 22u, 23u, 21u, 24u },
  std::array<std::uint_fast8_t, 4u>{ 21u, 23u, 22u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 23u, 24u, 22u, 25u },
  std::array<std::uint_fast8_t, 4u>{ 21u, 22u, 23u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 22u, 24u, 23u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 24u, 25u, 23u, 26u },
  std::array<std::uint_fast8_t, 4u>{ 24u, 20u, 23u, 26u },
  std::array<std::uint_fast8_t, 4u>{ 22u, 23u, 24u, 21u },
  std::array<std::uint_fast8_t, 4u>{ 23u, 25u, 24u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 23u, 20u, 24u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 25u, 26u, 24u, 27u },
  std::array<std::uint_fast8_t, 4u>{ 20u, 26u, 24u, 27u },
  std::array<std::uint_fast8_t, 4u>{ 23u, 24u, 25u, 22u },
  std::array<std::uint_fast8_t, 4u>{ 23u, 24u, 20u, 22u },
  std::array<std::uint_fast8_t, 4u>{ 24u, 26u, 25u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 24u, 26u, 20u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 26u, 27u, 25u, 28u },
  std::array<std::uint_fast8_t, 4u>{ 26u, 27u, 20u, 28u },
  std::array<std::uint_fast8_t, 4u>{ 24u, 25u, 26u, 23u },
  std::array<std::uint_fast8_t, 4u>{ 24u, 20u, 26u, 23u },
  std::array<std::uint_fast8_t, 4u>{ 25u, 27u, 26u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 20u, 27u, 26u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 27u, 28u, 26u, 29u },
  std::array<std::uint_fast8_t, 4u>{ 25u, 26u, 27u, 24u },
  std::array<std::uint_fast8_t, 4u>{ 20u, 26u, 27u, 24u },
  std::array<std::uint_fast8_t, 4u>{ 26u, 28u, 27u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 28u, 29u, 27u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 26u, 27u, 28u, 25u },
  std::array<std::uint_fast8_t, 4u>{ 27u, 29u, 28u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 27u, 28u, 29u, 26u }
};

inline constexpr std::array<std::array<std::uint_fast8_t, 3u>, 40u> PENG_ENCODE = {
  std::array<std::uint_fast8_t, 3u>{ 1u, 1u, 1u },
  std::array<std::uint_fast8_t, 3u>{ 2u, 2u, 2u },
  std::array<std::uint_fast8_t, 3u>{ 3u, 3u, 3u },
  std::array<std::uint_fast8_t, 3u>{ 4u, 4u, 4u },
  std::array<std::uint_fast8_t, 3u>{ 5u, 5u, 5u },
  std::array<std::uint_fast8_t, 3u>{ 0u, 5u, 5u },
  std::array<std::uint_fast8_t, 3u>{ 5u, 5u, 0u },
  std::array<std::uint_fast8_t, 3u>{ 6u, 6u, 6u },
  std::array<std::uint_fast8_t, 3u>{ 7u, 7u, 7u },
  std::array<std::uint_fast8_t, 3u>{ 8u, 8u, 8u },
  std::array<std::uint_fast8_t, 3u>{ 9u, 9u, 9u },
  std::array<std::uint_fast8_t, 3u>{ 11u, 11u, 11u },
  std::array<std::uint_fast8_t, 3u>{ 12u, 12u, 12u },
  std::array<std::uint_fast8_t, 3u>{ 13u, 13u, 13u },
  std::array<std::uint_fast8_t, 3u>{ 14u, 14u, 14u },
  std::array<std::uint_fast8_t, 3u>{ 15u, 15u, 15u },
  std::array<std::uint_fast8_t, 3u>{ 10u, 15u, 15u },
  std::array<std::uint_fast8_t, 3u>{ 15u, 15u, 10u },
  std::array<std::uint_fast8_t, 3u>{ 16u, 16u, 16u },
  std::array<std::uint_fast8_t, 3u>{ 17u, 17u, 17u },
  std::array<std::uint_fast8_t, 3u>{ 18u, 18u, 18u },
  std::array<std::uint_fast8_t, 3u>{ 19u, 19u, 19u },
  std::array<std::uint_fast8_t, 3u>{ 21u, 21u, 21u },
  std::array<std::uint_fast8_t, 3u>{ 22u, 22u, 22u },
  std::array<std::uint_fast8_t, 3u>{ 23u, 23u, 23u },
  std::array<std::uint_fast8_t, 3u>{ 24u, 24u, 24u },
  std::array<std::uint_fast8_t, 3u>{ 25u, 25u, 25u },
  std::array<std::uint_fast8_t, 3u>{ 20u, 25u, 25u },
  std::array<std::uint_fast8_t, 3u>{ 25u, 25u, 20u },
  std::array<std::uint_fast8_t, 3u>{ 26u, 26u, 26u },
  std::array<std::uint_fast8_t, 3u>{ 27u, 27u, 27u },
  std::array<std::uint_fast8_t, 3u>{ 28u, 28u, 28u },
  std::array<std::uint_fast8_t, 3u>{ 29u, 29u, 29u },
  std::array<std::uint_fast8_t, 3u>{ 30u, 30u, 30u },
  std::array<std::uint_fast8_t, 3u>{ 31u, 31u, 31u },
  std::array<std::uint_fast8_t, 3u>{ 32u, 32u, 32u },
  std::array<std::uint_fast8_t, 3u>{ 33u, 33u, 33u },
  std::array<std::uint_fast8_t, 3u>{ 34u, 34u, 34u },
  std::array<std::uint_fast8_t, 3u>{ 35u, 35u, 35u },
  std::array<std::uint_fast8_t, 3u>{ 36u, 36u, 36u }
};

inline constexpr std::array<std::array<std::uint_fast8_t, 3u>, 37u> DAMINGGANG_ENCODE = {
  std::array<std::uint_fast8_t, 3u>{ 5u, 5u, 5u },
  std::array<std::uint_fast8_t, 3u>{ 1u, 1u, 1u },
  std::array<std::uint_fast8_t, 3u>{ 2u, 2u, 2u },
  std::array<std::uint_fast8_t, 3u>{ 3u, 3u, 3u },
  std::array<std::uint_fast8_t, 3u>{ 4u, 4u, 4u },
  std::array<std::uint_fast8_t, 3u>{ 5u, 5u, 0u },
  std::array<std::uint_fast8_t, 3u>{ 6u, 6u, 6u },
  std::array<std::uint_fast8_t, 3u>{ 7u, 7u, 7u },
  std::array<std::uint_fast8_t, 3u>{ 8u, 8u, 8u },
  std::array<std::uint_fast8_t, 3u>{ 9u, 9u, 9u },
  std::array<std::uint_fast8_t, 3u>{ 15u, 15u, 15u },
  std::array<std::uint_fast8_t, 3u>{ 11u, 11u, 11u },
  std::array<std::uint_fast8_t, 3u>{ 12u, 12u, 12u },
  std::array<std::uint_fast8_t, 3u>{ 13u, 13u, 13u },
  std::array<std::uint_fast8_t, 3u>{ 14u, 14u, 14u },
  std::array<std::uint_fast8_t, 3u>{ 15u, 15u, 10u },
  std::array<std::uint_fast8_t, 3u>{ 16u, 16u, 16u },
  std::array<std::uint_fast8_t, 3u>{ 17u, 17u, 17u },
  std::array<std::uint_fast8_t, 3u>{ 18u, 18u, 18u },
  std::array<std::uint_fast8_t, 3u>{ 19u, 19u, 19u },
  std::array<std::uint_fast8_t, 3u>{ 25u, 25u, 25u },
  std::array<std::uint_fast8_t, 3u>{ 21u, 21u, 21u },
  std::array<std::uint_fast8_t, 3u>{ 22u, 22u, 22u },
  std::array<std::uint_fast8_t, 3u>{ 23u, 23u, 23u },
  std::array<std::uint_fast8_t, 3u>{ 24u, 24u, 24u },
  std::array<std::uint_fast8_t, 3u>{ 25u, 25u, 20u },
  std::array<std::uint_fast8_t, 3u>{ 26u, 26u, 26u },
  std::array<std::uint_fast8_t, 3u>{ 27u, 27u, 27u },
  std::array<std::uint_fast8_t, 3u>{ 28u, 28u, 28u },
  std::array<std::uint_fast8_t, 3u>{ 29u, 29u, 29u },
  std::array<std::uint_fast8_t, 3u>{ 30u, 30u, 30u },
  std::array<std::uint_fast8_t, 3u>{ 31u, 31u, 31u },
  std::array<std::uint_fast8_t, 3u>{ 32u, 32u, 32u },
  std::array<std::uint_fast8_t, 3u>{ 33u, 33u, 33u },
  std::array<std::uint_fast8_t, 3u>{ 34u, 34u, 34u },
  std::array<std::uint_fast8_t, 3u>{ 35u, 35u, 35u },
  std::array<std::uint_fast8_t, 3u>{ 36u, 36u, 36u }
};

inline int tile34(int const tile37) {
  if (tile37 == TILE37_NONE) return NULL_TILE_IDX;
  if (tile37 < 0 || tile37 >= 37) throw std::invalid_argument("Invalid 37-tile ID.");
  if (tile37 < 30) {
    return (tile37 / 10) * 9 + (tile37 % 10 == 0 ? 4 : tile37 % 10 - 1);
  }
  return tile37 - 3;
}

inline int base34_to_tile37(int const base34) {
  if (base34 == NULL_TILE_IDX) return TILE37_NONE;
  if (base34 < 0 || base34 >= 34) throw std::invalid_argument("Invalid base34 ID.");
  if (base34 < 27) return (base34 / 9) * 10 + (base34 % 9) + 1;
  return base34 + 3;
}

inline int next_dora_base34(int const ind_b34) {
  if (ind_b34 < 27) {
    int const s = ind_b34 / 9;
    int const r = ind_b34 % 9;
    return s * 9 + (r == 8 ? 0 : r + 1);
  }
  if (ind_b34 < 31) {
    return 27 + ((ind_b34 - 27 + 1) % 4);
  }
  return 31 + ((ind_b34 - 31 + 1) % 3);
}

inline int idx136_to_tile37(int const idx136) {
  if (idx136 < 0 || idx136 >= 136) throw std::invalid_argument("Invalid 136-tile ID.");
  for (int t = 36; t >= 0; --t) {
    if (idx136 >= OFFSETS_136[t]) return t;
  }
  return 0;
}

inline bool is_red5_tile37(int const tile37) {
  return tile37 == 0 || tile37 == 10 || tile37 == 20;
}

inline int red5_tile37_of_base34(int const b34) {
  if (b34 == 4) return 0;
  if (b34 == 13) return 10;
  if (b34 == 22) return 20;
  return -1;
}

inline int max_copies_tile37(int const tile37) {
  if (tile37 < 30 && tile37 % 10 == 0) return 1;
  if (tile37 < 30 && tile37 % 10 == 5) return 3;
  return 4;
}

inline int action_cand_mode(int const action_type) {
  switch (action_type) {
    case 10: return 0; // PAD
    case 0: return 1;  // POST_DISCARD_EXACT
    case 6:
    case 7: return 2;  // PRE_DISCARD_MELD_ENVELOPE
    case 1:
    case 2:
    case 8: return 3;  // PENDING_RINSHAN_KAN
    case 5: return 4;  // PASS_CURRENT_STATE
    case 3:
    case 9: return 5;  // WIN_EVAL
    case 4: return 6;  // KYUSHU_DRAW_EVAL
    default: throw std::invalid_argument("Unknown action_type.");
  }
}

struct ManifestRow {
  int action_type;
  int target_base34;
  int c0_base34;
  int c1_base34;
  int source_rel;
  int target_is_red5;
  int consumed_has_red5;
  int is_moqie;
  int is_riichi;
  int cand_mode;
};

inline std::array<ManifestRow, 547> build_action_manifest() {
  std::array<ManifestRow, 547> m{};
  for (int aid = 0; aid < 547; ++aid) {
    ManifestRow r{0, NULL_TILE_IDX, NULL_TILE_IDX, NULL_TILE_IDX, NULL_SOURCE_IDX, 0, 0, 0, 0, 0};
    if (aid < 148) {
      int const t37 = aid / 4;
      r.action_type = 0;
      r.target_base34 = tile34(t37);
      r.target_is_red5 = is_red5_tile37(t37) ? 1 : 0;
      r.is_moqie = (aid / 2) & 1;
      r.is_riichi = aid & 1;
    } else if (aid < 182) {
      int const b34 = aid - 148;
      r.action_type = 1;
      r.target_base34 = b34;
      r.consumed_has_red5 = (b34 == 4 || b34 == 13 || b34 == 22) ? 1 : 0;
    } else if (aid < 219) {
      int const t37 = aid - 182;
      r.action_type = 2;
      r.target_base34 = tile34(t37);
      r.target_is_red5 = is_red5_tile37(t37) ? 1 : 0;
    } else if (aid == 219) {
      r.action_type = 3;
    } else if (aid == 220) {
      r.action_type = 4;
    } else if (aid == 221) {
      r.action_type = 5;
    } else if (aid < 312) {
      auto const& row = CHI_ENCODE[aid - 222];
      r.action_type = 6;
      r.target_base34 = tile34(row[2]);
      r.c0_base34 = tile34(row[0]);
      r.c1_base34 = tile34(row[1]);
      r.source_rel = 3;
      r.target_is_red5 = is_red5_tile37(row[2]) ? 1 : 0;
      r.consumed_has_red5 = (is_red5_tile37(row[0]) || is_red5_tile37(row[1])) ? 1 : 0;
    } else if (aid < 432) {
      int const rel_from = (aid - 312) / 40;
      int const code = (aid - 312) % 40;
      auto const& row = PENG_ENCODE[code];
      r.action_type = 7;
      r.target_base34 = tile34(row[2]);
      r.c0_base34 = tile34(row[0]);
      r.c1_base34 = tile34(row[1]);
      r.source_rel = rel_from + 1;
      r.target_is_red5 = is_red5_tile37(row[2]) ? 1 : 0;
      r.consumed_has_red5 = (is_red5_tile37(row[0]) || is_red5_tile37(row[1])) ? 1 : 0;
    } else if (aid < 543) {
      int const rel_from = (aid - 432) / 37;
      int const t37 = (aid - 432) % 37;
      int const b34 = tile34(t37);
      r.action_type = 8;
      r.target_base34 = b34;
      r.source_rel = rel_from + 1;
      r.target_is_red5 = is_red5_tile37(t37) ? 1 : 0;
      r.consumed_has_red5 = ((b34 == 4 || b34 == 13 || b34 == 22) && !is_red5_tile37(t37)) ? 1 : 0;
    } else if (aid < 546) {
      r.action_type = 9;
      r.source_rel = (aid - 543) + 1;
    } else {
      r.action_type = 10;
    }
    r.cand_mode = action_cand_mode(r.action_type);
    m[aid] = r;
  }
  return m;
}

inline std::array<ManifestRow, 547> const ACTION_MANIFEST = build_action_manifest();

} // namespace Kanachan::StructuredV2
