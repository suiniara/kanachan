#pragma once

#include "mjsoul_hand_eval.hpp"
#include "structured_v2_tables.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace Kanachan::StructuredV2 {

struct EncodedSample {
  std::array<float, 34 * TILE_FEATS_DIM> tile_feats{};
  int event_len = 0;
  std::array<std::uint8_t, MAX_PROGRESSION_LEN * EVENT_BYTES> events{};
  std::array<float, GLOBAL_FEATS_DIM> global_feats{};
  std::array<float, 4 * PLAYER_FEATS_DIM> player_feats{};
  std::array<std::uint8_t, MAX_CANDIDATES> cand_mode{};
  std::array<float, MAX_CANDIDATES * CAND_DYN_DIM> cand_dyn{};
};

void encode_single_sample(
    int32_t const* sparse,       // [29]
    int32_t const* numeric,      // [6]
    int32_t const* progression,  // [prog_stride]
    int prog_stride,
    int32_t const* candidates,   // [MAX_CANDIDATES]
    bool enable_early_shape_hint,
    std::array<float, 4> const& reward_weights,
    EncodedSample& out);

} // namespace Kanachan::StructuredV2
