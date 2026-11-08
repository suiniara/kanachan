// Public-observation discard afterstates. No simulator/private-state access.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdexcept>
#include <nyanten/replacement_number.hpp>
#include <array>
#include <algorithm>
#include <numeric>
#include <vector>

namespace {
constexpr int width = 46;

int tile34(int tile) {
  if (tile < 0 || tile >= 37) throw std::invalid_argument("Invalid 37-tile ID.");
  if (tile < 30) return tile / 10 * 9 + (tile % 10 == 0 ? 4 : tile % 10 - 1);
  return tile - 3;
}

std::vector<int> integers(PyObject* object, int expected = -1) {
  PyObject* sequence = PySequence_Fast(object, "Expected an integer sequence.");
  if (sequence == nullptr) throw std::invalid_argument("Expected an integer sequence.");
  Py_ssize_t const size = PySequence_Fast_GET_SIZE(sequence);
  if ((expected >= 0 && size != expected) || size > 546) {
    Py_DECREF(sequence);
    throw std::invalid_argument("Invalid native candidate-feature input width.");
  }
  std::vector<int> result;
  result.reserve(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    long const value = PyLong_AsLong(PySequence_Fast_GET_ITEM(sequence, i));
    if (PyErr_Occurred() || value < 0 || value > 546) {
      Py_DECREF(sequence);
      throw std::invalid_argument("Invalid native candidate-feature integer.");
    }
    result.push_back(static_cast<int>(value));
  }
  Py_DECREF(sequence);
  return result;
}

PyObject* discard_features(PyObject*, PyObject* arguments) {
  PyObject *hand_object, *visible_object, *dora_object, *actions_object;
  int melds;
  if (!PyArg_ParseTuple(arguments, "OOOOi", &hand_object, &visible_object,
                       &dora_object, &actions_object, &melds)) return nullptr;
  try {
    auto hand37 = integers(hand_object, 37);
    auto visible = integers(visible_object, 34);
    auto dora = integers(dora_object, 34);
    auto actions = integers(actions_object);
    if (melds < 0 || melds > 4 || actions.size() > 32) {
      throw std::invalid_argument("Invalid meld count or candidate count.");
    }
    std::array<int, 34> hand{};
    for (int tile = 0; tile < 37; ++tile) {
      int const limit = tile < 30 && tile % 10 == 0 ? 1 :
                        tile < 30 && tile % 10 == 5 ? 3 : 4;
      if (hand37[tile] > limit) throw std::invalid_argument("Impossible hand multiplicity.");
      hand[tile34(tile)] += hand37[tile];
    }
    for (int tile = 0; tile < 34; ++tile) {
      if (visible[tile] > 4 || visible[tile] < hand[tile] || dora[tile] > 5) {
        throw std::invalid_argument("Impossible observed tile count or dora factor.");
      }
    }
    bool const has_discard = std::any_of(actions.begin(), actions.end(), [](int a) { return a < 148; });
    if (has_discard && std::accumulate(hand.begin(), hand.end(), 0) != 14 - melds * 3) {
      throw std::invalid_argument("Discard hand size does not match observed melds.");
    }
    std::vector<std::array<float, width>> output(actions.size());
    for (std::size_t i = 0; i < actions.size(); ++i) {
      int const action = actions[i];
      if (action >= 148) continue; // Explicit missingness for non-discard actions.
      int const discard37 = action / 4;
      int const discard34 = tile34(discard37);
      if (hand37[discard37] == 0) throw std::invalid_argument("Discard tile is not in the observed hand.");
      auto after = hand;
      --after[discard34];
      int const replacement = Nyanten::calculateReplacementNumber(after);
      auto& features = output[i];
      features[0] = 1.f;
      features[1] = static_cast<float>(replacement - 1) / 8.f;
      features[2] = replacement == 1 ? 1.f : 0.f;
      int live_tiles = 0, live_types = 0;
      for (int tile = 0; tile < 34; ++tile) {
        if (visible[tile] == 4) continue;
        ++after[tile];
        bool const improves = Nyanten::calculateReplacementNumber(after) < replacement;
        --after[tile];
        if (improves) {
          live_tiles += 4 - visible[tile];
          ++live_types;
          features[5 + tile] = 1.f;
        }
      }
      features[3] = static_cast<float>(live_tiles) / 136.f;
      features[4] = static_cast<float>(live_types) / 34.f;
      for (int suit = 0; suit < 3; ++suit) {
        features[39 + suit] = static_cast<float>(hand37[suit * 10] - (discard37 == suit * 10));
      }
      int retained = 0;
      for (int tile = 0; tile < 34; ++tile) retained += after[tile] * dora[tile];
      features[42] = static_cast<float>(retained) / 70.f;
      features[43] = discard37 < 30 && discard37 % 10 == 0 ? 1.f : 0.f;
      features[44] = static_cast<float>(dora[discard34]) / 5.f;
      features[45] = static_cast<float>(std::accumulate(after.begin(), after.end(), 0)) / 13.f;
    }
    return PyBytes_FromStringAndSize(reinterpret_cast<char const*>(output.data()),
                                    static_cast<Py_ssize_t>(output.size() * sizeof(output[0])));
  } catch (std::exception const& error) {
    PyErr_Clear();
    PyErr_SetString(PyExc_ValueError, error.what());
    return nullptr;
  }
}

PyMethodDef methods[] = {
  {"discard_features", discard_features, METH_VARARGS,
   "Generate observed-information discard afterstates as packed float32 rows."},
  {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_candidate_features", nullptr, -1, methods};
} // namespace

PyMODINIT_FUNC PyInit__candidate_features() {
  PyObject* result = PyModule_Create(&module);
  if (result == nullptr) return nullptr;
  if (PyModule_AddStringConstant(result, "SCHEMA_ID", "discard_afterstate_v1") < 0 ||
      PyModule_AddIntConstant(result, "FEATURE_WIDTH", width) < 0) {
    Py_DECREF(result);
    return nullptr;
  }
  return result;
}
