"""Compile the real responsibility methods in a minimal progression fixture.

This checks the counter guards without requiring a game, Python extension or
GPU. It does not establish full simulator or scoring acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


HEADER = r'''
#include <cstdint>
#include <vector>
#include <utility>
#include <limits>
#include <stdexcept>
#include <iostream>
template<class E> struct ErrorStream {
  template<class T> ErrorStream &operator<<(T const &) { return *this; }
  ~ErrorStream() noexcept(false) { throw E("invalid progression"); }
};
template<class E> ErrorStream<E> fail(int) { return {}; }
#define KANACHAN_THROW fail
#define KANACHAN_ASSERT(x) do { if (!(x)) throw std::logic_error("counter bound"); } while (false)
constexpr int _1 = 0;
namespace Kanachan {
struct RoundState {
  std::vector<std::uint_fast16_t> progression_;
  std::pair<std::uint_fast8_t,std::uint_fast8_t> checkDaSanyuanPao_() const;
  std::pair<std::uint_fast8_t,std::uint_fast8_t> checkDaSixiPao_() const;
};
'''

FIXTURES = r'''
}
int main() {
  using K = Kanachan::RoundState;
  auto const none = std::make_pair(std::numeric_limits<std::uint_fast8_t>::max(),std::numeric_limits<std::uint_fast8_t>::max());
  int cases=0, failures=0;
  for (bool winds : {false,true}) for (int seat=0;seat<4;++seat) for(int relative=0;relative<3;++relative) {
    int const total=winds?4:3;
    auto run=[&](K const &state, bool responsible) {
      ++cases;
      try {
        auto answer=winds?state.checkDaSixiPao_():state.checkDaSanyuanPao_();
        auto expected=responsible?std::make_pair(static_cast<std::uint_fast8_t>((seat+relative+1)%4),static_cast<std::uint_fast8_t>(seat)):none;
        if(answer!=expected) ++failures;
      } catch(std::logic_error const &) { ++failures; }
    };
    for(int count=0;count<=total;++count) {
      K concealed, open, mixed;
      for(int tile=0;tile<count;++tile) {
        concealed.progression_.push_back(1881+seat*34+(winds?27:31)+tile);
        open.progression_.push_back(957+seat*120+relative*40+(winds?33:37)+tile);
        mixed.progression_.push_back(tile==count-1?1881+seat*34+(winds?27:31)+tile:957+seat*120+relative*40+(winds?33:37)+tile);
      }
      run(concealed,false);
      run(open,count==total);
      run(mixed,false);
    }
    K final_open, final_open_kan;
    for(int tile=0;tile<total-1;++tile) {
      final_open.progression_.push_back(1881+seat*34+(winds?27:31)+tile);
      final_open_kan.progression_.push_back(1881+seat*34+(winds?27:31)+tile);
    }
    final_open.progression_.push_back(957+seat*120+relative*40+(winds?33:37)+total-1);
    final_open_kan.progression_.push_back(1437+seat*111+relative*37+(winds?30:34)+total-1);
    run(final_open,true);
    run(final_open_kan,true);
  }
  std::cout<<"{\"cases\":"<<cases<<",\"failures\":"<<failures<<"}"<<std::endl;
}
'''


def methods(source):
    result = []
    for name in ("checkDaSanyuanPao_", "checkDaSixiPao_"):
        start = source.index(
            "std::pair<std::uint_fast8_t, std::uint_fast8_t> RoundState::" + name
        )
        opening = source.index("{", start)
        depth = 1
        end = opening + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        result.append(source[start:end])
    return "\n".join(result)


def compile_and_run(source, prefix, compiler):
    cpp = prefix.with_suffix(".cpp")
    executable = prefix.with_suffix(".exe")
    cpp.write_text(HEADER + methods(source) + FIXTURES, encoding="utf-8")
    command = [compiler, "/nologo", "/std:c++20", "/EHsc", "/utf-8", str(cpp),
               "/Fe:" + str(executable), "/Fo:" + str(prefix.with_suffix(".obj"))]
    compiled = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=60)
    if compiled.returncode:
        raise RuntimeError(compiled.stdout + compiled.stderr)
    tested = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(tested.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    candidate = options.source.read_text(encoding="utf-8")
    original = options.reference_source.read_text(encoding="utf-8")
    new = compile_and_run(candidate, options.output / "candidate", options.compiler)
    old = compile_and_run(original, options.output / "original", options.compiler)
    if new["failures"] or not old["failures"]:
        raise RuntimeError(f"Counter regression did not reproduce and resolve: {old}, {new}")
    receipt = {"status":"complete", "scope":"real_pao_method_counter_component_only",
               "original":old,"candidate":new,
               "candidate_sha256":hashlib.sha256(options.source.read_bytes()).hexdigest(),
               "original_sha256":hashlib.sha256(options.reference_source.read_bytes()).hexdigest(),
               "full_native_simulation_verified":False}
    (options.output / "acceptance.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(receipt))
