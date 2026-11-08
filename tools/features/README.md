# Native discard afterstate features

This small extension reuses the project's Nyanten headers. Its inputs are
the observer's concealed/drawn tiles, public visible counts, public dora
factors and candidate IDs. It has no simulator object or hidden-hand input.
Training and inference both call `kanachan.candidate_features`.

Build with the same Python ABI that will load the model. In an environment
with a C++20 compiler, CMake and the existing Nyanten checkout:

```bash
cmake -S tools/features -B build/features -DCMAKE_BUILD_TYPE=Release \
  -DKANACHAN_NYANTEN_ROOT=/path/to/nyanten \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build/features --parallel 2
python -c "from kanachan import _candidate_features as f; print(f.SCHEMA_ID, f.FEATURE_WIDTH)"
```

An MSVC developer shell can instead specify
`KANACHAN_PYTHON_INCLUDE_DIR` and `KANACHAN_PYTHON_LIBRARY` for the target
interpreter. The output is `kanachan/_candidate_features.pyd` on Windows or
`kanachan/_candidate_features.so` on Linux. The root build and standalone
simulation runtime build also include this target. Wheel packaging retains
the built extension.

The head option `decoder.candidate_features=discard_v1` requires this module.
The default `none` loads existing models without it. A mismatched native
schema is rejected. A training source archive includes this build description
and the C++ source; it does not contain a platform-independent binary.
