#!/usr/bin/env bash
# Build inside an existing CUDA 12.8/PyTorch 2.9 Linux image.
set -euo pipefail

source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${KANACHAN_PYTHON:-/root/miniconda3/envs/torch_env/bin/python}
build_root=/root/autodl-tmp/kanachan-build
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONDONTWRITEBYTECODE=1

"$python_bin" - <<'PY'
import torch
if torch.__version__ != "2.9.0+cu128":
    raise RuntimeError("This image recipe reuses the PyTorch 2.9.0+cu128 base wheel.")
if "sm_120" not in torch._C._cuda_getArchFlags().split():
    raise RuntimeError("The installed wheel does not include RTX 5090 kernels.")
PY

mkdir -p "$build_root"
"$python_bin" -m pip install --no-cache-dir \
  -c "$source_root/tools/runtime/requirements-cu128.txt" \
  'cmake>=3.24,<4' ninja hatchling editables
"$python_bin" -m pip install --no-cache-dir --no-build-isolation \
  -c "$source_root/tools/runtime/requirements-cu128.txt" -e "$source_root"

# Downloads and intermediate objects stay on the data volume, outside the image.
if [[ ! -d "$build_root/boost_1_85_0" ]]; then
  curl --fail --location --retry 3 \
    https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.bz2 \
    -o "$build_root/boost_1_85_0.tar.bz2"
  printf '%s  %s\n' \
    7009fe1faa1697476bdc7027703a2badb84e849b7b0baad5086b087b971f8617 \
    "$build_root/boost_1_85_0.tar.bz2" | sha256sum -c -
  tar -xjf "$build_root/boost_1_85_0.tar.bz2" -C "$build_root"
fi
if [[ ! -d "$build_root/nyanten" ]]; then
  git clone https://github.com/Cryolite/nyanten.git "$build_root/nyanten"
  git -C "$build_root/nyanten" checkout 581aa72ce53e0ffeb0ef8d1ed14df2521ca5fb9b
fi
if [[ $(git -C "$build_root/nyanten" rev-parse HEAD) != 581aa72ce53e0ffeb0ef8d1ed14df2521ca5fb9b ]]; then
  printf '%s\n' 'Existing Nyanten source differs from this image recipe.' >&2
  exit 1
fi

python_prefix=$($python_bin -c 'import sys; print(sys.prefix)')
python_include=$($python_bin -c 'import sysconfig; print(sysconfig.get_path("include"))')
python_library=$($python_bin -c 'import sysconfig; from pathlib import Path; print(Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY"))')
cmake_bin="$python_prefix/bin/cmake"
"$cmake_bin" -S "$source_root/tools/runtime" -B "$build_root/native" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$python_prefix/bin/ninja" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
  -DKANACHAN_BOOST_ROOT="$build_root/boost_1_85_0" \
  -DKANACHAN_NYANTEN_ROOT="$build_root/nyanten" \
  -DKANACHAN_PYTHON_INCLUDE_DIR="$python_include" \
  -DKANACHAN_PYTHON_LIBRARY="$python_library" \
  -DKANACHAN_BUILD_ANNOTATION=ON
"$cmake_bin" --build "$build_root/native" --parallel "${KANACHAN_BUILD_JOBS:-8}"
"$python_bin" -m pip check
"$python_bin" -B "$source_root/tools/runtime/verify_environment.py"
