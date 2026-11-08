# Isolated native Python runtime

This CMake entry point builds the real simulation Python extension directly
from the repository sources. It also supports an x64 MSVC build without WSL.
It requires a complete Boost source tree (headers and `libs/python/src`),
Nyanten headers, and the target Python interpreter's headers and import library.
Boost.Python is compiled into the extension; Protobuf is not needed here.

In an x64 Visual Studio developer shell, with paths for those dependencies:

```powershell
cmake -S tools/runtime -B build/runtime -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DKANACHAN_BOOST_ROOT=C:/deps/boost_1_85_0 `
  -DKANACHAN_NYANTEN_ROOT=C:/deps/nyanten `
  -DKANACHAN_PYTHON_INCLUDE_DIR=C:/Python312/include `
  -DKANACHAN_PYTHON_LIBRARY=C:/Python312/libs/python312.lib
cmake --build build/runtime --parallel 2
```

The Windows result is `kanachan/simulation/_simulation.pyd`. Run from the
repository with the matching Python interpreter and training dependencies.
This standalone build enables the native assertions. Its MSVC/Python 3.12
runtime was exercised on 2026-10-30 using real CUDA online sampling, optimizer
updates, checkpoint recovery, and four complete evaluation games. This does
not establish compatibility with every Windows/Python/GPU configuration.

## Linux CUDA 12.8 training image

`requirements-cu128.txt` selects the PyTorch 2.9, TensorDict 0.10 and TorchRL
0.10 dependency line. The checked AutoDL base already contains PyTorch
2.9.0+cu128, Python 3.11 and the `sm_120` kernels for RTX 5090.

Keep the checkout at `/opt/kanachan` and the Python environment on the system
disk so both are included when saving the image. Build intermediates and
training data belong on `/root/autodl-tmp`, which is a separate data volume.

From a base with GCC, Git, curl, Protobuf headers and `protoc` installed:

```bash
bash /opt/kanachan/tools/runtime/prepare_image.sh
```

The Linux build adds `/opt/kanachan/bin/annotate` and `bin/paishan`, with
Protobuf code regenerated for the installed system library. Native simulation
retains assertions and uses a Release build with link-time optimization.

`verify_environment.py` uses the included complete real-game fixture, checks
BC and MC+CQL training through the normal entry points, compares interrupted
recovery states, loads the retained models, checks the 28.7M-parameter model's
forward and backward passes, and completes four native evaluation games.
The included fixture and models are for environment acceptance.

```bash
python /opt/kanachan/tools/runtime/verify_environment.py
python /opt/kanachan/tools/runtime/verify_environment.py --require-cuda
```

CPU-mode image preparation records CUDA verification as pending. Run the
second command after mounting the GPU to exercise BF16 training and recovery.
It also runs `verify_attention.py` to compare the existing mask's forward and
gradients, require an actual CUDA FlexAttention kernel in the profiler, and
measure BF16 AdamW model steps for the 0.83M and 28.7M parameter models.

### Fused decoder attention

BC and CQL accept `encoder.attention_backend=flex`. This uses compiled PyTorch
FlexAttention on CUDA with the existing causal-history/current-state mask.
It needs no separate `flash-attn` wheel. Unset `TORCHDYNAMO_DISABLE` for this
path. CPU execution and the short append-only inference cache use SDPA.
`sdpa` remains the default until the target GPU benchmark is available.
Weights and state keys are unchanged. Non-default backend choices are bound
to new checkpoint recipes so a resume cannot silently change the configured backend.

```bash
# CPU mode: actual real-game Q/K/V and gradient reference check, no fused claim.
python /opt/kanachan/tools/runtime/verify_attention.py --cpu-reference
# GPU mode only: runtime dispatch, correctness, memory and model-step timings.
python /opt/kanachan/tools/runtime/verify_attention.py --batch-sizes 32 128 256
```

Timings exclude data loading, checkpoint writing and initial compilation.
The report lists compilation cost separately. A built-in FlashAttention
capability flag alone is not dispatch or performance evidence. FlexAttention
is a PyTorch prototype API, so retain the SDPA option for comparison.
See [PyTorch FlexAttention](https://docs.pytorch.org/docs/2.9/nn.attention.flex_attention.html).

The verification process scopes Inductor's `triton.descriptive_names` to
`torch` and records this naming policy in its report. PyTorch 2.9's default
ATen naming can otherwise label Flex templates `select` or `zeros`, causing
the profiler gate to reject an executing CUDA kernel. The gate still requires
a CUDA event with the Flex source name; it does not accept a CPU event or a
generic Triton kernel. This changes verification names, not the training
backend default. See [PyTorch's naming implementation](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/_inductor/utils.py).
