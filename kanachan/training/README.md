# `kanachan.training` Python submodule

The [online placement training guide](online/README.md) covers the named
scalar CQL recipe, persistent replay, fixed-seed evaluation and the single-host
online learner. See [examples/training/online.yaml](../../examples/training/online.yaml).

## Encoder variants

New experiments can select `encoder.architecture=rmsnorm_swiglu` with any
encoder size preset. This uses per-token Pre-Norm RMSNorm, a gated SwiGLU
feedforward network, and PyTorch scaled dot-product attention (SDPA). It keeps
bidirectional attention over the visible state and legal candidates. Sparse,
progression and candidate padding tokens are excluded as attention keys;
numeric tokens remain visible. Progression position embeddings and the data
format are unchanged.

The gated hidden width is two thirds of `encoder.dim_feedforward`, rounded
to the nearest multiple of eight, with a minimum of eight. This keeps the
three SwiGLU projections near the parameter budget of the original two FFN
projections. The variant always uses SwiGLU in place of the legacy encoder's
`activation_function`; decoder activation settings remain independent. Leave
`encoder.layer_normalization=false`: this variant supplies its own final
per-token RMSNorm. SDPA selects its backend according to device, dtype and
mask support; choosing this variant does not guarantee a FlashAttention
kernel or a speed improvement.

The default `encoder.architecture=transformer` retains the original Post-LN
encoder and its checkpoint layout. Old exported models without an
`architecture` field load through this default. Modern exports record the
variant so BC, CQL/IQL/ILQL, AWR, ranking and online model loading reconstruct
the correct network. Use the same variant when resuming or transferring
`encoder.pth`: legacy and modern encoder weights are not interchangeable.
Train a separate BC baseline before comparing RL policies from this variant.

This design follows the [Pre-Norm analysis](https://arxiv.org/abs/2002.04745),
[RMSNorm](https://arxiv.org/abs/1910.07467),
[GLU variants](https://arxiv.org/abs/2002.05202), and the
[Llama reference implementation](https://github.com/meta-llama/llama3/blob/main/llama/model.py).
Attention mask and dropout semantics follow the
[PyTorch 2.6 SDPA documentation](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html).
Architecture tests and short training runs establish functionality, not
playing strength; use held-out games and paired native matches to measure it.

## Decision-decoder research reference (planned)

The next decoder design uses [Qwen4-Exp](https://huggingface.co/docs/transformers/main/en/model_doc/qwen4_exp)
as its main reference. Qwen identifies
[Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next) as an early
preview of the Qwen4 architecture. The existing `rmsnorm_swiglu` option
remains a bidirectional baseline; it does not implement Qwen4-Exp.

Planned, separately controlled experiments are gated residual streams (GR),
compact event-history n-gram features (PLE), and a GDN/attention hybrid.
Sparse attention needs a separate long-history cost comparison. Event memory
must respect game, seat and round boundaries. These are adaptations whose
benefits in Mahjong remain unverified.

The [architecture report](https://arxiv.org/html/2608.30320v1) also motivates
a separate Muon experiment. Our packed QKV and gate/up projections need
logical splitting before orthogonalization; applying Muon to each packed
tensor as one matrix would not follow that recipe. Compare playing quality,
training/inference cost and stability, with explicit parameter budgets.

## Submodules

### [`kanachan.training.bc`](bc) Python submodule

An implementation of training for supervised learning to mimic actions recorded in game records (a.k.a. behavioral cloning).

### [`kanachan.training.cql`](cql) Python submodule

An implementation of training for [conservative Q-learning (CQL)](https://arxiv.org/abs/2006.04779).

### [`kanachan.training.iql`](iql) Python submodule

An implementation of training for [implicit Q-learning (IQL)](https://arxiv.org/abs/2110.06169).

## Check the training pipeline before a large run

Use annotations and snapshots from the same project version. The v2 BC reader
expects eight tab-separated columns. Legacy seven-column BC annotations and
legacy model vocabularies need their matching code, or fresh annotations made
with the v2 annotation tool. Changing the column count alone is insufficient.

Use Python 3.10–3.12 with the versions in `pyproject.toml`: PyTorch 2.6.0,
TensorDict 0.7.0 and TorchRL 0.7.0. For CUDA, the Dockerfile uses CUDA 12.4
and PyTorch's cu124 wheels. CPU training uses native optimizers; Apex is
optional. Enabling it in Docker requires `INSTALL_APEX=1` and an explicit
`APEX_REVISION`. The portable Adam loader also accepts legacy Apex group
step counters. Keep architecture and optimizer choices consistent when resuming.

Run the regression suite in that environment:

```sh
python -m unittest discover -s test/training -v
```

The suite includes short CPU runs of BC, IQL, ILQL, scalar and quantile CQL,
AWR, ranking and end-of-round ranking, followed by loading their exported
models. It also checks replay input and checkpoint recovery. CUDA tests run
when a CUDA device is available. None of these short runs measures playing
strength.

Real Gloo multiprocessing tests are opt-in because rendezvous needs a working
local communication backend. Set `KANACHAN_TEST_DISTRIBUTED=1` when running
`test_replay_distributed.py` and `test_distributed_training.py`; without it,
replay sharding and termination
still have unit coverage using a controlled transport. An initialization
timeout in Gloo is not a successful distributed-training check.

Set `KANACHAN_TEST_COMPILE=1` for the TorchInductor BC/CQL test, which includes
AMP and gradient checkpointing. The normal regression runs disable compilation
to keep small tests fast. RL autocast disables the weight cache because a
source network is used with and without autograd; sharing cached casts can
detach gradients and break checkpoint recomputation
([PyTorch issue](https://github.com/pytorch/pytorch/issues/141896)).

Before a large run, inspect a bounded sample using the actual reader:

```sh
python -m kanachan.training.audit_data --format bc --data /data/bc.tsv.bz2
python -m kanachan.training.audit_data --format rl --data /data/rl.tsv.bz2
```

The default limit is 10,000 rows; `--max-samples 0` validates the full file.
An all-terminal RL sample is a warning: eight-column BC rows also parse as
terminal RL transitions. IQL, ILQL and CQL need transitions produced from
complete games **before** row shuffling:

```sh
python bin/annotation/rl/rl.py --stream /data/bc-complete-games.tsv > /data/rl.tsv
```

Streaming requires all rows of each UUID to be consecutive and rejects a UUID
that reappears. It holds one game in memory and accepts gzip/bzip2 input.
Without `--stream`, the converter sorts the entire input in memory. AWR and
the state-ranking trainer use the eight-column BC input; end-of-round ranking
uses the separate four-column output of `bin/annotation/eor/eor.py`.

Training logs and TensorBoard now distinguish samples from completed optimizer
steps and target updates. The counters are labelled `this run`; they restart
when the training program is restarted. Gradient accumulation of 1000 means one
optimizer update per 1000 batches. Target update intervals count completed
optimizer steps, including saved history when resuming a new checkpoint.
At EOF, a partial accumulation window is applied using its actual sample
count. A run completing zero optimizer steps fails without exporting a final
model. Check for exhausted input or non-finite gradients in that case.
A target update rate of 0.1 uses 10% of the source parameters
and retains 90% of the previous target parameters at each update.

BC weights are not picked up automatically by an RL preset. Use
`encoder.load_from=/data/bc/snapshots/encoder.pth` to initialize the RL encoder
from BC, with the same encoder architecture and vocabulary. The RL value and
Q heads are separate models. `initial_model_prefix` resumes snapshots from
the current trainer; a BC snapshot directory is not an IQL/ILQL checkpoint.
When extracting an AWR policy, also set
`decoder.load_from=/data/bc/snapshots/decoder.pth` with the matching decoder
architecture to retain the BC policy head. This option is also honored by the
ranking trainers. IQL/ILQL reject a single `decoder.load_from` because their
separate Q/V heads are not a BC policy head. No load option is silently ignored.

## Resume training from a checkpoint

New snapshots include `training-state.json` (or `training-state.N.json`)
alongside the weights and optimizer files. Keep these files together when
using `initial_model_prefix`. The state records the input position and the
completed optimizer step count and AMP scaler state. Resume using the same
data ordering. Random-generator state and replay contents are not saved, so
this is not a guarantee of bit-for-bit reproducibility.

Periodic snapshots wait for a completed gradient accumulation window. With
multiple input workers they also wait for a complete worker cycle. An
incomplete accumulation window at EOF is flushed, including a smaller final
batch. Periodic and final snapshots both retain AMP scaling state. ZeRO
optimizer state is consolidated across ranks before global rank zero writes it.

To reread a completed epoch or start a new data file, set
`start_new_epoch=true` together with `initial_model_prefix`. This resets the
input cursor while keeping optimizer counters, scheduler and AMP state.
`optimizer.initialize=true` instead starts fresh optimizer, scheduler, scaler
and input history, while retaining the model weights. It takes precedence
over `start_new_epoch`.

Old snapshots without this state use the sample count from a numbered
filename and emit a warning. An old unnumbered final snapshot has no known
input position and starts at zero; prefer a numbered snapshot for recovery.

Replay checkpoints do not contain the shuffle buffer. An unfinished
multi-worker accumulation window can also lack an exact input position.
Recovery rejects those cases rather than silently skipping arbitrary samples.
Use `start_new_epoch=true` to reread from the beginning with saved optimizer
history, or `optimizer.initialize=true` to reset that history as well.

## Replay and reward-model checks

Set `contiguous_training_data=true` only for complete, consecutive games.
Replay then reads sequentially, overriding multiple workers, computes
episode returns, and rejects an incomplete final episode. With shuffled
transitions use `false`; no episode return is inferred between neighboring
rows, and the reward function must support independent transitions.
Single-process replay includes the final partial batch. Distributed replay
uses equal batch sizes per participating rank and reports any remainder
smaller than the number of ranks.

End-of-round ranking supervises the final rank of all four players. Tied
scores follow starting-seat order. If an earlier reward model was trained
with the incorrect seat selection or round feature conversion, fixing the
code does not repair its saved weights: retrain and validate that reward
model before using it for RL. Enabling encoder checkpointing must preserve
the final layer normalization and its gradients; this is regression-tested.
The end-of-round encoder also honors `encoder.layer_normalization`, including
checkpoint recomputation and saved model metadata. For quantile CQL,
`kappa=0` uses ordinary quantile regression rather than a zero TD loss, as
defined in the [QR-DQN paper](https://arxiv.org/abs/1710.10044).

## Evaluate BC on separate games

Split training and validation annotations by complete game UUID before
shuffling. Keep all decisions from a game in the same split. Evaluate candidate
checkpoints on the same fixed validation file:

```sh
python -m kanachan.training.bc.evaluate \
  --model /data/snapshots/model.kanachan \
  --data /data/validation.tsv.bz2 \
  --device cuda:0 --batch-size 256
```

The JSON result reports sample count, negative log likelihood, action agreement
and the uniform legal-action baseline. `--max-samples N` limits a quick check;
omit it for the complete file. Partial final batches are included. These are
offline behavior-cloning metrics, not game win rates. Preserve the BC policy
as a fixed opponent when evaluating RL checkpoints in the simulator.
In `kanachan.simulation.run` output, a lower `ranking.average` is better.
`difference.ranking.average` computes baseline average rank minus
proposed average rank, so a positive difference favors the proposed model.
All `variance` fields are unbiased sample variances. A single observation has
`variance: null`; these descriptive variances are not confidence intervals
and do not account for dependence between players or duplicated games.

The evaluator also accepts legacy BC exports producing raw `decode` scores
and an argmax action. Known legacy defaults (no layer normalization and
`noise_init_std=0` meaning ordinary linear layers) are adapted while loading;
the snapshot files themselves are unchanged. Other vocabularies still require
their matching code and data.

## Evidence from the 2026-10-29 audit

The Windows CPU suite, real WSL Gloo training/recovery and replay tests,
single RTX 3090 CUDA/AMP training, and TorchInductor BC/CQL training were
exercised separately. The final Windows run contains 73 tests: 68 passed and
five environment-specific tests were skipped there and passed separately in
WSL. Gloo includes two-layer checkpointing for CQL and end-of-round ranking.
Both Windows and WSL dependency checks pass. C++ annotation, simulation, tile-wall restoration and
fixture-generation targets were built with GCC 15.3 and Boost 1.92. CMake
uses Boost's basic stacktrace backend when libbacktrace is unavailable. All
eight complete historical games pass the native annotation/simulation
comparison after correcting player-grade and kan-dora feature encodings.

Eight historical raw game records were copied from the read-only F drive
and reannotated: 4,251 decisions, split by complete games into 3,146 training
rows and 1,105 validation rows. Their RL conversion contains 3,957 ordinary
transitions, 262 round-end transitions and 32 game-end transitions. Short GPU
runs exercise BC, IQL, ILQL, CQL, AWR and replay on these real annotations.
The same raw annotations yield 50 training and 16 validation end-of-round
rows. Both ranking models train on real data and produce normalized finite
four-seat probabilities. The learned end-of-round model was also used to
produce rewards for a complete 50-update IQL short run.

The small BC pilot reaches NLL 1.7353 and 41.63% action agreement on the
1,105 held-out decisions, compared with uniform NLL 2.0372 and 17.71%
agreement. AWR initialized with both BC components reaches NLL 1.7070 and
43.08% agreement after 50 updates. These small-corpus observations validate
the transfer and evaluation path, not a general improvement in playing strength.

The historical BC snapshot `model.90003456.kanachan` achieves NLL 0.7652 and
70.32% action agreement on those two validation games (uniform legal-action
agreement 17.71%). Its original training-data overlap is unknown, so these
numbers establish a working model/data path; they do not prove unseen-game
generalization or playing strength. Old seven-column datasets must be
regenerated from raw records for v2 training.

The native comparison framework checks every recorded decision, round score
and final result. Generate complete fixtures with `paishan`, `annotate` and
`test/annotation_vs_simulation/generate`, then run:

```sh
python test/annotation_vs_simulation/run.py /data/complete-game-fixtures
KANACHAN_NATIVE_FIXTURES=/data/complete-game-fixtures python -m unittest discover -s test/training -p test_native_replay.py -v
```

One physical GPU is available on this host. CPU Gloo multiprocessing and
single-GPU AMP checks do not establish multi-GPU NCCL behavior. Docker's
`dockerDesktopLinuxEngine` pipe is absent on this host, so the full container
image rebuild is unverified. A statistically adequate playing-strength
comparison remains a separate long-running experiment.

## Remote six-game pipeline acceptance, 2026-10-30

An independent Windows/Python 3.12 environment with an RTX 3070 Laptop
8 GB and PyTorch 2.6.0 CUDA 12.4 processed six raw Jade/Throne half-games.
Four training games produced 2,298 BC and RL decisions; validation and test
used different complete games (504 and 949 decisions). BC completed 72
optimizer updates and scalar placement CQL completed 70 successful updates.
Both restored from sample 256 and finished with tensor-identical source and
target weights. For this strict comparison, flash, memory-efficient and
cuDNN SDPA were disabled: enabling the cuDNN attention backend produced
small numerical differences after resuming despite identical replay/RNG state.

The real native simulator then collected 531 learner decisions over two
online iterations, performed eight CUDA/AMP optimizer updates, restored its
model/replay/optimizer/scaler state, and exported the policy. Evaluation used
a held-out wall seed with four seat rotations against the fixed BC policy.
The native build retained assertions; this exposed and fixed an invalid
assertion about completed `std::jthread` objects by explicitly joining them
before returning. These bounded checks establish that the stages connect and
resume correctly. They do not establish playing strength or promotion
eligibility.
