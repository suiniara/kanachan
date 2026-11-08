# Dense Mahjong decision decoder and discrete CQL

Use `--config-name decision` in the existing BC or CQL entry point.
This selects a separate `DenseDecisionDecoder` backbone and
`GlobalActionHead`. Legacy encoder models, checkpoints and optimizer presets
remain supported. No language tokenizer, pretrained language weights or MoE
is required.

## Actual feature and model sizes

| Input | Current shape | Padding |
| --- | --- | --- |
| Sparse observation IDs | integer [B,29] | 510 |
| Honba, deposits, four seat scores | integer [B,6] | none |
| Visible progression events | integer [B,109], right padded | 2165 |
| Original legal candidate IDs | integer [B,32] | 546 |

Shapes derive from `kanachan.constants`. The exported constructor records
the feature schema and rejects incompatible runtimes. Raw feature files do
not need conversion solely to change the model architecture.

Tokens are BOS, visible history, current sparse/numeric state, one unordered
legal-action-set summary, and a final decision token: currently 147 tokens.
History sees earlier history; the simultaneous current-state block reads
history and itself; the decision token reads both. Padding keys are hidden,
and padding query outputs are zeroed. All current-state tokens share one
logical RoPE position, while numeric fields retain their individual identities.

Blocks use dense MHA, an explicit allowed-attention mask, token-wise Pre-Norm
RMSNorm, RoPE, SwiGLU and zero dropout. Numeric bin arithmetic, clipping and
float64 intermediates retain the existing semantics, followed by six learned
projections. For this architecture `encoder.dim_feedforward` is the actual
SwiGLU hidden width; it is not converted from a legacy FFN width.

| Preset | Width | Layers | Heads | SwiGLU hidden | Numeric bins/field | Actual parameters, including head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decision_tiny | 128 | 2 | 2 | 192 | 64 | 829,730 |
| decision_small | 256 | 6 | 4 | 704 | 128 | 6,051,874 |

Tiny is about 0.7% larger than the existing 823,681-parameter BC baseline.
Its smaller gated FFN accounts for the larger global action head. These are
starting presets. `encoder.numeric_dimension` explicitly selects the number
of bins if needed; changing it changes the numeric/model contract.

The backbone returns [B,D]. A D-to-hidden-to-546 head produces finite global
scores and gathers them to the original [B,32] candidate order. The final
adapter returns an int32 **candidate index**. Exact ties use the smallest
global action ID. BC logits and CQL return estimates share shapes, not semantics.

## Training and continuation

Example single-line commands, using independently audited train data:

```sh
python -m kanachan.training.bc.train --config-name decision training_data=/data/train.bc.tsv.gz batch_size=128 num_workers=0 optimizer.learning_rate=0.0003 hydra.run.dir=runs/decision-bc
python -m kanachan.training.cql.train --config-name decision training_data=/data/train.rl.tsv.gz batch_size=128 num_workers=0 bc_initial_model=runs/decision-bc/snapshots/model.kanachan hydra.run.dir=runs/decision-cql
```

Use `encoder=decision_small` to select the larger preset. Transfer from BC
copies the backbone, legal-set embedding and hidden action-head feature layer
when `bc_initial_model` is supplied. The scalar output is always reinitialized
to zero; BC logits are never treated as return estimates. Every online-network
parameter participates in training from the start. There is no feature-freezing
phase. The default MC recipe has no target network. The explicit TD comparator
uses a separate non-trainable reference updated by EMA after successful steps.

The older `encoder.load_from` option transfers the backbone alone, keeping a
fresh hidden head; it conflicts with `bc_initial_model`. Complete BC-head loading
via `decoder.load_from` remains rejected. Full CQL resume uses
`initial_model_prefix`; retain the same BC reference bytes and training recipe.

The decoder default is `target_kind=mc`, `double_q_learning=false`: one
trainable online Q network regresses complete observed half-game returns,
plus alpha times `T*logsumexp(Q_legal/T)-Q_logged`, alpha=0.5 and T=1.
There is no next-state Q forward or EMA target in this mode. Illegal actions
and padding never enter these reductions. New decoder CQL exports **online**;
legacy presets retain their explicit TD and target-export defaults.

The decoder CQL default is `reward_scheme=round_and_placement`: at a round
boundary add this observer's settled net point change / `round_score_unit`
(default 10,000), times `round_reward_weight` (default 0.2). On the final
transition also add half-game placement [1.5,0.5,-0.5,-1.5], with discount=1.
This deliberately optimizes score plus placement. At the default scale,
50,000 net points equal one rank-utility step, rather than the previous
10,000 points. This is a starting configuration, not an empirically optimal
tradeoff or a guarantee of pure rank optimization. It includes the last round
once, riichi costs, deposits and draw payments; consecutive settlement
snapshots also cover intervening rounds without an observer action. The final
snapshot uses actual half-game scores, even when the last observed round was
earlier. Returns are computed on complete observer trajectories **before**
replay shuffling. Round boundaries are not terminals.

`reward_scheme=placement` explicitly retains the old terminal-only objective;
the generic `placement` preset and current online recipe keep that behavior.
Round-score rewards reject EOR shaping/plugins and incomplete replay. Reward
scheme/scale/weight enter the decoder checkpoint recipe and replay identity;
old terminal-reward optimizer/replay states cannot resume as the new objective.
MC returns are accumulated before replay shuffling and exposed as the separate
`mc_return` field. Forward passes may overwrite `action_value` without changing
these labels. The target is detached `G_t`, the sum of the actual remaining
round and placement rewards (gamma=1). This is observed-return regression plus
CQL regularization, not the original Bellman-backup CQL objective. It predicts
the logged behavior's continuation returns and retains sampling variance and
function-approximation error; it does not inherit TD-CQL improvement guarantees.
It removes the feedback from predicted future Q into the training target.

For the explicit comparator `target_kind=td double_q_learning=true`, the
detached target is standard one-step Double DQN:
`y = r + gamma * (1 - done) * Q_target(s', argmax_legal Q_online(s'))`.
Only active nonterminal next observations enter either next-state forward.
In TD mode complete logged returns are diagnostics only. MC and TD are separate
objectives, never blended. CQL reduces over legal actions exactly; illegal actions and
padding have no contribution. No GRP or learned reward predictor is needed.

The former `mc_weight` and `critic_warmup_steps` options have been removed.
Checkpoints binding either former experiment are rejected as different recipes;
their retained models and historical evidence are not rewritten. Higher BC
action agreement caused by retaining or freezing BC features is not evidence
that CQL improves play. Playing strength requires independent game outcomes.

With gamma=1 in TD mode, a shared offset in both Q networks cancels from nonterminal
TD residuals and the CQL regularizer. The implementation reports terminal and
nonterminal diagnostics rather than hiding this property by mixing objectives.
This standard TD recipe does not guarantee that values remain well calibrated
or that the resulting policy surpasses BC.

MC uses `Loss (MC Return)` and `MC/*` residual metrics. An additive Q offset
leaves MC labels unchanged and creates a return-regression correction on all
observed actions. It does not merely clamp Q or change its displayed scale.
Target kind and the absence/presence of a target network bind checkpoints;
old TD optimizer/replay states cannot resume as MC. MC snapshots contain
`encoder.pth` and `decoder.pth`, with no target-network parameter files.

The 2026-11-05 MC functional check passed 47 CPU checks and real CUDA FP16
training on 627 decisions. Continuous training and interrupted/resumed training
matched model, optimizer, scheduler, scaler, replay, cursor and RNG exactly.
Independent raw-row settlement accounting reproduced every observed-return
label (maximum absolute error 1.85e-7), and only one current-state forward per
batch occurred: no next-state Q or learned reward model was called.

A separate real-data check added 20 to the output bias and trained every
parameter for 300 FP32 updates. Mean observed-action Q fell from 19.97996 to
1.92668; return MSE fell from 403.75117 to 12.51020. Labels remained bitwise
unchanged. An independent checkpoint readback reproduced these measurements.
This establishes a working correction for the injected offset on these data;
it does not establish full calibration or improved playing strength. Four
native games exercised inference after only 20 main updates and all finished
fourth; that functional checkpoint is not a replacement for the retained BC.
No new long MC training or matched strength comparison was performed.

Receipts: [MC CUDA and native functional check](../../runs/remote-data/mc-cql-acceptance.json),
[independent readback](../../runs/remote-data/mc-cql-independent-acceptance.json).

In TD mode `TD/Terminal count`, `TD/Nonterminal count`, their separate residuals and the
overall absolute residual are logged alongside the Q/target/behavior-return
diagnostics. Zero-count groups report zero with an explicit count of zero.
Compare values against the **current** reward objective, not the old terminal
placement support of [-1.5, 1.5]. Changed reward and training recipes reject
old decoder resumes rather than silently reusing incompatible optimizer states.

BC/CQL presets use
AdamW: matrix weights decay, while embeddings, scalar vectors, biases and
normalization scales do not. New online runs use the same parameter grouping
and explicitly select fused AdamW on CUDA. An old online checkpoint retains
its original flat group, decay settings and execution flags when resumed;
it is never silently regrouped. Master parameters remain FP32; fp16/bf16
autocast is supported, with scalar heads/loss reductions in FP32.

For offline BC/CQL, dense decoder training requires
`device.dtype=float32`. Use `device.amp_dtype=float16` or
`device.amp_dtype=bfloat16` (`bf16` is an alias) for CUDA mixed precision;
`float32` or `null` disables autocast. BF16 checks the selected GPU's
support and reports an error if unavailable. CPU training retains FP32
execution. FP16/FP64 master-parameter configurations are rejected before
reading data, rather than failing later in the FP32 action head or CQL loss.
Legacy encoder precision configurations retain their existing behavior.

Resume uses the checkpoint's original precision recipe. Enabling BF16 does
not silently convert an existing FP16 training checkpoint.

`examples/training/decision-online.yaml` configures the existing online
entry point with Huber CQL and a separate immutable offline pool. The online
FIFO cannot overwrite it. A deterministic Algorithm R reservoir samples
transitions from the entire audited train file with bounded memory. All
source decisions have equal inclusion probability; this is transition
sampling, not equal weighting of games. The initial build reads the complete
source, so preparation time grows with corpus size.

`offline-pool.pt` saves the source SHA256, cursor, retained source row indices,
selection RNG and replay sampling RNG. Construction checkpoints every minute
and at completion. If interrupted before `latest.pt` exists, rerun the same
configuration in the same output directory; only a directory containing pool
preparation files is accepted. Once `latest.pt` exists, use normal `--resume`.
Both completed replay pools are self-contained in that checkpoint. Old
prefix-pool checkpoints keep their saved members and sampling state.

By default half each batch comes from offline replay; before online replay
reaches learning_starts, all samples come from the offline pool.
Offline/online CQL coefficients and pool RNGs are independent.

The Q-only training view shares candidate validation between backbone and
head, and omits the inference actor's unused action selection. It shares
parameters with the portable actor, preserving export/state keys. RoPE sine
and cosine are computed in FP32 once per forward and shared by Q/K and all
layers, including checkpointed layers. Online metric tensors transfer to the
host together at the end of an iteration. Finite-loss/gradient checks and
terminal-safe next-state filtering remain active.

## Caches, export and exact recovery

`DenseDecisionDecoder.incremental` and `cached_decision` accept one frozen
eval observer and an explicit `DecisionContext(environment, game, seat,
round_instance, model_version)`. The round identity must distinguish honba
instances of the same nominal round.

Only BOS/history K/V persist. Current scores, hand fields, legal summary and
decision token are recomputed. Changed prefixes, truncation/rollback, stream
identity, weight version, device/dtype or autocast invalidate caches.

```python
actor.eval().requires_grad_(False)
index, cache = cached_decision(
    backbone, head, observation, context=context, cache=cache
)
```

The portable TensorDict actor at `snapshots/model.kanachan` uses the complete
stateless path, including in the native simulator. Its current four-field
interface does not provide reliable game/round identities, so it must not
reuse caches by batch-slot position. Applications with those identities may
opt into the explicit cache API.

New BC/CQL checkpoints bind schema, backbone/head settings, data SHA256,
loss/reward/optimizer recipe, RNG and hashes of snapshot components.
Exact resume rejects a changed mask/recipe or mismatched bytes even when
tensor shapes agree. Online snapshots include both replay pools, optimizer,
target, scaler, exploration, opponent and RNG state. Deterministic synchronous
sampling is required for exact online comparisons.

Tests cover causality, permutations, padding, legal/tied actions, terminals,
Double DQN/CQL mathematics and gradients, AMP, activation checkpointing, cache
equivalence/isolation, exports, and interrupted BC/CQL/online training.
`encoder.history_attention=bidirectional` supplies the controlled no-cache
ablation while keeping tokens, normalization and head fixed.

Functional tests and short pilot training do not establish stronger play.
Measure held-out BC metrics, Q/TD/CQL stability, actual throughput/latency
and complete native games with paired seeds and seat rotations. Larger
training also requires independently accepted data and a validated rules engine.

References:
[discrete CQL implementation](https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/qlearning/torch/cql_impl.py),
[CQL paper](https://arxiv.org/abs/2006.04779),
[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
[LitGPT](https://github.com/Lightning-AI/litgpt),
[Transformers](https://github.com/huggingface/transformers),
[Mortal](https://github.com/Equim-chan/Mortal).
These are implementation references, not dependency version pins.

## Verified functional scope (2026-11-04)

Tiny was checked on the existing RTX 3070 Laptop GPU in a separate remote
source copy. The production sources, task definitions and old BC model were
preserved.

- fp16/bf16 forward and checkpointed gradients passed on 32 real observations.
- The CUDA pilot performed 16 BC updates and 20 CQL updates. Interrupted CQL
  reproduced online/target parameters, optimizer, replay, scaler, cursor and RNG.
- Eight native evaluation half-games completed. The real online entry also
  completed continuous and interrupted runs with a 50/50 offline/online mix;
  all training states matched exactly after four updates per final run.
- Cache equivalence passed FP32/fp16/bf16, including a 74-event real history.
  The short-history, batch-one microbenchmark showed no cache speed gain:
  full 11.614 ms versus cached 11.912 ms (20 timed repeats). This does not
  measure complete training speed or establish performance on longer streams.

Exact recovery is verified for single-rank, zero-worker offline training and
deterministic synchronous online collection. Distributed/asynchronous exact
recovery is not established by these checks. The pilot models are functional
test artifacts, not strength-qualified replacements for the previous BC model.

Receipts:
[functional CUDA/native](../../runs/remote-data/decision-decoder-remote-acceptance-v1.json),
[independent state/cache checks](../../runs/remote-data/decision-decoder-independent-check-v1.json),
[real online entry and recovery](../../runs/remote-data/decision-decoder-online-entry-acceptance-v1.json).

## Implementation optimization verification (2026-11-04)

The existing remote development source was updated in place. No new
experiment directory or model/replay artifact was created. The original
source archive, BC model, production conversion and independent audit source
were retained.

With the same Tiny weights and real observations on the RTX 3070 Laptop,
FP32/fp16/bf16 candidate values and gradients matched the previous code
bitwise. The old flat-group AdamW checkpoint also reproduced its next CUDA
update exactly. One native half-game supplied 75 decisions; four successful
mixed CQL updates reproduced model, target, optimizer, scaler, both replay
pools and RNG after an in-memory checkpoint/resume.

Alternating measurements under fp16 and the explicitly selected SDPA math
backend gave these median wall times:

| Microbenchmark | Previous | Updated |
| --- | ---: | ---: |
| Q forward, batch 1 | 14.68 ms | 10.85 ms |
| Q forward, batch 32 | 14.47 ms | 10.49 ms |
| CQL forward + backward, batch 32 | 74.37 ms | 51.03 ms |

Forward used 24 repeats and forward/backward used 12. These measurements
exclude simulation, input I/O and optimizer steps. They establish a local
implementation improvement, not complete training throughput or playing
strength. The Q training view benefits from sharing validation and omitting
unused action selection; portable actor export remains compatible.

[Optimization receipt](../../runs/remote-data/decision-optimization-acceptance.json)
records the checks and benchmark scope.

## Offline throughput and diagnostics

BC and CQL compute the gradient clipping norm once per update. Nonfinite
gradient values still skip the update and reduce the AMP scale; an overflowing
norm with otherwise finite gradients remains an error. CQL batches its scalar
metric transfer without changing TD or conservative losses.

New episode replay buffers use uniform sampling without replacement with
constant-time swap removal (`format_version=2`, `swap_v2`). Loading a version-1
checkpoint retains its original ordered removal and version-1 serialization,
so its subsequent batches and RNG sequence remain reproducible.

For repeated sequential CQL scans with episode replay, `data_cache` can name a
single SQLite file in an existing directory. `data_cache_bytes` bounds that
database (default 256 MiB). The cache stores a fixed window of parsed raw
batches, then falls through to the original reader when full. It binds the
source SHA256, reader implementation and parsing settings; identifying a cache
still hashes the complete source. Cold, warm and disabled cache paths preserve
sample order and learner RNG. This is optional and requires zero-worker input;
it does not copy an entire large corpus or modify training data.

Online `offline_pool_cache` can point to a completed pool from an earlier run.
Source identity, capacity, seed and selection policy must agree. The pointer
never overrides a pool already checkpointed in the current output directory.

CQL `q_monitor_interval` (default 100 batches; 0 disables it) records legal-Q
quantiles and standard deviation, legal-action count and conservative gap
minus `temperature * log(number_of_legal_actions)`. The subtraction is only a
diagnostic. With contiguous episode returns, Q bias/MAE is measured against the
logged behavior return, not the value of the current policy. Distributed
quantiles are labeled as averages of rank-local statistics.

The FP16/BF16 CUDA check compared real BC (512 decisions) and CQL (627
decisions) training before/after this change. Model, target, optimizer, scaler,
replay, cursor and RNG matched bitwise, including checkpoint/resume and cold/
warm cache paths. Four alternating short-loop timing repeats measured BC
0.594 to 0.562 seconds and CQL 1.929 to 2.011 seconds. Thus these measurements
do not establish a CQL speedup or full-corpus throughput gain.

[Throughput verification receipt](../../runs/remote-data/decision-throughput-acceptance.json).

### Ordered afterstate preparation (2026-11-09)

For single-process `discard_v1` BC with `num_workers=0`,
`candidate_feature_workers=4 candidate_prefetch_batches=8` moves raw-line
parsing and native afterstate derivation to an ordered, bounded worker queue.
It preserves reader order, RNG, labels and the exported raw-observation model
interface. The default worker count is zero. Precomputed features are temporary
training tensors and use the same public-observation derivation as inference.

Sequential, single-process MC+CQL can also use `candidate_feature_workers=4`.
Before updates, it prepares canonical complete-return replay batches in RAM,
including afterstates. `prepared_mc_bytes` caps retained tensor bytes (default
3 GiB); leave additional RAM for replay, workers and the trainer. Exceeding the
cap raises an error. This option is for a bounded input subset that fits in RAM,
not an entire large corpus. It writes no feature cache to disk.

`prepared_mc_epochs` defaults to one. Additional epochs reuse the same prepared
batches, shuffling their order with `prepared_mc_seed + epoch`; they do not
resample individual transitions. This sampling policy, seed and epoch count
are bound to the checkpoint recipe. Recovery regenerates the cache, verifies
source/reward/tensor/order hashes, and restores its batch cursor. Checkpoints
contain the small identity and cursor, not the RAM cache. Prepared mode cannot
resume a conventional replay checkpoint or use `start_new_epoch`; retain its
original complete input and exact settings. TD+CQL and the default MC reader
remain available without prepared mode.

On one RTX 5090 with Torch 2.9/CUDA 12.8, a controlled 65,536-row BC comparison
used the same 512-dimensional, eight-layer model, batch 256 and BF16/SDPA.
Ordered parsing/afterstates increased steady throughput from 1,813 to 2,529
samples/s (39.5%); final model weights were bitwise equal. GPU utilization
averaged about 63% and 87% during the corresponding measurement windows.
A separate bounded MC+CQL pilot with batch 512 sustained about 2,560 samples/s
and 95–96% GPU utilization after preparation. These are workload-specific
throughput measurements, not playing-strength results. Strict recovery and
canonical first-epoch update equality are also covered by regression tests.

Keep checkpoint spacing based on measured wall time and a disk reserve; these
options do not enable more frequent checkpoints or relax integrity checks.

## IQL and optional final-rank supervision

The [Mahjong features and AWR diagnostics guide](decision-features.md) covers
the optional `decoder.candidate_features=discard_v1` branch, its native build,
observer-only derivation/cache contract and global advantage-weight checks.

The existing IQL and AWR entry points now accept `--config-name decision`.
See [the decision IQL guide](iql/decision-decoder.md) for the paper/author-code
comparison, BC transfer, separate V export, policy extraction, checkpoint
requirements and the default-disabled final-rank auxiliary experiment.
The BC, MC+CQL and TD+CQL paths above remain available. Functional acceptance
does not establish that IQL or the auxiliary objective improves play.

## Historical CQL feature-freezing trial (2026-11-04)

This retained trial used logged-return mixing 0.5 and a 2,048-update
feature-freezing phase. Both mechanisms have since been removed from the
current training implementation. Its results do not validate the current
standard TD-CQL recipe, and higher action agreement under frozen features
must not be interpreted as evidence of policy improvement.

The first controlled repair stabilized values but still produced weaker play.
It was retained as a failed result. A subsequent fixed-budget comparison used
the same 65,536 sampled training transitions and 2,048 successful updates per
CQL arm, with new native evaluation seeds. Each arm played 64 half-games
against three copies of the retained encoder BC, grouped into 16 matched
four-seat rotations:

| Policy | Mean rank (lower is better) | Fourth-place rate |
| --- | ---: | ---: |
| Original decoder BC | 2.796875 | 31.25% |
| CQL without feature protection | 3.375000 | 48.4375% |
| CQL with transferred hidden features and warmup | 2.937500 | 37.50% |

The protected candidate's paired rank difference from unprotected CQL was
-0.4375, with a 95% seed-group bootstrap interval [-0.75, -0.1875]. Its
difference from BC was +0.140625, interval [-0.28125, 0.4375]. This supports
an improvement over this CQL control but does not establish superiority or
non-inferiority to BC. The candidate ended exactly at the warmup boundary;
these games do not establish strength after extended joint training.

An independent readback reproduced the complete sampled-transition sequence,
all 152,703 validation decisions, candidate checkpoint/export weights, and
native statistics/paired intervals. The BC backbone and hidden head remained
bitwise identical during warmup. The real CUDA functional pilot crossed the
freeze/unfreeze boundary and exactly resumed model, target, optimizer,
scheduler, scaler, replay and RNG. All 45 CPU checks passed. Existing BC/CQL
models and production data/audit sources were preserved; no model was promoted.

Receipts: [controlled comparison](../../runs/remote-data/cql-policy-acceptance.json),
[independent verification](../../runs/remote-data/cql-policy-independent-acceptance.json).
