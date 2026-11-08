# Placement-first offline-to-online training

The first implementation is a single-host, single-GPU learner with batched
native simulation. It supports four-player Mahjong Soul half-games in Jade
or Throne room. The initial model must be a **scalar CQL export**; BC logits,
quantile exports and IQL twin actors are not interchangeable with this Q head.

Build the project's native extension on the training host before running
native simulation and online training. Python CPU unit tests do not replace
native/CUDA verification. The dense decoder has also been checked with the
existing Windows/MSVC native build.

The placement recipe, online entry point and evaluation CLI default to eager
execution. Enable compilation explicitly with `enable_compile=true` in the
training configuration or `--compile` in the evaluation CLI, after checking
that TorchInductor's compiler and CUDA toolchain work on the training host.

## Offline preparation

Keep the BC actor as an immutable evaluation baseline. Initialize the CQL
encoder from its encoder checkpoint; initialize and train the Q decoder
separately. The architecture must match the BC encoder.

For example, using shell-appropriate line continuation or one command:

```sh
python -m kanachan.training.cql.train --config-name placement training_data=/data/train.rl.tsv batch_size=128 encoder=bert_mini encoder.load_from=/models/bc/encoder.pth hydra.run.dir=/output/cql
```

This named recipe sets scalar Double Q, discount 1, contiguous complete games,
sequential input and bounded episode shuffling. The built-in reward is
`2.5 - final_rank`, or `[1.5, 0.5, -0.5, -1.5]`. Round boundaries do not end
the half-game. Optional `reward_model=/models/eor/model.kanachan` adds a public
round-state potential; cumulative shaping rewards equal the final utility
minus the initial potential. It must output `(N, 4, 4)` rank probabilities
or logits. This potential is a heuristic; its predictions do not replace the
actual terminal ranking. Use the same shaping model in offline and online
training, or use no shaping in both stages.

Existing external `reward_plugin=...` files remain supported. Explicitly
choose the discount appropriate to those rewards. The legacy unnamed CQL
configuration still defaults to discount 0.99 and 200 quantile intervals.

Single-rank sequential offline CQL now saves `replay-state[.N].pth` alongside
the model, optimizer and input cursor. It includes unread episode fragments,
the shuffled buffer, prefetched batch and RNG. Resume with the original
source file, buffer size and batch size:

```sh
python -m kanachan.training.cql.train --config-name placement training_data=/data/train.rl.tsv batch_size=128 encoder=bert_mini initial_model_prefix=/output/cql/snapshots hydra.run.dir=/output/cql-continued
```

The source path, size and modification time must match. Distributed replay
retains its previous restart behavior; it is not an exact multi-rank replay
checkpoint. An explicit new epoch restarts the data stream.

## Online training

Copy and edit [the example](../../../examples/training/online.yaml). Paths in
the YAML file are resolved relative to that file. Configuration numbers are
starting values, not validated playing-strength or cloud-budget estimates.

```sh
python -m kanachan.training.online.train --config /path/to/online.yaml
python -m kanachan.training.online.train --resume /output/online/latest.pt --iterations 2000
```

The loop freezes a policy for each collection batch, records its version and
opponent, generates complete half-games, applies rewards and terminal-safe
multi-step targets, and then performs Double DQN updates. CQL regularization
and epsilon exploration decay by collected decisions. The target network
updates only after successful optimizer steps. Evaluation never explores.

The default uses one-step Double DQN targets. Optional `td_steps > 1` uses
uncorrected sampled multi-step returns, which can be biased when replayed
behavior differs from the current policy. It is an experimental option,
not a requirement of this training route.

New runs use the same AdamW parameter groups as offline BC/CQL: matrix weights
decay, while embeddings, biases, normalization scales and vectors do not.
CUDA selects fused AdamW explicitly. A resumed old flat-group checkpoint
retains its original optimizer settings rather than changing algorithms
during recovery.

Optional `offline_data` builds an immutable Algorithm R reservoir over the
complete train file. Capacity bounds memory, not how much of the file is
examined. Every transition has equal inclusion probability, including the
tail. The initial full scan can take substantial time on a large corpus.
Its `offline-pool.pt` persists the source SHA256, source row indices, cursor
and independent selection/sampling RNGs. Before the first `latest.pt`, an
interrupted build continues by rerunning the same configuration and output
path. After that, `--resume` restores the saved pool without rebuilding it.
Existing prefix-pool checkpoints continue using their original members.

`latest.pt` is the authoritative atomic checkpoint. It contains the model
description, source and target weights, optimizer, AMP scaler, complete CPU
replay, exploration schedule, Python/Torch/CUDA RNG, opponent pool and
incumbent evaluation. `model.kanachan` is the current inference export;
`best.kanachan` is the initial or last accepted model. Opponents and an optional
reward model are copied and hashed inside the output directory. Retain the
whole directory when moving a run. Resume can override the total iteration
count, output directory and device; algorithm and evaluation settings stay
fixed. Restoring saved CUDA RNG requires the original CUDA device count.

Both sampled and promoted historical opponents are recorded. Fixed baselines
are retained; the active historical pool is bounded. Promotion requires the
upper bound of the paired average-rank difference to be below the configured
improvement threshold and the fourth-place increase upper bound to satisfy
the configured tolerance. The example uses zero tolerated fourth-place
increase. A failure to establish improvement keeps the incumbent.

Paired comparisons use a fixed baseline and identical wall seeds/seat
arrangements. Bootstrap resampling operates on independent seed sets, not
on the correlated seat rotations within a set. One seed set cannot provide
a confidence interval. Development evaluations are reused for checkpoint
selection and therefore do not constitute a final unbiased strength result.

## Independent evaluation

Rebuild the native simulation library after pulling these changes. The old
12-argument native call remains supported; the new optional seed fixes walls
and seat arrangements. Results are returned in simulation-ID order, independent
of thread completion order.

```sh
python -m kanachan.simulation.run --device cuda --baseline-model /models/fixed.kanachan --baseline-grade 12 --proposed-model /output/online/best.kanachan --proposed-grade 12 --room jade --mode 1vs3 -n 1000 --seed 902731 --output /output/final-evaluation.json
```

Use seeds excluded from training and development evaluation. The JSON includes
model hashes, seeds, per-game results, group IDs, fourth-place rate and cluster
confidence intervals. Deterministic inference is required for repeatable
decisions; different hardware or floating-point kernels can still affect
nearly tied Q values. Parallel exploratory sampling also depends on inference
order; restoring RNG does not promise bitwise identical asynchronous games.
Use inference batch size and concurrency 1 for controlled trajectory checks.

## Verification

```sh
python -B -m unittest discover -s test/training -p 'test_online_training.py' -v
KANACHAN_TEST_ONLINE_NATIVE=1 python -B -m unittest discover -s test/training -p 'test_online_native.py' -v
```

The second command runs actual complete games, tests seed reproducibility
across thread counts, and exercises native collection, CUDA AMP when available,
Q updates and recovery. It is a functional acceptance test, not a strength
benchmark. No formal cloud training is launched by these tests.
