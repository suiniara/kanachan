# Decision decoder: IQL, policy extraction and outcome supervision

For optional discard afterstate features and globally aggregated AWR weight
diagnostics, see [the feature guide](../decision-features.md).

The decoder uses the existing `training.iql` critic trainer and `training.awr`
policy trainer. BC, MC+CQL and TD+CQL remain selectable. No second IQL loop
or JAX dependency is introduced; the shared factories construct and describe
both legacy encoders and decision representations.

## Research and algorithm boundaries

[IQL equations 5–7](https://arxiv.org/html/2110.06169v1#S4) fit an expectile V
against dataset-action Q, regress Q to `r + discount * V(next)`, then extract
a policy by advantage-weighted BC. The
[author's critic](https://github.com/ikostrikov/implicit_q_learning/blob/master/critic.py)
uses minimum target-twin Q for V. The
[author's actor](https://github.com/ikostrikov/implicit_q_learning/blob/master/actor.py)
clips exponential weights at 100. A legal-action categorical policy can use
these objectives without porting the Gaussian actor, JAX/Flax, Gym or D4RL.

Only the logged action contributes to Q/V losses. Although the dense readout
computes an action vector, no maximum over unlogged actions or CQL penalty
enters IQL's target. V is a separate scalar readout. Terminal and zero-discount
next states are removed before the V forward: all-padding terminal records
are not valid decoder inputs.

The existing trainer accumulates Q/V gradients using the same pre-step V,
then steps all optimizers. This schedule is retained and checkpointed. It
differs from the [author's interleaved update](https://github.com/ikostrikov/implicit_q_learning/blob/master/learner.py),
which updates V before computing the Q target. This is an adaptation, not a
claim of numerical reproduction of the continuous-control experiments.

Policy extraction reuses AWR with the exported expectile V. This project's
`beta` is a temperature: `exp((Q - V) / beta)`; the paper's beta and the
author's `temperature` multiply the advantage. Use
`beta = 1 / author_temperature`. Existing AWR defaults (beta 1, clipping 20)
remain; `weight_clipping=100` selects the author's clipping. Omitting
`value_model` retains the older policy-expectation baseline, which is not
IQL's learned expectile baseline.

## Existing entry points

Run from the repository root in the project's supported training environment.
Substitute real paths. The included reward plugin adapts `PlacementReward`
to IQL's existing plugin interface without duplicating its reward definition.

```bash
python -m kanachan.training.iql.train --config-name decision \
  training_data=/data/train.rl.tsv batch_size=128 \
  reward_plugin=examples/training/iql-placement-reward.py \
  bc_initial_model=/models/bc/model.kanachan \
  expectile=0.7 hydra.run.dir=/runs/decision-iql

python -m kanachan.training.awr.train --config-name decision \
  training_data=/data/train.bc.tsv batch_size=128 \
  q_model=/runs/decision-iql/snapshots/model.kanachan \
  value_model=/runs/decision-iql/snapshots/value-model.kanachan \
  encoder.load_from=/models/bc/encoder.pth \
  decoder.load_from=/models/bc/decoder.pth \
  beta=1.0 weight_clipping=100 hydra.run.dir=/runs/decision-iql-policy
```

Transfer requires matching architecture dimensions. IQL copies the BC
backbone and hidden head features, initializes independent fresh Q readouts,
and starts V's scalar output at zero. BC logits are never imported as Q/V.
Independent Q readouts avoid keeping both deterministic critics identical
forever after transferring identical features. Each target starts as an exact
copy of its corresponding source critic.

IQL's `model.kanachan` is the target twin critic, exporting `action_value` and
a greedy `action` for diagnostics. Evaluate the subsequent AWR
`model.kanachan` for the complete IQL policy. Exact decision-action ties use
global action IDs to preserve candidate permutations. The V export is
`value-model.kanachan`. IQL requires two critics and rejects the formerly
ignored `double_q_learning=false` flag.

Decision defaults are discount 1 for final-placement returns and expectile
0.7; these are starting configurations, not validated Mahjong settings.
Contiguous replay remains opt-in through the existing `replay_buffer_size`
and `contiguous_training_data` options.

To create a reviewable training source archive, including the reward example
used above and a file-digest manifest, run:

```bash
python tools/package_training_source.py --output /runs/kanachan-training-source.zip
```

The output must be a new file. The bundle includes Python training sources,
documentation, examples and training tests. Native simulator binaries and
datasets are supplied separately.

## Optional final-rank auxiliary head

[Suphx section 3.2](https://arxiv.org/html/2003.13590v2#S3.SS2) trains a separate
GRU to predict final game reward from current and previous round summaries,
then uses differences in predictions as round rewards. This does not establish
that a rank head improves this decoder. Existing EOR/ranking models and
optional potential shaping remain available.

`final_rank_weight=0.1` enables an experimental four-class head sharing V's
representation and hidden readout. Zero, the default, creates no auxiliary
parameters. Its target uses the existing score-to-rank helper, observer
starting seat and starting-seat tie break. V's objective becomes expectile
loss plus weighted final-rank cross entropy, normalized over all decisions
for consistent gradient accumulation. The V export additionally exposes
`final_rank_logits` when enabled.

The current RL format contains final scores only on `next.end_of_game`
records. Only those records receive auxiliary supervision, independently of
reward-plugin changes to `next.done`. Intermediate zero-filled results never
mean first place. The model receives only sparse/numeric/progression/candidate
features: future outcomes are targets, never inputs. A batch without a final
record contributes zero auxiliary gradient. This initial experiment has
deliberately sparse supervision and does not propagate labels through
unverified shuffled trajectories.

A later dense experiment could label every prefix of a complete observer
trajectory before shuffling, with game identity and perspective checks. A
mean-return auxiliary head is another option; the upper-expectile V must not
be relabeled a mean-return predictor. Either extension needs held-out game
splits, calibration and loss-weight ablations. A frozen validated predictor
could later use the existing reward/shaping interface; the live auxiliary
head is not automatically a reward plugin.

## Recovery and evaluation

Dense IQL/AWR checkpoints bind schema, architecture, precision, hyperparameters,
data and relevant teacher/reward digests. IQL binds all five networks, three
optimizers, optional schedulers, critic and V exports, cursor and RNG. BC
transfer provenance is inherited on resume without reinitializing features.
Changed recipes or saved bytes are rejected. Resume with the same overrides,
`initial_model_prefix=/runs/decision-iql/snapshots initial_model_index=N`,
and leave `bc_initial_model` unset.

After DDP synchronizes the source networks, both target critics' parameters
and buffers are broadcast from rank 0 before any training forward. This also
covers Q readouts randomized after BC transfer. On checkpoint recovery, the
loaded targets are synchronized directly, preserving their saved Polyak lag;
they are not reset to the current sources. Old checkpoints produced by the
affected distributed BC-initialized path contain only rank 0's saved targets;
synchronization cannot undo V/optimizer updates made with divergent targets.

The opt-in `test_decision_iql_distributed.py` regression uses two CPU/Gloo
processes with different random seeds. It checks both target states and V's
logged-action Q supervision before and after updates, then resumes a saved
checkpoint and checks that target lag is preserved. Enable it with
`KANACHAN_TEST_DISTRIBUTED=1`. This checks distributed state handling;
it does not validate a multi-GPU/NCCL training run.

Exact continuation is tested for single-process, zero-worker streaming
training. IQL replay checkpoints still omit the shuffle buffer and retain
their existing restart/new-epoch requirements; exact distributed replay
recovery is not established.

Before claiming stronger play, compare BC, MC+CQL, TD+CQL, IQL+AWR, and
IQL+AWR with the auxiliary head under matched data, update budgets, opponents
and rotated seats. Use held-out games/seeds, multiple training seeds and
paired intervals for placement utility, mean rank and fourth-place rate.
Loss, action agreement, smoke games and auxiliary accuracy alone cannot
establish a playing-strength improvement.
