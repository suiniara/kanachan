# Mahjong candidate features and AWR diagnostics

This stage implements the first two priorities: an optional discard afterstate
branch and global AWR weight diagnostics. BC, IQL/AWR, MC+CQL and TD+CQL reuse
their existing training loops. No ensemble algorithm or new reward is added.

## Optional candidate branch

The current v2 sparse tokens contain the observer's concealed tiles, drawn
tile and visible dora indicators. Public progression records encode discards,
chi, pon and kans. The new branch derives features from these same inputs;
it does not assert that the old token representation lacked the information.

Build [the native module](../../tools/features/README.md), then add
`decoder.candidate_features=discard_v1` to a dense decision configuration:

```bash
python -m kanachan.training.bc.train --config-name decision \
  decoder.candidate_features=discard_v1 \
  training_data=/data/train.bc.tsv batch_size=128 hydra.run.dir=/runs/bc-afterstate

python -m kanachan.training.iql.train --config-name decision \
  decoder.candidate_features=discard_v1 \
  bc_initial_model=/runs/bc-afterstate/snapshots/model.kanachan \
  reward_plugin=examples/training/iql-placement-reward.py \
  training_data=/data/train.rl.tsv batch_size=128 hydra.run.dir=/runs/iql-afterstate

python -m kanachan.training.awr.train --config-name decision \
  decoder.candidate_features=discard_v1 weight_diagnostics_interval=100 \
  q_model=/runs/iql-afterstate/snapshots/model.kanachan \
  value_model=/runs/iql-afterstate/snapshots/value-model.kanachan \
  encoder.load_from=/runs/bc-afterstate/snapshots/encoder.pth \
  decoder.load_from=/runs/bc-afterstate/snapshots/decoder.pth \
  training_data=/data/train.bc.tsv batch_size=128 hydra.run.dir=/runs/awr-afterstate
```

Architecture dimensions must match the source checkpoints. MC/TD+CQL accept
the same decoder option with their existing target/reward settings. An
explicit BC/AWR `decoder.load_from` may import an old global BC head into
the new head: global logits are retained and the added correction starts at
zero. Full checkpoint resumes remain strict. IQL/CQL BC feature transfer
continues to initialize fresh Q outputs rather than importing policy logits.

Each native discard ID below 148 removes its specified 37-type tile from the
observed concealed/drawn hand. Nyanten computes structural shanten. Each of
34 possible draws is checked for a strict shanten decrease, weighted by
`4 - observed_count[type]`. Observed counts include own tiles, current
indicators, discards and newly exposed meld tiles. A called discard is counted
once; added kan adds one tile, not a second four-tile meld. Red fives map to
their ordinary type for shape/ukeire and remain distinct for retention.

The 46 normalized columns comprise:

- a discard applicability mask, shanten, structural tenpai, live ukeire and
  live effective-type count;
- 34 live effective-type bits;
- three concealed red-five retention flags, concealed dora count, discarded
  red/dora indicators and remaining concealed-tile count.

Tenpai here describes hand shape. It is not a legal-ron label and does not
include yaku or furiten checks. Retention columns count the concealed hand;
meld bonuses are outside that count. Ukeire is availability under the public
observation, not a probability of winning or a single-player expected value.
Chi, pon, kan, wins and pass receive zero afterstate columns with mask zero;
their action identity and the decision representation still reach the scorer.

The original global logits are supplemented by a shared candidate MLP taking
the decision hidden vector, action-ID embedding and derived columns. It has
no candidate-position embedding. Padding stays masked, and permuting
candidates permutes scores. A zero-initialized final correction preserves
the original scoring function at migration.

`discard_afterstate_v1` binds the rules, column names and native width in
the head export and checkpoint schema. The raw BC/RL formats and backbone
input schema stay unchanged. A process-local LRU holds 4096 canonical
observation/candidate sets; returned feature tensors cannot mutate it. Existing
parsed-batch caches store raw observations and remain valid. No derived
columns are silently inserted into those caches.

The derivation runs on CPU, including when the scorer uses CUDA AMP. It
therefore introduces host transfers and a compiler graph boundary. Isolated
feature timing is not an end-to-end training throughput or FlexAttention
speedup claim. A bounded, read-only real-data audit is available:

```bash
python tools/audit_candidate_features.py --input /data/test.rl.tsv.gz \
  --limit 4096 --min-rows 4096 --output /runs/afterstate-audit.json
```

Empty input fails with a nonzero exit status. `--min-rows` defaults to one
and must be positive and no greater than `--limit`. For a release gate,
set both to the required coverage and require successful process exit before
accepting the report. Reports record the required minimum and actual checked
rows. Visible availability includes tiles that may be in opponents' hands or
the dead wall; it is not the actual drawable live-wall count.

## AWR diagnostics

`weight_diagnostics_interval=100` records the first batch, every hundredth
batch and the final batch. Zero disables it. Metrics are written to TensorBoard
under `AWR/` and to `awr-weight-diagnostics.jsonl` in the run directory.
The diagnostic interval is not an optimization hyperparameter and can change
on resume without changing the policy objective.

Recorded quantities include advantage/weight quantiles, nonfinite advantages,
weight sum/mean, clipping and underflow fractions, largest-sample and top-one-
percent weight share, and effective sample size:

`ESS = (sum(weight))**2 / sum(weight**2)`.

ESS fraction divides by global sample count. Clipping fraction counts
`advantage / beta > log(weight_clipping)` before clipping. Grouped sample
fractions and weight shares use the logged global action type, current round
stage, score band, current rank and number of publicly declared opponent
riichis. These are public-state bins, not hidden opponent labels.

All ranks first gather the detached sample rows, including differing local
row counts. Quantiles, ratios, clipping fractions and ESS are then computed
from the union. Averaging local ESS or normalized local weight shares would
give a different result. Diagnostics do not rescale weights, consume learner
RNG or alter the existing advantage-weighted BC loss. Large ESS does not
prove that Q/V estimates are correct.

## Subsequent label work

The existing four-class V auxiliary head remains terminal-only in this stage.
The current RL conversion already groups complete decisions by `(uuid, seat)`
in `bin/annotation/rl/rl.py`; this is the appropriate place to validate terminal
scores and attach the same own-final-rank target to each earlier decision,
before replay shuffling discards identity. Partial trajectories need explicit
missing labels. Split by whole game and normalize supervision over complete
games/observer trajectories so long games do not dominate. Global loss
normalization must also account for unequal labeled mass across ranks.
This should reuse the existing V head and version the extra label metadata.

Opponent-threat supervision needs a replay label generator before a head is
useful. It must reconstruct each opponent's hand, legal waits, yaku, permanent
and temporary furiten, and conditional ron payment at the observation point.
A missing ron event is not a safe-tile label. Unknown or incomplete replay
state must be masked; hidden hands may produce training targets but cannot
enter the observer input or feature cache. The current afterstate function
cannot accept those hidden inputs. These heads and dense rank labels are
not implemented or claimed validated in this stage.

The included IQL placement reward optimizes mean rank through utilities
`[1.5, 0.5, -0.5, -1.5]`. Fourth-place avoidance and platform-point objectives
require their own explicitly chosen utilities. Auxiliary rank cross entropy
does not select or change that reward objective.

## Research and acceptance boundaries

[Mortal observations](https://github.com/Equim-chan/Mortal/blob/main/libriichi/src/state/obs_repr.rs)
provide a reference for explicit shape, waits, furiten and single-player
tables. The feature branch here is a project adaptation, not those tables.
[Mortal GRP](https://github.com/Equim-chan/Mortal/blob/main/mortal/model.py)
and its [reward calculator](https://github.com/Equim-chan/Mortal/blob/main/mortal/reward_calculator.py)
are references for later full-game rank supervision and reward prediction;
this stage does not reproduce GRP's 24-permutation model.

[Mizukami and Tsuruoka](https://www.logos.ic.i.u-tokyo.ac.jp/~tsuruoka/papers/cig2015mizukami.pdf)
separate opponent waiting, winning-tile and score predictions.
[IQL](https://arxiv.org/abs/2110.06169) motivates checking concentration in
advantage-weighted policy extraction.
[EDAC](https://arxiv.org/abs/2110.01548) and its
[implementation](https://github.com/snu-mllab/EDAC) are references for a later
independent-model disagreement experiment. No uncertainty reweighting is
added here, and model agreement is not a confidence guarantee.

Functional tests and replay feature audits establish calculation and interface
behavior. They do not establish stronger Mahjong play. Strength evaluation
still needs matched data/budgets/opponents, held-out whole games, rotated
seats, multiple training seeds and paired placement/fourth-place intervals.
