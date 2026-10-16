# Improvement work — 13 October 2026

**Outcome: harder successor implemented and prospectively frozen; numerical work remains blocked.**

## Existing study

The actual isolated v2 reproduction attempt stopped before model import at
324.328125 / 304.1875 MiB free-plus-speculative memory, below 512 MiB.
Pageouts and swap were stable during that sample. The original guard was not
weakened, and no unrelated process was terminated. Its receipt is
`../../successor/itfm_v2/publication/itfm-v2-paper-20261011/replays/attempt-20261013-01/REPLAY_RECEIPT.json`.
The fresh source/scalar audit passed 228/228 (`V2_VERIFICATION.json`).
That is not a completed numerical reproduction or clean-environment replay.

## New work

`successor/itfm_v3` contains a separate 30-cell development protocol: five seeds,
six conditions, noisy cues, hidden value drift, distractors, and capacity stress.
It implements a frozen learned base and tensor residual retrieval against base,
last residual, kNN1, uniform kNN4, functioning recent history, finite-history MLP,
GRU and local attention. Erasure, sign, spatial-mean and value-permutation
interventions hold retrieval keys fixed. The past-only privileged oracle is
isolated from ordinary model inputs. Model and retrieval choices use validation
only, with selection saved before development evaluation generation.

The stream is synthetic cue-response data with hidden temporal state. It is not
a PDE or public-dataset substitute. Parameters/state/FLOPs are not claimed equal;
the implementation records parameters, update counts, state sizes and timings.
The same-bank kNN controls can falsify tensor-memory advantage. No architecture
novelty or empirical improvement is claimed before execution.

Source includes causal/leakage tests, a miniature end-to-end engineering fixture,
raw tensors/checkpoints/loss retention, an independent retrieval and metric
calculation, checkpoint reexecution, exclusive output/ownership checks, and
resource/time/output-budget enforcement. One 30-minute, one-thread development
campaign is permitted only after exact-source runtime tests pass. Failed or
partial campaigns cannot automatically rerun or tune thresholds.

## Verification and actual limits

- 11 standard-library contract tests passed, both normally and under optimized Python.
- All eight Python files parsed/compiled in memory; Ruff lint and format checks passed.
- Numerical tests and the end-to-end fixture were **not executed**.
- The initial source freeze and blocked receipt remain. Final static review made
  validation survive optimized Python and added saved-checkpoint reexecution;
  the final source is bound by `freezes/development-2.json`.
- Final numerical-test admission stopped at 106.09375 / 99.4375 MiB; pageouts were
  stable within the sample and swap used was 1480.44 MiB. No Torch worker launched.
- No v3 development cell ran, no campaign budget was consumed, and no score exists.
- All 440 protected files from v1, v2 and the predecessor component results are
  byte-for-byte unchanged (`PRESERVATION_AFTER.json`). The existing dirty relocation
  and root Git identity were preserved; no commit, push or public release occurred.

## Resumption

From `successor/itfm_v3`, the README supplies exact guarded test/run/verify commands
for the final freeze. Use fresh output directories. First pass runtime tests;
then run the declared campaign once; then verify all raw outputs. If a condition
fails its scientific rule, retain NOT_SUPPORTED. If execution stops, retain
INCOMPLETE/FAILED rather than infer a scientific result.

V2 numerical reproduction, a clean dependency environment, independent external
review/reproduction and manuscript PDF/release work remain separate unfinished
work. Public ITFM-R and protected confirmation remain unopened.
