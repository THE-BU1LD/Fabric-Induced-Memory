# ITFM v3: noisy and drifting residual-recall development study

This is a separately versioned successor to the frozen ITFM v2 exact-lookup
study. The original FIM negative result and v1/v2 evidence are unchanged.
**No v3 scientific result exists yet.** Source implementation and static checks
do not establish runtime correctness, a memory advantage, or publication readiness.

## Scientific question

Does a distance-weighted, spatial residual memory retain a useful forecasting
effect under imperfect cues, drifting values, distractors and finite capacity,
when compared with simple lookup and trained sequence controls? Can value-only
interventions remove that effect without changing keys, training or data?

This is a robustness/diagnostic development study, not a claim of architectural
novelty. It is a constructed continuous event stream, **not a physical PDE,
natural dataset, or public-benchmark reproduction**. Public ITFM-R remains a
separate unadmitted campaign. No external dataset or third-party implementation
is silently substituted with this generator.

## What is actually implemented

- Six predeclared conditions: clean, noisy cue, value drift, distractors,
  capacity stress, and their combination; five development seeds each.
- Fresh per-owner hidden residual processes; disjoint training/validation/
  evaluation owners; owner IDs are opaque and independent of the latent values.
- A learned cue-only MLP base, frozen before all residual predictors.
- Tensor memory, top-1 kNN, uniform top-4 kNN, actual recent-window retrieval,
  last observed residual, finite-history MLP, full-history GRU, local attention.
- Erasure, sign inversion, spatial averaging and within-owner value permutation
  using unchanged keys and retrieval decisions. A privileged anchor-ID oracle
  provides a separate past-only diagnostic, with the same FIFO capacity.
- Validation-only choice of width/temperature and residual scale; explicit
  selection receipt written before evaluation data generation.
- Raw cues, targets, predictions, source indices, owner IDs, masks, training
  losses, all candidate checkpoints, selected settings and output hashes.
- A second retrieval calculation and float64 metric recomputation from saved
  tensors; failures, resource blocks and partial fits stay on disk.

See [METHOD.md](METHOD.md) for equations, information access, estimands,
stopping rules and limitations. The machine-readable contract is `protocol.json`.

## Safe execution

Commands below run from this directory. No packages are installed automatically.
The intended numerical environment is Python 3.11 with Torch 2.12.1, matching
the existing v2 replay requirement. This is a version constraint, not a claimed
clean environment or a complete historical dependency lock.

```sh
rtk python3 run.py check
rtk python3 -m unittest -v test_standard
rtk python3 run.py test --freeze freezes/development-2.json --output runs/runtime-tests-1
rtk python3 run.py run --freeze freezes/development-2.json \
  --test-receipt runs/runtime-tests-1/RECEIPT.json --output runs/development-1
rtk python3 run.py verify --freeze freezes/development-2.json \
  --input runs/development-1 --output runs/verification-1
```

Existing output paths are refused. After a blocked attempt choose a fresh name.
`freezes/development-2.json` binds the supplied source. The first freeze and its
blocked test receipt are preserved; before any numerical execution, static review
strengthened optimized-Python validation and checkpoint verification. For a later
pre-outcome engineering revision, `run.py freeze --output freezes/UNUSED.json`
creates a new source identity without replacing either earlier freeze.
A real campaign consumes the one-campaign budget before launching; a failed or
partial campaign must be adjudicated, not silently rerun. A changed scientific
protocol after outcome access needs a new version. Standard-library contract
tests and static compilation do not import Torch. Numerical tests, data creation,
training and raw verification require two samples above 512 MiB, stable pageouts
and swap, 3 GiB scratch, and an exclusive study lock. Admission is rechecked
before numerical import and during the campaign; the parent enforces a 30-minute
process-group timeout and one thread. Tests have a separate 120-second ceiling.
No paid compute, protected confirmation or public run is authorized by this file.

`runs/` and `freezes/` contain exclusive receipts, not generated success claims.
Do not remove a live lock. `RESEARCH_STATE.json` records the actual execution state.

## Prior-art boundary

Primary abstracts checked on 13 October 2026:

- [Zoology / MQAR](https://arxiv.org/abs/2312.04927): associative recall is an
  established evaluation setting, with attention versus gated-convolution
  comparisons. This study does not reproduce their language-model experiments.
- [MemNO](https://arxiv.org/abs/2409.02313): memory in time-dependent PDE models
  and partial/noisy observations already have theoretical and empirical prior
  art. This generator does not supply evidence about their PDE claims.

These sources motivate stronger controls; checking their abstracts is not a
systematic novelty review. No new architecture or state-of-the-art claim is made.
