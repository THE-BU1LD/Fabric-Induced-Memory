# FIM bounded engineering evidence

**End state: BLOCKED BEFORE EXECUTION.**

Validated 0 of 5 expected cells. Missing mandatory cells: periodic-seed-1601, periodic-seed-1709, periodic-seed-1801, periodic-seed-1907, periodic-seed-2003.

## Methods

One existing residual MLP checkpoint per seed is selected by base-model validation MSE. Fixed observed-context memory controls share that checkpoint; their scalar correction is selected on validation and frozen for test. Baseline comparisons include the same base model, last observed residual, and persistence. Source, configuration, data, and declared artifact checksums are verified by the maintained completion validator before a row is admitted.

Protocol groups hash all configuration fields except seed. Incompatible shapes, precision, device, budgets, or purposes remain separate. Paired effects use independent seeds, with negative memory-minus-comparator MSE favoring memory. Exact two-sided sign-flip tests (up to 20 seeds) and a deterministic 10,000-resample seed-pair bootstrap are descriptive; Holm adjustment covers the three declared comparisons within each protocol group.

## Results

No validated paired results are available. No result chart or numerical finding was generated.

All seed outcomes, including baseline wins, are retained in per_seed_metrics.csv and paired_differences.csv. Autonomous base rollout metrics are separate in autonomous_base_horizons.csv. Runtime and CUDA peak allocation are copied from actual retained training/evaluation logs; unavailable host memory remains blank. Shared training times must not be summed across memory conditions.

## Missing and failed cells

- periodic-seed-1601: PENDING; validated=False; Upstream access/readiness gates blocked; never launched.
- periodic-seed-1709: PENDING; validated=False; Upstream access/readiness gates blocked; never launched.
- periodic-seed-1801: PENDING; validated=False; Upstream access/readiness gates blocked; never launched.
- periodic-seed-1907: PENDING; validated=False; Upstream access/readiness gates blocked; never launched.
- periodic-seed-2003: PENDING; validated=False; Upstream access/readiness gates blocked; never launched.

## Claim and evidence map

| Claim | Evidence | Scope |
|---|---|---|
| Completed engineering execution | 0 validated completion receipts and artifact inventories | Synthetic fixture only |
| Memory-relative empirical effects | Per-seed MSE, paired differences, and seed-level summaries | Observed seeds and frozen settings only |
| Autonomous rollout behavior | Per-horizon base-only errors and retained raw predictions | Finite fixture horizon; no stability certificate |
| Scientific efficacy or general superiority | Not established | No public benchmark or independent replication |
| Durable retrieval, final quota, and stopped platform sessions | Not established by this JSON-only analysis | Require separate platform receipts |

## Limitations

- This is a synthetic engineering/development fixture, not a public benchmark or confirmatory study.
- Independent repetitions are seeds. Timesteps, trajectories within a seed, batches, and spatial points are not statistical replicates.
- Each seed trains one MLP; memory variants reuse its frozen checkpoint. Training time and parameter counts are shared, not separate matched compute trials.
- Memory scale is selected on validation only; test results are descriptive and must not be used to revise this protocol.
- Few seeds provide coarse sign-flip inference and unstable bootstrap intervals. Inconclusive differences do not establish equivalence.
- Exact sign-flip p-values require independent, sign-exchangeable paired effects under the null; that assumption is not established by this fixture.
- Autonomous base rollout is a separate finite-horizon diagnostic with memory disabled; it does not establish long-term stability or memory efficacy.
- No novelty, general architecture ranking, venue readiness, acceptance, or independent scientific replication is established.

## Manuscript and venue boundary

An engineering methods description and honest bounded descriptive results can be drafted from valid rows. A scientific efficacy preprint is not justified by this fixture alone. A named conference requires a suitable public or independently validated task, strong external comparators, adequate repetitions, independent reproducibility, literature/novelty review, and venue-specific expert review.

Observed final Kaggle quota, GPU/session duration, durable retrieval, and platform session-stop confirmation are unavailable to this analyzer. Local output files or an archive do not establish a second persistent copy.
