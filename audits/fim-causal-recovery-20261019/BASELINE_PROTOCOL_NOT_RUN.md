# Baseline-first falsification protocol — NOT RUN

19 October 2026. This is a prospective development protocol for the maintained
v2 estimator. It is not a new v4 concept, an extension of a consumed allowance,
a confirmatory protocol, or evidence that any added comparison has executed.
The existing one-seed software smoke has already been inspected and is development
data. Its results must never become a protected test for a redesigned method.

## Question and admission

Does matching observed contexts improve residual correction beyond inexpensive
predictors and correctors that receive the same mature observations, after
accounting for the base model, cold starts and total cost?

The current estimator is fixed context-cosine kernel regression on residuals.
The broad combination of a frozen forecaster, refreshed memory and calibrated
correction is established prior art. No distinct scientific contribution has
passed IDEA READY. Necessary correctness repair is admitted; implementing a new
learned architecture to chase the smoke score is not the next step.

The strongest falsifier is that a validation-selected simple forecaster and
conventional corrector match or beat retrieval under the same information and
budget. A full-versus-weak-MLP improvement alone does not support the hypothesis.

## Exact prospective comparison

Use the existing `bounded_periodic_field_v1` generation contract as a development
diagnostic only. Pin the implementation commit, complete resolved configuration,
raw tensor hashes and split streams before each admitted execution. Do not call
this a PDE benchmark, a public forecasting dataset, or evidence of distribution
shift robustness. Public data require a separate identity, license/terms, split,
baseline-reproduction and cost admission.

One full generated repetition is the independent unit. Owner trajectories are
nested within that repetition; temporal samples and spatial sites are dependent.
If subsequently admitted, seeds 101, 202 and 303 are a three-repetition development
design. This is not a claim of sufficient sample size for confirmation. Seed 101
has already been inspected; all three remain development-only.

### Forecasters

Compare persistence, the current MLP and an explicit finite-history linear
predictor fitted using training owners only. The predictor uses the same three
context frames. Fit a ridge predictor with an intercept, no target-dependent
normalization and the fixed candidate penalties `{0.0001, 0.01, 1}`; select the
penalty using validation only. No test-owner labels enter its fitted parameters.

The two-epoch MLP is a software fixture, not a strong benchmark. A future MLP
training budget must be declared before comparison and applied equally to every
control sharing that backbone. Select a backbone using validation, with ties
resolved by the fixed order persistence, linear predictor, MLP. Retain all
candidate results and fitting costs. The selection rule is proposed, not executed.

### Residual correctors

For each frozen, identified backbone, form its own residuals. Never reuse an old
backbone's residual bank after changing weights or representation. Use only
same-owner entries whose explicit availability time is at or before the query,
and whose forecast-origin age meets the same minimum-age rule. Capacity is 16
per owner with FIFO eviction in this first bounded protocol.

| Corrector | Matched information | Distinguishing computation |
|---|---|---|
| None | Same current observed context | Zero correction |
| Last residual | Same mature history access; one value used | Most recent eligible residual |
| FIFO mean | All retained eligible residuals | Uniform mean |
| Uniform top-k | Same selected keys and values as full retrieval | Equal weights within selected top-k |
| Exact cosine kernel | Same candidates, keys, values, capacity and timestamps | Independent implementation of current formula |
| Finite-window ridge | Same retained contexts and residuals | Regularized local linear residual predictor |
| Current retrieval | Same legal pool | Deterministic cosine top-4, temperature .2 |

For local residual ridge, use the same fixed penalty candidates above, select
using validation only and refit only on each query's legal retained history.
Include an intercept and define an empty history as zero correction. Report
conditioning failures and solve failures; no silent fallback may hide them.
The linear predictor and local ridge are different baselines with different
fit costs, both of which must be disclosed.

For the main estimator comparison, select each corrector's scale from
`{0, .25, .5, 1}` using validation MSE; ties take the smaller scale. Select the
strongest conventional comparator on validation before viewing test results.
Comparator ties use the table order, excluding the exact implementation check
and current retrieval. The exact-kernel comparison checks implementation
equivalence and cannot establish a novel mechanism.

### Mechanism interventions

Apply no memory, no retrieval with writes retained, reset before every query,
same-owner mature-pool key mispairing, same-owner mature-pool value mispairing,
and predeclared relevant/irrelevant deletion where the task defines relevance
without its unseen outcome. Freeze the full method's validation-selected scale
for these interventions so a second calibration cannot hide the effect.
Separately recalibrated intervention results may be a labeled secondary analysis
only if declared before execution.

The repaired `random_value` API is deterministic cyclic mispairing, not a uniform
random permutation. Its eligible donor pool and actual donor weights are tested.
It was not one of the five smoke comparisons. Report how many queries have zero,
one and multiple eligible donors: a singleton cannot test association. Preserve
all owner, maturity, capacity and candidate constraints in both key and value
controls. Do not describe an intervention that introduces future or foreign-owner
labels as a fair memory control.

## Metrics, uncertainty and costs

The proposed primary endpoint is the paired difference in overall test MSE:
current retrieval minus the validation-selected conventional comparator. Compute
per-owner MSE first and weight owners equally within a repetition, then weight
repetitions equally. For equal-sized owners this equals pooled element MSE;
unequal-sized future data must retain the declared weighting.

Report the three paired differences individually and their mean/range. With
three development repetitions, do not supply a confirmatory significance claim
or treat a bootstrap over time points as independent evidence. A later protected
study needs independent-unit precision planning, a practical effect threshold,
fixed multiplicity handling and an admitted sample size before its outcomes are
viewed. Those choices are unresolved here; confirmatory methodology is NOT READY.

Secondary diagnostics: cold-start and warm-memory error, eligible count, donor
age and weights, owner-isolation violations, retrieval coverage, occupancy,
forgetting after a separately declared shift, and per-horizon autonomous error.
The cold/warm split motivated by the current smoke is post hoc for that smoke;
predeclare it for future development. Never remove cold starts from the primary
metric. Autonomous residual correction must not write unobserved future targets.

Record parameters, fit/update count, all search trials, distance work, retained
bytes, reads, writes, evictions, end-to-end elapsed time, retrieval/update time,
and process/device peak memory. The backbone is trained once per declared trial
and shared by its controls; report both total study cost and incremental cost.
Equal optimizer updates are not equal compute. A more expensive corrector also
needs a comparison with the same additional budget spent on a conventional
predictor before an efficiency advantage can be claimed.

## Budget and stop rules

Proposed maximum: one compute lane, three development repetitions, no more than
300 seconds per repetition and 1,200 seconds for the complete process including
setup and shutdown. This is a proposed ceiling, not a runtime estimate or a
newly launched allowance. Admit only after the added baseline implementations,
source-bound dry run and resource/cost check exist. This document does not execute
the repository's older multi-hour `development.json`.

Stop on memory/disk refusal, nonfinite values, information leakage, ambiguous
optimizer or representation identity, corrupt/incomplete output, or budget
expiry. Preserve every failure. Do not add seeds, tune after test inspection,
reopen a completed/consumed predecessor, or retry simply to obtain a positive
result. A conceptual change requires a separately recorded hypothesis and gate.

## Prior art and competing explanations

1. [FreshCast, arXiv:2610.07834v2](https://arxiv.org/html/2610.07834v2): frozen
   forecasting with refreshed memory, relational weighting and calibrated
   correction. Its learned/readout details differ from this fixed cosine
   estimator; broad architectural ingredients do not establish FIM novelty.
2. [Delay-aware adaptation audit, arXiv:2610.12232v1](https://arxiv.org/html/2610.12232v1):
   actual observation availability and inexpensive linear correction baselines
   matter. Its numerical findings are not transferred to FIM.
3. [TAFAS, arXiv:2501.04970v1](https://arxiv.org/html/2501.04970v1): partial-feedback
   calibration is relevant prior art. Original forecasts and later revisions
   require separate scores. The current fixture has fully observed one-step
   arrivals and does not test the partial-feedback task.

A stronger initial predictor plus conventional correction is the preferred
diagnostic direction. Level/amplitude/age descriptors are established alternatives
to normalized shape keys, but no current measurement identifies missing features
as the cause. Coordinate-level partial-feedback storage addresses a different
task and cannot be admitted as a scientific improvement from this fixture.

**Status: NOT RUN; additional baselines and interventions above are not yet
implemented in the maintained five-variant smoke runner. Scientific contribution
and confirmatory admission remain blocked.**
