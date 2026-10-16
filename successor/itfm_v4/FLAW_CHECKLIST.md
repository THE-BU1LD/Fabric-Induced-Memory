# FIM successor: flaw and repair checklist

Identified flaws and unresolved threats as of 15 October 2026; not a claim that every possible flaw has been discovered. Repair proposals are not recorded as completed fixes.

P0 blocks a trustworthy central claim or execution. P1 is required for the associated mechanism, robustness or release claim. An unchecked box means the repair and its required evidence are still outstanding. Source findings are scoped to the named version; they do not automatically invalidate a different historical experiment.

The executable audit receipt is `audit/SOURCE_DEFECTS_REFINED.json`. It confirms seven source/scalar findings and passes four audit controls. It does not establish a repaired model.

## F01 — Novelty

- [ ] **P0 — Frozen backbone plus refreshed retrieval and calibrated correction substantially overlaps existing work.**
- Cause: FreshCast specifies that combination. TAFAS, COSA and the 2026 delay-aware audit also cover partial feedback, gated adaptation and causal-delay measurement.
- Repair: Reject that combination as the sole new contribution; compare explicit material differences against primary methods.
- Required evidence: A claim-by-claim distinction and a falsifying experiment that does not reduce to an existing baseline.
- Current finding: `CONFIRMED_PRIOR_ART`.
- Source: https://arxiv.org/html/2610.07834v1

## F02 — Goal

- [ ] **P0 — Perfect forecasting is not an attainable universal acceptance criterion.**
- Cause: Irreducible conditional variance and potentially adverse retrieval corrections.
- Repair: Use a named task, fixed practical improvement, risk limits and finite budget.
- Required evidence: Conditional-risk derivation plus frozen empirical criteria.
- Current finding: `MATHEMATICAL_BOUNDARY`.
- Source: MATH_AND_FALSIFIERS.md

## F03 — Observation visibility

- [ ] **P0 — Public residual writer subtracts the entire target and ignores its observation mask.**
- Cause: Scorer truth and model observations share ForecastBatch.
- Repair: Separate ObservationEvent from scorer targets; gather visible coordinates before subtraction; retain validity and arrival metadata.
- Required evidence: Mutating every hidden target, including NaN/Inf, leaves predictions, keys, normalization and memory checksum unchanged.
- Current finding: `CONFIRMED_SOURCE_NOT_REPAIRED`.
- Source: ../itfm_v2/src/fim_sota/adapter.py:264

## F04 — Timing

- [ ] **P0 — Default target availability is event_time+1 despite variable lead_times.**
- Cause: A missing explicit availability field silently assumes one-step observations.
- Repair: Require coordinate-level arrival times and original forecast ID; reject ambiguous availability.
- Required evidence: Multi-horizon delayed and out-of-order reveal tests; zero future contributions.
- Current finding: `CONFIRMED_SOURCE_NOT_REPAIRED`.
- Source: ../itfm_v2/src/fim_sota/adapter.py:276

## F05 — Metrics

- [ ] **P0 — Evaluator and verifier compute different relative L2 reductions.**
- Cause: Mean sample ratios versus pooled energy ratio.
- Repair: Freeze one estimand and reduction order; use independent implementations of that same definition.
- Required evidence: Unequal-energy fixture: mean 0.5 versus pooled sqrt(1/101); repaired paths must agree.
- Current finding: `CONFIRMED_EXECUTABLE_AUDIT`.
- Source: audit/SOURCE_DEFECTS.json

## F06 — Numerical validation

- [ ] **P0 — Non-finite claimed or recomputed metrics can pass verification.**
- Cause: Comparisons involving NaN evaluate false; infinity minus infinity becomes NaN.
- Repair: Reject nonfinite inputs and nonfinite intermediates before tolerance comparisons.
- Required evidence: NaN on either side and infinity on both sides must all fail.
- Current finding: `CONFIRMED_EXECUTABLE_AUDIT`.
- Source: audit/SOURCE_DEFECTS.json

## F07 — Matrix integrity

- [ ] **P0 — Duplicate cells and families can satisfy required counts.**
- Cause: Decision checks list length rather than exact unique expected identities.
- Repair: Construct expected matrix from frozen protocol and derive outcomes from verified raw rows.
- Required evidence: Duplicates, missing IDs, extra IDs, wrong seeds and mismatched sources must fail.
- Current finding: `CONFIRMED_EXECUTABLE_AUDIT`.
- Source: audit/SOURCE_DEFECTS.json

## F08 — Decision completeness

- [ ] **P0 — Secondary regression and uncertainty gates are not fully enforced.**
- Cause: Summary booleans and means substitute for all protocol requirements.
- Repair: Bind each gate to raw arithmetic, planned uncertainty and full per-family results.
- Required evidence: 100 percent supplied family regression and unsupported confidence must never pass.
- Current finding: `CONFIRMED_EXECUTABLE_AUDIT`.
- Source: audit/SOURCE_DEFECTS.json

## F09 — Independence

- [ ] **P0 — A boolean can assert independent reproduction.**
- Cause: Identity and provenance are not independently established by the decision function.
- Repair: Require attributable external reproduction artifacts and human-reviewed independence; retain same-operator label locally.
- Required evidence: An independent operator reproduces frozen results; self-assertion cannot unlock claims.
- Current finding: `CONFIRMED_SOURCE_NOT_REPAIRED`.
- Source: ../itfm_v2/src/fim_sota/verification.py:76

## F10 — Aggregation

- [ ] **P0 — Bootstrap weighting differs from equal-cell macro weighting.**
- Cause: Families contribute flattened trajectory rows.
- Repair: Define trajectory-to-seed-to-cell reductions, then dependence-preserving resampling.
- Required evidence: Unequal trajectory counts retain the predeclared estimand: audit fixture expects 0.5, current function gives 0.1.
- Current finding: `CONFIRMED_EXECUTABLE_AUDIT`.
- Source: audit/SOURCE_DEFECTS.json

## F11 — Missing failures

- [ ] **P0 — Pairing silently excludes unmatched or failed units.**
- Cause: Only finite PASS rows and shared intersections enter statistics.
- Repair: Validate complete expected observations before computing paired summaries; retain failures explicitly.
- Required evidence: Removing one candidate row must block complete-study adjudication.
- Current finding: `CONFIRMED_EXECUTABLE_AUDIT`.
- Source: audit/SOURCE_DEFECTS.json

## F12 — Trainability

- [ ] **P0 — Historical value/salience/temperature paths cannot all learn through forecast loss.**
- Cause: Detached/no_grad writes, hard threshold and scalar conversion.
- Repair: Provide explicit learning losses or mark each policy fixed; keep trainable temperatures as tensors.
- Required evidence: Finite nonzero gradients and actual parameter changes on informative delayed-recall unit cases.
- Current finding: `CONFIRMED_SOURCE_CAUSAL_ROLE_UNTESTED`.
- Source: ../../FIM/fim_experiments/systems.py:256

## F13 — Owner identity

- [ ] **P0 — Historical owner IDs are minibatch positions.**
- Cause: Stable episode identity is not carried end to end.
- Repair: Use immutable episode IDs with explicit reset and split namespaces.
- Required evidence: Batch reorder and batch-size changes preserve each episode's outputs.
- Current finding: `CONFIRMED_SOURCE_CAUSAL_ROLE_UNTESTED`.
- Source: ../../FIM/fim_experiments/systems.py:263

## F14 — Batch isolation

- [ ] **P0 — One owner without traces disables retrieval for the entire batch.**
- Cause: The eligibility test requires every row to have a candidate.
- Repair: Return per-row and per-coordinate valid masks with independent base fallback.
- Required evidence: Adding an empty unrelated episode leaves another episode's prediction unchanged.
- Current finding: `CONFIRMED_SOURCE_CAUSAL_ROLE_UNTESTED`.
- Source: ../../FIM/fim_experiments/systems.py:188

## F15 — Capacity fairness

- [ ] **P1 — Owners compete for a shared bank and may evict one another.**
- Cause: Global capacity with owner-filtered retrieval.
- Repair: Choose per-owner quotas for isolation; evaluate shared capacity separately and account all bytes.
- Required evidence: Owner isolation, eviction traces, fixed total and per-owner byte comparisons.
- Current finding: `CONFIRMED_SOURCE_CAUSAL_ROLE_UNTESTED`.
- Source: ../../FIM/fim_experiments/systems.py:144

## F16 — Spatial information

- [ ] **P1 — Historical compression discards location and amplitude information.**
- Cause: Global pooling, normalized values and uniformly broadcast feedback.
- Repair: Preserve masked spatial residuals or a reconstructable basis; normalize keys separately.
- Required evidence: Known-source reconstruction, amplitude scaling and localized intervention tests.
- Current finding: `SOURCE_DIAGNOSIS_REQUIRES_RUNTIME`.
- Source: ../../FIM/fim_experiments/models.py

## F17 — Mechanism description

- [ ] **P1 — V3 tensor arm is a fixed weighted retrieval estimator.**
- Cause: Raw cue distances, top-k softmax, FIFO and scalar calibration.
- Repair: Name it accurately and retain it as a reference.
- Required evidence: Any more complex candidate must beat matching kNN/kernel/local-linear controls.
- Current finding: `CONFIRMED_SPECIFICATION`.
- Source: ../itfm_v3/models.py

## F18 — Information advantage

- [ ] **P1 — Short-context baselines may lose because they receive less history.**
- Cause: Memory exposes past observations outside their window.
- Repair: Match legally available information and include genuinely longer-history controls.
- Required evidence: Gain survives context and information matching.
- Current finding: `OPEN_SCIENTIFIC_QUESTION`.
- Source: ../../UPGRADE_PLAN_2026-10-14.md

## F19 — Task mismatch

- [ ] **P1 — Online revealed-target prediction is not open-loop forecasting.**
- Cause: True future observations may be appended between predictions.
- Repair: Separate assimilation, delayed feedback and open-loop tracks.
- Required evidence: Open-loop memory receives no true future values; report each track separately.
- Current finding: `CONFIRMED_SCOPE_LIMIT`.
- Source: ../itfm_v3/README.md

## F20 — Missingness bias

- [ ] **P1 — Mask normalization does not correct informative missingness.**
- Cause: Observed residuals may be selected by unobserved outcomes.
- Repair: Declare MCAR/MAR assumptions, examine support and positivity; do not claim MNAR identification without extra information.
- Required evidence: Missingness stress tests and sensitivity analysis; propensity corrections compared with established estimators.
- Current finding: `MATHEMATICAL_LIMIT`.
- Source: MATH_AND_FALSIFIERS.md

## F21 — Drift

- [ ] **P1 — Similarity need not imply future residual agreement.**
- Cause: Stale observations or regime changes break conditional stationarity.
- Repair: Diagnose residual correlation by age and regime before adding age weights or gates.
- Required evidence: Gain under declared shifts and bounded harm; no universal guarantee.
- Current finding: `OPEN_SCIENTIFIC_QUESTION`.
- Source: https://arxiv.org/html/2606.04135v1

## F22 — Ablation interpretation

- [ ] **P1 — Erasing an additive correction automatically removes its gain.**
- Cause: Full-minus-base arithmetic is reused as evidence of information content.
- Repair: Add relevant versus irrelevant deletion, matched replacements and key/value mispairing.
- Required evidence: Distinct predictions for informative interventions with coverage reported.
- Current finding: `CONFIRMED_LOGICAL_LIMIT`.
- Source: MATH_AND_FALSIFIERS.md

## F23 — Generalization

- [ ] **P1 — New owners do not necessarily mean new dynamics.**
- Cause: V3 shares cue dictionary and mixing across partitions.
- Repair: Separate owner, dictionary, dynamics, geometry and noise shifts.
- Required evidence: Complete unseen-dynamics and dictionary results beyond in-distribution owners.
- Current finding: `CONFIRMED_SPECIFICATION`.
- Source: ../itfm_v3/data.py

## F24 — Statistical units

- [ ] **P1 — Optimization seeds, trajectories and domains are different units.**
- Cause: A single seed can jointly change task generation and optimization.
- Repair: Separate RNG streams and define crossed/nested uncertainty.
- Required evidence: Prospective power and uncertainty plan with preserved pairing and dependence.
- Current finding: `OPEN_METHODOLOGY`.
- Source: ../../UPGRADE_PLAN_2026-10-14.md

## F25 — Small sample significance

- [ ] **P1 — Five paired signs cannot yield two-sided exact p below 0.0625.**
- Cause: The smallest two-sided probability is 2/2^5.
- Repair: Choose adequate independent units prospectively; report effect sizes and uncertainty.
- Required evidence: Declared sample-size justification; no seed additions selected for significance.
- Current finding: `MATHEMATICAL_BOUNDARY`.
- Source: MATH_AND_FALSIFIERS.md

## F26 — Baseline validity

- [ ] **P1 — Historical SelectiveSSMSystem does not establish an official temporal SSM reproduction.**
- Cause: The inspected implementation is a feedforward convolutional stack.
- Repair: Verify state and use the admitted upstream recipe and longer-history controls.
- Required evidence: History perturbation and independent upstream validation reproduction.
- Current finding: `SOURCE_DIAGNOSIS_REQUIRES_RUNTIME`.
- Source: ../../FIM/fim_experiments/systems.py:320

## F27 — Runtime feasibility

- [ ] **P1 — Small runtime tests did not establish full-campaign memory fit.**
- Cause: V3 stopped mid-cell under global host pressure.
- Repair: Full-shape load probe before freeze; isolate candidate workers and stream checkpoints when equivalent.
- Required evidence: Measured peak memory/time/output safety margin; retain resource failures.
- Current finding: `CONFIRMED_RETAINED_EXECUTION`.
- Source: ../../audits/seed307-diagnosis-20261014/REPORT.md

## F28 — Consumed budget

- [ ] **P0 — Frozen v3 cannot be silently restarted.**
- Cause: One-campaign marker was consumed before partial execution.
- Repair: Keep v3 intact; any new campaign needs new source, hypothesis, budget and failure policy.
- Required evidence: Consumed-marker preservation and no duplicate execution.
- Current finding: `CONFIRMED_RETAINED_EXECUTION`.
- Source: ../itfm_v3/runs/CAMPAIGN_CONSUMED.json

## F29 — Recovery

- [ ] **P1 — Continuation equivalence is not demonstrated.**
- Cause: Optimizer, RNG, sampler, memory and budget state may diverge.
- Repair: Specify full checkpoint state and test interruption against uninterrupted execution on disposable development data.
- Required evidence: Matching predictions and state within frozen tolerances without extra training opportunities.
- Current finding: `OPEN_ENGINEERING`.
- Source: ../../UPGRADE_PLAN_2026-10-14.md

## F30 — Public evidence admission

- [ ] **P0 — Pinned source/data terms and complete data manifests are not admitted.**
- Cause: Two upstream pinned trees have no named license file; datasets lack admitted hashes.
- Repair: Obtain applicable source/data terms and official split manifests before integration.
- Required evidence: Authoritative permission/license receipts plus checksums; public visibility alone is insufficient.
- Current finding: `FRESH_FILENAME_SCAN_NOT_LEGAL_CONCLUSION`.
- Source: audit/UPSTREAM_LICENSE_SCAN.json

## F31 — Provenance

- [ ] **P1 — Current historical source is not established as identical to every frozen experiment.**
- Cause: Relocation, deletions and untracked successor files coexist.
- Repair: Keep exact source/config/environment hashes per result and preserve old trees.
- Required evidence: Every historical attribution uses its executed source, not a similar current path.
- Current finding: `CONFIRMED_REPOSITORY_STATE`.
- Source: ../../audits/status-20261015-01/AUDIT.json

## F32 — Optimization

- [ ] **P1 — Hard-clamped dynamics parameters can leave the active gradient region.**
- Cause: Clamping after unconstrained updates can zero gradients.
- Repair: Compare smooth constraints or projected optimization in a bounded successor only.
- Required evidence: Measured gradients, activity and held-out contribution.
- Current finding: `HYPOTHESIS_NOT_ESTABLISHED_CAUSE`.
- Source: ../../FIM/fim_experiments/models.py

## F33 — Compute matching

- [ ] **P1 — Equal updates or parameter counts do not imply equal resources.**
- Cause: Retrieval, adaptation, indexing, storage and tuning cost extra.
- Repair: Report full training/tuning/inference costs and Pareto comparisons.
- Required evidence: Accuracy-cost advantage survives complete accounting.
- Current finding: `OPEN_METHODOLOGY`.
- Source: ../../UPGRADE_PLAN_2026-10-14.md

## F34 — Release

- [ ] **P0 — Verified source and tests do not establish publication completion.**
- Cause: Independent review, PDF inspection and immutable release requirements remain open.
- Repair: Tie each paper claim to raw artifacts and complete separate release gates.
- Required evidence: Reviewed manuscript, reproducible figures, verified package and explicit submission authority.
- Current finding: `OPEN_RELEASE`.
- Source: ../../PUBLICATION_READINESS_CHECKLIST_2026-10-11.md

## F35 — Oracle interpretation

- [ ] **P1 — A privileged approximate oracle is not necessarily an information-theoretic optimum.**
- Cause: A noisy most-recent observation can lose to averaging.
- Repair: Distinguish diagnostic upper capability from a proven Bayes bound.
- Required evidence: Analytical assumptions or a documented proxy; failed proxy does not prove all history useless.
- Current finding: `MATHEMATICAL_LIMIT`.
- Source: MATH_AND_FALSIFIERS.md

