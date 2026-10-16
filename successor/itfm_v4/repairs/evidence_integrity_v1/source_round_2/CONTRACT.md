# Evidence integrity repair v1

This is a correction to research evidence tooling, not a new forecasting model or a novelty claim. It addresses the already demonstrated V01–V06 failure modes in an isolated package. V07 (observation-mask use by the tensor writer) is outside this package. Frozen source and results must remain unchanged. Research checkpoints do not advance from these engineering checks.

The renewed user request authorizes fixing documented flaws while retaining prior outcomes. This repair follows the invalid-evidence correction branch; it does not implement the unapproved v4 scientific mechanism. Maximum three repair rounds, one process, 30 seconds per test invocation, zero model imports/runs, zero paid cost. Fresh host admission failed at 81.48/109.47 MiB with rising pageouts. Only small standard-library integrity tests are permitted here.

## Input contract

A separately frozen plan declares a schema version, study ID, source and data SHA-256 identities, distinct candidate/baseline method names, unique cells and their family identities, unique integer seeds, and every expected `(cell, seed, trajectory, origin)` sample. Every cell/seed block must occur in the plan. The caller provides the expected plan SHA-256 separately to the CLI. A matching digest binds the input to those bytes; it does not authenticate the author, establish when the plan was frozen, or verify actual data provenance.

Evidence declares matching source/data identities and contains exactly one candidate and one baseline row for every planned sample. Both rows must contain the same target vector. IDs, vector dimensions, finite real values and successful statuses are required. Repeated, absent, unexpected, failed or malformed rows invalidate aggregate adjudication. No intersection, filtering, imputation or post-hoc exclusion is permitted. Failed input remains identified by its path and SHA-256 in the refusal receipt. Files are never modified by the validator.

JSON object keys must be unique; JSON NaN and infinity are forbidden. Booleans and numeric strings are not numbers. Non-finite intermediate arithmetic is rejected. Each input file is capped at 8 MiB for this reference CLI; larger studies require a separately profiled streaming implementation. This is not a full-size resource profile.

## Metric and reduction specification

For sample vectors `p,y`, compute `MSE = sum((p-y)^2)/D` and `L2 = sqrt(sum((p-y)^2)/max(sum(y^2), epsilon))`. `epsilon` is a positive finite energy floor in the frozen plan. Both scalar audit and evidence reduction call this one definition. This is full-vector scoring, not observed-mask-only scoring.

For each method and metric, average origins within trajectory; trajectories within cell/seed; seeds within cell; then cells equally. Unequal trajectory counts cannot change cell weight. This prospective reduction does not relabel or recompute historical outcomes. Optional supplied metric claims must exactly cover both methods and both metric names and agree with recomputation to fixed absolute tolerance `1e-10`, with all values finite.

For each cell, relative gain is `(baseline_L2-candidate_L2)/baseline_L2`. When both are exactly zero the declared convention is zero gain; a zero baseline with positive candidate error makes the relative endpoint undefined and rejects adjudication. Macro gain averages cell gains equally. Family gain averages its cell gains equally. Seed win fractions count strict error improvements; ties do not count as wins.

## Decision boundary

Every declared numerical gate is evaluated: minimum macro gain, minimum winning cells, maximum cell regression, minimum seed win fraction, minimum winning families, maximum family regression, and an optional maximum macro candidate error. Decisions derive from raw arrays, never supplied win flags or gain summaries.

A numerical-gate pass is a descriptive calculation. The tool **never authorizes a scientific, SOTA, publication, or independent-reproduction claim**. It accepts no independence boolean. Confidence intervals and multiplicity cannot be used for confirmation until a new scientific protocol defines the relevant population, exchangeability assumptions, hypotheses and precision design. This limitation is explicit rather than silently skipping those gates.

The exploratory bootstrap resamples entire families, retaining all cells/seeds/trajectories in a sampled family. Each replicate recomputes the equal-cell macro as the sum of sampled cell gains divided by the number of sampled cells (with multiplicities). It does not flatten individual trajectories. With fewer than two families it returns no interval. With two or more, its percentile interval remains exploratory, never confirmatory; small cluster counts may be severely underpowered. Deterministic RNG seed, repetitions and confidence belong to the plan. Point-estimate equality is exact by construction; interval coverage is not claimed or proved.

## Proof obligations and acceptance

1. Exact-set equality and duplicate rejection imply one record per method/planned sample before reduction. A missing or failed sample therefore cannot disappear into a valid summary.
2. Shared target-vector equality prevents candidate and baseline from silently being scored against different truths.
3. Cell reduction and every bootstrap draw operate on cell statistics; duplicating a trajectory count in one cell cannot increase that cell's macro weight.
4. All accepted inputs/intermediates are finite before arithmetic comparison. NaN's comparison semantics cannot turn a mismatch into a pass.
5. Family regressions are computed and checked explicitly; passing average gain cannot conceal a prohibited individual family loss.
6. No API path returns scientific claim authorization, regardless of descriptive results. External independence remains a separate real-world evidence requirement.

Executable acceptance includes the six prior adversarial cases, benign controls, wrong source/data/plan hashes, duplicates/missing/extra IDs, failed statuses, target mismatch, zero denominators, overflow, malformed JSON, strict types, row-order invariance, equal-cell weighting, family resampling, CLI success/refusal receipts, and no-overwrite behavior. Tests use explicitly synthetic software fixtures, not research data or performance results.

## Remaining integration work

This module can validate files matching this contract today. Existing runners do not emit this schema yet and are not silently redirected. Future integration requires source/data admission, a complete raw export and prospective sample inventory, the same metric implementation or tested equivalence, full-size profiling, and the unresolved scientific protocol. Tensor observation visibility and model mechanism repairs remain open.
