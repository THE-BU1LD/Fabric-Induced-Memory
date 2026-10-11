# Retention analysis v2 — engineering contract

This separate 10 October 2026 session is authorized by the user's request to
find and execute remaining work. It preserves all closed predecessor budgets.
Parent: FIM reviewed engineering PR38 at
`92bcc3eb5322d967684673ca61f22a8bbebdc16b`. Independent implicit-diffusion PR39
remains separate and is not a dependency of this change.

## Defect and intended correction

The retained retention functions cast float64 timestamps to float32. Distinct
observations at 1e9, 1e9+1 and 1e9+2 become indistinguishable, yielding zero area
and zero exponential rate for responses 1, 0.5 and 0.25. The exact trapezoidal
area is 1.125 and the decay rate is log(2) per supplied time unit.

The versioned module preserves input precision, computes analysis in float64,
requires finite one-dimensional paired observations with strictly increasing
times, and rejects undefined logarithmic fits instead of clipping observations.
Centered, scaled time coordinates avoid artificial variance floors. A fitted
negative decay rate remains a valid growth result. Trapezoidal area is signed
and retains the user's time/response units. Normalization accepts a signed
nonzero initial response and preserves later signs/zeros. It returns float64.

The API accepts real float32/float64 tensors. Precision already lost in an input
float32 timestamp cannot be recovered; supply float64 or a relative time axis.
Unrepresentable float64 results fail explicitly. This is ordinary descriptive
analysis, not statistical uncertainty estimation or independent-run inference.

## Prospective verification and scientific boundaries

One artificial CPU regression collection, with at most one repair rerun if a
concrete failure occurs, and at most 120 seconds per command. Compare analytical
exponentials, power laws, hand-calculated trapezoids, tiny/large time units,
amplitude rescaling, signed responses, finite arithmetic and invalid inputs.
Retain actual failures and raw command receipts. No protected runs, retained
scientific outcomes, memory-benefit analysis, training campaign or paid compute.

Import explicitly from `fim.analysis.retention_v2`. Existing retention functions,
exports, callers, frozen protocols, benchmark/recall metric, results and workflow
conditions stay unchanged. A future study must adopt the new endpoint before
outcome access. Software correctness does not complete FIM's scientific matrix
or replace its adverse stored comparisons.

## Source-review correction

The initial 54-case collection passed. Independent source review found that a
smallest-subnormal response could be rounded to zero while forming the midpoint,
even when multiplication by the time interval should yield a representable area.
The revised trapezoids use bounded mantissas and `math.ldexp`, then `math.fsum`
for the final scalar area. Four independent 100-digit Decimal-oracle fixtures
cover equal, unequal, opposite and zero/subnormal endpoints. The scalar integral
performs this arithmetic on the host; GPU execution/performance is not claimed.
If an individual area or the accumulation exceeds supported finite arithmetic,
the function rejects it rather than attempting an arbitrary-precision sum.
Pre-review source, tests and the original 54-case receipt remain retained.
