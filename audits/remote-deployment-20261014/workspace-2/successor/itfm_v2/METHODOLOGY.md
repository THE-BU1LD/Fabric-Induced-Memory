# ITFM v2 Development Methodology

## Independent units and splits

Families are split before anchors, trajectories, events, or windows are
constructed. A family template and its paired regimes occur in exactly one of
train, validation, or test. Metrics aggregate first within owner, then across
paired families. Windows are never treated as independent replicates.

## Information structure

At each calibration or query event, both regimes receive the same complete
visible context. Their targets differ by opposite, family-specific, zero-mean
spatial innovations. Calibration targets are observed after prediction and can
be committed to memory. Query events occur after a washout interval longer than
the visible context and minimum admissible memory age.

Nuisance transitions between calibration and query events have distinct keys
and zero innovation. They prevent a last-value state from acting as a keyed
associative memory.

## Frozen base and memory calibration

The development base is the final visible frame. Under balanced paired regimes,
it is the conditional mean for the ambiguous anchor transition. This removes
base-training variance and makes every stored residual stationary.

The tensor-memory correction is computed online from past completed
transitions. A single coefficient in `[0,1]` is fitted in closed form on the
validation split and then frozen for test. Test targets never affect calibration.

## Declared controls

1. `no_memory`: frozen base only.
2. `tensor_memory`: exact same-owner, past-only, minimum-age tensor retrieval.
3. `erase`: primary retrieval with the value set to zero.
4. `sign_flip`: primary retrieval with its value sign reversed.
5. `vector_broadcast`: spatial mean of the retrieved tensor broadcast globally.
6. `recent_only`: retrieval restricted to entries no older than the visible
   context window.
7. `last_value`: most recently completed residual without key matching.
8. `paired_swap`: primary retrieved values exchanged between oracle-matched
   paired regimes only for causal evaluation.
9. `oracle_anchor`: exact past entry selected using the development-only anchor
   label; diagnostic ceiling, never a learned result.

## Pre-fit identifiability gate

Before any fitted model is allowed, the generated corpus must show:

- zero paired full-context difference at every query;
- nonzero paired future difference;
- zero family overlap across splits;
- calibration-to-query age strictly greater than visible context length;
- frozen-base query MSE above zero;
- oracle past-memory query MSE at least 20% lower than frozen base;
- analytical last-frame/zero-shift prediction equivalent to the frozen base.

## Primary endpoint and decision rule

Primary endpoint: query-only normalized MSE, normalized by training-target
variance and aggregated within owner then paired family.

V2 passes bounded development only if all are true on the untouched test split:

1. tensor memory improves query NMSE by at least 5% over every non-oracle
   information control;
2. tensor memory recovers at least 50% of the oracle-memory gap;
3. erasure removes at least 80% of the tensor-memory gain;
4. paired swaps have donor-direction score above 0.9;
5. sign-flipped and vector-broadcast values do not reproduce the gain;
6. non-query MSE increases by no more than 1% of training-target variance;
7. retrieval precision for the matching anchor is at least 0.99;
8. every integrity and artifact check passes.

The thresholds are frozen before numerical execution. A failure is retained; no
seed, family, event, or control may be silently removed.

## Confirmation boundary

The synthetic task is only a mechanism test. Confirmation requires a new frozen
protocol on a public low-resolution or noisy dynamical-system dataset where
unresolved state creates a measurable oracle memory gap. Strong recurrent,
state-space, neural-operator, retrieval, and analytical controls remain required.

