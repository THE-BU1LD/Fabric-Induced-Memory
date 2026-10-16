# Observation pipeline repair v1

This is an engineering correction authorized by the renewed 15 October request
to improve ITFM. It does not implement a new research mechanism, reopen v3, or
advance the scientific contribution gate. Frozen predecessors and the completed
evidence_integrity_v1 package remain byte-identical.

Budget: three repair rounds, one process, 30 seconds per standard-library test
invocation, no model import without resource admission, no scientific trials,
no paid compute. Tensor integration is source-only until actual tensor tests
pass on an admitted host. Fixtures are software tests, never experiment results.

## Forecast and observation contract

The deployment API has no scorer-target argument. A forecast registers immutable
owner, ID, origin, coordinate target times, saved base prediction, original key,
and a frozen model/normalizer identity. Recomputing keys from later model state
is forbidden. Predictions at a timestamp precede reveals at that timestamp;
time never moves backward. Multiple prediction batches may precede the first
reveal. No prediction is allowed after any reveal at the same timestamp.

An observation event contains only revealed coordinate indices, finite values,
their availability times, and actual arrival time. Each coordinate must mature
at or after its declared target time and arrive no earlier than availability.
The event must be applied at its arrival time. Late/out-of-order target reveals
are supported through monotonic arrival times. Owners, forecasts, events and
coordinates are checked; duplicates and contradictory updates are refused.
The environment owns the authenticity of arrival/availability declarations;
this library cannot establish that an external source reported truthful times.

The existing tensor bank cannot represent missing residual coordinates.
Therefore partial observations are staged, not zero-imputed into that bank.
A residual is released exactly once, only when every coordinate is observed.
Its availability is the final actual arrival time, not an earlier nominal
target time. Stored residuals are observed values minus the original saved base
prediction. This conservative bridge sacrifices partial-vector retrieval until
a separately specified masked-value bank exists; it makes no accuracy claim.

Registration and reveal validation are transactional. Failed inputs leave the
ledger unchanged. Fixed forecast, event, and scalar limits bound accumulation;
exhaustion fails explicitly rather than silently dropping evidence. There is no
automatic resume; a poisoned tensor bridge must be discarded with its outputs.

## Tensor bridge

The new bridge builds the historical backbone batch with a zero placeholder
target of explicitly supplied output shape. Scorer truth is never accepted.
The backbone and adapter must be in evaluation mode. Parameters and buffers are
hashed across calls to reject mutation; lazy modules must first be initialized
on admitted non-protected data. The supplied source identity still requires a
real external source freeze. The bridge uses the historical adapter and bank
without editing them. One bank belongs to one source identity and split;
prepopulated banks are refused. A bank-write exception permanently poisons the
bridge, because the historical bank does not promise transactional writes.

## Prediction export

A recorder is constructed from exact externally frozen plan bytes and their
separately supplied SHA-256. It captures candidate and baseline predictions
before a sample's scorer truth can be supplied. It never derives the expected
inventory from successful outputs. Scorer truth remains exclusively in this
recorder, never in the observation ledger. Missing, extra, duplicate, nonfinite,
failed or mismatched rows cannot produce an aggregate pass.

The recorder calls the unchanged evidence_integrity_v1 validator on its actual
raw export. A single atomic non-overwriting bundle holds input evidence, plan
identity and the verdict; failures retain partial evidence and an explicit
refusal. Source/data claims in a plan are identity declarations, not an
independent authentication of provenance. Numerical gates never authorize a
scientific claim. Inputs/exports are bounded to the reference 8 MiB contract.

Acceptance: hand-calculated residuals, hidden-value noninterference at the
observation projection boundary, delayed and out-of-order events, same-time
ordering, owner/ID isolation, overflow/type/budget rejection, atomic rejection,
and genuine recorder-to-existing-validator integration with both positive and
negative software controls. Tensor tests must exercise the actual torch bank
and adapter; mocks do not count as tensor validation.
