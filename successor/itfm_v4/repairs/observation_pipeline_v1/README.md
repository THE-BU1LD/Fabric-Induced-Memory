# Observation pipeline repair v1

This repair provides an explicit observation boundary and connects prospective
prediction exports to the retained evidence validator. It preserves every
historical runner and result. This is engineering work; no accuracy improvement,
new scientific mechanism or contribution-gate pass is claimed.

## Implemented behavior

- `observations.py`: immutable forecast snapshots, sparse observed events,
  coordinate maturity and actual-arrival checks, stable owners, duplicate
  rejection, transactional validation, and bounded storage. Partial reveals
  accumulate until a complete residual can be written safely to the existing
  full-value memory bank. Hidden coordinates are never inspected by the
  environment's mask projection helper.
- `recorder.py`: captures predictions before scorer truth, checks a separately
  supplied frozen plan digest and complete expected inventory, then calls the
  exact retained `evidence_integrity_v1` validator. Atomic bundles preserve raw
  outputs, hashes, numerical verdicts, and failed/incomplete rows. Bad scores
  can have valid evidence; failed numerical gates remain failed.
- `tensor_bridge.py`: source implementation of an evaluation-only wrapper for
  the actual public `ITFMRAdapter`, `TensorMemoryBank`, and backbone API. The
  predictor accepts `ForecastInputs` with no target field. The historical
  batch receives a zero shape placeholder. The bridge retains original keys
  and base predictions, refuses model/normalizer/configuration changes, binds
  one fresh bank to a split/source, and automatically exports actual model
  outputs when a recorder is attached. Bank-write failure poisons the bridge
  instead of permitting an unsafe retry.

## Verification status

The standard-library regression suite passed 36 tests in rounds 1 and 2;
the final source receipt is `TEST_ROUND_3.json`. These are adversarial software
fixtures, not research observations. No tensor/model package is imported by
that suite. `FINAL_VERIFICATION.json` records final checks and preservation.

Six actual tensor integration tests are present in `test_tensor.py`. They are
**not executed**: local memory/pageout/swap admission failed before tensor
imports. The initial sandbox-limited refusal and subsequent complete admission
refusal are retained in `TENSOR_ADMISSION*.json`. Do not describe the tensor
writer defect as fully fixed until those tests pass and the real experiment
runner is wired and validated. Source connection is not runtime verification.

## Use in a prospectively admitted runner

Put this directory and `successor/itfm_v2/src` on the runner's Python import
path. Instantiate `PredictionRecorder` with actual pre-frozen plan bytes and
their separately recorded SHA-256. Instantiate `SafeOnlineAdapter` with an
initialized evaluation-mode backbone, adapter, fresh bank, source identity,
split, and that recorder. The backbone must implement the existing `identity()`
contract; it must disclose mutable non-tensor execution settings there.

For each timestamp, call `predict(inputs, forecast_ids=..., now=..., samples=...)`
for all forecasts before supplying any observation events. Sample owners and
origins must match the actual forecast rows. Candidate and base predictions are
exported automatically. Scorer truth goes only to `recorder.record_target()`;
the deployment loop supplies independently available sparse observations to
`bridge.observe()`. Finalize to a fresh bundle path even after a run fails;
record known errors with `record_failure()` so missing outcomes stay visible.
Do not create a prospective plan by enumerating successful outputs afterward.

Run software checks with a new receipt path:

```sh
rtk proxy python3 -B successor/itfm_v4/repairs/observation_pipeline_v1/run_tests.py --receipt /tmp/itfm-observation-tests-new.json
```

On a host with sufficient resources, the guarded tensor entry point performs
two admission samples, disk/owner/lock checks, and a mandatory pre-import
recheck before importing Torch. It refuses unknown admission information and
keeps the unchanged 512 MiB floor. Choose a new receipt path:

```sh
rtk proxy python3 -B successor/itfm_v4/repairs/observation_pipeline_v1/run_tensor_tests.py --receipt /tmp/itfm-tensor-tests-new.json
```

## Remaining limits

Partial-value retrieval is not implemented: incompletely observed forecasts
stay staged. The bridge currently assigns one forecast horizon per row; the
underlying ledger supports coordinate-specific target times. State hashing
and CPU event staging emphasize auditability and have not been profiled at
full scale. Storage refuses its declared limits; there is no resumable stream
checkpoint, automatic rollover, or silent eviction of pending evidence.
Callers must retain input observation streams for replay and close out the
recorder on exceptions. A process crash before closeout can lose buffered
predictions; the prospective inventory still prevents a complete-evidence pass.

The old public writer remains unchanged and unsafe for partial observations;
existing callers must explicitly migrate to this bridge after verification.
The new validator integration is tested at the recorder boundary; real tensor
export execution, public datasets/baselines, resource profiling, scientific
contribution, confirmation and independent reproduction remain open.
