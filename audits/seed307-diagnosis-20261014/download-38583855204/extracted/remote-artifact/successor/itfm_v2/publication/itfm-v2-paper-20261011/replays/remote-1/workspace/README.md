# ITFM v2 — Interventional Episodic Tensor Memory

ITFM v2 is a separate successor to the completed negative FIM study and the
unexecuted ITFM v1 development protocol. It is designed to test a narrower,
identifiable claim: a past-only keyed tensor memory can reuse a previously
observed spatial innovation after the complete visible context has become
ambiguous.

The development task contains paired hidden regimes whose complete visible
contexts are identical at query events while their futures differ. Earlier
calibration transitions, outside the visible context window, reveal the missing
spatial innovation. The primary model may read only same-owner, strictly past
entries. Erasure, sign-reversal, vector-broadcast, recent-only, and paired-value
swap controls determine whether any gain is caused by the stored tensor value.

This is a positive-capable diagnostic, not a guaranteed positive result and not
natural-domain evidence. A result is accepted only when every preregistered
predictive, intervention, integrity, and artifact gate passes.

The frozen synthetic development study subsequently passed all three declared
seeds and all 96 historical local verification checks. This is a bounded mechanism
result, not a public-benchmark or state-of-the-art result. The next campaign is
defined in `protocols/sota_confirmation_v1.json` and `SOTA_STRATEGY.md`.

## Public benchmark implementation

The production-facing stack now lives in the separate `fim_sota` package. It
does not modify the frozen `fim_v2` synthetic implementation or its retained
results. The package provides:

- versioned typed run configuration and a resolved-config command;
- manifest/hash/license-bound POBench-PDE loaders and explicit masks;
- a locked-upstream HS-FNO generation bridge;
- common forecast batch, backbone output, and rollout contracts;
- a vectorized, chunked, GPU-native causal residual bank with exact reference
  retrieval, deterministic ties, serialization, checksums, telemetry, four
  eviction policies, approximate search, and causal interventions;
- key projection, mask/lead/parameter conditioning, residual normalization,
  per-channel calibration, monotone similarity gating, uncertainty gating, and
  exact backbone fallback;
- fail-closed LANO/HS-FNO wrappers and an external-baseline registry;
- budgeted training/checkpoint/resume and statistical evaluation utilities;
- immutable stage manifests, protected-test unlocks, artifact verification,
  report generation, model/data cards, and an archival checksum builder.

This is engineering infrastructure, not public-data evidence. Exact upstream
reproduction, real dataset admission, oracle-gap triage, training, protected
evaluation, and independent reproduction remain unexecuted until their frozen
gates pass.

## Synthetic paper verification

The publication revision uses the exact recovered executed source in
`publication/itfm-v2-paper-20261011/frozen_source`. Run the non-overwriting
checker with `python3 scripts/build_paper_release.py verify`; use the isolated
reproduction wrapper documented in that publication directory after resource
admission. Do not run the historical pipeline directly against frozen results.
The public stack and its environment are separate from the synthetic archive.

Public-stack commands are available after installation:

```bash
fim-sota print-config --config configs/sota/base.json
fim-sota fetch-data --registry admission/data_registry.json
fim-sota verify-data --manifest admission/datasets/navier_stokes.json \
  --root datasets/pobench/navier_stokes
fim-sota reproduce-lano --stage-spec admission/stages/reproduce-lano.json
fim-sota reproduce-hsfno --stage-spec admission/stages/reproduce-hsfno.json
fim-sota run-oracle-gap --stage-spec admission/stages/oracle-gap.json
fim-sota train-adapter --stage-spec admission/stages/train-adapter.json
fim-sota run-ablation --stage-spec admission/stages/ablation.json
fim-sota run-pobench-final --stage-spec admission/stages/pobench-final.json
fim-sota run-hsfno-final --stage-spec admission/stages/hsfno-final.json
fim-sota verify-results --manifest RUN/manifest.json --receipt RUN/verified.json
fim-sota build-report --verification-receipt RUN/verified.json --output REPORT.md
```

Stage commands are dry runs unless `--execute` is supplied. `--execute` still
fails closed on resources, identities, receipts, output existence, and protected
test authorization.

## Layout

- `PROJECT_CONTRACT.md`: claim, falsifiers, and evidence boundary.
- `METHODOLOGY.md`: dataset, controls, endpoints, and decision rule.
- `protocols/development_v2.json`: machine-readable frozen development plan.
- `src/fim_v2`: generator, memory, evaluation, audit, and runner.
- `src/fim_sota`: public-data, backbone, memory, training, evaluation, and
  orchestration APIs.
- `tests`: invariants and expected failure-mode tests.
- `results`: generated receipts and immutable run artifacts.
- `paper`: evidence-bound LaTeX manuscript, bibliography, retained-results
  table, and claim-to-artifact map.
- `admission`: current fail-closed S0 receipt and exact requirements for opening
  public execution; it intentionally contains no dummy dataset or stage specs.
- `scripts/build_paper_release.py`: fail-closed manuscript evidence verifier and
  non-overwriting arXiv source stager. It does not certify compilation, visual
  quality, public confirmation, or submission.

## Manuscript status

The revised `paper/main.tex` describes the exact synthetic implementation,
including the empty recent-only restriction, actual donor metric, full-context
keys, calibration subset, analytic bounds, and scalar-record verification
limits. All 20 executed source files are recovered and hash-verified separately
from later public-stack packaging. See the dated publication workspace for
preservation checks and open gates. The current revision is not publication-ready:
raw replay, PDF compilation/visual QA, independent reproduction/review and final
release archive/terms/destination remain open. Ryan is the sole displayed author;
optional affiliation/funding/conflict fields are omitted at the author's request.
