# ITFM-R model card

## Status

The independent ITFM-R source stack is statically implemented; its new runtime
tests, official integrations, public-data training, and evaluation have not
executed.
The only positive evidence is the separately frozen synthetic `fim_v2`
mechanism study. No public advantage, SOTA, natural-system, publication, or
independent-reproduction claim is available.

## Intended use

Research on whether strictly past, same-trajectory, already-observable forecast
residual tensors can safely improve partial-observation and non-Markovian PDE
forecasters. ITFM-R is not intended for safety-critical weather, epidemiology,
engineering control, or operational forecasting.

## Architecture

A registered backbone returns its ordinary prediction and a latent context key.
The key is conditioned on mask geometry, lead time, and optional physical
parameters. A bounded tensor memory retrieves old observation-space residuals.
A validation-only per-channel calibration and monotone similarity/uncertainty
gate apply the correction. Invalid retrieval is exactly the backbone output.

## Required evidence before release

- admitted sources, licenses, datasets, checksums, and official split manifests;
- exact upstream LANO and HS-FNO reproduction receipts;
- validation-only oracle-gap triage;
- complete staged gates and every declared baseline/control;
- frozen multiseed public evaluation with no dropped failures;
- independent clean-environment and separate-operator reproduction;
- verified artifact, privacy, security, and license reports.

## Known limitations

Current LANO, HS-FNO, and external-comparison entries are fail-closed wrappers,
not paper-faithful reproduced models. Approximate retrieval must be checked
against exact retrieval in each admitted environment. Energy telemetry remains
unavailable until a hardware meter is admitted.
