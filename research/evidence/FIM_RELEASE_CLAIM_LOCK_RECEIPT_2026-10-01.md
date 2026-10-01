# FIM completion-release claim lock — 2026-10-01

This is a sanitized reproducibility receipt only. It does not contain learner/collaborator data, checkpoints, raw protected data, or new scientific outcomes.

## Source identity
- Canonical repository: `THE-BU1LD/Fabric-Induced-Memory`
- Base main SHA: `b6167d481df69fb0b783a06368060a0e15a75d21`
- Base tree SHA: `08012bcd8f0317b86631ee300c6e2a730d98e279`
- Existing checkpoint/config identity work inspected first: branch `repro/checkpoint-config-identity-20261001`, commit `3dfdf9484e9ece54ecec126b8a9b01984230be3b`

## October 1 release package identity
The completed research package is `FIM-spatial-recall-completion-r1-2026-10-01`. It is a research artifact package, not a GitHub Release, and no upload/submission/acceptance status is inferred.

Byte-verified Library artifacts:
- PDF: `4ab95e7b1c3ae0210c757c8e1cac5765ab048ca95b72f8fd87a5788aea8a1194` (855712 bytes)
- TeX: `d5ae109508ded66eef17640884d8a8bc1ccd00ca3feeb2e29a60ce8883044165` (59603 bytes)
- completion manifest: `baf03b399bdae51758261fe84e83a9cc13360d59f4f46f5e22730edee82081ca` (5209 bytes)
- research state: `f0587a18d53300e22235ce6ec2e03f59a6eff43ec04e347333a656c8af32e674` (10448 bytes)
- claim/evidence ledger: `2d083376d1573188a5c2fe80550afd7129267c8e73c2319419d118cb5015b57b` (3323 bytes)

## Repair
A fail-closed completion-release claim validator was implemented outside the repository and bound to the five artifacts above. It checks:
- exact artifact SHA-256 identities;
- exact release version;
- retained numerical claims;
- primary terminal dispositions;
- the development-only boundary for Study II;
- rejection of accidental promotion of the unresolved external-memory claim.

During test construction, an initially weak status check was found: searching for `INCONCLUSIVE` anywhere in the manifest could miss a promoted primary disposition if another later occurrence stayed inconclusive. The validator was tightened to structured/document-specific primary-disposition checks before final verification.

## Executed checks
- `python -m py_compile verify_fim_release_claims.py test_verify_fim_release_claims.py`
- `python test_verify_fim_release_claims.py` → 4/4 passed
- direct verification against the five actual materialized release files → `PASS`

Mutation regressions verify fail-closed behavior for byte drift, numerical-claim drift after intentional hash rebasing, and terminal-status promotion after intentional hash rebasing.

## Scientific boundary
Preserved exactly:
- original 80-update Study I endpoint: **NEGATIVE**;
- Study II: development-only optimization/checkpoint diagnostic;
- external trace-bank benefit beyond recurrence: **INCONCLUSIVE**;
- broad OOD/PDE/real-data advantage: **UNTESTED**;
- no final protected confirmatory `H_mem` result.

No training, protected evaluation, held-out tuning, confirmatory rerun, release, submission, merge, or external message was performed.

## UNRUN
Repository-wide pytest, hosted CI, GPU execution, successor protected protocol, independent-machine reproduction, and paper submission/release publication remain UNRUN.

## Persistent private artifact
The exact validator, spec, tests, raw verification receipt and run logs were stored as `FIM_release_claim_lock_2026-10-01.zip` in the authorized private Library. Package SHA-256:
`fa301ef4c3baa60bdb4df7f84f9cc54b6fd3e12fe37f9d0ba60e6867e0025ae5`.

## Next narrow action
Review this isolated branch against main, then—without opening protected outcomes—copy the validator/spec/tests into the repository and run the normal local non-outcome-bearing test suite before any human-reviewed merge.
