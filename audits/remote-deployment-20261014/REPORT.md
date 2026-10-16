# Remote v2/v3 execution — 14 October 2026

**V2 executed; strict historical raw comparison failed. V3 withheld.**

Published dedicated branch `codex/itfm-remote-20261014` in THE-BU1LD/Fabric-Induced-Memory. Current execution commit: `b7a97d1fcaa5a61d7511781f670d7324b8616932`. Main was not changed. Run: https://github.com/THE-BU1LD/Fabric-Induced-Memory/actions/runs/38578599860 .

## What actually completed

- Standard macOS ARM64 runner; exact Python 3.11.15 and direct pinned requirements installed in a fresh venv. Resolved transitive environment retained. Python came from the checksum-bound Astral build, distinct from local Homebrew.
- Unchanged admission passed at about 2.5 GiB free-plus-speculative, stable pageouts and zero swap use.
- V2 original 10 runtime tests, frozen static audit, all three seeds, and 96 internal result checks passed.
- All 27 seed/control raw tensor captures and the instrumented evaluator were retained. The separate exact-key reference passed all three seeds.
- Historical raw comparison passed 48/66 predicates. All 18 failed predicates belong to seed 307: nine raw-row and nine summary comparisons.
- Seeds 101 and 211 match historical scalar records exactly. Seed 307 has small differences: base query NMSE shifts from 0.48768470861860486 to 0.487684704790684 (about -3.83e-9); tensor query NMSE remains zero; sign-flip NMSE shifts by about -7.66e-9. Maximum observed per-row squared-error difference is 0.00018310546875. The cause is not established by this scalar inspection. All three original development decisions are unchanged, but the fixed 1e-10 raw-comparison checks still fail.
- V3 numerical tests, campaign and raw verification did not start; the one-campaign budget is unconsumed.

## Retained failures and repairs

1. Run 38577058691: GitHub setup-python lacked exact 3.11.15 ARM64; no numerical stage. Draft assets and logs downloaded/hash-verified.
2. Run 38577627585: fresh runner memory guard failed around 300 MiB before runtime download. Draft assets and logs downloaded/hash-verified.
3. Run 38577967203: disposable-runner cache purge enabled admission; 10 v2 tests passed, then static audit found the replay copy omitted its required README. No experiment seed ran. Draft assets and logs downloaded/hash-verified.
4. Run 38578599860: a narrowly scoped copy helper added the retained v2 README to the fresh isolated workspace; all original 20 executed source files, original wrapper/worker and guards stayed intact. V2 numerical execution then completed, with the strict raw discrepancy described above. This outcome-bearing run is NOT adjudicated for automatic retry.

The first three pre-experiment failures were explicitly adjudicated with retained receipts before continuing. No scientific outcome was discarded or retuned. Fifteen local orchestration tests pass, including exact copy/audit validation and rejection of outcome-bearing retries.

## Preservation and cost

All 440 protected original files remain unchanged. All 28 raw-file hashes and the complete downloaded 113-file collection match their receipts. All four attempts remain both locally and in unpublished draft-release assets. No standard Actions artifact upload, cache or LFS was used; standard public macOS runner time is free and release assets have no total storage/bandwidth quota under the referenced GitHub documentation. No paid resources were launched.

The draft is an execution archive, not a public scientific release or independent reproduction. The next step is to diagnose the preserved seed-307 discrepancy. Relaxing tolerances or treating the raw verifier as passed would violate the requested unchanged criteria. V3 remains dependent on that unresolved gate.

See SUMMARY.json, SCALAR_DISCREPANCY_ANALYSIS.json, RUN_*.log, RUN_*_JOBS.json, and download-*/extracted/remote-artifact. Frozen inputs and earlier failed attempts are retained.
