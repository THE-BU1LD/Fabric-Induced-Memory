# Remote reproduction outcome — 14 October 2026

V2 is fully reproduced under its original strict checks. V3 runtime tests passed, but its campaign is incomplete and its single-campaign allowance is consumed.

## Seed 307 diagnosis and correction

The previous runner used macOS 15.7.9 while the historical environment was macOS 26.5.2. Errors occurred already in the seed-307 calibration rows before retrieval. The runner now uses standard macOS 26 and an added OS-family preflight. On macOS 26.6.2, the original v2 replay passed all 66/66 raw checks, including all seed-307 historical comparisons, at the unchanged 1e-10 thresholds. All 10 tests, three seeds and 96 internal checks completed. This resolves the observed discrepancy through environment alignment; the exact low-level math-library cause was not isolated. No frozen numerical code or results were edited. The prior failed run remains retained.

## V3 actual execution

All 12 numerical tests passed. The campaign passed initial admission, began clean/seed-401, and retained base, finite-history (64/128), and GRU (64/128) training/checkpoints. Before the next candidate, the original resource check rejected 440.88/421.45 MiB, below 512 MiB. Zero of 30 complete cell results exist; no campaign summary or full raw verification exists. This is an incomplete resource failure, not a scientific negative result.

The original consumed marker and all partial outputs were copied into local v3 runs. Repeating this campaign would violate the frozen max_campaigns=1 guard. Additional compute cannot finish this frozen version through an unauthorized restart; a separately authorized prospective version would be necessary.

## Preservation and cost

All 167 collected files, raw v2 tensor hashes, 440 predecessor files and all 12 v3 frozen source files verified unchanged/intact. All prior attempts remain retained. Outputs are stored locally and in an unpublished draft release. Only the standard free public GitHub runner was used; no paid compute or Actions artifact storage was used. Main was not changed.

Run: https://github.com/THE-BU1LD/Fabric-Induced-Memory/actions/runs/38583855204
Commit: e1555619c8992f4faf46fe442d3751623fd57852
Source manifest: 4ac81e7c97e13a071c2fd7b565a799dc4827769c9f6890236bcf0f1d01cbb436

Same-operator remote execution is not independent external replication.
