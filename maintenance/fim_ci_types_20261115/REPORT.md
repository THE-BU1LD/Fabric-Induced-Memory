# FIM successor strict typing repair — 15 November 2026

The existing ITFM-R workflow stopped at 133 strict Mypy diagnostics on PR #51 head `092bb378446add05f8fd5ddb685dc1496d83f5e3`. A local copy matched every source blob to that head and reproduced all 133 diagnostics using the pinned type checker and numerical library versions. The candidate has zero diagnostics in 17 source files with the existing strict configuration.

## Changes

- Add explicit engineering-control function, callback, manifest, receipt and source-identity types, while retaining the lightweight import path.
- Declare registered Torch buffers and the existing checkpoint marker; retain module calls and hooks. Narrow library return/call boundaries with precise types rather than broad ignores.
- Bind the dataclass helper to actual dataclass instances and use a typed archival record and retrieval-validity protocol.
- Separate same-scope variables that have different roles (patch order, resource admission, signal restoration and failure receipts).
- Explicitly request the legacy NumPy RNG state, rejecting unsupported non-tuple state before checkpoint serialization. Reject macOS resource output without a page-size declaration through the existing blocking admission result. Two focused negative controls cover these error paths.

No type-check flag, workflow, dependency lock, protocol, evaluation threshold, existing test assertion, scientific result or historical evidence is changed. Casts preserve runtime values. Existing prediction, loss, accumulation, memory, budget and checkpoint arithmetic remains the same; the worst-group key lookup uses dictionary indexing for keys already known to exist. The standard-library engineering entry point still imports neither Torch nor NumPy.

## Verification

- Strict Mypy: 0 issues / 17 source files, compared with 133 on the byte-verified baseline.
- Ruff lint: pass. Format: 52 files pass.
- Full successor engineering unit suite: 118 passed, 4 explicitly gated CUDA cases skipped, exit code 0.
- Frozen protocol audit: 11/11 checks pass.

The local Python is 3.12.14. Torch 2.11.0+cpu, NumPy 2.4.4, SciPy 1.17.1, Mypy 2.4.0, Ruff 0.17.0 and Pytest 9.1.1 match the pinned project versions. The temporary runtime also exposes system packages; it is not a clean reproduction of the entire hosted lock. Hosted source checks, including the dependency license gate, remain a separate check after publishing this candidate.

No GPU or scientific campaign ran. The protected 40-cell matrix remains held, and no scientific claim is upgraded by this engineering repair. Source hashes change normally, so old identity-bound artifacts must not be silently reclassified as produced by this candidate.
