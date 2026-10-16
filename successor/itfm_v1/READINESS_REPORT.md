# ITFM v1 readiness report

Date: 2026-10-11

## Verdict

The successor is **implemented, runtime-tested, and data-materialized, but not
run-ready on the current host state**. The immutable 512 MiB admission boundary
paused the smoke experiment before cell 1/9. No successor fit or performance
result exists.

## Completed

- Canonical project state, predecessor boundary, falsifiers, claim ledger, and
  development-only protocol.
- Transport-aligned tensor residual memory with strict owner and event-time
  eligibility, detached FIFO writes, local top-k retrieval, bounded gating, and
  explicit interventions.
- Shared bounded normalized cross-correlation transport and age-aware co-moving
  periodic coordinates for 1D, 2D, and 3D fields, with unique-radius validation
  and identity-first tie handling.
- Transport-off, no-memory, vector-broadcast, sham-transport,
  finite-history-CNN, ConvGRU, stable diagonal-SSM, and zero-parameter periodic
  transport controls.
- Event-pure training, deterministic seeds, resource checks before each cell,
  hash-bound checkpoints/predictions, strict resume validation, robustness/OOD
  evaluation, intervention analysis, figures, tables, and artifact manifests.
- Deterministic synthetic-data materialization and official PDEBench
  registry/license staging, plus capped exact-shard downloading and lazy
  schema-declared HDF5 loading.

## Evidence

- Runtime tests: 37/37 pass under Python 3.11 and Torch 2.12.1.
- Formatting: pass.
- Ruff lint: pass.
- Python bytecode compilation: pass.
- State and protocol JSON parsing: pass.
- Static contract audit: 15/15 checks pass in `results/static_audit.json`.
- PDEBench registry SHA-256:
  `469f43609413c475740a54f1246a09be16bdffcc8b9c064f307fcf02f12a8d6a`.
- PDEBench software-license SHA-256:
  `53015b22f7d9eccb67ab5aeb991b1b705d367ee70efa2311d34e72d2a36b9cc9`.
- Synthetic archive: 252,996 bytes, SHA-256
  `c76c1d747cb6ab2cf6a142150b7e1753b01cdd2ae10173673260f3d99b0be54b`.

These are static and provenance checks, not runtime or scientific evidence.

## Failures and blocks

- The smoke subprocess paused at 473,333,760 free-plus-speculative bytes; the
  final host check reported 277,643,264 bytes versus a required 536,870,912,
  and cumulative pageouts increased to 605,390.
- A Python 3.14 run failed before collection because pytest was absent. A later
  Python 3.11 run found one malformed 3D test fixture (missing channel axis);
  that invalid run is retained and the corrected suite passes 37/37.
- The previous low-pressure pipeline invocation also stopped before tests and
  retained a `blocked_resource` receipt.
- No model fit, smoke result, development result, protected evaluation,
  independent reproduction, or release exists.
- The outer checkout remains a dirty relocation. It was not reset, staged, or
  rewritten.

## Next execution

After the host is stably above the admission threshold:

```bash
cd /Volumes/PRO-BLADE/GitHubRepos/Personal-Research/Fabric-Induced-Memory/successor/itfm_v1
PYTHONPATH=src python3.11 scripts/run_pipeline.py
```

Only if that receipt is complete and failure-free:

```bash
PYTHONPATH=src python3.11 scripts/run_pipeline.py --full-development
```

The second command executes 27 declared paired cells. It is development work,
not confirmatory evidence. Protected public evaluation remains uncreated.
