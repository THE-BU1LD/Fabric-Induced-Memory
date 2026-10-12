# Checkpoint returns no longer alias active training state

This engineering change starts from FIM PR45 commit
`323fc2738c08daed9dcac48a429cc4b1dd8b0e4a` and repairs only the public
`fim.utils.checkpoint.load_checkpoint` helper. The frozen experiment loader and
scientific records retain their original bytes.

The published candidate targets current PR44
`2c0d63959defe5544ee0f40b86f7de2f430d0125` and includes PR45's unchanged
cancellation repair. PR45 had developed a canonical-state conflict because PR44
added a verification-only followup. This candidate imports all eight new PR44
artifacts exactly, preserves both complete predecessor states, and unions the
source-specific maintenance records. Neither existing branch is rewritten.
See `integration.json` for identities and the preservation receipt for the
complete tree comparison. This closes that conflict in the combined proposal.

## Reproduced problem

The existing loader returned the same objects that it passed to native optimizer
and scheduler loaders. Those loaders retain some incoming tensors and containers.
Editing a returned Adam, AdamW or momentum-SGD checkpoint could therefore change
live moments without an optimizer step. Conversely, an optimizer step could
silently rewrite the supposedly saved values in the returned checkpoint.
Scheduler learning-rate lists, milestone counters and nested child state were
also shared. An `extra` metadata tensor aliased to an optimizer moment provided
another ordinary, serializable path to the same defect.

The exact parent source produced **11 failures and one passing weights-only
control** in the initial 12-case regression collection. The baseline source,
test source, raw log, JUnit output and source hashes remain in this directory.
The archived test source is the exact baseline prefix, verified against its
executed SHA-256 receipt before preservation.

## Change

After every requested component passes admission, make a private deep copy of
each requested model, optimizer and scheduler state. Complete all these copies
before touching live components. Apply those copies and return the original
deserialized payload as a caller-owned snapshot. This also separates the returned
snapshot from inputs retained by an admitted native load hook.

The checkpoint schema, validation, destination behavior, restricted
deserialization, parameter and child-scheduler objects, and existing rollback
logic are preserved. This adds transient memory proportional to the requested
component payloads during loading. It does not capture RNG, standalone EMA,
data-loader positions or unregistered FIM episode state; the existing public
checkpoint recovery contract continues to apply.

## Verification actually executed

The corrected focused run passed **38 tests**, with **zero failures, errors,
skips or warnings**: 13 new ownership/admission cases and all 25 existing public
checkpoint recovery tests. Warnings were treated as errors. Pytest reported
2.14 seconds; the complete command took 3.2576 seconds. The candidate adds one
copy-failure case after the baseline to verify that allocation failure precedes
any live mutation; it is not counted among the baseline's 12 cases.

The checks cover both mutation directions, exact next-update equality for
Adam/AdamW/momentum SGD, three native scheduler types including SequentialLR,
metadata aliases, a native input-retaining hook, model-only loads and an
injected pre-application copy failure. Existing malformed-checkpoint and
rollback tests also pass. These tests use only artificial small linear models.

Reproduce from this candidate checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
  python -m pytest -q -W error \
  tests/test_checkpoint_payload_isolation.py \
  tests/test_public_checkpoint_recovery.py
```

The recorded environment is Python 3.12.14, CPU PyTorch 2.14.1+cpu and pytest
9.1.1. Scoped compilation and maintained-source whitespace checks pass.
The raw baseline log retains pytest's original trace formatting, including
trailing spaces; it is preserved rather than rewritten to satisfy whitespace
lint. This is focused local verification, not a full-repository or hosted-CI
claim. An independent root-agent source/test review found no substantive issue
and performed no additional test execution.

## Research boundary and remaining work

No retained scientific checkpoint was loaded. No held outcomes, benchmark
campaign, scientific workflow dispatch, paid compute, main merge or scientific
release occurred. Historical adverse and mixed findings, incomplete evidence,
closed session budgets and protected holds remain authoritative.

Review the combined draft and its PR44 → PR38 prerequisites before any
integration into main. PR45's cancellation work is included in the combined
proposal; no PR closure or main merge follows from this integration. Scientific
execution still requires its separate
prospective protocol and authorization gates. This repair makes resumed state
ownership reliable; it establishes no model-performance benefit.
