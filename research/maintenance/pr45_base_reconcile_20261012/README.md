# FIM PR45: reconcile the current PR44 base

## Resolved integration blocker

On 12 October 2026, GitHub reported draft PR45 as conflicting. Its earlier
description recorded a conflict-free state before PR44 gained its full ordinary
repository acceptance record. Both branches had appended a different session
to `research/RESEARCH_STATE.json` at the same position.

This continuation reconciles the existing draft. It retains PR45's exact
publisher, tests, and evidence, imports the eight new PR44 evidence files with
their existing Git blob identities, and preserves both independently appended
canonical sessions. Complete predecessor state files are retained verbatim.

The reconciliation commit has the previous PR45 head
`323fc2738c08daed9dcac48a429cc4b1dd8b0e4a` and current PR44 head
`2c0d63959defe5544ee0f40b86f7de2f430d0125` as parents. Their common ancestor is
`58c0b5b1249b44456052de4ddda7adbbb0a7acae`. The existing PR45 branch is advanced
by an ordinary lease-checked fast-forward. PR44 and main remain unchanged.

## Exact source preservation

All three parent trees were independently reconstructed from their complete
path/mode/blob inventories and matched the immutable Git commit tree hashes.
Only the canonical index conflicts. All **460 other PR45 leaves and modes**
remain exact. Every incoming PR44 file retains its original blob and mode,
including the compressed full-suite JUnit. `preservation.json` lists those
eight identities and the precise parent trees.

The canonical index retains both source-specific records unchanged and appends
this continuation. Its current-session pointer and date identify this work;
the scientific holds, adverse findings, earlier closed budgets and other
canonical fields retain their previous meaning.

## Verification performed

One bounded standard-library run passed **27 test methods**, with no failures
or skips, in **0.154 seconds** in unittest and **0.234 seconds** for the complete
command. These are the unchanged publisher, cancellation, CLI and workflow
tests. Source hashes were identical before and after. Scoped compilation passed.
An independent assistant reviewed the tree union and history preservation;
the final current-session pointer resolves to exactly one retained record.
This review is recorded in `independent_review.json` and is not human
maintainer or scientific acceptance.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_evidence_publication*.py' -v
```

`test.log` and `test_receipt.json` record the command, actual Python environment,
timing and seven runtime/test/workflow hashes. Fourteen selected original files
were materialized and checked against their Git blobs; this was a selected
source snapshot, not a complete local checkout. Complete remote-tree identity
checks separately cover inherited paths and modes.

The **873 passed / 7 skipped / 204 passing subtests** receipt imported from PR44
remains bound to source `58c0b5b1249b44456052de4ddda7adbbb0a7acae`. It is not
reclassified as full-suite acceptance of PR45's cancellation fix. Current and
historical publisher test counts overlap and are not summed. No new test,
numerical function or runtime implementation was authored in this continuation.

## Remaining gate

The draft remains open for maintainer review with PR44 and PR38. No main merge,
scientific workflow dispatch, protected outcome access, model fitting, paid
compute or submission occurred. The 40-cell matrix and delayed-recall successor
remain **HELD**. This resolves a branch integration blocker and does not establish
memory benefit or scientific completion.

The commit preserves the existing `[skip ci]` and `skip-checks: true` convention.
No new hosted or full-repository acceptance result is claimed.
