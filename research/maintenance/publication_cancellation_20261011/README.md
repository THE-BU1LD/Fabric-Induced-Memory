# FIM evidence publisher: recover interrupted replacements

## Result

The existing evidence publisher now recovers Python cancellation during a
multi-file replacement. Successful recovery reraises the original exception
object. Interrupted recovery keeps available original bytes and explicit
recovery instructions, while continuing to recover other destinations.

Source base: draft [PR44](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/44)
at `58c0b5b1249b44456052de4ddda7adbbb0a7acae`. The only changed existing runtime
file is `scripts/evidence_publication.py`. This proposal is a separate draft
stacked above that inspected source, with its canonical research history
preserved verbatim in `predecessor_RESEARCH_STATE.json`.

## Reproduced defect

The original replacement and rollback handlers caught `Exception` only.
`KeyboardInterrupt`, `SystemExit` and `GeneratorExit` therefore skipped recovery.
Earlier output files remained replaced, but `finally` deleted their backups.
The first baseline recorded 30 failing constructed subcases before a deliberately
injected recovery `KeyboardInterrupt` escaped the test's expected-error
assertion and terminated that baseline process. This was an incomplete baseline,
not a completed test collection. Exact original source, test source, output and
receipt are retained.

The candidate test harness catches `BaseException` before asserting the required
recovery-error type so this witness cannot terminate a future test runner.
The baseline was not rerun or rewritten.

## Repair

The publisher journals an attempt before calling atomic `os.replace`. Recovery
checks whether the staging source was consumed; this covers an exception both
before replacement and immediately after replacement completed. An uncommitted
attempt does not need restoration.

Recovery handles Python cancellations as well as ordinary errors. A consumed
backup or successfully removed new output counts as completed recovery even
if that operation raises immediately afterward. A failed recovery-status probe
retains the available backup and reports unresolved recovery. If recovery is
complete, the original publication exception is reraised unchanged.

## Verification and retained iteration

The final focused command is:

```sh
python -m unittest discover -s tests -p 'test_evidence_publication*.py' -v
```

It passed **27 test methods** in **0.172 seconds** (0.279071 seconds for the
complete command): 17 unchanged publisher/CLI/workflow methods and 10 new
methods containing 54 constructed cancellation subcases. The 54 are nested
within those 10 methods; they are not additional independent tests. Python is
3.12.14. Source hashes matched before and after execution. `final.log` and
`final_receipt.json` retain the exact output, command and hashes.

The first candidate passed 24 methods. Source review then identified two gaps:
an exception after a completed rollback could report a consumed backup, and a
failure in the recovery status probe could skip preservation. The one reserved
repair cycle added the correction and three focused test methods. Exact first
candidate source/tests/receipts remain under `before_review/`; the original
24-method result overlaps the final collection and is not added.

Cases cover cancellation before and after each atomic replacement, existing and
new files, interrupted backup restoration/removal, completed rollback followed
by cancellation, failed recovery-status probes, staging interruption, original
exception identity, permissions, successful publication and actual CLI wiring.
No backend installation, Torch import, model fit, generated trajectory, retained
scientific outcome or protected evaluation was used. This is focused software
verification; no current full-repository or hosted CI pass is claimed.

## Limits and scientific state

The publisher still requires one writer and quiescent readers. Separate files
do not become visible atomically as a set. SIGKILL, power loss, arbitrary
asynchronous interruption and concurrent mutation remain outside this repair's
guarantee. An incomplete recovery raises `PublicationRollbackError` carrying
the original exception and actionable remaining recovery records.

Frozen protocols, scientific/numerical functions, model/trainer source,
workflows, source-bound historical outcomes and adverse findings are unchanged.
The 40-cell trajectory-isolated matrix and delayed-recall successor remain held.
Passing these engineering tests does not establish a memory benefit, complete a
scientific checkpoint or make FIM submission-ready. Maintainer review of this
draft and its prerequisites is the next integration action.

## Source and formatting preservation

All 441 unrelated parent leaves and modes retain their exact Git identities.
Only the publisher and appended canonical index change among existing paths.
Compilation and scoped whitespace checks pass. The full whitespace check
reports four original unittest-emitted trailing spaces in `baseline.log`; raw
failure evidence is retained exactly. See `preservation.json`.
