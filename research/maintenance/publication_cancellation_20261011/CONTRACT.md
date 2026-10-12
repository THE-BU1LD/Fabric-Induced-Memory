# Evidence-publication cancellation repair — 11 October 2026

This is a new bounded engineering session under the user's request to find and
complete outstanding work. Base: FIM draft PR44, commit
`58c0b5b1249b44456052de4ddda7adbbb0a7acae`, tree
`cdd6c51d661a119c4a836429b2a06ce042ed1998`.

## Concrete failure and allowed change

`scripts/evidence_publication.py` catches only `Exception` around replacement
and rollback. A Python cancellation such as `KeyboardInterrupt` during a later
replacement can leave earlier outputs replaced and then delete their backups
in `finally`. The publication journal is also updated after the replacement,
so a cancellation delivered immediately after an atomic rename can omit that
destination from recovery.

Repair only this filesystem publisher's cancellation/recovery behavior. Admit
all Python `BaseException` failures to rollback, journal a replacement before
attempting it, distinguish consumed staging files from uncommitted attempts,
and preserve recovery bytes if rollback itself is interrupted. Reraise the
original failure after successful recovery. Keep the normal successful bytes,
permissions, existing output admission and API unchanged.

## Bounded verification

Use only constructed temporary UTF-8 files. Retain the exact original source,
predecessor canonical state, one baseline run of the new regression collection,
and one candidate run containing the new tests plus existing publisher tests.
Allow one repair/review cycle and one additional focused candidate run only if
a concrete remaining defect is identified. Each test command has a 60-second
ceiling. Use Python stdlib; install no dependencies, import no Torch backend,
read no scientific outcomes and execute no model or benchmark.

Check Python compilation, scoped whitespace and exact preservation of all
unrelated tracked blobs/modes. Request independent source review before draft
publication. Record actual failures, counts, timings and source identities.
Historical test totals are not current verification and are not added to this
session's distinct count.

## Scientific and operational boundary

Frozen method/protocols, numerical/statistical functions, model/training code,
workflows and retained results remain byte-identical. Protected execution and
outcome holds remain in force. No scientific dispatch, paid job, retained demo
rerun, main merge, manuscript edit, submission or scientific claim follows.

The publisher still requires one writer and quiescent readers. This repair
does not establish crash-atomic multi-file visibility, SIGKILL/power-loss
recovery, concurrent-writer safety or general async-signal safety at arbitrary
Python instructions. Its tests inject exceptions at filesystem call boundaries,
including after an atomic rename has completed.

## Review artifact

Create an isolated draft above the inspected PR44 head. Append the actual
session to the canonical index and preserve its complete predecessor verbatim.
Do not overwrite the existing PR44 branch or close/merge any proposal.
