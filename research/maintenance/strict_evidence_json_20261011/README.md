# FIM evidence JSON admission repair

The two maintained evidence readers previously used ordinary `json.loads`.
An object containing both `"execution_authorized": true` and a later
`"execution_authorized": false` silently became a single false value. The
unchanged frozen-method validator accepted that constructed contradictory copy
of the method. Duplicate run-status, metric or budget fields were similarly
invisible after decoding. These are input interpretation failures, not scientific
findings or evidence that a protected run occurred.

## Implemented behavior

`scripts/evidence_json.py` provides a shared UTF-8 JSON-object decoder. It rejects
duplicate keys at every depth, including identical values and equivalent Unicode
escape spellings; NaN/Infinity tokens; and real-number overflow to infinity.
Both the trajectory-isolated analyzer and delayed-recall successor verifier now
use it before semantic admission, backend loading or evidence publication.

Ordinary valid JSON retains its values and ordering. Python integer precision,
subnormal floats and signed zero are preserved. The analyzer continues to hash
the exact original manifest bytes. No numerical formula, method requirement or
frozen protocol is changed. The helper has no external dependencies and does not
authorize execution. Direct callers supplying an already decoded dictionary
still own their parsing boundary; this repair covers the two maintained CLIs.

## Verification actually performed

The same nine new stdlib test methods were run against original and repaired
readers. Original source failed **34 constructed subcases** of the stronger
parse-boundary assertions. The repair passed all nine methods. The tests cover
admission before semantic calls, malformed UTF-8/JSON, duplicate fields,
nonfinite values, original-byte identity, valid numerical values, and both
standalone script invocations.

Existing neighbors then passed: **17 evidence-publication methods** and
**13 frozen-method admission methods**, including their existing subtests.
There are **39 distinct passing test methods** across these three scoped runs.
The historical 864-test integration result was not rerun or added to this count.
The new tests and method-admission neighbors use constructed metadata and backend
doubles; no Torch backend, checkpoint, generated trajectory or outcome was loaded.

Python was 3.12.14. Baseline and candidate logs, commands, environment/version,
source hashes and elapsed times are retained beside this file. The direct method
probe separately demonstrates actual original acceptance of a contradictory
method copy using only the unchanged metadata validator. Independent root source
review found no blocking issue before publication. Compilation and scoped
source/test/metadata whitespace checks passed. The complete staged whitespace
check reports six trailing spaces emitted by unittest in the retained baseline
failure log; `whitespace.json` records that failure and the original bytes remain.

## Source and history

Base: draft PR44 at `a659bc621f38a527397ea7ae7a18f2f144e1c7b2`.
Only two existing runtime files and the appended canonical index change. The
complete predecessor canonical index is retained byte for byte. All other
existing files retain their identities, including frozen protocols, model,
training, benchmarks, numerical analysis, scientific results and workflows.

The old integration source verifier remains an exact historical artifact for its
original implementation tree. Its two reader-file expectations are superseded
only for this narrow maintenance continuation. `preservation.json` records the
current source boundary; it does not relabel old test results as current tests.

## Remaining work

Maintain the draft review and integration gates for PR44 and prerequisite PR38.
The frozen 40-cell matrix and delayed-recall successor remain held. No scientific
workflow dispatch, paid compute, main merge, manuscript update or submission
occurred. The repair establishes unambiguous software admission and makes no
memory-benefit, paper-readiness or scientific-completion claim.
