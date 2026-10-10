# Reject invalid public rollouts before state changes

The public `autoregressive_rollout` helper previously checked sequence lengths
inside its loop. A short control sequence could write memory before its later
missing entry raised. A short teacher-forcing sequence could advance the model
again before the missing entry was discovered. Other malformed teacher inputs
could likewise fail after the first stateful call.

The helper now checks step count, nonempty finite initial inputs, sequence batch
and time coverage, matching devices, teacher feature shape and dtype, and finite
used prefixes before the first model call. Control feature axes and dtype remain
model-specific. Extra unused time entries remain ignored. The original
teacher-forcing convention, recurrence, output list and memory flag are retained.

## Verified results

| Collection | Result |
| --- | --- |
| New tests against PR38's original helper | 21 failed, 9 passed |
| New tests against the repair | 30 passed |
| Complete ordinary repository collection | 715 passed, 7 skipped, 164 subtests passed |

The complete collection reported 16 existing public-trainer GradScaler warnings,
30.74 seconds in pytest, and 35.012145 seconds for the whole command. This includes
the focused tests; overlapping counts are not added. The collection is based on
PR38, not on the separate diffusion proposal PR39. Both branches happen to add 30
tests; their equal totals do not represent the same suite.

Stateful fixtures verify that invalid input causes zero model calls and zero
memory writes. Valid tensor and tuple outputs, autoregression, teacher-forcing
and control order are checked analytically. The real public FIMModel matches
manual stepping bitwise across predictions and final persistent state.

Raw logs, JUnit XML and source-hashed receipts are retained here. The baseline
uses unchanged test source and the original helper at
`92bcc3eb5322d967684673ca61f22a8bbebdc16b`; the later receipts verify source identity
before and after execution. Independent source review found no blocking issue
and did not rerun tests. All three processes used CPU and one numerical thread.

## Limits and follow-up

This protects against invalid caller inputs before execution. It cannot undo a
model exception after an admitted step has begun, and it does not reset memory
between independent episodes. Finite input scans can add overhead and accelerator
synchronization; only CPU behavior was qualified.

No model/objective, benchmark, protocol, historical result, scientific workflow
or protected hold changes. No retained demonstration or scientific run occurred.
The canonical index retains its exact predecessor and appends this maintenance
session. Review this draft with its PR38 prerequisite before integration.
