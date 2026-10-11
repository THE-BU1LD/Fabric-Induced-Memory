# Public rollout input preflight — 10 October 2026

This bounded engineering session starts from FIM integration PR #38 at
`92bcc3eb5322d967684673ca61f22a8bbebdc16b`. Open proposals were inspected;
none changes `fim/training/rollout.py`. This is independent of opt-in diffusion
PR #39 and does not import or rerun that proposal.

## Reproduced contract risk

The public autoregressive helper checks control/teacher-forcing length inside
its step loop. Invalid future input can therefore raise only after the model
has advanced its recurrent cache or written memory. An invalid teacher-forcing
batch or spatial shape can also fail after an otherwise valid first step.

Validate caller-owned input structure and finite used prefixes before the first
model call. Keep the recurrence, teacher-forcing convention, control forwarding,
output format and caller-controlled memory behavior unchanged for valid calls.
Control feature axes remain model-specific. This does not promise rollback when
the model itself fails during an admitted rollout, or reset independent episodes.

## Verification and scope

Use artificial stateful models to witness call counts, state, inputs and outputs;
also compare the real public FIMModel with a manual sequence from identical
initial state. Preserve the pre-repair failing receipt and tested source hashes.
Allow one focused baseline, one focused repaired suite, and one complete ordinary
suite, with at most one additional focused run for a concrete defect. Each test
process is bounded at 180 seconds, on CPU with one numerical thread. No protected
data, scientific workflow dispatch, retained demo, paid compute or main merge.

The changes are caller-admission checks, not a new scientific model or objective.
Every frozen source, protocol, historical result and existing execution hold
remains unchanged. Append this session to the canonical research index and retain
its exact predecessor. Publish a separate draft for review if authorized access
is available.
