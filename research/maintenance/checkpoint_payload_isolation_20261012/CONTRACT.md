# Public checkpoint payload isolation — 12 October 2026

Source parent: `323fc2738c08daed9dcac48a429cc4b1dd8b0e4a`, FIM PR45.
The existing public recovery contract is
`research/PUBLIC_CHECKPOINT_RECOVERY_20261010.md`.

## Scope and expected behavior

Investigate the public `fim.utils.checkpoint.load_checkpoint` return value.
Native optimizer and scheduler loaders may retain references to incoming tensor
or container state. Returning those same objects lets a caller accidentally
change active training state by editing checkpoint metadata, and can make the
returned checkpoint change as training advances.

On success, return a caller-owned snapshot of the deserialized checkpoint.
Prepare private copies of every requested component before any live mutation;
apply only those copies. Preserve validation, rollback, parameter identity,
child-scheduler identity, checkpoint format, and exact next-update behavior.
No default training or scientific behavior changes.

## Prospective verification

Use artificial small linear models only, with real CPU Adam, AdamW and momentum
SGD updates. Check both directions of mutation, independent weight snapshots,
shared metadata aliases, nested native scheduler containers, and admitted load
hooks that retain their input. Run the new cases first on the exact parent,
retain failures, then run the repaired cases with the existing public recovery
tests. Inspect the diff and preserve every unrelated parent blob and mode.

Allow one baseline process, one candidate process, and at most one additional
targeted process if a concrete test or review finding requires correction.
Each process has a 120-second ceiling and one numerical thread. Stop once the
identified ownership risk and neighboring recovery behavior are verified.

## Research boundary

This is a new engineering session. All previous source-specific budgets,
receipts, failed/negative outcomes and research holds retain their original
meaning. Do not load retained scientific checkpoints, run the held matrix,
inspect protected outcomes, repeat retained demos, perform benchmark fitting,
dispatch scientific workflows, spend money, merge, or make efficacy claims.
Keep the ordinary scientific workflow guard unchanged. Finish with the exact
source/test receipts and a separate draft for review.

## Metadata integration discovered before publication

Fresh readback found that PR44 advanced to
`2c0d63959defe5544ee0f40b86f7de2f430d0125` after the PR45 source parent was
created. Its nine changed paths are only the canonical index and eight new
verification artifacts. Import those artifacts as exact Git objects and retain
the complete current PR44 state snapshot. Union the source-specific maintenance
entries with PR45 and this session, leaving the earlier records unchanged.

The separate draft targets current PR44 and includes the original PR45
cancellation implementation plus the new checkpoint ownership fix. Do not modify
either existing branch. Confirm that the entire runtime/test source still equals
the tested PR45-derived candidate; this metadata union does not require another
test process or extend the closed engineering-test budget.
