# FIM checkpoint/config identity receipt — 2026-10-01

## Scope

This receipt records one bounded reproducibility repair. It does not change, rerun, or reinterpret any scientific outcome.

Canonical repository: `THE-BU1LD/Fabric-Induced-Memory`  
Base `main` commit inspected: `b6167d481df69fb0b783a06368060a0e15a75d21`  
Existing checkpoint evaluator blob inspected: `scripts/evaluate_checkpoint.py` = `b2808deef7ac4d948cbdaec196d1d8d16fb87f6f`

The canonical evaluator can load either `model_state_dict` or trainer `model` payloads, but its provenance output records only checkpoint/config paths and does not bind the evaluated checkpoint to a content digest or verify that an explicit config matches the checkpoint-embedded config.

## Repair

Added `scripts/verify_checkpoint_identity.py`, a read-only preflight that:

- loads checkpoints with `torch.load(..., weights_only=True)`;
- records checkpoint SHA-256, byte size, and state-dict key;
- records the checkpoint-declared source Git SHA when one is present;
- canonicalizes embedded/external configs to a stable SHA-256 identity;
- fails closed when an external config disagrees with the embedded checkpoint config;
- optionally requires or checks an expected source Git SHA;
- can emit a machine-readable JSON receipt without training or evaluating the model.

Added `tests/test_verify_checkpoint_identity.py` covering digest stability, restricted checkpoint loading, source-SHA checks, and config mismatch rejection.

## Verification actually executed

Environment used for the bounded helper test: Python runtime with PyTorch `2.10.0+cpu`.

Command:

```text
python -m pytest -q tests/test_verify_checkpoint_identity.py
```

Observed result:

```text
5 passed in 1.04s
```

A missing-checkpoint invocation was also executed and failed closed with `FileNotFoundError`, as intended.

These are synthetic serialization/config identity regression fixtures only. They are not scientific validation and do not constitute a FIM efficacy rerun.

Local pre-commit file digests of the exact tested helper sources:

- `scripts/verify_checkpoint_identity.py`: `b0df167f95935bea355add375975f1b0a017ccdf944ba17fb0030c70dab26b78`
- `tests/test_verify_checkpoint_identity.py`: `c4267767f1dac7ee88e2e05cc535da0257e30ec0735a38e67213a9d1e118b96d`

## October 1 release evidence bound in this review

Materialized release artifacts were hashed byte-for-byte before this receipt was prepared:

- `FIM_Spatial_Recall_Completion_Release.pdf`: `4ab95e7b1c3ae0210c757c8e1cac5765ab048ca95b72f8fd87a5788aea8a1194` (855,712 bytes)
- `FIM_Spatial_Recall_Completion_Release.tex`: `d5ae109508ded66eef17640884d8a8bc1ccd00ca3feeb2e29a60ce8883044165` (59,603 bytes)
- `COMPLETION_MANIFEST.md`: `baf03b399bdae51758261fe84e83a9cc13360d59f4f46f5e22730edee82081ca` (5,209 bytes)

The completion manifest labels the release `FIM-spatial-recall-completion-r1-2026-10-01`. It explicitly preserves the negative Study I endpoint, classifies the slow-EMA finding as a development diagnosis, and leaves the external-memory advantage inconclusive. No final protected confirmatory `H_mem` result is claimed.

The broader retained FIM paper reports a 24-cell component-study source-tree identity of `e8071b2243247361e3ea0da1eaa412e69f204593f80102fe5f837dead5300f46`, while its Git commit field is reported as unknown. That source-tree digest must not be silently equated with the current canonical Git commit.

## Scientific boundary preserved

- Original 80-update sparse-recall endpoint: negative.
- 320-update raw-vs-EMA study: development diagnostic, not confirmatory.
- External trace-bank advantage beyond recurrence/no-memory: unresolved/inconclusive.
- No new outcome training, protected-data access, held-out tuning, confirmatory rerun, release, submission, or merge was performed.
- This branch does not establish that the October 1 release was executed from current `main`.

## Remaining evidence gap and next narrow action

Run the new verifier against each retained historical checkpoint/config pair that is actually available in the archived FIM evidence package and save the JSON receipts beside those artifacts. If a historical checkpoint lacks a declared source SHA, retain that absence explicitly rather than assigning the current repository SHA retroactively. The successor delayed-recall protocol is separate and remains frozen/unexecuted.
