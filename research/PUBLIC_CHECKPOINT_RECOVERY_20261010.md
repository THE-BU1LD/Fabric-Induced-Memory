# Public checkpoint save/load recovery

The inspected source parent is `22329862aaeff7da61d30e6aba3041f445ad0e4d`.
This repair concerns `fim.utils.checkpoint`. The separate experiment checkpoint
implementation, benchmark generators, frozen protocols, held trajectory matrix,
retained scientific results and paper artifacts are unchanged.

## Demonstrated failures

The original helper saved directly over the destination. A failed serialization
could therefore destroy the previous checkpoint or leave a partial first file.
It also applied model weights before discovering a malformed later tensor or
optimizer component, leaving the caller partly resumed after an exception.
The initial regression suite recorded **17 failures and one exact-update control
pass**. The complete output is retained in `public_checkpoint_baseline.txt`.

Independent review then found that native optimizer/scheduler loaders accept
some malformed state that fails only on the next update: missing Adam moments,
a primitive negative Adam step, a dictionary-valued scheduler gamma and a
negative squared moment. An optimizer post-load hook could also raise again
during rollback and leave the learning rate corrupted. Those five additional
failures are retained in `public_checkpoint_peer_failures.txt`.

A further independent witness found that assigning a scheduler's serialized
state directly to its attributes replaces `SequentialLR` children with plain
dictionaries. The repair now snapshots the live scheduler object graph and
preserves the original child scheduler/optimizer references. The final test
checks those identities and executes the next update after rejection. An
attempted local before-fix capture overlapped the edit and passed; it is excluded
as a failing baseline. The independent witness remains in the review receipt.

Finally, a primitive SGD momentum buffer was accepted and failed on the next
step. Its separate failing case is retained in `public_checkpoint_sgd_failure.txt`.
The final suite contains 25 distinct tests, including the original exact AdamW
and scheduler continuation control.

## Current public contract

```python
from fim.utils.checkpoint import save_checkpoint, load_checkpoint

save_checkpoint(path, model, optimizer, scheduler, epoch=completed_epoch,
                extra={"run": "caller-owned run identity"})
payload = load_checkpoint(path, model, optimizer, scheduler)
```

Saving retains the existing replacement-on-success behavior, but first writes
an exclusive unpredictable file in the destination directory, flushes and
fsyncs it, and then atomically replaces the destination. Serialization or
publication errors clean only that temporary file and preserve the previous
destination. This is not a claim of directory-entry durability after power loss.

Loading uses explicit restricted tensor deserialization. Epochs are nonnegative
integers. Model state must contain exactly matching dense materialized tensor
keys, shapes and dtypes, finite values, and consistent entries for shared
parameter aliases. Extra state must consist of tensors and primitive containers.
Passing an optimizer or scheduler requires that component to exist in the
checkpoint; a weights-only warm start should omit those arguments.

The helper supports Adam, AdamW and SGD. It verifies complete Adam moments,
nonnegative squared moments, scalar tensor steps, matching momentum shapes and
dtypes, parameter ordering/group structure and finite nonnegative controls.
Native scheduler state must match the current state structure. Validation
finishes before caller-owned state changes. If application raises, registered
model tensors, optimizer state/groups and the scheduler object graph are
restored without invoking the failed optimizer load hook again. Parameter and
child-scheduler references remain connected to the original objects.

The caller must stop concurrent model/optimizer mutation during save/load.
Callbacks that modify unrelated external objects are outside the restoration
contract. This helper does not capture global or external RNGs, data-loader
cursors, standalone EMA objects, or unregistered episode state. In particular,
loading these weights is not an exact continuation of FIM episode memory.
The caller still controls episode reset/recovery and the frozen experiment path.

## Verification and scientific boundary

The committed synthetic suite exercises real AdamW/SGD updates, scheduler
continuation, compound scheduler identity, actual post-load hooks, incomplete
serialization, and rejected model/optimizer/scheduler state. All 25 recovery
cases pass. The complete repository suite and exact hosted revision are recorded
in the PR and canonical state; earlier failing receipts remain unchanged.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
  python -m pytest -q
```

No retained scientific checkpoint is loaded by the new cases. This is engineering
evidence only: it neither releases a protected hold nor establishes memory
benefit, GPU execution, paper readiness or research completion.
