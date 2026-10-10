# Standalone trace-bank admission repair

Base: `84160703cc733c6deb150cf7b6ee5bb3a84d699b` (main).

## Reproduced failures

The standalone `fim.memory.trace_bank.TraceBank` retained aliased storage after
`detach().to()`: zeroing a caller's value tensor changed stored `[3, 4]` into
`[0, 0]`. A two-row batch with one metadata mapping raised `IndexError` after
inserting its first row. NaN salience was accepted into persistent memory.

The public `add` and `batch_add` paths now prepare owned copies of keys, values
and nested metadata. Admission validates dimensions, finite numbers (also after
dtype conversion), integer timestamps/layers and batch metadata length. Merges
and pruning are staged against a private list; entries, clock and inferred device
are published only when the entire batch succeeds. Invalid configuration is
rejected when constructing the bank. Signed finite salience remains supported;
existing merge weights still floor salience at `1e-6`.

Valid-input cosine merge selection, salience-weighted averaging, decay-aware
capacity pruning and retrieval equations retain their existing definitions.
This is transactionality for one method invocation, not concurrent-writer
isolation. The existing `recent`/`iter_traces` inspection interfaces still expose
trace objects; callers must treat those as read-only. Owning the original write
inputs does not make the entire public Python object immutable.

## Local engineering verification

`python -m pytest -q tests/test_trace_bank_admission.py tests/test_memory.py`:
**35 passed**. This includes 32 new parameterized cases and the three existing
memory tests. Targeted Ruff (`E4,E7,E9,F,I`) passes for both changed Python files.

Tests cover caller storage and nested metadata ownership, bad later rows and
batch metadata, first-write device rollback, dtype overflow, deterministic
weighted merge/prune/top-1 retrieval, valid batch/sequential equivalence and
an injected execution failure after a staged merge. Existing tests exercise the
real retrieval, consolidation and model helpers on small synthetic tensors.

## Scientific boundary

This change affects the standalone exported memory subsystem; the maintained
models/core have distinct trace-bank implementations. The protected 40-cell
trajectory-isolated runner, model, benchmark/metric source, protocol, workflows,
checkpoints and retained adverse outcomes are unchanged. No scientific matrix,
protected outcome, paid compute or research-completion claim is introduced.
Open trainer, evidence-admission and fractional-history PRs do not overlap these
changed implementation and test paths.
