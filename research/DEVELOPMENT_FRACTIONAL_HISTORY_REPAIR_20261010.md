# Development repair: bounded fractional histories

Date: 10 October 2026. Base: `84160703cc733c6deb150cf7b6ee5bb3a84d699b`.

## Observed defect and scientific scope

The exported `fim.core.fractional.FractionalMemory` stores time on axis `-2`,
but its capacity branch used five hardcoded indices and sliced the last axis.
On the unmodified base, three pushes into a two-state memory produced:

- `(2, 3)` states: `IndexError: too many indices for tensor of dimension 3`.
- `(2, 3, 4, 5)` states: reported history length **3**, above its limit of **2**.
- Mutating the first input after pushing it changed a subsequent integer-order
  difference from expected `[3, 4]` to `[-995, -993]` because `detach()` alone
  retained the caller's storage.

These witnesses use artificial tensors and the exact base module loaded from
Git. They do not use scientific trajectories or retained outcomes.

## Implemented contract

For a state of shape `(..., D)`, the helper retains time on axis `-2`. Before
concatenation it keeps at most `max_history - 1` earlier states, then snapshots
the new detached state. This bounds both temporal length and the backing tensor
allocation for vector, batched and spatial inputs. Shape, dtype and device must
remain fixed within an episode; `reset()` starts a new episode.

The finite-history Grunwald–Letnikov formula is unchanged:

`dt**(-alpha) * sum_k w_k * x_(t-k)`, with `w_0=1` and
`w_k=-(alpha-(k-1))*w_(k-1)/k`, for the retained history only.

This is a truncated GL difference, with no Caputo initial-value correction.
A constant finite history can have a nonzero difference. The standalone
`fractional_difference` retains gradients; episodic `FractionalMemory` remains
detached by design. Invalid counts, time steps, shapes and non-finite values
raise clearly, and a rejected push does not change the previously valid history.

## Verification

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_fractional_memory_history.py tests/test_fractional.py
```

Result: **33 passed in 4.41 seconds**: 31 added cases and two existing tiny
fractional-PDE cases. Runtime: Python 3.12.14, PyTorch 2.14.1+cpu, pytest 9.1.1.
The CPU runtime was supplied through the workspace dependency target; no host
Python environment or repository dependency specification changed.

The tests cover nine rank/capacity combinations with nine pushes each, direct
binomial coefficients, an independently calculated rolling reference, gradients,
detached snapshot isolation, failed-push preservation, overflow, reset, invalid
budgets/time steps, the finite-history constant case and complex dtype support.
Focused Ruff (`E4,E7,E9,F,I`), Python compilation and whitespace checks pass.
The first local test attempt lacked SymPy in the temporary Torch runtime; those
dependency failures disappeared when the complete existing runtime was supplied.

## Preserved boundaries

No FIM architecture, benchmark source, protected 40-cell protocol, workflow,
retained metric, checkpoint, paper, or earlier negative finding is changed.
The helper is an exported API; repository search found no maintained model
caller of this class. A future study that uses its repaired behavior must record
this source revision, especially if it previously exceeded the history limit.

The scientific result remains unestablished by this repair. The existing
trajectory-isolated execution hold and the delayed-recall successor protocol
remain in force. Open PRs #26–#28 are separate work. The new canonical state
explicitly indexes the historical truth record and the already-merged dispatch
guard, without rewriting the pending truth-reconciliation patch.
