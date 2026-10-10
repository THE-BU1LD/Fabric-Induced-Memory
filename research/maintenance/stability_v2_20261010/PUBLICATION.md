# Opt-in stable FIM diagnostics: publication follow-up

## Scope and algorithm
This adopts the previously delivered, locally tested stability_v2 implementation. It is prospective numerical maintenance, not a memory-benefit result. No existing model, trainer, metric, benchmark or protected protocol is rewired. The retained `fim/core/stability.py` remains Git blob `1f28c9494319c17cc8e0038eddfda55b804565da` on inspected publication parent `b417f3c4b0487b09f948eea70c5d8b94a21cf3d5` (PR #34).

For a finite nonzero field, represent its norm as scale s=max(abs(x)) and r=norm(x/s), without materializing s*r. Compare r against threshold/s; calculate log norm as log(s)+log(r). Half/bfloat16 reductions use float32; double inputs retain float64. Clipping computes (x/s)*(cap/r)*norm/(norm+eps), using log-space damping, so an underflowed global multiplier does not erase representable output. The original mathematical epsilon rule is retained; bitwise equality at all rounding boundaries is not promised.

Growth is the average difference of log(norm+eps), not a sequence of raw ratios. Every frame is validated. It measures growth per observation, not physical-time growth or a Lyapunov exponent. Diagnostics are no_grad. In-place clipping requires contiguous storage and validates before writing; it does not isolate concurrent external aliases. Final dtype underflow and floating-point comparison error remain possible. One normalized field-sized temporary is used, and scalar reads can synchronize accelerators. No runtime-speedup claim is made.

## Runnable adoption and tests

```python
from fim.core.stability_v2 import clamp_norm_, estimate_growth_rate, is_bounded
clamp_norm_(state, max_norm=5.0, eps=1e-12)
within_cap = is_bounded(state, threshold=5.0)
growth = estimate_growth_rate([previous_state, state], eps=1e-12)
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_stability_v2.py
```

The original delivery contained 45 passing candidate cases; 33 failures and 12 passes against unchanged source. The failure count includes deliberately added validation/API expectations, not 33 independent bugs. The entire original 140-case package was rerun successfully before this publication; the 45 FIM cases are included, not additional unique coverage. Tests use only artificial tensors, no scientific checkpoints or outcomes. Source SHA-256: ba6f579d0dca65fac7b4799cda391d5caa42e0d166842d44a5431aae8f74aeb6; test SHA-256: f9708f5c9c14b92507ecb7a6e7ff8d2753b6ea4a443861690c7e151fdab476df.

## State and release boundaries
The canonical session index preserves its predecessor as an exact Git-blob snapshot and links this note. The historical state and all its linked receipts retain their original source scope. PR #34's trace-bank code is inherited, not rewritten; the FIM diagnostic tests do not establish combined full-repository validation. The separate fractional-history, training, metrics, and scientific-run branches remain separate proposals.

The publishing commit uses [skip ci] to avoid unrequested hosted work. No hosted pass, scientific checkpoint promotion, protected execution, merge, deployment, or research completion is claimed. A scientific use must explicitly pin this version and its numerical policy in a prospective study. Next gate: independent code review and authorized repository-wide integration checks.
