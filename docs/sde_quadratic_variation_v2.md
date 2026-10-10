# Diagonal Ito integration with noise-scaled quadratic variation

This opt-in development correction is selected with
`from fim.stochastic.sde_v2 import SDEIntegratorV2`. The legacy
`fim.stochastic.sde.SDEIntegrator`, callers and historical outcomes are unchanged.

## Reproduced defect and model contract

For `dX = f(X,t)dt + g(X,t)dW`, a Milstein step with increment variance `v` is
`x_next = x + f*dt + g*dw + 0.5*g*g_prime*(dw**2-v)`.
The default sampler in the legacy implementation uses
`dw = noise_scale*sqrt(dt)*normal`, but its correction always subtracts `dt`.
Its correction therefore has nonzero conditional mean when `noise_scale != 1`.
With `x0=2`, `f=0`, `g=x`, `dt=0.1` and **zero noise**, the legacy one-step
result is **1.9**; the deterministic equation requires **2.0**, returned by v2.

V2 binds `v = noise_scale**2*dt` using the sampled amplitude before squaring,
which also avoids an intermediate square overflow at tiny dt. Custom noise
increments must explicitly declare their conditional variance for Milstein.
They are already scaled; the default noise_scale is not applied a second time.
This implementation covers scalar/componentwise Ito diffusion with independent
Brownian drivers. It does not supply iterated cross-noise integrals or a Levy
jump integrator, and a variance declaration does not establish Brownian noise.

Input and all coefficient fields have matching batch tensor shape, device and
dtype; coefficients must be finite. Zero requested steps return the initial
state without invoking fields. Zero noise does not consume the supplied RNG.
Nonfinite numerical steps fail before clipping could conceal them. Autograd
passes through the trajectory, fields and custom variance tensors. Configuration
is copied on construction so mutating the caller's dataclass does not alter it.

## Use

```python
import torch
from fim.stochastic.sde import SDEConfig
from fim.stochastic.sde_v2 import SDEIntegratorV2

solver = SDEIntegratorV2(
    SDEConfig(dt=0.01, steps=10, noise_scale=0.5, stability_clip=None),
    generator=torch.Generator().manual_seed(17),
)
x0 = torch.ones(2, 1)
trajectory = solver.milstein(
    x0, drift=lambda x, t: -0.1*x, diffusion=lambda x, t: 0.2*x,
    diffusion_grad=lambda x, t: torch.full_like(x, 0.2),
)
# [batch, steps+1, state dimensions...]
```

## Verification and scientific boundary

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q tests/test_sde_quadratic_variation_v2.py tests/test_dynamics.py`
returned **28 passed** in the isolated live-main snapshot (23 new integrator
fixtures plus 5 existing dynamics tests). Checks include hand-derived non-unit
noise steps, deterministic limits, unit-noise legacy Euler equivalence, state
gradients, independent-trajectory shape protection and exact zero-step behavior.

This is a prospective correction and a new implementation version, not a new
model efficacy finding. Historical values are not recomputed or relabeled.
Any scientific execution selecting v2 needs its own pinned protocol/source
identity. Protected studies remain held. No paid compute, full scientific
matrix, historical benchmark replay, paper update or submission was performed.
