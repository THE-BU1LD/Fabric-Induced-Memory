# Implicit diffusion v2

The retained implicit method did not solve its stated linear system. On a
2-by-2 periodic field with dt=1 and diffusivity 0.5, it maps the constant 2 to
1.3333333333 and doubles a checkerboard's norm from 2 to 4. The correct
backward-Euler checkerboard multiplier is 0.2. These diagnostics use artificial
fields and do not establish a defect in any retained scientific result without
checking that result's exact caller/source.

## Explicit use

```python
import torch
from fim.operators.implicit_diffusion_v2 import (
    ImplicitDiffusionOperatorV2, periodic_backward_euler,
)
x = torch.randn(2, 3, 5, 7, dtype=torch.float64)
d = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
y = periodic_backward_euler(x, d, dt=0.5)
operator = ImplicitDiffusionOperatorV2(
    channels=3, diffusivity=0.2, dt=0.5,
    learnable=False, use_residual=False,
).double()
y_module = operator(x)
```

The explicit import leaves existing models and default exports unchanged. For an
isotropic, nonfractional, implicit legacy operator with matching learnability,
strict `load_state_dict(old_operator.state_dict())` preserves parameter names and
values; the intended prospective change is the inner linear solver. Configuration
is still required because a state dict does not encode dt or output conventions.

## Equation and scope

The unit-spacing periodic five-point stencil is
`L(u)[i,j]=u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]-4*u[i,j]`.
Wrap indices independently in each spatial direction. A two-cell axis correctly
has repeated neighbors; a singleton axis contributes zero.

Backward Euler requires `(I-dt*d*L)y=x`. A Fourier mode has Laplacian eigenvalue
`-4*sin(pi*ky/H)^2-4*sin(pi*kx/W)^2`, so its update is division by
`1+dt*d*(4*sin(pi*ky/H)^2+4*sin(pi*kx/W)^2)`. No epsilon is inserted into the
zero mode. The two real transforms use matched orthogonal normalization and the
explicit original spatial shape, including odd widths. The coordinate-loop dense
matrix/linear-solve tests supply an independent oracle for this derivation.

For admitted nonnegative controls, the mathematical inner solve preserves the
mean and contracts each nonconstant mode. FFT arithmetic introduces ordinary
floating-point rounding; bitwise equality is not promised. Shape, dtype, controls,
finite values and representability are checked. CPU float32/float64 were tested;
GPU behavior, half precision, anisotropy, fractional diffusion and nonperiodic
boundaries are outside this verification.

The module retains the old softplus diffusivity parameterization, diffusivity cap,
residual scale, stability scale and optional clipping. With `use_residual=True`
it returns a scaled correction, not the evolved field. Learned postprocessing
does not inherit the mathematical mean/energy guarantees of the inner solve.
Zero dt returns owned identity storage before FFT execution after validation;
the diffusivity derivative is zero, represented by a disconnected gradient at
that branch. Nonfinite intermediate/output behavior raises instead of saturating.

## Reproduction and evidence

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  --import-mode=importlib tests/test_implicit_diffusion_v2.py
```

The preimplementation contract, raw logs, JUnit XML, source identities and actual
command receipts are under `research/verification/implicit_diffusion_v2_20261010/`.
The canonical state retains the exact previous state and appends this session.
The previous adverse FIM/ablation records and all frozen execution holds remain.
A reviewed development implementation does not resolve the memory-benefit
hypothesis, establish model superiority, advance a scientific checkpoint or
make the paper submission-ready.

## References

The Fourier-mode derivation uses the standard central stencil discussed in
[MIT 18.336 lecture 14](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/c7c12accea163007923288875bb42604_MIT18_336S09_lec14.pdf).
The precise multiplier above is derived for this declared periodic discrete
operator. PyTorch documents the matched transform
[normalization](https://docs.pytorch.org/docs/2.14/generated/torch.fft.rfft2.html)
and the inverse transform's explicit
[shape requirement](https://docs.pytorch.org/docs/2.14/generated/torch.fft.irfft2.html).

Current local gate: **715 passed, 7 skipped, 16 existing GradScaler warnings**, 23.71 seconds in pytest; full command 95.2197 seconds. The earlier new fixture failure was a 4.44e-16 FFT rounding error against an unjustified bitwise assertion; only that new tolerance was repaired, and all earlier failure artifacts remain.
