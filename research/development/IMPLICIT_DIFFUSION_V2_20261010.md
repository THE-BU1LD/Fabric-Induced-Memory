# Prospective implicit diffusion repair — 10 October 2026

## Contract before implementation or execution

Parent: FIM PR38 head `92bcc3eb5322d967684673ca61f22a8bbebdc16b`.
This is a separately authorized engineering continuation of the user's request to
execute the research pipeline. Earlier sessions, budgets and artifacts stay closed.

The retained `DiffusionOperator.implicit_step` computes
`(x + dt*d*L(x))/(1 + dt*d)`. For the documented periodic five-point Laplacian,
constant fields have `L(x)=0`; therefore this expression changes a constant field
when `dt*d>0` and is not the backward-Euler solve it is named for. Large steps can
also amplify a high-frequency mode. These are algebraic implementation findings,
not observations from protected model outcomes.

Implement an explicit opt-in `ImplicitDiffusionOperatorV2`, plus a reusable
`periodic_backward_euler` function. Its core must solve exactly the existing
unit-spacing periodic stencil equation `(I - dt*d*L)y=x`, using Fourier multipliers
`1/(1 + dt*d*(4*sin(pi*ky/H)^2 + 4*sin(pi*kx/W)^2))`. The zero mode remains zero in
the Laplacian; no epsilon floor may damp it. Return the inverse transform with the
explicit original `(H,W)` shape. Support nonempty BCHW float32/float64 tensors and
nonnegative finite scalar/per-channel diffusivity. Preserve input and parameter
gradients. Validate controls before execution and reject unrepresentable products
or nonfinite outputs instead of silently publishing them.

The opt-in module must preserve the retained module's parameter names, diffusivity
parameterization, residual scale, stability scale and optional output clipping for
compatible isotropic nonfractional implicit configurations. Mean preservation and
contractivity apply to its inner diffusion solve; learned postprocessing is not
misrepresented as a heat-equation property. Existing source, default exports,
model callers, protocols, results and workflows are not edited. A future model
experiment choosing this repaired operator is a separately versioned development
hypothesis and must retain its own source/configuration identity.

## Verification budget and acceptance

- One tiny baseline diagnostic on the exact retained module.
- At most 35 new artificial tensor cases and one targeted regression process.
- One existing complete ordinary repository suite, maximum 180 seconds.
- One repair/review cycle and one conditional targeted rerun for a concrete defect.
- One numerical thread; CPU only; no training, benchmark campaign, protected
  outcome read, held seed, paid compute, release, merge or submission.

Independent oracles: construct the periodic Laplacian by coordinate loops, solve
the dense linear system, check residuals, exact constant/Fourier modes, non-growth,
input/diffusivity gradients and dtype/shape behavior. Retain raw results and all
failures. Passing these checks establishes only a bounded numerical implementation
contract; it neither resolves memory benefit nor completes the research project.

References used for the derivation and API contract:
- Benjamin Seibold, MIT 18.336 Spring 2009 lecture 14, Fourier-mode analysis of the
  heat equation and the central periodic stencil:
  https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/c7c12accea163007923288875bb42604_MIT18_336S09_lec14.pdf
- PyTorch rfft2 normalization:
  https://docs.pytorch.org/docs/2.14/generated/torch.fft.rfft2.html
- PyTorch irfft2 explicit shape requirement for odd spatial widths:
  https://docs.pytorch.org/docs/2.14/generated/torch.fft.irfft2.html
