# Additive colored Gaussian noise V2

## Defect and prospective scope

At integration PR38 commit `92bcc3eb5322d967684673ca61f22a8bbebdc16b`,
`fim/stochastic/noise.py::ColoredNoise` Fourier-filters the input itself. For
inputs with two or more dimensions it never samples a random field: zero stays
zero and repeated inputs give identical outputs. Its sample-standard-deviation
normalization also produces nonfinite output for a field with one spatial site.

`fim.stochastic.colored_noise_v2.ColoredNoiseV2` is an explicit development-only
alternative. Existing noise classes, model factories, training/evaluation callers,
frozen protocols, retained outcomes and manuscript values are preserved. Using
this changed stochastic mechanism in a scientific study requires a separately
declared version and prospective protocol. These component checks do not show
memory benefit or improve the historical negative/mixed results.

## Exact distribution contract

For each independent field, draw independent standard real Gaussian values
`w` on the requested grid. With orthonormal forward and inverse DFTs, set

`H(f) = (sum_j f_j**2 + eps)**(-beta / 4)`

and return `x + sigma * inverse_DFT(H * DFT(w))`. Frequencies are measured in
cycles per grid sample. `eps > 0` regularizes DC; positive beta emphasizes low
frequencies and negative beta emphasizes high frequencies. The power multiplier
is `H**2`, so it approaches `|f|**(-beta)` away from the regularization scale.
The real, even multiplier retains Hermitian symmetry. No input-value-derived
noise, cross-field averaging or data-dependent normalization is used.

With `normalize=True`, replace `H` by `H / sqrt(mean(H**2))`. For independent
unit-variance Gaussian inputs, this gives **unit ensemble marginal variance at
every site**, hence variance `sigma**2` after scaling. This is deterministic
spectral normalization and preserves the Gaussian distribution. An individual
draw is not forced to have zero spatial mean or unit sample standard deviation.
That distinction matters for highly correlated fields and a one-site field.
The one-site normalized case is ordinary scalar Gaussian noise.

The normalized path computes the equivalent ratio after subtracting the maximum
log amplitude. This avoids premature overflow when the common spectral scale
cancels. Values that cannot be represented at the input dtype fail explicitly.
With `normalize=False`, the raw multiplier is used; the marginal variance before
`sigma` is `mean(H**2)` and depends on beta, eps and grid shape.

`spatial_dims` selects only the last axes. Leading axes index independent fields:
use `spatial_dims=1` for a `(batch, channels, time)` tensor and `spatial_dims=2`
for `(batch, channels, height, width)`. The default is **one** spatial dimension.
No batch or channel axis is inferred or transformed unless explicitly included.

```python
import torch
from fim.stochastic.colored_noise_v2 import ColoredNoiseV2

generator = torch.Generator(device="cpu").manual_seed(29)
noise = ColoredNoiseV2(beta=1.0, eps=1e-3, sigma=0.1, spatial_dims=2)
sample = noise(torch.zeros(2, 3, 5, 7), generator=generator)
```

## Execution semantics and limitations

- A supplied generator must match the tensor device. Its state belongs to the
  caller and advances once per nonzero-amplitude draw. Store and restore that
  state when exact replay is needed. A local generator does not reseed global RNG.
- Without a generator, PyTorch's default device generator is used. Reproducibility
  across different PyTorch versions, devices or FFT backends is not promised.
- Beta zero bypasses the FFT and yields exact scaled white noise for the same
  generator. Sigma zero is an identity and consumes no random values.
- Inputs must be finite, nonempty float32/float64 tensors. No dtype/device copy,
  input mutation, clipping or hidden repair occurs. Parameters must be finite,
  with positive eps and nonnegative sigma. Beta may have either sign.
- Noise is independent of input values, so the input Jacobian is the identity.
  Configuration scalars are not trainable parameters.
- Deterministic filter validation occurs before drawing. A failure after drawing
  (for example output overflow) may advance RNG state; no rollback is promised.
- This is a periodic-grid Gaussian spectral filter. It does not implement a
  causal temporal noise generator, nonperiodic boundary conditions, cross-channel
  covariance, or exact unregularized power-law noise at DC.

## Bounded verification plan and receipts

New engineering session: `fim-colored-noise-v2-20261010`.

This session permits local syntax/identity checks and the existing ordinary
pull-request test workflows, with one corrective cycle only for a concrete
implementation/test failure. Tests use analytic basis fields and tiny generated
arrays. No training, protected data, scientific outcome run, paid compute, main
merge, manuscript change or release is part of this session. The inherited
scientific job is still conditional on manual dispatch; no dispatch is requested.

`tests/test_colored_noise_v2.py` retains positive assertions documenting all three
legacy failures and checks the new stochastic behavior, exact seeded replay,
input-gradient identity, independent leading axes, beta-zero identity, one-site
handling, malformed input rejection and pre-draw filter validation. The four-site
analytic covariance oracle has first row `[1, 4/11, 1/11, 4/11]` for beta 2 and
eps 1/16. It tests the normalization exactly without a statistical pass threshold.
The tests also check a closed-form raw Fourier-mode gain and relative gains on
odd, even and singleton-height two-dimensional grids.

Actual command outcomes and exact source identities are appended to
`research/verification/colored_noise_v2_20261010/receipt.json`; inherited PR38
verification remains attached to its original source and is not counted as a
test result for this new implementation.
