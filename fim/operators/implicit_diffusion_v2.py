"""Opt-in backward Euler for the retained unit-spacing periodic stencil.

The historical ``diffusion.DiffusionOperator`` and its scientific callers are
unchanged. Import this module explicitly when choosing the corrected solver.
"""

from __future__ import annotations

import math
import operator
from numbers import Real

import torch

from .diffusion import DiffusionOperator


def _nonnegative(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    return result


def _field(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor) or x.layout != torch.strided:
        raise ValueError("x must be a dense BCHW tensor")
    if x.device.type == "meta" or x.dtype not in (torch.float32, torch.float64):
        raise ValueError("x must be a materialized float32 or float64 tensor")
    if x.ndim != 4 or any(size == 0 for size in x.shape):
        raise ValueError("x must have nonempty BCHW dimensions")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("x must contain only finite values")


def periodic_backward_euler(
    x: torch.Tensor, diffusivity: float | torch.Tensor, dt: float
) -> torch.Tensor:
    """Solve ``(I - dt*d*L)y=x`` without constructing the dense matrix.

    ``L`` is the five-point periodic Laplacian with unit grid spacing, including
    repeated neighbors on a two-cell axis and cancellation on a singleton axis.
    ``x`` is nonempty BCHW float32/float64. ``diffusivity`` is a finite nonnegative
    scalar or C-vector (also accepted as 1,C,1,1); tensor controls must be on x's
    device. Gradients pass to x and tensor diffusivity. dt is a scalar control.

    The zero-frequency multiplier is exactly one. Zero dt returns owned identity
    storage after validation, without evaluating an FFT. Unrepresentable control
    products or nonfinite FFT results raise; they are never silently clipped.
    Work and temporary storage are O(B*C*H*W*log(H*W)) and O(B*C*H*W).
    """
    _field(x)
    dt = _nonnegative(dt, "dt")
    if isinstance(diffusivity, torch.Tensor):
        d = diffusivity
        if d.layout != torch.strided or d.device != x.device:
            raise ValueError("diffusivity must be dense and on x's device")
        if d.dtype not in (torch.float32, torch.float64):
            raise ValueError("tensor diffusivity must be float32 or float64")
        if d.shape not in (torch.Size([]), torch.Size([x.shape[1]]), torch.Size([1, x.shape[1], 1, 1])):
            raise ValueError("diffusivity must be scalar, C, or 1,C,1,1")
        if not bool(torch.isfinite(d).all()) or bool((d < 0).any()):
            raise ValueError("diffusivity must be finite and nonnegative")
        d = d.to(dtype=x.dtype)
    else:
        d = x.new_tensor(_nonnegative(diffusivity, "diffusivity"))
    if not bool(torch.isfinite(d).all()):
        raise ValueError("diffusivity is not representable in x's dtype")
    d = d.reshape(1, -1, 1, 1)

    if dt == 0:
        return x.clone()

    h, w = x.shape[-2:]
    ky = torch.arange(h, device=x.device, dtype=x.dtype).reshape(h, 1)
    kx = torch.arange(w // 2 + 1, device=x.device, dtype=x.dtype).reshape(1, -1)
    eigenvalues = 4 * torch.sin(math.pi * ky / h).square() + 4 * torch.sin(math.pi * kx / w).square()
    rate = d * dt
    if not bool(torch.isfinite(rate).all()):
        raise ValueError("dt*diffusivity is not representable in x's dtype")
    denominator = 1 + rate * eigenvalues
    if not bool(torch.isfinite(denominator).all()):
        raise ValueError("implicit diffusion multiplier is not representable")

    spectrum = torch.fft.rfft2(x, norm="ortho")
    out = torch.fft.irfft2(spectrum / denominator, s=(h, w), norm="ortho")
    if not bool(torch.isfinite(out).all()):
        raise ValueError("implicit diffusion produced a nonfinite result")
    return out


class ImplicitDiffusionOperatorV2(DiffusionOperator):
    """Corrected opt-in implicit solve with compatible retained state keys.

    This interface deliberately supports isotropic, nonfractional, implicit
    diffusion only. The inherited diffusivity parameterization, residual/stability
    scales and optional clipping are preserved. Therefore conservation and
    contractivity describe ``implicit_step``; they do not describe arbitrary
    learned postprocessing. Existing models/default exports are not switched.
    """

    def __init__(
        self,
        channels: int,
        diffusivity: float = 0.1,
        dt: float = 1.0,
        learnable: bool = True,
        max_diffusivity: float = 2.0,
        eps: float = 1e-6,
        use_residual: bool = True,
        clamp_output: bool = False,
    ) -> None:
        if isinstance(channels, bool):
            raise ValueError("channels must be a positive integer")
        try:
            channels = operator.index(channels)
        except TypeError as exc:
            raise ValueError("channels must be a positive integer") from exc
        if channels <= 0:
            raise ValueError("channels must be a positive integer")
        diffusivity = _nonnegative(diffusivity, "diffusivity")
        dt = _nonnegative(dt, "dt")
        max_diffusivity = _nonnegative(max_diffusivity, "max_diffusivity")
        eps = _nonnegative(eps, "eps")
        if eps == 0:
            raise ValueError("eps must be positive")
        for name, value in (("learnable", learnable), ("use_residual", use_residual), ("clamp_output", clamp_output)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a Boolean")
        # Preserve the legacy parameterization rather than silently remapping
        # checkpoints. Reject initial values that overflow that parameterization.
        super().__init__(channels=channels, diffusivity=diffusivity, dt=dt,
                         mode="implicit", fractional=False, anisotropic=False,
                         learnable=learnable, max_diffusivity=max_diffusivity,
                         eps=eps, use_residual=use_residual, clamp_output=clamp_output)
        if not all(bool(torch.isfinite(value).all()) for value in self.state_dict().values()):
            raise ValueError("initial diffusivity is not representable by the retained parameterization")

    def implicit_step(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return periodic_backward_euler(x, d, self.dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _field(x)
        if x.shape[1] != self.channels:
            raise ValueError("x channel count must match the operator")
        if self.mode != "implicit" or self.anisotropic or self.fractional:
            raise ValueError("v2 supports isotropic nonfractional implicit diffusion only")
        if not all(bool(torch.isfinite(value).all()) for value in self.state_dict().values()):
            raise ValueError("operator state must be finite")
        out = super().forward(x)
        if not bool(torch.isfinite(out).all()):
            raise ValueError("diffusion postprocessing produced a nonfinite result")
        return out
