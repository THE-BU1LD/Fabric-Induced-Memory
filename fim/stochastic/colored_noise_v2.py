"""Opt-in additive Gaussian noise with a declared spatial power spectrum.

This prospective component is not selected by the maintained scientific runners.
See docs/colored-noise-v2.md for normalization and evidence boundaries.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn


def _finite_real(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


class ColoredNoiseV2(nn.Module):
    """Add independent Gaussian fields colored over the last spatial axes.

    ``spatial_dims`` identifies the final axes to filter. Every preceding index
    identifies an independent field, so batch and channel axes are not mixed.
    The amplitude filter is ``(sum(frequency**2) + eps)**(-beta / 4)``.
    ``normalize=True`` gives unit *ensemble marginal variance* before ``sigma``;
    it does not impose a sample mean or sample standard deviation on each draw.

    Only finite, nonempty float32/float64 input tensors are admitted. The noise
    depends on input shape/device/dtype, not input values. Gradients with respect
    to input are therefore the identity. Pass a device-compatible generator for
    caller-owned reproducibility. A zero ``sigma`` does not consume random state.
    """

    def __init__(
        self,
        beta: float = 1.0,
        eps: float = 1e-6,
        normalize: bool = True,
        *,
        sigma: float = 1.0,
        spatial_dims: int = 1,
    ) -> None:
        super().__init__()
        self.beta = _finite_real("beta", beta)
        self.eps = _finite_real("eps", eps)
        self.sigma = _finite_real("sigma", sigma)
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be nonnegative")
        if type(normalize) is not bool:
            raise ValueError("normalize must be a Boolean")
        if type(spatial_dims) is not int or spatial_dims < 1:
            raise ValueError("spatial_dims must be a positive integer")
        self.normalize = normalize
        self.spatial_dims = spatial_dims

    def _filter(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[-self.spatial_dims:]
        radius2 = torch.zeros(shape, dtype=x.dtype, device=x.device)
        for axis, size in enumerate(shape):
            frequency = torch.fft.fftfreq(size, dtype=x.dtype, device=x.device)
            view = [1] * self.spatial_dims
            view[axis] = size
            radius2 = radius2 + frequency.reshape(view).square()
        radius2 = radius2 + self.eps
        if not bool(torch.isfinite(radius2).all()) or not bool((radius2 > 0).all()):
            raise ValueError("eps is not representable for the input dtype")

        log_amplitude = (-self.beta / 4.0) * radius2.log()
        if not bool(torch.isfinite(log_amplitude).all()):
            raise ValueError("spectral exponent is not representable for the input dtype")
        if self.normalize:
            # The common scale cancels. Subtract first to avoid overflow before
            # computing the deterministic Gaussian-field variance normalization.
            amplitude = (log_amplitude - log_amplitude.max()).exp()
            amplitude = amplitude / amplitude.square().mean().sqrt()
        else:
            amplitude = log_amplitude.exp()
        if not bool(torch.isfinite(amplitude).all()) or not bool((amplitude > 0).any()):
            raise ValueError("spectral amplitude is not representable for the input dtype")
        return amplitude

    def forward(
        self,
        x: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.dtype not in (torch.float32, torch.float64):
            raise ValueError("x must be a float32 or float64 tensor")
        if x.ndim < self.spatial_dims or x.numel() == 0:
            raise ValueError("x must contain a nonempty field for every spatial axis")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("x must contain only finite values")
        if self.sigma == 0:
            return x

        # Validate the deterministic filter before consuming random state. Beta
        # zero is exactly white noise; avoid an unnecessary FFT round trip.
        amplitude = None if self.beta == 0 else self._filter(x)
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        if amplitude is not None:
            axes = tuple(range(x.ndim - self.spatial_dims, x.ndim))
            spectrum = torch.fft.fftn(noise, dim=axes, norm="ortho")
            noise = torch.fft.ifftn(spectrum * amplitude, dim=axes, norm="ortho").real
        result = x + self.sigma * noise
        if not bool(torch.isfinite(result).all()):
            raise ValueError("colored noise produced nonfinite output")
        return result


__all__ = ["ColoredNoiseV2"]
