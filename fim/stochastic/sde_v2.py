"""Opt-in diagonal Itô integrators with explicit increment quadratic variation.

The maintained sde.SDEIntegrator remains the historical implementation. New
experiments must select this version and bind their configuration/source identity.
Milstein here requires componentwise diffusion driven by independent Brownian
increments; it does not implement cross-noise iterated integrals or Levy jumps.
"""
from __future__ import annotations

import math
import operator
from typing import Callable

import torch

from .sde import DiffusionFn, DiffusionGradFn, DriftFn, NoiseFn, SDEConfig

VarianceFn = Callable[[torch.Tensor, float], torch.Tensor | float]


def _step_count(value: int) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError
        count = operator.index(value)
    except TypeError as exc:
        raise ValueError('steps must be a nonnegative integer') from exc
    if count < 0:
        raise ValueError('steps must be a nonnegative integer')
    return count


class SDEIntegratorV2:
    """Euler--Maruyama and scalar/diagonal Milstein, preserving autograd.

    The default increment has variance ``noise_scale**2 * dt``. A custom
    ``noise_fn(x, dt)`` returns already-scaled increments; for Milstein it must
    additionally declare their conditional variance using ``noise_variance_fn``.
    Supplying variance does not make non-Brownian increments Milstein-compatible.
    Drift, diffusion and its componentwise derivative return tensors shaped as x.
    """

    version = 'diagonal-ito-v2'

    def __init__(self, config: SDEConfig, noise_fn: NoiseFn | None = None, *,
                 noise_variance_fn: VarianceFn | None = None,
                 generator: torch.Generator | None = None) -> None:
        if not math.isfinite(config.dt) or config.dt <= 0:
            raise ValueError('dt must be finite and positive')
        _step_count(config.steps)
        if not math.isfinite(config.noise_scale) or config.noise_scale < 0:
            raise ValueError('noise_scale must be finite and nonnegative')
        if config.stability_clip is not None and (
                not math.isfinite(config.stability_clip) or config.stability_clip <= 0):
            raise ValueError('stability_clip must be finite and positive or None')
        if noise_variance_fn is not None and noise_fn is None:
            raise ValueError('noise_variance_fn requires custom noise_fn')
        if generator is not None and noise_fn is not None:
            raise ValueError('generator applies only to the default Brownian noise')
        # Copy the scalar configuration so later caller mutation cannot change a run.
        self.config = SDEConfig(dt=float(config.dt), steps=_step_count(config.steps),
                                stability_clip=config.stability_clip,
                                noise_scale=float(config.noise_scale))
        self.noise_fn = noise_fn
        self.noise_variance_fn = noise_variance_fn
        self.generator = generator

    @staticmethod
    def _field(value: torch.Tensor, x: torch.Tensor, name: str) -> torch.Tensor:
        if (not isinstance(value, torch.Tensor) or value.shape != x.shape
                or value.device != x.device or value.dtype != x.dtype):
            raise ValueError(f'{name} must match state shape, device and dtype')
        if not torch.isfinite(value).all():
            raise ValueError(f'{name} must be finite')
        return value

    def _increment(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        if self.noise_fn is not None:
            return self._field(self.noise_fn(x, dt), x, 'noise increment')
        if self.config.noise_scale == 0:
            return torch.zeros_like(x)
        dw = torch.randn(x.shape, dtype=x.dtype, device=x.device,
                         generator=self.generator)
        return dw * (math.sqrt(dt) * self.config.noise_scale)

    def _variance(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        if self.noise_fn is None:
            # Scale before squaring, matching _increment even when sigma**2
            # alone would overflow but sigma**2 * dt remains representable.
            amplitude = self.config.noise_scale * math.sqrt(dt)
            value = amplitude * amplitude
        else:
            value = self.noise_variance_fn(x, dt)
        variance = torch.as_tensor(value, device=x.device, dtype=x.dtype)
        try:
            variance = torch.broadcast_to(variance, x.shape)
        except RuntimeError as exc:
            raise ValueError('noise variance must broadcast exactly to state shape') from exc
        if not torch.isfinite(variance).all() or torch.any(variance < 0):
            raise ValueError('noise variance must be finite and nonnegative')
        return variance

    def integrate(self, x0: torch.Tensor, drift: DriftFn, diffusion: DiffusionFn,
                  method: str = 'euler', diffusion_grad: DiffusionGradFn | None = None,
                  steps: int | None = None, t0: float = 0.0) -> torch.Tensor:
        if method not in {'euler', 'milstein'}:
            raise ValueError(f'Unknown method: {method}')
        count = self.config.steps if steps is None else _step_count(steps)
        if (not isinstance(x0, torch.Tensor) or x0.ndim < 1 or x0.numel() == 0
                or not x0.is_floating_point() or not torch.isfinite(x0).all()):
            raise ValueError('x0 must be a finite nonempty floating batch tensor')
        if not math.isfinite(t0):
            raise ValueError('t0 must be finite')
        if method == 'milstein' and diffusion_grad is None:
            raise ValueError('Milstein requires the componentwise diffusion derivative')
        if method == 'milstein' and self.noise_fn is not None and self.noise_variance_fn is None:
            raise ValueError('custom Milstein increments require noise_variance_fn')
        dt = self.config.dt
        states = [x0]
        x = x0
        for index in range(count):
            t = float(t0) + index * dt
            if not math.isfinite(t):
                raise ValueError('integration time exceeds finite range')
            f = self._field(drift(x, t), x, 'drift')
            g = self._field(diffusion(x, t), x, 'diffusion')
            dw = self._increment(x, dt)
            next_x = x + f * dt + g * dw
            if method == 'milstein':
                gp = self._field(diffusion_grad(x, t), x, 'diffusion derivative')
                # d[X] = variance(dw); dt alone is correct only for unit noise.
                next_x = next_x + 0.5 * g * gp * (dw.square() - self._variance(x, dt))
            if not torch.isfinite(next_x).all():
                raise ValueError(f'nonfinite state at integration step {index + 1}')
            if self.config.stability_clip is not None:
                next_x = next_x.clamp(-self.config.stability_clip, self.config.stability_clip)
            states.append(next_x)
            x = next_x
        return torch.stack(states, dim=1)

    def euler_maruyama(self, x0: torch.Tensor, drift: DriftFn, diffusion: DiffusionFn,
                       steps: int | None = None, t0: float = 0.0) -> torch.Tensor:
        return self.integrate(x0, drift, diffusion, 'euler', steps=steps, t0=t0)

    def milstein(self, x0: torch.Tensor, drift: DriftFn, diffusion: DiffusionFn,
                 diffusion_grad: DiffusionGradFn | None = None, steps: int | None = None,
                 t0: float = 0.0) -> torch.Tensor:
        return self.integrate(x0, drift, diffusion, 'milstein', diffusion_grad, steps, t0)
