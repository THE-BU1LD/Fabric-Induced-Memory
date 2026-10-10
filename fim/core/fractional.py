from __future__ import annotations

import math
from numbers import Integral
from typing import Optional

import torch


def _validate_alpha(alpha: float) -> None:
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be finite and in (0, 1]")


def _validate_dt(dt: float) -> None:
    if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")


def _validate_tensor(name: str, value: torch.Tensor, min_ndim: int) -> None:
    if value.ndim < min_ndim or value.numel() == 0:
        raise ValueError(f"{name} must be nonempty with at least {min_ndim} dimensions")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a real floating dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _gl_weights(alpha: float, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    w = torch.empty(n, device=device, dtype=dtype)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = w[k - 1] * (-(alpha - (k - 1)) / k)
    return w


def grunwald_letnikov_weights(
    alpha: float,
    order: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    _validate_alpha(alpha)
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
        raise ValueError("order must be a positive integer")

    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    if not dtype.is_floating_point:
        raise ValueError("weights must use a real floating dtype")

    return _gl_weights(alpha, order, device, dtype)


def fractional_difference(
    history: torch.Tensor,
    alpha: float,
    dt: float = 1.0,
) -> torch.Tensor:
    """Return the finite-window GL difference with time on axis ``-2``.

    Observations are ordered oldest to newest. The omitted prehistory is zero;
    this is the GL finite difference, not an initial-value-corrected Caputo rule.
    """
    _validate_tensor("history", history, min_ndim=2)
    _validate_dt(dt)

    T = history.shape[-2]

    weights = grunwald_letnikov_weights(
        alpha,
        T,
        device=history.device,
        dtype=history.dtype,
    )

    scale = (dt ** (-alpha))
    weights = weights * scale

    flipped = torch.flip(history, dims=[-2])

    view_shape = [1] * history.ndim
    view_shape[-2] = T
    weights = weights.view(*view_shape)

    return torch.sum(flipped * weights, dim=-2)


class FractionalMemory:
    """Detached observation snapshots for a bounded, finite-window GL difference.

    A pushed observation has shape ``(..., features)``. The internal history
    inserts time immediately before the final feature axis. Reset the memory
    before starting an independent trajectory or changing shape, device or dtype.
    """

    def __init__(
        self,
        alpha: float,
        max_history: int = 128,
        dt: float = 1.0,
    ):
        _validate_alpha(alpha)
        _validate_dt(dt)
        if isinstance(max_history, bool) or not isinstance(max_history, Integral) or max_history < 1:
            raise ValueError("max_history must be a positive integer")

        self.alpha = float(alpha)
        self.max_history = int(max_history)
        self.dt = float(dt)

        self._history: Optional[torch.Tensor] = None

    def push(self, x: torch.Tensor) -> torch.Tensor:
        _validate_tensor("observation", x, min_ndim=1)
        if self._history is not None:
            expected_shape = self._history.shape[:-2] + self._history.shape[-1:]
            if x.shape != expected_shape:
                raise ValueError("observation shape changed; reset memory before a new trajectory")
            if x.dtype != self._history.dtype or x.device != self._history.device:
                raise ValueError("observation dtype or device changed; reset memory first")

        frame = x.detach().unsqueeze(-2)
        if self._history is None or self.max_history == 1:
            # detach alone aliases the caller's storage and can rewrite history.
            history = frame.clone()
        else:
            keep = min(self._history.shape[-2], self.max_history - 1)
            recent = self._history.narrow(-2, self._history.shape[-2] - keep, keep)
            # Concatenate only the admitted window, so no obsolete storage is kept.
            history = torch.cat([recent, frame], dim=-2)

        result = fractional_difference(history, self.alpha, self.dt)
        self._history = history
        return result

    def reset(self) -> None:
        self._history = None

    @property
    def history_length(self) -> int:
        if self._history is None:
            return 0
        return self._history.shape[-2]
