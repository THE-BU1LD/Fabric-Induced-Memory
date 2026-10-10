from __future__ import annotations

import math
import operator

import torch


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _fractional_parameters(alpha: float, dt: float = 1.0) -> tuple[float, float]:
    if isinstance(alpha, bool) or isinstance(dt, bool):
        raise TypeError("alpha and dt must be finite real numbers")
    alpha, dt = float(alpha), float(dt)
    if not math.isfinite(alpha) or not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be finite and in (0, 1]")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    return alpha, dt


def _finite_real_tensor(x: torch.Tensor, name: str) -> None:
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError(f"{name} must be a real floating-point tensor")
    if x.numel() == 0:
        raise ValueError(f"{name} must be nonempty")
    if not torch.isfinite(x).all():
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
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    alpha, _ = _fractional_parameters(alpha)
    order = _positive_integer(order, "order")

    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    if not dtype.is_floating_point:
        raise TypeError("weights require a real floating-point dtype")

    return _gl_weights(alpha, order, device, dtype)


def fractional_difference(
    history: torch.Tensor,
    alpha: float,
    dt: float = 1.0,
) -> torch.Tensor:
    """Finite-history GL difference; time is the penultimate axis.

    This uses the supplied history with zero extension before its first sample.
    It does not subtract the initial condition or implement a Caputo derivative.
    """
    _finite_real_tensor(history, "history")
    if history.ndim < 2:
        raise ValueError("history must include a time dimension")
    alpha, dt = _fractional_parameters(alpha, dt)

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

    result = torch.sum(flipped * weights, dim=-2)
    if not torch.isfinite(result).all():
        raise FloatingPointError("fractional difference produced non-finite values")
    return result


class FractionalMemory:
    """Detached, bounded GL history for feature vectors or spatial fields.

    Each pushed state retains its full shape. The inserted history axis is -2,
    independently of state rank. Starting a new trajectory requires ``reset``.
    """

    def __init__(
        self,
        alpha: float,
        max_history: int = 128,
        dt: float = 1.0,
    ):
        self.alpha, self.dt = _fractional_parameters(alpha, dt)
        self.max_history = _positive_integer(max_history, "max_history")

        self._history: torch.Tensor | None = None

    def push(self, x: torch.Tensor) -> torch.Tensor:
        _finite_real_tensor(x, "state")
        if x.ndim < 1:
            raise ValueError("state must include a feature dimension")

        if self._history is None:
            # detach alone aliases the caller's input storage.
            history = x.detach().clone().unsqueeze(-2)
        else:
            expected_shape = self._history.shape[:-2] + self._history.shape[-1:]
            if x.shape != expected_shape:
                raise ValueError("state shape changed; reset memory before a new trajectory")
            if x.device != self._history.device or x.dtype != self._history.dtype:
                raise ValueError("state dtype/device changed; reset memory before a new trajectory")
            history = torch.cat([self._history, x.detach().unsqueeze(-2)], dim=-2)

        if history.shape[-2] > self.max_history:
            history = history[..., -self.max_history:, :].contiguous()

        # An invalid or overflowing update must not poison the retained state.
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
