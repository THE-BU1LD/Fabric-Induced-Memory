from __future__ import annotations

import math
import operator
from typing import Optional

import torch


def _positive_count(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if count < 1:
        raise ValueError(f"{name} must be a positive integer")
    return count


def _time_step(dt: float) -> float:
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    return dt


def _finite_history(history: torch.Tensor) -> None:
    if not torch.is_tensor(history):
        raise TypeError("history must be a tensor")
    if history.ndim < 2 or history.numel() == 0:
        raise ValueError("history must be nonempty and include a time dimension")
    if not (history.is_floating_point() or history.is_complex()):
        raise TypeError("history must have a floating or complex dtype")
    if not torch.isfinite(history).all():
        raise FloatingPointError("history contains non-finite values")


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
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    order = _positive_count(order, "order")

    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    if not (dtype.is_floating_point or dtype.is_complex):
        raise TypeError("weights require a floating or complex dtype")

    return _gl_weights(alpha, order, device, dtype)


def fractional_difference(
    history: torch.Tensor,
    alpha: float,
    dt: float = 1.0,
) -> torch.Tensor:
    """Finite-history GL difference, with time on axis -2 and newest state last.

    This is the truncated Grunwald--Letnikov formula, without a Caputo initial
    value correction. A constant finite history need not have zero difference.
    Unlike ``FractionalMemory``, this function retains the input gradient graph.
    """
    _finite_history(history)
    dt = _time_step(dt)

    T = history.shape[-2]

    weights = grunwald_letnikov_weights(
        alpha,
        T,
        device=history.device,
        dtype=history.dtype,
    )

    scale = dt ** (-alpha)
    weights = weights * scale

    flipped = torch.flip(history, dims=[-2])

    view_shape = [1] * history.ndim
    view_shape[-2] = T
    weights = weights.view(*view_shape)

    result = torch.sum(flipped * weights, dim=-2)
    if not torch.isfinite(result).all():
        raise FloatingPointError("fractional difference contains non-finite values")
    return result


class FractionalMemory:
    """Detached rolling GL history for states of shape ``(..., features)``.

    Each push snapshots the supplied state. Shape, dtype and device are fixed
    within one episode; call ``reset`` before starting an incompatible episode.
    The history is only committed after the new difference is valid.
    """

    def __init__(
        self,
        alpha: float,
        max_history: int = 128,
        dt: float = 1.0,
    ):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

        self.alpha = float(alpha)
        self.max_history = _positive_count(max_history, "max_history")
        self.dt = _time_step(dt)

        self._history: Optional[torch.Tensor] = None

    def push(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError("state must be a tensor")
        if x.ndim < 1:
            raise ValueError("state must include a feature dimension")
        sample = x.detach().unsqueeze(-2)
        _finite_history(sample)

        if self._history is None:
            history = sample.clone()
        else:
            expected = self._history.shape[:-2] + self._history.shape[-1:]
            if x.shape != expected:
                raise ValueError("state shape changed; reset memory for a new episode")
            if x.dtype != self._history.dtype or x.device != self._history.device:
                raise ValueError("state dtype/device changed; reset memory for a new episode")
            # Slice the declared temporal axis for every supported state rank.
            # Trimming before concatenation also bounds the backing allocation.
            keep = self.max_history - 1
            if keep:
                history = torch.cat([self._history[..., -keep:, :], sample], dim=-2)
            else:
                history = sample.clone()

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
