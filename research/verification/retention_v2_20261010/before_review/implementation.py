"""Opt-in retention analysis with explicit observations and float64 arithmetic.

Historical ``fim.analysis.retention`` functions and frozen callers stay intact.
No observations are clipped, reordered, discarded or replaced by this module.
"""
from __future__ import annotations

import math

import torch


def _vector(name: str, value: torch.Tensor, minimum: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must have float32 or float64 dtype")
    if value.numel() < minimum:
        raise ValueError(f"{name} requires at least {minimum} observations")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value.to(dtype=torch.float64)


def _observations(times: torch.Tensor, response: torch.Tensor):
    x = _vector("times", times, 2)
    y = _vector("response", response, 2)
    if x.shape != y.shape or x.device != y.device:
        raise ValueError("times and response must have identical shapes and devices")
    if not torch.all(x[1:] > x[:-1]):
        raise ValueError("times must be strictly increasing")
    return x, y


def _finite_float(value: torch.Tensor, quantity: str) -> float:
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError(f"{quantity} is not representable as a finite float64")
    return result


def _negative_slope(x: torch.Tensor, y: torch.Tensor) -> float:
    # Remove the irrelevant time origin before centering; scaling avoids
    # squaring tiny/large time intervals and never floors their actual variance.
    x = x - x[0]
    if not torch.isfinite(x).all():
        raise ValueError("time span is not representable as a finite float64")
    scale = x.abs().max()
    if scale <= 0:
        raise ValueError("transformed times must be distinct")
    scaled = x / scale
    centered_x = scaled - scaled.mean()
    centered_y = y - y.mean()
    denominator = centered_x.square().sum()
    if denominator <= 0:
        raise ValueError("transformed times have zero variance")
    slope = (centered_x * centered_y).sum() / denominator / scale
    return _finite_float(-slope, "decay rate")


def decay_curve(response: torch.Tensor) -> torch.Tensor:
    """Return the signed response divided by its nonzero initial value.

    Output is float64. Zero initial values are undefined and rejected; positive,
    negative and zero later values remain unaltered apart from normalization.
    """
    y = _vector("response", response, 1)
    if y[0] == 0:
        raise ValueError("initial response must be nonzero")
    result = y / y[0]
    if not torch.isfinite(result).all():
        raise ValueError("normalized response is not finite in float64")
    return result


def estimate_exponential_decay(times: torch.Tensor, response: torch.Tensor) -> float:
    """OLS rate in log(response) = intercept - rate * time, per time unit.

    Every response must be strictly positive. Increasing responses yield a
    negative rate; no decay assumption is imposed on the fitted coefficient.
    """
    x, y = _observations(times, response)
    if not torch.all(y > 0):
        raise ValueError("logarithmic fitting requires strictly positive responses")
    return _negative_slope(x, torch.log(y))


def estimate_power_law_decay(times: torch.Tensor, response: torch.Tensor) -> float:
    """OLS exponent in log(response) = intercept - exponent * log(time).

    Times and responses must be strictly positive. Time origins are scientifically
    meaningful for this model and are never shifted before taking logarithms.
    """
    x, y = _observations(times, response)
    if not torch.all(x > 0) or not torch.all(y > 0):
        raise ValueError("power-law fitting requires positive times and responses")
    # log1p retains close timestamps; log differences handle a ratio too large
    # to represent. Both are log(t/t0), which only changes the fitted intercept.
    ratio = (x - x[0]) / x[0]
    logged_x = torch.where(
        torch.isfinite(ratio), torch.log1p(ratio), torch.log(x) - torch.log(x[0])
    )
    return _negative_slope(logged_x, torch.log(y))


def retention_auc(times: torch.Tensor, response: torch.Tensor) -> float:
    """Signed trapezoidal integral on strictly increasing observed times.

    The result uses the supplied time and response units; it is not implicitly
    normalized. At least two observations are required.
    """
    x, y = _observations(times, response)
    gaps = x[1:] - x[:-1]
    if not torch.isfinite(gaps).all():
        raise ValueError("time intervals are not representable as finite float64")
    midpoint_response = 0.5 * y[:-1] + 0.5 * y[1:]
    return _finite_float((gaps * midpoint_response).sum(), "retention AUC")


__all__ = [
    "decay_curve", "estimate_exponential_decay", "estimate_power_law_decay", "retention_auc"
]
