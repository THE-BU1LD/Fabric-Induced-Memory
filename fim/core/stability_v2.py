"""Opt-in, scale-stable diagnostics; the retained stability module is unchanged.

All norms flatten a real floating-point field. Diagnostic functions do not retain
an autograd graph. ``clamp_norm_`` explicitly mutates a contiguous tensor under
no_grad, just as an optimizer-side clipping operation does. It is not a
substitute for a differentiable training loss. No scientific caller is rewired.
"""
from __future__ import annotations

import math
from numbers import Real
from collections.abc import Sequence

import torch


def _scalar(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _field(tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        raise ValueError("expected a materialized dense strided tensor")
    if not tensor.is_floating_point() or tensor.numel() == 0:
        raise ValueError("expected a nonempty real floating-point tensor")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("tensor contains non-finite values")


def _parts(tensor: torch.Tensor) -> tuple[float, float]:
    # Only the normalized array is squared: no raw large/small squared values.
    scale = float(tensor.abs().amax().item())
    if scale == 0:
        return 0.0, 0.0
    # float32 avoids half precision sum overflow; float64 preserves double input.
    dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
    normalized = tensor.to(dtype=dtype) / scale
    unit_norm = float(torch.sqrt(normalized.square().sum()).item())
    return scale, unit_norm


@torch.no_grad()
def log_norm(tensor: torch.Tensor) -> float:
    """Return log(||tensor||_2), including -inf for a valid all-zero field."""
    _field(tensor)
    scale, unit_norm = _parts(tensor)
    return -math.inf if scale == 0 else math.log(scale) + math.log(unit_norm)


@torch.no_grad()
def is_bounded(tensor: torch.Tensor, *, threshold: float) -> bool:
    """Check a finite nonnegative bound without forming an overflowing norm.

The comparison is floating-point, not a certified interval bound.
"""
    threshold = _scalar(threshold, "threshold")
    _field(tensor)
    scale, unit_norm = _parts(tensor)
    if scale == 0:
        return True
    if threshold == 0 or scale > threshold:
        return False
    return unit_norm <= threshold / scale


@torch.no_grad()
def clamp_norm_(tensor: torch.Tensor, max_norm: float, eps: float = 1e-12) -> torch.Tensor:
    """Apply the retained max_norm/(norm+eps) clipping rule stably, in place.

The result keeps direction up to rounding and final dtype underflow. Contiguous
storage is required so overlapping views cannot partially mutate shared state.
Invalid arguments are rejected before any write. The returned object is tensor.
"""
    max_norm = _scalar(max_norm, "max_norm")
    eps = _scalar(eps, "eps")
    _field(tensor)
    if not tensor.is_contiguous():
        raise ValueError("in-place clamp requires contiguous non-overlapping storage")
    scale, unit_norm = _parts(tensor)
    if scale == 0 or (scale <= max_norm and unit_norm <= max_norm / scale):
        return tensor
    if max_norm == 0:
        tensor.zero_()
        return tensor
    ln = math.log(scale) + math.log(unit_norm)
    # Equivalent norm/(norm+eps), but do not overflow either norm or eps/scale.
    damping = 1.0 if eps == 0 else math.exp(ln - _logadd(ln, math.log(eps)))
    dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
    # Normalize before rescaling: multiplying x by an underflowed ratio would
    # erase representable clipped values for very large dynamic-range changes.
    candidate = (tensor.to(dtype=dtype) / scale) * (max_norm / unit_norm)
    candidate = candidate * damping
    if not bool(torch.isfinite(candidate).all().item()):
        raise FloatingPointError("clipped result is not representable")
    tensor.copy_(candidate)
    return tensor


def _logadd(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = max(a, b), min(a, b)
    return hi + math.log1p(math.exp(lo - hi))


@torch.no_grad()
def estimate_growth_rate(sequence: Sequence[torch.Tensor], *, eps: float = 1e-12) -> float:
    """Average log((||x[t+1]||+eps)/(||x[t]||+eps)) without raw ratios.

This is growth per observation, not per unit physical time or a Lyapunov
exponent. eps is positive and has the same units as the state norm. Every
frame is checked, including interior frames that cancel in the algebraic sum.
"""
    eps = _scalar(eps, "eps", positive=True)
    if not isinstance(sequence, (list, tuple)):
        raise TypeError("sequence must be a list or tuple of tensors")
    logs: list[float] = []
    signature = None
    for tensor in sequence:
        _field(tensor)
        this = (tensor.shape, tensor.dtype, tensor.device)
        if signature is not None and this != signature:
            raise ValueError("all sequence frames must share shape, dtype and device")
        signature = this
        logs.append(_logadd(log_norm(tensor), math.log(eps)))
    if len(logs) < 2:
        return 0.0
    return math.fsum(b - a for a, b in zip(logs[:-1], logs[1:])) / (len(logs) - 1)
