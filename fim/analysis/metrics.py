from __future__ import annotations

import math
from numbers import Real

import torch


def _validate_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    """Reject ambiguous observations without changing admitted metric formulas."""
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("metric inputs must be torch.Tensor instances")
    if pred.shape != target.shape:
        raise ValueError("metric inputs must have the same shape; broadcasting is not allowed")
    if pred.numel() == 0:
        raise ValueError("metric inputs must be nonempty")
    if pred.is_complex() or target.is_complex():
        raise ValueError("metric inputs must be real")
    if not torch.isfinite(pred).all().item() or not torch.isfinite(target).all().item():
        raise ValueError("metric inputs must contain only finite values")


def _validate_eps(eps: float) -> None:
    if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive real number")


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _validate_pair(pred, target)
    return torch.mean((pred - target) ** 2)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _validate_pair(pred, target)
    return torch.mean(torch.abs(pred - target))


def normalized_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    _validate_eps(eps)
    error = mse(pred, target)
    denom = torch.mean(target ** 2).clamp_min(eps)
    return error / denom


def rare_event_recall(pred_events: torch.Tensor, true_events: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Recall on binary event masks."""
    _validate_pair(pred_events, true_events)
    _validate_eps(eps)
    tp = torch.sum((pred_events > 0) & (true_events > 0)).float()
    fn = torch.sum((pred_events <= 0) & (true_events > 0)).float()
    return tp / (tp + fn).clamp_min(eps)


def rare_event_precision(pred_events: torch.Tensor, true_events: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    _validate_pair(pred_events, true_events)
    _validate_eps(eps)
    tp = torch.sum((pred_events > 0) & (true_events > 0)).float()
    fp = torch.sum((pred_events > 0) & (true_events <= 0)).float()
    return tp / (tp + fp).clamp_min(eps)


def memory_efficiency_score(recall: float, stored_traces: int) -> float:
    """A compactness-aware score in [0, 1+] depending on recall and storage."""
    return float(recall) / (1.0 + math.log1p(max(int(stored_traces), 0)))
