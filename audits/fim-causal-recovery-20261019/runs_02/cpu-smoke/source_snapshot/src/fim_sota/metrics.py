from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch


def _sample_dimensions(value: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(1, value.ndim))


def relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("relative L2 requires paired [B,...] tensors")
    error = (prediction.float() - target.float()).square()
    denominator = target.float().square()
    if mask is not None:
        if mask.shape != target.shape:
            try:
                mask = torch.broadcast_to(mask, target.shape)
            except RuntimeError as error_value:
                raise ValueError(
                    "metric mask is not broadcast-compatible"
                ) from error_value
        weights = mask.to(dtype=error.dtype)
        error = error * weights
        denominator = denominator * weights
    dimensions = _sample_dimensions(target)
    numerator_sum = error.sum(dim=dimensions)
    denominator_sum = denominator.sum(dim=dimensions)
    return torch.sqrt(numerator_sum / denominator_sum.clamp_min(epsilon))


def mean_squared_error(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("MSE tensors must match")
    squared = (prediction.float() - target.float()).square()
    if mask is None:
        return squared.reshape(squared.shape[0], -1).mean(dim=1)
    weights = torch.broadcast_to(mask, squared.shape).float()
    dimensions = _sample_dimensions(squared)
    return (squared * weights).sum(dim=dimensions) / weights.sum(
        dim=dimensions
    ).clamp_min(1.0)


def spectral_relative_error(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim < 3:
        raise ValueError("spectral error requires paired spatial tensors")
    predicted_spectrum = torch.fft.rfftn(prediction.float(), dim=(-2, -1))
    target_spectrum = torch.fft.rfftn(target.float(), dim=(-2, -1))
    difference = torch.abs(predicted_spectrum - target_spectrum).square()
    denominator = torch.abs(target_spectrum).square()
    dimensions = _sample_dimensions(difference)
    return torch.sqrt(
        difference.sum(dim=dimensions)
        / denominator.sum(dim=dimensions).clamp_min(1e-12)
    )


def gaussian_calibration(
    prediction: torch.Tensor,
    target: torch.Tensor,
    standard_deviation: torch.Tensor,
) -> Mapping[str, float]:
    if prediction.shape != target.shape:
        raise ValueError("calibration prediction/target shape mismatch")
    scale = torch.broadcast_to(standard_deviation, target.shape).float().clamp_min(1e-8)
    standardized = (target.float() - prediction.float()) / scale
    nll = 0.5 * (
        standardized.square() + 2.0 * torch.log(scale) + math.log(2.0 * math.pi)
    )
    return {
        "gaussian_nll": float(nll.mean().item()),
        "coverage_68": float((standardized.abs() <= 1.0).float().mean().item()),
        "coverage_95": float((standardized.abs() <= 1.96).float().mean().item()),
        "standardized_rmse": float(standardized.square().mean().sqrt().item()),
    }


def horizon_conditioned_error(
    prediction: torch.Tensor, target: torch.Tensor
) -> list[float]:
    if prediction.shape != target.shape or prediction.ndim < 3:
        raise ValueError("rollout tensors must have shape [B,H,...]")
    return [
        float(relative_l2(prediction[:, horizon], target[:, horizon]).mean().item())
        for horizon in range(prediction.shape[1])
    ]


@dataclasses.dataclass(frozen=True)
class MetricRow:
    trajectory_id: str
    family_id: str
    seed: int
    cell_id: str
    event_time: int
    horizon: int
    query: bool
    full_relative_l2: float
    observed_relative_l2: float
    unobserved_relative_l2: float
    mse: float
    spectral_error: float
    physics_residual: float | None
    retrieved: bool
    similarity: float | None
    source_age: int | None


def metric_rows(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    observation_mask: torch.Tensor,
    trajectory_ids: Sequence[str],
    family_ids: Sequence[str],
    event_times: torch.Tensor,
    horizons: torch.Tensor,
    query_flags: torch.Tensor,
    seed: int,
    cell_id: str,
    retrieval_valid: torch.Tensor | None = None,
    similarities: torch.Tensor | None = None,
    source_times: torch.Tensor | None = None,
    physics_residual: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    | None = None,
) -> list[MetricRow]:
    batch = int(prediction.shape[0])
    if target.shape != prediction.shape or len(trajectory_ids) != batch:
        raise ValueError("metric batch shape mismatch")
    mask = torch.broadcast_to(observation_mask, target.shape).bool()
    full = relative_l2(prediction, target)
    observed = relative_l2(prediction, target, mask)
    unobserved = relative_l2(prediction, target, ~mask)
    mse = mean_squared_error(prediction, target)
    spectral = spectral_relative_error(prediction, target)
    physics = None if physics_residual is None else physics_residual(prediction, target)
    rows: list[MetricRow] = []
    for index in range(batch):
        retrieved = (
            bool(retrieval_valid[index].item())
            if retrieval_valid is not None
            else False
        )
        source_age = None
        if retrieved and source_times is not None:
            source_age = int(event_times[index].item() - source_times[index].item())
        rows.append(
            MetricRow(
                trajectory_id=str(trajectory_ids[index]),
                family_id=str(family_ids[index]),
                seed=int(seed),
                cell_id=str(cell_id),
                event_time=int(event_times[index].item()),
                horizon=int(horizons[index].item()),
                query=bool(query_flags[index].item()),
                full_relative_l2=float(full[index].item()),
                observed_relative_l2=float(observed[index].item()),
                unobserved_relative_l2=float(unobserved[index].item()),
                mse=float(mse[index].item()),
                spectral_error=float(spectral[index].item()),
                physics_residual=None
                if physics is None
                else float(physics[index].item()),
                retrieved=retrieved,
                similarity=(
                    float(similarities[index].item())
                    if retrieved and similarities is not None
                    else None
                ),
                source_age=source_age,
            )
        )
    return rows


def intervention_effects(
    summaries: Mapping[str, Mapping[str, float]], *, primary_metric: str
) -> Mapping[str, float]:
    required = {"no_memory", "tensor_memory", "erase", "paired_swap"}
    missing = sorted(required - set(summaries))
    if missing:
        raise ValueError(f"missing intervention summaries: {missing}")
    base = float(summaries["no_memory"][primary_metric])
    primary = float(summaries["tensor_memory"][primary_metric])
    erase = float(summaries["erase"][primary_metric])
    denominator = max(abs(base), 1e-12)
    return {
        "relative_gain": (base - primary) / denominator,
        "erasure_gain_removal": (erase - primary) / max(abs(base - primary), 1e-12),
        "paired_swap_metric": float(summaries["paired_swap"][primary_metric]),
        "nonquery_degradation": float(
            summaries["tensor_memory"].get("nonquery_error", float("nan"))
            - summaries["no_memory"].get("nonquery_error", float("nan"))
        ),
    }


@dataclasses.dataclass
class RuntimeMeter:
    device: torch.device
    started: float = dataclasses.field(init=False)
    finished: float | None = dataclasses.field(default=None, init=False)

    def __enter__(self) -> RuntimeMeter:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self.finished = time.perf_counter()

    def receipt(self, examples: int) -> Mapping[str, Any]:
        if self.finished is None:
            raise RuntimeError("runtime meter has not finished")
        elapsed = self.finished - self.started
        return {
            "elapsed_seconds": elapsed,
            "examples": int(examples),
            "throughput_examples_per_second": int(examples) / max(elapsed, 1e-12),
            "peak_accelerator_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda" and torch.cuda.is_available()
                else 0
            ),
            "energy_joules": None,
            "energy_status": "UNAVAILABLE_NO_ADMITTED_METER",
        }
