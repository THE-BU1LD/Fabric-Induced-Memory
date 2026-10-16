from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Literal

import torch
import torch.nn.functional as F

PublicIntervention = Literal[
    "none",
    "erase",
    "sign_flip",
    "spatial_shuffle",
    "channel_mean_broadcast",
]


@dataclasses.dataclass(frozen=True)
class PublicMemoryEntry:
    key: torch.Tensor
    residual: torch.Tensor
    trajectory_id: str
    event_time: int
    available_time: int


@dataclasses.dataclass(frozen=True)
class PublicRetrieval:
    residual: torch.Tensor
    valid: bool
    similarity: float
    source_event_time: int | None
    source_available_time: int | None
    eligible_count: int


class CausalResidualAdapter:
    """Backbone-agnostic, test-time-causal episodic residual memory.

    Keys may live in a latent space while residuals remain observation-space
    tensors. Retrieval is restricted to the same trajectory and to residuals
    whose targets were already observable at the query time.
    """

    def __init__(
        self,
        *,
        capacity_per_trajectory: int,
        minimum_age: int,
        topk: int,
        temperature: float,
        minimum_similarity: float = -1.0,
    ) -> None:
        if min(capacity_per_trajectory, minimum_age, topk) < 1:
            raise ValueError("capacity, minimum_age, and topk must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must lie in [-1, 1]")
        self.capacity_per_trajectory = int(capacity_per_trajectory)
        self.minimum_age = int(minimum_age)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.minimum_similarity = float(minimum_similarity)
        self._entries: list[PublicMemoryEntry] = []

    @property
    def entries(self) -> tuple[PublicMemoryEntry, ...]:
        return tuple(self._entries)

    def write(
        self,
        *,
        key: torch.Tensor,
        residual: torch.Tensor,
        trajectory_id: str,
        event_time: int,
        available_time: int,
    ) -> None:
        if key.numel() == 0 or residual.numel() == 0:
            raise ValueError("key and residual must be non-empty")
        if not bool(torch.isfinite(key).all().item()):
            raise ValueError("key must be finite")
        if not bool(torch.isfinite(residual).all().item()):
            raise ValueError("residual must be finite")
        if available_time < event_time:
            raise ValueError("a residual cannot be available before its event")
        entry = PublicMemoryEntry(
            key=key.detach().clone(),
            residual=residual.detach().clone(),
            trajectory_id=str(trajectory_id),
            event_time=int(event_time),
            available_time=int(available_time),
        )
        self._entries.append(entry)
        indices = [
            index
            for index, item in enumerate(self._entries)
            if item.trajectory_id == entry.trajectory_id
        ]
        if len(indices) > self.capacity_per_trajectory:
            del self._entries[indices[0]]

    def _eligible(
        self,
        *,
        trajectory_id: str,
        query_time: int,
        maximum_age: int | None,
    ) -> list[PublicMemoryEntry]:
        return [
            entry
            for entry in self._entries
            if entry.trajectory_id == trajectory_id
            and entry.event_time <= query_time - self.minimum_age
            and entry.available_time <= query_time
            and (
                maximum_age is None or query_time - entry.event_time <= int(maximum_age)
            )
        ]

    @staticmethod
    def _intervene(
        residual: torch.Tensor, intervention: PublicIntervention
    ) -> torch.Tensor:
        if intervention == "none":
            return residual
        if intervention == "erase":
            return torch.zeros_like(residual)
        if intervention == "sign_flip":
            return -residual
        if intervention == "spatial_shuffle":
            if residual.ndim < 2:
                return torch.flip(residual, dims=(0,))
            return torch.flip(residual, dims=(-2, -1))
        if intervention == "channel_mean_broadcast":
            if residual.ndim < 2:
                return residual.mean().expand_as(residual)
            pooled = residual.mean(dim=(-2, -1), keepdim=True)
            return pooled.expand_as(residual)
        raise ValueError(f"unsupported intervention: {intervention}")

    def retrieve(
        self,
        *,
        query_key: torch.Tensor,
        trajectory_id: str,
        query_time: int,
        residual_shape: tuple[int, ...],
        intervention: PublicIntervention = "none",
        maximum_age: int | None = None,
    ) -> PublicRetrieval:
        eligible = self._eligible(
            trajectory_id=str(trajectory_id),
            query_time=int(query_time),
            maximum_age=maximum_age,
        )
        if not eligible:
            return PublicRetrieval(
                residual=torch.zeros(residual_shape, dtype=query_key.dtype),
                valid=False,
                similarity=0.0,
                source_event_time=None,
                source_available_time=None,
                eligible_count=0,
            )
        if any(entry.key.numel() != query_key.numel() for entry in eligible):
            raise ValueError("query key size does not match eligible memory keys")
        if any(tuple(entry.residual.shape) != residual_shape for entry in eligible):
            raise ValueError("eligible residual shape mismatch")
        query = F.normalize(query_key.detach().flatten(), dim=0, eps=1e-8)
        scores = torch.stack(
            [
                torch.dot(
                    query,
                    F.normalize(
                        entry.key.to(query_key.device).flatten(), dim=0, eps=1e-8
                    ),
                )
                for entry in eligible
            ]
        )
        k = min(self.topk, len(eligible))
        similarities, indices = torch.topk(scores, k=k)
        best = eligible[int(indices[0])]
        best_similarity = float(similarities[0].item())
        if best_similarity < self.minimum_similarity:
            return PublicRetrieval(
                residual=torch.zeros(residual_shape, dtype=query_key.dtype),
                valid=False,
                similarity=best_similarity,
                source_event_time=None,
                source_available_time=None,
                eligible_count=len(eligible),
            )
        weights = torch.softmax(similarities / self.temperature, dim=0)
        retrieved = sum(
            weights[index] * eligible[int(entry_index)].residual.to(query_key.device)
            for index, entry_index in enumerate(indices)
        )
        return PublicRetrieval(
            residual=self._intervene(retrieved, intervention),
            valid=True,
            similarity=best_similarity,
            source_event_time=best.event_time,
            source_available_time=best.available_time,
            eligible_count=len(eligible),
        )


def fit_clipped_alpha(
    forecast_errors: Iterable[torch.Tensor],
    retrieved_residuals: Iterable[torch.Tensor],
    *,
    maximum: float = 1.0,
) -> float:
    """Fit a scalar residual coefficient on validation tensors only."""

    if maximum <= 0:
        raise ValueError("maximum must be positive")
    errors = tuple(forecast_errors)
    residuals = tuple(retrieved_residuals)
    if not errors or len(errors) != len(residuals):
        raise ValueError("paired non-empty validation tensors are required")
    if any(left.shape != right.shape for left, right in zip(errors, residuals)):
        raise ValueError("validation tensor shapes must match")
    numerator = sum(
        float(torch.sum(error.detach() * residual.detach()).item())
        for error, residual in zip(errors, residuals)
    )
    denominator = sum(
        float(torch.sum(residual.detach() ** 2).item()) for residual in residuals
    )
    if denominator <= 0:
        return 0.0
    return min(float(maximum), max(0.0, numerator / denominator))
