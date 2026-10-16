from __future__ import annotations

import dataclasses
from typing import Literal

import torch
import torch.nn.functional as F

Intervention = Literal["none", "erase", "sign_flip", "vector_broadcast"]


@dataclasses.dataclass(frozen=True)
class MemoryEntry:
    key: torch.Tensor
    value: torch.Tensor
    owner_id: int
    event_time: int
    anchor_id: int


@dataclasses.dataclass(frozen=True)
class Retrieval:
    value: torch.Tensor
    valid: bool
    similarity: float
    source_time: int | None
    source_anchor: int | None
    eligible_count: int


class TensorResidualMemory:
    """Bounded exact memory for detached spatial residual tensors."""

    def __init__(
        self,
        *,
        capacity: int = 128,
        minimum_age: int = 1,
        topk: int = 1,
        temperature: float = 0.05,
        minimum_similarity: float = -1.0,
    ) -> None:
        if min(capacity, minimum_age, topk) < 1 or temperature <= 0:
            raise ValueError("invalid memory configuration")
        self.capacity = int(capacity)
        self.minimum_age = int(minimum_age)
        self.topk = int(topk)
        self.temperature = float(temperature)
        if not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must lie in [-1,1]")
        self.minimum_similarity = float(minimum_similarity)
        self._entries: list[MemoryEntry] = []

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def add(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        owner_id: int,
        event_time: int,
        anchor_id: int,
    ) -> None:
        if key.ndim != 3 or value.ndim != 3:
            raise ValueError("key and value must have shape [C,H,W]")
        if key.shape[-2:] != value.shape[-2:]:
            raise ValueError("key and value spatial shapes must match")
        if not bool(torch.isfinite(key).all().item()) or not bool(
            torch.isfinite(value).all().item()
        ):
            raise ValueError("memory entries must be finite")
        self._entries.append(
            MemoryEntry(
                key.detach().clone(),
                value.detach().clone(),
                int(owner_id),
                int(event_time),
                int(anchor_id),
            )
        )
        owner_indices = [
            index
            for index, entry in enumerate(self._entries)
            if entry.owner_id == int(owner_id)
        ]
        if len(owner_indices) > self.capacity:
            del self._entries[owner_indices[0]]

    def _eligible(
        self,
        *,
        owner_id: int,
        query_time: int,
        maximum_age: int | None,
    ) -> list[MemoryEntry]:
        rows = [
            entry
            for entry in self._entries
            if entry.owner_id == owner_id
            and entry.event_time <= query_time - self.minimum_age
            and (maximum_age is None or query_time - entry.event_time <= maximum_age)
        ]
        return rows

    def retrieve(
        self,
        query: torch.Tensor,
        *,
        owner_id: int,
        query_time: int,
        intervention: Intervention = "none",
        maximum_age: int | None = None,
    ) -> Retrieval:
        if query.ndim != 3:
            raise ValueError("query must have shape [C,H,W]")
        if intervention not in {"none", "erase", "sign_flip", "vector_broadcast"}:
            raise ValueError(f"unsupported intervention: {intervention}")
        eligible = self._eligible(
            owner_id=owner_id, query_time=query_time, maximum_age=maximum_age
        )
        if not eligible:
            return Retrieval(
                torch.zeros(
                    1,
                    query.shape[-2],
                    query.shape[-1],
                    dtype=query.dtype,
                    device=query.device,
                ),
                False,
                0.0,
                None,
                None,
                0,
            )
        normalized_query = F.normalize(query.flatten(), dim=0, eps=1e-8)
        scores = torch.stack(
            [
                torch.dot(
                    normalized_query,
                    F.normalize(entry.key.to(query.device).flatten(), dim=0, eps=1e-8),
                )
                for entry in eligible
            ]
        )
        k = min(self.topk, len(eligible))
        values, indices = torch.topk(scores, k=k)
        if float(values[0].item()) < self.minimum_similarity:
            return Retrieval(
                torch.zeros_like(eligible[0].value).to(query.device),
                False,
                float(values[0].item()),
                None,
                None,
                len(eligible),
            )
        weights = torch.softmax(values / self.temperature, dim=0)
        retrieved = sum(
            weights[index] * eligible[int(entry_index)].value.to(query.device)
            for index, entry_index in enumerate(indices)
        )
        best = eligible[int(indices[0])]
        if intervention == "erase":
            retrieved = torch.zeros_like(retrieved)
        elif intervention == "sign_flip":
            retrieved = -retrieved
        elif intervention == "vector_broadcast":
            pooled = retrieved.mean(dim=(-2, -1), keepdim=True)
            retrieved = pooled.expand_as(retrieved)
        return Retrieval(
            retrieved,
            True,
            float(values[0].item()),
            best.event_time,
            best.anchor_id,
            len(eligible),
        )

    def last_value(
        self, *, owner_id: int, query_time: int, shape: tuple[int, int, int]
    ) -> Retrieval:
        eligible = self._eligible(
            owner_id=owner_id, query_time=query_time, maximum_age=None
        )
        if not eligible:
            return Retrieval(torch.zeros(shape), False, 0.0, None, None, 0)
        entry = max(eligible, key=lambda item: item.event_time)
        return Retrieval(
            entry.value.clone(),
            True,
            1.0,
            entry.event_time,
            entry.anchor_id,
            len(eligible),
        )
