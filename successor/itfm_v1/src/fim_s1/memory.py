from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Intervention = Literal["none", "erase", "random_value", "value_swap", "oracle_value"]


@dataclass(frozen=True)
class RetrievalResult:
    value: torch.Tensor
    valid: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor
    eligible_counts: torch.Tensor
    intervention: str


class TensorFabricMemory(nn.Module):
    """Fixed-capacity spatial key/value memory with fail-closed eligibility.

    Keys and values retain an arbitrary number of spatial axes. Entries are eligible only when
    they belong to the same trajectory and were completed strictly before the
    query time. Values are detached on write, so the bank is evidence-bearing
    state rather than an implicit backpropagation path through history.
    """

    _INTERVENTIONS = {"none", "erase", "random_value", "value_swap", "oracle_value"}

    def __init__(
        self,
        *,
        capacity: int,
        key_channels: int,
        value_channels: int,
        height: int | None = None,
        width: int | None = None,
        spatial_shape: tuple[int, ...] | None = None,
        topk: int = 4,
        temperature: float = 0.2,
        spatial_addressing: bool = True,
        spatial_values: bool = True,
    ) -> None:
        super().__init__()
        if spatial_shape is None:
            if height is None or width is None:
                raise ValueError("provide spatial_shape or both height and width")
            spatial_shape = (int(height), int(width))
        elif height is not None or width is not None:
            raise ValueError("spatial_shape cannot be combined with height or width")
        spatial_shape = tuple(int(size) for size in spatial_shape)
        if not spatial_shape:
            raise ValueError("spatial_shape must contain at least one axis")
        if min(capacity, key_channels, value_channels, topk, *spatial_shape) < 1:
            raise ValueError("all dimensions, capacity, and topk must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.capacity = int(capacity)
        self.key_channels = int(key_channels)
        self.value_channels = int(value_channels)
        self.spatial_shape = spatial_shape
        self.spatial_rank = len(spatial_shape)
        self.height = spatial_shape[0] if self.spatial_rank == 2 else None
        self.width = spatial_shape[1] if self.spatial_rank == 2 else None
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.spatial_addressing = bool(spatial_addressing)
        self.spatial_values = bool(spatial_values)

        self.register_buffer(
            "keys", torch.zeros(capacity, key_channels, *self.spatial_shape)
        )
        self.register_buffer(
            "values", torch.zeros(capacity, value_channels, *self.spatial_shape)
        )
        self.register_buffer("owners", torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer(
            "event_times", torch.full((capacity,), -1, dtype=torch.long)
        )
        self.register_buffer("regimes", torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer("valid_entries", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("write_ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("size", torch.zeros((), dtype=torch.long))

    def __len__(self) -> int:
        return int(self.size.item())

    def clear(self) -> None:
        self.keys.zero_()
        self.values.zero_()
        self.owners.fill_(-1)
        self.event_times.fill_(-1)
        self.regimes.fill_(-1)
        self.valid_entries.zero_()
        self.write_ptr.zero_()
        self.size.zero_()

    def _validate_field(self, tensor: torch.Tensor, channels: int, name: str) -> None:
        expected = (channels, *self.spatial_shape)
        if tensor.ndim != self.spatial_rank + 2 or tuple(tensor.shape[1:]) != expected:
            raise ValueError(
                f"{name} must have shape [B,{channels},{','.join(map(str, self.spatial_shape))}], "
                f"got {tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains non-finite values")

    def add(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        owners: torch.Tensor,
        event_times: torch.Tensor,
        regimes: torch.Tensor | None = None,
    ) -> None:
        self._validate_field(keys, self.key_channels, "keys")
        self._validate_field(values, self.value_channels, "values")
        if keys.device != self.keys.device or values.device != self.values.device:
            raise ValueError("keys and values must be on the memory device")
        if keys.shape[0] != values.shape[0]:
            raise ValueError("keys and values must have the same batch size")
        batch = keys.shape[0]
        owners = owners.detach().to(device=keys.device, dtype=torch.long).flatten()
        event_times = (
            event_times.detach().to(device=keys.device, dtype=torch.long).flatten()
        )
        if owners.numel() != batch or event_times.numel() != batch:
            raise ValueError("owners and event_times must contain one item per entry")
        if bool((owners < 0).any().item()) or bool((event_times < 0).any().item()):
            raise ValueError("owners and event_times must be non-negative")
        if regimes is None:
            regimes = torch.full_like(owners, -1)
        regimes = regimes.detach().to(device=keys.device, dtype=torch.long).flatten()
        if regimes.numel() != batch:
            raise ValueError("regimes must contain one item per entry")

        original_batch = batch
        keys = F.normalize(keys.detach(), dim=1, eps=1e-8)
        values = values.detach()
        skipped = max(0, batch - self.capacity)
        if skipped:
            keys = keys[skipped:]
            values = values[skipped:]
            owners = owners[skipped:]
            event_times = event_times[skipped:]
            regimes = regimes[skipped:]
            batch = self.capacity
        start = (int(self.write_ptr.item()) + skipped) % self.capacity
        indices = (
            torch.arange(batch, device=self.keys.device, dtype=torch.long) + start
        ) % self.capacity
        self.keys.index_copy_(0, indices, keys)
        self.values.index_copy_(0, indices, values)
        self.owners.index_copy_(0, indices, owners)
        self.event_times.index_copy_(0, indices, event_times)
        self.regimes.index_copy_(0, indices, regimes)
        self.valid_entries[indices] = True
        self.write_ptr.fill_(
            (int(self.write_ptr.item()) + original_batch) % self.capacity
        )
        self.size.fill_(min(self.capacity, len(self) + original_batch))

    def _gather_values(self, indices: torch.Tensor, count: int) -> torch.Tensor:
        spatial_axes = tuple(range(2, 2 + self.spatial_rank))
        point_count = math.prod(self.spatial_shape)
        values_by_point = self.values[:count].permute(
            *range(2, 2 + self.spatial_rank), 0, 1
        )
        values_by_point = values_by_point.reshape(
            point_count, count, self.value_channels
        )
        indices_by_point = indices.permute(*spatial_axes, 0, 1).reshape(
            point_count, indices.shape[0], indices.shape[1]
        )
        locations = torch.arange(point_count, device=indices.device).view(-1, 1, 1)
        gathered = values_by_point[locations, indices_by_point]
        gathered = gathered.permute(1, 2, 3, 0)
        return gathered.reshape(
            indices.shape[0], indices.shape[1], self.value_channels, *self.spatial_shape
        )

    def _randomize_indices(
        self, indices: torch.Tensor, eligible: torch.Tensor, count: int
    ) -> torch.Tensor:
        randomized = indices.clone()
        for batch_index in range(indices.shape[0]):
            allowed = torch.nonzero(
                eligible[batch_index, :count], as_tuple=False
            ).flatten()
            if allowed.numel() <= 1:
                continue
            mapping = torch.arange(count, device=indices.device)
            mapping[allowed] = torch.roll(allowed, shifts=1)
            randomized[batch_index] = mapping[indices[batch_index]]
        return randomized

    def retrieve(
        self,
        query: torch.Tensor,
        owners: torch.Tensor,
        query_times: torch.Tensor,
        *,
        intervention: Intervention = "none",
        replacement: torch.Tensor | None = None,
    ) -> RetrievalResult:
        self._validate_field(query, self.key_channels, "query")
        if intervention not in self._INTERVENTIONS:
            raise ValueError(f"unsupported intervention: {intervention}")
        batch = query.shape[0]
        owners = owners.to(device=query.device, dtype=torch.long).flatten()
        query_times = query_times.to(device=query.device, dtype=torch.long).flatten()
        if owners.numel() != batch or query_times.numel() != batch:
            raise ValueError("owners and query_times must contain one item per query")
        if bool((owners < 0).any().item()) or bool((query_times < 0).any().item()):
            raise ValueError("owners and query_times must be non-negative")

        empty_value = query.new_zeros((batch, self.value_channels, *self.spatial_shape))
        empty_indices = torch.full(
            (batch, 1, *self.spatial_shape),
            -1,
            device=query.device,
            dtype=torch.long,
        )
        empty_weights = query.new_zeros((batch, 1, *self.spatial_shape))
        if len(self) == 0:
            return RetrievalResult(
                empty_value,
                torch.zeros(batch, device=query.device, dtype=torch.bool),
                empty_indices,
                empty_weights,
                torch.zeros(batch, device=query.device, dtype=torch.long),
                intervention,
            )

        count = len(self)
        eligible = (
            self.valid_entries[:count].unsqueeze(0)
            & (owners.unsqueeze(1) == self.owners[:count].unsqueeze(0))
            & (self.event_times[:count].unsqueeze(0) < query_times.unsqueeze(1))
        )
        eligible_counts = eligible.sum(dim=1)
        valid = eligible_counts > 0

        if self.spatial_addressing:
            normalized_query = F.normalize(query, dim=1, eps=1e-8)
            similarity = (
                normalized_query.unsqueeze(1) * self.keys[:count].unsqueeze(0)
            ).sum(dim=2) / self.temperature
        else:
            spatial_dims = tuple(range(2, 2 + self.spatial_rank))
            pooled_query = F.normalize(query.mean(dim=spatial_dims), dim=1, eps=1e-8)
            pooled_keys = F.normalize(
                self.keys[:count].mean(dim=spatial_dims), dim=1, eps=1e-8
            )
            global_similarity = (
                torch.einsum("bk,nk->bn", pooled_query, pooled_keys) / self.temperature
            )
            similarity = global_similarity.view(
                batch, count, *([1] * self.spatial_rank)
            ).expand(batch, count, *self.spatial_shape)
        eligibility_shape = (batch, count, *([1] * self.spatial_rank))
        similarity = similarity.masked_fill(
            ~eligible.view(eligibility_shape), float("-inf")
        )
        similarity = torch.where(
            valid.view(batch, 1, *([1] * self.spatial_rank)),
            similarity,
            torch.zeros_like(similarity),
        )

        selected_k = min(self.topk, count)
        selected_scores, indices = torch.topk(similarity, k=selected_k, dim=1)
        selected_eligible = torch.gather(
            eligible.view(eligibility_shape).expand(batch, count, *self.spatial_shape),
            1,
            indices,
        )
        weights = torch.softmax(selected_scores, dim=1)
        weights = weights * selected_eligible.to(weights.dtype)
        normalizer = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights = torch.where(
            valid.view(batch, 1, *([1] * self.spatial_rank)),
            weights / normalizer,
            torch.zeros_like(weights),
        )

        value_indices = indices
        if intervention == "random_value":
            value_indices = self._randomize_indices(indices, eligible, count)
        gathered = self._gather_values(value_indices, count)
        if not self.spatial_values:
            gathered_spatial_dims = tuple(range(3, 3 + self.spatial_rank))
            gathered = gathered.mean(dim=gathered_spatial_dims, keepdim=True).expand_as(
                gathered
            )
        retrieved = (weights.unsqueeze(2) * gathered).sum(dim=1)

        if intervention == "random_value":
            # A one-entry history cannot be permuted. Negating that singleton
            # preserves its magnitude and compute path without leaking the true
            # residual through a nominal sham intervention.
            singleton = eligible_counts == 1
            retrieved = torch.where(
                singleton.view(batch, 1, *([1] * self.spatial_rank)),
                -retrieved,
                retrieved,
            )

        if intervention == "erase":
            retrieved = torch.zeros_like(retrieved)
        elif intervention in {"value_swap", "oracle_value"}:
            if replacement is None:
                raise ValueError(f"{intervention} requires a replacement tensor")
            self._validate_field(replacement, self.value_channels, "replacement")
            if replacement.shape[0] != batch:
                raise ValueError("replacement batch size must match query batch")
            retrieved = replacement
        elif replacement is not None:
            raise ValueError("replacement is only valid for value_swap or oracle_value")

        retrieved = torch.where(
            valid.view(batch, 1, *([1] * self.spatial_rank)),
            retrieved,
            torch.zeros_like(retrieved),
        )
        if not bool(torch.isfinite(retrieved).all().item()):
            raise RuntimeError("retrieval produced non-finite values")
        return RetrievalResult(
            retrieved,
            valid,
            indices,
            weights,
            eligible_counts,
            intervention,
        )
