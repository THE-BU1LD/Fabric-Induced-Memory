from __future__ import annotations

import dataclasses
import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

EvictionPolicy = Literal["fifo", "reservoir", "diversity", "utility"]
SelectionControl = Literal[
    "cosine",
    "recent_only",
    "last_residual",
    "wrong_trajectory",
    "time_permuted",
    "random_key",
    "random_value",
    "paired_swap",
    "oracle_past_residual",
]
ValueIntervention = Literal[
    "none", "erase", "sign_flip", "spatial_shuffle", "channel_mean_broadcast"
]


@dataclasses.dataclass(frozen=True)
class BatchRetrieval:
    residual: torch.Tensor
    valid: torch.Tensor
    similarity: torch.Tensor
    source_event_time: torch.Tensor
    source_available_time: torch.Tensor
    source_serial: torch.Tensor
    eligible_count: torch.Tensor
    latency_seconds: float
    peak_accelerator_bytes: int
    # source_* identifies the value paired with the strongest selected key.
    # Keep key identity separate when a causal value permutation is requested.
    key_source_serial: torch.Tensor | None = None
    value_source_serials: torch.Tensor | None = None
    value_source_event_times: torch.Tensor | None = None
    value_source_available_times: torch.Tensor | None = None
    value_weights: torch.Tensor | None = None

    def telemetry(self) -> dict[str, object]:
        """Return target-free retrieval telemetry suitable for a run receipt."""

        valid = self.valid.detach().cpu()
        similarities = self.similarity.detach().cpu()[valid]
        source_age = (
            self.source_available_time.detach().cpu()[valid]
            - self.source_event_time.detach().cpu()[valid]
        )
        return {
            "queries": int(self.valid.numel()),
            "retrieved": int(valid.sum().item()),
            "retrieval_rate": float(valid.float().mean().item()),
            "mean_similarity": (
                float(similarities.mean().item()) if similarities.numel() else None
            ),
            "minimum_similarity": (
                float(similarities.min().item()) if similarities.numel() else None
            ),
            "mean_source_availability_lag": (
                float(source_age.float().mean().item()) if source_age.numel() else None
            ),
            "latency_seconds": self.latency_seconds,
            "peak_accelerator_bytes": self.peak_accelerator_bytes,
        }


def _as_long(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Reject ambiguous time values instead of truncating observation arrivals."""
    values = values.detach()
    if values.ndim != 1 or values.dtype == torch.bool or values.is_complex():
        raise ValueError("memory times must be integer-valued tensors with shape [B]")
    if values.is_floating_point():
        admitted = (
            torch.isfinite(values)
            & (values == values.trunc())
            & (values >= -(2**63))
            & (values < 2**63)
        )
        if not bool(admitted.all().item()):
            raise ValueError("memory times must be finite int64-range integer values")
    elif values.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("memory times require a supported integer dtype")
    return values.to(device=device, dtype=torch.long)


def masked_cosine_scores(
    queries: torch.Tensor,
    entries: torch.Tensor,
    *,
    query_masks: torch.Tensor | None = None,
    entry_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise mixed-precision-safe cosine similarity.

    Similarities are always accumulated in float32, even under autocast.  A
    feature contributes only when both the query and stored key mark it valid.
    """

    if queries.ndim != 2 or entries.ndim != 2 or queries.shape[1] != entries.shape[1]:
        raise ValueError("queries and entries must have shapes [B,D] and [N,D]")
    q = queries.detach().float()
    e = entries.detach().float()
    if query_masks is None and entry_masks is None:
        return F.normalize(q, dim=1, eps=1e-8) @ F.normalize(
            e, dim=1, eps=1e-8
        ).transpose(0, 1)
    q_mask = (
        torch.ones_like(q, dtype=torch.bool)
        if query_masks is None
        else query_masks.detach().to(device=q.device, dtype=torch.bool)
    )
    e_mask = (
        torch.ones_like(e, dtype=torch.bool)
        if entry_masks is None
        else entry_masks.detach().to(device=e.device, dtype=torch.bool)
    )
    if q_mask.shape != q.shape or e_mask.shape != e.shape:
        raise ValueError("key masks must match their corresponding keys")
    combined = q_mask[:, None, :] & e_mask[None, :, :]
    product = (q[:, None, :] * e[None, :, :] * combined).sum(dim=-1)
    q_norm = torch.sqrt((q[:, None, :].square() * combined).sum(dim=-1))
    e_norm = torch.sqrt((e[None, :, :].square() * combined).sum(dim=-1))
    return product / (q_norm * e_norm).clamp_min(1e-8)


def _deterministic_topk(
    scores: torch.Tensor, serials: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort by descending score, then ascending immutable serial."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [B,N]")
    serial_order = torch.argsort(serials, stable=True)
    ordered = scores[:, serial_order]
    score_order = torch.argsort(ordered, dim=1, descending=True, stable=True)
    chosen = score_order[:, : min(k, scores.shape[1])]
    indices = serial_order[chosen]
    values = torch.gather(scores, 1, indices)
    return values, indices


def _deterministic_candidate_topk(
    scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    global_serials: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k over per-query candidate lists using global serial tie breaks."""

    serial_values = global_serials[candidate_indices]
    serial_order = torch.argsort(serial_values, dim=1, stable=True)
    ordered_scores = torch.gather(scores, 1, serial_order)
    ordered_indices = torch.gather(candidate_indices, 1, serial_order)
    score_order = torch.argsort(ordered_scores, dim=1, descending=True, stable=True)[
        :, : min(k, scores.shape[1])
    ]
    return (
        torch.gather(ordered_scores, 1, score_order),
        torch.gather(ordered_indices, 1, score_order),
    )


class TensorMemoryBank:
    """GPU-native, causally filtered tensor-residual memory.

    Writes are detached and immutable.  Retrieval is vectorized across both the
    query batch and each candidate chunk.  Python iteration occurs only across
    bounded chunks or low-frequency eviction maintenance, never per candidate.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        key_dimension: int,
        residual_shape: tuple[int, ...],
        capacity_per_trajectory: int,
        minimum_age: int,
        topk: int,
        temperature: float,
        minimum_similarity: float = -1.0,
        chunk_size: int = 16384,
        eviction: EvictionPolicy = "fifo",
        device: str | torch.device = "cpu",
        storage_dtype: torch.dtype = torch.float32,
        seed: int = 0,
        ann_bits: int = 32,
    ) -> None:
        if (
            min(
                key_dimension,
                capacity_per_trajectory,
                minimum_age,
                topk,
                chunk_size,
            )
            < 1
        ):
            raise ValueError("memory dimensions and counts must be positive")
        if not residual_shape or min(residual_shape) < 1:
            raise ValueError("residual_shape must be non-empty and positive")
        if temperature <= 0 or not -1 <= minimum_similarity <= 1:
            raise ValueError("invalid temperature or minimum similarity")
        if eviction not in {"fifo", "reservoir", "diversity", "utility"}:
            raise ValueError(f"unsupported eviction policy: {eviction}")
        if not 4 <= ann_bits <= 64:
            raise ValueError("ann_bits must lie in [4,64]")
        self.key_dimension = int(key_dimension)
        self.residual_shape = tuple(int(value) for value in residual_shape)
        self.capacity_per_trajectory = int(capacity_per_trajectory)
        self.minimum_age = int(minimum_age)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.minimum_similarity = float(minimum_similarity)
        self.chunk_size = int(chunk_size)
        self.eviction = eviction
        self.device = torch.device(device)
        self.storage_dtype = storage_dtype
        self.seed = int(seed)
        self.ann_bits = int(ann_bits)
        self._trajectory_to_code: dict[str, int] = {}
        self._serial_counter = 0
        self._generator = torch.Generator(device="cpu").manual_seed(self.seed)
        projection = torch.randn(
            self.key_dimension, self.ann_bits, generator=self._generator
        )
        self._ann_projection = F.normalize(projection, dim=0).to(self.device)
        self._reset_tensors()

    def _reset_tensors(self) -> None:
        self.keys = torch.empty(
            (0, self.key_dimension), device=self.device, dtype=self.storage_dtype
        )
        self.key_masks = torch.empty(
            (0, self.key_dimension), device=self.device, dtype=torch.bool
        )
        self.residuals = torch.empty(
            (0, *self.residual_shape), device=self.device, dtype=self.storage_dtype
        )
        self.trajectory_codes = torch.empty(0, device=self.device, dtype=torch.long)
        self.event_times = torch.empty(0, device=self.device, dtype=torch.long)
        self.available_times = torch.empty(0, device=self.device, dtype=torch.long)
        self.utilities = torch.empty(0, device=self.device, dtype=torch.float32)
        self.serials = torch.empty(0, device=self.device, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    @property
    def memory_bytes(self) -> int:
        tensors = (
            self.keys,
            self.key_masks,
            self.residuals,
            self.trajectory_codes,
            self.event_times,
            self.available_times,
            self.utilities,
            self.serials,
            self._ann_projection,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def clear(self) -> None:
        self._trajectory_to_code.clear()
        self._serial_counter = 0
        self._reset_tensors()

    def _codes(self, trajectory_ids: tuple[str, ...], *, create: bool) -> torch.Tensor:
        codes: list[int] = []
        for identity in trajectory_ids:
            normalized = str(identity)
            if normalized not in self._trajectory_to_code:
                if not create:
                    codes.append(-1)
                    continue
                self._trajectory_to_code[normalized] = len(self._trajectory_to_code)
            codes.append(self._trajectory_to_code[normalized])
        return torch.tensor(codes, device=self.device, dtype=torch.long)

    def write_batch(
        self,
        *,
        keys: torch.Tensor,
        residuals: torch.Tensor,
        trajectory_ids: tuple[str, ...],
        event_times: torch.Tensor,
        available_times: torch.Tensor,
        key_masks: torch.Tensor | None = None,
        utilities: torch.Tensor | None = None,
    ) -> None:
        batch = int(keys.shape[0])
        if keys.shape != (batch, self.key_dimension):
            raise ValueError("keys must have shape [B,key_dimension]")
        if residuals.shape != (batch, *self.residual_shape):
            raise ValueError("residuals do not match configured residual_shape")
        if len(trajectory_ids) != batch:
            raise ValueError("trajectory_ids batch mismatch")
        event = _as_long(event_times, self.device)
        available = _as_long(available_times, self.device)
        if event.shape != (batch,) or available.shape != (batch,):
            raise ValueError("event and availability times must have shape [B]")
        if bool((available < event).any().item()):
            raise ValueError("a residual cannot be available before its event")
        key_values = keys.detach().to(self.device, self.storage_dtype)
        residual_values = residuals.detach().to(self.device, self.storage_dtype)
        if not bool(torch.isfinite(key_values).all().item()) or not bool(
            torch.isfinite(residual_values).all().item()
        ):
            raise ValueError("memory writes must be finite")
        masks = (
            torch.ones_like(key_values, dtype=torch.bool)
            if key_masks is None
            else key_masks.detach().to(self.device, torch.bool)
        )
        if masks.shape != key_values.shape or not bool(masks.any(dim=1).all().item()):
            raise ValueError("each key mask must expose at least one feature")
        utility_values = (
            torch.zeros(batch, device=self.device, dtype=torch.float32)
            if utilities is None
            else utilities.detach().to(self.device, torch.float32).reshape(-1)
        )
        if utility_values.shape != (batch,) or not bool(
            torch.isfinite(utility_values).all().item()
        ):
            raise ValueError("utilities must be finite with shape [B]")
        codes = self._codes(trajectory_ids, create=True)
        serial = torch.arange(
            self._serial_counter,
            self._serial_counter + batch,
            device=self.device,
            dtype=torch.long,
        )
        self._serial_counter += batch
        self.keys = torch.cat((self.keys, key_values), dim=0)
        self.key_masks = torch.cat((self.key_masks, masks), dim=0)
        self.residuals = torch.cat((self.residuals, residual_values), dim=0)
        self.trajectory_codes = torch.cat((self.trajectory_codes, codes), dim=0)
        self.event_times = torch.cat((self.event_times, event), dim=0)
        self.available_times = torch.cat((self.available_times, available), dim=0)
        self.utilities = torch.cat((self.utilities, utility_values), dim=0)
        self.serials = torch.cat((self.serials, serial), dim=0)
        self._enforce_capacity(torch.unique(codes))

    def _diverse_subset(self, indices: torch.Tensor, count: int) -> torch.Tensor:
        if indices.numel() <= count:
            return indices
        vectors = F.normalize(self.keys[indices].float(), dim=1, eps=1e-8)
        first = int(torch.argmin(self.serials[indices]).item())
        chosen = [first]
        minimum_distance = torch.full(
            (indices.numel(),), float("inf"), device=self.device
        )
        for _ in range(1, count):
            similarity = vectors @ vectors[chosen[-1]]
            minimum_distance = torch.minimum(minimum_distance, 1.0 - similarity)
            minimum_distance[torch.tensor(chosen, device=self.device)] = -1.0
            chosen.append(int(torch.argmax(minimum_distance).item()))
        return indices[torch.tensor(chosen, device=self.device)]

    def _enforce_capacity(self, affected_codes: torch.Tensor) -> None:
        keep = torch.ones(len(self), device=self.device, dtype=torch.bool)
        for code in affected_codes.detach().cpu().tolist():
            indices = torch.nonzero(self.trajectory_codes == int(code)).flatten()
            excess = int(indices.numel()) - self.capacity_per_trajectory
            if excess <= 0:
                continue
            if self.eviction == "fifo":
                order = torch.argsort(self.serials[indices], stable=True)
                selected = indices[order[-self.capacity_per_trajectory :]]
            elif self.eviction == "reservoir":
                permutation = torch.randperm(indices.numel(), generator=self._generator)
                selected = indices[
                    permutation[: self.capacity_per_trajectory].to(indices.device)
                ]
            elif self.eviction == "utility":
                serial_order = torch.argsort(self.serials[indices], stable=True)
                candidate = indices[serial_order]
                utility_order = torch.argsort(
                    self.utilities[candidate], descending=True, stable=True
                )
                selected = candidate[utility_order[: self.capacity_per_trajectory]]
            else:
                selected = self._diverse_subset(indices, self.capacity_per_trajectory)
            keep[indices] = False
            keep[selected] = True
        if bool((~keep).any().item()):
            for name in (
                "keys",
                "key_masks",
                "residuals",
                "trajectory_codes",
                "event_times",
                "available_times",
                "utilities",
                "serials",
            ):
                setattr(self, name, getattr(self, name)[keep])

    def _eligibility(
        self,
        query_codes: torch.Tensor,
        query_times: torch.Tensor,
        *,
        maximum_age: int | None,
        selection: SelectionControl,
        donor_codes: torch.Tensor | None,
        event_times: torch.Tensor,
    ) -> torch.Tensor:
        owner_codes = query_codes
        if selection == "wrong_trajectory":
            owner_ok = self.trajectory_codes[None, :] != owner_codes[:, None]
        elif selection == "paired_swap":
            if donor_codes is None:
                raise ValueError("paired_swap requires donor_trajectory_ids")
            owner_ok = self.trajectory_codes[None, :] == donor_codes[:, None]
        else:
            owner_ok = self.trajectory_codes[None, :] == owner_codes[:, None]
        age = query_times[:, None] - event_times[None, :]
        eligible = (
            owner_ok
            & (owner_codes[:, None] >= 0)
            & (age >= self.minimum_age)
            & (self.available_times[None, :] <= query_times[:, None])
        )
        if maximum_age is not None:
            eligible &= age <= int(maximum_age)
        return eligible

    def _selection_scores(
        self,
        query_keys: torch.Tensor,
        query_masks: torch.Tensor | None,
        selection: SelectionControl,
        oracle_residuals: torch.Tensor | None,
        control_seed: int,
        start: int = 0,
        stop: int | None = None,
    ) -> torch.Tensor:
        stop = len(self) if stop is None else int(stop)
        keys = self.keys[start:stop]
        entry_masks = self.key_masks[start:stop]
        if selection == "random_key":
            generator = torch.Generator(device="cpu").manual_seed(int(control_seed))
            all_random_keys = torch.randn(
                self.keys.shape,
                generator=generator,
                dtype=torch.float32,
            ).to(self.device)
            keys = all_random_keys[start:stop]
        if selection == "oracle_past_residual":
            if oracle_residuals is None:
                raise ValueError(
                    "oracle control requires validation-only oracle_residuals"
                )
            query = (
                oracle_residuals.detach()
                .to(self.device)
                .reshape(oracle_residuals.shape[0], -1)
            )
            values = self.residuals[start:stop].reshape(stop - start, -1)
            return masked_cosine_scores(query, values)
        if selection in {"last_residual", "recent_only"}:
            return (
                self.event_times[start:stop]
                .float()[None, :]
                .expand(query_keys.shape[0], -1)
            )
        return masked_cosine_scores(
            query_keys,
            keys,
            query_masks=query_masks,
            entry_masks=entry_masks,
        )

    def _apply_value_intervention(
        self, values: torch.Tensor, intervention: ValueIntervention
    ) -> torch.Tensor:
        if intervention == "none":
            return values
        if intervention == "erase":
            return torch.zeros_like(values)
        if intervention == "sign_flip":
            return -values
        if intervention == "spatial_shuffle":
            if values.ndim < 3:
                return torch.flip(values, dims=(-1,))
            return torch.flip(values, dims=(-2, -1))
        if intervention == "channel_mean_broadcast":
            if values.ndim < 3:
                return values.mean(dim=-1, keepdim=True).expand_as(values)
            return values.mean(dim=(-2, -1), keepdim=True).expand_as(values)
        raise ValueError(f"unsupported value intervention: {intervention}")

    def _causal_value_indices(
        self,
        eligible: torch.Tensor,
        selected_indices: torch.Tensor,
        selected_valid: torch.Tensor,
        control_seed: int,
    ) -> torch.Tensor:
        """Cyclic mispairing within each query's mature, same-owner donor pool.

        Order by immutable serial, then rotate by a seed-dependent nonzero
        offset. A singleton pool is unchanged and cannot test association.
        Neither future rows nor other owners change the permutation. This is
        a deterministic intervention, not a claim of a uniform random shuffle.
        """
        serial_order = torch.argsort(self.serials, stable=True)
        pool_order = torch.argsort(
            (~eligible[:, serial_order]).to(torch.int8), dim=1, stable=True
        )
        pools = serial_order[pool_order]
        ranks = torch.empty_like(pools)
        ranks.scatter_(
            1, pools, torch.arange(len(self), device=self.device).expand_as(pools)
        )
        selected_ranks = torch.gather(ranks, 1, selected_indices)
        counts = eligible.sum(dim=1).clamp_min(1)
        shifts = torch.where(
            counts > 1,
            1 + int(control_seed) % (counts - 1).clamp_min(1),
            torch.zeros_like(counts),
        )
        donor_ranks = (selected_ranks - shifts[:, None]) % counts[:, None]
        donors = torch.gather(pools, 1, donor_ranks)
        return torch.where(selected_valid, donors, selected_indices)

    def retrieve_batch(
        self,
        *,
        query_keys: torch.Tensor,
        trajectory_ids: tuple[str, ...],
        query_times: torch.Tensor,
        query_key_masks: torch.Tensor | None = None,
        maximum_age: int | None = None,
        selection: SelectionControl = "cosine",
        value_intervention: ValueIntervention = "none",
        donor_trajectory_ids: tuple[str, ...] | None = None,
        oracle_residuals: torch.Tensor | None = None,
        control_seed: int = 0,
        approximate: bool = False,
    ) -> BatchRetrieval:
        started = time.perf_counter()
        query = query_keys.detach().to(self.device)
        if query.ndim != 2 or query.shape[1] != self.key_dimension:
            raise ValueError("query_keys must have shape [B,key_dimension]")
        batch = int(query.shape[0])
        if len(trajectory_ids) != batch:
            raise ValueError("trajectory_ids batch mismatch")
        if donor_trajectory_ids is not None and len(donor_trajectory_ids) != batch:
            raise ValueError("donor_trajectory_ids batch mismatch")
        times = _as_long(query_times, self.device)
        if times.shape != (batch,):
            raise ValueError("query_times must have shape [B]")
        masks = (
            None
            if query_key_masks is None
            else query_key_masks.detach().to(self.device, torch.bool)
        )
        if masks is not None and masks.shape != query.shape:
            raise ValueError("query_key_masks must match query_keys")
        if selection == "recent_only" and maximum_age is None:
            raise ValueError("recent_only requires an explicit maximum_age")
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if len(self) == 0:
            return self._empty_retrieval(batch, query.dtype, started)
        query_codes = self._codes(trajectory_ids, create=False)
        donor_codes = (
            None
            if donor_trajectory_ids is None
            else self._codes(donor_trajectory_ids, create=False)
        )
        effective_events = self.event_times
        if selection == "time_permuted" and len(self) > 1:
            shift = 1 + int(control_seed) % (len(self) - 1)
            effective_events = torch.roll(self.event_times, shifts=shift)
        eligible = self._eligibility(
            query_codes,
            times,
            maximum_age=maximum_age,
            selection=selection,
            donor_codes=donor_codes,
            event_times=effective_events,
        )
        eligible_count = eligible.sum(dim=1)
        top_scores: torch.Tensor
        top_indices: torch.Tensor
        if approximate and selection not in {"last_residual", "recent_only"}:
            signatures_q = (query.float() @ self._ann_projection) >= 0
            signatures_e = (self.keys.float() @ self._ann_projection) >= 0
            hamming = (signatures_q[:, None, :] != signatures_e[None, :, :]).sum(dim=-1)
            candidate_count = min(len(self), max(64, self.topk * 16))
            hamming = hamming.masked_fill(~eligible, self.ann_bits + 1)
            candidate_indices = torch.argsort(hamming, dim=1, stable=True)[
                :, :candidate_count
            ]
            candidate_eligible = torch.gather(eligible, 1, candidate_indices)
            if selection == "oracle_past_residual":
                if oracle_residuals is None:
                    raise ValueError(
                        "oracle control requires validation-only oracle_residuals"
                    )
                candidate_keys = self.residuals[candidate_indices].reshape(
                    batch, candidate_count, -1
                )
                query_values = (
                    oracle_residuals.detach().to(self.device).reshape(batch, -1)
                )
                combined = torch.ones_like(candidate_keys, dtype=torch.bool)
            else:
                candidate_keys = self.keys[candidate_indices].float()
                if selection == "random_key":
                    generator = torch.Generator(device="cpu").manual_seed(
                        int(control_seed)
                    )
                    random_keys = torch.randn(
                        self.keys.shape, generator=generator, dtype=torch.float32
                    ).to(self.device)
                    candidate_keys = random_keys[candidate_indices]
                query_values = query.float()
                candidate_masks = self.key_masks[candidate_indices]
                combined = (
                    candidate_masks
                    if masks is None
                    else masks[:, None, :] & candidate_masks
                )
            products = (query_values[:, None, :] * candidate_keys * combined).sum(
                dim=-1
            )
            query_norms = torch.sqrt(
                (query_values[:, None, :].square() * combined).sum(dim=-1)
            )
            entry_norms = torch.sqrt((candidate_keys.square() * combined).sum(dim=-1))
            candidate_scores = products / (query_norms * entry_norms).clamp_min(1e-8)
            candidate_scores = candidate_scores.masked_fill(
                ~candidate_eligible, -torch.inf
            )
            top_scores, top_indices = _deterministic_candidate_topk(
                candidate_scores, candidate_indices, self.serials, self.topk
            )
        else:
            top_scores = torch.empty((batch, 0), device=self.device)
            top_indices = torch.empty((batch, 0), device=self.device, dtype=torch.long)
            for start in range(0, len(self), self.chunk_size):
                stop = min(len(self), start + self.chunk_size)
                chunk_scores = self._selection_scores(
                    query,
                    masks,
                    selection,
                    oracle_residuals,
                    control_seed,
                    start,
                    stop,
                ).masked_fill(~eligible[:, start:stop], -torch.inf)
                chunk_indices = torch.arange(
                    start, stop, device=self.device, dtype=torch.long
                ).expand(batch, -1)
                candidate_scores = torch.cat((top_scores, chunk_scores), dim=1)
                candidate_indices = torch.cat((top_indices, chunk_indices), dim=1)
                top_scores, top_indices = _deterministic_candidate_topk(
                    candidate_scores,
                    candidate_indices,
                    self.serials,
                    self.topk,
                )
        selected_valid = torch.isfinite(top_scores)
        best_similarity = top_scores[:, 0]
        if selection in {"last_residual", "recent_only"}:
            best_similarity = torch.where(
                selected_valid[:, 0],
                torch.ones_like(best_similarity),
                best_similarity,
            )
        valid = selected_valid[:, 0] & (best_similarity >= self.minimum_similarity)
        # Invalid slots must never receive softmax mass. Center before dividing
        # by temperature, and handle an empty eligible row without NaNs.
        has_values = selected_valid.any(dim=1, keepdim=True)
        masked_scores = top_scores.masked_fill(~selected_valid, -torch.inf)
        maximum = torch.where(
            has_values, masked_scores.max(dim=1, keepdim=True).values, 0.0
        )
        safe_scores = torch.where(
            has_values, (masked_scores - maximum) / self.temperature, 0.0
        )
        weights = torch.softmax(safe_scores, dim=1).masked_fill(~selected_valid, 0.0)
        value_indices = top_indices
        if selection == "random_value":
            value_indices = self._causal_value_indices(
                eligible, top_indices, selected_valid, control_seed
            )
        selected_values = self.residuals[value_indices]
        weight_shape = (batch, weights.shape[1]) + (1,) * len(self.residual_shape)
        residual = (selected_values * weights.reshape(weight_shape)).sum(dim=1)
        residual = self._apply_value_intervention(residual, value_intervention)
        residual = torch.where(
            valid.reshape((batch,) + (1,) * len(self.residual_shape)),
            residual,
            torch.zeros_like(residual),
        )
        best_indices = value_indices[:, 0]
        minus_one = torch.full((batch,), -1, device=self.device, dtype=torch.long)
        donor_valid = selected_valid & valid[:, None]
        donor_absent = torch.full_like(value_indices, -1)
        peak = (
            int(torch.cuda.max_memory_allocated(self.device))
            if torch.cuda.is_available() and self.device.type == "cuda"
            else 0
        )
        return BatchRetrieval(
            residual=residual.to(dtype=query.dtype),
            valid=valid,
            similarity=torch.where(
                valid, best_similarity, torch.zeros_like(best_similarity)
            ),
            source_event_time=torch.where(
                valid, self.event_times[best_indices], minus_one
            ),
            source_available_time=torch.where(
                valid, self.available_times[best_indices], minus_one
            ),
            source_serial=torch.where(valid, self.serials[best_indices], minus_one),
            eligible_count=eligible_count,
            latency_seconds=time.perf_counter() - started,
            peak_accelerator_bytes=peak,
            key_source_serial=torch.where(
                valid, self.serials[top_indices[:, 0]], minus_one
            ),
            value_source_serials=torch.where(
                donor_valid, self.serials[value_indices], donor_absent
            ),
            value_source_event_times=torch.where(
                donor_valid, self.event_times[value_indices], donor_absent
            ),
            value_source_available_times=torch.where(
                donor_valid, self.available_times[value_indices], donor_absent
            ),
            value_weights=weights * valid[:, None],
        )

    def _empty_retrieval(
        self, batch: int, dtype: torch.dtype, started: float
    ) -> BatchRetrieval:
        zeros = torch.zeros(
            (batch, *self.residual_shape), device=self.device, dtype=dtype
        )
        long_zeros = torch.zeros(batch, device=self.device, dtype=torch.long)
        return BatchRetrieval(
            residual=zeros,
            valid=torch.zeros(batch, device=self.device, dtype=torch.bool),
            similarity=torch.zeros(batch, device=self.device),
            source_event_time=long_zeros - 1,
            source_available_time=long_zeros - 1,
            source_serial=long_zeros - 1,
            eligible_count=long_zeros,
            latency_seconds=time.perf_counter() - started,
            peak_accelerator_bytes=0,
            key_source_serial=long_zeros - 1,
            value_source_serials=torch.empty(
                (batch, 0), device=self.device, dtype=torch.long
            ),
            value_source_event_times=torch.empty(
                (batch, 0), device=self.device, dtype=torch.long
            ),
            value_source_available_times=torch.empty(
                (batch, 0), device=self.device, dtype=torch.long
            ),
            value_weights=torch.empty((batch, 0), device=self.device),
        )

    def exact_reference(
        self,
        *,
        query_key: torch.Tensor,
        trajectory_id: str,
        query_time: int,
        query_key_mask: torch.Tensor | None = None,
        maximum_age: int | None = None,
    ) -> BatchRetrieval:
        """Slow one-query reference used only to verify the vectorized path."""

        return self.retrieve_batch(
            query_keys=query_key.reshape(1, -1),
            trajectory_ids=(trajectory_id,),
            query_times=torch.tensor([query_time]),
            query_key_masks=(
                None if query_key_mask is None else query_key_mask.reshape(1, -1)
            ),
            maximum_age=maximum_age,
            approximate=False,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "configuration": {
                "key_dimension": self.key_dimension,
                "residual_shape": self.residual_shape,
                "capacity_per_trajectory": self.capacity_per_trajectory,
                "minimum_age": self.minimum_age,
                "topk": self.topk,
                "temperature": self.temperature,
                "minimum_similarity": self.minimum_similarity,
                "chunk_size": self.chunk_size,
                "eviction": self.eviction,
                "seed": self.seed,
                "ann_bits": self.ann_bits,
                "storage_dtype": str(self.storage_dtype).removeprefix("torch."),
            },
            "trajectory_to_code": self._trajectory_to_code,
            "serial_counter": self._serial_counter,
            "keys": self.keys.detach().cpu(),
            "key_masks": self.key_masks.detach().cpu(),
            "residuals": self.residuals.detach().cpu(),
            "trajectory_codes": self.trajectory_codes.detach().cpu(),
            "event_times": self.event_times.detach().cpu(),
            "available_times": self.available_times.detach().cpu(),
            "utilities": self.utilities.detach().cpu(),
            "serials": self.serials.detach().cpu(),
            "ann_projection": self._ann_projection.detach().cpu(),
            "generator_state": self._generator.get_state(),
        }

    def checksum(self) -> str:
        digest = hashlib.sha256()
        state = self.state_dict()
        for name in sorted(state):
            value = state[name]
            digest.update(name.encode())
            if isinstance(value, torch.Tensor):
                digest.update(str(value.dtype).encode())
                digest.update(str(tuple(value.shape)).encode())
                digest.update(value.contiguous().numpy().tobytes())
            else:
                digest.update(repr(value).encode())
        return digest.hexdigest()

    def save(self, path: str | Path) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(self.state_dict(), temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return bank_artifact_sha256(destination)

    @classmethod
    def load(
        cls, path: str | Path, *, device: str | torch.device = "cpu"
    ) -> TensorMemoryBank:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != cls.SCHEMA_VERSION
        ):
            raise ValueError("unsupported memory-bank schema")
        configuration = dict(state["configuration"])
        dtype_name = str(configuration.pop("storage_dtype", "float32"))
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unsupported serialized storage dtype: {dtype_name}")
        configuration["storage_dtype"] = dtype
        bank = cls(**configuration, device=device)
        bank._trajectory_to_code = {
            str(key): int(value) for key, value in state["trajectory_to_code"].items()
        }
        bank._serial_counter = int(state["serial_counter"])
        for name in (
            "keys",
            "key_masks",
            "residuals",
            "trajectory_codes",
            "event_times",
            "available_times",
            "utilities",
            "serials",
        ):
            setattr(bank, name, state[name].to(bank.device))
        bank._ann_projection = state["ann_projection"].to(bank.device)
        bank._generator.set_state(state["generator_state"].cpu())
        if len(bank) and int(bank.keys.shape[1]) != bank.key_dimension:
            raise ValueError("serialized bank key dimension mismatch")
        return bank


def bank_artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
