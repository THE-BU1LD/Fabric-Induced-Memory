from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MemoryTrace:
    key: torch.Tensor
    value: torch.Tensor
    salience: float
    timestamp: int
    layer: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TraceBank:
    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        max_traces: int = 4096,
        merge_threshold: float = 0.98,
        decay_rate: float = 0.0,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.key_dim = self._integer(key_dim, "key_dim", minimum=1)
        self.value_dim = self._integer(value_dim, "value_dim", minimum=1)
        self.max_traces = self._integer(max_traces, "max_traces", minimum=1)
        self.merge_threshold = float(merge_threshold)
        self.decay_rate = float(decay_rate)
        if not math.isfinite(self.merge_threshold):
            raise ValueError("merge_threshold must be finite")
        if not math.isfinite(self.decay_rate) or self.decay_rate < 0:
            raise ValueError("decay_rate must be finite and nonnegative")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError("dtype must be a real floating-point dtype")
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self._traces: List[MemoryTrace] = []
        self._clock = 0

    def __len__(self) -> int:
        return len(self._traces)

    def clear(self) -> None:
        self._traces.clear()
        self._clock = 0

    @staticmethod
    def _integer(value: int, name: str, minimum: int = 0) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer >= {minimum}")
        try:
            result = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be an integer >= {minimum}") from exc
        if result < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return result

    def _owned_vector(self, x: torch.Tensor, width: int, name: str, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(x) or x.layout != torch.strided or x.is_complex():
            raise ValueError(f"{name} must be a dense real tensor")
        if x.numel() != width:
            raise ValueError(f"Expected {name}_dim={width}, got {x.numel()}")
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} must contain only finite values")
        # detach()/to() can share storage with a caller, including a batch view.
        result = x.detach().to(device=device, dtype=self.dtype).reshape(-1).clone()
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} is not finite after conversion to {self.dtype}")
        return result

    def _prepare_trace(self, key, value, salience, timestamp, layer, metadata, device) -> MemoryTrace:
        key_1d = self._owned_vector(key, self.key_dim, "key", device)
        value_1d = self._owned_vector(value, self.value_dim, "value", device)
        try:
            sal = float(salience.item() if torch.is_tensor(salience) else salience)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("salience must be a finite scalar") from exc
        if not math.isfinite(sal) or abs(sal) > torch.finfo(self.dtype).max:
            raise ValueError(f"salience must be finite in {self.dtype}")
        ts = self._integer(timestamp, "timestamp")
        if ts > torch.finfo(self.dtype).max:
            raise ValueError(f"timestamp must be finite in {self.dtype}")
        layer = self._integer(layer, "layer")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping or None")
        meta = deepcopy(dict(metadata)) if metadata is not None else {}
        return MemoryTrace(key_1d, value_1d, sal, ts, layer, meta)

    def _effective_score(self, trace: MemoryTrace, now: Optional[int] = None) -> float:
        current = self._clock if now is None else int(now)
        age = max(0, current - int(trace.timestamp))
        decay = math_exp_safe(-self.decay_rate * age)
        return float(trace.salience * decay)

    def _prune_if_needed(self) -> None:
        if len(self._traces) <= self.max_traces:
            return
        self._traces.sort(key=lambda t: (self._effective_score(t), t.timestamp), reverse=True)
        self._traces = self._traces[: self.max_traces]

    def _admit(self, prepared: List[MemoryTrace], device: torch.device) -> None:
        """Stage all merges/pruning before publishing any change to live memory."""
        if not prepared:
            return
        staged = list(self._traces)
        clock = self._clock
        for trace in prepared:
            clock = max(clock, trace.timestamp)
            if staged:
                keys = torch.stack([item.key for item in staged])
                query = F.normalize(trace.key.unsqueeze(0), dim=-1, eps=1e-6)
                bank_keys = F.normalize(keys, dim=-1, eps=1e-6)
                sim = torch.matmul(query, bank_keys.t()).squeeze(0)
                if not torch.isfinite(sim).all():
                    raise ValueError("non-finite similarity during trace admission")
                best_idx = int(torch.argmax(sim).item())
                if float(sim[best_idx].item()) >= self.merge_threshold:
                    existing = staged[best_idx]
                    w_new = max(trace.salience, 1e-6)
                    w_old = max(existing.salience, 1e-6)
                    total = w_new + w_old
                    alpha = w_new / total if math.isfinite(total) else 1.0 / (1.0 + w_old / w_new)
                    merged_key = (1.0 - alpha) * existing.key + alpha * trace.key
                    merged_value = (1.0 - alpha) * existing.value + alpha * trace.value
                    if not torch.isfinite(merged_key).all() or not torch.isfinite(merged_value).all():
                        raise ValueError("non-finite merged trace")
                    staged[best_idx] = MemoryTrace(
                        key=merged_key,
                        value=merged_value,
                        salience=max(existing.salience, trace.salience),
                        timestamp=max(existing.timestamp, trace.timestamp),
                        layer=trace.layer,
                        metadata={**existing.metadata, **trace.metadata},
                    )
                    continue
            staged.append(trace)
            if len(staged) > self.max_traces:
                staged.sort(key=lambda item: (self._effective_score(item, clock), item.timestamp), reverse=True)
                staged = staged[: self.max_traces]
        self._traces = staged
        self._clock = clock
        self.device = device

    def add(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        salience: float | torch.Tensor,
        timestamp: int,
        layer: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not torch.is_tensor(key):
            raise ValueError("key must be a dense real tensor")
        device = self.device or key.device
        trace = self._prepare_trace(key, value, salience, timestamp, layer, metadata, device)
        self._admit([trace], device)

    def batch_add(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        saliences: torch.Tensor,
        timestamp: int,
        layer: int = 0,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not all(torch.is_tensor(x) for x in (keys, values, saliences)):
            raise ValueError("keys, values and saliences must be tensors")
        if keys.ndim == 1:
            keys = keys.unsqueeze(0)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if keys.ndim < 2 or values.ndim < 2 or saliences.ndim < 1:
            raise ValueError("Batch add requires a leading batch dimension")
        if not torch.isfinite(saliences).all() or saliences.is_complex():
            raise ValueError("saliences must contain only finite real values")
        if saliences.ndim > 1:
            saliences = saliences.reshape(saliences.shape[0], -1).mean(dim=-1)
        if keys.shape[0] != values.shape[0] or keys.shape[0] != saliences.shape[0]:
            raise ValueError("Batch add requires matching batch sizes")

        if metadata is not None and len(metadata) != keys.shape[0]:
            raise ValueError("metadata must have one mapping per batch row")
        meta_list = metadata if metadata is not None else [None] * keys.shape[0]
        device = self.device or keys.device
        prepared = [
            self._prepare_trace(keys[i], values[i], saliences[i], timestamp, layer, meta_list[i], device)
            for i in range(keys.shape[0])
        ]
        self._admit(prepared, device)

    def as_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self._traces) == 0:
            device = self.device or torch.device("cpu")
            return (
                torch.empty(0, self.key_dim, device=device, dtype=self.dtype),
                torch.empty(0, self.value_dim, device=device, dtype=self.dtype),
                torch.empty(0, 1, device=device, dtype=self.dtype),
                torch.empty(0, 1, device=device, dtype=self.dtype),
            )

        keys = torch.stack([t.key for t in self._traces], dim=0)
        values = torch.stack([t.value for t in self._traces], dim=0)
        saliences = torch.tensor([[t.salience] for t in self._traces], device=keys.device, dtype=self.dtype)
        timestamps = torch.tensor([[t.timestamp] for t in self._traces], device=keys.device, dtype=self.dtype)
        return keys, values, saliences, timestamps

    def statistics(self) -> Dict[str, float]:
        if len(self._traces) == 0:
            return {
                "size": 0.0,
                "mean_salience": 0.0,
                "max_salience": 0.0,
                "mean_age": 0.0,
                "max_age": 0.0,
            }

        now = self._clock
        ages = [max(0, now - t.timestamp) for t in self._traces]
        saliences = [t.salience for t in self._traces]
        return {
            "size": float(len(self._traces)),
            "mean_salience": float(sum(saliences) / len(saliences)),
            "max_salience": float(max(saliences)),
            "mean_age": float(sum(ages) / len(ages)),
            "max_age": float(max(ages)),
        }

    def recent(self, k: int) -> List[MemoryTrace]:
        if k <= 0:
            return []
        return self._traces[-k:]

    def iter_traces(self) -> Iterable[MemoryTrace]:
        return iter(self._traces)

    def _retrieve_impl(
        self,
        query: torch.Tensor,
        topk: int = 32,
        temperature: float = 0.2,
        salience_scale: float = 0.05,
        recency_scale: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.ndim == 1:
            query = query.unsqueeze(0)
        query = query.reshape(query.shape[0], -1)

        keys, values, saliences, timestamps = self.as_tensors()
        if keys.numel() == 0:
            batch = query.shape[0]
            device = query.device
            dtype = query.dtype
            return (
                torch.zeros(batch, self.value_dim, device=device, dtype=dtype),
                torch.zeros(batch, 0, device=device, dtype=dtype),
                torch.empty(batch, 0, device=device, dtype=torch.long),
                torch.empty(batch, 0, device=device, dtype=dtype),
            )

        device = query.device
        dtype = query.dtype
        keys = keys.to(device=device, dtype=dtype)
        values = values.to(device=device, dtype=dtype)
        saliences = saliences.to(device=device, dtype=dtype).view(-1)
        timestamps = timestamps.to(device=device, dtype=dtype).view(-1)

        q = F.normalize(query, dim=-1, eps=1e-6)
        k = F.normalize(keys, dim=-1, eps=1e-6)

        sim = torch.matmul(q, k.t())
        if saliences.numel() > 0:
            sim = sim + float(salience_scale) * saliences.unsqueeze(0)
        if timestamps.numel() > 0:
            age = timestamps.max() - timestamps
            sim = sim - float(recency_scale) * age.unsqueeze(0)

        temp = max(float(temperature), 1e-6)
        if topk is not None and topk > 0 and topk < sim.shape[-1]:
            top_sim, top_idx = torch.topk(sim, k=topk, dim=-1)
            weights = torch.softmax(top_sim / temp, dim=-1)
            gathered = values.unsqueeze(0).expand(query.shape[0], -1, -1)
            gathered = torch.gather(
                gathered,
                dim=1,
                index=top_idx.unsqueeze(-1).expand(-1, -1, self.value_dim),
            )
            context = torch.sum(weights.unsqueeze(-1) * gathered, dim=1)
            return context, weights, top_idx, top_sim

        weights = torch.softmax(sim / temp, dim=-1)
        context = torch.matmul(weights, values)
        indices = torch.arange(values.shape[0], device=device).unsqueeze(0).expand(query.shape[0], -1)
        return context, weights, indices, sim

    def retrieve(
        self,
        query: torch.Tensor,
        topk: int = 32,
        temperature: float = 0.2,
        salience_scale: float = 0.05,
        recency_scale: float = 0.01,
    ) -> torch.Tensor:
        context, _, _, _ = self._retrieve_impl(
            query=query,
            topk=topk,
            temperature=temperature,
            salience_scale=salience_scale,
            recency_scale=recency_scale,
        )
        return context

    def retrieve_detailed(
        self,
        query: torch.Tensor,
        topk: int = 32,
        temperature: float = 0.2,
        salience_scale: float = 0.05,
        recency_scale: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._retrieve_impl(
            query=query,
            topk=topk,
            temperature=temperature,
            salience_scale=salience_scale,
            recency_scale=recency_scale,
        )


def math_exp_safe(x: float) -> float:
    if x < -60.0:
        return 0.0
    if x > 60.0:
        return float("inf")
    import math

    return math.exp(x)
