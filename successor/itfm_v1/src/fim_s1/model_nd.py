from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import Intervention, TensorFabricMemory
from .model import ITFMOutput
from .transport import (
    apply_periodic_translation,
    estimate_periodic_translation,
    transport_coordinates,
)


def convolution_for_rank(rank: int):
    try:
        return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[rank]
    except KeyError as error:
        raise ValueError("spatial rank must be 1, 2, or 3") from error


class NDResidualSpatialBlock(nn.Module):
    def __init__(self, channels: int, spatial_rank: int) -> None:
        super().__init__()
        convolution = convolution_for_rank(spatial_rank)
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.depthwise = convolution(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.mix = convolution(channels, channels, kernel_size=1)
        self.gate = convolution(channels, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(field)
        update = self.mix(F.gelu(self.depthwise(hidden)))
        gate = torch.sigmoid(self.gate(hidden))
        return field + torch.tanh(self.scale) * gate * update


class InterventionalNDTensorFabricPredictor(nn.Module):
    """Rank-generic ITFM predictor for 1D, 2D, or 3D spatial fields."""

    def __init__(
        self,
        *,
        context_channels: int,
        output_channels: int,
        spatial_shape: tuple[int, ...],
        hidden_channels: int = 32,
        key_channels: int = 16,
        memory_capacity: int = 128,
        topk: int = 4,
        temperature: float = 0.2,
        max_feedback: float = 1.0,
        depth: int = 3,
        memory_mode: str = "tensor",
        transport_mode: str = "identity",
        max_transport_shift: int = 2,
    ) -> None:
        super().__init__()
        spatial_shape = tuple(int(size) for size in spatial_shape)
        if not spatial_shape or any(size < 1 for size in spatial_shape):
            raise ValueError("spatial_shape must contain positive dimensions")
        spatial_rank = len(spatial_shape)
        convolution = convolution_for_rank(spatial_rank)
        if (
            min(context_channels, output_channels, hidden_channels, key_channels, depth)
            < 1
        ):
            raise ValueError("model dimensions and depth must be positive")
        if max_feedback <= 0:
            raise ValueError("max_feedback must be positive")
        if memory_mode not in {"tensor", "vector_broadcast"}:
            raise ValueError("memory_mode must be tensor or vector_broadcast")
        if transport_mode not in {"identity", "bounded_xcorr"}:
            raise ValueError("transport_mode must be identity or bounded_xcorr")
        if max_transport_shift < 0:
            raise ValueError("max_transport_shift must be non-negative")
        if transport_mode == "bounded_xcorr" and any(
            2 * max_transport_shift >= size for size in spatial_shape
        ):
            raise ValueError(
                "max_transport_shift must be less than half of each spatial dimension"
            )
        if context_channels % output_channels:
            raise ValueError("context_channels must be divisible by output_channels")
        if transport_mode == "bounded_xcorr" and context_channels < 2 * output_channels:
            raise ValueError("bounded_xcorr requires at least two observed frames")

        self.context_channels = int(context_channels)
        self.output_channels = int(output_channels)
        self.spatial_shape = spatial_shape
        self.spatial_rank = spatial_rank
        self.max_feedback = float(max_feedback)
        self.memory_mode = memory_mode
        self.transport_mode = transport_mode
        self.max_transport_shift = int(max_transport_shift)

        self.stem = convolution(
            context_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.blocks = nn.ModuleList(
            [
                NDResidualSpatialBlock(hidden_channels, spatial_rank)
                for _ in range(depth)
            ]
        )
        self.base_head = convolution(
            hidden_channels, output_channels, kernel_size=3, padding=1
        )
        self.query_head = convolution(hidden_channels, key_channels, kernel_size=1)
        self.correction_gate = convolution(
            hidden_channels + output_channels, output_channels, kernel_size=1
        )
        self.feedback_logit = nn.Parameter(torch.tensor(0.1))
        self.memory = TensorFabricMemory(
            capacity=memory_capacity,
            key_channels=key_channels,
            value_channels=output_channels,
            spatial_shape=spatial_shape,
            topk=topk,
            temperature=temperature,
            spatial_addressing=memory_mode == "tensor",
            spatial_values=memory_mode == "tensor",
        )

    def clear_memory(self) -> None:
        self.memory.clear()

    def _encode(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (self.context_channels, *self.spatial_shape)
        if (
            context.ndim != self.spatial_rank + 2
            or tuple(context.shape[1:]) != expected
        ):
            raise ValueError(
                f"context must have shape [B,{','.join(map(str, expected))}], "
                f"got {tuple(context.shape)}"
            )
        if not bool(torch.isfinite(context).all().item()):
            raise ValueError("context contains non-finite values")
        latent = F.gelu(self.stem(context))
        for block in self.blocks:
            latent = block(latent)
        base = self.base_head(latent)
        query = F.normalize(self.query_head(latent), dim=1, eps=1e-8)
        return latent, base, query

    @torch.no_grad()
    def estimate_transport_shifts(self, context: torch.Tensor) -> torch.Tensor:
        batch = context.shape[0]
        if self.transport_mode == "identity" or self.max_transport_shift == 0:
            return torch.zeros(
                batch,
                self.spatial_rank,
                device=context.device,
                dtype=torch.long,
            )
        channels = self.output_channels
        return estimate_periodic_translation(
            context[:, -2 * channels : -channels],
            context[:, -channels:],
            max_shift=self.max_transport_shift,
        )

    def forward(
        self,
        context: torch.Tensor,
        owners: torch.Tensor,
        query_times: torch.Tensor,
        *,
        use_memory: bool = True,
        intervention: Intervention = "none",
        replacement: torch.Tensor | None = None,
    ) -> ITFMOutput:
        latent, base, query = self._encode(context)
        transport_shifts = self.estimate_transport_shifts(context)
        memory_query = query
        if self.transport_mode == "bounded_xcorr":
            memory_query = apply_periodic_translation(
                query,
                -transport_coordinates(transport_shifts, query_times),
            )
        retrieval = self.memory.retrieve(
            memory_query,
            owners,
            query_times,
            intervention=intervention if use_memory else "erase",
            replacement=replacement if use_memory else None,
        )
        retrieved = retrieval.value
        if self.transport_mode == "bounded_xcorr" and replacement is None:
            retrieved = apply_periodic_translation(
                retrieved,
                transport_coordinates(
                    transport_shifts,
                    query_times,
                    temporal_offset=1,
                ),
            )
        gate = torch.sigmoid(
            self.correction_gate(torch.cat([latent, retrieved], dim=1))
        )
        scale = self.max_feedback * torch.tanh(self.feedback_logit)
        prediction = base + scale * gate * retrieved
        if not use_memory:
            prediction = base
            retrieved = torch.zeros_like(retrieved)
            gate = torch.zeros_like(gate)
        if not bool(torch.isfinite(prediction).all().item()):
            raise RuntimeError("prediction contains non-finite values")
        return ITFMOutput(
            prediction,
            base,
            memory_query,
            retrieved,
            retrieval,
            gate,
            transport_shifts,
        )

    @torch.no_grad()
    def commit_from_output(
        self,
        output: ITFMOutput,
        target: torch.Tensor,
        owners: torch.Tensor,
        event_times: torch.Tensor,
        regimes: torch.Tensor | None = None,
    ) -> None:
        expected = (self.output_channels, *self.spatial_shape)
        if target.ndim != self.spatial_rank + 2 or tuple(target.shape[1:]) != expected:
            raise ValueError(
                f"target must have shape [B,{','.join(map(str, expected))}], "
                f"got {tuple(target.shape)}"
            )
        residual = target - output.base_prediction.detach()
        if self.transport_mode == "bounded_xcorr":
            if output.transport_shifts is None:
                raise RuntimeError("transport output is missing shifts")
            residual = apply_periodic_translation(
                residual,
                -transport_coordinates(
                    output.transport_shifts,
                    event_times,
                    temporal_offset=1,
                ),
            )
        self.memory.add(
            output.query.detach(), residual, owners, event_times, regimes=regimes
        )
