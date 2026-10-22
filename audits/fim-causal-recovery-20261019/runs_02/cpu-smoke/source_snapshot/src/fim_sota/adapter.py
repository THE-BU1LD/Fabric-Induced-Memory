from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import BackboneOutput, ForecastBatch
from .memory import BatchRetrieval, TensorMemoryBank


class NormalizedKeyProjection(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int) -> None:
        super().__init__()
        if min(input_dimension, output_dimension) < 1:
            raise ValueError("projection dimensions must be positive")
        self.projection = nn.Linear(input_dimension, output_dimension, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        flattened = values.reshape(values.shape[0], -1)
        return F.normalize(self.projection(flattened).float(), dim=1, eps=1e-8)


class ContextKeyEncoder(nn.Module):
    """Encode latent state, mask geometry, lead time, and physical parameters."""

    def __init__(
        self,
        *,
        latent_dimension: int,
        key_dimension: int,
        mask_features: int = 16,
        lead_features: int = 8,
        physical_dimension: int = 0,
    ) -> None:
        super().__init__()
        if min(latent_dimension, key_dimension, mask_features, lead_features) < 1:
            raise ValueError("encoder dimensions must be positive")
        self.mask_features = int(mask_features)
        self.lead_features = int(lead_features)
        self.physical_dimension = int(physical_dimension)
        self.mask_projection = nn.LazyLinear(self.mask_features)
        combined = (
            int(latent_dimension)
            + self.mask_features
            + self.lead_features
            + self.physical_dimension
        )
        self.project = NormalizedKeyProjection(combined, key_dimension)

    def _lead_embedding(self, lead_times: torch.Tensor) -> torch.Tensor:
        half = self.lead_features // 2
        if half == 0:
            return lead_times.float().reshape(-1, 1)
        frequencies = torch.exp(
            torch.linspace(
                0.0,
                -math.log(10_000.0),
                half,
                device=lead_times.device,
            )
        )
        phase = lead_times.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        encoded = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
        if encoded.shape[1] < self.lead_features:
            encoded = F.pad(encoded, (0, self.lead_features - encoded.shape[1]))
        return encoded

    def forward(self, output: BackboneOutput, batch: ForecastBatch) -> torch.Tensor:
        latent = output.latent_key.reshape(output.latent_key.shape[0], -1)
        mask = output.observation_mask.float().reshape(
            output.observation_mask.shape[0], -1
        )
        mask_features = torch.tanh(self.mask_projection(mask))
        parts = [latent, mask_features, self._lead_embedding(batch.lead_times)]
        if self.physical_dimension:
            if batch.physical_parameters is None:
                raise ValueError("physical parameters are required by this key encoder")
            physical = batch.physical_parameters.reshape(batch.context.shape[0], -1)
            if physical.shape[1] != self.physical_dimension:
                raise ValueError("physical parameter dimension mismatch")
            parts.append(physical)
        return self.project(torch.cat(parts, dim=1))


class ResidualNormalizer(nn.Module):
    def __init__(
        self,
        channels: int,
        mode: str = "none",
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if mode not in {"none", "channel_std", "global_rms"}:
            raise ValueError(f"unsupported residual normalization: {mode}")
        self.mode = mode
        self.epsilon = float(epsilon)
        self.register_buffer("channel_scale", torch.ones(channels))
        self.register_buffer("global_scale", torch.ones(()))

    @torch.no_grad()
    def fit(self, residuals: torch.Tensor) -> None:
        if self.mode == "channel_std":
            dimensions = (0, *range(2, residuals.ndim))
            self.channel_scale.copy_(
                residuals.float()
                .std(dim=dimensions, unbiased=False)
                .clamp_min(self.epsilon)
            )
        elif self.mode == "global_rms":
            self.global_scale.copy_(
                residuals.float().square().mean().sqrt().clamp_min(self.epsilon)
            )

    def normalize(self, residuals: torch.Tensor) -> torch.Tensor:
        if self.mode == "channel_std":
            shape = (1, -1) + (1,) * (residuals.ndim - 2)
            return residuals / self.channel_scale.reshape(shape)
        if self.mode == "global_rms":
            return residuals / self.global_scale
        return residuals

    def denormalize(self, residuals: torch.Tensor) -> torch.Tensor:
        if self.mode == "channel_std":
            shape = (1, -1) + (1,) * (residuals.ndim - 2)
            return residuals * self.channel_scale.reshape(shape)
        if self.mode == "global_rms":
            return residuals * self.global_scale
        return residuals


class ChannelCalibrator(nn.Module):
    """Validation-fit nonnegative clipped per-channel calibration."""

    def __init__(self, channels: int, maximum: float = 1.0) -> None:
        super().__init__()
        if channels < 1 or maximum <= 0:
            raise ValueError("invalid channel calibration configuration")
        self.maximum = float(maximum)
        self.register_buffer("alpha", torch.zeros(channels))

    @torch.no_grad()
    def fit(self, errors: torch.Tensor, residuals: torch.Tensor) -> None:
        if errors.shape != residuals.shape or errors.ndim < 2:
            raise ValueError("calibration tensors must be paired [B,C,...]")
        dimensions = (0, *range(2, errors.ndim))
        numerator = (errors.float() * residuals.float()).sum(dim=dimensions)
        denominator = residuals.float().square().sum(dim=dimensions)
        fitted = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1e-12),
            torch.zeros_like(numerator),
        )
        self.alpha.copy_(fitted.clamp(0.0, self.maximum))

    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        shape = (1, -1) + (1,) * (residuals.ndim - 2)
        return residuals * self.alpha.reshape(shape)


class MonotoneSimilarityGate(nn.Module):
    """A monotone gate: softplus-constrained slope times similarity."""

    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.raw_slope = nn.Parameter(torch.zeros(()))
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(
        self, similarity: torch.Tensor, uncertainty: torch.Tensor | None = None
    ) -> torch.Tensor:
        slope = F.softplus(self.raw_slope)
        logits = slope * (similarity - self.threshold) + self.offset
        gate = torch.sigmoid(logits)
        if uncertainty is not None:
            reduced = uncertainty.float().reshape(uncertainty.shape[0], -1).mean(dim=1)
            gate = gate * torch.exp(-reduced.clamp_min(0.0))
        return gate


@dataclasses.dataclass(frozen=True)
class AdapterOutput:
    prediction: torch.Tensor
    backbone_prediction: torch.Tensor
    correction: torch.Tensor
    retrieval: BatchRetrieval
    gate: torch.Tensor
    diagnostics: Mapping[str, object]


class ITFMRAdapter(nn.Module):
    """Apply ITFM-R while preserving exact backbone fallback semantics."""

    def __init__(
        self,
        *,
        key_encoder: ContextKeyEncoder,
        bank: TensorMemoryBank,
        normalizer: ResidualNormalizer,
        calibrator: ChannelCalibrator,
        gate: MonotoneSimilarityGate,
    ) -> None:
        super().__init__()
        self.key_encoder = key_encoder
        self.bank = bank
        self.normalizer = normalizer
        self.calibrator = calibrator
        self.gate = gate

    def forward(
        self,
        output: BackboneOutput,
        batch: ForecastBatch,
        *,
        selection: str = "cosine",
        value_intervention: str = "none",
        maximum_age: int | None = None,
        approximate: bool = False,
    ) -> AdapterOutput:
        batch.validate()
        output.validate_against(batch)
        keys = self.key_encoder(output, batch)
        retrieval = self.bank.retrieve_batch(
            query_keys=keys,
            trajectory_ids=batch.trajectory_ids,
            query_times=batch.event_times,
            maximum_age=maximum_age,
            selection=selection,  # type: ignore[arg-type]
            value_intervention=value_intervention,  # type: ignore[arg-type]
            approximate=approximate,
        )
        retrieved = self.normalizer.denormalize(retrieval.residual)
        correction = self.calibrator(retrieved)
        gate = self.gate(retrieval.similarity, output.uncertainty)
        valid = retrieval.valid.to(correction.dtype)
        scale_shape = (gate.shape[0],) + (1,) * (correction.ndim - 1)
        correction = correction * (gate * valid).reshape(scale_shape)
        prediction = output.prediction + correction
        invalid_shape = (retrieval.valid.shape[0],) + (1,) * (
            output.prediction.ndim - 1
        )
        prediction = torch.where(
            retrieval.valid.reshape(invalid_shape), prediction, output.prediction
        )
        return AdapterOutput(
            prediction=prediction,
            backbone_prediction=output.prediction,
            correction=correction,
            retrieval=retrieval,
            gate=gate,
            diagnostics={
                "memory_checksum": self.bank.checksum(),
                "memory_entries": len(self.bank),
                "memory_bytes": self.bank.memory_bytes,
                "retrieval": retrieval.telemetry(),
            },
        )

    @torch.no_grad()
    def write_observed(
        self,
        output: BackboneOutput,
        batch: ForecastBatch,
        *,
        observation_time: int,
        utilities: torch.Tensor | None = None,
    ) -> int:
        """Write fully observed, mature residual rows only.

        This full-value bank cannot represent partially observed residuals.
        Such rows must use the separate observation-event staging bridge; the
        historical shortcut of writing hidden coordinates is forbidden.
        """

        batch.validate()
        output.validate_against(batch)
        if batch.target_available_times is None:
            raise ValueError(
                "residual writes require explicit target availability times"
            )
        available = batch.target_available_times
        admitted = available <= int(observation_time)
        if batch.target_observation_mask is None:
            raise ValueError("writes require an explicit target observation mask")
        mask = torch.broadcast_to(batch.target_observation_mask, batch.target.shape)
        if not bool(((mask == 0) | (mask == 1)).all().item()):
            raise ValueError("target observation mask must contain only zero/one")
        admitted = admitted & mask.bool().flatten(1).all(dim=1)
        if not bool(admitted.any().item()):
            return 0
        keys = self.key_encoder(output, batch)
        indices = torch.nonzero(admitted).flatten()
        residual = self.normalizer.normalize(
            batch.target[indices] - output.prediction[indices]
        )
        utility_values = None if utilities is None else utilities[indices]
        self.bank.write_batch(
            keys=keys[indices],
            residuals=residual,
            trajectory_ids=tuple(batch.trajectory_ids[int(index)] for index in indices),
            event_times=batch.event_times[indices],
            available_times=available[indices],
            utilities=utility_values,
        )
        return int(indices.numel())
