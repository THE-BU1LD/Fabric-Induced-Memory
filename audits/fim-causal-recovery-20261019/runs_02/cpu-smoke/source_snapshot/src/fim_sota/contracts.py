from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import torch


@dataclasses.dataclass(frozen=True)
class ForecastBatch:
    """Common batch contract used by public-data backbones and evaluators."""

    context: torch.Tensor
    target: torch.Tensor
    observation_mask: torch.Tensor
    trajectory_ids: tuple[str, ...]
    event_times: torch.Tensor
    lead_times: torch.Tensor
    split: str
    target_available_times: torch.Tensor | None = None
    target_observation_mask: torch.Tensor | None = None
    physical_parameters: torch.Tensor | None = None
    provenance_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        batch = int(self.context.shape[0])
        if batch < 1 or int(self.target.shape[0]) != batch:
            raise ValueError("context and target must share a non-empty batch axis")
        if int(self.observation_mask.shape[0]) != batch:
            raise ValueError("observation_mask batch mismatch")
        if len(self.trajectory_ids) != batch:
            raise ValueError("trajectory_ids batch mismatch")
        if self.event_times.shape != (batch,) or self.lead_times.shape != (batch,):
            raise ValueError("event_times and lead_times must have shape [B]")
        if self.target_available_times is not None and (
            self.target_available_times.shape != (batch,)
        ):
            raise ValueError("target_available_times must have shape [B]")
        if self.target_observation_mask is not None:
            try:
                torch.broadcast_shapes(
                    self.target_observation_mask.shape, self.target.shape
                )
            except RuntimeError as error:
                raise ValueError(
                    "target_observation_mask is not target-compatible"
                ) from error
        if self.physical_parameters is not None and (
            int(self.physical_parameters.shape[0]) != batch
        ):
            raise ValueError("physical_parameters batch mismatch")
        if self.provenance_ids and len(self.provenance_ids) != batch:
            raise ValueError("provenance_ids batch mismatch")
        for name, value in (
            ("context", self.context),
            ("target", self.target),
            ("observation_mask", self.observation_mask),
        ):
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must be finite")


@dataclasses.dataclass(frozen=True)
class BackboneOutput:
    prediction: torch.Tensor
    latent_key: torch.Tensor
    observation_mask: torch.Tensor
    diagnostics: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    uncertainty: torch.Tensor | None = None

    def validate_against(self, batch: ForecastBatch) -> None:
        if self.prediction.shape != batch.target.shape:
            raise ValueError("backbone prediction/target shape mismatch")
        if int(self.latent_key.shape[0]) != int(batch.target.shape[0]):
            raise ValueError("latent key batch mismatch")
        if self.observation_mask.shape != batch.observation_mask.shape:
            raise ValueError("backbone mask differs from the input mask")
        if not bool(torch.isfinite(self.prediction).all().item()):
            raise FloatingPointError("backbone prediction is not finite")
        if not bool(torch.isfinite(self.latent_key).all().item()):
            raise FloatingPointError("backbone latent key is not finite")


@dataclasses.dataclass(frozen=True)
class RolloutOutput:
    predictions: torch.Tensor
    steps: tuple[BackboneOutput, ...]
    diagnostics: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@runtime_checkable
class ForecastBackbone(Protocol):
    def forward_batch(self, batch: ForecastBatch) -> BackboneOutput: ...

    def rollout(
        self, initial: ForecastBatch, horizon: int, *, teacher_forcing: bool = False
    ) -> RolloutOutput: ...

    def identity(self) -> Mapping[str, Any]: ...

    def parameters(self, recurse: bool = True) -> Sequence[torch.nn.Parameter]: ...


@dataclasses.dataclass(frozen=True)
class ParameterCounts:
    total: int
    trainable: int
    frozen: int


def parameter_counts(module: torch.nn.Module) -> ParameterCounts:
    parameters = tuple(module.parameters())
    total = sum(int(parameter.numel()) for parameter in parameters)
    trainable = sum(
        int(parameter.numel()) for parameter in parameters if parameter.requires_grad
    )
    return ParameterCounts(total=total, trainable=trainable, frozen=total - trainable)
