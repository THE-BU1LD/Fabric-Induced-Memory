from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .contracts import BackboneOutput, ForecastBatch, RolloutOutput, parameter_counts
from .provenance import require_sha256


class BackboneAdmissionError(RuntimeError):
    """Raised when an upstream backbone is not identity- or interface-admitted."""


def load_checkpoint_verified(
    path: str | Path,
    *,
    expected_sha256: str,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    require_sha256(path, expected_sha256)
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise BackboneAdmissionError("checkpoint is not a state mapping")
    return payload


class WrappedBackbone(nn.Module):
    """Fail-closed adapter around an admitted upstream implementation."""

    def __init__(
        self,
        module: nn.Module,
        *,
        identity: Mapping[str, Any],
        forward_adapter: Callable[[nn.Module, ForecastBatch], BackboneOutput],
        rollout_adapter: Callable[[nn.Module, ForecastBatch, int, bool], RolloutOutput],
        expected_latent_dimension: int,
        preprocessing_sha256: str,
    ) -> None:
        super().__init__()
        required = {
            "name",
            "source_commit",
            "checkpoint_sha256",
            "reproduction_receipt",
        }
        missing = sorted(required - set(identity))
        if missing:
            raise BackboneAdmissionError(f"backbone identity missing {missing}")
        if not Path(str(identity["reproduction_receipt"])).is_file():
            raise BackboneAdmissionError("frontier reproduction receipt is missing")
        self.module = module
        self._identity = dict(identity)
        self.forward_adapter = forward_adapter
        self.rollout_adapter = rollout_adapter
        self.expected_latent_dimension = int(expected_latent_dimension)
        self.preprocessing_sha256 = preprocessing_sha256

    def forward_batch(self, batch: ForecastBatch) -> BackboneOutput:
        output = self.forward_adapter(self.module, batch)
        output.validate_against(batch)
        if output.latent_key.reshape(output.latent_key.shape[0], -1).shape[1] != (
            self.expected_latent_dimension
        ):
            raise BackboneAdmissionError("backbone latent dimension changed")
        return output

    def rollout(
        self, initial: ForecastBatch, horizon: int, *, teacher_forcing: bool = False
    ) -> RolloutOutput:
        if horizon < 1:
            raise ValueError("rollout horizon must be positive")
        return self.rollout_adapter(self.module, initial, horizon, teacher_forcing)

    def identity(self) -> Mapping[str, Any]:
        counts = dataclasses.asdict(parameter_counts(self.module))
        return {
            **self._identity,
            "preprocessing_sha256": self.preprocessing_sha256,
            "parameters": counts,
        }

    def freeze(self, frozen: bool = True) -> None:
        for parameter in self.module.parameters():
            parameter.requires_grad_(not frozen)


class LANOBackbone(WrappedBackbone):
    """Interface marker for an independently admitted LANO reproduction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self._identity.get("name") != "LANO":
            raise BackboneAdmissionError("LANO adapter received a different backbone")


class HSFNOBackbone(WrappedBackbone):
    """History-space adapter that preserves deterministic shift-append transport."""

    def __init__(self, *args: Any, history_length: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self._identity.get("name") != "HS-FNO":
            raise BackboneAdmissionError("HS-FNO adapter received a different backbone")
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.history_length = int(history_length)

    @staticmethod
    def shift_append(
        history: torch.Tensor, new_observation: torch.Tensor
    ) -> torch.Tensor:
        if history.ndim < 3 or new_observation.shape != history[:, -1].shape:
            raise ValueError("shift-append tensor shape mismatch")
        return torch.cat((history[:, 1:], new_observation.unsqueeze(1)), dim=1)

    @staticmethod
    def observed_append(
        history: torch.Tensor,
        candidate: torch.Tensor,
        *,
        target_observed: torch.Tensor,
    ) -> torch.Tensor:
        if target_observed.shape != (history.shape[0],):
            raise ValueError("target_observed must have shape [B]")
        if not bool(target_observed.all().item()):
            raise BackboneAdmissionError(
                "predicted future values may not be appended as observed memory"
            )
        return HSFNOBackbone.shift_append(history, candidate)


class PersistenceBackbone(nn.Module):
    def forward_batch(self, batch: ForecastBatch) -> BackboneOutput:
        prediction = batch.context[:, -1]
        if prediction.shape != batch.target.shape:
            raise ValueError("last context frame does not match target shape")
        return BackboneOutput(
            prediction=prediction,
            latent_key=batch.context.reshape(batch.context.shape[0], -1),
            observation_mask=batch.observation_mask,
            diagnostics={"baseline": "persistence"},
        )

    def rollout(
        self, initial: ForecastBatch, horizon: int, *, teacher_forcing: bool = False
    ) -> RolloutOutput:
        output = self.forward_batch(initial)
        predictions = output.prediction.unsqueeze(1).expand(
            -1, horizon, *output.prediction.shape[1:]
        )
        return RolloutOutput(
            predictions=predictions, steps=tuple(output for _ in range(horizon))
        )

    def identity(self) -> Mapping[str, Any]:
        return {
            "name": "persistence",
            "parameters": {"total": 0, "trainable": 0, "frozen": 0},
        }


class ClimatologyBackbone(nn.Module):
    def __init__(self, mean: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.detach().clone())

    def forward_batch(self, batch: ForecastBatch) -> BackboneOutput:
        prediction = self.mean.expand_as(batch.target)
        return BackboneOutput(
            prediction=prediction,
            latent_key=batch.context.reshape(batch.context.shape[0], -1),
            observation_mask=batch.observation_mask,
            diagnostics={"baseline": "climatology"},
        )

    def rollout(
        self, initial: ForecastBatch, horizon: int, *, teacher_forcing: bool = False
    ) -> RolloutOutput:
        output = self.forward_batch(initial)
        return RolloutOutput(
            predictions=output.prediction.unsqueeze(1).expand(
                -1, horizon, *output.prediction.shape[1:]
            ),
            steps=tuple(output for _ in range(horizon)),
        )

    def identity(self) -> Mapping[str, Any]:
        return {
            "name": "climatology",
            "parameters": {"total": 0, "trainable": 0, "frozen": 0},
        }


class ResidualMLPControl(nn.Module):
    """Learned non-episodic residual control for parameter matching."""

    def __init__(
        self, input_dimension: int, output_shape: tuple[int, ...], hidden: int
    ) -> None:
        super().__init__()
        if min(input_dimension, hidden, *output_shape) < 1:
            raise ValueError("invalid residual MLP dimensions")
        self.output_shape = output_shape
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, math_product(output_shape)),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent.reshape(latent.shape[0], -1)).reshape(
            latent.shape[0], *self.output_shape
        )


def math_product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


REQUIRED_EXTERNAL_BASELINES = {
    "current_state_fno",
    "lag_stack_fno",
    "history_to_history_fno",
    "ffno_fno_mpt",
    "lno_mpt",
    "oformer",
    "gnot",
    "ipot",
    "mionet",
    "unet",
    "convlstm",
    "convgru",
    "transformer_history",
    "memno",
    "poseidon_optional",
}


class BackboneRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., nn.Module]] = {
            "persistence": PersistenceBackbone,
            "climatology": ClimatologyBackbone,
            "residual_mlp": ResidualMLPControl,
        }
        self._receipts: dict[str, str] = {}

    def register_external(
        self,
        name: str,
        factory: Callable[..., nn.Module],
        reproduction_receipt: str | Path,
    ) -> None:
        if name not in REQUIRED_EXTERNAL_BASELINES | {"lano", "hs_fno"}:
            raise KeyError(f"unregistered comparison model: {name}")
        receipt = Path(reproduction_receipt)
        if not receipt.is_file():
            raise BackboneAdmissionError(f"missing reproduction receipt for {name}")
        payload = __import__("json").loads(receipt.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS_REPRODUCED":
            raise BackboneAdmissionError(f"{name} has not passed upstream reproduction")
        self._factories[name] = factory
        self._receipts[name] = str(receipt.resolve())

    def create(self, name: str, **kwargs: Any) -> nn.Module:
        if name not in self._factories:
            raise BackboneAdmissionError(
                f"{name} unavailable: exact implementation/reproduction not admitted"
            )
        return self._factories[name](**kwargs)

    def availability(self) -> Mapping[str, Mapping[str, Any]]:
        all_names = (
            set(self._factories) | REQUIRED_EXTERNAL_BASELINES | {"lano", "hs_fno"}
        )
        return {
            name: {
                "available": name in self._factories,
                "reproduction_receipt": self._receipts.get(name),
            }
            for name in sorted(all_names)
        }
