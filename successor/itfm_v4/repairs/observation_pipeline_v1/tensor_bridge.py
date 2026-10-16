"""Prospective target-free bridge to the retained public ITFM adapter.

Import only after numerical resource admission. No historical runner is changed.
"""

from dataclasses import dataclass
import hashlib
import json
import math

import torch

from fim_sota.contracts import ForecastBatch
from integrity import ev
from observations import Forecast, ObservationLedger


@dataclass(frozen=True)
class ForecastInputs:
    context: torch.Tensor
    observation_mask: torch.Tensor
    trajectory_ids: tuple[str, ...]
    event_times: torch.Tensor
    lead_times: torch.Tensor
    residual_shape: tuple[int, ...]
    split: str
    physical_parameters: torch.Tensor | None = None


def _state_digest(backbone, adapter):
    digest = hashlib.sha256()
    configuration = {
        "backbone": backbone.identity(),
        "normalizer": [adapter.normalizer.mode, adapter.normalizer.epsilon],
        "gate_threshold": adapter.gate.threshold,
        "encoder": [
            adapter.key_encoder.mask_features,
            adapter.key_encoder.lead_features,
            adapter.key_encoder.physical_dimension,
        ],
    }
    digest.update(json.dumps(configuration, sort_keys=True, allow_nan=False).encode())
    for prefix, module in (("backbone", backbone), ("adapter", adapter)):
        if any(m.training for m in module.modules()):
            raise ValueError("Every component must be in evaluation mode")
        for name, value in sorted(module.state_dict().items()):
            if isinstance(value, torch.nn.parameter.UninitializedParameter):
                raise ValueError("Lazy parameters must be initialized before binding")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("Model state must be finite")
            item = value.detach().cpu().contiguous()
            digest.update(
                repr((prefix, name, str(item.dtype), tuple(item.shape))).encode()
            )
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class SafeOnlineAdapter:
    """Evaluation-only wrapper: predict with no targets; ingest explicit reveals."""

    def __init__(
        self, backbone, adapter, *, source_sha256, split, ledger=None, recorder=None
    ):
        self.backbone, self.adapter = backbone, adapter
        self.source_sha256 = ev.digest(source_sha256)
        self.split = ev.identifier(split)
        if len(adapter.bank):
            raise ValueError("A fresh bank is required for a bound source and split")
        self.ledger = ledger if ledger is not None else ObservationLedger()
        self.recorder = recorder
        self._state = _state_digest(backbone, adapter)
        self._bank_state = adapter.bank.checksum()
        self._poisoned = False
        self.write_receipts = []

    def _check(self):
        if self._poisoned:
            raise RuntimeError(
                "Bridge poisoned after an operation failure; no retry permitted"
            )
        if _state_digest(self.backbone, self.adapter) != self._state:
            raise ValueError("Model or normalizer changed after binding")
        if self.adapter.bank.checksum() != self._bank_state:
            raise ValueError("Bank mutated outside this observation bridge")

    @torch.no_grad()
    def predict(self, inputs, *, forecast_ids, now, samples=None):
        self._check()
        self.ledger.check_prediction_time(now)
        if not isinstance(inputs, ForecastInputs) or inputs.split != self.split:
            raise ValueError("Target-free inputs in the bound split required")
        shape = tuple(ev.integer(x, "residual shape", 1) for x in inputs.residual_shape)
        if shape != self.adapter.bank.residual_shape:
            raise ValueError("Output shape must match the declared memory bank")
        batch_size = int(inputs.context.shape[0])
        if len(forecast_ids) != batch_size or len(set(forecast_ids)) != batch_size:
            raise ValueError("Unique forecast ID for each row required")
        if self.recorder is not None:
            if samples is None or len(samples) != batch_size:
                raise ValueError(
                    "Configured recorder requires one prospective sample per row"
                )
            for sample, owner in zip(samples, inputs.trajectory_ids, strict=True):
                self.recorder._key(sample)
                if sample["trajectory_id"] != owner or sample["origin"] != now:
                    raise ValueError(
                        "Export identities must match actual owners and origins"
                    )
            if self.recorder._plan["source_sha256"] != self.source_sha256:
                raise ValueError("Recorder and bridge source identities differ")
        elif samples is not None:
            raise ValueError("Samples require a configured recorder")
        if inputs.event_times.shape != (batch_size,) or inputs.lead_times.shape != (
            batch_size,
        ):
            raise ValueError("Event and lead times must have shape [B]")
        if (
            inputs.event_times.dtype != torch.long
            or inputs.lead_times.dtype != torch.long
        ):
            raise ValueError("Integer tick tensors required")
        if not bool((inputs.event_times == now).all().item()) or not bool(
            (inputs.lead_times > 0).all().item()
        ):
            raise ValueError("All origins equal now and all lead times are positive")
        # The historical batch type requires target shape, but never sees scorer truth.
        batch = ForecastBatch(
            context=inputs.context,
            target=torch.zeros(
                (batch_size, *shape),
                device=inputs.context.device,
                dtype=inputs.context.dtype,
            ),
            observation_mask=inputs.observation_mask,
            trajectory_ids=inputs.trajectory_ids,
            event_times=inputs.event_times,
            lead_times=inputs.lead_times,
            split=inputs.split,
            physical_parameters=inputs.physical_parameters,
        )
        batch.validate()
        try:
            output = self.backbone.forward_batch(batch)
            prediction = self.adapter(output, batch)
            if not bool(torch.isfinite(prediction.prediction).all().item()):
                raise ValueError("Adapted prediction must be finite")
            keys = self.adapter.key_encoder(output, batch)
            self._check()
            forecasts = [
                Forecast(
                    forecast_ids[i],
                    inputs.trajectory_ids[i],
                    now,
                    (now + int(inputs.lead_times[i].item()),) * math.prod(shape),
                    tuple(output.prediction[i].detach().cpu().reshape(-1).tolist()),
                    tuple(keys[i].detach().cpu().reshape(-1).tolist()),
                    self._state,
                )
                for i in range(batch_size)
            ]
            self.ledger.register_many(forecasts, now=now)
            if self.recorder is not None:
                self.record_predictions(self.recorder, samples, prediction)
            return prediction
        except BaseException:
            self._poisoned = True
            raise

    @torch.no_grad()
    def observe(self, event, *, now):
        self._check()
        ready = self.ledger.observe(event, now=now)
        if ready is None:
            return None
        bank = self.adapter.bank
        try:
            # Arithmetic uses the original base and only fully observed coordinates.
            device = self.adapter.normalizer.channel_scale.device
            dtype = self.adapter.normalizer.channel_scale.dtype
            residual = torch.tensor(ready.residual, device=device, dtype=dtype).reshape(
                1, *bank.residual_shape
            )
            normalized = self.adapter.normalizer.normalize(residual)
            if not bool(torch.isfinite(normalized).all().item()):
                raise ValueError("Normalized residual must be finite")
            before = bank.checksum()
            bank.write_batch(
                keys=torch.tensor([ready.forecast.key], device=device, dtype=dtype),
                residuals=normalized,
                trajectory_ids=(ready.forecast.owner,),
                event_times=torch.tensor(
                    [ready.forecast.origin], device=device, dtype=torch.long
                ),
                available_times=torch.tensor(
                    [ready.available_at], device=device, dtype=torch.long
                ),
            )
            self._bank_state = bank.checksum()
            receipt = {
                "forecast_id": ready.forecast.forecast_id,
                "owner": ready.forecast.owner,
                "available_at": ready.available_at,
                "source_sha256": self.source_sha256,
                "state_sha256": self._state,
                "bank_before": before,
                "bank_after": self._bank_state,
                "ledger_sha256": self.ledger.checksum(),
            }
            self.write_receipts.append(receipt)
            return dict(receipt)
        except BaseException:
            self._poisoned = True
            raise

    def record_predictions(self, recorder, samples, prediction):
        """Export actual candidate/base tensors; scorer truth is a separate call."""
        self._check()
        if len(samples) != prediction.prediction.shape[0]:
            raise ValueError("Export sample count mismatch")
        for i, sample in enumerate(samples):
            recorder.record_prediction(
                recorder._methods[0],
                sample,
                prediction.prediction[i].detach().cpu().reshape(-1).tolist(),
            )
            recorder.record_prediction(
                recorder._methods[1],
                sample,
                prediction.backbone_prediction[i].detach().cpu().reshape(-1).tolist(),
            )
