"""Bounded synthetic execution fixture, never a public scientific benchmark.

This module is imported only after the stdlib entry point admits resources and
the requested device. It reuses the existing MLP, trainer, and tensor memory;
there is no learned retrieval head or new architecture in this fixture.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .backbones import ResidualMLPControl
from .config import TrainingConfig
from .contracts import ForecastBatch
from .memory import TensorMemoryBank
from .provenance import JsonlLogger, atomic_write_json, sha256_file
from .training import (
    UnifiedTrainer,
    _atomic_torch_save,
    _batch_fingerprint,
    move_batch,
    seed_everything,
)

FIXTURE_VERSION = "bounded-periodic-field-v1"
VARIANTS = ("base", "memory", "erased", "spatial_mean", "last_residual")
ALPHA_GRID = (0.0, 0.25, 0.5, 1.0)
SPLIT_OFFSETS = {"train": 10_003, "validation": 20_003, "test": 30_007}


def _verify_identities(
    config: Mapping[str, Any],
    source_sha256: str,
    config_sha256: str,
    data_sha256: str,
) -> None:
    from .engineering import data_identity, digest, resolve, source_identity

    if resolve(dict(config)) != dict(config):
        raise ValueError("runtime requires a fully resolved immutable configuration")
    if digest(dict(config)) != config_sha256:
        raise ValueError("runtime configuration identity mismatch")
    if data_identity(config) != data_sha256:
        raise ValueError("runtime data identity mismatch")
    if source_identity()["sha256"] != source_sha256:
        raise ValueError("runtime source identity mismatch")


def _validate_runtime_config(config: Mapping[str, Any]) -> None:
    for name in (
        "epochs",
        "batch_size",
        "train_trajectories",
        "validation_trajectories",
        "test_trajectories",
        "context_steps",
        "sequence_steps",
        "channels",
        "spatial_points",
        "hidden",
        "gradient_accumulation",
        "wall_seconds",
        "example_budget",
        "memory_capacity",
        "memory_topk",
        "threads",
    ):
        if (
            isinstance(config[name], bool)
            or not isinstance(config[name], int)
            or config[name] < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if config["sequence_steps"] <= config["context_steps"]:
        raise ValueError("sequence_steps must exceed context_steps")
    if config["variant"] not in VARIANTS:
        raise ValueError("unknown engineering memory variant")
    if config["device"] not in {"cpu", "cuda"}:
        raise ValueError("engineering fixture supports explicit cpu or cuda only")
    if config["precision"] not in {"fp32", "bf16", "fp16"}:
        raise ValueError("unknown precision")
    if config["device"] == "cpu" and config["precision"] != "fp32":
        raise ValueError("CPU engineering fixture requires fp32")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable; CPU fallback is forbidden")
    if (
        config["device"] == "cuda"
        and config["precision"] == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("requested CUDA bf16 is unavailable")
    for name in ("learning_rate", "checkpoint_seconds", "memory_temperature"):
        if not math.isfinite(float(config[name])) or float(config[name]) <= 0:
            raise ValueError(f"{name} must be finite and positive")


def generate_fixture(config: Mapping[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    """Return raw CPU [time, channel, space] fields with disjoint owners.

    Split RNGs are independent of the global training RNG and one another. No
    fitted normalization, estimated physical parameter, or outcome-based
    selection occurs. All coefficients below are fixed engineering choices.
    """
    result: dict[str, dict[str, torch.Tensor]] = {}
    space = torch.arange(int(config["spatial_points"]), dtype=torch.float32) * (
        2 * math.pi / int(config["spatial_points"])
    )
    times = torch.arange(int(config["sequence_steps"]), dtype=torch.float32)
    for split, offset in SPLIT_OFFSETS.items():
        generator = torch.Generator(device="cpu").manual_seed(
            int(config["seed"]) + offset
        )
        trajectories: dict[str, torch.Tensor] = {}
        for owner in range(int(config[f"{split}_trajectories"])):
            phase = (
                2 * math.pi * torch.rand(int(config["channels"]), generator=generator)
            )
            amplitude = 0.3 + 0.3 * torch.rand(
                int(config["channels"]), generator=generator
            )
            drift = 0.04 + 0.08 * torch.rand(
                int(config["channels"]), generator=generator
            )
            position = space[None, None, :] + phase[None, :, None]
            clock = times[:, None, None] * drift[None, :, None]
            periodic = amplitude[None, :, None] * torch.sin(position + clock)
            secondary = 0.15 * torch.cos(2 * position - 0.7 * clock)
            innovations = 0.01 * torch.randn(periodic.shape, generator=generator)
            values = torch.tanh(periodic + secondary + innovations).contiguous()
            trajectories[f"{split}/trajectory-{owner:04d}"] = values
        result[split] = trajectories
    return result


def raw_data_sha256(fixture: Mapping[str, Mapping[str, torch.Tensor]]) -> str:
    """Hash actual generated values separately from the generation-spec hash."""
    digest = hashlib.sha256(FIXTURE_VERSION.encode())
    for split in SPLIT_OFFSETS:
        for owner, values in fixture[split].items():
            digest.update(owner.encode())
            digest.update(str(tuple(values.shape)).encode())
            digest.update(str(values.dtype).encode())
            digest.update(values.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_batches(
    fixture: Mapping[str, Mapping[str, torch.Tensor]],
    config: Mapping[str, Any],
    split: str,
    *,
    batch_size: int | None = None,
) -> tuple[ForecastBatch, ...]:
    """Canonical owner-major, increasing forecast-origin order, without shuffle."""
    size = int(config["batch_size"] if batch_size is None else batch_size)
    length = int(config["context_steps"])
    examples = [
        (owner, event, values[event - length + 1 : event + 1], values[event + 1])
        for owner, values in fixture[split].items()
        for event in range(length - 1, len(values) - 1)
    ]
    batches: list[ForecastBatch] = []
    for start in range(0, len(examples), size):
        group = examples[start : start + size]
        context = torch.stack([row[2] for row in group])
        target = torch.stack([row[3] for row in group])
        events = torch.tensor([row[1] for row in group], dtype=torch.long)
        batch = ForecastBatch(
            context=context,
            target=target,
            observation_mask=torch.ones_like(context, dtype=torch.bool),
            trajectory_ids=tuple(row[0] for row in group),
            event_times=events,
            lead_times=torch.ones_like(events),
            split=split,
            target_available_times=events + 1,
            target_observation_mask=torch.ones_like(target, dtype=torch.bool),
            provenance_ids=tuple(
                f"{FIXTURE_VERSION}/{row[0]}/{row[1]}" for row in group
            ),
        )
        batch.validate()
        batches.append(batch)
    return tuple(batches)


def build_model(config: Mapping[str, Any]) -> ResidualMLPControl:
    model = ResidualMLPControl(
        input_dimension=int(
            config["context_steps"] * config["channels"] * config["spatial_points"]
        ),
        output_shape=(int(config["channels"]), int(config["spatial_points"])),
        hidden=int(config["hidden"]),
    )
    # output_shape is reconstructed from the identity-checked configuration;
    # this existing module carries no mutable memory outside its state_dict.
    model.checkpoint_stateless = True
    return model


def forecast_step(model: nn.Module, batch: ForecastBatch) -> torch.Tensor:
    """Sample-mean scalar loss, allowing exact example-weighted accumulation."""
    prediction = model(batch.context)
    if prediction.shape != batch.target.shape:
        raise ValueError("engineering prediction/target shape mismatch")
    return (
        (prediction.float() - batch.target.float()).square().flatten(1).mean(1).mean()
    )


def _bank(config: Mapping[str, Any], variant: str) -> TensorMemoryBank:
    return TensorMemoryBank(
        key_dimension=int(
            config["context_steps"] * config["channels"] * config["spatial_points"]
        ),
        residual_shape=(int(config["channels"]), int(config["spatial_points"])),
        capacity_per_trajectory=int(config["memory_capacity"]),
        minimum_age=1,
        topk=1 if variant == "last_residual" else int(config["memory_topk"]),
        temperature=float(config["memory_temperature"]),
        device=str(config["device"]),
        seed=int(config["seed"]),
    )


def _metric(squared_error_sum: float, element_count: int) -> dict[str, float | int]:
    if element_count < 1 or not math.isfinite(squared_error_sum):
        raise FloatingPointError("nonfinite or empty engineering metric")
    return {
        "squared_error_sum": squared_error_sum,
        "element_count": element_count,
        "mse": squared_error_sum / element_count,
    }


@torch.no_grad()
def evaluate_stream(
    model: nn.Module,
    batches: Sequence[ForecastBatch],
    config: Mapping[str, Any],
    *,
    variant: str,
    alpha: float = 1.0,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One fresh split-local bank, persistent through each owner's chronology.

    A forecast is committed before its target is used for a write. The write's
    availability is the next observation time; future retrieval enforces this
    time and same-owner filtering. Keys contain only the observed raw context.
    """
    if variant not in VARIANTS:
        raise ValueError("unknown engineering memory variant")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("memory alpha must be finite and in [0,1]")
    model.eval()
    bank = None if variant == "base" else _bank(config, variant)
    device = torch.device(str(config["device"]))
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        config["precision"]
    ]
    totals = {name: 0.0 for name in ("selected", "base", "persistence")}
    count = 0
    records: list[dict[str, Any]] = []
    last_events: dict[str, int] = {}
    split: str | None = None
    retrieval_hits = 0
    for original in batches:
        original.validate()
        if not bool((original.observation_mask == 1).all().item()):
            raise ValueError("engineering stream requires fully observed context")
        if original.target_observation_mask is None or not bool(
            (original.target_observation_mask == 1).all().item()
        ):
            raise ValueError(
                "engineering stream requires fully observed residual targets"
            )
        if len(original.trajectory_ids) != 1:
            raise ValueError(
                "online engineering evaluation requires singleton chronological batches"
            )
        if split is not None and original.split != split:
            raise ValueError("a memory bank cannot span evaluation splits")
        split = original.split
        owner = original.trajectory_ids[0]
        event = int(original.event_times.item())
        if event <= last_events.get(owner, -1):
            raise ValueError("evaluation events must increase for each owner")
        last_events[owner] = event
        if (
            original.target_available_times is None
            or int(original.target_available_times.item()) != event + 1
        ):
            raise ValueError("engineering targets must arrive at the next observation")
        batch = move_batch(original, device)
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda" and dtype != torch.float32,
        ):
            base = model(batch.context).float()
        if base.shape != batch.target.shape:
            raise ValueError("engineering prediction/target shape mismatch")
        key = F.normalize(batch.context.float().flatten(1), dim=1, eps=1e-8)
        correction = torch.zeros_like(base)
        eligible_count = 0
        source_event = source_available = -1
        bank_size_before = 0 if bank is None else len(bank)
        if bank is not None:
            retrieval = bank.retrieve_batch(
                query_keys=key,
                trajectory_ids=batch.trajectory_ids,
                query_times=batch.event_times,
                selection="last_residual" if variant == "last_residual" else "cosine",
                value_intervention="erase" if variant == "erased" else "none",
            )
            correction = retrieval.residual
            if variant == "spatial_mean":
                # Keep channels distinct; remove only residual spatial geometry.
                correction = correction.mean(dim=-1, keepdim=True).expand_as(correction)
            eligible_count = int(retrieval.eligible_count.item())
            source_event = int(retrieval.source_event_time.item())
            source_available = int(retrieval.source_available_time.item())
            retrieval_hits += int(retrieval.valid.item())
            if bool(retrieval.valid.item()) and not (
                source_event < event and source_available <= event
            ):
                raise RuntimeError("memory retrieval violated causal availability")
        prediction = base + alpha * correction
        persistence = batch.context[:, -1].float()
        target = batch.target.float()
        for name, values in (
            ("selected", prediction),
            ("base", base),
            ("persistence", persistence),
        ):
            if not bool(torch.isfinite(values).all().item()):
                raise FloatingPointError(f"nonfinite {name} predictions")
            totals[name] += float(
                (values.double() - target.double()).square().sum().item()
            )
        count += target.numel()
        # This is the first target-dependent operation in the memory path.
        # Store full base residuals, never residuals of an earlier correction.
        if bank is not None:
            bank.write_batch(
                keys=key,
                residuals=target - base,
                trajectory_ids=batch.trajectory_ids,
                event_times=batch.event_times,
                available_times=batch.target_available_times,
            )
        records.append(
            {
                "owner": owner,
                "split": split,
                "event_time": event,
                "target_time": event + 1,
                "variant": variant,
                "alpha": alpha,
                "prediction": prediction[0].cpu(),
                "target": target[0].cpu(),
                "base_prediction": base[0].cpu(),
                "persistence_prediction": persistence[0].cpu(),
                "correction": correction[0].cpu(),
                "context_key": key[0].cpu(),
                "eligible_count": eligible_count,
                "source_event_time": source_event,
                "source_available_time": source_available,
                "bank_size_before_write": bank_size_before,
                "bank_size_after_write": 0 if bank is None else len(bank),
            }
        )
        if progress is not None and len(records) % 100 == 0:
            progress(
                {
                    "forecast_count": len(records),
                    "retrieval_hits": retrieval_hits,
                    "selected": _metric(totals["selected"], count),
                }
            )
    metrics: dict[str, Any] = {
        name: _metric(value, count) for name, value in totals.items()
    }
    metrics.update(
        {
            "forecast_count": len(records),
            "retrieval_hits": retrieval_hits,
            "variant": variant,
            "split": split,
            "alpha": alpha,
        }
    )
    return metrics, records


def _select_validation_alpha(
    records: Sequence[Mapping[str, Any]],
) -> tuple[float, list[dict[str, float | int]]]:
    """Choose one predeclared scale using validation residuals only."""
    if not records or any(row["split"] != "validation" for row in records):
        raise ValueError("memory scale selection requires validation records only")
    scores: list[dict[str, float | int]] = []
    for alpha in ALPHA_GRID:
        error = 0.0
        count = 0
        for row in records:
            prediction = row["base_prediction"] + alpha * row["correction"]
            target = row["target"]
            error += float(
                (prediction.double() - target.double()).square().sum().item()
            )
            count += target.numel()
        scores.append({"alpha": alpha, **_metric(error, count)})
    chosen = min(scores, key=lambda row: (float(row["mse"]), float(row["alpha"])))
    return float(chosen["alpha"]), scores


def _apply_validation_alpha(
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    alpha: float,
) -> None:
    error = 0.0
    count = 0
    for row in records:
        row["alpha"] = alpha
        row["prediction"] = row["base_prediction"] + alpha * row["correction"]
        error += float(
            (row["prediction"].double() - row["target"].double()).square().sum().item()
        )
        count += row["target"].numel()
    summary["selected"] = _metric(error, count)
    summary["alpha"] = alpha


@torch.no_grad()
def autonomous_base_diagnostic(
    model: nn.Module,
    test_trajectories: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Finite-horizon base rollout using only initial observations and predictions.

    Future targets are used only to score already committed predictions. Memory
    is disabled because the unobserved future cannot supply residual writes.
    This is a diagnostic on this fixture horizon, not a stability certificate.
    """
    model.eval()
    length = int(config["context_steps"])
    device = torch.device(str(config["device"]))
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        config["precision"]
    ]
    records: list[dict[str, Any]] = []
    errors: dict[int, float] = {}
    counts: dict[int, int] = {}
    maximum_absolute_prediction = 0.0
    for owner, values in test_trajectories.items():
        context = values[:length].unsqueeze(0).to(device).clone()
        for target_index in range(length, len(values)):
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda" and dtype != torch.float32,
            ):
                prediction = model(context).float()
            if prediction.shape != (
                1,
                int(config["channels"]),
                int(config["spatial_points"]),
            ):
                raise ValueError("autonomous prediction shape mismatch")
            if not bool(torch.isfinite(prediction).all().item()):
                raise FloatingPointError("nonfinite autonomous base prediction")
            prediction_cpu = prediction[0].cpu()
            # Target access follows prediction; it never changes subsequent context.
            target = values[target_index].cpu()
            horizon = target_index - length + 1
            errors[horizon] = errors.get(horizon, 0.0) + float(
                (prediction_cpu.double() - target.double()).square().sum().item()
            )
            counts[horizon] = counts.get(horizon, 0) + target.numel()
            maximum_absolute_prediction = max(
                maximum_absolute_prediction, float(prediction.abs().max().item())
            )
            records.append(
                {
                    "owner": owner,
                    "split": "test",
                    "horizon": horizon,
                    "target_time": target_index,
                    "initial_observed_steps": length,
                    "context": context[0].cpu(),
                    "prediction": prediction_cpu,
                    "target": target,
                    "memory_writes": 0,
                }
            )
            if progress is not None and len(records) % 100 == 0:
                progress(
                    {"forecast_count": len(records), "owner": owner, "horizon": horizon}
                )
            context = torch.cat((context[:, 1:], prediction.unsqueeze(1)), dim=1)
    summary = {
        "split": "test",
        "mode": "autonomous_base_diagnostic",
        "selection": "diagnostic_only_not_used_for_selection",
        "memory": "disabled_no_unobserved_residual_writes",
        "all_predictions_finite": True,
        "initial_observed_steps": length,
        "horizon_steps": max(counts, default=0),
        "owner_count": len(test_trajectories),
        "forecast_count": len(records),
        "maximum_absolute_prediction": maximum_absolute_prediction,
        "overall": _metric(sum(errors.values()), sum(counts.values())),
        "per_horizon": {
            str(horizon): _metric(errors[horizon], counts[horizon])
            for horizon in sorted(errors)
        },
    }
    return summary, records


def _load_frozen_model(
    config: Mapping[str, Any],
    checkpoint: str | Path,
    source_sha256: str,
    config_sha256: str,
    data_sha256: str,
) -> tuple[nn.Module, Mapping[str, Any]]:
    # Fail closed on old torch: never fall back to unrestricted pickle loading.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must have a dictionary envelope")
    if payload.get("schema_version") != 2:
        raise ValueError("engineering evaluation requires safe checkpoint schema 2")
    runtime = payload.get("runtime", {})
    if (
        runtime.get("torch") != str(torch.__version__)
        or runtime.get("precision") != config["precision"]
        or str(runtime.get("device")).split(":")[0] != config["device"]
    ):
        raise ValueError(
            "frozen evaluation requires matching Torch, device and precision"
        )
    if not isinstance(payload.get("batch_sha256"), str):
        raise ValueError("frozen checkpoint is missing actual training data identity")
    for name, expected in (
        ("source_sha256", source_sha256),
        ("config_sha256", config_sha256),
        ("data_sha256", data_sha256),
    ):
        if payload.get(name) != expected:
            raise ValueError(f"checkpoint {name} identity mismatch")
    if not isinstance(payload.get("model_state"), dict):
        raise ValueError("checkpoint missing model_state")
    model = build_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(str(config["device"]))
    model.eval()
    model.requires_grad_(False)
    return model, payload


def evaluate_checkpoint(
    config: Mapping[str, Any],
    checkpoint: str | Path,
    output_directory: str | Path,
    source_sha256: str,
    config_sha256: str,
    data_sha256: str,
) -> dict[str, Any]:
    """Frozen checkpoint replay and all predeclared synthetic diagnostic controls."""
    _verify_identities(config, source_sha256, config_sha256, data_sha256)
    _validate_runtime_config(config)
    torch.set_num_threads(int(config["threads"]))
    seed_everything(int(config["seed"]))
    model, checkpoint_payload = _load_frozen_model(
        config, checkpoint, source_sha256, config_sha256, data_sha256
    )
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    checkpoint_digest = sha256_file(checkpoint)
    event_context = {
        "benchmark": str(config["benchmark"]),
        "model": "ResidualMLPControl",
        "seed": int(config["seed"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
    }
    logger = JsonlLogger(destination / "events.jsonl", **event_context)

    def event(name: str, **fields: Any) -> None:
        data = {
            "time_unix_ns": time.time_ns(),
            "elapsed_seconds": time.monotonic() - started,
            **fields,
        }
        logger.log(name, **data)
        print(
            json.dumps({"event": name, **event_context, **data}, sort_keys=True),
            flush=True,
        )

    event("evaluation_start", phase="generate_fixture")
    fixture = generate_fixture(config)
    if (
        _batch_fingerprint(
            make_batches(fixture, config, "train"),
            make_batches(fixture, config, "validation"),
        )
        != checkpoint_payload["batch_sha256"]
    ):
        raise ValueError(
            "regenerated data differs from frozen checkpoint training data"
        )
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_ENGINEERING",
        "evidence_class": "SYNTHETIC_ENGINEERING_FIXTURE",
        "scientific_claim": "NOT_EVALUATED",
        "fixture_version": FIXTURE_VERSION,
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "raw_data_sha256": raw_data_sha256(fixture),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_step": int(checkpoint_payload["step"]),
        "selected_variant": str(config["variant"]),
        "selection_rule": "caller-supplied frozen checkpoint; each fixed control scale selected on validation only",
        "alpha_grid": list(ALPHA_GRID),
        "retrieval": "fixed normalized observed-context cosine; no learned query or gate",
        "normalization": "none; key L2 normalization is per observed context only",
        "ablations": {},
    }
    raw_data_artifact = destination / "raw_data.pt"
    _atomic_torch_save(
        raw_data_artifact,
        {
            "schema_version": 1,
            "evidence_class": "SYNTHETIC_ENGINEERING_FIXTURE",
            "fixture_version": FIXTURE_VERSION,
            "config": dict(config),
            "source_sha256": source_sha256,
            "config_sha256": config_sha256,
            "data_sha256": data_sha256,
            "raw_data_sha256": metrics["raw_data_sha256"],
            "trajectories": fixture,
        },
    )
    metrics["raw_data_path"] = str(raw_data_artifact)
    metrics["raw_data_artifact_sha256"] = sha256_file(raw_data_artifact)
    records: list[dict[str, Any]] = []
    for variant in VARIANTS:
        event("evaluation_split_start", phase="validation", variant=variant)
        validation, validation_raw = evaluate_stream(
            model,
            make_batches(fixture, config, "validation", batch_size=1),
            config,
            variant=variant,
            progress=lambda fields: event(
                "evaluation_progress", phase="validation", variant=variant, **fields
            ),
        )
        alpha, alpha_scores = _select_validation_alpha(validation_raw)
        _apply_validation_alpha(validation, validation_raw, alpha)
        event("evaluation_split_complete", phase="validation", **validation)
        event("evaluation_split_start", phase="test", variant=variant, alpha=alpha)
        test, test_raw = evaluate_stream(
            model,
            make_batches(fixture, config, "test", batch_size=1),
            config,
            variant=variant,
            alpha=alpha,
            progress=lambda fields: event(
                "evaluation_progress", phase="test", variant=variant, **fields
            ),
        )
        event("evaluation_split_complete", phase="test", **test)
        metrics["ablations"][variant] = {
            "validation": validation,
            "test": test,
            "alpha": alpha,
            "alpha_validation_scores": alpha_scores,
        }
        records.extend(validation_raw)
        records.extend(test_raw)
    metrics["selected"] = metrics["ablations"][str(config["variant"])]
    event("autonomous_base_start", phase="test_rollout")
    autonomous, autonomous_raw = autonomous_base_diagnostic(
        model,
        fixture["test"],
        config,
        progress=lambda fields: event(
            "evaluation_progress", phase="test_rollout", **fields
        ),
    )
    metrics["autonomous_base_diagnostic"] = autonomous
    event("autonomous_base_complete", phase="test_rollout", **autonomous)
    artifact = destination / "predictions.pt"
    _atomic_torch_save(
        artifact,
        {
            "schema_version": 1,
            "evidence_class": metrics["evidence_class"],
            "source_sha256": source_sha256,
            "config_sha256": config_sha256,
            "data_sha256": data_sha256,
            "raw_data_sha256": metrics["raw_data_sha256"],
            "checkpoint_sha256": metrics["checkpoint_sha256"],
            "records": records,
            "autonomous_base_diagnostic": autonomous_raw,
        },
    )
    metrics["predictions_path"] = str(artifact)
    metrics["predictions_sha256"] = sha256_file(artifact)
    atomic_write_json(destination / "metrics.json", metrics, overwrite=False)
    event(
        "evaluation_complete",
        phase="export",
        status="PASS_ENGINEERING",
        predictions_sha256=metrics["predictions_sha256"],
        forecast_records=len(records),
    )
    return metrics


def run_experiment(
    config: Mapping[str, Any],
    output_root: Path,
    run_id: str,
    source_sha256: str,
    config_sha256: str,
    data_sha256: str,
    admission: Callable[[], Mapping[str, Any]],
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Train the existing small MLP, then evaluate only its validation selection."""
    _verify_identities(config, source_sha256, config_sha256, data_sha256)
    _validate_runtime_config(config)
    initial_admission = admission()
    if initial_admission.get("status") != "PASS":
        run_directory = Path(output_root) / run_id
        run_directory.mkdir(parents=True, exist_ok=False)
        receipt = run_directory / "resource_block.json"
        atomic_write_json(receipt, initial_admission, overwrite=False)
        return {
            "status": "BLOCKED_RESOURCE",
            "best_validation_loss": None,
            "epochs_completed": 0,
            "steps_completed": 0,
            "examples_seen": 0,
            "elapsed_seconds": 0.0,
            "checkpoint": None,
            "failure_receipt": str(receipt),
            "evidence_class": "SYNTHETIC_ENGINEERING_FIXTURE",
            "scientific_claim": "NOT_EVALUATED",
        }
    torch.set_num_threads(int(config["threads"]))
    seed_everything(int(config["seed"]))
    fixture = generate_fixture(config)
    model = build_model(config).to(str(config["device"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    training = TrainingConfig(
        mode="joint",
        gate="full",
        seed=int(config["seed"]),
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        gradient_accumulation=int(config["gradient_accumulation"]),
        precision=str(config["precision"]),
        early_stopping_patience=int(config["epochs"]) + 1,
        checkpoint_every_steps=100,
        checkpoint_every_seconds=float(config["checkpoint_seconds"]),
        wall_clock_budget_seconds=int(config["wall_seconds"]),
        example_budget=int(config["example_budget"]),
    )
    trainer = UnifiedTrainer(
        model=model,
        optimizer=optimizer,
        config=training,
        run_id=run_id,
        output_root=output_root,
        source_sha256=source_sha256,
        config_sha256=config_sha256,
        data_sha256=data_sha256,
        resource_admission=admission,
        device=str(config["device"]),
    )
    outcome = trainer.fit(
        train_batches=make_batches(fixture, config, "train"),
        validation_batches=make_batches(fixture, config, "validation"),
        step=forecast_step,
        resume_from=resume_from,
    )
    result: dict[str, Any] = dataclasses.asdict(outcome)
    result.update(
        {
            "evidence_class": "SYNTHETIC_ENGINEERING_FIXTURE",
            "scientific_claim": "NOT_EVALUATED",
            "raw_data_sha256": raw_data_sha256(fixture),
        }
    )
    # Release training arrays, optimizer state, and embedded best snapshot before
    # allocating the frozen replay and its retained exports.
    del trainer, optimizer, model, fixture
    if outcome.status != "PASS_ENGINEERING":
        return result
    best_checkpoint = Path(output_root) / run_id / "checkpoints" / "best.pt"
    if not best_checkpoint.is_file():
        raise RuntimeError(
            "completed trainer did not save its validation-selected checkpoint"
        )
    result["best_checkpoint"] = str(best_checkpoint)
    result["checkpoint_selection"] = (
        "lowest base-model validation MSE saved by UnifiedTrainer"
    )
    readmission = admission()
    if readmission.get("status") != "PASS":
        receipt = Path(output_root) / run_id / "evaluation_resource_block.json"
        atomic_write_json(receipt, readmission, overwrite=False)
        result.update({"status": "BLOCKED_RESOURCE", "failure_receipt": str(receipt)})
        return result
    result["evaluation"] = evaluate_checkpoint(
        config,
        best_checkpoint,
        Path(output_root) / run_id / "evaluation",
        source_sha256,
        config_sha256,
        data_sha256,
    )
    return result
