from __future__ import annotations

import dataclasses
import os
import platform
import random
import signal
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import TrainingConfig
from .contracts import ForecastBatch
from .provenance import CHECKPOINT_SCHEMA_VERSION, JsonlLogger, atomic_write_json


@dataclasses.dataclass(frozen=True)
class TrainResult:
    status: str
    best_validation_loss: float | None
    epochs_completed: int
    steps_completed: int
    examples_seen: int
    elapsed_seconds: float
    checkpoint: str | None
    failure_receipt: str | None


def seed_everything(seed: int, deterministic: bool = True) -> Mapping[str, Any]:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return {
        "seed": int(seed),
        "deterministic_algorithms": deterministic,
        "torch_initial_seed": int(torch.initial_seed()),
    }


def random_state() -> Mapping[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def restore_random_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _trajectory_gate(config: TrainingConfig, batches: Sequence[ForecastBatch]) -> None:
    limits = {
        "one_sequence": 1,
        "eight_trajectory": 8,
        "sixty_four_trajectory": 64,
        "full": None,
    }
    identities = {identity for batch in batches for identity in batch.trajectory_ids}
    limit = limits[config.gate]
    if limit is not None and len(identities) > limit:
        raise ValueError(
            f"{config.gate} gate admits at most {limit} trajectories, "
            f"got {len(identities)}"
        )
    if config.gate != "full" and len(identities) < 1:
        raise ValueError("training gate has no trajectories")


def configure_training_mode(
    *,
    backbone: nn.Module,
    adapter: nn.Module,
    mode: str,
) -> list[nn.Parameter]:
    if mode not in {"frozen_backbone", "joint"}:
        raise ValueError(f"unsupported training mode: {mode}")
    backbone_trainable = mode == "joint"
    for parameter in backbone.parameters():
        parameter.requires_grad_(backbone_trainable)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    parameters = [
        parameter
        for module in (backbone, adapter)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("training mode produced no trainable parameters")
    return parameters


class UnifiedTrainer:
    """Budgeted trainer shared by admitted backbones and adapters."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        run_id: str,
        output_root: str | Path,
        source_sha256: str,
        config_sha256: str,
        resource_admission: Callable[[], Mapping[str, Any]],
        device: str | torch.device,
    ) -> None:
        config.validate()
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.run_id = run_id
        self.output_root = Path(output_root) / run_id
        self.source_sha256 = source_sha256
        self.config_sha256 = config_sha256
        self.resource_admission = resource_admission
        self.device = torch.device(device)
        self.logger = JsonlLogger(self.output_root / "events.jsonl", run_id=run_id)
        self._interrupted = False

    def _checkpoint_payload(self, step: int, epoch: int) -> Mapping[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "source_sha256": self.source_sha256,
            "config_sha256": self.config_sha256,
            "step": int(step),
            "epoch": int(epoch),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "random_state": random_state(),
        }

    def save_checkpoint(self, step: int, epoch: int, name: str = "latest.pt") -> Path:
        destination = self.output_root / "checkpoints" / name
        _atomic_torch_save(destination, self._checkpoint_payload(step, epoch))
        self.logger.log(
            "checkpoint_saved", path=str(destination), step=step, epoch=epoch
        )
        return destination

    def resume(self, path: str | Path) -> tuple[int, int]:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema")
        if payload.get("source_sha256") != self.source_sha256:
            raise ValueError("checkpoint source identity mismatch")
        if payload.get("config_sha256") != self.config_sha256:
            raise ValueError("checkpoint configuration identity mismatch")
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        restore_random_state(payload["random_state"])
        return int(payload["step"]), int(payload.get("epoch", 0))

    def fit(
        self,
        *,
        train_batches: Sequence[ForecastBatch],
        validation_batches: Sequence[ForecastBatch],
        step: Callable[[nn.Module, ForecastBatch], torch.Tensor],
        resume_from: str | Path | None = None,
    ) -> TrainResult:
        _trajectory_gate(self.config, train_batches)
        if not validation_batches:
            raise ValueError("validation batches are required for early stopping")
        initial_admission = self.resource_admission()
        if initial_admission.get("status") != "PASS":
            receipt = self.output_root / "resource_block.json"
            atomic_write_json(receipt, initial_admission, overwrite=False)
            return TrainResult(
                status="BLOCKED_RESOURCE",
                best_validation_loss=None,
                epochs_completed=0,
                steps_completed=0,
                examples_seen=0,
                elapsed_seconds=0.0,
                checkpoint=None,
                failure_receipt=str(receipt),
            )
        if self.config.distributed and not torch.distributed.is_initialized():
            raise RuntimeError("distributed mode requires an initialized process group")
        seed_receipt = seed_everything(self.config.seed)
        self.output_root.mkdir(parents=True, exist_ok=False)
        atomic_write_json(
            self.output_root / "environment.json",
            {
                "schema_version": 1,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "device": str(self.device),
                "seed": seed_receipt,
                "budgets": dataclasses.asdict(self.config),
                "energy": "UNAVAILABLE_NO_ADMITTED_METER",
            },
            overwrite=False,
        )
        started = time.monotonic()
        completed_steps = 0
        start_epoch = 0
        examples_seen = 0
        if resume_from is not None:
            completed_steps, start_epoch = self.resume(resume_from)
        self.model.to(self.device)
        best_loss: float | None = None
        best_checkpoint: Path | None = None
        epochs_without_improvement = 0
        precision_dtype = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.config.precision]
        autocast_enabled = self.config.precision != "fp32" and self.device.type in {
            "cuda",
            "mps",
        }
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.config.precision == "fp16" and self.device.type == "cuda",
        )
        previous_handlers: dict[int, Any] = {}

        def request_stop(_signal: int, _frame: object) -> None:
            self._interrupted = True

        for signal_value in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signal_value] = signal.signal(signal_value, request_stop)
        try:
            for epoch in range(start_epoch, self.config.epochs):
                admission = self.resource_admission()
                if admission.get("status") != "PASS":
                    checkpoint = self.save_checkpoint(
                        completed_steps, epoch, "resource-stop.pt"
                    )
                    atomic_write_json(
                        self.output_root / "resource_stop.json",
                        admission,
                        overwrite=False,
                    )
                    return TrainResult(
                        "BLOCKED_RESOURCE",
                        best_loss,
                        epoch,
                        completed_steps,
                        examples_seen,
                        time.monotonic() - started,
                        str(checkpoint),
                        str(self.output_root / "resource_stop.json"),
                    )
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                for batch_index, batch in enumerate(train_batches):
                    if self._interrupted:
                        checkpoint = self.save_checkpoint(
                            completed_steps, epoch, "preemption-safe.pt"
                        )
                        return TrainResult(
                            "PREEMPTED",
                            best_loss,
                            epoch,
                            completed_steps,
                            examples_seen,
                            time.monotonic() - started,
                            str(checkpoint),
                            None,
                        )
                    if (
                        time.monotonic() - started
                        > self.config.wall_clock_budget_seconds
                    ):
                        checkpoint = self.save_checkpoint(
                            completed_steps, epoch, "budget-stop.pt"
                        )
                        return TrainResult(
                            "BUDGET_EXHAUSTED",
                            best_loss,
                            epoch,
                            completed_steps,
                            examples_seen,
                            time.monotonic() - started,
                            str(checkpoint),
                            None,
                        )
                    batch.validate()
                    incoming = int(batch.context.shape[0])
                    if examples_seen + incoming > self.config.example_budget:
                        checkpoint = self.save_checkpoint(
                            completed_steps, epoch, "budget-stop.pt"
                        )
                        return TrainResult(
                            "BUDGET_EXHAUSTED",
                            best_loss,
                            epoch,
                            completed_steps,
                            examples_seen,
                            time.monotonic() - started,
                            str(checkpoint),
                            None,
                        )
                    moved = move_batch(batch, self.device)
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=precision_dtype,
                        enabled=autocast_enabled,
                    ):
                        loss = (
                            step(self.model, moved) / self.config.gradient_accumulation
                        )
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError("non-finite training loss")
                    scaler.scale(loss).backward()
                    should_update = (
                        (batch_index + 1) % self.config.gradient_accumulation == 0
                        or batch_index + 1 == len(train_batches)
                    )
                    if should_update:
                        scaler.unscale_(self.optimizer)
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_norm
                        )
                        if not bool(torch.isfinite(gradient_norm).item()):
                            raise FloatingPointError("non-finite gradient norm")
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        completed_steps += 1
                    examples_seen += incoming
                    if completed_steps and (
                        completed_steps % self.config.checkpoint_every_steps == 0
                    ):
                        self.save_checkpoint(completed_steps, epoch)
                validation_loss = self._validate(validation_batches, step)
                improved = best_loss is None or validation_loss < best_loss
                self.logger.log(
                    "epoch_complete",
                    epoch=epoch,
                    validation_loss=validation_loss,
                    improved=improved,
                )
                if improved:
                    best_loss = validation_loss
                    best_checkpoint = self.save_checkpoint(
                        completed_steps, epoch + 1, "best.pt"
                    )
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if (
                        epochs_without_improvement
                        >= self.config.early_stopping_patience
                    ):
                        break
            return TrainResult(
                "PASS_ENGINEERING",
                best_loss,
                epoch + 1,
                completed_steps,
                examples_seen,
                time.monotonic() - started,
                None if best_checkpoint is None else str(best_checkpoint),
                None,
            )
        except BaseException as error:
            checkpoint: Path | None = None
            try:
                checkpoint = self.save_checkpoint(
                    completed_steps, start_epoch, "failure.pt"
                )
            except BaseException:
                checkpoint = None
            failure_type = (
                "OOM"
                if isinstance(error, torch.OutOfMemoryError)
                else type(error).__name__
            )
            failure = {
                "schema_version": 1,
                "status": "FAILED",
                "failure_type": failure_type,
                "message": str(error),
                "steps_completed": completed_steps,
                "examples_seen": examples_seen,
                "silent_retry": False,
                "settings_changed": False,
            }
            receipt = self.output_root / "failure.json"
            atomic_write_json(receipt, failure, overwrite=False)
            self.logger.log("failed", **failure)
            raise
        finally:
            for signal_value, handler in previous_handlers.items():
                signal.signal(signal_value, handler)

    @torch.no_grad()
    def _validate(
        self,
        batches: Sequence[ForecastBatch],
        step: Callable[[nn.Module, ForecastBatch], torch.Tensor],
    ) -> float:
        self.model.eval()
        losses: list[float] = []
        for batch in batches:
            loss = step(self.model, move_batch(batch, self.device))
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite validation loss")
            losses.append(float(loss.item()))
        return sum(losses) / len(losses)


def move_batch(batch: ForecastBatch, device: torch.device) -> ForecastBatch:
    return dataclasses.replace(
        batch,
        context=batch.context.to(device),
        target=batch.target.to(device),
        observation_mask=batch.observation_mask.to(device),
        event_times=batch.event_times.to(device),
        lead_times=batch.lead_times.to(device),
        target_available_times=(
            None
            if batch.target_available_times is None
            else batch.target_available_times.to(device)
        ),
        target_observation_mask=(
            None
            if batch.target_observation_mask is None
            else batch.target_observation_mask.to(device)
        ),
        physical_parameters=(
            None
            if batch.physical_parameters is None
            else batch.physical_parameters.to(device)
        ),
    )
