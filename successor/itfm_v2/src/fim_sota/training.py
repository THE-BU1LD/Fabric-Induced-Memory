from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import signal
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import TrainingConfig
from .contracts import ForecastBatch
from .provenance import JsonlLogger, atomic_write_json

TRAINING_CHECKPOINT_SCHEMA_VERSION = 2


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
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def restore_random_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _assert_safe_payload(value: Any, path: str = "checkpoint") -> None:
    """Reject arbitrary Python objects before writing a weights-only checkpoint."""
    if (
        value is None
        or type(value) in {bool, int, float, str, bytes}
        or torch.is_tensor(value)
    ):
        return
    if type(value) in {dict, OrderedDict}:
        for key, item in value.items():
            if type(key) not in {str, int}:
                raise TypeError(f"unsupported checkpoint key at {path}")
            _assert_safe_payload(item, f"{path}.{key}")
        return
    if type(value) in {list, tuple}:
        for index, item in enumerate(value):
            _assert_safe_payload(item, f"{path}[{index}]")
        return
    raise TypeError(f"unsafe checkpoint state at {path}: {type(value).__name__}")


def _cpu_snapshot(value: Any) -> Any:
    """Snapshot state without retaining a second model on the accelerator."""
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if type(value) in {dict, OrderedDict}:
        result = type(value)((key, _cpu_snapshot(item)) for key, item in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = _cpu_snapshot(value._metadata)
        return result
    if type(value) is tuple:
        return tuple(_cpu_snapshot(item) for item in value)
    if type(value) is list:
        return [_cpu_snapshot(item) for item in value]
    return value


def _check_model_state_contract(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if module._non_persistent_buffers_set:
            raise ValueError(f"nonpersistent buffers cannot be resumed exactly: {name}")
        extra = type(module).get_extra_state is not nn.Module.get_extra_state
        restore = type(module).set_extra_state is not nn.Module.set_extra_state
        if extra != restore:
            raise ValueError(f"get_extra_state/set_extra_state must be paired: {name}")
        if extra and not callable(getattr(module, "reset_evaluation_state", None)):
            raise ValueError(
                f"stateful module must define reset_evaluation_state(): {name}"
            )
        builtin = type(module).__module__.startswith("torch.nn.")
        if (
            not builtin
            and not extra
            and not getattr(module, "checkpoint_stateless", False)
        ):
            raise ValueError(
                f"custom module {name or '<root>'} must serialize nonparameter state "
                "or explicitly declare checkpoint_stateless=True"
            )
    _assert_safe_payload(model.state_dict(), "model_state")


def _check_loss(loss: torch.Tensor, phase: str) -> None:
    if loss.ndim != 0:
        raise ValueError(f"{phase} step must return a scalar per-example mean")
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError(f"non-finite {phase} loss")


def _batch_fingerprint(*groups: Sequence[ForecastBatch]) -> str:
    digest = hashlib.sha256()
    for group_index, batches in enumerate(groups):
        digest.update(f"group:{group_index}:{len(batches)}\n".encode())
        for batch in batches:
            for field in dataclasses.fields(batch):
                value = getattr(batch, field.name)
                digest.update(field.name.encode() + b"\0")
                if torch.is_tensor(value):
                    tensor = value.detach().cpu().contiguous()
                    digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
                    # uint8 view also handles dtypes NumPy cannot represent directly.
                    digest.update(
                        tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                    )
                else:
                    digest.update(json.dumps(value, sort_keys=True).encode())
                digest.update(b"\0")
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
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
    """Fixed-order, sample-weighted training with optimizer-boundary recovery.

    ``step`` must return a scalar *per-example mean*. It must not mutate data,
    external state, or its own RNG; mutable model state belongs in state_dict
    (including get_extra_state/set_extra_state). Custom modules without extra
    state must explicitly declare ``checkpoint_stateless = True``. Accumulation
    is not claimed equivalent to a large batch for batch/state-dependent models.
    """

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
        data_sha256: str | None = None,
        scheduler: Any | None = None,
    ) -> None:
        config.validate()
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be a single non-empty path component")
        self.model, self.optimizer, self.config = model, optimizer, config
        self.scheduler = scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise ValueError(
                "metric-driven schedulers are unsupported; scheduler.step runs per optimizer update"
            )
        self.run_id = run_id
        self.output_root = Path(output_root) / run_id
        self.source_sha256, self.config_sha256 = source_sha256, config_sha256
        self.data_sha256 = data_sha256
        self.resource_admission = resource_admission
        self.device = torch.device(device)
        self.logger: JsonlLogger | None = None  # No directory mutation in constructor.
        self._interrupted = False
        self._at_boundary = True
        self._started: float | None = None
        self._elapsed_before = 0.0
        self._last_checkpoint_time = 0.0
        self._batch_sha256: str | None = None
        self._best_snapshot: dict[str, Any] | None = None
        self._progress: dict[str, Any] = {
            "epoch": 0,
            "next_batch_index": 0,
            "steps_completed": 0,
            "examples_seen": 0,
            "best_validation_loss": None,
            "epochs_without_improvement": 0,
            "early_stopped": False,
        }
        self._validate_runtime()
        _check_model_state_contract(model)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")

    def _validate_runtime(self) -> None:
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("only explicit CPU and CUDA devices are supported")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            if self.device.index >= torch.cuda.device_count():
                raise ValueError("requested CUDA device does not exist")
            with torch.cuda.device(self.device):
                if (
                    self.config.precision == "bf16"
                    and not torch.cuda.is_bf16_supported()
                ):
                    raise ValueError("requested CUDA device does not support bf16")
        elif self.config.precision != "fp32":
            raise ValueError("CPU training currently supports fp32 only")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            raise ValueError("initialized distributed execution is unsupported")
        if any(p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError("model parameters must use the fp32 reference dtype")
        if any(p.device != self.device for p in self.model.parameters()):
            raise ValueError(
                "move model to the explicit device before creating optimizer"
            )

    def _elapsed(self) -> float:
        return self._elapsed_before + (
            0.0 if self._started is None else time.monotonic() - self._started
        )

    def _runtime_identity(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "precision": self.config.precision,
            "torch": str(torch.__version__),
            "optimizer": type(self.optimizer).__module__
            + "."
            + type(self.optimizer).__qualname__,
            "scheduler": (
                None
                if self.scheduler is None
                else type(self.scheduler).__module__
                + "."
                + type(self.scheduler).__qualname__
            ),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "cuda_device_name": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else None
            ),
        }

    def _log(self, event: str, **fields: Any) -> None:
        payload = {
            "run_id": self.run_id,
            "seed": self.config.seed,
            "elapsed_seconds": self._elapsed(),
            **fields,
        }
        if self.logger is not None:
            self.logger.log(event, **payload)
        print(
            json.dumps(
                {"time_unix_ns": time.time_ns(), "event": event, **payload},
                sort_keys=True,
            ),
            flush=True,
        )

    def _checkpoint_payload(self, step: int, epoch: int) -> dict[str, Any]:
        if self._batch_sha256 is None:
            raise RuntimeError("checkpoint requires a validated fixed batch sequence")
        if not self._at_boundary:
            raise RuntimeError("checkpoint requested inside an accumulation window")
        if (
            step != self._progress["steps_completed"]
            or epoch != self._progress["epoch"]
        ):
            raise ValueError("checkpoint counters must match the safe restart cursor")
        payload = {
            "schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
            "resume_boundary": "completed_optimizer_window_fixed_batch_order",
            "run_id": self.run_id,
            "source_sha256": self.source_sha256,
            "config_sha256": self.config_sha256,
            "data_sha256": self.data_sha256,
            "batch_sha256": self._batch_sha256,
            "training_config": dataclasses.asdict(self.config),
            "runtime": self._runtime_identity(),
            "step": int(step),
            "epoch": int(epoch),
            "progress": dict(self._progress),
            "elapsed_seconds": self._elapsed(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": None
            if self.scheduler is None
            else self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "random_state": random_state(),
            "best_snapshot": self._best_snapshot,
        }
        _assert_safe_payload(payload)
        return payload

    def save_checkpoint(self, step: int, epoch: int, name: str = "latest.pt") -> Path:
        if Path(name).name != name or not name.endswith(".pt"):
            raise ValueError("checkpoint name must be a local .pt filename")
        destination = self.output_root / "checkpoints" / name
        _atomic_torch_save(destination, self._checkpoint_payload(step, epoch))
        self._last_checkpoint_time = self._elapsed()
        self._log("checkpoint_saved", path=str(destination), step=step, epoch=epoch)
        return destination

    def resume(self, path: str | Path) -> tuple[int, int]:
        # No pickle fallback: old/unsafe payloads require an explicit migration.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        _assert_safe_payload(payload)
        if payload.get("schema_version") != TRAINING_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported training checkpoint schema")
        for key in ("source_sha256", "config_sha256", "data_sha256", "batch_sha256"):
            expected = (
                self._batch_sha256 if key == "batch_sha256" else getattr(self, key)
            )
            if payload.get(key) != expected:
                raise ValueError(f"checkpoint {key} identity mismatch")
        if payload.get("training_config") != dataclasses.asdict(self.config):
            raise ValueError("checkpoint training configuration mismatch")
        if payload.get("runtime") != self._runtime_identity():
            raise ValueError(
                "exact resume requires the same device, precision, optimizer, scheduler and runtime"
            )
        if (
            payload.get("resume_boundary")
            != "completed_optimizer_window_fixed_batch_order"
        ):
            raise ValueError("unsupported checkpoint restart boundary")
        if (payload.get("scheduler_state") is None) != (self.scheduler is None):
            raise ValueError("checkpoint scheduler configuration mismatch")
        progress = payload["progress"]
        for key in (
            "epoch",
            "next_batch_index",
            "steps_completed",
            "examples_seen",
            "epochs_without_improvement",
        ):
            if type(progress.get(key)) is not int or progress[key] < 0:
                raise ValueError(f"invalid checkpoint progress: {key}")
        if (
            progress["epoch"] > self.config.epochs
            or progress["examples_seen"] > self.config.example_budget
        ):
            raise ValueError("checkpoint exceeds configured budget")
        if (
            payload.get("epoch") != progress["epoch"]
            or payload.get("step") != progress["steps_completed"]
            or type(progress.get("early_stopped")) is not bool
        ):
            raise ValueError("inconsistent checkpoint progress")
        best_loss = progress.get("best_validation_loss")
        if best_loss is not None and not math.isfinite(best_loss):
            raise ValueError("invalid best validation loss")
        if (
            not math.isfinite(payload["elapsed_seconds"])
            or payload["elapsed_seconds"] < 0
        ):
            raise ValueError("invalid checkpoint elapsed time")
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler_state"])
        self.scaler.load_state_dict(payload["scaler_state"])
        self._progress = dict(progress)
        self._elapsed_before = float(payload["elapsed_seconds"])
        self._best_snapshot = payload["best_snapshot"]
        # A best.pt is itself a complete recovery snapshot, without recursion.
        if self._best_snapshot is None and progress["best_validation_loss"] is not None:
            self._best_snapshot = _cpu_snapshot(payload)
        restore_random_state(payload["random_state"])
        self._last_checkpoint_time = self._elapsed_before
        return progress["steps_completed"], progress["epoch"]

    def _save_latest(self) -> Path:
        return self.save_checkpoint(
            self._progress["steps_completed"], self._progress["epoch"]
        )

    def _result(
        self, status: str, checkpoint: Path | None, receipt: Path | None = None
    ) -> TrainResult:
        self._log("training_finished", status=status, **self._progress)
        return TrainResult(
            status,
            self._progress["best_validation_loss"],
            self._progress["epoch"],
            self._progress["steps_completed"],
            self._progress["examples_seen"],
            self._elapsed(),
            None if checkpoint is None else str(checkpoint),
            None if receipt is None else str(receipt),
        )

    def fit(
        self,
        *,
        train_batches: Sequence[ForecastBatch],
        validation_batches: Sequence[ForecastBatch],
        step: Callable[[nn.Module, ForecastBatch], torch.Tensor],
        resume_from: str | Path | None = None,
    ) -> TrainResult:
        if not train_batches or not validation_batches:
            raise ValueError("non-empty training and validation batches are required")
        _trajectory_gate(self.config, train_batches)
        if any(b.split != "train" for b in train_batches):
            raise ValueError("training batches must use the train split")
        if any(b.split != "validation" for b in validation_batches):
            raise ValueError("validation batches must use the validation split")
        train_ids = {
            identity for batch in train_batches for identity in batch.trajectory_ids
        }
        validation_ids = {
            identity
            for batch in validation_batches
            for identity in batch.trajectory_ids
        }
        if train_ids & validation_ids:
            raise ValueError("training and validation trajectory identities overlap")
        if self.output_root.exists():
            raise FileExistsError(f"refusing to reuse run directory {self.output_root}")
        self.output_root.mkdir(parents=True, exist_ok=False)
        self.logger = JsonlLogger(self.output_root / "events.jsonl", run_id=self.run_id)
        admission = self.resource_admission()
        if admission.get("status") != "PASS":
            receipt = self.output_root / "resource_block.json"
            atomic_write_json(receipt, admission, overwrite=False)
            return self._result("BLOCKED_RESOURCE", None, receipt)
        previous_handlers: dict[int, Any] = {}
        latest: Path | None = None
        try:
            for batch in (*train_batches, *validation_batches):
                batch.validate()
            self._batch_sha256 = _batch_fingerprint(train_batches, validation_batches)
            seed_receipt = seed_everything(self.config.seed)
            if resume_from is not None:
                self.resume(resume_from)
            cursor = self._progress["next_batch_index"]
            if cursor > len(train_batches) or (
                cursor != len(train_batches)
                and cursor % self.config.gradient_accumulation
            ):
                raise ValueError("checkpoint is not at an optimizer-window boundary")
            if self._progress["epoch"] == self.config.epochs and cursor != 0:
                raise ValueError("completed checkpoint has a nonzero batch cursor")
            expected_examples = self._progress["epoch"] * sum(
                int(b.context.shape[0]) for b in train_batches
            ) + sum(int(b.context.shape[0]) for b in train_batches[:cursor])
            expected_steps = self._progress["epoch"] * math.ceil(
                len(train_batches) / self.config.gradient_accumulation
            ) + math.ceil(cursor / self.config.gradient_accumulation)
            if (
                self._progress["examples_seen"] != expected_examples
                or self._progress["steps_completed"] != expected_steps
            ):
                raise ValueError("checkpoint counters disagree with its batch cursor")
            self._started = time.monotonic()
            atomic_write_json(
                self.output_root / "environment.json",
                {
                    "schema_version": 2,
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "platform": platform.platform(),
                    "device": str(self.device),
                    "seed": seed_receipt,
                    "budgets": dataclasses.asdict(self.config),
                    "source_sha256": self.source_sha256,
                    "config_sha256": self.config_sha256,
                    "data_sha256": self.data_sha256,
                    "batch_sha256": self._batch_sha256,
                    "resume_from": None if resume_from is None else str(resume_from),
                    "energy": "UNAVAILABLE_NO_ADMITTED_METER",
                    "accumulation": "sample-weighted; state-dependent large-batch equivalence not claimed",
                },
                overwrite=False,
            )
            if self._best_snapshot is not None:
                _atomic_torch_save(
                    self.output_root / "checkpoints/best.pt", self._best_snapshot
                )
            # Recovery always has a pre-window checkpoint, even on first-window failure.
            latest = self._save_latest()

            def request_stop(_signal: int, _frame: object) -> None:
                self._interrupted = True

            if threading.current_thread() is threading.main_thread():
                for sig in (signal.SIGINT, signal.SIGTERM):
                    previous_handlers[sig] = signal.signal(sig, request_stop)
            last_admission_epoch, last_admission_elapsed = -1, self._elapsed()
            while (
                self._progress["epoch"] < self.config.epochs
                and not self._progress["early_stopped"]
            ):
                self.model.train()
                while self._progress["next_batch_index"] < len(train_batches):
                    cursor = self._progress["next_batch_index"]
                    window = train_batches[
                        cursor : cursor + self.config.gradient_accumulation
                    ]
                    incoming = sum(int(b.context.shape[0]) for b in window)
                    status = None
                    if self._interrupted:
                        status = "PREEMPTED"
                    elif (
                        self._elapsed() >= self.config.wall_clock_budget_seconds
                        or self._progress["examples_seen"] + incoming
                        > self.config.example_budget
                    ):
                        status = "BUDGET_EXHAUSTED"
                    admission = None
                    if status is None and (
                        self._progress["epoch"] != last_admission_epoch
                        or self._elapsed() - last_admission_elapsed >= 30.0
                    ):
                        admission = self.resource_admission()
                        last_admission_epoch = self._progress["epoch"]
                        last_admission_elapsed = self._elapsed()
                    if admission is not None and admission.get("status") != "PASS":
                        receipt = self.output_root / "resource_stop.json"
                        atomic_write_json(receipt, admission, overwrite=False)
                        return self._result(
                            "BLOCKED_RESOURCE", self._save_latest(), receipt
                        )
                    if status is not None:
                        return self._result(status, self._save_latest())
                    self._at_boundary = False
                    self.optimizer.zero_grad(set_to_none=True)
                    weighted_loss = 0.0
                    window_started = time.monotonic()
                    for batch in window:
                        n = int(batch.context.shape[0])
                        with self._autocast():
                            loss = step(self.model, move_batch(batch, self.device))
                        _check_loss(loss, "training")
                        self.scaler.scale(loss * (n / incoming)).backward()
                        weighted_loss += float(loss.detach().item()) * n
                    self.scaler.unscale_(self.optimizer)
                    if not any(
                        p.grad is not None
                        for p in self.model.parameters()
                        if p.requires_grad
                    ):
                        raise RuntimeError("training loss produced no model gradients")
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                        error_if_nonfinite=True,
                    )
                    if not bool(torch.isfinite(gradient_norm).item()):
                        raise FloatingPointError("non-finite gradient norm")
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._progress["steps_completed"] += 1
                    self._progress["examples_seen"] += incoming
                    self._progress["next_batch_index"] += len(window)
                    self._at_boundary = True
                    count = self._progress["steps_completed"]
                    if count % self.config.log_every_steps == 0:
                        self._log(
                            "train_step",
                            phase="train",
                            epoch=self._progress["epoch"],
                            global_step=count,
                            training_loss=weighted_loss / incoming,
                            effective_batch_size=incoming,
                            examples_seen=self._progress["examples_seen"],
                            learning_rate=[
                                g["lr"] for g in self.optimizer.param_groups
                            ],
                            gradient_norm=float(gradient_norm.item()),
                            examples_per_second=incoming
                            / max(time.monotonic() - window_started, 1e-9),
                            cuda_peak_bytes=(
                                torch.cuda.max_memory_allocated(self.device)
                                if self.device.type == "cuda"
                                else None
                            ),
                        )
                    if (
                        count % self.config.checkpoint_every_steps == 0
                        or self._elapsed() - self._last_checkpoint_time
                        >= self.config.checkpoint_every_seconds
                    ):
                        latest = self._save_latest()
                # Validation is replayable from the cursor at len(train_batches).
                # Its model state and RNG are restored so validation cannot mutate training memory.
                if self._interrupted:
                    return self._result("PREEMPTED", self._save_latest())
                if self._elapsed() >= self.config.wall_clock_budget_seconds:
                    return self._result("BUDGET_EXHAUSTED", self._save_latest())
                if (
                    self._progress["epoch"] != last_admission_epoch
                    or self._elapsed() - last_admission_elapsed >= 30.0
                ):
                    admission = self.resource_admission()
                    last_admission_epoch = self._progress["epoch"]
                    last_admission_elapsed = self._elapsed()
                    if admission.get("status") != "PASS":
                        receipt = self.output_root / "resource_stop.json"
                        atomic_write_json(receipt, admission, overwrite=False)
                        return self._result(
                            "BLOCKED_RESOURCE", self._save_latest(), receipt
                        )
                validation_loss = self._validate(validation_batches, step)
                best = self._progress["best_validation_loss"]
                improved = best is None or validation_loss < best
                self._progress["epoch"] += 1
                self._progress["next_batch_index"] = 0
                if improved:
                    self._progress["best_validation_loss"] = validation_loss
                    self._progress["epochs_without_improvement"] = 0
                    old_best = self._best_snapshot
                    self._best_snapshot = None
                    try:
                        self._best_snapshot = _cpu_snapshot(
                            self._checkpoint_payload(
                                self._progress["steps_completed"],
                                self._progress["epoch"],
                            )
                        )
                    except BaseException:
                        self._best_snapshot = old_best
                        raise
                    _atomic_torch_save(
                        self.output_root / "checkpoints/best.pt", self._best_snapshot
                    )
                else:
                    self._progress["epochs_without_improvement"] += 1
                    self._progress["early_stopped"] = (
                        self._progress["epochs_without_improvement"]
                        >= self.config.early_stopping_patience
                    )
                self._log(
                    "epoch_complete",
                    phase="validation",
                    validation_loss=validation_loss,
                    improved=improved,
                    **self._progress,
                )
                latest = self._save_latest()
            return self._result("PASS_ENGINEERING", latest)
        except BaseException as error:
            # Never serialize partially accumulated gradients or changed mutable state.
            # Recover from the last completed, persisted boundary; work after it replays.
            failure = {
                "schema_version": 2,
                "status": "FAILED",
                "failure_type": "OOM"
                if isinstance(error, torch.OutOfMemoryError)
                else type(error).__name__,
                "message": str(error),
                "progress": dict(self._progress),
                "recovery_checkpoint": None if latest is None else str(latest),
                "partial_window_discarded": not self._at_boundary,
                "silent_retry": False,
                "settings_changed": False,
            }
            atomic_write_json(
                self.output_root / "failure.json", failure, overwrite=False
            )
            self._log("failed", **failure)
            raise
        finally:
            self._elapsed_before = self._elapsed()
            self._started = None
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)

    def _autocast(self) -> Any:
        return torch.autocast(
            device_type=self.device.type,
            dtype={
                "fp32": torch.float32,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[self.config.precision],
            enabled=self.config.precision != "fp32",
        )

    @torch.no_grad()
    def _validate(
        self,
        batches: Sequence[ForecastBatch],
        step: Callable[[nn.Module, ForecastBatch], torch.Tensor],
    ) -> float:
        if not batches:
            raise ValueError("validation batches cannot be empty")
        saved_model = _cpu_snapshot(self.model.state_dict())
        saved_rng = random_state()
        modes = [(module, module.training) for module in self.model.modules()]
        total, count = 0.0, 0
        try:
            self.model.eval()
            for module in self.model.modules():
                reset = getattr(module, "reset_evaluation_state", None)
                if callable(reset):
                    reset()
            for batch in batches:
                with self._autocast():
                    loss = step(self.model, move_batch(batch, self.device))
                _check_loss(loss, "validation")
                n = int(batch.context.shape[0])
                total += float(loss.item()) * n
                count += n
            return total / count
        finally:
            self.model.load_state_dict(saved_model)
            restore_random_state(saved_rng)
            for module, training in modes:
                module.training = training


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
