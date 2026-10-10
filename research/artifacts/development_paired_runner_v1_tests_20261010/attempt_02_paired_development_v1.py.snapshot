"""Bounded, opt-in paired FIM integration runs on caller-pinned DEVELOPMENT banks.

Use a fresh, single-Python-thread process and an immutable checkout. This runner
records source bytes, measured outcomes and reset checks; it does not attest an
arbitrarily modified interpreter, enforce secrecy, or classify scientific claims.
It never calls the existing trainer or any frozen scientific runner.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import resource
import sys
import threading
import time
import traceback
import types
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import episode_bank_v1 as banks

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("full", "no_retrieval", "no_memory")
SCHEMA = "FIM_PAIRED_DEVELOPMENT_RUN_V1"
PURPOSE = "INTEGRATION_ONLY"
MODEL = {
    "in_channels": 1,
    "hidden": 4,
    "trace_dim": 4,
    "memory_capacity": 16,
    "retrieval_topk": 4,
    "memory_decay": 0.02,
    "salience_threshold": 0.0,
}
OPTIMIZER = {
    "name": "SGD",
    "lr": 0.001,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "clip_gradient_norm": 1.0,
}
MODEL_SOURCES = tuple(
    f"fim_experiments/{name}.py" for name in ("models", "systems", "ablation_systems")
)
SOURCE_PATHS = (
    *banks.SOURCE_PATHS,
    *MODEL_SOURCES,
    "fim_experiments/paired_development_v1.py",
)
BUFFER_SHAPES = {"keys": (16, 4), "values": (16, 4), "scores": (16,), "ages": (16,)}
DEMO_BANK = "research/artifacts/development_episode_bank_v1_demo_20261010/bank"
DEMO_BANK_SHA256 = "ae8e143064f27be469e69dc19fe1fac41d30737ecdcd4acff7a6de4a91afc938"
MAX_FILE_BYTES = 8_000_000


class RunnerError(ValueError):
    """A bounded run or retained evidence does not satisfy this version's contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def _sources() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_PATHS
    }


_IMPORTED_SOURCES = _sources()


def _checked_sources() -> dict[str, str]:
    current = _sources()
    _require(
        current == _IMPORTED_SOURCES,
        "source changed since runner import; use a fresh process and immutable checkout",
    )
    banks._verified_sources()
    return current


def _existing_implementation():
    """Execute exact trusted source snapshots, resolving only their legacy imports.

    The maintained modules use ``from models``/``from systems``. Temporary aliases
    make those imports unambiguous without sys.path edits or importing a foreign
    cached module. Restore every prior binding even if execution fails. Private
    modules are needed only while dataclass decorators execute, not for pickle.
    """
    _checked_sources()
    _require(
        threading.active_count() == 1,
        "legacy import adapter requires one active Python thread",
    )
    payloads = {name: (ROOT / name).read_bytes() for name in MODEL_SOURCES}
    _require(
        all(
            hashlib.sha256(raw).hexdigest() == _IMPORTED_SOURCES[name]
            for name, raw in payloads.items()
        ),
        "model source snapshot differs",
    )
    missing, prior, loaded = object(), {}, {}
    try:
        for path, raw in payloads.items():
            short = Path(path).stem
            private = "_fim_paired_v1_" + _IMPORTED_SOURCES[path] + "_" + short
            module = types.ModuleType(private)
            module.__file__ = str(ROOT / path)
            prior[private] = sys.modules.get(private, missing)
            sys.modules[private] = module
            exec(compile(raw, str(ROOT / path), "exec"), module.__dict__)
            loaded[short] = module
            if short in ("models", "systems"):
                prior[short] = sys.modules.get(short, missing)
                sys.modules[short] = module
    finally:
        for name, previous in reversed(list(prior.items())):
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    _checked_sources()
    return loaded["ablation_systems"]


@dataclass(frozen=True)
class RunConfig:
    model_seed: int = 37
    horizon: int = 4
    epochs: int = 1


def _configuration(config: RunConfig, bank: banks.EpisodeBank) -> dict[str, Any]:
    _require(type(config) is RunConfig, "config must be a v1 RunConfig")
    banks._integer(config.model_seed, "model seed", 0, 2**63 - 1)
    banks._integer(config.horizon, "prefix horizon", 1, 16)
    banks._integer(config.epochs, "epochs", 1, 2)
    manifest = bank.manifest
    _require(
        config.model_seed not in banks.HELD_SEEDS
        and config.model_seed != manifest["seed"],
        "model seed must be a separate non-held development seed",
    )
    _require(config.horizon <= manifest["horizon"], "prefix exceeds bank horizon")
    shape = manifest["array"]["shape"]
    _require(
        shape[0] <= 16 and 2 <= shape[2] <= 16,
        "v1 is limited to 16 episodes and 2..16 observation features",
    )
    episodes = (
        config.epochs * len(manifest["splits"]["train"])
        + len(manifest["splits"]["validation"])
        + len(manifest["splits"]["development_eval"])
    )
    _require(
        3 * config.horizon * episodes <= 768, "planned model-step budget exceeds 768"
    )
    return {
        **asdict(config),
        "model": MODEL.copy(),
        "optimizer": OPTIMIZER.copy(),
        "batch_size": 1,
        "device": "cpu",
        "dtype": "float32",
        "training": "teacher_forced_mean_observation_mse_float32",
        "evaluation": "final_state_autoregressive_observation_mse_float64_diagnostic",
        "checkpoint_selection": "fixed_final_no_selection",
        "order": "bank_split_order",
        "planned_steps": 3 * config.horizon * episodes,
        "planned_optimizer_updates": 3
        * config.epochs
        * len(manifest["splits"]["train"]),
    }


def _episode_plan(bank: banks.EpisodeBank, config: RunConfig) -> list[dict[str, Any]]:
    manifest, result = bank.manifest, []
    for split in banks.SPLITS:
        for epoch in range(config.epochs if split == "train" else 1):
            for position, index in enumerate(manifest["splits"][split]):
                result.append(
                    {
                        "ordinal": len(result),
                        "split": split,
                        "epoch": epoch if split == "train" else None,
                        "position": position,
                        "episode_index": index,
                        "episode_sha256": manifest["episode_sha256"][index],
                    }
                )
    return result


@contextmanager
def _execution_context():
    _require(
        torch.get_default_dtype() == torch.float32
        and torch.get_default_device().type == "cpu",
        "v1 requires process defaults CPU and float32",
    )
    _require(torch.get_num_threads() == 1, "v1 requires torch.set_num_threads(1)")
    _require(
        sys.platform.startswith("linux"), "v1 filesystem/RSS contract requires Linux"
    )
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=[]):
        torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn)


def _new_model(implementation, arm: str, config: RunConfig):
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(config.model_seed)
        torch.random.set_rng_state(generator.get_state())
        return implementation.AblatedFIMSystem(
            **MODEL, **implementation.variant_switches(arm)
        )


def _arrays(model, *, parameters_only: bool = False) -> dict[str, np.ndarray]:
    tensors = dict(model.named_parameters()) if parameters_only else model.state_dict()
    return {
        name: tensor.detach().cpu().contiguous().numpy().copy()
        for name, tensor in tensors.items()
    }


def _array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = arrays[name]
        _require(value.dtype == np.float32, "checkpoint tensors must be float32")
        digest.update(
            banks._canonical({"name": name, "shape": list(value.shape), "dtype": "<f4"})
            + b"\n"
        )
        digest.update(value.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _parameter_digest(model) -> str:
    return _array_digest(_arrays(model, parameters_only=True))


def _checkpoint_metadata(model) -> dict[str, Any]:
    parameters = _arrays(model, parameters_only=True)
    return {
        "parameter_sha256": _array_digest(parameters),
        "state_sha256": _array_digest(_arrays(model)),
        "parameter_names": sorted(parameters),
        "allocated_parameters": sum(value.size for value in parameters.values()),
    }


def _memory_state(model) -> dict[str, int]:
    bank = model.bank
    _require(
        model.bank is model.trace_bank is model.memory_bank,
        "memory aliases must identify the same bank",
    )
    _require(
        set(dict(model.named_buffers())) == {"bank." + name for name in BUFFER_SHAPES}
        and set(bank._buffers) == set(BUFFER_SHAPES),
        "memory buffer inventory differs",
    )
    for name, shape in BUFFER_SHAPES.items():
        tensor = getattr(bank, name)
        _require(
            tuple(tensor.shape) == shape
            and tensor.dtype == torch.float32
            and tensor.device.type == "cpu",
            f"memory buffer contract differs: {name}",
        )
    _require(
        type(bank.ptr) is int
        and 0 <= bank.ptr < 16
        and type(bank.size) is int
        and 0 <= bank.size <= 16,
        "invalid bank pointer/size",
    )
    return {"size": bank.size, "ptr": bank.ptr}


def strict_reset(model) -> dict[str, Any]:
    """Reset the actual model; raise on any exception or incomplete/impure clearing."""
    before = _memory_state(model)
    parameter_sha = _parameter_digest(model)
    model.reset_state()  # Deliberately no broad catch or fallback reset method.
    after = _memory_state(model)
    _require(after == {"size": 0, "ptr": 0}, "reset did not clear memory indices")
    _require(
        all(
            not bool(torch.count_nonzero(getattr(model.bank, name)))
            for name in BUFFER_SHAPES
        ),
        "reset did not clear all memory buffers",
    )
    _require(
        _parameter_digest(model) == parameter_sha, "reset changed learned parameters"
    )
    return {
        "status": "VERIFIED_CLEAR",
        "before": before,
        "after": after,
        "zero_buffers": sorted(BUFFER_SHAPES),
        "parameter_sha256": parameter_sha,
    }


def _write_json(path: Path, value: Any) -> None:
    banks._write_exclusive(path, banks._canonical(value) + b"\n")
    banks._fsync_directory(path.parent)


def _save_checkpoint(path: Path, model) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **_arrays(model))  # Float32 tensors only; no object arrays/pickle.
    banks._write_exclusive(path, buffer.getvalue())
    banks._fsync_directory(path.parent)


def _error(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    # Raw float32 bytes also preserve NaN/Inf failures, which strict JSON cannot.
    array = (
        tensor.detach()
        .cpu()
        .to(torch.float32)
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "f32_hex": array.tobytes().hex(),
    }


class _Journal:
    def __init__(self, path: Path):
        self.handle = path.open("xb")
        banks._fsync_directory(path.parent)
        self.records: list[dict[str, Any]] = []

    def append(self, value: dict[str, Any]) -> None:
        self.handle.write(banks._canonical(value) + b"\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.records.append(value)

    def close(self) -> None:
        self.handle.close()


def _run_episode(model, optimizer, bank, episode, config, journal, update_index):
    key = episode["ordinal"]
    training = episode["split"] == "train"
    parameters_before = _parameter_digest(model)
    journal.append(
        {
            "event": "episode_start",
            "episode": episode,
            "parameter_sha256": parameters_before,
        }
    )
    errors, losses, diagnostic = [], [], []
    optimization = None
    interruption = None
    try:
        journal.append(
            {
                "event": "reset",
                "episode_ordinal": key,
                "when": "before",
                **strict_reset(model),
            }
        )
        model.train(training)
        if training:
            optimizer.zero_grad(set_to_none=True)
        # The bank dataset returns independent clones. No shuffling, worker RNG,
        # sampler padding, minibatch sharing, or accidental train/eval mixture.
        x, y = bank.dataset(episode["split"])[episode["position"]]
        current = x[0].reshape(1, 1, 1, -1)
        with torch.set_grad_enabled(training):
            for step in range(config.horizon):
                observed = x[step].reshape(1, 1, 1, -1) if training else current
                target = y[step].reshape(1, 1, 1, -1)
                memory_before = _memory_state(model)
                journal.append(
                    {"event": "step_start", "episode_ordinal": key, "step": step}
                )
                output = model.step(observed, store_traces=True, retrieve=True)
                prediction = output.prediction
                valid = (
                    prediction.shape == target.shape
                    and prediction.dtype == torch.float32
                    and prediction.device.type == "cpu"
                    and bool(torch.isfinite(prediction).all())
                )
                mse = None
                if valid:
                    delta = prediction.detach().double() - target.double()
                    mse = float(delta.square().mean())
                journal.append(
                    {
                        "event": "step",
                        "episode_ordinal": key,
                        "step": step,
                        "input": _tensor_record(observed),
                        "target": _tensor_record(target),
                        "prediction": _tensor_record(prediction),
                        "finite_and_shape_valid": valid,
                        "diagnostic_observation_mse_float64": mse,
                        "memory_before": memory_before,
                        "memory_after": _memory_state(model),
                        "retrieved": output.retrieved is not None,
                    }
                )
                _require(
                    valid,
                    "model returned a nonfinite prediction or incompatible tensor",
                )
                losses.append(F.mse_loss(prediction, target))
                diagnostic.append(mse)
                current = prediction
            if training:
                loss = torch.stack(losses).mean()
                _require(bool(torch.isfinite(loss)), "nonfinite optimization loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                gradient_parameters = sum(
                    p.numel() for p in model.parameters() if p.grad is not None
                )
                optimization = {
                    "update_index": update_index,
                    "loss_float32": float(loss.detach()),
                    "gradient_norm_before_clip": float(norm),
                    "gradient_parameters": gradient_parameters,
                    "lr": 0.001,
                }
                journal.append(
                    {
                        "event": "optimizer_attempt",
                        "episode_ordinal": key,
                        "optimization": optimization,
                    }
                )
                optimizer.step()
                journal.append(
                    {
                        "event": "optimizer_applied",
                        "episode_ordinal": key,
                        "parameter_sha256": _parameter_digest(model),
                        "optimization": optimization,
                    }
                )
                _require(
                    all(bool(torch.isfinite(p).all()) for p in model.parameters()),
                    "optimizer produced nonfinite parameters",
                )
        if not training:
            _require(
                _parameter_digest(model) == parameters_before,
                "evaluation changed learned parameters",
            )
    except BaseException as exc:
        errors.append({"stage": "episode", **_error(exc)})
        if not isinstance(exc, Exception):
            interruption = exc
    finally:
        try:
            journal.append(
                {
                    "event": "reset",
                    "episode_ordinal": key,
                    "when": "after",
                    **strict_reset(model),
                }
            )
        except BaseException as exc:
            errors.append({"stage": "cleanup_reset", **_error(exc)})
            if not isinstance(exc, Exception) and interruption is None:
                interruption = exc
    if errors:
        journal.append(
            {
                "event": "episode_failed",
                "episode_ordinal": key,
                "errors": errors,
                "optimization": optimization,
            }
        )
        if interruption is not None:
            # Cancellation must propagate after both primary and cleanup errors
            # have been retained. Do not interpret it as an ordinary failed arm
            # and continue the remaining plan.
            raise interruption
        raise RunnerError(
            "episode failed; primary and cleanup errors are retained in events.jsonl"
        )
    journal.append(
        {
            "event": "episode_end",
            "episode_ordinal": key,
            "parameter_sha256_before": parameters_before,
            "parameter_sha256_after": _parameter_digest(model),
            "steps": config.horizon,
            "optimization": optimization,
            "diagnostic_observation_mse_float64": sum(diagnostic) / len(diagnostic),
        }
    )


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float32",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def _file_inventory(directory: Path) -> dict[str, Any]:
    result = {}
    for path in sorted(directory.rglob("*")):
        _require(not path.is_symlink(), "run cannot contain symlinks")
        if path.is_file() and path != directory / "manifest.json":
            raw = banks._read_regular(path, MAX_FILE_BYTES)
            result[path.relative_to(directory).as_posix()] = {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        elif not path.is_dir() and path != directory / "manifest.json":
            raise RunnerError("run contains a nonregular artifact")
    return result


def run_development(
    bank_directory: str | Path,
    destination: str | Path,
    *,
    expected_bank_sha256: str,
    config: RunConfig = RunConfig(),
    phase: str = "DEVELOPMENT",
) -> dict[str, Any]:
    """Run every planned arm once. Failed attempts remain; inspect returned status.

    Preflight rejections do not reserve an output. Once reserved, the pre-run plan
    and all written outcomes remain, including failures and partial checkpoints.
    Existing output directories are never overwritten. Measurements cannot stop
    external commands or authenticate a maliciously edited process or receipt.
    """
    _require(phase == "DEVELOPMENT", "only DEVELOPMENT integration is authorized by v1")
    source_snapshot = _checked_sources()
    bank = banks.load_development_bank(
        bank_directory, expected_bank_sha256=expected_bank_sha256
    )
    settings = _configuration(config, bank)
    with _execution_context():
        implementation = _existing_implementation()
        episodes = _episode_plan(bank, config)
        plan = {
            "schema": SCHEMA,
            "phase": "DEVELOPMENT",
            "purpose": PURPOSE,
            "status": "PLANNED",
            "bank_sha256": expected_bank_sha256,
            "source_sha256": source_snapshot,
            "config": settings,
            "environment": _environment(),
            "episodes": episodes,
            "arms": [
                {"name": arm, "switches": implementation.variant_switches(arm)}
                for arm in ARMS
            ],
            "scope": "Equal allocated parameters and SGD updates; no active-parameter or compute-match claim",
        }
        destination = Path(destination).absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir(exist_ok=False)
        banks._fsync_directory(destination.parent)
        started = time.perf_counter()
        try:
            _write_json(destination / "plan.json", plan)
            states, initial = {}, None
            for arm in ARMS:
                arm_path = destination / arm
                arm_path.mkdir()
                banks._fsync_directory(destination)
                journal = _Journal(arm_path / "events.jsonl")
                model, failures = None, []
                arm_started = time.perf_counter()
                state = {
                    "name": arm,
                    "phase": "DEVELOPMENT",
                    "purpose": PURPOSE,
                    "status": "FAILED",
                    "switches": implementation.variant_switches(arm),
                }
                try:
                    _checked_sources()
                    model = _new_model(implementation, arm, config)
                    metadata = _checkpoint_metadata(model)
                    state["initial"] = metadata
                    if initial is None:
                        initial = metadata
                        _save_checkpoint(destination / "initial.npz", model)
                        _write_json(destination / "initial_metadata.json", initial)
                    _require(
                        metadata == initial,
                        "paired initial tensors/allocated parameters differ",
                    )
                    journal.append({"event": "initialized", "initial": metadata})
                    optimizer = torch.optim.SGD(
                        model.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0
                    )
                    update_index = 0
                    for episode in episodes:
                        _run_episode(
                            model,
                            optimizer,
                            bank,
                            episode,
                            config,
                            journal,
                            update_index,
                        )
                        update_index += episode["split"] == "train"
                    _checked_sources()
                    state["status"] = "COMPLETE"
                except BaseException as exc:
                    failures.append({"stage": "arm", **_error(exc)})
                    if not isinstance(exc, Exception):
                        raise
                finally:
                    if model is not None:
                        try:
                            _save_checkpoint(arm_path / "final.npz", model)
                            state["final"] = _checkpoint_metadata(model)
                        except Exception as exc:
                            failures.append(
                                {"stage": "final_checkpoint", **_error(exc)}
                            )
                    state["model_steps_retained"] = sum(
                        r["event"] == "step" for r in journal.records
                    )
                    state["model_steps_attempted"] = sum(
                        r["event"] == "step_start" for r in journal.records
                    )
                    state["optimizer_updates_attempted"] = sum(
                        r["event"] == "optimizer_attempt" for r in journal.records
                    )
                    state["optimizer_updates_retained"] = sum(
                        r["event"] == "optimizer_applied" for r in journal.records
                    )
                    state["seconds"] = time.perf_counter() - arm_started
                    state["failures"] = failures
                    if failures:
                        state["status"] = "FAILED"
                    journal.close()
                    _write_json(arm_path / "status.json", state)
                    states[arm] = state["status"]
            _checked_sources()
            manifest = {
                "schema": SCHEMA,
                "phase": "DEVELOPMENT",
                "purpose": PURPOSE,
                "status": "COMPLETE"
                if all(value == "COMPLETE" for value in states.values())
                else "FAILED",
                "arms": states,
                "bank_sha256": expected_bank_sha256,
                "files": _file_inventory(destination),
                "seconds": time.perf_counter() - started,
                "process_peak_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                )
                * 1024,
                "rss_scope": "Cumulative Linux process high-water mark including imports and all prior arms; not per-arm memory",
            }
            manifest["run_sha256"] = banks._digest(manifest)
            _write_json(destination / "manifest.json", manifest)
            return manifest
        except BaseException as exc:
            # Directory reservation and successful writes are never rolled back.
            # If the filesystem itself fails, propagate the original exception;
            # do not assert that this best-effort receipt was saved.
            try:
                _write_json(
                    destination / "run_failure.json",
                    {
                        "status": "FAILED",
                        "phase": "DEVELOPMENT",
                        "purpose": PURPOSE,
                        "error": _error(exc),
                    },
                )
            except OSError:
                pass
            raise


def _finite_number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _near(
    actual: Any, expected: float, label: str, *, tolerance: float = 1e-12
) -> None:
    _require(
        _finite_number(actual)
        and math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
        f"{label} differs from retained raw evidence",
    )


def _read_checkpoint(raw: bytes) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        _require(
            len(entries) <= 256
            and len({entry.filename for entry in entries}) == len(entries),
            "checkpoint entry count/uniqueness differs",
        )
        _require(
            all(
                entry.compress_type == zipfile.ZIP_STORED
                and entry.file_size <= MAX_FILE_BYTES
                for entry in entries
            )
            and sum(entry.file_size for entry in entries) <= MAX_FILE_BYTES,
            "checkpoint must contain bounded uncompressed arrays",
        )
        # Validate NPY headers before np.load can allocate a declared shape.
        # A small ZIP entry may otherwise claim a huge array with missing data.
        for entry in entries:
            _require(
                entry.filename.endswith(".npy") and "/" not in entry.filename,
                "checkpoint entries must be flat NPY tensor names",
            )
            with archive.open(entry) as handle:
                _require(
                    np.lib.format.read_magic(handle) == (1, 0),
                    "checkpoint requires the v1 NPY header emitted by this writer",
                )
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                _require(
                    dtype == np.dtype("<f4")
                    and not fortran
                    and all(type(value) is int and value >= 0 for value in shape)
                    and math.prod(shape) <= 1_000_000,
                    "checkpoint header exceeds float32 shape/dtype bounds",
                )
                _require(
                    entry.file_size - handle.tell() == math.prod(shape) * 4,
                    "checkpoint payload length differs from its NPY shape",
                )
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    _require(
        all(
            value.dtype == np.float32
            and value.size <= 1_000_000
            and bool(np.isfinite(value).all())
            for value in result.values()
        ),
        "checkpoint tensors must be bounded finite float32 arrays",
    )
    return result


def _vector(record: Any, dimension: int) -> np.ndarray:
    _require(
        isinstance(record, dict)
        and set(record) == {"shape", "dtype", "f32_hex"}
        and record["shape"] == [1, 1, 1, dimension]
        and record["dtype"] == "torch.float32"
        and isinstance(record["f32_hex"], str)
        and len(record["f32_hex"]) == dimension * 8,
        "raw step tensor contract differs",
    )
    raw = bytes.fromhex(record["f32_hex"])
    _require(len(raw) == dimension * 4, "raw step tensor byte length differs")
    result = np.frombuffer(raw, dtype="<f4")
    _require(bool(np.isfinite(result).all()), "completed step has nonfinite raw values")
    return result


def _check_reset(
    record: dict[str, Any],
    *,
    ordinal: int,
    when: str,
    before: dict[str, int],
    parameter_sha: str,
) -> None:
    _require(
        record
        == {
            "event": "reset",
            "episode_ordinal": ordinal,
            "when": when,
            "status": "VERIFIED_CLEAR",
            "before": before,
            "after": {"size": 0, "ptr": 0},
            "zero_buffers": sorted(BUFFER_SHAPES),
            "parameter_sha256": parameter_sha,
        },
        "reset evidence differs from the strict trajectory-isolation contract",
    )


def _verify_arm(records, status, arm, episodes, bank, config, initial):
    """Check complete event grammar and raw values; no forward or optimizer calls."""
    position, parameter_sha, update = 0, initial["parameter_sha256"], 0
    diagnostics = {split: [] for split in banks.SPLITS}

    def take(event):
        nonlocal position
        _require(position < len(records), f"{arm} event log is incomplete")
        value = records[position]
        position += 1
        _require(
            isinstance(value, dict) and value.get("event") == event,
            f"{arm} event order differs; expected {event}",
        )
        return value

    _require(
        take("initialized") == {"event": "initialized", "initial": initial},
        "paired initialization evidence differs",
    )
    dimension = bank.manifest["array"]["shape"][2]
    for episode in episodes:
        ordinal, split = episode["ordinal"], episode["split"]
        training = split == "train"
        before_parameter = parameter_sha
        _require(
            take("episode_start")
            == {
                "event": "episode_start",
                "episode": episode,
                "parameter_sha256": parameter_sha,
            },
            "episode order/identity differs",
        )
        _check_reset(
            take("reset"),
            ordinal=ordinal,
            when="before",
            before={"size": 0, "ptr": 0},
            parameter_sha=parameter_sha,
        )
        x, y = bank.dataset(split)[episode["position"]]
        current, errors, losses = x[0].numpy(), [], []
        for step in range(config.horizon):
            _require(
                take("step_start")
                == {"event": "step_start", "episode_ordinal": ordinal, "step": step},
                "model-step attempt order differs",
            )
            record = take("step")
            _require(
                set(record)
                == {
                    "event",
                    "episode_ordinal",
                    "step",
                    "input",
                    "target",
                    "prediction",
                    "finite_and_shape_valid",
                    "diagnostic_observation_mse_float64",
                    "memory_before",
                    "memory_after",
                    "retrieved",
                }
                and record["episode_ordinal"] == ordinal
                and record["step"] == step
                and record["finite_and_shape_valid"] is True,
                "step coverage/finite status differs",
            )
            observed, target, prediction = (
                _vector(record[name], dimension)
                for name in ("input", "target", "prediction")
            )
            expected_input = x[step].numpy() if training else current
            _require(
                observed.tobytes() == expected_input.tobytes()
                and target.tobytes() == y[step].numpy().tobytes(),
                "step input/target differs from pinned episode and teacher-forcing policy",
            )
            size_before, size_after = (step, step + 1) if arm != "no_memory" else (0, 0)
            _require(
                record["memory_before"]
                == {"size": size_before, "ptr": size_before % 16}
                and record["memory_after"]
                == {"size": size_after, "ptr": size_after % 16}
                and record["retrieved"] is (arm == "full" and step > 0),
                "within-episode storage/retrieval differs from the declared intervention",
            )
            mse = float(
                np.mean(
                    (prediction.astype(np.float64) - target.astype(np.float64)) ** 2
                )
            )
            _near(
                record["diagnostic_observation_mse_float64"], mse, "step diagnostic MSE"
            )
            errors.append(mse)
            losses.append(
                F.mse_loss(
                    torch.from_numpy(prediction.copy()), torch.from_numpy(target.copy())
                )
            )
            current = prediction
        optimization = None
        if training:
            attempted, applied = take("optimizer_attempt"), take("optimizer_applied")
            optimization = attempted.get("optimization")
            _require(
                set(attempted) == {"event", "episode_ordinal", "optimization"}
                and attempted["episode_ordinal"] == ordinal
                and isinstance(optimization, dict)
                and set(optimization)
                == {
                    "update_index",
                    "loss_float32",
                    "gradient_norm_before_clip",
                    "gradient_parameters",
                    "lr",
                }
                and type(optimization["update_index"]) is int
                and optimization["update_index"] == update
                and optimization["lr"] == 0.001,
                "optimizer policy/coverage differs",
            )
            _near(
                optimization["loss_float32"],
                float(torch.stack(losses).mean()),
                "float32 training loss",
                tolerance=1e-6,
            )
            _require(
                _finite_number(optimization["gradient_norm_before_clip"])
                and optimization["gradient_norm_before_clip"] >= 0
                and type(optimization["gradient_parameters"]) is int
                and 1
                <= optimization["gradient_parameters"]
                <= initial["allocated_parameters"],
                "gradient diagnostics differ",
            )
            parameter_sha = applied.get("parameter_sha256")
            banks._sha(parameter_sha, "post-update parameter identity")
            _require(
                applied
                == {
                    "event": "optimizer_applied",
                    "episode_ordinal": ordinal,
                    "parameter_sha256": parameter_sha,
                    "optimization": optimization,
                },
                "optimizer application differs",
            )
            update += 1
        final_size = config.horizon if arm != "no_memory" else 0
        _check_reset(
            take("reset"),
            ordinal=ordinal,
            when="after",
            before={"size": final_size, "ptr": final_size % 16},
            parameter_sha=parameter_sha,
        )
        end = take("episode_end")
        expected = {
            "event": "episode_end",
            "episode_ordinal": ordinal,
            "parameter_sha256_before": before_parameter,
            "parameter_sha256_after": parameter_sha,
            "steps": config.horizon,
            "optimization": optimization,
            "diagnostic_observation_mse_float64": end.get(
                "diagnostic_observation_mse_float64"
            ),
        }
        _require(end == expected, "episode completion/parameter identity differs")
        mse = sum(errors) / len(errors)
        _near(end["diagnostic_observation_mse_float64"], mse, "episode diagnostic MSE")
        diagnostics[split].append(mse)
    _require(
        position == len(records), "unexpected events or extra model/optimizer work"
    )
    steps = len(episodes) * config.horizon
    _require(
        status["model_steps_retained"] == status["model_steps_attempted"] == steps
        and status["optimizer_updates_retained"]
        == status["optimizer_updates_attempted"]
        == update
        and status["final"]["parameter_sha256"] == parameter_sha,
        "retained model/optimizer counts or final checkpoint identity differ",
    )
    return {split: sum(values) / len(values) for split, values in diagnostics.items()}


def verify_development_run(
    directory: str | Path,
    *,
    expected_run_sha256: str,
    bank_directory: str | Path,
    expected_bank_sha256: str,
) -> dict[str, Any]:
    """Admit only a complete, pinned integration receipt; recompute raw diagnostics.

    Initialization is reconstructed once from the declared seed. There are no
    forward/training calls, inferential tests, outcome selection or claim upgrades.
    Raw artifacts remain necessary; a self-declared metric summary is insufficient.
    """
    try:
        banks._sha(expected_run_sha256, "expected_run_sha256")
        source_snapshot = _checked_sources()
        bank = banks.load_development_bank(
            bank_directory, expected_bank_sha256=expected_bank_sha256
        )
        directory = Path(directory).absolute()
        _require(
            directory.is_dir() and not directory.is_symlink(),
            "run must be a regular directory",
        )
        _require(
            {p.name for p in directory.iterdir()}
            == {
                "plan.json",
                "initial.npz",
                "initial_metadata.json",
                "manifest.json",
                *ARMS,
            },
            "run is incomplete, failed or has unexpected artifacts",
        )
        raw_manifest = banks._read_regular(directory / "manifest.json", 200_000)
        manifest = banks._strict_json(raw_manifest)
        _require(
            isinstance(manifest, dict)
            and set(manifest)
            == {
                "schema",
                "phase",
                "purpose",
                "status",
                "arms",
                "bank_sha256",
                "files",
                "seconds",
                "process_peak_rss_bytes",
                "rss_scope",
                "run_sha256",
            },
            "run manifest schema differs",
        )
        _require(
            manifest["schema"] == SCHEMA
            and manifest["phase"] == "DEVELOPMENT"
            and manifest["purpose"] == PURPOSE
            and manifest["status"] == "COMPLETE"
            and manifest["arms"] == {arm: "COMPLETE" for arm in ARMS},
            "failed/non-development run cannot qualify",
        )
        _require(
            banks._digest({k: v for k, v in manifest.items() if k != "run_sha256"})
            == manifest["run_sha256"]
            == expected_run_sha256,
            "pinned run identity differs",
        )
        _require(
            manifest["bank_sha256"] == expected_bank_sha256,
            "run and caller bank identities differ",
        )
        files = {}
        for name in ("plan.json", "initial.npz", "initial_metadata.json"):
            files[name] = banks._read_regular(directory / name, MAX_FILE_BYTES)
        for arm in ARMS:
            path = directory / arm
            _require(
                path.is_dir()
                and not path.is_symlink()
                and {p.name for p in path.iterdir()}
                == {"events.jsonl", "status.json", "final.npz"},
                "arm artifact coverage differs",
            )
            for name in ("events.jsonl", "status.json", "final.npz"):
                files[f"{arm}/{name}"] = banks._read_regular(
                    path / name, MAX_FILE_BYTES
                )
        inventory = {
            name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            for name, raw in files.items()
        }
        _require(
            manifest["files"] == inventory,
            "retained artifact bytes differ from run manifest",
        )
        plan = banks._strict_json(files["plan.json"])
        _require(
            isinstance(plan, dict)
            and set(plan)
            == {
                "schema",
                "phase",
                "purpose",
                "status",
                "bank_sha256",
                "source_sha256",
                "config",
                "environment",
                "episodes",
                "arms",
                "scope",
            },
            "plan schema differs",
        )
        _require(
            plan["schema"] == SCHEMA
            and plan["phase"] == "DEVELOPMENT"
            and plan["purpose"] == PURPOSE
            and plan["status"] == "PLANNED"
            and plan["bank_sha256"] == expected_bank_sha256
            and plan["source_sha256"] == source_snapshot,
            "plan source/phase/bank identity differs",
        )
        settings = plan["config"]
        config = RunConfig(
            **{key: settings[key] for key in ("model_seed", "horizon", "epochs")}
        )
        _require(
            settings == _configuration(config, bank), "fixed configuration differs"
        )
        episodes = _episode_plan(bank, config)
        _require(
            plan["episodes"] == episodes, "planned episode membership/order differs"
        )
        environment = plan["environment"]
        _require(
            environment["device"] == "cpu"
            and environment["dtype"] == "float32"
            and type(environment["torch_num_threads"]) is int
            and environment["torch_num_threads"] == 1
            and environment["deterministic_algorithms"] is True,
            "recorded numerical environment differs",
        )
        with _execution_context():
            implementation = _existing_implementation()
            _require(
                plan["arms"]
                == [
                    {"name": arm, "switches": implementation.variant_switches(arm)}
                    for arm in ARMS
                ],
                "planned existing intervention switches differ",
            )
            model = _new_model(implementation, "full", config)
            initial = banks._strict_json(files["initial_metadata.json"])
            _require(
                initial == _checkpoint_metadata(model),
                "initial parameter allocation/seed identity differs",
            )
            initial_arrays = _read_checkpoint(files["initial.npz"])
            expected_arrays = _arrays(model)
            _require(
                set(initial_arrays) == set(expected_arrays)
                and all(
                    np.array_equal(initial_arrays[name], value)
                    and initial_arrays[name].shape == value.shape
                    for name, value in expected_arrays.items()
                ),
                "initial checkpoint differs from seeded model",
            )
            diagnostics = {}
            for arm in ARMS:
                status = banks._strict_json(files[f"{arm}/status.json"])
                _require(
                    set(status)
                    == {
                        "name",
                        "phase",
                        "purpose",
                        "status",
                        "switches",
                        "initial",
                        "final",
                        "model_steps_retained",
                        "model_steps_attempted",
                        "optimizer_updates_retained",
                        "optimizer_updates_attempted",
                        "seconds",
                        "failures",
                    }
                    and status["name"] == arm
                    and status["phase"] == "DEVELOPMENT"
                    and status["purpose"] == PURPOSE
                    and status["status"] == "COMPLETE"
                    and status["failures"] == []
                    and status["switches"] == implementation.variant_switches(arm)
                    and status["initial"] == initial,
                    "arm status/intervention/pairing differs",
                )
                _require(
                    _finite_number(status["seconds"]) and status["seconds"] >= 0,
                    "invalid arm duration",
                )
                records = [
                    banks._strict_json(line)
                    for line in files[f"{arm}/events.jsonl"].splitlines()
                ]
                diagnostics[arm] = _verify_arm(
                    records, status, arm, episodes, bank, config, initial
                )
                final_arrays = _read_checkpoint(files[f"{arm}/final.npz"])
                _require(
                    set(final_arrays) == set(initial_arrays)
                    and all(
                        final_arrays[name].shape == value.shape
                        for name, value in initial_arrays.items()
                    ),
                    "final checkpoint tensor inventory/shape differs",
                )
                for name in set(final_arrays) - set(initial["parameter_names"]):
                    _require(
                        np.count_nonzero(final_arrays[name]) == 0,
                        "final checkpoint memory is not cleared",
                    )
                expected_final = {
                    **initial,
                    "state_sha256": _array_digest(final_arrays),
                    "parameter_sha256": _array_digest(
                        {
                            name: final_arrays[name]
                            for name in initial["parameter_names"]
                        }
                    ),
                }
                _require(
                    status["final"] == expected_final,
                    "final checkpoint evidence differs",
                )
        _require(
            _finite_number(manifest["seconds"])
            and manifest["seconds"] >= 0
            and type(manifest["process_peak_rss_bytes"]) is int
            and manifest["process_peak_rss_bytes"] > 0,
            "invalid runtime/process RSS measurement",
        )
        _checked_sources()
        return {
            "schema": SCHEMA,
            "status": "VERIFIED_INTEGRATION",
            "phase": "DEVELOPMENT",
            "purpose": PURPOSE,
            "run_sha256": expected_run_sha256,
            "bank_sha256": expected_bank_sha256,
            "model_steps": settings["planned_steps"],
            "optimizer_updates": settings["planned_optimizer_updates"],
            "allocated_parameters_per_arm": initial["allocated_parameters"],
            "initial_parameter_sha256": initial["parameter_sha256"],
            "diagnostic_observation_mse_float64_by_arm_split": diagnostics,
            "scientific_support_inferred": False,
            "compute_matched": False,
            "seconds": manifest["seconds"],
            "process_peak_rss_bytes": manifest["process_peak_rss_bytes"],
            "rss_scope": manifest["rss_scope"],
        }
    except RunnerError:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OSError,
        OverflowError,
        zipfile.BadZipFile,
    ) as exc:
        raise RunnerError(f"invalid retained integration evidence: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="New bounded DEVELOPMENT integration; never overwrite an attempt"
    )
    run.add_argument("--bank", type=Path, required=True)
    run.add_argument("--bank-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model-seed", type=int, required=True)
    run.add_argument("--horizon", type=int, default=4)
    run.add_argument("--epochs", type=int, default=1)
    demo = commands.add_parser(
        "demo", help="Use the unchanged retained PR35 bank: 9 updates, 60 model steps"
    )
    demo.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify",
        help="Recompute raw diagnostics and check pinned evidence; no model steps",
    )
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--run-sha256", required=True)
    verify.add_argument("--bank", type=Path, required=True)
    verify.add_argument("--bank-sha256", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    try:
        if args.command == "verify":
            result = verify_development_run(
                args.run,
                expected_run_sha256=args.run_sha256,
                bank_directory=args.bank,
                expected_bank_sha256=args.bank_sha256,
            )
        else:
            bank = ROOT / DEMO_BANK if args.command == "demo" else args.bank
            pin = DEMO_BANK_SHA256 if args.command == "demo" else args.bank_sha256
            config = (
                RunConfig()
                if args.command == "demo"
                else RunConfig(args.model_seed, args.horizon, args.epochs)
            )
            result = run_development(
                bank, args.output, expected_bank_sha256=pin, config=config
            )
            if result["status"] == "COMPLETE":
                result = verify_development_run(
                    args.output,
                    expected_run_sha256=result["run_sha256"],
                    bank_directory=bank,
                    expected_bank_sha256=pin,
                )
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0 if result["status"] == "VERIFIED_INTEGRATION" else 1
    except (RunnerError, banks.EpisodeBankError, OSError, RuntimeError) as exc:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "phase": "DEVELOPMENT",
                    "purpose": PURPOSE,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
