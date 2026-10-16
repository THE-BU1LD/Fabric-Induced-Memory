"""Finite DEVELOPMENT profiling using the maintained FIM engineering functions.

Importing this file uses only the standard library. Public API:
    run(source_root, config, output) -> dict
source_root is the itfm_v2 directory containing src/fim_sota; config is a
resolved/partial engineering config dict or JSON path; output is a NEW directory.
The CLI supplies an independent 300-second process limit plus 15-second cleanup.
Direct API callers must provide their own process supervisor for a hard deadline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time


WARMUP_WINDOWS = 3
MEASURED_WINDOWS = 5
MAX_PROFILE_SECONDS = 300
DEVELOPMENT_SEEDS = {101, 202, 303}


def _json(path, payload):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _host_peak():
    # ru_maxrss is a process-lifetime high-water mark, not incremental usage.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _gpu_inventory():
    """Report what the driver exposes; do not infer reservation or quota policy."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if result.returncode:
            return {"status": "UNAVAILABLE", "exit_code": result.returncode}
        rows = []
        for line in result.stdout.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 4:
                raise ValueError("unexpected nvidia-smi inventory format")
            rows.append(
                {
                    "driver_index": int(fields[0]),
                    "name": fields[1],
                    "vram_bytes": int(float(fields[2]) * 1024**2),
                    "driver_version": fields[3],
                }
            )
        return {"status": "OBSERVED", "driver_visible_count": len(rows), "devices": rows}
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {"status": "UNAVAILABLE", "reason": type(error).__name__}


def _summarize(values):
    return {
        "count": len(values),
        "total_seconds": sum(values),
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def run(source_root, config, output):
    """Profile 3 warmup + 5 measured optimizer windows; never evaluate test scores.

    Raises on invalid config, failed resource admission, unavailable CUDA,
    multiple CUDA-visible devices, nonfinite arithmetic or source drift. A
    unique output directory retains profile.json and live events.jsonl even on
    failure. Only the three existing development seeds are accepted.
    """
    source_root = Path(source_root).resolve()
    output = Path(output).resolve()
    source_path = source_root / "src"
    if not (source_path / "fim_sota/engineering_runtime.py").is_file():
        raise ValueError("source_root must contain src/fim_sota/engineering_runtime.py")
    if "fim_sota" in sys.modules:
        loaded = Path(sys.modules["fim_sota"].__file__).resolve()
        if not loaded.is_relative_to(source_path):
            raise RuntimeError("a different fim_sota source tree is already imported")
    if isinstance(config, (str, os.PathLike)):
        config = json.loads(Path(config).read_text())
    if not isinstance(config, dict):
        raise TypeError("config must be an engineering configuration dict or JSON path")
    sys.path.insert(0, str(source_path))
    # These maintained modules are standard-library-only.
    from fim_sota import engineering

    if Path(engineering.__file__).resolve() != source_path / "fim_sota/engineering.py":
        raise RuntimeError("engineering module source mismatch")
    configuration = engineering.resolve(dict(config))
    if configuration["seed"] not in DEVELOPMENT_SEEDS:
        raise ValueError("profile only accepts development seeds 101, 202 or 303")
    if configuration["purpose"] != "development" or configuration["precision"] != "fp32":
        raise ValueError("profile requires explicit purpose=development and precision=fp32")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = engineering.source_identity()
    receipt = {
        "schema_version": 1,
        "status": "RUNNING",
        "evidence_class": "DEVELOPMENT_PROFILE",
        "scientific_claim": "NOT_EVALUATED",
        "configuration": configuration,
        "source": source,
        "profile_source_sha256": _sha(__file__),
        "config_sha256": engineering.digest(configuration),
        "data_spec_sha256": engineering.data_identity(configuration),
        "warmup_optimizer_windows": WARMUP_WINDOWS,
        "measured_optimizer_windows": MEASURED_WINDOWS,
        "runtime_executed": False,
        "placement_verified": False,
        "test_scores_accessed": False,
        "full_training_or_evaluation_measured": False,
        "timing_scope": "Small repeated training-window profile with explicit device synchronization; not a complete campaign runtime prediction.",
        "quota_scope": "Driver/visible GPU counts do not prove the scheduler allocation or charging policy. This process does not stop or release a notebook accelerator allocation.",
        "host_process_lifetime_peak_bytes_before": _host_peak(),
        "phases": {},
    }

    def event(name, **fields):
        record = {
            "event": name,
            "time_unix_ns": time.time_ns(),
            "elapsed_seconds": time.perf_counter() - started,
            "evidence_class": "DEVELOPMENT_PROFILE",
            **fields,
        }
        with (output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)

    def guard(phase):
        if time.perf_counter() - started >= MAX_PROFILE_SECONDS:
            raise TimeoutError("profile soft deadline reached before " + phase)
        result = engineering.admission()
        receipt.setdefault("resource_checks", []).append({"phase": phase, **result})
        if result["status"] != "PASS":
            receipt["status"] = "BLOCKED_RESOURCE"
            raise RuntimeError("resource admission blocked before " + phase)

    _json(output / "profile.json", receipt)
    try:
        guard("tensor_import")
        if shutil.disk_usage(output).free < 3 * 1024**3:
            raise RuntimeError("less than 3 GiB scratch free")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("OMP_NUM_THREADS", str(configuration["threads"]))
        os.environ.setdefault("MKL_NUM_THREADS", str(configuration["threads"]))
        check = engineering.preflight(configuration, runtime=True)
        receipt["environment_preflight"] = check
        receipt["runtime_env"] = check
        if check["status"] == "BLOCKED":
            receipt["status"] = "BLOCKED_RESOURCE_OR_ENVIRONMENT"
            raise RuntimeError("maintained runtime preflight blocked")
        # First direct tensor imports occur only after the maintained admission.
        import torch
        from fim_sota.engineering_runtime import (
            build_model,
            forecast_step,
            generate_fixture,
            make_batches,
        )
        from fim_sota.training import (
            _atomic_torch_save,
            _cpu_snapshot,
            move_batch,
            random_state,
            seed_everything,
        )

        receipt["runtime_executed"] = True
        torch.set_num_threads(configuration["threads"])
        device = torch.device("cuda:0" if configuration["device"] == "cuda" else "cpu")
        visible_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        receipt["accelerators"] = {
            "driver_inventory": _gpu_inventory(),
            "torch_visible_count": visible_count,
            "selected_device": str(device),
            "assigned_device_count": "UNKNOWN_SCHEDULER_ALLOCATION_NOT_INFERRED",
        }
        if device.type == "cuda":
            if visible_count != 1:
                raise RuntimeError("CUDA profile requires exactly one visible GPU; select visibility before starting the process")
            torch.cuda.set_device(device)
            properties = torch.cuda.get_device_properties(device)
            receipt["accelerators"].update(
                selected_name=properties.name,
                selected_vram_bytes=int(properties.total_memory),
                compute_capability=[properties.major, properties.minor],
                selected_free_bytes_at_start=int(torch.cuda.mem_get_info(device)[0]),
            )
            torch.cuda.reset_peak_memory_stats(device)
        receipt["software"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "numpy": importlib.metadata.version("numpy"),
            "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "deterministic_algorithms": True,
            "precision": "fp32",
            "compilation": False,
            "distributed": False,
        }
        seed_everything(configuration["seed"])

        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def timed(function):
            synchronize()
            before = time.perf_counter()
            value = function()
            synchronize()
            return value, time.perf_counter() - before

        guard("fixture_materialization")
        fixture, seconds = timed(lambda: generate_fixture(configuration))
        receipt["phases"]["generate_fixture_seconds"] = seconds
        prepared, seconds = timed(
            lambda: {
                split: make_batches(fixture, configuration, split)
                for split in ("train", "validation", "test")
            }
        )
        receipt["phases"]["make_all_batches_seconds"] = seconds
        receipt["data_inventory"] = {
            split: {
                "owners": len(fixture[split]),
                "examples": sum(int(batch.context.shape[0]) for batch in prepared[split]),
                "batches": len(prepared[split]),
            }
            for split in prepared
        }
        # Test arrays are generated to measure materialization only; no prediction,
        # target summary, model selection or score is computed from these arrays.
        model, seconds = timed(lambda: build_model(configuration).to(device))
        receipt["phases"]["model_build_and_transfer_seconds"] = seconds
        if not all(parameter.device == device and parameter.dtype == torch.float32 for parameter in model.parameters()):
            raise RuntimeError("model parameters are not FP32 on the requested device")
        optimizer = torch.optim.Adam(model.parameters(), lr=configuration["learning_rate"])
        model.train()
        receipt["model"] = {
            "class": type(model).__name__,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_devices": sorted({str(parameter.device) for parameter in model.parameters()}),
            "input_shape": list(prepared["train"][0].context.shape),
            "target_shape": list(prepared["train"][0].target.shape),
        }
        train = prepared["train"]
        accumulation = configuration["gradient_accumulation"]
        measurements = []
        total_windows = WARMUP_WINDOWS + MEASURED_WINDOWS
        receipt["repeats_batches_if_needed"] = total_windows * accumulation > len(train)
        for index in range(total_windows):
            guard("optimizer_window_" + str(index))
            group = [train[(index * accumulation + offset) % len(train)] for offset in range(accumulation)]
            samples = sum(int(batch.context.shape[0]) for batch in group)
            measured = {"transfer_seconds": 0.0, "forward_seconds": 0.0, "backward_seconds": 0.0, "optimizer_seconds": 0.0}
            optimizer.zero_grad(set_to_none=True)
            synchronize()
            window_start = time.perf_counter()
            loss_value = 0.0
            for batch in group:
                moved, seconds = timed(lambda: move_batch(batch, device))
                measured["transfer_seconds"] += seconds
                tensor_values = [moved.context, moved.target, moved.observation_mask, moved.event_times, moved.lead_times, moved.target_available_times, moved.target_observation_mask]
                if any(value is not None and value.device != device for value in tensor_values):
                    raise RuntimeError("batch tensor placement differs from selected device")
                if moved.context.dtype != torch.float32 or moved.target.dtype != torch.float32:
                    raise RuntimeError("profile context and target must remain FP32")
                loss, seconds = timed(lambda: forecast_step(model, moved))
                measured["forward_seconds"] += seconds
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("nonfinite profile loss")
                weight = int(moved.context.shape[0]) / samples
                _, seconds = timed(lambda: (loss * weight).backward())
                measured["backward_seconds"] += seconds
                loss_value += float(loss.detach().item()) * weight
            if not all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item()) for parameter in model.parameters()):
                raise FloatingPointError("missing or nonfinite profile gradients")

            def update():
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                return float(norm.item())

            norm, seconds = timed(update)
            measured["optimizer_seconds"] = seconds
            synchronize()
            measured["window_seconds"] = time.perf_counter() - window_start
            measured.update(optimizer_window=index, effective_batch_size=samples, loss=loss_value, gradient_norm=norm, phase="warmup" if index < WARMUP_WINDOWS else "measurement")
            if not all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()):
                raise FloatingPointError("nonfinite parameters after optimizer step")
            event("optimizer_window_complete", **measured)
            if index >= WARMUP_WINDOWS:
                measurements.append(measured)
        receipt["measurements"] = measurements
        receipt["placement_verified"] = True
        with (output / "measured_timing.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(measurements[0]))
            writer.writeheader()
            writer.writerows(measurements)
            handle.flush()
            os.fsync(handle.fileno())
        receipt["timing_summary"] = {
            field: _summarize([row[field] for row in measurements])
            for field in ("transfer_seconds", "forward_seconds", "backward_seconds", "optimizer_seconds", "window_seconds")
        }
        receipt["measured_examples_per_second"] = sum(row["effective_batch_size"] for row in measurements) / sum(row["window_seconds"] for row in measurements)
        guard("checkpoint_timing")
        checkpoint = output / "profile_checkpoint.pt"

        def save_snapshot():
            payload = _cpu_snapshot({
                "schema_version": "development-profile-snapshot-v1",
                "evidence_class": "DEVELOPMENT_PROFILE",
                "not_a_resumable_training_checkpoint": True,
                "source_sha256": source["sha256"],
                "config_sha256": receipt["config_sha256"],
                "optimizer_windows": total_windows,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "random_state": random_state(),
            })
            _atomic_torch_save(checkpoint, payload)

        _, seconds = timed(save_snapshot)
        receipt["phases"]["cpu_snapshot_and_atomic_checkpoint_seconds"] = seconds
        loaded, seconds = timed(lambda: torch.load(checkpoint, map_location="cpu", weights_only=True))
        receipt["phases"]["weights_only_checkpoint_load_seconds"] = seconds
        if loaded["schema_version"] != "development-profile-snapshot-v1":
            raise RuntimeError("profile checkpoint envelope mismatch")
        if not all(torch.equal(value.detach().cpu(), loaded["model_state"][name]) for name, value in model.state_dict().items()):
            raise RuntimeError("profile checkpoint model state changed on safe reload")
        receipt["checkpoint"] = {"path": checkpoint.name, "sha256": _sha(checkpoint), "bytes": checkpoint.stat().st_size, "kind": "Timing snapshot, not a training-resume artifact"}
        if device.type == "cuda":
            receipt["gpu_memory"] = {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "free_bytes_at_end": int(torch.cuda.mem_get_info(device)[0]),
                "scope": "This process since peak reset after CUDA initialization; reserved context/other process allocations are not fully represented by allocated_bytes.",
            }
        else:
            receipt["gpu_memory"] = {"status": "NOT_APPLICABLE_CPU_PROFILE"}
        if engineering.source_identity() != source:
            raise RuntimeError("maintained source changed during profile")
        if _sha(__file__) != receipt["profile_source_sha256"]:
            raise RuntimeError("profile helper source changed during execution")
        _json(output / "source.json", source)
        _json(output / "configuration.json", configuration)
        _json(output / "environment.json", {"preflight": check, "software": receipt["software"], "accelerators": receipt["accelerators"]})
        receipt["status"] = "PASS_PROFILE"
        event("profile_complete", status=receipt["status"])
    except BaseException as error:
        if receipt["status"] == "RUNNING":
            receipt["status"] = "FAILED"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error), "silent_retry": False}
        event("profile_failed", **receipt["failure"])
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        receipt["host_process_lifetime_peak_bytes_after"] = _host_peak()
        receipt["host_peak_scope"] = "Process-lifetime ru_maxrss; includes imports and preparation, not a per-phase incremental allocation measurement."
        receipt["artifacts"] = {
            name: _sha(output / name)
            for name in ("events.jsonl", "measured_timing.csv", "profile_checkpoint.pt", "source.json", "configuration.json", "environment.json")
            if (output / name).is_file()
        }
        _json(output / "profile.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run(args.source_root, args.config, args.output)
        return
    if Path(args.output).exists():
        raise FileExistsError("profile output already exists")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--source-root", args.source_root, "--config", args.config, "--output", args.output]
    process = subprocess.Popen(command, start_new_session=True)
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        code = process.wait(timeout=MAX_PROFILE_SECONDS)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        code = 124
    finally:
        signal.signal(signal.SIGTERM, previous)
    output = Path(args.output)
    if output.is_dir():
        _json(output / "supervisor.json", {"exit_code": code, "hard_wall_seconds": MAX_PROFILE_SECONDS, "cleanup_reserve_seconds": 15, "evidence_class": "DEVELOPMENT_PROFILE", "accelerator_allocation_released": "NOT_CLAIMED"})
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
