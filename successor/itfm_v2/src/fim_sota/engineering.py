"""Finite engineering campaigns for the existing trainer. No tensor imports here.

This interface is deliberately separate from frozen scientific stage admission.
It never consumes or changes a historical campaign allowance.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

from .provenance import atomic_write_json, canonical_json_bytes, sha256_file

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1
DEFAULTS = dict(
    schema_version=1,
    purpose="smoke",
    benchmark="bounded_periodic_field_v1",
    seed=101,
    device="cpu",
    precision="fp32",
    epochs=2,
    batch_size=4,
    train_trajectories=4,
    validation_trajectories=2,
    test_trajectories=2,
    context_steps=3,
    sequence_steps=12,
    channels=1,
    spatial_points=8,
    hidden=16,
    learning_rate=0.001,
    gradient_accumulation=2,
    wall_seconds=120,
    example_budget=10000,
    checkpoint_seconds=30,
    memory_capacity=16,
    memory_topk=4,
    memory_temperature=0.2,
    variant="memory",
    threads=1,
)
VARIANTS = {"base", "memory", "erased", "spatial_mean", "last_residual"}


def digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def resolve(raw):
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("unknown engineering configuration fields")
    config = {**DEFAULTS, **raw}
    if type(config["schema_version"]) is not int or config["schema_version"] != SCHEMA:
        raise ValueError("unsupported engineering schema")
    if config["purpose"] not in {"smoke", "development"}:
        raise ValueError("confirmatory campaigns are not admitted by this interface")
    if config["benchmark"] != "bounded_periodic_field_v1":
        raise ValueError(
            "only the explicitly synthetic engineering benchmark is admitted"
        )
    if config["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be explicitly cpu or cuda")
    if config["precision"] not in {"fp32", "bf16", "fp16"}:
        raise ValueError("unsupported precision")
    if config["device"] == "cpu" and config["precision"] != "fp32":
        raise ValueError("CPU engineering reference requires fp32")
    if config["variant"] not in VARIANTS:
        raise ValueError("unknown model variant")
    for key, default in DEFAULTS.items():
        if (
            type(default) is int
            and key not in {"seed", "schema_version"}
            and (type(config[key]) is not int or config[key] < 1)
        ):
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**31:
        raise ValueError("seed must be an integer in [0,2**31)")
    for key in ("learning_rate", "memory_temperature", "checkpoint_seconds"):
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{key} must be finite and positive")
    if config["sequence_steps"] <= config["context_steps"]:
        raise ValueError("sequence_steps must exceed context_steps")
    if config["threads"] > 4 or config["memory_topk"] > config["memory_capacity"]:
        raise ValueError("threads >4 or topk > capacity is not admitted")
    # Bound eager fixture materialization; these are engineering guardrails.
    elements = (
        sum(
            config[k]
            for k in (
                "train_trajectories",
                "validation_trajectories",
                "test_trajectories",
            )
        )
        * config["sequence_steps"]
        * config["channels"]
        * config["spatial_points"]
        * (config["context_steps"] + 2)
    )
    if elements > 16_000_000:
        raise ValueError("fixture exceeds the 16-million-element engineering cap")
    if config["wall_seconds"] > 3600:
        raise ValueError("individual engineering cells are capped at one hour")
    input_dim = config["context_steps"] * config["channels"] * config["spatial_points"]
    output_dim = config["channels"] * config["spatial_points"]
    if config["hidden"] * (input_dim + output_dim + 1) + output_dim > 2_000_000:
        raise ValueError("model exceeds two-million-parameter engineering cap")
    memory_bytes = (
        max(config[k] for k in ("validation_trajectories", "test_trajectories"))
        * config["memory_capacity"]
        * (5 * input_dim + 4 * output_dim + 64)
    )
    if memory_bytes > 64 * 1024**2:
        raise ValueError("declared memory exceeds 64 MiB engineering cap")
    rows = config["sequence_steps"] - config["context_steps"]
    raw_elements = (
        config["validation_trajectories"] + config["test_trajectories"]
    ) * rows * 5 * (input_dim + 5 * output_dim) + config["test_trajectories"] * rows * (
        input_dim + 2 * output_dim
    )
    if raw_elements > 16_000_000:
        raise ValueError("retained evaluation tensors exceed 16-million-element cap")
    return config


def load_config_file(path):
    return resolve(json.loads(Path(path).read_text()))


def source_identity():
    paths = sorted((PACKAGE_ROOT / "src/fim_sota").glob("*.py"))
    paths += [PACKAGE_ROOT / "scripts/kaggle.py"]
    paths += [PACKAGE_ROOT / "src/fim_v2" / n for n in ("__init__.py", "resources.py")]
    records = {str(p.relative_to(PACKAGE_ROOT)): sha256_file(p) for p in paths}
    return {"sha256": digest(records), "files": records}


def data_identity(config):
    keys = (
        "benchmark",
        "seed",
        "train_trajectories",
        "validation_trajectories",
        "test_trajectories",
        "context_steps",
        "sequence_steps",
        "channels",
        "spatial_points",
    )
    return digest(
        {
            "generation": {k: config[k] for k in keys},
            "normalization": "none",
            "split": "disjoint-owner-and-generator-stream",
            "generator_source": sha256_file(
                PACKAGE_ROOT / "src/fim_sota/engineering_runtime.py"
            ),
        }
    )


def sample_resources():
    if sys.platform == "darwin":
        raw = subprocess.check_output(["vm_stat"], text=True)
        page = int(re.search(r"page size of (\d+) bytes", raw).group(1))
        values = {
            k.strip(): int(v)
            for k, v in re.findall(r"^([^:\n]+):\s+(\d+)\.", raw, re.M)
        }
        return dict(
            available_bytes=page * (values["Pages free"] + values["Pages speculative"]),
            paging=values["Pageouts"],
            swapouts=values["Swapouts"],
        )
    if sys.platform.startswith("linux"):
        mem = dict(
            re.findall(r"^(\w+):\s+(\d+)", Path("/proc/meminfo").read_text(), re.M)
        )
        vm = dict(
            line.split() for line in Path("/proc/vmstat").read_text().splitlines()
        )
        available = int(mem["MemAvailable"]) * 1024
        for limit_path, used_path in (
            ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
            (
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            ),
        ):
            if Path(limit_path).is_file() and Path(used_path).is_file():
                limit = Path(limit_path).read_text().strip()
                if limit != "max":
                    available = min(
                        available, max(0, int(limit) - int(Path(used_path).read_text()))
                    )
        return dict(
            available_bytes=available,
            paging=int(vm["pswpout"]),
            swapouts=int(vm["pswpout"]),
        )
    raise RuntimeError("no admitted resource sampler for this platform")


def admission(sample=sample_resources, pause=time.sleep):
    try:
        first = sample()
        pause(0.25)
        second = sample()
        passed = min(
            first["available_bytes"], second["available_bytes"]
        ) >= 512 * 1024**2 and all(
            second[k] == first[k] for k in ("paging", "swapouts")
        )
        return {
            "status": "PASS" if passed else "BLOCK",
            "first": first,
            "second": second,
            "floor_bytes": 512 * 1024**2,
        }
    except Exception as error:
        return {"status": "BLOCK", "reason": str(error), "floor_bytes": 512 * 1024**2}


def preflight(config, runtime=False):
    packages = {}
    for name in ("torch", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    resources = admission()
    receipt = dict(
        status="PASS_STATIC",
        python=platform.python_version(),
        packages=packages,
        resource=resources,
        device=config["device"],
        precision=config["precision"],
        source=source_identity(),
        config_sha256=digest(config),
        data_sha256=data_identity(config),
        runtime_executed=False,
    )
    if resources["status"] != "PASS" or not all(packages.values()):
        receipt["status"] = "BLOCKED"
    if runtime and receipt["status"] != "BLOCKED":
        # Mandatory immediate recheck before the first tensor import.
        receipt["preimport"] = admission()
        if receipt["preimport"]["status"] != "PASS":
            receipt["status"] = "BLOCKED"
            return receipt
        import torch

        if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (
            2,
            6,
        ):
            raise RuntimeError("weights-only checkpoint interface requires torch>=2.6")
        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA explicitly requested but unavailable; no CPU fallback"
            )
        if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA bf16 unsupported")
        receipt.update(
            runtime_executed=True,
            status="PASS_RUNTIME_ENVIRONMENT",
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=(torch.cuda.get_device_name(0) if config["device"] == "cuda" else None),
        )
    return receipt


class Budget:
    def __init__(self, seconds, clock=time.monotonic):
        self.clock, self.started, self.seconds = clock, clock(), seconds

    def remaining(self):
        return max(0, self.seconds - (self.clock() - self.started))

    def admits(self, seconds):
        return self.remaining() >= seconds


def load_manifest(path):
    raw = json.loads(Path(path).read_text())
    if (
        set(raw) != {"schema_version", "purpose", "wall_seconds", "cells"}
        or raw["schema_version"] != 1
    ):
        raise ValueError("invalid campaign manifest")
    if (
        raw["purpose"] not in {"smoke", "development"}
        or type(raw["wall_seconds"]) is not int
        or not 1 <= raw["wall_seconds"] <= 21600
    ):
        raise ValueError("campaign budget must be 1..21600 s and non-confirmatory")
    if not isinstance(raw["cells"], list) or not 1 <= len(raw["cells"]) <= 24:
        raise ValueError("campaign must contain 1..24 finite cells")
    ids, cells = set(), []
    for cell in raw["cells"]:
        if (
            set(cell) != {"id", "config"}
            or not isinstance(cell["id"], str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", cell["id"])
            or cell["id"] in ids
        ):
            raise ValueError("campaign IDs must be unique safe names")
        ids.add(cell["id"])
        config = resolve(cell["config"])
        if config["purpose"] != raw["purpose"]:
            raise ValueError("cell and campaign purpose differ")
        cells.append({"id": cell["id"], "config": config})
    raw["cells"] = cells
    if sum(c["config"]["wall_seconds"] + 30 for c in cells) > raw["wall_seconds"]:
        raise ValueError(
            "sum of cell limits and shutdown reserves exceeds campaign budget"
        )
    return raw


def artifact_records(root):
    # Only declared artifacts are eligible; arbitrary files are excluded.
    names = [
        "config.json",
        "source.json",
        "environment_preflight.json",
        "environment.json",
        "events.jsonl",
        "training_result.json",
        "checkpoints/latest.pt",
        "checkpoints/best.pt",
        "evaluation/metrics.json",
        "evaluation/predictions.pt",
        "evaluation/events.jsonl",
        "evaluation/raw_data.pt",
    ]
    required = set(names)
    missing = sorted(n for n in required if not (root / n).is_file())
    if missing:
        raise ValueError(f"incomplete run artifacts: {missing}")
    snapshot = json.loads((root / "source.json").read_text())
    names += [f"source_snapshot/{name}" for name in snapshot.get("files", {})]
    for name in names:
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or (root / name).is_symlink()
        ):
            raise ValueError("unsafe artifact path")
    return {n: sha256_file(root / n) for n in names if (root / n).is_file()}


def validate_completed(root, config, source):
    root = Path(root)
    receipt = json.loads((root / "completion.json").read_text())
    if (
        receipt.get("status") != "PASS_ENGINEERING"
        or receipt.get("source_sha256") != source
        or receipt.get("config_sha256") != digest(config)
        or receipt.get("data_sha256") != data_identity(config)
    ):
        raise ValueError("completed cell identity/status mismatch")
    if artifact_records(root) != receipt.get("artifacts"):
        raise ValueError("completed cell artifacts changed or are missing")
    snapshot = json.loads((root / "source.json").read_text())
    if snapshot.get("sha256") != source or digest(snapshot.get("files")) != source:
        raise ValueError("embedded source inventory mismatch")
    for name, expected in snapshot["files"].items():
        if sha256_file(root / "source_snapshot" / name) != expected:
            raise ValueError("source snapshot bytes mismatch")
    if digest(json.loads((root / "config.json").read_text())) != digest(config):
        raise ValueError("embedded configuration mismatch")
    metrics = json.loads((root / "evaluation/metrics.json").read_text())
    return receipt, metrics


def supervise(command, seconds, env=None):
    """One foreground child, bounded wall time and explicit graceful-stop reserve."""
    child = subprocess.Popen(command, env=env, start_new_session=True)

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return child.wait(timeout=seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        return 124
    finally:
        signal.signal(signal.SIGTERM, previous)


def execute_run(config, root, resume=None):
    root = Path(root).resolve()
    if (
        root.exists()
        or any(
            (root.parent / (root.name + suffix)).exists()
            for suffix in (".execution.json", ".blocked.json")
        )
        or list(root.parent.glob(f".{root.name}.request-*.json"))
    ):
        raise FileExistsError(f"refusing existing run: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    request = root.parent / f".{root.name}.request-{uuid.uuid4().hex}.json"
    source = source_identity()
    atomic_write_json(
        request,
        {
            "config": config,
            "root": str(root),
            "resume": None if resume is None else str(Path(resume).resolve()),
            "source_sha256": source["sha256"],
            "source": source,
        },
        overwrite=False,
    )
    command = [
        sys.executable,
        "-u",
        str(PACKAGE_ROOT / "scripts/kaggle.py"),
        "_worker",
        "--request",
        str(request),
    ]
    code = supervise(
        command,
        config["wall_seconds"],
        env={
            **os.environ,
            "OMP_NUM_THREADS": str(config["threads"]),
            "MKL_NUM_THREADS": str(config["threads"]),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    # Retain request/stop evidence even if the process never starts training.
    failure = root.parent / f"{root.name}.execution.json"
    atomic_write_json(
        failure,
        {
            "exit_code": code,
            "request": request.name,
            "source_sha256": source["sha256"],
            "config_sha256": digest(config),
            "wall_limit_seconds": config["wall_seconds"],
        },
        overwrite=False,
    )
    if code:
        raise RuntimeError(f"cell failed/stopped (exit {code}); retained {failure}")
    validate_completed(root, config, source["sha256"])


def worker(request):
    payload = json.loads(Path(request).read_text())
    config, root = resolve(payload["config"]), Path(payload["root"])
    if source_identity()["sha256"] != payload["source_sha256"]:
        raise RuntimeError("source changed since dispatch")
    source_bytes = {
        name: (PACKAGE_ROOT / name).read_bytes() for name in payload["source"]["files"]
    }
    if {
        name: hashlib.sha256(data).hexdigest() for name, data in source_bytes.items()
    } != payload["source"]["files"]:
        raise RuntimeError("dispatch source inventory mismatch")
    check = preflight(config, runtime=True)
    if check["status"] == "BLOCKED":
        atomic_write_json(
            root.parent / (root.name + ".blocked.json"), check, overwrite=False
        )
        raise RuntimeError("resource/environment admission blocked before model import")
    if shutil.disk_usage(root.parent).free < 3 * 1024**3:
        raise RuntimeError("less than 3 GiB scratch free")
    from .engineering_runtime import run_experiment

    result = run_experiment(
        config,
        root.parent,
        root.name,
        payload["source_sha256"],
        digest(config),
        data_identity(config),
        admission,
        resume_from=payload["resume"],
    )
    atomic_write_json(root / "config.json", config, overwrite=False)
    atomic_write_json(root / "source.json", payload["source"], overwrite=False)
    atomic_write_json(root / "environment_preflight.json", check, overwrite=False)
    atomic_write_json(root / "training_result.json", result, overwrite=False)
    for name, data in source_bytes.items():
        path = root / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
    if result["status"] != "PASS_ENGINEERING":
        raise RuntimeError(f"training did not finish: {result['status']}")
    if source_identity() != payload["source"]:
        raise RuntimeError("source changed during execution; no completion accepted")
    atomic_write_json(
        root / "completion.json",
        {
            "schema_version": 1,
            "status": "PASS_ENGINEERING",
            "source_sha256": payload["source_sha256"],
            "config_sha256": digest(config),
            "data_sha256": data_identity(config),
            "artifacts": artifact_records(root),
        },
        overwrite=False,
    )


def campaign(manifest, root, reuse=False):
    root = Path(root).resolve()
    if root.exists() and not reuse:
        raise FileExistsError(
            "campaign exists; --reuse only validates already completed cells"
        )
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".campaign.lock"
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    try:
        source = source_identity()["sha256"]
        identity = digest({"manifest": manifest, "source": source})
        statepath = root / "campaign.json"
        if statepath.exists():
            state = json.loads(statepath.read_text())
            if state["identity"] != identity:
                raise ValueError("campaign source/config changed")
        else:
            state = {
                "identity": identity,
                "status": "RUNNING",
                "cells": {},
                "elapsed_seconds": 0,
                "reserved_seconds": 0,
            }
        budget = Budget(
            max(
                0,
                manifest["wall_seconds"]
                - max(state["elapsed_seconds"], state.get("reserved_seconds", 0)),
            )
        )
        atomic_write_json(root / "manifest.json", manifest)
        state["status"] = "RUNNING"
        baseline = state["elapsed_seconds"]
        try:
            for cell in manifest["cells"]:
                name, config = cell["id"], cell["config"]
                destination = root / name
                if destination.exists():
                    validate_completed(destination, config, source)
                    state["cells"][name] = "REUSED_VERIFIED"
                    continue
                if name in state["cells"]:
                    raise RuntimeError(
                        f"attempted incomplete cell {name}; "
                        "use explicit checkpoint resume into a new run"
                    )
                if not budget.admits(config["wall_seconds"] + 30):
                    state["status"] = "BUDGET_EXHAUSTED"
                    raise RuntimeError(
                        "remaining campaign budget cannot admit the next cell"
                    )
                state["cells"][name] = "RUNNING"
                state["reserved_seconds"] = (
                    state.get("reserved_seconds", 0) + config["wall_seconds"] + 30
                )
                atomic_write_json(statepath, state)
                execute_run(config, destination)
                state["cells"][name] = "PASS_ENGINEERING"
            state["status"] = "PASS_ENGINEERING"
        except BaseException:
            if state["status"] == "RUNNING":
                state["status"] = "FAILED"
            raise
        finally:
            state["elapsed_seconds"] = baseline + budget.clock() - budget.started
            atomic_write_json(statepath, state)
    finally:
        lock.unlink()


def analyze(root, output):
    root = Path(root)
    if (root / "completion.json").is_file():
        config = load_config_file(root / "config.json")
        receipt, metrics = validate_completed(root, config, source_identity()["sha256"])
        atomic_write_json(
            output,
            {
                "scope": "synthetic engineering/development only",
                "rows": [
                    {
                        "id": root.name,
                        "seed": config["seed"],
                        "variant": config["variant"],
                        "metrics": metrics,
                        "source_sha256": receipt["source_sha256"],
                    }
                ],
            },
            overwrite=False,
        )
        return
    manifest = load_manifest(root / "manifest.json")
    source = source_identity()["sha256"]
    rows = []
    for cell in manifest["cells"]:
        receipt, metrics = validate_completed(root / cell["id"], cell["config"], source)
        rows.append(
            {
                "id": cell["id"],
                "seed": cell["config"]["seed"],
                "variant": cell["config"]["variant"],
                "metrics": metrics,
                "source_sha256": receipt["source_sha256"],
            }
        )
    atomic_write_json(
        output,
        {"scope": "synthetic engineering/development only", "rows": rows},
        overwrite=False,
    )


def export_bundle(output, manifest_path, run_root=None):
    output = Path(output)
    manifest = load_manifest(manifest_path)
    files = {
        str(p.relative_to(PACKAGE_ROOT)): p
        for p in (PACKAGE_ROOT / "src/fim_sota").glob("*.py")
    }
    for name in ("__init__.py", "resources.py"):
        files[f"src/fim_v2/{name}"] = PACKAGE_ROOT / "src/fim_v2" / name
    for pattern in (
        "scripts/kaggle.py",
        "tests/test_engineering_control.py",
        "tests/test_engineering_runtime.py",
        "tests/test_memory_causality.py",
        "tests/test_training_recovery.py",
        "kaggle/HANDOFF.md",
        "kaggle/requirements-runtime.txt",
        "kaggle/smoke-cpu.json",
        "kaggle/smoke-cuda.json",
        "kaggle/development.json",
        "kaggle/GPU_READINESS.md",
        "kaggle/CHANGE_LEDGER.md",
    ):
        for p in PACKAGE_ROOT.glob(pattern):
            if p.is_file():
                files[str(p.relative_to(PACKAGE_ROOT))] = p
    if run_root:
        run_root = Path(run_root)
        for cell in manifest["cells"]:
            run = run_root / cell["id"]
            receipt, _ = validate_completed(
                run, cell["config"], source_identity()["sha256"]
            )
            for name in (*receipt["artifacts"], "completion.json"):
                files[f"artifacts/{cell['id']}/{name}"] = run / name
    if any(p.is_symlink() for p in files.values()):
        raise ValueError("bundle refuses symlinks")
    contents = {name: path.read_bytes() for name, path in files.items()}
    inventory = {
        name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
    }
    source = source_identity()
    if any(inventory[name] != expected for name, expected in source["files"].items()):
        raise RuntimeError("source changed during bundle construction")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    descriptor, name = tempfile.mkstemp(
        prefix=".bundle-", suffix=".zip", dir=output.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            with zipfile.ZipFile(
                handle, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for name, data in sorted(contents.items()):
                    archive.writestr(name, data)
                archive.writestr("campaign.json", canonical_json_bytes(manifest))
                archive.writestr(
                    "BUNDLE_MANIFEST.json",
                    canonical_json_bytes(
                        {
                            "source": source,
                            "files": inventory,
                            "campaign_sha256": digest(manifest),
                        }
                    ),
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)  # atomic publication; refuses any existing target
    finally:
        temporary.unlink(missing_ok=True)


def handle(args):
    command = args.engineering_command
    if command in {"test", "evaluate"} and not args.worker_child:
        arguments = [
            sys.executable,
            "-u",
            str(PACKAGE_ROOT / "scripts/kaggle.py"),
            command,
            "--worker-child",
        ]
        keys = (
            ("config", "checkpoint", "output")
            if command == "evaluate"
            else ("device", "receipt")
        )
        for key in keys:
            arguments += ["--" + key, str(getattr(args, key))]
        limit = (
            load_config_file(args.config)["wall_seconds"]
            if command == "evaluate"
            else 300
        )
        code = supervise(
            arguments,
            limit,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        if code:
            receipt = (
                Path(args.output + ".execution.json")
                if command == "evaluate"
                else Path(args.receipt)
            )
            if not receipt.exists():
                atomic_write_json(
                    receipt,
                    {
                        "status": "BLOCKED_OR_FAILED",
                        "exit_code": code,
                        "runtime_pass": False,
                        "wall_limit_seconds": limit,
                    },
                    overwrite=False,
                )
            raise RuntimeError(f"{command} failed or blocked (exit {code})")
        return
    if command == "preflight":
        result = preflight(load_config_file(args.config), args.runtime)
        if args.receipt:
            atomic_write_json(args.receipt, result, overwrite=False)
        print(json.dumps(result, indent=2), flush=True)
        if result["status"] == "BLOCKED":
            raise RuntimeError("preflight blocked")
    elif command == "run":
        execute_run(load_config_file(args.config), args.output, args.resume)
    elif command == "campaign":
        manifest = load_manifest(args.manifest)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "source": source_identity(),
                        "manifest": manifest,
                        "maximum_cell_seconds_including_shutdown": sum(
                            c["config"]["wall_seconds"] + 30 for c in manifest["cells"]
                        ),
                        "runtime_prediction": "UNMEASURED",
                    },
                    indent=2,
                ),
                flush=True,
            )
        else:
            campaign(manifest, args.output, args.reuse)
    elif command == "_worker":
        worker(args.request)
    elif command == "evaluate":
        config = load_config_file(args.config)
        check = preflight(config, runtime=True)
        if check["status"] == "BLOCKED":
            raise RuntimeError("evaluation admission blocked")
        from .engineering_runtime import evaluate_checkpoint

        evaluate_checkpoint(
            config,
            Path(args.checkpoint),
            Path(args.output),
            source_identity()["sha256"],
            digest(config),
            data_identity(config),
        )
    elif command == "analyze":
        analyze(args.root, args.output)
    elif command == "export":
        export_bundle(args.output, args.manifest, args.run_root)
    elif command == "test":
        patterns = (
            "test_training_recovery.py",
            "test_engineering_runtime.py",
            "test_memory_causality.py",
        )
        expected = set()
        for name in patterns:
            tree = ast.parse((PACKAGE_ROOT / "tests" / name).read_text())
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    expected.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    expected.update(
                        method.name
                        for method in node.body
                        if isinstance(method, ast.FunctionDef)
                        and method.name.startswith("test_")
                    )
        required = "test_requested_device_training_evaluation"
        if not expected or required not in expected:
            raise RuntimeError("required tensor test inventory absent")
        check = preflight(resolve({"device": args.device}), runtime=True)
        if check["status"] == "BLOCKED":
            atomic_write_json(
                args.receipt,
                {"status": "BLOCKED", "tests_executed": 0, "environment": check},
                overwrite=False,
            )
            raise RuntimeError("tensor checks blocked before imports")
        os.environ["FIM_TEST_DEVICE"] = args.device
        import pytest

        class ReceiptPlugin:
            def __init__(self):
                self.inventory, self.passed, self.skipped, self.failed = [], [], [], []

            def pytest_collection_finish(self, session):
                self.inventory = [item.nodeid for item in session.items]
                actual = {name.split("::")[-1].split("[")[0] for name in self.inventory}
                if actual != expected:
                    raise RuntimeError(
                        "loaded test inventory differs from source inventory"
                    )

            def pytest_runtest_logreport(self, report):
                if report.failed:
                    self.failed.append(report.nodeid)
                if report.skipped:
                    self.skipped.append(report.nodeid)
                if report.when == "call" and report.passed:
                    self.passed.append(report.nodeid)

        plugin = ReceiptPlugin()
        code = pytest.main(
            [
                "-q",
                "-o",
                "addopts=",
                *[str(PACKAGE_ROOT / "tests" / name) for name in patterns],
            ],
            plugins=[plugin],
        )
        passed = (
            code == 0
            and len(plugin.passed) > 0
            and any(name.endswith("::" + required) for name in plugin.passed)
        )
        receipt = {
            "status": "PASS" if passed else "FAIL",
            "inventory": plugin.inventory,
            "test_sources": {
                name: sha256_file(PACKAGE_ROOT / "tests" / name) for name in patterns
            },
            "passed": plugin.passed,
            "failed": plugin.failed,
            "skipped": plugin.skipped,
            "requested_device": args.device,
            "environment": check,
        }
        atomic_write_json(args.receipt, receipt, overwrite=False)
        if not passed:
            raise RuntimeError(
                "tensor tests failed or required device test did not execute"
            )


def register(subcommands):
    parser = subcommands.add_parser(
        "engineering", help="finite engineering-only Kaggle preparation"
    )
    sub = parser.add_subparsers(dest="engineering_command", required=True)
    for name in ("preflight", "run", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "preflight":
            p.add_argument("--runtime", action="store_true")
            p.add_argument("--receipt")
        else:
            p.add_argument("--output", required=True)
        if name == "run":
            p.add_argument("--resume")
        if name == "evaluate":
            p.add_argument("--checkpoint", required=True)
            p.add_argument(
                "--worker-child", action="store_true", help=argparse.SUPPRESS
            )
        p.set_defaults(handler=handle)
    p = sub.add_parser("campaign")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reuse", action="store_true")
    p.set_defaults(handler=handle)
    p = sub.add_parser("_worker", help=argparse.SUPPRESS)
    p.add_argument("--request", required=True)
    p.set_defaults(handler=handle)
    p = sub.add_parser("analyze")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(handler=handle)
    p = sub.add_parser("export")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--run-root")
    p.set_defaults(handler=handle)
    p = sub.add_parser("test")
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--receipt", required=True)
    p.add_argument("--worker-child", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(handler=handle)
