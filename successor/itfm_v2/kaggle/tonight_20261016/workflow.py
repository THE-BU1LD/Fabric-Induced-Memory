"""Operator-supervised private Kaggle workflow; no tensor imports at import time.

The platform observations are supplied by a human/browser operator. This module
cannot stop a Kaggle allocation, read its quota, or establish remote persistence.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import stat
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

from budget import AdmissionError

DEADLINE = "2026-10-16T16:30:00+00:00"
SOURCE_SHA256 = "97a31bff8d785c37b0f2af6bfc374900670ed66bd554e17bb1d6525253a719f2"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def _zip_sha(archive_file, name):
    digest = hashlib.sha256()
    with archive_file.open(name) as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    from fim_sota.provenance import atomic_write_json
    atomic_write_json(Path(path), value)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _source_identity(source_root=None):
    from fim_sota import engineering
    if source_root is not None and Path(engineering.PACKAGE_ROOT).resolve() != Path(source_root).resolve():
        raise ValueError("imported maintained package differs from requested source root")
    source = engineering.source_identity()
    if source["sha256"] != SOURCE_SHA256:
        raise ValueError("source differs from the readiness snapshot")
    return source


def _versions():
    versions = {}
    for name in ("torch", "numpy", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_test_receipt(source_root, receipt, device):
    """Verify the actual maintained CLI receipt, including required executed test."""
    source = _source_identity(source_root)
    value = json.loads(Path(receipt).read_text())
    if value.get("status") != "PASS" or value.get("requested_device") != device:
        raise RuntimeError("requested-device test receipt did not pass")
    inventory, passed, failed, skipped = (value.get(key) for key in ("inventory", "passed", "failed", "skipped"))
    if not all(isinstance(items, list) for items in (inventory, passed, failed, skipped)):
        raise ValueError("test inventory fields are missing")
    if (not inventory or not passed or failed or len(set(inventory)) != len(inventory)
            or set(passed) & set(skipped) or set(passed) | set(skipped) != set(inventory)
            or not any(name.endswith("::test_requested_device_training_evaluation") for name in passed)):
        raise ValueError("test inventory incomplete or required device test was skipped")
    expected = {name: sha(Path(source_root) / "tests" / name)
                for name in ("test_training_recovery.py", "test_engineering_runtime.py")}
    if value.get("test_sources") != expected:
        raise ValueError("test receipt source inventory changed")
    check = value.get("environment", {})
    if (check.get("status") != "PASS_RUNTIME_ENVIRONMENT" or check.get("runtime_executed") is not True
            or check.get("device") != device or check.get("source") != source
            or check.get("packages") != {key: value for key, value in _versions().items() if key in {"torch", "numpy"}}
            or check.get("python") != platform.python_version()):
        raise ValueError("test receipt source/environment/runtime differs from current session")
    return value


def validate_profile(source_root, receipt, config=None):
    """Validate a real representative profile without importing a tensor library."""
    from fim_sota.engineering import digest
    source = _source_identity(source_root)
    path = Path(receipt)
    value = json.loads(path.read_text())
    configuration = value.get("configuration")
    if (value.get("status") != "PASS_PROFILE" or value.get("source") != source
            or not isinstance(configuration, dict) or value.get("config_sha256") != digest(configuration)
            or (config is not None and configuration != config)
            or value.get("profile_source_sha256") != sha(Path(__file__).with_name("device_profile.py"))
            or value.get("placement_verified") is not True):
        raise ValueError("representative profile identity or device placement failed")
    check = value.get("environment_preflight", {})
    if (check.get("status") != "PASS_RUNTIME_ENVIRONMENT" or check.get("runtime_executed") is not True
            or check.get("device") != configuration.get("device") or check.get("source") != source
            or check.get("packages") != {key: value for key, value in _versions().items() if key in {"torch", "numpy"}}
            or check.get("python") != platform.python_version()):
        raise ValueError("profile source/environment differs from this session")
    required = {"events.jsonl", "measured_timing.csv", "profile_checkpoint.pt", "source.json", "configuration.json", "environment.json"}
    if set(value.get("artifacts", {})) != required:
        raise ValueError("profile has no complete verified artifact inventory")
    accelerators = value.get("accelerators", {})
    if (str(accelerators.get("selected_device", "")).split(":")[0] != configuration["device"]
            or (configuration["device"] == "cuda" and accelerators.get("torch_visible_count") != 1)):
        raise ValueError("profile did not verify the declared single visible device")
    for name, expected in value.get("artifacts", {}).items():
        artifact = Path(name)
        if artifact.is_absolute() or ".." in artifact.parts or (path.parent / artifact).is_symlink():
            raise ValueError("unsafe profile artifact path")
        if sha(path.parent / artifact) != expected:
            raise ValueError("profile artifact changed")
    return value


def _profile_receipt(work):
    paths = [Path(work) / "representative_profile/profile.json", Path(work) / "representative_profile.json"]
    present = [path for path in paths if path.exists()]
    if len(present) != 1:
        raise ValueError("exactly one representative profile receipt must exist")
    return present[0]


def validate_cuda_gate(source_root, work):
    """Require the targeted executed CUDA tests before a finite matrix freeze."""
    import xml.etree.ElementTree as ET
    work, source_root = Path(work), Path(source_root)
    source = _source_identity(source_root)
    env = json.loads((work / "gpu_environment.json").read_text())
    if (env.get("status") != "PASS_RUNTIME_ENVIRONMENT" or env.get("source") != source
            or env.get("device") != "cuda" or env.get("runtime_executed") is not True):
        raise ValueError("actual CUDA runtime gate is absent or mismatched")
    receipt = json.loads((work / "cuda_selected_tests.json").read_text())
    expected = {name: sha(source_root / "tests" / name)
                for name in ("test_training_recovery.py", "test_engineering_runtime.py")}
    if (receipt.get("status") != "PASS" or receipt.get("source_sha256") != SOURCE_SHA256
            or receipt.get("requested_device") != "cuda" or receipt.get("precision") != "fp32"
            or receipt.get("test_sources") != expected
            or receipt.get("junit_sha256") != sha(work / "cuda_selected_tests.xml")):
        raise ValueError("targeted CUDA test gate does not identify the executed source")
    cases = ET.parse(work / "cuda_selected_tests.xml").findall(".//testcase")
    names = {"test_requested_device_training_evaluation",
             "test_train_select_checkpoint_and_replay_requested_cuda", "test_cuda_fp32_tiny_training"}
    if (len(cases) != 3 or {case.attrib.get("name") for case in cases} != names
            or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped"))):
        raise ValueError("the required CUDA tests did not all execute successfully")
    return receipt


def environment(source_root, output):
    source = _source_identity(source_root)
    versions = _versions()
    record = {"python": sys.version, "packages": versions, "source": source,
              "observed_at": now(), "device": "NOT_YET_INSPECTED"}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "environment_cpu_metadata.json"
    if target.exists():
        saved = json.loads(target.read_text())
        if any(saved.get(key) != record[key] for key in ("python", "packages", "source")):
            raise ValueError("resolved environment/source changed; old gates cannot be reused")
        record = saved
    else:
        write(target, record)
    requirements = "".join(f"{key}=={value}\n" for key, value in versions.items() if value)
    requirements_path = output / "requirements-resolved.txt"
    if requirements_path.exists() and requirements_path.read_text() != requirements:
        raise ValueError("resolved dependency lock changed")
    requirements_path.write_text(requirements)
    if not all(versions.values()):
        raise RuntimeError("missing dependency; stop on CPU and resolve explicitly, do not upgrade torch blindly")
    major_minor = tuple(int(x) for x in versions["torch"].split("+")[0].split(".")[:2])
    if major_minor < (2, 6):
        raise RuntimeError("installed torch is older than the maintained safe checkpoint requirement 2.6")
    return record


def supervised(command, logfile, timeout, *, budget=None, session_id=None):
    """Stream real child output; check wall/account/deadline guards every second."""
    import threading
    if not isinstance(command, (list, tuple)) or not command or any(not isinstance(item, str) or not item for item in command):
        raise ValueError("an explicit nonempty child argument list is required")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if budget is not None and not session_id:
        raise ValueError("budgeted child requires its reserved session ID")
    start = time.monotonic()
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("x") as handle:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, start_new_session=True,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1",
                                      "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                      "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                                      "PYTHONDONTWRITEBYTECODE": "1"})
        stream_errors = []
        def stream():
            try:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
                    handle.flush()
            except Exception as error:
                stream_errors.append(error)
        reader = threading.Thread(target=stream, daemon=True)
        reader.start()
        reason = None
        heartbeat = -30.0
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - start
                if stream_errors:
                    raise RuntimeError("child output stream failed") from stream_errors[0]
                if elapsed >= timeout:
                    reason = "per-run timeout"
                state = budget.snapshot() if budget else None
                if state:
                    estimate = state["estimates"]
                    session = state["sessions"][session_id]
                    observation = state.get("session_observation")
                    active = set(observation["active_session_ids"]) if observation else None
                    if active is None or active - {session_id}:
                        reason = "other or unknown active sessions require coordination; stopping only this child"
                    if session["state"] != "ALLOCATED_OBSERVED":
                        reason = "this allocation is no longer admitted"
                    reserve = state["authorized_hours"] * state["reserve_fraction"]
                    margin = max(120, reserve * 3600 / session["quota_multiplier"])
                    if (time.time() + margin >= state["deadline_utc_timestamp"] or
                            estimate["quota_spent_conservative_estimate"] >= state["authorized_hours"] - reserve or
                            estimate["quota_remaining_estimate"] <= 20 + reserve):
                        reason = "account/deadline guard requires export and shutdown"
                if elapsed - heartbeat >= 30:
                    print(json.dumps({"project": "FIM", "phase": "heartbeat", "time": now(),
                                      "elapsed_seconds": elapsed,
                                      "remaining_run_limit_seconds": max(0, timeout-elapsed),
                                      "budget": state["estimates"] if state else None}), flush=True)
                    heartbeat = elapsed
                if reason:
                    raise TimeoutError(reason)
                time.sleep(1)
            if child.returncode:
                raise RuntimeError(f"child exit {child.returncode}; see {logfile}")
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
        finally:
            reader.join(timeout=5)
            if reader.is_alive():
                # A descendant retaining stdout is still our launched process
                # group; never leave it running after its leader exits.
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                reader.join(timeout=5)
                if reader.is_alive():
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    reader.join(timeout=5)
            child.stdout.close()
            handle.flush()
            os.fsync(handle.fileno())
    if stream_errors:
        raise RuntimeError("child output stream failed") from stream_errors[0]
    return {"elapsed_seconds": time.monotonic()-start, "exit_code": child.returncode}


def cpu_gate(source_root, work):
    from fim_sota.engineering import load_config_file, validate_completed
    source_root, work = Path(source_root), Path(work)
    # The frozen recovery suite contains CUDA-availability-conditioned tests.
    # Refuse a visible driver GPU here: this phase must run before allocation.
    try:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10, check=False)
    except FileNotFoundError:
        probe = None
    if probe is not None and probe.returncode == 0 and probe.stdout.strip():
        raise RuntimeError("CPU gate must run with platform accelerator None before any GPU allocation")
    work.mkdir(parents=True, exist_ok=True)
    environment(source_root, work)
    script = source_root / "scripts/kaggle.py"
    receipt = work / "cpu_tests.json"
    if not receipt.exists():
        supervised([sys.executable, "-u", str(script), "test", "--device", "cpu",
                    "--receipt", str(receipt)], work / "cpu_tests.log", 340)
    result = validate_test_receipt(source_root, receipt, "cpu")
    cfg = load_config_file(source_root / "kaggle/smoke-cpu.json")
    destination = work / "cpu_smoke"
    if not destination.exists():
        supervised([sys.executable, "-u", str(script), "run", "--config",
                    str(source_root / "kaggle/smoke-cpu.json"), "--output", str(destination)],
                   work / "cpu_smoke.log", cfg["wall_seconds"]+35)
    validate_completed(destination, cfg, SOURCE_SHA256)
    write(work / "cpu_gate.json", {"status": "PASS", "source_sha256": SOURCE_SHA256,
                                  "source_root": str(source_root.resolve()),
                                  "environment_sha256": sha(work / "environment_cpu_metadata.json"),
                                  "smoke_completion_sha256": sha(destination / "completion.json"),
                                  "tests_sha256": sha(receipt), "time": now()})
    return result


def create_probe(work):
    work = Path(work)
    probe = work / "persistence_probe.json"
    if not probe.exists():
        write(probe, {"nonce": uuid.uuid4().hex, "source_sha256": SOURCE_SHA256,
                      "created_at": now(), "purpose": "private notebook output retrieval test"})
    return {"path": str(probe), "sha256": sha(probe)}


def verify_retrieval(original, retrieved, receipt, *, private_version_url):
    original, retrieved = Path(original).resolve(), Path(retrieved).resolve()
    if not str(retrieved).startswith("/kaggle/input/") or original == retrieved:
        raise ValueError("retrieve a saved private notebook output through Add Input; working copy is insufficient")
    if not private_version_url.startswith("https://www.kaggle.com/code/"):
        raise ValueError("record the actual private saved notebook version URL")
    if sha(original) != sha(retrieved):
        raise ValueError("retrieved artifact bytes differ")
    value = {"status": "PASS_RETRIEVED", "sha256": sha(original), "observed_at": now(),
             "original": str(original), "retrieved": str(retrieved),
             "private_version_url": private_version_url}
    write(receipt, value)
    return value


def require_gates(work):
    work = Path(work)
    cpu = json.loads((work / "cpu_gate.json").read_text())
    persist = json.loads((work / "persistence_verified.json").read_text())
    if cpu.get("status") != "PASS" or cpu.get("source_sha256") != SOURCE_SHA256:
        raise RuntimeError("CPU gate missing or stale")
    source_root = Path(cpu["source_root"])
    environment(source_root, work)
    validate_test_receipt(source_root, work / "cpu_tests.json", "cpu")
    if sha(work / "cpu_tests.json") != cpu.get("tests_sha256") or sha(work / "environment_cpu_metadata.json") != cpu.get("environment_sha256"):
        raise ValueError("CPU gate evidence changed")
    from fim_sota.engineering import load_config_file, validate_completed
    smoke = work / "cpu_smoke"
    validate_completed(smoke, load_config_file(source_root / "kaggle/smoke-cpu.json"), SOURCE_SHA256)
    if sha(smoke / "completion.json") != cpu.get("smoke_completion_sha256"):
        raise ValueError("CPU smoke completion evidence changed")
    if persist.get("status") != "PASS_RETRIEVED":
        raise RuntimeError("persistence retrieval gate missing")
    if sha(work / "persistence_probe.json") != persist.get("sha256"):
        raise RuntimeError("persistence probe changed")
    retrieved = Path(persist["retrieved"]).resolve()
    if not retrieved.is_relative_to(Path("/kaggle/input")) or sha(retrieved) != persist["sha256"]:
        raise RuntimeError("retrieved private persistence evidence is unavailable or changed")


def gpu_job(source_root, work, budget, session_id, run_id, command, seconds, *, verify_artifacts):
    _source_identity(source_root)
    require_gates(work)
    if not callable(verify_artifacts):
        raise ValueError("GPU job requires an artifact validator")
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run ID must be a safe path component")
    if (not isinstance(command, (list, tuple)) or not command
            or any(not isinstance(item, str) or not item for item in command)
            or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds <= 0 or seconds > 1200):
        raise ValueError("development readiness/profiling job exceeds small 20-minute bound")
    state = budget.snapshot()
    authorized = state["authorized_hours"]
    spent = state["estimates"]["quota_spent_conservative_estimate"]
    multiplier = state["sessions"][session_id]["quota_multiplier"]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in (authorized, spent, multiplier)):
        raise AdmissionError("readiness allocation needs finite observed-budget estimates")
    if authorized <= 0 or spent < 0 or multiplier <= 0:
        raise AdmissionError("readiness allocation budget is not admitted")
    # This account estimate includes setup, prior GPU checks, failures, and all
    # elapsed allocation idle time. The ceiling is shared, never reset per job.
    readiness_ceiling = authorized * 0.20
    proposed_debit = (seconds + 35) * multiplier / 3600
    if spent + proposed_debit > readiness_ceiling:
        raise AdmissionError("GPU readiness including elapsed allocation and this job would exceed 20% of authorized quota")
    reservation = budget.reserve_run(session_id, run_id, estimated_seconds=seconds+35,
                                     export_shutdown_seconds=120)
    try:
        result = supervised(command, Path(work) / f"{run_id}.log", seconds,
                            budget=budget, session_id=session_id)
        evidence = verify_artifacts()
        if not isinstance(evidence, dict) or evidence.get("status") not in {"PASS", "PASS_PROFILE", "PASS_ENGINEERING", "PASS_RUNTIME_ENVIRONMENT"}:
            raise ValueError("GPU child returned no validated successful artifact receipt")
        evidence_source = evidence.get("source_sha256") or evidence.get("source", {}).get("sha256") or evidence.get("environment", {}).get("source", {}).get("sha256")
        if evidence_source != SOURCE_SHA256:
            raise ValueError("GPU artifact receipt does not identify the frozen source")
        write(Path(work) / f"{run_id}.timing.json", {**result, "reservation": reservation,
                                                   "validated_evidence": evidence,
                                                   "class": "DEVELOPMENT", "time": now()})
        budget.finish_run(session_id, run_id, status="SUCCEEDED", artifacts_verified=True)
        return result
    except BaseException as error:
        budget.finish_run(session_id, run_id, status="INTERRUPTED" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "FAILED")
        raise


def freeze(candidate_path, profile_run, work, budget, session_id):
    from fim_sota.engineering import resolve, validate_completed, digest, data_identity
    require_gates(work)
    _source_identity()
    candidate = json.loads(Path(candidate_path).read_text())
    profile_run, work = Path(profile_run), Path(work)
    pilot_cfg = json.loads((profile_run / "config.json").read_text())
    validate_completed(profile_run, pilot_cfg, SOURCE_SHA256)
    from fim_sota.engineering import PACKAGE_ROOT
    validate_cuda_gate(PACKAGE_ROOT, work)
    cpu_profile = validate_profile(PACKAGE_ROOT, work / "cpu_representative_profile/profile.json")
    cpu_cfg = json.loads((work / "cpu_pilot/config.json").read_text())
    validate_completed(work / "cpu_pilot", cpu_cfg, SOURCE_SHA256)
    cpu_timing = json.loads((work / "cpu_pilot.timing.json").read_text())
    if (cpu_timing.get("exit_code") != 0 or cpu_cfg.get("device") != "cpu"
            or cpu_profile["configuration"].get("device") != "cpu"):
        raise ValueError("representative CPU timing and completed artifacts are required")
    profile_path = _profile_receipt(work)
    profile = validate_profile(PACKAGE_ROOT, profile_path)
    if pilot_cfg["seed"] not in (101, 202, 303):
        raise ValueError("pilot seed was not reserved for development")
    base = candidate["candidate_common_config"]
    if base.get("device") != "cuda" or pilot_cfg.get("device") != "cuda":
        raise ValueError("this GPU campaign requires a representative CUDA pilot")
    for key in base:
        if key not in ("epochs", "wall_seconds", "checkpoint_seconds", "example_budget") and base[key] != pilot_cfg[key]:
            raise ValueError(f"pilot did not use representative {key}")
        if key not in ("epochs", "wall_seconds", "checkpoint_seconds", "example_budget") and base[key] != profile["configuration"][key]:
            raise ValueError(f"microprofile did not use representative {key}")
        if key not in ("device", "epochs", "wall_seconds", "checkpoint_seconds", "example_budget"):
            if base[key] != cpu_cfg[key] or base[key] != cpu_profile["configuration"][key]:
                raise ValueError(f"CPU pilot did not use representative {key}")
    timing = json.loads((work / "pilot.timing.json").read_text())
    if timing.get("exit_code") != 0 or timing.get("validated_evidence", {}).get("status") != "PASS_ENGINEERING":
        raise ValueError("pilot timing has no verified end-to-end completion")
    elapsed = timing["elapsed_seconds"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("pilot elapsed time must be measured, finite and positive")
    # Deliberately pessimistic: scales the entire measured run, including its
    # evaluation, by the epoch ratio, then doubles and adds startup overhead.
    seconds = math.ceil(2 * elapsed * base["epochs"] / pilot_cfg["epochs"] + 120)
    if seconds > 3600:
        raise AdmissionError("measured full matrix exceeds supported one-hour cell cap; revise prospectively")
    cells = [{"id": f"periodic-seed-{seed}", "mandatory": True,
              "config": resolve({**base, "seed": seed, "wall_seconds": seconds,
                                  "checkpoint_seconds": 300})}
             for seed in candidate["outcome_access"]["candidate_seeds"]]
    if not cells or len({cell["id"] for cell in cells}) != len(cells):
        raise ValueError("candidate requires nonempty unique independent seeds")
    state = budget.snapshot()
    budget._guard_observations(state, expected_active=(session_id,))
    multiplier = state["sessions"][session_id]["quota_multiplier"]
    reserve = state["authorized_hours"] * state["reserve_fraction"]
    total = sum(cell["config"]["wall_seconds"]+35 for cell in cells)
    margin = max(120, reserve*3600/multiplier)
    if (state["estimates"]["quota_spent_conservative_estimate"]+total*multiplier/3600 > state["authorized_hours"]-reserve
            or state["estimates"]["quota_remaining_estimate"] - (total+margin)*multiplier/3600 < 20
            or time.time()+total+margin > state["deadline_utc_timestamp"]):
        raise AdmissionError("complete matrix cannot fit; no candidate test cells launched")
    value = {"schema_version": 1, "status": "FROZEN_DEVELOPMENT", "study_kind": "DEVELOPMENT",
             "source_sha256": SOURCE_SHA256, "created_at": now(), "cells": cells,
             "completion_deadline_utc_timestamp": state["deadline_utc_timestamp"],
             "account_budget_snapshot_at_freeze": state,
             "candidate_sha256": sha(candidate_path), "protocol": candidate,
             "expected_cost_seconds": total, "estimation_rule": "2x end-to-end pilot scaled by epoch ratio plus120s/cell",
             "environment_sha256": sha(work / "environment_cpu_metadata.json"),
             "profile_sha256": sha(profile_path),
             "analysis_sha256": sha(Path(__file__).parent / "analysis.py"),
             "data_identities": {c["id"]: data_identity(c["config"]) for c in cells},
             "outcome_access": "New cells remain DEVELOPMENT; no promotion to protected confirmation",
             "planned_value_interventions": "Existing erased/spatial_mean controls only; additional derived interventions are not admitted",
             "primary_comparison": "memory_vs_base"}
    target = work / "frozen_manifest.json"
    if target.exists():
        raise FileExistsError("manifest already frozen; never silently refreeze")
    write(target, value)
    (work / "frozen_manifest.sha256").write_text(digest(value)+"\n")
    return value


def run_next(source_root, work, budget, session_id):
    from fim_sota.engineering import validate_completed, digest
    work, source_root = Path(work), Path(source_root)
    _source_identity(source_root)
    require_gates(work)
    manifest = json.loads((work / "frozen_manifest.json").read_text())
    if digest(manifest) != (work / "frozen_manifest.sha256").read_text().strip():
        raise ValueError("frozen manifest changed")
    if (manifest.get("source_sha256") != SOURCE_SHA256
            or manifest.get("environment_sha256") != sha(work / "environment_cpu_metadata.json")
            or manifest.get("profile_sha256") != sha(_profile_receipt(work))
            or manifest.get("analysis_sha256") != sha(Path(__file__).with_name("analysis.py"))):
        raise ValueError("frozen source, environment, profile or analysis changed")
    validate_profile(source_root, _profile_receipt(work))
    ledger_path = work / "run_ledger.json"
    identity = digest(manifest)
    state = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "source_sha256": SOURCE_SHA256, "manifest_sha256": identity,
        "cells": {cell["id"]: {"state": "PENDING", "path": f"cells/{cell['id']}", "seed": cell["config"]["seed"]}
                  for cell in manifest["cells"]}}
    if (state.get("source_sha256") != SOURCE_SHA256 or state.get("manifest_sha256") != identity
            or set(state["cells"]) != {cell["id"] for cell in manifest["cells"]}):
        raise ValueError("run ledger source/manifest/expected cell inventory changed")
    write(ledger_path, state)
    for cell in manifest["cells"]:
        name, cfg = cell["id"], cell["config"]
        prior = state["cells"].get(name)
        root = work / "cells" / name
        if prior:
            if prior["state"] == "SUCCEEDED":
                validate_completed(root, cfg, manifest["source_sha256"])
                continue
            if prior["state"] != "PENDING":
                raise RuntimeError("attempted cell retained; diagnose and use explicit resume lineage, no silent retry")
        if root.exists() or (work / f"{name}.log").exists():
            raise RuntimeError("partial artifacts require explicit checkpoint-resume lineage; no silent retry")
        write(ledger_path, state)
        try:
            budget.reserve_run(session_id, name, estimated_seconds=cfg["wall_seconds"]+35,
                               export_shutdown_seconds=120)
        except AdmissionError as error:
            for entry in state["cells"].values():
                if entry["state"] == "PENDING":
                    entry.update(state="SKIPPED_BUDGET", reason=str(error))
            write(ledger_path, state)
            raise
        state["cells"][name].update(state="RUNNING", started_at=now())
        write(ledger_path, state)
        config_path = work / f"{name}.config.json"
        write(config_path, cfg)
        try:
            timing = supervised([sys.executable, "-u", str(source_root / "scripts/kaggle.py"),
                                 "run", "--config", str(config_path), "--output", str(root)],
                                work / f"{name}.log", cfg["wall_seconds"]+30,
                                budget=budget, session_id=session_id)
            validate_completed(root, cfg, manifest["source_sha256"])
            state["cells"][name].update(state="SUCCEEDED", timing=timing, finished_at=now())
            budget.finish_run(session_id, name, status="SUCCEEDED", artifacts_verified=True)
        except BaseException as error:
            status = "INTERRUPTED" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "FAILED"
            state["cells"][name].update(state=status, reason=str(error), traceback=traceback.format_exc())
            budget.finish_run(session_id, name, status=status)
            raise
        finally:
            write(ledger_path, state)
        return name  # Persist this cell and refresh real platform observations before next.
    return "ALL_REQUIRED_CELLS_VERIFIED"


def archive(work, destination):
    """Only this dedicated campaign directory, never arbitrary repository files."""
    work, destination = Path(work).resolve(), Path(destination).resolve()
    if destination.is_relative_to(work) or destination.exists():
        raise ValueError("archive must be new and outside the campaign directory")
    files = sorted(p for p in work.rglob("*") if p.is_file() and not p.name.endswith(".lock"))
    if (any(p.is_symlink() for p in work.rglob("*")) or (work / "ARCHIVE_MANIFEST.json").exists()
            or sum(p.stat().st_size for p in files) > 10*1024**3):
        raise ValueError("unsafe path or archive exceeds conservative10GiB ceiling")
    inventory = {str(p.relative_to(work)): sha(p) for p in files}
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as output:
        for p in files:
            output.write(p, p.relative_to(work))
        output.writestr("ARCHIVE_MANIFEST.json", json.dumps(inventory, sort_keys=True))
    with zipfile.ZipFile(destination) as check:
        if any(_zip_sha(check, n) != h for n, h in inventory.items()):
            raise ValueError("archive readback failed")
    return {"path": str(destination), "sha256": sha(destination), "files": len(files),
            "local_readback_verified": True, "remote_retrieval_verified": False}


def restore(archive_path, destination, expected_sha256, *, source_root=None):
    """Verify and extract a fresh copy; preserve original attempts and stop states.

    No GPU session, retry, or resume is launched. Re-run cpu_gate explicitly to
    rebind paths and validate the current environment, then recheck platform quota
    and session state. A restored local archive does not prove remote persistence.
    """
    from fim_sota.engineering import PACKAGE_ROOT, digest, validate_completed
    archive_path, destination = Path(archive_path).resolve(), Path(destination).resolve()
    source_root = Path(PACKAGE_ROOT if source_root is None else source_root)
    _source_identity(source_root)
    if destination.exists():
        raise FileExistsError("restore requires a new destination")
    if sha(archive_path) != expected_sha256:
        raise ValueError("retrieved archive checksum mismatch")
    with zipfile.ZipFile(archive_path) as source:
        members = source.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)) or len(names) > 20000 or sum(member.file_size for member in members) > 10*1024**3:
            raise ValueError("duplicate entries or excessive archive expansion")
        if "ARCHIVE_MANIFEST.json" not in names:
            raise ValueError("archive has no checksum inventory")
        if source.getinfo("ARCHIVE_MANIFEST.json").file_size > 8*1024**2:
            raise ValueError("archive inventory exceeds 8 MiB bound")
        for member in members:
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts or member.is_dir() or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("unsafe archive member")
        inventory = json.loads(source.read("ARCHIVE_MANIFEST.json"))
        if not isinstance(inventory, dict) or set(inventory) != set(names) - {"ARCHIVE_MANIFEST.json"}:
            raise ValueError("archive contents differ from checksum inventory")
        # Validate every member before extracting any byte.
        for name, expected in inventory.items():
            if _zip_sha(source, name) != expected:
                raise ValueError("archive member checksum mismatch")
        destination.mkdir(parents=True, exist_ok=False)
        for name in inventory:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(name) as incoming, target.open("xb") as outgoing:
                for block in iter(lambda: incoming.read(1024*1024), b""):
                    outgoing.write(block)
                outgoing.flush()
                os.fsync(outgoing.fileno())
    if any(sha(destination / name) != expected for name, expected in inventory.items()):
        raise ValueError("restored bytes changed on readback")
    manifest_path = destination / "frozen_manifest.json"
    verified = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("source_sha256") != SOURCE_SHA256
                or digest(manifest) != (destination / "frozen_manifest.sha256").read_text().strip()
                or manifest.get("analysis_sha256") != sha(Path(__file__).with_name("analysis.py"))):
            raise ValueError("restored frozen protocol/source/analysis changed")
        ledger_path = destination / "run_ledger.json"
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
            if ledger.get("manifest_sha256") != digest(manifest) or ledger.get("source_sha256") != SOURCE_SHA256:
                raise ValueError("restored ledger does not match frozen protocol")
            expected = {cell["id"]: cell for cell in manifest["cells"]}
            if set(ledger["cells"]) != set(expected):
                raise ValueError("restored ledger expected-cell inventory changed")
            for name, state in ledger["cells"].items():
                if state["state"] == "SUCCEEDED":
                    validate_completed(destination / "cells" / name, expected[name]["config"], SOURCE_SHA256)
                    verified.append(name)
    receipt = {"status": "PASS_LOCAL_RESTORE", "archive_sha256": expected_sha256,
               "files_verified": len(inventory), "completed_cells_verified": verified,
               "source_sha256": SOURCE_SHA256, "remote_retrieval_verified": False,
               "launch_or_resume_performed": False, "platform_state_requires_fresh_observation": True}
    write(destination / "restore_receipt.json", receipt)
    return receipt
