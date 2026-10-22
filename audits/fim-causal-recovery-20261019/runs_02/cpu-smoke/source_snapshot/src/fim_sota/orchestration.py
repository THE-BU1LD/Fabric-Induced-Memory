from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fim_v2.resources import resource_admission

from .provenance import (
    JsonlLogger,
    atomic_write_json,
    immutable_run_id,
    sha256_file,
)


class StageAdmissionError(RuntimeError):
    """Raised when an orchestration stage is not authorized to execute."""


@dataclasses.dataclass(frozen=True)
class BoundInput:
    path: str
    sha256: str


@dataclasses.dataclass(frozen=True)
class StageSpec:
    schema_version: int
    stage_id: str
    command: tuple[str, ...]
    working_directory: str
    inputs: tuple[BoundInput, ...]
    expected_outputs: tuple[str, ...]
    protected_test: bool = False
    frozen_development_receipt: str | None = None
    wall_clock_budget_seconds: int = 3600

    def validate(self) -> None:
        if self.schema_version != 1 or not self.stage_id or not self.command:
            raise StageAdmissionError("invalid stage specification")
        if self.wall_clock_budget_seconds < 1:
            raise StageAdmissionError("stage wall-clock budget must be positive")
        if self.protected_test and not self.frozen_development_receipt:
            raise StageAdmissionError(
                "protected stage requires a frozen development receipt"
            )
        for item in self.inputs:
            if len(item.sha256) != 64:
                raise StageAdmissionError(f"input has invalid SHA-256: {item.path}")


def load_stage_spec(path: str | Path) -> StageSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    inputs = tuple(BoundInput(**item) for item in payload.pop("inputs", []))
    payload["command"] = tuple(payload["command"])
    payload["expected_outputs"] = tuple(payload.get("expected_outputs", []))
    spec = StageSpec(inputs=inputs, **payload)
    spec.validate()
    return spec


def verify_frozen_development_receipt(path: str | Path) -> Mapping[str, Any]:
    receipt_path = Path(path)
    if not receipt_path.is_file():
        raise StageAdmissionError("frozen development receipt is missing")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "status": "FROZEN_DEVELOPMENT_PASS",
        "protected_test_unlocked": True,
        "protocol_frozen": True,
    }
    failed = {
        key: value for key, value in required.items() if payload.get(key) != value
    }
    if failed:
        raise StageAdmissionError(
            f"development receipt does not unlock protected test: {failed}"
        )
    for identity in ("source_sha256", "config_sha256", "protocol_sha256"):
        if len(str(payload.get(identity, ""))) != 64:
            raise StageAdmissionError(f"development receipt lacks {identity}")
    return payload


class StageRunner:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _validate(self, spec: StageSpec) -> Mapping[str, Any]:
        working = Path(spec.working_directory).resolve()
        if not working.is_dir():
            raise StageAdmissionError("stage working directory does not exist")
        verified_inputs: list[Mapping[str, Any]] = []
        for item in spec.inputs:
            path = Path(item.path).resolve()
            if not path.is_file():
                raise StageAdmissionError(f"bound input is missing: {path}")
            actual = sha256_file(path)
            if actual != item.sha256:
                raise StageAdmissionError(
                    f"bound input changed: expected {item.sha256}, got {actual}: {path}"
                )
            verified_inputs.append({"path": str(path), "sha256": actual})
        frozen = None
        if spec.protected_test:
            frozen = verify_frozen_development_receipt(
                str(spec.frozen_development_receipt)
            )
        return {
            "working_directory": str(working),
            "inputs": verified_inputs,
            "protected_test": spec.protected_test,
            "frozen_development": frozen,
        }

    def run(
        self,
        spec: StageSpec,
        *,
        execute: bool,
        run_id: str | None = None,
    ) -> Mapping[str, Any]:
        spec.validate()
        validated = self._validate(spec)
        identity = run_id or immutable_run_id(spec.stage_id)
        run_root = self.root / identity
        if run_root.exists():
            raise FileExistsError(f"immutable run already exists: {run_root}")
        run_root.mkdir(parents=True)
        logger = JsonlLogger(run_root / "events.jsonl", run_id=identity)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": identity,
            "stage_id": spec.stage_id,
            "status": "VALIDATING",
            "created_at_unix_ns": time.time_ns(),
            "spec": dataclasses.asdict(spec),
            "validation": validated,
            "failures": [],
            "outputs": [],
        }
        manifest_path = run_root / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        if not execute:
            manifest["status"] = "DRY_RUN_VALIDATED"
            manifest["note"] = "No training, evaluation, or protected access executed."
            atomic_write_json(manifest_path, manifest)
            logger.log("dry_run_validated", stage_id=spec.stage_id)
            return manifest
        admission = resource_admission()
        manifest["resource_admission"] = admission
        if admission.get("status") != "PASS":
            manifest["status"] = "BLOCKED_RESOURCE"
            atomic_write_json(manifest_path, manifest)
            logger.log("blocked_resource", resource=admission)
            return manifest
        manifest["status"] = "RUNNING"
        manifest["started_at_unix_ns"] = time.time_ns()
        atomic_write_json(manifest_path, manifest)
        logger.log("stage_started", command=list(spec.command))
        try:
            completed = subprocess.run(
                list(spec.command),
                cwd=spec.working_directory,
                check=False,
                capture_output=True,
                text=True,
                timeout=spec.wall_clock_budget_seconds,
            )
        except subprocess.TimeoutExpired as error:
            manifest["status"] = "FAILED_TIMEOUT"
            manifest["failures"].append(
                {
                    "type": "TimeoutExpired",
                    "message": str(error),
                    "settings_changed": False,
                    "silent_retry": False,
                }
            )
            atomic_write_json(manifest_path, manifest)
            return manifest
        (run_root / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (run_root / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            manifest["status"] = "FAILED_COMMAND"
            manifest["failures"].append(
                {
                    "type": "NonZeroExit",
                    "returncode": completed.returncode,
                    "settings_changed": False,
                    "silent_retry": False,
                }
            )
        else:
            missing_outputs: list[str] = []
            for output_name in spec.expected_outputs:
                output = (Path(spec.working_directory) / output_name).resolve()
                if not output.is_file():
                    missing_outputs.append(str(output))
                    continue
                manifest["outputs"].append(
                    {
                        "path": str(output),
                        "sha256": sha256_file(output),
                        "bytes": output.stat().st_size,
                    }
                )
            if missing_outputs:
                manifest["status"] = "FAILED_MISSING_OUTPUT"
                manifest["failures"].append(
                    {"type": "MissingOutput", "paths": missing_outputs}
                )
            else:
                manifest["status"] = "PASS_EXECUTED_UNVERIFIED"
        manifest["finished_at_unix_ns"] = time.time_ns()
        manifest["stdout_sha256"] = sha256_file(run_root / "stdout.txt")
        manifest["stderr_sha256"] = sha256_file(run_root / "stderr.txt")
        atomic_write_json(manifest_path, manifest)
        logger.log("stage_finished", status=manifest["status"])
        return manifest


def fetch_asset(
    *,
    url: str,
    destination: str | Path,
    expected_sha256: str,
    expected_bytes: int,
    license_receipt: str | Path,
) -> Mapping[str, Any]:
    license_payload = json.loads(Path(license_receipt).read_text(encoding="utf-8"))
    if license_payload.get("status") != "ADMITTED":
        raise StageAdmissionError("data license is not admitted")
    if not url.startswith("https://"):
        raise StageAdmissionError("only HTTPS dataset sources are allowed")
    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset asset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".download", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            temporary.open("wb") as handle,
        ):
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        actual_sha = sha256_file(temporary)
        actual_bytes = temporary.stat().st_size
        if actual_sha != expected_sha256 or actual_bytes != expected_bytes:
            raise StageAdmissionError(
                f"download identity mismatch: sha={actual_sha} bytes={actual_bytes}"
            )
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "status": "FETCHED_VERIFIED",
        "url": url,
        "path": str(output.resolve()),
        "sha256": expected_sha256,
        "bytes": expected_bytes,
        "license_receipt": str(Path(license_receipt).resolve()),
    }
