from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

CHECKPOINT_SCHEMA_VERSION = 1


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_write_json(
    path: str | Path, payload: object, *, overwrite: bool = True
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            temporary.replace(destination)
        else:
            # Atomic no-replace publication, including concurrent writers.
            os.link(temporary, destination)
            temporary.unlink()
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def immutable_run_id(prefix: str = "run") -> str:
    entropy = random.SystemRandom().getrandbits(48)
    return f"{prefix}-{time.time_ns()}-{os.getpid()}-{entropy:012x}"


def hash_manifest(paths: list[Path], root: Path) -> dict[str, Any]:
    records = [
        {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
        for path in sorted(paths)
        if path.is_file()
    ]
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["path"].encode())
        digest.update(record["sha256"].encode())
    return {"sha256": digest.hexdigest(), "files": records}


class JsonlLogger:
    """Small structured logger with one fsync-safe JSON event per line."""

    def __init__(self, path: str | Path, **context: object) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context = dict(context)

    def log(self, event: str, **fields: object) -> None:
        payload = {
            "schema_version": 1,
            "time_unix_ns": time.time_ns(),
            "event": event,
            **self.context,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def require_sha256(path: str | Path, expected: str) -> None:
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError("expected digest is not lowercase SHA-256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"identity mismatch for {path}: expected {expected}, got {actual}"
        )


def validate_checkpoint_envelope(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    required = {
        "run_id",
        "source_sha256",
        "config_sha256",
        "step",
        "model_state",
        "optimizer_state",
        "random_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"checkpoint missing fields: {missing}")


@contextlib.contextmanager
def stage_timer() -> Iterator[dict[str, float]]:
    started = time.perf_counter()
    result: dict[str, float] = {}
    try:
        yield result
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
