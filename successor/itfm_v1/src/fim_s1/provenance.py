from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(project_root: Path) -> dict[str, Any]:
    paths = {
        path
        for parent in (
            project_root / "src",
            project_root / "protocols",
            project_root / "scripts",
            project_root / "tests",
            project_root / "datasets" / "external",
        )
        for path in parent.rglob("*")
        if path.is_file() and path.suffix in {".py", ".json", ".csv", ".txt"}
    }
    for relative in (
        "pyproject.toml",
        "PROJECT_CONTRACT.md",
        "METHODOLOGY.md",
        "LITERATURE_BOUNDARY.md",
        "datasets/README.md",
        "paper/SPECIFICATION.md",
    ):
        candidate = project_root / relative
        if candidate.is_file():
            paths.add(candidate)
    digest = hashlib.sha256()
    files = []
    for path in sorted(paths):
        relative = path.relative_to(project_root).as_posix()
        file_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        files.append({"path": relative, "sha256": file_hash})
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_head = None
        git_dirty = None
    return {
        "sha256": digest.hexdigest(),
        "files": files,
        "git_head": git_head,
        "git_tracked_dirty": git_dirty,
    }


def environment_record() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device_mps_available": torch.backends.mps.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "pid": os.getpid(),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
