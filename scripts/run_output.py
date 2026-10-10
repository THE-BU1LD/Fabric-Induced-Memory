"""Fresh-run admission and single-writer, failure-preserving JSON publication."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


def _require_absent(path: Path) -> None:
    # exists() alone does not detect a dangling symlink.
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Run output already exists; choose a fresh destination: {path}")


def write_run_manifest(path: Path, payload: dict[str, Any], *, create: bool = False) -> None:
    """Publish a complete JSON snapshot without truncating an earlier snapshot.

    Initial publication uses an exclusive hard link, so an intervening creator
    cannot be overwritten. Updates assume one writer and reject nonregular
    destinations. Serialization, flush and replacement failures leave earlier
    manifest bytes intact. This is not a multi-file or power-loss transaction.
    """
    path = Path(path)
    if create:
        _require_absent(path)
        mode = None
    else:
        destination = path.lstat()
        if not stat.S_ISREG(destination.st_mode):
            raise ValueError(f"Manifest destination must be a regular file: {path}")
        mode = stat.S_IMODE(destination.st_mode)

    staged: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            staged = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            staged.chmod(mode)
        if create:
            # Both paths are siblings, so publication cannot cross filesystems.
            # link() refuses an existing file, directory or dangling symlink.
            os.link(staged, path)
        else:
            destination = path.lstat()
            if not stat.S_ISREG(destination.st_mode):
                raise ValueError(f"Manifest destination must be a regular file: {path}")
            os.replace(staged, path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def prepare_run_outputs(
    results_root: Path, manifest: Path, initial_payload: dict[str, Any],
) -> None:
    """Reserve fresh results and manifest destinations before any experiment."""
    results_root, manifest = Path(results_root), Path(manifest)
    _require_absent(results_root)
    _require_absent(manifest)
    resolved_root, resolved_manifest = results_root.resolve(), manifest.resolve()
    if resolved_root == resolved_manifest or resolved_root.is_relative_to(resolved_manifest):
        raise ValueError("The manifest must not equal or contain the results directory")

    # No exist_ok: a competing run cannot turn this into reuse of its directory.
    results_root.mkdir(parents=True, exist_ok=False)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_run_manifest(manifest, initial_payload, create=True)
