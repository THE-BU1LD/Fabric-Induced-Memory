"""Stage related evidence files and roll back ordinary publication failures.

This does not claim crash-atomic visibility for files in different directories.
Readers must wait for the producing command to succeed. No scientific data is
read here; callers supply fully rendered UTF-8 payloads after admission.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Iterable


class PublicationRollbackError(RuntimeError):
    """Publication failed and one or more previous outputs need recovery."""

    def __init__(self, original_error: Exception, recovery: list[dict]) -> None:
        self.original_error = original_error
        self.recovery = recovery
        super().__init__(
            "Evidence publication failed and rollback was incomplete; "
            f"preserved recovery files: {recovery!r}"
        )


def _stage_bytes(destination: Path, payload: bytes) -> Path:
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    staged = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return staged
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def publish_evidence(
    outputs: Iterable[tuple[Path, str]], *, protected_inputs: Iterable[Path] = ()
) -> None:
    """Publish a rendered set, preserving previous bytes on recoverable errors.

    Every file is staged before replacing an output. Backups are staged on the
    same filesystem as each destination, so rollback uses rename rather than a
    second content write. Existing symlinks and non-regular destinations are
    rejected. Outputs may not alias one another or a protected input by path.

    Requires a single writer and quiescent readers. A killed process, power
    loss, concurrent writers/readers, and a failed rollback are outside the
    all-or-restored guarantee; failed rollback leaves explicit recovery paths.
    """
    protected = {Path(path).resolve() for path in protected_inputs}
    targets: list[tuple[Path, bytes, int | None]] = []
    seen: set[Path] = set()
    for path, payload in outputs:
        destination = Path(path).absolute()
        identity = destination.resolve()
        if identity in seen:
            raise ValueError(f"Evidence destinations must be distinct: {destination}")
        if identity in protected:
            raise ValueError(f"Evidence output would replace its input: {destination}")
        seen.add(identity)
        mode = None
        try:
            info = destination.lstat()
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"Evidence destination must be a regular file: {destination}")
            mode = stat.S_IMODE(info.st_mode)
        targets.append((destination, payload.encode("utf-8"), mode))
    if not targets:
        raise ValueError("At least one evidence output is required")

    staged: list[dict] = []
    published: list[dict] = []
    keep: set[Path] = set()
    try:
        for destination, payload, mode in targets:
            destination.parent.mkdir(parents=True, exist_ok=True)
            entry = {"destination": destination, "next": None, "previous": None}
            staged.append(entry)
            if mode is not None:
                entry["previous"] = _stage_bytes(destination, destination.read_bytes())
                os.chmod(entry["previous"], mode)
            entry["next"] = _stage_bytes(destination, payload)
            if mode is not None:
                os.chmod(entry["next"], mode)
        try:
            for entry in staged:
                os.replace(entry["next"], entry["destination"])
                published.append(entry)
        except Exception as original_error:
            recovery = []
            for entry in reversed(published):
                try:
                    if entry["previous"] is None:
                        entry["destination"].unlink()
                    else:
                        os.replace(entry["previous"], entry["destination"])
                except Exception as rollback_error:
                    if entry["previous"] is not None:
                        keep.add(entry["previous"])
                    recovery.append({
                        "destination": str(entry["destination"]),
                        "previous_bytes": str(entry["previous"]) if entry["previous"] else None,
                        "action": "restore previous_bytes" if entry["previous"] else "remove new output",
                        "error": repr(rollback_error),
                    })
            if recovery:
                raise PublicationRollbackError(original_error, recovery) from original_error
            raise
    finally:
        for entry in staged:
            for kind in ("next", "previous"):
                temporary = entry[kind]
                if temporary is not None and temporary not in keep:
                    temporary.unlink(missing_ok=True)
