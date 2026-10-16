"""Bounded, single-writer, append-only evidence journal for local POSIX disks.

Hashes detect changes only relative to a separately retained anchor. They do not
authenticate the writer. Uncommitted temporary files are retained after faults.
"""

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from integrity import ev

MAX_BYTES = 32 * 1024 * 1024
MAX_RECORDS = 16384
ZERO = "0" * 64


def encode(value):
    return json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def read_bounded(path, limit=ev.MAX_INPUT_BYTES):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("Regular bounded file required")
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("File grew beyond size limit")
    return raw


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(path, raw):
    """Publish one fsynced file without replacement; preserve ambiguous failures."""
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    sync_directory(path.parent)
    os.unlink(temporary)
    sync_directory(path.parent)


class Journal:
    def __init__(self, directory, *, create=False, expected_anchor=None):
        self.directory = Path(directory)
        self._fd = None
        self._poisoned = False
        self._records = []
        self._bytes = 0
        self._tip = ZERO
        if create:
            self.directory.mkdir(mode=0o700)
            sync_directory(self.directory.parent)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise ValueError("Journal must be a real directory")
        try:
            self._fd = os.open(
                self.directory / ".writer.lock",
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
            )
            if not stat.S_ISREG(os.fstat(self._fd).st_mode):
                raise ValueError("Regular lock file required")
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._load()
            if create and self._records:
                raise ValueError("New journal unexpectedly contains records")
            if not create:
                if expected_anchor is None:
                    raise ValueError("Recovery requires a separately retained anchor")
                ev.object_fields(expected_anchor, ("records", "sha256"))
                ev.integer(expected_anchor["records"], "anchor records", 1, MAX_RECORDS)
                ev.digest(expected_anchor["sha256"])
                if expected_anchor != self.anchor:
                    raise ValueError(
                        "Journal does not match the expected complete history"
                    )
        except BaseException:
            self.close()
            raise

    def _load(self):
        names = []
        total_entries = 0
        with os.scandir(self.directory) as entries:
            for entry in entries:
                total_entries += 1
                if total_entries > 2 * MAX_RECORDS + 1:
                    raise ValueError("Journal directory entry budget exceeded")
                if entry.name == ".writer.lock" or entry.name.startswith(".pending-"):
                    continue
                if not re.fullmatch(r"[0-9]{8}\.json", entry.name):
                    raise ValueError("Unexpected journal entry")
                names.append(entry.name)
        if len(names) > MAX_RECORDS:
            raise ValueError("Journal record budget exceeded")
        for sequence, name in enumerate(sorted(names)):
            if name != f"{sequence:08d}.json":
                raise ValueError("Missing or reordered journal record")
            raw = read_bounded(self.directory / name)
            self._bytes += len(raw)
            if self._bytes > MAX_BYTES:
                raise ValueError("Journal byte budget exceeded")
            record = ev.strict_json(raw)
            ev.object_fields(record, ("body", "sha256"))
            body = record["body"]
            ev.object_fields(body, ("sequence", "previous", "event"))
            ev.integer(body["sequence"], "sequence", 0, MAX_RECORDS - 1)
            if body["sequence"] != sequence or body["previous"] != self._tip:
                raise ValueError("Journal chain mismatch")
            if ev.digest(record["sha256"]) != digest(body):
                raise ValueError("Journal checksum mismatch")
            self._records.append(body["event"])
            self._tip = record["sha256"]

    @property
    def anchor(self):
        return {"records": len(self._records), "sha256": self._tip}

    @property
    def events(self):
        return copy.deepcopy(self._records)

    def append(self, event):
        if self._fd is None or self._poisoned:
            raise RuntimeError(
                "Journal closed or poisoned; recover from a retained anchor"
            )
        body = {"sequence": len(self._records), "previous": self._tip, "event": event}
        value = {"body": body, "sha256": digest(body)}
        raw = encode(value)
        if len(raw) > ev.MAX_INPUT_BYTES or self._bytes + len(raw) > MAX_BYTES:
            raise ValueError("Journal byte budget exhausted")
        if len(self._records) >= MAX_RECORDS:
            raise ValueError("Journal record budget exhausted")
        # Detach all caller-owned objects before the durable commit.
        event = ev.strict_json(raw)["body"]["event"]
        try:
            publish(self.directory / f"{len(self._records):08d}.json", raw)
        except BaseException:
            self._poisoned = True
            raise
        self._records.append(event)
        self._bytes += len(raw)
        self._tip = value["sha256"]
        return self.anchor

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
