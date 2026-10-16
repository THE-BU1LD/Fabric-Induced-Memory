from __future__ import annotations

import csv
import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .data import AliasedAdvectionCorpus, WindowRecord
from .provenance import atomic_json, sha256_file


@dataclass(frozen=True)
class PDEBenchFile:
    equation: str
    filename: str
    url: str
    path: str
    md5: str


class PDEBenchRegistry:
    """Validated view of PDEBench's authoritative download registry."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"PDE", "Filename", "URL", "Path", "MD5"}
        if not rows or set(rows[0]) != required:
            raise ValueError("unexpected PDEBench registry schema")
        self.files = tuple(
            PDEBenchFile(
                equation=row["PDE"],
                filename=row["Filename"],
                url=row["URL"],
                path=row["Path"],
                md5=row["MD5"].lower(),
            )
            for row in rows
        )
        for item in self.files:
            if len(item.md5) != 32 or any(
                character not in "0123456789abcdef" for character in item.md5
            ):
                raise ValueError(f"invalid MD5 for {item.filename}")
            if not item.url.startswith("https://"):
                raise ValueError(f"non-HTTPS URL for {item.filename}")

    def select(
        self, *, equation: str, filename: str | None = None
    ) -> list[PDEBenchFile]:
        selected = [item for item in self.files if item.equation == equation]
        if filename is not None:
            selected = [item for item in selected if item.filename == filename]
        if not selected:
            raise KeyError(
                f"no PDEBench files for equation={equation!r}, filename={filename!r}"
            )
        return selected


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_pdebench_file(
    item: PDEBenchFile,
    destination_root: Path,
    *,
    max_bytes: int,
    timeout_seconds: float = 30.0,
) -> Path:
    """Download one explicitly selected shard with a hard byte cap and checksum."""

    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    destination = destination_root / item.path / item.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if md5_file(destination) != item.md5:
            raise RuntimeError(f"existing file checksum mismatch: {destination}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        raise RuntimeError(f"retained partial download requires audit: {partial}")
    request = urllib.request.Request(
        item.url,
        headers={"User-Agent": "fim-itfm-dataset-loader/1.0"},
    )
    written = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > max_bytes:
                raise RuntimeError(
                    f"declared file size {declared} exceeds cap {max_bytes}"
                )
            with partial.open("xb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    written += len(block)
                    if written > max_bytes:
                        raise RuntimeError(
                            f"download exceeded hard byte cap {max_bytes}"
                        )
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
        observed = md5_file(partial)
        if observed != item.md5:
            raise RuntimeError(
                f"checksum mismatch for {item.filename}: {observed} != {item.md5}"
            )
        partial.replace(destination)
    except Exception:
        # Preserve a non-empty partial file as failure evidence.
        if partial.exists() and partial.stat().st_size == 0:
            partial.unlink()
        raise
    return destination


def materialize_aliased_advection(
    corpus: AliasedAdvectionCorpus, destination: Path
) -> dict[str, Any]:
    """Write the deterministic synthetic corpus and a hash-bound manifest."""

    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "trajectories.npz"
    manifest_path = destination / "manifest.json"
    if archive.exists() or manifest_path.exists():
        if not archive.is_file() or not manifest_path.is_file():
            raise RuntimeError("dataset materialization is incomplete")
        with manifest_path.open("r", encoding="utf-8") as handle:
            retained = json.load(handle)
        if sha256_file(archive) != retained.get("archive_sha256"):
            raise RuntimeError("retained synthetic archive hash mismatch")
        if retained.get("corpus_identity") != corpus.identity():
            raise RuntimeError("retained synthetic corpus identity mismatch")
        return retained
    temporary = archive.with_suffix(".npz.partial")
    trajectories = corpus.trajectories()
    with temporary.open("xb") as handle:
        np.savez_compressed(
            handle,
            **{
                f"owner_{owner_id}": tensor.detach().cpu().numpy()
                for owner_id, tensor in trajectories.items()
            },
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(archive)
    manifest = {
        "schema_version": 1,
        "dataset_id": "aliased-advection-v1",
        "corpus_identity": corpus.identity(),
        "validation": corpus.validate(),
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "owners": sorted(trajectories),
    }
    atomic_json(manifest_path, manifest)
    return manifest


class HDF5TrajectoryStore:
    """Lazy, schema-explicit adapter for public HDF5 trajectory arrays."""

    def __init__(
        self,
        path: Path,
        *,
        dataset_key: str,
        sample_axis: int,
        time_axis: int,
        channel_axis: int | None,
    ) -> None:
        self.path = path
        self.dataset_key = dataset_key
        self.sample_axis = sample_axis
        self.time_axis = time_axis
        self.channel_axis = channel_axis

    def _normalized_axes(self, ndim: int) -> tuple[int, int, int | None]:
        raw_axes = [self.sample_axis, self.time_axis]
        if self.channel_axis is not None:
            raw_axes.append(self.channel_axis)
        if any(axis < -ndim or axis >= ndim for axis in raw_axes):
            raise ValueError(f"axis is out of range for rank-{ndim} dataset")
        normalized = [axis % ndim for axis in raw_axes]
        if len(set(normalized)) != len(normalized):
            raise ValueError("sample, time, and channel axes must be distinct")
        return (
            normalized[0],
            normalized[1],
            normalized[2] if self.channel_axis is not None else None,
        )

    def _h5py(self):
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError(
                "HDF5 loading requires the optional 'datasets' dependency"
            ) from error
        return h5py

    def describe(self) -> dict[str, Any]:
        h5py = self._h5py()
        with h5py.File(self.path, "r") as handle:
            if self.dataset_key not in handle:
                raise KeyError(f"missing HDF5 dataset {self.dataset_key}")
            dataset = handle[self.dataset_key]
            sample_axis, time_axis, channel_axis = self._normalized_axes(dataset.ndim)
            return {
                "path": str(self.path),
                "sha256": sha256_file(self.path),
                "dataset_key": self.dataset_key,
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "sample_axis": sample_axis,
                "time_axis": time_axis,
                "channel_axis": channel_axis,
            }

    def trajectories(self, indices: list[int]) -> Iterator[tuple[int, torch.Tensor]]:
        h5py = self._h5py()
        with h5py.File(self.path, "r") as handle:
            dataset = handle[self.dataset_key]
            ndim = dataset.ndim
            sample_axis, original_time_axis, original_channel_axis = (
                self._normalized_axes(ndim)
            )
            sample_count = dataset.shape[sample_axis]
            for index in indices:
                if index < 0 or index >= sample_count:
                    raise IndexError(
                        f"sample index {index} is outside [0,{sample_count})"
                    )
                selection = [slice(None)] * ndim
                selection[sample_axis] = index
                array = np.asarray(dataset[tuple(selection)])
                remaining_axes = [axis for axis in range(ndim) if axis != sample_axis]
                time_axis = remaining_axes.index(original_time_axis)
                if original_channel_axis is None:
                    array = np.expand_dims(array, axis=-1)
                    channel_axis = array.ndim - 1
                else:
                    channel_axis = remaining_axes.index(original_channel_axis)
                spatial_axes = [
                    axis
                    for axis in range(array.ndim)
                    if axis not in {time_axis, channel_axis}
                ]
                ordered = np.transpose(
                    array, axes=[time_axis, channel_axis, *spatial_axes]
                )
                yield index, torch.from_numpy(np.ascontiguousarray(ordered)).float()


def trajectory_windows(
    owner_id: int,
    trajectory: torch.Tensor,
    *,
    context_length: int,
    family_id: int | None = None,
) -> list[WindowRecord]:
    if trajectory.ndim < 3:
        raise ValueError("trajectory must have shape [T,C,*spatial]")
    if not 1 <= context_length < trajectory.shape[0]:
        raise ValueError("invalid context_length")
    family = owner_id if family_id is None else family_id
    rows = []
    for event_time in range(context_length - 1, trajectory.shape[0] - 1):
        start = event_time - context_length + 1
        context = trajectory[start : event_time + 1].flatten(0, 1)
        rows.append(
            WindowRecord(
                context=context,
                target=trajectory[event_time + 1],
                owner_id=owner_id,
                family_id=family,
                event_time=event_time,
                regime=0,
                aliased=False,
            )
        )
    return rows
