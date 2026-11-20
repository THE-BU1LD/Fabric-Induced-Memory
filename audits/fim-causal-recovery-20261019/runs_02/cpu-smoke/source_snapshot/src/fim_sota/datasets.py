from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import ForecastBatch
from .provenance import atomic_write_json, sha256_file

POBENCH_DATASETS = {"navier_stokes", "diffusion_reaction", "era5"}
HSFNO_FAMILIES = {
    "delayed_reaction_diffusion",
    "epidemic_delay",
    "nonlocal_neural_field",
    "delayed_wave",
    "distributed_memory",
}


class DatasetAdmissionError(RuntimeError):
    """Raised when source, license, hash, or split admission is incomplete."""


@dataclasses.dataclass(frozen=True)
class SourceFile:
    path: str
    sha256: str
    bytes: int


@dataclasses.dataclass(frozen=True)
class DatasetManifest:
    schema_version: int
    benchmark: Literal["pobench_pde", "hs_fno"]
    dataset: str
    upstream_commit: str
    license_status: Literal["ADMITTED", "BLOCKED", "UNKNOWN"]
    preprocessing_id: str
    preprocessing_sha256: str
    normalization: Mapping[str, Any]
    splits: Mapping[str, tuple[str, ...]]
    source_files: tuple[SourceFile, ...]
    generated_by: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    schema_only_fixture: bool = False

    def validate(self) -> None:
        if self.schema_version != 1:
            raise DatasetAdmissionError("unsupported dataset manifest schema")
        allowed = (
            POBENCH_DATASETS if self.benchmark == "pobench_pde" else HSFNO_FAMILIES
        )
        if self.dataset not in allowed:
            raise DatasetAdmissionError(
                f"dataset {self.dataset!r} is not registered for {self.benchmark}"
            )
        if len(self.upstream_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.upstream_commit
        ):
            raise DatasetAdmissionError("upstream commit must be a full SHA-1")
        if self.license_status != "ADMITTED" and not self.schema_only_fixture:
            raise DatasetAdmissionError("dataset license is not admitted")
        if len(self.preprocessing_sha256) != 64:
            raise DatasetAdmissionError("preprocessing identity must be SHA-256")
        split_names = set(self.splits)
        if split_names != {"train", "validation", "test"}:
            raise DatasetAdmissionError("manifest must define train/validation/test")
        detect_split_overlap(self.splits)
        if not self.source_files:
            raise DatasetAdmissionError("manifest has no source files")


@dataclasses.dataclass(frozen=True)
class SampleProvenance:
    sample_id: str
    dataset: str
    source_path: str
    source_sha256: str
    split: str
    trajectory_id: str
    event_time: int
    preprocessing_sha256: str
    schema_only_fixture: bool


def detect_split_overlap(splits: Mapping[str, Sequence[str]]) -> None:
    owner: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []
    for split, identities in splits.items():
        for identity in identities:
            if identity in owner and owner[identity] != split:
                collisions.append((identity, owner[identity], split))
            owner[str(identity)] = str(split)
    if collisions:
        raise DatasetAdmissionError(f"trajectory split overlap: {collisions[:10]}")


def load_manifest(path: str | Path) -> DatasetManifest:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DatasetAdmissionError("manifest must contain an object")
    try:
        files = tuple(SourceFile(**item) for item in payload.pop("source_files"))
        splits = {
            str(name): tuple(str(identity) for identity in values)
            for name, values in payload.pop("splits").items()
        }
        manifest = DatasetManifest(source_files=files, splits=splits, **payload)
    except (KeyError, TypeError) as error:
        raise DatasetAdmissionError(f"invalid dataset manifest: {error}") from error
    manifest.validate()
    return manifest


def verify_source_files(manifest: DatasetManifest, root: str | Path) -> dict[str, str]:
    base = Path(root).resolve()
    verified: dict[str, str] = {}
    for record in manifest.source_files:
        candidate = (base / record.path).resolve()
        if base not in candidate.parents and candidate != base:
            raise DatasetAdmissionError(f"source escapes dataset root: {record.path}")
        if not candidate.is_file():
            raise DatasetAdmissionError(f"missing dataset source: {record.path}")
        actual_size = candidate.stat().st_size
        actual_sha = sha256_file(candidate)
        if actual_size != record.bytes or actual_sha != record.sha256:
            raise DatasetAdmissionError(
                f"dataset source identity mismatch: {record.path}"
            )
        verified[record.path] = actual_sha
    return verified


def deterministic_mask(
    shape: tuple[int, ...],
    *,
    missing_percent: int,
    pattern: Literal["point_wise", "patch_wise"],
    seed: int,
    patch_size: tuple[int, int] = (4, 4),
) -> torch.Tensor:
    """Return an explicit observed=True mask with deterministic missingness."""

    if len(shape) < 2 or min(shape) < 1:
        raise ValueError("mask shape must include positive spatial dimensions")
    if not 0 <= missing_percent <= 100:
        raise ValueError("missing_percent must lie in [0,100]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    observed = torch.ones(shape, dtype=torch.bool)
    spatial_shape = shape[-2:]
    if pattern == "point_wise":
        spatial = torch.ones(spatial_shape, dtype=torch.bool)
        count = round(spatial.numel() * missing_percent / 100.0)
        if count:
            order = torch.randperm(spatial.numel(), generator=generator)
            spatial.flatten()[order[:count]] = False
        return spatial.expand(shape).clone()
    if pattern != "patch_wise":
        raise ValueError(f"unsupported mask pattern: {pattern}")
    height, width = spatial_shape
    patch_height, patch_width = patch_size
    patches = [
        (row, column)
        for row in range(0, height, patch_height)
        for column in range(0, width, patch_width)
    ]
    target = round(observed.numel() * missing_percent / 100.0)
    removed = 0
    order = torch.randperm(len(patches), generator=generator).tolist()
    prefix_shape = shape[:-2]
    prefix_count = math_product(prefix_shape)
    for index in order:
        if removed >= target:
            break
        row, column = patches[index]
        row_stop = min(height, row + patch_height)
        column_stop = min(width, column + patch_width)
        observed[..., row:row_stop, column:column_stop] = False
        removed += prefix_count * (row_stop - row) * (column_stop - column)
    return observed


def math_product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def trajectory_mask_seed(global_seed: int, trajectory_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{trajectory_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def temporally_consistent_masks(
    trajectory_ids: Sequence[str],
    sample_shape: tuple[int, ...],
    *,
    missing_percent: int,
    pattern: Literal["point_wise", "patch_wise"],
    seed: int,
) -> torch.Tensor:
    cache: dict[str, torch.Tensor] = {}
    rows: list[torch.Tensor] = []
    for identity in trajectory_ids:
        if identity not in cache:
            cache[identity] = deterministic_mask(
                sample_shape,
                missing_percent=missing_percent,
                pattern=pattern,
                seed=trajectory_mask_seed(seed, identity),
            )
        rows.append(cache[identity])
    return torch.stack(rows)


class POBenchPDEDataset(Dataset[dict[str, Any]]):
    """Manifest-bound loader for admitted POBench-compatible NPZ shards.

    This loader does not assert that arbitrary NPZ files are official.  The
    manifest must bind the locked upstream commit, exact preprocessing source
    hash, normalization statistics, trajectory split, and every shard hash.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        manifest_path: str | Path,
        split: Literal["train", "validation", "test"],
        mask_pattern: Literal["point_wise", "patch_wise"],
        missing_percent: int,
        mask_seed: int,
    ) -> None:
        self.root = Path(root).resolve()
        self.manifest = load_manifest(manifest_path)
        if self.manifest.benchmark != "pobench_pde":
            raise DatasetAdmissionError("manifest is not a POBench-PDE manifest")
        self.split = split
        self.mask_pattern = mask_pattern
        self.missing_percent = int(missing_percent)
        self.mask_seed = int(mask_seed)
        verify_source_files(self.manifest, self.root)
        self._samples = self._load_npz()

    def _load_npz(self) -> list[dict[str, Any]]:
        allowed = set(self.manifest.splits[self.split])
        samples: list[dict[str, Any]] = []
        required = {"context", "target", "trajectory_id", "event_time"}
        for record in self.manifest.source_files:
            if not record.path.endswith(".npz"):
                continue
            path = self.root / record.path
            with np.load(path, allow_pickle=False) as payload:
                missing = required - set(payload.files)
                if missing:
                    raise DatasetAdmissionError(
                        f"{record.path} missing arrays {missing}"
                    )
                contexts = payload["context"]
                targets = payload["target"]
                trajectories = payload["trajectory_id"].astype(str)
                times = payload["event_time"].astype(np.int64)
                if not (
                    len(contexts) == len(targets) == len(trajectories) == len(times)
                ):
                    raise DatasetAdmissionError(f"{record.path} array length mismatch")
                for index, identity in enumerate(trajectories):
                    if identity not in allowed:
                        continue
                    samples.append(
                        {
                            "context": torch.from_numpy(contexts[index]).float(),
                            "target": torch.from_numpy(targets[index]).float(),
                            "trajectory_id": str(identity),
                            "event_time": int(times[index]),
                            "source_path": record.path,
                            "source_sha256": record.sha256,
                            "source_index": int(index),
                        }
                    )
        if not samples:
            raise DatasetAdmissionError(f"no samples found for split {self.split}")
        return sorted(
            samples, key=lambda row: (row["trajectory_id"], row["event_time"])
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self._samples[index])
        identity = str(sample["trajectory_id"])
        context = normalize_tensor(sample["context"], self.manifest)
        target = normalize_tensor(sample["target"], self.manifest)
        mask = deterministic_mask(
            tuple(context.shape),
            missing_percent=self.missing_percent,
            pattern=self.mask_pattern,
            seed=trajectory_mask_seed(self.mask_seed, identity),
        )
        sample["observation_mask"] = mask
        sample["observed_context"] = torch.where(
            mask, context, torch.zeros_like(context)
        )
        sample["context"] = context
        sample["target"] = target
        sample["target_observation_mask"] = (
            mask[-1] if mask.ndim == target.ndim + 1 else mask
        )
        sample["provenance"] = SampleProvenance(
            sample_id=f"{sample['source_sha256']}:{sample['source_index']}",
            dataset=self.manifest.dataset,
            source_path=str(sample["source_path"]),
            source_sha256=str(sample["source_sha256"]),
            split=self.split,
            trajectory_id=identity,
            event_time=int(sample["event_time"]),
            preprocessing_sha256=self.manifest.preprocessing_sha256,
            schema_only_fixture=self.manifest.schema_only_fixture,
        )
        return sample


class NavierStokesLoader(POBenchPDEDataset):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.manifest.dataset != "navier_stokes":
            raise DatasetAdmissionError("expected navier_stokes manifest")


class DiffusionReactionLoader(POBenchPDEDataset):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.manifest.dataset != "diffusion_reaction":
            raise DatasetAdmissionError("expected diffusion_reaction manifest")


class ERA5Loader(POBenchPDEDataset):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.manifest.dataset != "era5":
            raise DatasetAdmissionError("expected era5 manifest")


def normalize_tensor(value: torch.Tensor, manifest: DatasetManifest) -> torch.Tensor:
    method = manifest.normalization.get("method")
    if method == "identity":
        return value
    if method == "channel_standardize":
        mean = torch.as_tensor(manifest.normalization["mean"], dtype=value.dtype)
        scale = torch.as_tensor(manifest.normalization["std"], dtype=value.dtype)
        channel_axis = int(manifest.normalization.get("channel_axis", -3))
        channel_axis %= value.ndim
        if mean.numel() != value.shape[channel_axis]:
            raise DatasetAdmissionError("normalization channel count mismatch")
        shape = [1] * value.ndim
        shape[channel_axis] = mean.numel()
        return (value - mean.reshape(shape)) / scale.reshape(shape).clamp_min(1e-12)
    raise DatasetAdmissionError(f"unsupported registered normalization: {method!r}")


def make_forecast_batch(
    samples: Sequence[Mapping[str, Any]], split: str
) -> ForecastBatch:
    contexts = torch.stack([sample["observed_context"] for sample in samples])
    targets = torch.stack([sample["target"] for sample in samples])
    masks = torch.stack([sample["observation_mask"] for sample in samples])
    target_masks = torch.stack(
        [sample["target_observation_mask"] for sample in samples]
    )
    event_times = torch.tensor([int(sample["event_time"]) for sample in samples])
    available_times = torch.tensor(
        [
            int(sample.get("target_available_time", int(sample["event_time"]) + 1))
            for sample in samples
        ]
    )
    provenance = tuple(sample["provenance"].sample_id for sample in samples)
    batch = ForecastBatch(
        context=contexts,
        target=targets,
        observation_mask=masks,
        trajectory_ids=tuple(str(sample["trajectory_id"]) for sample in samples),
        event_times=event_times,
        lead_times=torch.tensor(
            [int(sample.get("lead_time", 1)) for sample in samples]
        ),
        split=split,
        target_available_times=available_times,
        target_observation_mask=target_masks,
        provenance_ids=provenance,
    )
    batch.validate()
    return batch


def construct_history_windows(
    values: torch.Tensor,
    *,
    history: int,
    horizon: int,
    stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if values.ndim < 2 or min(history, horizon, stride) < 1:
        raise ValueError("invalid history window configuration")
    origins = torch.arange(history, values.shape[0] - horizon + 1, stride)
    if origins.numel() == 0:
        raise ValueError("trajectory is too short for requested windows")
    past = torch.stack([values[index - history : index] for index in origins])
    future = torch.stack([values[index : index + horizon] for index in origins])
    return past, future, origins


class LockedUpstreamHSFNOGenerator:
    """Execute an admitted upstream generator without copying its source.

    The caller supplies the exact command from the locked upstream checkout.
    Completion is accepted only when the checkout commit, license receipt, exit
    code, output file, and output checksum are all recorded.
    """

    def __init__(
        self,
        *,
        checkout: str | Path,
        expected_commit: str,
        license_receipt: str | Path,
    ) -> None:
        self.checkout = Path(checkout).resolve()
        self.expected_commit = expected_commit
        self.license_receipt = Path(license_receipt).resolve()

    def verify(self) -> None:
        if not self.license_receipt.is_file():
            raise DatasetAdmissionError("upstream license receipt is missing")
        receipt = json.loads(self.license_receipt.read_text(encoding="utf-8"))
        if receipt.get("status") != "ADMITTED":
            raise DatasetAdmissionError("upstream source license is not admitted")
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if commit != self.expected_commit:
            raise DatasetAdmissionError(
                "upstream checkout mismatch: "
                f"expected {self.expected_commit}, got {commit}"
            )

    def generate(
        self,
        *,
        family: str,
        command: Sequence[str],
        output: str | Path,
        configuration: Mapping[str, Any],
        receipt_path: str | Path,
    ) -> Mapping[str, Any]:
        if family not in HSFNO_FAMILIES:
            raise ValueError(f"unsupported HS-FNO family: {family}")
        self.verify()
        destination = Path(output).resolve()
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite generated data: {destination}"
            )
        completed = subprocess.run(
            list(command),
            cwd=self.checkout,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not destination.is_file():
            payload = {
                "schema_version": 1,
                "status": "FAILED",
                "family": family,
                "command": list(command),
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
                "configuration": dict(configuration),
            }
            atomic_write_json(receipt_path, payload, overwrite=False)
            raise RuntimeError(f"upstream generator failed; receipt: {receipt_path}")
        payload = {
            "schema_version": 1,
            "status": "GENERATED_UNVALIDATED",
            "family": family,
            "upstream_commit": self.expected_commit,
            "command": list(command),
            "configuration": dict(configuration),
            "output": str(destination),
            "output_sha256": sha256_file(destination),
            "output_bytes": destination.stat().st_size,
        }
        atomic_write_json(receipt_path, payload, overwrite=False)
        return payload


def cache_key(manifest: DatasetManifest, configuration: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(manifest.preprocessing_sha256.encode())
    for record in sorted(manifest.source_files, key=lambda item: item.path):
        digest.update(record.path.encode())
        digest.update(record.sha256.encode())
    digest.update(json.dumps(configuration, sort_keys=True).encode())
    return digest.hexdigest()


def atomic_cache_directory(
    cache_root: str | Path,
    identity: str,
    build: Callable[[Path], None],
) -> Path:
    root = Path(cache_root)
    destination = root / identity
    if destination.is_dir():
        return destination
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{identity}.", dir=root))
    try:
        build(temporary)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def schema_only_fixture_manifest(
    *,
    benchmark: Literal["pobench_pde", "hs_fno"],
    dataset: str,
    source_file: str | Path,
    splits: Mapping[str, Sequence[str]],
) -> DatasetManifest:
    """Create metadata for tests; the fixture is explicitly non-result evidence."""

    source = Path(source_file)
    manifest = DatasetManifest(
        schema_version=1,
        benchmark=benchmark,
        dataset=dataset,
        upstream_commit="0" * 40,
        license_status="BLOCKED",
        preprocessing_id="schema-only-fixture",
        preprocessing_sha256=hashlib.sha256(b"schema-only-fixture").hexdigest(),
        normalization={"method": "identity"},
        splits={name: tuple(values) for name, values in splits.items()},
        source_files=(
            SourceFile(
                path=source.name,
                sha256=sha256_file(source),
                bytes=source.stat().st_size,
            ),
        ),
        generated_by={"kind": "schema_only_fixture", "results_admissible": False},
        schema_only_fixture=True,
    )
    manifest.validate()
    return manifest
