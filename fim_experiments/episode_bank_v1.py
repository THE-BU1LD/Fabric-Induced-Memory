"""Opt-in, DEVELOPMENT-only delayed-recall episode artifacts and paired loaders.

This module does not change or execute a frozen runner. Model memory must still
be reset by the caller between episodes. See the implementation contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import re
import stat
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .runtime_benchmarks import DelayedRecallBenchmark, DelayedRecallConfig

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "fim_experiments/benchmark.py",
    "fim_experiments/runtime_benchmarks.py",
    "fim_experiments/episode_bank_v1.py",
)
SPLITS = ("train", "validation", "development_eval")
HELD_SEEDS = frozenset({101, 211, 307, 401, 503, 607, 701, 809, 907, 1009})
MAX_VALUES = 1_000_000
MANIFEST_KEYS = {
    "schema_version",
    "artifact_type",
    "phase",
    "seed",
    "benchmark_config",
    "horizon",
    "array",
    "episode_sha256",
    "splits",
    "source_sha256",
    "environment",
    "bank_sha256",
}


class EpisodeBankError(ValueError):
    """A development episode artifact does not satisfy its pinned contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EpisodeBankError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, name: str) -> None:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{name} must be a full lowercase SHA-256",
    )


def _integer(value: Any, name: str, minimum: int, maximum: int) -> None:
    _require(
        type(value) is int and minimum <= value <= maximum,
        f"{name} must be an integer in [{minimum}, {maximum}]",
    )


def _config(payload: dict[str, Any]) -> DelayedRecallConfig:
    _require(
        isinstance(payload, dict)
        and set(payload) == set(asdict(DelayedRecallConfig())),
        "benchmark config fields differ from the delayed-recall contract",
    )
    for key in ("memory_dim", "delay", "steps"):
        _integer(payload[key], key, 1, 64)
    _integer(payload["dimension"], "dimension", 2, 65)
    for key in ("cue_noise", "distractor_scale", "reveal_sharpness"):
        value = payload[key]
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        _require(valid, f"{key} must be finite and nonnegative")
    _require(payload["reveal_sharpness"] > 0, "reveal_sharpness must be positive")
    return DelayedRecallConfig(**payload)


def _sources() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_PATHS
    }


_IMPORTED_SOURCE_SHA256 = _sources()


def _verified_sources() -> dict[str, str]:
    current = _sources()
    _require(
        current == _IMPORTED_SOURCE_SHA256,
        "source changed since episode-bank import; use a fresh process with an immutable checkout",
    )
    return current


def _episode_ids(raw: bytes, shape: list[int]) -> list[str]:
    row_bytes = shape[1] * shape[2] * 4
    header = _canonical({"dtype": "<f4", "shape": shape[1:]}) + b"\n"
    return [
        hashlib.sha256(header + raw[i * row_bytes : (i + 1) * row_bytes]).hexdigest()
        for i in range(shape[0])
    ]


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_exclusive(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _read_regular(path: Path, max_bytes: int) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            _require(
                stat.S_ISREG(os.fstat(handle.fileno()).st_mode),
                f"{path.name} is not a regular file",
            )
            raw = handle.read(max_bytes + 1)
        _require(
            len(raw) <= max_bytes, f"{path.name} exceeds the bounded artifact size"
        )
        return raw
    except OSError as exc:
        raise EpisodeBankError(f"cannot read retained {path.name}: {exc}") from exc


def _strict_json(raw: bytes) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"duplicate manifest key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise EpisodeBankError(f"nonfinite manifest number: {value}")

    try:
        result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        _canonical(result)
        return result
    except (UnicodeError, ValueError, OverflowError) as exc:
        raise EpisodeBankError(f"invalid manifest JSON: {exc}") from exc


def create_development_bank(
    destination: str | Path,
    *,
    seed: int,
    config: DelayedRecallConfig | None = None,
    horizon: int = 20,
    train_count: int = 2,
    validation_count: int = 1,
    evaluation_count: int = 1,
) -> dict[str, Any]:
    """Materialize a fresh bounded bank without using or mutating global RNG.

    Validation precedes destination reservation. After reservation, every failed
    attempt leaves its directory intact; no existing bank is replaced.
    """
    source_snapshot = _verified_sources()
    config = _config(asdict(config if config is not None else DelayedRecallConfig()))
    _integer(seed, "seed", 0, 2**63 - 1)
    _require(
        seed not in HELD_SEEDS,
        "held scientific seeds cannot be used by the development producer",
    )
    _integer(horizon, "horizon", 1, 64)
    counts = [train_count, validation_count, evaluation_count]
    for split, count in zip(SPLITS, counts):
        _integer(count, split + " count", 1, 128)
    total = sum(counts)
    _require(total <= 128, "bank exceeds 128 episodes")
    shape = [total, horizon + 1, config.dimension]
    _require(math.prod(shape) <= MAX_VALUES, "bank exceeds the stored-value budget")
    destination = Path(destination).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)  # Atomic reservation, including racing writers.
    stage = "generation"
    try:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # Native sample_initial_state has no dtype parameter. Explicit allocation
        # preserves its N(0,I)/zero-clock law without changing the global dtype.
        memory = torch.randn(
            total,
            config.memory_dim,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        hidden = torch.cat(
            (memory, torch.zeros(total, 1, dtype=torch.float32, device="cpu")),
            dim=-1,
        )
        observations = DelayedRecallBenchmark(config).rollout(
            hidden, steps=horizon, generator=generator
        )
        stage = "source_freshness_after_generation"
        _verified_sources()
        _require(
            list(observations.shape) == shape, "generator returned a different shape"
        )
        _require(
            observations.dtype == torch.float32
            and bool(torch.isfinite(observations).all()),
            "generator returned nonfinite or non-float32 observations",
        )
        raw = (
            observations.detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype("<f4", copy=False)
            .tobytes(order="C")
        )
        ids = _episode_ids(raw, shape)
        _require(
            len(set(ids)) == total,
            "duplicate trajectory contents cannot define distinct bank episode identities",
        )
        splits, offset = {}, 0
        for split, count in zip(SPLITS, counts):
            splits[split] = list(range(offset, offset + count))
            offset += count
        manifest = {
            "schema_version": 1,
            "artifact_type": "FIM_DELAYED_RECALL_EPISODE_BANK_V1",
            "phase": "DEVELOPMENT",
            "seed": seed,
            "benchmark_config": asdict(config),
            "horizon": horizon,
            "array": {
                "file": "observations.f32",
                "dtype": "<f4",
                "order": "C",
                "shape": shape,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            "episode_sha256": ids,
            "splits": splits,
            "source_sha256": source_snapshot,
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "device": "cpu",
                "generation_dtype": "float32",
            },
        }
        manifest["bank_sha256"] = _digest(manifest)
        stage = "data_write"
        _write_exclusive(destination / "observations.f32", raw)
        stage = "manifest_write"
        _write_exclusive(destination / "manifest.pending", _canonical(manifest) + b"\n")
        os.rename(destination / "manifest.pending", destination / "manifest.json")
        stage = "directory_sync_after_installation"
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
        return {
            "phase": "DEVELOPMENT",
            "bank_sha256": manifest["bank_sha256"],
            "observations_sha256": manifest["array"]["sha256"],
            "shape": shape,
            "splits": {name: len(indices) for name, indices in splits.items()},
        }
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "phase": "DEVELOPMENT",
            "status": "FAILED",
            "stage": stage,
            "seed": seed,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        try:
            _write_exclusive(destination / "failure.json", _canonical(failure) + b"\n")
            _fsync_directory(destination)
        except OSError:
            pass  # The original error and partial directory remain; never claim a saved receipt.
        raise


class EpisodeDataset(Dataset):
    """Independent input/target snapshots compatible with the maintained trainer."""

    def __init__(self, observations: torch.Tensor, indices: tuple[int, ...]):
        self._observations = observations
        self.episode_indices = indices

    def __len__(self) -> int:
        return len(self.episode_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        _require(type(index) is int, "dataset index must be an integer")
        row = self._observations[self.episode_indices[index]]
        return row[:-1].clone(), row[1:].clone()


class EpisodeBank:
    def __init__(self, manifest: dict[str, Any], observations: torch.Tensor):
        self._manifest = copy.deepcopy(manifest)
        self._observations = observations

    @property
    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._manifest)

    def dataset(self, split: str) -> EpisodeDataset:
        _require(split in SPLITS, f"unknown development split: {split}")
        return EpisodeDataset(
            self._observations, tuple(self._manifest["splits"][split])
        )

    def loader(
        self, split: str, *, shuffle: bool = False, order_seed: int = 0
    ) -> DataLoader:
        """Always batch_size=1/num_workers=0; model state reset remains the caller's job."""
        _require(type(shuffle) is bool, "shuffle must be boolean")
        _integer(order_seed, "order_seed", 0, 2**63 - 1)
        _require(
            torch.get_default_device().type == "cpu",
            "loader requires the process default device to remain cpu during construction and iteration",
        )
        generator = torch.Generator(device="cpu").manual_seed(order_seed)
        return DataLoader(
            self.dataset(split),
            batch_size=1,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
            generator=generator,
        )


def load_development_bank(
    directory: str | Path, *, expected_bank_sha256: str
) -> EpisodeBank:
    """Verify the pinned artifact, current generator sources, and complete split contract."""
    source_snapshot = _verified_sources()
    _sha(expected_bank_sha256, "expected_bank_sha256")
    directory = Path(directory).absolute()
    _require(
        not directory.is_symlink() and directory.is_dir(),
        "bank must be a regular directory",
    )
    _require(
        {path.name for path in directory.iterdir()}
        == {"manifest.json", "observations.f32"},
        "bank is incomplete, failed, or contains unexpected files",
    )
    manifest = _strict_json(_read_regular(directory / "manifest.json", 128_000))
    _require(
        isinstance(manifest, dict) and set(manifest) == MANIFEST_KEYS,
        "manifest schema fields differ",
    )
    _require(
        type(manifest["schema_version"]) is int and manifest["schema_version"] == 1,
        "unsupported schema version",
    )
    _require(
        manifest["artifact_type"] == "FIM_DELAYED_RECALL_EPISODE_BANK_V1"
        and manifest["phase"] == "DEVELOPMENT",
        "only the development episode-bank contract is admitted",
    )
    _sha(manifest["bank_sha256"], "bank_sha256")
    body = {key: value for key, value in manifest.items() if key != "bank_sha256"}
    _require(
        _digest(body) == manifest["bank_sha256"] == expected_bank_sha256,
        "pinned bank identity differs",
    )
    _integer(manifest["seed"], "seed", 0, 2**63 - 1)
    _require(
        manifest["seed"] not in HELD_SEEDS,
        "development bank cannot use held scientific seeds",
    )
    config = _config(manifest["benchmark_config"])
    _integer(manifest["horizon"], "horizon", 1, 64)
    array = manifest["array"]
    _require(
        isinstance(array, dict)
        and set(array) == {"file", "dtype", "order", "shape", "bytes", "sha256"},
        "array fields differ",
    )
    _require(
        array["file"] == "observations.f32"
        and array["dtype"] == "<f4"
        and array["order"] == "C",
        "array filename, dtype, or order differs",
    )
    shape = array["shape"]
    _require(
        isinstance(shape, list) and len(shape) == 3, "array shape must have three axes"
    )
    _integer(shape[0], "episode count", 3, 128)
    _require(
        type(shape[1]) is int
        and shape[1] == manifest["horizon"] + 1
        and type(shape[2]) is int
        and shape[2] == config.dimension,
        "array shape differs from benchmark horizon/dimension",
    )
    _require(math.prod(shape) <= MAX_VALUES, "stored-value budget exceeded")
    _require(
        type(array["bytes"]) is int and array["bytes"] == math.prod(shape) * 4,
        "array byte length differs",
    )
    _sha(array["sha256"], "observations SHA-256")
    splits = manifest["splits"]
    _require(
        isinstance(splits, dict) and set(splits) == set(SPLITS), "split names differ"
    )
    all_indices = []
    for split in SPLITS:
        indices = splits[split]
        _require(
            isinstance(indices, list) and indices,
            f"{split} must contain episode indices",
        )
        for index in indices:
            _integer(index, "episode index", 0, shape[0] - 1)
        all_indices.extend(indices)
    _require(
        sorted(all_indices) == list(range(shape[0])),
        "splits must partition every episode exactly once",
    )
    _require(
        manifest["source_sha256"] == source_snapshot,
        "current generator/consumer source hashes differ from the bank",
    )
    environment = manifest["environment"]
    _require(
        isinstance(environment, dict)
        and set(environment)
        == {"python", "torch", "numpy", "device", "generation_dtype"},
        "generation environment fields differ",
    )
    _require(
        environment["device"] == "cpu" and environment["generation_dtype"] == "float32",
        "generation device/dtype differ",
    )
    _require(
        all(isinstance(value, str) and value.strip() for value in environment.values()),
        "environment values must be nonempty text",
    )
    raw = _read_regular(directory / "observations.f32", MAX_VALUES * 4)
    _require(
        len(raw) == array["bytes"]
        and hashlib.sha256(raw).hexdigest() == array["sha256"],
        "observation byte identity differs",
    )
    ids = _episode_ids(raw, shape)
    _require(
        manifest["episode_sha256"] == ids and len(set(ids)) == shape[0],
        "episode identities differ or duplicate contents exist",
    )
    values = (
        np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32, copy=True)
    )
    _require(bool(np.isfinite(values).all()), "observations must be finite")
    clock = np.zeros(shape[1], dtype=np.float32)
    increment = np.float32(1.0 / config.steps)
    for index in range(1, shape[1]):
        clock[index] = clock[index - 1] + increment
    _require(
        np.array_equal(values[:, :, -1], np.broadcast_to(clock, (shape[0], shape[1]))),
        "benchmark clock chronology differs",
    )
    _verified_sources()
    return EpisodeBank(manifest, torch.from_numpy(values))


def development_demo(destination: str | Path) -> dict[str, Any]:
    """Real generator -> retained bytes -> trainer-compatible paired consumers; no model run."""
    destination = Path(destination).absolute()
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        receipt = create_development_bank(
            destination / "bank",
            seed=20261010,
            config=DelayedRecallConfig(dimension=5, memory_dim=4, delay=3, steps=8),
            horizon=12,
            train_count=3,
            validation_count=1,
            evaluation_count=1,
        )
        first = load_development_bank(
            destination / "bank", expected_bank_sha256=receipt["bank_sha256"]
        )
        rows_a = list(first.loader("train", shuffle=True, order_seed=29))
        with torch.random.fork_rng(devices=[]):
            torch.rand(
                1000
            )  # Unrelated initialization/activity must not determine paired data.
            second = load_development_bank(
                destination / "bank", expected_bank_sha256=receipt["bank_sha256"]
            )
            rows_b = list(second.loader("train", shuffle=True, order_seed=29))
        _require(len(rows_a) == len(rows_b) == 3, "paired loader coverage differs")
        for (xa, ya), (xb, yb) in zip(rows_a, rows_b):
            _require(
                xa.shape == ya.shape == (1, 12, 5),
                "maintained trainer batch contract differs",
            )
            _require(
                torch.equal(xa, xb) and torch.equal(ya, yb),
                "paired consumer bytes differ",
            )
            _require(torch.equal(xa[:, 1:], ya[:, :-1]), "temporal overlap differs")
        saved = rows_b[0][0].clone()
        rows_a[0][0].fill_(float("nan"))
        _require(
            torch.equal(rows_b[0][0], saved), "consumers share returned tensor storage"
        )
        fresh = next(iter(first.loader("train", shuffle=True, order_seed=29)))[0]
        _require(torch.equal(fresh, saved), "a consumer mutated retained bank data")
        result = {
            **receipt,
            "status": "VERIFIED_DEVELOPMENT_DEMO",
            "paired_rows": 3,
            "batch_size": 1,
            "global_rng_independent": True,
            "clone_isolation": True,
            "model_training_runs": 0,
            "scientific_outcome_runs": 0,
            "seconds": time.perf_counter() - started,
        }
        _write_exclusive(destination / "demo_receipt.json", _canonical(result) + b"\n")
        _fsync_directory(destination)
        return result
    except Exception as exc:
        failure = {
            "phase": "DEVELOPMENT",
            "status": "FAILED",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        try:
            _write_exclusive(
                destination / "demo_failure.json", _canonical(failure) + b"\n"
            )
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser(
        "create", help="Create a new DEVELOPMENT-only episode bank"
    )
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--seed", required=True, type=int)
    create.add_argument("--train-count", default=2, type=int)
    create.add_argument("--validation-count", default=1, type=int)
    create.add_argument("--evaluation-count", default=1, type=int)
    create.add_argument("--horizon", default=20, type=int)
    verify = commands.add_parser(
        "verify", help="Verify a caller-pinned existing development bank"
    )
    verify.add_argument("--bank", required=True, type=Path)
    verify.add_argument("--expected-sha256", required=True)
    demo = commands.add_parser(
        "demo", help="Run the bounded real-generator development demonstration"
    )
    demo.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            result = create_development_bank(
                args.output,
                seed=args.seed,
                horizon=args.horizon,
                train_count=args.train_count,
                validation_count=args.validation_count,
                evaluation_count=args.evaluation_count,
            )
        elif args.command == "verify":
            bank = load_development_bank(
                args.bank, expected_bank_sha256=args.expected_sha256
            )
            result = {
                "status": "VERIFIED_DEVELOPMENT_EPISODE_BANK",
                "bank_sha256": bank.manifest["bank_sha256"],
                "phase": "DEVELOPMENT",
                "shape": bank.manifest["array"]["shape"],
            }
        else:
            result = development_demo(args.output)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
