from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypeVar

CURRENT_CONFIG_VERSION = 1
T = TypeVar("T")


class ConfigurationError(ValueError):
    """Raised when a run configuration is ambiguous or unsafe."""


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    benchmark: Literal["pobench_pde", "hs_fno"]
    name: str
    root: str
    manifest: str
    split: Literal["train", "validation", "test"] = "train"
    mask_pattern: Literal["point_wise", "patch_wise"] | None = None
    missing_percent: int | None = None
    observed_only_loss: bool = False

    def validate(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("dataset.name must be non-empty")
        if self.missing_percent is not None and not 0 <= self.missing_percent <= 100:
            raise ConfigurationError("dataset.missing_percent must lie in [0, 100]")
        if (self.mask_pattern is None) != (self.missing_percent is None):
            raise ConfigurationError(
                "mask_pattern and missing_percent must be supplied together"
            )


@dataclasses.dataclass(frozen=True)
class MemoryConfig:
    key_dimension: int
    residual_shape: tuple[int, ...]
    capacity_per_trajectory: int = 4096
    minimum_age: int = 1
    topk: int = 4
    temperature: float = 0.05
    minimum_similarity: float = 0.0
    chunk_size: int = 16384
    eviction: Literal["fifo", "reservoir", "diversity", "utility"] = "fifo"
    residual_normalization: Literal["none", "channel_std", "global_rms"] = "none"
    approximate_search: bool = False
    ann_bits: int = 32

    def validate(self) -> None:
        if (
            min(
                self.key_dimension,
                self.capacity_per_trajectory,
                self.minimum_age,
                self.topk,
                self.chunk_size,
            )
            < 1
        ):
            raise ConfigurationError("memory dimensions and counts must be positive")
        if not self.residual_shape or min(self.residual_shape) < 1:
            raise ConfigurationError("memory.residual_shape must be positive")
        if self.temperature <= 0:
            raise ConfigurationError("memory.temperature must be positive")
        if not -1.0 <= self.minimum_similarity <= 1.0:
            raise ConfigurationError("minimum_similarity must lie in [-1, 1]")
        if self.approximate_search and not 4 <= self.ann_bits <= 64:
            raise ConfigurationError("ann_bits must lie in [4, 64]")


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    mode: Literal["frozen_backbone", "joint"] = "frozen_backbone"
    gate: Literal[
        "one_sequence", "eight_trajectory", "sixty_four_trajectory", "full"
    ] = "one_sequence"
    seed: int = 101
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_accumulation: int = 1
    gradient_clip_norm: float = 1.0
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    early_stopping_patience: int = 10
    checkpoint_every_steps: int = 100
    distributed: bool = False
    wall_clock_budget_seconds: int = 3600
    example_budget: int = 100_000
    tuning_trial_budget: int = 1

    def validate(self) -> None:
        if (
            min(
                self.epochs,
                self.gradient_accumulation,
                self.early_stopping_patience,
                self.checkpoint_every_steps,
                self.wall_clock_budget_seconds,
                self.example_budget,
                self.tuning_trial_budget,
            )
            < 1
        ):
            raise ConfigurationError("training counts and budgets must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ConfigurationError("invalid optimizer configuration")
        if self.gradient_clip_norm <= 0:
            raise ConfigurationError("gradient_clip_norm must be positive")


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    seeds: tuple[int, ...]
    protected_test: bool = False
    frozen_development_receipt: str | None = None
    bootstrap_repetitions: int = 10_000
    confidence: float = 0.95
    fail_on_missing_run: bool = True

    def validate(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ConfigurationError("evaluation seeds must be non-empty and unique")
        if self.bootstrap_repetitions < 100:
            raise ConfigurationError("at least 100 bootstrap repetitions are required")
        if not 0.5 < self.confidence < 1.0:
            raise ConfigurationError("confidence must lie in (0.5, 1)")
        if self.protected_test and not self.frozen_development_receipt:
            raise ConfigurationError(
                "protected evaluation requires a frozen development receipt"
            )


@dataclasses.dataclass(frozen=True)
class RunConfig:
    schema_version: int
    run_name: str
    output_root: str
    backbone: str
    backbone_checkpoint: str | None
    backbone_identity_sha256: str | None
    dataset: DatasetConfig
    memory: MemoryConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    metadata: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != CURRENT_CONFIG_VERSION:
            raise ConfigurationError(
                f"resolved schema must be {CURRENT_CONFIG_VERSION}, got "
                f"{self.schema_version}"
            )
        if not self.run_name.strip() or not self.backbone.strip():
            raise ConfigurationError("run_name and backbone must be non-empty")
        if (self.backbone_checkpoint is None) != (
            self.backbone_identity_sha256 is None
        ):
            raise ConfigurationError(
                "checkpoint path and identity sha256 must be supplied together"
            )
        self.dataset.validate()
        self.memory.validate()
        self.training.validate()
        self.evaluation.validate()

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def resolved_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"


def migrate_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate only explicitly supported historical schemas.

    Version zero was used in early local scaffolding and named the output key
    ``results_root``.  No other semantic migration is inferred.
    """

    migrated = dict(payload)
    version = migrated.get("schema_version")
    if version == 0:
        if "results_root" in migrated and "output_root" not in migrated:
            migrated["output_root"] = migrated.pop("results_root")
        migrated["schema_version"] = 1
        version = 1
    if version != CURRENT_CONFIG_VERSION:
        raise ConfigurationError(
            f"unsupported configuration schema {version!r}; expected "
            f"{CURRENT_CONFIG_VERSION}"
        )
    return migrated


def _strict_dataclass(cls: type[T], payload: Mapping[str, Any]) -> T:
    allowed = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(payload) - allowed)
    missing = sorted(
        field.name
        for field in dataclasses.fields(cls)
        if field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING  # type: ignore[comparison-overlap]
        and field.name not in payload
    )
    if unknown or missing:
        raise ConfigurationError(
            f"{cls.__name__}: unknown={unknown or 'none'} missing={missing or 'none'}"
        )
    return cls(**payload)  # type: ignore[arg-type]


def resolve_config(payload: Mapping[str, Any]) -> RunConfig:
    raw = migrate_config(payload)
    nested = dict(raw)
    for name, cls in (
        ("dataset", DatasetConfig),
        ("memory", MemoryConfig),
        ("training", TrainingConfig),
        ("evaluation", EvaluationConfig),
    ):
        value = nested.get(name)
        if not isinstance(value, Mapping):
            raise ConfigurationError(f"{name} must be an object")
        converted = dict(value)
        if name == "memory" and "residual_shape" in converted:
            converted["residual_shape"] = tuple(converted["residual_shape"])
        if name == "evaluation" and "seeds" in converted:
            converted["seeds"] = tuple(converted["seeds"])
        nested[name] = _strict_dataclass(cls, converted)
    config = _strict_dataclass(RunConfig, nested)
    config.validate()
    return config


def load_config(path: str | Path) -> RunConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ConfigurationError("top-level configuration must be an object")
    return resolve_config(payload)
