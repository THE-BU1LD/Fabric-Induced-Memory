from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .provenance import atomic_write_json, sha256_file

INTERVENTIONS = (
    "none",
    "erase",
    "sign_flip",
    "spatial_shuffle",
    "channel_mean_broadcast",
    "recent_only",
    "last_residual",
    "wrong_trajectory",
    "time_permuted",
    "random_key",
    "random_value",
    "paired_swap",
    "oracle_past_residual",
)


@dataclasses.dataclass(frozen=True)
class OracleGapResult:
    dataset: str
    family: str
    minimum_age: int
    backbone_error: float
    oracle_error: float
    relative_recoverable_gain: float
    visible_window_only_error: float | None
    older_than_window_error: float | None
    observation_noise: float


def oracle_gap(
    *,
    dataset: str,
    family: str,
    minimum_age: int,
    backbone_prediction: torch.Tensor,
    oracle_prediction: torch.Tensor,
    target: torch.Tensor,
    error: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    observation_noise: float = 0.0,
    visible_window_prediction: torch.Tensor | None = None,
) -> OracleGapResult:
    base = float(error(backbone_prediction, target).mean().item())
    oracle = float(error(oracle_prediction, target).mean().item())
    visible = (
        None
        if visible_window_prediction is None
        else float(error(visible_window_prediction, target).mean().item())
    )
    return OracleGapResult(
        dataset=dataset,
        family=family,
        minimum_age=int(minimum_age),
        backbone_error=base,
        oracle_error=oracle,
        relative_recoverable_gain=(base - oracle) / max(abs(base), 1e-12),
        visible_window_only_error=visible,
        older_than_window_error=None if visible is None else visible - oracle,
        observation_noise=float(observation_noise),
    )


def triage_decision(
    rows: Sequence[OracleGapResult],
    *,
    track: str,
    threshold: float = 0.10,
) -> Mapping[str, Any]:
    if track not in {"pobench_pde", "hs_fno"}:
        raise ValueError("unknown triage track")
    required = 2 if track == "pobench_pde" else 3
    best_by_family: dict[str, float] = {}
    for row in rows:
        best_by_family[row.family] = max(
            best_by_family.get(row.family, float("-inf")), row.relative_recoverable_gain
        )
    passing = sorted(
        family for family, value in best_by_family.items() if value >= threshold
    )
    passed = len(passing) >= required
    return {
        "status": "PASS_TRIAGE" if passed else "STOP_TRACK_NEGATIVE_TRIAGE",
        "track": track,
        "threshold": threshold,
        "required_families": required,
        "passing_families": passing,
        "best_relative_gain_by_family": best_by_family,
        "negative_results_retained": True,
    }


def assert_future_leakage_sentinel_rejected(
    retrieve: Callable[[int], object], *, query_time: int, future_time: int
) -> None:
    if future_time <= query_time:
        raise ValueError("sentinel must be in the future")
    result = retrieve(future_time)
    valid = bool(result.valid)
    if valid:
        raise AssertionError("future-leakage sentinel was retrieved")


def persist_intervention_predictions(
    root: str | Path,
    *,
    run_id: str,
    intervention: str,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if intervention not in INTERVENTIONS:
        raise ValueError(f"unregistered intervention: {intervention}")
    destination = Path(root) / run_id / "interventions" / f"{intervention}.json"
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "intervention": intervention,
        "rows": list(rows),
    }
    atomic_write_json(destination, payload, overwrite=False)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "rows": len(rows),
    }


def intervention_summary(
    artifacts: Mapping[str, str | Path],
    *,
    metric: str,
) -> Mapping[str, Any]:
    values: dict[str, float] = {}
    hashes: dict[str, str] = {}
    for intervention, path_value in artifacts.items():
        path = Path(path_value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        measured = [float(row[metric]) for row in rows if metric in row]
        if not measured:
            raise ValueError(f"{intervention} has no metric {metric}")
        values[intervention] = sum(measured) / len(measured)
        hashes[intervention] = sha256_file(path)
    base = values.get("none")
    erase = values.get("erase")
    sign = values.get("sign_flip")
    if base is None or erase is None or sign is None:
        raise ValueError("none, erase, and sign_flip artifacts are required")
    return {
        "schema_version": 1,
        "metric": metric,
        "means": values,
        "artifact_sha256": hashes,
        "erase_delta": erase - base,
        "sign_flip_delta": sign - base,
        "causal_effect_claim_allowed": erase > base and sign > base,
    }


def robustness_grid() -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for pattern in ("point_wise", "patch_wise"):
        for missingness in (5, 25, 50, 75):
            rows.append({"mask_pattern": pattern, "missing_percent": missingness})
    rows.extend(
        {"stress": name, "values": values}
        for name, values in (
            ("observation_noise", (0.0, 0.01, 0.05, 0.10)),
            ("key_noise", (0.0, 0.01, 0.05, 0.10)),
            ("residual_noise", (0.0, 0.01, 0.05, 0.10)),
            ("rollout_horizon_multiplier", (1, 2, 4)),
            ("memory_capacity", (0, 32, 128, 512, 4096)),
            ("minimum_age", (1, 2, 4, 8, 16)),
            ("similarity_threshold", (-1.0, 0.0, 0.25, 0.5, 0.75)),
            ("topk", (1, 2, 4, 8, 16)),
            ("corrupted_memory_fraction", (0.0, 0.05, 0.25, 1.0)),
            (
                "distribution_shift",
                ("in_distribution", "ood_parameters", "ood_trajectory"),
            ),
        )
    )
    return tuple(rows)


def worst_group(
    rows: Sequence[Mapping[str, Any]], metric: str, group: str
) -> Mapping[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row[group]), []).append(float(row[metric]))
    means = {key: sum(values) / len(values) for key, values in grouped.items()}
    if not means:
        raise ValueError("no groups available")
    identity = max(means, key=means.get)
    return {"group": identity, "value": means[identity], "all_groups": means}
