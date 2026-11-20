from __future__ import annotations

import dataclasses
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


@dataclasses.dataclass(frozen=True)
class EvaluationCell:
    method: str
    seed: int
    family: str
    trajectory: str
    cell: str
    value: float | None
    status: str = "PASS"


def aggregate_within_trajectory(
    rows: Sequence[EvaluationCell],
) -> dict[tuple[str, int, str, str, str], float]:
    grouped: dict[tuple[str, int, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.status == "PASS" and row.value is not None and math.isfinite(row.value):
            grouped[
                (row.method, row.seed, row.family, row.trajectory, row.cell)
            ].append(float(row.value))
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def paired_values(
    rows: Sequence[EvaluationCell], candidate: str, baseline: str
) -> tuple[list[float], list[float], list[tuple[int, str, str, str]]]:
    aggregated = aggregate_within_trajectory(rows)
    candidate_map = {
        key[1:]: value for key, value in aggregated.items() if key[0] == candidate
    }
    baseline_map = {
        key[1:]: value for key, value in aggregated.items() if key[0] == baseline
    }
    shared = sorted(set(candidate_map) & set(baseline_map))
    return (
        [candidate_map[key] for key in shared],
        [baseline_map[key] for key in shared],
        shared,
    )


def paired_effect(
    candidate: Sequence[float], baseline: Sequence[float]
) -> Mapping[str, float]:
    if not candidate or len(candidate) != len(baseline):
        raise ValueError("paired non-empty samples are required")
    differences = [
        base - model for model, base in zip(candidate, baseline, strict=True)
    ]
    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / max(
        1, len(differences) - 1
    )
    standard_deviation = math.sqrt(variance)
    return {
        "mean_absolute_improvement": mean,
        "mean_relative_improvement": sum(
            (base - model) / max(abs(base), 1e-12)
            for model, base in zip(candidate, baseline, strict=True)
        )
        / len(candidate),
        "paired_standardized_effect": mean / max(standard_deviation, 1e-12),
        "win_fraction": sum(
            model < base for model, base in zip(candidate, baseline, strict=True)
        )
        / len(candidate),
        "tie_fraction": sum(
            model == base for model, base in zip(candidate, baseline, strict=True)
        )
        / len(candidate),
    }


def family_cluster_bootstrap(
    rows: Sequence[EvaluationCell],
    *,
    candidate: str,
    baseline: str,
    repetitions: int,
    confidence: float,
    seed: int,
) -> Mapping[str, Any]:
    if repetitions < 100 or not 0.5 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")
    candidate_values, baseline_values, keys = paired_values(rows, candidate, baseline)
    if not keys:
        raise ValueError("no paired units available")
    by_family: dict[str, list[float]] = defaultdict(list)
    for model, base, key in zip(candidate_values, baseline_values, keys, strict=True):
        by_family[key[1]].append((base - model) / max(abs(base), 1e-12))
    families = sorted(by_family)
    generator = random.Random(seed)
    distribution: list[float] = []
    for _ in range(repetitions):
        sampled = [generator.choice(families) for _ in families]
        values = [value for family in sampled for value in by_family[family]]
        distribution.append(sum(values) / len(values))
    distribution.sort()
    tail = (1.0 - confidence) / 2.0
    lower_index = min(repetitions - 1, max(0, int(tail * repetitions)))
    upper_index = min(repetitions - 1, max(0, int((1.0 - tail) * repetitions) - 1))
    point = sum(value for values in by_family.values() for value in values) / sum(
        len(values) for values in by_family.values()
    )
    return {
        "estimate": point,
        "confidence": confidence,
        "lower": distribution[lower_index],
        "upper": distribution[upper_index],
        "clusters": len(families),
        "independent_unit": "family",
        "repetitions": repetitions,
    }


def holm_adjust(
    p_values: Mapping[str, float], alpha: float = 0.05
) -> Mapping[str, Any]:
    if not 0 < alpha < 1 or any(not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("invalid alpha or p-value")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, (count - rank) * value)
        adjusted[name] = min(1.0, running)
    rejected: dict[str, bool] = {}
    continue_rejecting = True
    for rank, (name, value) in enumerate(ordered):
        threshold = alpha / (count - rank)
        continue_rejecting = continue_rejecting and value <= threshold
        rejected[name] = continue_rejecting
    return {"adjusted_p": adjusted, "rejected": rejected, "alpha": alpha}


def cell_win_loss_tie(
    rows: Sequence[EvaluationCell], candidate: str, baseline: str
) -> Mapping[str, Mapping[str, int]]:
    model, base, keys = paired_values(rows, candidate, baseline)
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"win": 0, "loss": 0, "tie": 0}
    )
    for model_value, base_value, key in zip(model, base, keys, strict=True):
        outcome = (
            "win"
            if model_value < base_value
            else "loss"
            if model_value > base_value
            else "tie"
        )
        counts[key[3]][outcome] += 1
    return dict(counts)


def missing_run_accounting(
    rows: Sequence[EvaluationCell],
    *,
    expected_methods: Iterable[str],
    expected_seeds: Iterable[int],
    expected_cells: Iterable[str],
) -> Mapping[str, Any]:
    expected = {
        (method, int(seed), cell)
        for method in expected_methods
        for seed in expected_seeds
        for cell in expected_cells
    }
    observed = {(row.method, row.seed, row.cell) for row in rows}
    failures = [dataclasses.asdict(row) for row in rows if row.status != "PASS"]
    return {
        "expected": len(expected),
        "observed": len(observed & expected),
        "missing": [list(key) for key in sorted(expected - observed)],
        "failures": failures,
        "post_hoc_exclusions_allowed": False,
    }
