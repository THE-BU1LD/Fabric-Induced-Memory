from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PredictionRow:
    owner_id: int
    family_id: int
    event_time: int
    regime: int
    aliased: bool
    squared_error: float
    element_count: int
    retrieval_valid: bool = False
    retrieval_eligible_count: int = 0
    retrieval_regime_precision: float | None = None
    transport_shift: tuple[int, ...] | None = None


def training_target_variance(targets: list[torch.Tensor]) -> float:
    if not targets:
        raise ValueError("at least one target is required")
    flattened = torch.cat([target.detach().reshape(-1).float() for target in targets])
    variance = float(flattened.var(unbiased=False).item())
    if not variance > 0:
        raise ValueError("training target variance must be positive")
    return variance


def summarize_rows(
    rows: list[PredictionRow], normalization: float
) -> dict[str, float | int | None]:
    if not rows:
        raise ValueError("cannot summarize an empty prediction set")
    if normalization <= 0:
        raise ValueError("normalization must be positive")

    by_owner: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"all": [], "alias": [], "non_alias": []}
    )
    valid_retrieval = 0
    eligible_total = 0
    regime_precision = []
    transport_magnitudes = []
    for row in rows:
        mse = row.squared_error / row.element_count
        by_owner[row.owner_id]["all"].append(mse)
        by_owner[row.owner_id]["alias" if row.aliased else "non_alias"].append(mse)
        valid_retrieval += int(row.retrieval_valid)
        eligible_total += row.retrieval_eligible_count
        if row.retrieval_regime_precision is not None:
            regime_precision.append(row.retrieval_regime_precision)
        if row.transport_shift is not None:
            transport_magnitudes.append(
                sum(abs(value) for value in row.transport_shift)
            )

    def clustered(bucket: str) -> float:
        owner_means = [
            sum(parts[bucket]) / len(parts[bucket])
            for parts in by_owner.values()
            if parts[bucket]
        ]
        if not owner_means:
            return float("nan")
        return (sum(owner_means) / len(owner_means)) / normalization

    return {
        "trajectory_count": len(by_owner),
        "row_count": len(rows),
        "alias_row_count": sum(row.aliased for row in rows),
        "overall_nmse": clustered("all"),
        "alias_conditioned_nmse": clustered("alias"),
        "non_alias_nmse": clustered("non_alias"),
        "retrieval_valid_fraction": valid_retrieval / len(rows),
        "mean_eligible_entries": eligible_total / len(rows),
        "retrieval_regime_precision": (
            sum(regime_precision) / len(regime_precision) if regime_precision else None
        ),
        "mean_transport_l1": (
            sum(transport_magnitudes) / len(transport_magnitudes)
            if transport_magnitudes
            else None
        ),
    }


def donor_direction_score(
    ordinary: torch.Tensor,
    swapped: torch.Tensor,
    recipient_target: torch.Tensor,
    donor_target: torch.Tensor,
) -> torch.Tensor:
    """Positive values mean swapping moved predictions toward the donor future."""

    before = (ordinary - donor_target).square().flatten(1).mean(dim=1)
    after = (swapped - donor_target).square().flatten(1).mean(dim=1)
    recipient_distance = (
        (ordinary - recipient_target).square().flatten(1).mean(dim=1).clamp_min(1e-12)
    )
    return (before - after) / recipient_distance
