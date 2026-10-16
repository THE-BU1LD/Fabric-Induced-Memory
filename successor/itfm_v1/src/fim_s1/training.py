from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn

from .data import WindowRecord
from .metrics import PredictionRow, donor_direction_score
from .model import ITFMOutput, InterventionalTensorFabricPredictor
from .model_nd import InterventionalNDTensorFabricPredictor

MemoryPredictor = (
    InterventionalTensorFabricPredictor | InterventionalNDTensorFabricPredictor
)
MEMORY_PREDICTOR_TYPES = (
    InterventionalTensorFabricPredictor,
    InterventionalNDTensorFabricPredictor,
)


@dataclass(frozen=True)
class WindowBatch:
    context: torch.Tensor
    target: torch.Tensor
    owners: torch.Tensor
    families: torch.Tensor
    event_times: torch.Tensor
    regimes: torch.Tensor
    aliases: torch.Tensor


def chronological_batches(
    records: list[WindowRecord],
    batch_size: int,
    device: torch.device,
    *,
    shuffle_within_time: bool = False,
) -> Iterator[WindowBatch]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ordered = sorted(records, key=lambda row: (row.event_time, row.owner_id))
    for _, grouped in itertools.groupby(ordered, key=lambda row: row.event_time):
        event_rows = list(grouped)
        if shuffle_within_time:
            permutation = torch.randperm(len(event_rows)).tolist()
            event_rows = [event_rows[index] for index in permutation]
        for start in range(0, len(event_rows), batch_size):
            rows = event_rows[start : start + batch_size]
            yield WindowBatch(
                context=torch.stack([row.context for row in rows]).to(device),
                target=torch.stack([row.target for row in rows]).to(device),
                owners=torch.tensor([row.owner_id for row in rows], device=device),
                families=torch.tensor([row.family_id for row in rows], device=device),
                event_times=torch.tensor(
                    [row.event_time for row in rows], device=device
                ),
                regimes=torch.tensor([row.regime for row in rows], device=device),
                aliases=torch.tensor([row.aliased for row in rows], device=device),
            )


def set_deterministic_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def retrieval_regime_precision(
    model: MemoryPredictor,
    output: ITFMOutput,
    query_regimes: torch.Tensor,
) -> list[float | None]:
    indices = output.retrieval.indices.clamp_min(0)
    selected_regimes = model.memory.regimes[indices]
    correct = selected_regimes == query_regimes.view(-1, 1, 1, 1)
    weights = output.retrieval.weights
    numerator = (weights * correct.to(weights.dtype)).sum(dim=(1, 2, 3))
    denominator = weights.sum(dim=(1, 2, 3)).clamp_min(1e-12)
    scores = numerator / denominator
    return [
        float(scores[index].item())
        if bool(output.retrieval.valid[index].item())
        else None
        for index in range(query_regimes.numel())
    ]


def train_one_epoch(
    model: nn.Module,
    records: list[WindowRecord],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    device: torch.device,
    gradient_clip_norm: float,
    use_memory: bool,
    intervention: str = "none",
) -> float:
    model.train()
    if isinstance(model, MEMORY_PREDICTOR_TYPES):
        model.clear_memory()
    total_squared_error = 0.0
    total_elements = 0
    for batch in chronological_batches(
        records, batch_size, device, shuffle_within_time=True
    ):
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, MEMORY_PREDICTOR_TYPES):
            output = model(
                batch.context,
                batch.owners,
                batch.event_times,
                use_memory=use_memory,
                intervention=intervention,
            )
            prediction = output.prediction
        else:
            output = None
            prediction = model(batch.context)
        loss = torch.mean((prediction - batch.target).square())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        if isinstance(model, MEMORY_PREDICTOR_TYPES) and output is not None:
            model.commit_from_output(
                output,
                batch.target,
                batch.owners,
                batch.event_times,
                regimes=batch.regimes,
            )
        total_squared_error += float(
            (prediction.detach() - batch.target).square().sum().item()
        )
        total_elements += batch.target.numel()
    return total_squared_error / max(1, total_elements)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    records: list[WindowRecord],
    *,
    batch_size: int,
    device: torch.device,
    use_memory: bool,
    intervention: str = "none",
) -> list[PredictionRow]:
    model.eval()
    if isinstance(model, MEMORY_PREDICTOR_TYPES):
        model.clear_memory()
    rows: list[PredictionRow] = []
    for batch in chronological_batches(records, batch_size, device):
        if isinstance(model, MEMORY_PREDICTOR_TYPES):
            ordinary = model(
                batch.context,
                batch.owners,
                batch.event_times,
                use_memory=use_memory,
                intervention=intervention,
            )
            prediction = ordinary.prediction
            retrieval_valid = ordinary.retrieval.valid
            eligible_counts = ordinary.retrieval.eligible_counts
            regime_precision = retrieval_regime_precision(
                model, ordinary, batch.regimes
            )
            transport_shifts = ordinary.transport_shifts
            model.commit_from_output(
                ordinary,
                batch.target,
                batch.owners,
                batch.event_times,
                regimes=batch.regimes,
            )
        else:
            prediction = model(batch.context)
            retrieval_valid = torch.zeros_like(batch.aliases)
            eligible_counts = torch.zeros_like(batch.event_times)
            regime_precision = [None] * batch.context.shape[0]
            transport_shifts = None

        per_item = (prediction - batch.target).square().flatten(1).sum(dim=1)
        element_count = batch.target[0].numel()
        for index in range(batch.context.shape[0]):
            rows.append(
                PredictionRow(
                    owner_id=int(batch.owners[index].item()),
                    family_id=int(batch.families[index].item()),
                    event_time=int(batch.event_times[index].item()),
                    regime=int(batch.regimes[index].item()),
                    aliased=bool(batch.aliases[index].item()),
                    squared_error=float(per_item[index].item()),
                    element_count=element_count,
                    retrieval_valid=bool(retrieval_valid[index].item()),
                    retrieval_eligible_count=int(eligible_counts[index].item()),
                    retrieval_regime_precision=regime_precision[index],
                    transport_shift=(
                        tuple(int(value) for value in transport_shifts[index].tolist())
                        if transport_shifts is not None
                        else None
                    ),
                )
            )
    return rows


@torch.no_grad()
def evaluate_interventions(
    model: MemoryPredictor,
    records: list[WindowRecord],
    *,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate declared memory-only interventions in chronological event batches.

    Counterpart swaps exchange retrieved residual fields between the two regimes
    of an oracle-matched family at the same event time. The ordinary pass alone
    is committed, so interventions cannot mutate later memory state.
    """

    model.eval()
    model.clear_memory()
    rows: dict[str, list[PredictionRow]] = {
        name: []
        for name in ("none", "erase", "random_value", "value_swap", "oracle_value")
    }
    direction_scores: list[float] = []
    ordered = sorted(
        records, key=lambda row: (row.event_time, row.family_id, row.regime)
    )
    event_times = sorted({row.event_time for row in ordered})
    for event_time in event_times:
        event_records = [row for row in ordered if row.event_time == event_time]
        batch = next(chronological_batches(event_records, len(event_records), device))
        ordinary = model(batch.context, batch.owners, batch.event_times)
        erased = model(
            batch.context,
            batch.owners,
            batch.event_times,
            intervention="erase",
        )
        randomized = model(
            batch.context,
            batch.owners,
            batch.event_times,
            intervention="random_value",
        )

        family_regime_to_index = {
            (int(batch.families[index].item()), int(batch.regimes[index].item())): index
            for index in range(batch.context.shape[0])
        }
        donor_indices = []
        for index in range(batch.context.shape[0]):
            family = int(batch.families[index].item())
            regime = int(batch.regimes[index].item())
            donor_indices.append(family_regime_to_index[(family, -regime)])
        donor_index = torch.tensor(donor_indices, device=device, dtype=torch.long)
        donor_retrieved = ordinary.retrieved[donor_index]
        swapped = model(
            batch.context,
            batch.owners,
            batch.event_times,
            intervention="value_swap",
            replacement=donor_retrieved,
        )
        oracle_residual = batch.target - ordinary.base_prediction
        oracle = model(
            batch.context,
            batch.owners,
            batch.event_times,
            intervention="oracle_value",
            replacement=oracle_residual,
        )

        predictions = {
            "none": ordinary.prediction,
            "erase": erased.prediction,
            "random_value": randomized.prediction,
            "value_swap": swapped.prediction,
            "oracle_value": oracle.prediction,
        }
        regime_precision = retrieval_regime_precision(model, ordinary, batch.regimes)
        for intervention, prediction in predictions.items():
            per_item = (prediction - batch.target).square().flatten(1).sum(dim=1)
            for index in range(batch.context.shape[0]):
                rows[intervention].append(
                    PredictionRow(
                        owner_id=int(batch.owners[index].item()),
                        family_id=int(batch.families[index].item()),
                        event_time=int(batch.event_times[index].item()),
                        regime=int(batch.regimes[index].item()),
                        aliased=bool(batch.aliases[index].item()),
                        squared_error=float(per_item[index].item()),
                        element_count=batch.target[index].numel(),
                        retrieval_valid=bool(ordinary.retrieval.valid[index].item()),
                        retrieval_eligible_count=int(
                            ordinary.retrieval.eligible_counts[index].item()
                        ),
                        retrieval_regime_precision=regime_precision[index],
                        transport_shift=(
                            tuple(
                                int(value)
                                for value in ordinary.transport_shifts[index].tolist()
                            )
                            if ordinary.transport_shifts is not None
                            else None
                        ),
                    )
                )

        scores = donor_direction_score(
            ordinary.prediction,
            swapped.prediction,
            batch.target,
            batch.target[donor_index],
        )
        direction_scores.extend(
            float(scores[index].item())
            for index in range(batch.context.shape[0])
            if bool(batch.aliases[index].item())
            and bool(ordinary.retrieval.valid[index].item())
        )
        model.commit_from_output(
            ordinary,
            batch.target,
            batch.owners,
            batch.event_times,
            regimes=batch.regimes,
        )
    return {
        "rows": rows,
        "swap_donor_direction_scores": direction_scores,
    }


def fit_model(
    model: nn.Module,
    records: list[WindowRecord],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    device: torch.device,
    use_memory: bool,
    intervention: str = "none",
) -> list[float]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    for _ in range(epochs):
        history.append(
            train_one_epoch(
                model,
                records,
                optimizer,
                batch_size=batch_size,
                device=device,
                gradient_clip_norm=gradient_clip_norm,
                use_memory=use_memory,
                intervention=intervention,
            )
        )
    return history
