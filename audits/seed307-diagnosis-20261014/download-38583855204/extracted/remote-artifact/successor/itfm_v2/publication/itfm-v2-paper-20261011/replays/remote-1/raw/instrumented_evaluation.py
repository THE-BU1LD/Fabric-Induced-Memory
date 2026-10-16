from __future__ import annotations
import dataclasses
from collections import defaultdict
import torch
from .data import EpisodicAliasCorpus, EventRecord
from .memory import Retrieval, TensorResidualMemory

@dataclasses.dataclass(frozen=True)
class PredictionRow:
    owner_id: int
    family_id: int
    event_time: int
    regime: int
    anchor_id: int
    kind: str
    query: bool
    squared_error: float
    element_count: int
    retrieval_valid: bool
    retrieval_similarity: float
    retrieval_source_time: int | None
    retrieval_source_anchor: int | None
    retrieval_eligible_count: int
    donor_direction_success: bool | None

@dataclasses.dataclass(frozen=True)
class StreamResult:
    rows: tuple[PredictionRow, ...]
    calibration_numerator: float
    calibration_denominator: float

def _oracle_anchor(memory: TensorResidualMemory, row: EventRecord) -> Retrieval:
    eligible = [entry for entry in memory.entries if entry.owner_id == row.owner_id and entry.anchor_id == row.anchor_id and (entry.event_time <= row.event_time - memory.minimum_age)]
    if not eligible:
        return Retrieval(torch.zeros_like(row.target), False, 0.0, None, None, 0)
    entry = max(eligible, key=lambda item: item.event_time)
    return Retrieval(entry.value.clone(), True, 1.0, entry.event_time, entry.anchor_id, len(eligible))

def _retrieve(memory: TensorResidualMemory, row: EventRecord, *, mode: str, context_length: int) -> Retrieval:
    if mode == 'no_memory' or mode == 'erase':
        intervention = 'erase'
    elif mode == 'sign_flip':
        intervention = 'sign_flip'
    elif mode == 'vector_broadcast':
        intervention = 'vector_broadcast'
    else:
        intervention = 'none'
    if mode == 'last_value':
        return memory.last_value(owner_id=row.owner_id, query_time=row.event_time, shape=tuple(row.target.shape))
    if mode == 'oracle_anchor':
        return _oracle_anchor(memory, row)
    owner_id = row.owner_id
    if mode == 'paired_swap':
        owner_id = row.owner_id + 1 if row.regime == -1 else row.owner_id - 1
    return memory.retrieve(row.context, owner_id=owner_id, query_time=row.event_time, intervention=intervention, maximum_age=context_length if mode == 'recent_only' else None)

def evaluate_stream(records: list[EventRecord], corpus: EpisodicAliasCorpus, memory_config: dict[str, float | int], *, mode: str, alpha: float) -> StreamResult:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha must lie in [0,1]')
    allowed = {'no_memory', 'tensor_memory', 'erase', 'sign_flip', 'vector_broadcast', 'recent_only', 'last_value', 'paired_swap', 'oracle_anchor'}
    if mode not in allowed:
        raise ValueError(f'unknown evaluation mode: {mode}')
    memory = TensorResidualMemory(capacity=int(memory_config['capacity']), minimum_age=int(memory_config['minimum_age']), topk=int(memory_config['topk']), temperature=float(memory_config['temperature']), minimum_similarity=float(memory_config['minimum_similarity']))
    rows: list[PredictionRow] = []
    numerator = 0.0
    denominator = 0.0
    by_time: dict[int, list[EventRecord]] = defaultdict(list)
    for row in records:
        by_time[row.event_time].append(row)
    for event_time in sorted(by_time):
        pending: list[tuple[EventRecord, torch.Tensor, torch.Tensor]] = []
        target_index = {(row.family_id, row.regime): row.target for row in by_time[event_time]}
        for row in sorted(by_time[event_time], key=lambda item: (item.family_id, item.regime)):
            base = corpus.base_prediction(row, corpus.channels)
            retrieval = _retrieve(memory, row, mode=mode, context_length=corpus.context_length)
            correction = retrieval.value if retrieval.valid else torch.zeros_like(base)
            prediction = base + alpha * correction
            error = row.target - base
            if row.query and retrieval.valid:
                numerator += float((error * correction).sum().item())
                denominator += float(correction.square().sum().item())
            donor_success: bool | None = None
            if mode == 'paired_swap' and row.query and retrieval.valid:
                donor_target = target_index[row.family_id, -row.regime]
                ordinary_distance = float((base - donor_target).square().mean().item())
                swapped_distance = float((prediction - donor_target).square().mean().item())
                donor_success = swapped_distance < ordinary_distance
            _capture.append({'owner_id': row.owner_id, 'family_id': row.family_id, 'event_time': row.event_time, 'query': row.query, 'base': base.detach().clone(), 'prediction': prediction.detach().clone(), 'target': row.target.detach().clone(), 'correction': correction.detach().clone(), 'donor_target': target_index[row.family_id, -row.regime].detach().clone(), 'valid': retrieval.valid, 'source_time': retrieval.source_time})
            squared_error = float((prediction - row.target).square().sum().item())
            rows.append(PredictionRow(owner_id=row.owner_id, family_id=row.family_id, event_time=row.event_time, regime=row.regime, anchor_id=row.anchor_id, kind=row.kind, query=row.query, squared_error=squared_error, element_count=row.target.numel(), retrieval_valid=retrieval.valid, retrieval_similarity=retrieval.similarity, retrieval_source_time=retrieval.source_time, retrieval_source_anchor=retrieval.source_anchor, retrieval_eligible_count=retrieval.eligible_count, donor_direction_success=donor_success))
            pending.append((row, row.context, row.target - base))
        for row, key, residual in pending:
            memory.add(key, residual, owner_id=row.owner_id, event_time=row.event_time, anchor_id=row.anchor_id)
    return StreamResult(tuple(rows), numerator, denominator)

def calibrate_alpha(result: StreamResult) -> float:
    if result.calibration_denominator <= 0:
        return 0.0
    value = result.calibration_numerator / result.calibration_denominator
    return min(1.0, max(0.0, value))

def summarize(result: StreamResult, *, normalization_variance: float) -> dict[str, float]:
    if normalization_variance <= 0:
        raise ValueError('normalization_variance must be positive')
    query = [row for row in result.rows if row.query]
    nonquery = [row for row in result.rows if not row.query]
    if not query or not nonquery:
        raise ValueError('evaluation requires query and non-query rows')

    def mse(rows: list[PredictionRow]) -> float:
        return sum((row.squared_error for row in rows)) / sum((row.element_count for row in rows))
    valid_queries = [row for row in query if row.retrieval_valid]
    precision_rows = [row for row in valid_queries if row.retrieval_source_anchor is not None and row.anchor_id >= 0]
    donor_rows = [row for row in query if row.donor_direction_success is not None]
    return {'query_mse': mse(query), 'query_nmse': mse(query) / normalization_variance, 'nonquery_mse': mse(nonquery), 'nonquery_nmse': mse(nonquery) / normalization_variance, 'retrieval_rate': len(valid_queries) / len(query), 'retrieval_anchor_precision': sum((row.retrieval_source_anchor == row.anchor_id for row in precision_rows)) / len(precision_rows) if precision_rows else 0.0, 'swap_donor_direction': sum((bool(row.donor_direction_success) for row in donor_rows)) / len(donor_rows) if donor_rows else 0.0, 'query_rows': float(len(query)), 'nonquery_rows': float(len(nonquery))}

def training_target_variance(records: list[EventRecord]) -> float:
    flattened = torch.cat([row.target.flatten() for row in records])
    variance = float(flattened.var(unbiased=False).item())
    if variance <= 0:
        raise RuntimeError('training-target variance is not positive')
    return variance
