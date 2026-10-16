from __future__ import annotations

import torch

from fim_sota.adapter import (
    ChannelCalibrator,
    ContextKeyEncoder,
    ITFMRAdapter,
    MonotoneSimilarityGate,
    ResidualNormalizer,
)
from fim_sota.backbones import HSFNOBackbone, PersistenceBackbone
from fim_sota.contracts import ForecastBatch
from fim_sota.memory import TensorMemoryBank


def batch() -> ForecastBatch:
    target = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
    return ForecastBatch(
        context=torch.stack((target, target), dim=1),
        target=target,
        observation_mask=torch.ones_like(target, dtype=torch.bool),
        trajectory_ids=("a", "b"),
        event_times=torch.tensor([2, 2]),
        lead_times=torch.ones(2, dtype=torch.long),
        split="validation",
        target_available_times=torch.tensor([3, 3]),
    )


def test_empty_memory_is_bitwise_backbone_equivalent() -> None:
    data = batch()
    backbone = PersistenceBackbone()
    output = backbone.forward_batch(data)
    bank = TensorMemoryBank(
        key_dimension=4,
        residual_shape=(1, 2, 2),
        capacity_per_trajectory=4,
        minimum_age=1,
        topk=1,
        temperature=0.1,
    )
    adapter = ITFMRAdapter(
        key_encoder=ContextKeyEncoder(
            latent_dimension=8,
            key_dimension=4,
            mask_features=2,
            lead_features=2,
        ),
        bank=bank,
        normalizer=ResidualNormalizer(1),
        calibrator=ChannelCalibrator(1),
        gate=MonotoneSimilarityGate(),
    )
    adapted = adapter(output, data)
    assert torch.equal(adapted.prediction, output.prediction)
    assert torch.equal(adapted.correction, torch.zeros_like(adapted.correction))


def test_hsfno_never_appends_unobserved_prediction() -> None:
    history = torch.zeros(2, 3, 1, 2, 2)
    candidate = torch.ones(2, 1, 2, 2)
    try:
        HSFNOBackbone.observed_append(
            history, candidate, target_observed=torch.tensor([True, False])
        )
    except RuntimeError as error:
        assert "may not be appended" in str(error)
    else:
        raise AssertionError("unobserved predicted future was appended")
