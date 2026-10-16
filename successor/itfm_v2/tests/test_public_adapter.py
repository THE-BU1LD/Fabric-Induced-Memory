from __future__ import annotations

import pytest
import torch

from fim_v2.public_adapter import CausalResidualAdapter, fit_clipped_alpha


def adapter() -> CausalResidualAdapter:
    return CausalResidualAdapter(
        capacity_per_trajectory=4,
        minimum_age=2,
        topk=1,
        temperature=0.05,
        minimum_similarity=0.9,
    )


def test_retrieval_is_same_trajectory_strictly_past_and_available() -> None:
    memory = adapter()
    memory.write(
        key=torch.tensor([1.0, 0.0]),
        residual=torch.ones(1, 2, 2),
        trajectory_id="a",
        event_time=2,
        available_time=3,
    )
    memory.write(
        key=torch.tensor([1.0, 0.0]),
        residual=torch.full((1, 2, 2), 9.0),
        trajectory_id="b",
        event_time=1,
        available_time=2,
    )
    blocked = memory.retrieve(
        query_key=torch.tensor([1.0, 0.0]),
        trajectory_id="a",
        query_time=3,
        residual_shape=(1, 2, 2),
    )
    assert not blocked.valid
    found = memory.retrieve(
        query_key=torch.tensor([1.0, 0.0]),
        trajectory_id="a",
        query_time=4,
        residual_shape=(1, 2, 2),
    )
    assert found.valid
    assert found.source_event_time == 2
    assert torch.equal(found.residual, torch.ones(1, 2, 2))


def test_latent_key_and_spatial_residual_can_have_different_shapes() -> None:
    memory = adapter()
    memory.write(
        key=torch.arange(8, dtype=torch.float32).reshape(2, 4),
        residual=torch.arange(16, dtype=torch.float32).reshape(1, 4, 4),
        trajectory_id="trajectory",
        event_time=0,
        available_time=1,
    )
    found = memory.retrieve(
        query_key=torch.arange(8, dtype=torch.float32),
        trajectory_id="trajectory",
        query_time=2,
        residual_shape=(1, 4, 4),
    )
    assert found.valid
    assert found.residual.shape == (1, 4, 4)


def test_interventions_change_only_the_retrieved_value() -> None:
    memory = adapter()
    residual = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    memory.write(
        key=torch.tensor([1.0, 0.0]),
        residual=residual,
        trajectory_id="a",
        event_time=0,
        available_time=0,
    )
    common = {
        "query_key": torch.tensor([1.0, 0.0]),
        "trajectory_id": "a",
        "query_time": 2,
        "residual_shape": (1, 2, 2),
    }
    assert torch.equal(
        memory.retrieve(**common, intervention="erase").residual,
        torch.zeros_like(residual),
    )
    assert torch.equal(
        memory.retrieve(**common, intervention="sign_flip").residual, -residual
    )
    assert torch.equal(
        memory.retrieve(**common, intervention="spatial_shuffle").residual,
        torch.tensor([[[4.0, 3.0], [2.0, 1.0]]]),
    )


def test_validation_alpha_is_clipped_and_requires_pairs() -> None:
    alpha = fit_clipped_alpha([torch.tensor([2.0, 4.0])], [torch.tensor([1.0, 2.0])])
    assert alpha == pytest.approx(1.0)
    assert fit_clipped_alpha([torch.ones(2)], [torch.zeros(2)]) == 0.0
    with pytest.raises(ValueError):
        fit_clipped_alpha([], [])
