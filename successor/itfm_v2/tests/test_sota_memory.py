from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fim_sota.memory import TensorMemoryBank


def bank(eviction: str = "fifo", capacity: int = 4) -> TensorMemoryBank:
    return TensorMemoryBank(
        key_dimension=2,
        residual_shape=(1, 2, 2),
        capacity_per_trajectory=capacity,
        minimum_age=2,
        topk=2,
        temperature=0.05,
        minimum_similarity=0.5,
        chunk_size=1,
        eviction=eviction,  # type: ignore[arg-type]
        seed=7,
    )


def write_two(memory: TensorMemoryBank) -> None:
    memory.write_batch(
        keys=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        residuals=torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]]
        ),
        trajectory_ids=("a", "a"),
        event_times=torch.tensor([0, 1]),
        available_times=torch.tensor([1, 2]),
    )


def test_batched_chunked_retrieval_matches_reference_and_is_causal() -> None:
    memory = bank()
    write_two(memory)
    batch = memory.retrieve_batch(
        query_keys=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16),
        trajectory_ids=("a", "a"),
        query_times=torch.tensor([2, 3]),
    )
    assert batch.valid.tolist() == [True, True]
    for index, time in enumerate((2, 3)):
        reference = memory.exact_reference(
            query_key=torch.tensor([1.0, 0.0])
            if index == 0
            else torch.tensor([0.0, 1.0]),
            trajectory_id="a",
            query_time=time,
        )
        assert torch.allclose(
            batch.residual[index].float(), reference.residual[0].float()
        )
        assert batch.source_event_time[index] < time
        assert batch.source_available_time[index] <= time


def test_unknown_owner_and_similarity_rejection_fall_back_to_zero() -> None:
    memory = bank()
    write_two(memory)
    unknown = memory.retrieve_batch(
        query_keys=torch.tensor([[1.0, 0.0]]),
        trajectory_ids=("unknown",),
        query_times=torch.tensor([10]),
    )
    assert not unknown.valid.item()
    assert torch.equal(unknown.residual, torch.zeros_like(unknown.residual))


def test_deterministic_tie_break_and_value_interventions_do_not_mutate_bank() -> None:
    memory = bank()
    memory.write_batch(
        keys=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        residuals=torch.stack((torch.ones(1, 2, 2), torch.full((1, 2, 2), 2.0))),
        trajectory_ids=("a", "a"),
        event_times=torch.tensor([0, 0]),
        available_times=torch.tensor([1, 1]),
    )
    before = memory.checksum()
    common = {
        "query_keys": torch.tensor([[1.0, 0.0]]),
        "trajectory_ids": ("a",),
        "query_times": torch.tensor([2]),
    }
    ordinary = memory.retrieve_batch(**common)
    erased = memory.retrieve_batch(**common, value_intervention="erase")
    flipped = memory.retrieve_batch(**common, value_intervention="sign_flip")
    assert ordinary.source_serial.item() == 0
    assert torch.equal(erased.residual, torch.zeros_like(erased.residual))
    assert torch.allclose(flipped.residual, -ordinary.residual)
    assert memory.checksum() == before


@pytest.mark.parametrize("eviction", ["fifo", "reservoir", "diversity", "utility"])
def test_eviction_never_exceeds_per_trajectory_capacity(eviction: str) -> None:
    memory = bank(eviction=eviction, capacity=2)
    memory.write_batch(
        keys=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        residuals=torch.ones(3, 1, 2, 2),
        trajectory_ids=("a", "a", "a"),
        event_times=torch.arange(3),
        available_times=torch.arange(3) + 1,
        utilities=torch.tensor([0.0, 1.0, 2.0]),
    )
    assert len(memory) == 2


def test_serialization_round_trip_preserves_checksum(tmp_path: Path) -> None:
    memory = bank()
    write_two(memory)
    path = tmp_path / "bank.pt"
    artifact_sha = memory.save(path)
    restored = TensorMemoryBank.load(path)
    assert len(artifact_sha) == 64
    assert restored.checksum() == memory.checksum()


def test_recent_only_requires_bounded_age() -> None:
    memory = bank()
    write_two(memory)
    with pytest.raises(ValueError, match="maximum_age"):
        memory.retrieve_batch(
            query_keys=torch.tensor([[1.0, 0.0]]),
            trajectory_ids=("a",),
            query_times=torch.tensor([3]),
            selection="recent_only",
        )
