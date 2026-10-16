import pytest
import torch

from fim_s1.memory import TensorFabricMemory


def make_memory(*, topk: int = 1) -> TensorFabricMemory:
    return TensorFabricMemory(
        capacity=8,
        key_channels=1,
        value_channels=1,
        height=1,
        width=2,
        topk=topk,
        temperature=0.05,
    )


def test_retrieval_is_spatial_and_query_aligned() -> None:
    memory = make_memory()
    memory.add(
        keys=torch.tensor([[[[1.0, -1.0]]], [[[-1.0, 1.0]]]]),
        values=torch.tensor([[[[10.0, 10.0]]], [[[20.0, 20.0]]]]),
        owners=torch.tensor([7, 7]),
        event_times=torch.tensor([0, 1]),
    )
    result = memory.retrieve(
        torch.tensor([[[[1.0, 1.0]]]]),
        owners=torch.tensor([7]),
        query_times=torch.tensor([2]),
    )
    assert result.valid.tolist() == [True]
    torch.testing.assert_close(result.value, torch.tensor([[[[10.0, 20.0]]]]))


def test_cross_owner_and_future_entries_are_ineligible() -> None:
    memory = make_memory()
    memory.add(
        torch.ones(1, 1, 1, 2),
        torch.ones(1, 1, 1, 2),
        owners=torch.tensor([4]),
        event_times=torch.tensor([5]),
    )
    cross_owner = memory.retrieve(
        torch.ones(1, 1, 1, 2), torch.tensor([9]), torch.tensor([6])
    )
    future = memory.retrieve(
        torch.ones(1, 1, 1, 2), torch.tensor([4]), torch.tensor([5])
    )
    assert not cross_owner.valid.item()
    assert not future.valid.item()
    assert torch.count_nonzero(cross_owner.value) == 0
    assert torch.count_nonzero(future.value) == 0


def test_interventions_are_explicit_and_fail_closed() -> None:
    memory = make_memory()
    memory.add(
        torch.ones(1, 1, 1, 2),
        torch.full((1, 1, 1, 2), 3.0),
        owners=torch.tensor([1]),
        event_times=torch.tensor([0]),
    )
    erased = memory.retrieve(
        torch.ones(1, 1, 1, 2),
        torch.tensor([1]),
        torch.tensor([1]),
        intervention="erase",
    )
    swapped = memory.retrieve(
        torch.ones(1, 1, 1, 2),
        torch.tensor([1]),
        torch.tensor([1]),
        intervention="value_swap",
        replacement=torch.full((1, 1, 1, 2), 9.0),
    )
    nuisance = memory.retrieve(
        torch.ones(1, 1, 1, 2),
        torch.tensor([1]),
        torch.tensor([1]),
        intervention="random_value",
    )
    assert torch.count_nonzero(erased.value) == 0
    torch.testing.assert_close(swapped.value, torch.full((1, 1, 1, 2), 9.0))
    torch.testing.assert_close(nuisance.value, torch.full((1, 1, 1, 2), -3.0))
    with pytest.raises(ValueError, match="requires a replacement"):
        memory.retrieve(
            torch.ones(1, 1, 1, 2),
            torch.tensor([1]),
            torch.tensor([1]),
            intervention="oracle_value",
        )


def test_writes_are_detached_and_fifo_bounded() -> None:
    memory = TensorFabricMemory(
        capacity=2,
        key_channels=1,
        value_channels=1,
        height=1,
        width=1,
    )
    for event_time in range(3):
        key = torch.tensor([[[[float(event_time + 1)]]]], requires_grad=True)
        value = torch.tensor([[[[float(event_time + 10)]]]], requires_grad=True)
        memory.add(key, value, torch.tensor([0]), torch.tensor([event_time]))
    assert len(memory) == 2
    assert set(memory.event_times.tolist()) == {1, 2}
    assert not memory.keys.requires_grad
    assert not memory.values.requires_grad


def test_vector_broadcast_removes_spatial_address_and_value_structure() -> None:
    memory = TensorFabricMemory(
        capacity=4,
        key_channels=1,
        value_channels=1,
        height=1,
        width=2,
        topk=1,
        temperature=0.05,
        spatial_addressing=False,
        spatial_values=False,
    )
    memory.add(
        keys=torch.tensor([[[[1.0, -1.0]]]]),
        values=torch.tensor([[[[2.0, 6.0]]]]),
        owners=torch.tensor([3]),
        event_times=torch.tensor([0]),
    )
    result = memory.retrieve(
        torch.tensor([[[[-1.0, 1.0]]]]),
        owners=torch.tensor([3]),
        query_times=torch.tensor([1]),
    )
    torch.testing.assert_close(result.value, torch.full((1, 1, 1, 2), 4.0))


@pytest.mark.parametrize("spatial_shape", [(5,), (2, 3, 4)])
def test_empty_memory_preserves_arbitrary_spatial_rank(spatial_shape) -> None:
    memory = TensorFabricMemory(
        capacity=4,
        key_channels=2,
        value_channels=3,
        spatial_shape=spatial_shape,
    )
    query = torch.ones(2, 2, *spatial_shape)
    result = memory.retrieve(query, torch.tensor([0, 1]), torch.tensor([1, 1]))
    assert result.value.shape == (2, 3, *spatial_shape)
    assert result.indices.shape == (2, 1, *spatial_shape)
    assert torch.count_nonzero(result.value) == 0


def test_three_dimensional_retrieval_is_location_specific() -> None:
    spatial_shape = (2, 2, 2)
    memory = TensorFabricMemory(
        capacity=4,
        key_channels=1,
        value_channels=1,
        spatial_shape=spatial_shape,
        topk=1,
        temperature=0.05,
    )
    pattern = torch.tensor([[[[[1.0, -1.0], [-1.0, 1.0]], [[-1.0, 1.0], [1.0, -1.0]]]]])
    memory.add(
        keys=torch.cat([pattern, -pattern], dim=0),
        values=torch.cat(
            [torch.full_like(pattern, 10.0), torch.full_like(pattern, 20.0)], dim=0
        ),
        owners=torch.tensor([4, 4]),
        event_times=torch.tensor([0, 1]),
    )
    result = memory.retrieve(
        torch.ones(1, 1, *spatial_shape),
        owners=torch.tensor([4]),
        query_times=torch.tensor([2]),
    )
    expected = torch.where(pattern > 0, 10.0, 20.0)
    torch.testing.assert_close(result.value, expected)


def test_vectorized_write_keeps_exact_last_capacity_for_large_batch() -> None:
    memory = TensorFabricMemory(
        capacity=3,
        key_channels=1,
        value_channels=1,
        spatial_shape=(1,),
    )
    memory.add(
        keys=torch.arange(1, 6, dtype=torch.float32).reshape(5, 1, 1),
        values=torch.arange(10, 15, dtype=torch.float32).reshape(5, 1, 1),
        owners=torch.arange(5),
        event_times=torch.arange(5),
    )
    assert len(memory) == 3
    assert memory.write_ptr.item() == 2
    assert set(memory.owners.tolist()) == {2, 3, 4}
    assert set(memory.event_times.tolist()) == {2, 3, 4}
    assert set(memory.values.flatten().tolist()) == {12.0, 13.0, 14.0}
