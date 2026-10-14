import torch

from fim_v2.memory import TensorResidualMemory


def make_memory(**overrides) -> TensorResidualMemory:
    config = {
        "capacity": 2,
        "minimum_age": 2,
        "topk": 1,
        "temperature": 0.05,
        "minimum_similarity": 0.999,
    }
    config.update(overrides)
    return TensorResidualMemory(**config)


def test_owner_and_time_contract_is_fail_closed() -> None:
    memory = make_memory()
    key = torch.ones(3, 4, 4)
    value = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    memory.add(key, value, owner_id=4, event_time=3, anchor_id=2)
    assert not memory.retrieve(key, owner_id=5, query_time=8).valid
    assert not memory.retrieve(key, owner_id=4, query_time=4).valid
    result = memory.retrieve(key, owner_id=4, query_time=5)
    assert result.valid
    torch.testing.assert_close(result.value, value)


def test_similarity_threshold_rejects_nuisance_keys() -> None:
    memory = make_memory()
    key = torch.zeros(3, 4, 4)
    key[:, 0, 0] = 1.0
    other = torch.zeros_like(key)
    other[:, 1, 1] = 1.0
    memory.add(key, torch.ones(1, 4, 4), owner_id=1, event_time=0, anchor_id=0)
    assert not memory.retrieve(other, owner_id=1, query_time=3).valid


def test_interventions_change_only_retrieved_value() -> None:
    memory = make_memory()
    key = torch.randn(3, 4, 4)
    value = torch.randn(1, 4, 4)
    value = value - value.mean()
    memory.add(key, value, owner_id=1, event_time=0, anchor_id=0)
    ordinary = memory.retrieve(key, owner_id=1, query_time=3)
    erased = memory.retrieve(key, owner_id=1, query_time=3, intervention="erase")
    flipped = memory.retrieve(key, owner_id=1, query_time=3, intervention="sign_flip")
    broadcast = memory.retrieve(
        key, owner_id=1, query_time=3, intervention="vector_broadcast"
    )
    torch.testing.assert_close(flipped.value, -ordinary.value)
    assert torch.count_nonzero(erased.value) == 0
    torch.testing.assert_close(
        broadcast.value, torch.zeros_like(value), atol=1e-6, rtol=0
    )


def test_capacity_is_enforced_per_owner() -> None:
    memory = make_memory(capacity=2, minimum_similarity=-1.0)
    for owner in (1, 2):
        for time in range(3):
            memory.add(
                torch.full((3, 2, 2), float(time + 1)),
                torch.full((1, 2, 2), float(owner)),
                owner_id=owner,
                event_time=time,
                anchor_id=time,
            )
    assert len([entry for entry in memory.entries if entry.owner_id == 1]) == 2
    assert len([entry for entry in memory.entries if entry.owner_id == 2]) == 2
