"""Synthetic admission regressions; no training, data fetch or benchmark runs."""

from copy import deepcopy

import pytest
import torch

from fim.memory.trace_bank import TraceBank


def snapshot(bank):
    return ([t.clone() for t in bank.as_tensors()], bank._clock, bank.device,
            [deepcopy(t.metadata) for t in bank.iter_traces()])


def assert_snapshot(bank, before):
    tensors, clock, device, metadata = before
    for actual, expected in zip(bank.as_tensors(), tensors):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (bank._clock, bank.device) == (clock, device)
    assert [t.metadata for t in bank.iter_traces()] == metadata


@pytest.mark.parametrize("batched", [False, True])
def test_admitted_trace_owns_caller_tensors_and_nested_metadata(batched):
    bank = TraceBank(2, 2)
    key = torch.tensor([1., 0.], requires_grad=True)
    value = torch.tensor([3., 4.], requires_grad=True)
    metadata = {"provenance": {"ids": ["original"]}}
    if batched:
        bank.batch_add(key[None], value[None], torch.ones(1), 1, metadata=[metadata])
    else:
        bank.add(key, value, 1., 1, metadata=metadata)
    before = snapshot(bank)
    with torch.no_grad():
        key.zero_()
        value.zero_()
    metadata["provenance"]["ids"].append("changed")
    assert_snapshot(bank, before)
    assert not bank.as_tensors()[0].requires_grad
    torch.testing.assert_close(bank.retrieve(torch.tensor([1., 0.])), torch.tensor([[3., 4.]]))


@pytest.mark.parametrize("metadata", [[], [{}], [{}, {}, {}], [{}, 3]])
def test_malformed_batch_metadata_does_not_partially_insert(metadata):
    bank = TraceBank(2, 2)
    bank.add(torch.tensor([-1., 0.]), torch.tensor([8., 9.]), 2., 1)
    before = snapshot(bank)
    with pytest.raises(ValueError, match="metadata"):
        bank.batch_add(torch.eye(2), torch.eye(2), torch.ones(2), 9, metadata=metadata)
    assert_snapshot(bank, before)


@pytest.mark.parametrize("field", ["keys", "values", "saliences"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_bad_later_row_rolls_back_entire_batch(field, bad):
    bank = TraceBank(2, 2)
    bank.add(torch.tensor([1., 0.]), torch.tensor([3., 4.]), 1., 1)
    args = {"keys": torch.eye(2), "values": torch.eye(2), "saliences": torch.ones(2)}
    args[field][1] = bad
    before = snapshot(bank)
    with pytest.raises(ValueError, match="finite"):
        bank.batch_add(**args, timestamp=11)
    assert_snapshot(bank, before)


@pytest.mark.parametrize("field, bad", [("key", torch.ones(3)), ("value", torch.ones(3)),
    ("salience", float("nan")), ("timestamp", 1.5), ("timestamp", -1), ("layer", True)])
def test_rejected_first_write_leaves_device_unbound(field, bad):
    bank = TraceBank(2, 2)
    args = dict(key=torch.ones(2), value=torch.ones(2), salience=1., timestamp=1)
    args[field] = bad
    before = snapshot(bank)
    with pytest.raises(ValueError):
        bank.add(**args)
    assert_snapshot(bank, before)
    assert bank.device is None


def test_dtype_overflow_is_rejected_before_changing_existing_memory():
    bank = TraceBank(2, 2, dtype=torch.float32)
    bank.add(torch.tensor([1., 0.]), torch.tensor([3., 4.]), 1., 1)
    before = snapshot(bank)
    with pytest.raises(ValueError, match="conversion"):
        bank.batch_add(torch.eye(2, dtype=torch.float64),
                       torch.tensor([[1., 2.], [1e100, 1.]], dtype=torch.float64), torch.ones(2), 8)
    assert_snapshot(bank, before)


def test_merge_prune_and_top1_retrieval_have_known_answers():
    bank = TraceBank(2, 2, max_traces=2, decay_rate=0., dtype=torch.float64)
    bank.add(torch.tensor([1., 0.]), torch.tensor([0., 4.]), 1., 1, metadata={"old": True})
    bank.add(torch.tensor([1., 0.]), torch.tensor([4., 0.]), 3., 2, metadata={"new": True})
    assert len(bank) == 1
    torch.testing.assert_close(bank.as_tensors()[1], torch.tensor([[3., 1.]], dtype=torch.float64))
    assert next(bank.iter_traces()).metadata == {"old": True, "new": True}
    bank.add(torch.tensor([0., 1.]), torch.tensor([8., 9.]), 2., 3)
    bank.add(torch.tensor([-1., 0.]), torch.tensor([10., 11.]), .5, 4)
    assert len(bank) == 2
    torch.testing.assert_close(bank.retrieve(torch.tensor([[1., 0.]], dtype=torch.float64), topk=1),
                               torch.tensor([[3., 1.]], dtype=torch.float64))
    assert bank._clock == 4


def test_batch_and_sequential_admission_agree_for_valid_merge_prune_sequence():
    keys = torch.tensor([[1., 0.], [1., .01], [0., 1.], [-1., 0.]])
    values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    saliences = torch.tensor([[1., 1.], [3., 3.], [2., 2.], [.5, .5]])
    metadata = [{"row": i} for i in range(4)]
    batch = TraceBank(2, 2, max_traces=2)
    single = TraceBank(2, 2, max_traces=2)
    batch.batch_add(keys, values, saliences, 5, metadata=metadata)
    for i in range(4):
        single.add(keys[i], values[i], saliences[i].mean(), 5, metadata=metadata[i])
    assert_snapshot(batch, snapshot(single))


def test_staging_failure_after_an_earlier_merge_does_not_publish(monkeypatch):
    bank = TraceBank(2, 2)
    bank.add(torch.tensor([1., 0.]), torch.tensor([1., 2.]), 1., 1)
    before = snapshot(bank)
    original = torch.matmul
    calls = 0
    def fail_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated device failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "matmul", fail_on_second)
    with pytest.raises(RuntimeError, match="device failure"):
        bank.batch_add(torch.eye(2), torch.eye(2), torch.ones(2), 7)
    assert_snapshot(bank, before)


@pytest.mark.parametrize("kwargs", [{"key_dim": 0}, {"max_traces": -1}, {"value_dim": True},
    {"merge_threshold": float("nan")}, {"decay_rate": -1.}, {"dtype": torch.int64}])
def test_invalid_constructor_contract_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TraceBank(**(dict(key_dim=2, value_dim=2) | kwargs))


def test_empty_batch_does_not_bind_device_or_advance_clock():
    bank = TraceBank(2, 2)
    bank.batch_add(torch.empty(0, 2), torch.empty(0, 2), torch.empty(0), timestamp=4)
    assert len(bank) == 0 and bank._clock == 0 and bank.device is None
