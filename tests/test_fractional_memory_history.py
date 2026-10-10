import pytest
import torch

from fim.core.fractional import (
    FractionalMemory,
    fractional_difference,
    grunwald_letnikov_weights,
)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 4, 5)])
@pytest.mark.parametrize("capacity", [1, 2, 4])
def test_rolling_history_matches_explicit_reference(shape, capacity):
    memory = FractionalMemory(alpha=0.5, max_history=capacity, dt=0.25)
    samples = []
    for step in range(9):
        x = torch.full(shape, float(step + 1), dtype=torch.float64)
        samples.append(x)
        actual = memory.push(x)
        retained = samples[-capacity:]
        # Independent scalar binomial recurrence for this hand-filled sequence.
        weight = 1.0
        expected = torch.zeros_like(x)
        for lag, sample in enumerate(reversed(retained)):
            if lag:
                weight *= -(0.5 - (lag - 1)) / lag
            expected += weight * sample * 2.0
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
        assert actual.shape == x.shape
        assert memory.history_length == min(step + 1, capacity)
        assert memory._history.numel() == x.numel() * memory.history_length


def test_gl_weights_and_integer_order_difference_are_hand_verifiable():
    weights = grunwald_letnikov_weights(0.5, 4, dtype=torch.float64)
    torch.testing.assert_close(weights, torch.tensor([1, -0.5, -0.125, -0.0625], dtype=torch.float64))
    history = torch.tensor([[100.0, 200.0], [2.0, 5.0], [6.0, 11.0]], dtype=torch.float64)
    torch.testing.assert_close(fractional_difference(history, alpha=1.0, dt=0.5), torch.tensor([8.0, 12.0], dtype=torch.float64))


def test_fractional_difference_preserves_gradient_graph():
    history = torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.float64, requires_grad=True)
    fractional_difference(history, alpha=0.5, dt=0.25).sum().backward()
    torch.testing.assert_close(history.grad[:, 0], torch.tensor([-0.25, -1.0, 2.0], dtype=torch.float64))


def test_memory_snapshots_detached_inputs_including_first_push():
    memory = FractionalMemory(alpha=1.0, max_history=2)
    first = torch.tensor([2.0, 3.0], requires_grad=True)
    assert not memory.push(first).requires_grad
    with torch.no_grad():
        first.fill_(1000.0)
    torch.testing.assert_close(memory.push(torch.tensor([5.0, 7.0])), torch.tensor([3.0, 4.0]))


@pytest.mark.parametrize("invalid", [
    torch.tensor([float("nan"), 2.0]),
    torch.tensor([float("inf"), 2.0]),
    torch.ones(3),
    torch.ones(2, dtype=torch.float64),
    torch.ones(2, dtype=torch.int64),
    torch.tensor(1.0),
    torch.empty(0),
])
def test_rejected_push_preserves_previous_episode(invalid):
    memory = FractionalMemory(alpha=1.0, max_history=2)
    memory.push(torch.tensor([2.0, 3.0]))
    before = memory._history.clone()
    with pytest.raises((ValueError, TypeError, FloatingPointError)):
        memory.push(invalid)
    torch.testing.assert_close(memory._history, before)
    assert memory.history_length == 1
    torch.testing.assert_close(memory.push(torch.tensor([5.0, 7.0])), torch.tensor([3.0, 4.0]))


def test_computation_failure_preserves_history():
    memory = FractionalMemory(alpha=1.0, max_history=2)
    memory.push(torch.tensor([-3e38]))
    before = memory._history.clone()
    with pytest.raises(FloatingPointError, match="fractional difference"):
        memory.push(torch.tensor([3e38]))
    torch.testing.assert_close(memory._history, before)


def test_reset_allows_new_shape_and_dtype():
    memory = FractionalMemory(alpha=0.5, max_history=2)
    memory.push(torch.ones(2))
    memory.reset()
    assert memory.history_length == 0
    state = torch.ones(2, 3, dtype=torch.float64)
    torch.testing.assert_close(memory.push(state), state)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_history_budget_is_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        FractionalMemory(0.5, max_history=value)
    with pytest.raises(ValueError, match="positive integer"):
        grunwald_letnikov_weights(0.5, value)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_dt_is_rejected(value):
    with pytest.raises(ValueError, match="dt"):
        FractionalMemory(0.5, dt=value)
    with pytest.raises(ValueError, match="dt"):
        fractional_difference(torch.ones(2, 1), 0.5, dt=value)


def test_constant_history_uses_truncated_gl_not_caputo_correction():
    history = torch.ones(3, 1, dtype=torch.float64)
    torch.testing.assert_close(fractional_difference(history, 0.5), torch.tensor([0.375], dtype=torch.float64))


def test_complex_history_is_retained_without_dtype_coercion():
    memory = FractionalMemory(alpha=1.0, max_history=2)
    memory.push(torch.tensor([1 + 2j], dtype=torch.complex128))
    actual = memory.push(torch.tensor([3 + 5j], dtype=torch.complex128))
    torch.testing.assert_close(actual, torch.tensor([2 + 3j], dtype=torch.complex128))
