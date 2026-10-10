"""Analytic GL references and streaming-window regressions, without scientific runs."""

import math

import pytest
import torch

from fim.core.fractional import (
    FractionalMemory,
    fractional_difference,
    grunwald_letnikov_weights,
)


def _reference(states, alpha, dt):
    # Closed-form generalized binomial coefficients, independent of the
    # implementation's recursive weight construction and history layout.
    result = torch.zeros_like(states[-1])
    for lag, state in enumerate(reversed(states)):
        numerator = math.prod(alpha - j for j in range(lag))
        coefficient = (-1) ** lag * numerator / math.factorial(lag)
        result = result + coefficient * state
    return result / dt**alpha


@pytest.mark.parametrize("shape", [(5,), (2, 5), (2, 3, 5), (2, 1, 3, 5)])
@pytest.mark.parametrize("capacity", [1, 2, 4])
@pytest.mark.parametrize("alpha", [0.25, 0.8, 1.0])
def test_bounded_history_preserves_all_state_axes(shape, capacity, alpha):
    memory = FractionalMemory(alpha, max_history=capacity, dt=0.25)
    states = []
    base = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
    for step in range(9):
        value = base + step * 0.75
        states.append(value)
        actual = memory.push(value)
        assert actual.shape == shape
        assert memory.history_length == min(step + 1, capacity)
        torch.testing.assert_close(
            actual, _reference(states[-capacity:], alpha, 0.25),
            rtol=1e-12, atol=1e-12,
        )


def test_first_order_difference_has_expected_units_and_gradient():
    history = torch.tensor([[[2., 4.], [5., 10.], [9., 16.]]], requires_grad=True)
    actual = fractional_difference(history, alpha=1.0, dt=0.5)
    torch.testing.assert_close(actual, torch.tensor([[8., 12.]]))
    actual.sum().backward()
    torch.testing.assert_close(history.grad, torch.tensor([[[0., 0.], [-2., -2.], [2., 2.]]]))


def test_closed_form_half_order_weights():
    actual = grunwald_letnikov_weights(0.5, 5, dtype=torch.float64)
    torch.testing.assert_close(actual, torch.tensor([1., -0.5, -0.125, -0.0625, -0.0390625], dtype=torch.float64))


def test_streaming_memory_owns_detached_input_and_reset_isolates_episode():
    memory = FractionalMemory(1.0, max_history=2)
    value = torch.tensor([2., 3.], requires_grad=True)
    initial = memory.push(value)
    assert not initial.requires_grad
    with torch.no_grad():
        value.add_(100)
    torch.testing.assert_close(memory.push(torch.tensor([5., 7.])), torch.tensor([3., 4.]))
    memory.reset()
    assert memory.history_length == 0
    # After reset, dimensions and dtype can change as well as trajectory.
    other = torch.ones(2, 3, dtype=torch.float64)
    torch.testing.assert_close(memory.push(other), other)


@pytest.mark.parametrize("field,value", [
    ("alpha", 0.), ("alpha", -0.5), ("alpha", 1.01),
    ("alpha", float("nan")), ("alpha", float("inf")), ("alpha", True),
    ("dt", 0.), ("dt", -1.), ("dt", float("nan")), ("dt", float("inf")),
    ("dt", True), ("max_history", 0), ("max_history", -1),
    ("max_history", 1.5), ("max_history", True),
])
def test_invalid_configuration_fails_at_construction(field, value):
    config = {"alpha": 0.8, "dt": 1., "max_history": 3}
    config[field] = value
    with pytest.raises((TypeError, ValueError)):
        FractionalMemory(**config)


@pytest.mark.parametrize("bad", [
    torch.ones(2, 1), torch.ones(2, dtype=torch.float64),
    torch.tensor([float("nan"), 1.]), torch.tensor([float("inf"), 1.]),
    torch.empty(0), torch.tensor(1.), torch.ones(2, dtype=torch.int64),
    torch.ones(2, dtype=torch.complex64),
])
def test_failed_push_preserves_previous_history(bad):
    memory = FractionalMemory(1., max_history=2)
    memory.push(torch.tensor([2., 3.]))
    with pytest.raises((TypeError, ValueError)):
        memory.push(bad)
    assert memory.history_length == 1
    torch.testing.assert_close(memory.push(torch.tensor([5., 7.])), torch.tensor([3., 4.]))


def test_overflow_does_not_advance_history():
    memory = FractionalMemory(1., max_history=2, dt=0.5)
    value = torch.tensor([1.])
    memory.push(value)
    with pytest.raises(FloatingPointError):
        memory.push(torch.tensor([torch.finfo(torch.float32).max]))
    assert memory.history_length == 1
    torch.testing.assert_close(memory.push(torch.tensor([2.])), torch.tensor([2.]))


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex64])
def test_weights_reject_nonreal_float_dtype(dtype):
    with pytest.raises(TypeError):
        grunwald_letnikov_weights(0.8, 4, dtype=dtype)


@pytest.mark.parametrize("order", [0, -1, 2.5, True])
def test_weights_reject_invalid_order(order):
    with pytest.raises((TypeError, ValueError)):
        grunwald_letnikov_weights(0.8, order)


@pytest.mark.parametrize("dt", [0., -1., float("nan"), float("inf")])
def test_direct_difference_rejects_invalid_dt(dt):
    with pytest.raises(ValueError):
        fractional_difference(torch.ones(1, 2, 3), 0.8, dt)
