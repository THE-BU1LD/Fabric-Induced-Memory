"""Hand-verifiable checks for the public finite-window GL memory."""

import pytest
import torch

from fim.core.fractional import (
    FractionalMemory,
    fractional_difference,
    grunwald_letnikov_weights,
)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 1, 3, 5), (2, 3, 1, 4, 5)])
def test_window_truncates_time_and_preserves_all_feature_axes(shape):
    memory = FractionalMemory(alpha=0.5, max_history=3, dt=0.25)
    for step in range(8):
        value = torch.full(shape, float(step + 1), dtype=torch.float64)
        actual = memory.push(value)
        # w_0=1, w_1=-1/2, w_2=-1/8; dt**(-1/2)=2.
        recent = list(range(max(1, step - 1), step + 2))[::-1]
        expected = 2 * sum(w * x for w, x in zip((1.0, -0.5, -0.125), recent))
        assert memory.history_length == min(step + 1, 3)
        assert actual.shape == value.shape
        torch.testing.assert_close(actual, torch.full_like(actual, expected))


def test_first_push_owns_history_snapshot():
    memory = FractionalMemory(alpha=1.0, max_history=2)
    value = torch.tensor([2.0, 4.0])
    memory.push(value)
    value.add_(100.0)
    actual = memory.push(torch.tensor([7.0, 10.0]))
    torch.testing.assert_close(actual, torch.tensor([5.0, 6.0]))


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 1, 3, 5)])
def test_single_entry_window_replaces_old_observation(shape):
    memory = FractionalMemory(alpha=0.5, max_history=1, dt=0.25)
    for step in range(4):
        value = torch.full(shape, float(step + 1))
        torch.testing.assert_close(memory.push(value), 2 * value)
        assert memory.history_length == 1


def test_fractional_weights_match_binomial_coefficients():
    actual = grunwald_letnikov_weights(0.5, 4, dtype=torch.float64)
    torch.testing.assert_close(actual, torch.tensor([1, -0.5, -0.125, -0.0625], dtype=torch.float64))


def test_direct_difference_preserves_autograd_and_integer_order_limit():
    history = torch.tensor([[2.0, 4.0], [7.0, 10.0]], dtype=torch.float64, requires_grad=True)
    actual = fractional_difference(history, alpha=1.0, dt=0.5)
    torch.testing.assert_close(actual, torch.tensor([10.0, 12.0], dtype=torch.float64))
    actual.sum().backward()
    torch.testing.assert_close(history.grad, torch.tensor([[-2.0, -2.0], [2.0, 2.0]], dtype=torch.float64))
    assert torch.autograd.gradcheck(lambda h: fractional_difference(h, 0.5, 0.25), (history,))


def test_reset_releases_history_and_allows_new_geometry():
    memory = FractionalMemory(alpha=1.0, max_history=2)
    source = torch.ones(2, 3, requires_grad=True)
    assert not memory.push(source).requires_grad
    memory.reset()
    assert memory.history_length == 0
    value = torch.full((1, 2, 3, 5), 7.0, dtype=torch.float64).transpose(-1, -2)
    torch.testing.assert_close(memory.push(value), value)
    assert memory.history_length == 1


@pytest.mark.parametrize("invalid", [torch.ones(3), torch.ones(2, dtype=torch.float64), torch.tensor([float("nan"), 0.0])])
def test_invalid_push_leaves_previous_history_intact(invalid):
    memory = FractionalMemory(alpha=1.0, max_history=2)
    memory.push(torch.tensor([2.0, 4.0]))
    with pytest.raises(ValueError):
        memory.push(invalid)
    assert memory.history_length == 1
    torch.testing.assert_close(memory.push(torch.tensor([7.0, 10.0])), torch.tensor([5.0, 6.0]))


@pytest.mark.parametrize("max_history", [0, -1, 1.5, True])
def test_invalid_window_is_rejected_before_allocating(max_history):
    with pytest.raises(ValueError, match="positive integer"):
        FractionalMemory(alpha=0.5, max_history=max_history)


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_time_scale_is_rejected(dt):
    with pytest.raises(ValueError, match="dt"):
        FractionalMemory(alpha=0.5, dt=dt)
    with pytest.raises(ValueError, match="dt"):
        fractional_difference(torch.ones(2, 3), alpha=0.5, dt=dt)


@pytest.mark.parametrize("alpha", [0.0, -0.5, 1.5, float("nan"), float("inf"), True])
def test_invalid_fractional_order_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        FractionalMemory(alpha=alpha)
    with pytest.raises(ValueError, match="alpha"):
        grunwald_letnikov_weights(alpha, 3)


@pytest.mark.parametrize("order", [0, -1, 1.5, True])
def test_invalid_weight_count_is_rejected(order):
    with pytest.raises(ValueError, match="positive integer"):
        grunwald_letnikov_weights(0.5, order)


@pytest.mark.parametrize("history", [torch.ones(3), torch.empty(2, 0, 3), torch.ones(2, 3, dtype=torch.int64), torch.ones(2, 3, dtype=torch.complex64)])
def test_invalid_history_does_not_silently_change_gl_arithmetic(history):
    with pytest.raises(ValueError):
        fractional_difference(history, 0.5)


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_weights_reject_dtypes_that_change_real_fractional_coefficients(dtype):
    with pytest.raises(ValueError, match="floating dtype"):
        grunwald_letnikov_weights(0.5, 3, dtype=dtype)
