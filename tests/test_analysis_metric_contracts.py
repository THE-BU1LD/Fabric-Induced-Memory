"""Artificial tensor checks; no benchmark or retained outcome is loaded."""
import importlib.util
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "fim_metric_contract_target",
    Path(__file__).resolve().parents[1] / "fim" / "analysis" / "metrics.py",
)
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)
NAMES = ["mse", "mae", "normalized_mse", "rare_event_recall", "rare_event_precision"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("shape", [(2,), (1, 2), (2, 1, 1)])
def test_rejects_broadcasting(name, shape):
    with pytest.raises(ValueError, match="same shape"):
        getattr(metrics, name)(torch.ones(2, 1), torch.zeros(shape))


@pytest.mark.parametrize("name", NAMES)
def test_rejects_empty_tensors(name):
    with pytest.raises(ValueError, match="nonempty"):
        getattr(metrics, name)(torch.empty(0), torch.empty(0))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", [0, 1])
def test_rejects_nonfinite_input(name, value, side):
    pair = [torch.ones(3), torch.zeros(3)]
    pair[side][1] = value
    with pytest.raises(ValueError, match="finite"):
        getattr(metrics, name)(*pair)


@pytest.mark.parametrize("name", NAMES)
def test_rejects_complex_input(name):
    with pytest.raises(ValueError, match="real"):
        getattr(metrics, name)(torch.ones(3, dtype=torch.complex64), torch.zeros(3))


@pytest.mark.parametrize("name", ["normalized_mse", "rare_event_recall", "rare_event_precision"])
@pytest.mark.parametrize("eps", [0, -1, float("nan"), float("inf"), True, "0.01"])
def test_rejects_invalid_epsilon(name, eps):
    with pytest.raises(ValueError, match="eps"):
        getattr(metrics, name)(torch.ones(2), torch.ones(2), eps=eps)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_valid_regression_values_and_gradients_match_previous_formula(dtype):
    pred = torch.tensor([[1., 4.], [2., -1.]], dtype=dtype, requires_grad=True)
    target = torch.tensor([[0., 2.], [3., -2.]], dtype=dtype)
    for name, expected in [
        ("mse", torch.mean((pred - target) ** 2)),
        ("mae", torch.mean(torch.abs(pred - target))),
        ("normalized_mse", torch.mean((pred-target)**2) / torch.mean(target**2).clamp_min(1e-8)),
    ]:
        result = getattr(metrics, name)(pred, target)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(result, pred, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, pred, retain_graph=True)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32, torch.float64])
def test_event_masks_retain_hand_verifiable_scores(dtype):
    pred = torch.tensor([1, 0, 1, 1], dtype=dtype)
    target = torch.tensor([1, 1, 0, 1], dtype=dtype)
    for name in ["rare_event_precision", "rare_event_recall"]:
        torch.testing.assert_close(getattr(metrics, name)(pred, target), torch.tensor(2/3))


@pytest.mark.parametrize("name", ["rare_event_recall", "rare_event_precision"])
def test_no_events_retains_zero_convention(name):
    assert getattr(metrics, name)(torch.zeros(3), torch.zeros(3)).item() == 0


def test_zero_target_retains_epsilon_denominator():
    pred, target = torch.ones(2), torch.zeros(2)
    torch.testing.assert_close(metrics.normalized_mse(pred, target, eps=.25), torch.tensor(4.))


@pytest.mark.parametrize("name", NAMES)
def test_rejects_non_tensor(name):
    with pytest.raises(TypeError, match="torch.Tensor"):
        getattr(metrics, name)([1, 2], torch.ones(2))
