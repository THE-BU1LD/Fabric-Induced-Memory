from pathlib import Path
import sys

import pytest
import torch

EXP = str(Path(__file__).resolve().parents[1] / 'fim_experiments')
_added = EXP not in sys.path
if _added:
    sys.path.insert(0, EXP)
try:
    from systems import AdaptiveMemoryBank, FIMSystem
finally:
    if _added:
        sys.path.remove(EXP)


def _bank():
    bank = AdaptiveMemoryBank(2, 4, age_decay=0.0).double()
    bank.add(torch.eye(2, dtype=torch.double), torch.tensor([[1., 0.], [0., 2.]], dtype=torch.double))
    return bank


def test_temperature_gradient_matches_finite_difference():
    bank = _bank()
    q = torch.tensor([[1., 0.2]], dtype=torch.double)
    temperature = torch.tensor(0.3, dtype=torch.double, requires_grad=True)
    loss = bank.retrieve(q, topk=2, temperature=temperature).sum()
    loss.backward()
    eps = 1e-6
    numerical = (bank.retrieve(q, topk=2, temperature=0.3 + eps).sum() - bank.retrieve(q, topk=2, temperature=0.3 - eps).sum()) / (2 * eps)
    assert temperature.grad is not None and temperature.grad.abs() > 0
    torch.testing.assert_close(temperature.grad, numerical, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize('temperature', [0.01, 0.2, 0.7, 2.0])
def test_float_and_tensor_temperature_agree(temperature):
    bank = _bank()
    q = torch.tensor([[1., 0.2]], dtype=torch.double)
    actual = bank.retrieve(q, topk=2, temperature=temperature)
    expected = bank.retrieve(q, topk=2, temperature=torch.tensor(temperature, dtype=torch.double))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    clamped = min(1., max(0.05, temperature))
    scores = torch.nn.functional.normalize(q, dim=-1) + 0.05
    reference = torch.softmax(scores / clamped, dim=-1) @ bank.values[:2]
    torch.testing.assert_close(actual, reference)


def test_model_prediction_loss_reaches_temperature():
    torch.manual_seed(17)
    model = FIMSystem(1, 8, 4, memory_capacity=16, retrieval_topk=4, salience_threshold=-1.)
    with torch.no_grad():
        for _ in range(4):
            model.step(torch.randn(1, 1, 1, 8))
    out = model.step(torch.randn(1, 1, 1, 8), store_traces=False)
    out.prediction.square().mean().backward()
    grad = model.retrieval_temperature.grad
    assert grad is not None and torch.isfinite(grad) and grad.abs() > 0
