"""Graph propagation must remain usable across evaluation and training modes."""

import pytest
import torch

from fim.geometry.graph import GraphPropagation, GridGraph
from fim.models.fim_model import FIMModel


def identity_propagation():
    propagation = GraphPropagation(1)
    with torch.no_grad():
        propagation.lin.weight.fill_(1.0)
        propagation.lin.bias.zero_()
    return propagation


def test_inference_warmup_does_not_poison_training_topology():
    propagation = identity_propagation()
    with torch.inference_mode():
        propagation(torch.ones(1, 1, 3, 3))

    field = torch.ones(1, 1, 3, 3, requires_grad=True)
    actual = propagation(field)
    degree = torch.tensor([[2., 3., 2.], [3., 4., 3.], [2., 3., 2.]])
    torch.testing.assert_close(actual, degree[None, None])
    actual.sum().backward()
    torch.testing.assert_close(field.grad, degree[None, None])
    torch.testing.assert_close(propagation.lin.weight.grad, torch.tensor([[24.]]))
    assert set(propagation.state_dict()) == {"lin.weight", "lin.bias"}


@pytest.mark.parametrize("implicit_topology", [False, True])
def test_autocast_preserves_field_dtype_and_accumulates_every_edge(implicit_topology):
    propagation = identity_propagation()
    field = torch.ones(1, 1, 3, 3, requires_grad=True)
    edges = None if implicit_topology else GridGraph(3, 3).edge_index
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = propagation(field, edges)

    degree = torch.tensor([[2., 3., 2.], [3., 4., 3.], [2., 3., 2.]])
    assert actual.dtype == field.dtype
    torch.testing.assert_close(actual, degree[None, None])
    actual.sum().backward()
    torch.testing.assert_close(field.grad, degree[None, None])
    torch.testing.assert_close(propagation.lin.weight.grad, torch.tensor([[24.]]))


@pytest.mark.parametrize("layers", [1, 2])
def test_full_model_trains_after_graph_inference_warmup(layers):
    torch.manual_seed(29)
    propagation = GraphPropagation(8)
    model = FIMModel(
        in_channels=1, out_channels=1, latent_channels=8, trace_dim=4,
        hidden_channels=8, num_layers=layers, max_traces=8,
        propagation=propagation,
    ).eval()
    with torch.inference_mode():
        model(
            torch.ones(1, 1, 6, 8), update_memory=False,
            cache_state=False, stochastic=False,
        )

    model.train()
    field = torch.randn(1, 1, 6, 8, requires_grad=True)
    prediction, _ = model(
        field, update_memory=False, cache_state=False, stochastic=False,
    )
    prediction.square().mean().backward()
    assert prediction.shape == field.shape
    assert torch.isfinite(prediction).all()
    assert field.grad is not None and torch.isfinite(field.grad).all()
    assert propagation.lin.weight.grad is not None
    assert torch.isfinite(propagation.lin.weight.grad).all()
    assert propagation.lin.weight.grad.abs().sum().item() > 0.0
    for bank in model.trace_banks:
        assert bank.fast_size.item() == 0
        assert bank.slow_size.item() == 0
