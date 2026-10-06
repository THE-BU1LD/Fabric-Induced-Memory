import copy

import pytest
import torch

from fim.geometry.graph import GraphPropagation, GridGraph
from fim.models.fim_model import FIMModel


def identity_propagation(connectivity=4):
    propagation = GraphPropagation(1, connectivity=connectivity)
    with torch.no_grad():
        propagation.lin.weight.fill_(1.0)
        propagation.lin.bias.zero_()
    return propagation


def test_repeated_destinations_sum_all_incoming_messages():
    propagation = GraphPropagation(1)
    with torch.no_grad():
        propagation.lin.weight.fill_(1.0)
        propagation.lin.bias.zero_()
    field = torch.ones(2, 1, 3, 3, requires_grad=True)
    actual = propagation(field, GridGraph(3, 3).edge_index)
    degree = torch.tensor([[2., 3., 2.], [3., 4., 3.], [2., 3., 2.]])
    torch.testing.assert_close(actual, degree.expand(2, 1, 3, 3))
    actual.sum().backward()
    torch.testing.assert_close(field.grad, degree.expand(2, 1, 3, 3))
    torch.testing.assert_close(propagation.lin.weight.grad, torch.tensor([[48.]]))


@pytest.mark.parametrize("connectivity, expected_center", [(4, 4.0), (8, 8.0)])
def test_default_grid_matches_explicit_topology(connectivity, expected_center):
    propagation = identity_propagation(connectivity)
    field = torch.ones(1, 1, 3, 3)
    implicit = propagation(field)
    explicit = propagation(field, GridGraph(3, 3, connectivity).edge_index)
    torch.testing.assert_close(implicit, explicit)
    assert implicit[0, 0, 1, 1].item() == expected_center


def test_shape_changes_and_noncontiguous_fields_preserve_grid_semantics():
    propagation = identity_propagation()
    first = torch.ones(1, 1, 2, 6)
    second = torch.ones(1, 1, 4, 3).transpose(-1, -2)
    assert not second.is_contiguous()
    propagation(first)
    actual = propagation(second)
    expected = torch.tensor([[2., 3., 3., 2.], [3., 4., 4., 3.], [2., 3., 3., 2.]])
    torch.testing.assert_close(actual, expected[None, None])
    # Derived topology must not become a shape-bound learned checkpoint entry.
    assert set(propagation.state_dict()) == {"lin.weight", "lin.bias"}


def test_single_cell_and_explicit_empty_graph_have_zero_messages():
    propagation = identity_propagation()
    with torch.no_grad():
        propagation.lin.bias.fill_(5.0)
    assert GridGraph(1, 1).edge_index.shape == (2, 0)
    torch.testing.assert_close(propagation(torch.ones(2, 1, 1, 1)), torch.zeros(2, 1, 1, 1))
    empty = torch.empty((2, 0), dtype=torch.long)
    torch.testing.assert_close(propagation(torch.ones(1, 1, 3, 3), empty), torch.zeros(1, 1, 3, 3))


def test_custom_multigraph_matches_independent_edge_loop():
    torch.manual_seed(27)
    propagation = GraphPropagation(2).double()
    reference = copy.deepcopy(propagation)
    field = torch.randn(2, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    reference_field = field.detach().clone().requires_grad_()
    edges = torch.tensor([[0, 0, 1, 2, 4, 5], [3, 3, 3, 0, 3, 0]])
    actual = propagation(field, edges)
    nodes = reference_field.flatten(2).transpose(1, 2)
    expected = torch.zeros_like(nodes)
    for source, destination in edges.t().tolist():
        expected[:, destination] = expected[:, destination] + reference.lin(nodes[:, source])
    expected = expected.transpose(1, 2).reshape_as(reference_field)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(field.grad, reference_field.grad)
    torch.testing.assert_close(propagation.lin.weight.grad, reference.lin.weight.grad)
    torch.testing.assert_close(propagation.lin.bias.grad, reference.lin.bias.grad)


@pytest.mark.parametrize("edges, error", [
    (torch.zeros((3, 2), dtype=torch.long), ValueError),
    (torch.zeros((2, 1), dtype=torch.float32), TypeError),
    (torch.tensor([[-1], [0]]), ValueError),
    (torch.tensor([[0], [4]]), ValueError),
])
def test_invalid_custom_edges_fail_before_indexing(edges, error):
    with pytest.raises(error):
        GraphPropagation(1)(torch.ones(1, 1, 2, 2), edges)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (-1, 2)])
def test_grid_rejects_nonpositive_dimensions(shape):
    with pytest.raises(ValueError):
        GridGraph(*shape)


def test_grid_rejects_unsupported_connectivity():
    with pytest.raises(ValueError):
        GridGraph(3, 3, connectivity=6)
    with pytest.raises(ValueError):
        GraphPropagation(1, connectivity=6)


@pytest.mark.parametrize("field", [torch.ones(1, 2, 3, 3), torch.ones(1, 3, 3), torch.ones(1, 1, 0, 3)])
def test_field_shape_and_channels_are_checked(field):
    with pytest.raises(ValueError):
        GraphPropagation(1)(field)


@pytest.mark.parametrize("layers", [1, 2])
def test_graph_propagation_runs_through_full_model_without_memory_writes(layers):
    torch.manual_seed(28)
    propagation = GraphPropagation(8)
    model = FIMModel(
        in_channels=1, out_channels=1, latent_channels=8, trace_dim=4,
        hidden_channels=8, num_layers=layers, max_traces=8,
        propagation=propagation,
    ).eval()
    field = torch.randn(1, 1, 6, 8, requires_grad=True)
    prediction, state = model(field, update_memory=False, stochastic=False)
    assert prediction.shape == field.shape
    assert state.stacked_state.shape == (1, layers, 8, 6, 8)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(state.stacked_state).all()
    prediction.square().mean().backward()
    assert propagation.lin.weight.grad is not None
    assert torch.isfinite(propagation.lin.weight.grad).all()
    assert propagation.lin.weight.grad.abs().sum().item() > 0.0
    for bank in model.trace_banks:
        assert bank.fast_size.item() == 0
        assert bank.slow_size.item() == 0
