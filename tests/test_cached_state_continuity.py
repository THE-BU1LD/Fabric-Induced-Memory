import copy

import pytest
import torch

from fim.models.fim_model import FIMModel


@pytest.mark.parametrize('layers', [1, 2])
def test_cached_state_matches_explicit_state(layers):
    torch.manual_seed(17)
    model = FIMModel(in_channels=1, out_channels=1, latent_channels=8,
                     trace_dim=4, hidden_channels=8, num_layers=layers,
                     memory_threshold=1.0).eval()
    x = torch.randn(1, 1, 8, 8)
    with torch.no_grad():
        _, first = model.step(x, update_memory=False)
        explicit_model = copy.deepcopy(model)
        y_cached, cached = model.step(x, update_memory=False)
        y_explicit, explicit = explicit_model.step(
            x, state=first.stacked_state.clone(), update_memory=False)
    torch.testing.assert_close(cached.stacked_state, explicit.stacked_state)
    torch.testing.assert_close(y_cached, y_explicit)


def test_cached_state_reinitializes_for_new_input_shape():
    torch.manual_seed(18)
    model = FIMModel(in_channels=1, out_channels=1, latent_channels=8,
                     trace_dim=4, hidden_channels=8, num_layers=1,
                     memory_threshold=1.0).eval()
    with torch.no_grad():
        model.step(torch.randn(1, 1, 8, 8), update_memory=False)
        _, out = model.step(torch.randn(2, 1, 10, 12), update_memory=False)
    assert out.stacked_state.shape == (2, 1, 8, 10, 12)
    assert torch.isfinite(out.stacked_state).all()
