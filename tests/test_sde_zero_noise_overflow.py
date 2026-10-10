"""Analytic zero-noise checks with finite but square-overflowing coefficients."""

import pytest
import torch

from fim.stochastic.sde import SDEConfig
from fim.stochastic.sde_v2 import SDEIntegratorV2


@pytest.mark.parametrize("method", ["euler", "milstein"])
@pytest.mark.parametrize("dtype,magnitude", [(torch.float32, 1e20), (torch.float64, 1e200)])
def test_zero_noise_large_coefficients_match_deterministic_values_and_gradients(method, dtype, magnitude):
    x0 = torch.tensor([[1.0, 2.0]], dtype=dtype, requires_grad=True)
    drift_rate = torch.tensor(0.125, dtype=dtype, requires_grad=True)
    diffusion_rate = torch.tensor(magnitude, dtype=dtype, requires_grad=True)
    generator = torch.Generator().manual_seed(529)
    rng_before = generator.get_state().clone()
    solver = SDEIntegratorV2(
        SDEConfig(dt=0.1, steps=3, noise_scale=0.0, stability_clip=None),
        generator=generator,
    )
    trajectory = solver.integrate(
        x0,
        lambda x, t: drift_rate * x + t,
        lambda x, t: diffusion_rate * x,
        method,
        lambda x, t: diffusion_rate.expand_as(x),
    )
    reference_x = x0.detach().clone().requires_grad_(True)
    reference_rate = drift_rate.detach().clone().requires_grad_(True)
    expected = [reference_x]
    for index in range(3):
        expected.append(expected[-1] + (reference_rate * expected[-1] + index * 0.1) * 0.1)
    expected = torch.stack(expected, dim=1)
    torch.testing.assert_close(trajectory, expected, rtol=0, atol=0)
    trajectory[:, -1].sum().backward()
    expected[:, -1].sum().backward()
    torch.testing.assert_close(x0.grad, reference_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(drift_rate.grad, reference_rate.grad, rtol=0, atol=0)
    assert diffusion_rate.grad.item() == 0.0
    assert torch.equal(rng_before, generator.get_state())


@pytest.mark.parametrize("invalid_field", ["diffusion", "derivative"])
def test_zero_noise_still_validates_all_requested_coefficient_fields(invalid_field):
    solver = SDEIntegratorV2(SDEConfig(steps=1, noise_scale=0.0))
    def diffusion(x, t):
        return torch.full_like(x, float("nan")) if invalid_field == "diffusion" else x
    def derivative(x, t):
        return torch.full_like(x, float("inf")) if invalid_field == "derivative" else torch.ones_like(x)
    with pytest.raises(ValueError, match="diffusion.*finite"):
        solver.milstein(torch.ones(1, 1), lambda x, t: x, diffusion, derivative)


def test_zero_configured_scale_does_not_disable_custom_milstein_correction():
    solver = SDEIntegratorV2(
        SDEConfig(dt=0.1, steps=1, noise_scale=0.0, stability_clip=None),
        noise_fn=lambda x, dt: torch.full_like(x, 0.5),
        noise_variance_fn=lambda x, dt: 0.1,
    )
    x0 = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    actual = solver.milstein(x0, lambda x, t: torch.zeros_like(x), lambda x, t: x,
                             lambda x, t: torch.ones_like(x))[:, -1]
    expected = x0 * (1.0 + 0.5 + 0.5 * (0.5**2 - 0.1))
    torch.testing.assert_close(actual, expected, rtol=1e-15, atol=0)
    actual.sum().backward()
    torch.testing.assert_close(x0.grad, torch.full_like(x0, 1.575), rtol=1e-15, atol=0)
