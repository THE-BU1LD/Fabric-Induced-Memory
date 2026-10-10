"""Generated/analytic integrator fixtures; no scientific datasets or outcomes."""
import pytest
import torch

from fim.stochastic.sde import SDEConfig, SDEIntegrator
from fim.stochastic.sde_v2 import SDEIntegratorV2


@pytest.mark.parametrize('method', ['euler', 'milstein'])
def test_zero_noise_reduces_exactly_to_deterministic_euler(method):
    x0 = torch.tensor([[2.0], [3.0]], dtype=torch.float64, requires_grad=True)
    solver = SDEIntegratorV2(SDEConfig(dt=0.1, steps=3, noise_scale=0, stability_clip=None))
    y = solver.integrate(x0, lambda x, t: 0.3*x, lambda x, t: x,
                         method, lambda x, t: torch.ones_like(x))
    expected = x0 * (1.03**3)
    torch.testing.assert_close(y[:, -1], expected, rtol=1e-14, atol=1e-14)
    y[:, -1].sum().backward()
    torch.testing.assert_close(x0.grad, torch.full_like(x0, 1.03**3))


@pytest.mark.parametrize('scale', [0.5, 1.0, 2.0])
def test_milstein_matches_hand_computed_multiplicative_noise_step(scale):
    seed, dt = 113, 0.125
    x0 = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    dw = torch.randn(x0.shape, dtype=x0.dtype,
                     generator=torch.Generator().manual_seed(seed)) * (dt**0.5*scale)
    solver = SDEIntegratorV2(SDEConfig(dt=dt, steps=1, noise_scale=scale, stability_clip=None),
                            generator=torch.Generator().manual_seed(seed))
    actual = solver.milstein(x0, lambda x,t: 0.2*x, lambda x,t: 0.7*x,
                            lambda x,t: torch.full_like(x,0.7))[:, -1]
    expected = x0*(1 + 0.2*dt + 0.7*dw + 0.5*0.7**2*(dw**2-scale**2*dt))
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)


def test_custom_milstein_uses_declared_variance_not_unit_dt():
    x0 = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    variance = torch.tensor([0.1, 0.4], dtype=torch.float64)
    solver = SDEIntegratorV2(SDEConfig(dt=0.1, steps=1, stability_clip=None),
                            noise_fn=lambda x,dt: torch.zeros_like(x),
                            noise_variance_fn=lambda x,dt: variance)
    result = solver.milstein(x0, lambda x,t: torch.zeros_like(x), lambda x,t: x,
                            lambda x,t: torch.ones_like(x))
    torch.testing.assert_close(result[:, -1], x0*(1-0.5*variance))


def test_milstein_variance_remains_finite_when_noise_scale_squared_overflows():
    dt, scale, seed = 1e-320, 1e160, 7
    x0 = torch.ones(2, 1, dtype=torch.float64)
    amplitude = dt**0.5 * scale
    dw = torch.randn(x0.shape, dtype=x0.dtype,
                     generator=torch.Generator().manual_seed(seed)) * amplitude
    solver = SDEIntegratorV2(SDEConfig(dt=dt, noise_scale=scale, steps=1, stability_clip=None),
                            generator=torch.Generator().manual_seed(seed))
    actual = solver.milstein(x0, lambda x,t:torch.zeros_like(x),lambda x,t:x,
                            lambda x,t:torch.ones_like(x))[:, -1]
    torch.testing.assert_close(actual, x0 + x0*dw + 0.5*x0*(dw**2-amplitude**2))


def test_custom_milstein_rejects_undeclared_quadratic_variation():
    solver = SDEIntegratorV2(SDEConfig(), noise_fn=lambda x,dt: torch.zeros_like(x))
    with pytest.raises(ValueError, match='noise_variance_fn'):
        solver.milstein(torch.ones(2,1), lambda x,t:x, lambda x,t:x, lambda x,t:x)


def test_zero_steps_preserves_initial_state_without_calling_fields():
    solver = SDEIntegratorV2(SDEConfig(steps=7))
    def forbidden(x,t):
        raise AssertionError('field evaluated for zero steps')
    x0=torch.ones(2,3)
    torch.testing.assert_close(solver.euler_maruyama(x0,forbidden,forbidden,steps=0), x0[:,None])


def test_zero_noise_does_not_consume_generator_state():
    generator = torch.Generator().manual_seed(8)
    before = generator.get_state().clone()
    solver = SDEIntegratorV2(SDEConfig(noise_scale=0, steps=2), generator=generator)
    solver.euler_maruyama(torch.ones(1,1),lambda x,t:x,lambda x,t:x)
    assert torch.equal(before, generator.get_state())


def test_seeded_euler_reproduces_historical_unit_noise_path():
    config=SDEConfig(dt=0.01,steps=3,stability_clip=None)
    x0=torch.ones(2,3,dtype=torch.float64)
    torch.manual_seed(98)
    previous=SDEIntegrator(config).euler_maruyama(x0,lambda x,t:-x,lambda x,t:0.2*x)
    current=SDEIntegratorV2(config,generator=torch.Generator().manual_seed(98))
    actual=current.euler_maruyama(x0,lambda x,t:-x,lambda x,t:0.2*x)
    torch.testing.assert_close(actual,previous,rtol=0,atol=0)


@pytest.mark.parametrize('changes', [{'dt':0},{'dt':float('nan')},{'steps':-1},
    {'steps':True},{'noise_scale':-1},{'noise_scale':float('inf')},{'stability_clip':0}])
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        SDEIntegratorV2(SDEConfig(**changes))


@pytest.mark.parametrize('variance', [-1.0,float('nan'),torch.ones(3,2)])
def test_invalid_custom_variance_is_rejected(variance):
    solver=SDEIntegratorV2(SDEConfig(steps=1),noise_fn=lambda x,dt:torch.zeros_like(x),
                           noise_variance_fn=lambda x,dt:variance)
    with pytest.raises(ValueError, match='variance'):
        solver.milstein(torch.ones(2,1),lambda x,t:x,lambda x,t:x,lambda x,t:x)


def test_shape_broadcast_cannot_mix_independent_trajectories():
    solver=SDEIntegratorV2(SDEConfig(steps=1))
    with pytest.raises(ValueError,match='drift must match'):
        solver.euler_maruyama(torch.ones(2,1),lambda x,t:torch.ones(2),lambda x,t:x)


def test_nonfinite_step_is_rejected_before_clipping_can_hide_it():
    solver=SDEIntegratorV2(SDEConfig(dt=2.0,steps=1,noise_scale=0))
    with pytest.raises(ValueError,match='nonfinite state'):
        solver.euler_maruyama(torch.full((1,1),1e308,dtype=torch.float64),
                              lambda x,t:x,lambda x,t:x)
