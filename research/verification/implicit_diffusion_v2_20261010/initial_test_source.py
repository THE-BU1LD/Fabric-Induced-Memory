"""Bounded artificial numerical contracts; no retained outcomes or study runs."""

import math

import pytest
import torch

from fim.operators.diffusion import DiffusionOperator
from fim.operators.implicit_diffusion_v2 import (
    ImplicitDiffusionOperatorV2,
    periodic_backward_euler,
)


def dense_laplacian(h, w, *, dtype=torch.float64):
    """Coordinate construction independent of FFT frequencies and torch.roll."""
    matrix = torch.zeros(h * w, h * w, dtype=dtype)
    for y in range(h):
        for x in range(w):
            row = y * w + x
            matrix[row, row] -= 4
            for j, k in (((y - 1) % h, x), ((y + 1) % h, x), (y, (x - 1) % w), (y, (x + 1) % w)):
                matrix[row, j * w + k] += 1
    return matrix


def dense_solve(x, d, dt):
    h, w = x.shape[-2:]
    lap = dense_laplacian(h, w, dtype=x.dtype)
    identity = torch.eye(h * w, dtype=x.dtype)
    d = torch.as_tensor(d, dtype=x.dtype).reshape(-1).expand(x.shape[1])
    result = []
    for sample in x:
        channels = []
        for channel, coefficient in zip(sample, d):
            channels.append(torch.linalg.solve(identity - dt * coefficient * lap, channel.reshape(-1)).reshape(h, w))
        result.append(torch.stack(channels))
    return torch.stack(result)


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (4, 1), (2, 2), (3, 5), (4, 6)])
def test_fft_matches_independent_dense_solve_and_equation_residual(shape):
    h, w = shape
    x = torch.linspace(-1.5, 2.5, 2 * 3 * h * w, dtype=torch.float64).reshape(2, 3, h, w)
    d = torch.tensor([0., .2, 2.], dtype=torch.float64)
    expected = dense_solve(x, d, .7)
    actual = periodic_backward_euler(x, d, .7)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
    lap = dense_laplacian(h, w)
    residual = actual.reshape(2, 3, -1) - .7 * d.reshape(1, 3, 1) * (actual.reshape(2, 3, -1) @ lap.T)
    torch.testing.assert_close(residual.reshape_as(x), x, atol=5e-14, rtol=5e-14)


def test_constant_mode_is_preserved_and_checkerboard_is_damped():
    x = torch.full((1, 1, 2, 2), 2., dtype=torch.float64)
    torch.testing.assert_close(periodic_backward_euler(x, .5, 1.), x, atol=0, rtol=0)
    mode = torch.tensor([[[[1., -1.], [-1., 1.]]]], dtype=torch.float64)
    torch.testing.assert_close(periodic_backward_euler(mode, .5, 1.), mode / 5, atol=0, rtol=0)


def test_general_fourier_mode_has_derived_attenuation():
    h, w = 5, 7
    y = torch.arange(h, dtype=torch.float64).reshape(h, 1)
    x = torch.arange(w, dtype=torch.float64).reshape(1, w)
    mode = torch.cos(2 * math.pi * (2 * y / h + 3 * x / w)).reshape(1, 1, h, w)
    eigenvalue = 4 * math.sin(2 * math.pi / h) ** 2 + 4 * math.sin(3 * math.pi / w) ** 2
    expected = mode / (1 + 2.3 * .9 * eigenvalue)
    torch.testing.assert_close(periodic_backward_euler(mode, .9, 2.3), expected, atol=3e-15, rtol=3e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shape_dtype_mean_and_energy_preserved_as_required(dtype):
    generator = torch.Generator().manual_seed(17)
    x = torch.randn((2, 3, 7, 5), generator=generator, dtype=dtype).transpose(-1, -2)
    before = x.clone()
    y = periodic_backward_euler(x, torch.tensor([.01, .7, 3.], dtype=dtype), 5.)
    assert y.shape == x.shape and y.dtype == dtype
    torch.testing.assert_close(y.mean((-2, -1)), x.mean((-2, -1)))
    assert bool((y.square().sum((-2, -1)) <= x.square().sum((-2, -1))).all())
    assert torch.equal(x, before)
    assert y.data_ptr() != x.data_ptr()


def test_input_and_diffusivity_gradients_match_dense_solve():
    x = torch.linspace(-.7, 1.4, 30, dtype=torch.float64).reshape(1, 2, 3, 5).requires_grad_()
    d = torch.tensor([.13, .63], dtype=torch.float64, requires_grad=True)
    weight = torch.cos(torch.arange(x.numel(), dtype=torch.float64)).reshape_as(x)
    actual = periodic_backward_euler(x, d, .8)
    expected = dense_solve(x, d, .8)
    grad = torch.autograd.grad((actual * weight).sum(), (x, d))
    reference = torch.autograd.grad((expected * weight).sum(), (x, d))
    for value, oracle in zip(grad, reference):
        torch.testing.assert_close(value, oracle, atol=3e-14, rtol=3e-14)


def test_zero_dt_validates_then_returns_owned_identity_without_fft(monkeypatch):
    x = torch.full((1, 1, 3, 5), 3e38, requires_grad=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("zero dt must not compute an FFT")
    monkeypatch.setattr(torch.fft, "rfft2", forbidden)
    y = periodic_backward_euler(x, .3, 0.)
    assert torch.equal(x, y) and x.data_ptr() != y.data_ptr()
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    with pytest.raises(ValueError, match="diffusivity"):
        periodic_backward_euler(x, -1., 0.)


@pytest.mark.parametrize("bad_dt", [-1., float("inf"), float("nan"), True])
def test_reject_invalid_time_before_transform(bad_dt):
    with pytest.raises(ValueError, match="dt"):
        periodic_backward_euler(torch.ones(1, 1, 2, 3), .2, bad_dt)


@pytest.mark.parametrize("bad_d", [-.1, float("inf"), torch.tensor([.1, .2]), torch.ones(2, 2), torch.tensor(1)])
def test_reject_invalid_diffusivity_without_mutation(bad_d):
    x = torch.ones(1, 1, 2, 3)
    with pytest.raises(ValueError, match="diffusivity"):
        periodic_backward_euler(x, bad_d, 1.)
    assert torch.equal(x, torch.ones_like(x))


@pytest.mark.parametrize("x", [torch.ones(2, 3), torch.ones(1, 1, 0, 2), torch.ones(1, 1, 2, 3, dtype=torch.float16), torch.full((1, 1, 2, 3), float("nan"))])
def test_reject_invalid_field(x):
    with pytest.raises(ValueError, match="x"):
        periodic_backward_euler(x, .2, 1.)


def test_reject_unrepresentable_control_product():
    with pytest.raises(ValueError, match="representable"):
        periodic_backward_euler(torch.ones(1, 1, 2, 3), 3e38, 2.)


@pytest.mark.parametrize("learnable", [False, True])
def test_state_dict_compatibility_and_postprocessing(learnable):
    legacy = DiffusionOperator(2, diffusivity=.3, dt=.7, mode="implicit", learnable=learnable).double()
    candidate = ImplicitDiffusionOperatorV2(2, diffusivity=.3, dt=.7, learnable=learnable).double()
    candidate.load_state_dict(legacy.state_dict(), strict=True)
    assert set(candidate.state_dict()) == set(legacy.state_dict())
    x = torch.linspace(-1, 1, 30, dtype=torch.float64).reshape(1, 2, 3, 5)
    expected = .5 * (dense_solve(x, candidate.get_diff(), .7) - x)
    torch.testing.assert_close(candidate(x), expected, atol=1e-14, rtol=1e-14)
    candidate(x).square().sum().backward()
    assert candidate.residual_scale.grad is not None
    if learnable:
        assert candidate.raw_diff.grad is not None and bool(torch.isfinite(candidate.raw_diff.grad).all())


def test_opt_in_module_rejects_unsupported_mode_and_bad_channel_state():
    model = ImplicitDiffusionOperatorV2(2)
    with pytest.raises(ValueError, match="channel"):
        model(torch.ones(1, 1, 2, 3))
    model.mode = "spectral"
    with pytest.raises(ValueError, match="isotropic"):
        model(torch.ones(1, 2, 2, 3))


def test_constructor_rejects_invalid_or_unrepresentable_configuration():
    for kwargs in ({"channels": True}, {"channels": 1.2}, {"channels": 0}, {"channels": 1, "diffusivity": -1}, {"channels": 1, "diffusivity": 1000}, {"channels": 1, "eps": 0}):
        with pytest.raises(ValueError):
            ImplicitDiffusionOperatorV2(**kwargs)
