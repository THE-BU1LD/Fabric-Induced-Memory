"""Analytic and artificial noise fixtures; no protected scientific outcomes."""
import math

import pytest
import torch

from fim.stochastic.colored_noise_v2 import ColoredNoiseV2
from fim.stochastic.noise import ColoredNoise


def _generator(seed=29):
    return torch.Generator().manual_seed(seed)


def test_legacy_witness_zero_field_receives_no_noise_or_rng_advance():
    x = torch.zeros(2, 8)
    before = torch.random.get_rng_state().clone()
    result = ColoredNoise()(x)
    assert torch.equal(result, x)
    assert torch.equal(before, torch.random.get_rng_state())


def test_legacy_witness_repeated_nonzero_input_is_deterministic():
    x = torch.arange(16, dtype=torch.float64).reshape(2, 8)
    layer = ColoredNoise()
    assert torch.equal(layer(x), layer(x))


def test_legacy_witness_one_site_normalization_is_nonfinite():
    with pytest.warns(UserWarning, match="degrees of freedom"):
        result = ColoredNoise()(torch.zeros(2, 1))
    assert not torch.isfinite(result).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_input_gets_finite_nonzero_noise_and_successive_draws_differ(dtype):
    x = torch.zeros(3, 16, dtype=dtype)
    layer = ColoredNoiseV2()
    generator = _generator()
    first, second = layer(x, generator=generator), layer(x, generator=generator)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert torch.count_nonzero(first) > 0
    assert not torch.equal(first, second)
    assert first.shape == x.shape and first.dtype == x.dtype


def test_identical_generator_states_replay_exactly_without_global_rng_use():
    x = torch.zeros(2, 5, 7, dtype=torch.float64)
    layer = ColoredNoiseV2(spatial_dims=2)
    before = torch.random.get_rng_state().clone()
    first = layer(x, generator=_generator())
    second = layer(x, generator=_generator())
    assert torch.equal(first, second)
    assert torch.equal(before, torch.random.get_rng_state())


def test_noise_is_independent_of_input_values_and_input_gradient_is_identity():
    zero = torch.zeros(2, 5, 7, dtype=torch.float64)
    x = torch.linspace(-1, 1, zero.numel(), dtype=zero.dtype).reshape_as(zero)
    x.requires_grad_()
    layer = ColoredNoiseV2(spatial_dims=2)
    actual = layer(x, generator=_generator())
    noise = layer(zero, generator=_generator())
    torch.testing.assert_close(actual - x, noise, rtol=1e-14, atol=1e-14)
    actual.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_beta_zero_is_exact_scaled_white_noise(normalize, dtype):
    x = torch.zeros(2, 9, dtype=dtype)
    expected = 0.5 * torch.randn(x.shape, dtype=dtype, generator=_generator())
    actual = ColoredNoiseV2(beta=0, sigma=0.5, normalize=normalize)(x, generator=_generator())
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape,spatial_dims", [((3, 1), 1), ((3, 1, 1), 2)])
def test_normalized_single_site_is_finite_unit_variance_gaussian(dtype, shape, spatial_dims):
    x = torch.zeros(shape, dtype=dtype)
    expected = torch.randn(shape, dtype=dtype, generator=_generator())
    actual = ColoredNoiseV2(spatial_dims=spatial_dims)(x, generator=_generator())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_normalized_four_site_covariance_matches_closed_form(monkeypatch):
    # Four independent basis fields expose the exact linear transform without
    # Monte Carlo thresholds. beta=2, eps=1/16 gives unnormalized spectral power
    # [16, 8, 16/5, 8], mean 44/5. Its normalized covariance has rows below.
    basis = torch.eye(4, dtype=torch.float64)
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: basis.clone())
    transformed = ColoredNoiseV2(beta=2, eps=1 / 16)(torch.zeros_like(basis))
    covariance = transformed.T @ transformed
    expected = torch.tensor([
        [1, 4/11, 1/11, 4/11],
        [4/11, 1, 4/11, 1/11],
        [1/11, 4/11, 1, 4/11],
        [4/11, 1/11, 4/11, 1],
    ], dtype=torch.float64)
    torch.testing.assert_close(covariance, expected, rtol=1e-13, atol=1e-13)


def test_raw_four_site_mode_has_analytic_amplitude(monkeypatch):
    mode = torch.tensor([[1.0, 0.0, -1.0, 0.0]], dtype=torch.float64)
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: mode.clone())
    actual = ColoredNoiseV2(beta=2, eps=1 / 16, normalize=False)(torch.zeros_like(mode))
    torch.testing.assert_close(actual, math.sqrt(8) * mode, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("shape", [(5, 7), (6, 8), (1, 7)])
def test_two_dimensional_filter_respects_analytic_mode_ratio(shape, monkeypatch):
    height, width = shape
    xx = torch.arange(width, dtype=torch.float64)[None, :]
    low = torch.cos(2 * math.pi * xx / width).expand(height, width)
    high = torch.cos(4 * math.pi * xx / width).expand(height, width)
    white = (low + high).unsqueeze(0)
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: white.clone())
    beta, eps = 1.5, 0.03
    actual = ColoredNoiseV2(beta=beta, eps=eps, spatial_dims=2)(torch.zeros_like(white))[0]
    low_coefficient = (actual * low).sum() / low.square().sum()
    high_coefficient = (actual * high).sum() / high.square().sum()
    expected_ratio = (((2 / width)**2 + eps) / ((1 / width)**2 + eps))**(-beta / 4)
    torch.testing.assert_close(high_coefficient / low_coefficient,
                               torch.tensor(expected_ratio, dtype=torch.float64),
                               rtol=1e-12, atol=1e-12)


def test_filter_does_not_mix_batch_or_channel_axes(monkeypatch):
    white = torch.zeros(2, 3, 5, 7, dtype=torch.float64)
    white[0, 1, 2, 3] = 1
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: white.clone())
    actual = ColoredNoiseV2(spatial_dims=2)(torch.zeros_like(white))
    assert torch.count_nonzero(actual[0, 1]) > 1
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.count_nonzero(actual[0, 0]) == 0
    assert torch.count_nonzero(actual[0, 2]) == 0


def test_zero_sigma_is_identity_and_preserves_random_state():
    x = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    generator = _generator()
    before = generator.get_state().clone()
    result = ColoredNoiseV2(sigma=0)(x, generator=generator)
    assert result is x
    assert torch.equal(before, generator.get_state())
    result.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("config", [
    {"beta": float("nan")}, {"beta": float("inf")}, {"beta": True},
    {"eps": 0}, {"eps": -1}, {"eps": float("inf")},
    {"sigma": -1}, {"sigma": float("nan")}, {"sigma": True},
    {"normalize": 1}, {"spatial_dims": 0}, {"spatial_dims": True},
    {"spatial_dims": 1.0},
])
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        ColoredNoiseV2(**config)


@pytest.mark.parametrize("x", [
    torch.ones(4, dtype=torch.int64), torch.ones(4, dtype=torch.complex64),
    torch.ones(4, dtype=torch.float16), torch.tensor(float("nan")),
    torch.tensor([float("inf")]), torch.tensor(1.0), torch.empty(0, 3),
])
def test_invalid_input_is_rejected_before_rng_use(x):
    generator = _generator()
    before = generator.get_state().clone()
    with pytest.raises(ValueError):
        ColoredNoiseV2()(x, generator=generator)
    assert torch.equal(before, generator.get_state())


def test_unrepresentable_filter_is_rejected_before_rng_use():
    generator = _generator()
    before = generator.get_state().clone()
    with pytest.raises(ValueError, match="eps is not representable"):
        ColoredNoiseV2(eps=1e-100)(torch.zeros(2, 8), generator=generator)
    assert torch.equal(before, generator.get_state())


def test_nonfinite_output_is_rejected(monkeypatch):
    monkeypatch.setattr(torch, "randn", lambda shape, **kwargs: torch.full(shape, 2.0))
    with pytest.raises(ValueError, match="nonfinite output"):
        ColoredNoiseV2(beta=0, sigma=3e38)(torch.zeros(1, 4))
