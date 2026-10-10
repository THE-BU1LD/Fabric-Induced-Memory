import math

import pytest
import torch

from fim.analysis import retention as retained
from fim.analysis.retention_v2 import (
    decay_curve, estimate_exponential_decay, estimate_power_law_decay, retention_auc,
)


def t(values):
    return torch.tensor(values, dtype=torch.float64)


def test_retained_large_time_origin_loses_area_and_decay_witness():
    times = 1e9 + t([0, 1, 2])
    response = t([1, 0.5, 0.25])
    assert retained.retention_auc(times, response) == 0.0
    assert retained.estimate_exponential_decay(times, response) == 0.0
    assert retention_auc(times, response) == pytest.approx(1.125, abs=0)
    assert estimate_exponential_decay(times, response) == pytest.approx(math.log(2), rel=1e-14)


@pytest.mark.parametrize("spacing", [1e-200, 1e-9, 1.0, 1e9, 1e200])
def test_decay_rate_preserves_time_units_without_variance_floor(spacing):
    times = t([0, 1, 2, 3]) * spacing
    response = torch.exp(-0.5 * t([0, 1, 2, 3]))
    assert estimate_exponential_decay(times, response) == pytest.approx(0.5 / spacing, rel=1e-13, abs=0)


@pytest.mark.parametrize("amplitude", [1e-200, 1.0, 1e200])
def test_exponential_amplitude_does_not_change_the_rate(amplitude):
    times = t([0, 1, 3, 5])
    response = amplitude * torch.exp(-0.7 * times)
    assert estimate_exponential_decay(times, response) == pytest.approx(0.7, rel=1e-12)


@pytest.mark.parametrize("exponent", [-1.25, 0.0, 1.5])
def test_power_law_matches_analytic_irregular_times(exponent):
    times = t([0.5, 1, 3, 10, 100])
    response = 3 * times.pow(-exponent)
    assert estimate_power_law_decay(times, response) == pytest.approx(exponent, rel=1e-12, abs=1e-14)


def test_power_law_handles_unrepresentable_direct_time_ratio():
    times = t([1e-300, 1.0, 1e300])
    response = torch.exp(-0.1 * torch.log(times))
    assert estimate_power_law_decay(times, response) == pytest.approx(0.1, rel=1e-13)


def test_growth_is_reported_as_negative_decay():
    times = t([0, 1, 2])
    assert estimate_exponential_decay(times, torch.exp(0.2 * times)) == pytest.approx(-0.2)


def test_signed_normalization_and_small_scale_are_not_clipped():
    response = t([-1e-200, -5e-201, 0, 1e-200])
    torch.testing.assert_close(decay_curve(response), t([1, 0.5, 0, -1]), rtol=0, atol=0)
    assert decay_curve(torch.tensor([2., 1.], dtype=torch.float32)).dtype == torch.float64


def test_auc_agrees_with_hand_trapezoids_and_keeps_response_units():
    times, response = t([0, 0.5, 2, 5]), t([4, 2, -2, 1])
    expected = 0.5 * (4 + 2) / 2 + 1.5 * (2 - 2) / 2 + 3 * (-2 + 1) / 2
    assert retention_auc(times, response) == pytest.approx(expected, abs=1e-15)
    assert retention_auc(times + 1e9, response) == pytest.approx(expected, abs=1e-15)


def test_auc_midpoint_does_not_overflow_for_finite_representable_integral():
    assert retention_auc(t([0, 0.25]), t([1e308, 1e308])) == pytest.approx(2.5e307, rel=1e-15)


@pytest.mark.parametrize("function", [estimate_exponential_decay, estimate_power_law_decay, retention_auc])
@pytest.mark.parametrize("times,response", [
    (t([]), t([])), (t([1]), t([1])), (t([1, 1]), t([1, 2])),
    (t([2, 1]), t([1, 2])), (t([1, 2]), t([1, float("nan")])),
    (t([1, float("inf")]), t([1, 2])), (t([1, 2]), t([1, 2, 3])),
    (t([[1, 2]]), t([[1, 2]])), (torch.tensor([1, 2]), t([1, 2])),
])
def test_ambiguous_observations_rejected_without_mutation(function, times, response):
    before_t, before_y = times.clone(), response.clone()
    with pytest.raises(ValueError):
        function(times, response)
    torch.testing.assert_close(times, before_t, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(response, before_y, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("function", [estimate_exponential_decay, estimate_power_law_decay])
@pytest.mark.parametrize("response", [t([1, 0]), t([1, -1])])
def test_log_fits_reject_zero_or_negative_observations(function, response):
    with pytest.raises(ValueError):
        function(t([1, 2]), response)


@pytest.mark.parametrize("response", [t([]), t([0, 1]), t([1, float("nan")]), t([[1, 2]])])
def test_invalid_normalization_rejected(response):
    with pytest.raises(ValueError):
        decay_curve(response)


def test_power_law_requires_a_positive_time_origin():
    with pytest.raises(ValueError):
        estimate_power_law_decay(t([0, 1]), t([1, 0.5]))


def test_unrepresentable_auc_rejected():
    with pytest.raises(ValueError, match="not representable"):
        retention_auc(t([0, 10]), t([1e308, 1e308]))
