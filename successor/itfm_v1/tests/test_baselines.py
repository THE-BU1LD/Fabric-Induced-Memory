import torch

from fim_s1.baselines import (
    GatedDiagonalSSMPredictor,
    PeriodicTranslationPredictor,
    trainable_parameter_count,
)


def test_periodic_xcorr_exactly_extrapolates_periodic_translation() -> None:
    predictor = PeriodicTranslationPredictor(
        input_channels=1,
        context_length=3,
        max_shift=2,
    )
    first = torch.randn(2, 1, 8, 8)
    second = torch.roll(first, shifts=(1, -2), dims=(-2, -1))
    third = torch.roll(second, shifts=(1, -2), dims=(-2, -1))
    context = torch.cat([first, second, third], dim=1)
    prediction = predictor(context)
    expected = torch.roll(third, shifts=(1, -2), dims=(-2, -1))
    torch.testing.assert_close(prediction, expected)
    assert trainable_parameter_count(predictor) == 0


def test_gated_diagonal_ssm_is_finite_stable_and_differentiable() -> None:
    model = GatedDiagonalSSMPredictor(
        input_channels=1,
        output_channels=1,
        context_length=4,
        hidden_channels=6,
        depth=1,
    )
    context = torch.randn(3, 4, 8, 8)
    prediction = model(context)
    assert prediction.shape == (3, 1, 8, 8)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert model.decay_log.grad is not None
    assert torch.isfinite(model.decay_log.grad).all()
