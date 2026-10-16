import pytest
import torch

from fim_s1.model_nd import InterventionalNDTensorFabricPredictor


@pytest.mark.parametrize("spatial_shape", [(9,), (3, 4, 5)])
def test_nd_predictor_forward_commit_and_retrieve(spatial_shape) -> None:
    model = InterventionalNDTensorFabricPredictor(
        context_channels=2,
        output_channels=1,
        spatial_shape=spatial_shape,
        hidden_channels=4,
        key_channels=2,
        memory_capacity=8,
        topk=2,
        depth=1,
    )
    context = torch.randn(1, 2, *spatial_shape)
    first = model(context, torch.tensor([3]), torch.tensor([0]))
    assert first.prediction.shape == (1, 1, *spatial_shape)
    assert not first.retrieval.valid.item()

    target = torch.randn(1, 1, *spatial_shape)
    model.commit_from_output(first, target, torch.tensor([3]), torch.tensor([0]))
    second = model(context, torch.tensor([3]), torch.tensor([1]))
    assert second.retrieval.valid.item()
    second.prediction.square().mean().backward()
    assert model.feedback_logit.grad is not None


def test_nd_predictor_rejects_unsupported_rank() -> None:
    with pytest.raises(ValueError, match="rank must be 1, 2, or 3"):
        InterventionalNDTensorFabricPredictor(
            context_channels=1,
            output_channels=1,
            spatial_shape=(2, 2, 2, 2),
        )


@pytest.mark.parametrize(
    ("spatial_shape", "shift"),
    [((9,), (2,)), ((5, 6, 7), (1, -1, 2))],
)
def test_nd_transport_recovers_rank_generic_periodic_translation(
    spatial_shape, shift
) -> None:
    previous = torch.randn(1, 1, *spatial_shape)
    current = torch.roll(
        previous,
        shifts=shift,
        dims=tuple(range(-len(spatial_shape), 0)),
    )
    model = InterventionalNDTensorFabricPredictor(
        context_channels=2,
        output_channels=1,
        spatial_shape=spatial_shape,
        hidden_channels=4,
        key_channels=2,
        depth=1,
        transport_mode="bounded_xcorr",
        max_transport_shift=2,
    )
    observed = model.estimate_transport_shifts(torch.cat([previous, current], dim=1))
    assert tuple(observed[0].tolist()) == shift
