import torch

from fim_s1.model import InterventionalTensorFabricPredictor


def make_model() -> InterventionalTensorFabricPredictor:
    return InterventionalTensorFabricPredictor(
        context_channels=3,
        output_channels=1,
        height=4,
        width=4,
        hidden_channels=8,
        key_channels=4,
        memory_capacity=8,
        topk=2,
        depth=2,
    )


def test_empty_memory_is_exact_base_path() -> None:
    model = make_model()
    context = torch.randn(2, 3, 4, 4)
    output = model(context, torch.tensor([0, 1]), torch.tensor([2, 2]))
    torch.testing.assert_close(output.prediction, output.base_prediction)
    assert not output.retrieval.valid.any()


def test_commit_and_erase_preserve_base_prediction() -> None:
    model = make_model()
    context = torch.randn(1, 3, 4, 4)
    first = model(context, torch.tensor([3]), torch.tensor([1]))
    model.commit_from_output(
        first,
        first.base_prediction + 2.0,
        torch.tensor([3]),
        torch.tensor([1]),
        regimes=torch.tensor([1]),
    )
    model.feedback_logit.data.fill_(2.0)
    ordinary = model(context, torch.tensor([3]), torch.tensor([2]))
    erased = model(
        context,
        torch.tensor([3]),
        torch.tensor([2]),
        intervention="erase",
    )
    assert ordinary.retrieval.valid.item()
    assert torch.count_nonzero(ordinary.retrieved) > 0
    torch.testing.assert_close(erased.prediction, erased.base_prediction)


def test_model_gradients_do_not_enter_stored_memory() -> None:
    model = make_model()
    context = torch.randn(1, 3, 4, 4)
    initial = model(context, torch.tensor([0]), torch.tensor([1]))
    model.commit_from_output(
        initial,
        torch.randn(1, 1, 4, 4),
        torch.tensor([0]),
        torch.tensor([1]),
    )
    model.feedback_logit.data.fill_(1.0)
    output = model(context, torch.tensor([0]), torch.tensor([2]))
    output.prediction.square().mean().backward()
    assert model.stem.weight.grad is not None
    assert not model.memory.keys.requires_grad
    assert not model.memory.values.requires_grad


def test_shape_and_owner_errors_are_rejected() -> None:
    model = make_model()
    try:
        model(torch.randn(1, 2, 4, 4), torch.tensor([0]), torch.tensor([1]))
    except ValueError as error:
        assert "context must have shape" in str(error)
    else:
        raise AssertionError("invalid context shape was accepted")


def test_bounded_xcorr_recovers_periodic_translation() -> None:
    model = InterventionalTensorFabricPredictor(
        context_channels=2,
        output_channels=1,
        height=6,
        width=6,
        hidden_channels=4,
        key_channels=2,
        memory_capacity=4,
        depth=1,
        transport_mode="bounded_xcorr",
        max_transport_shift=2,
    )
    previous = torch.randn(1, 1, 6, 6)
    current = torch.roll(previous, shifts=(1, -2), dims=(-2, -1))
    context = torch.cat([previous, current], dim=1)
    shifts = model.estimate_transport_shifts(context)
    assert shifts.tolist() == [[1, -2]]


def test_bounded_xcorr_prefers_identity_on_an_ambiguous_score_tie() -> None:
    model = InterventionalTensorFabricPredictor(
        context_channels=2,
        output_channels=1,
        height=8,
        width=8,
        hidden_channels=4,
        key_channels=2,
        depth=1,
        transport_mode="bounded_xcorr",
        max_transport_shift=2,
    )
    shifts = model.estimate_transport_shifts(torch.zeros(1, 2, 8, 8))
    assert shifts.tolist() == [[0, 0]]


def test_transport_aligns_retrieved_residual_and_replacements_are_post_transport() -> (
    None
):
    model = InterventionalTensorFabricPredictor(
        context_channels=2,
        output_channels=1,
        height=6,
        width=6,
        hidden_channels=4,
        key_channels=2,
        memory_capacity=4,
        topk=1,
        depth=1,
        transport_mode="bounded_xcorr",
        max_transport_shift=2,
    )
    previous = torch.randn(1, 1, 6, 6)
    current = torch.roll(previous, shifts=(1, -1), dims=(-2, -1))
    context = torch.cat([previous, current], dim=1)
    first = model(context, torch.tensor([2]), torch.tensor([0]))
    residual = torch.arange(36, dtype=torch.float32).reshape(1, 1, 6, 6)
    model.commit_from_output(
        first,
        first.base_prediction + residual,
        torch.tensor([2]),
        torch.tensor([0]),
    )
    ordinary = model(context, torch.tensor([2]), torch.tensor([1]))
    torch.testing.assert_close(
        ordinary.retrieved,
        torch.roll(residual, shifts=(1, -1), dims=(-2, -1)),
    )
    replacement = torch.full_like(residual, 7.0)
    swapped = model(
        context,
        torch.tensor([2]),
        torch.tensor([1]),
        intervention="value_swap",
        replacement=replacement,
    )
    torch.testing.assert_close(swapped.retrieved, replacement)


def test_transport_alignment_accounts_for_memory_age() -> None:
    model = InterventionalTensorFabricPredictor(
        context_channels=2,
        output_channels=1,
        height=7,
        width=7,
        hidden_channels=4,
        key_channels=2,
        memory_capacity=4,
        topk=1,
        depth=1,
        transport_mode="bounded_xcorr",
        max_transport_shift=2,
    )
    previous = torch.randn(1, 1, 7, 7)
    current = torch.roll(previous, shifts=(1, -1), dims=(-2, -1))
    context = torch.cat([previous, current], dim=1)
    written = model(context, torch.tensor([2]), torch.tensor([0]))
    residual = torch.arange(49, dtype=torch.float32).reshape(1, 1, 7, 7)
    model.commit_from_output(
        written,
        written.base_prediction + residual,
        torch.tensor([2]),
        torch.tensor([0]),
    )
    retrieved_two_steps_later = model(context, torch.tensor([2]), torch.tensor([2]))
    torch.testing.assert_close(
        retrieved_two_steps_later.retrieved,
        torch.roll(residual, shifts=(2, -2), dims=(-2, -1)),
    )
