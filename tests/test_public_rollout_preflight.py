"""Public rollout admission must precede any stateful model call."""

import copy
from types import SimpleNamespace

import pytest
import torch

from fim.models.fim_model import FIMModel
from fim.training.rollout import autoregressive_rollout


class StatefulStepModel:
    def __init__(self, tuple_output=True):
        self.calls = []
        self.writes = 0
        self.tuple_output = tuple_output

    def step(self, x, *, control=None, update_memory=False):
        self.calls.append((x.clone(), None if control is None else control.clone()))
        self.writes += int(update_memory)
        prediction = x + (1.0 if control is None else control)
        output = SimpleNamespace(step=len(self.calls), prediction=prediction)
        return (prediction, output) if self.tuple_output else prediction


@pytest.mark.parametrize("sequence", ["control_sequence", "teacher_forcing"])
def test_short_sequence_is_rejected_before_any_state_change(sequence):
    model = StatefulStepModel()
    with pytest.raises(ValueError, match=sequence):
        autoregressive_rollout(
            model, torch.zeros(2, 3), 3, update_memory=True,
            **{sequence: torch.zeros(2, 1, 3)},
        )
    assert model.calls == []
    assert model.writes == 0


@pytest.mark.parametrize("sequence", ["control_sequence", "teacher_forcing"])
@pytest.mark.parametrize("kind", ["batch", "rank", "not_tensor", "nan", "inf"])
def test_invalid_sequence_is_rejected_without_entering_model(sequence, kind):
    values = {
        "batch": torch.zeros(1, 3, 3),
        "rank": torch.zeros(3),
        "not_tensor": [[[0.0]]],
        "nan": torch.tensor([[[0.0] * 3] * 2 + [[float("nan")] * 3]] * 2),
        "inf": torch.tensor([[[0.0] * 3] * 2 + [[float("inf")] * 3]] * 2),
    }
    model = StatefulStepModel()
    with pytest.raises((TypeError, ValueError), match=sequence):
        autoregressive_rollout(
            model, torch.zeros(2, 3), 3, update_memory=True,
            **{sequence: values[kind]},
        )
    assert model.calls == []
    assert model.writes == 0


@pytest.mark.parametrize("shape", [(2, 3, 1), (2, 3, 1, 3)])
def test_teacher_forcing_feature_shape_is_checked_before_first_step(shape):
    model = StatefulStepModel()
    with pytest.raises(ValueError, match="teacher_forcing"):
        autoregressive_rollout(
            model, torch.zeros(2, 3), 3, teacher_forcing=torch.zeros(shape),
        )
    assert model.calls == []


def test_teacher_forcing_dtype_is_not_silently_cast_between_steps():
    model = StatefulStepModel()
    with pytest.raises(ValueError, match="teacher_forcing"):
        autoregressive_rollout(
            model, torch.zeros(2, 3), 3,
            teacher_forcing=torch.zeros(2, 3, 3, dtype=torch.float64),
        )
    assert model.calls == []


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_invalid_step_count_is_rejected_before_execution(steps):
    model = StatefulStepModel()
    with pytest.raises((TypeError, ValueError), match="steps"):
        autoregressive_rollout(model, torch.zeros(2, 3), steps)
    assert model.calls == []


@pytest.mark.parametrize("x0", [None, torch.tensor(1.0), torch.empty(0, 3), torch.empty(2, 0), torch.full((2, 3), float("nan"))])
def test_invalid_initial_tensor_does_not_enter_model(x0):
    model = StatefulStepModel()
    with pytest.raises((TypeError, ValueError), match="x0"):
        autoregressive_rollout(model, x0, 2)
    assert model.calls == []


@pytest.mark.parametrize("tuple_output", [False, True])
@pytest.mark.parametrize("return_outputs", [False, True])
def test_valid_autoregression_and_output_contract(tuple_output, return_outputs):
    model = StatefulStepModel(tuple_output=tuple_output)
    initial = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    predictions, outputs = autoregressive_rollout(
        model, initial, 3, return_outputs=return_outputs, update_memory=True,
    )
    expected = torch.tensor([[[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]], dtype=torch.float64)
    torch.testing.assert_close(predictions, expected, rtol=0, atol=0)
    assert model.writes == 3
    assert not predictions.requires_grad
    if return_outputs:
        assert len(outputs) == 3
        if tuple_output:
            assert [value.step for value in outputs] == [1, 2, 3]
        else:
            assert outputs == [None, None, None]
    else:
        assert outputs is None
    torch.testing.assert_close(initial, torch.tensor([[1.0, 2.0]], dtype=torch.float64))


def test_teacher_forcing_control_order_and_unused_suffix_are_preserved():
    model = StatefulStepModel()
    initial = torch.tensor([[1.0, 2.0]])
    teacher = torch.tensor([[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [float("nan"), 0.0]]])
    control = torch.tensor([[[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [float("nan"), 0.0]]])
    predictions, _ = autoregressive_rollout(
        model, initial, 3, teacher_forcing=teacher, control_sequence=control,
    )
    expected = torch.tensor([[[3.0, 5.0], [14.0, 25.0], [36.0, 47.0]]])
    torch.testing.assert_close(predictions, expected, rtol=0, atol=0)
    assert model.writes == 0


def test_real_fim_matches_manual_admitted_recurrence():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        model = FIMModel(
            in_channels=1, out_channels=1, latent_channels=8, trace_dim=4,
            hidden_channels=8, num_layers=1, max_traces=8,
        ).eval()
        manual = copy.deepcopy(model)
        initial = torch.randn(1, 1, 4, 4)
        predictions, outputs = autoregressive_rollout(model, initial, 3, update_memory=True)
        current, expected = initial, []
        with torch.no_grad():
            for _ in range(3):
                current, _ = manual.step(current, control=None, update_memory=True)
                expected.append(current)
        torch.testing.assert_close(predictions, torch.stack(expected, dim=1), rtol=0, atol=0)
        assert len(outputs) == 3
        assert model.step_index == manual.step_index == 3
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, manual.state_dict()[name], rtol=0, atol=0)
