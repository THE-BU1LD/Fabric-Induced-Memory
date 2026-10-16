"""Small implementation fixtures; no retained scientific checkpoint is read."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from fim_experiments import train as training


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fim_checkpoint_evaluation_test", ROOT / "scripts" / "evaluate_checkpoint.py"
)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
_import_path = list(sys.path)
try:
    SPEC.loader.exec_module(evaluator)
finally:
    # The legacy experiment entrypoint appends its import root. Do not leave
    # that collection-time path change behind for the standalone runner tests.
    sys.path[:] = _import_path


def _model() -> torch.nn.Module:
    model = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.2)
    return model


def _trained_checkpoint(tmp_path, *, use_ema=True, evaluate_ema=True):
    model = _model()
    x = torch.ones(1, 1, 2, 2)
    target = torch.full_like(x, 2.0)
    history = training.train(
        model,
        train_loader=[(x, target)],
        val_loader=[(x, target)],
        epochs=2,
        device="cpu",
        lr=0.1,
        weight_decay=0.0,
        use_amp=False,
        rollout_steps=1,
        teacher_forcing_ratio=0.0,
        use_ema=use_ema,
        evaluate_ema=evaluate_ema,
        ema_decay=0.5,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    path = tmp_path / "checkpoints" / "best_fim_checkpoint.pt"
    return path, model, history


def _run_cli(monkeypatch, checkpoint, output_dir, *, weights=None):
    argv = [
        "evaluate_checkpoint.py", "--checkpoint", str(checkpoint),
        "--config", str(output_dir / "fixture.yaml"),
        "--device", "cpu", "--rollout_steps", "1", "--batch_size", "1",
        "--output_dir", str(output_dir),
    ]
    if weights is not None:
        argv.extend(["--weights", weights])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(evaluator, "load_yaml_config", lambda _: {})
    monkeypatch.setattr(evaluator, "select_benchmark", lambda _: object())
    monkeypatch.setattr(evaluator, "build_system", lambda *args: _model())
    observed = {}

    def evaluate_fixture(model, benchmark, **kwargs):
        del benchmark, kwargs
        value = model(torch.ones(1, 1, 2, 2)).detach()
        observed["prediction"] = float(value.mean())
        observed["mse"] = float((value - 2.0).square().mean())
        return {"fixture_mse": observed["mse"]}

    monkeypatch.setattr(evaluator, "run", evaluate_fixture)
    evaluator.main()
    observed["provenance"] = json.loads(
        (output_dir / "evaluation_provenance.json").read_text()
    )
    return observed


def test_new_best_checkpoint_replays_the_weights_used_for_validation(tmp_path, monkeypatch):
    checkpoint, _, history = _trained_checkpoint(tmp_path)
    payload = torch.load(checkpoint, weights_only=True)
    assert not torch.equal(payload["model"]["weight"], payload["ema"]["weight"])
    observed = _run_cli(monkeypatch, checkpoint, tmp_path / "evaluation")
    assert observed["mse"] == pytest.approx(history["val_loss"][-1], abs=1e-7)
    assert payload["validation_weight_source"] == "ema"
    assert observed["provenance"]["weight_source"] == "ema"
    assert observed["provenance"]["requested_weight_source"] == "auto"


@pytest.mark.parametrize("use_ema,evaluate_ema", [(False, False), (False, True), (True, False)])
def test_raw_validation_replays_raw_weights(tmp_path, monkeypatch, use_ema, evaluate_ema):
    checkpoint, _, history = _trained_checkpoint(
        tmp_path, use_ema=use_ema, evaluate_ema=evaluate_ema,
    )
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["validation_weight_source"] == "raw"
    observed = _run_cli(monkeypatch, checkpoint, tmp_path / "evaluation")
    assert observed["mse"] == pytest.approx(history["val_loss"][-1], abs=1e-7)
    assert observed["provenance"]["weight_source"] == "raw"


@pytest.mark.parametrize("weights,expected_mse", [("raw", 2.890000104904175), ("ema", 3.0625)])
def test_explicit_cli_weight_selection_is_recorded(tmp_path, monkeypatch, weights, expected_mse):
    checkpoint, _, _ = _trained_checkpoint(tmp_path)
    observed = _run_cli(monkeypatch, checkpoint, tmp_path / "evaluation", weights=weights)
    assert observed["mse"] == pytest.approx(expected_mse, abs=1e-7)
    assert observed["provenance"]["weight_source"] == weights
    assert observed["provenance"]["requested_weight_source"] == weights


def test_legacy_checkpoint_stays_raw_even_when_ema_is_present(tmp_path, monkeypatch):
    checkpoint, _, _ = _trained_checkpoint(tmp_path)
    payload = torch.load(checkpoint, weights_only=True)
    del payload["validation_weight_source"]
    torch.save(payload, checkpoint)
    observed = _run_cli(monkeypatch, checkpoint, tmp_path / "evaluation")
    assert observed["mse"] == pytest.approx(2.890000104904175, abs=1e-7)
    assert observed["provenance"]["weight_source"] == "raw"
    assert "legacy" in observed["provenance"]["weight_source_reason"]


def test_final_state_checkpoint_uses_its_raw_model_state_dict():
    model = _model()
    state = {"weight": torch.full_like(model.weight, 0.75)}
    result = evaluator._load_evaluation_weights(model, {"model_state_dict": state})
    torch.testing.assert_close(model.weight, state["weight"], rtol=0, atol=0)
    assert result["weight_source"] == "raw"


@pytest.mark.parametrize("source", [None, True, 0, "averaged", []])
def test_invalid_or_missing_ema_never_falls_back_to_raw(source):
    model = _model()
    payload = {"model": deepcopy(model.state_dict()), "validation_weight_source": "ema", "ema": source}
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="no EMA mapping"):
        evaluator._load_evaluation_weights(model, payload)
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape", "dtype", "not_tensor", "non_string_key"])
def test_ema_parameter_admission_precedes_model_mutation(corruption):
    model = _model()
    raw = {"weight": torch.full_like(model.weight, 0.9)}
    ema = {"weight": torch.full_like(model.weight, 0.5)}
    if corruption == "missing":
        ema.clear()
    elif corruption == "unexpected":
        ema["not_a_parameter"] = torch.tensor(1.0)
    elif corruption == "shape":
        ema["weight"] = torch.ones(2, 2)
    elif corruption == "dtype":
        ema["weight"] = ema["weight"].double()
    elif corruption == "non_string_key":
        ema[0] = torch.tensor(1.0)
        ema["other"] = torch.tensor(1.0)
    else:
        ema["weight"] = [0.5]
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="EMA"):
        evaluator._load_evaluation_weights(model, {"model": raw, "ema": ema}, "ema")
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)


@pytest.mark.parametrize("declaration", [True, 0, "unknown", []])
def test_malformed_validation_metadata_is_rejected(declaration):
    model = _model()
    with pytest.raises(ValueError, match="validation_weight_source"):
        evaluator._load_evaluation_weights(
            model, {"model": model.state_dict(), "validation_weight_source": declaration},
        )


def test_missing_ema_cli_fails_before_any_evaluation_or_output(tmp_path, monkeypatch):
    checkpoint = tmp_path / "invalid.pt"
    torch.save({"model": _model().state_dict(), "validation_weight_source": "ema"}, checkpoint)
    output_dir = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="no EMA mapping"):
        _run_cli(monkeypatch, checkpoint, output_dir)
    assert not output_dir.exists()


def test_ema_does_not_replace_frozen_parameters_or_persistent_buffers():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    model[0].bias.requires_grad_(False)
    raw = deepcopy(model.state_dict())
    raw["0.bias"].fill_(3.0)
    raw["1.running_mean"].fill_(7.0)
    raw["1.num_batches_tracked"].fill_(19)
    ema = {name: torch.full_like(param, 0.5) for name, param in model.named_parameters() if param.requires_grad}
    evaluator._load_evaluation_weights(model, {"model": raw, "ema": ema}, "ema")
    assert torch.all(model[0].bias == 3.0)
    assert torch.all(model[1].running_mean == 7.0)
    assert int(model[1].num_batches_tracked) == 19
    for name, param in model.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param, ema[name], rtol=0, atol=0)


def test_tied_parameter_aliases_receive_the_same_ema_value():
    model = torch.nn.Module()
    model.left = torch.nn.Linear(2, 2, bias=False)
    model.right = model.left
    raw = deepcopy(model.state_dict())
    ema = {"left.weight": torch.full_like(model.left.weight, 0.5)}
    evaluator._load_evaluation_weights(model, {"model": raw, "ema": ema}, "ema")
    torch.testing.assert_close(model.left.weight, ema["left.weight"], rtol=0, atol=0)
    torch.testing.assert_close(model.right.weight, ema["left.weight"], rtol=0, atol=0)


def test_resume_retains_raw_optimizer_and_ema_state(tmp_path):
    checkpoint, trained_model, _ = _trained_checkpoint(tmp_path)
    payload = torch.load(checkpoint, weights_only=True)
    resumed = _model()
    optimizer = training.build_optimizer(resumed.parameters())
    ema = training.EMA(resumed, decay=0.5)
    state = training.load_checkpoint(checkpoint, resumed, optimizer=optimizer, ema=ema)
    assert state.epoch == 1
    torch.testing.assert_close(resumed.state_dict(), trained_model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict(), payload["optimizer"], rtol=0, atol=0)
    torch.testing.assert_close(ema.state_dict(), payload["ema"], rtol=0, atol=0)
    assert not torch.equal(resumed.weight, ema.shadow["weight"])


def test_save_checkpoint_rejects_ema_declaration_without_state(tmp_path):
    model = _model()
    path = tmp_path / "must_not_exist.pt"
    with pytest.raises(ValueError, match="EMA checkpoint state"):
        training.save_checkpoint(
            path, model, training.build_optimizer(model.parameters()), None,
            epoch=0, best_val_loss=1.0, history={}, validation_weight_source="ema",
        )
    assert not path.exists()


def test_checkpoint_without_validation_declares_no_weight_source(tmp_path):
    model = _model()
    x = torch.ones(1, 1, 2, 2)
    training.train(
        model, train_loader=[(x, 2.0 * x)], epochs=1, use_amp=False,
        checkpoint_dir=tmp_path, use_ema=True, rollout_steps=1,
    )
    payload = torch.load(tmp_path / "fim_checkpoint.pt", weights_only=True)
    assert payload["validation_weight_source"] is None
    result = evaluator._load_evaluation_weights(_model(), payload)
    assert result["weight_source"] == "raw"


def test_real_fim_system_checkpoint_matches_ema_parameter_application(tmp_path):
    # FIMSystem owns aliased bank modules. Exercise its actual state dictionary
    # and prediction path in addition to the scalar checkpoint fixtures.
    from fim_experiments.systems import FIMSystem

    def make_fim():
        return FIMSystem(
            in_channels=1, hidden=8, trace_dim=4, memory_capacity=8, retrieval_topk=2,
        ).eval()

    reference = make_fim()
    ema = training.EMA(reference, decay=0.5)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(0.01)
    ema.update(reference)
    x = torch.linspace(-1.0, 1.0, 8).reshape(1, 1, 1, 8)
    ema.apply(reference)
    with torch.no_grad():
        expected = reference.step(x, store_traces=False).prediction.clone()
    ema.restore(reference)

    checkpoint = tmp_path / "fim_fixture.pt"
    training.save_checkpoint(
        checkpoint, reference, training.build_optimizer(reference.parameters()), None,
        epoch=0, best_val_loss=0.0, history={}, ema=ema, validation_weight_source="ema",
    )
    restored = make_fim()
    payload = evaluator._load_checkpoint(checkpoint, torch.device("cpu"))
    result = evaluator._load_evaluation_weights(restored, payload)
    with torch.no_grad():
        actual = restored.step(x, store_traces=False).prediction
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert result["weight_source"] == "ema"
