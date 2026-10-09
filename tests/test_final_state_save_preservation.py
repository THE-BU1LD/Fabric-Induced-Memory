"""Exercise final-state orchestration with training/evaluation fully stubbed."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


_import_path = list(sys.path)
try:
    from fim_experiments import main as experiment
finally:
    sys.path[:] = _import_path


@pytest.fixture
def artificial_experiment(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.5)
    config = {
        "experiment": {"name": "artificial_save_fixture"},
        "runtime": {"device": "cpu", "seed": 11, "results_root": str(tmp_path)},
        "model": {"name": "fixture"},
        "benchmark": {"name": "fixture"},
        "train": {"dynamic_data": True},
        "eval": {"enabled": False},
    }
    history = {"fixture_only": [1.0]}
    monkeypatch.setattr(experiment, "select_benchmark", lambda _: SimpleNamespace(metrics=lambda: {"fixture_only": True}))
    monkeypatch.setattr(experiment, "build_system", lambda *arguments: model)
    monkeypatch.setattr(experiment, "train", lambda *arguments, **kwargs: history)
    monkeypatch.setattr(experiment, "setup_logging", lambda _: SimpleNamespace(info=lambda *arguments: None))

    def forbidden(*arguments, **kwargs):
        raise AssertionError("Artificial final-state fixture must not generate data or evaluate a model")

    monkeypatch.setattr(experiment, "build_loader_static", forbidden)
    monkeypatch.setattr(experiment, "run", forbidden)
    path = tmp_path / "artificial_save_fixture" / "fixture_fixture" / "checkpoints" / "final_state.pt"
    return config, model, history, path


@pytest.mark.parametrize("existed", [False, True])
@pytest.mark.parametrize("exception", [OSError, KeyboardInterrupt, SystemExit])
def test_failed_final_state_save_keeps_prior_checkpoint(artificial_experiment, monkeypatch, existed, exception):
    config, model, _history, path = artificial_experiment
    original = None
    if existed:
        experiment.run_experiment(config)
        original = path.read_bytes()
    with torch.no_grad():
        model.weight.fill_(0.9)

    def interrupted(_payload, destination):
        if hasattr(destination, "write"):
            destination.write(b"partial final-state fixture")
        else:
            Path(destination).write_bytes(b"partial final-state fixture")
        raise exception("artificial final-state serialization failure")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(exception, match="artificial final-state serialization"):
        experiment.run_experiment(config)
    if existed:
        assert path.read_bytes() == original
        payload = torch.load(path, weights_only=True)
        restored = torch.nn.Linear(1, 1, bias=False)
        restored.load_state_dict(payload["model_state_dict"], strict=True)
        torch.testing.assert_close(restored.weight, torch.tensor([[0.5]]))
    else:
        assert not path.exists()
    assert sorted(item.name for item in path.parent.iterdir()) == ([path.name] if existed else [])


def test_final_state_preserves_the_original_payload_schema(artificial_experiment):
    config, model, history, path = artificial_experiment
    metrics = experiment.run_experiment(config)
    payload = torch.load(path, weights_only=True)
    assert set(payload) == {"model_state_dict", "config", "history", "metrics"}
    assert payload["config"] == config
    assert payload["history"] == history
    assert payload["metrics"] == metrics
    torch.testing.assert_close(payload["model_state_dict"], model.state_dict(), rtol=0, atol=0)
    with torch.no_grad():
        model.weight.fill_(0.75)
    experiment.run_experiment(config)
    replacement = torch.load(path, weights_only=True)
    assert set(replacement) == set(payload)
    torch.testing.assert_close(replacement["model_state_dict"]["weight"], torch.tensor([[0.75]]))
    assert sorted(item.name for item in path.parent.iterdir()) == [path.name]
