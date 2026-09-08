from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_trajectory_isolated_ablations.py"
SPEC = importlib.util.spec_from_file_location("trajectory_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.reset_count = 0

    def eval(self):
        super().eval()
        return self

    def reset_state(self):
        self.reset_count += 1

    def step(self, x, store_traces=True, retrieve=True):
        self.calls.append((bool(store_traces), bool(retrieve), int(x.shape[0])))
        return x


def test_protocol_rejects_multi_episode_batch():
    with pytest.raises(ValueError, match="batch_size=1"):
        runner.require_trajectory_isolation(2)


def test_protocol_accepts_exactly_one_episode():
    runner.require_trajectory_isolation(1)


def test_validation_uses_memory_within_single_trajectory():
    model = RecordingModel()
    x = torch.zeros(1, 2, 1, 1, 4)
    y = torch.zeros_like(x)
    loader = [(x, y)]
    metrics = runner._isolated_validate_from_loader(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        rollout_steps=2,
        horizon_decay=1.0,
    )
    assert metrics["loss"] == pytest.approx(0.0)
    assert model.reset_count == 1
    assert model.calls
    assert all(store and retrieve and batch == 1 for store, retrieve, batch in model.calls)


def test_validation_rejects_accidental_batch_two():
    model = RecordingModel()
    x = torch.zeros(2, 2, 1, 1, 4)
    y = torch.zeros_like(x)
    with pytest.raises(RuntimeError, match="more than one independent episode"):
        runner._isolated_validate_from_loader(
            model=model,
            loader=[(x, y)],
            device=torch.device("cpu"),
            rollout_steps=2,
            horizon_decay=1.0,
        )


def test_final_evaluator_is_forced_to_batch_one(monkeypatch):
    seen = {}

    def fake_run(*args, **kwargs):
        seen["batch_size"] = kwargs.get("batch_size")
        return {"rollout_mse": 0.0}

    monkeypatch.setattr(runner, "CANONICAL_EVAL_RUN", fake_run)
    model = RecordingModel()
    out = runner._isolated_eval_run(model, object(), batch_size=1)
    assert out["rollout_mse"] == 0.0
    assert seen["batch_size"] == 1


def test_final_evaluator_rejects_batch_two(monkeypatch):
    monkeypatch.setattr(runner, "CANONICAL_EVAL_RUN", lambda *a, **k: {})
    with pytest.raises(ValueError, match="batch_size=1"):
        runner._isolated_eval_run(RecordingModel(), object(), batch_size=2)
