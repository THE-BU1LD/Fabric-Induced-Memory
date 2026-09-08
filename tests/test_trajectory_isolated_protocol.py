from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


def frozen_args(**overrides):
    values = dict(
        benchmarks=list(runner.FROZEN_BENCHMARKS),
        seeds=list(runner.FROZEN_SEEDS),
        variants=list(runner.FROZEN_VARIANTS),
        epochs=runner.FROZEN_EPOCHS,
        batch_size=runner.FROZEN_BATCH_SIZE,
        dataset_size=runner.FROZEN_DATASET_SIZE,
        rollout_steps=runner.FROZEN_ROLLOUT_STEPS,
        eval_steps=runner.FROZEN_EVAL_STEPS,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_protocol_rejects_multi_episode_batch():
    with pytest.raises(ValueError, match="batch_size=1"):
        runner.require_trajectory_isolation(2)


def test_protocol_accepts_exactly_one_episode():
    runner.require_trajectory_isolation(1)


def test_frozen_protocol_exact_budget_is_accepted():
    runner.validate_frozen_args(frozen_args())


def test_frozen_protocol_rejects_post_freeze_horizon_change():
    with pytest.raises(ValueError, match="protocol drift"):
        runner.validate_frozen_args(frozen_args(rollout_steps=4))


def test_delayed_recall_training_horizon_crosses_frozen_delay():
    assert runner.FROZEN_ROLLOUT_STEPS >= 8


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


def _run_actual_episode(model, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    model.reset_state()
    assert len(model.bank) == 0
    model.step(first, store_traces=True, retrieve=True)
    out = model.step(second, store_traces=True, retrieve=True)
    assert first.shape[0] == second.shape[0] == 1
    return out.prediction.detach().clone()


def test_actual_fim_episode_result_is_invariant_to_unrelated_episode_order():
    torch.manual_seed(991)
    model = runner.AblatedFIMSystem(
        in_channels=1,
        hidden=8,
        trace_dim=4,
        memory_capacity=8,
        retrieval_topk=2,
        memory_decay=0.0,
        salience_threshold=-1.0,
        memory_enabled=True,
        retrieval_enabled=True,
        salience_gating_enabled=True,
    ).eval()

    a0 = torch.tensor([[0.2, -0.4, 0.6, 0.1]], dtype=torch.float32)
    a1 = torch.tensor([[0.3, -0.1, 0.5, 0.0]], dtype=torch.float32)
    b0 = torch.tensor([[-0.7, 0.9, -0.2, 0.4]], dtype=torch.float32)
    b1 = torch.tensor([[-0.5, 0.8, -0.3, 0.6]], dtype=torch.float32)

    a_when_first = _run_actual_episode(model, a0, a1)
    _run_actual_episode(model, b0, b1)
    a_when_second = _run_actual_episode(model, a0, a1)

    torch.testing.assert_close(a_when_first, a_when_second, rtol=0.0, atol=0.0)


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
