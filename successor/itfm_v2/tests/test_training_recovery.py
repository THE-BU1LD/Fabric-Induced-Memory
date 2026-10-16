"""Numerical recovery checks for an admitted CPU/CUDA host; never run on import."""

from __future__ import annotations

import dataclasses
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from fim_sota.config import ConfigurationError, TrainingConfig
from fim_sota.contracts import ForecastBatch
from fim_sota.training import UnifiedTrainer, _atomic_torch_save


def batches(sizes: tuple[int, ...], split: str) -> list[ForecastBatch]:
    result = []
    offset = 0
    for n in sizes:
        x = (
            torch.arange(offset + 1, offset + n + 1, dtype=torch.float32).reshape(n, 1)
            / 10
        )
        result.append(
            ForecastBatch(
                context=x,
                target=2 * x + 0.1,
                observation_mask=torch.ones_like(x),
                trajectory_ids=tuple(f"{split}-{i}" for i in range(offset, offset + n)),
                event_times=torch.arange(offset, offset + n),
                lead_times=torch.ones(n),
                split=split,
            )
        )
        offset += n
    return result


def loss(model: nn.Module, batch: ForecastBatch) -> torch.Tensor:
    return (model(batch.context) - batch.target).square().mean()


def trainer(
    root: Path,
    name: str,
    *,
    model: nn.Module | None = None,
    config: TrainingConfig | None = None,
    scheduler: bool = False,
    admission=None,
) -> UnifiedTrainer:
    torch.manual_seed(13)
    model = nn.Linear(1, 1) if model is None else model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    schedule = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        if scheduler
        else None
    )
    return UnifiedTrainer(
        model=model,
        optimizer=optimizer,
        config=config
        or TrainingConfig(
            gate="full",
            epochs=3,
            gradient_accumulation=2,
            gradient_clip_norm=1e6,
            checkpoint_every_steps=2,
        ),
        run_id=name,
        output_root=root,
        source_sha256="a" * 64,
        config_sha256="b" * 64,
        data_sha256="c" * 64,
        resource_admission=admission or (lambda: {"status": "PASS"}),
        device="cpu",
        scheduler=schedule,
    )


def assert_nested_equal(left, right) -> None:
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


class StatefulRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.dropout = nn.Dropout(0.2)
        self.register_buffer("calls", torch.zeros((), dtype=torch.int64))
        self.visits = 0

    def forward(self, x):
        self.calls.add_(1)
        self.visits += 1
        noise = 0.001 * (random.random() + float(np.random.rand()) + torch.rand(()))
        return self.linear(self.dropout(x)) + noise + 1e-5 * self.visits

    def get_extra_state(self):
        return {"visits": self.visits}

    def set_extra_state(self, state):
        self.visits = state["visits"]

    def reset_evaluation_state(self):
        self.visits = 0


def test_mid_epoch_resume_matches_uninterrupted_state_rng_metrics_and_scheduler(
    tmp_path,
):
    train, validation = batches((3, 1, 2, 1, 2), "train"), batches((2, 1), "validation")
    torch.manual_seed(42)
    full_model = StatefulRegressor()
    initial = {
        key: value.clone() if torch.is_tensor(value) else dict(value)
        for key, value in full_model.state_dict().items()
    }
    full = trainer(tmp_path, "full", model=full_model, scheduler=True)
    full_result = full.fit(
        train_batches=train, validation_batches=validation, step=loss
    )
    interrupted_model = StatefulRegressor()
    interrupted_model.load_state_dict(initial)
    interrupted = trainer(
        tmp_path, "interrupted", model=interrupted_model, scheduler=True
    )
    original_log = interrupted._log

    def stop_after_first(event, **fields):
        original_log(event, **fields)
        if event == "train_step":
            interrupted._interrupted = True

    interrupted._log = stop_after_first
    stopped = interrupted.fit(
        train_batches=train, validation_batches=validation, step=loss
    )
    assert stopped.status == "PREEMPTED"
    payload = torch.load(stopped.checkpoint, weights_only=True)
    assert payload["progress"]["next_batch_index"] == 2
    assert payload["progress"]["examples_seen"] == 4
    assert isinstance(payload["random_state"]["numpy"]["keys"], list)
    resumed = trainer(tmp_path, "resumed", model=StatefulRegressor(), scheduler=True)
    resumed_result = resumed.fit(
        train_batches=train,
        validation_batches=validation,
        step=loss,
        resume_from=stopped.checkpoint,
    )
    assert full_result.status == resumed_result.status == "PASS_ENGINEERING"
    assert full_result.best_validation_loss == resumed_result.best_validation_loss
    assert full_result.steps_completed == resumed_result.steps_completed == 9
    assert full_result.examples_seen == resumed_result.examples_seen == 27
    assert_nested_equal(full.model.state_dict(), resumed.model.state_dict())
    assert_nested_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_nested_equal(full.scheduler.state_dict(), resumed.scheduler.state_dict())
    a, b = [
        torch.load(r.checkpoint, weights_only=True)
        for r in (full_result, resumed_result)
    ]
    assert_nested_equal(a["random_state"], b["random_state"])
    assert_nested_equal(
        a["best_snapshot"]["model_state"], b["best_snapshot"]["model_state"]
    )
    assert Path(resumed_result.checkpoint).name == "latest.pt"
    assert (tmp_path / "resumed/checkpoints/best.pt").is_file()


def test_unequal_microbatches_and_partial_window_are_sample_weighted(tmp_path):
    config = TrainingConfig(
        gate="full", epochs=1, gradient_accumulation=2, gradient_clip_norm=1e6
    )
    worker = trainer(tmp_path, "weighted", config=config)
    reference = nn.Linear(1, 1)
    reference.load_state_dict(worker.model.state_dict())
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
    train = batches((3, 1, 2), "train")
    for window in (train[:2], train[2:]):
        optimizer.zero_grad()
        x = torch.cat([b.context for b in window])
        target = torch.cat([b.target for b in window])
        (reference(x) - target).square().mean().backward()
        optimizer.step()
    result = worker.fit(
        train_batches=train, validation_batches=batches((1,), "validation"), step=loss
    )
    for actual, expected in zip(worker.model.parameters(), reference.parameters()):
        assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert result.steps_completed == 2 and result.examples_seen == 6
    events = [
        json.loads(line)
        for line in (tmp_path / "weighted/events.jsonl").read_text().splitlines()
    ]
    assert [
        e["effective_batch_size"] for e in events if e["event"] == "train_step"
    ] == [4, 2]


def test_validation_sample_weight_and_mutable_state_isolation(tmp_path):
    model = StatefulRegressor()
    worker = trainer(tmp_path, "validation", model=model)
    model.visits = 17
    data = batches((3, 1), "validation")
    metric = worker._validate(data, lambda m, b: b.target.square().mean())
    expected = torch.cat([b.target for b in data]).square().mean().item()
    assert metric == pytest.approx(expected)
    assert model.visits == 17
    assert model.training


def test_constructor_has_no_filesystem_side_effect_and_run_ids_cannot_overwrite(
    tmp_path,
):
    worker = trainer(tmp_path, "unique")
    assert not (tmp_path / "unique").exists()
    train, validation = batches((1,), "train"), batches((1,), "validation")
    worker.fit(train_batches=train, validation_batches=validation, step=loss)
    with pytest.raises(FileExistsError):
        trainer(tmp_path, "unique").fit(
            train_batches=train, validation_batches=validation, step=loss
        )


def test_data_or_source_mismatch_rejects_resume(tmp_path):
    train, validation = batches((1, 1), "train"), batches((1,), "validation")
    first = trainer(tmp_path, "first")
    result = first.fit(train_batches=train, validation_batches=validation, step=loss)
    changed = list(train)
    changed[0] = dataclasses.replace(changed[0], context=changed[0].context + 1)
    with pytest.raises(ValueError, match="batch_sha256 identity mismatch"):
        trainer(tmp_path, "bad-data").fit(
            train_batches=changed,
            validation_batches=validation,
            step=loss,
            resume_from=result.checkpoint,
        )
    bad_source = trainer(tmp_path, "bad-source")
    bad_source.source_sha256 = "d" * 64
    with pytest.raises(ValueError, match="source_sha256 identity mismatch"):
        bad_source.fit(
            train_batches=train,
            validation_batches=validation,
            step=loss,
            resume_from=result.checkpoint,
        )


def test_example_budget_never_stops_inside_an_optimizer_window(tmp_path):
    worker = trainer(
        tmp_path,
        "budget",
        config=TrainingConfig(
            gate="full", epochs=2, gradient_accumulation=2, example_budget=3
        ),
    )
    result = worker.fit(
        train_batches=batches((2, 2), "train"),
        validation_batches=batches((1,), "validation"),
        step=loss,
    )
    assert result.status == "BUDGET_EXHAUSTED"
    assert result.steps_completed == result.examples_seen == 0
    assert (
        torch.load(result.checkpoint, weights_only=True)["progress"]["next_batch_index"]
        == 0
    )


def test_elapsed_budget_survives_resume_with_simulated_time(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("fim_sota.training.time.monotonic", lambda: clock[0])
    config = TrainingConfig(gate="full", epochs=2, wall_clock_budget_seconds=10)
    worker = trainer(tmp_path, "time", config=config)

    def slow_step(model, batch):
        clock[0] = 12.0
        return loss(model, batch)

    train, validation = batches((1, 1), "train"), batches((1,), "validation")
    stopped = worker.fit(
        train_batches=train, validation_batches=validation, step=slow_step
    )
    assert stopped.status == "BUDGET_EXHAUSTED" and stopped.examples_seen == 1
    resumed = trainer(tmp_path, "time-resumed", config=config)
    result = resumed.fit(
        train_batches=train,
        validation_batches=validation,
        step=loss,
        resume_from=stopped.checkpoint,
    )
    assert result.status == "BUDGET_EXHAUSTED" and result.examples_seen == 1
    assert result.elapsed_seconds >= 12


def test_failure_does_not_write_partial_window_as_recovery(tmp_path):
    worker = trainer(tmp_path, "failure")
    seen = [0]

    def fail_second(model, batch):
        seen[0] += 1
        if seen[0] == 2:
            return model(batch.context).mean() * float("nan")
        return loss(model, batch)

    with pytest.raises(FloatingPointError):
        worker.fit(
            train_batches=batches((1, 1), "train"),
            validation_batches=batches((1,), "validation"),
            step=fail_second,
        )
    saved = torch.load(tmp_path / "failure/checkpoints/latest.pt", weights_only=True)
    assert saved["progress"]["examples_seen"] == saved["step"] == 0
    receipt = json.loads((tmp_path / "failure/failure.json").read_text())
    assert receipt["partial_window_discarded"] is True
    assert receipt["settings_changed"] is False


def test_resource_block_writes_receipt_without_training(tmp_path):
    worker = trainer(
        tmp_path, "blocked", admission=lambda: {"status": "BLOCKED", "reason": "test"}
    )

    def forbidden(*args):
        raise AssertionError("training must not start")

    result = worker.fit(
        train_batches=batches((1,), "train"),
        validation_batches=batches((1,), "validation"),
        step=forbidden,
    )
    assert result.status == "BLOCKED_RESOURCE"
    assert Path(result.failure_receipt).is_file()
    assert result.checkpoint is None


def test_atomic_checkpoint_failure_preserves_previous_bytes(tmp_path, monkeypatch):
    destination = tmp_path / "latest.pt"
    destination.write_bytes(b"prior-checkpoint")

    def broken_save(payload, handle):
        handle.write(b"partial")
        raise OSError("simulated interrupted serialization")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="simulated"):
        _atomic_torch_save(destination, {"safe": True})
    assert destination.read_bytes() == b"prior-checkpoint"
    assert list(tmp_path.iterdir()) == [destination]


def test_unsupported_modes_state_and_cpu_precision_fail_explicitly(tmp_path):
    with pytest.raises(ConfigurationError, match="distributed"):
        TrainingConfig(distributed=True).validate()
    with pytest.raises(ValueError, match="CPU.*fp32"):
        trainer(tmp_path, "fp16", config=TrainingConfig(precision="fp16"))

    class Undeclared(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.memory = []

    with pytest.raises(ValueError, match="serialize nonparameter state"):
        trainer(tmp_path, "undeclared", model=Undeclared())


def test_split_owners_must_be_disjoint(tmp_path):
    train, validation = batches((1,), "train"), batches((1,), "validation")
    validation[0] = dataclasses.replace(
        validation[0], trajectory_ids=train[0].trajectory_ids
    )
    with pytest.raises(ValueError, match="trajectory identities overlap"):
        trainer(tmp_path, "overlap").fit(
            train_batches=train, validation_batches=validation, step=loss
        )


def test_resource_admission_is_amortized_per_epoch(tmp_path):
    checks = []
    worker = trainer(
        tmp_path,
        "admission-frequency",
        config=TrainingConfig(gate="full", epochs=1),
        admission=lambda: checks.append(1) or {"status": "PASS"},
    )
    worker.fit(
        train_batches=batches((1, 1, 1, 1), "train"),
        validation_batches=batches((1,), "validation"),
        step=loss,
    )
    assert len(checks) == 2  # Initial admission and epoch entry, not each microbatch.


@pytest.mark.parametrize(
    "field,value",
    [
        ("learning_rate", float("nan")),
        ("gradient_clip_norm", float("inf")),
        ("gradient_accumulation", 1.5),
        ("epochs", True),
        ("seed", -1),
    ],
)
def test_invalid_training_numbers_fail(field, value):
    with pytest.raises(ConfigurationError):
        dataclasses.replace(TrainingConfig(), **{field: value}).validate()


def test_time_interval_saves_only_completed_windows(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("fim_sota.training.time.monotonic", lambda: clock[0])
    worker = trainer(
        tmp_path,
        "interval",
        config=TrainingConfig(
            gate="full",
            gradient_accumulation=2,
            checkpoint_every_steps=100,
            checkpoint_every_seconds=2,
            epochs=1,
        ),
    )

    def timed_step(model, batch):
        clock[0] += 1.0
        return loss(model, batch)

    worker.fit(
        train_batches=batches((1, 1, 1), "train"),
        validation_batches=batches((1,), "validation"),
        step=timed_step,
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "interval/events.jsonl").read_text().splitlines()
    ]
    saved = [event["step"] for event in events if event["event"] == "checkpoint_saved"]
    assert saved.count(1) == 1


@pytest.mark.skipif(
    os.environ.get("FIM_TEST_DEVICE") != "cuda",
    reason="CUDA was not explicitly requested; this is not a GPU pass",
)
def test_cuda_fp32_tiny_training(tmp_path):
    assert torch.cuda.is_available(), "explicit CUDA tests require CUDA"
    torch.manual_seed(1)
    model = nn.Linear(1, 1).to("cuda:0")
    worker = UnifiedTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        config=TrainingConfig(gate="full"),
        run_id="cuda",
        output_root=tmp_path,
        source_sha256="a" * 64,
        config_sha256="b" * 64,
        data_sha256="c" * 64,
        resource_admission=lambda: {"status": "PASS"},
        device="cuda:0",
    )
    result = worker.fit(
        train_batches=batches((2, 1), "train"),
        validation_batches=batches((1,), "validation"),
        step=loss,
    )
    assert result.status == "PASS_ENGINEERING"
    assert all(torch.isfinite(p).all() for p in model.parameters())
