"""Synthetic checkpoint recovery; no retained experiment is loaded or rerun."""
from __future__ import annotations

import copy

import pytest
import torch

from fim.utils.checkpoint import load_checkpoint, save_checkpoint


def _model():
    torch.manual_seed(72)
    return torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1))


def _snapshot(model, optimizer=None, scheduler=None):
    return copy.deepcopy((model.state_dict(),
                          optimizer.state_dict() if optimizer is not None else None,
                          scheduler.state_dict() if scheduler is not None else None))


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def _step(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    loss = (model(torch.tensor([[0.1, -0.3], [0.2, 0.7]])) - 0.4).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()


def test_failed_save_preserves_previous_checkpoint_bytes(tmp_path):
    path = tmp_path / "checkpoint.pt"
    before = b"previous evidence must survive a failed serialization"
    path.write_bytes(before)
    with pytest.raises(Exception):
        save_checkpoint(path, _model(), extra={"bad": lambda: None})
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == {path}


def test_failed_first_save_leaves_no_partial_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"

    def fail_after_write(payload, destination):
        if hasattr(destination, "write"):
            destination.write(b"partial")
        else:
            path.write_bytes(b"partial")
        raise OSError("injected storage failure")

    monkeypatch.setattr(torch, "save", fail_after_write)
    with pytest.raises(OSError, match="injected"):
        save_checkpoint(path, _model())
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("defect", ["shape", "dtype", "nonfinite", "unexpected_key"])
def test_rejected_model_state_cannot_partially_load(tmp_path, defect):
    model = _model()
    before = _snapshot(model)
    state = copy.deepcopy(model.state_dict())
    state["0.weight"].fill_(9)
    if defect == "shape":
        state["2.weight"] = torch.ones(7, 7)
    elif defect == "dtype":
        state["2.weight"] = state["2.weight"].double()
    elif defect == "nonfinite":
        state["2.weight"].fill_(float("nan"))
    else:
        state["unknown"] = torch.ones(1)
    path = tmp_path / "bad.pt"
    torch.save({"model": state, "epoch": 1}, path)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        load_checkpoint(path, model)
    _equal(_snapshot(model), before)


@pytest.mark.parametrize("defect", ["groups", "moment_shape", "moment_nonfinite", "missing"])
def test_rejected_optimizer_state_preserves_model_and_optimizer(tmp_path, defect):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    _step(model, optimizer, scheduler)
    before = _snapshot(model, optimizer, scheduler)
    payload = {"model": copy.deepcopy(model.state_dict()), "epoch": 2,
               "optimizer": copy.deepcopy(optimizer.state_dict())}
    payload["model"]["0.weight"].fill_(7)
    if defect == "groups":
        payload["optimizer"]["param_groups"][0]["params"] = []
    elif defect.startswith("moment"):
        key = next(iter(payload["optimizer"]["state"]))
        if defect == "moment_shape":
            payload["optimizer"]["state"][key]["exp_avg"] = torch.ones(7, 7)
        else:
            payload["optimizer"]["state"][key]["exp_avg"].fill_(float("inf"))
    else:
        del payload["optimizer"]
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        load_checkpoint(path, model, optimizer=optimizer)
    _equal(_snapshot(model, optimizer, scheduler), before)


def test_rejected_scheduler_state_preserves_all_components(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    _step(model, optimizer, scheduler)
    before = _snapshot(model, optimizer, scheduler)
    payload = {"model": copy.deepcopy(model.state_dict()), "epoch": 2,
               "optimizer": copy.deepcopy(optimizer.state_dict()),
               "scheduler": copy.deepcopy(scheduler.state_dict())}
    payload["model"]["0.weight"].fill_(7)
    payload["scheduler"]["gamma"] = float("nan")
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="finite"):
        load_checkpoint(path, model, optimizer=optimizer, scheduler=scheduler)
    _equal(_snapshot(model, optimizer, scheduler), before)


@pytest.mark.parametrize("epoch", [True, -1, 1.5, float("nan")])
def test_invalid_epoch_is_rejected_before_creating_output(tmp_path, epoch):
    path = tmp_path / "not_created" / "checkpoint.pt"
    with pytest.raises((ValueError, TypeError)):
        save_checkpoint(path, _model(), epoch=epoch)
    assert not path.parent.exists()


def test_conflicting_shared_parameter_entries_are_rejected(tmp_path):
    layer = torch.nn.Linear(2, 2)
    model = torch.nn.ModuleDict({"left": layer, "right": layer})
    before = _snapshot(model)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    state["left.weight"].fill_(3)
    state["right.weight"].fill_(4)
    path = tmp_path / "bad.pt"
    torch.save({"model": state, "epoch": 0}, path)
    with pytest.raises(ValueError, match="alias"):
        load_checkpoint(path, model)
    _equal(_snapshot(model), before)


def test_sgd_primitive_momentum_is_rejected_before_resuming(tmp_path):
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    _step(model, optimizer, scheduler)
    before = _snapshot(model, optimizer, scheduler)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler)
    payload = torch.load(path, weights_only=True)
    key = next(iter(payload["optimizer"]["state"]))
    payload["optimizer"]["state"][key]["momentum_buffer"] = 0.0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="momentum"):
        load_checkpoint(path, model, optimizer, scheduler)
    _equal(_snapshot(model, optimizer, scheduler), before)
    _step(model, optimizer, scheduler)


def test_load_explicitly_requests_restricted_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    model = _model()
    save_checkpoint(path, model)
    actual = torch.load
    observed = []

    def record(*args, **kwargs):
        observed.append(kwargs.get("weights_only"))
        return actual(*args, **kwargs)

    monkeypatch.setattr(torch, "load", record)
    load_checkpoint(path, model)
    assert observed == [True]


def test_successful_roundtrip_resumes_exact_next_adamw_and_scheduler_update(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    _step(model, optimizer, scheduler)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, epoch=1, extra={"fixture": "synthetic"})
    restored = _model()
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.ExponentialLR(restored_opt, gamma=0.8)
    payload = load_checkpoint(path, restored, restored_opt, restored_scheduler)
    assert payload["epoch"] == 1
    assert payload["extra"] == {"fixture": "synthetic"}
    _step(model, optimizer, scheduler)
    _step(restored, restored_opt, restored_scheduler)
    _equal(_snapshot(model, optimizer, scheduler), _snapshot(restored, restored_opt, restored_scheduler))


@pytest.mark.parametrize("defect", ["missing_moment", "primitive_step", "invalid_gamma", "negative_variance"])
def test_rejected_resume_state_cannot_poison_the_next_update(tmp_path, defect):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    _step(model, optimizer, scheduler)
    before = _snapshot(model, optimizer, scheduler)
    payload = {"model": copy.deepcopy(model.state_dict()), "epoch": 2,
               "optimizer": copy.deepcopy(optimizer.state_dict()),
               "scheduler": copy.deepcopy(scheduler.state_dict())}
    payload["model"]["0.weight"].fill_(7)
    key = next(iter(payload["optimizer"]["state"]))
    if defect == "missing_moment":
        del payload["optimizer"]["state"][key]["exp_avg"]
    elif defect == "primitive_step":
        payload["optimizer"]["state"][key]["step"] = -1.0
    elif defect == "negative_variance":
        payload["optimizer"]["state"][key]["exp_avg_sq"].fill_(-1)
    else:
        payload["scheduler"]["gamma"] = {"invalid": "gamma"}
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises((ValueError, TypeError)):
        load_checkpoint(path, model, optimizer, scheduler)
    _equal(_snapshot(model, optimizer, scheduler), before)
    _step(model, optimizer, scheduler)
    assert all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())


def test_rollback_does_not_reinvoke_a_failing_optimizer_load_hook(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    _step(model, optimizer, scheduler)
    before = _snapshot(model, optimizer, scheduler)
    parameter_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, epoch=1)

    def reject(actual):
        if actual is optimizer:
            actual.param_groups[0]["lr"] = 999.0
            raise RuntimeError("injected optimizer load hook")

    handle = optimizer.register_load_state_dict_post_hook(reject)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            load_checkpoint(path, model, optimizer, scheduler)
    finally:
        handle.remove()
    _equal(_snapshot(model, optimizer, scheduler), before)
    assert [id(parameter) for group in optimizer.param_groups for parameter in group["params"]] == parameter_ids
    _step(model, optimizer, scheduler)


def test_failed_load_preserves_composite_scheduler_objects(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008)
    children = [torch.optim.lr_scheduler.LinearLR(optimizer),
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)]
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, children, milestones=[2])
    _step(model, optimizer, scheduler)
    before = _snapshot(model, optimizer, scheduler)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, epoch=1)

    def reject(actual):
        if actual is optimizer:
            actual.param_groups[0]["lr"] = 999.0
            raise RuntimeError("injected composite scheduler resume failure")

    handle = optimizer.register_load_state_dict_post_hook(reject)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            load_checkpoint(path, model, optimizer, scheduler)
    finally:
        handle.remove()
    assert all(actual is original for actual, original in zip(scheduler._schedulers, children))
    assert scheduler.optimizer is optimizer
    assert all(child.optimizer is optimizer for child in children)
    _equal(_snapshot(model, optimizer, scheduler), before)
    _step(model, optimizer, scheduler)
