"""Artificial save/load fixtures only; no training or retained checkpoint access."""

from __future__ import annotations

from pathlib import Path
import stat

import pytest
import torch

from fim_experiments import train as training


def _model(value=0.5):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


def _save(path, *, value=0.5, epoch=3, ema=None):
    model = _model(value)
    training.save_checkpoint(
        path, model, training.build_optimizer(model.parameters()), None,
        epoch=epoch, best_val_loss=0.25, history={"train_loss": [0.25]},
        ema=ema, validation_weight_source="ema" if ema is not None else "raw",
    )


def _partial_failure(exception):
    def fail(_payload, destination):
        if hasattr(destination, "write"):
            destination.write(b"incomplete serialization fixture")
        else:
            Path(destination).write_bytes(b"incomplete serialization fixture")
        raise exception("artificial interrupted checkpoint save")
    return fail


@pytest.mark.parametrize("existed", [False, True])
@pytest.mark.parametrize("exception", [OSError, KeyboardInterrupt, SystemExit])
def test_interrupted_serialization_preserves_previous_checkpoint(tmp_path, monkeypatch, existed, exception):
    path = tmp_path / "checkpoint.pt"
    original = None
    if existed:
        _save(path)
        original = path.read_bytes()
    monkeypatch.setattr(training.torch, "save", _partial_failure(exception))
    with pytest.raises(exception, match="artificial interrupted"):
        _save(path, value=0.9, epoch=4)
    if existed:
        assert path.read_bytes() == original
        restored = _model(0.0)
        state = training.load_checkpoint(path, restored)
        assert state.epoch == 3
        torch.testing.assert_close(restored.weight, torch.full_like(restored.weight, 0.5))
    else:
        assert not path.exists()
    assert sorted(item.name for item in tmp_path.iterdir()) == ([path.name] if existed else [])


@pytest.mark.parametrize("existed", [False, True])
@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_filesystem_failure_preserves_destination(tmp_path, monkeypatch, existed, operation):
    path = tmp_path / "checkpoint.pt"
    original = None
    if existed:
        _save(path)
        original = path.read_bytes()

    def fail(*_arguments):
        raise OSError(f"artificial {operation} failure")

    monkeypatch.setattr(training.os, operation, fail)
    with pytest.raises(OSError, match=f"artificial {operation}"):
        _save(path, value=0.9, epoch=4)
    assert path.read_bytes() == original if existed else not path.exists()
    assert sorted(item.name for item in tmp_path.iterdir()) == ([path.name] if existed else [])


def test_success_preserves_payload_permissions_and_unrelated_staging(tmp_path):
    path = tmp_path / "nested" / "checkpoint.pt"
    _save(path)
    path.chmod(0o640)
    unrelated = path.parent / f".{path.name}.stage-unrelated.tmp"
    unrelated.write_bytes(b"another operation's temporary bytes")
    ema = training.EMA(_model(0.75))
    _save(path, value=0.9, epoch=4, ema=ema)
    payload = torch.load(path, weights_only=True)
    assert payload["epoch"] == 4
    assert payload["history"] == {"train_loss": [0.25]}
    assert payload["best_val_loss"] == 0.25
    assert payload["validation_weight_source"] == "ema"
    assert set(payload) == {"epoch", "best_val_loss", "history", "model", "optimizer", "validation_weight_source", "ema"}
    torch.testing.assert_close(payload["model"]["weight"], torch.tensor([[0.9]]))
    torch.testing.assert_close(payload["ema"]["weight"], torch.tensor([[0.75]]))
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert unrelated.read_bytes() == b"another operation's temporary bytes"
    assert sorted(item.name for item in path.parent.iterdir()) == sorted([path.name, unrelated.name])


@pytest.mark.parametrize("kind", ["directory", "symlink", "dangling_symlink"])
def test_nonregular_destinations_are_rejected_without_touching_targets(tmp_path, kind):
    path = tmp_path / "checkpoint.pt"
    target = tmp_path / "target.pt"
    if kind == "directory":
        path.mkdir()
        marker = path / "keep.txt"
        marker.write_bytes(b"unrelated directory entry")
    else:
        if kind == "symlink":
            _save(target)
        path.symlink_to(target)
    original = target.read_bytes() if target.exists() else None
    with pytest.raises(ValueError, match="Checkpoint destination must be a regular file"):
        _save(path)
    assert not list(tmp_path.glob("*.stage-*"))
    if kind == "directory":
        assert marker.read_bytes() == b"unrelated directory entry"
    else:
        assert path.is_symlink()
        assert path.readlink() == target
        assert target.read_bytes() == original if original is not None else not target.exists()
