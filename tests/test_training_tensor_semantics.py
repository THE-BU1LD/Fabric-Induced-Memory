from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "fim_experiments" / "train.py"
SPEC = importlib.util.spec_from_file_location("fim_train_semantics", TRAIN_PATH)
assert SPEC is not None and SPEC.loader is not None
train_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = train_module
SPEC.loader.exec_module(train_module)


def test_4d_field_batch_is_not_temporal_sequence():
    x = torch.randn(4, 1, 4, 4)
    y = 0.5 * x + 0.1
    assert not train_module._is_sequence_batch(x, y, rollout_steps=4)


def test_3d_vector_trajectory_is_temporal_sequence():
    x = torch.randn(4, 16, 33)
    y = torch.randn_like(x)
    assert train_module._is_sequence_batch(x, y, rollout_steps=16)


def test_5d_field_trajectory_is_temporal_sequence():
    x = torch.randn(2, 8, 2, 16, 16)
    y = torch.randn_like(x)
    assert train_module._is_sequence_batch(x, y, rollout_steps=8)


def test_static_4d_training_loss_preserves_field_shape():
    model = torch.nn.Conv2d(1, 1, kernel_size=1)
    x = torch.randn(4, 1, 4, 4)
    y = 0.5 * x + 0.1
    loss, _, stats = train_module._rollout_loss(
        model=model,
        x=x,
        y=y,
        rollout_steps=4,
        teacher_forcing_ratio=0.0,
        horizon_decay=1.0,
        update_memory=False,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert stats["horizon"] == 1.0
