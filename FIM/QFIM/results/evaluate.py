from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _canonical_output(out: Any) -> torch.Tensor:
    if hasattr(out, "prediction"):
        return out.prediction
    if isinstance(out, tuple):
        return out[0]
    return out


def _forward(model, x, *, store_traces: bool = True, retrieve: bool = True):
    if hasattr(model, "step"):
        try:
            out = model.step(x, store_traces=store_traces, retrieve=retrieve)
        except TypeError:
            out = model.step(x)
    else:
        out = model(x)
    return _canonical_output(out)


def _reset_model_state(model):
    for name in ("reset_state", "clear_memory", "detach_state"):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            break


def _prepare(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(1).unsqueeze(1)
    if x.ndim == 3:
        return x.unsqueeze(1)
    return x


def _flatten_trajectory(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise ValueError(f"Expected [batch,time,...] trajectory, got {tuple(x.shape)}")
    return x.reshape(x.shape[0], x.shape[1], -1).float()


@torch.no_grad()
def rollout(
    model,
    x0,
    steps,
    *,
    store_traces: bool = True,
    retrieve: bool = True,
):
    _reset_model_state(model)
    preds = []
    x = _prepare(x0)
    for _ in range(steps):
        pred = _forward(model, x, store_traces=store_traces, retrieve=retrieve)
        preds.append(pred)
        x = _prepare(pred.detach())
    return torch.stack(preds, dim=1)


def compute_mse(traj_pred, traj_true):
    pred = _flatten_trajectory(traj_pred)
    true = _flatten_trajectory(traj_true)
    t = min(pred.shape[1], true.shape[1])
    d = min(pred.shape[2], true.shape[2])
    return ((pred[:, :t, :d] - true[:, :t, :d]) ** 2).mean(dim=-1)


def evaluate(
    model,
    benchmark,
    device="cpu",
    steps=100,
    save_dir="results",
    *,
    store_traces: bool = True,
    retrieve: bool = True,
):
    """Batch rollout consistency check with explicit memory-mode controls."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    (save_dir / "metrics").mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        x0 = benchmark.sample_initial_state(batch_size=8, device=device)
        traj_true = benchmark.rollout(x0, steps)
        traj_pred = rollout(
            model,
            x0,
            steps,
            store_traces=store_traces,
            retrieve=retrieve,
        )
        mse = compute_mse(traj_pred, traj_true[:, 1:])

        torch.save(traj_true, save_dir / "trajectories" / "true.pt")
        torch.save(traj_pred, save_dir / "trajectories" / "pred.pt")

        metrics = {
            "mse_mean": mse.mean().item(),
            "mse_final": mse[:, -1].mean().item(),
            "store_traces": bool(store_traces),
            "retrieve": bool(retrieve),
        }
        with (save_dir / "metrics" / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return metrics
