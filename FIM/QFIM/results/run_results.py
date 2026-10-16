from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Tuple

import os
import torch

from results.evaluate import evaluate

os.environ.setdefault("MPLBACKEND", "Agg")


@dataclass
class RolloutMetrics:
    model_name: str
    rollout_steps: int
    params: int
    mse: float
    mae: float
    final_step_mse: float
    final_step_mae: float
    divergence_time: float
    stability_under_perturbation: float
    latent_norm_drift: float
    retrieval_utilization: float
    throughput_items_per_sec: float
    trajectory_fidelity: float
    rollout_score: float


def _params_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _canonical_output(out: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    prediction = None
    latent = None
    retrieved = None

    if hasattr(out, "prediction"):
        prediction = out.prediction
        latent = getattr(out, "latent", None)
        retrieved = getattr(out, "retrieved", None)
    elif isinstance(out, tuple):
        prediction = out[0]
        if len(out) > 1:
            aux = out[1]
            if torch.is_tensor(aux):
                latent = aux
            elif hasattr(aux, "latent"):
                latent = getattr(aux, "latent", None)
                retrieved = getattr(aux, "retrieved", None)
    else:
        prediction = out

    if prediction is None:
        raise TypeError("Model output did not contain a prediction tensor")

    return prediction, latent, retrieved


def _model_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    store_traces: bool = True,
    retrieve: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Forward one step without silently disabling an implemented memory path."""
    if hasattr(model, "step"):
        try:
            out = model.step(x, store_traces=store_traces, retrieve=retrieve)
        except TypeError:
            out = model.step(x)
    else:
        out = model(x)
    return _canonical_output(out)


def _reset_model_state(model: torch.nn.Module) -> None:
    for name in ("reset_state", "clear_memory", "detach_state"):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            break


def _prepare_rollout_input(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(1).unsqueeze(1)
    if x.ndim == 3:
        return x.unsqueeze(1)
    return x


def _flatten_traj(x: torch.Tensor) -> torch.Tensor:
    if x.ndim >= 3:
        return x.reshape(x.shape[0], x.shape[1], -1)
    if x.ndim == 2:
        return x.unsqueeze(0).reshape(x.shape[0], 1, -1)
    return x.reshape(1, 1, -1)


def _trajectory_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    pred_f = _flatten_traj(pred).float()
    gt_f = _flatten_traj(gt).float()
    t = min(pred_f.shape[1], gt_f.shape[1])
    d = min(pred_f.shape[2], gt_f.shape[2])
    pred_f = pred_f[:, :t, :d]
    gt_f = gt_f[:, :t, :d]

    mse_per_step = torch.mean((pred_f - gt_f) ** 2, dim=-1)
    mae_per_step = torch.mean(torch.abs(pred_f - gt_f), dim=-1)

    gt_centered = gt_f - gt_f.mean(dim=1, keepdim=True)
    pred_centered = pred_f - pred_f.mean(dim=1, keepdim=True)
    numerator = torch.sum(gt_centered * pred_centered, dim=-1)
    denom = torch.sqrt(
        torch.sum(gt_centered**2, dim=-1).clamp_min(1e-12)
        * torch.sum(pred_centered**2, dim=-1).clamp_min(1e-12)
    )
    corr = (numerator / denom).clamp(-1.0, 1.0)

    return {
        "mse": mse_per_step.mean().item(),
        "mae": mae_per_step.mean().item(),
        "final_step_mse": mse_per_step[:, -1].mean().item(),
        "final_step_mae": mae_per_step[:, -1].mean().item(),
        "trajectory_fidelity": corr.mean().item(),
    }


def _divergence_time(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.15) -> float:
    pred_f = _flatten_traj(pred).float()
    gt_f = _flatten_traj(gt).float()
    t = min(pred_f.shape[1], gt_f.shape[1])
    d = min(pred_f.shape[2], gt_f.shape[2])
    err = torch.mean(torch.abs(pred_f[:, :t, :d] - gt_f[:, :t, :d]), dim=-1)
    for i in range(t):
        if torch.any(err[:, i] > threshold):
            return float(i)
    return float(t)


def _latent_norm_drift(latents: List[torch.Tensor]) -> float:
    norms = []
    for z in latents:
        if z is None:
            continue
        z = z.float()
        norms.append(torch.norm(z.reshape(z.shape[0], -1), dim=-1).mean().item())
    if len(norms) < 2:
        return 0.0
    return float(abs(norms[-1] - norms[0]))


def _retrieval_utilization(retrieved_outputs: List[Optional[torch.Tensor]]) -> float:
    if not retrieved_outputs:
        return 0.0
    return float(
        sum(
            1
            for r in retrieved_outputs
            if torch.is_tensor(r) and r.numel() > 0
        )
        / len(retrieved_outputs)
    )


@torch.no_grad()
def _autoregressive_rollout(
    model: torch.nn.Module,
    x0: torch.Tensor,
    steps: int,
    device: torch.device,
    *,
    store_traces: bool = True,
    retrieve: bool = True,
) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]], List[Optional[torch.Tensor]], float]:
    model.eval()
    _reset_model_state(model)
    current = _prepare_rollout_input(x0)
    preds: List[torch.Tensor] = []
    latents: List[Optional[torch.Tensor]] = []
    retrieved_outputs: List[Optional[torch.Tensor]] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = perf_counter()

    for _ in range(steps):
        pred, latent, retrieved = _model_forward(
            model,
            current,
            store_traces=store_traces,
            retrieve=retrieve,
        )
        preds.append(pred)
        latents.append(latent)
        retrieved_outputs.append(retrieved)
        current = _prepare_rollout_input(pred)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    return torch.stack(preds, dim=1), latents, retrieved_outputs, elapsed


def _metrics_for_rollout(
    model: torch.nn.Module,
    pred: torch.Tensor,
    gt: torch.Tensor,
    latents: List[Optional[torch.Tensor]],
    retrieved_outputs: List[Optional[torch.Tensor]],
    elapsed: float,
    steps: int,
    divergence_threshold: float,
) -> Dict[str, float]:
    traj = _trajectory_metrics(pred, gt)
    return {
        "mse": traj["mse"],
        "mae": traj["mae"],
        "final_step_mse": traj["final_step_mse"],
        "final_step_mae": traj["final_step_mae"],
        "divergence_time": _divergence_time(pred, gt, threshold=divergence_threshold),
        "latent_norm_drift": _latent_norm_drift([z for z in latents if z is not None]),
        "retrieval_utilization": _retrieval_utilization(retrieved_outputs),
        "throughput_items_per_sec": float(steps / max(elapsed, 1e-8)),
        "trajectory_fidelity": traj["trajectory_fidelity"],
        "rollout_score": 1.0 / (1.0 + traj["mse"]),
    }


@torch.no_grad()
def _perturbation_stability(
    model: torch.nn.Module,
    x0: torch.Tensor,
    gt: torch.Tensor,
    steps: int,
    device: torch.device,
    *,
    store_traces: bool,
    retrieve: bool,
    noise_scale: float = 0.01,
    trials: int = 5,
) -> float:
    scores = []
    base = x0.float()
    for _ in range(trials):
        x_perturbed = base + noise_scale * torch.randn_like(base)
        pred, _, _, _ = _autoregressive_rollout(
            model,
            x_perturbed,
            steps,
            device,
            store_traces=store_traces,
            retrieve=retrieve,
        )
        stats = _trajectory_metrics(pred, gt)
        scores.append(1.0 / (1.0 + stats["mse"]))
    return float(sum(scores) / max(len(scores), 1))


def _plot_rollout(gt: torch.Tensor, pred: torch.Tensor, title: str, save_path: Optional[str | Path] = None) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gt_f = _flatten_traj(gt.detach().cpu())[0]
    pred_f = _flatten_traj(pred.detach().cpu())[0]
    t = min(gt_f.shape[0], pred_f.shape[0])
    d = min(gt_f.shape[1], pred_f.shape[1])

    plt.figure(figsize=(10, 4))
    for j in range(min(d, 4)):
        plt.plot(gt_f[:t, j].numpy(), label=f"truth {j}" if j == 0 else None, alpha=0.8)
        plt.plot(pred_f[:t, j].numpy(), label=f"pred {j}" if j == 0 else None, alpha=0.8, linestyle="--")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


def _is_memory_model(model: torch.nn.Module) -> bool:
    return any(hasattr(model, name) for name in ("memory_bank", "trace_bank", "bank"))


def _run_mode(
    model: torch.nn.Module,
    x0: torch.Tensor,
    gt: torch.Tensor,
    steps: int,
    device: torch.device,
    *,
    store_traces: bool,
    retrieve: bool,
    perturbation_scale: float,
    perturbation_trials: int,
    divergence_threshold: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred, latents, retrieved_outputs, elapsed = _autoregressive_rollout(
        model,
        x0,
        steps,
        device,
        store_traces=store_traces,
        retrieve=retrieve,
    )
    metrics = _metrics_for_rollout(
        model,
        pred,
        gt,
        latents,
        retrieved_outputs,
        elapsed,
        steps,
        divergence_threshold,
    )
    metrics["stability_under_perturbation"] = _perturbation_stability(
        model,
        x0,
        gt,
        steps,
        device,
        store_traces=store_traces,
        retrieve=retrieve,
        noise_scale=perturbation_scale,
        trials=perturbation_trials,
    )
    return pred, metrics


def _run_baseline(
    model: torch.nn.Module,
    benchmark: Any,
    device: torch.device,
    steps: int,
) -> Dict[str, float]:
    x0 = benchmark.sample_initial_state(1, device=device)
    gt = benchmark.rollout(x0, steps=steps)
    _, metrics = _run_mode(
        model,
        x0,
        gt,
        steps,
        device,
        store_traces=True,
        retrieve=True,
        perturbation_scale=0.01,
        perturbation_trials=5,
        divergence_threshold=0.15,
    )
    return metrics


def run(
    model: torch.nn.Module,
    benchmark: Any,
    device: str | torch.device = "cpu",
    rollout_steps: int = 100,
    perturbation_scale: float = 0.01,
    perturbation_trials: int = 5,
    divergence_threshold: float = 0.15,
    plot_path: Optional[str | Path] = None,
    baselines: Optional[Mapping[str, torch.nn.Module]] = None,
    save_results_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Evaluate the full model; for memory models also run paired mechanism controls.

    The previous implementation called ``step(..., store_traces=False, retrieve=False)``
    for every model with a ``step`` method, which disabled FIM's memory during normal
    evaluation. This function treats full memory as the default and records no-retrieval
    and no-memory controls on the same checkpoint and initial trajectory.
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    _reset_model_state(model)

    x0 = benchmark.sample_initial_state(1, device=device)
    gt = benchmark.rollout(x0, steps=rollout_steps)

    full_pred, full = _run_mode(
        model,
        x0,
        gt,
        rollout_steps,
        device,
        store_traces=True,
        retrieve=True,
        perturbation_scale=perturbation_scale,
        perturbation_trials=perturbation_trials,
        divergence_threshold=divergence_threshold,
    )

    metrics = RolloutMetrics(
        model_name=type(model).__name__,
        rollout_steps=rollout_steps,
        params=_params_count(model),
        mse=full["mse"],
        mae=full["mae"],
        final_step_mse=full["final_step_mse"],
        final_step_mae=full["final_step_mae"],
        divergence_time=full["divergence_time"],
        stability_under_perturbation=full["stability_under_perturbation"],
        latent_norm_drift=full["latent_norm_drift"],
        retrieval_utilization=full["retrieval_utilization"],
        throughput_items_per_sec=full["throughput_items_per_sec"],
        trajectory_fidelity=full["trajectory_fidelity"],
        rollout_score=full["rollout_score"],
    )

    _plot_rollout(gt, full_pred, title="Rollout Comparison", save_path=plot_path)

    result: Dict[str, Any] = {
        "model": type(model).__name__,
        "metrics": asdict(metrics),
        "evaluation_protocol": "memory_enabled_full_model_v2",
    }

    if _is_memory_model(model):
        _, no_retrieval = _run_mode(
            model,
            x0,
            gt,
            rollout_steps,
            device,
            store_traces=True,
            retrieve=False,
            perturbation_scale=perturbation_scale,
            perturbation_trials=perturbation_trials,
            divergence_threshold=divergence_threshold,
        )
        _, no_memory = _run_mode(
            model,
            x0,
            gt,
            rollout_steps,
            device,
            store_traces=False,
            retrieve=False,
            perturbation_scale=perturbation_scale,
            perturbation_trials=perturbation_trials,
            divergence_threshold=divergence_threshold,
        )
        result["memory_modes"] = {
            "full": full,
            "no_retrieval": no_retrieval,
            "no_memory": no_memory,
            "paired_effects": {
                "no_retrieval_minus_full_mse": no_retrieval["mse"] - full["mse"],
                "no_memory_minus_full_mse": no_memory["mse"] - full["mse"],
            },
        }

    # Independent batch evaluation is retained as a consistency check only.
    result["evaluation"] = evaluate(
        model,
        benchmark,
        device=device,
        steps=rollout_steps,
        store_traces=True,
        retrieve=True,
    )

    if baselines:
        result["baselines"] = {}
        for name, baseline_model in dict(baselines).items():
            result["baselines"][name] = _run_baseline(
                baseline_model.to(device), benchmark, device, rollout_steps
            )

    if save_results_path is not None:
        import json
        save_results_path = Path(save_results_path)
        save_results_path.parent.mkdir(parents=True, exist_ok=True)
        with save_results_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result
