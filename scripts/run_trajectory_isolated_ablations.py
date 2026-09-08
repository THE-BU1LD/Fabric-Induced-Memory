#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "fim_experiments"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import main as experiment_main
import train as train_module
from ablation_systems import AblatedFIMSystem, variant_switches

PROTOCOL = ROOT / "research" / "protocols" / "FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1.md"
DEFAULT_BENCHMARKS = ["delayed_recall", "lorenz96"]
DEFAULT_SEEDS = [101, 211, 307, 401, 503]
DEFAULT_VARIANTS = ["full", "no_memory", "no_retrieval", "no_salience_gating"]
CANONICAL_EVAL_RUN = experiment_main.run


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except Exception:
        return None


def require_trajectory_isolation(batch_size: int) -> None:
    if int(batch_size) != 1:
        raise ValueError(
            "Trajectory-isolated FIM protocol requires batch_size=1 for every arm; "
            f"received batch_size={batch_size}. Refusing cross-trajectory memory exposure."
        )


def _assert_single_episode_batch(x: torch.Tensor) -> None:
    if x.shape[0] != 1:
        raise RuntimeError(
            "Trajectory-isolated validation received more than one independent episode in a batch."
        )


@torch.no_grad()
def _isolated_validate_from_loader(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    rollout_steps: int,
    horizon_decay: float,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    rollout_losses: list[float] = []

    for batch in loader:
        train_module._reset_model_state(model)
        x, y = train_module._unpack_batch(batch)
        _assert_single_episode_batch(x)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss, _, stats = train_module._rollout_loss(
            model=model,
            x=x,
            y=y,
            rollout_steps=rollout_steps,
            teacher_forcing_ratio=0.0,
            horizon_decay=horizon_decay,
            update_memory=True,
        )
        losses.append(float(loss.item()))
        rollout_losses.append(float(stats["rollout_loss"]))

    if not losses:
        raise RuntimeError("Trajectory-isolated validation loader produced zero batches.")
    return {
        "loss": sum(losses) / len(losses),
        "rollout_loss": sum(rollout_losses) / len(rollout_losses),
    }


@torch.no_grad()
def _isolated_validate_from_benchmark(
    model: torch.nn.Module,
    benchmark: Any,
    device: torch.device,
    batch_size: int,
    batches: int,
    rollout_steps: int,
    horizon_decay: float,
) -> dict[str, float]:
    require_trajectory_isolation(batch_size)
    model.eval()
    losses: list[float] = []
    rollout_losses: list[float] = []

    for _ in range(int(batches)):
        train_module._reset_model_state(model)
        x, y = benchmark.generate_batch(batch_size=1, device=device)
        _assert_single_episode_batch(x)
        loss, _, stats = train_module._rollout_loss(
            model=model,
            x=x,
            y=y,
            rollout_steps=rollout_steps,
            teacher_forcing_ratio=0.0,
            horizon_decay=horizon_decay,
            update_memory=True,
        )
        losses.append(float(loss.item()))
        rollout_losses.append(float(stats["rollout_loss"]))

    if not losses:
        raise RuntimeError("Trajectory-isolated benchmark validation produced zero batches.")
    return {
        "loss": sum(losses) / len(losses),
        "rollout_loss": sum(rollout_losses) / len(rollout_losses),
    }


def _isolated_eval_run(*args, **kwargs):
    requested = kwargs.get("batch_size", 1)
    require_trajectory_isolation(requested)
    kwargs["batch_size"] = 1
    model = args[0] if args else kwargs.get("model")
    if model is not None:
        train_module._reset_model_state(model)
    return CANONICAL_EVAL_RUN(*args, **kwargs)


def install_trajectory_isolation_hooks() -> None:
    # The historical training functions remain unchanged on disk. This frozen
    # lane overrides validation and final evaluation semantics explicitly so
    # memory is usable within one trajectory but never shared across episodes.
    train_module._validate_from_loader = _isolated_validate_from_loader
    train_module._validate_from_benchmark = _isolated_validate_from_benchmark
    experiment_main.run = _isolated_eval_run


def model_factory(switches: dict[str, bool]):
    def factory(**kwargs):
        return AblatedFIMSystem(**kwargs, **switches)

    return factory


def metric_scalar(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def manifest_payload(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "protocol_name": "FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1",
        "protocol_path": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "created_or_updated_utc": utc_now(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "scientific_boundary": (
            "Fresh trajectory-isolated current-code FIM evidence only. Historical shared-bank results are "
            "not rewritten. Null, mixed, adverse, and failed cells are retained."
        ),
        "trajectory_isolation": {
            "batch_size": 1,
            "memory_reset_between_independent_episodes": True,
            "validation_memory_enabled_within_episode": True,
            "test_memory_enabled_within_episode": True,
            "checkpoint_selection": "fixed_final_epoch_no_best_selection",
            "ema": False,
        },
        "benchmarks": list(args.benchmarks),
        "seeds": list(args.seeds),
        "variants": list(args.variants),
        "budget": {
            "epochs": args.epochs,
            "dataset_size": args.dataset_size,
            "batch_size": 1,
            "train_rollout_steps": args.rollout_steps,
            "eval_rollout_steps": args.eval_steps,
        },
        "expected_cells": len(args.benchmarks) * len(args.seeds) * len(args.variants),
        "runs": rows,
    }


def write_manifest(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest_payload(args, rows), indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.manifest)


def run_one(args: argparse.Namespace, benchmark: str, seed: int, variant: str) -> dict[str, Any]:
    switches = variant_switches(variant)
    experiment_main.FIMSystem = model_factory(switches)

    cfg = experiment_main.default_config()
    cfg["experiment"]["name"] = f"trajectory_isolated_{benchmark}_{variant}_s{seed}"
    cfg["runtime"]["seed"] = seed
    cfg["runtime"]["device"] = args.device
    cfg["runtime"]["results_root"] = str(args.results_root)
    cfg["runtime"]["deterministic"] = True
    cfg["benchmark"]["name"] = benchmark
    cfg["model"]["name"] = "fim_plus"
    cfg["train"]["epochs"] = args.epochs
    cfg["train"]["batch_size"] = 1
    cfg["train"]["dataset_size"] = args.dataset_size
    cfg["train"]["rollout_steps"] = args.rollout_steps
    cfg["train"]["workers"] = 0
    cfg["train"]["dynamic_data"] = False
    cfg["train"]["use_ema"] = False
    cfg["train"]["evaluate_ema"] = False
    cfg["train"]["save_best"] = False
    cfg["train"]["save_last"] = True
    cfg["eval"]["rollout_steps"] = args.eval_steps

    started = utc_now()
    metrics = experiment_main.run_experiment(cfg)
    ended = utc_now()

    row: dict[str, Any] = {
        "status": "success",
        "benchmark": benchmark,
        "seed": seed,
        "variant": variant,
        **switches,
        "started_utc": started,
        "ended_utc": ended,
        "git_commit": git_commit(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "experiment_name": cfg["experiment"]["name"],
        "results_root": str(args.results_root),
        "epochs": args.epochs,
        "batch_size": 1,
        "dataset_size": args.dataset_size,
        "train_rollout_steps": args.rollout_steps,
        "eval_rollout_steps": args.eval_steps,
        "validation_memory_enabled": True,
        "evaluation_memory_enabled": True,
    }
    for key in ["rollout_mse", "final_step_mse", "rollout_mae", "final_step_mae"]:
        value = metric_scalar(metrics, key)
        if value is not None:
            row[key] = value
    if "rollout_mse" not in row:
        raise RuntimeError("Successful run did not produce required rollout_mse metric.")
    return row


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the frozen trajectory-isolated FIM ablation matrix.")
    ap.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--dataset-size", type=int, default=32)
    ap.add_argument("--rollout-steps", type=int, default=4)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results" / "trajectory_isolated_v1",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results" / "trajectory_isolated_v1" / "manifest.json",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    require_trajectory_isolation(args.batch_size)
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)

    # Resolve every label before any compute. Unsupported historical labels fail
    # closed instead of silently mapping onto a current mechanism.
    for variant in args.variants:
        variant_switches(variant)

    install_trajectory_isolation_hooks()
    args.results_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    write_manifest(args, rows)

    for benchmark in args.benchmarks:
        for seed in args.seeds:
            for variant in args.variants:
                print(f"RUN benchmark={benchmark} seed={seed} variant={variant}", flush=True)
                try:
                    row = run_one(args, benchmark, seed, variant)
                except Exception as exc:
                    row = {
                        "status": "failed",
                        "benchmark": benchmark,
                        "seed": seed,
                        "variant": variant,
                        **variant_switches(variant),
                        "git_commit": git_commit(),
                        "protocol_sha256": sha256_file(PROTOCOL),
                        "failed_utc": utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    print(f"FAILED benchmark={benchmark} seed={seed} variant={variant}: {exc}", flush=True)
                rows.append(row)
                write_manifest(args, rows)

    expected = len(args.benchmarks) * len(args.seeds) * len(args.variants)
    failures = [row for row in rows if row.get("status") != "success"]
    print(f"WROTE {args.manifest} rows={len(rows)} expected={expected} failures={len(failures)}", flush=True)
    if len(rows) != expected or failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
