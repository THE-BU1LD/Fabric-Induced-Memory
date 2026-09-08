#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import traceback
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "fim_experiments"
for path in (str(ROOT), str(EXP)):
    if path not in sys.path:
        sys.path.insert(0, path)

import main as experiment_main
from ablation_systems import AblatedFIMSystem, variant_switches

PROTOCOL = ROOT / "research" / "FRESH_MULTI_SEED_REPLICATION_PROTOCOL_V2_20260908.md"
OUT = ROOT / "results" / "fresh_paired_multiseed_v2_20260908"
SEEDS = [101, 202, 303, 404, 505]
BENCHMARKS = ["lorenz96", "delayed_recall"]
BENCHMARK_OFFSETS = {"lorenz96": 10_000, "delayed_recall": 20_000}
VARIANTS = ["full", "no_memory", "no_retrieval", "no_salience_gating"]
BASELINES = ["fim_plus", "transformer", "ssm", "deeponet", "mlp"]
EPOCHS = 6
BATCH_SIZE = 64
DATASET_SIZE = 1024
TRAIN_ROLLOUT_STEPS = 4
EVAL_ROLLOUT_STEPS = 30
HIDDEN = 64
TRACE_DIM = 32

_ORIGINAL_EVAL_RUN = experiment_main.run
_CANONICAL_FIM_SYSTEM = experiment_main.FIMSystem
_ACTIVE_DATA_SEED: int | None = None
_ACTIVE_EVAL_SEED: int | None = None


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def git_status() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def paired_seeds(benchmark: str, seed: int) -> tuple[int, int]:
    offset = BENCHMARK_OFFSETS[benchmark]
    return 1_000_000 + offset + int(seed), 2_000_000 + offset + int(seed)


@contextmanager
def isolated_rng(seed: int) -> Iterator[None]:
    """Isolate Python/Torch RNG so model-specific draws cannot shift data/eval rows."""
    py_state = random.getstate()
    with torch.random.fork_rng(devices=[]):
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        try:
            yield
        finally:
            random.setstate(py_state)


def set_active_pairing(data_seed: int, eval_seed: int) -> None:
    global _ACTIVE_DATA_SEED, _ACTIVE_EVAL_SEED
    _ACTIVE_DATA_SEED = int(data_seed)
    _ACTIVE_EVAL_SEED = int(eval_seed)


def paired_build_loader_static(
    benchmark: Any,
    batch_size: int,
    device: torch.device,
    dataset_size: int,
    workers: int,
    steps: int | None = None,
):
    if _ACTIVE_DATA_SEED is None:
        raise RuntimeError("paired data seed is not set")
    if dataset_size < 2:
        raise ValueError("dataset size is too small to create train/validation splits")

    # Synthetic data generation gets an isolated stream independent of model init/dropout.
    with isolated_rng(_ACTIVE_DATA_SEED):
        x, y = benchmark.generate_batch(batch_size=dataset_size, steps=steps, device="cpu")
    x, y = experiment_main.reshape_if_needed(x, y)
    dataset = TensorDataset(x, y)

    split = max(1, int(0.85 * len(dataset)))
    split = min(split, len(dataset) - 1)
    split_generator = torch.Generator().manual_seed(_ACTIVE_DATA_SEED + 1)
    train_set, val_set = random_split(
        dataset, [split, len(dataset) - split], generator=split_generator
    )

    loader_kwargs = experiment_main._loader_kwargs(device, workers)
    shuffle_generator = torch.Generator().manual_seed(_ACTIVE_DATA_SEED + 2)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=shuffle_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def paired_eval_run(*args, **kwargs):
    if _ACTIVE_EVAL_SEED is None:
        raise RuntimeError("paired evaluation seed is not set")
    with isolated_rng(_ACTIVE_EVAL_SEED):
        return _ORIGINAL_EVAL_RUN(*args, **kwargs)


def install_pairing_hooks() -> None:
    experiment_main.build_loader_static = paired_build_loader_static
    experiment_main.run = paired_eval_run


def mechanism_factory(switches: dict[str, bool]):
    def factory(**kwargs):
        return AblatedFIMSystem(**kwargs, **switches)

    return factory


def make_config(
    benchmark: str,
    model: str,
    seed: int,
    experiment_name: str,
) -> dict[str, Any]:
    cfg = experiment_main.default_config()
    cfg["experiment"]["name"] = experiment_name
    cfg["runtime"]["seed"] = int(seed)
    cfg["runtime"]["device"] = "cpu"
    cfg["runtime"]["results_root"] = str(OUT / "runs")
    cfg["benchmark"]["name"] = benchmark
    cfg["model"]["name"] = model
    cfg["model"]["hidden"] = HIDDEN
    cfg["model"]["trace_dim"] = TRACE_DIM
    cfg["train"]["epochs"] = EPOCHS
    cfg["train"]["batch_size"] = BATCH_SIZE
    cfg["train"]["dataset_size"] = DATASET_SIZE
    cfg["train"]["workers"] = 0
    cfg["train"]["rollout_steps"] = TRAIN_ROLLOUT_STEPS
    cfg["eval"]["rollout_steps"] = EVAL_ROLLOUT_STEPS
    return cfg


def finite_metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing/non-numeric {key}: {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {value}")
    return value


def result_dir(experiment_name: str, benchmark: str, model: str) -> str:
    return str(OUT / "runs" / experiment_name / f"{benchmark}_{model}")


def base_row(
    *, kind: str, benchmark: str, seed: int, arm: str, experiment_name: str, model: str
) -> dict[str, Any]:
    data_seed, eval_seed = paired_seeds(benchmark, seed)
    return {
        "kind": kind,
        "benchmark": benchmark,
        "seed": int(seed),
        "arm": arm,
        "model_name": model,
        "model_seed": int(seed),
        "data_seed": data_seed,
        "split_seed": data_seed + 1,
        "shuffle_seed": data_seed + 2,
        "eval_seed": eval_seed,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "dataset_size": DATASET_SIZE,
        "train_rollout_steps": TRAIN_ROLLOUT_STEPS,
        "eval_rollout_steps": EVAL_ROLLOUT_STEPS,
        "hidden": HIDDEN,
        "trace_dim": TRACE_DIM if model == "fim_plus" else None,
        "experiment_name": experiment_name,
        "result_dir": result_dir(experiment_name, benchmark, model),
        "git_commit": git_commit(),
    }


def execute_cell(
    *, kind: str, benchmark: str, seed: int, arm: str, model: str, switches: dict[str, bool] | None
) -> dict[str, Any]:
    experiment_name = f"paired_v2_{kind}_{benchmark}_{arm}_s{seed}"
    row = base_row(
        kind=kind,
        benchmark=benchmark,
        seed=seed,
        arm=arm,
        experiment_name=experiment_name,
        model=model,
    )
    if switches is not None:
        row.update(switches)

    data_seed, eval_seed = paired_seeds(benchmark, seed)
    set_active_pairing(data_seed, eval_seed)
    experiment_main.FIMSystem = (
        mechanism_factory(switches) if switches is not None else _CANONICAL_FIM_SYSTEM
    )

    cfg = make_config(benchmark, model, seed, experiment_name)
    row["started_utc"] = utcnow()
    try:
        metrics = experiment_main.run_experiment(cfg)
        row.update(
            {
                "status": "success",
                "ended_utc": utcnow(),
                "rollout_mse": finite_metric(metrics, "rollout_mse"),
                "rollout_mae": finite_metric(metrics, "rollout_mae"),
                "final_step_mse": finite_metric(metrics, "final_step_mse"),
                "final_step_mae": finite_metric(metrics, "final_step_mae"),
                "parameters": int(metrics["parameters"]),
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "ended_utc": utcnow(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return row


def pairing_self_check() -> dict[str, Any]:
    """Prove data/split/shuffle and eval streams ignore ambient RNG consumption."""
    benchmark_cfg = {"name": "lorenz96", "config": {}}
    data_seed, eval_seed = paired_seeds("lorenz96", 101)

    def loader_snapshot(ambient_draws: int):
        torch.manual_seed(7)
        _ = torch.randn(ambient_draws)
        set_active_pairing(data_seed, eval_seed)
        benchmark = experiment_main.select_benchmark(benchmark_cfg)
        loader, val = paired_build_loader_static(
            benchmark=benchmark,
            batch_size=8,
            device=torch.device("cpu"),
            dataset_size=32,
            workers=0,
            steps=3,
        )
        tensors = tuple(t.clone() for t in loader.dataset.dataset.tensors)
        train_indices = list(loader.dataset.indices)
        val_indices = list(val.dataset.indices)
        sampler_order = list(iter(loader.sampler))
        return tensors, train_indices, val_indices, sampler_order

    first = loader_snapshot(3)
    second = loader_snapshot(113)
    if len(first[0]) != len(second[0]) or any(
        not torch.equal(a, b) for a, b in zip(first[0], second[0])
    ):
        raise RuntimeError("paired data generation depends on ambient RNG consumption")
    if first[1:] != second[1:]:
        raise RuntimeError("paired split/shuffle depends on ambient RNG consumption")

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

        def reset_state(self):
            return None

    def eval_snapshot(ambient_draws: int):
        torch.manual_seed(9)
        _ = torch.randn(ambient_draws)
        set_active_pairing(data_seed, eval_seed)
        benchmark = experiment_main.select_benchmark(benchmark_cfg)
        return paired_eval_run(
            Identity(), benchmark, device="cpu", rollout_steps=3, batch_size=8
        )

    eval_a = eval_snapshot(5)
    eval_b = eval_snapshot(157)
    if eval_a != eval_b:
        raise RuntimeError("paired evaluation depends on ambient RNG consumption")

    return {
        "data_tensor_identity": True,
        "split_identity": True,
        "shuffle_identity": True,
        "evaluation_identity": True,
        "checked_data_seed": data_seed,
        "checked_eval_seed": eval_seed,
    }


def grouped_summary(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["kind"] == kind and row["status"] == "success":
            groups[(row["benchmark"], row["arm"])].append(row)
    out: dict[str, Any] = {}
    for (benchmark, arm), group in sorted(groups.items()):
        for metric in ("rollout_mse", "rollout_mae"):
            values = [float(row[metric]) for row in group]
            out[f"{benchmark}/{arm}/{metric}"] = {
                "n": len(values),
                "mean": statistics.mean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values_by_seed": {str(row["seed"]): row[metric] for row in group},
            }
    return out


def paired_contrast(
    rows: list[dict[str, Any]], *, kind: str, benchmark: str, a: str, b: str, metric: str
) -> dict[str, Any]:
    lookup = {
        (row["seed"], row["arm"]): row
        for row in rows
        if row["kind"] == kind and row["benchmark"] == benchmark and row["status"] == "success"
    }
    deltas: list[float] = []
    a_values: list[float] = []
    b_values: list[float] = []
    wins = ties = losses = 0
    missing: list[int] = []
    for seed in SEEDS:
        ra = lookup.get((seed, a))
        rb = lookup.get((seed, b))
        if ra is None or rb is None:
            missing.append(seed)
            continue
        av, bv = float(ra[metric]), float(rb[metric])
        delta = av - bv
        a_values.append(av)
        b_values.append(bv)
        deltas.append(delta)
        if delta < 0:
            wins += 1
        elif delta > 0:
            losses += 1
        else:
            ties += 1
    if not deltas:
        return {
            "kind": kind,
            "benchmark": benchmark,
            "metric": metric,
            "a": a,
            "b": b,
            "missing_seeds": missing,
        }
    a_mean = statistics.mean(a_values)
    b_mean = statistics.mean(b_values)
    relative_improvement = (b_mean - a_mean) / b_mean if b_mean != 0 else 0.0
    return {
        "kind": kind,
        "benchmark": benchmark,
        "metric": metric,
        "a": a,
        "b": b,
        "paired_deltas_a_minus_b": deltas,
        "mean_delta": statistics.mean(deltas),
        "sd_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "a_mean": a_mean,
        "b_mean": b_mean,
        "relative_improvement_a_vs_b": relative_improvement,
        "a_wins": wins,
        "ties": ties,
        "a_losses": losses,
        "missing_seeds": missing,
    }


def write_outputs(rows: list[dict[str, Any]], self_check: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    expected_mechanism = len(BENCHMARKS) * len(SEEDS) * len(VARIANTS)
    expected_baseline = len(BENCHMARKS) * len(SEEDS) * len(BASELINES)
    failures = [row for row in rows if row["status"] != "success"]
    complete = len(rows) == expected_mechanism + expected_baseline and not failures

    manifest = {
        "schema_version": 2,
        "created_utc": utcnow(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(Path(__file__)),
        "git_commit": git_commit(),
        "git_status_before_results": git_status(),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "scientific_boundary": (
            "Fresh paired synthetic replication/triage only. Prior mixed/adverse and near-null evidence remains. "
            "No failed cell, baseline win, null, or adverse result may be omitted or rescue-tuned under this protocol."
        ),
        "rng_contract": {
            "benchmark_offsets": BENCHMARK_OFFSETS,
            "data_seed_formula": "1000000 + benchmark_offset + seed",
            "split_seed_formula": "data_seed + 1",
            "shuffle_seed_formula": "data_seed + 2",
            "eval_seed_formula": "2000000 + benchmark_offset + seed",
            "model_seed_formula": "seed",
        },
        "self_check": self_check,
        "frozen": {
            "seeds": SEEDS,
            "benchmarks": BENCHMARKS,
            "variants": VARIANTS,
            "baselines": BASELINES,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "dataset_size": DATASET_SIZE,
            "train_rollout_steps": TRAIN_ROLLOUT_STEPS,
            "eval_rollout_steps": EVAL_ROLLOUT_STEPS,
            "hidden": HIDDEN,
            "trace_dim": TRACE_DIM,
            "primary_metric": "rollout_mse",
            "minimum_relative_improvement": 0.02,
            "minimum_seed_wins": 4,
        },
        "expected_cells": {"mechanism": expected_mechanism, "baseline": expected_baseline},
        "complete": complete,
        "failures": failures,
        "runs": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    mechanism_summary = grouped_summary(rows, "mechanism")
    baseline_summary = grouped_summary(rows, "baseline")
    contrasts: list[dict[str, Any]] = []
    for benchmark in BENCHMARKS:
        for comparator in ("no_retrieval", "no_memory", "no_salience_gating"):
            for metric in ("rollout_mse", "rollout_mae"):
                contrasts.append(
                    paired_contrast(
                        rows,
                        kind="mechanism",
                        benchmark=benchmark,
                        a="full",
                        b=comparator,
                        metric=metric,
                    )
                )
        for comparator in ("transformer", "ssm", "deeponet", "mlp"):
            for metric in ("rollout_mse", "rollout_mae"):
                contrasts.append(
                    paired_contrast(
                        rows,
                        kind="baseline",
                        benchmark=benchmark,
                        a="fim_plus",
                        b=comparator,
                        metric=metric,
                    )
                )

    primary = paired_contrast(
        rows,
        kind="mechanism",
        benchmark="delayed_recall",
        a="full",
        b="no_retrieval",
        metric="rollout_mse",
    )
    primary_passed = bool(
        not primary.get("missing_seeds")
        and primary.get("relative_improvement_a_vs_b", float("-inf")) >= 0.02
        and primary.get("a_wins", 0) >= 4
    )
    primary["minimum_relative_improvement"] = 0.02
    primary["minimum_seed_wins"] = 4
    primary["advancement_rule_passed"] = primary_passed

    summary = {
        "schema_version": 1,
        "complete": complete,
        "failure_count": len(failures),
        "primary_retrieval_replication": primary,
        "mechanism_summary": mechanism_summary,
        "baseline_summary": baseline_summary,
        "contrasts": contrasts,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# FIM fresh paired multi-seed paper evidence — protocol v2",
        "",
        "Generated only from the retained v2 manifest. This is bounded synthetic replication/triage evidence, not a broad-superiority or SOTA claim.",
        "",
        f"- Matrix complete: **{complete}**",
        f"- Failed cells: **{len(failures)}**",
        f"- Paired RNG self-check passed: **{all(bool(v) for k, v in self_check.items() if k.endswith('_identity'))}**",
        "",
        "## Primary retrieval-feedback replication",
        "",
    ]
    if "a_mean" in primary:
        lines += [
            f"- Full delayed-recall rollout MSE mean: `{primary['a_mean']:.8g}`",
            f"- No-retrieval delayed-recall rollout MSE mean: `{primary['b_mean']:.8g}`",
            f"- Full relative improvement: `{100 * primary['relative_improvement_a_vs_b']:.3f}%`",
            f"- Paired directions (full wins / ties / losses): `{primary['a_wins']} / {primary['ties']} / {primary['a_losses']}`",
            f"- Frozen advancement rule passed: **{primary_passed}**",
        ]
    else:
        lines.append(f"- Primary contrast incomplete; missing seeds: `{primary.get('missing_seeds', SEEDS)}`")
    lines += [
        "",
        "The frozen rule requires >=2% mean relative MSE improvement and >=4/5 paired-seed wins. A failed rule is retained as a null/adverse replication outcome and does not authorize rescue tuning.",
        "",
        "## Mechanism means",
        "",
        "| Benchmark | Arm | MSE mean | MSE SD | MAE mean | MAE SD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for benchmark in BENCHMARKS:
        for arm in VARIANTS:
            mse = mechanism_summary.get(f"{benchmark}/{arm}/rollout_mse")
            mae = mechanism_summary.get(f"{benchmark}/{arm}/rollout_mae")
            if mse and mae:
                lines.append(
                    f"| {benchmark} | {arm} | {mse['mean']:.8g} | {mse['sd']:.8g} | {mae['mean']:.8g} | {mae['sd']:.8g} |"
                )
    lines += [
        "",
        "## Maintained baseline means",
        "",
        "| Benchmark | Arm | Params (seed-specific in manifest) | MSE mean | MSE SD | MAE mean | MAE SD |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for benchmark in BENCHMARKS:
        for arm in BASELINES:
            mse = baseline_summary.get(f"{benchmark}/{arm}/rollout_mse")
            mae = baseline_summary.get(f"{benchmark}/{arm}/rollout_mae")
            if mse and mae:
                lines.append(
                    f"| {benchmark} | {arm} | see manifest | {mse['mean']:.8g} | {mse['sd']:.8g} | {mae['mean']:.8g} | {mae['sd']:.8g} |"
                )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "All model/ablation cells use paired synthetic data, split/shuffle streams, and evaluation initial conditions within benchmark+seed. Parameter counts are not matched and are retained in the manifest. Five seeds are descriptive here; no tiny-n significance claim is authorized. Historical paper-reference values remain historical unless separately reproduced.",
        "",
    ]
    (OUT / "paper_evidence.md").write_text("\n".join(lines), encoding="utf-8")
    return summary, complete


def main() -> int:
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    OUT.mkdir(parents=True, exist_ok=True)
    install_pairing_hooks()
    check = pairing_self_check()
    (OUT / "pairing_self_check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []

    # Mechanism matrix. All labels are resolved before first outcome-bearing cell.
    resolved_switches = {variant: variant_switches(variant) for variant in VARIANTS}
    for benchmark in BENCHMARKS:
        for seed in SEEDS:
            for variant in VARIANTS:
                print(f"MECHANISM benchmark={benchmark} seed={seed} variant={variant}", flush=True)
                row = execute_cell(
                    kind="mechanism",
                    benchmark=benchmark,
                    seed=seed,
                    arm=variant,
                    model="fim_plus",
                    switches=resolved_switches[variant],
                )
                rows.append(row)
                (OUT / "manifest.partial.json").write_text(json.dumps({"runs": rows}, indent=2) + "\n", encoding="utf-8")

    # Maintained baselines under the same paired-data/evaluation contract.
    experiment_main.FIMSystem = _CANONICAL_FIM_SYSTEM
    for benchmark in BENCHMARKS:
        for seed in SEEDS:
            for model in BASELINES:
                print(f"BASELINE benchmark={benchmark} seed={seed} model={model}", flush=True)
                row = execute_cell(
                    kind="baseline",
                    benchmark=benchmark,
                    seed=seed,
                    arm=model,
                    model=model,
                    switches=None,
                )
                rows.append(row)
                (OUT / "manifest.partial.json").write_text(json.dumps({"runs": rows}, indent=2) + "\n", encoding="utf-8")

    _, complete = write_outputs(rows, check)
    print(
        f"WROTE v2 paired evidence: cells={len(rows)} failures={sum(r['status'] != 'success' for r in rows)} complete={complete}",
        flush=True,
    )
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
