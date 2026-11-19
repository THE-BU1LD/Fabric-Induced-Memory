#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import random
import re
from pathlib import Path
from statistics import mean, median
from typing import Iterable

try:
    from scripts.evidence_publication import publish_evidence
except ModuleNotFoundError:
    from evidence_publication import publish_evidence

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BENCHMARKS = ["delayed_recall", "lorenz96"]
EXPECTED_SEEDS = [101, 211, 307, 401, 503]
EXPECTED_VARIANTS = ["full", "no_memory", "no_retrieval", "no_salience_gating"]
EXPECTED_SWITCHES = {
    "full": (True, True, True),
    "no_memory": (False, False, False),
    "no_retrieval": (True, False, True),
    "no_salience_gating": (True, True, False),
}
SWITCH_FIELDS = ("memory_enabled", "retrieval_enabled", "salience_gating_enabled")
EXPECTED_ISOLATION = {
    "batch_size": 1,
    "memory_reset_between_independent_episodes": True,
    "validation_memory_enabled_within_episode": True,
    "test_memory_enabled_within_episode": True,
    "checkpoint_selection": "fixed_final_epoch_no_best_selection",
    "ema": False,
}
EXPECTED_PROTOCOL_PATH = "research/protocols/FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1.md"
# Bind admission to the preserved frozen protocol, not merely matching hashes
# supplied by the manifest itself. Any successor needs a separate protocol.
EXPECTED_PROTOCOL_SHA256 = "637e0e6b3ea895bcbc14fc80ced6b12f420d6355de2e6440cb7bbfb482b880c9"
EXPECTED_BUDGET = {
    "epochs": 4,
    "dataset_size": 32,
    "batch_size": 1,
    "train_rollout_steps": 16,
    "eval_rollout_steps": 20,
}


def exact_sign_flip_p(deltas: list[float]) -> float:
    if not deltas:
        return float("nan")
    observed = abs(mean(deltas))
    magnitudes = [abs(x) for x in deltas]
    total = 0
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(magnitudes)):
        total += 1
        stat = abs(mean([s * x for s, x in zip(signs, magnitudes)]))
        if stat + 1e-15 >= observed:
            extreme += 1
    return extreme / total


def paired_bootstrap_ci(deltas: list[float], samples: int = 10000, seed: int = 20260908) -> tuple[float, float]:
    if not deltas:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(deltas)
    draws = []
    for _ in range(samples):
        draws.append(mean(deltas[rng.randrange(n)] for _ in range(n)))
    draws.sort()
    lo = draws[max(0, int(0.025 * samples) - 1)]
    hi = draws[min(samples - 1, int(0.975 * samples))]
    return lo, hi


def fmt(x: float) -> str:
    if math.isnan(x):
        return "NA"
    return f"{x:.6g}"


def _require_fixed_fields(values: dict, expected: dict, label: str) -> None:
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be an object")
    for field, frozen in expected.items():
        actual = values.get(field)
        # JSON booleans compare equal to 0/1 in Python. Scientific switches and
        # integer budgets must retain their declared types without coercion.
        if type(actual) is not type(frozen) or actual != frozen:
            raise ValueError(f"{label}.{field} differs from frozen protocol: {actual!r}")


def _unique_manifest_object(pairs: list[tuple[str, object]]) -> dict:
    """Reject ambiguous declarations before the JSON decoder discards a key."""
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"Duplicate manifest field: {key!r}")
        values[key] = value
    return values


def validate_manifest(data: dict) -> list[dict]:
    if not isinstance(data, dict):
        raise ValueError("Manifest must be an object")
    if data.get("protocol_name") != "FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1":
        raise ValueError("Unexpected protocol_name")
    expected_cells = set(itertools.product(EXPECTED_BENCHMARKS, EXPECTED_SEEDS, EXPECTED_VARIANTS))
    _require_fixed_fields(data, {
        "schema_version": 2,
        "protocol_path": EXPECTED_PROTOCOL_PATH,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "expected_cells": len(expected_cells),
    }, "Manifest")
    if data.get("benchmarks") != EXPECTED_BENCHMARKS:
        raise ValueError(f"Benchmark set differs from frozen protocol: {data.get('benchmarks')}")
    if data.get("seeds") != EXPECTED_SEEDS or any(type(seed) is not int for seed in data["seeds"]):
        raise ValueError(f"Seed set differs from frozen protocol: {data.get('seeds')}")
    if data.get("variants") != EXPECTED_VARIANTS:
        raise ValueError(f"Variant set differs from frozen protocol: {data.get('variants')}")
    if data.get("budget") != EXPECTED_BUDGET:
        raise ValueError(f"Execution budget differs from frozen protocol: {data.get('budget')}")
    _require_fixed_fields(data["budget"], EXPECTED_BUDGET, "Budget")
    _require_fixed_fields(data.get("trajectory_isolation"), EXPECTED_ISOLATION, "Trajectory isolation")
    if data.get("git_dirty") is not False:
        raise ValueError(f"Paper-facing analysis requires a clean executed tree; git_dirty={data.get('git_dirty')}")
    commit = data.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Manifest git_commit must be a concrete 40-character commit SHA")

    rows = data.get("runs", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Manifest runs must be a list of objects")
    if len(rows) != len(expected_cells):
        raise ValueError(f"Expected {len(expected_cells)} rows, got {len(rows)}")
    if any(row.get("status") != "success" for row in rows):
        failures = [row for row in rows if row.get("status") != "success"]
        raise ValueError(f"Manifest contains {len(failures)} failed cells; analysis gate remains closed")

    cells = []
    for index, row in enumerate(rows):
        benchmark, seed, variant = row.get("benchmark"), row.get("seed"), row.get("variant")
        if not isinstance(benchmark, str) or type(seed) is not int or not isinstance(variant, str):
            raise ValueError(f"Run {index} requires string benchmark/variant and integer seed")
        cell = (benchmark, seed, variant)
        if cell not in expected_cells:
            raise ValueError(f"Unexpected benchmark/seed/variant cell: {cell!r}")
        cells.append(cell)
        _require_fixed_fields(row, {
            **EXPECTED_BUDGET,
            **dict(zip(SWITCH_FIELDS, EXPECTED_SWITCHES[variant])),
            "git_commit": commit,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "validation_memory_enabled": True,
            "evaluation_memory_enabled": True,
        }, f"Run {cell!r}")
        for metric in ("rollout_mse", "final_step_mse", "rollout_mae", "final_step_mae"):
            value = row.get(metric)
            if type(value) not in (int, float) or value < 0 or not math.isfinite(value):
                raise ValueError(f"Missing/non-finite/non-negative numeric {metric} required in {cell!r}")
    if set(cells) != expected_cells or len(set(cells)) != len(cells):
        raise ValueError("Duplicate or missing benchmark/seed/variant cells")
    return rows


def build_comparisons(rows: list[dict]) -> list[dict]:
    lookup = {(r["benchmark"], int(r["seed"]), r["variant"]): r for r in rows}
    results: list[dict] = []
    for benchmark in EXPECTED_BENCHMARKS:
        full = [float(lookup[(benchmark, seed, "full")]["rollout_mse"]) for seed in EXPECTED_SEEDS]
        for variant in ("no_memory", "no_retrieval", "no_salience_gating"):
            other = [float(lookup[(benchmark, seed, variant)]["rollout_mse"]) for seed in EXPECTED_SEEDS]
            deltas = [f - o for f, o in zip(full, other)]
            lo, hi = paired_bootstrap_ci(deltas)
            results.append(
                {
                    "benchmark": benchmark,
                    "comparison": f"full_vs_{variant}",
                    "n_seeds": len(EXPECTED_SEEDS),
                    "full_mean_rollout_mse": mean(full),
                    "control_mean_rollout_mse": mean(other),
                    "mean_delta_full_minus_control": mean(deltas),
                    "median_delta_full_minus_control": median(deltas),
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                    "exact_sign_flip_p": exact_sign_flip_p(deltas),
                    "full_better_seed_count": sum(d < 0 for d in deltas),
                    "control_better_seed_count": sum(d > 0 for d in deltas),
                    "ties": sum(d == 0 for d in deltas),
                }
            )
    return results


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_csv(rows: Iterable[dict]) -> str:
    rows = list(rows)
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def render_table(comparisons: list[dict]) -> str:
    lines = [
        "| Benchmark | Comparison | n | Full MSE | Control MSE | Delta (full-control) | 95% paired bootstrap CI | Exact sign-flip p | Full-better seeds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comparisons:
        ci = f"[{fmt(r['bootstrap95_low'])}, {fmt(r['bootstrap95_high'])}]"
        lines.append(
            f"| {r['benchmark']} | {r['comparison']} | {r['n_seeds']} | "
            f"{fmt(r['full_mean_rollout_mse'])} | {fmt(r['control_mean_rollout_mse'])} | "
            f"{fmt(r['mean_delta_full_minus_control'])} | {ci} | "
            f"{fmt(r['exact_sign_flip_p'])} | {r['full_better_seed_count']}/{r['n_seeds']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "results" / "trajectory_isolated_v1" / "manifest.json")
    ap.add_argument("--csv", type=Path, default=ROOT / "results" / "trajectory_isolated_v1" / "paired_comparisons.csv")
    ap.add_argument("--table", type=Path, default=ROOT / "paper" / "generated" / "trajectory_isolated_ablation_table.md")
    ap.add_argument("--evidence", type=Path, default=ROOT / "research" / "TRAJECTORY_ISOLATED_EVIDENCE.md")
    args = ap.parse_args()

    manifest_bytes = args.manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    data = json.loads(manifest_bytes.decode("utf-8"), object_pairs_hook=_unique_manifest_object)
    rows = validate_manifest(data)
    comparisons = build_comparisons(rows)

    table = render_table(comparisons)
    source_note = (
        f"Source manifest: `{args.manifest.resolve()}`\n\n"
        f"Source manifest SHA-256: `{manifest_sha256}`\n\n"
    )
    table_content = (
        "<!-- GENERATED; do not hand-edit numerical values. -->\n\n"
        + source_note + table
    )

    evidence = [
        "# Trajectory-isolated FIM evidence\n",
        "Status: generated only after the frozen 40-cell manifest passes completeness, clean-tree, budget, and provenance checks.\n",
        source_note,
        f"Exact run commit: `{rows[0]['git_commit']}`\n",
        f"Protocol SHA-256: `{data['protocol_sha256']}`\n",
        "## Claim boundary\n",
        "This is a bounded current-code mechanism study with five paired seeds, not a broad publication-scale superiority claim. "
        "Historical shared-minibatch evidence remains historical and is not rewritten. Negative, null, or baseline-favoring results are retained.\n",
        "## Paired rollout-MSE comparisons\n",
        table,
        "## Interpretation rule\n",
        "A negative `full-control` delta means lower rollout MSE for full FIM. With five seeds, exact sign-flip inference has limited resolution; "
        "effect direction, interval width, and per-seed consistency should be emphasized over thresholded significance. Any architecture or tuning change after these outcomes requires a separately frozen protocol.\n",
    ]
    publish_evidence(
        [(args.csv, render_csv(comparisons)),
         (args.table, table_content),
         (args.evidence, "\n".join(evidence))],
        protected_inputs=[args.manifest],
    )
    print(f"validated {len(rows)} cells; wrote {args.csv}, {args.table}, {args.evidence}")


if __name__ == "__main__":
    main()
