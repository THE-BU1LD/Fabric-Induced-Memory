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
PROTOCOL = ROOT / "research" / "protocols" / "FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1.md"
EXPECTED_BENCHMARKS = ["delayed_recall", "lorenz96"]
EXPECTED_SEEDS = [101, 211, 307, 401, 503]
EXPECTED_VARIANTS = ["full", "no_memory", "no_retrieval", "no_salience_gating"]
EXPECTED_BUDGET = {
    "epochs": 4,
    "dataset_size": 32,
    "batch_size": 1,
    "train_rollout_steps": 16,
    "eval_rollout_steps": 20,
}
EXPECTED_ISOLATION = {
    "batch_size": 1,
    "memory_reset_between_independent_episodes": True,
    "validation_memory_enabled_within_episode": True,
    "test_memory_enabled_within_episode": True,
    "checkpoint_selection": "fixed_final_epoch_no_best_selection",
    "ema": False,
}
# These are the frozen arm semantics, kept independent of the train/eval imports.
EXPECTED_SWITCHES = {
    "full": (True, True, True),
    "no_memory": (False, False, False),
    "no_retrieval": (True, False, True),
    "no_salience_gating": (True, True, False),
}
SWITCH_NAMES = ("memory_enabled", "retrieval_enabled", "salience_gating_enabled")


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


def validate_manifest(data: dict) -> list[dict]:
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a JSON object")
    if data.get("protocol_name") != "FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1":
        raise ValueError("Unexpected protocol_name")
    if data.get("benchmarks") != EXPECTED_BENCHMARKS:
        raise ValueError(f"Benchmark set differs from frozen protocol: {data.get('benchmarks')}")
    if data.get("seeds") != EXPECTED_SEEDS:
        raise ValueError(f"Seed set differs from frozen protocol: {data.get('seeds')}")
    if any(type(seed) is not int for seed in data["seeds"]):
        raise ValueError("Frozen seeds must be integers, without coercion")
    if data.get("variants") != EXPECTED_VARIANTS:
        raise ValueError(f"Variant set differs from frozen protocol: {data.get('variants')}")
    budget = data.get("budget")
    if not isinstance(budget, dict) or budget != EXPECTED_BUDGET or any(
        type(budget[key]) is not int for key in EXPECTED_BUDGET
    ):
        raise ValueError(f"Execution budget differs from frozen protocol: {data.get('budget')}")
    isolation = data.get("trajectory_isolation")
    if not isinstance(isolation, dict) or any(
        type(isolation.get(key)) is not type(expected) or isolation.get(key) != expected
        for key, expected in EXPECTED_ISOLATION.items()
    ):
        raise ValueError("Trajectory-isolation or checkpoint-selection semantics differ from frozen protocol")
    if data.get("git_dirty") is not False:
        raise ValueError(f"Paper-facing analysis requires a clean executed tree; git_dirty={data.get('git_dirty')}")
    commit = data.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Manifest git_commit must be one full 40-character commit SHA")
    try:
        expected_protocol_hash = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"Cannot verify the frozen protocol bytes: {PROTOCOL}") from exc
    if data.get("protocol_sha256") != expected_protocol_hash:
        raise ValueError("Manifest protocol hash does not match the frozen protocol bytes")

    rows = data.get("runs", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Manifest runs must be a list of JSON objects")
    expected = len(EXPECTED_BENCHMARKS) * len(EXPECTED_SEEDS) * len(EXPECTED_VARIANTS)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, got {len(rows)}")
    if any(row.get("status") != "success" for row in rows):
        failures = [row for row in rows if row.get("status") != "success"]
        raise ValueError(f"Manifest contains {len(failures)} failed cells; analysis gate remains closed")

    expected_cells = set(itertools.product(EXPECTED_BENCHMARKS, EXPECTED_SEEDS, EXPECTED_VARIANTS))
    cells = []
    for row in rows:
        if (row.get("benchmark") not in EXPECTED_BENCHMARKS
                or type(row.get("seed")) is not int or row["seed"] not in EXPECTED_SEEDS
                or row.get("variant") not in EXPECTED_VARIANTS):
            raise ValueError("A run contains an unexpected benchmark/seed/variant cell")
        cells.append((row["benchmark"], row["seed"], row["variant"]))
    if len(set(cells)) != expected or set(cells) != expected_cells:
        raise ValueError("Duplicate or missing benchmark/seed/variant cells")

    commits = {r.get("git_commit") for r in rows}
    protocols = {r.get("protocol_sha256") for r in rows}
    if len(commits) != 1 or None in commits:
        raise ValueError(f"Expected one exact git commit, saw {commits}")
    if commits != {data.get("git_commit")}:
        raise ValueError("Run commit does not match manifest commit")
    if len(protocols) != 1 or None in protocols:
        raise ValueError("Expected one exact protocol hash across all cells")
    if protocols != {data.get("protocol_sha256")}:
        raise ValueError("Run protocol hash does not match manifest protocol hash")

    for row in rows:
        for field, expected_value in EXPECTED_BUDGET.items():
            if type(row.get(field)) is not int or row[field] != expected_value:
                raise ValueError(f"A run used the wrong {field} budget (integer required)")
        for field, expected_value in zip(SWITCH_NAMES, EXPECTED_SWITCHES[row["variant"]]):
            if row.get(field) is not expected_value:
                raise ValueError(f"A run's {field} switch does not match variant {row['variant']!r}")
        if row.get("validation_memory_enabled") is not True or row.get("evaluation_memory_enabled") is not True:
            raise ValueError("Validation/test memory semantics are not aligned")
        for metric in ("rollout_mse", "final_step_mse", "rollout_mae", "final_step_mae"):
            value = row.get(metric)
            try:
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"Missing/invalid nonnegative finite {metric} in {(row['benchmark'], row['seed'], row['variant'])}")
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
    data = json.loads(manifest_bytes.decode("utf-8"))
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
