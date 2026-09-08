#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BENCHMARKS = ["delayed_recall", "lorenz96"]
EXPECTED_SEEDS = [101, 211, 307, 401, 503]
EXPECTED_VARIANTS = ["full", "no_memory", "no_retrieval", "no_salience_gating"]


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
    if data.get("protocol_name") != "FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1":
        raise ValueError("Unexpected protocol_name")
    if data.get("benchmarks") != EXPECTED_BENCHMARKS:
        raise ValueError(f"Benchmark set differs from frozen protocol: {data.get('benchmarks')}")
    if data.get("seeds") != EXPECTED_SEEDS:
        raise ValueError(f"Seed set differs from frozen protocol: {data.get('seeds')}")
    if data.get("variants") != EXPECTED_VARIANTS:
        raise ValueError(f"Variant set differs from frozen protocol: {data.get('variants')}")
    if data.get("trajectory_isolation", {}).get("batch_size") != 1:
        raise ValueError("Manifest is not trajectory-isolated")

    rows = data.get("runs", [])
    expected = len(EXPECTED_BENCHMARKS) * len(EXPECTED_SEEDS) * len(EXPECTED_VARIANTS)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, got {len(rows)}")
    if any(row.get("status") != "success" for row in rows):
        failures = [row for row in rows if row.get("status") != "success"]
        raise ValueError(f"Manifest contains {len(failures)} failed cells; analysis gate remains closed")

    cells = [(r["benchmark"], int(r["seed"]), r["variant"]) for r in rows]
    if len(set(cells)) != expected:
        raise ValueError("Duplicate or missing benchmark/seed/variant cells")

    commits = {r.get("git_commit") for r in rows}
    protocols = {r.get("protocol_sha256") for r in rows}
    if len(commits) != 1 or None in commits:
        raise ValueError(f"Expected one exact git commit, saw {commits}")
    if len(protocols) != 1 or None in protocols:
        raise ValueError("Expected one exact protocol hash across all cells")
    if protocols != {data.get("protocol_sha256")}:
        raise ValueError("Run protocol hash does not match manifest protocol hash")

    for row in rows:
        if int(row.get("batch_size", 0)) != 1:
            raise ValueError("A run was not executed with batch_size=1")
        if not row.get("validation_memory_enabled") or not row.get("evaluation_memory_enabled"):
            raise ValueError("Validation/test memory semantics are not aligned")
        for metric in ("rollout_mse", "final_step_mse", "rollout_mae", "final_step_mae"):
            if metric not in row or not math.isfinite(float(row[metric])):
                raise ValueError(f"Missing/non-finite {metric} in {cells[rows.index(row)]}")
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

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = validate_manifest(data)
    comparisons = build_comparisons(rows)
    write_csv(args.csv, comparisons)

    table = render_table(comparisons)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(
        "<!-- GENERATED from results/trajectory_isolated_v1/manifest.json; do not hand-edit numerical values. -->\n\n"
        + table,
        encoding="utf-8",
    )

    evidence = [
        "# Trajectory-isolated FIM evidence\n",
        "Status: generated only after the frozen 40-cell manifest passes completeness and provenance checks.\n",
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
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text("\n".join(evidence), encoding="utf-8")
    print(f"validated {len(rows)} cells; wrote {args.csv}, {args.table}, {args.evidence}")


if __name__ == "__main__":
    main()
