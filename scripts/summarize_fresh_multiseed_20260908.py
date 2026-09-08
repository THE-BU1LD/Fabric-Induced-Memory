#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fresh_multiseed_20260908"
SEEDS = [101, 202, 303, 404, 505]
BENCHMARKS = ["lorenz96", "delayed_recall"]
VARIANTS = ["full", "no_memory", "no_retrieval", "no_salience_gating"]
MODELS = ["fim_plus", "transformer", "ssm", "deeponet", "mlp"]


def finite_number(value):
    if not isinstance(value, (int, float)):
        raise ValueError(f"expected numeric metric, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric: {value}")
    return value


def summarize(values):
    values = [finite_number(v) for v in values]
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def load_mechanism_rows():
    path = OUT / "ablations" / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("runs", [])
    expected = {(b, s, v) for b in BENCHMARKS for s in SEEDS for v in VARIANTS}
    observed = {(r["benchmark"], int(r["seed"]), r["variant"]) for r in rows}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"mechanism matrix incomplete: missing={missing} extra={extra}")
    for row in rows:
        finite_number(row["rollout_mse"])
        finite_number(row["rollout_mae"])
    return rows


def load_baseline_rows():
    rows = []
    for seed in SEEDS:
        path = OUT / "baselines" / f"baseline_seed_{seed}" / "suite_summary.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for run in data.get("runs", []):
            result = run.get("result", {})
            rows.append({
                "seed": seed,
                "benchmark": run["benchmark"],
                "model": run["model"],
                "rollout_mse": finite_number(result["rollout_mse"]),
                "rollout_mae": finite_number(result["rollout_mae"]),
            })
    expected = {(b, s, m) for b in BENCHMARKS for s in SEEDS for m in MODELS}
    observed = {(r["benchmark"], r["seed"], r["model"]) for r in rows}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"baseline matrix incomplete: missing={missing} extra={extra}")
    return rows


def paired_contrast(rows, key_name, a, b, benchmark, metric):
    by = {(r["seed"], r[key_name]): r for r in rows if r["benchmark"] == benchmark}
    deltas = []
    a_values = []
    b_values = []
    wins = ties = losses = 0
    for seed in SEEDS:
        av = finite_number(by[(seed, a)][metric])
        bv = finite_number(by[(seed, b)][metric])
        d = av - bv
        a_values.append(av)
        b_values.append(bv)
        deltas.append(d)
        if d < 0:
            wins += 1
        elif d > 0:
            losses += 1
        else:
            ties += 1
    mean_a = statistics.mean(a_values)
    mean_b = statistics.mean(b_values)
    relative_improvement = (mean_b - mean_a) / mean_b if mean_b != 0 else 0.0
    return {
        "benchmark": benchmark,
        "metric": metric,
        "a": a,
        "b": b,
        "paired_deltas_a_minus_b": deltas,
        "mean_delta": statistics.mean(deltas),
        "sd_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "a_mean": mean_a,
        "b_mean": mean_b,
        "relative_improvement_a_vs_b": relative_improvement,
        "a_wins": wins,
        "ties": ties,
        "a_losses": losses,
    }


def main():
    mechanism = load_mechanism_rows()
    baselines = load_baseline_rows()

    mechanism_groups = defaultdict(list)
    for r in mechanism:
        mechanism_groups[(r["benchmark"], r["variant"], "rollout_mse")].append(r["rollout_mse"])
        mechanism_groups[(r["benchmark"], r["variant"], "rollout_mae")].append(r["rollout_mae"])

    baseline_groups = defaultdict(list)
    for r in baselines:
        baseline_groups[(r["benchmark"], r["model"], "rollout_mse")].append(r["rollout_mse"])
        baseline_groups[(r["benchmark"], r["model"], "rollout_mae")].append(r["rollout_mae"])

    mech_summary = {
        f"{b}/{v}/{metric}": summarize(values)
        for (b, v, metric), values in sorted(mechanism_groups.items())
    }
    baseline_summary = {
        f"{b}/{m}/{metric}": summarize(values)
        for (b, m, metric), values in sorted(baseline_groups.items())
    }

    contrasts = []
    for benchmark in BENCHMARKS:
        for comparator in ["no_retrieval", "no_memory", "no_salience_gating"]:
            for metric in ["rollout_mse", "rollout_mae"]:
                contrasts.append(paired_contrast(mechanism, "variant", "full", comparator, benchmark, metric))
        for comparator in ["transformer", "ssm", "deeponet", "mlp"]:
            for metric in ["rollout_mse", "rollout_mae"]:
                contrasts.append(paired_contrast(baselines, "model", "fim_plus", comparator, benchmark, metric))

    primary = paired_contrast(mechanism, "variant", "full", "no_retrieval", "delayed_recall", "rollout_mse")
    primary["minimum_mean_relative_improvement"] = 0.02
    primary["minimum_seed_wins"] = 4
    primary["advancement_rule_passed"] = (
        primary["relative_improvement_a_vs_b"] >= 0.02 and primary["a_wins"] >= 4
    )

    payload = {
        "schema_version": 1,
        "protocol": "research/FRESH_MULTI_SEED_REPLICATION_PROTOCOL_20260908.md",
        "scientific_boundary": (
            "Bounded fresh-seed replication/triage only. Historical paper-reference values are not fresh evidence. "
            "All baseline wins, null effects, adverse effects, and failed cells must be preserved."
        ),
        "expected_cells": {"mechanism": 40, "baseline": 50},
        "mechanism_summary": mech_summary,
        "baseline_summary": baseline_summary,
        "contrasts": contrasts,
        "primary_retrieval_replication": primary,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# FIM fresh multi-seed evidence summary — 2026-09-08",
        "",
        "This report is generated from retained fresh run artifacts. It is a bounded replication/triage result, not a state-of-the-art or general-superiority claim.",
        "",
        "## Primary retrieval-feedback replication",
        "",
        f"- Full mean delayed-recall rollout MSE: `{primary['a_mean']:.8g}`",
        f"- No-retrieval mean delayed-recall rollout MSE: `{primary['b_mean']:.8g}`",
        f"- Mean relative improvement of full: `{100*primary['relative_improvement_a_vs_b']:.3f}%`",
        f"- Seed directions (full wins / ties / losses): `{primary['a_wins']} / {primary['ties']} / {primary['a_losses']}`",
        f"- Predeclared bounded advancement rule passed: **{primary['advancement_rule_passed']}**",
        "",
        "The rule required >=2% mean relative MSE improvement and at least 4/5 paired-seed wins. Failure to pass is retained as a null/adverse replication outcome; it must not trigger rescue tuning under this protocol.",
        "",
        "## Mechanism means",
        "",
        "| Benchmark | Variant | MSE mean | MSE SD | MAE mean | MAE SD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for benchmark in BENCHMARKS:
        for variant in VARIANTS:
            mse = mech_summary[f"{benchmark}/{variant}/rollout_mse"]
            mae = mech_summary[f"{benchmark}/{variant}/rollout_mae"]
            lines.append(f"| {benchmark} | {variant} | {mse['mean']:.8g} | {mse['sd']:.8g} | {mae['mean']:.8g} | {mae['sd']:.8g} |")

    lines += [
        "",
        "## Maintained baseline means",
        "",
        "| Benchmark | Model | MSE mean | MSE SD | MAE mean | MAE SD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for benchmark in BENCHMARKS:
        for model in MODELS:
            mse = baseline_summary[f"{benchmark}/{model}/rollout_mse"]
            mae = baseline_summary[f"{benchmark}/{model}/rollout_mae"]
            lines.append(f"| {benchmark} | {model} | {mse['mean']:.8g} | {mse['sd']:.8g} | {mae['mean']:.8g} | {mae['sd']:.8g} |")

    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Five seeds are insufficient for strong significance claims here. Use paired deltas, direction counts, and effect magnitude descriptively. Do not promote historical `paper_reference_results.json` into current evidence. Do not hide a baseline win or an adverse ablation result.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"validated 40 mechanism + 50 baseline cells; wrote {OUT / 'summary.json'} and {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
