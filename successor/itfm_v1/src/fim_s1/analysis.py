from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .provenance import atomic_json, atomic_text, sha256_file


def exact_sign_flip_p(deltas: list[float]) -> float:
    nonzero = [value for value in deltas if value != 0.0]
    if not nonzero:
        return 1.0
    observed = abs(mean(nonzero))
    extreme = 0
    total = 2 ** len(nonzero)
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        statistic = abs(mean(sign * value for sign, value in zip(signs, nonzero)))
        extreme += int(statistic >= observed - 1e-15)
    return extreme / total


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _load_rows(output: Path, run: dict[str, Any]) -> list[dict[str, Any]]:
    path = output / run["predictions"]
    if sha256_file(path) != run["predictions_sha256"]:
        raise RuntimeError(f"prediction hash mismatch for {run['run_id']}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["rows"]


def _family_alias_mse(rows: list[dict[str, Any]]) -> dict[int, float]:
    by_family: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["aliased"]:
            by_family[int(row["family_id"])].append(
                float(row["squared_error"]) / int(row["element_count"])
            )
    return {family: mean(values) for family, values in by_family.items() if values}


def hierarchical_bootstrap_delta(
    proposed: dict[int, dict[int, float]],
    control: dict[int, dict[int, float]],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    shared_seeds = sorted(set(proposed) & set(control))
    if not shared_seeds:
        raise ValueError("no shared seeds for bootstrap")
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sampled_seeds = [rng.choice(shared_seeds) for _ in shared_seeds]
        seed_deltas = []
        for run_seed in sampled_seeds:
            families = sorted(set(proposed[run_seed]) & set(control[run_seed]))
            if not families:
                continue
            sampled_families = [rng.choice(families) for _ in families]
            seed_deltas.append(
                mean(
                    control[run_seed][family] - proposed[run_seed][family]
                    for family in sampled_families
                )
            )
        if seed_deltas:
            draws.append(mean(seed_deltas))
    if not draws:
        raise ValueError("bootstrap produced no draws")
    draws.sort()
    lower = draws[int(0.025 * (len(draws) - 1))]
    upper = draws[int(0.975 * (len(draws) - 1))]
    return lower, upper


def _write_svg(path: Path, model_means: dict[str, float]) -> None:
    width = 760
    height = 90 + 55 * len(model_means)
    maximum = max(model_means.values()) if model_means else 1.0
    bars = []
    for index, (name, value) in enumerate(sorted(model_means.items())):
        y = 55 + index * 55
        bar_width = 500 * value / max(maximum, 1e-12)
        bars.append(
            f'<text x="10" y="{y + 18}" font-size="14">{name}</text>'
            f'<rect x="190" y="{y}" width="{bar_width:.2f}" height="24" fill="#315b7d"/>'
            f'<text x="{200 + bar_width:.2f}" y="{y + 18}" font-size="13">{value:.6g}</text>'
        )
    content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="10" y="28" font-size="19" font-weight="bold">Alias-conditioned NMSE</text>'
        + "".join(bars)
        + "</svg>"
    )
    atomic_text(path, content)


def analyze(output: Path, resamples: int = 10_000) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete" or manifest.get("failures"):
        raise RuntimeError("analysis requires a complete failure-free manifest")
    if manifest.get("completed_runs") != manifest.get("expected_runs"):
        raise RuntimeError("analysis requires every declared run cell")

    runs_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_metrics: dict[str, dict[int, dict[int, float]]] = defaultdict(dict)
    for run in manifest["runs"]:
        runs_by_model[run["model"]].append(run)
        family_metrics[run["model"]][int(run["seed"])] = _family_alias_mse(
            _load_rows(output, run)
        )

    aggregates: dict[str, dict[str, float | int]] = {}
    for model, runs in sorted(runs_by_model.items()):
        values = [float(run["validation"]["alias_conditioned_nmse"]) for run in runs]
        aggregates[model] = {
            "seeds": len(values),
            "alias_nmse_mean": mean(values),
            "alias_nmse_sd": stdev(values) if len(values) > 1 else 0.0,
            "alias_nmse_min": min(values),
            "alias_nmse_max": max(values),
            "parameters_mean": mean(float(run["parameters"]) for run in runs),
            "runtime_seconds_mean": mean(float(run["runtime_seconds"]) for run in runs),
            "peak_memory_bytes_max": max(
                int(run["process_peak_memory_bytes"]) for run in runs
            ),
        }

    primary_model = str(manifest.get("primary_model", "itfm"))
    if primary_model not in runs_by_model:
        raise RuntimeError(f"complete manifest lacks primary model {primary_model}")
    comparisons: dict[str, dict[str, Any]] = {}
    raw_p: dict[str, float] = {}
    proposed_by_seed = {
        int(run["seed"]): float(run["validation"]["alias_conditioned_nmse"])
        for run in runs_by_model[primary_model]
    }
    declared_controls = {
        "no_memory",
        "vector_broadcast",
        "finite_history_cnn",
        "conv_gru",
        "sham_transport",
        "itfm",
        "periodic_xcorr",
        "gated_diagonal_ssm",
    }
    for model, runs in sorted(runs_by_model.items()):
        if model not in declared_controls:
            continue
        control_by_seed = {
            int(run["seed"]): float(run["validation"]["alias_conditioned_nmse"])
            for run in runs
        }
        shared = sorted(set(proposed_by_seed) & set(control_by_seed))
        deltas = [control_by_seed[seed] - proposed_by_seed[seed] for seed in shared]
        lower, upper = hierarchical_bootstrap_delta(
            family_metrics[primary_model],
            family_metrics[model],
            resamples=resamples,
            seed=20261011,
        )
        p_value = exact_sign_flip_p(deltas)
        raw_p[model] = p_value
        comparisons[model] = {
            "control_minus_primary_mean": mean(deltas),
            "seed_deltas": deltas,
            "bootstrap_95_interval": [lower, upper],
            "exact_sign_flip_p": p_value,
        }
    adjusted = holm_adjust(raw_p)
    for model, value in adjusted.items():
        comparisons[model]["holm_p"] = value

    controls = [name for name in aggregates if name in declared_controls]
    if set(controls) != declared_controls:
        missing = sorted(declared_controls - set(controls))
        raise RuntimeError(f"complete manifest lacks declared controls: {missing}")
    strongest = min(controls, key=lambda name: aggregates[name]["alias_nmse_mean"])
    primary_mean = float(aggregates[primary_model]["alias_nmse_mean"])
    control_mean = float(aggregates[strongest]["alias_nmse_mean"])
    relative_improvement = (control_mean - primary_mean) / max(control_mean, 1e-12)

    primary_runs = runs_by_model[primary_model]
    erasure_deltas = [
        float(run["interventions"]["erase"]["alias_conditioned_nmse"])
        - float(run["interventions"]["none"]["alias_conditioned_nmse"])
        for run in primary_runs
    ]
    random_deltas = [
        float(run["interventions"]["random_value"]["alias_conditioned_nmse"])
        - float(run["interventions"]["none"]["alias_conditioned_nmse"])
        for run in primary_runs
    ]
    swap_scores = [
        float(run["interventions"]["swap_donor_direction"]["mean"])
        for run in primary_runs
        if run["interventions"]["swap_donor_direction"]["mean"] is not None
    ]
    threshold = 0.05
    max_non_alias_harm = 0.02
    strongest_runs = {int(run["seed"]): run for run in runs_by_model[strongest]}
    non_alias_harms = []
    for run in primary_runs:
        seed = int(run["seed"])
        if seed in strongest_runs:
            baseline = float(strongest_runs[seed]["validation"]["non_alias_nmse"])
            proposed = float(run["validation"]["non_alias_nmse"])
            non_alias_harms.append((proposed - baseline) / max(baseline, 1e-12))

    gates = {
        "practical_improvement": relative_improvement >= threshold,
        "erasure_direction": bool(erasure_deltas) and mean(erasure_deltas) > 0,
        "swap_direction": bool(swap_scores) and mean(swap_scores) > 0,
        "random_specificity": bool(random_deltas) and mean(random_deltas) > 0,
        "non_alias_harm": bool(non_alias_harms)
        and mean(non_alias_harms) <= max_non_alias_harm,
    }
    if manifest.get("smoke"):
        decision = "SMOKE_ONLY_NOT_SCIENTIFIC"
    else:
        decision = (
            "ADVANCE_TO_CONFIRMATORY_DESIGN"
            if all(gates.values())
            else "V1_NOT_SUPPORTED"
        )
    analysis = {
        "schema_version": 1,
        "manifest_sha256": sha256_file(manifest_path),
        "aggregates": aggregates,
        "comparisons": comparisons,
        "primary_model": primary_model,
        "strongest_control": strongest,
        "relative_improvement_vs_strongest": relative_improvement,
        "mechanism": {
            "erasure_alias_nmse_delta_mean": mean(erasure_deltas),
            "random_alias_nmse_delta_mean": mean(random_deltas),
            "swap_donor_direction_mean": mean(swap_scores) if swap_scores else None,
            "non_alias_relative_harm_mean": (
                mean(non_alias_harms) if non_alias_harms else None
            ),
        },
        "gates": gates,
        "decision": decision,
        "claim_boundary": "Development evidence only; no confirmatory or publication claim.",
    }
    atomic_json(output / "analysis.json", analysis)

    summary_csv = output / "summary.csv"
    temporary_csv = summary_csv.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "seeds", "alias_nmse_mean", "alias_nmse_sd", "parameters_mean"]
        )
        for model, values in sorted(aggregates.items()):
            writer.writerow(
                [
                    model,
                    values["seeds"],
                    values["alias_nmse_mean"],
                    values["alias_nmse_sd"],
                    values["parameters_mean"],
                ]
            )
    temporary_csv.replace(summary_csv)
    _write_svg(
        output / "alias_nmse.svg",
        {name: float(values["alias_nmse_mean"]) for name, values in aggregates.items()},
    )
    lines = [
        "# ITFM development analysis",
        "",
        f"Decision: **{decision}**",
        "",
        "| Model | Seeds | Alias NMSE mean | SD | Parameters |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, values in sorted(aggregates.items()):
        lines.append(
            f"| {model} | {values['seeds']} | {values['alias_nmse_mean']:.8g} | "
            f"{values['alias_nmse_sd']:.8g} | {values['parameters_mean']:.0f} |"
        )
    lines.extend(
        [
            "",
            f"Strongest control: `{strongest}`.",
            f"Relative improvement versus strongest: {relative_improvement:.3%}.",
            "",
            "This is development evidence only. Failed gates are retained and must not be tuned away.",
        ]
    )
    atomic_text(output / "SUMMARY.md", "\n".join(lines) + "\n")
    artifacts = {
        name: sha256_file(output / name)
        for name in ("analysis.json", "summary.csv", "alias_nmse.svg", "SUMMARY.md")
    }
    atomic_json(
        output / "analysis_artifacts.json",
        {
            "schema_version": 1,
            "manifest_sha256": sha256_file(manifest_path),
            "artifacts": artifacts,
        },
    )
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    result = analyze(args.output, resamples=args.resamples)
    print(
        json.dumps(
            {"decision": result["decision"], "gates": result["gates"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
