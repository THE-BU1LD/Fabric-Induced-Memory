"""Stdlib-only analysis of validated, completed FIM engineering cells.

API: run(root, manifest, output). ``manifest`` may be a dict or JSON path.
The default ledger is root/run_ledger.json; entries are cells[id] dictionaries
with state and optional relative path/reason. Default cell path is cells/id.
Only SUCCEEDED entries passing the maintained completion validator are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
from xml.sax.saxutils import escape


STATES = {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "INTERRUPTED", "SKIPPED_BUDGET"}
VARIANTS = ("base", "memory", "erased", "spatial_mean", "last_residual")
COMPARISONS = ("base", "last_residual", "persistence")
LIMITATIONS = [
    "This is a synthetic engineering/development fixture, not a public benchmark or confirmatory study.",
    "Independent repetitions are seeds. Timesteps, trajectories within a seed, batches, and spatial points are not statistical replicates.",
    "Each seed trains one MLP; memory variants reuse its frozen checkpoint. Training time and parameter counts are shared, not separate matched compute trials.",
    "Memory scale is selected on validation only; test results are descriptive and must not be used to revise this protocol.",
    "Few seeds provide coarse sign-flip inference and unstable bootstrap intervals. Inconclusive differences do not establish equivalence.",
    "Exact sign-flip p-values require independent, sign-exchangeable paired effects under the null; that assumption is not established by this fixture.",
    "Autonomous base rollout is a separate finite-horizon diagnostic with memory disabled; it does not establish long-term stability or memory efficacy.",
    "No novelty, general architecture ranking, venue readiness, acceptance, or independent scientific replication is established.",
]


def _json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON value: {value}")))


def _digest(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _validator(root, config, source):
    # Maintained source is supplied by the verified extracted snapshot's sys.path.
    # engineering and provenance are stdlib-only; never import torch for analysis.
    from fim_sota.engineering import validate_completed
    return validate_completed(root, config, source)


def _number(value, label, nonnegative=True):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{label} cannot be negative")
    return float(value)


def _metric(value, expected_count=None):
    count = value["element_count"]
    if type(count) is not int or count < 1 or (expected_count is not None and count != expected_count):
        raise ValueError("invalid metric element count")
    total = _number(value["squared_error_sum"], "squared_error_sum")
    mse = _number(value["mse"], "mse")
    if not math.isclose(total / count, mse, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("MSE does not match retained sum/count")
    return mse


def _safe_cell_path(root, relative):
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("cell path must remain relative to the campaign root")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("cell path escaped campaign root")
    return resolved


def _read_events(path):
    return [_json_line(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _json_line(line):
    return json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite event: {value}")))


def _extract(cell, path, source):
    config = cell["config"]
    receipt, metrics = _validator(path, config, source)
    for name in ("source_sha256", "config_sha256", "data_sha256"):
        if metrics.get(name) != receipt.get(name):
            raise ValueError(f"metric envelope {name} differs from validated receipt")
    if metrics.get("status") != "PASS_ENGINEERING" or metrics.get("evidence_class") != "SYNTHETIC_ENGINEERING_FIXTURE":
        raise ValueError("analysis accepts only identified engineering fixture evidence")
    if metrics.get("scientific_claim") != "NOT_EVALUATED":
        raise ValueError("unsupported scientific-claim classification")
    for field, artifact in (
        ("checkpoint_sha256", "checkpoints/best.pt"),
        ("predictions_sha256", "evaluation/predictions.pt"),
        ("raw_data_artifact_sha256", "evaluation/raw_data.pt"),
    ):
        if metrics.get(field) != _file_hash(path / artifact):
            raise ValueError(f"metric {field} artifact identity mismatch")
    training = _json(path / "training_result.json")
    if training.get("status") != "PASS_ENGINEERING":
        raise ValueError("training did not complete")
    if training.get("raw_data_sha256") != metrics.get("raw_data_sha256"):
        raise ValueError("training and evaluation raw data identities differ")
    seconds = _number(training["elapsed_seconds"], "training elapsed seconds")
    for name in ("steps_completed", "examples_seen", "epochs_completed"):
        if type(training[name]) is not int or training[name] < 1:
            raise ValueError(f"missing positive training counter {name}")
    group_config = {key: value for key, value in config.items() if key != "seed"}
    group = _digest(group_config)
    parameters = config["hidden"] * (config["context_steps"] * config["channels"] * config["spatial_points"] + 1) + config["channels"] * config["spatial_points"] * (config["hidden"] + 1)
    events = _read_events(path / "events.jsonl")
    eval_events = _read_events(path / "evaluation/events.jsonl")
    peaks = [_number(e["cuda_peak_bytes"], "CUDA peak bytes") for e in events if e.get("cuda_peak_bytes") is not None]
    evaluation_times = [_number(e["elapsed_seconds"], "evaluation seconds") for e in eval_events if "elapsed_seconds" in e]
    common = dict(cell_id=cell["id"], seed=config["seed"], protocol_group=group,
                  source_sha256=source, config_sha256=receipt["config_sha256"],
                  data_sha256=receipt["data_sha256"], raw_data_sha256=metrics["raw_data_sha256"],
                  shared_training_seconds=seconds, shared_evaluation_seconds=max(evaluation_times) if evaluation_times else None,
                  peak_cuda_allocated_bytes=max(peaks) if peaks else None,
                  peak_host_bytes=None, trained_parameter_count=parameters,
                  steps_completed=training["steps_completed"], examples_seen=training["examples_seen"],
                  epochs_completed=training["epochs_completed"], device=config["device"], precision=config["precision"])
    rows = []
    for variant in VARIANTS:
        block = metrics["ablations"][variant]
        alpha = _number(block["alpha"], "alpha")
        if alpha not in metrics["alpha_grid"] or not 0 <= alpha <= 1:
            raise ValueError("alpha is outside the frozen grid")
        for split in ("validation", "test"):
            expected = config[f"{split}_trajectories"] * (config["sequence_steps"] - config["context_steps"]) * config["channels"] * config["spatial_points"]
            if block[split]["alpha"] != alpha:
                raise ValueError("alpha differs between validation selection and test")
            for condition in ("selected", "base", "persistence"):
                _metric(block[split][condition], expected)
            if block[split]["base"] != metrics["ablations"]["base"][split]["selected"]:
                raise ValueError("base predictions are inconsistent across paired conditions")
            if block[split]["persistence"] != metrics["ablations"]["base"][split]["persistence"]:
                raise ValueError("persistence differs across paired conditions")
        rows.append({**common, "condition": variant, "alpha": alpha,
                     "validation_mse": block["validation"]["selected"]["mse"],
                     "test_mse": block["test"]["selected"]["mse"],
                     "test_sse": block["test"]["selected"]["squared_error_sum"],
                     "test_element_count": block["test"]["selected"]["element_count"]})
    base = metrics["ablations"]["base"]
    if metrics["ablations"]["erased"]["test"]["selected"] != base["test"]["selected"]:
        raise ValueError("disabled-memory control differs from the same frozen base")
    rows.append({**common, "condition": "persistence", "alpha": None,
                 "validation_mse": base["validation"]["persistence"]["mse"],
                 "test_mse": base["test"]["persistence"]["mse"],
                 "test_sse": base["test"]["persistence"]["squared_error_sum"],
                 "test_element_count": base["test"]["persistence"]["element_count"]})
    autonomous = metrics["autonomous_base_diagnostic"]
    _metric(autonomous["overall"])
    if not autonomous["all_predictions_finite"] or autonomous["memory"] != "disabled_no_unobserved_residual_writes":
        raise ValueError("autonomous diagnostic is nonfinite or changed information policy")
    horizons = [{"cell_id": cell["id"], "seed": config["seed"], "protocol_group": group,
                 "horizon": int(horizon), "mse": _metric(values),
                 "squared_error_sum": values["squared_error_sum"], "element_count": values["element_count"]}
                for horizon, values in autonomous["per_horizon"].items()]
    return rows, horizons


def _percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_statistics(effects, bootstrap_seed=101, bootstrap_repetitions=10000):
    """Independent paired seed effects; positive values favor the comparator."""
    values = [_number(value, "paired effect", nonnegative=False) for value in effects]
    if not values:
        raise ValueError("paired statistics require completed independent seeds")
    n = len(values)
    result = dict(n_seeds=n, mean_difference=statistics.fmean(values),
                  sample_sd=statistics.stdev(values) if n > 1 else None,
                  memory_wins=sum(value < 0 for value in values),
                  comparator_wins=sum(value > 0 for value in values), ties=sum(value == 0 for value in values),
                  exact_two_sided_sign_flip_p=None, bootstrap_low=None, bootstrap_high=None,
                  interval_method="unavailable_with_one_seed" if n == 1 else "descriptive_95pct_percentile_bootstrap_of_seed_pairs",
                  bootstrap_seed=bootstrap_seed, bootstrap_repetitions=bootstrap_repetitions if n > 1 else 0)
    if n > 1:
        rng = random.Random(bootstrap_seed)
        means = [statistics.fmean(rng.choices(values, k=n)) for _ in range(bootstrap_repetitions)]
        result["bootstrap_low"] = _percentile(means, 0.025)
        result["bootstrap_high"] = _percentile(means, 0.975)
        if n <= 20:
            observed = abs(math.fsum(values))
            exceed = sum(abs(math.fsum(sign * value for sign, value in zip(signs, values))) >= observed - 2 * math.ulp(observed)
                         for signs in itertools.product((-1, 1), repeat=n))
            result["exact_two_sided_sign_flip_p"] = exceed / (2 ** n)
    return result


def _comparisons(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["protocol_group"], {}).setdefault(row["cell_id"], {})[row["condition"]] = row
    aggregates, pairs = [], []
    for group, cells in sorted(grouped.items()):
        seeds = [conditions["memory"]["seed"] for conditions in cells.values()]
        if len(set(seeds)) != len(seeds):
            raise ValueError("duplicate seed within a protocol group; do not count retries as independent replication")
        local = []
        for comparator in COMPARISONS:
            effects = []
            for cell_id, conditions in sorted(cells.items()):
                memory = conditions["memory"]["test_mse"]
                reference = conditions[comparator]["test_mse"]
                effect = memory - reference
                effects.append(effect)
                pairs.append(dict(protocol_group=group, cell_id=cell_id, seed=conditions["memory"]["seed"],
                                  comparator=comparator, memory_mse=memory, comparator_mse=reference,
                                  difference=effect, direction="memory_better" if effect < 0 else "comparator_better" if effect > 0 else "tie"))
            local.append(dict(protocol_group=group, comparator=comparator, **paired_statistics(effects)))
        # Holm controls this declared three-comparison family; no best-seed choice.
        ordered = sorted((item for item in local if item["exact_two_sided_sign_flip_p"] is not None), key=lambda item: item["exact_two_sided_sign_flip_p"])
        prior = 0.0
        for index, item in enumerate(ordered):
            prior = max(prior, min(1.0, (len(COMPARISONS) - index) * item["exact_two_sided_sign_flip_p"]))
            item["holm_adjusted_p"] = prior
        for item in local:
            item.setdefault("holm_adjusted_p", None)
        aggregates.extend(local)
    return aggregates, pairs


def _csv(path, rows, fields):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _svg(path, title, rows, effect=False):
    """Small scientific SVG from retained values only; no placeholder numbers."""
    if not rows:
        return False
    width, left, right, row_height = 980, 370, 900, 29
    height = 90 + row_height * len(rows)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="20" y="25" font-family="sans-serif" font-size="18">{escape(title)}</text>']
    if effect:
        extent = max(abs(float(row[key])) for row in rows for key in ("mean_difference", "bootstrap_low", "bootstrap_high") if row.get(key) is not None) or 1e-12
        scale = lambda value: (left + right) / 2 + value / extent * (right - left) / 2
        parts.append(f'<line x1="{scale(0)}" y1="48" x2="{scale(0)}" y2="{height - 20}" stroke="#777"/>')
    else:
        extent = max(row["test_mse"] for row in rows) or 1e-12
        scale = lambda value: left + value / extent * (right - left)
    for index, row in enumerate(rows):
        y = 65 + index * row_height
        label = f'{row["protocol_group"][:8]} / {row["comparator"]} / n={row["n_seeds"]}' if effect else f'{row["cell_id"]} / {row["condition"]}'
        value = row["mean_difference"] if effect else row["test_mse"]
        parts.append(f'<text x="20" y="{y + 4}" font-family="sans-serif" font-size="12">{escape(label)}</text>')
        if effect:
            if row["bootstrap_low"] is not None:
                parts.append(f'<line x1="{scale(row["bootstrap_low"])}" y1="{y}" x2="{scale(row["bootstrap_high"])}" y2="{y}" stroke="#444" stroke-width="2"/>')
            parts.append(f'<circle cx="{scale(value)}" cy="{y}" r="4" fill="{"#176b47" if value < 0 else "#a53b32"}"/>')
        else:
            parts.append(f'<rect x="{left}" y="{y - 7}" width="{scale(value) - left}" height="14" fill="#35699a"/>')
        parts.append(f'<text x="{right + 8}" y="{y + 4}" font-family="monospace" font-size="11">{value:.6g}</text>')
    parts.append('</svg>')
    Path(path).write_text("\n".join(parts) + "\n")
    return True


def run(root, manifest, output):
    root = Path(root).resolve()
    protocol = _json(manifest) if not isinstance(manifest, dict) else manifest
    source = protocol["source_sha256"]
    if not isinstance(source, str) or len(source) != 64 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("manifest must identify one frozen source SHA-256")
    cells = protocol["cells"]
    if not cells or len({cell["id"] for cell in cells}) != len(cells):
        raise ValueError("manifest requires nonempty unique cell IDs")
    ledger_path = _safe_cell_path(root, protocol.get("run_ledger", "run_ledger.json"))
    ledger = _json(ledger_path) if ledger_path.exists() else {"cells": {}}
    if ledger.get("manifest_sha256") not in (None, _digest(protocol)):
        raise ValueError("ledger belongs to another manifest")
    if ledger.get("source_sha256") not in (None, source):
        raise ValueError("ledger belongs to another source")
    unknown = sorted(set(ledger["cells"]) - {cell["id"] for cell in cells})
    if unknown:
        raise ValueError(f"ledger has unmanifested cells: {unknown}")
    statuses, rows, horizons = [], [], []
    for cell in cells:
        if type(cell.get("mandatory", True)) is not bool:
            raise ValueError("mandatory must be a Boolean")
        entry = ledger["cells"].get(cell["id"], {"state": "PENDING"})
        state = entry["state"]
        if state not in STATES:
            raise ValueError(f"unknown run state: {state}")
        status = dict(cell_id=cell["id"], seed=cell["config"]["seed"], mandatory=bool(cell.get("mandatory", True)),
                      ledger_state=state, artifact_valid=False, reason=str(entry.get("reason", "")))
        if state == "SUCCEEDED":
            try:
                path = _safe_cell_path(root, entry.get("path", f'cells/{cell["id"]}'))
                extracted, rollout = _extract(cell, path, source)
                rows.extend(extracted)
                horizons.extend(rollout)
                status["artifact_valid"] = True
            except (ValueError, KeyError, OSError, TypeError) as error:
                status["reason"] = f"invalid completed artifact: {type(error).__name__}: {error}"
        statuses.append(status)
    aggregates, pairs = _comparisons(rows)
    completed = sum(row["artifact_valid"] for row in statuses)
    mandatory_missing = [row["cell_id"] for row in statuses if row["mandatory"] and not row["artifact_valid"]]
    attempted = any(row["ledger_state"] in {"RUNNING", "SUCCEEDED", "FAILED", "INTERRUPTED"} for row in statuses)
    end_state = ("COMPLETE ENGINEERING/PILOT VALIDATION ONLY" if completed and not mandatory_missing else
                 "PARTIAL CAMPAIGN" if completed or attempted else "BLOCKED BEFORE EXECUTION")
    result = dict(schema_version=1, end_state=end_state, complete_confirmatory_study=False,
                  evidence_class="SYNTHETIC_ENGINEERING_FIXTURE", source_sha256=source,
                  manifest_sha256=_digest(protocol), expected_cells=len(cells), valid_completed_cells=completed,
                  mandatory_missing=mandatory_missing, statistical_unit="independent_seed",
                  effect_direction="memory_test_mse_minus_comparator_test_mse; negative favors memory",
                  cells=statuses, paired_aggregates=aggregates, limitations=LIMITATIONS,
                  preprint_supported="engineering methods and bounded descriptive results only; no scientific efficacy claim",
                  durable_storage_verified=False, session_stop_verified=False,
                  observed_final_quota_hours=None, gpu_session_seconds=None)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "analysis.json", result)
    _write_json(destination / "manifest.json", protocol)
    _csv(destination / "cell_status.csv", statuses, list(statuses[0]))
    _csv(destination / "per_seed_metrics.csv", rows, list(rows[0]) if rows else ["cell_id", "seed", "condition", "test_mse"])
    _csv(destination / "paired_differences.csv", pairs, list(pairs[0]) if pairs else ["cell_id", "seed", "comparator", "difference"])
    _csv(destination / "paired_summary.csv", aggregates, list(aggregates[0]) if aggregates else ["protocol_group", "comparator", "n_seeds", "mean_difference"])
    _csv(destination / "autonomous_base_horizons.csv", horizons, list(horizons[0]) if horizons else ["cell_id", "seed", "horizon", "mse"])
    _svg(destination / "per_seed_mse.svg", "Held-out one-step MSE (lower is better)", rows)
    _svg(destination / "paired_effects.svg", "Memory minus comparator MSE; negative favors memory (seed bootstrap intervals)", aggregates, effect=True)
    report = ["# FIM bounded engineering evidence", "", f"**End state: {end_state}.**", "",
              f"Validated {completed} of {len(cells)} expected cells. Missing mandatory cells: {', '.join(mandatory_missing) or 'none'}.", "",
              "## Methods", "", "One existing residual MLP checkpoint per seed is selected by base-model validation MSE. Fixed observed-context memory controls share that checkpoint; their scalar correction is selected on validation and frozen for test. Baseline comparisons include the same base model, last observed residual, and persistence. Source, configuration, data, and declared artifact checksums are verified by the maintained completion validator before a row is admitted.", "",
              "Protocol groups hash all configuration fields except seed. Incompatible shapes, precision, device, budgets, or purposes remain separate. Paired effects use independent seeds, with negative memory-minus-comparator MSE favoring memory. Exact two-sided sign-flip tests (up to 20 seeds) and a deterministic 10,000-resample seed-pair bootstrap are descriptive; Holm adjustment covers the three declared comparisons within each protocol group.", "",
              "## Results", ""]
    if not aggregates:
        report.append("No validated paired results are available. No result chart or numerical finding was generated.")
    for item in aggregates:
        report.append(f"- Protocol {item['protocol_group'][:12]}, memory versus {item['comparator']}: n={item['n_seeds']} seeds; mean difference {item['mean_difference']:.8g}; memory wins {item['memory_wins']}, comparator wins {item['comparator_wins']}, ties {item['ties']}. " + ("One seed cannot estimate replication uncertainty." if item['n_seeds'] == 1 else f"Descriptive bootstrap interval [{item['bootstrap_low']:.8g}, {item['bootstrap_high']:.8g}]; Holm-adjusted exact p={item['holm_adjusted_p']} under the stated sign-exchangeability assumption."))
    report += ["", "All seed outcomes, including baseline wins, are retained in per_seed_metrics.csv and paired_differences.csv. Autonomous base rollout metrics are separate in autonomous_base_horizons.csv. Runtime and CUDA peak allocation are copied from actual retained training/evaluation logs; unavailable host memory remains blank. Shared training times must not be summed across memory conditions.", "", "## Missing and failed cells", ""]
    report += [f"- {row['cell_id']}: {row['ledger_state']}; validated={row['artifact_valid']}; {row['reason'] or 'no additional reason recorded'}." for row in statuses]
    report += ["", "## Claim and evidence map", "", "| Claim | Evidence | Scope |", "|---|---|---|",
               f"| Completed engineering execution | {completed} validated completion receipts and artifact inventories | Synthetic fixture only |",
               "| Memory-relative empirical effects | Per-seed MSE, paired differences, and seed-level summaries | Observed seeds and frozen settings only |",
               "| Autonomous rollout behavior | Per-horizon base-only errors and retained raw predictions | Finite fixture horizon; no stability certificate |",
               "| Scientific efficacy or general superiority | Not established | No public benchmark or independent replication |",
               "| Durable retrieval, final quota, and stopped platform sessions | Not established by this JSON-only analysis | Require separate platform receipts |",
               "", "## Limitations", ""] + [f"- {item}" for item in LIMITATIONS]
    report += ["", "## Manuscript and venue boundary", "", "An engineering methods description and honest bounded descriptive results can be drafted from valid rows. A scientific efficacy preprint is not justified by this fixture alone. A named conference requires a suitable public or independently validated task, strong external comparators, adequate repetitions, independent reproducibility, literature/novelty review, and venue-specific expert review.", "",
               "Observed final Kaggle quota, GPU/session duration, durable retrieval, and platform session-stop confirmation are unavailable to this analyzer. Local output files or an archive do not establish a second persistent copy."]
    (destination / "report.md").write_text("\n".join(report) + "\n")
    _write_json(destination / "checksums.json", {path.name: _file_hash(path) for path in sorted(destination.iterdir()) if path.is_file()})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.manifest, args.output), sort_keys=True, indent=2))
