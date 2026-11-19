"""Artificial manifest fixtures; no scientific training or retained outcomes."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import analyze_trajectory_isolated_ablations as analyzer


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = "research/protocols/FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1.md"
SWITCHES = {
    "full": (True, True, True),
    "no_memory": (False, False, False),
    "no_retrieval": (True, False, True),
    "no_salience_gating": (True, True, False),
}


def artificial_manifest():
    """Use the current runner's JSON contract with explicitly artificial values."""
    data = {
        "schema_version": 2,
        "protocol_name": "FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1",
        "protocol_path": PROTOCOL_PATH,
        "protocol_sha256": hashlib.sha256((ROOT / PROTOCOL_PATH).read_bytes()).hexdigest(),
        "created_or_updated_utc": "2026-10-08T00:00:02+00:00",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "python": "3.12.14",
        "torch": "2.14.1+cpu",
        "scientific_boundary": "ARTIFICIAL TEST ONLY; not an executed matrix",
        "trajectory_isolation": {
            "batch_size": 1,
            "memory_reset_between_independent_episodes": True,
            "validation_memory_enabled_within_episode": True,
            "test_memory_enabled_within_episode": True,
            "checkpoint_selection": "fixed_final_epoch_no_best_selection",
            "ema": False,
        },
        "benchmarks": ["delayed_recall", "lorenz96"],
        "seeds": [101, 211, 307, 401, 503],
        "variants": list(SWITCHES),
        "budget": dict(analyzer.EXPECTED_BUDGET),
        "expected_cells": 40,
        "runs": [],
    }
    for benchmark, seed, variant in itertools.product(data["benchmarks"], data["seeds"], data["variants"]):
        memory, retrieval, salience = SWITCHES[variant]
        data["runs"].append({
            "status": "success", "benchmark": benchmark, "seed": seed, "variant": variant,
            "memory_enabled": memory, "retrieval_enabled": retrieval,
            "salience_gating_enabled": salience,
            "started_utc": "2026-10-08T00:00:00+00:00",
            "ended_utc": "2026-10-08T00:00:01+00:00",
            "git_commit": data["git_commit"], "protocol_sha256": data["protocol_sha256"],
            "experiment_name": f"ARTIFICIAL_{benchmark}_{variant}_s{seed}",
            "results_root": "ARTIFICIAL_TEST_ONLY",
            **data["budget"],
            "validation_memory_enabled": True, "evaluation_memory_enabled": True,
            "rollout_mse": 1.0, "final_step_mse": 1.0,
            "rollout_mae": 1.0, "final_step_mae": 1.0,
        })
    return data


def test_complete_artificial_manifest_is_accepted_without_rewriting_rows():
    data = artificial_manifest()
    data["runs"].reverse()
    before = copy.deepcopy(data)
    assert analyzer.validate_manifest(data) == data["runs"]
    assert data == before


@pytest.mark.parametrize("field,value", [
    ("memory_enabled", False), ("retrieval_enabled", False),
    ("salience_gating_enabled", False), ("memory_enabled", 1),
    ("validation_memory_enabled", "false"), ("evaluation_memory_enabled", 1),
    ("epochs", 4.9), ("batch_size", True), ("dataset_size", "32"),
    ("train_rollout_steps", 16.5), ("eval_rollout_steps", 20.9),
    ("seed", 101.5), ("seed", "101"), ("seed", 999),
    ("benchmark", "unregistered"), ("variant", "unregistered"),
    ("rollout_mse", -1.0), ("rollout_mse", True), ("rollout_mse", "1.0"),
])
def test_corrupt_run_cannot_enter_the_analysis_gate(field, value):
    data = artificial_manifest()
    data["runs"][0][field] = value
    with pytest.raises(ValueError):
        analyzer.validate_manifest(data)


@pytest.mark.parametrize("field,value", [
    ("memory_reset_between_independent_episodes", False),
    ("validation_memory_enabled_within_episode", "true"),
    ("test_memory_enabled_within_episode", False),
    ("checkpoint_selection", "best_validation"), ("ema", True), ("batch_size", True),
])
def test_declared_isolation_and_selection_semantics_are_required(field, value):
    data = artificial_manifest()
    data["trajectory_isolation"][field] = value
    with pytest.raises(ValueError):
        analyzer.validate_manifest(data)


@pytest.mark.parametrize("field", ["memory_enabled", "retrieval_enabled", "salience_gating_enabled"])
def test_missing_switch_provenance_is_rejected(field):
    data = artificial_manifest()
    del data["runs"][0][field]
    with pytest.raises(ValueError):
        analyzer.validate_manifest(data)


def test_matching_but_unbound_protocol_hash_is_rejected():
    data = artificial_manifest()
    data["protocol_sha256"] = "b" * 64
    for row in data["runs"]:
        row["protocol_sha256"] = data["protocol_sha256"]
    with pytest.raises(ValueError):
        analyzer.validate_manifest(data)


def test_unknown_commit_cannot_be_reported_as_exact_source():
    data = artificial_manifest()
    data["git_commit"] = "UNKNOWN"
    for row in data["runs"]:
        row["git_commit"] = "UNKNOWN"
    with pytest.raises(ValueError):
        analyzer.validate_manifest(data)


def test_duplicate_cell_is_rejected_even_when_count_is_40():
    data = artificial_manifest()
    data["runs"][-1] = copy.deepcopy(data["runs"][0])
    with pytest.raises(ValueError):
        analyzer.validate_manifest(data)


def test_corrupt_switch_cli_leaves_existing_publication_untouched(tmp_path):
    data = artificial_manifest()
    data["runs"][0]["retrieval_enabled"] = False
    manifest = tmp_path / "artificial.json"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    outputs = [tmp_path / name for name in ("paired.csv", "table.md", "evidence.md")]
    old = b"previous artificial payload\n"
    for output in outputs:
        output.write_bytes(old)
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/analyze_trajectory_isolated_ablations.py"),
        "--manifest", str(manifest), "--csv", str(outputs[0]),
        "--table", str(outputs[1]), "--evidence", str(outputs[2]),
    ], capture_output=True, text=True, cwd=tmp_path, timeout=15)
    assert result.returncode != 0
    assert all(output.read_bytes() == old for output in outputs)
