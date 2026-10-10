"""Analysis-admission regression tests using artificial metadata and zero metrics.

No retained outcomes, checkpoints, training paths, or scientific matrix are read.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import analyze_trajectory_isolated_ablations as analysis


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/protocols/FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1.md"
SWITCH_NAMES = ("memory_enabled", "retrieval_enabled", "salience_gating_enabled")
SWITCHES = {
    "full": (True, True, True),
    "no_memory": (False, False, False),
    "no_retrieval": (True, False, True),
    "no_salience_gating": (True, True, False),
}


@pytest.fixture
def manifest():
    protocol_hash = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    rows = []
    for benchmark, seed, variant in itertools.product(
        ["delayed_recall", "lorenz96"], [101, 211, 307, 401, 503], SWITCHES
    ):
        rows.append({
            "status": "success", "benchmark": benchmark, "seed": seed,
            "variant": variant, "git_commit": "a" * 40,
            "protocol_sha256": protocol_hash,
            "epochs": 4, "dataset_size": 32, "batch_size": 1,
            "train_rollout_steps": 16, "eval_rollout_steps": 20,
            "validation_memory_enabled": True, "evaluation_memory_enabled": True,
            **dict(zip(SWITCH_NAMES, SWITCHES[variant])),
            "rollout_mse": 0.0, "final_step_mse": 0.0,
            "rollout_mae": 0.0, "final_step_mae": 0.0,
        })
    return {
        "schema_version": 2,
        "protocol_name": "FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1",
        "protocol_path": "research/protocols/FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1.md",
        "protocol_sha256": protocol_hash, "git_commit": "a" * 40,
        "git_dirty": False, "expected_cells": 40,
        "benchmarks": ["delayed_recall", "lorenz96"],
        "seeds": [101, 211, 307, 401, 503], "variants": list(SWITCHES),
        "budget": {"epochs": 4, "dataset_size": 32, "batch_size": 1,
                   "train_rollout_steps": 16, "eval_rollout_steps": 20},
        "trajectory_isolation": {
            "batch_size": 1, "memory_reset_between_independent_episodes": True,
            "validation_memory_enabled_within_episode": True,
            "test_memory_enabled_within_episode": True,
            "checkpoint_selection": "fixed_final_epoch_no_best_selection", "ema": False,
        },
        "runs": rows,
    }


def test_complete_artificial_manifest_is_accepted_without_reordering(manifest):
    manifest["runs"].reverse()
    assert analysis.validate_manifest(manifest) is manifest["runs"]


@pytest.mark.parametrize("field,value", [
    ("benchmark", "unknown_benchmark"), ("seed", 999),
    ("variant", "unknown_variant"), ("seed", 101.9), ("seed", "101"),
])
def test_forty_unique_rows_must_be_exact_frozen_cells(manifest, field, value):
    manifest["runs"][0][field] = value
    with pytest.raises(ValueError):
        analysis.validate_manifest(manifest)


def test_duplicate_cell_rejected(manifest):
    manifest["runs"][0] = manifest["runs"][1].copy()
    with pytest.raises(ValueError):
        analysis.validate_manifest(manifest)


def test_consistently_wrong_protocol_hash_rejected(manifest):
    manifest["protocol_sha256"] = "0" * 64
    for row in manifest["runs"]:
        row["protocol_sha256"] = manifest["protocol_sha256"]
    with pytest.raises(ValueError):
        analysis.validate_manifest(manifest)


@pytest.mark.parametrize("variant", SWITCHES)
@pytest.mark.parametrize("field", SWITCH_NAMES)
def test_variant_labels_require_exact_boolean_switches(manifest, variant, field):
    row = next(row for row in manifest["runs"] if row["variant"] == variant)
    row[field] = not row[field]
    with pytest.raises(ValueError):
        analysis.validate_manifest(manifest)


@pytest.mark.parametrize("bad", [1, "true", None])
def test_truthy_values_are_not_switch_evidence(manifest, bad):
    manifest["runs"][0]["memory_enabled"] = bad
    with pytest.raises(ValueError):
        analysis.validate_manifest(manifest)


@pytest.mark.parametrize("field,value", [
    ("memory_reset_between_independent_episodes", False),
    ("validation_memory_enabled_within_episode", False),
    ("test_memory_enabled_within_episode", False),
    ("checkpoint_selection", "best_validation"), ("ema", True), ("batch_size", True),
])
def test_isolation_and_model_selection_semantics_must_match(manifest, field, value):
    manifest["trajectory_isolation"][field] = value
    with pytest.raises(ValueError):
        analysis.validate_manifest(manifest)


@pytest.mark.parametrize("field", ["validation_memory_enabled", "evaluation_memory_enabled"])
def test_row_memory_flags_require_true_boolean(manifest, field):
    manifest["runs"][0][field] = "false"
    with pytest.raises(ValueError):
        analysis.validate_manifest(manifest)


@pytest.mark.parametrize("field,value", [
    ("batch_size", 1.9), ("batch_size", True), ("epochs", 4.5),
    ("dataset_size", "32"), ("train_rollout_steps", 16.1), ("eval_rollout_steps", 20.9),
])
def test_budget_values_are_not_silently_coerced(manifest, field, value):
    manifest["runs"][0][field] = value
    with pytest.raises(ValueError):
        analysis.validate_manifest(manifest)


@pytest.mark.parametrize("bad", [-0.1, True, "0.0", float("nan"), float("inf")])
def test_error_metrics_require_nonnegative_finite_numbers(manifest, bad):
    manifest["runs"][0]["rollout_mse"] = bad
    with pytest.raises(ValueError):
        analysis.validate_manifest(manifest)


@pytest.mark.parametrize("commit", ["", "UNKNOWN", "abc123", None])
def test_matching_placeholder_commits_are_not_execution_identity(manifest, commit):
    manifest["git_commit"] = commit
    for row in manifest["runs"]:
        row["git_commit"] = commit
    with pytest.raises(ValueError):
        analysis.validate_manifest(manifest)


def test_bad_manifest_cannot_replace_existing_paper_outputs(manifest, tmp_path):
    # All rows share the same incorrect hash: formerly admitted and published.
    manifest["protocol_sha256"] = "0" * 64
    for row in manifest["runs"]:
        row["protocol_sha256"] = "0" * 64
    source = tmp_path / "ARTIFICIAL_MANIFEST.json"
    source.write_text(json.dumps(manifest))
    outputs = [tmp_path / name for name in ("paired.csv", "table.md", "evidence.md")]
    for path in outputs:
        path.write_bytes(b"existing artificial evidence\n")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/analyze_trajectory_isolated_ablations.py"),
         "--manifest", str(source), "--csv", str(outputs[0]),
         "--table", str(outputs[1]), "--evidence", str(outputs[2])],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert "protocol" in result.stderr.lower()
    assert all(path.read_bytes() == b"existing artificial evidence\n" for path in outputs)
