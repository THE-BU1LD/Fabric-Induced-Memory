from itertools import product

from scripts.validate_component_ablation_manifest import (
    EXPECTED_BENCHMARKS,
    EXPECTED_SEEDS,
    EXPECTED_SWITCHES,
    EXPECTED_VARIANTS,
    validate_manifest,
)


def _valid_manifest():
    commit = "a" * 40
    runs = []
    for benchmark, seed, variant in product(
        EXPECTED_BENCHMARKS, EXPECTED_SEEDS, EXPECTED_VARIANTS
    ):
        memory_enabled, retrieval_enabled, salience_gating_enabled = EXPECTED_SWITCHES[variant]
        runs.append(
            {
                "benchmark": benchmark,
                "seed": seed,
                "variant": variant,
                "memory_enabled": memory_enabled,
                "retrieval_enabled": retrieval_enabled,
                "salience_gating_enabled": salience_gating_enabled,
                "started_utc": "2026-08-29T00:00:00+00:00",
                "ended_utc": "2026-08-29T00:01:00+00:00",
                "git_commit": commit,
                "experiment_name": f"current_ablation_{benchmark}_{variant}_s{seed}",
                "results_root": "results/current_component_ablations",
                "epochs": 12,
                "batch_size": 64,
                "dataset_size": 2048,
                "train_rollout_steps": 4,
                "eval_rollout_steps": 30,
            }
        )
    return {
        "schema_version": 1,
        "git_commit": commit,
        "benchmarks": EXPECTED_BENCHMARKS.copy(),
        "seeds": EXPECTED_SEEDS.copy(),
        "variants": EXPECTED_VARIANTS.copy(),
        "runs": runs,
    }


def test_valid_manifest_passes():
    assert validate_manifest(_valid_manifest()) == []


def test_missing_cell_fails_closed():
    manifest = _valid_manifest()
    manifest["runs"].pop()
    errors = validate_manifest(manifest)
    assert any("missing 1 expected cell" in error for error in errors)
    assert any("runs length must be 24" in error for error in errors)


def test_duplicate_cell_fails_closed():
    manifest = _valid_manifest()
    manifest["runs"][-1] = dict(manifest["runs"][0])
    errors = validate_manifest(manifest)
    assert any("duplicate" in error for error in errors)
    assert any("missing 1 expected cell" in error for error in errors)


def test_mixed_commit_fails_closed():
    manifest = _valid_manifest()
    manifest["runs"][7]["git_commit"] = "b" * 40
    errors = validate_manifest(manifest)
    assert any("!= manifest git_commit" in error for error in errors)


def test_protocol_drift_fails_closed():
    manifest = _valid_manifest()
    manifest["runs"][3]["epochs"] = 99
    errors = validate_manifest(manifest)
    assert any("mixes experiment protocol settings" in error for error in errors)


def test_wrong_switch_semantics_fail_closed():
    manifest = _valid_manifest()
    target = next(row for row in manifest["runs"] if row["variant"] == "no_retrieval")
    target["retrieval_enabled"] = True
    errors = validate_manifest(manifest)
    assert any("does not match variant 'no_retrieval'" in error for error in errors)


def test_unknown_commit_is_not_valid_evidence():
    manifest = _valid_manifest()
    manifest["git_commit"] = "UNKNOWN"
    for row in manifest["runs"]:
        row["git_commit"] = "UNKNOWN"
    errors = validate_manifest(manifest)
    assert any("concrete commit SHA" in error for error in errors)
