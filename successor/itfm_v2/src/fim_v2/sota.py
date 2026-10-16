from __future__ import annotations

import re
from typing import Any


def audit_sota_protocol(
    protocol: dict[str, Any], frontier: dict[str, Any]
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["frozen_status"] = (
        protocol.get("status") == "FROZEN_BEFORE_PUBLIC_BENCHMARK_EXECUTION"
    )
    checks["stage_order"] = [
        stage.get("id") for stage in protocol.get("stages", [])
    ] == [
        "S0",
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
    ]
    primary = protocol.get("primary_track", {})
    secondary = protocol.get("secondary_track", {})
    checks["primary_36_cells"] = (
        len(primary.get("datasets", []))
        * len(primary.get("mask_patterns", []))
        * sum(
            len(values)
            for values in primary.get("test_missing_percent_by_train", {}).values()
        )
        == primary.get("cells")
        == 36
    )
    primary_gate = primary.get("material_sota_gate", {})
    checks["material_primary_gate"] = (
        primary_gate.get("minimum_macro_relative_error_reduction_vs_reproduced_lano", 0)
        >= 0.10
        and primary_gate.get("minimum_cells_won_of_36", 0) >= 30
        and primary_gate.get("maximum_relative_regression_any_cell", 1) <= 0.02
    )
    secondary_gate = secondary.get("material_sota_gate", {})
    checks["material_secondary_gate"] = (
        secondary_gate.get("minimum_relative_improvement_vs_reproduced_hs_fno", 0)
        >= 0.15
        and secondary_gate.get("minimum_family_wins_of_5", 0) >= 4
        and secondary_gate.get("maximum_target_rollout_error", 1) <= 0.0799
    )
    checks["multiseed"] = (
        len(primary.get("seeds", [])) >= 5 and len(secondary.get("seeds", [])) >= 10
    )
    checks["causal_controls"] = len(protocol.get("memory_causality_controls", [])) >= 10
    checks["protected_test_policy"] = (
        len(
            protocol.get("protected_test_policy", {}).get(
                "forbidden_before_final_run", []
            )
        )
        >= 4
    )
    frontiers = frontier.get("frontiers", {})
    commits = [item.get("code_commit", "") for item in frontiers.values()] + [
        item.get("commit", "") for item in frontier.get("supporting_sources", [])
    ]
    checks["source_commits_locked"] = len(commits) >= 5 and all(
        re.fullmatch(r"[0-9a-f]{40}", commit) for commit in commits
    )
    lano = frontiers.get("pobench_pde", {}).get("reported_lano", {})
    checks["frontier_36_values"] = (
        sum(len(values) for dataset in lano.values() for values in dataset.values())
        == 36
    )
    checks["external_results_labeled"] = (
        frontier.get("status") == "EXTERNAL_REPORTED_NOT_LOCALLY_REPRODUCED"
    )
    return checks
