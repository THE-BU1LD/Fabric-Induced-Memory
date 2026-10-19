"""Conservative, stdlib-only preflight for the FIM one-shot run budget.

This module validates a caller-declared resource plan. It does not measure,
reserve, contain, or terminate a workload. A plan that fits is still NOT
admitted for launch: the external enforcement needed for this one-shot budget
is not observable or verifiable from this candidate.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = 1
BUDGET_ID = "FIM_DELAYED_RECALL_ONCE_V1"
CAPS = {
    "gpu_seconds": 3_600,
    "wall_seconds": 3_600,
    "aggregate_cpu_seconds": 7_200,
    "ram_peak_bytes": 8 * 1024**3,
    "output_bytes": 1 * 1024**3,
}
PHASES = ("setup", "tests", "experiment", "verification", "export")
PHASE_FIELDS = frozenset(CAPS)
CUMULATIVE_FIELDS = ("gpu_seconds", "wall_seconds", "aggregate_cpu_seconds", "output_bytes")
SPENT_FIELDS = frozenset(CUMULATIVE_FIELDS)

ENFORCEMENT_GAPS = (
    "trusted_cumulative_accounting_source_unverified",
    "external_single_gpu_hour_lease_unverified",
    "external_3600_second_wall_kill_unverified",
    "aggregate_cpu_counter_and_kill_unverified",
    "8_gib_memory_cgroup_unverified",
    "1_gib_prewrite_output_quota_unverified",
)


def _uint(value: Any, field: str) -> int:
    # bool is an int subclass, so reject it explicitly. Floats (including
    # integral-looking floats) are rejected to keep budget math exact.
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def evaluate_plan(document: Any) -> dict[str, Any]:
    """Return a JSON-safe preflight report; never grants launch permission.

    All five phases are mandatory, including phases with an estimated cost of
    zero. Wall and GPU time, aggregate CPU time, and output are summed across
    phases. RAM peak estimates are also summed, deliberately conservative if
    phase/process overlap has not been ruled out. Caller-declared cumulative
    usage is mandatory for GPU, wall, aggregate CPU, and output so a new plan
    cannot appear to restore previously spent allowance. RAM is an instantaneous
    peak and is checked against its cap independently, not added to historical
    RAM peaks. All declared inputs remain unverified estimates/observations.
    """
    errors: list[str] = []
    totals = {name: 0 for name in CAPS}
    consumed: dict[str, int] | None = None

    if not isinstance(document, Mapping):
        return _report("INVALID_PLAN", totals, {"document": "expected an object"})

    expected_top = {"schema_version", "budget_id", "already_consumed", "phases"}
    unknown_top = set(document) - expected_top
    missing_top = expected_top - set(document)
    if unknown_top:
        errors.append("unknown top-level fields: " + ", ".join(sorted(map(str, unknown_top))))
    if missing_top:
        errors.append("missing top-level fields: " + ", ".join(sorted(missing_top)))
    if document.get("schema_version") != SCHEMA_VERSION or type(document.get("schema_version")) is not int:
        errors.append("schema_version must be integer 1")
    if document.get("budget_id") != BUDGET_ID:
        errors.append(f"budget_id must be {BUDGET_ID}")

    spent = document.get("already_consumed")
    if not isinstance(spent, Mapping):
        errors.append("already_consumed must be an object with cumulative counters")
    else:
        unknown = set(spent) - SPENT_FIELDS
        missing = SPENT_FIELDS - set(spent)
        if unknown:
            errors.append("already_consumed has unknown fields: " + ", ".join(sorted(map(str, unknown))))
        if missing:
            errors.append("already_consumed is missing fields: " + ", ".join(sorted(missing)))
        parsed: dict[str, int] = {}
        # Use the schema tuple rather than a set intersection so diagnostic
        # ordering does not depend on Python's per-process hash seed.
        for field in CUMULATIVE_FIELDS:
            if field not in spent:
                continue
            try:
                parsed[field] = _uint(spent[field], f"already_consumed.{field}")
            except ValueError as exc:
                errors.append(str(exc))
        if not unknown and not missing and len(parsed) == len(SPENT_FIELDS):
            consumed = parsed

    phases = document.get("phases")
    if not isinstance(phases, Mapping):
        errors.append("phases must be an object")
    else:
        supplied = set(phases)
        missing_phases = set(PHASES) - supplied
        extra_phases = supplied - set(PHASES)
        if missing_phases:
            errors.append("missing phases: " + ", ".join(sorted(missing_phases)))
        if extra_phases:
            errors.append("unknown phases: " + ", ".join(sorted(map(str, extra_phases))))
        for phase in PHASES:
            if phase not in phases:
                continue
            costs = phases[phase]
            if not isinstance(costs, Mapping):
                errors.append(f"{phase} must be an object")
                continue
            unknown = set(costs) - PHASE_FIELDS
            missing = PHASE_FIELDS - set(costs)
            if unknown:
                errors.append(f"{phase} has unknown resource fields: " + ", ".join(sorted(map(str, unknown))))
            if missing:
                errors.append(f"{phase} is missing resource fields: " + ", ".join(sorted(missing)))
            # CAPS preserves the public schema order and keeps errors stable
            # across processes with different hash seeds.
            for field in CAPS:
                if field not in costs:
                    continue
                try:
                    amount = _uint(costs[field], f"{phase}.{field}")
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                totals[field] += amount

    if errors:
        return _report("INVALID_PLAN", totals, {"plan": errors}, already_consumed=consumed)

    # consumed is non-None after the strict validation above.
    assert consumed is not None
    cumulative = {field: consumed[field] + totals[field] for field in CUMULATIVE_FIELDS}
    overages: dict[str, Any] = {}
    for field in CUMULATIVE_FIELDS:
        cap = CAPS[field]
        if cumulative[field] > cap:
            overages[field] = {
                "already_consumed": consumed[field],
                "planned": totals[field],
                "cumulative_total": cumulative[field],
                "cap": cap,
            }
    if totals["ram_peak_bytes"] > CAPS["ram_peak_bytes"]:
        overages["ram_peak_bytes"] = {
            "planned_peak": totals["ram_peak_bytes"],
            "cap": CAPS["ram_peak_bytes"],
        }
    return _report(
        "OVER_BUDGET" if overages else "WITHIN_DECLARED_BUDGET",
        totals,
        overages,
        already_consumed=consumed,
        cumulative_totals=cumulative,
        remaining_budget={field: CAPS[field] - cumulative[field] for field in CUMULATIVE_FIELDS},
    )


def _report(
    plan_status: str,
    totals: dict[str, int],
    problems: dict[str, Any],
    *,
    already_consumed: dict[str, int] | None = None,
    cumulative_totals: dict[str, int] | None = None,
    remaining_budget: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "budget_id": BUDGET_ID,
        "plan_status": plan_status,
        "planned_totals": totals,
        "already_consumed": already_consumed,
        "cumulative_totals": cumulative_totals,
        "remaining_budget": remaining_budget,
        "accounting_status": "SELF_DECLARED_UNVERIFIED",
        "limits": dict(CAPS),
        "problems": problems,
        "enforcement_status": "UNVERIFIED",
        "launch_admissible": False,
        "launch_blockers": list(ENFORCEMENT_GAPS),
        "scope": "declared-plan preflight only; plan and prior-usage inputs are not verified accounting or hard controls",
    }


def require_launch_admission(report: Any) -> None:
    """Fail closed: this candidate has no trusted external enforcer adapter."""
    raise RuntimeError(
        "launch blocked: this candidate can check declared budget arithmetic only; "
        "external GPU, wall, aggregate-CPU, RAM, and output enforcement is unverified"
    )
