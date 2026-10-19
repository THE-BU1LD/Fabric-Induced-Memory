"""Tiny deterministic tests. Standard library only; no scientific workload."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from validator import CAPS, CUMULATIVE_FIELDS, ENFORCEMENT_GAPS, PHASES, evaluate_plan, require_launch_admission


def plan_each(value: int = 0) -> dict:
    return {
        "schema_version": 1,
        "budget_id": "FIM_DELAYED_RECALL_ONCE_V1",
        "already_consumed": {field: 0 for field in CUMULATIVE_FIELDS},
        "phases": {
            phase: {
                "gpu_seconds": value,
                "wall_seconds": value,
                "aggregate_cpu_seconds": value,
                "ram_peak_bytes": value,
                "output_bytes": value,
            }
            for phase in PHASES
        },
    }


class AdmissionTests(unittest.TestCase):
    def test_fitting_plan_still_never_admits_launch(self):
        report = evaluate_plan(plan_each())
        self.assertEqual(report["plan_status"], "WITHIN_DECLARED_BUDGET")
        self.assertFalse(report["launch_admissible"])
        self.assertEqual(report["enforcement_status"], "UNVERIFIED")
        self.assertEqual(report["launch_blockers"], list(ENFORCEMENT_GAPS))
        with self.assertRaisesRegex(RuntimeError, "launch blocked"):
            require_launch_admission(report)

    def test_exact_boundaries_fit_but_are_not_admitted(self):
        doc = plan_each()
        doc["phases"]["experiment"] = dict(CAPS)
        report = evaluate_plan(doc)
        self.assertEqual(report["plan_status"], "WITHIN_DECLARED_BUDGET")
        self.assertEqual(report["planned_totals"], CAPS)
        self.assertFalse(report["launch_admissible"])

    def test_each_total_cap_is_checked(self):
        for field, cap in CAPS.items():
            with self.subTest(field=field):
                doc = plan_each()
                doc["phases"]["experiment"][field] = cap + 1
                report = evaluate_plan(doc)
                self.assertEqual(report["plan_status"], "OVER_BUDGET")
                if field in CUMULATIVE_FIELDS:
                    self.assertEqual(report["problems"][field], {
                        "already_consumed": 0, "planned": cap + 1,
                        "cumulative_total": cap + 1, "cap": cap,
                    })
                else:
                    self.assertEqual(report["problems"][field], {"planned_peak": cap + 1, "cap": cap})
                self.assertFalse(report["launch_admissible"])

    def test_costs_are_summed_across_setup_tests_and_export(self):
        doc = plan_each()
        for phase in ("setup", "tests", "export"):
            doc["phases"][phase]["aggregate_cpu_seconds"] = 2_401
        report = evaluate_plan(doc)
        self.assertEqual(report["planned_totals"]["aggregate_cpu_seconds"], 7_203)
        self.assertEqual(report["plan_status"], "OVER_BUDGET")

    def test_exhausted_allowance_cannot_be_replenished(self):
        doc = plan_each()
        doc["already_consumed"]["gpu_seconds"] = CAPS["gpu_seconds"]
        doc["phases"]["experiment"]["gpu_seconds"] = 1
        report = evaluate_plan(doc)
        self.assertEqual(report["plan_status"], "OVER_BUDGET")
        self.assertEqual(report["problems"]["gpu_seconds"]["cumulative_total"], 3_601)
        self.assertEqual(report["remaining_budget"]["gpu_seconds"], -1)
        self.assertFalse(report["launch_admissible"])

    def test_spent_plus_plan_must_fit_each_cumulative_cap(self):
        for field, cap in ((name, CAPS[name]) for name in CUMULATIVE_FIELDS):
            with self.subTest(field=field):
                doc = plan_each()
                doc["already_consumed"][field] = cap - 1
                doc["phases"]["experiment"][field] = 2
                report = evaluate_plan(doc)
                self.assertEqual(report["plan_status"], "OVER_BUDGET")
                self.assertEqual(report["cumulative_totals"][field], cap + 1)
                self.assertFalse(report["launch_admissible"])

    def test_exact_spent_plus_plan_boundary_has_zero_remaining(self):
        doc = plan_each()
        doc["already_consumed"]["gpu_seconds"] = CAPS["gpu_seconds"] - 100
        doc["phases"]["experiment"]["gpu_seconds"] = 100
        report = evaluate_plan(doc)
        self.assertEqual(report["plan_status"], "WITHIN_DECLARED_BUDGET")
        self.assertEqual(report["remaining_budget"]["gpu_seconds"], 0)
        self.assertFalse(report["launch_admissible"])

    def test_missing_prior_usage_counters_are_invalid_not_zeroed(self):
        doc = plan_each()
        del doc["already_consumed"]
        report = evaluate_plan(doc)
        self.assertEqual(report["plan_status"], "INVALID_PLAN")
        self.assertIsNone(report["already_consumed"])
        self.assertIsNone(report["cumulative_totals"])
        self.assertFalse(report["launch_admissible"])

    def test_boolean_prior_usage_counter_is_invalid(self):
        doc = plan_each()
        doc["already_consumed"]["wall_seconds"] = True
        report = evaluate_plan(doc)
        self.assertEqual(report["plan_status"], "INVALID_PLAN")
        self.assertIsNone(report["already_consumed"])

    def test_ram_peak_is_not_added_to_historical_cumulative_counters(self):
        doc = plan_each()
        doc["phases"]["experiment"]["ram_peak_bytes"] = CAPS["ram_peak_bytes"]
        report = evaluate_plan(doc)
        self.assertEqual(report["plan_status"], "WITHIN_DECLARED_BUDGET")
        self.assertNotIn("ram_peak_bytes", report["already_consumed"])
        self.assertNotIn("ram_peak_bytes", report["cumulative_totals"])
        self.assertNotIn("ram_peak_bytes", report["remaining_budget"])

    def test_ram_is_summed_conservatively_across_phases(self):
        doc = plan_each()
        doc["phases"]["setup"]["ram_peak_bytes"] = 5 * 1024**3
        doc["phases"]["experiment"]["ram_peak_bytes"] = 4 * 1024**3
        self.assertEqual(evaluate_plan(doc)["plan_status"], "OVER_BUDGET")

    def test_missing_phase_or_cost_is_invalid(self):
        doc = plan_each()
        del doc["phases"]["export"]
        self.assertEqual(evaluate_plan(doc)["plan_status"], "INVALID_PLAN")
        doc = plan_each()
        del doc["phases"]["tests"]["output_bytes"]
        self.assertEqual(evaluate_plan(doc)["plan_status"], "INVALID_PLAN")

    def test_unknown_or_boolean_enforcement_fields_cannot_self_certify(self):
        for injected in (
            {"enforced": True},
            {"gpu_limit_enforced": True, "cgroup_limit_enforced": True},
            {"enforcement_receipt": {"status": "PASS"}},
        ):
            with self.subTest(injected=injected):
                doc = plan_each()
                doc.update(injected)
                self.assertEqual(evaluate_plan(doc)["plan_status"], "INVALID_PLAN")
                self.assertFalse(evaluate_plan(doc)["launch_admissible"])

    def test_boolean_float_and_negative_costs_are_invalid(self):
        for bad in (True, 1.0, -1):
            with self.subTest(bad=bad):
                doc = plan_each()
                doc["phases"]["setup"]["gpu_seconds"] = bad
                self.assertEqual(evaluate_plan(doc)["plan_status"], "INVALID_PLAN")

    def test_unknown_phase_budget_and_schema_are_invalid(self):
        cases = []
        doc = plan_each()
        doc["phases"]["unbudgeted_setup"] = doc["phases"]["setup"]
        cases.append(doc)
        doc = plan_each()
        doc["budget_id"] = "other"
        cases.append(doc)
        doc = plan_each()
        doc["schema_version"] = True
        cases.append(doc)
        for doc in cases:
            with self.subTest(doc=doc):
                self.assertEqual(evaluate_plan(doc)["plan_status"], "INVALID_PLAN")

    def test_input_is_not_mutated(self):
        doc = plan_each(1)
        before = copy.deepcopy(doc)
        evaluate_plan(doc)
        self.assertEqual(doc, before)

    def test_invalid_field_diagnostics_are_hash_seed_independent(self):
        script = """
import json
from validator import CUMULATIVE_FIELDS, PHASES, evaluate_plan

document = {
    "schema_version": 1,
    "budget_id": "FIM_DELAYED_RECALL_ONCE_V1",
    "already_consumed": {field: True for field in CUMULATIVE_FIELDS},
    "phases": {
        phase: {
            "gpu_seconds": True,
            "wall_seconds": True,
            "aggregate_cpu_seconds": True,
            "ram_peak_bytes": True,
            "output_bytes": True,
        }
        for phase in PHASES
    },
}
print(json.dumps(evaluate_plan(document)["problems"]["plan"]))
"""
        reports = []
        for seed in ("1", "2", "3", "17"):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = seed
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path(__file__).resolve().parent,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            reports.append(json.loads(result.stdout))
        self.assertTrue(all(report == reports[0] for report in reports[1:]))


if __name__ == "__main__":
    unittest.main()
