"""Adversarial software fixtures. None are scientific observations."""
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import evidence as ev


def fixture(cells=2, seeds=(101, 211), counts=None, families=None, errors=None):
    counts = counts or [1] * cells
    families = families or [f"family-{i}" for i in range(cells)]
    errors = errors or [.5] * cells
    plan = {
        "schema_version": 1, "study_id": "SOFTWARE_FIXTURE_ONLY",
        "source_sha256": "a" * 64, "data_sha256": "b" * 64,
        "methods": {"candidate": "candidate", "baseline": "baseline"},
        "cells": [{"cell_id": f"cell-{i}", "family_id": families[i]} for i in range(cells)],
        "seeds": list(seeds), "samples": [],
        "metric": {"name": "mean_sample_relative_l2", "epsilon": 1e-12},
        "gates": {"minimum_macro_gain": .1, "minimum_cells_won": cells,
                  "maximum_cell_regression": .02, "minimum_seed_win_fraction": .8,
                  "minimum_families_won": len(set(families)), "maximum_family_regression": .02,
                  "maximum_candidate_error": .8},
        "bootstrap": {"repetitions": 100, "confidence": .95, "seed": 9},
    }
    payload = {"schema_version": 1, "source_sha256": "a" * 64, "data_sha256": "b" * 64, "rows": []}
    for cell in range(cells):
        for seed in seeds:
            for t in range(counts[cell]):
                sample = {"cell_id": f"cell-{cell}", "seed": seed,
                          "trajectory_id": f"trajectory-{t}", "origin": 0}
                plan["samples"].append(sample)
                for method, error in [("candidate", errors[cell]), ("baseline", 1.)]:
                    payload["rows"].append(dict(sample, method=method, status="PASS",
                                                prediction=[1. - error], target=[1.]))
    return plan, payload


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.payload = fixture()

    def rejects(self, code):
        return self.assertRaisesRegex(ev.EvidenceError, code)

    def assert_invalid(self, plan=None, payload=None, code=None):
        with self.assertRaises(ev.EvidenceError) as error:
            ev.analyze(self.plan if plan is None else plan, self.payload if payload is None else payload)
        if code:
            self.assertEqual(error.exception.code, code)

    def test_benign_raw_evidence_has_no_scientific_authority(self):
        result = ev.analyze(self.plan, self.payload)
        self.assertEqual(result["status"], "PASS_INTEGRITY")
        self.assertTrue(result["numerical_gates_pass"])
        self.assertEqual(result["claim_authorization"], ev.CLAIM_BOUNDARY)
        self.assertFalse(result["bootstrap"]["confirmatory_authorized"])
        self.assertNotIn("broad_sota_allowed", result)

    def test_row_order_invariance(self):
        expected = ev.analyze(self.plan, self.payload)
        random.Random(13).shuffle(self.payload["rows"])
        self.assertEqual(expected, ev.analyze(self.plan, self.payload))

    def test_duplicate_rows_cannot_replace_required_identities(self):
        self.payload["rows"] = [copy.deepcopy(self.payload["rows"][0])] * len(self.payload["rows"])
        self.assert_invalid(code="DUPLICATE")

    def test_duplicate_planned_cells_rejected(self):
        self.plan["cells"][1] = dict(self.plan["cells"][0])
        self.assert_invalid(code="DUPLICATE")

    def test_duplicate_planned_seeds_rejected(self):
        self.plan["seeds"].append(self.plan["seeds"][0])
        self.assert_invalid(code="DUPLICATE")

    def test_duplicate_planned_samples_rejected(self):
        self.plan["samples"].append(dict(self.plan["samples"][0]))
        self.assert_invalid(code="DUPLICATE")

    def test_plan_cannot_omit_a_cell_seed_block(self):
        self.plan["samples"] = [x for x in self.plan["samples"] if not (x["cell_id"] == "cell-1" and x["seed"] == 211)]
        self.assert_invalid(code="COVERAGE")

    def test_one_failed_candidate_blocks_aggregation_and_retains_failure(self):
        self.payload["rows"][0].update(status="OOM", error="fixture failure")
        with self.assertRaises(ev.EvidenceError) as error:
            ev.analyze(self.plan, self.payload)
        self.assertEqual(error.exception.code, "INCOMPLETE")
        self.assertEqual(error.exception.details["failed_rows"][0]["status"], "OOM")

    def test_one_missing_candidate_blocks_aggregation(self):
        self.payload["rows"].pop(0)
        self.assert_invalid(code="INCOMPLETE")

    def test_unexpected_method_cell_seed_and_origin_rejected(self):
        for field, value in [("method", "extra"), ("cell_id", "extra"), ("seed", 999), ("origin", 999)]:
            with self.subTest(field=field):
                payload = copy.deepcopy(self.payload)
                payload["rows"][0][field] = value
                self.assert_invalid(payload=payload, code="UNEXPECTED")

    def test_wrong_source_or_data_identity_rejected(self):
        for key in ("source_sha256", "data_sha256"):
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload[key] = "c" * 64
                self.assert_invalid(payload=payload, code="PROVENANCE")

    def test_candidate_and_baseline_target_mismatch_rejected(self):
        self.payload["rows"][0]["target"] = [2.]
        self.assert_invalid(code="TARGET_MISMATCH")

    def test_shape_mismatch_rejected(self):
        self.payload["rows"][0]["prediction"] = [1., 2.]
        self.assert_invalid(code="SHAPE")

    def test_nonfinite_raw_values_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for field in ("prediction", "target"):
                with self.subTest(value=value, field=field):
                    payload = copy.deepcopy(self.payload)
                    payload["rows"][0][field] = [value]
                    self.assert_invalid(payload=payload, code="NONFINITE")

    def test_nonfinite_claims_on_either_side_cannot_pass(self):
        for a, b in [(float("nan"), 1.), (1., float("nan")), (float("inf"), float("inf"))]:
            with self.subTest(a=a, b=b), self.assertRaises(ev.EvidenceError):
                ev.verify_metric_claims({"mse": a}, {"mse": b})

    def test_claim_comparison_checks_keys_values_and_tolerance(self):
        ev.verify_metric_claims({"mse": 1.}, {"mse": 1.})
        for a, b in [({}, {}), ({"mse": 1.}, {"mse": 2.}), ({"mse": 1.}, {"l2": 1.})]:
            with self.subTest(a=a, b=b), self.assertRaises(ev.EvidenceError):
                ev.verify_metric_claims(a, b)
        with self.assertRaises(ev.EvidenceError):
            ev.verify_metric_claims({"mse": 1.}, {"mse": 1.}, float("nan"))

    def test_raw_claims_recomputed_not_trusted(self):
        result = ev.analyze(self.plan, self.payload)
        self.payload["claimed_metrics"] = copy.deepcopy(result["metrics"])
        ev.analyze(self.plan, self.payload)
        self.payload["claimed_metrics"]["candidate"]["relative_l2"] = 0.
        self.assert_invalid(code="METRIC_MISMATCH")

    def test_mean_sample_l2_is_not_pooled_energy_ratio(self):
        values = [ev.sample_metrics([0.], [1.], 1e-12)["relative_l2"],
                  ev.sample_metrics([10.], [10.], 1e-12)["relative_l2"]]
        self.assertEqual(ev.mean(values), .5)
        self.assertNotAlmostEqual(ev.mean(values), math.sqrt(1 / 101))

    def test_full_pipeline_uses_same_sample_metric(self):
        plan, payload = fixture(cells=1, seeds=(1,), counts=[2])
        # Separate trajectories with equal weight: target energies differ 100-fold.
        for row in payload["rows"]:
            if row["trajectory_id"] == "trajectory-0":
                row.update(target=[1.], prediction=[0.])
            else:
                row.update(target=[10.], prediction=[10.] if row["method"] == "candidate" else [0.])
        self.assertEqual(ev.analyze(plan, payload)["metrics"]["candidate"]["relative_l2"], .5)

    def test_equal_cell_macro_survives_unequal_trajectory_counts(self):
        plan, payload = fixture(cells=2, seeds=(1,), counts=[1, 9], families=["same", "same"], errors=[0., 1.])
        result = ev.analyze(plan, payload)
        self.assertEqual(result["metrics"]["candidate"]["relative_l2"], .5)
        self.assertEqual(result["bootstrap"]["estimate"], .5)
        self.assertIsNone(result["bootstrap"]["lower"])

    def test_trajectory_then_seed_then_cell_reduction(self):
        plan, payload = fixture(cells=1, seeds=(1, 2), counts=[2], errors=[0.])
        extra = {"cell_id": "cell-0", "seed": 1, "trajectory_id": "trajectory-0", "origin": 1}
        plan["samples"].append(extra)
        for method in ("candidate", "baseline"):
            payload["rows"].append(dict(extra, method=method, status="PASS", prediction=[0.], target=[1.]))
        # candidate: seed 1 trajectory means .5 and 0 -> .25; seed 2 -> 0; cell -> .125
        self.assertEqual(ev.analyze(plan, payload)["metrics"]["candidate"]["relative_l2"], .125)

    def test_family_regression_cannot_hide_behind_good_average(self):
        plan, payload = fixture(cells=5, seeds=(1,), errors=[.1, .025, .025, .025, .025])
        for row in payload["rows"]:
            if row["method"] == "baseline":
                row["prediction"] = [.95]
        plan["gates"].update(minimum_cells_won=4, minimum_families_won=4,
                             maximum_cell_regression=2., minimum_seed_win_fraction=0.)
        result = ev.analyze(plan, payload)
        self.assertTrue(result["numerical_gates"]["minimum_macro_gain"])
        self.assertTrue(result["numerical_gates"]["maximum_cell_regression"])
        self.assertFalse(result["numerical_gates"]["maximum_family_regression"])
        self.assertFalse(result["numerical_gates_pass"])

    def test_each_other_numerical_gate_is_enforced(self):
        mutations = {"minimum_macro_gain": .6, "maximum_candidate_error": .4,
                     "maximum_cell_regression": 0., "minimum_seed_win_fraction": 1.,
                     "minimum_cells_won": 2, "minimum_families_won": 2}
        for name, threshold in mutations.items():
            with self.subTest(gate=name):
                plan, payload = fixture(errors=[1.01, .5])
                plan["gates"][name] = threshold
                self.assertFalse(ev.analyze(plan, payload)["numerical_gates"][name])

    def test_independence_boolean_is_not_an_input_authority(self):
        self.payload["independently_reproduced"] = True
        self.assert_invalid(code="SCHEMA")

    def test_supplied_win_flags_rejected(self):
        self.payload["rows"][0]["win"] = True
        self.assert_invalid(code="SCHEMA")

    def test_zero_error_baseline_policy_is_explicit(self):
        self.assertEqual(ev.relative_gain(0., 0.), 0.)
        with self.assertRaises(ev.EvidenceError):
            ev.relative_gain(1., 0.)
        self.assertEqual(ev.sample_metrics([0.], [0.], 1e-12), {"mse": 0., "relative_l2": 0.})

    def test_overflow_is_rejected(self):
        for prediction, target in [([1e308], [-1e308]), ([0.], [1e200])]:
            with self.subTest(prediction=prediction), self.assertRaises(ev.EvidenceError):
                ev.sample_metrics(prediction, target, 1e-12)

    def test_strict_numeric_types_and_bounds(self):
        for value in [True, "0.5", None]:
            with self.subTest(value=value):
                payload = copy.deepcopy(self.payload)
                payload["rows"][0]["prediction"] = [value]
                self.assert_invalid(payload=payload, code="TYPE")
        for path, value in [("seeds", [True]), ("schema_version", True)]:
            plan = copy.deepcopy(self.plan)
            plan[path] = value
            self.assert_invalid(plan=plan, code="TYPE")
        self.plan["metric"]["epsilon"] = 0
        self.assert_invalid(code="METRIC")

    def test_empty_arrays_and_method_alias_rejected(self):
        self.payload["rows"][0]["prediction"] = []
        self.assert_invalid(code="SCHEMA")
        plan, payload = fixture()
        plan["methods"]["baseline"] = plan["methods"]["candidate"]
        self.assert_invalid(plan=plan, payload=payload, code="IDENTITY")

    def test_bootstrap_uses_cells_within_whole_family_draws(self):
        gains, families = {"a": 1., "b": 0., "c": 0.}, {"a": "x", "b": "x", "c": "y"}
        result = ev.cell_macro_bootstrap(gains, families, repetitions=100, confidence=.95, seed=9)
        self.assertAlmostEqual(result["estimate"], 1 / 3)
        self.assertEqual((result["lower"], result["upper"]), (0., .5))
        self.assertEqual(result, ev.cell_macro_bootstrap(gains, families, repetitions=100, confidence=.95, seed=9))
        self.assertEqual(result["interval_status"], "EXPLORATORY_NOT_CONFIRMATORY")

    def test_strict_json_rejects_duplicate_keys_constants_and_malformed_input(self):
        for raw in [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{', b'\xff']:
            with self.subTest(raw=raw), self.assertRaises(ev.EvidenceError):
                ev.strict_json(raw)

    def test_malformed_status_and_failure_detail_are_refused(self):
        for key, value in [("status", {"unexpected": True}), ("error", {"bad": float("inf")})]:
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload["rows"][0]["status"] = "FAILED"
                payload["rows"][0][key] = value
                self.assert_invalid(payload=payload)

    def test_huge_integer_json_refusal_uses_evidence_error(self):
        if not hasattr(sys, "get_int_max_str_digits") or sys.get_int_max_str_digits() == 0:
            self.skipTest("Interpreter integer string limit unavailable")
        with self.assertRaises(ev.EvidenceError) as error:
            ev.strict_json('{"a":' + '9' * (sys.get_int_max_str_digits() + 1) + '}')
        self.assertEqual(error.exception.code, "JSON")

    def test_standalone_gain_and_bootstrap_reject_invalid_types(self):
        for a, b in [(float("inf"), 1.), (-1., 1.), (True, 1.)]:
            with self.subTest(a=a), self.assertRaises(ev.EvidenceError):
                ev.relative_gain(a, b)
        with self.assertRaises(ev.EvidenceError):
            ev.cell_macro_bootstrap({1: .5, "a": .5}, {1: "x", "a": "x"}, repetitions=100, confidence=.95, seed=1)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fim-evidence-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        plan, payload = fixture()
        self.plan, self.payload, self.receipt = [self.root / n for n in ("plan.json", "evidence.json", "receipt.json")]
        self.plan.write_text(json.dumps(plan), encoding="utf-8")
        self.payload.write_text(json.dumps(payload), encoding="utf-8")
        self.expected = hashlib.sha256(self.plan.read_bytes()).hexdigest()

    def run_cli(self, digest=None):
        return subprocess.run([sys.executable, "-B", str(Path(ev.__file__).resolve()),
            "--plan", str(self.plan), "--plan-sha256", self.expected if digest is None else digest,
            "--evidence", str(self.payload), "--receipt", str(self.receipt)],
            capture_output=True, text=True, timeout=5)

    def test_cli_success_hash_bound_receipt(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(receipt["provenance"]["plan_sha256"], self.expected)
        self.assertEqual(receipt["claim_authorization"], ev.CLAIM_BOUNDARY)

    def test_cli_wrong_plan_hash_retains_refusal_and_exits_nonzero(self):
        result = self.run_cli("0" * 64)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(json.loads(self.receipt.read_text())["error_code"], "PLAN_HASH")

    def test_cli_missing_row_retains_input_and_refusal(self):
        payload = json.loads(self.payload.read_text())
        payload["rows"].pop()
        self.payload.write_text(json.dumps(payload))
        before = self.payload.read_bytes()
        self.assertEqual(self.run_cli().returncode, 2)
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(receipt["error_code"], "INCOMPLETE")
        self.assertEqual(self.payload.read_bytes(), before)
        self.assertEqual(receipt["provenance"]["evidence_sha256"], hashlib.sha256(before).hexdigest())

    def test_cli_preserves_existing_receipt(self):
        self.receipt.write_text("existing evidence")
        self.assertEqual(self.run_cli().returncode, 3)
        self.assertEqual(self.receipt.read_text(), "existing evidence")

    def test_cli_malformed_json_retains_failure(self):
        self.payload.write_text('{"rows":')
        self.assertEqual(self.run_cli().returncode, 2)
        self.assertEqual(json.loads(self.receipt.read_text())["error_code"], "JSON")

    def test_cli_missing_input_retains_io_failure(self):
        self.payload.unlink()
        self.assertEqual(self.run_cli().returncode, 2)
        self.assertEqual(json.loads(self.receipt.read_text())["error_code"], "IO")

    def test_bounded_file_read_rejects_large_input(self):
        # Use a temporary small cap to test the branch without allocating a large file.
        original = ev.MAX_INPUT_BYTES
        try:
            ev.MAX_INPUT_BYTES = 4
            with self.assertRaises(ev.EvidenceError):
                ev.read_bounded(self.payload)
        finally:
            ev.MAX_INPUT_BYTES = original

    def test_cli_malformed_failure_detail_retains_valid_refusal_receipt(self):
        payload = json.loads(self.payload.read_text())
        payload["rows"][0].update(status="FAILED", error={"bad": 1.})
        self.payload.write_text(json.dumps(payload).replace('"bad": 1.0', '"bad": 1e999'))
        result = self.run_cli()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(json.loads(self.receipt.read_text())["error_code"], "SCHEMA")

    def test_atomic_receipt_does_not_leave_partial_or_replace_existing_file(self):
        with self.assertRaises(ValueError):
            ev.write_receipt(self.receipt, {"bad": float("inf")})
        self.assertFalse(self.receipt.exists())
        ev.write_receipt(self.receipt, {"ok": True})
        before = self.receipt.read_bytes()
        with self.assertRaises(FileExistsError):
            ev.write_receipt(self.receipt, {"replacement": True})
        self.assertEqual(self.receipt.read_bytes(), before)
        self.assertEqual(list(self.root.glob('.evidence-receipt-*')), [])


if __name__ == "__main__":
    unittest.main()
