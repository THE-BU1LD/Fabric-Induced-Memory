"""Adversarial software tests, not scientific performance measurements."""

import ast
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from integrity import ev
from observations import Forecast, Observation, ObservationLedger, project_observed
from recorder import PredictionRecorder


def forecast(identity="f", owner="owner", origin=0):
    return Forecast(identity, owner, origin, (origin + 2, origin + 3), (1., 2.), (0., 1.), "a" * 64)


def event(identity="e", indices=(0, 1), values=(4., 8.), times=(2, 3), now=3, owner="owner", fid="f"):
    return Observation(identity, fid, owner, now, indices, values, times)


def plan():
    return {
        "schema_version": 1, "study_id": "SOFTWARE_TEST_ONLY", "source_sha256": "a" * 64,
        "data_sha256": "b" * 64, "methods": {"candidate": "memory", "baseline": "base"},
        "cells": [{"cell_id": "cell", "family_id": "family"}], "seeds": [1],
        "samples": [{"cell_id": "cell", "seed": 1, "trajectory_id": "owner", "origin": 0}],
        "metric": {"name": "mean_sample_relative_l2", "epsilon": 1e-12},
        "gates": {"minimum_macro_gain": .1, "minimum_cells_won": 1,
                  "maximum_cell_regression": .02, "minimum_seed_win_fraction": .8,
                  "minimum_families_won": 1, "maximum_family_regression": .02,
                  "maximum_candidate_error": None},
        "bootstrap": {"repetitions": 100, "confidence": .95, "seed": 4},
    }


def recorder(p=None):
    p = p or plan()
    raw = json.dumps(p).encode()
    return PredictionRecorder(raw, expected_sha256=hashlib.sha256(raw).hexdigest())


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.ledger = ObservationLedger()
        self.ledger.register_many([forecast()], now=0)

    def rejected_unchanged(self, e, now=3):
        before = self.ledger.checksum()
        with self.assertRaises(ValueError):
            self.ledger.observe(e, now=now)
        self.assertEqual(before, self.ledger.checksum())

    def test_hand_calculated_residual_uses_saved_base(self):
        ready = self.ledger.observe(event(), now=3)
        self.assertEqual(ready.residual, (3., 6.))
        self.assertEqual(ready.forecast.key, (0., 1.))

    def test_partial_reveals_wait_for_completeness(self):
        self.assertIsNone(self.ledger.observe(event(indices=(0,), values=(4,), times=(2,), now=2), now=2))
        ready = self.ledger.observe(event(identity="second", indices=(1,), values=(8,), times=(3,), now=6), now=6)
        self.assertEqual(ready.available_at, 6)
        self.assertEqual(ready.residual, (3., 6.))

    def test_out_of_order_target_reveals_use_final_arrival(self):
        self.ledger.observe(event(indices=(1,), values=(8,), times=(3,), now=4), now=4)
        ready = self.ledger.observe(event(identity="late", indices=(0,), values=(4,), times=(2,), now=7), now=7)
        self.assertEqual(ready.available_at, 7)

    def test_hidden_nan_infinity_and_poison_never_read(self):
        class Poison:
            def __float__(self):
                raise AssertionError("Hidden value accessed")
        results = []
        for hidden in (math.nan, math.inf, -math.inf, Poison(), 1e30):
            ledger = ObservationLedger()
            ledger.register_many([forecast()], now=0)
            e = project_observed(event_id="visible", forecast_id="f", owner="owner", arrived_at=2,
                                 values=[4, hidden], mask=[True, False], available_at=[2, Poison()])
            ledger.observe(e, now=2)
            results.append(ledger.checksum())
        self.assertEqual(len(set(results)), 1)

    def test_non_boolean_masks_rejected(self):
        for mask in ([1, 0], [True, math.nan]):
            with self.assertRaises(ValueError):
                project_observed(event_id="e", forecast_id="f", owner="o", arrived_at=2,
                                 values=[1, 2], mask=mask, available_at=[2, 3])

    def test_empty_reveal_rejected(self):
        self.rejected_unchanged(event(indices=(), values=(), times=()))

    def test_unknown_forecast_and_wrong_owner_rejected(self):
        self.rejected_unchanged(event(fid="wrong"))
        self.rejected_unchanged(event(owner="wrong"))

    def test_duplicate_event_rejected(self):
        self.ledger.observe(event(indices=(0,), values=(4,), times=(2,)), now=3)
        self.rejected_unchanged(event(indices=(1,), values=(8,), times=(3,)))

    def test_duplicate_coordinate_transaction_is_atomic(self):
        self.rejected_unchanged(event(indices=(0, 0)))

    def test_revised_coordinate_rejected(self):
        self.ledger.observe(event(indices=(0,), values=(4,), times=(2,)), now=3)
        self.rejected_unchanged(event(identity="revise", indices=(0,), values=(99,), times=(2,)))

    def test_future_and_premature_availability_rejected(self):
        self.rejected_unchanged(event(times=(4, 3)))
        self.rejected_unchanged(event(times=(1, 3)))

    def test_actual_arrival_required_and_clock_cannot_reverse(self):
        self.rejected_unchanged(event(now=4), now=3)
        self.ledger.observe(event(indices=(0,), values=(4,), times=(2,), now=4), now=4)
        self.rejected_unchanged(event(identity="late", indices=(1,), values=(8,), times=(3,)), now=3)

    def test_predictions_before_reveals_at_same_time(self):
        self.ledger.register_many([forecast("g", origin=3)], now=3)
        self.ledger.observe(event(), now=3)
        with self.assertRaises(ValueError):
            self.ledger.register_many([forecast("h", origin=3)], now=3)
        self.ledger.register_many([forecast("h", origin=4)], now=4)

    def test_completed_forecast_released_once(self):
        self.ledger.observe(event(), now=3)
        self.rejected_unchanged(event(identity="again"))

    def test_nonfinite_and_bool_observed_values_rejected(self):
        for value in (math.nan, math.inf, True, "3"):
            self.rejected_unchanged(event(values=(4, value)))

    def test_residual_overflow_rejected_atomically(self):
        ledger = ObservationLedger()
        ledger.register_many([replace(forecast(), base=(-1e308, 2))], now=0)
        before = ledger.checksum()
        with self.assertRaises(ValueError):
            ledger.observe(event(values=(1e308, 3)), now=3)
        self.assertEqual(ledger.checksum(), before)

    def test_invalid_indices_and_times_rejected(self):
        for index in (-1, 2, True, 0.0):
            self.rejected_unchanged(event(indices=(index, 1)))
        self.rejected_unchanged(event(times=(True, 3)))

    def test_forecast_batch_rejection_is_atomic(self):
        before = self.ledger.checksum()
        with self.assertRaises(ValueError):
            self.ledger.register_many([forecast("g"), forecast("f")], now=0)
        self.assertEqual(self.ledger.checksum(), before)

    def test_forecast_snapshots_are_immutable(self):
        base = [1, 2]
        ledger = ObservationLedger()
        ledger.register_many([replace(forecast(), base=base)], now=0)
        base[0] = 999
        self.assertEqual(ledger.observe(event(), now=3).residual, (3, 6))

    def test_owner_isolation_with_interleaved_events(self):
        self.ledger.register_many([forecast("other", owner="other")], now=0)
        self.ledger.observe(event(identity="other", fid="other", owner="other", values=(100, 100)), now=3)
        self.assertEqual(self.ledger.observe(event(), now=3).residual, (3, 6))

    def test_budgets_fail_without_eviction(self):
        for kwargs in ({"max_forecasts": 1}, {"max_scalars": 14}):
            ledger = ObservationLedger(**kwargs)
            ledger.register_many([forecast()], now=0)
            before = ledger.checksum()
            with self.assertRaises(ValueError):
                ledger.register_many([forecast("other")], now=0)
            self.assertEqual(before, ledger.checksum())
        ledger = ObservationLedger(max_events=1)
        ledger.register_many([forecast()], now=0)
        ledger.observe(event(indices=(0,), values=(4,), times=(2,)), now=3)
        before = ledger.checksum()
        with self.assertRaises(ValueError):
            ledger.observe(event(identity="new", indices=(1,), values=(8,), times=(3,)), now=3)
        self.assertEqual(before, ledger.checksum())


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.r = recorder()
        self.sample = plan()["samples"][0]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/"bundle.json"

    def complete(self, candidate=(.5,), baseline=(0,), target=(1,)):
        self.r.record_prediction("memory", self.sample, candidate)
        self.r.record_prediction("base", self.sample, baseline)
        self.r.record_target(self.sample, target)

    def test_actual_export_passes_existing_validator_with_exact_metrics(self):
        self.complete()
        bundle = self.r.finalize(self.path)
        self.assertEqual(bundle["result"]["metrics"]["memory"]["relative_l2"], .5)
        self.assertEqual(bundle["result"], ev.analyze(bundle["plan"], bundle["evidence"]))
        self.assertEqual(bundle["claim_authorization"], ev.CLAIM_BOUNDARY)

    def test_worse_prediction_has_valid_evidence_and_fails_numerical_gates(self):
        self.complete(candidate=(-1,))
        result = self.r.finalize(self.path)["result"]
        self.assertEqual(result["status"], "PASS_INTEGRITY")
        self.assertFalse(result["numerical_gates_pass"])

    def test_missing_results_retained_as_failure_not_filtered(self):
        self.r.record_prediction("memory", self.sample, [.5])
        bundle = self.r.finalize(self.path)
        self.assertEqual(bundle["result"]["status"], "FAIL_INTEGRITY")
        self.assertEqual(len(bundle["evidence"]["rows"]), 2)
        self.assertEqual(bundle["evidence"]["rows"][0]["prediction"], [.5])

    def test_failed_run_blocks_aggregate_and_preserves_prediction(self):
        self.r.record_prediction("memory", self.sample, [.5])
        self.r.record_failure("memory", self.sample, "resource stop")
        result = self.r.finalize(self.path)
        self.assertEqual(result["result"]["status"], "FAIL_INTEGRITY")
        self.assertEqual(result["evidence"]["rows"][0]["error"], "resource stop")

    def test_targets_cannot_arrive_before_both_predictions(self):
        with self.assertRaises(ValueError):
            self.r.record_target(self.sample, [1])
        self.r.record_prediction("memory", self.sample, [.5])
        with self.assertRaises(ValueError):
            self.r.record_target(self.sample, [1])

    def test_prediction_after_truth_and_duplicate_truth_rejected(self):
        self.complete()
        with self.assertRaises(ValueError):
            self.r.record_prediction("memory", self.sample, [1])
        with self.assertRaises(ValueError):
            self.r.record_target(self.sample, [1])

    def test_wrong_plan_identity_rejected(self):
        with self.assertRaises(ValueError):
            PredictionRecorder(json.dumps(plan()).encode(), expected_sha256="c"*64)

    def test_plan_mutation_cannot_change_inventory(self):
        p = plan()
        r = recorder(p)
        p["samples"].clear()
        self.assertEqual(len(r.payload()["rows"]), 2)

    def test_duplicate_and_unplanned_rows_rejected(self):
        self.r.record_prediction("memory", self.sample, [1])
        for method, sample in (("memory", self.sample), ("bad", self.sample), ("base", dict(self.sample, origin=9))):
            with self.assertRaises(ValueError):
                self.r.record_prediction(method, sample, [1])

    def test_shape_mismatch_and_nonfinite_rejected(self):
        for values in ([math.nan], [True], []):
            with self.assertRaises(ValueError):
                self.r.record_prediction("memory", self.sample, values)
        self.r.record_prediction("memory", self.sample, [.5])
        with self.assertRaises(ValueError):
            self.r.record_prediction("base", self.sample, [1, 2])

    def test_mutating_input_and_payload_does_not_change_predictions(self):
        value = [.5]
        self.r.record_prediction("memory", self.sample, value)
        value[0] = 100
        payload = self.r.payload()
        payload["rows"][0]["prediction"][0] = 200
        self.assertEqual(self.r.payload()["rows"][0]["prediction"], [.5])

    def test_atomic_publication_never_overwrites(self):
        self.complete()
        self.path.write_text("preserved")
        with self.assertRaises(FileExistsError):
            self.r.finalize(self.path)
        self.assertEqual(self.path.read_text(), "preserved")
        self.assertEqual(sorted(p.name for p in self.path.parent.iterdir()), ["bundle.json"])

    def test_finalization_closes_recorder(self):
        self.complete()
        self.r.finalize(self.path)
        with self.assertRaises(ValueError):
            self.r.finalize(self.path.parent/"second.json")
        with self.assertRaises(ValueError):
            self.r.record_failure("memory", self.sample, "late")

    def test_bundle_raw_evidence_hash_is_recomputable(self):
        self.complete()
        self.r.finalize(self.path)
        b = json.loads(self.path.read_text())
        raw = json.dumps(b["evidence"], allow_nan=False, sort_keys=True).encode()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), b["evidence_sha256"])


class SourceTests(unittest.TestCase):
    def test_tensor_bridge_is_syntactically_valid_without_import(self):
        tree = ast.parse((Path(__file__).parent/"tensor_bridge.py").read_text())
        inputs = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ForecastInputs")
        fields = [n.target.id for n in inputs.body if isinstance(n, ast.AnnAssign)]
        self.assertNotIn("target", fields)
        self.assertIn("residual_shape", fields)


if __name__ == "__main__":
    unittest.main()
