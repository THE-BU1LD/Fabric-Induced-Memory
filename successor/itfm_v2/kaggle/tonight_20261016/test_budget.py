"""Standard-library tests; simulated clock and no cloud or tensor imports."""
import json
from pathlib import Path
import tempfile
import unittest

from budget import AccountBudget, AdmissionError, quota_policy, utc_timestamp


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "shared-account.json"
        self.now = [utc_timestamp("2026-10-16T06:36:53Z")]

    def tearDown(self):
        self.directory.cleanup()

    def create(self, **kwargs):
        values = dict(account_id="private-account-alias", q0_hours=30,
                      observed_at=self.now[0], active_sessions=[], sessions_observed_at=self.now[0],
                      deadline_utc="2026-10-16T16:30:00Z", clock=lambda: self.now[0])
        values.update(kwargs)
        return AccountBudget.create(self.path, **values)

    def allocated(self, count=1, **kwargs):
        ledger = self.create(**kwargs)
        ledger.reserve_session("fim-session", "FIM", count)
        ledger.confirm_allocation("fim-session", observed_at=self.now[0])
        return ledger

    def refresh(self, ledger, quota=30, sessions=None):
        ledger.observe(quota_hours=quota, observed_at=self.now[0],
                       active_sessions=["fim-session"] if sessions is None else sessions,
                       sessions_observed_at=self.now[0])

    def test_policy_uses_observed_q0_and_expansion_never_automatic(self):
        self.assertEqual(quota_policy(30), dict(preferred_budget_hours=6,
                         possible_expanded_budget_hours=10, expanded_authorized=False))
        self.assertEqual(quota_policy(25)["preferred_budget_hours"], 1)
        self.assertEqual(quota_policy(23)["preferred_budget_hours"], 0)
        self.assertIsNone(quota_policy(None)["preferred_budget_hours"])
        with self.assertRaises(AdmissionError):
            self.create(authorized_hours=7)

    def test_unknown_quota_or_session_inventory_blocks(self):
        ledger = self.create(q0_hours=None, observed_at=None, active_sessions=None, sessions_observed_at=None)
        self.assertIsNone(ledger.snapshot()["q0_hours"])
        with self.assertRaises(AdmissionError):
            ledger.reserve_session("s", "FIM", 1)
        self.refresh(ledger, quota=29, sessions=[])
        ledger.reserve_session("s", "FIM", 1)
        self.assertEqual(ledger.snapshot()["authorized_hours"], 5)

    def test_stale_observation_and_other_active_jobs_refuse_admission(self):
        ledger = self.create()
        self.now[0] += 301
        with self.assertRaises(AdmissionError):
            ledger.reserve_session("s", "FIM", 1)
        self.refresh(ledger, sessions=["unrelated-job"])
        with self.assertRaisesRegex(AdmissionError, "coordination"):
            ledger.reserve_session("s", "FIM", 1)
        self.assertEqual(ledger.snapshot()["sessions"], {})

    def test_allocation_confirmation_requires_fresh_evidence(self):
        ledger = self.create()
        ledger.reserve_session("s", "FIM", 1)
        self.now[0] += 301
        with self.assertRaises(AdmissionError):
            ledger.confirm_allocation("s", observed_at=self.now[0])
        self.assertEqual(ledger.snapshot()["sessions"]["s"]["state"], "PENDING_ALLOCATION")

    def test_zero_preferred_budget_requires_revised_user_allocation(self):
        ledger = self.create(q0_hours=24)
        with self.assertRaisesRegex(AdmissionError, "zero"):
            ledger.reserve_session("s", "FIM", 1)

    def test_shared_ledger_prevents_cross_project_concurrency(self):
        ledger = self.allocated()
        other = AccountBudget(self.path, clock=lambda: self.now[0])
        with self.assertRaisesRegex(AdmissionError, "not confirmed stopped"):
            other.reserve_session("apen-session", "APEN", 1)
        self.assertEqual(list(ledger.snapshot()["sessions"]), ["fim-session"])

    def test_run_reservation_is_persisted_before_child_and_cannot_duplicate(self):
        ledger = self.allocated()
        permit = ledger.reserve_run("fim-session", "pilot", estimated_seconds=600, export_shutdown_seconds=60)
        self.assertEqual(permit["state"], "RUNNING")
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["sessions"]["fim-session"]["runs"]["pilot"]["state"], "RUNNING")
        self.assertAlmostEqual(permit["export_shutdown_margin_seconds"], 2160)
        with self.assertRaisesRegex(AdmissionError, "double reservation"):
            ledger.reserve_run("fim-session", "pilot", estimated_seconds=600, export_shutdown_seconds=60)
        with self.assertRaisesRegex(AdmissionError, "concurrent"):
            ledger.reserve_run("fim-session", "parallel", estimated_seconds=600, export_shutdown_seconds=60)

    def test_ten_percent_reserve_is_not_training_capacity(self):
        ledger = self.allocated()
        with self.assertRaisesRegex(AdmissionError, "reserved"):
            ledger.reserve_run("fim-session", "too-long", estimated_seconds=5.5 * 3600, export_shutdown_seconds=60)
        run = ledger.snapshot()["sessions"]["fim-session"]["runs"]["too-long"]
        self.assertEqual(run["state"], "SKIPPED_BUDGET")

    def test_absolute_deadline_refuses_run_with_export_margin(self):
        ledger = self.allocated()
        self.now[0] = utc_timestamp("2026-10-16T16:00:00Z")
        self.refresh(ledger, quota=30)
        # Allocate a fresh ledger at this clock to isolate the deadline guard.
        path = self.path.parent / "fresh.json"
        ledger = AccountBudget.create(path, account_id="alias", q0_hours=30,
                    observed_at=self.now[0], active_sessions=[], sessions_observed_at=self.now[0],
                    deadline_utc="2026-10-16T16:30:00Z", authorized_hours=1,
                    clock=lambda: self.now[0])
        ledger.reserve_session("s", "FIM", 1)
        ledger.confirm_allocation("s", observed_at=self.now[0])
        with self.assertRaisesRegex(AdmissionError, "deadline"):
            ledger.reserve_run("s", "late", estimated_seconds=1500, export_shutdown_seconds=360)

    def test_allocated_device_count_charges_idle_time_conservatively(self):
        ledger = self.allocated(count=2)
        self.now[0] += 1800
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot["estimates"]["wall_clock_allocated_seconds_estimated"], 1800)
        self.assertEqual(snapshot["estimates"]["physical_device_hours_estimated"], 1)
        self.assertEqual(snapshot["estimates"]["quota_remaining_estimate"], 29)
        self.assertFalse(snapshot["platform_limit_guaranteed"])

    def test_observed_other_debit_plus_current_idle_time_counts(self):
        ledger = self.allocated()
        self.refresh(ledger, quota=25)
        self.now[0] += 1800
        self.assertEqual(ledger.snapshot()["estimates"]["quota_spent_conservative_estimate"], 5.5)

    def test_absolute_floor_blocks_new_run_after_observed_account_debit(self):
        ledger = self.allocated()
        self.refresh(ledger, quota=20.1)
        with self.assertRaises(AdmissionError):
            ledger.reserve_run("fim-session", "blocked", estimated_seconds=600, export_shutdown_seconds=60)

    def test_all_lifecycle_states_and_artifact_success_gate(self):
        ledger = self.allocated()
        for name, status in (("first", "SUCCEEDED"), ("second", "FAILED"), ("third", "INTERRUPTED")):
            ledger.reserve_run("fim-session", name, estimated_seconds=60, export_shutdown_seconds=60)
            if status == "SUCCEEDED":
                with self.assertRaises(AdmissionError):
                    ledger.finish_run("fim-session", name, status=status)
            ledger.finish_run("fim-session", name, status=status, artifacts_verified=status == "SUCCEEDED")
        states = ledger.snapshot()["sessions"]["fim-session"]["runs"]
        self.assertEqual([value["state"] for value in states.values()], ["SUCCEEDED", "FAILED", "INTERRUPTED"])

    def test_process_exit_does_not_stop_allocation_charging(self):
        ledger = self.allocated()
        ledger.reserve_run("fim-session", "r", estimated_seconds=60, export_shutdown_seconds=60)
        ledger.finish_run("fim-session", "r", status="INTERRUPTED")
        ledger.request_stop("fim-session")
        self.now[0] += 60
        first = ledger.snapshot()
        self.assertFalse(first["all_campaign_sessions_confirmed_stopped"])
        self.assertGreater(first["estimates"]["physical_device_hours_estimated"], 0)
        ledger.confirm_session_stopped("fim-session", observed_at=self.now[0])
        stopped_hours = ledger.snapshot()["estimates"]["physical_device_hours_estimated"]
        self.now[0] += 60
        self.assertEqual(ledger.snapshot()["estimates"]["physical_device_hours_estimated"], stopped_hours)
        self.assertTrue(ledger.snapshot()["all_campaign_sessions_confirmed_stopped"])

    def test_success_does_not_hide_timeout_violation(self):
        ledger = self.allocated()
        ledger.reserve_run("fim-session", "r", estimated_seconds=60, export_shutdown_seconds=60)
        self.now[0] += 61
        ledger.finish_run("fim-session", "r", status="SUCCEEDED", artifacts_verified=True)
        self.assertTrue(ledger.snapshot()["sessions"]["fim-session"]["runs"]["r"]["timeout_exceeded"])

    def test_existing_account_ledger_cannot_be_reset(self):
        ledger = self.create()
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaises(ValueError):
            ledger.observe(quota_hours=30, observed_at=self.now[0] + 1,
                           active_sessions=[], sessions_observed_at=self.now[0])

    def test_corrupt_loaded_policy_fails_closed_without_overwriting_evidence(self):
        self.create()
        valid = self.path.read_bytes()
        for key, value in (("authorized_hours", float("nan")), ("authorized_hours", 7),
                           ("reserve_fraction", 0.01), ("q0_hours", -1),
                           ("schema_version", True), ("platform_limit_guaranteed", True)):
            with self.subTest(field=key, value=value):
                state = json.loads(valid)
                state[key] = value
                self.path.write_text(json.dumps(state))
                corrupt = self.path.read_bytes()
                ledger = AccountBudget(self.path, clock=lambda: self.now[0])
                with self.assertRaises(AdmissionError):
                    ledger.reserve_session("blocked", "FIM", 1)
                self.assertEqual(self.path.read_bytes(), corrupt)

    def test_nan_quota_missing_fields_and_malformed_json_fail_closed(self):
        self.create()
        state = json.loads(self.path.read_text())
        state["quota_observation"]["hours"] = float("nan")
        cases = [json.dumps(state), "{broken", "[]", '{"schema_version": 1}']
        for serialized in cases:
            with self.subTest(serialized=serialized[:40]):
                self.path.write_text(serialized)
                ledger = AccountBudget(self.path, clock=lambda: self.now[0])
                with self.assertRaises(AdmissionError):
                    ledger.snapshot()
                self.assertEqual(self.path.read_text(), serialized)

    def test_loaded_device_multiplier_cannot_undercharge_t4_pair(self):
        self.allocated(count=2)
        state = json.loads(self.path.read_text())
        state["sessions"]["fim-session"]["quota_multiplier"] = 1
        self.path.write_text(json.dumps(state))
        ledger = AccountBudget(self.path, clock=lambda: self.now[0])
        with self.assertRaisesRegex(AdmissionError, "every allocated device"):
            ledger.reserve_run("fim-session", "forbidden", estimated_seconds=60, export_shutdown_seconds=60)

    def test_stop_observation_cannot_backdate_already_completed_work(self):
        ledger = self.allocated()
        start = self.now[0]
        ledger.reserve_run("fim-session", "r", estimated_seconds=60, export_shutdown_seconds=60)
        self.now[0] += 50
        ledger.finish_run("fim-session", "r", status="SUCCEEDED", artifacts_verified=True)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(AdmissionError, "preceded recorded session activity"):
            ledger.confirm_session_stopped("fim-session", observed_at=start)
        self.assertEqual(self.path.read_bytes(), before)

    def test_clock_regression_does_not_refund_account_usage(self):
        ledger = self.allocated()
        self.now[0] -= 1
        with self.assertRaises(AdmissionError):
            ledger.snapshot()


if __name__ == "__main__":
    unittest.main()
