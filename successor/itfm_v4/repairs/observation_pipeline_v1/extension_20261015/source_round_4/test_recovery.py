"""Fault-injection and replay tests; fixtures are not forecasting experiments."""

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from durable import DurableObservationLedger, DurableRecorder, audit_observations
from journal import digest, encode
from test_pipeline import event, forecast, plan
from verify_export import verify


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / "journal"
        self.raw = json.dumps(plan()).encode()
        self.sha = hashlib.sha256(self.raw).hexdigest()
        self.sample = plan()["samples"][0]

    def create(self):
        recorder = DurableRecorder.create(
            self.directory, self.raw, expected_sha256=self.sha
        )
        self.addCleanup(recorder.close)
        return recorder

    def resume(self, anchor):
        recorder = DurableRecorder.resume(
            self.directory, expected_anchor=anchor, expected_plan_sha256=self.sha
        )
        self.addCleanup(recorder.close)
        return recorder

    def predictions(self, recorder):
        recorder.record_prediction("memory", self.sample, [0.5])
        recorder.record_prediction("base", self.sample, [0])

    def test_actual_recorder_replay_and_scoring_after_restart(self):
        r = self.create()
        self.predictions(r)
        before, anchor = r.payload(), r.anchor
        r.close()
        recovered = self.resume(anchor)
        self.assertEqual(recovered.payload(), before)
        recovered.record_target(self.sample, [1])
        bundle = recovered.finalize(Path(self.tmp.name) / "bundle.json")
        result = verify(bundle, self.raw, expected_plan_sha256=self.sha)
        self.assertEqual(result["status"], "PASS_REANALYSIS")
        self.assertTrue(result["numerical_gates_pass"])

    def test_failures_retained_through_restart(self):
        r = self.create()
        r.record_prediction("memory", self.sample, [0.5])
        r.record_failure("memory", self.sample, "Interrupted after prediction")
        anchor = r.anchor
        r.close()
        r = self.resume(anchor)
        bundle = r.finalize(Path(self.tmp.name) / "bundle.json")
        self.assertEqual(bundle["result"]["status"], "FAIL_INTEGRITY")
        row = bundle["evidence"]["rows"][0]
        self.assertEqual(row["prediction"], [0.5])
        self.assertEqual(row["error"], "Interrupted after prediction")
        self.assertEqual(
            verify(bundle, self.raw, expected_plan_sha256=self.sha)["evidence_status"],
            "FAIL_INTEGRITY",
        )

    def test_recovery_cannot_generate_new_predictions_or_attach_model(self):
        r = self.create()
        r.record_prediction("memory", self.sample, [0.5])
        anchor = r.anchor
        r.close()
        r = self.resume(anchor)
        with self.assertRaises(ValueError):
            r.record_prediction("base", self.sample, [0])
        with self.assertRaises(ValueError):
            r._key(self.sample)

    def test_invalid_input_leaves_disk_and_memory_unchanged(self):
        r = self.create()
        before = r.payload(), r.anchor, sorted(self.directory.iterdir())
        with self.assertRaises(ValueError):
            r.record_target(self.sample, [1])
        self.assertEqual(
            (r.payload(), r.anchor, sorted(self.directory.iterdir())), before
        )

    def test_write_failure_does_not_commit_memory_and_poison_prevents_retry(self):
        r = self.create()
        before = r.payload(), r.anchor
        with patch("journal.publish", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                r.record_prediction("memory", self.sample, [0.5])
        self.assertEqual((r.payload(), r.anchor), before)
        with self.assertRaises(RuntimeError):
            r.record_prediction("memory", self.sample, [0.5])

    def test_crash_after_publication_is_recoverable_but_old_anchor_rejected(self):
        r = self.create()
        old_anchor = r.anchor
        with patch("journal.sync_directory", side_effect=OSError("crash after link")):
            with self.assertRaises(OSError):
                r.record_prediction("memory", self.sample, [0.5])
        r.close()
        with self.assertRaises(ValueError):
            self.resume(old_anchor)
        record = json.loads((self.directory / "00000001.json").read_text())
        inspected_anchor = {"records": 2, "sha256": record["sha256"]}
        recovered = self.resume(inspected_anchor)
        self.assertEqual(recovered.payload()["rows"][0]["prediction"], [0.5])
        self.assertTrue(list(self.directory.glob(".pending-*")))

    def test_create_never_replaces_existing_journal(self):
        r = self.create()
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(r.anchor["records"], 1)

    def test_concurrent_writer_refused_then_lock_released(self):
        r = self.create()
        anchor = r.anchor
        with self.assertRaises(BlockingIOError):
            self.resume(anchor)
        r.close()
        self.assertEqual(self.resume(anchor).anchor, anchor)

    def test_truncation_detected_by_external_anchor(self):
        r = self.create()
        r.record_prediction("memory", self.sample, [0.5])
        anchor = r.anchor
        r.close()
        (self.directory / "00000001.json").unlink()
        with self.assertRaises(ValueError):
            self.resume(anchor)

    def test_middle_gap_detected(self):
        r = self.create()
        self.predictions(r)
        anchor = r.anchor
        r.close()
        (self.directory / "00000001.json").unlink()
        with self.assertRaises(ValueError):
            self.resume(anchor)

    def test_record_edit_detected(self):
        r = self.create()
        r.record_prediction("memory", self.sample, [0.5])
        anchor = r.anchor
        r.close()
        path = self.directory / "00000001.json"
        record = json.loads(path.read_text())
        record["body"]["event"]["values"] = [9]
        path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            self.resume(anchor)

    def test_rehashed_edit_still_fails_retained_anchor(self):
        r = self.create()
        r.record_prediction("memory", self.sample, [0.5])
        anchor = r.anchor
        r.close()
        path = self.directory / "00000001.json"
        record = json.loads(path.read_text())
        record["body"]["event"]["values"] = [9]
        record["sha256"] = digest(record["body"])
        path.write_bytes(encode(record))
        with self.assertRaises(ValueError):
            self.resume(anchor)

    def test_duplicate_json_keys_refused(self):
        r = self.create()
        anchor = r.anchor
        r.close()
        (self.directory / "00000000.json").write_text(
            '{"body":{},"body":{},"sha256":"a"}'
        )
        with self.assertRaises(ValueError):
            self.resume(anchor)

    def test_symlinked_record_refused(self):
        r = self.create()
        anchor = r.anchor
        r.close()
        record = self.directory / "00000000.json"
        external = Path(self.tmp.name) / "external.json"
        record.rename(external)
        record.symlink_to(external)
        with self.assertRaises(OSError):
            self.resume(anchor)

    def test_changed_plan_and_implementation_refused(self):
        r = self.create()
        anchor = r.anchor
        r.close()
        with self.assertRaises(ValueError):
            DurableRecorder.resume(
                self.directory, expected_anchor=anchor, expected_plan_sha256="c" * 64
            )
        with patch("durable.implementation_identity", return_value={}):
            with self.assertRaises(ValueError):
                self.resume(anchor)

    def test_size_and_record_budgets_fail_before_commit(self):
        r = self.create()
        before = r.anchor
        for name, value in (("MAX_BYTES", 1), ("MAX_RECORDS", 1)):
            with patch("journal." + name, value):
                with self.assertRaises(ValueError):
                    r.record_prediction("memory", self.sample, [0.5])
            self.assertEqual(r.anchor, before)
        r.record_prediction("memory", self.sample, [0.5])

    def test_failed_export_is_sealed_and_can_replay_exactly(self):
        r = self.create()
        self.predictions(r)
        r.record_target(self.sample, [1])
        occupied = Path(self.tmp.name) / "occupied.json"
        occupied.write_text("preserve me")
        with self.assertRaises(FileExistsError):
            r.finalize(occupied)
        with self.assertRaises(ValueError):
            r.record_failure("memory", self.sample, "cannot alter sealed evidence")
        anchor = r.anchor
        r.close()
        r = self.resume(anchor)
        result = r.finalize(Path(self.tmp.name) / "recovered.json")
        self.assertEqual(result["result"]["status"], "PASS_INTEGRITY")
        self.assertEqual(r.anchor, anchor)
        self.assertEqual(occupied.read_text(), "preserve me")

    def test_inputs_and_payload_do_not_alias(self):
        r = self.create()
        sample, values = dict(self.sample), [0.5]
        r.record_prediction("memory", sample, values)
        sample["origin"], values[0] = 5, 9
        payload = r.payload()
        payload["rows"][0]["prediction"][0] = 99
        self.assertEqual(r.payload()["rows"][0]["prediction"], [0.5])

    def test_abrupt_process_exit_preserves_committed_record_and_releases_lock(self):
        script = """import json,os,sys
from durable import DurableRecorder
r=DurableRecorder.create(sys.argv[1],sys.argv[2].encode(),expected_sha256=sys.argv[3])
r.record_prediction("memory",json.loads(sys.argv[2])["samples"][0],[0.5])
print(json.dumps(r.anchor),flush=True)
os._exit(7)
"""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(self.directory),
                self.raw.decode(),
                self.sha,
            ],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        recovered = self.resume(json.loads(result.stdout))
        self.assertEqual(recovered.payload()["rows"][0]["prediction"], [0.5])

    def test_verifier_detects_metric_and_raw_edits(self):
        r = self.create()
        self.predictions(r)
        r.record_target(self.sample, [1])
        bundle = r.finalize(Path(self.tmp.name) / "bundle.json")
        altered = copy.deepcopy(bundle)
        altered["result"]["numerical_gates_pass"] = False
        with self.assertRaises(ValueError):
            verify(altered, self.raw, expected_plan_sha256=self.sha)
        altered = copy.deepcopy(bundle)
        altered["evidence"]["rows"][0]["prediction"][0] = 99
        with self.assertRaises(ValueError):
            verify(altered, self.raw, expected_plan_sha256=self.sha)

    def test_export_verifier_runs_as_real_cli(self):
        r = self.create()
        self.predictions(r)
        r.record_target(self.sample, [1])
        bundle = Path(self.tmp.name) / "bundle.json"
        r.finalize(bundle)
        plan_path = Path(self.tmp.name) / "plan.json"
        plan_path.write_bytes(self.raw)
        receipt = Path(self.tmp.name) / "receipt.json"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "verify_export.py"),
                "--bundle",
                str(bundle),
                "--plan",
                str(plan_path),
                "--expected-plan-sha256",
                self.sha,
                "--receipt",
                str(receipt),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(receipt.read_text())["status"], "PASS_REANALYSIS")


class ObservationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / "observations"
        self.identity = {
            "source_sha256": "b" * 64,
            "split": "development",
            "model_state_sha256": "a" * 64,
        }
        self.ledger = DurableObservationLedger.create(self.directory, **self.identity)
        self.addCleanup(self.ledger.close)
        self.ledger.register_many([forecast()], now=0)

    def test_partial_and_completed_replay_match_saved_base_and_arrival(self):
        self.ledger.observe(event(indices=(1,), values=(8,), times=(3,), now=4), now=4)
        self.ledger.observe(
            event(identity="late", indices=(0,), values=(4,), times=(2,), now=7), now=7
        )
        anchor, checksum = self.ledger.anchor, self.ledger.checksum()
        self.ledger.close()
        result = audit_observations(
            self.directory, expected_anchor=anchor, expected_identity=self.identity
        )
        self.assertEqual(result["ledger_sha256"], checksum)
        self.assertEqual(result["completed_residuals"][0]["residual"], (3, 6))
        self.assertEqual(result["completed_residuals"][0]["available_at"], 7)

    def test_partial_history_is_not_claimed_complete(self):
        self.ledger.observe(event(indices=(0,), values=(4,), times=(2,), now=2), now=2)
        anchor = self.ledger.anchor
        self.ledger.close()
        result = audit_observations(
            self.directory, expected_anchor=anchor, expected_identity=self.identity
        )
        self.assertEqual(result["completed_residuals"], [])

    def test_wrong_source_split_or_model_identity_refused(self):
        anchor = self.ledger.anchor
        self.ledger.close()
        for key in self.identity:
            changed = dict(self.identity, **{key: "wrong"})
            with self.assertRaises(ValueError):
                audit_observations(
                    self.directory, expected_anchor=anchor, expected_identity=changed
                )

    def test_invalid_observation_atomic_on_disk_and_memory(self):
        before = self.ledger.anchor, self.ledger.checksum()
        with self.assertRaises(ValueError):
            self.ledger.observe(event(owner="wrong"), now=3)
        self.assertEqual((self.ledger.anchor, self.ledger.checksum()), before)

    def test_invalid_forecast_identity_atomic(self):
        before = self.ledger.anchor, self.ledger.checksum()
        with self.assertRaises(ValueError):
            self.ledger.register_many(
                [replace(forecast("g"), state_id="c" * 64)], now=0
            )
        self.assertEqual((self.ledger.anchor, self.ledger.checksum()), before)

    def test_reveal_disk_failure_cannot_advance_visible_state(self):
        before = self.ledger.anchor, self.ledger.checksum()
        with patch("journal.publish", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.ledger.observe(event(), now=3)
        self.assertEqual((self.ledger.anchor, self.ledger.checksum()), before)


if __name__ == "__main__":
    unittest.main()
