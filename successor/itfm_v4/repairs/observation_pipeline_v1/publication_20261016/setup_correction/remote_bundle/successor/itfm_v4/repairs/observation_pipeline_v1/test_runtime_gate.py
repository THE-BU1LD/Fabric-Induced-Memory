"""Test the real supervisor acceptance rules without tensor imports."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from runtime_verification import (
    InventoryResult,
    assess,
    test_ids as suite_ids,
    stable,
    verify_sources,
    bounded_process,
    admission,
    validate_worker_receipt,
    stop_owned_process,
)


class RuntimeGateTests(unittest.TestCase):
    def outcome(self, body):
        class Case(unittest.TestCase):
            test_example = body

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Case)
        expected = list(suite_ids(suite))
        result = unittest.TextTestRunner(
            stream=io.StringIO(), resultclass=InventoryResult
        ).run(suite)
        return result, expected

    def test_pass_requires_actual_nonempty_exact_inventory(self):
        result, expected = self.outcome(lambda self: self.assertTrue(True))
        self.assertTrue(assess(result, expected))
        self.assertFalse(assess(result, []))
        self.assertFalse(assess(result, ["wrong"]))
        self.assertFalse(assess(result, expected * 2))

    def test_empty_discovery_cannot_pass(self):
        result = InventoryResult(io.StringIO(), True, 1)
        self.assertTrue(result.wasSuccessful())
        self.assertFalse(assess(result, ["required"]))
        self.assertFalse(assess(result, []))

    def test_skipped_test_cannot_pass(self):
        result, expected = self.outcome(unittest.skip("unavailable")(lambda self: None))
        self.assertTrue(result.wasSuccessful())
        self.assertFalse(assess(result, expected))

    def test_expected_failure_cannot_pass(self):
        result, expected = self.outcome(
            unittest.expectedFailure(lambda self: self.fail("defect"))
        )
        self.assertTrue(result.wasSuccessful())
        self.assertFalse(assess(result, expected))

    def test_failure_and_error_cannot_pass(self):
        for body in (lambda self: self.fail("failure"), lambda self: 1 / 0):
            result, expected = self.outcome(body)
            self.assertFalse(assess(result, expected))

    def test_missing_executed_identity_cannot_pass(self):
        result, expected = self.outcome(lambda self: None)
        result.executed.clear()
        self.assertFalse(assess(result, expected))

    def test_resource_floor_and_pressure_fail_closed(self):
        sample = {"free_mib": 512, "pageouts": 2, "swap_mib": 10}
        self.assertTrue(stable(sample, sample))
        for changed in ({"free_mib": 511}, {"pageouts": 3}, {"swap_mib": 11}):
            self.assertFalse(stable(sample, dict(sample, **changed)))

    def test_preimport_recheck_can_reject_initial_success(self):
        good = {"free_mib": 1024, "pageouts": 2, "swap_mib": 10}
        with (
            patch("runtime_verification.platform.system", return_value="Darwin"),
            patch("runtime_verification.time.sleep"),
            patch(
                "runtime_verification.sample",
                side_effect=[good, good, dict(good, free_mib=100)],
            ),
        ):
            receipt = {}
            with self.assertRaisesRegex(ValueError, "Pre-import"):
                admission(receipt)
            self.assertEqual(receipt["status"], "BLOCKED_ADMISSION")

    def test_source_hash_and_path_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.py"
            source.write_text("pass\n")
            manifest = root / "manifest.json"
            value = {
                "schema_version": 1,
                "files": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()},
                "expected_tests": ["test.case"],
                "expected_scalar_tests": ["test.scalar"],
            }

            def write():
                manifest.write_text(json.dumps(value))
                return hashlib.sha256(manifest.read_bytes()).hexdigest()

            sha = write()
            self.assertEqual(verify_sources(manifest, sha), value)
            source.write_text("changed\n")
            with self.assertRaises(ValueError):
                verify_sources(manifest, sha)
            with self.assertRaises(ValueError):
                verify_sources(manifest, "a" * 64)
            value["files"] = {"../outside.py": "a" * 64}
            with self.assertRaises(ValueError):
                verify_sources(manifest, write())

    def test_subprocess_deadline_kills_only_owned_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "worker.log"
            code, timed_out = bounded_process(
                [sys.executable, "-B", "-S", "-c", "import time; time.sleep(10)"],
                log,
                seconds=0.1,
            )
            self.assertTrue(timed_out)
            self.assertLess(code, 0)
            self.assertTrue(log.exists())

    def test_worker_exit_status_and_log_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "worker.log"
            code, timed_out = bounded_process(
                [
                    sys.executable,
                    "-B",
                    "-S",
                    "-c",
                    "print('retained'); raise SystemExit(7)",
                ],
                log,
                seconds=3,
            )
            self.assertEqual(code, 7)
            self.assertFalse(timed_out)
            self.assertEqual(log.read_text().strip(), "retained")

    def test_missing_manifest_cli_fails_without_tensor_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "failure.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-S",
                    str(Path(__file__).parent / "runtime_verification.py"),
                    "preflight",
                    "--manifest-sha256",
                    "a" * 64,
                    "--receipt",
                    str(receipt),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertNotEqual(result.returncode, 0)
            saved = json.loads(receipt.read_text())
            self.assertEqual(saved["model_modules_imported"], [])
            self.assertFalse(saved["status"].startswith("PASS"))


class WorkerReceiptTests(unittest.TestCase):
    def valid(self):
        return dict(
            status="PASS_TENSOR_SOFTWARE_ONLY",
            manifest_sha256="a" * 64,
            tests_run=1,
            executed=["required"],
            skipped=0,
            failures=0,
            errors=0,
            expected_failures=0,
            unexpected_successes=0,
            model_modules_imported=["torch", "numpy"],
        )

    def check(self, detail):
        validate_worker_receipt(detail, ["required"], "a" * 64)

    def test_complete_receipt_accepted(self):
        self.check(self.valid())

    def test_missing_or_nonzero_outcomes_rejected(self):
        for name in (
            "skipped",
            "failures",
            "errors",
            "expected_failures",
            "unexpected_successes",
        ):
            for value in (None, True, False, 1, 0.0, "0"):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    self.check(dict(self.valid(), **{name: value}))
            detail = self.valid()
            del detail[name]
            with self.assertRaises(ValueError):
                self.check(detail)

    def test_invalid_count_inventory_identity_rejected(self):
        for name, value in (
            ("tests_run", True),
            ("tests_run", 1.0),
            ("tests_run", 0),
            ("executed", []),
            ("executed", ["required", "required"]),
            ("executed", [None]),
            ("manifest_sha256", "b" * 64),
            ("status", "PASS_PREFLIGHT_ONLY"),
            ("model_modules_imported", []),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.check(dict(self.valid(), **{name: value}))

    def test_missing_receipt_and_empty_inventory_rejected(self):
        for detail in (None, [], {}):
            with self.assertRaises(ValueError):
                self.check(detail)
        with self.assertRaises(ValueError):
            validate_worker_receipt(self.valid(), [], "a" * 64)

    def test_exit_race_still_reaps_worker(self):
        process = Mock(pid=123)
        with patch("runtime_verification.os.killpg", side_effect=ProcessLookupError):
            stop_owned_process(process)
        process.wait.assert_called_once_with()

    def test_interruption_cleans_up_owned_process(self):
        process = Mock(pid=123)
        process.wait.side_effect = [KeyboardInterrupt, -9]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("runtime_verification.subprocess.Popen", return_value=process),
            patch("runtime_verification.os.killpg") as kill,
        ):
            with self.assertRaises(KeyboardInterrupt):
                bounded_process(["worker"], Path(tmp) / "log")
            kill.assert_called_once()
            self.assertEqual(kill.call_args.args[0], 123)
            self.assertEqual(process.wait.call_count, 2)
