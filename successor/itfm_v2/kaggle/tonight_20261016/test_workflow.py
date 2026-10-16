"""Stdlib safety tests; no numerical imports, device work, or cloud requests."""

import copy
import importlib.util
import json
from pathlib import Path
import platform
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile


HELPERS = Path(__file__).resolve().parent
sys.path.insert(0, str(HELPERS))
sys.path.insert(0, str(HELPERS.parents[1] / "src"))
spec = importlib.util.spec_from_file_location("fim_tonight_workflow", HELPERS / "workflow.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
from fim_sota.engineering import digest

SOURCE = {"sha256": workflow.SOURCE_SHA256, "files": {"source.py": "a" * 64}}
VERSIONS = {"torch": "2.11.0", "numpy": "2.4.0", "pytest": "9.0.0"}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_receipt(root):
    sources = {}
    for name in ("test_training_recovery.py", "test_engineering_runtime.py"):
        path = root / "tests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# unit fixture only\n")
        sources[name] = workflow.sha(path)
    node = "test_engineering_runtime.py::test_requested_device_training_evaluation"
    return {"status": "PASS", "requested_device": "cpu", "inventory": [node],
            "passed": [node], "failed": [], "skipped": [], "test_sources": sources,
            "environment": {"status": "PASS_RUNTIME_ENVIRONMENT", "runtime_executed": True,
                            "device": "cpu", "source": SOURCE, "packages": {key: VERSIONS[key] for key in ("torch", "numpy")},
                            "python": platform.python_version()}}


class WorkflowTests(unittest.TestCase):
    def test_targeted_cuda_gate_requires_all_three_executed_tests_and_current_source(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_source_identity", return_value=SOURCE):
            root = Path(temp)
            cpu = test_receipt(root)
            environment = {**cpu["environment"], "device": "cuda"}
            write(root / "gpu_environment.json", environment)
            names = ("test_requested_device_training_evaluation",
                     "test_train_select_checkpoint_and_replay_requested_cuda", "test_cuda_fp32_tiny_training")
            xml = root / "cuda_selected_tests.xml"
            receipt_path = root / "cuda_selected_tests.json"

            def save_cases(selected, skipped=False):
                xml.write_text('<testsuites><testsuite>' + ''.join(
                    f'<testcase name="{name}">' + ('<skipped message="not executed"/>' if skipped and index == 0 else '') + '</testcase>'
                    for index, name in enumerate(selected)) + '</testsuite></testsuites>')
                value = {"status": "PASS", "source_sha256": workflow.SOURCE_SHA256,
                         "requested_device": "cuda", "precision": "fp32",
                         "test_sources": cpu["test_sources"], "junit_sha256": workflow.sha(xml)}
                write(receipt_path, value)
                return value

            expected = save_cases(names)
            self.assertEqual(workflow.validate_cuda_gate(root, root), expected)
            save_cases(names, skipped=True)
            with self.assertRaisesRegex(ValueError, "all execute"):
                workflow.validate_cuda_gate(root, root)
            save_cases(names[:2])
            with self.assertRaisesRegex(ValueError, "all execute"):
                workflow.validate_cuda_gate(root, root)
            changed = save_cases(names)
            changed["source_sha256"] = "b" * 64
            write(receipt_path, changed)
            with self.assertRaisesRegex(ValueError, "executed source"):
                workflow.validate_cuda_gate(root, root)
            save_cases(names)
            (root / "tests/test_training_recovery.py").write_text("changed source after run")
            with self.assertRaisesRegex(ValueError, "executed source"):
                workflow.validate_cuda_gate(root, root)

    def test_actual_receipt_schema_passes_and_fabricated_legacy_or_skipped_receipts_fail(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_source_identity", return_value=SOURCE), patch.object(workflow, "_versions", return_value=VERSIONS):
            root = Path(temp)
            receipt = test_receipt(root)
            target = root / "receipt.json"
            write(target, receipt)
            self.assertEqual(workflow.validate_test_receipt(root, target, "cpu"), receipt)
            for changed in (
                {"status": "PASS", "device": "cpu", "runtime_pass": True},
                {**receipt, "passed": [], "skipped": receipt["inventory"]},
                {**receipt, "failed": ["failed-test"]},
                {**receipt, "inventory": receipt["inventory"] + ["omitted-test"]},
            ):
                write(target, changed)
                with self.assertRaises((ValueError, RuntimeError)):
                    workflow.validate_test_receipt(root, target, "cpu")
            write(target, receipt)
            (root / "tests/test_training_recovery.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "source inventory"):
                workflow.validate_test_receipt(root, target, "cpu")

    def test_environment_reuse_preserves_identity_and_rejects_dependency_drift(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_source_identity", return_value=SOURCE), patch.object(workflow, "_versions", return_value=VERSIONS):
            root = Path(temp)
            first = workflow.environment(root, root / "work")
            digest_before = workflow.sha(root / "work/environment_cpu_metadata.json")
            self.assertEqual(first, workflow.environment(root, root / "work"))
            self.assertEqual(digest_before, workflow.sha(root / "work/environment_cpu_metadata.json"))
            with patch.object(workflow, "_versions", return_value={**VERSIONS, "torch": "2.12.0"}):
                with self.assertRaisesRegex(ValueError, "environment/source changed"):
                    workflow.environment(root, root / "work")

    def test_cpu_gate_refuses_detected_gpu_before_launch(self):
        probe = Mock(returncode=0, stdout="GPU-observed\n")
        with patch.object(workflow.subprocess, "run", return_value=probe), patch.object(workflow, "supervised") as launch:
            with self.assertRaisesRegex(RuntimeError, "accelerator None"):
                workflow.cpu_gate("/unused", "/unused")
            launch.assert_not_called()

    def test_gpu_exit_zero_is_not_success_without_validated_artifacts(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_source_identity", return_value=SOURCE), patch.object(workflow, "require_gates"), patch.object(workflow, "supervised", return_value={"elapsed_seconds": 2, "exit_code": 0}):
            budget = Mock()
            budget.reserve_run.return_value = {"state": "RUNNING"}
            budget.snapshot.return_value = {"authorized_hours": 6.0,
                "estimates": {"quota_spent_conservative_estimate": 0.0},
                "sessions": {"session": {"quota_multiplier": 1}}}
            with self.assertRaisesRegex(ValueError, "no validated"):
                workflow.gpu_job(temp, temp, budget, "session", "run-a", ["real-command"], 60, verify_artifacts=lambda: None)
            budget.finish_run.assert_called_once_with("session", "run-a", status="FAILED")
            budget.reset_mock()
            with self.assertRaisesRegex(ValueError, "frozen source"):
                workflow.gpu_job(temp, temp, budget, "session", "run-b", ["real-command"], 60,
                                 verify_artifacts=lambda: {"status": "PASS_ENGINEERING", "source_sha256": "b" * 64})
            budget.finish_run.assert_called_once_with("session", "run-b", status="FAILED")
            budget.reset_mock()
            result = workflow.gpu_job(temp, temp, budget, "session", "run-c", ["real-command"], 60,
                                      verify_artifacts=lambda: {"status": "PASS_ENGINEERING", "source_sha256": workflow.SOURCE_SHA256})
            self.assertEqual(result["exit_code"], 0)
            budget.finish_run.assert_called_once_with("session", "run-c", status="SUCCEEDED", artifacts_verified=True)
            self.assertTrue((Path(temp) / "run-c.timing.json").exists())

    def test_gpu_readiness_shared_twenty_percent_ceiling_counts_idle_and_devices(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_source_identity", return_value=SOURCE), patch.object(workflow, "require_gates"), patch.object(workflow, "supervised", return_value={"elapsed_seconds": 1, "exit_code": 0}) as launch:
            budget = Mock()
            budget.reserve_run.return_value = {"state": "RUNNING"}
            evidence = lambda: {"status": "PASS_ENGINEERING", "source_sha256": workflow.SOURCE_SHA256}
            for spent, multiplier in ((0.15, 1), (0.0, 2)):
                budget.snapshot.return_value = {"authorized_hours": 1.0,
                    "estimates": {"quota_spent_conservative_estimate": spent},
                    "sessions": {"session": {"quota_multiplier": multiplier}}}
                with self.assertRaisesRegex(workflow.AdmissionError, "20%"):
                    workflow.gpu_job(temp, temp, budget, "session", "blocked", ["not-launched"], 600, verify_artifacts=evidence)
            budget.reserve_run.assert_not_called()
            launch.assert_not_called()
            # Exactly the shared boundary is admissible: 0.1h already allocated
            # plus (325+35)/3600h estimated for this one-device child = 0.2h.
            budget.snapshot.return_value = {"authorized_hours": 1.0,
                "estimates": {"quota_spent_conservative_estimate": 0.1},
                "sessions": {"session": {"quota_multiplier": 1}}}
            workflow.gpu_job(temp, temp, budget, "session", "boundary", ["verified-command"], 325, verify_artifacts=evidence)
            budget.reserve_run.assert_called_once()
            launch.assert_called_once()

    def test_empty_command_and_invalid_timeout_do_not_reserve_or_launch(self):
        with patch.object(workflow, "_source_identity", return_value=SOURCE), patch.object(workflow, "require_gates"), patch.object(workflow.subprocess, "Popen") as launch:
            budget = Mock()
            with self.assertRaises(ValueError):
                workflow.gpu_job("/unused", "/unused", budget, "s", "r", [], 30, verify_artifacts=lambda: {})
            for timeout in (0, -1, float("nan"), True):
                with self.assertRaises(ValueError):
                    workflow.supervised(["not-launched"], "/unused", timeout)
            budget.reserve_run.assert_not_called()
            launch.assert_not_called()

    def test_profile_requires_exact_artifacts_environment_and_single_device(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_source_identity", return_value=SOURCE), patch.object(workflow, "_versions", return_value=VERSIONS):
            root = Path(temp)
            config = {"device": "cuda", "seed": 101}
            names = ("events.jsonl", "measured_timing.csv", "profile_checkpoint.pt", "source.json", "configuration.json", "environment.json")
            for name in names:
                (root / name).write_text("unit fixture only")
            receipt = {"status": "PASS_PROFILE", "source": SOURCE, "configuration": config,
                       "config_sha256": digest(config), "profile_source_sha256": workflow.sha(HELPERS / "device_profile.py"),
                       "placement_verified": True, "accelerators": {"selected_device": "cuda:0", "torch_visible_count": 1},
                       "environment_preflight": {"status": "PASS_RUNTIME_ENVIRONMENT", "runtime_executed": True,
                            "device": "cuda", "source": SOURCE, "packages": {key: VERSIONS[key] for key in ("torch", "numpy")}, "python": platform.python_version()},
                       "artifacts": {name: workflow.sha(root / name) for name in names}}
            write(root / "profile.json", receipt)
            self.assertEqual(workflow.validate_profile(root, root / "profile.json"), receipt)
            changed = copy.deepcopy(receipt)
            changed["accelerators"]["torch_visible_count"] = 2
            write(root / "profile.json", changed)
            with self.assertRaisesRegex(ValueError, "single visible"):
                workflow.validate_profile(root, root / "profile.json")
            write(root / "profile.json", receipt)
            (root / "measured_timing.csv").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                workflow.validate_profile(root, root / "profile.json")

    def test_budget_rejection_records_every_expected_cell_without_launch(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_source_identity", return_value=SOURCE), patch.object(workflow, "require_gates"), patch.object(workflow, "validate_profile"), patch.object(workflow, "supervised") as launch:
            work = Path(temp)
            write(work / "environment_cpu_metadata.json", {})
            write(work / "representative_profile/profile.json", {})
            manifest = {"source_sha256": workflow.SOURCE_SHA256,
                        "environment_sha256": workflow.sha(work / "environment_cpu_metadata.json"),
                        "profile_sha256": workflow.sha(work / "representative_profile/profile.json"),
                        "analysis_sha256": workflow.sha(HELPERS / "analysis.py"),
                        "cells": [{"id": name, "config": {"seed": seed, "wall_seconds": 100}} for name, seed in (("a", 101), ("b", 102))]}
            write(work / "frozen_manifest.json", manifest)
            (work / "frozen_manifest.sha256").write_text(digest(manifest))
            budget = Mock()
            budget.reserve_run.side_effect = workflow.AdmissionError("protect remaining reserve")
            with self.assertRaises(workflow.AdmissionError):
                workflow.run_next(work, work, budget, "session")
            ledger = json.loads((work / "run_ledger.json").read_text())
            self.assertEqual(set(ledger["cells"]), {"a", "b"})
            self.assertTrue(all(row["state"] == "SKIPPED_BUDGET" for row in ledger["cells"].values()))
            self.assertEqual(ledger["manifest_sha256"], digest(manifest))
            launch.assert_not_called()

    def test_archive_restore_checks_bytes_and_never_launches_or_claims_remote_persistence(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_source_identity", return_value=SOURCE):
            root = Path(temp)
            work = root / "work"
            write(work / "nested/evidence.json", {"status": "PENDING"})
            result = workflow.archive(work, root / "evidence.zip")
            restored = workflow.restore(root / "evidence.zip", root / "restored", result["sha256"], source_root=root)
            self.assertEqual((root / "restored/nested/evidence.json").read_bytes(), (work / "nested/evidence.json").read_bytes())
            self.assertFalse(restored["launch_or_resume_performed"])
            self.assertFalse(restored["remote_retrieval_verified"])
            with self.assertRaises(FileExistsError):
                workflow.restore(root / "evidence.zip", root / "restored", result["sha256"], source_root=root)
            with self.assertRaisesRegex(ValueError, "checksum"):
                workflow.restore(root / "evidence.zip", root / "wrong", "0" * 64, source_root=root)

    def test_restore_rejects_path_traversal_before_creating_destination(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_source_identity", return_value=SOURCE):
            root = Path(temp)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../outside.txt", "untrusted")
                output.writestr("ARCHIVE_MANIFEST.json", json.dumps({"../outside.txt": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "unsafe archive"):
                workflow.restore(archive, root / "restored", workflow.sha(archive), source_root=root)
            self.assertFalse((root / "restored").exists())
            self.assertFalse((root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
