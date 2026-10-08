"""Artificial filesystem fixtures only; no retained FIM outcomes are opened."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import evidence_publication as publisher


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate = load_module("candidate_analyzer", SCRIPTS / "analyze_trajectory_isolated_ablations.py")


class PublicationFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = [self.root / directory / name for directory, name in (
            ("results", "paired.csv"), ("paper", "table.md"), ("research", "evidence.md")
        )]
        self.old = [b"old csv\r\n", b"old table\n", b"old evidence\n"]
        self.outputs = [(path, f"new artificial payload {index}\n") for index, path in enumerate(self.paths)]

    def seed(self):
        for index, path in enumerate(self.paths):
            path.parent.mkdir(parents=True)
            path.write_bytes(self.old[index])
            path.chmod(0o640)

    def assert_old(self):
        self.assertEqual([path.read_bytes() for path in self.paths], self.old)

    def assert_no_temps(self):
        self.assertFalse(list(self.root.rglob("*.stage-*")))


class PublicationTests(PublicationFixture):
    def test_success_new_set(self):
        publisher.publish_evidence(self.outputs)
        self.assertEqual([p.read_text() for p in self.paths], [text for _, text in self.outputs])
        self.assert_no_temps()

    def test_success_replaces_set_and_preserves_permissions(self):
        self.seed()
        publisher.publish_evidence(self.outputs)
        self.assertEqual([p.read_text() for p in self.paths], [text for _, text in self.outputs])
        self.assertEqual([stat.S_IMODE(p.stat().st_mode) for p in self.paths], [0o640] * 3)
        self.assert_no_temps()

    def test_staging_failure_never_publishes(self):
        self.seed()
        real_fsync = os.fsync
        calls = 0
        def fail_later(fd):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise OSError("artificial fsync failure while staging table")
            return real_fsync(fd)
        with patch.object(publisher.os, "fsync", side_effect=fail_later):
            with self.assertRaisesRegex(OSError, "artificial fsync"):
                publisher.publish_evidence(self.outputs)
        self.assert_old()
        self.assert_no_temps()

    def test_failure_at_each_publish_position_restores_previous_set(self):
        self.seed()
        real_replace = os.replace
        for position in (1, 2, 3):
            with self.subTest(position=position):
                calls = 0
                def fail_once(source, destination):
                    nonlocal calls
                    calls += 1
                    if calls == position:
                        raise OSError("artificial publication failure")
                    return real_replace(source, destination)
                with patch.object(publisher.os, "replace", side_effect=fail_once):
                    with self.assertRaisesRegex(OSError, "artificial publication"):
                        publisher.publish_evidence(self.outputs)
                self.assert_old()
                self.assert_no_temps()

    def test_failure_removes_new_outputs_during_rollback(self):
        real_replace = os.replace
        calls = 0
        def fail_second(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("artificial second publication failure")
            return real_replace(source, destination)
        with patch.object(publisher.os, "replace", side_effect=fail_second):
            with self.assertRaises(OSError):
                publisher.publish_evidence(self.outputs)
        self.assertFalse(any(path.exists() for path in self.paths))
        self.assert_no_temps()

    def test_incomplete_rollback_preserves_original_bytes_and_recovery_instruction(self):
        self.seed()
        real_replace = os.replace
        calls = 0
        def fail_commit_and_restore(source, destination):
            nonlocal calls
            calls += 1
            if calls in (2, 3):
                raise OSError("artificial commit/rollback failure")
            return real_replace(source, destination)
        with patch.object(publisher.os, "replace", side_effect=fail_commit_and_restore):
            with self.assertRaises(publisher.PublicationRollbackError) as caught:
                publisher.publish_evidence(self.outputs)
        recovery = caught.exception.recovery
        self.assertEqual(len(recovery), 1)
        self.assertEqual(recovery[0]["destination"], str(self.paths[0]))
        self.assertEqual(Path(recovery[0]["previous_bytes"]).read_bytes(), self.old[0])
        self.assertEqual(recovery[0]["action"], "restore previous_bytes")
        self.assertEqual([p.read_bytes() for p in self.paths[1:]], self.old[1:])
        real_replace(recovery[0]["previous_bytes"], recovery[0]["destination"])
        self.assert_old()
        self.assert_no_temps()

    def test_directory_destination_rejected_before_any_write(self):
        self.seed()
        self.paths[1].unlink()
        self.paths[1].mkdir()
        with self.assertRaisesRegex(ValueError, "regular file"):
            publisher.publish_evidence(self.outputs)
        self.assertEqual(self.paths[0].read_bytes(), self.old[0])
        self.assertEqual(self.paths[2].read_bytes(), self.old[2])
        self.assert_no_temps()

    def test_symlink_destination_rejected_without_touching_target(self):
        self.seed()
        target = self.root / "unrelated.txt"
        target.write_bytes(b"untouched unrelated bytes")
        self.paths[1].unlink()
        self.paths[1].symlink_to(target)
        with self.assertRaisesRegex(ValueError, "regular file"):
            publisher.publish_evidence(self.outputs)
        self.assertEqual(target.read_bytes(), b"untouched unrelated bytes")
        self.assertEqual(self.paths[0].read_bytes(), self.old[0])
        self.assert_no_temps()

    def test_duplicate_destinations_rejected_before_any_write(self):
        self.seed()
        duplicate = self.paths[0].parent / "." / self.paths[0].name
        with self.assertRaisesRegex(ValueError, "distinct"):
            publisher.publish_evidence(self.outputs + [(duplicate, "clobber")])
        self.assert_old()
        self.assert_no_temps()

    def test_protected_manifest_cannot_be_an_output(self):
        self.seed()
        with self.assertRaisesRegex(ValueError, "replace its input"):
            publisher.publish_evidence(self.outputs, protected_inputs=[self.paths[2]])
        self.assert_old()
        self.assert_no_temps()

    def test_invalid_text_is_rejected_before_any_write(self):
        self.seed()
        with self.assertRaises(UnicodeEncodeError):
            publisher.publish_evidence(self.outputs[:2] + [(self.paths[2], "\ud800")])
        self.assert_old()
        self.assert_no_temps()

    def test_empty_output_set_rejected(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            publisher.publish_evidence([])


class AnalyzerIntegrationTests(PublicationFixture):
    def run_artificial_writer(self, module, manifest):
        # The scientific admission and comparison functions are deliberately
        # not exercised with outcomes. This isolates actual CLI output wiring.
        argv = ["analyzer", "--manifest", str(manifest), "--csv", str(self.paths[0]),
                "--table", str(self.paths[1]), "--evidence", str(self.paths[2])]
        with patch.object(sys, "argv", argv), \
             patch.object(module, "validate_manifest", return_value=[{"git_commit": "ARTIFICIAL_TEST_ONLY"}]), \
             patch.object(module, "build_comparisons", return_value=[{"fixture": "ARTIFICIAL_TEST_ONLY"}]), \
             patch.object(module, "render_table", return_value="ARTIFICIAL TEST TABLE\n"), \
             redirect_stdout(io.StringIO()):
            module.main()

    def test_candidate_rejects_same_fault_without_partial_publication(self):
        self.seed()
        self.paths[1].unlink()
        self.paths[1].mkdir()
        manifest = self.root / "custom-source.json"
        manifest.write_text(json.dumps({"protocol_sha256": "ARTIFICIAL_TEST_ONLY"}))
        with self.assertRaises(ValueError):
            self.run_artificial_writer(candidate, manifest)
        self.assertEqual(self.paths[0].read_bytes(), self.old[0])
        self.assertEqual(self.paths[2].read_bytes(), self.old[2])

    def test_custom_manifest_path_and_exact_bytes_are_recorded(self):
        self.seed()
        manifest = self.root / "source with spaces.json"
        manifest.write_text('{"protocol_sha256": "ARTIFICIAL_TEST_ONLY"}\n')
        self.run_artificial_writer(candidate, manifest)
        for path in self.paths[1:]:
            text = path.read_text()
            self.assertIn(str(manifest.resolve()), text)
            self.assertIn(hashlib.sha256(manifest.read_bytes()).hexdigest(), text)
            self.assertNotIn("results/trajectory_isolated_v1/manifest.json", text)
        self.assertEqual(self.paths[0].read_bytes(), b"fixture\r\nARTIFICIAL_TEST_ONLY\r\n")

    def test_actual_cli_rejects_invalid_manifest_before_any_output(self):
        self.seed()
        manifest = self.root / "inadmissible.json"
        manifest.write_text("{}")
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "analyze_trajectory_isolated_ablations.py"),
             "--manifest", str(manifest), "--csv", str(self.paths[0]),
             "--table", str(self.paths[1]), "--evidence", str(self.paths[2])],
            capture_output=True, text=True, timeout=10, cwd=self.root,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unexpected protocol_name", result.stderr)
        self.assert_old()
        self.assert_no_temps()


class ImportAndWorkflowTests(unittest.TestCase):
    def test_repository_module_import_does_not_require_script_directory_on_path(self):
        result = subprocess.run(
            [sys.executable, "-c", "from scripts.analyze_trajectory_isolated_ablations import render_csv; print(render_csv([{'fixture': 'ARTIFICIAL_TEST_ONLY'}]))"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ARTIFICIAL_TEST_ONLY", result.stdout)

    def test_scientific_workflow_requires_explicit_dispatch(self):
        workflow = (ROOT / ".github/workflows/current-component-ablations.yml").read_text()
        scientific = workflow.split("\n  trajectory-isolated-v1:\n", 1)[1]
        scientific_header = scientific.split("\n    steps:\n", 1)[0]
        self.assertIn("    if: github.event_name == 'workflow_dispatch'\n", scientific_header)
        tests_job = workflow.split("\n  repository-tests:\n", 1)[1].split("\n  trajectory-isolated-v1:\n", 1)[0]
        tests_header = tests_job.split("\n    steps:\n", 1)[0]
        self.assertNotIn("    if:", tests_header)


if __name__ == "__main__":
    unittest.main(verbosity=2)
