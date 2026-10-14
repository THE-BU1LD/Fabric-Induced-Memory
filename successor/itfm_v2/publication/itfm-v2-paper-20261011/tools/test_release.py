"""Lightweight tests for preservation and fail-closed release behavior."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_pdf
import release
import reproduce


class ReleaseTests(unittest.TestCase):
    def test_all_output_wrappers_protect_prior_manuscript(self):
        out = release.PUB / "before/SHOULD_NOT_EXIST"
        for call in (
            release.stage,
            build_pdf.build,
            lambda p: reproduce.execute(p, sys.executable),
        ):
            with self.assertRaises(ValueError):
                call(out)
        self.assertFalse(out.exists())

    def test_pdf_build_refuses_existing_output(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(FileExistsError),
        ):
            build_pdf.build(directory)

    def test_pdf_resource_stop_creates_no_source_or_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "build"
            with (
                patch.object(
                    release, "verify", return_value={"failed": [], "manuscript": {}}
                ),
                patch.object(
                    release, "resource_admission", return_value={"status": "BLOCKED"}
                ),
                patch.object(release, "bounded_command") as command,
            ):
                receipt = build_pdf.build(out)
            self.assertEqual(receipt["status"], "BLOCKED_RESOURCE")
            command.assert_not_called()
            self.assertFalse((out / "source").exists())
            self.assertFalse((out / "main.pdf").exists())

    def test_command_budget_terminates_child(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = release.bounded_command(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                cwd=directory,
                timeout=0.1,
            )
            self.assertTrue(receipt["timed_out"])
            self.assertNotEqual(receipt["exit_code"], 0)

    def test_path_escape_rejected(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(ValueError),
        ):
            release.safe_path(Path(directory), "../secret")

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            (root / "outside").symlink_to(Path(directory))
            with self.assertRaises(ValueError):
                release.safe_path(root, "outside/file")

    def test_existing_receipt_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            release.write_new(path, {"original": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                release.write_new(path, {"original": False})
            self.assertEqual(path.read_bytes(), before)

    def test_frozen_result_write_rejected(self):
        with self.assertRaises(ValueError):
            release.write_new(release.ROOT / "results/SHOULD_NOT_EXIST.json", {})
        self.assertFalse((release.ROOT / "results/SHOULD_NOT_EXIST.json").exists())

    def test_stage_refuses_unmet_gates_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "new"
            with (
                patch.object(
                    release,
                    "release_gate",
                    return_value={
                        "status": "BLOCKED",
                        "missing_gates": ["independent_reproduction"],
                    },
                ),
                self.assertRaisesRegex(RuntimeError, "independent_reproduction"),
            ):
                release.stage(out)
            self.assertFalse(out.exists())

    def test_replay_refuses_existing_output_before_verification(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(FileExistsError),
        ):
            reproduce.execute(directory, "/does/not/exist")

    def test_replay_resource_stop_imports_no_model(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "replay"
            with (
                patch.object(
                    release, "verify", return_value={"failed": [], "manuscript": {}}
                ),
                patch.object(
                    release, "resource_admission", return_value={"status": "BLOCKED"}
                ),
            ):
                receipt = reproduce.execute(out, "/does/not/exist")
            self.assertEqual(receipt["status"], "BLOCKED_RESOURCE")
            self.assertFalse((out / "workspace").exists())
            self.assertTrue((out / "REPLAY_RECEIPT.json").is_file())

    def test_status_label_alone_cannot_clear_release_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "RELEASE_GATES.json").write_text(
                json.dumps({"author_metadata": {"status": "PASS"}})
            )
            with patch.object(
                release, "verify", return_value={"failed": [], "manuscript": {}}
            ):
                result = release.release_gate(root, root)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertIn("author_metadata:missing_receipt", result["missing_gates"])

    def test_generic_attestation_cannot_clear_any_gate(self):
        for name in (
            "author_metadata",
            "clean_environment_replay",
            "raw_output_verification",
            "independent_reproduction",
            "scientific_review",
            "pdf_build",
            "visual_inspection",
            "public_archive",
            "venue_requirements",
        ):
            self.assertFalse(
                release.substantive_receipt(
                    name,
                    {"status": "PASS", "checked_at": "2026-10-11", "manuscript": {}},
                )
            )

    def test_visual_review_must_cover_every_page(self):
        self.assertFalse(
            release.substantive_receipt(
                "visual_inspection",
                {
                    "checked_at": "2026-10-11",
                    "page_count": 3,
                    "pages_reviewed": [1, 3],
                    "unresolved_defects": 0,
                    "pdf_sha256": "digest",
                    "reviewer": "reviewer",
                },
            )
        )

    def test_stale_manuscript_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "approval.json"
            receipt.write_text(
                json.dumps({"status": "PASS", "manuscript": {"main.tex": "old"}})
            )
            (root / "RELEASE_GATES.json").write_text(
                json.dumps(
                    {
                        "author_metadata": {
                            "status": "PASS",
                            "receipt": receipt.name,
                            "sha256": release.sha(receipt),
                        }
                    }
                )
            )
            with patch.object(
                release,
                "verify",
                return_value={"failed": [], "manuscript": {"main.tex": "new"}},
            ):
                result = release.release_gate(root, root)
            self.assertIn("author_metadata:stale_or_failed", result["missing_gates"])


if __name__ == "__main__":
    unittest.main()
