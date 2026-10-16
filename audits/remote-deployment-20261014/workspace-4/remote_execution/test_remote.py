import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import collect
import once
import prepare
import run_remote
import retain_release


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def manifest(self):
        (self.root / "input.txt").write_text("frozen")
        (self.root / "REMOTE_MANIFEST.json").write_text(
            json.dumps(
                {"files": {"input.txt": prepare.digest(self.root / "input.txt")}}
            )
        )
        return prepare.digest(self.root / "REMOTE_MANIFEST.json")

    def test_manifest_accepts_exact_bytes(self):
        run_remote.verify_manifest(self.root, self.manifest())

    def test_manifest_rejects_tampering(self):
        expected = self.manifest()
        (self.root / "input.txt").write_text("changed")
        with self.assertRaises(RuntimeError):
            run_remote.verify_manifest(self.root, expected)

    def test_manifest_rejects_wrong_identity(self):
        self.manifest()
        with self.assertRaises(RuntimeError):
            run_remote.verify_manifest(self.root, "0" * 64)

    def test_path_traversal_rejected(self):
        for name in ("../escape", "/tmp/escape"):
            with self.assertRaises(ValueError):
                prepare.safe(self.root, name)

    def test_existing_receipt_not_overwritten(self):
        p = self.root / "receipt.json"
        run_remote.write(p, {"a": 1})
        with self.assertRaises(FileExistsError):
            run_remote.write(p, {"a": 2})

    def test_failure_receipt_is_not_accepted(self):
        p = self.root / "receipt.json"
        p.write_text(json.dumps({"status": "FAILED_RETAINED"}))
        with self.assertRaises(RuntimeError):
            run_remote.expect(p, "PASS_RUNTIME_TESTS")

    def test_non_mac_platform_rejected_before_model_imports(self):
        with (
            patch.object(run_remote, "verify_manifest"),
            patch.object(run_remote.platform, "system", return_value="Linux"),
        ):
            with self.assertRaisesRegex(RuntimeError, "macOS ARM64"):
                run_remote.preflight("x", self.root)

    def test_symlink_escape_rejected(self):
        (self.root / "link").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            prepare.safe(self.root, "link/escape")

    def test_first_dispatch_only(self):
        once.check("12", [{"id": 12}], "1")
        with self.assertRaises(RuntimeError):
            once.check("12", [{"id": 11}, {"id": 12}], "1")
        with self.assertRaises(RuntimeError):
            once.check("12", [{"id": 12}], "2")
        once.check("12", [{"id": 11}, {"id": 12}], "1", adjudicated=[11])
        with self.assertRaises(RuntimeError):
            once.check(
                "12", [{"id": 10}, {"id": 11}, {"id": 12}], "1", adjudicated=[11]
            )

    def test_blocked_preflight_launches_nothing(self):
        with (
            patch.object(run_remote, "ROOT", self.root),
            patch.object(
                run_remote, "preflight", side_effect=RuntimeError("BLOCKED_RESOURCE")
            ),
            patch.object(run_remote, "verify_manifest"),
            patch.object(run_remote, "stage") as stage,
        ):
            self.assertEqual(run_remote.execute("x", "run"), 3)
            stage.assert_not_called()

    def test_v2_failure_prevents_v3(self):
        with (
            patch.object(run_remote, "ROOT", self.root),
            patch.object(run_remote, "preflight"),
            patch.object(run_remote, "verify_manifest"),
            patch.object(
                run_remote.importlib.metadata,
                "version",
                side_effect=lambda n: {
                    "torch": "2.12.1",
                    "numpy": "2.4.6",
                    "pytest": "9.1.1",
                    "setuptools": "81.0.0",
                }[n],
            ),
            patch.object(
                run_remote, "stage", side_effect=RuntimeError("v2 failed")
            ) as stage,
        ):
            self.assertEqual(run_remote.execute("x", "run"), 3)
            self.assertEqual(stage.call_count, 1)
            self.assertEqual(stage.call_args.args[0], "v2")

    def test_collection_roundtrip_and_corruption(self):
        expected = self.manifest()
        folder = self.root / "remote-run"
        folder.mkdir()
        (folder / "failure.log").write_text("retained failure")
        output = self.root / "collected"
        collect.collect(self.root, output)
        collect.verify(output, expected)
        (output / "remote-run/failure.log").write_text("tampered")
        with self.assertRaises(RuntimeError):
            collect.verify(output, expected)

    def test_release_parts_roundtrip_and_hashes(self):
        import hashlib

        payload = b"0123456789" * 13
        archive = self.root / "archive"
        archive.write_bytes(payload)
        parts = retain_release.split_archive(archive, self.root, chunk_bytes=17)
        self.assertEqual(
            b"".join((self.root / p["file"]).read_bytes() for p in parts), payload
        )
        for part in parts:
            self.assertLessEqual(part["bytes"], 17)
            self.assertEqual(
                hashlib.sha256((self.root / part["file"]).read_bytes()).hexdigest(),
                part["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
