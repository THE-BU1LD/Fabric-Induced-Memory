"""Cancellation witnesses use artificial files only, never scientific outcomes."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import evidence_publication as publisher


CANCELLATIONS = (KeyboardInterrupt, SystemExit, GeneratorExit)


def bundle(root: Path, existing: bool = True):
    paths = [root / name for name in ("table.csv", "summary.md", "evidence.md")]
    old = [b"old artificial table\r\n", b"old artificial summary\n", b"old artificial evidence\n"]
    if existing:
        for path, payload in zip(paths, old):
            path.write_bytes(payload)
            path.chmod(0o640)
    outputs = [(path, f"new artificial payload {index}\n") for index, path in enumerate(paths)]
    return paths, old, outputs


class PublicationCancellationTests(unittest.TestCase):
    def assert_restored(self, root, paths, old, existing):
        if existing:
            self.assertEqual([p.read_bytes() for p in paths], old)
            self.assertEqual([stat.S_IMODE(p.stat().st_mode) for p in paths], [0o640] * 3)
        else:
            self.assertFalse(any(p.exists() for p in paths))
        self.assertEqual(list(root.glob("*.stage-*")), [])

    def check_cancelled_replace(self, *, existing, after_replace):
        real_replace = os.replace
        for exception_type in CANCELLATIONS:
            for position in (1, 2, 3):
                with self.subTest(exception=exception_type.__name__, position=position):
                    with tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        paths, old, outputs = bundle(root, existing)
                        error = exception_type("artificial cancellation")
                        calls = 0

                        def cancel(source, destination):
                            nonlocal calls
                            calls += 1
                            if calls == position:
                                if after_replace:
                                    real_replace(source, destination)
                                raise error
                            return real_replace(source, destination)

                        with patch.object(publisher.os, "replace", side_effect=cancel):
                            with self.assertRaises(exception_type) as caught:
                                publisher.publish_evidence(outputs)
                        self.assertIs(caught.exception, error)
                        self.assert_restored(root, paths, old, existing)

    def test_cancellation_before_replace_restores_existing_outputs(self):
        self.check_cancelled_replace(existing=True, after_replace=False)

    def test_cancellation_before_replace_removes_new_outputs(self):
        self.check_cancelled_replace(existing=False, after_replace=False)

    def test_cancellation_after_atomic_replace_restores_existing_outputs(self):
        self.check_cancelled_replace(existing=True, after_replace=True)

    def test_cancellation_after_atomic_replace_removes_new_outputs(self):
        self.check_cancelled_replace(existing=False, after_replace=True)

    def test_interrupted_restore_preserves_backup_and_continues_other_recovery(self):
        real_replace = os.replace
        for exception_type in CANCELLATIONS:
            with self.subTest(exception=exception_type.__name__):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    paths, old, outputs = bundle(root)
                    original = KeyboardInterrupt("artificial publication cancellation")
                    recovery_error = exception_type("artificial recovery cancellation")
                    calls = 0

                    def cancel_publish_and_recovery(source, destination):
                        nonlocal calls
                        calls += 1
                        if calls == 3:
                            raise original
                        if calls == 4:
                            raise recovery_error
                        return real_replace(source, destination)

                    with patch.object(publisher.os, "replace", side_effect=cancel_publish_and_recovery):
                        with self.assertRaises(BaseException) as caught:
                            publisher.publish_evidence(outputs)
                    failure = caught.exception
                    self.assertIsInstance(failure, publisher.PublicationRollbackError)
                    self.assertIs(failure.original_error, original)
                    self.assertIs(failure.__cause__, original)
                    self.assertEqual(len(failure.recovery), 1)
                    recovery = failure.recovery[0]
                    self.assertEqual(recovery["destination"], str(paths[1]))
                    self.assertEqual(recovery["action"], "restore previous_bytes")
                    self.assertIn(exception_type.__name__, recovery["error"])
                    backup = Path(recovery["previous_bytes"])
                    self.assertEqual(backup.read_bytes(), old[1])
                    self.assertEqual(stat.S_IMODE(backup.stat().st_mode), 0o640)
                    self.assertEqual(paths[0].read_bytes(), old[0])
                    self.assertEqual(paths[2].read_bytes(), old[2])
                    self.assertEqual(list(root.glob("*.stage-*")), [backup])
                    real_replace(backup, paths[1])
                    self.assert_restored(root, paths, old, True)

    def test_interrupted_remove_records_recovery_for_new_output(self):
        real_replace = os.replace
        real_unlink = Path.unlink
        for exception_type in CANCELLATIONS:
            with self.subTest(exception=exception_type.__name__):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    paths, old, outputs = bundle(root, existing=False)
                    original = OSError("artificial publication failure")
                    calls = 0

                    def fail_third(source, destination):
                        nonlocal calls
                        calls += 1
                        if calls == 3:
                            raise original
                        return real_replace(source, destination)

                    def cancel_unlink(path, *args, **kwargs):
                        if path == paths[1]:
                            raise exception_type("artificial recovery cancellation")
                        return real_unlink(path, *args, **kwargs)

                    with patch.object(publisher.os, "replace", side_effect=fail_third), \
                         patch.object(Path, "unlink", new=cancel_unlink):
                        with self.assertRaises(BaseException) as caught:
                            publisher.publish_evidence(outputs)
                    failure = caught.exception
                    self.assertIsInstance(failure, publisher.PublicationRollbackError)
                    self.assertIs(failure.original_error, original)
                    self.assertEqual(len(failure.recovery), 1)
                    self.assertEqual(failure.recovery[0]["destination"], str(paths[1]))
                    self.assertIsNone(failure.recovery[0]["previous_bytes"])
                    self.assertEqual(failure.recovery[0]["action"], "remove new output")
                    self.assertFalse(paths[0].exists())
                    self.assertFalse(paths[2].exists())
                    self.assertEqual(paths[1].read_text(), outputs[1][1])
                    real_unlink(paths[1])
                    self.assert_restored(root, paths, old, False)

    def test_cancelled_staging_never_changes_outputs(self):
        real_fsync = os.fsync
        for exception_type in CANCELLATIONS:
            with self.subTest(exception=exception_type.__name__):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    paths, old, outputs = bundle(root)
                    original = exception_type("artificial staging cancellation")
                    calls = 0

                    def cancel_staging(fd):
                        nonlocal calls
                        calls += 1
                        if calls == 4:
                            raise original
                        return real_fsync(fd)

                    with patch.object(publisher.os, "fsync", side_effect=cancel_staging):
                        with self.assertRaises(exception_type) as caught:
                            publisher.publish_evidence(outputs)
                    self.assertIs(caught.exception, original)
                    self.assert_restored(root, paths, old, True)

    def test_cancellation_after_atomic_restore_does_not_report_consumed_backup(self):
        real_replace = os.replace
        for exception_type in CANCELLATIONS:
            with self.subTest(exception=exception_type.__name__):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    paths, old, outputs = bundle(root)
                    original = OSError("artificial publication failure")
                    calls = 0

                    def cancel_after_restore(source, destination):
                        nonlocal calls
                        calls += 1
                        if calls == 3:
                            raise original
                        result = real_replace(source, destination)
                        if calls == 4:
                            raise exception_type("artificial cancellation after restore")
                        return result

                    with patch.object(publisher.os, "replace", side_effect=cancel_after_restore):
                        with self.assertRaises(BaseException) as caught:
                            publisher.publish_evidence(outputs)
                    self.assertIs(caught.exception, original)
                    self.assert_restored(root, paths, old, True)

    def test_cancellation_after_removal_does_not_report_absent_new_output(self):
        real_replace = os.replace
        real_unlink = Path.unlink
        for exception_type in CANCELLATIONS:
            with self.subTest(exception=exception_type.__name__):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    paths, old, outputs = bundle(root, existing=False)
                    original = KeyboardInterrupt("artificial publication cancellation")
                    calls = 0

                    def fail_third(source, destination):
                        nonlocal calls
                        calls += 1
                        if calls == 3:
                            raise original
                        return real_replace(source, destination)

                    def cancel_after_unlink(path, *args, **kwargs):
                        result = real_unlink(path, *args, **kwargs)
                        if path == paths[1]:
                            raise exception_type("artificial cancellation after removal")
                        return result

                    with patch.object(publisher.os, "replace", side_effect=fail_third), \
                         patch.object(Path, "unlink", new=cancel_after_unlink):
                        with self.assertRaises(BaseException) as caught:
                            publisher.publish_evidence(outputs)
                    self.assertIs(caught.exception, original)
                    self.assert_restored(root, paths, old, False)

    def test_failed_recovery_status_probe_preserves_available_backup(self):
        real_replace = os.replace
        real_exists = Path.exists
        for exception_type in CANCELLATIONS:
            with self.subTest(exception=exception_type.__name__):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    paths, old, outputs = bundle(root)
                    original = OSError("artificial publication failure")
                    calls = 0
                    probe_failed = False

                    def fail_third(source, destination):
                        nonlocal calls
                        calls += 1
                        if calls == 3:
                            raise original
                        return real_replace(source, destination)

                    def cancel_probe(path):
                        nonlocal probe_failed
                        if path.name.startswith(".summary.md.stage-") and not probe_failed:
                            probe_failed = True
                            raise exception_type("artificial recovery-status cancellation")
                        return real_exists(path)

                    with patch.object(publisher.os, "replace", side_effect=fail_third), \
                         patch.object(Path, "exists", new=cancel_probe):
                        with self.assertRaises(BaseException) as caught:
                            publisher.publish_evidence(outputs)
                    failure = caught.exception
                    self.assertIsInstance(failure, publisher.PublicationRollbackError)
                    self.assertIs(failure.original_error, original)
                    self.assertEqual(len(failure.recovery), 1)
                    recovery = failure.recovery[0]
                    self.assertEqual(recovery["destination"], str(paths[1]))
                    backup = Path(recovery["previous_bytes"])
                    self.assertEqual(backup.read_bytes(), old[1])
                    self.assertEqual(paths[0].read_bytes(), old[0])
                    self.assertEqual(paths[2].read_bytes(), old[2])
                    real_replace(backup, paths[1])
                    self.assert_restored(root, paths, old, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
