"""Standard-library control-plane tests; deliberately never import torch."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fim_sota import engineering as e


class ControlTests(unittest.TestCase):
    def test_atomic_json_refuses_existing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            e.atomic_write_json(path, {"original": True}, overwrite=False)
            with self.assertRaises(FileExistsError):
                e.atomic_write_json(path, {"original": False}, overwrite=False)
            self.assertEqual(json.loads(path.read_text()), {"original": True})

    def test_import_does_not_load_tensors(self):
        import subprocess
        import sys

        code = "import sys; from fim_sota import engineering; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_zero_not_replaced_with_default(self):
        self.assertEqual(e.resolve({"seed": 0})["seed"], 0)
        with self.assertRaises(ValueError):
            e.resolve({"epochs": 0})

    def test_reject_bad_configs(self):
        for config in (
            {"wat": 1},
            {"epochs": True},
            {"learning_rate": float("nan")},
            {"precision": "fp8"},
            {"device": "auto"},
            {"purpose": "confirmatory"},
            {"sequence_steps": 2},
            {"memory_topk": 100},
            {"wall_seconds": 3601},
            {"train_trajectories": 1000000},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                e.resolve(config)

    def test_stable_resource_floor(self):
        sample = {"available_bytes": 512 * 1024**2, "paging": 2, "swapouts": 8}
        self.assertEqual(e.admission(lambda: sample, lambda _: None)["status"], "PASS")
        self.assertEqual(
            e.admission(
                lambda: {**sample, "available_bytes": sample["available_bytes"] - 1},
                lambda _: None,
            )["status"],
            "BLOCK",
        )

    def test_resource_change_and_unknown_fail_closed(self):
        seq = iter(
            [
                {"available_bytes": 10**9, "paging": 1, "swapouts": 2},
                {"available_bytes": 10**9, "paging": 2, "swapouts": 2},
            ]
        )
        self.assertEqual(
            e.admission(lambda: next(seq), lambda _: None)["status"], "BLOCK"
        )
        self.assertEqual(e.admission(lambda: {}, lambda _: None)["status"], "BLOCK")

    def test_budget_simulated_clock(self):
        now = [10.0]
        b = e.Budget(50, lambda: now[0])
        self.assertTrue(b.admits(50))
        now[0] = 30
        self.assertFalse(b.admits(31))
        now[0] = 70
        self.assertEqual(b.remaining(), 0)

    def manifest(self, directory, **overrides):
        raw = {
            "schema_version": 1,
            "purpose": "smoke",
            "wall_seconds": 200,
            "cells": [{"id": "tiny", "config": {}}],
            **overrides,
        }
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(raw))
        return path

    def test_manifest_resolves_every_cell(self):
        with tempfile.TemporaryDirectory() as d:
            m = e.load_manifest(self.manifest(d))
            self.assertEqual(m["cells"][0]["config"], e.DEFAULTS)

    def test_manifest_rejects_traversal_duplicates_unbounded(self):
        with tempfile.TemporaryDirectory() as d:
            for cells in (
                [],
                [{"id": "../oops", "config": {}}],
                [{"id": "x", "config": {}}] * 2,
            ):
                with self.subTest(cells=cells), self.assertRaises(ValueError):
                    e.load_manifest(self.manifest(d, cells=cells))
            with self.assertRaises(ValueError):
                e.load_manifest(self.manifest(d, wall_seconds=100))

    def test_data_hash_independent_of_model_but_binds_generation(self):
        a = e.resolve({})
        self.assertEqual(
            e.data_identity(a), e.data_identity({**a, "variant": "erased"})
        )
        for key in ("seed", "spatial_points", "context_steps", "train_trajectories"):
            self.assertNotEqual(
                e.data_identity(a), e.data_identity({**a, key: a[key] + 1})
            )

    def make_completed(self, root):
        config = e.resolve({})
        for name in [
            "config.json",
            "source.json",
            "events.jsonl",
            "environment_preflight.json",
            "environment.json",
            "training_result.json",
            "checkpoints/latest.pt",
            "checkpoints/best.pt",
            "evaluation/metrics.json",
            "evaluation/predictions.pt",
            "evaluation/events.jsonl",
            "evaluation/raw_data.pt",
        ]:
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
        (root / "config.json").write_text(json.dumps(config))
        (root / "source.json").write_text(
            json.dumps({"sha256": e.digest({}), "files": {}})
        )
        receipt = dict(
            status="PASS_ENGINEERING",
            source_sha256=e.digest({}),
            config_sha256=e.digest(config),
            data_sha256=e.data_identity(config),
            artifacts=e.artifact_records(root),
        )
        (root / "completion.json").write_text(json.dumps(receipt))
        return config

    def test_reuse_checks_identity_and_every_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = self.make_completed(root)
            e.validate_completed(root, config, e.digest({}))
            with self.assertRaises(ValueError):
                e.validate_completed(root, config, "new-source")
            (root / "checkpoints/best.pt").write_text("changed")
            with self.assertRaises(ValueError):
                e.validate_completed(root, config, e.digest({}))

    def test_missing_artifact_is_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = self.make_completed(root)
            (root / "evaluation/predictions.pt").unlink()
            with self.assertRaises(ValueError):
                e.validate_completed(root, config, e.digest({}))

    def test_existing_run_refused_without_launch(self):
        with tempfile.TemporaryDirectory() as d, patch.object(e, "supervise") as runner:
            with self.assertRaises(FileExistsError):
                e.execute_run(e.resolve({}), d)
            runner.assert_not_called()

    def test_blocked_attempt_cannot_silently_run_again(self):
        with tempfile.TemporaryDirectory() as d, patch.object(e, "supervise") as runner:
            root = Path(d) / "blocked"
            (Path(d) / "blocked.blocked.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                e.execute_run(e.resolve({}), root)
            runner.assert_not_called()

    def test_orphan_request_also_prevents_retry(self):
        with tempfile.TemporaryDirectory() as d, patch.object(e, "supervise") as runner:
            root = Path(d) / "ambiguous"
            (Path(d) / ".ambiguous.request-token.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                e.execute_run(e.resolve({}), root)
            runner.assert_not_called()

    def test_allocation_caps_cover_model_and_raw_export(self):
        for config in (
            {"hidden": 1000000},
            {"memory_capacity": 10000000},
            {"spatial_points": 2000, "validation_trajectories": 100},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                e.resolve(config)

    def test_campaign_reserves_budget_before_launch_failure(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = e.load_manifest(self.manifest(d))
            root = Path(d) / "run"

            def fail(*args):
                state = json.loads((root / "campaign.json").read_text())
                self.assertEqual(state["reserved_seconds"], 150)
                self.assertEqual(state["cells"]["tiny"], "RUNNING")
                raise RuntimeError("simulated interruption")

            with (
                patch.object(e, "execute_run", side_effect=fail),
                self.assertRaises(RuntimeError),
            ):
                e.campaign(manifest, root)
            with (
                patch.object(e, "execute_run") as runner,
                self.assertRaises(RuntimeError),
            ):
                e.campaign(manifest, root, reuse=True)
            runner.assert_not_called()

    def test_embedded_source_inventory_is_verified(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = self.make_completed(root)
            (root / "source.json").write_text(
                json.dumps({"sha256": e.digest({}), "files": {"fake.py": "0" * 64}})
            )
            receipt = json.loads((root / "completion.json").read_text())
            receipt["artifacts"] = e.artifact_records(root)
            (root / "completion.json").write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                e.validate_completed(root, config, e.digest({}))

    def test_export_is_allowlisted_atomic_and_refuses_overwrite(self):
        import hashlib
        import zipfile

        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "bundle.zip"
            e.export_bundle(output, self.manifest(d))
            with zipfile.ZipFile(output) as archive:
                self.assertNotIn(".env", archive.namelist())
                manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
                for name, expected in manifest["files"].items():
                    self.assertEqual(
                        hashlib.sha256(archive.read(name)).hexdigest(), expected
                    )
                self.assertIn("src/fim_v2/resources.py", archive.namelist())
            previous = output.read_bytes()
            with self.assertRaises(FileExistsError):
                e.export_bundle(output, self.manifest(d))
            self.assertEqual(output.read_bytes(), previous)

    def test_supervisor_terminates_only_owned_group(self):
        import subprocess

        child = unittest.mock.Mock(pid=123456)
        child.wait.side_effect = [subprocess.TimeoutExpired("fake", 1), 0]
        child.poll.return_value = None
        with (
            patch.object(e.subprocess, "Popen", return_value=child),
            patch.object(e.os, "killpg") as kill,
        ):
            self.assertEqual(e.supervise(["fake"], 1), 124)
        kill.assert_called_once_with(child.pid, e.signal.SIGTERM)

    def test_dry_run_does_not_launch(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(e, "execute_run") as runner,
        ):
            from argparse import Namespace

            with contextlib.redirect_stdout(io.StringIO()) as out:
                e.handle(
                    Namespace(
                        engineering_command="campaign",
                        manifest=self.manifest(d),
                        dry_run=True,
                    )
                )
            self.assertEqual(len(json.loads(out.getvalue())["manifest"]["cells"]), 1)
            runner.assert_not_called()

    def test_live_lock_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            lock = root / ".campaign.lock"
            lock.write_text("existing")
            with self.assertRaises(FileExistsError):
                e.campaign({}, root, reuse=True)
            self.assertEqual(lock.read_text(), "existing")

    def test_unsafe_partial_reuse_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            (root / "tiny").mkdir(parents=True)
            manifest = e.load_manifest(self.manifest(d))
            with self.assertRaises(FileNotFoundError):
                e.campaign(manifest, root, reuse=True)
            self.assertEqual(
                json.loads((root / "campaign.json").read_text())["status"], "FAILED"
            )


if __name__ == "__main__":
    unittest.main()
