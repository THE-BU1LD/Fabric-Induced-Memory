"""Artificial filesystem and runner-control tests; no scientific backend runs."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts import run_output

ROOT = Path(__file__).resolve().parents[1]


def test_fresh_nested_output_and_complete_manifest_updates(tmp_path):
    root = tmp_path / "fresh"
    manifest = root / "manifest.json"
    run_output.prepare_run_outputs(root, manifest, {"runs": []})
    assert root.is_dir()
    assert json.loads(manifest.read_text()) == {"runs": []}
    manifest.chmod(0o640)
    run_output.write_run_manifest(manifest, {"runs": [{"status": "failed"}]})
    assert json.loads(manifest.read_text()) == {"runs": [{"status": "failed"}]}
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o640
    assert list(root.iterdir()) == [manifest]


def test_external_manifest_destination_remains_supported(tmp_path):
    root = tmp_path / "fresh"
    manifest = tmp_path / "receipts" / "run.json"
    run_output.prepare_run_outputs(root, manifest, {"fixture": True})
    assert json.loads(manifest.read_text()) == {"fixture": True}
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o600


@pytest.mark.parametrize("destination", ["root", "manifest"])
@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling"])
def test_existing_destinations_are_preserved(tmp_path, destination, kind):
    root = tmp_path / "run"
    manifest = tmp_path / "manifest.json"
    target = root if destination == "root" else manifest
    original = tmp_path / "original"
    original.write_bytes(b"prior evidence")
    if kind == "file":
        target.write_bytes(b"retained bytes")
    elif kind == "directory":
        target.mkdir()
        (target / "checkpoint").write_bytes(b"checkpoint bytes")
    else:
        target.symlink_to(original if kind == "symlink" else tmp_path / "missing")
    with pytest.raises(FileExistsError, match="fresh destination"):
        run_output.prepare_run_outputs(root, manifest, {"runs": []})
    assert original.read_bytes() == b"prior evidence"
    if kind == "file":
        assert target.read_bytes() == b"retained bytes"
    elif kind == "directory":
        assert (target / "checkpoint").read_bytes() == b"checkpoint bytes"
    else:
        assert target.is_symlink()
    if destination == "manifest":
        assert not root.exists()


@pytest.mark.parametrize("same_path", [True, False])
def test_manifest_cannot_be_results_root_or_its_ancestor(tmp_path, same_path):
    manifest = tmp_path / "new"
    root = manifest if same_path else manifest / "child"
    with pytest.raises(ValueError, match="must not equal or contain"):
        run_output.prepare_run_outputs(root, manifest, {"runs": []})
    assert not manifest.exists()


def test_competing_root_creation_cannot_reuse_other_run(tmp_path, monkeypatch):
    root = tmp_path / "run"
    original_mkdir = Path.mkdir

    def create_competing_directory(path, *args, **kwargs):
        if path == root:
            original_mkdir(path)
            (path / "other-run").write_bytes(b"other process")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", create_competing_directory)
    with pytest.raises(FileExistsError):
        run_output.prepare_run_outputs(root, root / "manifest.json", {"runs": []})
    assert (root / "other-run").read_bytes() == b"other process"
    assert not (root / "manifest.json").exists()


def test_competing_manifest_creation_cannot_be_overwritten(tmp_path, monkeypatch):
    root = tmp_path / "run"
    manifest = root / "manifest.json"
    original_link = os.link

    def create_competing_manifest(source, target):
        Path(target).write_bytes(b"other manifest")
        return original_link(source, target)

    monkeypatch.setattr(run_output.os, "link", create_competing_manifest)
    with pytest.raises(FileExistsError):
        run_output.prepare_run_outputs(root, manifest, {"runs": []})
    assert manifest.read_bytes() == b"other manifest"
    assert list(root.iterdir()) == [manifest]


@pytest.mark.parametrize("symlink", [False, True])
def test_fixed_temporary_name_is_never_overwritten(tmp_path, symlink):
    manifest = tmp_path / "manifest.json"
    old_temporary = tmp_path / "manifest.json.tmp"
    original = tmp_path / "original"
    original.write_bytes(b"unrelated evidence")
    if symlink:
        old_temporary.symlink_to(original)
    else:
        old_temporary.write_bytes(b"stale snapshot")
    run_output.write_run_manifest(manifest, {"runs": []}, create=True)
    run_output.write_run_manifest(manifest, {"runs": [{"status": "success"}]})
    assert original.read_bytes() == b"unrelated evidence"
    assert old_temporary.is_symlink() if symlink else old_temporary.read_bytes() == b"stale snapshot"
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


@pytest.mark.parametrize("failure", ["serialization", "fsync", "replace"])
def test_failed_update_preserves_previous_complete_manifest(tmp_path, monkeypatch, failure):
    manifest = tmp_path / "manifest.json"
    previous = b'{"runs": [{"status": "failed", "error": "retained"}]}\n'
    manifest.write_bytes(previous)
    payload = {"runs": []}

    def fail(*args, **kwargs):
        raise OSError("injected publication failure")

    if failure == "serialization":
        payload["invalid"] = object()
    else:
        monkeypatch.setattr(run_output.os, failure, fail)
    with pytest.raises((OSError, TypeError)):
        run_output.write_run_manifest(manifest, payload)
    assert manifest.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [manifest]


def test_failed_initial_flush_does_not_publish_partial_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"

    def fail(*args):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(run_output.os, "fsync", fail)
    with pytest.raises(OSError, match="injected"):
        run_output.write_run_manifest(manifest, {"runs": []}, create=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["symlink", "directory", "missing"])
def test_update_requires_an_existing_regular_manifest(tmp_path, kind):
    manifest = tmp_path / "manifest.json"
    original = tmp_path / "original"
    original.write_bytes(b"keep this")
    if kind == "symlink":
        manifest.symlink_to(original)
    elif kind == "directory":
        manifest.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        run_output.write_run_manifest(manifest, {"runs": []})
    assert original.read_bytes() == b"keep this"
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


@pytest.fixture
def artificial_runner(monkeypatch):
    # Import the real control flow with backend doubles only for import-time
    # definitions. No Torch installation, model or benchmark is selected.
    fake_torch = ModuleType("torch")
    fake_torch.no_grad = lambda: lambda function: function
    fake_torch.__version__ = "artificial-test-backend"
    fake_main = ModuleType("main")
    fake_main.run = lambda *args, **kwargs: None
    fake_ablation = ModuleType("ablation_systems")
    fake_ablation.AblatedFIMSystem = object
    fake_ablation.variant_switches = lambda name: {}
    before_path = sys.path[:]
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "torch", fake_torch)
        imports.setitem(sys.modules, "main", fake_main)
        imports.setitem(sys.modules, "train", ModuleType("train"))
        imports.setitem(sys.modules, "ablation_systems", fake_ablation)
        spec = importlib.util.spec_from_file_location(
            "artificial_trajectory_output_runner", ROOT / "scripts" / "run_trajectory_isolated_ablations.py",
        )
        runner = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(runner)
        finally:
            sys.path[:] = before_path
    monkeypatch.setattr(runner, "validate_frozen_args", lambda args: None)
    monkeypatch.setattr(runner, "manifest_payload", lambda args, rows: {"fixture": True, "runs": rows})
    return runner


def fixture_args(tmp_path):
    root = tmp_path / "artificial-run"
    return SimpleNamespace(
        results_root=root, manifest=root / "manifest.json",
        benchmarks=["artificial"], seeds=[1], variants=["first", "second"],
    )


def test_runner_rejects_reused_output_before_hooks_or_execution(tmp_path, monkeypatch, artificial_runner):
    args = fixture_args(tmp_path)
    args.results_root.mkdir()
    args.manifest.write_bytes(b"retained outcome")
    calls = []
    monkeypatch.setattr(artificial_runner, "parse_args", lambda: args)
    monkeypatch.setattr(artificial_runner, "install_trajectory_isolation_hooks", lambda: calls.append("hook"))
    monkeypatch.setattr(artificial_runner, "run_one", lambda *a: calls.append("run"))
    with pytest.raises(FileExistsError):
        artificial_runner.main()
    assert calls == []
    assert args.manifest.read_bytes() == b"retained outcome"


def test_runner_retains_failed_cell_before_continuing(tmp_path, monkeypatch, artificial_runner):
    args = fixture_args(tmp_path)
    seen = []
    monkeypatch.setattr(artificial_runner, "parse_args", lambda: args)
    monkeypatch.setattr(artificial_runner, "install_trajectory_isolation_hooks", lambda: None)

    def artificial_cell(args, benchmark, seed, variant):
        snapshot = json.loads(args.manifest.read_text())
        seen.append(snapshot)
        if variant == "first":
            raise RuntimeError("artificial backend failure")
        return {"status": "success", "variant": variant}

    monkeypatch.setattr(artificial_runner, "run_one", artificial_cell)
    with pytest.raises(SystemExit) as error:
        artificial_runner.main()
    assert error.value.code == 2
    assert seen[0]["runs"] == []
    assert seen[1]["runs"][0]["status"] == "failed"
    rows = json.loads(args.manifest.read_text())["runs"]
    assert [row["status"] for row in rows] == ["failed", "success"]
    assert rows[0]["error"] == "artificial backend failure"
