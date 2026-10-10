"""Filesystem fixtures only; no retained checkpoint or scientific outcome is read."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fim_evaluation_output_test", ROOT / "scripts" / "evaluate_checkpoint.py"
)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
_import_path = list(sys.path)
try:
    SPEC.loader.exec_module(evaluator)
finally:
    sys.path[:] = _import_path


@pytest.fixture
def fixture_cli(tmp_path, monkeypatch):
    checkpoint = tmp_path / "run" / "checkpoints" / "fixture.pt"
    checkpoint.parent.mkdir(parents=True)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.5)
    torch.save({"model": model.state_dict(), "config": {}}, checkpoint)
    original_checkpoint = checkpoint.read_bytes()
    args = Namespace(
        checkpoint=checkpoint, config=None, device="cpu", rollout_steps=1,
        batch_size=1, weights="auto", output_dir=tmp_path / "evaluation",
    )
    calls = {"loads": 0, "evaluations": 0}
    load_checkpoint = evaluator._load_checkpoint

    def counted_load(*arguments):
        calls["loads"] += 1
        return load_checkpoint(*arguments)

    def fake_evaluation(_model, _benchmark, **kwargs):
        calls["evaluations"] += 1
        # These literal artificial payloads test publication plumbing only.
        kwargs["plot_path"].write_bytes(b"artificial plot fixture")
        kwargs["save_results_path"].write_text('{"artificial_fixture": true}\n')
        return {"artificial_fixture": True}

    monkeypatch.setattr(evaluator, "parse_args", lambda: args)
    monkeypatch.setattr(evaluator, "_load_checkpoint", counted_load)
    monkeypatch.setattr(evaluator, "select_benchmark", lambda _: object())
    monkeypatch.setattr(evaluator, "build_system", lambda *arguments: model)
    monkeypatch.setattr(evaluator, "run", fake_evaluation)
    yield args, calls
    assert checkpoint.read_bytes() == original_checkpoint


@pytest.mark.parametrize(
    "kind", ["populated_directory", "empty_directory", "file", "directory_symlink", "file_symlink", "dangling_symlink"]
)
def test_existing_destination_is_rejected_before_checkpoint_load(fixture_cli, tmp_path, kind):
    args, calls = fixture_cli
    output = args.output_dir
    target = tmp_path / "unrelated"
    if kind in ("populated_directory", "empty_directory"):
        output.mkdir()
        if kind == "populated_directory":
            (output / "metrics.json").write_bytes(b"earlier metrics\r\n")
            (output / "evaluation_provenance.json").write_bytes(b"earlier provenance\n")
    elif kind == "file":
        output.write_bytes(b"unrelated file\n")
    else:
        if kind == "directory_symlink":
            target.mkdir()
            (target / "metrics.json").write_bytes(b"unrelated metrics\n")
        elif kind == "file_symlink":
            target.write_bytes(b"unrelated linked file\n")
        output.symlink_to(target, target_is_directory=kind == "directory_symlink")

    original_files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="Evaluation output already exists"):
        evaluator.main()
    assert calls == {"loads": 0, "evaluations": 0}
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == original_files
    if "symlink" in kind:
        assert output.is_symlink()
        assert output.readlink() == target


@pytest.mark.parametrize("use_default", [False, True])
def test_fresh_output_succeeds_and_a_retry_cannot_overwrite_it(fixture_cli, use_default):
    args, calls = fixture_cli
    output = args.output_dir
    if use_default:
        args.output_dir = None
        output = args.checkpoint.parent.parent / "evaluation"
    evaluator.main()
    assert calls == {"loads": 1, "evaluations": 1}
    assert json.loads((output / "metrics.json").read_text()) == {"artificial_fixture": True}
    provenance = json.loads((output / "evaluation_provenance.json").read_text())
    assert provenance["checkpoint"] == str(args.checkpoint)
    assert provenance["weight_source"] == "raw"
    original = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(FileExistsError, match="Evaluation output already exists"):
        evaluator.main()
    assert calls == {"loads": 1, "evaluations": 1}
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original


def test_destination_appearing_after_preflight_is_preserved(fixture_cli, monkeypatch):
    args, calls = fixture_cli
    original_build = evaluator.build_system

    def competing_creation(*arguments):
        args.output_dir.mkdir()
        (args.output_dir / "metrics.json").write_bytes(b"another completed run\n")
        return original_build(*arguments)

    monkeypatch.setattr(evaluator, "build_system", competing_creation)
    with pytest.raises(FileExistsError):
        evaluator.main()
    assert calls == {"loads": 1, "evaluations": 0}
    assert (args.output_dir / "metrics.json").read_bytes() == b"another completed run\n"
    assert sorted(path.name for path in args.output_dir.iterdir()) == ["metrics.json"]


def test_actual_cli_refuses_reuse_before_reading_a_missing_checkpoint(tmp_path):
    output = tmp_path / "existing evaluation"
    output.mkdir()
    original = output / "metrics.json"
    original.write_bytes(b"preserve these earlier metrics\n")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_checkpoint.py"),
         "--checkpoint", str(tmp_path / "not-a-real-checkpoint.pt"),
         "--output_dir", str(output)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "Evaluation output already exists" in result.stderr
    assert "Checkpoint not found" not in result.stderr
    assert original.read_bytes() == b"preserve these earlier metrics\n"
    assert sorted(path.name for path in output.iterdir()) == ["metrics.json"]
