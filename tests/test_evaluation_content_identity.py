"""Artificial checkpoint/config fixtures; no retained scientific data is read."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fim_evaluation_identity_test", ROOT / "scripts" / "evaluate_checkpoint.py"
)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
_import_path = list(sys.path)
try:
    SPEC.loader.exec_module(evaluator)
finally:
    sys.path[:] = _import_path


def _config_digest(config):
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@pytest.fixture
def fixture_cli(tmp_path, monkeypatch):
    checkpoint = tmp_path / "run" / "checkpoints" / "fixture.pt"
    checkpoint.parent.mkdir(parents=True)
    args = Namespace(
        checkpoint=checkpoint, config=None, device="cpu", rollout_steps=1,
        batch_size=1, weights="auto", output_dir=tmp_path / "evaluation",
    )
    calls = {"benchmarks": 0, "evaluations": 0, "loaded_weights": []}
    model = torch.nn.Linear(1, 1, bias=False)

    def save_fixture(weight=0.25, config=None, path=checkpoint):
        payload = {"model": {"weight": torch.tensor([[weight]])}}
        if config is not None:
            payload["config"] = config
        torch.save(payload, path)
        return path.read_bytes()

    def benchmark(_config):
        calls["benchmarks"] += 1
        return object()

    def evaluation(_model, _benchmark, **kwargs):
        calls["evaluations"] += 1
        calls["loaded_weights"].append(float(_model.weight.detach().item()))
        kwargs["save_results_path"].write_text('{"artificial_fixture": true}\n')
        return {"artificial_fixture": True}

    monkeypatch.setattr(evaluator, "parse_args", lambda: args)
    monkeypatch.setattr(evaluator, "set_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator, "select_benchmark", benchmark)
    monkeypatch.setattr(evaluator, "build_system", lambda *args: model)
    monkeypatch.setattr(evaluator, "run", evaluation)
    save_fixture(config={})
    return args, calls, save_fixture


def _run(args):
    evaluator.main()
    return json.loads((args.output_dir / "evaluation_provenance.json").read_text())


def test_same_checkpoint_path_with_different_bytes_has_distinct_identity(fixture_cli, tmp_path):
    args, calls, save = fixture_cli
    first_bytes = save(weight=0.25, config={})
    first = _run(args)
    second_bytes = save(weight=0.75, config={})
    args.output_dir = tmp_path / "second evaluation"
    second = _run(args)

    assert calls["loaded_weights"] == [0.25, 0.75]
    assert first != second, "Different checkpoint content must not receive identical provenance"
    assert first["checkpoint_sha256"] == hashlib.sha256(first_bytes).hexdigest()
    assert second["checkpoint_sha256"] == hashlib.sha256(second_bytes).hexdigest()
    assert first["checkpoint_size_bytes"] == len(first_bytes)
    assert second["checkpoint_size_bytes"] == len(second_bytes)
    assert first["checkpoint"] == second["checkpoint"] == str(args.checkpoint)


@pytest.mark.parametrize("source", ["explicit_yaml", "checkpoint_embedded", "sibling_yaml"])
def test_resolved_config_source_and_content_follow_existing_precedence(fixture_cli, tmp_path, source):
    args, _calls, save = fixture_cli
    sibling = args.checkpoint.parent.parent / "configs" / "resolved_config.yaml"
    sibling.parent.mkdir()
    configs = {
        "explicit_yaml": {"marker": "explicit", "runtime": {"seed": 17}},
        "checkpoint_embedded": {"marker": "embedded", "runtime": {"seed": 23}},
        "sibling_yaml": {"marker": "sibling", "runtime": {"seed": 29}},
    }
    sibling.write_text(yaml.safe_dump(configs["sibling_yaml"]))
    save(config=configs["checkpoint_embedded"] if source != "sibling_yaml" else None)
    expected_path = None
    if source == "explicit_yaml":
        args.config = tmp_path / "explicit.yaml"
        args.config.write_text(yaml.safe_dump(configs[source]))
        expected_path = str(args.config)
    elif source == "sibling_yaml":
        expected_path = str(sibling)

    result = _run(args)
    assert result["config_source"] == source
    assert result["config_source_path"] == expected_path
    assert result["resolved_config_sha256"] == _config_digest(configs[source])
    assert result["seed"] == configs[source]["runtime"]["seed"]


def test_equivalent_yaml_formatting_has_the_same_resolved_digest(fixture_cli, tmp_path):
    args, _calls, _save = fixture_cli
    args.config = tmp_path / "fixture.yaml"
    args.config.write_text('marker: "caf\u00e9"\nruntime: {seed: 17, deterministic: true}\n')
    first = _run(args)
    args.config.write_text('runtime:\n  deterministic: true\n  seed: 17\nmarker: caf\u00e9\n')
    args.output_dir = tmp_path / "equivalent evaluation"
    second = _run(args)
    assert first["resolved_config_sha256"] == second["resolved_config_sha256"]
    assert first["resolved_config_sha256"] == _config_digest(
        {"marker": "caf\u00e9", "runtime": {"seed": 17, "deterministic": True}}
    )


def test_yaml_replacement_after_parsing_does_not_change_consumed_config_identity(fixture_cli, tmp_path, monkeypatch):
    args, _calls, _save = fixture_cli
    args.config = tmp_path / "fixture.yaml"
    consumed = {"runtime": {"seed": 17}}
    replacement = {"runtime": {"seed": 99}}
    args.config.write_text(yaml.safe_dump(consumed))
    real_load = evaluator.load_yaml_config

    def replace_after_read(path):
        result = real_load(path)
        path.write_text(yaml.safe_dump(replacement))
        return result

    monkeypatch.setattr(evaluator, "load_yaml_config", replace_after_read)
    result = _run(args)
    assert result["seed"] == 17
    assert result["resolved_config_sha256"] == _config_digest(consumed)
    assert result["resolved_config_sha256"] != _config_digest(replacement)


@pytest.mark.parametrize("change", ["replace", "truncate_in_place"])
def test_checkpoint_identity_matches_snapshot_even_if_source_changes_before_deserialization(fixture_cli, tmp_path, monkeypatch, change):
    args, calls, save = fixture_cli
    consumed = save(weight=0.25, config={})
    replacement = tmp_path / "replacement.pt"
    save(weight=0.75, config={}, path=replacement)
    real_load = evaluator.torch.load

    def change_source_then_load(file, *arguments, **kwargs):
        if change == "replace":
            replacement.replace(args.checkpoint)
        else:
            args.checkpoint.write_bytes(b"artificial interrupted rewrite")
        return real_load(file, *arguments, **kwargs)

    monkeypatch.setattr(evaluator.torch, "load", change_source_then_load)
    result = _run(args)
    assert calls["loaded_weights"] == [0.25]
    assert result["checkpoint_sha256"] == hashlib.sha256(consumed).hexdigest()
    assert result["checkpoint_size_bytes"] == len(consumed)
    assert result["checkpoint_sha256"] != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()


def test_large_snapshot_spills_to_disk_and_is_closed_after_load(fixture_cli, monkeypatch):
    args, _calls, _save = fixture_cli
    monkeypatch.setattr(evaluator, "_CHECKPOINT_SPOOL_MAX_BYTES", 64, raising=False)
    captured = []
    real_load = evaluator.torch.load

    def inspect_snapshot(file, *arguments, **kwargs):
        assert not isinstance(file, (str, Path)), "Deserialization must consume the retained snapshot"
        assert file._rolled, "A checkpoint larger than the memory cap must spill to disk"
        captured.append(file)
        return real_load(file, *arguments, **kwargs)

    monkeypatch.setattr(evaluator.torch, "load", inspect_snapshot)
    _run(args)
    assert captured[0].closed


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt, SystemExit])
def test_snapshot_is_closed_and_no_outputs_exist_after_load_failure(fixture_cli, monkeypatch, failure):
    args, calls, _save = fixture_cli
    captured = []

    def fail_loading(file, *arguments, **kwargs):
        captured.append(file)
        raise failure("artificial load failure")

    monkeypatch.setattr(evaluator.torch, "load", fail_loading)
    with pytest.raises(failure, match="artificial load failure"):
        evaluator.main()
    assert not args.output_dir.exists()
    assert calls["benchmarks"] == calls["evaluations"] == 0
    assert captured and not isinstance(captured[0], (str, Path))
    assert captured[0].closed


@pytest.mark.parametrize("invalid", [{"extra": float("nan")}, {"extra": {1: "ambiguous JSON key"}}])
def test_ambiguous_config_identity_fails_before_benchmark_or_output(fixture_cli, invalid):
    args, calls, save = fixture_cli
    save(config=invalid)
    with pytest.raises((TypeError, ValueError)):
        evaluator.main()
    assert calls["benchmarks"] == calls["evaluations"] == 0
    assert not args.output_dir.exists()
