from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from scripts.verify_checkpoint_identity import _config_sha256, verify_checkpoint_identity


def _write_checkpoint(path: Path, *, config=None, source_sha=None) -> None:
    payload = {"model_state_dict": {"weight": torch.tensor([1.0, 2.0])}}
    if config is not None:
        payload["config"] = config
    if source_sha is not None:
        payload["source_git_sha"] = source_sha
    torch.save(payload, path)


def test_identity_receipt_hashes_checkpoint_and_matching_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "final_state.pt"
    config = {"model": {"hidden": 8}, "runtime": {"seed": 7, "deterministic": True}}
    _write_checkpoint(checkpoint, config=config, source_sha="abc123")
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    receipt = verify_checkpoint_identity(checkpoint, config_path, expected_source_sha="abc123")

    assert receipt["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert receipt["checkpoint"]["state_dict_key"] == "model_state_dict"
    assert receipt["checkpoint"]["declared_source_git_sha"] == "abc123"
    assert receipt["config"]["embedded_normalized_sha256"] == _config_sha256(config)
    assert receipt["config"]["embedded_external_match"] is True
    assert receipt["load_policy"] == "torch.load(weights_only=True)"


def test_config_digest_is_order_invariant() -> None:
    left = {"runtime": {"seed": 7, "deterministic": True}, "model": {"hidden": 8}}
    right = {"model": {"hidden": 8}, "runtime": {"deterministic": True, "seed": 7}}
    assert _config_sha256(left) == _config_sha256(right)


def test_mismatched_external_config_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "final_state.pt"
    _write_checkpoint(checkpoint, config={"runtime": {"seed": 7}})
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump({"runtime": {"seed": 8}}), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match checkpoint-embedded config"):
        verify_checkpoint_identity(checkpoint, config_path)


def test_expected_source_sha_fails_when_checkpoint_has_no_source_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "final_state.pt"
    _write_checkpoint(checkpoint, config={"runtime": {"seed": 7}})

    with pytest.raises(ValueError, match="declares no source Git SHA"):
        verify_checkpoint_identity(checkpoint, expected_source_sha="abc123")


def test_require_source_sha_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "final_state.pt"
    _write_checkpoint(checkpoint, config={"runtime": {"seed": 7}})

    with pytest.raises(ValueError, match="does not declare a source Git SHA"):
        verify_checkpoint_identity(checkpoint, require_source_sha=True)
