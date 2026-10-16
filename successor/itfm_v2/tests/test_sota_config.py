from __future__ import annotations

import json
from pathlib import Path

import pytest

from fim_sota.config import ConfigurationError, load_config, resolve_config

ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_configuration_resolves_stably() -> None:
    config = load_config(ROOT / "configs/sota/base.json")
    assert config.schema_version == 1
    assert config.dataset.benchmark == "pobench_pde"
    assert json.loads(config.resolved_json())["memory"]["topk"] == 4


def test_version_zero_migration_is_narrow_and_unknown_fields_fail() -> None:
    payload = json.loads((ROOT / "configs/sota/base.json").read_text())
    payload["schema_version"] = 0
    payload["results_root"] = payload.pop("output_root")
    assert resolve_config(payload).schema_version == 1
    payload["invented"] = True
    with pytest.raises(ConfigurationError):
        resolve_config(payload)


def test_protected_evaluation_requires_frozen_receipt() -> None:
    payload = json.loads((ROOT / "configs/sota/base.json").read_text())
    payload["evaluation"]["protected_test"] = True
    with pytest.raises(ConfigurationError, match="frozen development receipt"):
        resolve_config(payload)
