"""Synthetic CLI checks for ambiguous JSON declarations; no retained outcomes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_trajectory_manifest_admission import artificial_manifest


ROOT = Path(__file__).resolve().parents[1]


def run_cli(tmp_path, raw):
    manifest = tmp_path / "artificial.json"
    manifest.write_text(raw, encoding="utf-8")
    outputs = [tmp_path / name for name in ("paired.csv", "table.md", "evidence.md")]
    old = [b"existing artificial csv\n", b"existing artificial table\n", b"existing artificial evidence\n"]
    for path, content in zip(outputs, old):
        path.write_bytes(content)
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/analyze_trajectory_isolated_ablations.py"),
        "--manifest", str(manifest), "--csv", str(outputs[0]),
        "--table", str(outputs[1]), "--evidence", str(outputs[2]),
    ], capture_output=True, text=True, cwd=tmp_path, timeout=15)
    return result, outputs, old


@pytest.mark.parametrize("marker, replacement", [
    ('"git_dirty": false', '"git_dirty": true, "git_dirty": false'),
    ('"ema": false', '"ema": true, "ema": false'),
    ('"retrieval_enabled": true', '"retrieval_enabled": false, "retrieval_enabled": true'),
])
def test_duplicate_declaration_keys_rejected_before_publication(tmp_path, marker, replacement):
    raw = json.dumps(artificial_manifest())
    assert marker in raw
    raw = raw.replace(marker, replacement, 1)
    result, outputs, old = run_cli(tmp_path, raw)
    assert result.returncode != 0, "Ambiguous duplicate-key manifest was accepted"
    assert "Duplicate manifest field" in result.stderr
    assert [path.read_bytes() for path in outputs] == old


def test_unambiguous_complete_artificial_matrix_still_publishes(tmp_path):
    raw = json.dumps(artificial_manifest())
    result, outputs, old = run_cli(tmp_path, raw)
    assert result.returncode == 0, result.stderr
    assert all(path.read_bytes() != previous for path, previous in zip(outputs, old))
    assert "benchmark,comparison" in outputs[0].read_text()
    assert "Source manifest SHA-256" in outputs[1].read_text()
    assert "Source manifest SHA-256" in outputs[2].read_text()
