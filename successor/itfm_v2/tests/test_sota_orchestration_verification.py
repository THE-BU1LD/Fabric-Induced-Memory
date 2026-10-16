from __future__ import annotations

import json
from pathlib import Path

import pytest

from fim_sota.orchestration import (
    BoundInput,
    StageAdmissionError,
    StageRunner,
    StageSpec,
    verify_frozen_development_receipt,
)
from fim_sota.provenance import sha256_file
from fim_sota.verification import VerificationError, verify_bound_artifacts


def test_stage_dry_run_is_immutable_and_executes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("bound")
    specification = StageSpec(
        schema_version=1,
        stage_id="S2_ORACLE_GAP",
        command=("python", "-c", "raise SystemExit('must not execute')"),
        working_directory=str(tmp_path),
        inputs=(BoundInput(str(source), sha256_file(source)),),
        expected_outputs=(),
    )
    result = StageRunner(tmp_path / "runs").run(
        specification, execute=False, run_id="fixed-run"
    )
    assert result["status"] == "DRY_RUN_VALIDATED"
    with pytest.raises(FileExistsError):
        StageRunner(tmp_path / "runs").run(
            specification, execute=False, run_id="fixed-run"
        )


def test_protected_unlock_requires_exact_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"status": "PASS"}))
    with pytest.raises(StageAdmissionError):
        verify_frozen_development_receipt(receipt)


def test_corrupted_artifact_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    manifest = {
        "outputs": [
            {"path": str(artifact), "sha256": sha256_file(artifact), "bytes": 2}
        ]
    }
    assert verify_bound_artifacts(manifest)
    artifact.write_text('{"changed": true}')
    with pytest.raises(VerificationError, match="hash mismatch"):
        verify_bound_artifacts(manifest)
