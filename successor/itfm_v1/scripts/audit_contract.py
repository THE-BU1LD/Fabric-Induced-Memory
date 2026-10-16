from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> dict[str, Any]:
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    protocol = load_json("protocols/development_v1.json")
    state = load_json("RESEARCH_STATE.json")
    models = list(protocol["models"])
    seeds = list(protocol["training"]["seeds"])
    expected_cells = len(models) * len(seeds) + len(protocol.get("sensitivity", []))
    required_models = {
        "itfm_transport",
        "itfm",
        "no_memory",
        "vector_broadcast",
        "sham_transport",
        "finite_history_cnn",
        "conv_gru",
        "gated_diagonal_ssm",
        "periodic_xcorr",
    }
    required_files = {
        "PROJECT_CONTRACT.md",
        "METHODOLOGY.md",
        "LITERATURE_BOUNDARY.md",
        "paper/SPECIFICATION.md",
        "src/fim_s1/memory.py",
        "src/fim_s1/model.py",
        "src/fim_s1/model_nd.py",
        "src/fim_s1/transport.py",
        "src/fim_s1/datasets.py",
        "src/fim_s1/experiment.py",
        "src/fim_s1/analysis.py",
        "datasets/external/pdebench/pdebench_data_urls.csv",
        "datasets/external/pdebench/LICENSE.txt",
    }
    checks = {
        "canonical_state": state.get("canonical_state") is True,
        "development_protocol": protocol.get("status") == "development",
        "protected_test_absent": protocol.get("protected_test", {}).get("created")
        is False,
        "primary_model_declared": protocol.get("primary_model") in models,
        "models_unique": len(models) == len(set(models)),
        "required_controls_exact": set(models) == required_models,
        "expected_cells_27": expected_cells == 27,
        "budget_covers_matrix": int(state["budgets"]["development_training_runs"])
        >= expected_cells,
        "transport_named_exactly": protocol["model"].get("transport_mode")
        == "bounded_xcorr",
        "parameter_tolerance_predeclared": protocol["model"].get(
            "parameter_match_relative_tolerance"
        )
        == 0.05,
        "no_training_sensitivity": not protocol.get("sensitivity"),
        "required_files_present": all(
            (ROOT / path).is_file() for path in required_files
        ),
        "pdebench_registry_hash": sha256(
            ROOT / "datasets/external/pdebench/pdebench_data_urls.csv"
        )
        == "469f43609413c475740a54f1246a09be16bdffcc8b9c064f307fcf02f12a8d6a",
        "pdebench_license_hash": sha256(ROOT / "datasets/external/pdebench/LICENSE.txt")
        == "53015b22f7d9eccb67ab5aeb991b1b705d367ee70efa2311d34e72d2a36b9cc9",
    }
    inspected = sorted(
        path
        for parent in (ROOT / "src", ROOT / "scripts", ROOT / "tests")
        for path in parent.rglob("*.py")
        if path.resolve() != Path(__file__).resolve()
    )
    forbidden = ("TODO", "FIXME", "NotImplementedError", "placeholder", "scaffold")
    findings = {
        path.relative_to(ROOT).as_posix(): [
            token for token in forbidden if token in path.read_text(encoding="utf-8")
        ]
        for path in inspected
    }
    findings = {path: tokens for path, tokens in findings.items() if tokens}
    checks["no_placeholder_markers"] = not findings
    receipt = {
        "schema_version": 1,
        "status": "pass" if all(checks.values()) else "fail",
        "checked_at_unix": time.time(),
        "checks": checks,
        "expected_cells": expected_cells,
        "models": models,
        "seeds": seeds,
        "placeholder_findings": findings,
        "scope": "Static contract consistency only; not runtime or scientific evidence.",
    }
    destination = ROOT / "results" / "static_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(json.dumps({"status": receipt["status"], "checks": checks}, sort_keys=True))
    if receipt["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
