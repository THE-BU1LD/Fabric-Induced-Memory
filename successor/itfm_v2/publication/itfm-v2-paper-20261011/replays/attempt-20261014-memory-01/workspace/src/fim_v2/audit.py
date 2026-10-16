from __future__ import annotations

import json
from pathlib import Path


REQUIRED_CONTROLS = {
    "no_memory",
    "tensor_memory",
    "erase",
    "sign_flip",
    "vector_broadcast",
    "recent_only",
    "last_value",
    "paired_swap",
    "oracle_anchor",
}


def audit(root: Path) -> dict[str, object]:
    protocol_path = root / "protocols/development_v2.json"
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    required = [
        "README.md",
        "PROJECT_CONTRACT.md",
        "METHODOLOGY.md",
        "protocols/development_v2.json",
        "src/fim_v2/data.py",
        "src/fim_v2/memory.py",
        "src/fim_v2/evaluation.py",
        "src/fim_v2/analysis.py",
        "src/fim_v2/experiment.py",
        "tests/test_data.py",
        "tests/test_memory.py",
        "tests/test_decision.py",
    ]
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".md", ".json"}
        and "results" not in path.parts
        and path.name != "audit.py"
    ).lower()
    checks = {
        "protocol_identity": protocol.get("protocol_id") == "itfm-development-v2",
        "development_only": protocol.get("status") == "development",
        "controls_exact": set(protocol.get("controls", [])) == REQUIRED_CONTROLS,
        "three_seeds": len(protocol["dataset"]["seeds"]) == 3,
        "context_alias_required": bool(
            protocol["identifiability"]["require_full_context_alias"]
        ),
        "future_divergence_required": bool(
            protocol["identifiability"]["require_future_divergence"]
        ),
        "age_beyond_context_required": bool(
            protocol["identifiability"]["require_age_beyond_context"]
        ),
        "protected_confirmation_absent": not bool(
            protocol["protected_public_confirmation"]["created"]
        ),
        "required_files_present": all((root / path).is_file() for path in required),
        "no_placeholder_markers": not any(
            marker in text for marker in ("todo", "fixme", "placeholder")
        ),
    }
    return {
        "schema_version": 1,
        "scope": "static contract audit only; not runtime or scientific evidence",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
    }
