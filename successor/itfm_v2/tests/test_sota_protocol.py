from __future__ import annotations

import copy
import json
from pathlib import Path

from fim_v2.sota import audit_sota_protocol

ROOT = Path(__file__).resolve().parents[1]


def load() -> tuple[dict[str, object], dict[str, object]]:
    protocol = json.loads(
        (ROOT / "protocols/sota_confirmation_v1.json").read_text(encoding="utf-8")
    )
    frontier = json.loads(
        (ROOT / "protocols/frontier_registry_2026-10-11.json").read_text(
            encoding="utf-8"
        )
    )
    return protocol, frontier


def test_frozen_sota_protocol_passes_static_audit() -> None:
    protocol, frontier = load()
    assert all(audit_sota_protocol(protocol, frontier).values())


def test_audit_rejects_a_marginal_primary_target() -> None:
    protocol, frontier = load()
    weakened = copy.deepcopy(protocol)
    weakened["primary_track"]["material_sota_gate"][  # type: ignore[index]
        "minimum_macro_relative_error_reduction_vs_reproduced_lano"
    ] = 0.01
    assert not audit_sota_protocol(weakened, frontier)["material_primary_gate"]


def test_audit_rejects_unlocked_source_identity() -> None:
    protocol, frontier = load()
    unlocked = copy.deepcopy(frontier)
    unlocked["frontiers"]["pobench_pde"]["code_commit"] = "main"  # type: ignore[index]
    assert not audit_sota_protocol(protocol, unlocked)["source_commits_locked"]
