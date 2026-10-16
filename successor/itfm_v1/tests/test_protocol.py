import json
from pathlib import Path


def test_protocol_matrix_and_protected_boundary_are_consistent() -> None:
    protocol = json.loads(
        Path("protocols/development_v1.json").read_text(encoding="utf-8")
    )
    assert protocol["primary_model"] == "itfm_transport"
    assert len(protocol["models"]) == len(set(protocol["models"])) == 9
    assert len(protocol["models"]) * len(protocol["training"]["seeds"]) == 27
    assert protocol["sensitivity"] == []
    assert protocol["protected_test"]["created"] is False
    assert protocol["model"]["transport_mode"] == "bounded_xcorr"
    assert protocol["model"]["parameter_match_relative_tolerance"] == 0.05
