from __future__ import annotations

import hashlib

import pytest
import torch

from fim_experiments import ablation
from fim_experiments.ablation import ControlledFIMSystem, normalize_variants, variant_controls
from fim_experiments.systems import FIMSystem


def test_full_variant_enables_writes_and_retrieval() -> None:
    assert variant_controls("full") == (True, True)


def test_no_memory_disables_writes_and_retrieval() -> None:
    assert variant_controls("no_memory") == (False, False)


def test_no_retrieval_keeps_writes_but_disables_retrieval() -> None:
    assert variant_controls("no_retrieval") == (True, False)


def test_normalize_variants_accepts_hyphenated_names_and_deduplicates() -> None:
    assert normalize_variants(["full", "no-memory", "no_memory", "no-retrieval"]) == [
        "full",
        "no_memory",
        "no_retrieval",
    ]


def test_unknown_variant_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown ablation variant"):
        variant_controls("rescued_positive_result")


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("full", (True, True)),
        ("no_memory", (False, False)),
        ("no_retrieval", (True, False)),
    ],
)
def test_controlled_step_ignores_trainer_memory_flags(monkeypatch, variant, expected) -> None:
    calls = []

    def fake_step(self, x, store_traces=True, retrieve=True):
        calls.append((store_traces, retrieve))
        return x

    monkeypatch.setattr(FIMSystem, "step", fake_step)
    store_traces, retrieve = variant_controls(variant)
    model = ControlledFIMSystem(
        in_channels=1,
        hidden=4,
        trace_dim=2,
        memory_capacity=4,
        retrieval_topk=1,
        store_traces=store_traces,
        retrieve=retrieve,
    )

    x = torch.zeros(1, 1, 1, 4)
    model.step(x, store_traces=not store_traces, retrieve=not retrieve)

    assert calls == [expected]


def test_forward_uses_fixed_variant_controls(monkeypatch) -> None:
    calls = []

    def fake_step(self, x, store_traces=True, retrieve=True):
        calls.append((store_traces, retrieve))
        return x

    monkeypatch.setattr(FIMSystem, "step", fake_step)
    model = ControlledFIMSystem(
        in_channels=1,
        hidden=4,
        trace_dim=2,
        memory_capacity=4,
        retrieval_topk=1,
        store_traces=False,
        retrieve=False,
    )

    model(torch.zeros(1, 1, 1, 4))

    assert calls == [(False, False)]


def test_provenance_records_parameter_count_checkpoint_hash_and_packages(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "final_state.pt"
    checkpoint_bytes = b"deterministic-checkpoint-fixture"
    checkpoint.write_bytes(checkpoint_bytes)

    monkeypatch.setattr(ablation, "_git_sha", lambda: "abc123")
    monkeypatch.setattr(
        ablation,
        "_package_manifest",
        lambda: [{"name": "example-package", "version": "1.2.3"}],
    )

    provenance = ablation._provenance(
        "full",
        41,
        {"experiment": {"name": "fixture"}},
        {"parameters": 12345},
        checkpoint,
    )

    assert provenance["git_sha"] == "abc123"
    assert provenance["parameters"] == 12345
    assert provenance["packages"] == [{"name": "example-package", "version": "1.2.3"}]
    assert provenance["checkpoint"] == {
        "path": str(checkpoint),
        "exists": True,
        "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "bytes": len(checkpoint_bytes),
    }


def test_provenance_fails_closed_to_missing_checkpoint_without_fake_hash(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "missing.pt"
    monkeypatch.setattr(ablation, "_git_sha", lambda: None)
    monkeypatch.setattr(ablation, "_package_manifest", lambda: [])

    provenance = ablation._provenance(
        "no_memory",
        42,
        {},
        {"parameters": 99},
        checkpoint,
    )

    assert provenance["checkpoint"] == {
        "path": str(checkpoint),
        "exists": False,
    }
    assert "sha256" not in provenance["checkpoint"]
