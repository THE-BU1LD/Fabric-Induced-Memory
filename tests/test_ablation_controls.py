from __future__ import annotations

import pytest
import torch

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
