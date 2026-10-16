from __future__ import annotations

import torch

from fim_sota.metrics import relative_l2, spectral_relative_error
from fim_sota.statistics import (
    EvaluationCell,
    family_cluster_bootstrap,
    holm_adjust,
    missing_run_accounting,
    paired_effect,
)


def test_metrics_are_zero_for_exact_prediction() -> None:
    target = torch.randn(3, 1, 8, 8)
    assert torch.equal(relative_l2(target, target), torch.zeros(3))
    assert torch.equal(spectral_relative_error(target, target), torch.zeros(3))


def test_paired_statistics_preserve_trajectory_units() -> None:
    rows = [
        EvaluationCell(method, seed, family, trajectory, "cell", value)
        for method, seed, family, trajectory, value in (
            ("candidate", 1, "f1", "t1", 0.5),
            ("baseline", 1, "f1", "t1", 1.0),
            ("candidate", 2, "f2", "t2", 0.75),
            ("baseline", 2, "f2", "t2", 1.0),
        )
    ]
    effect = paired_effect([0.5, 0.75], [1.0, 1.0])
    assert effect["win_fraction"] == 1.0
    bootstrap = family_cluster_bootstrap(
        rows,
        candidate="candidate",
        baseline="baseline",
        repetitions=100,
        confidence=0.95,
        seed=4,
    )
    assert bootstrap["lower"] > 0


def test_holm_and_missing_runs_are_fail_closed() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.04}, alpha=0.05)
    assert adjusted["rejected"]["a"]
    accounting = missing_run_accounting(
        [], expected_methods=("candidate",), expected_seeds=(1,), expected_cells=("c",)
    )
    assert accounting["missing"] == [["candidate", 1, "c"]]
    assert not accounting["post_hoc_exclusions_allowed"]
