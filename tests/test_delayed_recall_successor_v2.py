from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from fim_experiments.benchmark import DelayedRecallBenchmark, DelayedRecallConfig
from fim_experiments.recall_metrics import (
    RecallWindowSpec,
    delayed_recall_observation_noise_floor_mse,
    recall_window_memory_mse,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "protocols" / "FIM_DELAYED_RECALL_SUCCESSOR_V2.json"
VERIFIER = ROOT / "scripts" / "verify_delayed_recall_successor_v2.py"
SPEC = importlib.util.spec_from_file_location("delayed_recall_v2_verifier", VERIFIER)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def load_protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_recall_window_maps_observation_times_to_target_indices() -> None:
    spec = RecallWindowSpec(delay=8, window=4, memory_dim=32)
    assert spec.observation_times == (8, 9, 10, 11)
    assert spec.target_indices(20) == (7, 8, 9, 10)
    assert spec.target_slice(20) == slice(7, 11)


def test_recall_window_rejects_short_target_sequence() -> None:
    spec = RecallWindowSpec(delay=8, window=4, memory_dim=32)
    with pytest.raises(ValueError, match="too short"):
        spec.target_indices(10)


def test_recall_metric_uses_only_memory_payload_channels() -> None:
    spec = RecallWindowSpec(delay=2, window=2, memory_dim=2)
    target = torch.zeros(1, 5, 3)
    prediction = torch.zeros_like(target)
    prediction[:, 1:3, :2] = 2.0
    prediction[:, 1:3, 2] = 1000.0

    observed = recall_window_memory_mse(prediction, target, spec)
    assert observed.item() == pytest.approx(4.0)


def test_recall_metric_rejects_missing_clock_channel() -> None:
    spec = RecallWindowSpec(delay=2, window=2, memory_dim=2)
    prediction = torch.zeros(1, 5, 2)
    with pytest.raises(ValueError, match="clock feature"):
        recall_window_memory_mse(prediction, prediction, spec)


def test_analytic_post_cue_noise_floor_is_frozen() -> None:
    assert delayed_recall_observation_noise_floor_mse(0.3) == pytest.approx(0.09)


def test_current_benchmark_reports_observation_stochasticity() -> None:
    benchmark = DelayedRecallBenchmark(DelayedRecallConfig())
    metrics = benchmark.metrics()
    assert metrics["latent_transition_stochastic"] is False
    assert metrics["observation_stochastic"] is True
    assert metrics["stochastic"] is True


def test_preoutcome_protocol_verifies_without_opening_outcomes() -> None:
    receipt = verifier.verify_protocol(load_protocol())
    assert receipt["status"] == "VERIFIED_PREOUTCOME_METHOD_FREEZE"
    assert receipt["execution_authorized"] is False
    assert receipt["outcome_access_allowed"] is False
    assert receipt["observation_times"] == [8, 9, 10, 11]
    assert receipt["target_indices"] == [7, 8, 9, 10]
    assert receipt["observation_noise_floor_mse"] == pytest.approx(0.09)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("execution_authorized",), True, "execution must remain unauthorized"),
        (
            ("paired_design", "identical_trajectory_bytes_per_seed_across_variants"),
            False,
            "paired trajectory identity",
        ),
        (
            ("claim_boundary", "successor_outcomes_may_rescue_v1"),
            True,
            "may not rescue",
        ),
    ],
)
def test_verifier_fails_closed_on_scientific_boundary_drift(path, value, message) -> None:
    payload = copy.deepcopy(load_protocol())
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(verifier.ProtocolError, match=message):
        verifier.verify_protocol(payload)
