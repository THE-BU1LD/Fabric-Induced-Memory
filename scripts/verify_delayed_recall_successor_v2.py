#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT / "research" / "protocols" / "FIM_DELAYED_RECALL_SUCCESSOR_V2.json"
)

# Named values from the existing frozen contract, Git blob
# 61e954763626eea5d82c56ac38981215919dfc73. Do not derive this expectation from
# the candidate being verified: mutually consistent edits must still fail.
# A changed scientific method requires a separately versioned protocol/review.
FROZEN_CONTRACT: dict[str, Any] = {'schema_version': 1,
 'protocol_name': 'FIM_DELAYED_RECALL_SUCCESSOR_V2',
 'status': 'FROZEN_METHOD_PREOUTCOME_REVIEW_REQUIRED',
 'execution_authorized': False,
 'outcome_access_allowed': False,
 'source': {'base_main_commit': 'ae6d8a9622e8b28615715e8b68ca04b2a6188653',
            'delayed_recall_benchmark_git_blob': 'c91a6afe9d86830c6bcfaa2d9187547d9688050d',
            'recall_metrics_git_blob': '7d7e0d1c3daa4d5f7f4d2e02792fa363e1826c84'},
 'benchmark': {'name': 'delayed_recall',
               'dimension': 33,
               'memory_dim': 32,
               'delay': 8,
               'steps': 16,
               'cue_noise': 0.02,
               'distractor_scale': 0.3,
               'reveal_sharpness': 0.35,
               'latent_transition_stochastic': False,
               'observation_stochastic': True},
 'endpoint': {'primary': 'recall_window_memory_mse',
              'observation_times': [8, 9, 10, 11],
              'target_indices': [7, 8, 9, 10],
              'window_length': 4,
              'channels': 'memory_payload_only_exclude_clock',
              'aggregation': 'mean_over_batch_time_memory_features',
              'secondary': ['rollout_mse', 'final_step_mse', 'rollout_mae', 'final_step_mae']},
 'observation_noise_reference': {'type': 'analytic_conditional_mean_floor',
                                 'scope': 'post_cue_memory_features',
                                 'expected_mse_per_memory_feature': 0.09,
                                 'formula': 'distractor_scale_squared',
                                 'is_trainable_baseline': False},
 'paired_design': {'seeds': [607, 701, 809, 907, 1009],
                   'variants': ['full', 'no_memory', 'no_retrieval', 'no_salience_gating'],
                   'identical_trajectory_bytes_per_seed_across_variants': True,
                   'generate_and_hash_trajectory_artifact_before_variant_execution': True,
                   'memory_reset_between_independent_episodes': True,
                   'batch_size': 1,
                   'no_best_checkpoint_selection': True},
 'budget': {'epochs': 4,
            'dataset_size': 32,
            'batch_size': 1,
            'train_rollout_steps': 16,
            'eval_rollout_steps': 20,
            'teacher_forcing_ratio': 0.5,
            'horizon_decay': 1,
            'dynamic_data': False,
            'use_ema': False,
            'evaluate_ema': False},
 'analysis': {'primary_comparison': 'full_vs_no_memory',
              'mechanism_comparison': 'full_vs_no_retrieval',
              'report_raw_per_seed_values': True,
              'paired_bootstrap_samples': 10000,
              'paired_bootstrap_seed': 20261005,
              'exact_sign_flip_descriptive': True,
              'claim_rule': 'No memory-benefit claim unless the paired mean primary delta favors '
                            'full and full wins at least 4 of 5 frozen seeds; retrieval-specific '
                            'claims additionally require the same directional rule versus '
                            'no_retrieval.',
              'failure_rule': 'Retain every failed cell; do not replace seeds, extend budgets, '
                              'move the recall window, or tune the mechanism after outcome '
                              'access.'},
 'claim_boundary': {'rewrites_frozen_v1': False,
                    'historical_v1_primary_remains_whole_rollout_mse': True,
                    'successor_outcomes_may_rescue_v1': False,
                    'negative_or_mixed_result_must_be_retained': True}}

# This real-valued setting happens to be encoded as an integer in the JSON.
REAL_FIELDS = {"budget.horizon_decay"}
BOUNDARY_ERRORS = {
    "execution_authorized": "execution must remain unauthorized",
    "outcome_access_allowed": "outcome access must remain closed",
    "paired_design.identical_trajectory_bytes_per_seed_across_variants": (
        "paired trajectory identity must be required"
    ),
    "claim_boundary.successor_outcomes_may_rescue_v1": (
        "successor outcomes may not rescue the frozen v1 result"
    ),
}


class ProtocolError(ValueError):
    pass


def git_blob_sha(path: Path) -> str:
    raw = path.read_bytes()
    header = f"blob {len(raw)}\0".encode()
    return hashlib.sha1(header + raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _verify_frozen_value(actual: Any, expected: Any, path: str) -> None:
    label = path or "protocol"
    message = BOUNDARY_ERRORS.get(path, f"{label} differs from the frozen method")

    if isinstance(expected, dict):
        require(type(actual) is dict, f"{label} must be an object")
        require(
            all(type(key) is str for key in actual),
            f"{label} field names must be strings",
        )
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        require(not missing, f"{label} is missing frozen fields: {missing}")
        require(not extra, f"{label} has unknown fields: {extra}")
        for key, value in expected.items():
            child = f"{path}.{key}" if path else key
            _verify_frozen_value(actual[key], value, child)
        return

    if isinstance(expected, list):
        require(type(actual) is list, f"{label} must be a list")
        require(len(actual) == len(expected), message)
        for index, value in enumerate(expected):
            _verify_frozen_value(actual[index], value, f"{label}[{index}]")
        return

    if type(expected) is float or path in REAL_FIELDS:
        # Counts are strict integers elsewhere; real parameters admit 1 and 1.0
        # equally, but never Boolean/string coercions or NaN/Infinity.
        require(
            type(actual) in (int, float)
            and actual == expected
            and math.isfinite(actual),
            message,
        )
        return

    require(type(actual) is type(expected) and actual == expected, message)


def verify_frozen_contract(payload: dict[str, Any]) -> None:
    """Check every declared method field before loading an execution backend."""
    _verify_frozen_value(payload, FROZEN_CONTRACT, "")


def verify_protocol(payload: dict[str, Any]) -> dict[str, Any]:
    verify_frozen_contract(payload)

    from fim_experiments.benchmark import DelayedRecallBenchmark, DelayedRecallConfig
    from fim_experiments.recall_metrics import (
        RecallWindowSpec,
        delayed_recall_observation_noise_floor_mse,
    )

    require(payload.get("schema_version") == 1, "unexpected schema_version")
    require(
        payload.get("protocol_name") == "FIM_DELAYED_RECALL_SUCCESSOR_V2",
        "unexpected protocol_name",
    )
    require(
        payload.get("status") == "FROZEN_METHOD_PREOUTCOME_REVIEW_REQUIRED",
        "protocol status is not the frozen pre-outcome review state",
    )
    require(payload.get("execution_authorized") is False, "execution must remain unauthorized")
    require(payload.get("outcome_access_allowed") is False, "outcome access must remain closed")

    source = payload.get("source", {})
    benchmark_path = ROOT / "fim_experiments" / "benchmark.py"
    metric_path = ROOT / "fim_experiments" / "recall_metrics.py"
    require(
        git_blob_sha(benchmark_path) == source.get("delayed_recall_benchmark_git_blob"),
        "DelayedRecallBenchmark source identity differs from the frozen contract",
    )
    require(
        git_blob_sha(metric_path) == source.get("recall_metrics_git_blob"),
        "recall endpoint source identity differs from the frozen contract",
    )

    frozen = payload.get("benchmark", {})
    config = DelayedRecallConfig(
        dimension=int(frozen["dimension"]),
        memory_dim=int(frozen["memory_dim"]),
        delay=int(frozen["delay"]),
        steps=int(frozen["steps"]),
        cue_noise=float(frozen["cue_noise"]),
        distractor_scale=float(frozen["distractor_scale"]),
        reveal_sharpness=float(frozen["reveal_sharpness"]),
    )
    benchmark = DelayedRecallBenchmark(config)
    metrics = benchmark.metrics()
    require(metrics["name"] == "DelayedRecallBenchmark", "unexpected benchmark implementation")
    require(metrics["latent_transition_stochastic"] is False, "latent transition drifted")
    require(metrics["observation_stochastic"] is True, "observation stochasticity drifted")
    require(metrics["stochastic"] is True, "aggregate stochastic metadata drifted")

    endpoint = payload.get("endpoint", {})
    spec = RecallWindowSpec(
        delay=int(frozen["delay"]),
        window=int(endpoint["window_length"]),
        memory_dim=int(frozen["memory_dim"]),
    )
    expected_times = list(spec.observation_times)
    expected_indices = list(spec.target_indices(int(payload["budget"]["eval_rollout_steps"])))
    require(endpoint.get("observation_times") == expected_times, "recall observation window drifted")
    require(endpoint.get("target_indices") == expected_indices, "recall target indexing drifted")
    require(
        endpoint.get("channels") == "memory_payload_only_exclude_clock",
        "primary endpoint channel scope drifted",
    )

    noise = payload.get("observation_noise_reference", {})
    expected_floor = delayed_recall_observation_noise_floor_mse(config.distractor_scale)
    require(
        abs(float(noise.get("expected_mse_per_memory_feature")) - expected_floor) <= 1e-15,
        "analytic observation-noise reference drifted",
    )
    require(noise.get("is_trainable_baseline") is False, "noise floor must remain diagnostic only")

    design = payload.get("paired_design", {})
    require(
        design.get("seeds") == [607, 701, 809, 907, 1009],
        "fresh successor seed set drifted",
    )
    require(
        design.get("variants")
        == ["full", "no_memory", "no_retrieval", "no_salience_gating"],
        "variant matrix drifted",
    )
    require(
        design.get("identical_trajectory_bytes_per_seed_across_variants") is True,
        "paired trajectory identity must be required",
    )
    require(
        design.get("generate_and_hash_trajectory_artifact_before_variant_execution") is True,
        "trajectory pre-materialization/hash gate must remain enabled",
    )
    require(design.get("batch_size") == 1, "trajectory isolation requires batch_size=1")

    claim = payload.get("claim_boundary", {})
    require(claim.get("rewrites_frozen_v1") is False, "successor may not rewrite v1")
    require(
        claim.get("historical_v1_primary_remains_whole_rollout_mse") is True,
        "v1 primary endpoint boundary drifted",
    )
    require(
        claim.get("successor_outcomes_may_rescue_v1") is False,
        "successor outcomes may not rescue the frozen v1 result",
    )

    return {
        "status": "VERIFIED_PREOUTCOME_METHOD_FREEZE",
        "protocol_name": payload["protocol_name"],
        "execution_authorized": False,
        "outcome_access_allowed": False,
        "observation_times": expected_times,
        "target_indices": expected_indices,
        "observation_noise_floor_mse": expected_floor,
        "benchmark_git_blob": git_blob_sha(benchmark_path),
        "recall_metrics_git_blob": git_blob_sha(metric_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    payload = json.loads(args.protocol.read_text(encoding="utf-8"))
    print(json.dumps(verify_protocol(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
