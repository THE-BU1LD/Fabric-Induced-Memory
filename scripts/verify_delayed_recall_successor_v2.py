#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from fim_experiments.benchmark import DelayedRecallBenchmark, DelayedRecallConfig
from fim_experiments.recall_metrics import (
    RecallWindowSpec,
    delayed_recall_observation_noise_floor_mse,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT / "research" / "protocols" / "FIM_DELAYED_RECALL_SUCCESSOR_V2.json"
)


class ProtocolError(ValueError):
    pass


def git_blob_sha(path: Path) -> str:
    raw = path.read_bytes()
    header = f"blob {len(raw)}\0".encode()
    return hashlib.sha1(header + raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def verify_protocol(payload: dict[str, Any]) -> dict[str, Any]:
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
