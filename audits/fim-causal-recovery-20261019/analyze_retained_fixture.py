"""Recompute descriptive diagnostics from the completed fixture; never train."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository / "successor/itfm_v2/src"))
    from fim_sota.engineering import (
        load_config_file,
        preflight,
        source_identity,
        validate_completed,
    )

    if args.output.exists():
        raise FileExistsError(args.output)
    config = load_config_file(args.run_root / "config.json")
    environment = preflight(config, runtime=True)
    admission_path = args.output.with_suffix(".admission.json")
    with admission_path.open("x") as handle:
        json.dump(environment, handle, indent=2)
        handle.write("\n")
    if environment["status"] != "PASS_RUNTIME_ENVIRONMENT":
        raise RuntimeError("retained-artifact analysis was not admitted")
    completion, metrics = validate_completed(
        args.run_root, config, source_identity()["sha256"]
    )
    import torch

    torch.set_num_threads(1)
    payload = torch.load(
        args.run_root / "evaluation/predictions.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert payload["source_sha256"] == completion["source_sha256"]
    rows = payload["records"]
    test_rows = [row for row in rows if row["split"] == "test"]
    methods = {
        name: [row for row in test_rows if row["variant"] == name]
        for name in metrics["ablations"]
    }
    memory_rows = methods["memory"]
    first_event = {}
    for row in memory_rows:
        owner = row["owner"]
        first_event[owner] = min(
            first_event.get(owner, row["event_time"]), row["event_time"]
        )

    def score(group, prediction="prediction"):
        error = sum(
            float((row[prediction].double() - row["target"].double()).square().sum())
            for row in group
        )
        count = sum(row["target"].numel() for row in group)
        return {
            "squared_error_sum": error,
            "element_count": count,
            "mse": error / count,
        }

    descriptive = {}
    for name, group in methods.items():
        cold = [row for row in group if row["event_time"] == first_event[row["owner"]]]
        warm = [row for row in group if row["event_time"] > first_event[row["owner"]]]
        actual = score(group)
        expected = metrics["ablations"][name]["test"]["selected"]["mse"]
        assert math.isclose(actual["mse"], expected, rel_tol=4e-14, abs_tol=0)
        descriptive[name] = {
            "overall": actual,
            "cold": score(cold),
            "warm": score(warm),
        }
    descriptive["persistence"] = {
        "overall": score(memory_rows, "persistence_prediction"),
        "cold": score(
            [r for r in memory_rows if r["event_time"] == first_event[r["owner"]]],
            "persistence_prediction",
        ),
        "warm": score(
            [r for r in memory_rows if r["event_time"] > first_event[r["owner"]]],
            "persistence_prediction",
        ),
    }
    last_by_key = {(r["owner"], r["event_time"]): r for r in methods["last_residual"]}
    departure_squared = error_alignment_twice = 0.0
    warm_elements = 0
    for row in memory_rows:
        if row["event_time"] == first_event[row["owner"]]:
            assert row["eligible_count"] == 0
            assert torch.equal(row["prediction"], row["base_prediction"])
            continue
        last = last_by_key[(row["owner"], row["event_time"])]
        assert torch.equal(row["target"], last["target"])
        delta = row["prediction"].double() - last["prediction"].double()
        remaining_error = row["target"].double() - last["prediction"].double()
        departure_squared += float(delta.square().sum())
        error_alignment_twice += float((2 * remaining_error * delta).sum())
        warm_elements += delta.numel()
    observed_gap = (
        descriptive["memory"]["warm"]["mse"]
        - descriptive["last_residual"]["warm"]["mse"]
    )
    identity_gap = (departure_squared - error_alignment_twice) / warm_elements
    assert math.isclose(observed_gap, identity_gap, rel_tol=4e-14, abs_tol=1e-18)
    checkpoint = torch.load(
        args.run_root / "checkpoints/best.pt", map_location="cpu", weights_only=True
    )
    parameter_count = sum(
        v.numel() for v in checkpoint["model_state"].values() if torch.is_tensor(v)
    )
    output = {
        "status": "PASS_DESCRIPTIVE_RECOMPUTATION",
        "analysis_class": "POST_HOC_ENGINEERING_DIAGNOSTIC_NOT_A_NEW_EXPERIMENT",
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_root": str(args.run_root),
        "source_sha256": completion["source_sha256"],
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "predictions_sha256": metrics["predictions_sha256"],
        "primary_metric_unchanged": True,
        "independent_repetitions": 1,
        "test_owners": len(first_event),
        "metrics": descriptive,
        "memory_cold_start_fraction_of_sse": descriptive["memory"]["cold"][
            "squared_error_sum"
        ]
        / descriptive["memory"]["overall"]["squared_error_sum"],
        "warm_memory_minus_last_identity": {
            "mean_squared_prediction_departure": departure_squared / warm_elements,
            "twice_mean_error_alignment": error_alignment_twice / warm_elements,
            "difference": identity_gap,
            "observed_mse_difference": observed_gap,
        },
        "retained_record_counts": {
            "all_online_records": len(rows),
            "test_records_per_variant": len(memory_rows),
        },
        "model_tensor_element_count": parameter_count,
        "maximum_bank_entries_recorded": max(r["bank_size_after_write"] for r in rows),
        "maximum_eligible_count_recorded": max(r["eligible_count"] for r in rows),
        "capacity_stress": "NOT_TESTED: nine writes per owner, capacity sixteen",
        "cost_limits": {
            "per_query_latency": "NOT_RETAINED_BY_STREAM",
            "process_peak_memory": "NOT_MEASURED",
            "energy": "NOT_MEASURED",
        },
        "protocol_deviation": "SMOKE_PROTOCOL mentioned relative L2; the existing fixture emits MSE only. No relative-L2 result was computed or claimed.",
    }
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": output["status"],
                "source_sha256": output["source_sha256"],
                "cold_start_fraction": output["memory_cold_start_fraction_of_sse"],
                "metrics": descriptive,
                "warm_identity": output["warm_memory_minus_last_identity"],
            }
        )
    )


if __name__ == "__main__":
    main()
