from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def summarize_rows(rows: list[dict[str, object]], variance: float) -> dict[str, float]:
    query = [row for row in rows if bool(row["query"])]
    nonquery = [row for row in rows if not bool(row["query"])]

    def mse(selected: list[dict[str, object]]) -> float:
        return sum(float(row["squared_error"]) for row in selected) / sum(
            int(row["element_count"]) for row in selected
        )

    valid = [row for row in query if bool(row["retrieval_valid"])]
    precision = [
        row
        for row in valid
        if row["retrieval_source_anchor"] is not None and int(row["anchor_id"]) >= 0
    ]
    donor = [row for row in query if row["donor_direction_success"] is not None]
    query_mse = mse(query)
    nonquery_mse = mse(nonquery)
    return {
        "query_mse": query_mse,
        "query_nmse": query_mse / variance,
        "nonquery_mse": nonquery_mse,
        "nonquery_nmse": nonquery_mse / variance,
        "retrieval_rate": len(valid) / len(query),
        "retrieval_anchor_precision": (
            sum(
                int(row["retrieval_source_anchor"]) == int(row["anchor_id"])
                for row in precision
            )
            / len(precision)
            if precision
            else 0.0
        ),
        "swap_donor_direction": (
            sum(bool(row["donor_direction_success"]) for row in donor) / len(donor)
            if donor
            else 0.0
        ),
        "query_rows": float(len(query)),
        "nonquery_rows": float(len(nonquery)),
    }


def independently_adjudicate(
    summaries: dict[str, dict[str, float]], decision: dict[str, float]
) -> dict[str, bool]:
    primary = summaries["tensor_memory"]
    base = summaries["no_memory"]
    oracle = summaries["oracle_anchor"]
    controls = (
        "no_memory",
        "sign_flip",
        "vector_broadcast",
        "recent_only",
        "last_value",
    )
    gains = [
        (summaries[name]["query_nmse"] - primary["query_nmse"])
        / summaries[name]["query_nmse"]
        if summaries[name]["query_nmse"] > 0
        else 0.0
        for name in controls
    ]
    oracle_gap = base["query_nmse"] - oracle["query_nmse"]
    oracle_recovery = (
        (base["query_nmse"] - primary["query_nmse"]) / oracle_gap
        if oracle_gap > 0
        else 0.0
    )
    primary_gain = base["query_nmse"] - primary["query_nmse"]
    erasure_removal = (
        (summaries["erase"]["query_nmse"] - primary["query_nmse"]) / primary_gain
        if primary_gain > 0
        else 0.0
    )
    return {
        "all_nonoracle_control_gains": min(gains)
        >= float(decision["minimum_control_relative_gain"]),
        "oracle_gap_recovery": oracle_recovery
        >= float(decision["minimum_oracle_gap_recovery"]),
        "erasure_removes_gain": erasure_removal
        >= float(decision["minimum_erasure_gain_removal"]),
        "swap_direction": summaries["paired_swap"]["swap_donor_direction"]
        >= float(decision["minimum_swap_donor_direction"]),
        "retrieval_precision": primary["retrieval_anchor_precision"]
        >= float(decision["minimum_retrieval_precision"]),
        "nonquery_preserved": primary["nonquery_nmse"] - base["nonquery_nmse"]
        <= float(decision["maximum_nonquery_variance_degradation"]),
        "sign_flip_not_equivalent": summaries["sign_flip"]["query_nmse"]
        > primary["query_nmse"],
        "vector_not_equivalent": summaries["vector_broadcast"]["query_nmse"]
        > primary["query_nmse"],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run_root = root / "results/development-v2"
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("missing development manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_path = root / "protocols/development_v2.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "manifest_pass": manifest.get("status") == "PASS_BOUNDED_DEVELOPMENT",
        "no_failures": not manifest.get("failures"),
        "run_count": manifest.get("completed_runs")
        == manifest.get("expected_runs")
        == 3,
        "protocol_hash": sha256_file(protocol_path) == manifest.get("protocol_sha256"),
    }
    for source in manifest.get("source", {}).get("files", []):
        path = root / source["path"]
        checks[f"source:{source['path']}"] = (
            path.is_file() and sha256_file(path) == source["sha256"]
        )
    for run in manifest.get("runs", []):
        result_path = run_root / run["result"]
        checks[f"result:{run['seed']}"] = (
            result_path.is_file() and sha256_file(result_path) == run["result_sha256"]
        )
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        checks[f"decision:{run['seed']}"] = (
            result.get("decision", {}).get("status") == "PASS_BOUNDED_DEVELOPMENT"
        )
        recomputed: dict[str, dict[str, float]] = {}
        for relative, expected in result.get("prediction_artifacts", {}).items():
            path = run_root / relative
            checks[f"artifact:{relative}"] = (
                path.is_file() and sha256_file(path) == expected
            )
            if not path.is_file():
                continue
            mode = path.stem.removeprefix("predictions-")
            rows = json.loads(path.read_text(encoding="utf-8"))
            recomputed[mode] = summarize_rows(
                rows, float(result["normalization_variance"])
            )
        checks[f"mode_count:{run['seed']}"] = set(recomputed) == set(
            protocol["controls"]
        )
        for mode, observed in recomputed.items():
            reported = result["summaries"][mode]
            checks[f"metrics:{run['seed']}:{mode}"] = all(
                close(float(reported[key]), value) for key, value in observed.items()
            )
        independent_gates = independently_adjudicate(recomputed, protocol["decision"])
        checks[f"independent_gates:{run['seed']}"] = all(independent_gates.values())
        checks[f"reported_gates:{run['seed']}"] = independent_gates == result.get(
            "decision", {}
        ).get("gates")
        checks[f"identifiability:{run['seed']}"] = all(
            result.get("identifiability", {}).get("gates", {}).values()
        )
    receipt = {
        "schema_version": 1,
        "status": "pass" if checks and all(checks.values()) else "fail",
        "checks": checks,
    }
    output = root / "results/result_verification.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "checks": len(checks)}))
    raise SystemExit(0 if receipt["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
