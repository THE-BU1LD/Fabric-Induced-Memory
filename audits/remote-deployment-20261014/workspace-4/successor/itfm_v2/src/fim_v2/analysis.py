from __future__ import annotations


def safe_relative_gain(reference: float, candidate: float) -> float:
    if reference <= 0:
        return 0.0 if candidate >= reference else float("inf")
    return (reference - candidate) / reference


def adjudicate(
    summaries: dict[str, dict[str, float]], decision: dict[str, float]
) -> dict[str, object]:
    primary = summaries["tensor_memory"]
    base = summaries["no_memory"]
    oracle = summaries["oracle_anchor"]
    controls = [
        "no_memory",
        "sign_flip",
        "vector_broadcast",
        "recent_only",
        "last_value",
    ]
    gains = {
        name: safe_relative_gain(summaries[name]["query_nmse"], primary["query_nmse"])
        for name in controls
    }
    oracle_denominator = base["query_nmse"] - oracle["query_nmse"]
    oracle_recovery = (
        (base["query_nmse"] - primary["query_nmse"]) / oracle_denominator
        if oracle_denominator > 0
        else 0.0
    )
    primary_gain = base["query_nmse"] - primary["query_nmse"]
    erased_gain_removal = (
        (summaries["erase"]["query_nmse"] - primary["query_nmse"]) / primary_gain
        if primary_gain > 0
        else 0.0
    )
    nonquery_degradation = primary["nonquery_nmse"] - base["nonquery_nmse"]
    gates = {
        "all_nonoracle_control_gains": min(gains.values())
        >= float(decision["minimum_control_relative_gain"]),
        "oracle_gap_recovery": oracle_recovery
        >= float(decision["minimum_oracle_gap_recovery"]),
        "erasure_removes_gain": erased_gain_removal
        >= float(decision["minimum_erasure_gain_removal"]),
        "swap_direction": summaries["paired_swap"]["swap_donor_direction"]
        >= float(decision["minimum_swap_donor_direction"]),
        "retrieval_precision": primary["retrieval_anchor_precision"]
        >= float(decision["minimum_retrieval_precision"]),
        "nonquery_preserved": nonquery_degradation
        <= float(decision["maximum_nonquery_variance_degradation"]),
        "sign_flip_not_equivalent": summaries["sign_flip"]["query_nmse"]
        > primary["query_nmse"],
        "vector_not_equivalent": summaries["vector_broadcast"]["query_nmse"]
        > primary["query_nmse"],
    }
    return {
        "status": "PASS_BOUNDED_DEVELOPMENT" if all(gates.values()) else "FAIL",
        "gates": gates,
        "relative_gains": gains,
        "oracle_gap_recovery": oracle_recovery,
        "erasure_gain_removal": erased_gain_removal,
        "nonquery_nmse_degradation": nonquery_degradation,
    }
