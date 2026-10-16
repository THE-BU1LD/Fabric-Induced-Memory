from fim_v2.analysis import adjudicate


def summary(query: float, nonquery: float = 0.1, **extra):
    payload = {
        "query_nmse": query,
        "nonquery_nmse": nonquery,
        "retrieval_anchor_precision": 0.0,
        "swap_donor_direction": 0.0,
    }
    payload.update(extra)
    return payload


def decision_config():
    return {
        "minimum_control_relative_gain": 0.05,
        "minimum_oracle_gap_recovery": 0.50,
        "minimum_erasure_gain_removal": 0.80,
        "minimum_swap_donor_direction": 0.90,
        "minimum_retrieval_precision": 0.99,
        "maximum_nonquery_variance_degradation": 0.01,
    }


def passing_summaries():
    return {
        "tensor_memory": summary(0.05, retrieval_anchor_precision=1.0),
        "no_memory": summary(1.0),
        "erase": summary(1.0),
        "sign_flip": summary(3.0),
        "vector_broadcast": summary(1.0),
        "recent_only": summary(1.0),
        "last_value": summary(1.0),
        "paired_swap": summary(3.0, swap_donor_direction=1.0),
        "oracle_anchor": summary(0.0),
    }


def test_joint_rule_passes_only_when_all_gates_pass() -> None:
    result = adjudicate(passing_summaries(), decision_config())
    assert result["status"] == "PASS_BOUNDED_DEVELOPMENT"
    assert all(result["gates"].values())


def test_joint_rule_fails_when_intervention_is_not_directional() -> None:
    summaries = passing_summaries()
    summaries["paired_swap"]["swap_donor_direction"] = 0.5
    result = adjudicate(summaries, decision_config())
    assert result["status"] == "FAIL"
    assert not result["gates"]["swap_direction"]
