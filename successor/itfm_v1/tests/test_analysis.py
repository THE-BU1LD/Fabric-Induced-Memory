from fim_s1.analysis import exact_sign_flip_p, holm_adjust


def test_exact_sign_flip_is_two_sided_and_preserves_small_sample_limit() -> None:
    assert exact_sign_flip_p([]) == 1.0
    assert exact_sign_flip_p([1.0, 1.0, 1.0]) == 0.25


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == 0.03
    assert adjusted["c"] == 0.06
    assert adjusted["b"] == 0.06
