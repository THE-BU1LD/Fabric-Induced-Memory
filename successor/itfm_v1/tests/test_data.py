import torch

from fim_s1.data import AliasedAdvectionCorpus, corrupt_records


def test_family_split_is_disjoint_and_aliases_have_different_futures() -> None:
    corpus = AliasedAdvectionCorpus(
        trajectory_count=12,
        time_steps=12,
        height=8,
        width=8,
        context_length=3,
        seed=17,
    )
    counts = corpus.validate()
    assert counts["train_aliases"] > 0
    assert counts["train_alias_current_max_abs_difference"] == 0.0
    assert counts["train_alias_future_min_mse"] > 0.0
    assert not (corpus.split_families["train"] & corpus.split_families["test"])

    records = corpus.records("train")
    grouped = {(row.family_id, row.event_time, row.regime): row for row in records}
    matched = 0
    for row in records:
        if not row.aliased or row.regime != -1:
            continue
        partner = grouped[(row.family_id, row.event_time, 1)]
        torch.testing.assert_close(
            row.context[-1], partner.context[-1], rtol=0.0, atol=0.0
        )
        assert not torch.equal(row.target, partner.target)
        matched += 1
    assert matched > 0


def test_records_are_chronological_and_owner_stable() -> None:
    corpus = AliasedAdvectionCorpus(
        trajectory_count=10,
        time_steps=10,
        height=4,
        width=4,
        context_length=2,
    )
    records = corpus.records("validation")
    observed_order = [(row.event_time, row.owner_id) for row in records]
    assert observed_order == sorted(observed_order)
    for row in records:
        assert row.owner_id // 2 == row.family_id


def test_identity_is_deterministic_and_changes_with_generation_contract() -> None:
    common = dict(
        trajectory_count=10,
        time_steps=10,
        height=4,
        width=4,
        context_length=2,
        seed=29,
    )
    first = AliasedAdvectionCorpus(**common)
    second = AliasedAdvectionCorpus(**common)
    shifted = AliasedAdvectionCorpus(**common, speed=2)
    assert first.identity() == second.identity()
    assert first.identity()["sha256"] != shifted.identity()["sha256"]


def test_corruption_is_reproducible_and_does_not_mutate_targets() -> None:
    corpus = AliasedAdvectionCorpus(
        trajectory_count=10,
        time_steps=10,
        height=4,
        width=4,
        context_length=2,
        seed=31,
    )
    records = corpus.records("validation")
    first = corrupt_records(records, noise_std=0.1, missing_fraction=0.25, seed=7)
    second = corrupt_records(records, noise_std=0.1, missing_fraction=0.25, seed=7)
    assert len(first) == len(records)
    for original, left, right in zip(records, first, second, strict=True):
        torch.testing.assert_close(left.context, right.context)
        torch.testing.assert_close(left.target, original.target)
        assert left.context.data_ptr() != original.context.data_ptr()
