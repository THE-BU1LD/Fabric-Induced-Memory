import torch

from fim_v2.data import EpisodicAliasCorpus


def small_corpus(seed: int = 7) -> EpisodicAliasCorpus:
    return EpisodicAliasCorpus(
        family_count=8,
        anchor_count=2,
        query_repeats=1,
        context_length=3,
        height=8,
        width=8,
        washout_events=5,
        seed=seed,
    )


def test_complete_context_is_aliased_but_future_diverges() -> None:
    corpus = small_corpus()
    report = corpus.validate()
    assert report["test_full_context_max_abs_difference"] == 0.0
    assert report["test_paired_future_min_mse"] > 0.0
    assert report["test_minimum_calibration_age"] > corpus.context_length


def test_family_splits_are_disjoint_and_complete() -> None:
    corpus = small_corpus()
    train = corpus.split_families["train"]
    validation = corpus.split_families["validation"]
    test = corpus.split_families["test"]
    assert not (train & validation or train & test or validation & test)
    assert train | validation | test == set(range(corpus.family_count))


def test_frozen_base_is_conditional_pair_mean() -> None:
    corpus = small_corpus()
    rows = corpus.records("train")
    index = {(row.family_id, row.event_time, row.regime): row for row in rows}
    for row in rows:
        if row.regime != -1 or row.anchor_id < 0:
            continue
        counterpart = index[(row.family_id, row.event_time, 1)]
        pair_mean = 0.5 * (row.target + counterpart.target)
        torch.testing.assert_close(
            pair_mean, corpus.base_prediction(row, corpus.channels)
        )
