from fim_v2.data import EpisodicAliasCorpus
from fim_v2.evaluation import calibrate_alpha, evaluate_stream, summarize


def test_end_to_end_memory_reuses_old_spatial_innovation() -> None:
    corpus = EpisodicAliasCorpus(
        family_count=8,
        anchor_count=2,
        query_repeats=1,
        context_length=3,
        height=8,
        width=8,
        washout_events=5,
        seed=17,
    )
    corpus.validate()
    config = {
        "capacity": 32,
        "minimum_age": 4,
        "topk": 1,
        "temperature": 0.05,
        "minimum_similarity": 0.999999,
    }
    validation = corpus.records("validation")
    raw = evaluate_stream(validation, corpus, config, mode="tensor_memory", alpha=1.0)
    alpha = calibrate_alpha(raw)
    assert alpha > 0.99
    primary = summarize(raw, normalization_variance=1.0)
    base = summarize(
        evaluate_stream(validation, corpus, config, mode="no_memory", alpha=0.0),
        normalization_variance=1.0,
    )
    recent = summarize(
        evaluate_stream(validation, corpus, config, mode="recent_only", alpha=alpha),
        normalization_variance=1.0,
    )
    assert primary["query_mse"] < 1e-10
    assert base["query_mse"] > 0.1
    assert recent["query_mse"] == base["query_mse"]
    assert primary["retrieval_anchor_precision"] == 1.0
