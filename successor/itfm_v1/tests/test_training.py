import torch

from fim_s1.data import AliasedAdvectionCorpus
from fim_s1.metrics import summarize_rows, training_target_variance
from fim_s1.model import InterventionalTensorFabricPredictor
from fim_s1.training import (
    chronological_batches,
    evaluate,
    fit_model,
    set_deterministic_seed,
)


def test_bounded_end_to_end_training_and_evaluation() -> None:
    set_deterministic_seed(3)
    corpus = AliasedAdvectionCorpus(
        trajectory_count=6,
        time_steps=7,
        height=4,
        width=4,
        context_length=2,
        seed=5,
    )
    train = corpus.records("train")
    validation = corpus.records("validation")
    model = InterventionalTensorFabricPredictor(
        context_channels=2,
        output_channels=1,
        height=4,
        width=4,
        hidden_channels=4,
        key_channels=2,
        memory_capacity=16,
        topk=2,
        depth=1,
    )
    history = fit_model(
        model,
        train,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        device=torch.device("cpu"),
        use_memory=True,
    )
    rows = evaluate(
        model,
        validation,
        batch_size=2,
        device=torch.device("cpu"),
        use_memory=True,
    )
    summary = summarize_rows(
        rows, training_target_variance([row.target for row in train])
    )
    assert len(history) == 1
    assert summary["row_count"] == len(validation)
    assert summary["alias_row_count"] > 0
    assert summary["retrieval_regime_precision"] is not None
    assert 0.0 <= summary["retrieval_regime_precision"] <= 1.0


def test_batches_never_mix_event_times() -> None:
    corpus = AliasedAdvectionCorpus(
        trajectory_count=12,
        time_steps=10,
        height=4,
        width=4,
        context_length=2,
    )
    batches = chronological_batches(
        corpus.records("train"), batch_size=5, device=torch.device("cpu")
    )
    for batch in batches:
        assert torch.unique(batch.event_times).numel() == 1
