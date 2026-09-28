from __future__ import annotations

import torch

from fim_experiments.benchmark import DelayedRecallBenchmark, DelayedRecallConfig


def test_delayed_recall_reports_observation_stochasticity_explicitly() -> None:
    benchmark = DelayedRecallBenchmark(DelayedRecallConfig())
    metrics = benchmark.metrics()

    assert benchmark.latent_transition_stochastic is False
    assert benchmark.observation_stochastic is True
    assert benchmark.stochastic is True
    assert metrics["stochastic"] is True
    assert metrics["latent_transition_stochastic"] is False
    assert metrics["observation_stochastic"] is True
    assert metrics["cue_noise"] == benchmark.config.cue_noise
    assert metrics["distractor_scale"] == benchmark.config.distractor_scale
    assert "fresh Gaussian distractor noise" in metrics["observation_noise_semantics"]


def test_delayed_recall_hidden_transition_remains_deterministic() -> None:
    benchmark = DelayedRecallBenchmark(DelayedRecallConfig())
    hidden = torch.zeros(2, benchmark.config.dimension)

    first = benchmark.step(hidden)
    second = benchmark.step(hidden)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_delayed_recall_observations_use_fresh_noise() -> None:
    benchmark = DelayedRecallBenchmark(
        DelayedRecallConfig(cue_noise=0.02, distractor_scale=0.3)
    )
    hidden = torch.zeros(2, benchmark.config.dimension)

    torch.manual_seed(20260928)
    first = benchmark.observe(hidden, t=1)
    second = benchmark.observe(hidden, t=1)

    assert not torch.equal(first[..., : benchmark.config.memory_dim], second[..., : benchmark.config.memory_dim])
    torch.testing.assert_close(first[..., -1:], second[..., -1:], rtol=0.0, atol=0.0)


def test_delayed_recall_zero_noise_reports_deterministic_observations() -> None:
    benchmark = DelayedRecallBenchmark(
        DelayedRecallConfig(cue_noise=0.0, distractor_scale=0.0)
    )
    metrics = benchmark.metrics()

    assert benchmark.latent_transition_stochastic is False
    assert benchmark.observation_stochastic is False
    assert benchmark.stochastic is False
    assert metrics["stochastic"] is False
    assert metrics["observation_stochastic"] is False
