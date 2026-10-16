"""Small CPU runtime contract fixtures; no training or scientific outcome runs.

Adapted from the retained native acceptance packet into normal pytest discovery
and imports from the installed repository package.
"""
from __future__ import annotations

import unittest

import pytest
import torch

from fim_experiments.runtime_benchmarks import (
    DelayedRecallBenchmark, DelayedRecallConfig, LevyStableConfig, LevyStableProcess,
)

from fim_experiments.benchmark import (
    DelayedRecallBenchmark as FrozenDelayedRecallBenchmark,
    DelayedRecallConfig as FrozenDelayedRecallConfig,
    LevyStableConfig as FrozenLevyStableConfig,
    LevyStableProcess as FrozenLevyStableProcess,
)


class LevyNativeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())

    def test_default_candidate_matches_canonical_tensor_draws(self):
        torch.manual_seed(20261007)
        frozen = FrozenLevyStableProcess(FrozenLevyStableConfig())
        expected = frozen.sample_increment(128, dtype=torch.float64)
        torch.manual_seed(20261007)
        candidate = LevyStableProcess(LevyStableConfig())
        actual = candidate.sample_increment(128, dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_alpha_two_is_exactly_beta_invariant(self):
        plus = LevyStableProcess(LevyStableConfig(alpha=2.0, beta=1.0))
        minus = LevyStableProcess(LevyStableConfig(alpha=2.0, beta=-1.0))
        torch.manual_seed(41)
        plus_draw = plus.sample_increment(1024, dtype=torch.float64)
        torch.manual_seed(41)
        minus_draw = minus.sample_increment(1024, dtype=torch.float64)
        torch.testing.assert_close(plus_draw, minus_draw, rtol=0, atol=0)

    def test_step_preserves_shape_dtype_and_finiteness(self):
        process = LevyStableProcess(LevyStableConfig(dim=3))
        for dtype in (torch.float32, torch.float64):
            state = torch.zeros(7, 3, dtype=dtype)
            result = process.step(state)
            self.assertEqual(result.shape, state.shape)
            self.assertEqual(result.dtype, dtype)
            self.assertTrue(torch.isfinite(result).all())

    def test_direct_increment_rejects_nonfloating_dtype(self):
        process = LevyStableProcess(LevyStableConfig())
        with self.assertRaises(TypeError):
            process.sample_increment(2, dtype=torch.int64)

    def test_rollout_override_rejects_negative_boolean_and_noninteger_steps(self):
        process = LevyStableProcess(LevyStableConfig())
        state = process.sample_initial_state(2)
        for steps in (-1, True, 1.5):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                process.rollout(state, steps=steps)


class DelayedRecallNativeRuntimeTests(unittest.TestCase):
    def test_exact_segment_continuation_restores_hidden_time_and_rng(self):
        benchmark = DelayedRecallBenchmark(DelayedRecallConfig())
        full_generator = torch.Generator(device="cpu").manual_seed(73)
        full_hidden = benchmark.sample_initial_state(2, generator=full_generator)
        full, full_final = benchmark.rollout_segment(
            full_hidden, start_time=0, observation_count=17, generator=full_generator
        )

        split_generator = torch.Generator(device="cpu").manual_seed(73)
        split_hidden = benchmark.sample_initial_state(2, generator=split_generator)
        first, continuation = benchmark.rollout_segment(
            split_hidden, start_time=0, observation_count=6, generator=split_generator
        )
        second, split_final = benchmark.rollout_segment(
            continuation, start_time=6, observation_count=11, generator=split_generator
        )
        torch.testing.assert_close(torch.cat((first, second), dim=1), full, rtol=0, atol=0)
        torch.testing.assert_close(split_final, full_final, rtol=0, atol=0)

    def test_missing_rng_state_changes_continuation(self):
        benchmark = DelayedRecallBenchmark(DelayedRecallConfig())
        generator = torch.Generator(device="cpu").manual_seed(79)
        hidden = benchmark.sample_initial_state(1, generator=generator)
        _, continuation = benchmark.rollout_segment(
            hidden, start_time=0, observation_count=4, generator=generator
        )
        exact, _ = benchmark.rollout_segment(
            continuation, start_time=4, observation_count=3, generator=generator
        )
        wrong_generator = torch.Generator(device="cpu").manual_seed(80)
        wrong, _ = benchmark.rollout_segment(
            continuation, start_time=4, observation_count=3, generator=wrong_generator
        )
        self.assertFalse(torch.equal(exact, wrong))

    def test_batch_shift_pairing_and_clock_channel(self):
        benchmark = DelayedRecallBenchmark(DelayedRecallConfig())
        generator = torch.Generator(device="cpu").manual_seed(83)
        inputs, targets = benchmark.generate_batch(3, generator=generator)
        self.assertEqual(inputs.shape, (3, 16, 33))
        self.assertEqual(targets.shape, (3, 16, 33))
        torch.testing.assert_close(inputs[:, 1:], targets[:, :-1], rtol=0, atol=0)
        expected_clock = torch.arange(16, dtype=inputs.dtype) / 16.0
        torch.testing.assert_close(inputs[0, :, -1], expected_clock, rtol=0, atol=0)

    def test_hidden_requires_nonempty_batched_real_floating_tensor(self):
        benchmark = DelayedRecallBenchmark(DelayedRecallConfig())
        invalid = (
            torch.zeros(33),
            torch.zeros(0, 33),
            torch.zeros(1, 33, dtype=torch.int64),
            torch.zeros(1, 33, dtype=torch.complex64),
        )
        for hidden in invalid:
            with self.subTest(shape=tuple(hidden.shape), dtype=str(hidden.dtype)), self.assertRaises(
                (TypeError, ValueError)
            ):
                benchmark.step(hidden)

    def test_start_time_must_match_hidden_clock(self):
        benchmark = DelayedRecallBenchmark(DelayedRecallConfig())
        generator = torch.Generator(device="cpu").manual_seed(89)
        hidden = benchmark.sample_initial_state(1, generator=generator)
        _, continuation = benchmark.rollout_segment(
            hidden, start_time=0, observation_count=4, generator=generator
        )
        with self.assertRaises(ValueError):
            benchmark.rollout_segment(
                continuation, start_time=3, observation_count=1, generator=generator
            )


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"dimension": 33.0}, "dimension"),
        ({"memory_dim": True, "dimension": 2}, "memory_dim"),
        ({"memory_dim": 1.5, "dimension": 2.5}, "memory_dim"),
        ({"delay": True}, "delay"),
        ({"delay": 1.5}, "delay"),
        ({"steps": True}, "steps"),
        ({"steps": 16.5}, "steps"),
    ],
)
def test_delayed_config_requires_positive_integers_before_generation(values, field):
    with pytest.raises(ValueError, match=field + " must be a positive integer"):
        DelayedRecallConfig(**values)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_adjacent_observation_time_is_rejected_without_consuming_rng(dtype):
    benchmark = DelayedRecallBenchmark(DelayedRecallConfig())
    hidden = torch.zeros(1, 33, dtype=dtype)
    hidden[:, -1] = 7 / 16
    generator = torch.Generator(device="cpu").manual_seed(97)
    before = generator.get_state().clone()
    with pytest.raises(ValueError, match="start_time does not match"):
        benchmark.rollout_segment(hidden, start_time=8, observation_count=1, generator=generator)
    torch.testing.assert_close(generator.get_state(), before, rtol=0, atol=0)


def test_ambiguous_low_precision_clock_is_rejected_before_observation():
    benchmark = DelayedRecallBenchmark(DelayedRecallConfig())
    hidden = torch.zeros(1, 33, dtype=torch.bfloat16)
    hidden[:, -1] = 256 / 16
    # In bfloat16 both time 256 and 257 round to this same clock value.
    assert torch.tensor(256 / 16, dtype=hidden.dtype) == torch.tensor(257 / 16, dtype=hidden.dtype)
    with pytest.raises(ValueError, match="clock precision"):
        benchmark.rollout_segment(hidden, start_time=256, observation_count=1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_default_split_continuation_preserves_real_floating_dtype(dtype):
    benchmark = DelayedRecallBenchmark(DelayedRecallConfig())
    hidden = torch.zeros(2, 33, dtype=dtype)
    full_rng = torch.Generator(device="cpu").manual_seed(101)
    full, full_final = benchmark.rollout_segment(hidden, start_time=0, observation_count=17, generator=full_rng)
    split_rng = torch.Generator(device="cpu").manual_seed(101)
    first, continuation = benchmark.rollout_segment(hidden, start_time=0, observation_count=8, generator=split_rng)
    second, split_final = benchmark.rollout_segment(continuation, start_time=8, observation_count=9, generator=split_rng)
    assert full.dtype == dtype
    torch.testing.assert_close(torch.cat((first, second), dim=1), full, rtol=0, atol=0)
    torch.testing.assert_close(split_final, full_final, rtol=0, atol=0)


@pytest.mark.parametrize("clock", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_restored_clock_is_rejected(clock):
    benchmark = DelayedRecallBenchmark(DelayedRecallConfig())
    hidden = torch.zeros(1, 33)
    hidden[:, -1] = clock
    with pytest.raises(ValueError, match="start_time does not match"):
        benchmark.rollout_segment(hidden, start_time=0, observation_count=1)


@pytest.mark.parametrize("state", [torch.zeros(1), torch.zeros(0, 1), torch.zeros(1, 1, dtype=torch.int64)])
def test_levy_zero_step_rollout_still_validates_state(state):
    process = LevyStableProcess(LevyStableConfig())
    with pytest.raises((TypeError, ValueError)):
        process.rollout(state, steps=0)


def test_levy_zero_step_rollout_preserves_initial_state_without_rng_draws():
    process = LevyStableProcess(LevyStableConfig(dim=3))
    initial = torch.ones(2, 3, dtype=torch.float64)
    before = torch.get_rng_state().clone()
    actual = process.rollout(initial, steps=0)
    torch.testing.assert_close(actual, initial[:, None], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


@pytest.mark.parametrize("steps", [16, 20])
def test_default_recall_tensors_match_frozen_source_at_declared_horizons(steps):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(107)
        frozen = FrozenDelayedRecallBenchmark(FrozenDelayedRecallConfig())
        expected_inputs, expected_targets = frozen.generate_batch(2, steps=steps)
        torch.manual_seed(107)
        candidate = DelayedRecallBenchmark(DelayedRecallConfig())
        actual_inputs, actual_targets = candidate.generate_batch(2, steps=steps)
    torch.testing.assert_close(actual_inputs, expected_inputs, rtol=0, atol=0)
    torch.testing.assert_close(actual_targets, expected_targets, rtol=0, atol=0)
