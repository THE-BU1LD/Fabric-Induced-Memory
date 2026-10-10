"""Metadata-only regressions for the complete delayed-recall method freeze.

Backend doubles deliberately isolate admission logic. These fixtures never
generate trajectories, train a model, open outcomes, or qualify tensor behavior.
The unchanged source-file hashes are still checked by the real verifier.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/protocols/FIM_DELAYED_RECALL_SUCCESSOR_V2.json"
VERIFIER = Path(
    os.environ.get(
        "FIM_PROTOCOL_VERIFIER_PATH",
        str(ROOT / "scripts/verify_delayed_recall_successor_v2.py"),
    )
)

# These mutations used to receive VERIFIED_PREOUTCOME_METHOD_FREEZE. Each changes
# a named method requirement or exploits a coercion of a frozen count/flag.
METHOD_DRIFT_CASES = (
    (("source", "base_main_commit"), "UNKNOWN"),
    (("benchmark", "name"), "different_benchmark"),
    (("benchmark", "dimension"), 34),
    (("benchmark", "dimension"), 33.75),
    (("benchmark", "delay"), 8.75),
    (("benchmark", "steps"), 16.75),
    (("benchmark", "cue_noise"), float("nan")),
    (("benchmark", "latent_transition_stochastic"), True),
    (("benchmark", "observation_stochastic"), False),
    (("endpoint", "primary"), "best_looking_endpoint"),
    (("endpoint", "aggregation"), "minimum_over_time"),
    (("endpoint", "secondary"), []),
    (("endpoint", "window_length"), 4.75),
    (("observation_noise_reference", "type"), "trainable_model"),
    (("observation_noise_reference", "scope"), "all_features"),
    (("observation_noise_reference", "formula"), "arbitrary_constant"),
    (("paired_design", "memory_reset_between_independent_episodes"), False),
    (("paired_design", "no_best_checkpoint_selection"), False),
    (("paired_design", "batch_size"), True),
    (("budget", "epochs"), 1000),
    (("budget", "dataset_size"), 0),
    (("budget", "batch_size"), 64),
    (("budget", "train_rollout_steps"), 1),
    (("budget", "eval_rollout_steps"), 999),
    (("budget", "teacher_forcing_ratio"), False),
    (("budget", "horizon_decay"), 0),
    (("budget", "dynamic_data"), True),
    (("budget", "use_ema"), True),
    (("budget", "evaluate_ema"), True),
    (("analysis", "primary_comparison"), "full_vs_weakest_arm"),
    (("analysis", "mechanism_comparison"), "full_vs_weakest_arm"),
    (("analysis", "report_raw_per_seed_values"), False),
    (("analysis", "paired_bootstrap_samples"), 0),
    (("analysis", "paired_bootstrap_seed"), 42),
    (("analysis", "exact_sign_flip_descriptive"), False),
    (("analysis", "claim_rule"), "Always claim a memory benefit."),
    (("analysis", "failure_rule"), "Discard failed cells."),
    (("claim_boundary", "negative_or_mixed_result_must_be_retained"), False),
)


def set_value(payload: dict, path: tuple[str, ...], value: object) -> None:
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def backend_doubles() -> dict[str, types.ModuleType]:
    """Provide deterministic metadata only, with no tensor backend."""
    benchmark = types.ModuleType("fim_experiments.benchmark")

    class Config:
        def __init__(self, **fields: object) -> None:
            self.__dict__.update(fields)

    class Benchmark:
        def __init__(self, config: Config) -> None:
            self.config = config

        def metrics(self) -> dict:
            stochastic = self.config.cue_noise > 0 or self.config.distractor_scale > 0
            return {
                "name": "DelayedRecallBenchmark",
                "latent_transition_stochastic": False,
                "observation_stochastic": stochastic,
                "stochastic": stochastic,
            }

    benchmark.DelayedRecallConfig = Config
    benchmark.DelayedRecallBenchmark = Benchmark
    metric = types.ModuleType("fim_experiments.recall_metrics")

    class Window:
        def __init__(self, delay: int, window: int, memory_dim: int) -> None:
            if min(delay, window, memory_dim) < 1:
                raise ValueError("invalid fixture window")
            self.delay = delay
            self.window = window

        @property
        def observation_times(self) -> tuple[int, ...]:
            return tuple(range(self.delay, self.delay + self.window))

        def target_indices(self, target_steps: int) -> tuple[int, ...]:
            if self.delay - 1 + self.window > target_steps:
                raise ValueError("fixture target sequence too short")
            return tuple(range(self.delay - 1, self.delay - 1 + self.window))

    def noise_floor(scale: float) -> float:
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("invalid fixture noise")
        return scale * scale

    metric.RecallWindowSpec = Window
    metric.delayed_recall_observation_noise_floor_mse = noise_floor
    return {
        "fim_experiments.benchmark": benchmark,
        "fim_experiments.recall_metrics": metric,
    }


class FrozenMethodAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch.dict(sys.modules, backend_doubles()))
        spec = importlib.util.spec_from_file_location("frozen_method_verifier", VERIFIER)
        assert spec is not None and spec.loader is not None
        self.verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.verifier)
        # A separately retained before-repair script uses the same exact files.
        self.verifier.ROOT = ROOT
        self.payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    def test_unchanged_method_passes_with_both_holds_closed(self) -> None:
        receipt = self.verifier.verify_protocol(self.payload)
        self.assertEqual(receipt["status"], "VERIFIED_PREOUTCOME_METHOD_FREEZE")
        self.assertIs(receipt["execution_authorized"], False)
        self.assertIs(receipt["outcome_access_allowed"], False)
        self.assertEqual(receipt["observation_times"], [8, 9, 10, 11])
        self.assertEqual(receipt["target_indices"], [7, 8, 9, 10])

    def test_named_method_drift_is_rejected(self) -> None:
        for path, value in METHOD_DRIFT_CASES:
            with self.subTest(path=".".join(path), value=value):
                candidate = copy.deepcopy(self.payload)
                set_value(candidate, path, value)
                with self.assertRaises(self.verifier.ProtocolError):
                    self.verifier.verify_protocol(candidate)

    def test_integer_counts_reject_coerced_values(self) -> None:
        for path in (
            ("benchmark", "dimension"),
            ("benchmark", "memory_dim"),
            ("benchmark", "delay"),
            ("benchmark", "steps"),
            ("endpoint", "window_length"),
            ("paired_design", "batch_size"),
            ("budget", "epochs"),
            ("budget", "dataset_size"),
            ("budget", "batch_size"),
            ("budget", "train_rollout_steps"),
            ("budget", "eval_rollout_steps"),
            ("analysis", "paired_bootstrap_samples"),
            ("analysis", "paired_bootstrap_seed"),
        ):
            original = self.payload[path[0]][path[1]]
            for value in (str(original), float(original), True):
                with self.subTest(path=".".join(path), value=value):
                    candidate = copy.deepcopy(self.payload)
                    set_value(candidate, path, value)
                    with self.assertRaises(self.verifier.ProtocolError):
                        self.verifier.verify_protocol(candidate)

    def test_coherent_rewrites_cannot_redefine_the_frozen_method(self) -> None:
        # These internally consistent rewrites pass the historical checks.
        # A frozen method must reject them even when its derived fields agree.
        rewrites = (
            {
                ("benchmark", "delay"): 9,
                ("endpoint", "observation_times"): [9, 10, 11, 12],
                ("endpoint", "target_indices"): [8, 9, 10, 11],
            },
            {
                ("benchmark", "distractor_scale"): 0.4,
                ("observation_noise_reference", "expected_mse_per_memory_feature"): 0.4 ** 2,
            },
            {
                ("benchmark", "memory_dim"): 16,
                ("benchmark", "dimension"): 17,
            },
        )
        for index, rewrite in enumerate(rewrites):
            with self.subTest(rewrite=index):
                candidate = copy.deepcopy(self.payload)
                for path, value in rewrite.items():
                    set_value(candidate, path, value)
                with self.assertRaises(self.verifier.ProtocolError):
                    self.verifier.verify_protocol(candidate)

    def test_real_settings_allow_equal_numeric_representations(self) -> None:
        self.payload["budget"]["horizon_decay"] = 1.0
        self.payload["observation_noise_reference"]["expected_mse_per_memory_feature"] = 9e-2
        self.verifier.verify_protocol(self.payload)

    def test_real_settings_reject_nonfinite_or_nonnumeric_values(self) -> None:
        for path in (
            ("benchmark", "cue_noise"),
            ("benchmark", "distractor_scale"),
            ("benchmark", "reveal_sharpness"),
            ("budget", "teacher_forcing_ratio"),
            ("budget", "horizon_decay"),
            ("observation_noise_reference", "expected_mse_per_memory_feature"),
        ):
            for value in (float("nan"), float("inf"), -float("inf"), "0.3", True):
                with self.subTest(path=".".join(path), value=value):
                    candidate = copy.deepcopy(self.payload)
                    set_value(candidate, path, value)
                    with self.assertRaises(self.verifier.ProtocolError):
                        self.verifier.verify_protocol(candidate)

    def test_missing_or_malformed_sections_raise_protocol_error(self) -> None:
        for key in (
            "source", "benchmark", "endpoint", "observation_noise_reference",
            "paired_design", "budget", "analysis", "claim_boundary",
        ):
            for value in (None, [], "invalid", {}):
                with self.subTest(section=key, value=value):
                    candidate = copy.deepcopy(self.payload)
                    candidate[key] = value
                    with self.assertRaises(self.verifier.ProtocolError):
                        self.verifier.verify_protocol(candidate)

    def test_missing_unused_analysis_section_is_rejected(self) -> None:
        del self.payload["analysis"]
        with self.assertRaises(self.verifier.ProtocolError):
            self.verifier.verify_protocol(self.payload)

    def test_unknown_fields_do_not_silently_extend_the_frozen_method(self) -> None:
        for section in (None, "benchmark", "budget", "analysis"):
            with self.subTest(section=section):
                candidate = copy.deepcopy(self.payload)
                target = candidate if section is None else candidate[section]
                target["select_best_seed"] = True
                with self.assertRaises(self.verifier.ProtocolError):
                    self.verifier.verify_protocol(candidate)

    def test_root_and_field_types_fail_with_protocol_error(self) -> None:
        for payload in (None, [], "invalid", 1, True):
            with self.subTest(payload=payload):
                with self.assertRaises(self.verifier.ProtocolError):
                    self.verifier.verify_protocol(payload)

    def test_key_order_is_not_part_of_the_method(self) -> None:
        reordered = dict(reversed(list(self.payload.items())))
        for key, value in reordered.items():
            if isinstance(value, dict):
                reordered[key] = dict(reversed(list(value.items())))
        self.verifier.verify_protocol(reordered)

    def test_existing_boundary_diagnostics_remain_specific(self) -> None:
        for path, value, message in (
            (("execution_authorized",), True, "execution must remain unauthorized"),
            (
                ("paired_design", "identical_trajectory_bytes_per_seed_across_variants"),
                False, "paired trajectory identity",
            ),
            (("claim_boundary", "successor_outcomes_may_rescue_v1"), True, "may not rescue"),
        ):
            with self.subTest(path=path):
                candidate = copy.deepcopy(self.payload)
                set_value(candidate, path, value)
                with self.assertRaisesRegex(self.verifier.ProtocolError, message):
                    self.verifier.verify_protocol(candidate)

    def test_invalid_method_is_rejected_before_backend_import(self) -> None:
        self.payload["budget"]["epochs"] = 1000
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "drifted-protocol.json"
            path.write_text(json.dumps(self.payload), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(VERIFIER), "--protocol", str(path)],
                text=True, capture_output=True, check=False, cwd=ROOT,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("VERIFIED_PREOUTCOME_METHOD_FREEZE", result.stdout)
        self.assertIn("budget.epochs", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
