"""Stdlib unit fixtures only. Fabricated fixture values are never run evidence."""

import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


spec = importlib.util.spec_from_file_location("tonight_analysis", Path(__file__).with_name("analysis.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read_csv(path):
    return list(csv.DictReader(io.StringIO(path.read_text())))


def configuration(seed=101, **changes):
    return dict(seed=seed, context_steps=2, sequence_steps=4, channels=1,
                spatial_points=2, train_trajectories=1, validation_trajectories=1,
                test_trajectories=1, hidden=3, device="cpu", precision="fp32", **changes)


def manifest(*seeds):
    return {"source_sha256": "a" * 64, "study_kind": "DEVELOPMENT",
            "cells": [{"id": f"seed-{seed}", "config": configuration(seed), "mandatory": True} for seed in seeds]}


def metric(value):
    return {"element_count": 4, "squared_error_sum": value * 4, "mse": value}


def fixture(root, cell, memory_mse=0.6):
    """TEST ONLY inert artifacts; the production validator is explicitly mocked."""
    directory = root / "cells" / cell["id"]
    receipt = {"source_sha256": "a" * 64, "config_sha256": analysis._digest(cell["config"]), "data_sha256": "d" * 64}
    values = {"base": .5, "memory": memory_mse, "erased": .5, "spatial_mean": .55, "last_residual": .4}
    metrics = {**receipt, "status": "PASS_ENGINEERING", "evidence_class": "SYNTHETIC_ENGINEERING_FIXTURE",
               "scientific_claim": "NOT_EVALUATED", "raw_data_sha256": "e" * 64,
               "alpha_grid": [0, .25, .5, 1], "ablations": {}}
    for variant, value in values.items():
        block = {"alpha": .5, "selected": metric(value), "base": metric(.5), "persistence": metric(.25)}
        metrics["ablations"][variant] = {"alpha": .5, "test": copy.deepcopy(block), "validation": copy.deepcopy(block)}
    metrics["autonomous_base_diagnostic"] = {"overall": metric(.8), "all_predictions_finite": True,
        "memory": "disabled_no_unobserved_residual_writes", "per_horizon": {"1": metric(.7), "2": metric(.9)}}
    for key, name in (("checkpoint_sha256", "checkpoints/best.pt"), ("predictions_sha256", "evaluation/predictions.pt"),
                      ("raw_data_artifact_sha256", "evaluation/raw_data.pt")):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"UNIT TEST INERT BYTES: not a tensor checkpoint or scientific result")
        metrics[key] = analysis._file_hash(path)
    write(directory / "training_result.json", {"status": "PASS_ENGINEERING", "raw_data_sha256": "e" * 64,
          "elapsed_seconds": 2.5, "steps_completed": 2, "examples_seen": 2, "epochs_completed": 1})
    (directory / "events.jsonl").write_text(json.dumps({"event": "train_step", "cuda_peak_bytes": None}) + "\n")
    (directory / "evaluation/events.jsonl").write_text(json.dumps({"event": "evaluation_complete", "elapsed_seconds": 1.5}) + "\n")
    return receipt, metrics


class AnalysisTests(unittest.TestCase):
    def test_no_tensor_import_and_empty_ledger_exports_no_fake_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(analysis, "_validator", side_effect=AssertionError("no completed cell may be read")):
                result = analysis.run(root, manifest(101, 102), root / "analysis")
            self.assertEqual(result["end_state"], "BLOCKED BEFORE EXECUTION")
            self.assertEqual(result["valid_completed_cells"], 0)
            self.assertEqual(result["mandatory_missing"], ["seed-101", "seed-102"])
            self.assertFalse(result["complete_confirmatory_study"])
            self.assertEqual(read_csv(root / "analysis/per_seed_metrics.csv"), [])
            self.assertEqual(len(read_csv(root / "analysis/cell_status.csv")), 2)
            self.assertFalse(list((root / "analysis").glob("*.svg")))
            self.assertNotIn("torch", sys.modules)

    def test_single_seed_complete_is_pilot_only_and_baseline_win_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = manifest(101)
            verified = fixture(root, protocol["cells"][0])
            write(root / "run_ledger.json", {"cells": {"seed-101": {"state": "SUCCEEDED"}}})
            with patch.object(analysis, "_validator", return_value=verified) as validator:
                result = analysis.run(root, protocol, root / "analysis")
            validator.assert_called_once_with((root / "cells/seed-101").resolve(), protocol["cells"][0]["config"], "a" * 64)
            self.assertEqual(result["end_state"], "COMPLETE ENGINEERING/PILOT VALIDATION ONLY")
            self.assertFalse(result["complete_confirmatory_study"])
            for summary in result["paired_aggregates"]:
                self.assertEqual(summary["n_seeds"], 1)
                self.assertEqual(summary["comparator_wins"], 1)
                self.assertIsNone(summary["bootstrap_low"])
                self.assertIsNone(summary["exact_two_sided_sign_flip_p"])
            rows = read_csv(root / "analysis/per_seed_metrics.csv")
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["condition"] for row in rows}, set(analysis.VARIANTS) | {"persistence"})
            for path in (root / "analysis").glob("*.svg"):
                ET.parse(path)
            checksums = json.loads((root / "analysis/checksums.json").read_text())
            self.assertTrue(all(analysis._file_hash(root / "analysis" / name) == digest for name, digest in checksums.items()))

    def test_missing_interrupted_and_budget_cells_block_completeness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = manifest(101, 102, 103, 104)
            verified = fixture(root, protocol["cells"][0])
            write(root / "run_ledger.json", {"cells": {
                "seed-101": {"state": "SUCCEEDED"}, "seed-102": {"state": "FAILED", "reason": "observed failure"},
                "seed-103": {"state": "INTERRUPTED"}, "seed-104": {"state": "SKIPPED_BUDGET"}}})
            with patch.object(analysis, "_validator", return_value=verified):
                result = analysis.run(root, protocol, root / "analysis")
            self.assertEqual(result["end_state"], "PARTIAL CAMPAIGN")
            self.assertEqual(len(result["mandatory_missing"]), 3)
            self.assertEqual(len(read_csv(root / "analysis/per_seed_metrics.csv")), 6)
            self.assertEqual(result["cells"][1]["reason"], "observed failure")

    def test_invalid_completed_artifact_is_retained_as_failure_not_zero_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = manifest(101)
            write(root / "run_ledger.json", {"cells": {"seed-101": {"state": "SUCCEEDED"}}})
            with patch.object(analysis, "_validator", side_effect=ValueError("artifact changed")):
                result = analysis.run(root, protocol, root / "analysis")
            self.assertEqual(result["end_state"], "PARTIAL CAMPAIGN")
            self.assertEqual(result["valid_completed_cells"], 0)
            self.assertIn("artifact changed", result["cells"][0]["reason"])
            self.assertEqual(result["paired_aggregates"], [])

    def test_source_envelope_and_metric_mutations_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cell = manifest(101)["cells"][0]
            verified = fixture(root, cell)
            bad = copy.deepcopy(verified)
            bad[1]["source_sha256"] = "f" * 64
            with patch.object(analysis, "_validator", return_value=bad):
                with self.assertRaisesRegex(ValueError, "envelope"):
                    analysis._extract(cell, root / "cells/seed-101", "a" * 64)
            verified[1]["ablations"]["memory"]["test"]["selected"]["mse"] = .99
            with patch.object(analysis, "_validator", return_value=verified):
                with self.assertRaisesRegex(ValueError, "sum/count"):
                    analysis._extract(cell, root / "cells/seed-101", "a" * 64)
            for value in (float("nan"), float("inf"), -1, True):
                with self.assertRaises(ValueError):
                    analysis._metric(metric(value))

    def test_seed_statistics_exact_coarse_inference_and_no_pseudoreplication(self):
        summary = analysis.paired_statistics([-1] * 5, bootstrap_repetitions=1000)
        self.assertEqual(summary["exact_two_sided_sign_flip_p"], 2 / 32)
        self.assertEqual(summary["n_seeds"], 5)
        self.assertEqual(summary["memory_wins"], 5)
        self.assertEqual(summary["bootstrap_low"], -1)
        self.assertEqual(analysis.paired_statistics([0] * 3)["exact_two_sided_sign_flip_p"], 1)
        mixed = analysis.paired_statistics([-.1, .2, .3], bootstrap_repetitions=1000)
        self.assertEqual(mixed, analysis.paired_statistics([-.1, .2, .3], bootstrap_repetitions=1000))
        self.assertEqual(mixed["comparator_wins"], 2)
        self.assertIsNone(analysis.paired_statistics([1])["sample_sd"])

    def test_distinct_configuration_groups_are_not_pooled_and_duplicates_fail(self):
        rows = []
        for group, cell_id, seed in (("group-a", "a", 101), ("group-b", "b", 101)):
            for variant in set(analysis.VARIANTS) | {"persistence"}:
                rows.append(dict(protocol_group=group, cell_id=cell_id, seed=seed,
                                 condition=variant, test_mse=.5 if variant == "memory" else .4))
        aggregates, pairs = analysis._comparisons(rows)
        self.assertEqual(len(aggregates), 6)
        self.assertTrue(all(item["n_seeds"] == 1 for item in aggregates))
        self.assertTrue(all(item["difference"] > 0 for item in pairs))
        for row in rows:
            row["protocol_group"] = "same-group"
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            analysis._comparisons(rows)

    def test_unknown_cells_states_path_escape_and_changed_manifest_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = manifest(101)
            for ledger in (
                {"cells": {"unknown": {"state": "SUCCEEDED"}}},
                {"cells": {"seed-101": {"state": "MADE_UP"}}},
                {"manifest_sha256": "wrong", "cells": {}},
                {"source_sha256": "b" * 64, "cells": {}},
            ):
                write(root / "run_ledger.json", ledger)
                with self.assertRaises(ValueError):
                    analysis.run(root, protocol, root / "analysis")
            with self.assertRaises(ValueError):
                analysis._safe_cell_path(root, "../outside")


if __name__ == "__main__":
    unittest.main()
