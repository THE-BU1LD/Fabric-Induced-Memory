"""Lightweight contract/failure tests; never imports numerical libraries."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import ROOT, admission, decide, exact_sign_flip, validate, write_new
from run import output_path


class Contracts(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / "protocol.json").read_text())

    def test_protocol(self):
        validate(self.protocol)

    def test_empty_recent_control_rejected(self):
        self.protocol["memory"]["minimum_age"] = 9
        with self.assertRaises(ValueError):
            validate(self.protocol)

    def test_duplicate_seeds_rejected(self):
        self.protocol["seeds"][1] = self.protocol["seeds"][0]
        with self.assertRaises(ValueError):
            validate(self.protocol)

    def test_resource_floor_cannot_change(self):
        self.protocol["budget"]["minimum_free_mib"] = 100
        with self.assertRaises(ValueError):
            validate(self.protocol)

    def test_protected_access_rejected(self):
        self.protocol["protected_evaluation"]["authorized"] = True
        with self.assertRaises(ValueError):
            validate(self.protocol)

    def test_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "evidence.json"
            write_new(path, {"negative": True})
            with self.assertRaises(FileExistsError):
                write_new(path, {"negative": False})
            self.assertEqual(json.loads(path.read_text()), {"negative": True})

    def test_refuse_old_evidence_output(self):
        with self.assertRaises(ValueError):
            output_path(ROOT.parent / "itfm_v2/results/new")

    def test_missing_resource_observation_blocks(self):
        with patch("core.subprocess.check_output", side_effect=OSError("denied")):
            self.assertEqual(admission()["status"], "BLOCKED")

    def test_seed_level_statistic(self):
        self.assertEqual(exact_sign_flip([1] * 5), 0.0625)
        self.assertEqual(exact_sign_flip([0] * 5), 1.0)
        with self.assertRaises(ValueError):
            exact_sign_flip([])

    def test_oracle_gap_is_required(self):
        metrics = {
            "base": 1.0,
            "tensor": 0.1,
            "oracle": 1.0,
            "strongest": 0.5,
            "erase": 1.0,
            "nonprimary_delta": 0.0,
            "permutation_fraction": 1.0,
            "value_permutation": 0.8,
        }
        self.assertEqual(
            decide(metrics, self.protocol["decision"])["status"], "NOT_SUPPORTED"
        )

    def test_strong_baseline_can_falsify(self):
        metrics = {
            "base": 1.0,
            "tensor": 0.1,
            "oracle": 0.05,
            "strongest": 0.09,
            "erase": 1.0,
            "nonprimary_delta": 0.0,
            "permutation_fraction": 1.0,
            "value_permutation": 0.8,
        }
        result = decide(metrics, self.protocol["decision"])
        self.assertFalse(result["checks"]["baseline_gain"])
        self.assertEqual(result["status"], "NOT_SUPPORTED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
