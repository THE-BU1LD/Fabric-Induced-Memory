"""Numerical contract tests. Execute only through the resource-guarded runner."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from core import ROOT
from data import causal_features, generate
from experiment import Budget, cell, score, select_alpha
from models import SequenceModel, retrieve

torch.set_num_threads(1)


class NumericalContracts(unittest.TestCase):
    def setUp(self):
        self.p = json.loads((ROOT / "protocol.json").read_text())
        self.c = self.p["conditions"][1]
        torch.manual_seed(918)

    def test_disjoint_owners_and_repeatability(self):
        a = generate(self.p, self.c, 401, "train")
        b = generate(self.p, self.c, 401, "validation")
        c = generate(self.p, self.c, 401, "train")
        self.assertTrue(set(a["owners"]).isdisjoint(b["owners"]))
        self.assertTrue(torch.equal(a["cue"], c["cue"]))
        self.assertTrue(torch.equal(a["target"], c["target"]))
        self.assertTrue((a["primary"].sum(1) >= 8).all())

    def test_noisy_keys_not_exact_and_masks_causal(self):
        a = generate(self.p, self.c, 401, "validation")
        owner, t = torch.nonzero(a["primary"])[0].tolist()
        prev = int(a["previous"][owner, t])
        self.assertGreater(t - prev, 8)
        self.assertFalse(torch.equal(a["cue"][owner, t], a["cue"][owner, prev]))

    def test_feature_shift_no_current_or_future_target(self):
        cue = torch.randn(2, 12, 3)
        target = torch.randn(2, 12, 4)
        features = causal_features(cue, target)
        changed = target.clone()
        changed[:, 6:] += 100
        self.assertTrue(
            torch.equal(features[:, :7], causal_features(cue, changed)[:, :7])
        )

    def test_hand_computed_memory_and_fifo_eviction(self):
        cue = torch.tensor([[[1.0], [2.0], [1.0], [9.0], [1.0]]])
        value = torch.tensor([[[3.0], [7.0], [5.0], [11.0], [99.0]]])
        result = retrieve(cue, value, capacity=2, topk=1, temperature=0.1)
        self.assertEqual(float(result["correction"][0, 2, 0]), 3.0)
        self.assertEqual(float(result["correction"][0, 4, 0]), 5.0)
        self.assertEqual(int(result["sources"][0, 4, 0]), 2)
        self.assertEqual(float(result["correction"][0, 0, 0]), 0.0)

    def test_actual_recent_window_and_empty_interval_rejection(self):
        cue = torch.ones(1, 12, 2)
        value = torch.arange(12.0).reshape(1, 12, 1)
        result = retrieve(cue, value, 32, 1, 0.1, maximum_age=3)
        self.assertGreaterEqual(int(result["sources"][0, 11, 0]), 8)
        self.assertLess(int(result["sources"][0, 11, 0]), 11)
        with self.assertRaises(ValueError):
            retrieve(cue, value, 32, 1, 0.1, minimum_age=4, maximum_age=3)

    def test_owner_isolation_and_batch_reordering(self):
        cue = torch.randn(3, 10, 2)
        value = torch.randn(3, 10, 4)
        a = retrieve(cue, value, 8, 4, 0.2)["correction"]
        order = torch.tensor([2, 0, 1])
        b = retrieve(cue[order], value[order], 8, 4, 0.2)["correction"]
        self.assertTrue(torch.equal(a[order], b))
        changed = value.clone()
        changed[1] += 500
        c = retrieve(cue, changed, 8, 4, 0.2)["correction"]
        self.assertTrue(torch.equal(a[0], c[0]))

    def test_current_and_future_values_cannot_affect_prediction(self):
        cue = torch.randn(2, 12, 3)
        value = torch.randn(2, 12, 4)
        a = retrieve(cue, value, 8, 4, 0.2)["correction"]
        value[:, 6:] += 999
        b = retrieve(cue, value, 8, 4, 0.2)["correction"]
        self.assertTrue(torch.equal(a[:, :7], b[:, :7]))

    def test_interventions_hold_keys_fixed(self):
        cue = torch.randn(2, 12, 3)
        value = torch.randn(2, 12, 4)
        ordinary = retrieve(cue, value, 8, 4, 0.2)
        for mode in ("erase", "sign_flip", "spatial_mean", "value_permutation"):
            result = retrieve(cue, value, 8, 4, 0.2, mode=mode)
            self.assertTrue(torch.equal(ordinary["sources"], result["sources"]))
            if mode == "erase":
                self.assertEqual(float(result["correction"].abs().sum()), 0.0)
            if mode == "sign_flip":
                self.assertTrue(
                    torch.equal(-ordinary["correction"], result["correction"])
                )

    def test_oracle_privilege_not_available_to_models(self):
        cue = torch.ones(1, 3, 1)
        value = torch.tensor([[[2.0], [7.0], [9.0]]])
        labels = torch.tensor([[0, 1, 0]])
        result = retrieve(cue, value, 8, 1, 0.2, mode="oracle", labels=labels)
        self.assertEqual(float(result["correction"][0, 1, 0]), 0.0)
        self.assertEqual(float(result["correction"][0, 2, 0]), 2.0)
        with self.assertRaises(ValueError):
            retrieve(cue, value, 8, 1, 0.2, labels=labels)

    def test_all_trained_baselines_are_causal_and_have_gradients(self):
        for kind in ("finite_history", "gru", "attention"):
            model = SequenceModel(kind, 7, 3, 16, 4).eval()
            x = torch.randn(2, 12, 7)
            a = model(x)
            changed = x.clone()
            changed[:, 6:] += 10
            b = model(changed)
            self.assertTrue(torch.allclose(a[:, :6], b[:, :6], atol=1e-6), kind)
            a.square().mean().backward()
            self.assertTrue(
                all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ),
                kind,
            )

    def test_selection_uses_only_supplied_validation(self):
        data = generate(self.p, self.c, 401, "validation")
        zero = torch.zeros_like(data["target"])
        alpha, value, _ = select_alpha(zero, data["target"], data, 1.0, self.p)
        self.assertEqual(alpha, 1.0)
        self.assertEqual(value, 0.0)
        with self.assertRaises(ValueError):
            score(torch.full_like(zero, float("nan")), data, 1.0, self.p)

    def test_small_end_to_end_and_independent_verifier(self):
        # Engineering fixture, seed outside the scientific protocol; no efficacy claim.
        from verify import verify

        p = self.p
        p["seeds"] = [19]
        p["conditions"] = [p["conditions"][0]]
        p["data"].update(
            train_owners=4, validation_owners=2, evaluation_owners=2, steps=96
        )
        p["training"].update(epochs=1, base_epochs=1, widths=[16, 32])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            experiment = root / "campaign"
            experiment.mkdir()
            result = cell(
                p, p["conditions"][0], 19, experiment / "clean-19", Budget(p, root)
            )
            self.assertEqual(
                set(result["scores"]),
                set(p["baselines"]) | set(p["interventions"]) | {"tensor", "oracle"},
            )
            output = root / "verification"
            output.mkdir()
            verify(experiment, output, p)
            receipt = json.loads((output / "VERIFICATION.json").read_text())
            self.assertEqual(receipt["status"], "PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
