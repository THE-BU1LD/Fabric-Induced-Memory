"""Actual tensor/durable integration fixtures, never scientific performance data."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from durable import DurableRecorder, DurableObservationLedger, audit_observations
from observations import Observation
from test_pipeline import plan
from test_tensor import setup_bridge
from verify_export import verify


class DurableTensorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_actual_predictions_survive_restart_and_reanalyze(self):
        raw = json.dumps(plan()).encode()
        sha = hashlib.sha256(raw).hexdigest()
        rec = DurableRecorder.create(
            self.root / "predictions", raw, expected_sha256=sha
        )
        self.addCleanup(rec.close)
        bridge, inputs = setup_bridge(rec)
        sample = plan()["samples"][0]
        prediction = bridge.predict(inputs, forecast_ids=["f"], now=0, samples=[sample])
        self.assertEqual(
            rec.payload()["rows"][0]["prediction"],
            prediction.prediction.flatten().tolist(),
        )
        anchor = rec.anchor
        rec.close()
        recovered = DurableRecorder.resume(
            self.root / "predictions", expected_anchor=anchor, expected_plan_sha256=sha
        )
        self.addCleanup(recovered.close)
        recovered.record_target(sample, [4, 8])
        bundle = recovered.finalize(self.root / "bundle.json")
        self.assertEqual(
            verify(bundle, raw, expected_plan_sha256=sha)["evidence_status"],
            "PASS_INTEGRITY",
        )

    def test_actual_bank_and_durable_observation_replay_agree(self):
        bridge, inputs = setup_bridge()
        identity = {
            "source_sha256": bridge.source_sha256,
            "split": bridge.split,
            "model_state_sha256": bridge._state,
        }
        ledger = DurableObservationLedger.create(self.root / "events", **identity)
        self.addCleanup(ledger.close)
        bridge.ledger = ledger
        bridge.predict(inputs, forecast_ids=["f"], now=0)
        bridge.observe(Observation("one", "f", "owner", 2, (0,), (4,), (2,)), now=2)
        self.assertEqual(len(bridge.adapter.bank), 0)
        bridge.observe(Observation("two", "f", "owner", 5, (1,), (8,), (2,)), now=5)
        anchor = ledger.anchor
        ledger.close()
        replay = audit_observations(
            self.root / "events", expected_anchor=anchor, expected_identity=identity
        )
        residual = torch.tensor(replay["completed_residuals"][0]["residual"]).reshape(
            1, 1, 2
        )
        torch.testing.assert_close(bridge.adapter.bank.residuals, residual)
        self.assertEqual(bridge.adapter.bank.available_times.tolist(), [5])

    def test_unrelated_empty_owner_does_not_disable_populated_owner(self):
        bridge, inputs = setup_bridge()
        bridge.predict(inputs, forecast_ids=["f"], now=0)
        bridge.observe(
            Observation("one", "f", "owner", 2, (0, 1), (4, 8), (2, 2)), now=2
        )
        later = replace(
            inputs,
            context=inputs.context.repeat(2, 1, 1),
            observation_mask=inputs.observation_mask.repeat(2, 1, 1),
            trajectory_ids=("owner", "new"),
            event_times=torch.tensor([3, 3]),
            lead_times=torch.tensor([2, 2]),
        )
        result = bridge.predict(later, forecast_ids=["g", "h"], now=3)
        self.assertEqual(result.retrieval.valid.tolist(), [True, False])
        torch.testing.assert_close(result.prediction[1], inputs.context[0])

    def test_owner_output_survives_batch_permutation(self):
        bridge, inputs = setup_bridge()
        bridge.predict(inputs, forecast_ids=["f"], now=0)
        bridge.observe(
            Observation("one", "f", "owner", 2, (0, 1), (4, 8), (2, 2)), now=2
        )
        later = replace(
            inputs,
            context=inputs.context.repeat(2, 1, 1),
            observation_mask=inputs.observation_mask.repeat(2, 1, 1),
            trajectory_ids=("owner", "new"),
            event_times=torch.tensor([3, 3]),
            lead_times=torch.tensor([2, 2]),
        )
        first = bridge.predict(later, forecast_ids=["g", "h"], now=3)
        second = bridge.predict(
            replace(later, trajectory_ids=("new", "owner")),
            forecast_ids=["i", "j"],
            now=3,
        )
        torch.testing.assert_close(first.prediction[0], second.prediction[1])
        torch.testing.assert_close(first.prediction[1], second.prediction[0])
