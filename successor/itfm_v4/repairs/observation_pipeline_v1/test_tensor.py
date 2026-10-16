"""Actual tensor integration checks; run only through run_tensor_tests.py.

Small deterministic software fixtures exercise the real adapter/bank, not
scientific training, independent reproduction or a benchmark.
"""

import dataclasses
import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn
from fim_sota.adapter import (
    ITFMRAdapter,
    ContextKeyEncoder,
    ResidualNormalizer,
    ChannelCalibrator,
    MonotoneSimilarityGate,
)
from fim_sota.contracts import BackboneOutput, ForecastBatch
from fim_sota.memory import TensorMemoryBank
from observations import Observation
from tensor_bridge import ForecastInputs, SafeOnlineAdapter
from test_pipeline import recorder


class FixtureBackbone(nn.Module):
    def identity(self):
        return {"name": "software_fixture_only", "version": 1}

    def forward_batch(self, batch):
        # Assertions inspect the actual object delivered to this backbone.
        if bool(batch.target.any().item()):
            raise AssertionError("Scorer truth reached predictor")
        return BackboneOutput(
            batch.context.clone(), batch.context.flatten(1), batch.observation_mask
        )


def setup_bridge(rec=None):
    torch.manual_seed(19)
    backbone = FixtureBackbone().eval()
    bank = TensorMemoryBank(
        key_dimension=2,
        residual_shape=(1, 2),
        capacity_per_trajectory=4,
        minimum_age=1,
        topk=1,
        temperature=1.0,
    )
    encoder = ContextKeyEncoder(
        latent_dimension=2, key_dimension=2, mask_features=2, lead_features=2
    )
    adapter = ITFMRAdapter(
        key_encoder=encoder,
        bank=bank,
        normalizer=ResidualNormalizer(1),
        calibrator=ChannelCalibrator(1),
        gate=MonotoneSimilarityGate(),
    ).eval()
    inputs = ForecastInputs(
        torch.tensor([[[1.0, 2.0]]]),
        torch.ones(1, 1, 2),
        ("owner",),
        torch.tensor([0]),
        torch.tensor([2]),
        (1, 2),
        "development",
    )
    dummy = ForecastBatch(
        inputs.context,
        torch.zeros(1, 1, 2),
        inputs.observation_mask,
        inputs.trajectory_ids,
        inputs.event_times,
        inputs.lead_times,
        inputs.split,
    )
    with torch.no_grad():
        encoder(backbone.forward_batch(dummy), dummy)
        adapter.calibrator.alpha.fill_(1.0)
    return SafeOnlineAdapter(
        backbone, adapter, source_sha256="a" * 64, split="development", recorder=rec
    ), inputs


class TensorIntegrationTests(unittest.TestCase):
    def test_partial_then_complete_reveal_writes_real_bank_once(self):
        bridge, inputs = setup_bridge()
        bridge.predict(inputs, forecast_ids=["f"], now=0)
        e = Observation("a", "f", "owner", 2, (0,), (4.0,), (2,))
        self.assertIsNone(bridge.observe(e, now=2))
        self.assertEqual(len(bridge.adapter.bank), 0)
        bridge.observe(Observation("b", "f", "owner", 5, (1,), (8.0,), (2,)), now=5)
        torch.testing.assert_close(
            bridge.adapter.bank.residuals, torch.tensor([[[3.0, 6.0]]])
        )
        self.assertEqual(bridge.adapter.bank.available_times.tolist(), [5])
        with self.assertRaises(ValueError):
            bridge.observe(Observation("c", "f", "owner", 6, (0,), (4.0,), (2,)), now=6)

    def test_actual_predictions_automatically_reach_validator(self):
        rec = recorder()
        bridge, inputs = setup_bridge(rec)
        sample = rec._plan["samples"][0]
        bridge.predict(inputs, forecast_ids=["f"], now=0, samples=[sample])
        rec.record_target(sample, [4.0, 8.0])
        with tempfile.TemporaryDirectory() as tmp:
            result = rec.finalize(Path(tmp) / "actual-tensor-export.json")["result"]
        self.assertEqual(result["status"], "PASS_INTEGRITY")
        self.assertFalse(result["numerical_gates_pass"])

    def test_same_tick_reveal_cannot_affect_prediction(self):
        bridge, inputs = setup_bridge()
        bridge.predict(inputs, forecast_ids=["f"], now=0)
        bridge.observe(
            Observation("a", "f", "owner", 2, (0, 1), (4.0, 8.0), (2, 2)), now=2
        )
        later = dataclasses.replace(inputs, event_times=torch.tensor([2]))
        with self.assertRaises(ValueError):
            bridge.predict(later, forecast_ids=["g"], now=2)
        later = dataclasses.replace(inputs, event_times=torch.tensor([3]))
        result = bridge.predict(later, forecast_ids=["g"], now=3)
        self.assertTrue(bool(result.retrieval.valid.all().item()))

    def test_model_change_and_split_change_are_refused(self):
        bridge, inputs = setup_bridge()
        with self.assertRaises(ValueError):
            bridge.predict(
                dataclasses.replace(inputs, split="protected"),
                forecast_ids=["f"],
                now=0,
            )
        with torch.no_grad():
            bridge.adapter.calibrator.alpha.zero_()
        with self.assertRaises(ValueError):
            bridge.predict(inputs, forecast_ids=["f"], now=0)

    def test_mutated_input_after_prediction_cannot_change_stored_base(self):
        bridge, inputs = setup_bridge()
        bridge.predict(inputs, forecast_ids=["f"], now=0)
        inputs.context.fill_(999.0)
        bridge.observe(
            Observation("a", "f", "owner", 2, (0, 1), (4.0, 8.0), (2, 2)), now=2
        )
        torch.testing.assert_close(
            bridge.adapter.bank.residuals, torch.tensor([[[3.0, 6.0]]])
        )

    def test_non_tensor_configuration_change_is_refused(self):
        bridge, inputs = setup_bridge()
        bridge.adapter.gate.threshold = 0.75
        with self.assertRaises(ValueError):
            bridge.predict(inputs, forecast_ids=["f"], now=0)


if __name__ == "__main__":
    unittest.main()
