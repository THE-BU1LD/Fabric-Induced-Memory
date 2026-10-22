"""Tensor tests: invoke only through the resource-admitted engineering runner."""

import dataclasses
import json
import math
import os
from itertools import pairwise

import pytest
import torch
from torch import nn

from fim_sota.adapter import (
    ChannelCalibrator,
    ContextKeyEncoder,
    ITFMRAdapter,
    MonotoneSimilarityGate,
    ResidualNormalizer,
)
from fim_sota.contracts import BackboneOutput
from fim_sota.engineering import data_identity, digest, resolve, source_identity
from fim_sota.engineering_runtime import (
    ALPHA_GRID,
    VARIANTS,
    _select_validation_alpha,
    autonomous_base_diagnostic,
    build_model,
    evaluate_checkpoint,
    evaluate_stream,
    forecast_step,
    generate_fixture,
    make_batches,
    raw_data_sha256,
    run_experiment,
)
from fim_sota.memory import TensorMemoryBank
from fim_sota.training import move_batch, seed_everything


def config(**changes):
    return resolve(
        {
            "epochs": 1,
            "batch_size": 2,
            "train_trajectories": 2,
            "validation_trajectories": 2,
            "test_trajectories": 2,
            "context_steps": 2,
            "sequence_steps": 6,
            "channels": 2,
            "spatial_points": 3,
            "hidden": 5,
            "gradient_accumulation": 2,
            "memory_capacity": 4,
            "memory_topk": 2,
            **changes,
        }
    )


def identities(configuration):
    return (
        source_identity()["sha256"],
        digest(configuration),
        data_identity(configuration),
    )


class ZeroModel(nn.Module):
    def forward(self, context):
        return torch.zeros_like(context[:, -1])


def controlled_fixture():
    values = torch.arange(36, dtype=torch.float32).reshape(6, 2, 3) / 40 + 0.1
    return {
        split: {
            f"{split}/owner-a": values.clone(),
            f"{split}/owner-b": values.clone() + 1,
        }
        for split in ("train", "validation", "test")
    }


def assert_records_equal(left, right):
    assert len(left) == len(right)
    for first, second in zip(left, right, strict=True):
        assert first.keys() == second.keys()
        for key in first:
            if isinstance(first[key], torch.Tensor):
                assert torch.equal(first[key], second[key]), key
            else:
                assert first[key] == second[key], key


def test_raw_split_streams_are_deterministic_independent_and_disjoint():
    cfg = config()
    first = generate_fixture(cfg)
    torch.manual_seed(7851)
    torch.randn(20)
    second = generate_fixture(cfg)
    assert raw_data_sha256(first) == raw_data_sha256(second)
    owners = [set(first[split]) for split in first]
    assert not any(owners[i] & owners[j] for i in range(3) for j in range(i))
    changed = generate_fixture(config(train_trajectories=3))
    for split in ("validation", "test"):
        for owner in first[split]:
            assert torch.equal(first[split][owner], changed[split][owner])
    assert raw_data_sha256(first) != raw_data_sha256(generate_fixture(config(seed=102)))
    assert all(
        bool((values.abs() <= 1).all())
        for owners_ in first.values()
        for values in owners_.values()
    )


def test_batches_use_only_past_context_and_next_raw_observation():
    cfg = config()
    fixture = generate_fixture(cfg)
    batches = make_batches(fixture, cfg, "train", batch_size=1)
    assert len(batches) == cfg["train_trajectories"] * (
        cfg["sequence_steps"] - cfg["context_steps"]
    )
    for batch in batches:
        owner = batch.trajectory_ids[0]
        event = int(batch.event_times.item())
        assert torch.equal(
            batch.context[0], fixture["train"][owner][event - 1 : event + 1]
        )
        assert torch.equal(batch.target[0], fixture["train"][owner][event + 1])
        assert int(batch.target_available_times.item()) == event + 1
        assert bool(batch.observation_mask.all())
    pairs = [(b.trajectory_ids[0], int(b.event_times.item())) for b in batches]
    assert pairs == sorted(pairs)


def test_training_loss_is_sample_mean_with_finite_nonzero_gradients():
    cfg = config()
    seed_everything(cfg["seed"])
    model = build_model(cfg)
    batch = make_batches(generate_fixture(cfg), cfg, "train")[0]
    loss = forecast_step(model, batch)
    expected = (model(batch.context) - batch.target).square().flatten(1).mean(1).mean()
    assert torch.equal(loss, expected)
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(
        gradient is not None and bool(torch.isfinite(gradient).all())
        for gradient in gradients
    )
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_memory_persists_causally_with_owner_and_split_isolation():
    cfg = config()
    fixture = controlled_fixture()
    batches = make_batches(fixture, cfg, "validation", batch_size=1)
    metrics, records = evaluate_stream(ZeroModel(), batches, cfg, variant="memory")
    assert metrics["retrieval_hits"] == 6
    for index in (0, 4):
        assert records[index]["eligible_count"] == 0
        assert torch.count_nonzero(records[index]["correction"]) == 0
    assert records[4]["bank_size_before_write"] > 0
    assert records[1]["eligible_count"] == 1
    assert torch.count_nonzero(records[1]["correction"]) > 0
    for row in records:
        if row["source_event_time"] >= 0:
            assert row["source_event_time"] < row["event_time"]
            assert row["source_available_time"] <= row["event_time"]
    _, again = evaluate_stream(ZeroModel(), batches, cfg, variant="memory")
    assert_records_equal(records, again)
    test_batches = make_batches(fixture, cfg, "test", batch_size=1)
    _, test = evaluate_stream(ZeroModel(), test_batches, cfg, variant="memory")
    assert test[0]["bank_size_before_write"] == 0
    with pytest.raises(ValueError, match="cannot span"):
        evaluate_stream(
            ZeroModel(), [batches[0], test_batches[0]], cfg, variant="memory"
        )
    with pytest.raises(ValueError, match="must increase"):
        evaluate_stream(ZeroModel(), [batches[1], batches[0]], cfg, variant="memory")


def test_same_step_target_is_absent_from_key_and_prediction():
    cfg = config()
    batches = make_batches(controlled_fixture(), cfg, "validation", batch_size=1)
    changed = list(batches)
    changed[1] = dataclasses.replace(changed[1], target=changed[1].target + 30)
    _, before = evaluate_stream(ZeroModel(), batches, cfg, variant="memory")
    _, after = evaluate_stream(ZeroModel(), changed, cfg, variant="memory")
    assert torch.equal(before[1]["context_key"], after[1]["context_key"])
    assert torch.equal(before[1]["prediction"], after[1]["prediction"])
    assert not torch.equal(before[2]["prediction"], after[2]["prediction"])
    with pytest.raises(ValueError, match="fully observed"):
        evaluate_stream(
            ZeroModel(),
            [dataclasses.replace(batches[0], target_observation_mask=None)],
            cfg,
            variant="memory",
        )


def test_public_writer_rejects_partial_hidden_targets_and_admits_complete_rows():
    cfg = config()
    batch = make_batches(controlled_fixture(), cfg, "validation", batch_size=1)[0]
    bank = TensorMemoryBank(
        key_dimension=2,
        residual_shape=(2, 3),
        capacity_per_trajectory=4,
        minimum_age=1,
        topk=1,
        temperature=0.2,
    )
    adapter = ITFMRAdapter(
        key_encoder=ContextKeyEncoder(
            latent_dimension=2, key_dimension=2, mask_features=2, lead_features=2
        ),
        bank=bank,
        normalizer=ResidualNormalizer(2),
        calibrator=ChannelCalibrator(2),
        gate=MonotoneSimilarityGate(),
    )
    output = BackboneOutput(
        prediction=torch.zeros_like(batch.target),
        latent_key=torch.ones(1, 2),
        observation_mask=batch.observation_mask,
    )
    mask = torch.ones_like(batch.target, dtype=torch.bool)
    mask[0, 0, 1] = False
    partial = dataclasses.replace(batch, target_observation_mask=mask)
    before = bank.checksum()
    assert adapter.write_observed(output, partial, observation_time=2) == 0
    hidden_changed = partial.target.clone()
    hidden_changed[0, 0, 1] += 10000
    poisoned = dataclasses.replace(partial, target=hidden_changed)
    assert adapter.write_observed(output, poisoned, observation_time=2) == 0
    assert bank.checksum() == before
    assert adapter.write_observed(output, batch, observation_time=1) == 0
    assert adapter.write_observed(output, batch, observation_time=2) == 1
    assert len(bank) == 1
    assert torch.equal(bank.residuals[0], batch.target[0])


def test_erasure_spatial_mean_and_last_residual_change_computation():
    cfg = config()
    batches = make_batches(controlled_fixture(), cfg, "validation", batch_size=1)
    outputs = {
        variant: evaluate_stream(ZeroModel(), batches, cfg, variant=variant)[1]
        for variant in VARIANTS
    }
    for baseline, erased in zip(outputs["base"], outputs["erased"], strict=True):
        assert torch.equal(baseline["prediction"], erased["prediction"])
    full = outputs["memory"][1]["correction"]
    mean = outputs["spatial_mean"][1]["correction"]
    assert not torch.equal(full, mean)
    assert torch.equal(mean, full.mean(dim=-1, keepdim=True).expand_as(full))
    assert not torch.equal(mean[0], mean[1])
    for index in (1, 2, 3, 5, 6, 7):
        assert torch.equal(
            outputs["last_residual"][index]["correction"],
            outputs["last_residual"][index - 1]["target"],
        )
    assert any(
        not torch.equal(a["correction"], b["correction"])
        for a, b in zip(outputs["memory"], outputs["last_residual"], strict=True)
    )


def test_alpha_is_selected_only_on_validation_and_zero_can_win():
    row = {
        "split": "validation",
        "base_prediction": torch.tensor([0.0]),
        "target": torch.tensor([1.0]),
        "correction": torch.tensor([1.0]),
    }
    alpha, scores = _select_validation_alpha([row])
    assert alpha == 1.0
    assert tuple(item["alpha"] for item in scores) == ALPHA_GRID
    assert (
        _select_validation_alpha([{**row, "correction": torch.tensor([-1.0])}])[0]
        == 0.0
    )
    with pytest.raises(ValueError, match="validation"):
        _select_validation_alpha([{**row, "split": "test"}])


def test_autonomous_rollout_never_consumes_hidden_future_targets():
    cfg = config(sequence_steps=20)
    seed_everything(cfg["seed"])
    model = build_model(cfg)
    fixture = generate_fixture(cfg)
    summary, original = autonomous_base_diagnostic(model, fixture["test"], cfg)
    poisoned = {owner: values.clone() for owner, values in fixture["test"].items()}
    for values in poisoned.values():
        values[cfg["context_steps"] :] += 100
    changed_summary, changed = autonomous_base_diagnostic(model, poisoned, cfg)
    assert summary["all_predictions_finite"]
    assert summary["horizon_steps"] == 18
    assert summary["forecast_count"] == 36
    assert summary["overall"]["mse"] != changed_summary["overall"]["mse"]
    for first, second in zip(original, changed, strict=True):
        assert torch.equal(first["prediction"], second["prediction"])
        assert torch.equal(first["context"], second["context"])
        assert first["memory_writes"] == second["memory_writes"] == 0
        assert bool(torch.isfinite(first["prediction"]).all())
    for owner in fixture["test"]:
        rows = [row for row in original if row["owner"] == owner]
        for previous, current in pairwise(rows):
            assert torch.equal(current["context"][-1], previous["prediction"])


def _minimal_checkpoint(path, cfg, **overrides):
    from fim_sota.training import _batch_fingerprint

    source, configuration, data = identities(cfg)
    seed_everything(cfg["seed"])
    fixture = generate_fixture(cfg)
    torch.save(
        {
            "schema_version": 2,
            "source_sha256": source,
            "config_sha256": configuration,
            "data_sha256": data,
            "runtime": {
                "torch": str(torch.__version__),
                "device": cfg["device"],
                "precision": cfg["precision"],
            },
            "batch_sha256": _batch_fingerprint(
                make_batches(fixture, cfg, "train"),
                make_batches(fixture, cfg, "validation"),
            ),
            "model_state": build_model(cfg).state_dict(),
            "step": 0,
            **overrides,
        },
        path,
    )
    return source, configuration, data


def test_frozen_replay_is_safe_exact_and_rejects_identity_drift(tmp_path):
    cfg = config()
    checkpoint = tmp_path / "model.pt"
    ids = _minimal_checkpoint(checkpoint, cfg)
    first = evaluate_checkpoint(cfg, checkpoint, tmp_path / "first", *ids)
    replay = evaluate_checkpoint(cfg, checkpoint, tmp_path / "replay", *ids)
    assert first["ablations"] == replay["ablations"]
    assert first["raw_data_sha256"] == replay["raw_data_sha256"]
    raw = torch.load(tmp_path / "first/predictions.pt", weights_only=True)
    reraw = torch.load(tmp_path / "replay/predictions.pt", weights_only=True)
    archived_data = torch.load(tmp_path / "first/raw_data.pt", weights_only=True)
    assert raw_data_sha256(archived_data["trajectories"]) == first["raw_data_sha256"]
    assert archived_data["config"] == cfg
    assert_records_equal(raw["records"], reraw["records"])
    assert_records_equal(
        raw["autonomous_base_diagnostic"], reraw["autonomous_base_diagnostic"]
    )
    for variant in VARIANTS:
        summary = first["ablations"][variant]
        assert summary["validation"]["alpha"] == summary["test"]["alpha"]
        assert math.isfinite(summary["test"]["selected"]["mse"])
        for split in ("validation", "test"):
            rows = [
                row
                for row in raw["records"]
                if row["variant"] == variant and row["split"] == split
            ]
            count = sum(row["target"].numel() for row in rows)
            for metric, key in (
                ("selected", "prediction"),
                ("base", "base_prediction"),
                ("persistence", "persistence_prediction"),
            ):
                total = sum(
                    float((row[key].double() - row["target"].double()).square().sum())
                    for row in rows
                )
                assert summary[split][metric]["element_count"] == count
                # Independent float64 reductions can differ with tensor rank
                # and reduction order. Bound only this arithmetic roundoff;
                # prediction, replay and optimizer equality remain bitwise.
                tolerance = 4 * count * torch.finfo(torch.float64).eps
                assert math.isclose(
                    summary[split][metric]["squared_error_sum"],
                    total,
                    rel_tol=tolerance,
                    abs_tol=0.0,
                )
                assert math.isclose(
                    summary[split][metric]["mse"],
                    total / count,
                    rel_tol=tolerance,
                    abs_tol=0.0,
                )
    with pytest.raises(FileExistsError):
        evaluate_checkpoint(cfg, checkpoint, tmp_path / "first", *ids)
    _minimal_checkpoint(checkpoint, cfg, data_sha256="0" * 64)
    with pytest.raises(ValueError, match="data_sha256 identity"):
        evaluate_checkpoint(cfg, checkpoint, tmp_path / "wrong-data", *ids)
    assert not (tmp_path / "wrong-data").exists()
    with pytest.raises(ValueError, match="configuration identity"):
        evaluate_checkpoint(
            config(seed=999), checkpoint, tmp_path / "wrong-config", *ids
        )


def _train_replay(tmp_path, device):
    cfg = config(
        device=device,
        train_trajectories=1,
        validation_trajectories=1,
        test_trajectories=1,
        sequence_steps=5,
        channels=1,
        spatial_points=2,
    )
    ids = identities(cfg)
    result = run_experiment(cfg, tmp_path, "train", *ids, lambda: {"status": "PASS"})
    assert result["status"] == "PASS_ENGINEERING"
    assert result["steps_completed"] > 0
    assert result["examples_seen"] == 3
    assert result["best_checkpoint"].endswith("best.pt")
    checkpoint = torch.load(
        result["best_checkpoint"], map_location="cpu", weights_only=True
    )
    seed_everything(cfg["seed"])
    initial = build_model(cfg).state_dict()
    assert any(
        not torch.equal(initial[name], value.cpu())
        for name, value in checkpoint["model_state"].items()
    )
    replay = evaluate_checkpoint(
        cfg, result["best_checkpoint"], tmp_path / "replay", *ids
    )
    assert result["evaluation"]["ablations"] == replay["ablations"]
    saved = json.loads((tmp_path / "train/evaluation/metrics.json").read_text())
    assert saved["scientific_claim"] == "NOT_EVALUATED"
    assert saved["evidence_class"] == "SYNTHETIC_ENGINEERING_FIXTURE"


def test_requested_device_training_evaluation(tmp_path):
    device = os.environ.get("FIM_TEST_DEVICE", "cpu")
    assert device in {"cpu", "cuda"}
    if device == "cuda":
        assert torch.cuda.is_available(), (
            "CUDA was explicitly requested and must not fall back"
        )
    _train_replay(tmp_path, device)


@pytest.mark.skipif(
    os.environ.get("FIM_TEST_DEVICE") != "cuda",
    reason="CUDA was not explicitly requested",
)
def test_train_select_checkpoint_and_replay_requested_cuda(tmp_path):
    assert torch.cuda.is_available(), (
        "CUDA was explicitly requested and must not fall back"
    )
    _train_replay(tmp_path, "cuda")


@pytest.mark.skipif(
    os.environ.get("FIM_TEST_DEVICE") != "cuda",
    reason="CUDA precision comparison was not explicitly requested",
)
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_requested_cuda_precision_gradients_and_online_stream_match_fp32(precision):
    assert torch.cuda.is_available(), (
        "CUDA was explicitly requested and must not fall back"
    )
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("requested CUDA device does not support bf16")
    reference_cfg = config(device="cuda", precision="fp32", sequence_steps=10)
    mixed_cfg = {**reference_cfg, "precision": precision}
    seed_everything(reference_cfg["seed"])
    reference = build_model(reference_cfg).to("cuda")
    mixed = build_model(mixed_cfg).to("cuda")
    mixed.load_state_dict(reference.state_dict())
    fixture = generate_fixture(reference_cfg)
    batch = move_batch(
        make_batches(fixture, reference_cfg, "train")[0], torch.device("cuda")
    )
    reference_loss = forecast_step(reference, batch)
    reference_loss.backward()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    with torch.autocast("cuda", dtype=dtype):
        mixed_loss = forecast_step(mixed, batch)
    mixed_loss.backward()
    assert torch.isfinite(mixed_loss)
    assert torch.allclose(mixed_loss, reference_loss, rtol=0.1, atol=0.02)
    for expected, actual in zip(
        reference.parameters(), mixed.parameters(), strict=True
    ):
        assert actual.grad is not None and bool(torch.isfinite(actual.grad).all())
        assert torch.allclose(actual.grad, expected.grad, rtol=0.15, atol=0.005)
    # Same fixed predictor, chronological observed-context inference, fresh banks.
    # This is not an unobserved autoregressive rollout or a stability claim.
    for split in ("validation", "test"):
        batches = make_batches(fixture, reference_cfg, split, batch_size=1)
        _, fp32_records = evaluate_stream(
            reference, batches, reference_cfg, variant="memory"
        )
        _, mixed_records = evaluate_stream(mixed, batches, mixed_cfg, variant="memory")
        for expected, actual in zip(fp32_records, mixed_records, strict=True):
            assert bool(torch.isfinite(actual["prediction"]).all())
            assert torch.allclose(
                actual["prediction"], expected["prediction"], rtol=0.15, atol=0.02
            )
    _, expected_rollout = autonomous_base_diagnostic(
        reference, fixture["test"], reference_cfg
    )
    _, actual_rollout = autonomous_base_diagnostic(mixed, fixture["test"], mixed_cfg)
    for expected, actual in zip(expected_rollout, actual_rollout, strict=True):
        assert bool(torch.isfinite(actual["prediction"]).all())
        assert torch.allclose(
            actual["prediction"], expected["prediction"], rtol=0.15, atol=0.02
        )
