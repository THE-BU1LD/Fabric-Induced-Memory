"""Small real CPU model fixtures; no protected outcomes or scientific comparison."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from fim_experiments import episode_bank_v1 as banks
from fim_experiments import paired_development_v1 as runner
from fim_experiments.runtime_benchmarks import DelayedRecallConfig


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def tiny_bank(tmp_path_factory):
    path = tmp_path_factory.mktemp("paired-data") / "bank"
    receipt = banks.create_development_bank(
        path,
        seed=29,
        config=DelayedRecallConfig(dimension=3, memory_dim=2, delay=2, steps=4),
        horizon=2,
        train_count=1,
        validation_count=1,
        evaluation_count=1,
    )
    return path, receipt["bank_sha256"]


def read_json(path):
    return json.loads(Path(path).read_text())


def events(path, arm):
    return [
        json.loads(line)
        for line in (path / arm / "events.jsonl").read_text().splitlines()
    ]


@pytest.fixture(scope="module")
def completed(tiny_bank, tmp_path_factory):
    path = tmp_path_factory.mktemp("paired-model") / "run"
    bank, pin = tiny_bank
    current_split, optimized_splits = [None], []
    original_episode = runner._run_episode
    original_step = torch.optim.SGD.step

    def episode(*args, **kwargs):
        current_split[0] = args[3]["split"]
        return original_episode(*args, **kwargs)

    def step(*args, **kwargs):
        optimized_splits.append(current_split[0])
        return original_step(*args, **kwargs)

    before = torch.random.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "_run_episode", episode)
        patch.setattr(torch.optim.SGD, "step", step)
        manifest = runner.run_development(
            bank, path, expected_bank_sha256=pin, config=runner.RunConfig(horizon=2)
        )
    assert manifest["status"] == "COMPLETE", [
        read_json(path / arm / "status.json") for arm in runner.ARMS
    ]
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    return path, manifest, optimized_splits


def verify(path, manifest, tiny_bank):
    return runner.verify_development_run(
        path,
        expected_run_sha256=manifest["run_sha256"],
        bank_directory=tiny_bank[0],
        expected_bank_sha256=tiny_bank[1],
    )


def test_real_three_arm_end_to_end_and_train_only_updates(completed, tiny_bank):
    path, manifest, optimized_splits = completed
    result = verify(path, manifest, tiny_bank)
    assert optimized_splits == ["train"] * 3
    assert result["status"] == "VERIFIED_INTEGRATION"
    assert result["model_steps"] == 18 and result["optimizer_updates"] == 3
    assert not result["scientific_support_inferred"] and not result["compute_matched"]
    assert result["process_peak_rss_bytes"] > 0
    assert "not per-arm" in result["rss_scope"]
    initial = read_json(path / "initial_metadata.json")
    for arm in runner.ARMS:
        status = read_json(path / arm / "status.json")
        assert status["initial"] == initial
        assert status["final"]["parameter_sha256"] != initial["parameter_sha256"]
        ends = [event for event in events(path, arm) if event["event"] == "episode_end"]
        assert ends[0]["optimization"] is not None
        for end in ends[1:]:
            assert end["optimization"] is None
            assert end["parameter_sha256_before"] == end["parameter_sha256_after"]


def test_same_inputs_and_existing_within_episode_interventions(completed):
    path = completed[0]
    for arm in runner.ARMS:
        rows = events(path, arm)
        steps = [row for row in rows if row["event"] == "step"]
        for row in steps:
            assert row["memory_before"]["size"] == (
                0 if arm == "no_memory" else row["step"]
            )
            assert row["memory_after"]["size"] == (
                0 if arm == "no_memory" else row["step"] + 1
            )
            assert row["retrieved"] is (arm == "full" and row["step"] > 0)
        if arm != "full":
            reference = [row for row in events(path, "full") if row["event"] == "step"]
            assert [r["target"] for r in steps] == [r["target"] for r in reference]
            # Teacher-forced train inputs pair exactly. Autoregressive later
            # inputs intentionally depend on the intervention's own prediction.
            assert [r["input"] for r in steps[:2]] == [
                r["input"] for r in reference[:2]
            ]
        resets = [row for row in rows if row["event"] == "reset"]
        assert len(resets) == 6
        assert all(row["after"] == {"size": 0, "ptr": 0} for row in resets)
        assert all(
            row["zero_buffers"] == ["ages", "keys", "scores", "values"]
            for row in resets
        )


def test_reproduction_with_unrelated_rng_activity_is_bitwise_identical(
    completed, tiny_bank, tmp_path
):
    torch.rand(31)
    before = torch.random.get_rng_state().clone()
    output = tmp_path / "repeat"
    manifest = runner.run_development(
        tiny_bank[0],
        output,
        expected_bank_sha256=tiny_bank[1],
        config=runner.RunConfig(horizon=2),
    )
    assert manifest["status"] == "COMPLETE"
    assert torch.equal(torch.random.get_rng_state(), before)
    assert read_json(output / "initial_metadata.json") == read_json(
        completed[0] / "initial_metadata.json"
    )
    for arm in runner.ARMS:
        assert events(output, arm) == events(completed[0], arm)
        assert (
            read_json(output / arm / "status.json")["final"]
            == read_json(completed[0] / arm / "status.json")["final"]
        )


def test_final_safe_tensor_checkpoint_loads_and_reproduces_inference(
    completed, tiny_bank
):
    implementation = runner._existing_implementation()
    first = runner._new_model(implementation, "full", runner.RunConfig())
    second = runner._new_model(implementation, "full", runner.RunConfig(model_seed=41))
    raw = (completed[0] / "full" / "final.npz").read_bytes()
    saved = runner._read_checkpoint(raw)
    for model in (first, second):
        model.load_state_dict(
            {name: torch.from_numpy(value.copy()) for name, value in saved.items()},
            strict=True,
        )
        assert (
            runner._checkpoint_metadata(model)
            == read_json(completed[0] / "full" / "status.json")["final"]
        )
        runner.strict_reset(model)
    bank = banks.load_development_bank(tiny_bank[0], expected_bank_sha256=tiny_bank[1])
    x, _ = bank.dataset("development_eval")[0]
    with torch.no_grad():
        a = first.step(x[0].reshape(1, 1, 1, -1)).prediction
        b = second.step(x[0].reshape(1, 1, 1, -1)).prediction
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.fixture
def model():
    return runner._new_model(
        runner._existing_implementation(), "full", runner.RunConfig()
    )


def test_strict_reset_checks_every_buffer_index_alias_and_parameters(model):
    before = runner._parameter_digest(model)
    for name in runner.BUFFER_SHAPES:
        getattr(model.bank, name).fill_(2)
    model.bank.ptr, model.bank.size = 7, 7
    result = runner.strict_reset(model)
    assert result["before"] == {"ptr": 7, "size": 7}
    assert result["after"] == {"ptr": 0, "size": 0}
    assert model.bank is model.trace_bank is model.memory_bank
    assert runner._parameter_digest(model) == before
    assert all(
        torch.count_nonzero(getattr(model.bank, name)) == 0
        for name in runner.BUFFER_SHAPES
    )


@pytest.mark.parametrize("name", ["keys", "values", "scores", "ages"])
def test_incomplete_buffer_reset_is_rejected(model, name, monkeypatch):
    reset = model.reset_state

    def broken():
        reset()
        getattr(model.bank, name).fill_(1)

    monkeypatch.setattr(model, "reset_state", broken)
    with pytest.raises(runner.RunnerError, match="all memory buffers"):
        runner.strict_reset(model)


@pytest.mark.parametrize("name", ["ptr", "size"])
def test_incomplete_index_reset_is_rejected(model, name, monkeypatch):
    reset = model.reset_state

    def broken():
        reset()
        setattr(model.bank, name, 1)

    monkeypatch.setattr(model, "reset_state", broken)
    with pytest.raises(runner.RunnerError, match="memory indices"):
        runner.strict_reset(model)


def test_reset_exceptions_are_not_suppressed(model, monkeypatch):
    def broken():
        raise RuntimeError("reset itself failed")

    monkeypatch.setattr(model, "reset_state", broken)
    with pytest.raises(RuntimeError, match="reset itself failed"):
        runner.strict_reset(model)


def test_reset_must_not_change_learned_parameters(model, monkeypatch):
    reset = model.reset_state

    def broken():
        reset()
        with torch.no_grad():
            next(model.parameters()).add_(1)

    monkeypatch.setattr(model, "reset_state", broken)
    with pytest.raises(runner.RunnerError, match="changed learned parameters"):
        runner.strict_reset(model)


def test_untracked_buffer_and_divergent_alias_are_rejected(model):
    model.register_buffer("forgotten_history", torch.ones(1))
    with pytest.raises(runner.RunnerError, match="buffer inventory"):
        runner.strict_reset(model)
    del model.forgotten_history
    model.trace_bank = copy.deepcopy(model.bank)
    with pytest.raises(runner.RunnerError, match="aliases"):
        runner.strict_reset(model)


def test_failed_forward_and_cleanup_are_both_retained_without_replacement(
    tiny_bank, tmp_path, monkeypatch
):
    factory = runner._new_model

    def broken(implementation, arm, config):
        model = factory(implementation, arm, config)
        if arm == "full":
            real_reset, calls = model.reset_state, [0]

            def reset():
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError("secondary cleanup failure")
                real_reset()

            def step(*args, **kwargs):
                raise ValueError("primary forward failure")

            model.reset_state, model.step = reset, step
        return model

    monkeypatch.setattr(runner, "_new_model", broken)
    output = tmp_path / "failed"
    manifest = runner.run_development(
        tiny_bank[0],
        output,
        expected_bank_sha256=tiny_bank[1],
        config=runner.RunConfig(horizon=2),
    )
    assert manifest["status"] == "FAILED"
    assert manifest["arms"] == {
        "full": "FAILED",
        "no_retrieval": "COMPLETE",
        "no_memory": "COMPLETE",
    }
    failure = next(
        row for row in events(output, "full") if row["event"] == "episode_failed"
    )
    assert [error["message"] for error in failure["errors"]] == [
        "primary forward failure",
        "secondary cleanup failure",
    ]
    status = read_json(output / "full" / "status.json")
    assert status["model_steps_attempted"] == 1 and status["model_steps_retained"] == 0
    assert (
        status["optimizer_updates_attempted"]
        == status["optimizer_updates_retained"]
        == 0
    )
    assert (output / "full" / "final.npz").is_file()
    with pytest.raises(runner.RunnerError, match="cannot qualify"):
        verify(output, manifest, tiny_bank)
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.run_development(
            tiny_bank[0],
            output,
            expected_bank_sha256=tiny_bank[1],
            config=runner.RunConfig(horizon=2),
        )
    assert (output / "manifest.json").read_bytes() == before


def test_nan_prediction_raw_bytes_and_attempt_survive_failure(
    tiny_bank, tmp_path, monkeypatch
):
    factory = runner._new_model

    def broken(implementation, arm, config):
        model = factory(implementation, arm, config)
        if arm == "full":
            step = model.step

            def nonfinite(*args, **kwargs):
                result = step(*args, **kwargs)
                result.prediction = torch.full_like(result.prediction, float("nan"))
                return result

            model.step = nonfinite
        return model

    monkeypatch.setattr(runner, "_new_model", broken)
    path = tmp_path / "nan"
    manifest = runner.run_development(
        tiny_bank[0],
        path,
        expected_bank_sha256=tiny_bank[1],
        config=runner.RunConfig(horizon=2),
    )
    assert manifest["status"] == "FAILED"
    raw = next(row for row in events(path, "full") if row["event"] == "step")
    assert (
        raw["finite_and_shape_valid"] is False
        and raw["diagnostic_observation_mse_float64"] is None
    )
    assert np.isnan(
        np.frombuffer(bytes.fromhex(raw["prediction"]["f32_hex"]), dtype="<f4")
    ).all()


@pytest.mark.parametrize(
    "primary,cleanup,expected",
    [
        (
            KeyboardInterrupt("primary cancellation"),
            RuntimeError("cleanup failure"),
            KeyboardInterrupt,
        ),
        (ValueError("primary failure"), SystemExit("cleanup cancellation"), SystemExit),
    ],
)
def test_interruption_retains_both_errors_in_nontraining_episode(
    model, tiny_bank, tmp_path, monkeypatch, primary, cleanup, expected
):
    bank = banks.load_development_bank(tiny_bank[0], expected_bank_sha256=tiny_bank[1])
    config = runner.RunConfig(horizon=2)
    episode = next(
        item
        for item in runner._episode_plan(bank, config)
        if item["split"] == "validation"
    )
    original_reset, resets = model.reset_state, [0]

    def reset():
        resets[0] += 1
        if resets[0] == 2:
            raise cleanup
        original_reset()

    def step(*args, **kwargs):
        raise primary

    monkeypatch.setattr(model, "reset_state", reset)
    monkeypatch.setattr(model, "step", step)
    journal = runner._Journal(tmp_path / "events.jsonl")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    try:
        with pytest.raises(expected):
            runner._run_episode(model, optimizer, bank, episode, config, journal, 0)
    finally:
        journal.close()
    failure = journal.records[-1]
    assert failure["event"] == "episode_failed"
    assert [error["message"] for error in failure["errors"]] == [
        str(primary),
        str(cleanup),
    ]
    assert not any(row["event"].startswith("optimizer_") for row in journal.records)
    assert not any(row["event"] == "step" for row in journal.records)
    assert (
        json.loads((tmp_path / "events.jsonl").read_text().splitlines()[-1]) == failure
    )


def test_cancellation_stops_later_arms_and_arm_failure_is_explicit(
    tiny_bank, tmp_path, monkeypatch
):
    factory, created = runner._new_model, []

    def cancelled(implementation, arm, config):
        created.append(arm)
        model = factory(implementation, arm, config)
        real_reset, calls = model.reset_state, [0]

        def reset():
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError("cleanup after cancellation")
            real_reset()

        def step(*args, **kwargs):
            raise KeyboardInterrupt("cancel requested")

        model.reset_state, model.step = reset, step
        return model

    monkeypatch.setattr(runner, "_new_model", cancelled)
    output = tmp_path / "cancelled"
    with pytest.raises(KeyboardInterrupt, match="cancel requested"):
        runner.run_development(
            tiny_bank[0],
            output,
            expected_bank_sha256=tiny_bank[1],
            config=runner.RunConfig(horizon=2),
        )
    assert created == ["full"]
    assert (
        not (output / "no_retrieval").exists() and not (output / "no_memory").exists()
    )
    assert not (output / "manifest.json").exists()
    status = read_json(output / "full" / "status.json")
    assert (
        status["status"] == "FAILED"
        and status["failures"][0]["type"] == "KeyboardInterrupt"
    )
    assert (
        status["model_steps_attempted"] == 1
        and status["optimizer_updates_retained"] == 0
    )
    failure = next(
        row for row in events(output, "full") if row["event"] == "episode_failed"
    )
    assert [error["message"] for error in failure["errors"]] == [
        "cancel requested",
        "cleanup after cancellation",
    ]
    assert (
        read_json(output / "run_failure.json")["error"]["type"] == "KeyboardInterrupt"
    )


@pytest.mark.parametrize(
    "config",
    [
        runner.RunConfig(model_seed=101, horizon=2),
        runner.RunConfig(model_seed=29, horizon=2),
        runner.RunConfig(model_seed=True, horizon=2),
        runner.RunConfig(horizon=3),
        runner.RunConfig(horizon=0),
        runner.RunConfig(horizon=2, epochs=3),
    ],
)
def test_preflight_bounds_reject_before_reservation(tiny_bank, tmp_path, config):
    path = tmp_path / "never-reserved"
    with pytest.raises((runner.RunnerError, banks.EpisodeBankError)):
        runner.run_development(
            tiny_bank[0], path, expected_bank_sha256=tiny_bank[1], config=config
        )
    assert not path.exists()


def test_protected_mode_and_wrong_bank_pin_rejected(tiny_bank, tmp_path):
    for options in (
        {"phase": "CONFIRMATORY", "expected_bank_sha256": tiny_bank[1]},
        {"expected_bank_sha256": "0" * 64},
    ):
        path = tmp_path / "no"
        with pytest.raises((runner.RunnerError, banks.EpisodeBankError)):
            runner.run_development(
                tiny_bank[0], path, config=runner.RunConfig(horizon=2), **options
            )
        assert not path.exists()


def test_non_float32_process_rejected_and_rng_unchanged(tiny_bank, tmp_path):
    before = torch.random.get_rng_state().clone()
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(runner.RunnerError, match="CPU and float32"):
            runner.run_development(
                tiny_bank[0],
                tmp_path / "no",
                expected_bank_sha256=tiny_bank[1],
                config=runner.RunConfig(horizon=2),
            )
    finally:
        torch.set_default_dtype(old)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert not (tmp_path / "no").exists()


def test_exact_model_source_imports_ignore_and_restore_foreign_cached_modules(
    monkeypatch,
):
    foreign_models, foreign_systems = (
        types.ModuleType("models"),
        types.ModuleType("systems"),
    )
    monkeypatch.setitem(sys.modules, "models", foreign_models)
    monkeypatch.setitem(sys.modules, "systems", foreign_systems)
    implementation = runner._existing_implementation()
    model = runner._new_model(implementation, "full", runner.RunConfig())
    assert (
        sys.modules["models"] is foreign_models
        and sys.modules["systems"] is foreign_systems
    )
    assert model.encoder.__class__.__module__.startswith("_fim_paired_v1_")
    assert model.bank.__class__.__module__.startswith("_fim_paired_v1_")
    assert model.memory_enabled and model.retrieval_enabled


def copy_sources(tmp_path, monkeypatch):
    root = tmp_path / "source-copy"
    for name in runner.SOURCE_PATHS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runner.ROOT / name, path)
    monkeypatch.setattr(runner, "ROOT", root)
    return root


def test_imported_source_edit_is_rejected_before_output(
    tiny_bank, tmp_path, monkeypatch
):
    root = copy_sources(tmp_path, monkeypatch)
    with (root / "fim_experiments/models.py").open("a") as handle:
        handle.write("\n# edited after import\n")
    with pytest.raises(runner.RunnerError, match="source changed"):
        runner.run_development(
            tiny_bank[0],
            tmp_path / "no",
            expected_bank_sha256=tiny_bank[1],
            config=runner.RunConfig(horizon=2),
        )
    assert not (tmp_path / "no").exists()


def test_source_edit_during_attempt_retains_failure_and_blocks_qualification(
    tiny_bank, tmp_path, monkeypatch
):
    root = copy_sources(tmp_path, monkeypatch)
    factory = runner._new_model

    def editing(implementation, arm, config):
        model = factory(implementation, arm, config)
        with (root / "fim_experiments/models.py").open("a") as handle:
            handle.write("\n# checkout changed during execution\n")
        return model

    monkeypatch.setattr(runner, "_new_model", editing)
    path = tmp_path / "changed"
    with pytest.raises(runner.RunnerError, match="source changed"):
        runner.run_development(
            tiny_bank[0],
            path,
            expected_bank_sha256=tiny_bank[1],
            config=runner.RunConfig(horizon=2),
        )
    assert (path / "plan.json").is_file() and (path / "run_failure.json").is_file()
    assert all(
        read_json(path / arm / "status.json")["status"] == "FAILED"
        for arm in runner.ARMS
    )
    assert not (path / "manifest.json").exists()


def reseal(path):
    manifest = read_json(path / "manifest.json")
    manifest["files"] = runner._file_inventory(path)
    manifest.pop("run_sha256")
    manifest["run_sha256"] = banks._digest(manifest)
    (path / "manifest.json").write_bytes(banks._canonical(manifest) + b"\n")
    return manifest


@pytest.mark.parametrize(
    "mutation",
    [
        "target",
        "input",
        "metric",
        "missing_step",
        "extra_step",
        "retrieval",
        "reset",
        "optimization",
        "allocation",
        "final_checkpoint",
    ],
)
def test_resealed_semantic_defects_rejected(completed, tiny_bank, tmp_path, mutation):
    path = tmp_path / mutation
    shutil.copytree(completed[0], path)
    rows = events(path, "full")
    step = next(row for row in rows if row["event"] == "step")
    if mutation in ("target", "input"):
        step[mutation]["f32_hex"] = np.zeros(3, dtype="<f4").tobytes().hex()
    elif mutation == "metric":
        step["diagnostic_observation_mse_float64"] += 0.01
    elif mutation == "missing_step":
        rows.remove(step)
    elif mutation == "extra_step":
        rows.append(step)
    elif mutation == "retrieval":
        step["retrieved"] = True
    elif mutation == "reset":
        next(row for row in rows if row["event"] == "reset")["zero_buffers"].remove(
            "ages"
        )
    elif mutation == "optimization":
        next(row for row in rows if row["event"] == "optimizer_attempt")[
            "optimization"
        ]["lr"] = 0.5
    elif mutation == "allocation":
        status = read_json(path / "full" / "status.json")
        status["initial"]["allocated_parameters"] += 1
        (path / "full" / "status.json").write_text(json.dumps(status))
    elif mutation == "final_checkpoint":
        arrays = runner._read_checkpoint((path / "full" / "final.npz").read_bytes())
        arrays[next(iter(arrays))] += np.float32(0.125)
        np.savez(path / "full" / "final.npz", **arrays)
    (path / "full" / "events.jsonl").write_bytes(
        b"".join(banks._canonical(row) + b"\n" for row in rows)
    )
    manifest = reseal(path)
    with pytest.raises(runner.RunnerError):
        verify(path, manifest, tiny_bank)


def test_external_run_pin_and_raw_integrity_are_both_required(
    completed, tiny_bank, tmp_path
):
    path = tmp_path / "corrupt"
    shutil.copytree(completed[0], path)
    manifest = completed[1]
    with pytest.raises(runner.RunnerError, match="pinned run identity"):
        verify(path, {"run_sha256": "0" * 64}, tiny_bank)
    with (path / "full" / "events.jsonl").open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(runner.RunnerError, match="artifact bytes"):
        verify(path, manifest, tiny_bank)


@pytest.mark.parametrize(
    "value",
    [
        np.array([{"object": "forbidden"}], dtype=object),
        np.array([float("inf")], dtype=np.float32),
    ],
)
def test_checkpoint_loader_rejects_object_and_nonfinite_arrays(value):
    handle = io.BytesIO()
    np.savez(handle, invalid=value)
    with pytest.raises((runner.RunnerError, ValueError)):
        runner._read_checkpoint(handle.getvalue())


@pytest.mark.parametrize("shape", [(1_000_001,), (3,)])
def test_malformed_npy_shape_is_rejected_before_array_allocation(shape, monkeypatch):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f4", "fortran_order": False, "shape": shape}
    )
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "tensor.npy", header.getvalue()
        )  # No payload for either claimed shape.

    def must_not_allocate(*args, **kwargs):
        pytest.fail("np.load was reached before the malformed header was rejected")

    monkeypatch.setattr(np, "load", must_not_allocate)
    with pytest.raises(runner.RunnerError, match="header|payload length"):
        runner._read_checkpoint(raw.getvalue())
