"""Bounded real-generator and adversarial disk tests; no model training/outcomes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
import pytest
import torch

from fim_experiments import episode_bank_v1 as bank
from fim_experiments.benchmark import DelayedRecallBenchmark as FrozenBenchmark
from fim_experiments.benchmark import DelayedRecallConfig as FrozenConfig
from fim_experiments.runtime_benchmarks import (
    DelayedRecallBenchmark,
    DelayedRecallConfig,
)
from fim_experiments.train import _unpack_batch


def make_bank(path, **kwargs):
    defaults = dict(
        seed=37,
        config=DelayedRecallConfig(dimension=5, memory_dim=4, delay=3, steps=7),
        horizon=12,
        train_count=3,
        validation_count=1,
        evaluation_count=1,
    )
    defaults.update(kwargs)
    receipt = bank.create_development_bank(path, **defaults)
    return receipt, bank.load_development_bank(
        path, expected_bank_sha256=receipt["bank_sha256"]
    )


def rewrite_manifest(path, transform, *, raw=None):
    """Reseal artificial corrupt fixtures to exercise semantics beyond hashes."""
    manifest = json.loads((path / "manifest.json").read_text())
    if raw is not None:
        (path / "observations.f32").write_bytes(raw)
        manifest["array"]["sha256"] = hashlib.sha256(raw).hexdigest()
        manifest["episode_sha256"] = bank._episode_ids(raw, manifest["array"]["shape"])
    transform(manifest)
    manifest.pop("bank_sha256")
    manifest["bank_sha256"] = bank._digest(manifest)
    (path / "manifest.json").write_bytes(bank._canonical(manifest) + b"\n")
    return manifest["bank_sha256"]


def test_exact_native_and_frozen_reference_data_at_float32(tmp_path):
    receipt, loaded = make_bank(tmp_path / "bank")
    config = DelayedRecallConfig(**loaded.manifest["benchmark_config"])
    native = DelayedRecallBenchmark(config)
    generator = torch.Generator().manual_seed(37)
    x, y = native.generate_batch(5, steps=12, generator=generator)
    raw = np.fromfile(tmp_path / "bank" / "observations.f32", dtype="<f4").reshape(
        5, 13, 5
    )
    expected = torch.from_numpy(raw.copy())
    torch.testing.assert_close(x, expected[:, :-1], rtol=0, atol=0)
    torch.testing.assert_close(y, expected[:, 1:], rtol=0, atol=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        frozen_x, frozen_y = FrozenBenchmark(
            FrozenConfig(**asdict(config))
        ).generate_batch(5, steps=12)
    torch.testing.assert_close(frozen_x, x, rtol=0, atol=0)
    torch.testing.assert_close(frozen_y, y, rtol=0, atol=0)
    assert receipt["shape"] == [5, 13, 5]


def test_rng_default_dtype_and_destination_do_not_change_bank_identity(tmp_path):
    before = torch.random.get_rng_state().clone()
    first, _ = make_bank(tmp_path / "first")
    assert torch.equal(before, torch.random.get_rng_state())
    previous_dtype = torch.get_default_dtype()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.rand(113)
            changed = torch.random.get_rng_state().clone()
            torch.set_default_dtype(torch.float64)
            second, loaded = make_bank(tmp_path / "second")
            assert torch.equal(changed, torch.random.get_rng_state())
            assert loaded.dataset("train")[0][0].dtype == torch.float32
    finally:
        torch.set_default_dtype(previous_dtype)
    assert first == second
    assert (tmp_path / "first" / "observations.f32").read_bytes() == (
        tmp_path / "second" / "observations.f32"
    ).read_bytes()


def test_cpu_generation_and_clear_loader_precondition_under_default_meta_device(
    tmp_path,
):
    previous_device = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        _, loaded = make_bank(tmp_path / "bank")
        assert loaded.dataset("train")[0][0].device.type == "cpu"
        assert torch.get_default_device().type == "meta"
        with pytest.raises(bank.EpisodeBankError, match="default device.*cpu"):
            loaded.loader("train")
    finally:
        torch.set_default_device(previous_device)
    assert next(iter(loaded.loader("train")))[0].device.type == "cpu"


def copied_sources(tmp_path, monkeypatch):
    source_root = tmp_path / "copied_sources"
    for name in bank.SOURCE_PATHS:
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bank.ROOT / name, target)
    monkeypatch.setattr(bank, "ROOT", source_root)
    return source_root / "fim_experiments/runtime_benchmarks.py"


def test_source_edit_after_import_rejects_creation_and_existing_bank_load(
    tmp_path, monkeypatch
):
    receipt, _ = make_bank(tmp_path / "existing")
    runtime = copied_sources(tmp_path, monkeypatch)
    runtime.write_text(runtime.read_text() + "\n# Source changed after import.\n")
    with pytest.raises(bank.EpisodeBankError, match="source changed since"):
        make_bank(tmp_path / "new")
    assert not (tmp_path / "new").exists()
    with pytest.raises(bank.EpisodeBankError, match="source changed since"):
        bank.load_development_bank(
            tmp_path / "existing", expected_bank_sha256=receipt["bank_sha256"]
        )


def test_source_edit_during_generation_retains_failure_before_data_publication(
    tmp_path, monkeypatch
):
    runtime = copied_sources(tmp_path, monkeypatch)
    original_rollout = DelayedRecallBenchmark.rollout

    def edit_after_rollout(self, *args, **kwargs):
        observations = original_rollout(self, *args, **kwargs)
        runtime.write_text(runtime.read_text() + "\n# Mid-generation checkout edit.\n")
        return observations

    monkeypatch.setattr(DelayedRecallBenchmark, "rollout", edit_after_rollout)
    path = tmp_path / "bank"
    with pytest.raises(bank.EpisodeBankError, match="source changed since"):
        make_bank(path)
    failure = json.loads((path / "failure.json").read_text())
    assert failure["stage"] == "source_freshness_after_generation"
    assert failure["status"] == "FAILED"
    assert not (path / "manifest.json").exists()
    assert not (path / "observations.f32").exists()


def test_source_edit_during_loading_is_detected_before_tensor_exposure(
    tmp_path, monkeypatch
):
    receipt, _ = make_bank(tmp_path / "bank")
    runtime = copied_sources(tmp_path, monkeypatch)
    original_read = bank._read_regular

    def edit_after_data_read(path, max_bytes):
        raw = original_read(path, max_bytes)
        if path.name == "observations.f32":
            runtime.write_text(runtime.read_text() + "\n# Mid-load checkout edit.\n")
        return raw

    monkeypatch.setattr(bank, "_read_regular", edit_after_data_read)
    with pytest.raises(bank.EpisodeBankError, match="source changed since"):
        bank.load_development_bank(
            tmp_path / "bank", expected_bank_sha256=receipt["bank_sha256"]
        )


@pytest.mark.parametrize("shuffle", [False, True])
def test_paired_loaders_use_local_order_and_preserve_global_rng(tmp_path, shuffle):
    receipt, first = make_bank(tmp_path / "bank", train_count=7)
    before = torch.random.get_rng_state().clone()
    a = list(first.loader("train", shuffle=shuffle, order_seed=43))
    assert torch.equal(before, torch.random.get_rng_state())
    with torch.random.fork_rng(devices=[]):
        torch.rand(273)
        second = bank.load_development_bank(
            tmp_path / "bank", expected_bank_sha256=receipt["bank_sha256"]
        )
        changed = torch.random.get_rng_state().clone()
        b = list(second.loader("train", shuffle=shuffle, order_seed=43))
        assert torch.equal(changed, torch.random.get_rng_state())
    assert len(a) == len(b) == 7
    for left, right in zip(a, b):
        x, y = _unpack_batch(left)
        xx, yy = _unpack_batch(right)
        assert x.shape == y.shape == (1, 12, 5)
        assert torch.equal(x, xx) and torch.equal(y, yy)
        assert torch.equal(x[:, 1:], y[:, :-1])
    assert first.loader("train").num_workers == 0
    assert first.loader("train").batch_size == 1


def test_independent_storage_and_manifest_copy_and_split_partition(tmp_path):
    _, loaded = make_bank(tmp_path / "bank")
    manifest = loaded.manifest
    ids = [index for name in bank.SPLITS for index in manifest["splits"][name]]
    assert sorted(ids) == list(range(5)) and len(set(ids)) == 5
    manifest["splits"]["train"].clear()
    assert len(loaded.dataset("train")) == 3
    dataset = loaded.dataset("train")
    x, y = dataset[0]
    original_x, original_y = x.clone(), y.clone()
    assert x.data_ptr() != y.data_ptr()
    x.fill_(float("nan"))
    assert torch.equal(y, original_y)
    next_x, next_y = dataset[0]
    assert torch.equal(next_x, original_x) and torch.equal(next_y, original_y)
    assert dataset.episode_indices == (0, 1, 2)


def test_explicit_loader_seed_changes_order_without_changing_membership(tmp_path):
    _, loaded = make_bank(tmp_path / "bank", train_count=9)

    def keys(seed):
        return [
            x.numpy().tobytes()
            for x, _ in loaded.loader("train", shuffle=True, order_seed=seed)
        ]

    a, b = keys(17), keys(19)
    assert a != b and set(a) == set(b) and len(a) == 9


@pytest.mark.parametrize("seed", sorted(bank.HELD_SEEDS) + [-1, True, 1.5, 2**63])
def test_invalid_or_held_seed_rejected_before_reservation(tmp_path, seed):
    with pytest.raises(bank.EpisodeBankError):
        bank.create_development_bank(tmp_path / "bank", seed=seed)
    assert not (tmp_path / "bank").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        dict(horizon=0),
        dict(horizon=65),
        dict(horizon=True),
        dict(train_count=0),
        dict(validation_count=True),
        dict(evaluation_count=-1),
        dict(train_count=127),
    ],
)
def test_shape_and_count_budget_rejected_before_reservation(tmp_path, arguments):
    with pytest.raises(bank.EpisodeBankError):
        bank.create_development_bank(tmp_path / "bank", seed=37, **arguments)
    assert not (tmp_path / "bank").exists()


def test_caller_pin_cannot_be_replaced_by_self_consistent_other_bank(tmp_path):
    first, _ = make_bank(tmp_path / "first")
    second, _ = make_bank(tmp_path / "second", seed=41)
    assert first["bank_sha256"] != second["bank_sha256"]
    with pytest.raises(bank.EpisodeBankError, match="pinned bank identity"):
        bank.load_development_bank(
            tmp_path / "second", expected_bank_sha256=first["bank_sha256"]
        )


@pytest.mark.parametrize(
    "mutation",
    ["byte", "truncate", "missing_data", "missing_manifest", "extra", "wrong_pin"],
)
def test_changed_or_incomplete_artifact_rejected(tmp_path, mutation):
    path = tmp_path / "bank"
    receipt, _ = make_bank(path)
    pin = receipt["bank_sha256"]
    data = path / "observations.f32"
    if mutation == "byte":
        raw = bytearray(data.read_bytes())
        raw[0] ^= 1
        data.write_bytes(raw)
    elif mutation == "truncate":
        data.write_bytes(data.read_bytes()[:-4])
    elif mutation == "missing_data":
        data.unlink()
    elif mutation == "missing_manifest":
        (path / "manifest.json").unlink()
    elif mutation == "extra":
        (path / "failure.json").write_text("retained failure")
    else:
        pin = "0" * 64
    with pytest.raises(bank.EpisodeBankError):
        bank.load_development_bank(path, expected_bank_sha256=pin)


@pytest.mark.parametrize(
    "mutation",
    [
        "overlap",
        "omission",
        "boolean_index",
        "clock",
        "nan",
        "duplicate_episode",
        "wrong_source",
        "wrong_shape",
        "wrong_phase",
        "wrong_filename",
        "large_noise",
        "config_dimension",
    ],
)
def test_resealed_invalid_semantics_rejected(tmp_path, mutation):
    path = tmp_path / "bank"
    make_bank(path)

    def transform(manifest):
        if mutation == "overlap":
            manifest["splits"]["validation"] = [0]
        elif mutation == "omission":
            manifest["splits"]["train"] = [0, 1]
        elif mutation == "boolean_index":
            manifest["splits"]["train"][0] = False
        elif mutation == "wrong_source":
            manifest["source_sha256"][bank.SOURCE_PATHS[0]] = "0" * 64
        elif mutation == "wrong_shape":
            manifest["array"]["shape"][1] += 1
        elif mutation == "wrong_phase":
            manifest["phase"] = "CONFIRMATORY"
        elif mutation == "wrong_filename":
            manifest["array"]["file"] = "../observations.f32"
        elif mutation == "large_noise":
            manifest["benchmark_config"]["cue_noise"] = 10**400
        elif mutation == "config_dimension":
            manifest["benchmark_config"]["dimension"] += 1

    raw = None
    if mutation in {"clock", "nan", "duplicate_episode"}:
        values = np.fromfile(path / "observations.f32", dtype="<f4").reshape(5, 13, 5)
        if mutation == "clock":
            values[0, 2, -1] = values[0, 1, -1]
        elif mutation == "nan":
            values[0, 0, 0] = np.nan
        else:
            values[1] = values[0]
        raw = values.tobytes()
    pin = rewrite_manifest(path, transform, raw=raw)
    with pytest.raises(ValueError):
        bank.load_development_bank(path, expected_bank_sha256=pin)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"seed":1,"seed":2}',
        b'{"seed":NaN}',
        b'{"seed":1e9999}',
        b"null",
        b"[1,2]",
        b"{" * 128001,
    ],
)
def test_malformed_or_oversized_manifest_rejected(tmp_path, raw):
    path = tmp_path / "bank"
    receipt, _ = make_bank(path)
    (path / "manifest.json").write_bytes(raw)
    with pytest.raises(bank.EpisodeBankError):
        bank.load_development_bank(path, expected_bank_sha256=receipt["bank_sha256"])


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_nonregular_data_rejected_without_following_or_blocking(tmp_path, kind):
    path = tmp_path / "bank"
    receipt, _ = make_bank(path)
    raw = path / "observations.f32"
    raw.rename(tmp_path / "moved-data")
    if kind == "symlink":
        raw.symlink_to(tmp_path / "moved-data")
    else:
        os.mkfifo(raw)
    with pytest.raises(bank.EpisodeBankError):
        bank.load_development_bank(path, expected_bank_sha256=receipt["bank_sha256"])


def test_no_overwrite_and_atomic_single_writer(tmp_path):
    path = tmp_path / "bank"

    def create():
        try:
            return make_bank(path)[0]
        except FileExistsError:
            return "destination already reserved"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: create(), range(2)))
    successes = [result for result in results if isinstance(result, dict)]
    assert len(successes) == 1 and results.count("destination already reserved") == 1
    before = {item.name: item.read_bytes() for item in path.iterdir()}
    with pytest.raises(FileExistsError):
        make_bank(path)
    assert before == {item.name: item.read_bytes() for item in path.iterdir()}
    bank.load_development_bank(path, expected_bank_sha256=successes[0]["bank_sha256"])


@pytest.mark.parametrize(
    "stage", ["data_write", "manifest_write", "directory_sync_after_installation"]
)
def test_failed_creation_retains_directory_and_failure_and_refuses_load(
    tmp_path, monkeypatch, stage
):
    path = tmp_path / "bank"
    original_write = bank._write_exclusive
    original_sync = bank._fsync_directory
    injected = False

    def fail_write(target, raw):
        nonlocal injected
        selected = target.name == (
            "observations.f32" if stage == "data_write" else "manifest.pending"
        )
        if stage != "directory_sync_after_installation" and selected and not injected:
            injected = True
            raise OSError("injected artifact write failure")
        return original_write(target, raw)

    def fail_sync(target):
        nonlocal injected
        if (
            stage == "directory_sync_after_installation"
            and target == path
            and not injected
        ):
            injected = True
            raise OSError("injected directory fsync failure after installation")
        return original_sync(target)

    monkeypatch.setattr(bank, "_write_exclusive", fail_write)
    monkeypatch.setattr(bank, "_fsync_directory", fail_sync)
    with pytest.raises(OSError, match="injected"):
        make_bank(path)
    failure = json.loads((path / "failure.json").read_text())
    assert failure["status"] == "FAILED" and failure["stage"] == stage
    assert (path / "manifest.json").exists() == (
        stage == "directory_sync_after_installation"
    )
    with pytest.raises(bank.EpisodeBankError, match="incomplete, failed"):
        bank.load_development_bank(path, expected_bank_sha256="0" * 64)
    with pytest.raises(FileExistsError):
        make_bank(path)


def test_real_cli_create_verify_demo_and_explicit_nonzero_failures(tmp_path, capsys):
    path = tmp_path / "created"
    assert (
        bank.main(["create", "--output", str(path), "--seed", "37", "--horizon", "4"])
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert (
        bank.main(
            ["verify", "--bank", str(path), "--expected-sha256", receipt["bank_sha256"]]
        )
        == 0
    )
    assert (
        json.loads(capsys.readouterr().out)["status"]
        == "VERIFIED_DEVELOPMENT_EPISODE_BANK"
    )
    assert (
        bank.main(["verify", "--bank", str(path), "--expected-sha256", "0" * 64]) == 2
    )
    assert "pinned bank identity" in capsys.readouterr().err
    assert bank.main(["demo", "--output", str(tmp_path / "demo")]) == 0
    demo = json.loads(capsys.readouterr().out)
    assert demo["paired_rows"] == 3 and demo["shape"] == [5, 13, 5]
    assert demo["model_training_runs"] == demo["scientific_outcome_runs"] == 0
    assert json.loads((tmp_path / "demo" / "demo_receipt.json").read_text()) == demo
    assert bank.main(["demo", "--output", str(tmp_path / "demo")]) == 2
    assert "FileExistsError" in capsys.readouterr().err
