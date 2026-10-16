from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fim_sota.datasets import (
    DatasetAdmissionError,
    NavierStokesLoader,
    detect_split_overlap,
    schema_only_fixture_manifest,
    temporally_consistent_masks,
)


def test_split_overlap_is_rejected() -> None:
    with pytest.raises(DatasetAdmissionError, match="overlap"):
        detect_split_overlap({"train": ("a",), "validation": ("a",), "test": ("b",)})


def test_masks_are_temporally_consistent_by_trajectory() -> None:
    masks = temporally_consistent_masks(
        ("a", "a", "b"),
        (1, 8, 8),
        missing_percent=25,
        pattern="patch_wise",
        seed=11,
    )
    assert torch.equal(masks[0], masks[1])
    assert not torch.equal(masks[0], masks[2])
    assert masks.dtype == torch.bool


def test_schema_only_fixture_is_loaded_but_never_result_admissible(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "fixture.npz"
    np.savez(
        shard,
        context=np.ones((3, 2, 1, 4, 4), dtype=np.float32),
        target=np.ones((3, 1, 4, 4), dtype=np.float32),
        trajectory_id=np.array(["train", "validation", "test"]),
        event_time=np.array([0, 0, 0], dtype=np.int64),
    )
    manifest = schema_only_fixture_manifest(
        benchmark="pobench_pde",
        dataset="navier_stokes",
        source_file=shard,
        splits={
            "train": ("train",),
            "validation": ("validation",),
            "test": ("test",),
        },
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(dataclasses.asdict(manifest)))
    dataset = NavierStokesLoader(
        root=tmp_path,
        manifest_path=manifest_path,
        split="train",
        mask_pattern="point_wise",
        missing_percent=25,
        mask_seed=3,
    )
    sample = dataset[0]
    assert sample["provenance"].schema_only_fixture
    assert sample["observed_context"].shape == sample["context"].shape
