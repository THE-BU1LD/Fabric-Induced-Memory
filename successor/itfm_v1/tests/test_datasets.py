from pathlib import Path

import pytest
import torch

from fim_s1.data import AliasedAdvectionCorpus
from fim_s1.datasets import HDF5TrajectoryStore, PDEBenchRegistry, trajectory_windows


def test_staged_pdebench_registry_is_valid() -> None:
    registry = PDEBenchRegistry(
        Path("datasets/external/pdebench/pdebench_data_urls.csv")
    )
    advection = registry.select(
        equation="Advection", filename="1D_Advection_Sols_beta0.4.hdf5"
    )
    assert len(advection) == 1
    assert advection[0].md5 == "d595bbfd2c659df995a93cd40d6ea568"


def test_trajectory_windows_support_one_and_three_spatial_dimensions() -> None:
    one_dimensional = torch.randn(6, 2, 8)
    three_dimensional = torch.randn(6, 1, 2, 3, 4)
    assert trajectory_windows(0, one_dimensional, context_length=2)[
        0
    ].context.shape == (
        4,
        8,
    )
    assert trajectory_windows(1, three_dimensional, context_length=2)[
        0
    ].context.shape == (2, 2, 3, 4)


def test_corpus_trajectory_export_is_copy_safe() -> None:
    corpus = AliasedAdvectionCorpus(
        trajectory_count=6,
        time_steps=7,
        height=4,
        width=4,
        context_length=2,
    )
    exported = corpus.trajectories("train")
    if not exported:
        pytest.fail("training split unexpectedly empty")
    owner = next(iter(exported))
    exported[owner].zero_()
    assert torch.count_nonzero(corpus.trajectories("train")[owner]) > 0


def test_hdf5_axis_contract_normalizes_and_rejects_collisions() -> None:
    valid = HDF5TrajectoryStore(
        Path("unused.h5"),
        dataset_key="data",
        sample_axis=0,
        time_axis=-1,
        channel_axis=1,
    )
    assert valid._normalized_axes(4) == (0, 3, 1)
    invalid = HDF5TrajectoryStore(
        Path("unused.h5"),
        dataset_key="data",
        sample_axis=0,
        time_axis=-1,
        channel_axis=3,
    )
    with pytest.raises(ValueError, match="distinct"):
        invalid._normalized_axes(4)
