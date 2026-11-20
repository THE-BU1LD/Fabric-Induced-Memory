"""Independent causal retrieval regressions; run through the admitted test CLI."""

import dataclasses
import math
import os

import pytest
import torch

from fim_sota.adapter import (
    ChannelCalibrator,
    ContextKeyEncoder,
    ITFMRAdapter,
    MonotoneSimilarityGate,
    ResidualNormalizer,
)
from fim_sota.contracts import BackboneOutput, ForecastBatch
from fim_sota.memory import TensorMemoryBank


def memory(*, topk=2, temperature=0.01, chunk_size=1):
    return TensorMemoryBank(
        key_dimension=2,
        residual_shape=(1,),
        capacity_per_trajectory=8,
        minimum_age=1,
        topk=topk,
        temperature=temperature,
        chunk_size=chunk_size,
        device=os.environ.get("FIM_TEST_DEVICE", "cpu"),
    )


@pytest.mark.parametrize("approximate", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 8])
def test_masked_softmax_preserves_single_eligible_negative_match(
    approximate, chunk_size
):
    bank = memory(chunk_size=chunk_size)
    bank.write_batch(
        keys=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        residuals=torch.tensor([[7.0], [99.0]]),
        trajectory_ids=("a", "a"),
        event_times=torch.tensor([0, 10]),
        available_times=torch.tensor([1, 11]),
    )
    result = bank.retrieve_batch(
        query_keys=torch.tensor([[-1.0, 0.0], [-1.0, 0.0]]),
        trajectory_ids=("a", "unknown"),
        query_times=torch.tensor([2, 2]),
        approximate=approximate,
    )
    assert result.valid.tolist() == [True, False]
    assert result.residual[:, 0].tolist() == [7.0, 0.0]
    assert result.eligible_count.tolist() == [1, 0]
    assert bool(torch.isfinite(result.residual).all())


@pytest.mark.parametrize("approximate", [False, True])
def test_value_shuffle_uses_only_mature_same_owner_donors(approximate):
    def populated(poison):
        bank = memory(topk=1)
        bank.write_batch(
            keys=torch.tensor([[1.0, 0.0]] * 4),
            residuals=torch.tensor([[3.0], [9.0], [70.0 + poison], [99.0 - poison]]),
            trajectory_ids=("a", "a", "b", "a"),
            event_times=torch.tensor([0, 1, 0, 10]),
            available_times=torch.tensor([1, 2, 1, 11]),
        )
        return bank

    arguments = dict(
        query_keys=torch.tensor([[1.0, 0.0]] * 3),
        trajectory_ids=("a", "a", "unknown"),
        query_times=torch.tensor([2, 1, 2]),
        selection="random_value",
        control_seed=0,
        approximate=approximate,
    )
    bank = populated(0)
    before = bank.checksum()
    result = bank.retrieve_batch(**arguments)
    # The first query swaps the two mature A values. The second has one donor.
    assert result.residual[:, 0].tolist() == [9.0, 3.0, 0.0]
    assert result.eligible_count.tolist() == [2, 1, 0]
    assert result.source_serial.tolist() == [1, 0, -1]
    assert result.source_event_time.tolist() == [1, 0, -1]
    assert result.source_available_time.tolist() == [2, 1, -1]
    assert result.key_source_serial.tolist() == [0, 0, -1]
    assert torch.equal(
        result.residual, populated(1000).retrieve_batch(**arguments).residual
    )
    assert torch.equal(result.residual, bank.retrieve_batch(**arguments).residual)
    assert bank.checksum() == before


@pytest.mark.parametrize("control_seed", [0, 1, 11])
def test_value_shuffle_preserves_eligible_multiset_and_reports_actual_donors(
    control_seed,
):
    bank = memory(topk=3, temperature=0.2)
    bank.write_batch(
        keys=torch.tensor([[1.0, 0.0]] * 4),
        residuals=torch.tensor([[2.0], [5.0], [11.0], [1000.0]]),
        trajectory_ids=("a", "a", "a", "other"),
        event_times=torch.tensor([0, 1, 2, 0]),
        available_times=torch.tensor([1, 2, 3, 1]),
    )
    result = bank.retrieve_batch(
        query_keys=torch.tensor([[1.0, 0.0]]),
        trajectory_ids=("a",),
        query_times=torch.tensor([4]),
        selection="random_value",
        control_seed=control_seed,
    )
    assert result.residual.item() == pytest.approx(6.0)
    assert sorted(result.value_source_serials[0].tolist()) == [0, 1, 2]
    assert bool((result.value_source_available_times <= 4).all())
    assert result.source_serial.item() in {0, 1, 2}


@pytest.mark.parametrize("approximate", [False, True])
def test_value_shuffle_preserves_selected_key_weight_to_donor_pairing(approximate):
    values = [2.0, 5.0, 11.0]
    bank = memory(topk=2, temperature=0.2)
    bank.write_batch(
        keys=torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]),
        residuals=torch.tensor(values).reshape(-1, 1),
        trajectory_ids=("a", "a", "a"),
        event_times=torch.tensor([0, 1, 2]),
        available_times=torch.tensor([1, 2, 3]),
    )
    result = bank.retrieve_batch(
        query_keys=torch.tensor([[1.0, 0.0]]),
        trajectory_ids=("a",),
        query_times=torch.tensor([4]),
        selection="random_value",
        control_seed=0,
        approximate=approximate,
    )
    # Key scores 1 and .8 keep their unequal weights; serials 2 and 0 donate.
    expected = (11.0 + math.exp(-1.0) * 2.0) / (1.0 + math.exp(-1.0))
    assert result.key_source_serial.item() == 0
    assert result.value_source_serials.tolist() == [[2, 0]]
    assert result.source_serial.item() == 2
    assert result.residual.item() == pytest.approx(expected, rel=2e-6)
    reconstructed = sum(
        values[serial] * weight
        for serial, weight in zip(
            result.value_source_serials[0].tolist(),
            result.value_weights[0].tolist(),
            strict=True,
        )
    )
    assert reconstructed == pytest.approx(expected, rel=2e-6)


def test_approximate_reranking_honors_stored_masks_without_query_mask():
    bank = memory(topk=1)
    bank.write_batch(
        keys=torch.tensor([[1.0, -100.0], [1.0, 0.0]]),
        key_masks=torch.tensor([[True, False], [True, True]]),
        residuals=torch.tensor([[7.0], [99.0]]),
        trajectory_ids=("a", "a"),
        event_times=torch.tensor([0, 0]),
        available_times=torch.tensor([1, 1]),
    )
    args = dict(
        query_keys=torch.tensor([[1.0, 1.0]]),
        trajectory_ids=("a",),
        query_times=torch.tensor([2]),
    )
    exact = bank.retrieve_batch(**args)
    approximate = bank.retrieve_batch(**args, approximate=True)
    assert exact.residual.item() == 7.0
    # Every entry is a candidate: no candidate approximation can explain a change.
    assert torch.equal(approximate.residual, exact.residual)
    assert torch.equal(approximate.similarity, exact.similarity)


@pytest.mark.parametrize("chunk_size", [1, 8])
def test_retrieval_matches_independent_python_kernel_reference(chunk_size):
    keys = [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [1.0, 0.0]]
    values = [2.0, 5.0, 11.0, 1000.0]
    bank = memory(topk=4, temperature=0.1, chunk_size=chunk_size)
    bank.write_batch(
        keys=torch.tensor(keys),
        residuals=torch.tensor(values).reshape(-1, 1),
        trajectory_ids=("a", "a", "a", "other"),
        event_times=torch.tensor([0, 1, 2, 0]),
        available_times=torch.tensor([1, 2, 3, 1]),
    )
    scores = [-k[0] / math.hypot(*k) for k in keys[:3]]
    masses = [math.exp((score - max(scores)) / 0.1) for score in scores]
    expected = sum(w * v for w, v in zip(masses, values[:3], strict=True)) / sum(masses)
    result = bank.retrieve_batch(
        query_keys=torch.tensor([[-1.0, 0.0]]),
        trajectory_ids=("a",),
        query_times=torch.tensor([4]),
    )
    assert result.residual.item() == pytest.approx(expected, rel=2e-6)


def adapter_fixture():
    bank = TensorMemoryBank(
        key_dimension=2,
        residual_shape=(1,),
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
        normalizer=ResidualNormalizer(1),
        calibrator=ChannelCalibrator(1),
        gate=MonotoneSimilarityGate(),
    )
    batch = ForecastBatch(
        context=torch.ones(1, 2),
        target=torch.ones(1, 1),
        observation_mask=torch.ones(1, 2),
        trajectory_ids=("a",),
        event_times=torch.tensor([0]),
        lead_times=torch.tensor([5]),
        split="validation",
        target_observation_mask=torch.ones(1, 1),
    )
    output = BackboneOutput(
        prediction=torch.zeros(1, 1),
        latent_key=torch.ones(1, 2),
        observation_mask=batch.observation_mask,
    )
    return adapter, batch, output


def test_public_writer_requires_explicit_arrival_and_waits_for_delayed_target():
    adapter, batch, output = adapter_fixture()
    before = adapter.bank.checksum()
    with pytest.raises(ValueError, match=r"explicit.*availab"):
        adapter.write_observed(output, batch, observation_time=1)
    assert adapter.bank.checksum() == before
    delayed = dataclasses.replace(batch, target_available_times=torch.tensor([7]))
    assert adapter.write_observed(output, delayed, observation_time=6) == 0
    assert adapter.bank.checksum() == before
    assert adapter.write_observed(output, delayed, observation_time=7) == 1
    assert adapter.bank.available_times.tolist() == [7]


@pytest.mark.parametrize("arrival", [1.5, float("nan"), float("inf")])
def test_memory_rejects_ambiguous_noninteger_arrival_before_mutation(arrival):
    bank = memory()
    before = bank.checksum()
    with pytest.raises(ValueError, match="integer"):
        bank.write_batch(
            keys=torch.ones(1, 2),
            residuals=torch.ones(1, 1),
            trajectory_ids=("a",),
            event_times=torch.tensor([0]),
            available_times=torch.tensor([arrival]),
        )
    assert bank.checksum() == before
