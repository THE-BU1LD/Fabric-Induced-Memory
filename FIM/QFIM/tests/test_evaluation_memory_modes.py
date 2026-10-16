from __future__ import annotations

import torch

from fim_experiments.main import FIMSystem
from results.run_results import _autoregressive_rollout, _retrieval_utilization


def test_full_rollout_stores_and_retrieves_traces():
    torch.manual_seed(0)
    model = FIMSystem(
        in_channels=1,
        hidden=8,
        trace_dim=4,
        memory_capacity=16,
        retrieval_topk=4,
        salience_threshold=-1.0,
    )
    x0 = torch.randn(2, 6)

    _, _, retrieved, _ = _autoregressive_rollout(
        model,
        x0,
        steps=4,
        device=torch.device("cpu"),
        store_traces=True,
        retrieve=True,
    )

    assert _retrieval_utilization(retrieved) > 0.0


def test_no_memory_rollout_never_retrieves():
    torch.manual_seed(0)
    model = FIMSystem(
        in_channels=1,
        hidden=8,
        trace_dim=4,
        memory_capacity=16,
        retrieval_topk=4,
        salience_threshold=-1.0,
    )
    x0 = torch.randn(2, 6)

    _, _, retrieved, _ = _autoregressive_rollout(
        model,
        x0,
        steps=4,
        device=torch.device("cpu"),
        store_traces=False,
        retrieve=False,
    )

    assert _retrieval_utilization(retrieved) == 0.0
    assert len(model.bank) == 0
