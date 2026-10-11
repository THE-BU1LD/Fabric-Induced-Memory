"""Opt-in, observed-input validation on an owned public FIMModel episode copy.

This helper never replaces the frozen experiment evaluator. Each invocation is
one independent episode and returns only existing per-step loss diagnostics.
"""

from __future__ import annotations

import copy

import torch

from fim.models.fim_model import FIMModel
from fim.training.trainer import Trainer, TrainStats


def _admit_episode(
    trainer: Trainer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: str,
) -> None:
    if type(trainer) is not Trainer or type(trainer.model) is not FIMModel:
        raise TypeError("v1 requires the concrete public Trainer and FIMModel")
    if weights not in ("raw", "ema"):
        raise ValueError("weights must explicitly be 'raw' or 'ema'")
    if weights == "ema" and trainer.ema is None:
        raise ValueError("EMA weights requested but the trainer has no EMA")
    if trainer.device.type != "cpu":
        raise ValueError("isolated evaluation v1 supports CPU only")

    parameters = tuple(trainer.model.parameters())
    if not parameters or any(
        p.device.type != "cpu" or p.dtype not in (torch.float32, torch.float64)
        for p in parameters
    ):
        raise ValueError("model parameters must be float32 or float64 on CPU")
    dtype = parameters[0].dtype
    if any(p.dtype != dtype for p in parameters):
        raise ValueError("model parameters must have a common floating dtype")
    for module in trainer.model.modules():
        if any(
            getattr(module, name, {})
            for name in (
                "_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
                "_backward_pre_hooks",
            )
        ):
            raise ValueError("isolated evaluation v1 does not accept execution hooks")
    if any(value.device.type != "cpu" for value in trainer.model.buffers()):
        raise ValueError("model buffers must be on CPU")

    for label, value, channels in (
        ("inputs", inputs, trainer.model.in_channels),
        ("targets", targets, trainer.model.out_channels),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label} must be a tensor")
        if value.ndim != 5 or value.shape[0] != 1 or value.shape[2] != channels:
            raise ValueError(f"{label} must be [1, T, {channels}, H, W]")
        if value.numel() == 0:
            raise ValueError(f"{label} must contain a nonempty episode")
        if value.layout != torch.strided or value.device.type != "cpu" or value.dtype != dtype:
            raise ValueError(f"{label} must be a dense CPU tensor matching model dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} must be finite")
    if inputs.shape[1] != targets.shape[1] or inputs.shape[-2:] != targets.shape[-2:]:
        raise ValueError("inputs and targets must have identical time and spatial dimensions")


@torch.no_grad()
def evaluate_isolated_episode_v1(
    trainer: Trainer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: str,
) -> tuple[TrainStats, ...]:
    """Evaluate one independent episode without advancing the training instance.

    ``inputs`` and ``targets`` are finite dense CPU tensors ``[1,T,C,H,W]``.
    Each input frame is observed; this is not an autoregressive rollout metric.
    Memory is cleared once, then remains available within the copied episode.
    Choose ``weights='raw'`` or ``weights='ema'`` explicitly. The caller must not
    mutate the model or EMA concurrently. One full model copy is required.
    """
    _admit_episode(trainer, inputs, targets, weights)
    with torch.random.fork_rng(devices=[]):
        # Cached training state can be a live, nonleaf autograd tensor, which
        # torch does not deepcopy. It must not enter an independent episode.
        memo = {} if trainer.model._state is None else {id(trainer.model._state): None}
        model = copy.deepcopy(trainer.model, memo)
        model.reset_state(clear_memory=True)
        model.eval()
        if weights == "ema":
            trainer.ema.apply_to(model)

        # Reuse the maintained loss interpretation without touching the owner's
        # model or applying/restoring EMA weights in-place on its parameters.
        evaluator = copy.copy(trainer)
        evaluator.model = model
        evaluator.ema = None
        return tuple(
            evaluator.eval_step((inputs[:, step], targets[:, step]))
            for step in range(inputs.shape[1])
        )
