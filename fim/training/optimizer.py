from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterable, Tuple

import torch


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    decay = []
    no_decay = []

    for p in parameters:
        if not p.requires_grad:
            continue
        if p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        fused=torch.cuda.is_available(),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_scale: float = 0.1,
):
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))

        progress = (step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def clip_gradients(
    model: torch.nn.Module,
    max_norm: float = 1.0,
):
    total_norm = torch.norm(
        torch.stack(
            [
                p.grad.detach().norm(2)
                for p in model.parameters()
                if p.grad is not None
            ]
        ),
        2,
    )

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(clip_coef)


class EMA:
    """Average trainable parameters; episode-memory buffers are not weights."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        if not math.isfinite(self.decay) or not 0.0 <= self.decay <= 1.0:
            raise ValueError("EMA decay must be finite and in [0, 1]")
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
            if value.requires_grad
        }

    def _parameters(self, model: torch.nn.Module):
        parameters = {
            name: value for name, value in model.named_parameters()
            if value.requires_grad
        }
        if parameters.keys() != self.shadow.keys():
            raise ValueError("EMA trainable parameter names changed after initialization")
        for name, value in parameters.items():
            shadow = self.shadow[name]
            if value.shape != shadow.shape or value.device != shadow.device or value.dtype != shadow.dtype:
                raise ValueError(f"EMA parameter contract changed: {name}")
        return parameters

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, value in self._parameters(model).items():
            self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module):
        for name, value in self._parameters(model).items():
            value.copy_(self.shadow[name])

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        """Temporarily apply EMA weights and restore raw weights on every exit."""
        parameters = self._parameters(model)
        original = {name: value.detach().clone() for name, value in parameters.items()}
        try:
            self.apply_to(model)
            yield
        finally:
            with torch.no_grad():
                for name, value in parameters.items():
                    value.copy_(original[name])
