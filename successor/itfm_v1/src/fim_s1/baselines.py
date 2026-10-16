from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ResidualSpatialBlock
from .transport import apply_periodic_translation, estimate_periodic_translation


class PeriodicTranslationPredictor(nn.Module):
    """Parameter-free periodic extrapolator from the final two observed frames."""

    def __init__(
        self,
        *,
        input_channels: int,
        context_length: int,
        max_shift: int = 2,
    ) -> None:
        super().__init__()
        if input_channels < 1 or context_length < 2 or max_shift < 0:
            raise ValueError("invalid periodic-translation dimensions")
        self.input_channels = int(input_channels)
        self.context_length = int(context_length)
        self.max_shift = int(max_shift)

    @torch.no_grad()
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        expected_channels = self.input_channels * self.context_length
        if context.ndim != 4 or context.shape[1] != expected_channels:
            raise ValueError(
                f"expected [B,{expected_channels},H,W], got {tuple(context.shape)}"
            )
        previous = context[:, -2 * self.input_channels : -self.input_channels]
        current = context[:, -self.input_channels :]
        shifts = estimate_periodic_translation(
            previous,
            current,
            max_shift=self.max_shift,
        )
        return apply_periodic_translation(current, shifts)


class FiniteHistoryCNN(nn.Module):
    def __init__(
        self,
        *,
        context_channels: int,
        output_channels: int,
        hidden_channels: int = 48,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(context_channels, hidden_channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualSpatialBlock(hidden_channels) for _ in range(depth)]
        )
        self.head = nn.Conv2d(hidden_channels, output_channels, 3, padding=1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.stem(context))
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        joined = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(joined, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(joined, hidden_channels, 3, padding=1)

    def forward(self, item: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        reset, update = torch.sigmoid(
            self.gates(torch.cat([item, state], dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([item, reset * state], dim=1)))
        return (1.0 - update) * state + update * candidate


class ConvGRUPredictor(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        context_length: int,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.context_length = int(context_length)
        self.hidden_channels = int(hidden_channels)
        self.cell = ConvGRUCell(input_channels, hidden_channels)
        self.head = nn.Conv2d(hidden_channels, output_channels, 3, padding=1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        expected_channels = self.input_channels * self.context_length
        if context.ndim != 4 or context.shape[1] != expected_channels:
            raise ValueError(
                f"expected {expected_channels} stacked context channels, got {tuple(context.shape)}"
            )
        batch, _, height, width = context.shape
        state = context.new_zeros((batch, self.hidden_channels, height, width))
        frames = context.reshape(
            batch, self.context_length, self.input_channels, height, width
        )
        for frame_index in range(self.context_length):
            state = self.cell(frames[:, frame_index], state)
        return self.head(state)


class GatedDiagonalSSMPredictor(nn.Module):
    """Stable input-selective diagonal state-space baseline over context frames."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        context_length: int,
        hidden_channels: int = 32,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if (
            min(input_channels, output_channels, context_length, hidden_channels, depth)
            < 1
        ):
            raise ValueError("SSM dimensions must be positive")
        self.input_channels = int(input_channels)
        self.context_length = int(context_length)
        self.hidden_channels = int(hidden_channels)
        self.input_projection = nn.Conv2d(input_channels, hidden_channels, 1)
        self.step_projection = nn.Conv2d(input_channels, hidden_channels, 1)
        self.decay_log = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        self.blocks = nn.ModuleList(
            [ResidualSpatialBlock(hidden_channels) for _ in range(depth)]
        )
        self.head = nn.Conv2d(hidden_channels, output_channels, 3, padding=1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        expected_channels = self.input_channels * self.context_length
        if context.ndim != 4 or context.shape[1] != expected_channels:
            raise ValueError(
                f"expected {expected_channels} stacked context channels, got {tuple(context.shape)}"
            )
        batch, _, height, width = context.shape
        state = context.new_zeros((batch, self.hidden_channels, height, width))
        frames = context.reshape(
            batch, self.context_length, self.input_channels, height, width
        )
        rate = F.softplus(self.decay_log).clamp_min(1e-6)
        for frame_index in range(self.context_length):
            frame = frames[:, frame_index]
            candidate = self.input_projection(frame)
            step = F.softplus(self.step_projection(frame)).clamp_max(20.0)
            decay = torch.exp(-rate * step)
            state = decay * state + (1.0 - decay) * candidate
        for block in self.blocks:
            state = block(state)
        return self.head(state)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def closest_width(
    factory,
    target_parameters: int,
    *,
    minimum: int = 4,
    maximum: int = 128,
) -> tuple[nn.Module, int]:
    """Select the declared integer width closest to a target parameter count."""

    best: tuple[int, int, int, nn.Module] | None = None
    for width in range(minimum, maximum + 1):
        model = factory(width)
        count = trainable_parameter_count(model)
        candidate = (abs(count - target_parameters), count, width, model)
        if best is None or (candidate[0], candidate[2]) < (best[0], best[2]):
            best = candidate
    if best is None:
        raise RuntimeError("no baseline width candidates were generated")
    _, _, width, model = best
    return model, width
