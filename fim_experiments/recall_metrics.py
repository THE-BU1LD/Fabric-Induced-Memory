from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RecallWindowSpec:
    """Pre-outcome specification for a delayed-recall evaluation window."""

    delay: int = 8
    window: int = 4
    memory_dim: int = 32

    def __post_init__(self) -> None:
        if self.delay < 1:
            raise ValueError("delay must be >= 1 because target index 0 corresponds to observation t=1")
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.memory_dim < 1:
            raise ValueError("memory_dim must be >= 1")

    @property
    def observation_times(self) -> tuple[int, ...]:
        return tuple(range(self.delay, self.delay + self.window))

    def target_indices(self, target_steps: int) -> tuple[int, ...]:
        if target_steps < 1:
            raise ValueError("target_steps must be >= 1")
        start = self.delay - 1
        stop = start + self.window
        if stop > target_steps:
            raise ValueError(
                "target sequence is too short for the complete frozen recall window: "
                f"need target_steps >= {stop}, received {target_steps}"
            )
        return tuple(range(start, stop))

    def target_slice(self, target_steps: int) -> slice:
        indices = self.target_indices(target_steps)
        return slice(indices[0], indices[-1] + 1)


def recall_window_memory_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    spec: RecallWindowSpec,
) -> torch.Tensor:
    """MSE over the frozen recall window and memory-payload channels only.

    DelayedRecallBenchmark.generate_batch returns targets whose index j corresponds
    to observation time t=j+1. The final clock feature is intentionally excluded:
    the endpoint is about recalling the hidden payload, not predicting the clock.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes must match, got {prediction.shape} and {target.shape}"
        )
    if prediction.ndim != 3:
        raise ValueError(
            "recall_window_memory_mse expects [batch, time, feature] tensors; "
            f"received rank {prediction.ndim}"
        )
    if prediction.shape[-1] <= spec.memory_dim:
        raise ValueError(
            "expected a clock feature after the memory payload; "
            f"feature_dim={prediction.shape[-1]} memory_dim={spec.memory_dim}"
        )

    window = spec.target_slice(int(prediction.shape[1]))
    delta = prediction[:, window, : spec.memory_dim] - target[:, window, : spec.memory_dim]
    return delta.square().mean()


def delayed_recall_observation_noise_floor_mse(distractor_scale: float) -> float:
    """Conditional-mean MSE floor from fresh post-cue Gaussian distractor noise.

    For t>0 the benchmark adds sigma * N(0, I) independently to every memory
    feature. Conditional on the latent memory and clock, the best mean predictor
    therefore has expected per-memory-feature MSE sigma^2.
    """

    sigma = float(distractor_scale)
    if not torch.isfinite(torch.tensor(sigma)):
        raise ValueError("distractor_scale must be finite")
    if sigma < 0.0:
        raise ValueError("distractor_scale must be non-negative")
    return sigma * sigma


__all__ = [
    "RecallWindowSpec",
    "recall_window_memory_mse",
    "delayed_recall_observation_noise_floor_mse",
]
