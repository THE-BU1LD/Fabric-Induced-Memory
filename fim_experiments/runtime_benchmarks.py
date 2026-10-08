"""Opt-in native runtime successors, separate from the frozen scientific source.

The maintained experiment CLI continues to use :mod:`fim_experiments.benchmark`.
Importing these classes does not authorize any scientific or outcome run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .benchmark import UnifiedBenchmark, _as_device


@dataclass
class LevyStableConfig:
    alpha: float = 1.5
    beta: float = 0.0
    scale: float = 1.0
    loc: float = 0.0
    dim: int = 1
    steps: int = 100


class LevyStableProcess(UnifiedBenchmark):
    def __init__(self, config: LevyStableConfig) -> None:
        super().__init__()
        self._validate_config(config)
        self.config = config
        self.dt = 1.0
        self.default_steps = config.steps
        self.dimension = config.dim
        self.state_shape = (config.dim,)
        self.stochastic = True

    @staticmethod
    def _validate_config(config: LevyStableConfig) -> None:
        if not math.isfinite(float(config.alpha)) or not (0 < float(config.alpha) <= 2):
            raise ValueError("alpha must be finite and in (0, 2].")
        if not math.isfinite(float(config.beta)) or not (-1 <= float(config.beta) <= 1):
            raise ValueError("beta must be finite and in [-1, 1].")
        if not math.isfinite(float(config.scale)) or float(config.scale) <= 0:
            raise ValueError("scale must be finite and positive.")
        if not math.isfinite(float(config.loc)):
            raise ValueError("loc must be finite.")
        if isinstance(config.dim, bool) or not isinstance(config.dim, int) or config.dim < 1:
            raise ValueError("dim must be a positive integer.")
        if isinstance(config.steps, bool) or not isinstance(config.steps, int) or config.steps < 1:
            raise ValueError("steps must be a positive integer.")

    @staticmethod
    def _validate_batch_size(batch_size: int) -> int:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        return batch_size

    @staticmethod
    def _validate_steps(steps: int) -> int:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer.")
        return steps

    def _validate_state(self, state: torch.Tensor) -> None:
        if state.ndim != 2 or state.shape[1] != self.dimension:
            raise ValueError(f"state must have shape [batch, {self.dimension}]")
        if state.shape[0] < 1:
            raise ValueError("state batch dimension must be positive.")
        if not state.is_floating_point():
            raise TypeError("state must use a floating-point dtype.")

    def sample_increment(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        batch_size = self._validate_batch_size(batch_size)
        if not getattr(dtype, "is_floating_point", False):
            raise TypeError("dtype must be a real floating-point torch dtype.")
        a = float(self.config.alpha)
        # Skewness is mathematically unidentified at alpha=2.  Canonicalize it
        # exactly instead of allowing tan(pi) roundoff to leak beta into samples.
        b = 0.0 if a == 2.0 else float(self.config.beta)
        scale = float(self.config.scale)
        device = _as_device(device)

        U = torch.empty(batch_size, self.config.dim, device=device, dtype=dtype).uniform_(
            -math.pi / 2, math.pi / 2
        )
        W = torch.empty(batch_size, self.config.dim, device=device, dtype=dtype).exponential_()

        if abs(a - 1.0) > 1e-6:
            tan_term = math.tan(math.pi * a / 2)
            phi = math.atan(b * tan_term) / a
            S = (1 + (b * tan_term) ** 2) ** (1 / (2 * a))
            num = torch.sin(a * (U + phi))
            den = torch.cos(U).clamp_min(1e-8) ** (1 / a)
            frac = num / den
            term = (torch.cos(U - a * (U + phi)).clamp_min(1e-8) / (W + 1e-8)) ** ((1 - a) / a)
            X = S * frac * term
        else:
            X = (2 / math.pi) * (
                (math.pi / 2 + b * U) * torch.tan(U)
                - b * torch.log(
                    (
                        (math.pi / 2) * W * torch.cos(U).clamp_min(1e-8)
                    )
                    / (math.pi / 2 + b * U).clamp_min(1e-8)
                    + 1e-8
                )
            )

        return scale * X + self.config.loc

    def step(self, state: torch.Tensor) -> torch.Tensor:
        self._validate_state(state)
        return state + self.sample_increment(state.shape[0], device=state.device, dtype=state.dtype)

    def sample_initial_state(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        batch_size = self._validate_batch_size(batch_size)
        device = _as_device(device)
        return torch.zeros(batch_size, self.config.dim, device=device)

    def rollout(self, x0: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        self._validate_state(x0)
        steps = self.default_steps if steps is None else self._validate_steps(steps)
        traj = [x0]
        x = x0
        for _ in range(steps):
            x = self.step(x)
            traj.append(x)
        return torch.stack(traj, dim=1)

    def generate_batch(
        self,
        batch_size: int,
        steps: Optional[int] = None,
        device: str | torch.device = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.sample_initial_state(batch_size, device=device)
        traj = self.rollout(x0, steps=steps)
        return traj[:, :-1], traj[:, 1:]

    def metrics(self) -> Dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "name": "LevyStableProcess",
                "state_shape": self.state_shape,
                "horizon": self.default_steps,
                "is_non_markovian": False,
                "stochastic": True,
                "dt": 1.0,
                "dimension": self.dimension,
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "scale": self.config.scale,
                "loc": self.config.loc,
                "stable_parameterization": "CMS/S1",
                "alpha2_beta_invariant": True,
                "increment_location_applied_each_step": True,
            }
        )
        return base


@dataclass
class DelayedRecallConfig:
    dimension: int = 33
    memory_dim: int = 32
    delay: int = 8
    steps: int = 16
    cue_noise: float = 0.02
    distractor_scale: float = 0.3
    reveal_sharpness: float = 0.35

    def __post_init__(self) -> None:
        for name in ("memory_dim", "dimension", "delay", "steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dimension != self.memory_dim + 1:
            raise ValueError("dimension must equal memory_dim + 1 for the clock channel")
        for name in ("cue_noise", "distractor_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(float(self.reveal_sharpness)) or self.reveal_sharpness <= 0.0:
            raise ValueError("reveal_sharpness must be finite and > 0")


class DelayedRecallBenchmark(UnifiedBenchmark):
    def __init__(self, config: DelayedRecallConfig) -> None:
        super().__init__()
        self.config = config
        self.dimension = config.dimension
        self.dt = 1.0
        self.default_steps = config.steps
        self.state_shape = (config.dimension,)
        self.latent_transition_stochastic = False
        self.observation_stochastic = bool(
            config.cue_noise > 0.0 or config.distractor_scale > 0.0
        )
        self.stochastic = self.observation_stochastic
        self.is_non_markovian = True

    def _validate_hidden(self, state: torch.Tensor) -> None:
        if state.ndim != 2 or state.shape[-1] != self.config.dimension:
            raise ValueError(
                "delayed-recall hidden state must have shape "
                f"[batch, {self.config.dimension}]; received {tuple(state.shape)}"
            )
        if state.shape[0] < 1:
            raise ValueError("delayed-recall hidden state batch must be non-empty")
        if not state.is_floating_point():
            raise TypeError("delayed-recall hidden state must use a real floating dtype")

    def _split(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_hidden(state)
        mem = state[..., : self.config.memory_dim]
        clock = state[..., -1:]
        return mem, clock

    @staticmethod
    def _randn_like(
        reference: torch.Tensor,
        *,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        return torch.randn(
            reference.shape,
            dtype=reference.dtype,
            device=reference.device,
            generator=generator,
        )

    def reveal_gate(
        self,
        t: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(t, int) or isinstance(t, bool) or t < 1:
            raise ValueError("post-cue observation time t must be an integer >= 1")
        return torch.sigmoid(
            torch.tensor(
                (t - self.config.delay) / self.config.reveal_sharpness,
                device=device,
                dtype=dtype,
            )
        )

    def sample_initial_state(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be an integer >= 1")
        device = _as_device(device)
        mem = torch.randn(
            batch_size,
            self.config.memory_dim,
            device=device,
            generator=generator,
        )
        clock = torch.zeros(batch_size, 1, device=device)
        return torch.cat([mem, clock], dim=-1)

    def observe(
        self,
        hidden: torch.Tensor,
        t: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if not isinstance(t, int) or isinstance(t, bool) or t < 0:
            raise ValueError("observation time t must be an integer >= 0")
        mem, clock = self._split(hidden)
        noise = self._randn_like(mem, generator=generator)
        if t == 0:
            visible = mem + self.config.cue_noise * noise
        else:
            gate = self.reveal_gate(t, device=hidden.device, dtype=hidden.dtype)
            visible = gate * mem + self.config.distractor_scale * noise
        return torch.cat([visible, clock], dim=-1)

    def step(self, state: torch.Tensor) -> torch.Tensor:
        mem, clock = self._split(state)
        clock_next = clock + (1.0 / self.config.steps)
        return torch.cat([mem, clock_next], dim=-1)

    def rollout_segment(
        self,
        hidden: torch.Tensor,
        *,
        start_time: int,
        observation_count: int,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a resumable observation segment.

        Exact continuation requires the returned hidden tensor, the next observation
        time, and the caller generator state. The explicit generator makes fresh
        observation noise restorable without process-global RNG state.
        """
        if not isinstance(start_time, int) or isinstance(start_time, bool) or start_time < 0:
            raise ValueError("start_time must be an integer >= 0")
        if (
            not isinstance(observation_count, int)
            or isinstance(observation_count, bool)
            or observation_count < 1
        ):
            raise ValueError("observation_count must be an integer >= 1")
        self._validate_hidden(hidden)
        expected_clock = torch.full_like(
            hidden[:, -1:], float(start_time) / self.config.steps
        )
        next_clock = torch.full_like(
            expected_clock, float(start_time + 1) / self.config.steps
        )
        previous_clock = torch.full_like(
            expected_clock, float(start_time - 1) / self.config.steps
        ) if start_time else None
        if torch.equal(expected_clock, next_clock) or (
            previous_clock is not None and torch.equal(expected_clock, previous_clock)
        ):
            raise ValueError(
                "hidden clock precision cannot distinguish adjacent observation times; "
                "restore a higher-precision hidden state"
            )
        # Roundoff allowance must stay below an observation interval. Otherwise
        # bfloat16 can accept an adjacent time even at the default 16-step horizon.
        tolerance = min(
            torch.finfo(hidden.dtype).eps * max(4.0, float(start_time + 1)),
            0.25 / self.config.steps,
        )
        if not torch.isfinite(hidden[:, -1:]).all() or not torch.allclose(
            hidden[:, -1:], expected_clock, rtol=0.0, atol=tolerance
        ):
            raise ValueError(
                "start_time does not match the clock channel in the restored hidden state"
            )
        traj = []
        current = hidden
        for offset in range(observation_count):
            traj.append(self.observe(current, start_time + offset, generator=generator))
            current = self.step(current)
        return torch.stack(traj, dim=1), current

    def rollout(
        self,
        x0: torch.Tensor,
        steps: Optional[int] = None,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        steps = self.default_steps if steps is None else steps
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("steps must be an integer >= 0")
        traj, _ = self.rollout_segment(
            x0,
            start_time=0,
            observation_count=steps + 1,
            generator=generator,
        )
        return traj

    def generate_batch(
        self,
        batch_size: int,
        steps: Optional[int] = None,
        device: str | torch.device = "cpu",
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.sample_initial_state(batch_size, device=device, generator=generator)
        traj = self.rollout(x0, steps=steps, generator=generator)
        return traj[:, :-1], traj[:, 1:]

    def metrics(self) -> Dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "name": "DelayedRecallBenchmark",
                "state_shape": self.state_shape,
                "horizon": self.default_steps,
                "is_non_markovian": True,
                "stochastic": self.stochastic,
                "latent_transition_stochastic": self.latent_transition_stochastic,
                "observation_stochastic": self.observation_stochastic,
                "observation_noise_semantics": (
                    "Gaussian cue noise at t=0 and fresh Gaussian distractor noise "
                    "for each observation at t>0"
                ),
                "cue_noise": self.config.cue_noise,
                "distractor_scale": self.config.distractor_scale,
                "reveal_sharpness": self.config.reveal_sharpness,
                "reveal_gate_semantics": (
                    "sigmoid centered on delay: gate(delay)=0.5 with a non-zero "
                    "pre-delay tail; preserved from the frozen v2 source"
                ),
                "dt": 1.0,
                "dimension": self.dimension,
                "memory_dim": self.config.memory_dim,
                "delay": self.config.delay,
            }
        )
        return base

__all__ = [
    "LevyStableConfig",
    "LevyStableProcess",
    "DelayedRecallConfig",
    "DelayedRecallBenchmark",
]
