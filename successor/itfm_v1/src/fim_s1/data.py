from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WindowRecord:
    context: torch.Tensor
    target: torch.Tensor
    owner_id: int
    family_id: int
    event_time: int
    regime: int
    aliased: bool


class AliasedAdvectionCorpus:
    """Paired periodic advection trajectories with oracle-defined alias events.

    Every family shares one initial field across two trajectories with opposite
    velocities. On a periodic grid, paired observations coincide at declared
    times while their next fields differ. Families, not windows, are assigned to
    splits before records are constructed.
    """

    def __init__(
        self,
        *,
        trajectory_count: int = 96,
        time_steps: int = 48,
        height: int = 16,
        width: int = 16,
        channels: int = 1,
        context_length: int = 3,
        train_fraction: float = 0.625,
        validation_fraction: float = 0.1875,
        seed: int = 20261011,
        observation_noise: float = 0.0,
        speed: int = 1,
        smoothness: float = 18.0,
    ) -> None:
        if trajectory_count < 6 or trajectory_count % 2:
            raise ValueError("trajectory_count must be an even integer of at least 6")
        if time_steps <= context_length + 2:
            raise ValueError("time_steps is too short for the requested context")
        if min(height, width, channels, context_length) < 1:
            raise ValueError(
                "spatial dimensions, channels, and context must be positive"
            )
        if width % 2:
            raise ValueError("width must be even so paired periodic aliases are exact")
        if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
            raise ValueError("split fractions must lie strictly between zero and one")
        if train_fraction + validation_fraction >= 1:
            raise ValueError("train and validation fractions must leave a test split")
        if observation_noise < 0:
            raise ValueError("observation_noise cannot be negative")
        if speed < 1:
            raise ValueError("speed must be a positive integer")
        if smoothness <= 0:
            raise ValueError("smoothness must be positive")

        self.trajectory_count = int(trajectory_count)
        self.time_steps = int(time_steps)
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.context_length = int(context_length)
        self.seed = int(seed)
        self.observation_noise = float(observation_noise)
        self.speed = int(speed)
        self.smoothness = float(smoothness)
        self.family_count = trajectory_count // 2

        generator = torch.Generator().manual_seed(seed)
        family_order = torch.randperm(self.family_count, generator=generator).tolist()
        train_count = max(1, int(self.family_count * train_fraction))
        validation_count = max(1, int(self.family_count * validation_fraction))
        if train_count + validation_count >= self.family_count:
            validation_count = max(1, self.family_count - train_count - 1)
        self.split_families = {
            "train": set(family_order[:train_count]),
            "validation": set(
                family_order[train_count : train_count + validation_count]
            ),
            "test": set(family_order[train_count + validation_count :]),
        }
        self._trajectories = self._generate(generator)

    def _template(self, generator: torch.Generator) -> torch.Tensor:
        raw = torch.randn(self.channels, self.height, self.width, generator=generator)
        spectrum = torch.fft.rfft2(raw)
        ky = torch.fft.fftfreq(self.height).view(self.height, 1)
        kx = torch.fft.rfftfreq(self.width).view(1, self.width // 2 + 1)
        low_pass = torch.exp(-self.smoothness * (kx.square() + ky.square()))
        smooth = torch.fft.irfft2(spectrum * low_pass, s=(self.height, self.width)).real
        smooth = (smooth - smooth.mean()) / smooth.std().clamp_min(1e-6)
        return smooth

    def _generate(self, generator: torch.Generator) -> dict[int, torch.Tensor]:
        trajectories: dict[int, torch.Tensor] = {}
        for family_id in range(self.family_count):
            template = self._template(generator)
            for regime_index, direction in enumerate((-1, 1)):
                owner = 2 * family_id + regime_index
                frames = []
                for event_time in range(self.time_steps):
                    shift = direction * self.speed * event_time
                    frame = torch.roll(template, shifts=shift, dims=-1)
                    if self.observation_noise:
                        noise = torch.randn(
                            frame.shape, generator=generator, dtype=frame.dtype
                        )
                        frame = frame + self.observation_noise * noise
                    frames.append(frame)
                trajectories[owner] = torch.stack(frames)
        return trajectories

    def _is_alias(self, event_time: int) -> bool:
        return (2 * self.speed * event_time) % self.width == 0

    def records(self, split: str) -> list[WindowRecord]:
        if split not in self.split_families:
            raise ValueError(f"unknown split: {split}")
        records: list[WindowRecord] = []
        for family_id in sorted(self.split_families[split]):
            for regime_index, regime in enumerate((-1, 1)):
                owner = 2 * family_id + regime_index
                trajectory = self._trajectories[owner]
                for event_time in range(self.context_length - 1, self.time_steps - 1):
                    start = event_time - self.context_length + 1
                    context = trajectory[start : event_time + 1].reshape(
                        self.context_length * self.channels, self.height, self.width
                    )
                    records.append(
                        WindowRecord(
                            context=context,
                            target=trajectory[event_time + 1],
                            owner_id=owner,
                            family_id=family_id,
                            event_time=event_time,
                            regime=regime,
                            aliased=self._is_alias(event_time),
                        )
                    )
        records.sort(key=lambda item: (item.event_time, item.owner_id))
        return records

    def validate(self) -> dict[str, int | float]:
        all_sets = list(self.split_families.values())
        if any(all_sets[i] & all_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError("family leakage across splits")
        if set.union(*all_sets) != set(range(self.family_count)):
            raise RuntimeError("family split is incomplete")
        counts: dict[str, int | float] = {}
        for split in self.split_families:
            split_records = self.records(split)
            alias_count = sum(record.aliased for record in split_records)
            if not split_records or alias_count == 0:
                raise RuntimeError(f"split {split} lacks records or alias cases")
            counts[f"{split}_records"] = len(split_records)
            counts[f"{split}_aliases"] = alias_count
            counts[f"{split}_families"] = len(self.split_families[split])
            indexed = {
                (row.family_id, row.event_time, row.regime): row
                for row in split_records
            }
            current_differences = []
            future_mse = []
            for row in split_records:
                if not row.aliased or row.regime != -1:
                    continue
                counterpart = indexed.get((row.family_id, row.event_time, 1))
                if counterpart is None:
                    raise RuntimeError(f"split {split} has an unmatched alias pair")
                current_differences.append(
                    float(
                        (
                            row.context[-self.channels :]
                            - counterpart.context[-self.channels :]
                        )
                        .abs()
                        .max()
                        .item()
                    )
                )
                future_mse.append(
                    float((row.target - counterpart.target).square().mean().item())
                )
            if not current_differences or max(current_differences) != 0.0:
                raise RuntimeError(
                    f"split {split} does not contain exact observed aliases"
                )
            if min(future_mse) <= 0.0:
                raise RuntimeError(f"split {split} alias futures are not distinct")
            counts[f"{split}_alias_pairs"] = len(current_differences)
            counts[f"{split}_alias_current_max_abs_difference"] = max(
                current_differences
            )
            counts[f"{split}_alias_future_min_mse"] = min(future_mse)
        return counts

    def identity(self) -> dict[str, object]:
        import hashlib

        digest = hashlib.sha256()
        for owner in sorted(self._trajectories):
            digest.update(str(owner).encode("ascii"))
            digest.update(self._trajectories[owner].contiguous().numpy().tobytes())
        return {
            "sha256": digest.hexdigest(),
            "trajectory_count": self.trajectory_count,
            "family_count": self.family_count,
            "time_steps": self.time_steps,
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
            "context_length": self.context_length,
            "seed": self.seed,
            "speed": self.speed,
            "smoothness": self.smoothness,
            "observation_noise": self.observation_noise,
            "splits": {
                name: sorted(families) for name, families in self.split_families.items()
            },
        }

    def trajectories(self, split: str | None = None) -> dict[int, torch.Tensor]:
        if split is None:
            owners = sorted(self._trajectories)
        else:
            if split not in self.split_families:
                raise ValueError(f"unknown split: {split}")
            owners = [
                owner
                for family in sorted(self.split_families[split])
                for owner in (2 * family, 2 * family + 1)
            ]
        return {owner: self._trajectories[owner].clone() for owner in owners}


def corrupt_records(
    records: list[WindowRecord],
    *,
    noise_std: float = 0.0,
    missing_fraction: float = 0.0,
    seed: int = 0,
) -> list[WindowRecord]:
    if noise_std < 0 or not 0 <= missing_fraction < 1:
        raise ValueError("invalid corruption parameters")
    generator = torch.Generator().manual_seed(seed)
    corrupted = []
    for row in records:
        context = row.context.clone()
        if noise_std:
            context = context + noise_std * torch.randn(
                context.shape, generator=generator, dtype=context.dtype
            )
        if missing_fraction:
            keep = torch.rand(context.shape, generator=generator) >= missing_fraction
            context = context * keep.to(context.dtype)
        corrupted.append(
            WindowRecord(
                context=context,
                target=row.target,
                owner_id=row.owner_id,
                family_id=row.family_id,
                event_time=row.event_time,
                regime=row.regime,
                aliased=row.aliased,
            )
        )
    return corrupted
