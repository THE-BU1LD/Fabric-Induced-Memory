from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class EventRecord:
    context: torch.Tensor
    target: torch.Tensor
    owner_id: int
    family_id: int
    event_time: int
    regime: int
    anchor_id: int
    kind: str
    query: bool


class EpisodicAliasCorpus:
    """Paired event streams with complete-context aliases and delayed innovations.

    The construction is intentionally diagnostic. Paired regimes share every
    visible context. At calibration and query events, their targets differ by
    opposite spatial innovations. Calibration targets become observable only
    after their prediction and may then be stored for later same-owner queries.
    """

    def __init__(
        self,
        *,
        family_count: int = 48,
        anchor_count: int = 3,
        query_repeats: int = 2,
        context_length: int = 3,
        height: int = 16,
        width: int = 16,
        channels: int = 1,
        washout_events: int = 12,
        residual_scale: float = 0.75,
        smoothness: float = 14.0,
        train_fraction: float = 0.625,
        validation_fraction: float = 0.1875,
        seed: int = 101,
    ) -> None:
        if family_count < 6:
            raise ValueError("family_count must be at least six")
        if anchor_count < 2 or query_repeats < 1:
            raise ValueError("at least two anchors and one query repeat are required")
        if min(context_length, height, width, channels, washout_events) < 1:
            raise ValueError("dimensions and washout_events must be positive")
        if residual_scale <= 0 or smoothness <= 0:
            raise ValueError("residual_scale and smoothness must be positive")
        if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
            raise ValueError("split fractions must lie in (0,1)")
        if train_fraction + validation_fraction >= 1:
            raise ValueError("split fractions must leave a test partition")

        self.family_count = int(family_count)
        self.anchor_count = int(anchor_count)
        self.query_repeats = int(query_repeats)
        self.context_length = int(context_length)
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.washout_events = int(washout_events)
        self.residual_scale = float(residual_scale)
        self.smoothness = float(smoothness)
        self.seed = int(seed)

        generator = torch.Generator().manual_seed(seed)
        family_order = torch.randperm(family_count, generator=generator).tolist()
        train_count = max(1, int(family_count * train_fraction))
        validation_count = max(1, int(family_count * validation_fraction))
        if train_count + validation_count >= family_count:
            validation_count = max(1, family_count - train_count - 1)
        self.split_families = {
            "train": set(family_order[:train_count]),
            "validation": set(
                family_order[train_count : train_count + validation_count]
            ),
            "test": set(family_order[train_count + validation_count :]),
        }
        self._records = self._generate(generator)

    def _smooth_field(self, generator: torch.Generator) -> torch.Tensor:
        raw = torch.randn(self.channels, self.height, self.width, generator=generator)
        spectrum = torch.fft.rfft2(raw)
        ky = torch.fft.fftfreq(self.height).view(self.height, 1)
        kx = torch.fft.rfftfreq(self.width).view(1, self.width // 2 + 1)
        low_pass = torch.exp(-self.smoothness * (kx.square() + ky.square()))
        field = torch.fft.irfft2(spectrum * low_pass, s=(self.height, self.width)).real
        field = field - field.mean(dim=(-2, -1), keepdim=True)
        return field / field.square().mean().sqrt().clamp_min(1e-6)

    def _context(self, field: torch.Tensor) -> torch.Tensor:
        return field.repeat(self.context_length, 1, 1)

    def _generate(self, generator: torch.Generator) -> list[EventRecord]:
        rows: list[EventRecord] = []
        for family_id in range(self.family_count):
            anchors = [self._smooth_field(generator) for _ in range(self.anchor_count)]
            residuals = [
                self.residual_scale * self._smooth_field(generator)
                for _ in range(self.anchor_count)
            ]
            nuisance_count = self.query_repeats * self.washout_events
            nuisances = [self._smooth_field(generator) for _ in range(nuisance_count)]

            schedule: list[tuple[str, int, torch.Tensor]] = []
            for anchor_id, anchor in enumerate(anchors):
                schedule.append(("calibration", anchor_id, anchor))
            nuisance_index = 0
            for _ in range(self.query_repeats):
                for _ in range(self.washout_events):
                    schedule.append(("nuisance", -1, nuisances[nuisance_index]))
                    nuisance_index += 1
                for anchor_id, anchor in enumerate(anchors):
                    schedule.append(("query", anchor_id, anchor))

            for regime_index, regime in enumerate((-1, 1)):
                owner_id = 2 * family_id + regime_index
                for event_time, (kind, anchor_id, field) in enumerate(schedule):
                    if anchor_id >= 0:
                        target = field + regime * residuals[anchor_id]
                    else:
                        target = field.clone()
                    rows.append(
                        EventRecord(
                            context=self._context(field),
                            target=target,
                            owner_id=owner_id,
                            family_id=family_id,
                            event_time=event_time,
                            regime=regime,
                            anchor_id=anchor_id,
                            kind=kind,
                            query=kind == "query",
                        )
                    )
        rows.sort(key=lambda row: (row.event_time, row.family_id, row.regime))
        return rows

    def records(self, split: str) -> list[EventRecord]:
        if split not in self.split_families:
            raise ValueError(f"unknown split: {split}")
        families = self.split_families[split]
        return [row for row in self._records if row.family_id in families]

    @staticmethod
    def base_prediction(row: EventRecord, channels: int) -> torch.Tensor:
        return row.context[-channels:]

    def identity(self) -> dict[str, object]:
        return {
            "dataset": "episodic-spatial-alias-v2",
            "seed": self.seed,
            "family_count": self.family_count,
            "anchor_count": self.anchor_count,
            "query_repeats": self.query_repeats,
            "context_length": self.context_length,
            "shape": [self.channels, self.height, self.width],
            "washout_events": self.washout_events,
            "residual_scale": self.residual_scale,
            "splits": {
                name: sorted(families) for name, families in self.split_families.items()
            },
        }

    def validate(self) -> dict[str, float | int]:
        split_sets = list(self.split_families.values())
        if any(
            split_sets[i] & split_sets[j]
            for i in range(len(split_sets))
            for j in range(i + 1, len(split_sets))
        ):
            raise RuntimeError("family leakage across splits")
        if set.union(*split_sets) != set(range(self.family_count)):
            raise RuntimeError("family split is incomplete")

        report: dict[str, float | int] = {}
        for split in ("train", "validation", "test"):
            rows = self.records(split)
            index = {(row.family_id, row.event_time, row.regime): row for row in rows}
            context_difference: list[float] = []
            future_difference: list[float] = []
            query_ages: list[int] = []
            query_count = 0
            for row in rows:
                if row.regime != -1:
                    continue
                counterpart = index[(row.family_id, row.event_time, 1)]
                context_difference.append(
                    float((row.context - counterpart.context).abs().max().item())
                )
                if row.anchor_id >= 0:
                    future_difference.append(
                        float((row.target - counterpart.target).square().mean().item())
                    )
                if row.query:
                    query_count += 2
                    calibration_times = [
                        candidate.event_time
                        for candidate in rows
                        if candidate.owner_id == row.owner_id
                        and candidate.anchor_id == row.anchor_id
                        and candidate.kind == "calibration"
                        and candidate.event_time < row.event_time
                    ]
                    if not calibration_times:
                        raise RuntimeError("query lacks a prior calibration event")
                    query_ages.append(row.event_time - max(calibration_times))
            if not rows or not query_count:
                raise RuntimeError(f"split {split} is empty or lacks queries")
            report[f"{split}_records"] = len(rows)
            report[f"{split}_queries"] = query_count
            report[f"{split}_families"] = len(self.split_families[split])
            report[f"{split}_full_context_max_abs_difference"] = max(context_difference)
            report[f"{split}_paired_future_min_mse"] = min(future_difference)
            report[f"{split}_minimum_calibration_age"] = min(query_ages)
            if max(context_difference) != 0.0:
                raise RuntimeError("paired visible contexts are not exact aliases")
            if min(future_difference) <= 0.0:
                raise RuntimeError("paired futures do not diverge")
            if min(query_ages) <= self.context_length:
                raise RuntimeError(
                    "calibration information lies inside visible context"
                )
        return report
