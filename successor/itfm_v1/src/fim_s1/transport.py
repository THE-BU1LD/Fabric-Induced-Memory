from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F


@torch.no_grad()
def estimate_periodic_translation(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    max_shift: int,
) -> torch.Tensor:
    """Return the bounded normalized-cross-correlation translation per item.

    The returned shift maps ``previous`` into ``current``. Spatial dimensions
    are rank-generic and periodic; channels are compared jointly.
    """

    if previous.shape != current.shape or previous.ndim < 3:
        raise ValueError("previous and current must share shape [B,C,*spatial]")
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    if not bool(torch.isfinite(previous).all().item()) or not bool(
        torch.isfinite(current).all().item()
    ):
        raise ValueError("transport inputs must be finite")
    spatial_rank = previous.ndim - 2
    if max_shift == 0:
        return torch.zeros(
            previous.shape[0], spatial_rank, device=previous.device, dtype=torch.long
        )

    normalized_previous = F.normalize(previous.flatten(1), dim=1, eps=1e-8).reshape_as(
        previous
    )
    normalized_current = F.normalize(current.flatten(1), dim=1, eps=1e-8).reshape_as(
        current
    )
    if any(2 * max_shift >= size for size in previous.shape[2:]):
        raise ValueError(
            "max_shift must be less than half of every periodic spatial dimension"
        )
    candidates = sorted(
        itertools.product(range(-max_shift, max_shift + 1), repeat=spatial_rank),
        key=lambda candidate: (sum(abs(value) for value in candidate), candidate),
    )
    spatial_dims = tuple(range(-spatial_rank, 0))
    scores = torch.stack(
        [
            (
                torch.roll(
                    normalized_previous,
                    shifts=candidate,
                    dims=spatial_dims,
                )
                * normalized_current
            )
            .flatten(1)
            .sum(dim=1)
            for candidate in candidates
        ],
        dim=1,
    )
    best = scores.argmax(dim=1)
    candidate_tensor = torch.tensor(
        candidates, device=previous.device, dtype=torch.long
    )
    return candidate_tensor[best]


def apply_periodic_translation(
    field: torch.Tensor, shifts: torch.Tensor
) -> torch.Tensor:
    """Apply one integer periodic translation to each batch item."""

    if field.ndim < 3:
        raise ValueError("field must have shape [B,C,*spatial]")
    spatial_rank = field.ndim - 2
    if shifts.shape != (field.shape[0], spatial_rank):
        raise ValueError(f"shifts must have shape [B,{spatial_rank}]")
    if shifts.device != field.device:
        raise ValueError("field and shifts must be on the same device")
    spatial_dims = tuple(range(-spatial_rank, 0))
    return torch.stack(
        [
            torch.roll(
                field[index],
                shifts=tuple(int(value) for value in shifts[index].tolist()),
                dims=spatial_dims,
            )
            for index in range(field.shape[0])
        ]
    )


def transport_coordinates(
    velocity_shifts: torch.Tensor,
    event_times: torch.Tensor,
    *,
    temporal_offset: int = 0,
) -> torch.Tensor:
    """Integrate a constant integer velocity into periodic coordinates."""

    if velocity_shifts.ndim != 2:
        raise ValueError("velocity_shifts must have shape [B,spatial_rank]")
    event_times = event_times.to(
        device=velocity_shifts.device, dtype=torch.long
    ).flatten()
    if event_times.numel() != velocity_shifts.shape[0]:
        raise ValueError("event_times must contain one value per shift")
    if bool((event_times < 0).any().item()):
        raise ValueError("event_times must be non-negative")
    return velocity_shifts * (event_times + temporal_offset).unsqueeze(1)
