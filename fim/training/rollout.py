from __future__ import annotations

import operator
from typing import List, Optional, Tuple

import torch


def _validate_rollout_inputs(
    x0: torch.Tensor,
    steps: int,
    control_sequence: Optional[torch.Tensor],
    teacher_forcing: Optional[torch.Tensor],
) -> int:
    """Reject malformed caller inputs before a model can advance its state."""
    if isinstance(steps, bool):
        raise TypeError("steps must be an integer, not a boolean")
    try:
        steps = operator.index(steps)
    except TypeError as exc:
        raise TypeError("steps must be an integer") from exc
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if not isinstance(x0, torch.Tensor):
        raise TypeError("x0 must be a tensor")
    if x0.ndim < 1 or x0.numel() == 0:
        raise ValueError("x0 must be a nonempty tensor with a batch axis")
    if not bool(torch.isfinite(x0).all()):
        raise ValueError("x0 must be finite")

    for name, sequence in (
        ("control_sequence", control_sequence),
        ("teacher_forcing", teacher_forcing),
    ):
        if sequence is None:
            continue
        if not isinstance(sequence, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if sequence.ndim < 2:
            raise ValueError(f"{name} must have batch and time axes")
        if sequence.shape[0] != x0.shape[0] or sequence.shape[1] < steps:
            raise ValueError(f"{name} must match the x0 batch and contain at least steps entries")
        if sequence.device != x0.device:
            raise ValueError(f"{name} must be on the same device as x0")
        if name == "teacher_forcing":
            if sequence.shape[2:] != x0.shape[1:]:
                raise ValueError("teacher_forcing feature shape must match x0")
            if sequence.dtype != x0.dtype:
                raise ValueError("teacher_forcing dtype must match x0")
        # Extra supplied time steps are deliberately unused, as in the original
        # recurrence. Control feature axes and dtype remain model-specific.
        if not bool(torch.isfinite(sequence[:, :steps]).all()):
            raise ValueError(f"{name} used prefix must be finite")
    return steps


@torch.no_grad()
def autoregressive_rollout(
    model,
    x0: torch.Tensor,
    steps: int,
    *,
    control_sequence: Optional[torch.Tensor] = None,
    teacher_forcing: Optional[torch.Tensor] = None,
    update_memory: bool = False,
    return_outputs: bool = True,
) -> Tuple[torch.Tensor, Optional[List]]:
    """Run an admitted sequence, leaving independent-episode resets to the caller.

    Input checks finish before the first ``model.step`` call. The helper does not
    roll model state back when an admitted model call itself raises. Teacher
    forcing entry ``t`` becomes the input after prediction ``t``; its batch,
    feature shape, device and dtype must match ``x0``. Controls retain their own
    feature shape and dtype, with a matching batch, device and time coverage.
    """
    steps = _validate_rollout_inputs(x0, steps, control_sequence, teacher_forcing)

    B = x0.shape[0]

    preds = torch.zeros(
        B,
        steps,
        *x0.shape[1:],
        device=x0.device,
        dtype=x0.dtype,
    )

    outputs: Optional[List] = [] if return_outputs else None

    x = x0

    for t in range(steps):
        if control_sequence is not None:
            control = control_sequence[:, t]
        else:
            control = None

        result = model.step(
            x,
            control=control,
            update_memory=update_memory,
        )

        if isinstance(result, tuple):
            y, out = result
        else:
            y = result
            out = None

        if y.shape != x.shape:
            raise RuntimeError(
                f"Shape mismatch: predicted {y.shape} vs input {x.shape}"
            )

        preds[:, t] = y

        if outputs is not None:
            outputs.append(out)

        if teacher_forcing is not None:
            x = teacher_forcing[:, t]
        else:
            x = y

    return preds, outputs
