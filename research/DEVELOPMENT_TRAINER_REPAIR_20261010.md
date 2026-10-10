# Public Trainer development repair — 10 October 2026

This change repairs the public one-step `fim.training.trainer.Trainer` helper.
It does not change the held trajectory-isolated protocol, the protected runner,
`fim_experiments/train.py`, the benchmark generators, any retained result or paper.
The canonical experiment runner already uses a distinct parameter-only EMA.

## Defects and behavior

The public helper expected a retired output wrapper (`out.state.tensor`), while
current `FIMModel.step` returns `(prediction, FIMStepOutput)` with tensor state.
Its scheduler passed names from a different scheduler API. Its EMA attempted
floating-point updates on integer memory counters and averaged episode-memory
buffers as if they were model weights. Evaluation permanently installed its EMA
weights, including when later code raised.

The helper now consumes current outputs, passes real pre/post retrieval states
to the existing consistency loss, rejects shape broadcasting/non-finite inputs,
losses and gradients, and calls the scheduler with its actual parameter names.
EMA averages trainable parameters only and provides a context that restores raw
weights on success or exception. Evaluation also restores the previous train/eval
mode. The loss coefficients are unchanged.

The caller still owns episode boundaries and should call
`model.reset_state(clear_memory=True)` between independent trajectories.
This one-step helper is not evidence of trajectory isolation or memory benefit.
Evaluation may advance model-owned episode state; the weight-restoration guarantee
concerns parameters and model mode, not a snapshot of an ongoing episode.

## Validation scope

`tests/test_public_trainer.py` includes a tiny actual FIM forward/backward and
validation cycle, exact arithmetic on a structured artificial model, scheduler
execution, integer-buffer/EMA checks, exception restoration, rejected shape
broadcasts, and non-finite input/gradient rejection. All inputs are artificial.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_public_trainer.py
```

Local compilation and targeted Ruff checks can run without PyTorch. The local
container does not contain PyTorch and its installation was blocked by automatic
approval review; runtime tests must be established by the existing hosted CI.
Until an actual passing hosted run is linked in the PR, runtime validation remains
pending. This development repair supplies no scientific efficacy result, releases
no execution hold, and does not establish research completion.

## Second development pass: rejected-gradient recovery and mixed modes

The inspected parent was `0ae2bf91fb1b88217637a5319f9c74628ffa4436`. Five additional artificial regression cases failed there while the 16 preceding Trainer cases passed. Evaluation restored only the root mode, recursively destroying intentionally mixed encoder/submodule modes on both successful and failed evaluation. Separately, a nonfinite gradient advanced an enabled GradScaler to its unscaled state; clipping raised without completing the scaler attempt, so the next valid batch failed with `unscale_() has already been called ... since the last update()`.

Evaluation now snapshots and restores every submodule mode. If gradient clipping raises, the public Trainer completes the scaler update, clears rejected gradients, and re-raises. It performs no optimizer, scheduler or EMA update for that rejected attempt. The same Trainer then accepts a valid batch.

The selected local suite now passes **21 tests** on the installed CPU PyTorch 2.14.1 runtime. Four new cases cover mixed modes on success/failure; the fifth exercises the actual CPU GradScaler state machine, a nonfinite-gradient hook, unchanged raw/EMA/optimizer/scheduler state, and a following valid update with the expected reduced scale. The older local-runtime limitation above is retained as history and is superseded for this pass. Existing deprecation warnings remain.

This does not roll back model-owned episode memory after a failed model step and does not claim CUDA execution. The caller still owns episode recovery. The protected 40-cell protocol, source-bound experiment trainer, data, scientific outcomes and paper artifacts remain untouched. Hosted evidence is recorded per exact revision in the PR.
