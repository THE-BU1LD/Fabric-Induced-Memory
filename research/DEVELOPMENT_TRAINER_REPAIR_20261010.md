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
