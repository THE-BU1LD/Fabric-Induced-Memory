# FIM Kaggle engineering handoff

16 October 2026. **BLOCKED** until the new tensor correctness and recovery suite
actually passes. This is a functioning implementation prepared for validation,
not a tested GPU release. No cloud job has been submitted. No GPU quota has
been consumed by this task.

## Scope and launch path

Use the existing `fim-sota engineering` CLI, exposed without an installation by
`successor/itfm_v2/scripts/kaggle.py`. The root historical `main.py`, archived
`FIM/scripts/run_full_suite.py`, consumed v3 protocol and v4 research gates are
not this campaign. This path is a synthetic software/development diagnostic,
not an admitted public-data experiment or a confirmatory study.

Commands below assume the repository root. On this Mac prefix shell commands
with `rtk proxy` according to the workspace instruction. Kaggle does not need
RTK. In an exported bundle, the package itself is the root: replace
`successor/itfm_v2/scripts/` with `scripts/` and
`successor/itfm_v2/kaggle/` with `kaggle/`.

```sh
# Read installed versions and perform two resource samples before tensor imports.
python -u successor/itfm_v2/scripts/kaggle.py preflight \
  --config successor/itfm_v2/kaggle/smoke-cpu.json --receipt cpu-preflight.json

# Tiny CPU correctness, actual interrupted recovery and artifact tests.
python -u successor/itfm_v2/scripts/kaggle.py test --device cpu --receipt cpu-tests.json

# Smallest GPU gate: explicit CUDA environment, then tiny training/evaluation.
python -u successor/itfm_v2/scripts/kaggle.py preflight --runtime \
  --config successor/itfm_v2/kaggle/smoke-cuda.json --receipt cuda-preflight.json
python -u successor/itfm_v2/scripts/kaggle.py test --device cuda --receipt cuda-tests.json

# One fully specified two-epoch experiment; fresh paths are mandatory.
python -u successor/itfm_v2/scripts/kaggle.py run \
  --config successor/itfm_v2/kaggle/smoke-cuda.json --output runs/cuda-smoke-001

# Enumerate every proposed development cell without importing Torch or launching.
python -u successor/itfm_v2/scripts/kaggle.py campaign \
  --manifest successor/itfm_v2/kaggle/development.json \
  --output runs/development-001 --dry-run

# Execute only after actual CPU/CUDA gates pass and future GPU use is authorized.
python -u successor/itfm_v2/scripts/kaggle.py campaign \
  --manifest successor/itfm_v2/kaggle/development.json --output runs/development-001

# Recovery goes into a new run; original receipts and failures remain unchanged.
python -u successor/itfm_v2/scripts/kaggle.py run \
  --config successor/itfm_v2/kaggle/smoke-cuda.json --output runs/cuda-resumed-001 \
  --resume runs/cuda-smoke-001/checkpoints/latest.pt

# Frozen-checkpoint evaluation never trains or selects weights using test data.
python -u successor/itfm_v2/scripts/kaggle.py evaluate \
  --config successor/itfm_v2/kaggle/smoke-cuda.json \
  --checkpoint runs/cuda-smoke-001/checkpoints/best.pt --output runs/replay-001

# Analysis refuses missing, corrupt, mismatched or incomplete cells.
python -u successor/itfm_v2/scripts/kaggle.py analyze \
  --root runs/development-001 --output development-summary.json

# Portable source/config/test bundle; optional --run-root includes verified raw results.
python -u successor/itfm_v2/scripts/kaggle.py export \
  --manifest successor/itfm_v2/kaggle/development.json --output fim-kaggle.zip
```

`analyze --root` also accepts one completed run. `campaign --reuse` verifies
completed cells against source/config/data/artifact hashes. It refuses failed,
partial, ambiguously running or previously blocked attempts. Explicit recovery
is supported by `run --resume` into a fresh directory; it does not rewrite an
old campaign's completion inventory. Reconcile that separate resumed result
explicitly before treating a failed campaign as complete.

## Environment

The inspected laptop has Python 3.14.7, PyTorch 2.11.0 and NumPy 2.4.4 by package
metadata; pytest is absent. Torch/CUDA runtime imports were refused by admission.
`requirements-runtime.txt` describes compatible API constraints, not a tested
cross-platform lock. Use Python 3.11+ on Kaggle and inspect its installed CUDA
build first. Do not replace it with a CPU wheel or blindly upgrade packages.
The core fixture needs Torch and NumPy; its numerical test command also needs
pytest. It needs no dataset download, network account or credentials.

CUDA requests fail if CUDA is unavailable. FP32 is the reference. FP16/BF16 are
explicit opt-ins in a copied config; CUDA tests check finite gradients and
one-step/autonomous outputs against FP32. Unsupported BF16 is a recorded skip,
not a pass. Do not use AMP for a campaign until that hardware-specific check
passes. Compilation, activation checkpointing, multiprocessing and distributed
training are disabled; unsupported distributed mode is rejected.

## Exactly what runs

The smoke config has seed 101; 4/2/2 train/validation/test owners; 12 steps;
3 observed context frames; 1 channel; 8 periodic spatial sites; hidden width 16;
2 epochs; batch 4; accumulation 2; learning rate .001; FP32; one CPU thread;
120-second cell limit; 10,000-example ceiling; memory capacity 16 per owner;
top-4 cosine retrieval and temperature .2. Full and partial accumulation windows
are weighted by actual example count. Effective batch sizes are logged.

Raw bounded periodic fields use independent split RNG streams and disjoint owner
IDs. Space uses distinct periodic sites, with fixed sinusoidal components,
per-owner phase/amplitude/drift and small independent noise, passed through tanh.
This is not a PDE solver or real-world dataset. No fitted normalization or data
cache is used. Source/config/generation hashes and actual raw tensor hashes are
retained; generated batches have a deterministic, fingerprinted order.

The existing `ResidualMLPControl` learns one-step forecasts from observed
context. The existing `TensorMemoryBank` stores detached, fully observed base
residuals, persisting across chronological events and isolating owner identity.
Each split/variant/evaluation starts with an empty bank. Retrieval is fixed
normalized context cosine, not a learned addressing claim. Memory writes happen
after the forecast, become eligible at their declared next observation time,
and never use a future target for the current prediction. Finite owner counts,
per-owner capacity and preflight allocation caps bound storage.

All five controls share the same trained checkpoint: base, full spatial residual
memory, erased values, spatially averaged residuals and last residual. Persistence
is also scored. A fixed scale grid `{0,.25,.5,1}` is selected on validation for
each control; frozen scales are then used on test. The separate autonomous base
rollout appends only its own predictions and reports each horizon; it neither
writes future observations nor measures a memory advantage. Spectral/geometry
architecture ablations are not represented by this fixture.

## Interruption, logging and evidence

Schema-2 checkpoints use atomic replacement, fsync, safe tensor/primitive payloads
and `weights_only=True` loading without a pickle fallback. They carry parameters,
declared mutable model state, optimizer, optional scheduler, scaler, all RNG
states, batch cursor, completed steps/examples, cumulative elapsed budget,
validation criterion, early-stop counters, and source/config/data/runtime identity.
`latest.pt` and `best.pt` are distinct. Exact restart is at a completed optimizer
accumulation window; a crash replays work after the last durable boundary. It
does not resume inside an unfinished backward pass. Identical Torch/device/thread
configuration is required. Custom mutable models must implement safe state
serialization and an evaluation reset hook; plain non-module memory is rejected.

Training uses no persistent bank; online evaluation resets and deterministically
replays the whole stream from a frozen checkpoint. There is no claim of exact
mid-evaluation continuation. Signal/time stops request a checkpoint, allow a
30-second shutdown reserve, then kill only the owned process group if necessary.
Output directories and receipts are exclusive. Campaigns reserve each cell's
maximum allowance durably before dispatch, including after ambiguous crashes.

Timestamped, flushed console events and fsynced JSONL contain real losses,
validation metrics, step/epoch counters, learning rates, effective batch sizes,
throughput, CUDA peak allocation, checkpoint paths and failures. Evaluation logs
split/variant progress and actual metrics. No ETA is invented. Local JSONL and
verified JSON summaries are the tracking interfaces; TensorBoard/W&B integration
is not required or enabled. No remote tracking is contacted.

## Quota decision

The preliminary development manifest has seeds 101/202/303, each training once
and evaluating all five controls. Each cell has a 3,600-second hard ceiling plus
30-second shutdown reserve; the whole campaign ceiling is 14,400 seconds.
These are stop limits, not predicted runtimes. Reserve a further hour for setup,
CPU/CUDA smoke and artifact download, keeping the planned envelope within five
hours if the gates pass. Do not fill quota merely to reach four hours.

Measure the same small FP32 experiment on CPU and CUDA first. Singleton online
memory queries and short MLP workloads may be faster on CPU. Record actual
end-to-end time, peak memory and error before enlarging anything. The three-seed
synthetic diagnostic cannot establish significance, public forecasting superiority
or novelty. Missing public data/baseline admission and v4 contribution work remain
separate scientific blockers.
