# Paired development runner v1

This executable path connects the retained development episode bank to the
existing `AblatedFIMSystem`. It is an opt-in integration tool for the three
existing interventions `full`, `no_retrieval`, and `no_memory`.

The tool does not execute, replace, or repair a frozen scientific protocol. The
held outcome boundary, original negative findings, invalid historical campaign,
and other open trainer, evaluator, metric, and memory changes remain unchanged.
The preimplementation specification is
[`DEVELOPMENT_PAIRED_RUNNER_V1_CONTRACT_20261010.md`](../research/DEVELOPMENT_PAIRED_RUNNER_V1_CONTRACT_20261010.md).

## Run a concrete integration check

Use a fresh Linux Python process, CPU PyTorch, NumPy, and an immutable checkout.
Set the numerical thread environment before importing PyTorch. Invoke the module
from the repository root; it resolves the maintained model's legacy imports
internally without adding unrelated directories to `sys.path`.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m fim_experiments.paired_development_v1 demo \
  --output /tmp/fim-paired-demo-new-attempt
```

The demo consumes the **existing** bank at
`research/artifacts/development_episode_bank_v1_demo_20261010/bank`, pinned to
`ae8e143064f27be469e69dc19fe1fac41d30737ecdcd4acff7a6de4a91afc938`.
It generates no new data. The predeclared model seed is 37, independent of that
bank's generation seed. Each arm uses its three training, one validation, and
one development-evaluation episode, restricted to the first four transitions.
Across all three arms, the demo makes **60 model-step calls and nine SGD
updates**. The CLI then verifies the retained raw evidence without further
model-step or optimizer calls.

This four-transition fixture is labeled `DEVELOPMENT / INTEGRATION_ONLY`. It
tests implementation, pairing, memory isolation, artifact retention, and
reproduction. It is not a test of delayed-recall benefit, a benchmark score, or
evidence supporting the scientific hypothesis.

An independently pinned development bank can also be supplied explicitly:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m fim_experiments.paired_development_v1 run \
  --bank /path/to/development-bank \
  --bank-sha256 FULL_INDEPENDENTLY_RECORDED_BANK_SHA256 \
  --model-seed 37 --horizon 4 --epochs 1 \
  --output /path/to/new-run-directory
```

`FULL_INDEPENDENTLY_RECORDED_BANK_SHA256` must be replaced with a trusted,
previously recorded bank identity. A caller-supplied pin is required; the runner
does not treat an untrusted artifact's declaration as its own external identity.
An existing output directory is always an error, including one containing a
failed attempt. Keep the failed directory and use a separately authorized new
attempt identity if additional work is justified.

## Exact model and optimization semantics

The existing model definitions are executed from their exact source bytes in
`fim_experiments/models.py`, `systems.py`, and `ablation_systems.py`. Their
`variant_switches` determine the interventions. No scientific model definition
is duplicated in this runner.

All arms allocate the same architecture: one input channel, hidden dimension 4,
trace dimension 4, memory capacity 16, retrieval top-k 4, and memory decay 0.02.
The salience threshold is fixed to **0.0** to exercise writes and retrieval in a
small integration fixture. That setting was declared before running models; it
is not a tuned efficacy claim.

Each arm is initialized independently from the same local CPU RNG seed. The
runner compares every initial parameter and state tensor, including allocated
parameter count. It preserves the caller's CPU RNG state. A held scientific
seed or the bank's own generation seed cannot be used as the model seed.

For a training episode with states $x_0,\ldots,x_H$, the runner feeds the real
state $x_t$ at each step and computes the float32 optimization loss

$$
L = \frac{1}{H}\sum_{t=0}^{H-1}
       \frac{1}{D}\sum_{j=1}^{D}(f_\theta(x_t;m_t)_j-x_{t+1,j})^2.
$$

Here $m_t$ is within-episode memory governed by the existing intervention; its
write, detach, and retrieval behavior remains the model's current code. The
bank is cleared between independent episodes. Training uses one gradient
backward pass and one SGD update per episode, learning rate 0.001, no momentum,
no weight decay, and global gradient norm clipping at 1.0. Gradients are cleared
before each training episode. The same sequential training episode order and
prefix horizon are used in every arm.

The current bank detaches stored keys, values, and salience scores. Observation
MSE can train current query/feedback paths, but it does not train stored value
or salience heads through those writes. A one-item retrieval softmax also has
zero query gradient. This fixture exercises the existing mechanism and does
not establish that all allocated memory parameters learn from its loss.

All training epochs finish before validation and development evaluation. Those
splits run once on the fixed final parameters, under `torch.no_grad` semantics:
the first input is $x_0$, and subsequent inputs are the arm's own preceding
prediction. Within-episode storage/retrieval remains active according to the
intervention. Parameters are checked to remain unchanged. These observations
cannot select an arm, seed, setting, epoch, or checkpoint.

The verifier recomputes **float64 observation MSE diagnostics** from retained
float32 prediction and target bytes. It separately verifies the float32
optimization-loss receipt. These diagnostics are explicitly different from
the held recall-only scientific endpoint. No confidence interval, significance
test, model-benefit inference, or scientific classification is produced.

Pairing controls initial tensors, allocated parameters, supplied observations,
order, horizon, and SGD update count. The interventions intentionally change
memory access and active computation. Equal active parameters, FLOPs, runtime,
memory use, or information use are **not** established by this fixture.

## Strict isolation and retained failures

Before and after every episode, the actual model's `reset_state()` is called.
The runner checks that `bank`, `trace_bank`, and `memory_bank` are aliases of
one object; that the only relevant buffers are keys, values, scores, and ages;
and that all those tensors, pointer, and size are cleared. Reset must preserve
every learned parameter. An exception, missing buffer, extra buffer, divergent
alias, incomplete clear, or parameter-changing reset fails the arm.

The generic maintained reset helper is not used because it suppresses reset
exceptions. This runner does not silently fall back to another reset method.
If a forward call and cleanup reset both fail, both exceptions and tracebacks
remain in `events.jsonl`.

`KeyboardInterrupt`, `SystemExit`, and other interruptions propagate after the
primary and cleanup errors are recorded. Cancellation stops the remaining arms;
it is not converted into an ordinary arm failure that permits further work.

Before any model is instantiated, `plan.json` declares every arm, episode,
source digest, setting, and planned update. Model-step attempts, returned raw
outcomes, optimizer attempts, completed optimizer calls, and failures are
recorded separately. A failed optimizer call may have partially mutated its
parameters; an attempted call is not silently counted as a completed update.
The final tensor snapshot and failure record are retained where writing
succeeds. Failure in one arm does not trigger a retry or replacement seed. Only
the other arms in the original plan are attempted.

A failed or partial run cannot pass the complete-run verifier. The CLI returns
a nonzero exit code for failure. Filesystem failures and interruption can leave
a partial directory; it remains a failed attempt, never an implicitly rolled
back or complete run. Writes flush file data and synchronize containing
directories on Linux. These operations cannot override filesystem, hardware,
or power-loss limitations.

## Evidence layout and verification

Each complete output directory contains:

| File | Meaning |
|---|---|
| `plan.json` | Pre-run declaration of source, data, environment, interventions, settings, and episode order |
| `initial.npz` | Shared initial float32 parameter and buffer tensors; no pickle |
| `initial_metadata.json` | Tensor and parameter identities, names, and allocated parameter count |
| `<arm>/events.jsonl` | Chronological attempts, raw float32 input/target/prediction bytes, resets, updates, episode diagnostics, and failures |
| `<arm>/final.npz` | That arm's final tensor snapshot, also retained on failure when possible |
| `<arm>/status.json` | Arm disposition, initial/final identities, failures, attempted/retained counts, and duration |
| `manifest.json` | Hash and size of every evidence file, caller bank identity, overall status, duration, process RSS, and run identity |

Raw tensors use their shape, original dtype, and little-endian float32 bytes
encoded as hexadecimal. This retains nonfinite outputs as evidence without
writing invalid JSON or quietly dropping the failing step. A complete run
requires finite, correctly shaped values. Array checkpoints use ordinary NumPy
archives with float32 tensors and `allow_pickle=False` when loading.

NPY headers, declared shapes, dtypes, and payload lengths are validated before
array allocation. Small malformed archives cannot request arbitrarily large
arrays merely by declaring a larger shape in their headers.

Save the `run_sha256` from the run's printed receipt in the canonical project
record. Independent verification requires that pin and the bank pin:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m fim_experiments.paired_development_v1 verify \
  --run /path/to/retained-run --run-sha256 RECORDED_RUN_SHA256 \
  --bank /path/to/development-bank --bank-sha256 RECORDED_BANK_SHA256
```

The verifier checks exact artifact coverage and hashes, reconstructs the
initial tensors from the declared seed, validates every episode and step
against the pinned bank, checks teacher forcing and autoregressive input
chronology, validates memory/intervention/reset/update events, loads final
tensor snapshots, and recomputes diagnostics. It does not rerun forward or
training calls. Repeating `run` under the same admitted source and environment
with a fresh output path is a separate execution; such a repeat requires its
own recorded development budget. Durations, RSS, and the encompassing run hash
can differ even when model predictions and tensor states reproduce exactly.

Runtime uses elapsed monotonic time. RSS is the **cumulative Linux process
high-water mark**, including imports and all earlier work in that process. It
is not isolated per-arm memory or a compute-matched comparison.

## Bounds and trust limits

This version permits at most two epochs, 16 transitions, 16 episodes, 16
observation features, and 768 planned model-step calls. It supports only the
three declared existing interventions and fixed model/optimizer semantics.
Changing this scope requires a new recorded version and budget. The CLI
records bounded local operations; it cannot prevent commands executed outside
it, infer scientific validity, or enforce protected-data secrecy.

Source digests include the generator, bank consumer, actual model sources, and
this runner. Model source bytes are compiled through temporary, restored legacy
import aliases; unrelated cached `models` and `systems` modules are not used.
Source freshness is checked before and after execution. These checks detect
ordinary checkout edits during a session. A fresh interpreter and immutable
checkout remain preconditions, particularly for bank dependencies already
cached before import. This is not general Python-runtime attestation or a
cryptographic proof that an untrusted party executed the declared code.

The original bank's duplicate-content rejection and development-only sampling
limitations still apply. This runner does not establish independent scientific
units, power, generalization, or benchmark fairness beyond its declared
integration controls. Frozen research remains held until its own justified,
versioned methodology is ready and authorized.
