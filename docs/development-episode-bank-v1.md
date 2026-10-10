# Development delayed-recall episode bank v1

This opt-in path makes the delayed-recall generator's actual observations a
retained input artifact. Independent variants can read the same pinned bytes
instead of generating a new dataset after model initialization has consumed
random numbers. Every admitted split is **DEVELOPMENT** data.

The maintained experiment CLI, benchmark implementations, training code,
evaluation code, frozen protocols, old results, and scientific holds are
unchanged. This module does not run a learner. It is an engineering step toward a
paired design, not qualification of the held successor-v2 study or evidence for
a memory advantage.

## Execute the complete bounded demonstration

From the repository root with its Python dependencies installed:

```bash
python -m fim_experiments.episode_bank_v1 demo --output /tmp/fim-development-bank-demo
```

Use a fresh destination. Existing directories are never replaced. The command
generates five real delayed-recall episodes using seed `20261010`, four memory
channels, a clock channel, and 12 transitions. It creates:

```text
bank/observations.f32
bank/manifest.json
demo_receipt.json
```

The demonstration loads the bank twice, checks all three training episodes in
the same seeded order while the process-global RNG states differ, and checks
independent tensor storage. Validation and development-evaluation each contain
one different episode. The receipt records the bank/data SHA-256, shape, split
sizes, measured duration, and zero model-training/scientific-outcome runs.

To create a separate small bank using the native default configuration:

```bash
python -m fim_experiments.episode_bank_v1 create \
  --output /tmp/fim-development-bank \
  --seed 37 --horizon 20 \
  --train-count 8 --validation-count 2 --evaluation-count 2
```

Save the returned `bank_sha256` in the experiment's reviewed configuration or
receipt **before** any variant runs. To verify a bank, provide that independently
retained value through `--expected-sha256`; the verifier never silently trusts a
bank's self-reported identity. The following executable Python example creates,
pins, and reads a bank without training:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from fim_experiments.episode_bank_v1 import (
    create_development_bank,
    load_development_bank,
)

with TemporaryDirectory() as temporary:
    destination = Path(temporary) / "bank"
    receipt = create_development_bank(destination, seed=37, horizon=12)
    pin = receipt["bank_sha256"]
    bank = load_development_bank(destination, expected_bank_sha256=pin)
    for inputs, targets in bank.loader("train", shuffle=True, order_seed=43):
        assert inputs.shape == targets.shape == (1, 12, 33)
        assert (inputs[:, 1:] == targets[:, :-1]).all()
    print(pin, bank.manifest["splits"])
```

`create_development_bank` also accepts a validated `DelayedRecallConfig` from
`fim_experiments.runtime_benchmarks`. The public CLI deliberately exposes only
seed, counts, and horizon. The `demo` command specifies its own small config.

## Concrete data contract

Initial hidden memory consists of explicit CPU float32 draws from `N(0, I)`,
with a zero clock. A dedicated CPU `torch.Generator` determines both initial
memory and all observation noise. The native delayed-recall `rollout` provides
the transition and observation equations. Explicit float32 allocation avoids
dependence on `torch.set_default_dtype`; generation and loading do not change
the process-global RNG. This does not promise identical regeneration under
arbitrary future PyTorch versions or hardware.

The observation array has shape `(N, H + 1, D)`, where `D = memory_dim + 1`.
The final feature is the benchmark clock. Raw storage is C-order, little-endian
IEEE float32 (`<f4`), with exactly `4 * N * (H + 1) * D` bytes. There is no pickle
or executable serialization. Episode `i` produces inputs `observations[i, :-1]`
and targets `observations[i, 1:]`; their shared temporal values are bit-identical.
The clock is checked against repeated float32 increments of `1 / config.steps`,
matching the maintained recurrence including its rounding.

The manifest records these exact fields:

| Field | Meaning |
| --- | --- |
| `schema_version`, `artifact_type`, `phase` | Version 1, this named artifact type, and DEVELOPMENT |
| `seed`, `benchmark_config`, `horizon` | Complete generator inputs |
| `array` | Fixed file name, dtype, order, shape, length, and data SHA-256 |
| `episode_sha256` | Ordered content identity of every episode |
| `splits` | Nonempty `train`, `validation`, `development_eval` index lists |
| `source_sha256` | Exact benchmark, native runtime, and bank-module source identities |
| `environment` | Python, PyTorch, NumPy, CPU device, float32 generation dtype |
| `bank_sha256` | Identity of the complete canonical manifest body |

Canonical JSON means UTF-8 JSON with sorted keys, compact separators, and no
NaN/infinity. The bank SHA-256 covers the canonical manifest **excluding** its
own `bank_sha256` field. Each episode SHA-256 covers canonical JSON
`{"dtype":"<f4","shape":[H+1,D]}`, a newline byte, and that episode's raw bytes.
Duplicate episode contents are refused rather than assigned distinct content
identities. This check is not a proof of statistical independence.
It is an explicit admission condition: a duplicate-content draw fails and is
retained; the producer does not silently redraw it. Any future scientific use
must account for this condition in its protocol rather than claim an
unconditioned sampling law or infer independent units from unequal hashes.

The producer assigns consecutive, disjoint indices to the three declared split
sizes. The loader checks that every episode appears exactly once across splits.
It checks the caller's pinned identity, source identities, finite values, shape,
clock, lengths, and hashes before exposing tensors. It refuses extra files,
missing files, duplicate JSON keys, unexpected fields, symbolic-link data files,
nonregular files, and oversized artifacts.

The consumer requires the exact source hashes recorded at creation. A later
source edit therefore requires using the original source checkout to verify an
old bank, or creating a new bank for the changed implementation. Do not modify a
historical manifest to make a new source version appear equivalent. Runtime
versions are retained as provenance; loading verifies bytes rather than
regenerating them with the current runtime.

Use a **fresh interpreter and an immutable checkout** when creating or loading
a bank. The module captures the source hashes at import, checks that snapshot
before and after generation, and checks it again before and after loading.
An ordinary checkout edit in a long-lived process therefore fails instead of
labeling cached execution with newly read source bytes. The module cannot attest
to a dependency that was already imported from different bytes before the bank
module itself was imported, arbitrary in-memory monkeypatches, or a malicious
writer. Restart the process after any source change; do not use module reloads
as a substitute for that precondition.

## Paired loading and model responsibilities

Every loader fixes `batch_size=1`, `num_workers=0`, and `drop_last=False`.
The loader adapter uses `torch.get_default_device()` in the tested PyTorch
runtime and requires the process default device to remain CPU during
construction and iteration; it rejects an incompatible default at construction. The producer and direct dataset reads use explicit CPU
storage even when the default device differs. The adapter does not change the
process device setting on the caller's behalf.
`shuffle` defaults to false. An explicit local `order_seed` determines shuffle
order and PyTorch's loader base seed without consuming the global RNG. Construct
fresh loaders with the same seed for paired variants. A reused loader's local
generator advances between iterations; use equal iteration schedules, or a
declared per-epoch order seed, when integrating a future paired runner.

Each dataset read returns fresh, mutually independent input and target tensors.
Mutating a returned tensor cannot change another sample, another consumer, or
the retained observations. The manifest property also returns a copy.

The returned `(inputs, targets)` batches are accepted by the existing trainer's
batch unpacker. The bank does **not** reset model memory between episodes, select
checkpoints, enforce training budgets, measure performance, or make the complete
scientific matrix runnable. Those responsibilities require a separately
reviewed prospective runner/protocol integration. Batch size one alone does not
establish trajectory isolation if a model carries state across calls.

## Write failures, bounds, and trust

Creation reserves a new directory atomically, writes and synchronizes raw data,
writes and synchronizes a pending manifest, installs the manifest, and
synchronizes the directory and its parent. A competing writer fails at directory
reservation. No existing destination is removed, truncated, or replaced.

After reservation, failures leave the directory intact. The producer attempts
to retain `failure.json` with its stage, exception, and seed; if storage itself
fails, it preserves the original error without claiming the receipt was saved.
The loader refuses failed or incomplete directories. A synchronization failure
after manifest installation is recorded as a failure, not described as a
rollback. The demonstration similarly retains a failure receipt if possible.

These filesystem operations target a local POSIX filesystem. They do not provide
a distributed transaction, protection against a malicious writer changing the
entire checkout and pins, hardware-level durability guarantees, or data secrecy.
A pin must be retained outside the mutable bank to detect a self-consistent
replacement. Finite values, hashes, and split membership establish artifact
integrity; they cannot attest that a stated scientific protocol was valid.

Creation bounds are 128 total episodes, 64 transitions, 64 memory channels, and
1,000,000 stored float32 values. All three split counts must be positive. It
refuses the ten held trajectory-isolated-v1 and delayed-recall-v2 scientific
seeds (`101, 211, 307, 401, 503, 607, 701, 809, 907, 1009`). No CONFIRMATORY phase,
protected-data input, network operation, accelerator, workflow dispatch, or paid
compute is provided. This local refusal is not a security boundary for other
programs or an authorization to run held experiments with different seeds.

## Verification and retained execution

The focused test command is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q tests/test_episode_bank_v1.py \
  tests/test_benchmark_runtime_contracts.py \
  tests/test_delayed_recall_successor_v2.py
```

Tests use small development fixtures and the real generator. They compare
canonical float32 observations to native and frozen-reference outputs, exercise
the trainer batch adapter, validate local RNG/ordering/storage isolation, and
attempt corrupted, resealed-invalid, overlapping, and failed disk writes.
Temporary test fixtures are not scientific datasets or research outcomes.

The implementation contract was recorded before code in
`research/DEVELOPMENT_EPISODE_BANK_V1_CONTRACT_20261010.md`. Actual attempts,
review, costs, and the retained demonstration are indexed by
`research/RESEARCH_STATE.json`; historical engineering and scientific records
remain separate, retained entries in that state.
