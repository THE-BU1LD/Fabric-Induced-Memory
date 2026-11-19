# Development episode bank v1 — verified engineering result

Recorded 2026-10-10. This pass implements a usable delayed-recall artifact and
paired loader, with a retained end-to-end development demonstration. It establishes
data-path behavior within the documented contract. It establishes no memory
benefit, comparative model result, scientific readiness, or protected outcome.

## Problem and implemented path

The successor-v2 protocol requires identical trajectory bytes across variants,
generated and hashed before variant execution. The existing static loader in
`fim_experiments/main.py` generates observations and randomly splits them inside
each call. Its configuration declarations alone do not retain or enforce that
paired-input identity.

The new opt-in `fim_experiments/episode_bank_v1.py` generates real observations
with an explicit CPU generator and float32 storage. It saves a canonical raw
array, content-addressed manifest, episode identities, and disjoint development
splits. A caller supplies an independently retained bank SHA-256; loading checks
the complete artifact before exposing isolated tensor copies. Independent
loaders use explicit local ordering seeds and batch size one.

The code also implements exclusive destination reservation, synchronized writes,
incomplete/failed artifact rejection, retained failure receipts when storage
permits, and source-freshness checks. It provides executable create, verify,
and demo commands. The API and format are documented in
`docs/development-episode-bank-v1.md`.

No existing benchmark, runtime benchmark, trainer, evaluator, protocol,
historical result, paper, or other open PR implementation was changed. The
previous fractional-history repair and its evidence remain on the parent branch.

## Actual verification history

The contract and new bounded session were recorded before implementation in
`research/DEVELOPMENT_EPISODE_BANK_V1_CONTRACT_20261010.md` and the canonical state.
Every verification attempt below is retained. Times are command durations; the
pytest-reported test duration is separately given where applicable.

| Attempt | Observation | Duration |
| --- | --- | --- |
| 1: initial focused suite | 104 passed, 7 subtests passed; initial contract checks passed | 24.855 s command; 19.36 s pytest |
| 2: targeted pre-repair regression | 1 failed: an ambient meta default device redirected initial allocation away from CPU | 28.639 s command; 23.27 s pytest |
| 3: independent provenance witness | Cached and reloaded generators admitted different observations under equal declared source hashes after an ordinary file edit | 3.886 s |
| 4: final focused suite | 108 passed, 7 subtests passed; both review repairs and the original contract checks passed | 9.856 s command; 8.17 s pytest |

The complete focused verification cost is **67.236 seconds**, within the
180-second bound. All four admitted verification commands and both review repair
cycles are used; no further search for a favorable outcome is authorized by this
session. The retained demonstration below is the one separately admitted demo.

The verification directory is
`research/artifacts/development_episode_bank_v1_tests_20261010/`.
`attempt_01_sources.json` preserves the first tested source and test file.
`attempt_02_test_source.json` preserves the failing regression's test source;
its producer version is in the first archive. The independent witness script,
stdout, and observed-source snapshot are also retained. Its snapshot explicitly
states that the original stdout did not print source hashes; it is not
retroactively presented as a hash printed by that original command. The final
test log and receipt identify the final source files by SHA-256.

The witness script reproduced the defect against the pre-repair sources. Running
it against the repaired checkout is expected to stop at the new freshness guard.
The archived source snapshot permits inspecting or reconstructing its historical
input version without overwriting current source or history.

### Review corrections

**CPU allocation.** The initial producer supplied a CPU generator and float32
dtype but omitted `device="cpu"` on the initial tensors. A default-meta regression
failed in the native rollout before data publication. The repair allocates both
memory and clock on CPU. Direct dataset reads remain CPU-resident. The loader
checks its explicit CPU default-device precondition instead of changing the
caller's process settings.

**Executed-source freshness.** Reading source hashes only at generation time
can describe newly edited disk files while Python executes cached definitions.
The independent witness reproduced that discrepancy. The repair captures a
source snapshot at bank-module import and requires it to remain unchanged before
reservation, after generation, and at both load boundaries. The manifest records
that verified snapshot. New tests make real copied-source edits before
creation/loading, during generation, and during data loading; invalid artifacts
are not exposed to consumers.

Independent review by the parent-assigned reviewer examined the producer,
consumer, tests, failure behavior, scientific boundary, and both corrections.
The reviewer found no remaining blocker before the final suite and demo.

## Retained real-generator demonstration

Command from the final source checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m fim_experiments.episode_bank_v1 demo \
  --output research/artifacts/development_episode_bank_v1_demo_20261010
```

This destination already exists in the released branch. To reproduce, use a new
destination; the command deliberately refuses to overwrite the retained record.

The retained bank is in
`research/artifacts/development_episode_bank_v1_demo_20261010/bank/`.
Its observations contain five episodes, 13 observations per episode, and five
features (four memory channels plus the clock): **1,300 raw bytes**. The splits
are three training, one validation, and one development-evaluation episode.
The seed is `20261010`; the benchmark has delay 3, clock denominator 8, and
12 transitions. No held scientific seed was used.

The demo verified identical ordered input/target values for two separately loaded
consumers despite unrelated RNG activity, adjacent temporal overlap, and clone
isolation after deliberately mutating one returned tensor. It passed with:

- Bank SHA-256: `ae8e143064f27be469e69dc19fe1fac41d30737ecdcd4acff7a6de4a91afc938`.
- Observation SHA-256: `6c274a11fd0cb8f1a0e46f82a391243e8b14e47c7cfdc17040e51e4ed29e94c5`.
- Demo-body duration: 0.059227 seconds.
- Complete command duration, including startup: 6.125824 seconds.
- Model-training runs: **0**. Scientific-outcome runs: **0**.

The demo receipt is retained beside the bank; the command receipt is retained
with the test attempts. The environment was Python 3.12.14, PyTorch 2.14.1+cpu,
and NumPy 2.3.5, with one numerical thread. These times describe this tiny
development demonstration, not a training-efficiency or scaling result.

## Scope and remaining scientific work

The producer and consumer require a fresh interpreter and immutable checkout.
The source snapshot does not attest to dependencies cached before the bank was
imported, in-memory monkeypatches, or a malicious writer. A future source version
must use a new artifact identity; an old bank is verified from its original
source checkout rather than silently relabeled.

The bank refuses duplicate episode contents, with failed attempts retained and
no automatic redraw. Any future scientific protocol must account for this
explicit admission condition. Content hashes and disjoint indices do not prove
statistical independence. The bank also does not reset model memory, perform
training, enforce another runner's budgets, or establish baseline fairness.

The held trajectory-isolated-v1 and delayed-recall-v2 protocols remain held.
The existing unfavorable and mixed findings are unchanged. Adopting this path
for a scientific study requires its own reviewed prospective integration,
qualified controls, model-state isolation, frozen protocol, and execution
decision before protected outcome access. The engineering pass is complete;
the research question remains open under its existing evidence.

## Publication parent inspection

The tested code base is `fd5161947c127797e7b0c2cda9dbfb7d0d378d97`. Before
publication, parent PR #30 advanced to
`1c2de9851b6d6858e25f6cdedb641ef9300343e2`. Its inspected delta adds only 19
lines to the canonical state: a separate development receipt and an explicit
incomplete-history-import limit. Both fields are retained verbatim. No source,
runtime, protocol, or tested artifact changed. The continuation uses that
metadata-only descendant as its publication parent while retaining the exact
tested generator, consumer, and data bindings.
