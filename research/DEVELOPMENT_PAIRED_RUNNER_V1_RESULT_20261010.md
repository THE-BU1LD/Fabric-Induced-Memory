# Paired development runner v1 — closed engineering result

## Outcome and scope

A new opt-in runner now executes the existing `full`, `no_retrieval`, and
`no_memory` interventions on independently caller-pinned development episodes.
It verifies actual memory clearing between trajectories, pairs initial tensors
and supplied inputs, restricts optimization to training data, retains raw
outcomes and failures, and verifies complete evidence against source/data pins.

This work starts at PR #35 head
`0ad7dc9766771cb3d9cb18bab649cfa2da3e4dc5`, in a separate worktree. No existing
model, bank producer/consumer, trainer, evaluator, metric, or protocol source
was changed. The prior fractional repair and episode-bank sessions and their
budgets remain closed and unchanged.

The new engineering session is **closed** after independent review, 124 passing
affected tests, and one verified integration demonstration. No scientific
checkpoint or research disposition is advanced. All existing adverse results,
invalid historical evidence, frozen protocols, and protected execution holds
remain in force. This receipt is not a completed hypothesis test or paper result.

## Fixed implementation and scientific boundary

The pre-code contract is
`research/DEVELOPMENT_PAIRED_RUNNER_V1_CONTRACT_20261010.md`; executable usage and
full semantics are in `docs/development-paired-runner-v1.md`.

The existing three model files are compiled from their exact bytes using
temporary legacy import aliases that are restored afterward. The runner checks
source freshness, initial tensor identities, strict memory resets before and
after every episode, raw input/target/prediction bytes, within-episode retrieval,
and optimizer applications. The complete verifier reconstructs initial tensors,
checks event grammar and bank membership, loads final safe tensor snapshots,
and recomputes diagnostics. It executes no forward or optimizer calls.

The fixed architecture has 2,372 allocated parameters per arm. The salience
threshold of 0.0 was declared before execution to exercise memory paths.
Training is teacher forced; validation and development evaluation are
autoregressive, on fixed final parameters without checkpoint selection. Stored
keys, values, and salience scores retain their existing detached behavior.
Equal allocated parameters and updates do not establish equal active parameters,
FLOPs, runtime, memory cost, or information use.

## Preserved verification and repair history

| Attempt | Evidence | Result |
|---|---|---|
| 1 | Initial new-runner suite | 40 passed in 18.38 s; 26.405267 s measured command |
| Source review | Interruption/cleanup handling | A cancellation could bypass `episode_failed` and lose a secondary cleanup error; no executable pre-repair witness was claimed |
| Source review | Bounded NPZ admission | A small NPY payload could declare a larger array before `np.load` checked it; archive byte bounds alone did not bound allocation |
| Repair cycle 1 | Both defects recorded before edits | Retain both BaseException paths and propagate cancellation without later arms; validate NPY header shape/dtype/length before loading |
| 2 | Final affected suite | **124 passed in 8.06 s**; 9.622930 s measured command |
| 3 | One retained CLI demonstration plus verification | **VERIFIED_INTEGRATION**; 7.642118 s measured command |

The final suite comprises 45 new runner tests and existing episode-bank,
current-intervention, retrieval-gradient, and training-tensor tests. It includes
real bounded model training, bitwise reproduction despite unrelated RNG use,
an optimizer spy proving training-only updates, safe checkpoint reload and
inference, every buffer/index/alias/reset-purity check, foreign cached-module
isolation, source-edit rejection, retained nonfinite output bytes, simultaneous
forward/cleanup errors, cancellation without further arms, and resealed raw
artifact/chronology/metric/checkpoint defects.

Cancellation and malformed-header repairs were independently reviewed before
the final gate and demonstration. The reviewer for model/reset/training/failure
semantics was the parent-assigned `atg_ngmt_olympus` agent; the parent separately
reviewed event grammar, raw diagnostics, initial/final checkpoint admission,
and bounded NPY headers. Neither review ran additional probes or model work.

The initial source, initial tests, final source, final tests, hashes, commands,
logs, and receipts remain in
`research/artifacts/development_paired_runner_v1_tests_20261010/`.
The initial runner SHA was
`4c02e60f85a05d5ef078bc6f56c357a44ecc1bc671be4c47a764c43ba64839cc`.
The final reviewed/tested/demo runner SHA is
`735af27ffbbc484ed286cd880dc784a8e7efe107dd724dadae4508b37cc49d73`.
The final test SHA is
`7aa6390a40fa60f088bef07e089849a4d65e46d43506274edad2709f588a9926`.
All seven execution-source digests are in the retained demo's `plan.json`.

## Single retained demonstration

Artifacts are in
`research/artifacts/development_paired_runner_v1_demo_20261010/`.

- Bank identity: `ae8e143064f27be469e69dc19fe1fac41d30737ecdcd4acff7a6de4a91afc938`.
- Run identity: `382d2e7febf87af361de2fc3c1ab2645d4aa6b2ff447a11f723f7159560617d4`.
- Shared initial parameter identity: `d3f82d4dcb72d1240162486afde58004b7e1a509745d657e3fd6ec2750fb775e`.
- Existing bank: five episodes; three training, one validation, one development evaluation. No new bank generation.
- Fixed prefix: four transitions; one epoch; model seed 37.
- All three arms completed, each with 20 model steps and three SGD updates.
- Total: **60 model steps, nine SGD updates**, 60 returned raw outcomes and complete reset/update traces.
- Referenced artifact payload: 230,698 bytes, with every file hash in `manifest.json`.
- Measured run body: 0.807719919 s. Entire command, including imports and verification: 7.642118101 s.
- Peak RSS: 331,022,336 bytes, the cumulative Linux process high-water mark including imports and all arms.

These float64 observation MSE values are preserved as **integration diagnostics**:

| Existing intervention | Training diagnostic | Validation diagnostic | Development-evaluation diagnostic |
|---|---:|---:|---:|
| full | 0.5497110425441943 | 0.3222115305374137 | 1.0289934449153166 |
| no_retrieval | 0.5513993717035174 | 0.3183459165284570 | 1.0287509765248437 |
| no_memory | 0.5513993717035174 | 0.3183459165284570 | 1.0287509765248437 |

They have no inferential interpretation. The raw observation-space diagnostic
does not replace the held recall-only endpoint. The matching no-retrieval and
no-memory predictions are consistent with the current detached storage path
being inactive in prediction when retrieval is disabled; this is an integration
observation, not a scientific efficacy finding. No arm or setting was selected
using these outcomes.

## Closed budget and concurrent work

Consumed: three of six allowed verification commands/probes (including the
single demo), **43.670314927 s** of the 240 s command budget, one of two allowed
review repair cycles, and the sole authorized retained demonstration. The demo's
60-step/nine-update cap was met exactly. No paid compute, protected outcomes,
held seeds, manual scientific workflow dispatch, broad training, or hypothesis
support claim occurred. No additional execution is authorized by unused budget
after this session closes.

PR #35 remained at the exact parent during the final readback. Open PRs #27,
#28, #32, and #34 edit separate maintained trainer/evaluator/metric/standalone
memory paths. Newly observed PR #36 adds standalone scale-stable diagnostics
and a parallel canonical-state proposal; its source was not integrated or
revalidated here. These open proposals must be reconciled explicitly when a
future integration is authorized. This branch preserves its exact tested model
and generator sources and all predecessor evidence.

Publication is a separate stacked draft on
`codex/fim-development-episode-bank-20261010`, using the new branch
`codex/fim-paired-development-runner-20261010`. No existing PR or branch is
force-updated or merged. Source hashes and artifact pins avoid circular commit
claims inside the release being created; final PR/head/CI receipts are recorded
by the publishing session.
