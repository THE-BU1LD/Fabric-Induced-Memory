# FIM trajectory-isolated bounded confirmatory protocol v1

Status: **FROZEN BEFORE OUTCOME ACCESS**

This protocol is a new current-code evidence lane. It does not rewrite, replace, or rescue historical FIM results. The retained shared-minibatch memory experiments remain historical evidence and must not be promoted as trajectory-isolated evidence.

## Motivation and P0 correction

The maintained `fim_experiments.systems.FIMSystem` uses one `AdaptiveMemoryBank` per model instance. With minibatches larger than one, independent trajectories can write into and retrieve from the same bank. Batch membership is not part of the scientific state. Therefore paper-facing memory experiments must make cross-trajectory retrieval impossible before new outcomes are used.

For this bounded protocol, the correction is deliberately minimal and predeclared: **all training, validation, and evaluation arms use `batch_size=1`**, memory is reset between independent episodes, and memory write/retrieval remains enabled during within-trajectory validation and test rollouts. This is the protocol-level correction explicitly allowed by the P0 audit; it avoids changing the memory algorithm itself after prior outcomes were observed.

A second pre-outcome audit finding is also frozen here: the default delayed-recall task uses `delay=8` and `steps=16`. A training rollout of only four steps never reaches the recall event and therefore cannot directly train the claimed long-delay mechanism. This protocol uses a **16-step training rollout** for both benchmarks so the delayed-recall arm actually crosses its memory delay. This change was made before any trajectory-isolated matrix outcome was observed.

No best-checkpoint selection is used. Training budget is fixed in advance and the final epoch is evaluated. EMA model selection is disabled.

## Research questions

Primary H1: on `delayed_recall`, the full current-code FIM mechanism has lower final fixed-budget rollout MSE than `no_memory` under paired fresh seeds.

Primary H2: on `delayed_recall`, the full current-code FIM mechanism has lower rollout MSE than `no_retrieval` under paired fresh seeds.

Generalization H3: on `lorenz96`, the full model's trajectory-isolated rollout MSE is compared with the same two memory controls under the identical fixed budget.

Secondary mechanism check: `no_salience_gating` tests whether selective storage helps relative to storing every trace that reaches the maintained memory path.

A null, mixed, or adverse result is acceptable. No threshold, architecture, seed, benchmark, metric, or budget may be changed after outcome access and still be called this protocol.

## Frozen matrix

Benchmarks:

- `delayed_recall`
- `lorenz96`

Fresh seeds:

- 101
- 211
- 307
- 401
- 503

Current-code variants:

- `full`
- `no_memory`
- `no_retrieval`
- `no_salience_gating`

Total expected cells: **40**.

## Fixed training and evaluation budget

- device: CPU for canonical CI execution
- deterministic mode: enabled
- batch size: **1 for every arm**
- epochs: **4**
- static generated dataset size: **32 independent trajectories per cell**
- train rollout steps: **16**
- delayed-recall default delay: **8**, so the training horizon crosses the recall event
- teacher forcing ratio: **0.5**
- horizon decay: **1.0**
- evaluation rollout steps: **20**
- optimizer/lr/weight decay/gradient clipping: repository defaults
- EMA training/evaluation: **disabled**
- best-checkpoint selection: **disabled**
- final evaluated state: fixed final epoch
- workers: 0

The bounded matrix is intentionally smaller than a final publication-scale study. It is designed to answer whether the corrected trajectory-isolated mechanism has enough signal to justify a larger separately frozen reproduction. It must not be described as a broad state-of-the-art comparison.

## Validation and test semantics

For every independent episode:

1. clear memory/state before the episode;
2. allow memory writes during the episode;
3. allow retrieval during subsequent steps of that same episode;
4. never expose traces from a different episode or batch element;
5. evaluate autoregressively with teacher forcing disabled.

Validation and final evaluation therefore use the same mechanism semantics. Validation is diagnostic only and is not used for early stopping or checkpoint selection.

## Primary metrics

- rollout MSE
- final-step MSE
- rollout MAE
- final-step MAE

Primary paired comparisons use `rollout_mse` by seed. Report raw per-seed values, paired deltas, deterministic paired bootstrap 95% intervals, and exact sign-flip p-values as descriptive inference. With only five seeds, statistical power is limited and must be stated explicitly.

## Failure handling

Every attempted cell must be retained in the manifest as `success` or `failed`. Failed cells are not silently removed. The analysis gate fails closed if any expected cell is missing, duplicated, failed, from a different Git commit, or bound to a different protocol hash.

## Provenance

Every row must retain at minimum:

- Git commit
- dirty-tree flag
- protocol SHA-256
- benchmark
- seed
- variant and exact switches
- training/evaluation budget
- Python version
- PyTorch version
- start/end timestamps
- experiment directory
- success/failure state
- generated metrics when successful

## Claim boundary

This protocol cannot validate the historical paper-reference values and cannot erase prior baseline wins or negative evidence. It tests only the maintained current-code FIM memory mechanism under trajectory-isolated execution. Any successor tuning or architecture change requires a new protocol version frozen before its outcomes are inspected.
