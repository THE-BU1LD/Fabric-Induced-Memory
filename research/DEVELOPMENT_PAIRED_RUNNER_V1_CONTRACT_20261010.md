# Pinned-bank paired development runner v1 — preimplementation contract

This is a new bounded engineering continuation authorized by the user's repeated
request for real research-code changes. The predecessor episode-bank session and
its exhausted budget remain closed and unchanged. No scientific hypothesis,
protected protocol, historical result, or competing implementation is revised.

## Observed gap and overlap audit

PR #35 provides retained development episodes but no model runner consumes them.
The maintained experiment reset helper catches and suppresses reset exceptions;
therefore calling it does not prove trajectory isolation. The current
`FIMSystem` owns one shared `AdaptiveMemoryBank`, so explicit verified clearing
between independent episodes remains essential.

This work starts in a clean worktree at PR #35 head
`0ad7dc9766771cb3d9cb18bab649cfa2da3e4dc5`. Open PR #28 edits the separate public
`fim/training` trainer; #27 edits maintained experiment training/evaluation and
admission; #34 edits the standalone `fim/memory/trace_bank.py`; #32 edits general
metrics. None consumes the pinned episode bank. Preserve all their files.

Implement only new runner/test/documentation/artifact files and append the
canonical session. Reuse the existing `models.py`, `systems.py`, and
`ablation_systems.py` without modifying or copying their scientific definitions
into a new model implementation.

## Exact prospective execution contract

1. Require a caller-provided bank directory and independently pinned bank SHA-256.
   Use the verified DEVELOPMENT-only v1 bank loader unchanged. Refuse held seeds,
   incompatible source snapshots, an existing output destination, oversized
   dimensions/counts, and any CONFIRMATORY mode before model execution.
2. Run exactly the existing `full`, `no_retrieval`, and `no_memory` interventions.
   They use the same architecture allocation and exact `variant_switches` values.
   Initialize each arm independently from the same fresh development model seed,
   isolated from bank generation and the caller's RNG. Compare complete initial
   parameter tensor identities and allocated parameter counts before training.
3. Use CPU float32, one numerical thread, fixed SGD with learning rate 0.001,
   no momentum/weight decay/scheduler/EMA/early stopping, and gradient clipping at
   norm 1.0. The fixed integration configuration is hidden size 4, trace dimension
   4, bank capacity 16, retrieval top-k 4, memory decay 0.02, and salience threshold
   0.0. The zero threshold is predeclared to exercise within-episode storage and
   retrieval; it is not a tuned scientific setting.
4. Use the same sequential episode order and same fixed prefix horizon in each
   arm. Inputs have batch size one. Training is fully teacher-forced, with one
   SGD step per training episode after averaging float32 observation MSE over
   its horizon. Zero gradients before every training episode. Do not optimize
   validation or development-evaluation data.
5. Validation and development evaluation are autoregressive from the first
   observed state. Continue within-episode memory writes and retrieval according
   to the intervention switches. Their values are diagnostic only and cannot
   select checkpoints, hyperparameters, arms, or seeds. Save the fixed final state.
6. Before and after every independent episode, call the real model reset method
   and verify its effects: all three bank aliases identify the same object;
   keys, values, scores, and ages are zero; pointer and size are zero; the known
   buffer inventory is complete; and learned parameters do not change during
   reset. Never swallow a reset error. Preserve primary and cleanup failures.
7. Keep raw per-step predictions, targets, split/epoch/episode identifiers, bank
   occupancy before/after, retrieval occurrence, and optimization diagnostics.
   Preserve every attempted arm and failure. Continue only the originally planned
   remaining arms after an arm fails; never replace seeds, retry an output path,
   or turn partial results into a complete run.
8. Persist a pre-run plan before models execute, initial and final tensor
   checkpoints in a non-pickle format, step logs, per-arm status, and a complete
   manifest binding source/config/data/environment and every retained file hash.
   Check model-source freshness around execution. Legacy imports must resolve to
   the exact reviewed source bytes, not an unrelated top-level `models` module.
9. Provide executable run/demo/verify commands. Verification checks the caller's
   pinned run identity, complete expected arms/episodes/steps, raw-file hashes,
   data membership, control settings, reset/optimization traces, and recomputed
   diagnostic metrics. Failed or incomplete runs are retained but cannot pass the
   complete-run verifier. It does not infer scientific support from scores.

Equal allocated parameters, initial tensors, input order, horizons and optimizer
updates are explicit controls. They do not establish equal active parameters,
FLOPs, runtime, or information use during the intended memory intervention.
Measure duration and report process peak RSS with its actual cumulative scope;
do not label it an isolated per-arm memory or compute comparison.

## Bounded demonstration and operating limits

The single retained demonstration consumes the unchanged PR #35 bank with SHA
`ae8e143064f27be469e69dc19fe1fac41d30737ecdcd4acff7a6de4a91afc938`.
It uses all five development episodes (3 training, 1 validation, 1 development
evaluation), the first four transitions, one epoch, and a separately declared
model seed. Across three arms this is exactly nine optimizer updates and 60
model step calls. No new bank is generated for the retained demonstration.

This four-transition fixture is an **integration check**, not the frozen
hypothesis test, a recall-benefit experiment, a benchmark comparison, or a paper
result. All numeric diagnostics remain labeled DEVELOPMENT / INTEGRATION_ONLY.

The public runner remains bounded: at most two epochs, 16 transitions, 16 input
episodes, 16 observation features, and 768 planned model steps. Architecture,
optimizer, interventions, training/evaluation semantics and threshold are fixed
for this version. Expanding these bounds or changing those semantics requires a
separately recorded successor configuration/version and budget.

## Verification and stopping budget

One implementation workstream; at most six focused verification commands or
independent executable probes, totaling at most 240 measured command seconds;
at most two independent-review repair cycles; one retained integration demo
with the nine-update/60-step budget above. Unit-test fixtures use smaller bounded
development data and are included in the measured verification budget.

No protected outcome access, held seeds, scientific matrix dispatch, paid
compute, broad model training, scientific support claim, or paper-result update.
Retain initial failures and exact tested source versions. Stop after independent
review, bounded verification, retained demonstration and a separate draft PR.
