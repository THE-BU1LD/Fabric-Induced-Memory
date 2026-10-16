# FIM delayed-recall successor v2

Status: **FROZEN METHOD / PRE-OUTCOME REVIEW REQUIRED**

This is a separately versioned successor to `FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1`.
It does not change that protocol, reinterpret its whole-rollout primary endpoint, or
authorize a rescue run. Outcome access remains disabled until the exact machine-readable
contract in `FIM_DELAYED_RECALL_SUCCESSOR_V2.json` is reviewed.

## Why a successor is necessary

`DelayedRecallBenchmark` has deterministic hidden-state evolution but stochastic
observations: Gaussian cue noise is used at t=0 and fresh Gaussian distractor noise is
drawn at every t>0 observation. Current source now reports those semantics explicitly.
The frozen v1 study remains valid as executed, but its whole-rollout MSE mixes pre-recall
distractor timesteps with the recall transition and therefore is not a direct
recall-specific endpoint.

## Frozen primary endpoint

The successor primary endpoint is **recall-window memory-payload MSE**.

- benchmark delay: t=8
- observation times: t in {8, 9, 10, 11}
- target tensor indices: {7, 8, 9, 10}, because target index j corresponds to
  observation time t=j+1
- channels: the 32 memory-payload features only
- excluded channel: the final clock feature
- aggregation: mean squared error over batch, the four frozen time points, and all
  memory-payload features

The window is intentionally fixed before outcome access. It may not be moved to where a
model happens to perform best.

Whole-rollout MSE, final-step MSE, rollout MAE, and final-step MAE remain required
secondary metrics.

## Observation-noise reference

For t>0 the target observation adds
`distractor_scale * N(0, I)` independently to each memory feature. With the frozen
`distractor_scale=0.3`, a conditional-mean predictor that knows the latent memory still
has an expected per-memory-feature MSE floor of 0.09 from observation noise alone.

This 0.09 value is a diagnostic reference, not a trainable baseline and not a target
that FIM is expected to beat on finite samples.

## Matched trajectory/noise contract

Fresh successor seeds are 607, 701, 809, 907, and 1009. For each seed, the complete
train/validation/evaluation trajectory tensors must be generated **once**, retained,
SHA-256 hashed, and reused byte-for-byte by all four variants:

1. full
2. no_memory
3. no_retrieval
4. no_salience_gating

A variant-specific regeneration, even from the same nominal seed, does not satisfy the
successor contract. Batch size remains one and model memory is reset between independent
episodes.

## Fixed budget

The successor retains the bounded v1 training scale so the endpoint change is not
confounded with a larger optimization budget:

- 4 epochs
- 32 retained trajectories per seed
- batch size 1
- 16 training rollout steps
- 20 evaluation rollout steps
- teacher forcing ratio 0.5
- horizon decay 1.0
- dynamic data disabled
- EMA disabled
- no best-checkpoint selection

## Analysis and falsification rule

The primary paired comparison is full versus no_memory. The mechanism comparison is full
versus no_retrieval. Report every per-seed value, paired mean deltas, a deterministic
10,000-sample paired bootstrap interval, and the exact sign-flip p-value as descriptive
inference.

A positive memory-benefit statement is not permitted unless the paired mean primary
delta favors full FIM and full wins at least four of the five frozen seeds. A
retrieval-specific statement additionally requires the same directional rule versus
no_retrieval. These rules are deliberately stronger than choosing the best-looking mean
after execution.

Every failed cell remains in the ledger. After outcome access, do not replace seeds,
increase the budget, move the recall window, change the endpoint, or tune the mechanism
and still call the result this protocol.

## Review gate

The machine-readable contract currently sets both
`execution_authorized=false` and `outcome_access_allowed=false`. A separate review
must verify source identities, endpoint indexing, matched-trajectory materialization,
and the frozen analysis rule before either flag can change in a later commit.
