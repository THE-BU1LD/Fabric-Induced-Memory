# Prospective development specification

## Generator and information sets

For seed s, a shared dictionary has K independent 16-dimensional Gaussian cues
normalized to unit RMS, and a shared Gaussian mixing matrix W/sqrt(16). Each
owner independently receives K hidden 4-by-4 residual fields r[a,0], with iid
Gaussian entries of standard deviation 0.75. At each event all fields evolve as

    r[a,t] = sqrt(1-d^2) r[a,t-1] + d * 0.75 * epsilon[a,t].

Choose a uniformly random anchor a, then observe x[t] = cue[a] + sigma * noise.
The next revealed target is y[t] = tanh(cue[a] W) + r[a,t] + 0.03 * noise.
Distractor events instead use a fresh Gaussian cue and independent residual,
assigned a unique evaluator-side label that never repeats. This is an online
cue/response process with time-varying hidden state; it is not continuous physics.

Conditions (sigma, d, distractor probability, K): clean (0,0,0,24), noisy
(0.15,0,0,24), drift (0,0.05,0,24), distractors (0,0,0.35,24), capacity
(0,0,0,64), combined (0.15,0.05,0.35,64). Every bank has 32 FIFO entries.
There are 48 training, 16 validation and 16 evaluation owners, 192 events each,
with the first 32 events excluded from loss/reporting endpoints. All owners
remain independent units; paired seed effects, not event counts, inform uncertainty.

The model receives current cue and previously revealed cue/target pairs only.
Anchor IDs, repeat masks, owner number, latent residual, and generator matrix
are never predictive features. Opaque owner IDs select isolated banks only;
the implementation operates on independent batch rows and resets state for
every split/owner. The privileged oracle alone uses anchor IDs. Training,
validation and evaluation use distinct RNG streams and disjoint owner IDs.
Sharing cue dictionaries across splits is intentional; this is new-owner
generalization, not unseen-cue generalization. Randomization and IID values
remove v2's regime-in-owner-parity shortcut.

## Models

The common base b(x) is Linear(16,64), tanh, Linear(64,16). It is fitted on all
post-warmup training targets and then frozen. Every history predictor uses the
same residual e[t]=y[t]-b(x[t]); it may write e[t] only after predicting y[t].
All trained candidates use AdamW, learning rate .001, weight decay .0001,
gradient clipping 1, 40 epochs, owner minibatches of 8, and MSE over all
post-warmup events. No candidate is trained against the primary repeat mask.
The base is trained once per seed/condition; no evaluation target fits a weight.

Tensor retrieval considers the last 32 strictly earlier events from this owner.
Squared mean cue distance selects up to four nearest entries; softmax of
-distance/temperature weights their entire 16-element residual fields.
Prediction is b(x[t]) + alpha * weighted_residual. Temperatures {.05,.2} and
alpha {0,.25,.5,.75,1} are selected on validation primary NMSE. The bank stores
raw noisy cues; no hidden matching labels or learned oracle routes are used.
Equal distances resolve by earlier event time. No similarity gate can discard
difficult cases; a genuinely empty bank returns zero correction.

Baselines:

| Baseline | Definition | Information/state |
|---|---|---|
| Base | Frozen b(x) | Current cue only |
| Last value | Latest observed residual | Past-only; same scale grid |
| kNN1 | Nearest residual | Same 32-entry bank |
| kNN4 | Uniform mean of four nearest residuals | Same bank and values |
| Recent | Weighted top-4 restricted to ages 1 through 8 inclusive | Nonempty after event 0 |
| Finite history | MLP on 8 causal feature tokens | Width 64 or 128 |
| GRU | One-layer GRU with linear residual head | Width/state 64 or 128; entire past stream |
| Attention | One transformer layer, 4 heads, FFN 2*width, sinusoidal positions, dropout 0, linear head | Local causal 32-token window; width 64 or 128 |

History tokens concatenate x[t], x[t-1], e[t-1]; first-event previous fields
are zero. Finite-history padding is zero; GRU initial state is zero. Attention
explicitly masks future tokens and those older than its declared window. For
each trained family, both widths are trained for equal updates and retained;
the best width and alpha are chosen on validation. Retrieval families retain
both temperature candidates, including redundant top1/uniform choices.

This matches training streams, targets, base and optimization-update budgets
among learned history candidates. It does **not** match parameter counts, state
slots or FLOPs exactly. Counts, state sizes and measured timings are reported;
the study cannot claim a compute-matched architecture ranking. The simple kNN
comparisons use exactly the same memory capacity and information. Tensor memory
may lose to these simpler comparators; that is a valid terminal answer.

## Interventions and oracle

Selected tensor settings remain fixed. Erasure replaces its retrieved value
with zero; sign inversion negates it; spatial mean replaces all 16 elements by
their mean; value permutation rotates eligible stored values by one position
within the same owner's FIFO bank, while keeping keys and weights unchanged.
Permutation validity is recorded (at least two eligible entries). It preserves
the bank's value multiset but destroys the key/value association. It is not a
cross-owner treatment and does not use current/future targets.

The oracle chooses the most recent matching hidden anchor among the same 32
strictly earlier entries and applies its stored observed residual with alpha=1.
Missing anchors give zero correction. This is a privileged diagnostic with
no fitting or unlimited history; it is not a mathematical lower bound on error.
Its gap may disappear under drift or eviction, which blocks a memory-recovery
claim in that condition. Donor/permutation values are retained as tensors;
an algebraically forced directional response is not presented as independent
evidence of practical benefit.

## Endpoints, decision, uncertainty

Primary rows are post-warmup recurring anchors whose most recent occurrence is
older than 8 events. This criterion is computed from event history, before
predictions, and is never a model feature. Average spatial squared error over
each owner's primary rows, divide by training target variance, then average
over all 16 owners. Require at least eight primary observations per owner;
otherwise retain INVALID/incomplete evidence rather than drop an owner.
Secondary endpoint uses all remaining post-warmup rows, with the same owner
aggregation. Report every baseline, control, seed, and condition.

Before evaluation data generation, select the strongest baseline by validation
primary NMSE (ties by protocol list order); save the settings, candidate scores,
training receipts and checkpoints. Do not select baselines or configurations
using evaluation scores. Evaluation here is a development holdout, not protected
confirmation. Raw arrays are retained for independent arithmetic and causal-
source verification.

For every seed in every non-clean condition require all of:

1. Oracle reduces base primary NMSE by at least 20%.
2. Tensor memory improves the validation-selected strongest baseline by at least 5%.
3. Tensor memory recovers at least 50% of the base-to-oracle error gap.
4. Erasure removes at least 80% of its base-relative gain.
5. Nonprimary NMSE exceeds base by no more than .01.
6. Permutation is valid on at least 90% of primary rows and raises NMSE by .01.

These are prospective development thresholds, not thresholds validated for a
natural task. A fully executed campaign failing any requirement is NOT_SUPPORTED.
Infrastructure failures are incomplete, never a negative scientific conclusion.
Clean-condition results are a calibration diagnostic, still fully reported.
Report seed-level paired differences and exact two-sided sign-flip p-values
descriptively. Five seeds have minimum possible two-sided p=.0625; there is no
population-significance claim and no timestep-as-independent-sample shortcut.

## Execution and evidence

One campaign, five seeds times six conditions, one CPU thread, 1800-second total
parent timeout, 2 GiB output ceiling, zero paid compute. A consumed budget or
failed fit cannot trigger automatic retuning/retry. Runtime tests must pass on
the exact frozen source before the campaign starts. Raw verification is a
separately guarded calculation; it cannot turn same-operator work into external
reproduction. All admission failures, partial epoch logs, checkpoints and source
snapshots remain available. Run and verification receipts are separate from the
scientific decision. Any external dataset study or protected confirmation must
be designed, licensed, budgeted and frozen separately.

The immediate admissible claim before execution is solely that this specification
and implementation exist. Novelty, runtime correctness, efficacy, independent
replication and publication readiness remain unestablished.
