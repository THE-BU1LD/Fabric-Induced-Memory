# Maintained FIM v2 estimator: exact specification and evidence boundary

19 October 2026. Implementation commit:
`7b9c2af4251248eb155e07ab88ea00cbca46184d`.
Runtime source SHA-256:
`601db1e7acc5be34d20e1913225b3e9ec6d108a36dfdda8b73f2307e6d461291`.

This specifies the existing maintained engineering estimator and its repaired
contracts. It is a technical research record, not a completed novelty paper.
It documents the existing implementation after its software smoke; it is not
a retrospective claim of preregistration. The next comparison is separately
specified as NOT RUN.
The original FIM fabric architecture and separately versioned v2 historical,
v3 and v4 studies are not silently replaced by this small estimator.

## 1. Problem, objective and prior-art boundary

A forecaster may make errors whose structure persists over time. Once the true
observation arrives, that error can become a legal correction for a later
forecast. The question is whether matching contexts extracts information beyond
what a recent residual, an average residual, or a conventional local regressor
already supplies, and whether any benefit justifies its total cost.

The current model predicts a one-step field from three fully observed frames,
then adds a weighted average of mature residuals from the same owner. It does
not train a new memory representation during online evaluation. The scientific
objective concerns additional predictive information; software correctness and
container persistence alone do not establish that information.

[FreshCast](https://arxiv.org/html/2610.07834v2) already combines frozen
forecasting with refreshed memory, relational weighting and calibrated
correction. The present fixed cosine weighting is a simpler estimator with
different readout details. The [delay-aware adaptation audit](https://arxiv.org/html/2610.12232v1)
makes actual label arrival and cheap linear correction important comparisons.
[TAFAS](https://arxiv.org/html/2501.04970v1) uses partial-feedback calibration;
its task and revised predictions must not be equated with this one-step fixture.
These sources constrain novelty; none supplies a performance result for FIM.

The testable current statement is: context matching may help when mature
residuals with similar observed contexts predict the next error more accurately
than equally informed conventional correction. The smoke does not establish
that premise. A distinct mechanism satisfying the v4 contribution gate has not
been identified.

## 2. Variables, shapes and availability

For owner `o` and integer forecast origin `t`, let the observed field be
`x[o,t]` with shape `[C,S]`. Context `X[o,t]` stacks the most recent `L` frames
and has shape `[L,C,S]`. Batched context is `[B,L,C,S]`, target and output are
`[B,C,S]`, and flattened key dimension is `D=L*C*S`.

The smoke has `L=3`, `C=1`, `S=8`, `D=24`, and output dimension `P=C*S=8`.
Targets are `x[o,t+1]`, with explicit availability `a=t+1`. Time denotes discrete
observation indices; no physical time-unit, sampling-transfer or PDE claim is
implied. The maintained stream requires fully observed contexts and targets,
singleton chronological evaluation batches, disjoint owner IDs across splits,
and strictly increasing origins within an owner.

Memory entry `i` contains owner, normalized context key, key-validity mask,
base residual, event time, availability time, utility and immutable serial.
An entry is eligible only for its owner, after availability, and with the
configured minimum origin age. The smoke uses minimum age one, no maximum-age
limit, FIFO capacity sixteen per owner and no explicit age decay. Training and
different split/variant evaluations never share a bank.

The public adapter now refuses residual writes without explicit availability.
Memory times must be finite, integer-valued, int64-range values; booleans and
ambiguous fractional/nonfinite arrivals are rejected before mutation. This
removes premature inferred writes but does not implement a complete arbitrary
delay or partially observed scientific evaluation task.

## 3. Data generation and deterministic order

For each owner and channel, independent split generators draw phase uniformly
on `[0,2π)`, amplitude on `[.3,.6)`, drift on `[.04,.12)` and independent normal
innovations. At spatial site `s`, let `p=2πs/S+phase` and `u=t*drift`. The field is

$$
x_{t,s}=\tanh\{A\sin(p+u)+0.15\cos(2p-0.7u)+0.01\epsilon_{t,s}\}.
$$

The split seeds are the run seed plus 10003 (train), 20003 (validation) or
30007 (test). Owners are disjoint and generated in fixed order. No fitted
normalization, cache or outcome-based data selection is used. Raw tensors have
their own retained hash in addition to the generation/config/source identities.
The fixture is a bounded noisy periodic field, not a discretized PDE solver.

With twelve frames and three context frames, each owner supplies nine forecasts,
origins two through ten. Owner-major, increasing-origin order is fixed; training
does not shuffle it. Four training owners give 36 examples per epoch. Validation
and test each have two owners and 18 forecasts, or 144 scored field elements.
These elements are not 144 independent experimental units.

## 4. Base model, initialization and training

The existing `ResidualMLPControl` flattens the context and applies

$$b_t=W_2\operatorname{GELU}(W_1\operatorname{vec}(X_t)+b_1)+b_2.$$

`W1` is `[16,24]`, `b1` is `[16]`, `W2` is `[8,16]`, and `b2` is `[8]`.
There are 536 trained scalars. The output reshapes to `[C,S]`. Initialization
is the pinned PyTorch `nn.Linear.reset_parameters` procedure after the declared
seed. Deterministic algorithms are enabled. The retained environment is Python
3.12.14, PyTorch 2.11.0+cpu, NumPy 2.4.4, FP32 and one CPU thread.

The loss is mean squared error over output elements per example, then the mean
over examples. Adam uses learning rate .001 and its pinned framework defaults;
no additional learned retrieval parameters or memory training loss is used.
The smoke has two epochs, microbatches of four and accumulation over two
microbatches. Accumulation uses the actual number of examples, including the
final partial window; effective batch sizes are 8,8,8,8,4 per epoch. Gradient
norm clipping is one. No scheduler, AMP, compilation, multiprocessing,
distributed training or external pretrained backbone is used by this smoke.

Validation uses the base-model MSE. The lowest validation checkpoint is saved
separately as `best.pt`; `latest.pt` retains recovery state. Early-stop patience
exceeds the two-epoch run, so it does not create an unreported search. The run
completed ten optimizer steps and 72 example presentations. A two-epoch fixture
is not a claim of optimized backbone quality.

## 5. Frozen online prediction, in chronological order

1. Load the validation-selected checkpoint after verifying source, config and
   data identity. Freeze model weights and start an empty bank for this split
   and variant. The detached raw-context keys remain in one fixed coordinate
   system throughout evaluation.
2. At origin `t`, obtain only the observed context. Compute base prediction
   `b_t` and key `q_t=normalize(vec(X_t))`, with the implemented norm floor.
3. Filter retained entries by owner and maturity before choosing donors. Compute
   cosine scores in float32. Stored masks remain active even when a query mask
   is omitted. For the normal smoke path, choose exact top-k with `k=4`, descending
   similarity and ascending immutable serial for ties.
4. Normalize valid weights at temperature `.2`, centering scores before dividing
   by temperature. Invalid slots receive exactly zero mass. With no eligible
   entry, return zero correction and absent provenance without NaNs.
5. Form correction `c_t=sum_i w_ti r_i` and commit `prediction=b_t+alpha*c_t`.
   Each stored `r_i=x[i+1]-b_i` is a base residual, never a residual of a previous
   corrected prediction.
6. Only after prediction, use the new target to form its residual. Store the
   declared availability time `t+1`; it cannot affect the current prediction.
   Apply FIFO capacity independently by owner. The next query repeats this
   process with the same frozen backbone.

In compact form, with `A_t` the legal retained set,

$$
\hat x_{t+1}=b_t+\alpha\sum_{i\in\operatorname{topk}(A_t)}
\frac{\exp((s_{ti}-s_{t,\max})/\tau)}
{\sum_{j\in\operatorname{topk}(A_t)}\exp((s_{tj}-s_{t,\max})/\tau)}r_i.
$$

The generic approximate path uses random-projection candidates followed by
masked reranking. It is not used by the smoke; its mask consistency is tested
when the candidate set contains every row. General approximate-search recall
or speed is unmeasured. Invalid or unavailable observations cannot be justified
by an approximation.

## 6. Controls, calibration and provenance

The five smoke variants share one frozen checkpoint. `base` has no bank.
`memory` uses the full spatial residual. `erased` keeps memory operations but
zeros the correction. `spatial_mean` broadcasts each channel's residual spatial
mean. `last_residual` uses the most recent eligible residual with top-k one.
Persistence predicts the last observed frame and is scored alongside them.

For each variant, evaluate scales `{0,.25,.5,1}` on validation predictions and
choose minimum MSE, breaking ties toward smaller scale. Freeze that scale on
test. This minimizes validation error over a finite grid; it gives no guarantee
of test improvement. Mechanism interventions in a future causal comparison
should instead hold the full model's scale fixed, as specified in the NOT_RUN
protocol. Separate no-retrieval/reset/key-shuffle/finite-window ridge runner
variants have not been implemented or executed by this five-control smoke.

The repaired value intervention sorts each query's mature same-owner pool by
serial and cyclically shifts donors by a seed-dependent nonzero offset. A pool
of one is unchanged. Future and foreign-owner values cannot enter or determine
this rotation. This is deterministic mispairing, not a uniform random shuffle.
Key selection and key weights remain fixed while actual value donors change.

`BatchRetrieval` now reports strongest-key identity separately from strongest
key's value donor, plus every selected value donor's serial, event time,
availability and weight. Empty slots use `-1` and zero weight. Tests reconstruct
an unequal-weight answer from these records. The existing stream archive retains
best-donor times and occupancy but does not yet persist all these new per-donor
vectors; full provenance instrumentation is a remaining evaluation improvement.

## 7. Recovery and numerical contract

Schema-2 checkpoints safely store tensor/primitive state with atomic replacement
and `weights_only=True` loading. They bind source/config/data/batch/runtime
identity; include model, declared mutable state, optimizer, optional scheduler,
scaler, RNGs, cursor, completed updates/examples and cumulative budget; and
resume only after a completed optimizer accumulation window.

Optimizer positions are now bound to ordered model parameter names. Foreign,
duplicate or omitted trainable parameters are rejected. A reordered optimizer
cannot silently assign momentum to a different same-shaped parameter on resume.
Identity is compared before loading model or optimizer state. Historical
checkpoints are preserved; incompatible older identities fail closed.

Bitwise model, optimizer, scheduler, RNG and frozen replay checks remain strict.
The independent scalar reconstruction test uses a float64 accumulation-roundoff
bound, `4*element_count*machine_epsilon` relative and zero absolute tolerance.
That adjustment addresses different reduction order, not predictive error.

Evaluation restarts by replaying the whole stream from the frozen checkpoint.
Exact mid-evaluation bank continuation and an interrupted backward-pass restart
are not promised. The smoke never updates its backbone while retaining an
evaluation bank. Generic learned keys and changed backbones would require explicit
bank invalidation/versioning; the detached retrieval path is not proof of learned
addressing.

## 8. What can be proved, and what requires measurement

Let `r` be the next base residual and `c` a legal, target-free correction. For any
fixed scale, the exact expected squared-error difference is

$$\Delta(\alpha)=\alpha^2\mathbb E\|c\|^2-2\alpha\mathbb E\langle r,c\rangle.$$

If `V=E||c||²>0`, an unconstrained optimal scale is `E<r,c>/V`, clipped to the
permitted interval when constrained. Improvement at a positive scale requires
sufficient positive alignment with the future error. This identity does not
assume that such alignment exists or that validation estimates transfer.
It is not a proof that adding memory must help.

At scale one and after warm-up, the last-residual forecast is exactly

$$b_t+(x_t-b_{t-1})=x_t+(b_t-b_{t-1}).$$

It is persistence plus the base model's predicted increment. Let
`d=prediction_memory-prediction_last` and
`e_last=target-prediction_last`. Per example,

$$\|e_{last}-d\|^2-\|e_{last}\|^2=\|d\|^2-2\langle e_{last},d\rangle.$$

The retained-artifact analysis checks this identity numerically. It separates
the size of memory's departure from a recent residual from whether that departure
points toward the remaining error. It proves an accounting relationship, not a
causal explanation for the learned data-generating process.

Kernel smoothing can reduce noise and introduce bias from old or mismatched
residuals. A simple bias-variance formula additionally requires fixed/exogenous
weights or explicit conditional noise assumptions. Overlapping contexts may
depend on past residual noise, so independence must not be assumed casually.
Temporal staleness, loss of amplitude information and a weak initial forecaster
are falsifiable explanations, not established causes of all observed error.

## 9. Complexity, limits and next evidence

For `N` retained rows, key dimension `D` and output dimension `P`, exact cosine
work is proportional to `B*N*D`; this implementation's stable sorting adds
sorting work rather than providing a linear-time top-k guarantee. Residual
combination costs `B*k*P`. Stored tensors scale with `N*(D+P)` plus masks and
metadata. The full eligibility mask uses `B*N` space, and candidate reranking
does not establish a global sublinear-search claim. Owner capacity does not by
itself bound an unlimited number of owners; the fixture's owner count is finite.
Masked cosine also materializes feature-level temporaries of size
`O(B * min(N, chunk_size) * D)`, in addition to the `O(B*N)` eligibility mask.
These allocations contribute to peak memory even when every stored feature is
valid; chunking does not eliminate that workspace.

The smoke makes nine writes per owner under capacity sixteen, so it does not
stress eviction or long-horizon forgetting. Per-query latency and process peak
memory are not retained by the existing stream logger; no efficiency superiority
is claimed. The generic reservoir option is not a verified uniform historical
reservoir and is excluded from this FIFO result.

There is one generated repetition, two test owners and nine one-step predictions
per owner. The separate nine-step autonomous diagnostic uses only base-model
predictions after its initial context; no unobserved residual is written. It
cannot establish autonomous memory advantage, long-horizon stability,
sampling-interval transfer, parameter transfer, solver sensitivity or OOD
robustness. Those questions require separate admitted tasks and protocols.

The next scientific design is the baseline-first NOT_RUN protocol. No completed
paper, protected confirmation, independent reproduction or new contribution is
claimed from the engineering evidence in this directory.
