# ITFM v1 Project Contract

## Precise statement

Existing sequence and neural-operator models struggle when partial observations
alias different latent states and futures. We propose an owner-safe,
transport-aligned tensor residual memory because a genuinely informative
external memory should remain useful after accounting for finite history,
recurrent/SSM state, analytical transport, parameters, and retrieval compute,
and should produce predictable localized forecast changes under value erasure
and oracle-defined value swaps.

## Fundamental problem

Partial observation can make observed dynamics non-Markovian. A current field
may be compatible with multiple latent states, forcing a predictor to infer
missing state from history. Larger context, recurrent state, SSMs, observers,
memory kernels, and retrieval can all address this. A lower forecast error from
a memory-bearing model does not identify memory as the cause because it may also
have more parameters, information, updates, or a different optimization path.

The project therefore tests two requirements jointly:

1. predictive necessity beyond matched alternatives;
2. intervention-localized use of stored information.

## Mechanism

For context tensor `X_t` and current observation `x_t`, a shared encoder produces
a spatial query `Q_t` and base forecast `b_t`. Each completed historical
transition may store:

- tensor key `K_s` derived from information available at time `s`;
- tensor residual value `V_s = x_(s+1) - b_s` after `x_(s+1)` becomes observed;
- immutable trajectory owner, event time, and optional regime label.

At query time, exact per-location similarity is computed across eligible entries.
Top-k values are combined independently at each spatial location to produce a
retrieved residual field. A bounded normalized cross-correlation operator
estimates periodic translation between the final two observed frames. Keys and
residual values are stored in the resulting constant-velocity co-moving frame
and transformed back at the query target time, so alignment accounts for memory
age. A learned bounded gate adds the aligned field to the base forecast.

Eligibility requires identical trajectory ownership and `write_time < query_time`.
No entry from another independent unit or the future can be read.

## Contribution boundary

Not claimed as novel:

- external differentiable memory;
- key-value retrieval;
- state-space memory;
- neural operators with temporal memory;
- continuous spatial memory fields;
- partial-observation prediction;
- nearest-neighbor forecasting.

Candidate contribution:

- an end-to-end tensor-memory reference model whose values remain spatial and
  observation-interpretable;
- a leakage-resistant event-time/ownership contract;
- a joint predictive and interventional decision rule for determining whether
  forecasts depend directionally on mechanism-relevant stored information;
- a parameter-free transport alignment that makes the spatial transformation
  explicit and falsifiable rather than asking a retrieval gate to learn it
  implicitly.

## Adversarial flaw register

| Flaw | Cause | Severity | Repair | Required evidence |
|---|---|---:|---|---|
| Novelty collision | Prior work already combines memory, fields, operators, and partial observations | Critical | Make intervention identification the contribution | Full-paper and code comparison before submission |
| Retrieval is just longer context | The key encoder can summarize recent frames | Critical | Match accessible context and add finite-history/recurrent controls | Information-budget table and matched protocol |
| Extra capacity explains gains | Memory path adds parameters and stored values | Critical | Parameter- and compute-matched controls | Parameter, FLOP, update, and wall-time accounting |
| Future leakage | A transition value contains `x_(s+1)` | Critical | Write only after the observation arrives and enforce `write_time < query_time` | Event audit with zero violations |
| Cross-unit leakage | A global bank can mix trajectories | Critical | Immutable owner IDs checked on every query | Shuffle/rebatch tests and zero cross-owner reads |
| Alias task is trivial | A synthetic identifier may reveal the regime | Critical | Remove direct regime identifiers and add nuisance-matched controls | Oracle Bayes analysis and simple lookup baselines |
| Transport alone solves the task | Two recent frames identify periodic velocity without memory | Critical | Add a zero-parameter bounded-cross-correlation extrapolator | Same-split analytical baseline result |
| Alignment adds compute rather than information | Transported retrieval costs more than raw retrieval | Critical | Train an identical sham-transport memory with permuted eligible values | Parameter/runtime/compute and accuracy comparison |
| Constant-translation assumption is wrong | Real dynamics can accelerate, deform, rotate, diffuse, or change topology | High | Bound claims to constant periodic translations and test noise/missingness/speed shift | Failure-stratified robustness results and later successor mechanism |
| Intervention is off-manifold | Arbitrary swaps can create nonsense values | High | Swap only between oracle matched alias pairs and report random swaps separately | Donor/recipient matching diagnostics |
| Memory optimization confound | End-to-end bank dynamics change gradients | High | Primary study uses detached completed-transition values | Gradient-path tests and frozen-bank comparison |
| Spatial locality is cosmetic | Full-field keys may retrieve globally similar entries | High | Compute similarity and top-k independently by location | Spatial intervention and locality metrics |
| Exact search does not scale | Cost grows linearly with entries | Medium | Bound science-scale capacity; defer ANN | Latency and memory curves before scaling claims |
| Current field may already suffice | Synthetic generator may not create true aliasing | Critical | Verify conditional future variance at fixed observed state | Oracle aliasing report before training |
| Strong controls may win | SSM/history models can infer hidden state | Critical | Treat this as a valid falsifier | Same-split matched baseline results |
| Synthetic-only effect | Hand-built task may encode the mechanism | Critical | Require later independent public-domain replication | Frozen second-system confirmation |
| Statistical pseudoreplication | Windows from one trajectory are correlated | Critical | Split and aggregate by trajectory | Clustered uncertainty over trajectory units |

## Chosen implementation

Selected: exact, owner-scoped, past-only tensor residual memory with local
top-k retrieval, age-aware co-moving periodic transport alignment, explicit inference
interventions, and analytical/history/recurrent/state-space controls.

Rejected for v1:

- repairing the predecessor vector bank;
- generic tensor memory without interventions;
- learned approximate search;
- trainable writes from protected outcomes;
- cross-trajectory retrieval;
- claims based only on aggregate rollout error.

## Checkpoint 1 decision

**PASS_BOUNDED_DEVELOPMENT.** The question, mechanism, falsifier, and novelty
boundary are explicit enough to justify a bounded implementation. Publication
novelty and confirmatory readiness remain unestablished.

