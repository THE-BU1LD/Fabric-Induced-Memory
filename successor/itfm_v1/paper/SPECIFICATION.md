# Transport-Aligned Interventional Tensor Fabric Memory

## 1. Introduction

Partially observed dynamical systems can be non-Markovian in observation space:
the same observed field may correspond to different hidden states and future
evolution. Finite windows, recurrent state, state-space models, observers,
memory kernels, and external retrieval are competing solutions. Merely obtaining
lower error with a memory-bearing model cannot show that external memory stores
uniquely useful information.

ITFM is designed to make memory use inspectable and falsifiable. It retains
spatial keys and observation-residual values, enforces independent-unit and
event-time eligibility, and exposes inference-time interventions that change
memory without changing model parameters or current inputs. Its development
candidate also applies an explicit bounded translation group action inferred
from recent observations; transport-off and analytical transport-only controls
separate memory content from alignment.

The hypothesis is not that tensors are universally superior to vectors. It is
that spatially structured external values can preserve localized information in
aliased partially observed dynamics, and that genuine use of this information
must be visible under directional counterfactual interventions.

### 1.1 Related-work boundary

External differentiable memory, key-value retrieval, recurrent tokens,
state-space memory, Mori--Zwanzig memory operators, spatiotemporal neural
operators, and continuous neural memory fields all predate this project. The
candidate contribution is therefore not "memory as a tensor." It is the joint
identification design: owner/time-safe spatial residual memory, explicit group
transport, matched sham information, and counterfactual memory interventions.
Novelty remains provisional until full-text and implementation comparison is
completed before submission.

### 1.2 Predeclared hypotheses

- H1: the transport-aligned model improves alias-conditioned NMSE by at least
  5% over the strongest declared control.
- H2: erasure and nuisance-value interventions degrade aliased predictions.
- H3: matched value swaps move predictions toward donor futures.
- H4: the effect does not require cross-owner or future access and does not
  materially harm non-aliased cases.

Failure of any required gate rejects advancement of this mechanism.

## 2. Problem formulation

Let latent state `s_t` evolve under unknown dynamics and let the observed field
be

`x_t = h(s_t) in R^(C x H x W)`.

Aliasing exists when two states `s_a != s_b` satisfy `h(s_a) approximately
h(s_b)` but have materially different conditional futures. A context tensor is

`X_t = concat(x_(t-L+1), ..., x_t) in R^(LC x H x W)`.

The predictor estimates `x_(t+1)` using `X_t` and an external memory containing
only transitions completed before time `t` for the same independent trajectory.

### Identifiability target

The memory contribution is operationally identified only when:

1. transport-aligned ITFM exceeds the practical margin against the strongest
   declared control, including analytical transport, on
   alias-conditioned error;
2. erasing eligible values degrades aliased predictions;
3. swapping values between oracle matched alias pairs shifts the prediction
   toward the donor future;
4. random or nuisance-matched values do not produce the same effect;
5. non-aliased controls remain materially unchanged.

## 3. Model specification

### 3.1 Base predictor

Input: `X_t` with shape `[B, LC, H, W]`.

An encoder `E_theta` produces

`Z_t = E_theta(X_t) in R^(B x D x H x W)`.

Residual convolutional blocks use group normalization, depthwise spatial
convolution, pointwise mixing, GELU activation, and bounded residual scales.

The base head produces

`b_t = D_theta(Z_t) in R^(B x C x H x W)`.

### 3.2 Tensor query

A `1 x 1` projection produces a query tensor

`Q_t = normalize_channel(W_q Z_t) in R^(B x K x H x W)`.

Normalization is per spatial location with epsilon `1e-8`.

### 3.3 Memory entry

After `x_(s+1)` becomes observed, transition `s` may be written as

`e_s = (K_s, V_s, owner_s, time_s, regime_s)`,

For the transport-aligned variant, let the bounded displacement estimate at
write time be `delta_hat_s` and define the constant-velocity coordinate
`c_s=s delta_hat_s`. The stored entry is defined by

- `K_s = stop_gradient(T_(-c_s) Q_s)`;
- `V_s = stop_gradient(T_(-c_(s+1)) (x_(s+1) - b_s))`;
- `owner_s` is the immutable trajectory identifier;
- `time_s` is the source event time;
- `regime_s` is optional and used only for development diagnostics or declared
  oracle interventions, never as a model input.

Keys have shape `[K,H,W]`; values have shape `[C,H,W]`. The fixed-capacity bank
uses FIFO replacement in v1. Replacement is deterministic and not learned.
The transport-off ablation stores the untransformed query and residual.

The implementation generalizes this operator to spatial shape
`S=(S_1,...,S_r)` for `r>=1`: keys are `[K,*S]`, values are `[C,*S]`, and
top-k selection is performed independently at each spatial index. Convolutional
predictors are supplied for ranks 1, 2, and 3. The v1 scientific protocol tests
only rank 2; other ranks are implementation scope and cannot support an
empirical claim until separately protocolized.

### 3.4 Eligibility

For query `(owner_b, time_t)`, entry `i` is eligible iff

`owner_i = owner_b and time_i < time_t`.

The implementation raises an error on malformed owner/time metadata and returns
zero retrieval with a false-validity mask if no eligible entry exists.

### 3.5 Query-aligned retrieval in a co-moving fabric

The query is first placed in the same canonical frame:

`Kquery_t = T_(-c_t) Q_t`, where `c_t=t delta_hat_t`.

At each spatial position `(u,v)`, similarity is

`a_(b,i,u,v) = <Kquery_(b,:,u,v), K_(i,:,u,v)> / tau`.

Ineligible entries receive negative infinity. The top `k` entries are selected
independently at every spatial position, and softmax weights are computed across
those entries. The retrieved residual field is

`Rbar_(b,:,u,v) = sum_j softmax(a_j) V_(j,:,u,v)`.

Exact search costs `O(B N K product(S))` time and
`O(N (K+C) product(S))` storage for bank size `N`. Approximate search is outside
v1.

### 3.6 Gated correction

Before correction, the development candidate estimates a bounded periodic
translation. Let `T_delta` be the discrete translation group action and let
`D={-d,...,d}^2`. For the final two observed frames,

`delta_hat_t = argmax_(delta in D) <normalize(T_delta x_(t-1)), normalize(x_t)>`.

Normalization is over all channels and spatial positions of each item. For
spatial rank `r`, exhaustive estimation costs
`O(B C product(S) (2d+1)^r)` time and stores only the score table and candidate
shifts. This exponential dependence on rank and radius is acceptable only for
the bounded development setting; no scalability claim is made.

The aligned memory field is

`Rtilde_t = T_(c_(t+1)) Rbar_t`.

Thus an entry written for target time `s+1` is transported by the net
coordinate difference `c_(t+1)-c_(s+1)`, rather than receiving an age-blind
one-step shift after retrieval.

The argmax is deterministic and parameter-free. The radius must be strictly
less than half of every periodic spatial dimension so candidate translations
remain unique. Ties prefer the minimum L1 displacement and then lexicographic
order, so an uninformative field selects the identity. The formulation assumes
the locally estimated integer displacement is constant from the time origin;
acceleration or deformation is a declared failure regime. The transport-off
ablation uses `Rtilde_t=R_t`; the analytical baseline predicts
`T_(delta_hat_t) x_t` without memory. This formulation is exact only for bounded
periodic translations and is not claimed to represent general deformation.

The model forms

`g_t = sigmoid(W_g concat(Z_t, Rtilde_t))`,

`xhat_(t+1) = b_t + tanh(alpha) g_t * Rtilde_t`,

where `alpha` is learned and bounded by the configured maximum feedback scale.
It is initialized to a small positive value (`feedback_logit = 0.1`) so the
memory path receives gradients from the first optimization step.
When retrieval is invalid or erased, `Rbar_t = 0`, so the prediction reduces
exactly to the base path.

### 3.7 Interventions

All interventions occur after eligibility and retrieval selection, without
changing parameters or current inputs:

- `none`: use eligible retrieved values;
- `erase`: replace values by zero;
- `random_value`: deterministically permute eligible values within owner; when
  only one eligible entry exists, negate it so the sham cannot silently equal
  ordinary retrieval while preserving magnitude and the retrieval compute path;
- `value_swap`: replace selected values with values from declared matched donor
  entries;
- `oracle_value`: inject a declared oracle residual as a diagnostic of the
  learned correction gate. It is not an upper bound because the learned gate
  and feedback scale remain active.

Matched donor and oracle replacements are defined in post-transport correction
coordinates and therefore are not transported a second time.

Intervention metadata and selected entry indices must be returned with every
prediction.

## 4. Training

The primary development implementation trains neural models using one-step MSE
over training trajectories. Batches contain only one event time; owners may be
shuffled within that event, so optimizer ordering cannot introduce a future
transition. Memory keys and completed-transition values are detached; gradients
do not pass through stored history. The zero-parameter analytical baseline is
not optimized. Other learned models use the same context, optimizer, updates,
data, and seeds.

No protected outcome is used for model selection. Development validation may
select declared hyperparameters within the fixed budget. All attempted
configurations and failures must be retained.

Development execution is restricted to CPU or Apple MPS, uses one numerical
thread by default, and requests strict deterministic PyTorch algorithms.
Unavailable deterministic kernels are execution failures rather than warnings.

## 5. Inference

For each independent trajectory:

1. clear the bank;
2. construct context using observations available at time `t`;
3. query entries with matching owner and earlier event time;
4. produce the base and memory-corrected forecast;
5. after `x_(t+1)` becomes available, commit the completed transition;
6. never write protected targets before their prediction is scored.

Rebatching or shuffling must not alter ownership semantics.

## 6. Development experiment

The synthetic aliased-advection system contains a visible scalar field moved by
an unobserved regime-dependent velocity. Observation construction creates
matched fields whose immediate appearance is insufficient to determine the
next displacement, while recent evolution or a past transition can reveal the
regime. Trajectory IDs are split before windows are generated.

Declared models are transport-aligned ITFM, its transport-off ITFM ablation,
its no-memory path, a vector-broadcast memory control, an identical
sham-transport model trained and evaluated with deterministically permuted
eligible values, a parameter-matched finite-history CNN, a ConvGRU, a stable
input-gated diagonal SSM, and a zero-parameter bounded-cross-correlation
transport extrapolator.
The vector control globally pools keys and values and therefore removes local
spatial addressing while preserving the external-memory path. Robustness cells
apply context noise, missingness, and a speed-2 dynamics shift without refitting.
Sensitivity evaluation reuses each primary checkpoint and varies top-k and the
maximum transport radius without retraining. The full development matrix is
nine models across three seeds, or 27 declared cells; one model is analytical
and is repeated by seed only to preserve paired evaluation identities. The future confirmatory study
must use an independently frozen public partial-observation domain.

## 7. Endpoints and decision rule

Primary: alias-conditioned normalized MSE, averaged within trajectory and then
across trajectories.

Secondary: overall and non-alias NMSE, retrieval regime precision, erasure
degradation, swap donor-direction score, random-value specificity gap, runtime,
and peak memory.

Development continuation requires both:

- at least 5% relative improvement over the strongest declared control,
  including the analytical transport baseline, on the primary endpoint; and
- correct directional erasure and value-swap responses.

Failure is scientifically valid and closes this v1 mechanism unless a recorded
implementation or validity defect is demonstrated.

## 8. Failure behavior

- Cross-owner or future entries are ineligible, not down-weighted.
- Empty eligible memory returns the exact base prediction.
- Non-finite keys, values, or predictions raise an error.
- Shape mismatch raises an error before mutation.
- A failed or interrupted experiment remains retained and is never silently
  converted to a completed cell.
- Every checkpoint, raw prediction family, result, source contract, protocol,
  and external dataset file is hash-bound before analysis.

## 9. Claim boundary

This specification makes no claim of empirical advantage, natural-domain
generalization, state of the art, or publication novelty. Those require valid
executed evidence and later independent verification.

