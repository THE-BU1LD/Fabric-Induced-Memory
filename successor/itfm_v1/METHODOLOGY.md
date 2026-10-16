# ITFM v1 Development Methodology

## Evidence target

The experiment asks whether transport-aligned ITFM both predicts aliased
futures better than matched learned and analytical controls and responds
correctly to interventions that modify only its retrieved memory value. Either
requirement can falsify the v1 hypothesis.

## Dataset

`aliased-advection-v1` generates smooth periodic fields in paired families. The
two members of each family share a template and move with opposite unobserved
velocities. At oracle-defined periodic events their current observed fields are
identical while their next fields differ. Recent context can reveal velocity,
making the finite-history and recurrent controls deliberately strong.

Paired family identifiers are divided into train, validation, and test before
trajectories or windows are constructed. A family and its shared template can
appear in only one split. Trajectory owner identifiers remain unique within a
family and are carried through every memory operation.

This is a development construction. It cannot support natural-domain claims.

## Models

1. `itfm_transport`: primary tensor residual memory with bounded deterministic
   periodic transport inferred from the final two observed frames. Keys and
   residuals are canonicalized into a constant-velocity co-moving coordinate
   frame and mapped back at the forecast target time.
2. `itfm`: transport-off ablation.
3. `no_memory`: identical ITFM neural parameters with the external correction
   disabled.
4. `vector_broadcast`: identical memory-bearing architecture with global
   addressing and spatially pooled values broadcast across the field. This is
   the mechanism-off control for spatial structure.
5. `sham_transport`: identical transport-aligned architecture and retrieval
   compute, but eligible values are deterministically permuted during training
   and evaluation. This is the parameter- and compute-matched information
   control.
6. `finite_history_cnn`: a declared-width convolutional history model; the
   runner selects the integer width closest to ITFM's trainable parameter count.
7. `conv_gru`: a recurrent convolutional state baseline with width selected by
   the same parameter-count rule.
8. `gated_diagonal_ssm`: a stable input-selective diagonal state-space baseline
   with parameter-matched width.
9. `periodic_xcorr`: zero-parameter periodic extrapolation from the final two
   observed frames using the same bounded normalized cross-correlation operator.
   This is the simplest explanation for success on advection.

All models receive exactly the same stacked context. Learned models use the same
training epochs, optimizer, learning rate, weight decay, event-pure batches,
trajectories, seeds, and validation outcomes. The analytical baseline is not
optimized. Parameter counts, selected widths, runtime, and memory are retained.
The three learned architectural baselines must fall within 5% of the primary
model's trainable parameter count or the affected cell fails.
Training examples, optimizer-step counts, and evaluation-row counts are retained
to expose update-budget mismatches directly.

## Memory integrity

- A value is committed only after its target observation arrives.
- Every entry records trajectory owner and source event time.
- Retrieval requires equal owner and strictly earlier event time.
- Stored keys and values are detached from autograd.
- Exact search is used; no ANN approximation or hidden cache is permitted.
- A prediction with no eligible memory reduces exactly to the base path.

## Intervention matrix

The full model is evaluated sequentially under:

- ordinary retrieval;
- erased retrieved value;
- deterministic nuisance eligible value (permutation, or magnitude-preserving
  sign reversal when only one eligible value exists);
- retrieved-field swap between the two oracle-matched family members;
- oracle residual insertion as a correction-gate diagnostic, not an upper bound.

Only the ordinary pass may write the completed transition. Intervention passes
cannot mutate later memory state.

## Endpoints

Primary endpoint:

- alias-conditioned normalized MSE, normalized by training-target variance,
  averaged within trajectory and then across paired families.

Secondary endpoints:

- overall and non-alias NMSE;
- erasure degradation;
- swap donor-direction score;
- random-value specificity gap;
- fraction of queries with eligible memory;
- eligible-entry counts;
- trainable parameters, runtime, and peak memory.

The oracle condition is diagnostic and cannot be used as evidence of learned
performance.

## Robustness, OOD, and sensitivity

Every primary model is evaluated without additional fitting under context
noise (`sigma=0.05`), 25% context missingness, and a held-out speed-2 dynamics
shift. Each primary ITFM checkpoint is also evaluated at top-k values 1 and 8
and maximum transport radii 1 and 3 without retraining. Nine models across three
seeds produce 27 declared experiment cells; sensitivity does not consume
additional training runs.

The speed-2 shift reuses the generator seed, family split, and templates from
the in-distribution corpus; only the physical displacement speed changes. This
prevents an initial-condition or split change from being misreported as a speed
effect.

## Development decision rule

Continue this mechanism toward a confirmatory design only if:

1. transport-aligned ITFM improves alias-conditioned NMSE by at least 5%
   relative to the strongest declared control, including analytical transport;
2. erasure degrades aliased performance in the predicted direction;
3. matched value swaps move predictions toward donor futures;
4. random/nuisance values do not reproduce the effect;
5. non-alias performance is not materially degraded;
6. no ownership, event-time, split, or artifact-integrity violation occurs.

Otherwise v1 is negative, inconclusive, or invalid according to the recorded
failure. The protocol is not to be expanded merely to find a positive result.

## Statistical treatment

Development results are descriptive. The eventual full development matrix uses
three optimization seeds. Windows are not treated as independent units.
Uncertainty is clustered at paired-family level using 10,000 bootstrap
resamples. Declared secondary pairwise comparisons use Holm adjustment.

No development p-value authorizes a confirmatory or publication claim.

## Execution and failure preservation

The runner writes a manifest before the first fit, checkpoints every completed
run, hashes protocol, source, checkpoints, ordinary predictions, robustness
predictions, and intervention predictions, and retains exception type, message,
traceback, and runtime for failures. Sensitivity predictions are retained too.
Resume validates every retained artifact and the expanded scientific source
identity. Pipeline receipts are append-only with a latest pointer.
A matrix is complete only when every expected cell has a completed artifact and
there are zero retained failures.

## Protected boundary

No protected public-domain dataset has been selected or created. Development
outcomes may inform one bounded revision within the declared budget. Any
materially changed mechanism becomes v2. A later confirmatory protocol must be
versioned and frozen before its protected outcomes are inspected.

