# Flaw, repair and required-evidence audit

19 October 2026. This ledger covers the inspected maintained v2 engineering
runtime and the research boundaries relevant to it. It is not a claim that all
possible flaws in every historical FIM implementation have been enumerated.
Code identity, actual test counts and retained failures are in `REPORT.md`.

## Repaired and locally tested

| Flaw | Cause and severity | Applied solution | Evidence and remaining limit |
|---|---|---|---|
| Value-shuffle control could use a future residual | Rotation over the entire bank; critical causal violation | Rotate only each query's mature same-owner donor pool | Pre-fix failure and passing future-value poisoning tests, exact/approximate |
| Value-shuffle control could use another owner's residual | Same global rotation; critical isolation violation | Owner eligibility constrains donors before permutation | Foreign-owner poisoning, unknown owner and singleton cases pass |
| Retrieved-value metadata could identify the key rather than actual donor | Keys selected before value swap; high auditability risk | Separate key identity and actual donor serial/time/availability/weight | Independent unequal-weight donor reconstruction passes |
| Invalid retrieval slots could absorb softmax mass | Invalid scores set to zero ahead of negative valid scores; high numerical correctness risk | Mask invalid logits, center valid scores, zero absent rows | Negative-similarity/low-temperature cases plus independent Python kernel calculation pass |
| Approximate reranking ignored stored masks without a query mask | Replaced candidate masks with all-valid; high masked-information risk | Preserve candidate masks | Exhaustive-candidate exact/approximate equality case passes; general ANN recall unmeasured |
| Adapter inferred target arrival one step later even for another lead | Missing availability default; high causal-interface risk | Require explicit arrival for residual writes | Five-step lead with delayed arrival seven is withheld until mature; ambiguous omission rejected |
| Fractional/nonfinite times could be truncated into legal arrivals | Blind cast to int64; high input-contract risk | Validate finite integral int64-range time values before mutation | Fractional/NaN/Inf regression cases pass |
| Optimizer state could bind to the wrong same-shaped parameter | Positional state without ordered named binding; high recovery risk | Bind optimizer groups to ordered model parameter names and compare before load | Foreign/missing/duplicate parameters rejected; reversed same-shape resume rejected before model mutation |
| Independent scalar metric reconstruction failed at roundoff level | Different float64 summation order; medium test-validity risk | Bound only independent-reduction roundoff by `4*n*epsilon`, absolute tolerance zero | Original failure retained; bitwise state/RNG/optimizer/prediction/replay checks unchanged and passing |
| Portable test inventory omitted the new causal suite | Export/CLI explicitly enumerate tests; medium handoff risk | Include new suite in both source inventory and export | 56 collected: 52 passed, four CUDA skips; portable inventory verified |

The regressions were written before functional repair. That run had 31 passed,
19 failed and four skipped. The final source also contains two additional
unequal-weight donor cases suggested by read-only review. Passing software tests
does not establish forecasting benefit, complete mathematical correctness for
every input, or CUDA execution.

## What the retained smoke explains

All five controls share one frozen MLP, one generated fixture and
validation-selected scales. Overall MSE remains the reported primary diagnostic.
The following cold/warm analysis is **post hoc**, not a new experiment or a
predeclared confirmatory subgroup.

Each owner's first prediction has no mature memory. The two initial forecasts
contribute 1.1197106749524406 of full memory's approximately 1.2028573354081566
squared error: 93.0876%. The empty-bank forecast equals the base model exactly.
This identifies a concrete cold-start weakness of this fixture's prediction
path, not merely a suspicion about optimization.

After warm-up, full memory MSE is approximately 0.0006495833, last residual
0.0003240339 and persistence 0.0003245119. The exact paired-loss decomposition
finds that retrieval's departure from last residual has mean squared magnitude
0.0002172664 and twice its alignment with the remaining error is -0.0001082830.
Their difference, 0.0003255494, exactly accounts for the warm MSE gap within
roundoff. The additional correction points away from the remaining error on
average in this retained fixture. This does not prove a general cause such as
staleness, amplitude loss or distribution shift.

Even hypothetical zero error on both cold-start forecasts would leave full
memory's warm squared error above persistence's total squared error. A fallback
alone therefore cannot make this full-memory run beat persistence. The backbone
and keys are frozen during evaluation, so optimizer-induced representation drift
cannot explain the full-versus-last difference within this run.

## Unresolved issues and discriminating next evidence

| Flaw or limitation | Cause / severity | Candidate solution or decision | Required evidence |
|---|---|---|---|
| Weak cold-start predictor | Empty memory falls back to a two-epoch MLP that is poor on this smooth fixture; high predictive limitation | Compare validation-selected persistence, linear predictor and adequately budgeted MLP before adding complexity | Fresh development protocol, shared data access and training/search costs; retain cold starts in overall metric |
| Full retrieval loses after warm-up | Older context matches alter the latest residual in an unhelpful direction here; high mechanism-identification limitation | Compare last residual, mature FIFO mean, uniform top-k and exact kernel/ridge with equal legal history | Paired errors, weights/ages and association interventions; no assertion that a proposed fix must win |
| No established novelty | Frozen-model memory plus calibrated residual correction overlaps prior art; critical scientific gate | Define a material mechanism difference and a falsifier against its closest conventional implementation | IDEA READY remains blocked until the distinction survives primary-literature and baseline review |
| No isolated evidence for information older than the visible context | Recent residual targets may already be in the three-frame observed context | If older information is the claim, require donor availability before the earliest visible frame in a separate protocol | With one-step targets and context length L, this requires origin age at least L+1, not merely L; compare an equally informed finite-history predictor |
| Missing strong task baseline | MLP has only two epochs; persistence dominates overall | Use the strongest validation-selected conventional predictor first | Upstream baseline reproduction for public benchmarks; equal observation and tuning opportunities |
| Incomplete mechanism controls in smoke runner | Five existing variants do not separately implement no-retrieval, query-reset or shuffled-key experiments | Implement only after the prospective baseline/control contract is admitted | Flags must change actual computation; hold information and full-model scale fixed where meaningful |
| Value intervention is not a uniform shuffle | Deterministic cyclic donor rotation | Describe it accurately and report zero/single/multiple-donor coverage | More than one eligible donor is necessary; scientific value-shuffle comparison has not run |
| Key loses absolute magnitude | L2-normalized raw context emphasizes shape; possible limitation, not diagnosed cause | Compare conventional level/amplitude/age features under the same information budget | Predeclared feature ablation and a matched kernel/regression reference; FreshCast overlap prevents casual novelty claims |
| Generic learned-addressing path is not verified | Retrieval detaches keys and bank tensors | Do not claim learned addressing; define a gradient/representation contract before using such a mode | Actual gradient tests, bank-version invalidation and end-to-end control evidence |
| Generic reservoir option lacks a historical-uniform guarantee | Sampling retained-plus-arriving entries is not a verified full-history reservoir | Keep this FIFO result separate; repair that option only if needed by an admitted task | Hand-verifiable replacement probabilities and long-stream inclusion-frequency checks |
| Partial feedback is not integrated in this fixture | Runner requires fully observed targets arriving at `t+1` | Separate coordinate-level masks/availability only for a declared partial-feedback task | Hidden-target mutation must not change legal state or prediction; compare identical masked estimators |
| No exact mid-evaluation continuation | Evaluation restarts from frozen checkpoint and replays whole stream | Keep the boundary explicit; persist bank/cursor only if a task requires it | Interrupt/restart replay equivalence including memory, ledger and provenance |
| Limited generic recovery proof | Specific model/optimizer/scheduler patterns tested; arbitrary schedulers and custom states can differ | Admit each additional mutable interface explicitly | State binding, serialization, pre-load validation and exact recovery for the actual configuration |
| Generic evaluator can overstate a causal claim | The unused `evaluation.py` helper sets `causal_effect_claim_allowed` from aggregate inequalities without matched identities or uncertainty | Treat that field as non-authoritative and the path as unadmitted; repair before any scientific use | Reject mismatched source/config/data/owner artifacts and apply a predeclared paired inference rule |
| No eviction or forgetting stress | Nine writes per owner, capacity sixteen | Do not infer capacity robustness or retention performance | Separate bounded stream exceeding capacity, explicit age/decay policies and matched-history baselines |
| Incomplete runtime cost telemetry | Stream retains occupancy but not per-query latency or process peak memory | Add instrumentation before an efficiency study | Measured total and incremental cost, retained bytes, all reads/writes/evictions, and peak memory |
| Approximate search quality unmeasured | Small tests exhaust the candidate pool | Keep exact retrieval as reference | Recall/error versus exact, real workload timing, identical masks and causal candidate filters |
| Single development repetition | One seed, two test owners, correlated spatial/time samples | Do not infer statistical support or universal negative performance | Independent repetitions and later protected precision/multiplicity planning |
| No OOD/dynamical-system evidence | Bounded periodic generator, one-step observed stream | Keep autonomous base diagnostic distinct from memory and PDE claims | Admitted datasets, shifts, solver/sampling/horizon definitions and frozen comparisons |
| GPU execution unavailable in this session | CPU-only build; Kaggle showed Login/Sign Up | Deliver source-bound portable launch package and use an authenticated admitted host later | Actual account/accelerator/quota/jobs, CUDA preflight, numerical test receipt, logs and artifacts |
| External verification absent | Two assistant reviews share project context; no outside reproduction | Provide exact source, environment, tests and raw artifacts | Independent environment/person executes the retained protocol and reconciles outputs |

## Execution limitations preserved

Initial memory admission failures remain retained. Only this task's clean setup
file cache was released; the 512 MiB guard and protected studies were not changed.
The first single-cell smoke manifest left no margin beyond its 120-second cell
and 30-second shutdown reserve. It was refused before any cell reservation.
A separate 180-second manifest added explicit setup margin within the original
session ceiling, with the same source/config/seed and no inspected outcome.
Both attempts remain in the evidence directory.

The prospective smoke declaration mentioned relative-L2 diagnostics, but the
existing runtime emitted MSE only. That missing secondary diagnostic is recorded
in `posthoc_diagnostics.json`; no value is fabricated and primary acceptance is
unchanged. Neither resource refusal nor a missing diagnostic is relabeled as
a passing test.

## Decision

The implemented causal and recovery defects are repaired and locally tested.
The broad question of useful context-matched residual memory is unresolved.
The current fixture's adverse baseline ordering is preserved. The preferred
next research step is an admitted strong-baseline/equal-history comparison,
not repeated tuning until a favorable number appears. No proposed improvement
above is represented as proved to improve predictive performance.
