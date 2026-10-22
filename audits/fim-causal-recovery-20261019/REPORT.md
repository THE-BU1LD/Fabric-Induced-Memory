# FIM causal memory and recovery: execution report

19 October 2026. **Implemented and locally tested.** The maintained v2 CPU
engineering blocker has been reduced by actual numerical execution: 52 tests
passed, four CUDA tests skipped, 23 standard-library controls passed, and one
bounded end-to-end fixture completed. This is not scientific contribution,
public-benchmark superiority, independent reproduction or research completion.

## Identity and canonical state

- Canonical repository: `THE-BU1LD/Fabric-Induced-Memory`.
- Base commit: `e1ec0de48b915002418fde7625bdf96659023815`.
- Isolated development branch: `codex/fim-causal-recovery-20261019`.
- Implementation/test commit: `7b9c2af4251248eb155e07ab88ea00cbca46184d`.
- Runtime source SHA-256: `601db1e7acc5be34d20e1913225b3e9ec6d108a36dfdda8b73f2307e6d461291`.
- The documentation/evidence commit is the commit containing this report; its
  parent is the implementation commit. No claim depends on an invented future SHA.
- Sole writer: `/root`; two read-only assistant reviewers did not run a second
  compute lane. They are not external independent reproduction.

Start future work with [`CURRENT_RESEARCH_STATE.json`](../../CURRENT_RESEARCH_STATE.json),
then the exact version-specific state and frozen evidence. The full prior tree
identity is retained in `BASE_TREE.txt`; `PROTECTED_FILES.json` checks preservation.
Exact changed paths are in `CHANGED_FILES.txt` and the two-commit Git diff.

| Evidence line | Inspected state | Relationship to this execution |
|---|---|---|
| Original FIM | Historical compact/paper evidence; retired shared-minibatch 24-cell study; corrected 40-cell confirmation held | Preserved; no fresh performance claim |
| Historical ITFM v2 | Retained synthetic study and 66/66 remote replay checks | Separate from maintained `fim-sota` runtime |
| Maintained v2 engineering | Dated Oct 16 report had 23 stdlib passes and no admitted tensor run | This report supplies fresh CPU numerical evidence |
| ITFM v3 | 0/30 complete cells, consumed campaign, frozen sources | Preserved; not restarted |
| ITFM v4 | Separate runtime branch at `266fe56ff9897e57b7e610fb7afc369fe767855b`; historical 87 scalar + 10 tensor passes; contribution blocked | No common ancestor with canonical main; not this runtime's validation |

The old v4 ALL_ITEMS counts and dated Mac handoff remain historical observations.
Later v4 software checks do not resolve its scientific contribution gate. Root
`RESEARCH_TRUTH.md` contains a dated workflow-guard observation; the current
scientific ablation job does have a manual-dispatch guard. That freshness finding
does not authorize dispatch. ENGRAM and Worldcastr remain outside this work.

## Repairs and why they matter

`memory.py` now limits value-control donors to each query's mature same-owner
pool. Previously, a global rotation could inject another owner's or a future
residual while reporting metadata for an eligible key. New donor records identify
the actual contributing values and their weights separately from key identity.
The operation is deterministic cyclic mispairing, not a uniform random shuffle.

The same file fixes invalid-slot softmax weighting, preserves stored masks in
approximate reranking and rejects ambiguous arrival times. Invalid scores no
longer absorb probability from valid negative similarities. Unknown owners and
empty histories return zero correction. `adapter.py` now requires explicit
target availability before residual writes, preventing an inferred one-step
arrival for a longer-lead target.

`training.py` binds optimizer positions to ordered model parameter names.
Foreign, duplicate or omitted trainable parameters are rejected, and incompatible
same-shaped reordering is detected before checkpoint state can mutate the model.
The existing strict optimizer/RNG/scheduler/state recovery checks remain intact.

`engineering.py` includes the new causal suite in the admitted test inventory
and portable archive. Three test files cover the repairs, including independent
Python retrieval calculations, future/owner poisoning and unequal donor weights.
The only tolerance adjustment bounds independent float64 summation roundoff;
bitwise replay, predictions and recovery assertions were not weakened.

Read [`FLAW_AUDIT.md`](FLAW_AUDIT.md) for the complete inspected
flaw → cause → severity → solution → required-evidence ledger, and
[`TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) for shapes, equations,
training, prediction order, controls, checkpoint boundaries and conditional
mathematical justification. No proof of universal predictive improvement exists.

## Actual verification, including failures

| Check | Passed | Failed | Skipped / blocked | Retained receipt |
|---|---:|---:|---|---|
| Original CPU numerical suite | 29 | 1 | 4 CUDA skipped | `baseline_cpu_tests.json` |
| New regression cases before functional fixes | 31 | 19 | 4 CUDA skipped | `regression_before.json` |
| First repaired invocation | 0 | 0 | Resource admission blocked; zero tests executed | `repaired_cpu_tests_01.json` |
| Repaired suite before final review cases/format | 50 | 0 | 4 CUDA skipped | `repaired_cpu_tests_02.json` |
| Final source numerical suite | 52 | 0 | 4 CUDA skipped | `final_cpu_tests.json` |
| Standard-library controls | 23 | 0 | 0 | `final_control_tests.json` |
| Ruff check and formatting, seven runtime/test files | — | 0 | 0 | `final_lint.json`, `final_format.json` |
| Single software smoke cell | 1 completed | 0 | No CUDA execution | `smoke_launch_02.json` |
| Source/config/data/artifact analysis | Verified | 0 | No new model run | `smoke_analysis.json` |
| Retained prediction recomputation | Verified | 0 | Post hoc descriptive analysis | `posthoc_diagnostics.json` |
| Fresh extracted portable directory | 52 numerical + 23 control | 0 | Same four CUDA skips; same host/environment | `BUNDLE_EXECUTION.json`, `bundle_cpu_tests.json` |

The final suite collected 56 cases and finished pytest execution in 4.59 seconds.
The four skipped cases are CUDA training, CUDA checkpoint replay, FP16 and BF16
checks. They are unexecuted hardware coverage, not passes. The first baseline
failure was a roughly 2e-15 discrepancy in an independently accumulated scalar
metric; its raw failure is retained. The pre-fix regressions separately reproduced
real causal, numerical, arrival and optimizer defects.

The host was measured as Linux x86-64, Python 3.12.14, PyTorch 2.11.0+cpu,
NumPy 2.4.4 and an 8 GiB cgroup, with one CPU computation lane, FP32 and one
thread. Pytest 9.1.1 and Ruff 0.17.0 were installed in an isolated environment.
Exact installed packages are in `environment-lock.txt`. The 512 MiB admission
floor was retained. Earlier low-memory refusals and task-owned clean setup-cache
release receipts are preserved; no unrelated caches or jobs were modified.

The session declared a 1,800-second child-compute ceiling. The smoke cell kept
its 120-second maximum and a 30-second shutdown reserve. An initial campaign
budget of exactly 150 seconds refused dispatch after about .002 seconds of
setup, with zero cells and zero reserved seconds. A new manifest added 30 seconds
of setup margin within the session budget. `SMOKE_ADMISSION_02.json` records this
before execution. It was not an outcome-driven retry, and the refused attempt
remains under `runs/`.

## What actually ran and what it found

The unchanged existing smoke configuration uses seed 101, four/two/two
train/validation/test owners, twelve frames, three observed context frames, eight
spatial sites, a 536-parameter MLP, two epochs, batch four and accumulation two.
Ten optimizer updates process 72 example presentations. The single completed
launch took **7.335284 seconds end to end**. Training's own retained elapsed
counter is about .556 seconds; that is not total process time or a speed claim.

All five controls share one validation-selected checkpoint. Each control's
scale is chosen from `{0,.25,.5,1}` on validation only and frozen for test.
Actual overall results, generated from retained metrics, are in
[`SMOKE_METRICS.md`](SMOKE_METRICS.md):

| Method | Test MSE |
|---|---:|
| Base MLP | 0.0725440009565 |
| Full residual retrieval | 0.00835317594033 |
| Erased values | 0.0725440009565 |
| Spatial mean residual | 0.0725251487591 |
| Last residual | 0.00806379866561 |
| Persistence | 0.000337322700398 |

**Full memory loses to both last residual and persistence in this diagnostic.**
It improves over the weak two-epoch base, which is insufficient evidence of a
useful retrieval mechanism. Memory has 16 hits among 18 test forecasts, with
two empty-history first forecasts. The independent retained-prediction analysis
finds that those cold forecasts account for 93.0876% of its squared error.

Warm-only MSE is .0006495833 for full memory, .0003240339 for last residual and
.0003245119 for persistence. These are labeled post hoc diagnostics, not a
replacement primary metric or independent experiments. Even perfect cold-start
predictions would not make this fixed full-memory run beat persistence overall.
The exact paired-loss identity confirms that its extra warm correction is
misaligned with the last-residual control's remaining error on these observations.

The nine-step autonomous **base** diagnostic remained finite and had MSE
.0815668354. It never wrote unobserved future residuals and does not measure an
autonomous memory benefit. No capacity stress occurred: nine writes per owner
under capacity sixteen. Per-query latency, CPU process peak memory and energy
were not measured; efficiency claims remain unsupported. The prospective smoke
note mentioned relative L2, but this existing fixture emits MSE only; no missing
secondary metric is fabricated.

## Evidence and reproduction

`runs_02/cpu-smoke/` retains raw synthetic fields, per-forecast predictions,
best/latest safe checkpoints, training/evaluation JSONL, resolved config,
environment/resource observations, exact source snapshot and completion hashes.
The four tensor artifact files total under 425 KiB; there is no private dataset
or large production checkpoint. `EVIDENCE_SUMMARY.json` maps source, config,
generation, actual raw data and selected checkpoint identities.

Commands below use repository root and an environment with the retained pins:

```sh
python -u successor/itfm_v2/scripts/kaggle.py test --device cpu --receipt cpu-tests-new.json
PYTHONPATH=successor/itfm_v2/src python -m unittest discover -s successor/itfm_v2/tests -p test_engineering_control.py -v
python -u successor/itfm_v2/scripts/kaggle.py campaign --manifest audits/fim-causal-recovery-20261019/smoke_manifest_02.json --output runs/fim-causal-smoke-new
python -u successor/itfm_v2/scripts/kaggle.py analyze --root audits/fim-causal-recovery-20261019/runs_02 --output smoke-analysis-new.json
python -u audits/fim-causal-recovery-20261019/analyze_retained_fixture.py --run-root audits/fim-causal-recovery-20261019/runs_02/cpu-smoke --output diagnostics-new.json
```

Fresh output names are mandatory. Use `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.
Every numerical entry point above performs existing resource admission. Analysis
checks identities and artifact hashes before accepting a completed run. The
portable ZIP is generated by the existing export command and verified against
its complete inventory; see `BUNDLE_VERIFICATION.json` and the current Kaggle
handoff for exact launch and download steps.
The ZIP has 66 files and is 297,714 bytes. Its SHA-256 is
`5c2095b29124bfba27f0c0904f526424e47d9170b2086b6446db93f593362fe3`.
A fresh extracted directory passed the same 52 numerical cases, four hardware
skips and 23 control tests, and validated its included completed artifacts.
This is portability verification on the same environment, not an independent
empirical replication or additional unique test cases.

## Claim-to-evidence boundary

| Claim | Evidence | Classification |
|---|---|---|
| Identified causal/weight/recovery defects repaired | Pre-fix failures, final source hashes, regression receipts | IMPLEMENTED and LOCALLY_TESTED |
| Bounded v2 fixture completes and retains verifiable outputs | Completed run, checkpoints, raw data/predictions, independent metric reconstruction | LOCALLY_TESTED |
| Current full memory beats simple alternatives | Retained fixture shows the opposite ordering | Not supported |
| CUDA/AMP ready | Four hardware skips; unauthenticated Kaggle session | Not remotely executed |
| Useful new scientific contribution | Prior-art overlap and missing equal-history/strong-baseline comparison | Not established |
| Independently reproduced | Two assistant source reviews, same-session local checks | Not independently reproduced |
| Final research/paper/release ready | No admitted protected confirmation or locked scientific claim | Not complete |

## Remaining blockers and decision

**Engineering:** expanded controls/baselines, full cost and donor instrumentation,
generic reservoir/learned-key/evaluator semantics, and mid-evaluation continuation
remain outside the repaired path. Each is listed with its required evidence in
the flaw audit.

**Scientific:** a distinct contribution, strong baseline reproduction,
information-matched comparisons, public-data admission, independent-unit
uncertainty, practical-effect threshold and protected protocol remain unresolved.
[`BASELINE_PROTOCOL_NOT_RUN.md`](BASELINE_PROTOCOL_NOT_RUN.md) gives the next
discriminating design. It has not run and does not reopen v3/v4 allowances.

**Runtime:** no Kaggle connector was exposed; the connected local browser request
timed out; the cloud Kaggle page showed Login/Sign Up. Account, accelerator, quota
and active jobs were therefore unverified. No remote job was launched. GitHub
Actions billing usage is unavailable through the repository connector. The
reviewed push/PR triggers are avoided with `[skip ci]`; workflow files are not
changed and a skipped/pending check is not green CI.

**External review:** no independent outside reproduction, final paper verification,
release or submission. No merge to main, deployment, paid run, outreach or schedule
is part of this session.

The next single runtime action is to open the verified bundle on an authenticated
Kaggle session, inspect actual accelerator/quota/jobs, then run explicit CUDA
preflight plus the tiny CUDA test command. Do not launch the older development
campaign first. The next scientific decision is whether the bounded strong-base,
equal-history protocol is admitted and worth implementing; repeated tuning until
a positive result is not an acceptable acceptance rule.
