# FIM / ITFM restart audit — 15 October 2026

**Current verdict: the project has a useful narrow synthetic result, but no demonstrated public-benchmark advantage. The prospective v4 contribution is not ready.** This directory contains an adversarial restart audit and evidence requirements. No v4 model, training campaign, protected evaluation or completed paper is claimed.

Read [RESEARCH_STATE.json](RESEARCH_STATE.json) before further work. The portfolio record is `../RESEARCH_STATE.json` relative to the repository root; its PR-012 entry points here for the current successor. Historical version states and frozen evidence remain authoritative for their own results.

The supplied request first asks for perfection and positive results, then explicitly requires finite budgets, acceptance of valid negative outcomes, and no implementation before a defensible contribution. This audit follows those explicit research gates. There is no valid guarantee that more iterations produce a better forecasting model.

## What is working, and what the results mean

| Version | Retained result | Defensible interpretation |
|---|---|---|
| Original FIM | Complete 24-cell component matrix; six comparisons split 3 favorable to full FIM and 3 favorable to removal; minimum exact p = 0.5 | Bounded negative component evidence; no reliable component superiority demonstrated |
| Original mini-suite | FIM 0.197582, DeepONet 0.157946, MLP 0.196575 on the retained error summary | FIM was not the strongest method in this suite |
| ITFM v2 synthetic | Query NMSE 0 for tensor memory on each of three frozen seeds; approximately 0.487685 without memory | An exact-residual repeated-key mechanism works on the designed task; this is not public generalization evidence |
| ITFM v2 replay | Retained aligned-environment strict checks 66/66; earlier mismatch history retained | Bounded repeatability on the tested environment; not independent scientific reproduction |
| ITFM v3 | Retained 12 runtime tests passed; zero of 30 complete campaign cells; partial clean/401 output before a resource stop | Software exercised, scientific campaign incomplete, one-campaign allowance consumed |
| Public ITFM-R | No admitted complete public campaign | Efficacy, superiority and SOTA remain not evaluated |
| Prospective v4 | Five concepts reviewed; 35 flaws or unresolved threats catalogued | Contribution gate not passed; no new scientific result |

These results are recorded in [the preceding status audit](../../audits/status-20261015-01/AUDIT.json) and [the retained execution diagnosis](../../audits/seed307-diagnosis-20261014/REPORT.md). Existing tests and artifact hashes establish their stated software/provenance properties only.

## What this audit established

The targeted literature review covers 15 primary-source records at the reading depths explicitly recorded in [CONCEPT_AUDIT.json](CONCEPT_AUDIT.json). It is not an exhaustive novelty certification. The prior proposal substantially overlaps [FreshCast](https://arxiv.org/html/2610.07834v1). Partial-feedback adaptation appears in [TAFAS](https://ojs.aaai.org/index.php/AAAI/article/view/33965), and causal-delay audits with simple correctors appear in [Training on the Future](https://arxiv.org/html/2610.12232v1). Their reported results are external claims, not our reproductions.

Five mechanisms were considered: streaming residual retrieval; drift-aware addressing and retention; compressed spatial residual state; coordinate-level observation-safe retrieval; and geometry-aware transport. None currently meets all conceptual requirements. The fourth is the strongest correctness reference, but a necessary bug fix is not by itself a new research contribution. The fifth lacks a defined valid transport law. The concept-review allowance was five and all five were reviewed; this audit does not authorize indefinite idea search.

The [executable source audit](audit_source.py) confirms seven specific findings in the existing public stack: duplicate matrix rows can obtain an affirmative decision; the secondary decision ignores a forbidden family regression; NaN/Inf pairs can pass metric comparison; two incompatible relative-L2 reductions are used; bootstrap weighting changes the equal-cell estimand; failed unmatched rows disappear from pairing; and the writer does not read the target observation mask. The last finding is static evidence, not a tensor leakage experiment. These findings are not asserted to apply to the separate executed v2 synthetic pipeline.

The refined audit uses separate duplicate and regression fixtures, four ordinary acceptance/rejection controls, and 27 exact rational identity checks. All passed their intended audit conditions. This means the defects were demonstrated, **not repaired**. Both original and refined receipts are retained under `audit/`.

Two resource samples gave 134.75 and 139.52 MiB free-plus-speculative memory against the unchanged 512 MiB floor. Pageouts and swap were stable across those samples; disk was sufficient; no matching study worker was found. Model execution remains unadmitted. Pinned upstream tree scans found no conventionally named license file in LANO or HS-FNO; that is a limited filename scan, not a legal conclusion about all possible terms. Applicable code/data terms remain to be established.

## Required work, in dependency order

1. **Resolve the scientific contribution.** Specify an identifiable deployment problem, a distinct mechanism beyond the nearest methods, its assumptions, and an experiment that can reject it. An observational correctness study is a possible different contribution, but must explain what the recent delay audits do not answer. Do not implement another architecture merely because it might score better.
2. **Freeze the specification and methodology.** Separate scorer truth from observable events, define exact arrival times and stable owners, preserve spatial residual information, define fallbacks and gradient expectations, and state the estimand. Admit real dataset versions, terms, splits and credible baselines. Record finite training/tuning/run/time/storage budgets and stopping rules before execution.
3. **Repair the measured validity defects in a new implementation.** Use observed-only events and causality tests; a single metric implementation with explicit reduction and zero-energy rules; complete unique cell/seed/trajectory inventories; finite values; all declared regression and uncertainty gates; evidence-backed reproduction status; and explicit failed-run dispositions. Never overwrite a frozen predecessor to make a new result appear historical.
4. **Demonstrate mechanism and engineering fidelity.** Check relevant gradients or declare parameters intentionally fixed; owner isolation under batch permutation; spatial/amplitude fidelity; full-size memory and latency; interrupted/resumed equivalence; and raw-result-to-table regeneration. The [35-item checklist](FLAW_CHECKLIST.md) specifies cause, severity, repair and evidence for each issue.
5. **Run a bounded development study only after admission.** First ask whether legally available history contains recoverable predictive value. Compare against frozen base, long-history models, simple residual/online regression, close retrieval methods, mechanism-off and information/compute-matched controls. A privileged oracle is a diagnostic, not a deployable score or necessarily the true optimum. If a simple equally informed control explains the gain, reject the complex mechanism.
6. **Freeze and perform confirmation, then audit and reproduce.** Keep protected outcomes unopened until the complete freeze. Retain every failure and seed. Classify the outcome honestly, trace each claim to raw evidence, complete the paper and reproduction package, and obtain actual independent verification before that claim is made.

This is an ordered acceptance plan, not an implemented model specification or an approved scientific-run budget. Checkpoint 1 must pass before the requested specification/implementation pipeline can advance. The public v2 protocol's 10% primary and 15% secondary improvement thresholds and other material gates remain unchanged; any materially different study requires its own prospective protocol.

## The ten checkpoints

| Checkpoint | Current state | Evidence needed to pass |
|---|---|---|
| 1 — Idea ready | BLOCKED: contribution not ready | Material distinction from close prior work, explicit assumptions and mechanism, meaningful falsifier and feasible bounded study |
| 2 — Model specification | NOT EVALUATED | Complete dimensions, state transitions, equations, initialization, training/inference, failure behavior and complexity |
| 3 — Methodology | NOT EVALUATED | Admitted data/baselines, independent units, controls, exact estimands, uncertainty, multiplicity, exclusions and finite budget |
| 4 — Model implemented | NOT EVALUATED | Faithful source, mathematical/shape/gradient/edge tests against the specification |
| 5 — Experiment system | NOT EVALUATED | Full methodology executable with raw outputs, provenance and retained failures |
| 6 — Run ready | NOT EVALUATED | Complete bounded dry run and full-shape resource/recovery profiling under admission |
| 7 — Confirmatory freeze | NOT EVALUATED | Immutable source/data/environment/protocol and untouched protected boundary |
| 8 — Evidence locked | NOT EVALUATED | Full frozen outcomes and claim-to-artifact audit, including negative or invalid evidence |
| 9 — Verified | NOT EVALUATED | Clean reproduction and a real independent review appropriate to the claim |
| 10 — Research complete | NOT EVALUATED | Defensible terminal conclusion plus complete inspected paper, evidence and reproducibility package |

The current blockers have separate meanings: an unresolved contribution blocks the research direction; code defects block trust in future public decisions; missing data/source admission blocks those integrations; insufficient memory blocks numerical execution. More compute alone resolves none of the conceptual or validity defects.

## What can be proved

[MATH_AND_FALSIFIERS.md](MATH_AND_FALSIFIERS.md) derives the exact residual-correction risk condition, the conditional-variance floor, and the observed-event state-invariance property. It also gives explicit counterexamples to universal improvement, unbiased recovery under arbitrary missingness, interchangeable metric reductions and overinterpreted erasure ablations. These distinguish correctness proofs from assumptions requiring empirical validation. No proof of future benchmark superiority exists.

To reproduce the source audit without importing a model, choose a fresh output path:

```sh
rtk python3 successor/itfm_v4/audit_source.py --output /tmp/fim-source-audit-new.json
```

The tool refuses to overwrite an existing receipt and fails if its recorded defect assumptions no longer hold. That exit behavior belongs to an audit of known defects, not a model-success gate.
