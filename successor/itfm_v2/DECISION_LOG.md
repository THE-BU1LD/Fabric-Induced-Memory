# ITFM v2 decision log

## 2026-10-11 — Separate v2 from v1

V1 is preserved because it has a hash-bound protocol and receipts. V2 changes
the scientific question before any v1 fit completed; no v1 outcome was used.

## 2026-10-11 — Replace current-frame aliasing with full-context aliasing

V1 paired only the current observation while exposing velocity in the preceding
frames. V2 requires the entire visible context to match across paired futures.

## 2026-10-11 — Remove exact analytical transport as the answer

V2 uses repeated visible anchor contexts and hidden spatial innovations. A
zero-shift or last-frame predictor is the frozen conditional-mean base, not an
exact future solver.

## 2026-10-11 — Freeze the base before memory calibration

The base is the final visible frame. Residual values are therefore stationary.
The memory coefficient is a single validation-only closed-form scalar clipped
to `[0,1]`; protected test targets cannot tune it.

## 2026-10-11 — Require old, keyed, spatial information

Retrieval excludes entries inside the visible context window. Washout events
make a last-value state uninformative. Zero-mean innovations make a globally
broadcast value uninformative. Exact anchor labels are reserved for the oracle
ceiling and never enter the primary retrieval query.

## 2026-10-11 — Keep the claim narrow

A synthetic pass is only a causal mechanism check. Recurrent/SSM/operator
superiority, natural-domain efficacy, novelty, and publication readiness remain
unsupported until a separately frozen public-data protocol passes.

## 2026-10-11 — Preserve resource failure

The first guarded pipeline admitted the test suite, which passed 10/10, and the
static contract audit, then stopped before the experiment at 527,171,584 bytes
free-plus-speculative versus the 536,870,912-byte floor. Zero seeds started.
The guard was then strengthened to require stable pageouts as well as headroom.

## 2026-10-11 — Accept the bounded synthetic result, not a SOTA claim

The later guarded pipeline completed. Ten tests passed, all three frozen seeds
passed the bounded-development rule, and the independent verifier passed 96 of
96 checks. This supersedes the resource block for the synthetic campaign only.
It does not establish public-data performance or state of the art.

## 2026-10-11 — Target partial-observation and non-Markovian frontiers

The public campaign targets POBench-PDE/LANO as the primary partial-observation
frontier and HS-FNO as the secondary delay-PDE frontier. Generic PDE-foundation
model superiority is out of scope because it would mix tasks, scale, and
compute. Upstream source identities, public splits, baselines, metrics, and a
material improvement threshold are frozen before any protected test execution.

## 2026-10-11 — Require a material, broad win

The candidate must improve the reproduced primary composite by at least 10%,
win at least 30 of 36 POBench-PDE cells, avoid material regression in any hard
regime, and pass causal memory ablations. The HS-FNO track requires at least a
15% aggregate rollout improvement and wins on at least four of five benchmark
families. One favorable cell cannot support a SOTA statement.

## 2026-10-11 - Author only the evidence-supported manuscript

The full research-lifecycle request does not authorize manufacturing missing
public results. The manuscript therefore reports the verified three-seed
synthetic mechanism study, includes exact intervention and integrity results,
and labels the public ITFM-R campaign as prospective. A complete LaTeX source,
bibliography, literature audit, claim-to-evidence map, and release verifier were
added. The source is not called arXiv-ready because compilation, rendered visual
inspection, public confirmation, independent reproduction, and final
authorship/license checks are incomplete.

## 2026-10-11 - Preserve the failed execution and paper-build gate

Two samples measured 282,214,400 and 193,888,256 bytes free-plus-speculative,
both below the 536,870,912-byte floor. Cumulative pageouts rose from 662,980 to
664,029 and swap use rose from 5,699.44 to 5,752.12 MiB. No Torch import,
dependency install, test suite, model execution, data fetch, training,
evaluation, container build, LaTeX compile, PDF render, or release staging was
started in this session.

## 2026-10-11 - Preserve the resumption audit and unchanged stop

The repeated pipeline attachment changed no research instruction; it only
removed the earlier prefixed implementation sentence. Fresh GitHub checks
resolved both locked frontier commits, but the exact root trees still exposed
no license file. No POBench-PDE or HS-FNO dataset bytes were found in the
relevant research workspaces, and no accelerator or monetary budget was
authorized. Two new resource samples measured 216,891,392 and 301,170,688
bytes free-plus-speculative, both below the 536,870,912-byte floor, while
cumulative pageouts rose from 670,581 to 670,995. The public campaign, runtime
suite, manuscript compilation, rendering, and release therefore remained
closed; no prior result was overwritten.

