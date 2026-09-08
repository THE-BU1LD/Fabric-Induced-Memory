# FIM fresh multi-seed replication protocol — frozen 2026-09-08

## Status and chronology

This is a **prospectively frozen fresh-seed replication/triage protocol**, not a blind first test. It was written after the retained 2026-08-29 three-seed component matrix was already known to be mixed/adverse on Lorenz-96 and near-null on delayed recall. Those earlier outcomes are development evidence and must remain visible.

This protocol must not be edited after its fresh outcomes are observed. Any later model change, threshold change, seed replacement, metric change, or benchmark change is a new protocol.

## Scientific questions

1. **Retrieval-feedback replication:** does feeding retrieved memory back into the latent state materially improve delayed-recall rollout error versus an otherwise identical write-only memory system?
2. **Salience-gating replication:** does salience-gated storage materially improve performance versus storing every produced trace?
3. **Baseline competitiveness:** under one matched bounded training/evaluation budget, how does current FIM compare with the repository's maintained Transformer, selective SSM, DeepONet, and MLP baselines on delayed recall and Lorenz-96?
4. **Benchmark specificity:** are any observed effects specific to a memory-oriented benchmark rather than general across chaotic dynamics?

## Frozen commit boundary

The protocol is authored against canonical `THE-BU1LD/Fabric-Induced-Memory` main SHA:

`68dcb42242ee6e11f2243dccd6f3e0f3460e1545`

The workflow must record the exact executed checkout SHA. A different SHA is acceptable only when the diff is the protocol/runner/workflow needed to execute this frozen design and does not change model mathematics, benchmark generation, training semantics, or outcome definitions.

## Frozen seeds

Fresh paired seeds:

`101, 202, 303, 404, 505`

No failed or unfavorable seed may be replaced. Infrastructure failures remain explicit failed cells.

## Frozen benchmarks

- `delayed_recall`
- `lorenz96`

The same benchmark implementation and seed are shared across paired model/ablation cells.

## Frozen mechanism matrix

Variants:

- `full`
- `no_memory`
- `no_retrieval`
- `no_salience_gating`

The current implementation semantics are defined by `fim_experiments/ablation_systems.py` and must pass `tests/test_current_component_ablations.py` before compute.

`no_memory` and `no_retrieval` are deliberately retained as an integrity pair. Because write-only memory does not feed back into prediction, performance identity between these two is not interpreted as a defect by itself. Their semantic distinction is verified structurally: `no_retrieval` stores traces while `no_memory` does not.

Total mechanism cells: `2 benchmarks × 5 seeds × 4 variants = 40`.

## Frozen baseline matrix

Models:

- `fim_plus`
- `transformer`
- `ssm`
- `deeponet`
- `mlp`

Total baseline cells: `2 benchmarks × 5 seeds × 5 models = 50`.

The purpose is not to claim state of the art. It is to prevent a component story from being interpreted without comparison to maintained simple/standard alternatives.

## Frozen bounded compute budget

All mechanism and baseline cells use:

- epochs: `6`
- batch size: `64`
- dataset size: `1024`
- training rollout steps: `4`
- evaluation rollout steps: `30`
- workers: `0`
- device: `cpu` in CI
- deterministic seed handling: repository default

The bounded budget is chosen as a high-information replication/triage run, not a final convergence study. Under-training remains a limitation and must be disclosed.

## Outcomes

Primary metric: **rollout MSE** (lower is better).

Secondary metric: rollout MAE.

### Primary replication contrast

On `delayed_recall`, paired by seed:

`delta_retrieval = MSE(full) - MSE(no_retrieval)`

A negative delta favors retrieval feedback.

A **practically meaningful retrieval effect** requires both:

1. mean relative MSE improvement of at least **2%** versus `no_retrieval`; and
2. `full` lower MSE on at least **4 of 5** paired seeds.

This is an advancement heuristic for this bounded replication, not a general proof or significance threshold.

### Secondary contrasts

- `full` vs `no_salience_gating` on each benchmark;
- `full` vs `no_memory` on each benchmark;
- `fim_plus` vs each maintained baseline on each benchmark.

With only five seeds, no tiny-n p-value is to be used to manufacture confidence. Report all five paired deltas, means, sample SDs, and direction counts. Confidence/significance claims require a separately powered protocol.

## Failure and missing-cell policy

- A failed cell is evidence and remains failed.
- No failed cell is dropped from the manifest.
- No replacement seed is permitted.
- The aggregate is marked incomplete if any expected cell is absent.
- NaN/Inf metrics fail the matrix.
- Historical `paper_reference_results.json` is not fresh evidence.

## Claim rules

Allowed if supported by this run:

- bounded fresh-seed replication statements tied to these two benchmarks and this budget;
- benchmark-specific positive, null, mixed, or adverse findings;
- direct reporting of baseline wins;
- confirmation that ablation switches execute distinct structural paths.

Not allowed from this run alone:

- broad component superiority;
- general long-horizon memory superiority;
- SOTA claims;
- convergence claims;
- historical-paper equivalence;
- external/general real-world validation;
- treating CI success as scientific success.

## Canonical execution

```bash
bash scripts/run_fresh_multiseed_20260908.sh
python scripts/summarize_fresh_multiseed_20260908.py
```

The summary must be generated only from retained run artifacts and the mechanism manifest.

## Freeze rule

Once the first fresh outcome-bearing cell begins, this document and the frozen runner arguments are immutable. Any follow-up after seeing results must use a new versioned protocol.
