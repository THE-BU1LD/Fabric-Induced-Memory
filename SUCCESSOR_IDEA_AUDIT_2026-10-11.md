# FIM successor idea audit — 2026-10-11

Status: **CHECKPOINT 1 NOT PASSED / RELEASE LINEAGE BLOCKED / NO IMPLEMENTATION AUTHORIZED**

This audit preserves the completed negative component study. It does not
reinterpret its null/adverse ablations or authorize rerunning them for a more
favorable result.

## Immutable predecessor result

The frozen maintained component study completed 24/24 cells over two
benchmarks, three seeds, and four variants. Across six paired comparisons, mean
direction favors the full model in 3/6 and a removal in 3/6; the smallest exact
sign-test p-value is 0.5. The current evidence therefore does not establish
general benefit from memory, retrieval, or salience gating.

Canonical evidence within the relocated evidence tree:

- `FIM/COMPLETION_MANIFEST.md`
- `FIM/results/current_component_ablations/manifest.json`
- `FIM/results/current_component_ablations/analysis.json`
- `FIM/docs/CURRENT_COMPONENT_ABLATION_ADJUDICATION_2026-08-29.md`

Read-only verification on 2026-10-11 reports 24 unique cells under one frozen
protocol/source identity, and all five entries in `FIM/output/fim-evidence.sha256`
verified. This checks retained-artifact integrity; it does not make the current
source tree identical to the frozen executable source or constitute an
independent reproduction.

## Source and release identity blocker

The outer repository is not release-clean. At audit time, Git reports 65
tracked deletions under `fim/` and a large untracked replacement/evidence tree
under `FIM/`. The untracked tree calls itself an archival line and points to a
different maintained sibling path. Meanwhile, the tracked root truth files are
older than the completed 24-cell record inside `FIM/`.

Consequences:

- the outer commit `32d8aa0b109f08a8ecd23c7bd97eaf29b5fe8989` does not by itself
  identify the complete relocated source/evidence package;
- the frozen result manifest records executable-source identity
  `e8071b2243247361e3ea0da1eaa412e69f204593f80102fe5f837dead5300f46`,
  whereas recomputing the same declared 107-file hashing procedure over the
  current `FIM/` tree gives
  `06712dd535f0a6a7e72622ed18239bf34482e0bf72b3d90cd8e3c9d5649ca5c6`;
- no source archive matching the frozen identity was found by the bounded
  archive inventory, so the retained outputs are hash-bound evidence but the
  exact executable source is not currently identified for rerun;
- no clean release, independent verification, or successor implementation may
  be claimed from this worktree state;
- the relocation must be reconciled non-destructively, with hashes and lineage
  retained, before Checkpoints 9 or 10 can pass.

## Mechanism identifiability audit

The maintained experimental `FIMSystem` in
`FIM/fim_experiments/systems.py` stores a compressed latent vector as both key
and value. Retrieval projects that vector to channel width and broadcasts the
same update over all spatial positions. Memory ownership is assigned by batch
position, not a persistent trajectory identifier supplied by the dataset.

This creates four severe interpretation limits:

1. the memory value does not preserve observation-space content that can be
   independently checked for recovery;
2. spatially uniform broadcast cannot establish location-specific recall;
3. batch-slot ownership is safe only if slot identity remains a valid trajectory
   identity across every step and batching operation;
4. final error cannot distinguish useful retrieval from capacity, recurrent
   history, or optimization effects.

The exact equality of retained `no_memory` and `no_retrieval` results is itself
an identifiability warning: storing traces without reading them has no observed
effect in that protocol, and the experiment does not localize a positive memory
mechanism.

## Exact successor question

Under observation aliasing where the same current observation corresponds to
different latent states and futures, can an observation-valued external memory
recover the missing state in a way that is causally localized by memory-swap,
memory-erasure, and oracle-value interventions, while outperforming matched
finite-history and state-space baselines?

Candidate statement, not yet a claim:

> Existing memory-augmented forecasters can appear useful without identifying
> information stored uniquely in external memory. We propose an
> intervention-identified observation-valued memory because, on oracle-defined
> alias pairs, swapping only the relevant memory value should swap the forecast
> toward the corresponding latent future while leaving non-aliased controls
> unchanged.

## Prior-art boundary checked on 2026-10-11

- Ruiz et al., *On the Benefits of Memory for Modeling Time-Dependent PDEs*
  (MemNO) already gives a state-space memory neural operator, theory, and
  partial/distorted-observation experiments. <https://arxiv.org/abs/2409.02313>
- Singh et al., *Learning Semilinear Neural Operators* (NODA) already frames
  sparse/noisy observation forecasting as prediction plus observer correction.
  <https://arxiv.org/abs/2402.15656>
- Koren and Lanthaler, *Merging Memory and Space: A Spatiotemporal State Space
  Neural Operator* already gives a factorized spatial/temporal SSM operator,
  a universality result, and partial-observability experiments.
  <https://arxiv.org/abs/2507.23428>
- Lin et al., *Data-driven learning for the Mori-Zwanzig formalism* already
  learns separated Markov and memory operators from reduced observables.
  <https://arxiv.org/abs/2101.05873>
- Neural Operator Processes address probabilistic operator prediction from
  sparse partial observations. <https://arxiv.org/abs/2606.22946>
- Retrieval-augmented forecasting methods including RAFT, RATL, and FreshCast
  already cover historical continuation retrieval, residual retrieval, routing,
  refresh, and validation-calibrated combination.

Therefore external memory, partial-observation memory, residual retrieval, and
prediction/correction are not novel. A FIM successor would need to contribute
an intervention-based memory-identifiability test or a formal observability
result, not another memory architecture with favorable benchmark scores.

## Adversarial flaw register

| Flaw | Cause | Severity | Possible solution | Required evidence |
|---|---|---:|---|---|
| Novelty collision | MemNO, NODA, Mori-Zwanzig learning, SSM operators, and retrieval forecasting cover the broad idea | Critical | Focus on intervention-identified external-memory information | Full-paper/theorem and implementation comparison |
| Synthetic construction may be trivial | A delayed-recall task can directly encode the answer in memory | Critical | Use alias pairs derived from a declared partially observed dynamical system plus nuisance-matched controls | Oracle observability analysis and simple lookup/recurrent baselines |
| Memory is not identifiable | More history/capacity can explain improvements | Critical | Same backbone and information budget; finite-window, SSM, parametric-state, fixed-kNN, and oracle-memory controls | Conditional effects under memory-only interventions |
| Current value semantics are opaque | Stored values are learned latent compressions | High | Observation-valued or explicitly residual-valued memory with inspectable ground truth | Value reconstruction and swap-localization metrics |
| Spatial mechanism is weak | A trace vector is broadcast uniformly over space | High | Query-aligned spatial values or explicitly nonspatial tasks | Spatial intervention maps and locality tests |
| Ownership can be confounded | Batch indices stand in for trajectory identity | Critical | Immutable dataset trajectory IDs passed through every memory operation | Tests under shuffling, variable batch size, and reordered trajectories |
| Leakage risk | Writes/readbacks can cross units or use unavailable future information | Critical | Train/development/test unit split before windowing; read-only protected bank; event-time audit | Zero cross-unit/future retrieval in saved logs |
| Causality is overclaimed | Ablation changes architecture and optimization | Critical | Key swap, value swap, pair swap, erasure, oracle insertion, and null-memory interventions at inference | Paired counterfactual forecasts with unchanged parameters |
| Baseline fairness | Weak feed-forward controls make memory look necessary | Critical | Selective SSM, transformer, finite-history operator, matched recurrent state, kNN, and oracle observer | Parameter/compute/information matched table |
| Scaling cost | Exact memory search is linear in bank size | Medium | Predeclare exact-search science scale; benchmark ANN only after equivalence | Retrieval latency/memory curves and neighbor-recall checks |
| Optimization instability | Memory updates and nonstationary retrieval alter gradients | High | Freeze bank construction for primary study or separate write/read phases | Seed stability, gradient norms, and bank-drift diagnostics |
| Generalization | Alias construction may not transfer beyond one simulator | High | One analytic controlled system and one public PDE/assimilation dataset | Predeclared cross-system replication |
| Simpler explanation | A sufficiently long context can resolve aliasing | Critical | Vary context length and include an oracle sufficient-history control | Error versus history length and conditional mutual-information proxy |

## Candidate implementations screened

1. **Repair the current latent memory and rerun.** Rejected as successor
   science. It may improve engineering but would not create novelty or repair
   the frozen negative result.
2. **Observation-valued spatial retrieval.** Mechanistically clearer, but alone
   is an architectural variant without a defensible contribution.
3. **Generic non-Markovian memory operator.** Rejected due direct prior-art
   collision with MemNO, SSM operators, and Mori-Zwanzig learning.
4. **Intervention-identified memory under oracle-defined aliasing.** Retained as
   the only candidate direction. It still requires an observability/novelty
   proof and exact public-data design before implementation.

## Required falsifier

The central candidate is falsified if any of the following occurs under a
frozen, adequately powered protocol:

- matched finite-history or state-space controls equal or outperform it within
  the practical-effect margin;
- swapping or erasing the allegedly relevant memory does not cause the
  predeclared directional change in forecast;
- random or nuisance-matched values perform equivalently;
- the benefit disappears after matching parameters, updates, information, and
  tuning budget;
- the effect exists only on the hand-built recall task and not the predeclared
  second system.

## Budgets and protected boundary

- Development budget: **0 model-training runs authorized** until Checkpoint 1
  and source-lineage reconciliation.
- Compute budget: **0 heavy jobs authorized** while the portfolio resource
  admission gate is below 512 MiB free-plus-speculative or pageouts rise.
- Protected evaluation: **not created**. The completed component matrix remains
  immutable predecessor evidence and cannot be reused for successor tuning.
- Release budget: read-only lineage and hash reconciliation only until the
  desired tracked/untracked source identity is explicitly resolved.

## Checkpoint decisions

- Checkpoint 1, idea ready: **FAIL / BLOCKED** — promising falsifier, but novelty
  and observability boundaries are incomplete.
- Checkpoints 2–7 for a successor: **NOT EVALUATED** — no model, methodology,
  code, dry run, development, or confirmation work is authorized.
- Checkpoints 8–10 for the predecessor: evidence is locked, but exact-source
  reproduction, independent verification, and release remain **BLOCKED** by
  the dirty relocation and the frozen-versus-current source-hash mismatch.

The next valid work is non-compute: reconcile lineage, audit full prior papers
and code, formalize the aliasing construction and information sets, and either
freeze a distinct successor contract or close the candidate as no-go.
