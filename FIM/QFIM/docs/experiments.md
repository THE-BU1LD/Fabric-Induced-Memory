# Experiments

## Experimental setup

### Benchmarks

The repository contains experiment paths for multiple dynamical systems, including:

- Lorenz-96 (chaotic ODE)
- diffusion equation (PDE)
- wave equation (PDE)
- fractional diffusion systems
- stochastic differential equations

The existence of an experiment path is not evidence that a current, complete, comparable run has been produced for every benchmark.

## Data generation

Each benchmark generates trajectories

\[
\{x_0, x_1, \dots, x_T\}.
\]

Training pairs are constructed as

\[
(x_t, x_{t+1}),
\]

and evaluation uses autoregressive rollout when supported by the maintained runner.

## Training configuration

The documented default training configuration includes:

- optimizer: AdamW
- learning rate: \(3 \times 10^{-4}\)
- batch size: 32
- cosine decay with warmup
- gradient clipping: 1.0
- mixed precision where enabled by the maintained execution path

Exact paper-facing settings must be read from the retained config for the corresponding run rather than inferred from these defaults.

## Objective

The implementation uses a composite objective of the form

\[
\mathcal{L} =
\lambda_p \mathcal{L}_{pred} +
\lambda_m \mathcal{L}_{mem} +
\lambda_s \mathcal{L}_{stab} +
\lambda_r \mathcal{L}_{retr} +
\lambda_c \mathcal{L}_{cons}.
\]

A component appearing in this expression does not by itself establish that the component improves predictive performance. Component value must be established through a valid intervention or ablation using the maintained evaluation path.

## Metrics

Available evaluation dimensions include:

- mean squared error (MSE)
- long-horizon error growth
- distribution alignment
- spectral consistency
- drift over time

Paper-facing claims should name the primary metric and retain the complete evaluated metric set so a favorable secondary metric cannot replace an adverse primary result.

## Evaluation protocol

Models are evaluated through autoregressive rollout where the benchmark runner supports it:

\[
\hat{x}_{t+1} = f(\hat{x}_t).
\]

The memory-aware evaluation path was repaired on 2026-08-29 because earlier evaluation code could disable FIM memory during evaluation. Results produced by an evaluation path that accidentally disables the mechanism under study cannot support a claim about the benefit of that mechanism.

The relevant repository history includes the fixes:

- `2743b0ba1284c4711f5fda0612a66ae38bfa49a8` — evaluate FIM memory modes instead of disabling memory;
- `554cb3971355b2dcaff6fa018699332b3f5381f1` — make rollout evaluation memory-aware and shape-safe;
- `c08058f923ae59d64aa1f5ef10b24d5cee370aba` — regression test against accidentally disabling FIM memory during evaluation;
- `1a24b43b4510ac8e1ae6778aa98f845a0eb2f7dd` — request a rerun of evidence with the repaired evaluator;
- `91d4d6917425bb810248579f9f9391f6e844b562` — extend the frozen paired memory diagnostic configuration to two seeds.

These commits are engineering/protocol evidence. They do **not** by themselves establish a positive scientific result.

## Component interventions

The research question for a component intervention is whether enabling the FIM memory mechanism changes held-out rollout behavior relative to a properly matched disabled/control condition when all other relevant settings are held fixed.

For every maintained intervention report, retain at minimum:

```text
git_commit:
benchmark:
dataset_or_generator_version:
train_seed:
eval_seed:
model_config:
memory_condition:
training_command:
evaluation_command:
primary_metric:
secondary_metrics:
raw_artifact_path:
started_at:
completed_at:
status:
```

A failed or interrupted run remains a failed/interrupted run. It must not be silently removed from an aggregate.

## Current evidence status

The previous version of this document stated that the full model **consistently outperforms ablations** in long-horizon stability, accumulated error, spectral fidelity, and noise robustness. That statement is too broad for the verified evidence and remains withdrawn.

A fresh external FIM current-component matrix has now executed across two benchmarks, three seeds, and four variants, with complete 24-cell manifest coverage and retained artifacts. The retained artifact was independently extracted and adjudicated seed-by-seed on 2026-08-31.

The current evidence is **mixed and does not establish broad FIM component superiority**:

- **Lorenz-96:** aggregate rollout error is mixed/adverse to the full condition. Relative to `no_memory` / `no_retrieval`, the full condition has mean rollout MSE higher by `0.021585` (about 4.22%) and mean rollout MAE higher by `0.010639` (about 3.04%). Relative to `no_salience_gating`, mean MSE is higher by `0.012399` (about 2.38%), while mean MAE is nearly tied but slightly adverse (`+0.000517`).
- **Delayed recall:** the full condition is numerically better than `no_memory` / `no_retrieval` on both stored metrics in all three seeds, but the mean differences are near-null: `-0.000154` MSE (about 0.0184%) and `-0.000049` MAE (about 0.0071%). The salience-gating comparison is even closer to zero.
- `no_memory` and `no_retrieval` are exactly identical on the retained MSE/MAE values across all six benchmark/seed combinations, so this matrix does not separately identify those conditions on the reported metrics.
- only three paired seeds exist per benchmark; no significance or general component-necessity claim is warranted.

The scientific verdict for broad FIM component superiority is therefore **mixed / not established**. Benchmark-specific adverse and near-null evidence must remain visible.

### Fresh retained FIM evidence receipt — 2026-08-29

A fresh current-code component-ablation execution exists in `THE-BU1LD/Fabric-Induced-Memory` and is recorded here so the evidence cannot be lost between repositories.

- GitHub Actions run: `33257399564` (`Current FIM component ablations`)
- executed checkout SHA: `d4fdca0189f071ba4ce38f874301ae8be8e586ba`
- source PR head recorded for the trigger: `f8455ca2cb15700c09b17745d4687246364a6639`
- benchmarks: `lorenz96`, `delayed_recall`
- seeds: `11`, `23`, `37`
- variants: `full`, `no_memory`, `no_retrieval`, `no_salience_gating`
- expected cells: 24
- validated cells: 24
- repository tests before execution: 23 passed, 8 skipped
- retained artifact name: `fim-current-component-ablations-d4fdca0189f071ba4ce38f874301ae8be8e586ba`
- artifact ID: `9716689517`
- artifact ZIP SHA-256: `81bb5c79f751cfc9fea9b4cf6e90879e1b7fbe81b48629657bdc30b8d492fd13`
- artifact retention configured by the run: 30 days
- detailed seed-paired adjudication: `THE-BU1LD/Fabric-Induced-Memory` PR #7, `docs/CURRENT_COMPONENT_ABLATION_ADJUDICATION_2026-08-29.md`

This receipt establishes a complete bounded rerun and its metric-level adjudication. It **does not** establish that the full model wins, that any component is necessary, historical-paper equivalence, or generalization beyond the executed benchmarks.

## Evidence gate for a positive component claim

A claim that a FIM component improves performance should require all of the following:

1. the evaluated condition actually enables the intended memory mechanism;
2. the comparator differs only in the declared intervention or is otherwise explicitly matched;
3. the benchmark, data split/generator, seeds, and training budget are frozen before result interpretation;
4. raw per-seed results are retained;
5. missing/failed seeds are visible;
6. the primary metric is predeclared;
7. uncertainty or paired effects are reported when multiple seeds are available;
8. the result is reproducible from the recorded commit and command;
9. simple or stronger baselines are not omitted because they win;
10. the paper text matches the actual scope of the evidence.

## Visualizations

The repository contains or can generate visualizations such as error curves, temporal/dimensional heatmaps, latent projections, spectral analyses, and trajectory comparisons. A visualization is paper evidence only when it is traceable to the same retained run set used for the corresponding numerical table or claim.

## Reproducibility rule

Fixed seeds and deterministic settings should be used where applicable, but deterministic configuration alone is not sufficient reproducibility. Every paper-facing result should also record the exact commit, command, configuration, data provenance, and output artifact.
