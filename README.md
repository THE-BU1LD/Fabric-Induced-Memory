# Fabric-Induced Memory (FIM)

FIM is a research framework for neural sequence models with a structured latent memory fabric rather than only a larger context window or a single compressed hidden state.

The implementation explores local information propagation, persistent traces, decay, salience, retrieval, latent geometry, stochastic dynamics, and long-horizon forecasting.

## Evidence boundary

Read [`RESEARCH_TRUTH.md`](RESEARCH_TRUTH.md) before quoting results.

The public repository contains real source code, tests, experiment runners, stored checkpoints/logs/configs, a delayed-recall mini-suite, historical paper-reference values, and an existing PDF manuscript. These artifacts are not all equivalent forms of evidence.

In particular:

- `results/mini_processed/mini_suite_summary.md` is a persisted compact delayed-recall comparison. In that stored artifact, FIM is competitive but does **not** have the lowest rollout MSE; DeepONet does.
- `paper_reference_results.json` explicitly identifies its values as compact results transcribed from the existing paper. Treat those as historical paper-reference values unless they are freshly reproduced from the current commit/configuration.
- The repository must not be described as showing that the full FIM model consistently beats every baseline or ablation. The committed artifacts do not support that blanket statement.

Negative results and baseline wins are part of the scientific record and should be preserved.

## Repository map

- `fim/` — core dynamics, geometry, memory, operators, stochastic modules, training, and utilities.
- `fim_experiments/` — benchmark systems, training/evaluation runners, and experiment orchestration.
- `tests/` and `fim/epistemic_tests_full/tests/` — implementation and scientific sanity checks.
- `scripts/` — train, evaluation, ablation, paper-Lorenz, and suite runners.
- `results/` — stored run artifacts and processed summaries.
- `docs/` — architecture, mathematics, and evidence-bounded experiment documentation.
- `paper/` — existing manuscript artifact.
- `RESEARCH_TRUTH.md` — current evidence and claim boundary.

## Reproducibility rule

For new paper evidence, preserve the Git commit, dirty-tree state, resolved configuration, seed, benchmark/data identity, raw metrics/predictions, logs/checkpoints, and failure state. Paper tables and figures should be regenerated from those retained artifacts rather than manually copied from historical summaries.

## Current status

**IMPLEMENTED / EVIDENCE_PARTIAL.** The repository is a substantive research implementation with stored experimental artifacts, but a fresh frozen multi-seed benchmark-and-ablation reproduction is still required before stronger submission claims are justified.
