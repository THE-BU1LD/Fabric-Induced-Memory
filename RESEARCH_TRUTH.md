# Research Truth

This document records what the current public repository artifacts support and what they do not yet support.

## Evidence currently present

- A delayed-recall mini-suite comparing FIM with DeepONet, a selective SSM, a shape-aware MLP, and a Transformer.
- Persisted checkpoints, resolved configs, run logs, metrics, and generated figures for the stored mini-suite runs.
- A `paper_reference_results.json` file whose own metadata identifies its values as paper-reported compact results from `paper/Final FIM.pdf`.
- Core and epistemic test code covering dynamics, memory, benchmark generation, fractional dynamics, Levy sampling, rollout stability, and training execution.
- Maintained train/eval entrypoints repaired and protected by CI.
- A current-code component-ablation system with explicit `full`, `no_memory`, `no_retrieval`, and `no_salience_gating` variants, semantic tests, a multi-seed runner, provenance manifest, and frozen interpretation protocol.

## What the stored mini-suite supports

For the persisted delayed-recall mini-suite, the stored rollout MSE values are:

- DeepONet: 0.1579457372
- FIM: 0.1975817829
- Shape-aware MLP: 0.1965747178
- Transformer: 0.2490188628
- Selective SSM: 0.3046883643

On this artifact, FIM is competitive but is not the best model by rollout MSE. DeepONet is best among the stored runs, and the shape-aware MLP is marginally lower than FIM.

## Current execution-integrity status

The maintained repository execution paths have been repaired and verified in GitHub Actions. The canonical training path now routes through the maintained experiment implementation, checkpoint evaluation supports the current structured model outputs, and regression tests cover the repaired evaluator behavior. The repository CI has passed the maintained compile/test path with **18 passed, 8 skipped, 0 failed**.

The legacy component-ablation wrapper that depended on unsupported flags was not treated as evidence. It has been replaced by a current-code ablation stack whose switches correspond to mechanisms that actually exist in `FIMSystem`:

- `full`;
- `no_memory`;
- `no_retrieval`;
- `no_salience_gating`.

The historical `no_spectral_mixing` label is intentionally unsupported in the current stack and must not be silently mapped to an unrelated mechanism.

A frozen 24-cell execution matrix is defined as **2 benchmarks × 3 seeds × 4 current-code variants**. The online execution workflow is intended to retain a manifest tying each cell to its commit, switch state, configuration, metrics, and generated evidence. Until that matrix finishes and its completeness/provenance checks pass, it is an execution protocol, not new paper evidence.

## Historical paper-reference boundary

`paper_reference_results.json` is explicitly labeled as a compact transcription of values reported in the existing PDF. It is not, by itself, proof that those values were freshly reproduced from the current commit/configuration.

The stored Lorenz ablation values in that reference file are:

- full: 1.317169
- no_memory: 1.317169
- no_spectral_mixing: 0.869918
- no_retrieval: 1.317169

Those values do not support a blanket claim that the full model consistently outperforms its ablations. They also must not be reinterpreted as results from the new current-code ablation variants unless a retained current-commit run independently establishes the correspondence.

## Claims that should not currently be made from this public repository alone

- That FIM consistently outperforms all included baselines.
- That every paper-reported number has been freshly reproduced from the current commit.
- That every ablation favors the full model.
- That the historical `no_spectral_mixing` result is equivalent to any new current-code ablation.
- That a successful smoke/CI path proves the 24-cell scientific matrix completed.
- That the current public repository is submission-ready solely because a PDF, workflow, and result artifacts exist.

## Next evidence gate

The immediate gate is to complete and validate the frozen current-commit experiment package rather than add more mechanisms or tune after results are visible.

Required before stronger paper-facing claims:

1. Complete the current 24-cell component-ablation matrix with no unexplained missing or duplicate benchmark × seed × variant cells.
2. Preserve raw metrics, resolved configs, seed, environment, logs, checkpoint identity, commit SHA, switch state, and failure state per cell.
3. Compute paired per-seed deltas against `full` and retain adverse ablations when a reduced variant wins.
4. Keep benchmark-specific conclusions separate when Lorenz96 and delayed recall disagree.
5. Run the broader frozen current-commit reproduction package required by issue #1, including the canonical paper Lorenz experiment and maintained benchmark/baseline suite.
6. Regenerate paper tables and figures only from validated current-run artifacts.
7. Update this truth document only after those retained artifacts exist and completeness checks pass.

Negative, mixed, baseline-winning, and mechanism-falsifying outcomes are all valid scientific endpoints. Missing execution is not a null result, and historical paper-reference values are not fresh reproduction evidence.
