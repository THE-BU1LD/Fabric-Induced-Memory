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

The maintained repository execution paths have been repaired and verified in GitHub Actions. The canonical training path now routes through the maintained experiment implementation, checkpoint evaluation supports the current structured model outputs, and regression tests cover the repaired evaluator behavior. An earlier maintained compile/test checkpoint recorded **18 passed, 8 skipped, 0 failed**; this historical count is not a claim about the current collection.

The legacy component-ablation wrapper that depended on unsupported flags was not treated as evidence. It has been replaced by a current-code ablation stack whose switches correspond to mechanisms that actually exist in `FIMSystem`:

- `full`;
- `no_memory`;
- `no_retrieval`;
- `no_salience_gating`.

The historical `no_spectral_mixing` label is intentionally unsupported in the current stack and must not be silently mapped to an unrelated mechanism.

The earlier **24-cell shared-minibatch matrix** is retired as the current evidence lane. Its retained artifacts remain historical evidence: independent trajectories could share a memory bank, so that execution cannot establish trajectory-isolated memory behavior. The old runner must not be repeated or its results relabeled as the corrected protocol.

The current frozen protocol is [`FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1`](research/protocols/FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1.md): **2 benchmarks × 5 seeds × 4 current-code variants = 40 cells**. Its benchmarks are `delayed_recall` and `lorenz96`; paired seeds are `101, 211, 307, 401, 503`. The fixed budget is four epochs, 32 trajectories per cell, batch size one, a 16-step training rollout and a 20-step evaluation rollout. Memory is reset between independent episodes; within-episode writes and retrieval remain enabled. Final-epoch evaluation replaces best-checkpoint selection.

These are protocol settings, not completed results. The protected scientific execution hold remains in force. This document does not authorize a workflow dispatch, release that hold, or assert that any of the 40 cells has completed. A future authorized execution must retain a manifest binding every cell to its exact commit, protocol hash, switches, configuration, environment, outputs and failure state.

### Engineering integration and scientific execution status

**Verified on 8 October 2026:** [PR #25](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/25) merged the reviewed engineering integration into `main` at commit [`84160703cc733c6deb150cf7b6ee5bb3a84d699b`](https://github.com/THE-BU1LD/Fabric-Induced-Memory/commit/84160703cc733c6deb150cf7b6ee5bb3a84d699b). GitHub records PRs #20–#24 as merged through that integration. The earlier 7 October review of `main` `08d2c3df4262612bffe38488f97e263769ff7ef6` is historical; the engineering changes and dispatch-only condition are now integrated.

| Source PR | Reviewed head | Integrated behavior |
| --- | --- | --- |
| [#20](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/20) | `c6c1188ded48f4cd83ade3553ca10914be7217cd` | Graph accumulation and full-model integration repair |
| [#21](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/21) | `3ff0fe1de64795e6241bf0ee54c5c5ad06a3850f` | Opt-in native benchmark runtime contracts |
| [#22](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/22) | `605ca23a3961fcbf95fabfd211ab69e8a77c2ca3` | Staged evidence publication and recovery after write failures |
| [#23](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/23) | `28f8da75642204f7a9447aaef73532ab65d7a5c8` | Explicit raw/EMA evaluation weights and preserved training-resume state |
| [#24](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/24) | `758bf83430d30cfb3f29b204650dafe3d13da803` | Current 40-cell protocol and historical-result boundaries |

The retained [integration report](docs/integration/20261008/README.md), [verification receipt](docs/integration/20261008/verification.json) and [test receipt](docs/integration/20261008/pytest.xml) document the qualification performed before that merge: **176 passed, 7 skipped and 10 subtests passed**, plus compilation, the frozen-v2 source verifier, wheel build and installed-package implementation fixtures. The seven skips remain skips. These are recorded integration checks; this status reconciliation did not rerun them. The [hosted `test` check](https://github.com/THE-BU1LD/Fabric-Induced-Memory/actions/runs/37771892154/job/113293236434) also completed successfully on merged commit `84160703cc733c6deb150cf7b6ee5bb3a84d699b`. The dated integration report retains its original candidate-review wording as historical provenance; this section supplies the later merged status.

The integrated [component workflow](https://github.com/THE-BU1LD/Fabric-Induced-Memory/blob/84160703cc733c6deb150cf7b6ee5bb3a84d699b/.github/workflows/current-component-ablations.yml) requires `github.event_name == 'workflow_dispatch'` for the `trajectory-isolated-v1` scientific job. Pull requests retain ordinary repository testing and cannot start that scientific job. Future engineering changes must preserve this condition.

The native runtime successor remains opt-in, and `fim_experiments/benchmark.py` retains frozen Git blob `c91a6afe9d86830c6bcfaa2d9187547d9688050d`. The retained verifier result is `VERIFIED_PREOUTCOME_METHOD_FREEZE`, with `execution_authorized: false` and `outcome_access_allowed: false`. Engineering integration is complete; the protected scientific execution and outcome holds remain in force. Neither the merge nor its successful checks establish a memory benefit, completed 40-cell matrix, closure of [reproduction issue #1](https://github.com/THE-BU1LD/Fabric-Induced-Memory/issues/1), or publication readiness.

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
- That a successful smoke/CI path proves the frozen 40-cell trajectory-isolated matrix completed.
- That the current public repository is submission-ready solely because a PDF, workflow, and result artifacts exist.

## Next evidence gate

The immediate gate is a qualified execution decision. Before any protected scientific dispatch, the hold must be explicitly released for this matrix, the exact execution revision must contain the dispatch-only guard, and its commit, protocol hash, environment and durable output destination must be bound in advance. Until then the matrix remains held, even when engineering tests pass.

Required before stronger paper-facing claims:

1. After those execution gates are satisfied, complete the frozen 40-cell trajectory-isolated matrix with no unexplained missing or duplicate benchmark × seed × variant cells. Do not restart the retired shared-minibatch matrix.
2. Preserve raw metrics, resolved configs, seed, environment, logs, checkpoint identity, commit SHA, switch state, and failure state per cell.
3. Compute paired per-seed deltas against `full` and retain adverse ablations when a reduced variant wins.
4. Keep benchmark-specific conclusions separate when Lorenz96 and delayed recall disagree.
5. Treat the broader reproduction package required by issue #1, including the canonical paper Lorenz experiment and maintained benchmark/baseline suite, as a separate readiness and authorization decision; the bounded 40-cell protocol does not establish it.
6. Regenerate paper tables and figures only from validated current-run artifacts.
7. Update scientific outcome claims only after retained artifacts exist and completeness checks pass. Keep source, protocol and integration status current without promoting those updates into scientific results.

Negative, mixed, baseline-winning, and mechanism-falsifying outcomes are all valid scientific endpoints. Missing execution is not a null result, and historical paper-reference values are not fresh reproduction evidence.
