# Interventional Tensor Fabric Memory (ITFM) v1

This directory is the separately versioned successor to the completed negative
Fabric-Induced Memory component study. It does not replace, reinterpret, or
modify the predecessor evidence under `../../FIM/`.

Read `RESEARCH_STATE.json` and `PROJECT_CONTRACT.md` before beginning work.
Update the state and append `DECISION_LOG.md` before ending a research session.

## Exact research question

Under partial observation with oracle-defined aliasing, does a query-aligned
tensor-valued external memory provide forecast-relevant information that is not
captured by matched finite-history, recurrent-state, stable state-space, and
analytical transport predictors, and is that information mechanism-localized
by value-swap, erasure, nuisance-value, and oracle interventions?

## Candidate contribution

The contribution is not the existence of an external memory or a tensor field.
It is an identification protocol and reference implementation in which:

1. memory keys and residual values retain spatial structure;
2. every entry has immutable trajectory ownership and event time;
3. eligibility rules prevent cross-unit and future retrieval;
4. memory-only interventions leave learned parameters and current inputs fixed;
5. success requires both predictive benefit and the predeclared directional
   intervention response.

The core fabric and an optional predictor support one-, two-, and
three-dimensional spatial domains through `spatial_shape=(S1, ..., Sr)`. The
bounded periodic transport estimator, co-moving coordinate map, and group action
are rank-generic too. The
frozen development experiment remains two-dimensional, so multidimensional
support is an engineering capability—not evidence of multidimensional efficacy.

The deterministic synthetic loader and official PDEBench metadata are described
in `datasets/README.md`. Public arrays are never downloaded implicitly.

## Current checkpoint state

- Checkpoint 1, idea ready: passed for a bounded development implementation.
- Checkpoint 2, model specification ready: passed.
- Checkpoint 3, development methodology ready: passed; confirmatory protocol is
  deliberately not frozen.
- Checkpoint 4: implementation passes all 37 numerical and semantic tests.
- Checkpoint 5: experiment system and deterministic data materialization are
  present; the smoke matrix is paused before cell 1/9 by resource admission.
- Checkpoints 6-10: not reached.

No scientific result is claimed by the presence of this code.

## Local verification

The fail-closed verifier checks resource admission, runs all numerical tests,
executes the bounded smoke matrix, validates artifact hashes, and regenerates
the smoke analysis:

```bash
PYTHONPATH=src python3.11 scripts/run_pipeline.py
```

Only after that receipt is complete may the declared 27-cell development matrix
be requested:

```bash
PYTHONPATH=src python3.11 scripts/run_pipeline.py --full-development
```

Both commands stop before tests or training when the host resource-admission
guard fails. They default to one numerical thread and niceness 10, and the
experiment rechecks admission before every model/seed cell. Neither command
creates confirmatory evidence.

