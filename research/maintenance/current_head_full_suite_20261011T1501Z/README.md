# Current-source full engineering verification — 11 October 2026

The strict JSON admission repair in existing draft PR44 had 39 focused test
methods, while its earlier 864-test integration receipt described an older
source. This separate verification closes the missing full repository collection
for commit `58c0b5b1249b44456052de4ddda7adbbb0a7acae` (tree
`cdd6c51d661a119c4a836429b2a06ce042ed1998`).

## Executed result

One unchanged ordinary `python -m pytest -q` collection with JUnit output passed:
**873 ordinary tests, 204 passing subtests, 7 skipped tests, zero failures or
errors**, with 33 inherited GradScaler deprecation warnings. Pytest took 11.08
seconds and the complete command took 11.951 seconds. These counts overlap prior
collections and must not be added to them.

Python 3.12.14, CPU PyTorch 2.14.1+cpu, pytest 9.1.1, NumPy 2.3.5, SciPy 1.17.0,
Matplotlib 3.10.8 and PyYAML 6.0.3 were used. OMP, OpenBLAS, MKL and NumExpr each
had one thread; MPLBACKEND was Agg. CUDA was unavailable. Dependencies were
installed in an isolated validation environment; repository requirements and
workflows were not changed.

The repository was clean at the exact source before and after the run. This
followup adds the raw log, JUnit, environment, prospective contract and receipt,
plus an exact predecessor-state snapshot and appended canonical session. Runtime,
test, model, protocol, result, and workflow bytes remain unchanged. No extra test
or retry was needed.

## Reading the evidence

- `prospective_contract.json` records the one-process, 240-second bound before
  execution. Earlier closed sessions and budgets remain unchanged.
- `test_receipt.json` binds the actual command, timestamps, source, environment,
  clean before/after state and log digest.
- `pytest.log` and `pytest.xml.gz` preserve the actual collection, skips, warnings
  and subtest entries. The gzip archive preserves the original XML bytes and their hash. They are local verification receipts.
- The canonical index appends this session without overwriting predecessor
  claims, adverse results or verification records.
- `independent_review.json` records a separate assistant's read-only integrity
  review. It is not a human maintainer acceptance or scientific review.

Seven inherited compatibility tests remain skipped because their required
benchmark or module APIs are unavailable. These are diffusion, Lorenz, wave,
fractional diffusion, Levy, persistent-memory and required-rollout coverage gaps;
the raw JUnit preserves the exact skip messages.

## Remaining work

Keep PR44 Draft and retain its PR38 prerequisite for maintainer review. This is
engineering acceptance, not a reproduction of the paper or evidence of memory
benefit. The frozen 40-cell matrix and delayed-recall successor remain held.
No protected campaign, retained demo, outcome analysis, scientific workflow,
paid compute, main merge or submission was executed.
