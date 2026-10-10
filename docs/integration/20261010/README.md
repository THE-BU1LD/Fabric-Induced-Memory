# FIM engineering integration — 10 October 2026

Ten existing draft proposals are combined in one review candidate. The complete
ordinary repository suite passes **685 tests, with 7 skips and 164 passing
subtests**. This closes the missing combined software check. Research remains
`IMPLEMENTED / EVIDENCE_PARTIAL`; no scientific checkpoint or execution hold
changes.

## What is combined

| Original proposal | Integrated behavior | Inspected head |
| --- | --- | --- |
| [#26](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/26) | Correct the dated truth record to reflect the already merged PR25 and its scientific dispatch guard | `c46d8f6078c05184c60e409acc9aa0acb7d51220` |
| [#27](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/27) | Validate scientific evidence admission and preserve checkpoint, evaluation and trajectory output bytes | `9503099f9952ef3af6bb92f7e75d006e37ff414f` |
| [#28](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/28) | Repair public training, structured losses, reversible EMA and validated checkpoint recovery | `7b64ef98770e0cf3fdffbab7c8ef01f071e219fb` |
| [#30](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/30) | Bound fractional histories and retain detached snapshots | `79a3941d2ccf8f321e3589cf69edce0ba37022dd` |
| [#32](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/32) | Reject ambiguous shapes and nonfinite general metric inputs | `3180112dd71534a6341edd1f9f8f12cceee85ad9` |
| [#33](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/33) | Add opt-in noise-scaled Milstein quadratic variation and zero-noise handling | `db791fb3f9d6a305f4b082f88b595e4767680a67` |
| [#34](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/34) | Admit standalone trace-bank batches transactionally and own their tensor/metadata storage | `b417f3c4b0487b09f948eea70c5d8b94a21cf3d5` |
| [#35](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/35) | Retain and verify paired development episode banks | `0ad7dc9766771cb3d9cb18bab649cfa2da3e4dc5` |
| [#36](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/36) | Add opt-in scale-stable field norms, clipping and growth diagnostics | `df0a0fce5ef5b28982a55428ebf0573f74c0fd45` |
| [#37](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/37) | Run bounded paired development episodes with verified resets and retained evidence | `67e2ad75f7b2bd39ed9d3f243c9c03234f7e334e` |

These are adopted implementations, not ten newly authored fixes. All selected
runtime, test and historical-artifact blobs are byte-identical to the inspected
proposals. Parent/child work is counted once. Alternative fractional-memory
[PR29](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/29) remains separate
because it overlaps PR30; neither its branch nor its history was overwritten.

## Integration decisions and exact preservation

The common main is `84160703cc733c6deb150cf7b6ee5bb3a84d699b`.
[sources.json](sources.json) records all ten selected heads, their trees, the
**98 adopted file identities**, **215 unchanged main files**, and **16 exact
predecessor metadata snapshots**. All implementation overlaps were identical
blobs or already shared ancestry. No runtime conflict required a rewrite.

Only three entrypoint/state paths and the truth file needed reconciliation:

1. `AGENTS.md` combines the existing session rules and points to one canonical
   `research/RESEARCH_STATE.json`.
2. The former root `RESEARCH_STATE.json` becomes a navigation pointer. Its
   complete original PR28 content is retained as an exact snapshot.
3. The canonical research index links every inspected predecessor state,
   distinguishes inherited receipts from this combined check, and records the
   active integration session separately from the closed earlier budgets.
4. `RESEARCH_TRUTH.md` contains PR26's exact integrated-status correction followed
   by PR27's exact evidence-admission append. Both complete source versions are
   retained. All unfavorable scientific findings remain.

The [protected source inventory](protected_source_identities.json) additionally
identifies the unchanged model/system definitions, benchmark and recall metric,
both frozen protocols, workflow definitions, and paper-reference values. Existing
retained results and earlier development artifacts retain their source identities.
The full source inventory can be verified without Git or PyTorch:

```bash
python docs/integration/20261010/verify_sources.py
```

This is an identity check against the committed inventory, not an external
signature or evidence that a study was executed.

## Verification actually executed

- Source-union verification passed: 98 adopted files, 215 unchanged main files,
  16 predecessor snapshots.
- Compilation passed for `fim`, `fim_experiments`, `scripts`, `tests`, and the
  integration verifier. Scoped staged whitespace checks passed for maintained
  code/tests and the authored integration metadata. The complete staged check
  reports original whitespace in five retained history files; those bytes remain
  unchanged. [whitespace_receipt.json](whitespace_receipt.json) retains that
  failed check and its exact scope.
- One complete ordinary test process passed: **685 passed, 7 skipped, 164 subtests
  passed**, with 16 existing public-trainer GradScaler deprecation warnings.
  Pytest took **27.98 seconds**; the complete command took **31.121531 seconds**.
- Runtime: Python 3.12.14, CPU PyTorch 2.14.1, pytest 9.1.1, NumPy 2.3.5,
  SciPy 1.17.0. Accelerators were disabled and numerical libraries used one thread.
- No test failed and no repair rerun was needed. The initial attempt to install
  pytest from the CPU PyTorch index failed before test execution; pytest was then
  supplied from the already available validation environment.

The [raw log](pytest.log), [JUnit receipt](pytest.xml), and
[complete command receipt](test_receipt.json) retain the evidence. Original
component-suite counts overlap the combined suite; they are not added together.
Existing ordinary hosted PR checks, if run, are recorded separately and do not
become additional distinct test coverage.

Independent parent-agent source review resolved all ten source trees against Git
objects and all 98 adopted blobs against an actual inspected source head. The
review also checked the canonical pointer/index, preserved closed budgets,
unfavorable claims, exact truth-file union, unchanged workflows, and whitespace.
It found no integration blocker. The review did not rerun the test collection.

To reproduce in an environment with the project requirements and pytest:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLBACKEND=Agg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
python -m pytest -q
```

## Remaining gates

The candidate requires maintainer review before integration into main. The
source-union check establishes preservation; it does not replace review of each
source proposal's known limitations. The standalone memory and opt-in numerical
helpers remain distinct from the maintained experiment-model implementations.
No caller is silently switched to the opt-in helpers.

The protected 40-cell trajectory matrix and delayed-recall successor remain held.
The scientific workflow still requires an explicit manual dispatch. No study
campaign, protected outcome, new retained demonstration, paid job, main merge,
submission or scientific result update was performed. Artificial test fixtures
may execute tiny model steps; their passing counts are software verification.

The stored full FIM model still does not beat every baseline or historical
ablation. A qualified prospective scientific execution and retained evidence are
still needed to resolve the research question.
