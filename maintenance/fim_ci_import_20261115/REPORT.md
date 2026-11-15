# FIM import-provenance and successor lint repair — 2026-11-15

The existing owned draft PR #51 was reviewed at commit
`8538e0c67d849ef1d247299df8694cef2f4eff5b`. Its hosted ITFM-R source-check
job `124131850922` stopped at nine inherited lint findings. The independent
review also identified cached-module collisions in the trajectory runner's
remaining experiment dependencies.

## Changes and evidence

- Package-qualify the trajectory runner's train/ablation dependencies and the
  maintained experiment module's benchmark, results, systems and train imports.
  The systems/ablation import chain now uses the same canonical package modules.
  Six subprocess provenance regressions fail on the prior production sources
  and pass after repair; unrelated cached module entries remain untouched.
- Bind each engineering progress callback to its phase and variant. A new
  regression defers all callbacks until evaluation finishes: the prior code
  wrongly reported `last_residual` for earlier variants; the repair retains each
  callback's original identity. Evaluation arithmetic is unchanged.
- Clear the nine lint findings with string/import formatting and explicit
  `strict=False` on two `zip` calls. The existing nonempty/length/shape guards and
  arithmetic are unchanged. Existing tests retain their assertions and behavior.

## Bounded validation

| Check | Result |
| --- | --- |
| Full successor lint | Zero findings; same nine-error baseline reproduced first |
| Full successor format check | 52 files pass |
| Runner/import and existing selected regressions | 62 passed |
| Successor control/runtime/public-adapter tests | 39 passed, 3 CUDA-only cases skipped |
| Repository test collection | 203 tests collect without errors |
| AST preservation checks | Pass; exceptions are only the stated imports/callback binding/explicit zip keyword and additive tests |

Commands, exact versions and limits are in `validation.json`; source-scope checks
are in `source_preservation.json`. The isolated local environment uses cached
Python 3.12.14, Torch 2.15.0+cpu, Pytest 9.1.1 and Ruff 0.15.20, with primary-runtime
NumPy 2.3.5 / SciPy 1.17.0. It is not the pinned successor lock environment.
The hosted workflows remain necessary before claiming a green CI gate.

## Preserved scientific boundary

All workflow, protocol, configuration, lock, admission, frozen source and
historical-evidence blobs are preserved. Every runner/model/training function
and frozen setting keeps its AST; the engineering runtime changes only callback
metadata binding. This maintenance pass adds no scientific outcomes, changes no
budgets or evaluator arithmetic, and does not run the held 40-cell matrix.
Existing negative outcomes and scientific limitations remain unchanged.

The full numerical suites, local type/license/static-protocol checks, GPU cases
and scientific campaigns were not run in this bounded local validation.

## First hosted readback and annotation follow-up

The pinned ITFM-R check on `50beedcd4dd3fdcde26237c1ce0da20f73b4c89e`
passes format and lint, then reports 135 type findings in 12 files. Two concern
the new callback's overly narrow `dict` annotation; the follow-up changes only
those annotations to `Mapping`, matching the existing callback interface.
Its runtime AST is unchanged and the deferred-callback regression passes again.
All 133 remaining reported source lines map byte-for-byte to the original head.
The type gate stays enabled; its final hosted readback is a separate validation
step, and no full-CI success is claimed here.
