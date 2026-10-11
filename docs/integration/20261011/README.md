# FIM followup integration — combined result

The five compatible followups to PR38 now share one tested candidate. The
complete ordinary local repository collection passed **864 tests**, with
**7 skips**, **164 passing subtests**, and **33 inherited GradScaler warnings**.
Pytest took 17.31 seconds; the complete test command took 18.224951 seconds.
There were no test failures or corrective reruns in this integration session.

This is an engineering readiness checkpoint. FIM's scientific memory-benefit
hypothesis remains unconfirmed and both protected evaluation protocols remain
held. Existing adverse, mixed and baseline-winning results retain their source
identities and interpretations.

## Source combination

Base: PR38, `92bcc3eb5322d967684673ca61f22a8bbebdc16b`.

| Source PR | Exact head | Behavior adopted |
| --- | --- | --- |
| #39 | `160f3d9d43919568f25db118211d7fb58be81ab0` | Opt-in periodic backward-Euler diffusion solve |
| #40 | `eee27200e6dd50bc774f7a10cfd572da517defaf` | Opt-in stochastic colored noise with analytic normalization |
| #41 | `765c8c58837cd3e6fc8ea86192ed768e382b2551` | Validate rollout inputs before stateful model calls |
| #42 | `a3da3cfc08f69dd0f2354196c2a3fd32b5b9dd72` | Validate an independent episode on an owned model copy |
| #43 | `e24a6a7e502db3f6cc5f6baeff375b36dc8f40e6` | Opt-in precise retention/decay analysis |

All **63 adopted paths** retain their exact inspected Git blob and mode. All
**341 unrelated base leaves** retain their exact identities. The two replaced
base files are `fim/training/rollout.py` from PR41 and `README.md` from PR42;
the canonical index is separately reconciled. No runtime conflict required a
new implementation. The numerical APIs and validation adapter remain opt-in.

All six complete canonical predecessor states are retained byte for byte under
`research/state_snapshots/integration_20261011/`. The current canonical index
unions each followup's source-specific records and points to this new session.
Every preceding closed budget and failure remains attached to its original
source. Initial source receipts that predate hosted CI are historical records;
they are not rewritten to appear to describe a later run.

## Validation and interpretation

- `verify_sources.py --git-sources` verifies each adopted blob against its
  original Git tree, all unchanged base leaves and canonical predecessor entries.
- Maintained Python modules, tests and the new verifier compile successfully.
- One combined full repository collection ran on Python 3.12.14, CPU PyTorch
  2.14.1+cpu, pytest 9.1.1, with one numerical thread and no accelerators.
- `tested_source.json` binds 141 runtime/test/configuration/workflow/verifier
  files to SHA-256 identities. The set hash before and after execution is
  `7dedae52521d3bb3c4b42c61e35c6afa4226528e69fdc2afdb326f030bbc4c2f`.
- `pytest.log`, `pytest.xml`, `environment.json` and `test_receipt.json` retain
  the actual command, environment, timing and outcomes.

Counts from earlier component PRs and any later hosted jobs overlap
this collection and are not added. Pytest emits separate JUnit entries for
subtests: 1,035 entries are 864 ordinary passes, 164 passing subtests and 7 skips.
The seven skips are retained optional/legacy benchmark, fractional, Levy and
persistent-memory module-availability checks; they remain unqualified.

The retained 20261010 source verifier applies to its original tree. This
descendant intentionally supersedes its README/rollout identity expectations
using the new source inventory. Its original script and receipts are unchanged;
a pass of the historical verifier on this descendant is not claimed.

The scoped whitespace check for maintained code, tests and new metadata passes.
The complete staged check reports original traceback whitespace in four imported
historical log/XML files. Their exact source bytes remain preserved, and the
failed all-files check is retained in `whitespace_receipt.json`.

## Remaining gates

Review this combined draft and its PR38 prerequisite before integration into
main. No constituent PR is closed or merged here. Both ordinary hosted workflows passed the same 864/7/164 collection.
Their generated merge checkout has exactly the reviewed implementation tree.
`hosted_verification.json` and two decoded logs retain the run/job URLs,
test totals, environment and source identity. The protected scientific job was
skipped. PR44 remains a draft; the implementation commit is
`82f724b725772bbd3e6d233b9d69c3d6ec1e0089`, tree
`6c694ff12268a1a6b7b0164ca85dcbe49369b1c9`. This later metadata followup records
publication and hosted acceptance without changing any tested source byte.

Scientific execution remains a separate, source-bound decision. No protected
matrix, held-seed campaign, protected outcome access, retained development-demo
rerun, paid compute, scientific dispatch, manuscript update or submission was
performed. Existing protocol/model/benchmark/metric/workflow/result bytes are
preserved. The 40-cell study and delayed-recall successor cannot be claimed
complete from this engineering gate.
