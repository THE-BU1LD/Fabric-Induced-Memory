# FIM CI import and format repair — 2026-10-30

## Outcome

The trajectory-isolated runner now imports the maintained experiment module by
its package name. An unrelated cached module named `main` can no longer prevent
test collection or silently replace the runner's canonical evaluator. The
successor CLI also passes the format step that stopped its latest hosted check.

This is a local repair candidate based on PR #50 head
`466a4f3d45b935bb6236399c9f7beccd69d99a2d`, on branch
`codex/fim-import-ci-repair-20261030`. The remote head was reread before closeout
and had not changed. No remote branch, PR, comment, workflow dispatch, merge,
release, deployment, or scientific outcome run was created by this repair.

## Demonstrated failures and changes

The [root CI job](https://github.com/THE-BU1LD/Fabric-Induced-Memory/actions/runs/39777488332/job/119820040003)
and [component-ablations repository-test job](https://github.com/THE-BU1LD/Fabric-Induced-Memory/actions/runs/39777488430/job/119820040658)
both stopped during collection because `import main` reused the root CLI wrapper,
which has no `run` attribute. Changing `sys.path` does not replace a module already
cached in `sys.modules`.

`scripts/run_trajectory_isolated_ablations.py` now imports
`fim_experiments.main` explicitly and adds the repository root to its import path
for direct invocation from another directory. Training-module imports and all
runner functions, budgets, variants, protocol checks, and evaluator logic remain
unchanged.

New subprocess regressions preload both the real root wrapper and an unrelated
`main` that has its own `run`. Both cases failed before the source repair. They
now verify the intended module and evaluator identities while preserving the
unrelated cache entry. A third test exercises direct `--help` invocation outside
the repository without `PYTHONPATH`.

The [ITFM-R format job](https://github.com/THE-BU1LD/Fabric-Induced-Memory/actions/runs/39777488434/job/119820040328)
stopped at one missing blank line after a local import in
`successor/itfm_v2/src/fim_sota/cli.py`. The repair adds that line. Its Python AST
is unchanged.

## Validation

| Check | Result |
| --- | --- |
| Both collision regressions before the source repair | 2 failed, including the hosted `AttributeError` |
| New regressions plus existing trajectory-isolation tests | 14 passed |
| Full repository test collection | 197 tests collected, no errors |
| Exact successor `ruff format --check src tests scripts` | 52 files pass |
| Ruff check of the new regression file and changed successor CLI | Pass |
| Runner function/class AST preservation | Pass |
| Successor CLI AST preservation | Pass |
| Existing scientific state, outcomes, workflows and inherited-lint files | Preserved |

Commands, outputs, and admission evidence are retained in `validation.json`.
Source identities and preservation checks are retained in
`source_preservation.json`. The temporary CPU environment used Python 3.12.14 with
Torch 2.11.0+cpu, Pytest 9.1.1 and Ruff 0.17.0; it inherited other installed
scientific dependencies. This bounded check does not reproduce the complete
successor lock environment. The full numerical suites, static protocol audit,
type check, license scan, hosted checks and scientific matrix were not rerun.

## Remaining gates

The exact successor lint command, previously skipped after the hosted format
failure, reports nine pre-existing errors in five unchanged files:

| File, relative to `successor/itfm_v2` | Existing errors |
| --- | --- |
| `scripts/check_sota_readiness.py` | E501 at line 31 |
| `src/fim_sota/engineering_runtime.py` | E501 at lines 645 and 647; B023 at lines 677 and 691 |
| `src/fim_v2/audit.py` | I001 at line 1 |
| `src/fim_v2/public_adapter.py` | B905 at lines 226 and 230 |
| `tests/test_engineering_control.py` | E501 at line 27 |

Those files remain byte-identical to the PR head. Historical source is protected;
this repair does not alter it or weaken workflow coverage to conceal failures.
Review of those inherited findings and a later hosted check of the reviewed
candidate remain necessary before claiming a green CI gate.

The original corrected 40-cell confirmatory matrix remains held and manually
dispatched only. Historical v2 evidence and consumed v3/v4 allowances remain
untouched. Existing scientific limitations and negative outcomes remain in
`CURRENT_RESEARCH_STATE.json`; its new entry records only this engineering work.
