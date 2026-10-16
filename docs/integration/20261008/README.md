# Combined FIM engineering integration — 8 October 2026

This candidate assembles the existing engineering repairs from PRs #20–#24 on
`main` commit `08d2c3df4262612bffe38488f97e263769ff7ef6` and verifies them
together. The complete maintained collection passes with **176 passed,
7 skipped and 10 subtests passed**. Compilation, the frozen v2 source verifier,
a wheel build and an installed-package integration fixture also pass.

This is the combined integration candidate for review. It does not establish
that the changes are on `main`, complete the scientific reproduction issue, or
release the protected scientific execution hold.

## Exact source inputs

| PR | Reviewed head | Integrated behavior |
| --- | --- | --- |
| [#20](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/20) | `c6c1188ded48f4cd83ade3553ca10914be7217cd` | Graph accumulation, inference/training cache behavior and model integration |
| [#21](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/21) | `3ff0fe1de64795e6241bf0ee54c5c5ad06a3850f` | Opt-in native benchmark runtime contracts |
| [#22](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/22) | `605ca23a3961fcbf95fabfd211ab69e8a77c2ca3` | Staged evidence publication and recovery after ordinary write failures |
| [#23](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/23) | `28f8da75642204f7a9447aaef73532ab65d7a5c8` | Explicit raw/EMA evaluation weights with preserved training-resume state |
| [#24](https://github.com/THE-BU1LD/Fabric-Induced-Memory/pull/24) | `758bf83430d30cfb3f29b204650dafe3d13da803` | Correct current protocol and historical-result boundaries in Research Truth |

The five drafts contribute 15 distinct paths. All functional source and test
files are byte-identical to their reviewed PR heads. Their only overlapping
path is `.github/workflows/current-component-ablations.yml`. PR #23 uses two
different explanatory comments; its complete file was selected. Removing
comment-only lines makes all four workflow proposals identical. The shared
condition remains `github.event_name == 'workflow_dispatch'` on the scientific
job. Ordinary repository tests remain enabled for pull requests.

The integration adds this report and its two receipts. It changes no additional
implementation, experiment CLI selection, frozen protocol, seed, budget,
checkpoint, raw metric, manuscript or result artifact.

## Combined verification

The test runtime was isolated from the repository: Python 3.12.14,
PyTorch 2.14.1+cpu and pytest 9.1.1 on Linux, with single-thread CPU settings.
The full collection completed in 8.05 seconds. [pytest.xml](pytest.xml) retains
the case results, including every skip reason; its host identifier was removed.
The seven skips are legacy optional benchmark, fractional-diffusion,
Levy-module and persistent-memory checks whose referenced components are not
available. They are not reported as passing.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -r s
python -m compileall -q fim fim_experiments scripts tests
python -m scripts.verify_delayed_recall_successor_v2
python -m build --wheel --no-isolation
```

Install the dependencies declared in `pyproject.toml`, plus `pytest` and
`build`, in an isolated environment before running these commands. The source
verifier is invoked as a module from the repository root; a bare direct-script
invocation in an uninstalled checkout cannot import `fim_experiments`.

The built wheel was installed into a fresh target outside the source checkout.
Imports for the graph repair, runtime successor, evidence publisher and
checkpoint evaluator all resolved inside that target. A small interoperability
fixture warmed graph topology in inference mode, restored explicit EMA weights
through the checkpoint evaluator, and then checked accumulation and finite
training gradients. The same installed package also generated two synthetic
observations and published three artificial UTF-8 files. These are bounded
implementation fixtures, with no scientific performance interpretation.

[verification.json](verification.json) records the base/source heads, every
integrated file's Git blob and SHA-256, protected original blob identities,
runtime, commands, counts and installed-package receipt. The whole existing
tree was checked: every original blob outside the declared 15 integration
paths retained its exact Git identity.

## Scientific boundary and remaining review

`fim_experiments/benchmark.py` retains Git blob
`c91a6afe9d86830c6bcfaa2d9187547d9688050d`. The v2 verifier returns
`VERIFIED_PREOUTCOME_METHOD_FREEZE`, `execution_authorized: false` and
`outcome_access_allowed: false`. The existing experiment CLI still selects
the frozen benchmark implementation; the native runtime successor is opt-in.
No scientific matrix or retained-checkpoint evaluation was executed for this
integration.

The existing adverse ablations and baseline-winning results remain as recorded
in [Research Truth](../../../RESEARCH_TRUTH.md). Passing these engineering
checks does not establish a memory benefit, forecasting advantage, completed
40-cell matrix or publication readiness.

The remaining engineering decision is acceptance of this combined candidate
and its exact hosted-check result. Existing PRs #20–#24 and `main` were not
changed by the local qualification. Scientific execution additionally requires
its separate exact-source, protocol, environment, destination and hold-release
decision; this integration does not make that decision.
