# Test discovery boundaries — 16 October 2026

## Rationale

A repository-root `pytest -q` recursively collected frozen source copies,
publication snapshots, descendant `runs/` bundles, and nested successor projects
under incompatible Python package roots. This produced 136 collection errors
in CI run 38869002547. It must not be addressed by hiding failures in maintained
suites or by changing frozen scientific evidence.

## Supported suites and verification ownership

| Component | Current role | Command / CI owner | Outcome claim |
| --- | --- | --- | --- |
| Root `tests/` | Maintained FIM implementation and trajectory-isolation checks | Root `.github/workflows/ci.yml`, `pytest -q tests`; also in root `pytest -q` via `testpaths` | No paper result |
| `fim/epistemic_tests_full/tests/` | Maintained epistemic/core sanity tests | Root CI, `pytest -q fim/epistemic_tests_full/tests`; also in root `pytest -q` | No scientific outcome |
| `successor/itfm_v2/tests/` | Independently maintained ITFM-R suite with locked CPU dependencies | `.github/workflows/itfm-sota-ci.yml`; working dir `successor/itfm_v2`, `pytest -q`; triggered on v2 changes | Synthetic/public evidence separate |
| `successor/itfm_v3/test_standard.py` | Active standard-library v3 contract tests | Root CI runs `python -m unittest -v test_standard` from `successor/itfm_v3` | No v3 campaign outcome |
| `successor/itfm_v3/test_runtime.py` | Resource-admitted numeric development tests | Run only in admitted v3 runtime per `successor/itfm_v3/README.md` | Previously consumed campaign allowance; no rerun authorized |
| `successor/itfm_v1/tests/` | Frozen pre-execution development study | Dedicated isolated `PYTHONPATH=src` and project environment; verified receipt still needed before claiming coverage | Not promoted by root CI |
| `successor/itfm_v4/repairs/observation_pipeline_v1/test*.py` | Separate v4 prospective engineering gate | v4 publication bundle/release verification, with exact source and runtime environment | Not swept into root pytest; re-verification remains separate |
| `FIM/**` | Historical archive | Immutable archive release instructions in `FIM/ARCHIVAL_RELEASE.md` | Archive not present-day root package |
| `audits/**`, `remote_execution/**`, nested `runs/**` and `publication/**/frozen_source/**` | Retained evidence, remote bundles or snapshots | Verify hashes / source-specific tools, not global test discovery | Keep immutable, do not delete |

Root discovery is **inclusion-based** via `pytest.ini` `testpaths`, not a
claim that every source file under this monorepo is tested. An explicit
`pytest -q successor/itfm_v2/tests` from the root is NOT a valid substitute
for running its independently locked environment and workdir. Do not use the
new root green status to certify frozen research campaigns or v1/v4 readiness.

## Engineering acceptance

- Confirm collection lists tests from both declared root paths (no silent
  removal of trajectory tests).
- Verify CI against an exact PR head after the dependency import fix in PR #47.
- Verify the independently maintained ITFM v2 workflow in its own matrix;
  record untested successors as explicit open gates, not passing suites.
- No workflow dispatch or scientific result-bearing run is authorized by this
  change. The protected 40-cell trajectory-isolated matrix remains on hold.

Reference: issue #48 and draft PR #47.
