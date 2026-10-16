# ITFM v2 synthetic paper revision

Scope is route B: the exact synthetic mechanism case study. The public ITFM-R
campaign is deferred and does not support this paper. The user confirmed sole
authorship and requested no further metadata. Ryan remains the displayed author;
optional affiliation/funding/conflict fields are omitted. Distribution terms
and the target destination remain unspecified, not silently chosen.

`frozen_source/` contains all 20 executed files, each matching the historical
manifest. The original `pyproject.toml` was recovered from a retained session
response and accepted only after its pre-existing SHA-256 matched.
`SOURCE_RECOVERY.json` records the recovery. `before/` preserves the earlier
paper and staging tool. `PRESERVATION_BASELINE.json` protects the original
source and every existing result/receipt from mutation.

The current public-stack `pyproject.toml` is deliberately preserved. The paper
uses the separately recovered source, not that later environment definition.
`REPLAY_REQUIREMENTS.txt` records proposed replay versions observed locally,
not a fabricated historical dependency lock. Numerical replay has not run in
this revision because resource admission failed.

## Safe commands

From `successor/itfm_v2`, run lightweight verification:

```sh
rtk python3 scripts/build_paper_release.py verify
rtk python3 scripts/build_paper_release.py gate
```

The `gate` command returns exit 3 until all required substantive receipts exist.
Neither command modifies frozen evidence. An optional `--receipt NEW_PATH`
saves a new receipt exclusively and refuses existing or protected paths.

After stable resource admission and checking process ownership, use a unique
output directory for the numerical replay:

```sh
rtk python3 publication/itfm-v2-paper-20261011/tools/reproduce.py execute \
  --output publication/itfm-v2-paper-20261011/replays/UNUSED_DIRECTORY
```

This copies only the recovered source into an isolated workspace, runs its
original 10-test pipeline, then uses an instrumented in-memory evaluator to
retain raw tensors and independently recompute their errors and an exact-key
reference. It does not install packages. It is a same-host/operator replay,
not a clean dependency installation or external reproduction. Its numerical
worker remains runtime-unverified until a successful admitted run exists.
Use Python 3.11.15 and Torch 2.12.1 for this exact replay; keep discrepancies.

Do not execute the historical `scripts/run_pipeline.py` or
`scripts/verify_results.py` directly in the live evidence tree: those historical
files overwrite their designated result paths. They remain unchanged for
provenance and are run only in fresh isolated copies by the new wrapper.

## Paper and release

`../../paper/main.tex`, `references.bib`, and `generated_results.tex` are the
revised paper inputs. The generated tables include all nine conditions and
all three seeds. The timeline and derivation explain the construction without
inventing measurements. `RELEASE_GATES.json` records remaining release gates.
Receipts are bound to the exact manuscript hashes and require gate-specific
evidence, not just a PASS label. External attestations still require human
identity/authority checks; the local tool cannot authenticate a reviewer.

Compilation, rendering, source staging, dependency installs and numerical work
remain subject to the repository resource contract. When admitted, build into
a fresh output directory with Tectonic or the documented LaTeX toolchain,
retain the log, and inspect every page. Never claim a server build from local
compilation. Publication readiness remains blocked until the open gates close.

## Guarded PDF build

From `successor/itfm_v2`, after checking ownership and disk, run:

```sh
rtk python3 publication/itfm-v2-paper-20261011/tools/build_pdf.py \
  --output publication/itfm-v2-paper-20261011/builds/UNUSED_DIRECTORY
```

The two-sample memory check runs before source copying or compilation. The build
has a 120-second process-group budget. `builds/attempt-1/BUILD_RECEIPT.json`
records a real blocked admission; no compiler ran. Rendering and review are
still required. `TOOL_TESTS_FINAL_V3.json` records 15 passing standard-library
preservation/tool tests and clean Ruff checks, not numerical worker execution.
