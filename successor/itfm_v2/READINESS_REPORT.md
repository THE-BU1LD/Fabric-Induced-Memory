# Current ITFM v2 publication status

The three-seed synthetic study completed earlier on 11 October and its
historical verifier passed 96 checks. The later publication audit recovered
all 20 executed source files exactly, corrected the manuscript definitions,
and added non-overwriting verification/reproduction tools.

Publication is still blocked on admitted numerical replay and PDF build/visual
QA, independent reproduction/review, an immutable archive, and unspecified
release terms/destination. Optional author fields were omitted at author request.
See `publication/itfm-v2-paper-20261011/README.md` inside the successor directory
and the repository publication checklist. The public campaign remains deferred
for this narrow paper and scientifically not evaluated.

The final local audit passes **228/228** source/scalar-record checks, and all
**15** preservation/release-tool tests pass. The failed replay/build admissions
are retained; no new numerical run or PDF is claimed. The revision report and
complete 59-item disposition are in the publication workspace.

## Earlier status retained as dated history

# ITFM v2 readiness

Date: 2026-10-11

## 17:04 IST public-stack implementation update

The dedicated `fim_sota` package, public-data/backbone contracts, production
memory bank, trainer, metrics/statistics, orchestration, verification, CLI,
platform pins, container definition, CI workflow, cards, and security/license
boundaries are now implemented at source level. Ruff 0.15.20 reports zero
issues and all 27 Python files in the new package plus SOTA test scope parse.
The bounded 28-file source bundle has SHA-256
`d6956a54741edda75a554df63917ab8bbfc5117631313fe05055a149b37efc97`.

Runtime verification was not admitted. The latest snapshot measured
188,940,288 free-plus-speculative bytes versus the immutable 536,870,912-byte
floor, 4,799.31 MiB swap used, rising pageouts, and active NeuroCAD/LGWM jobs. No Torch import,
pytest run, dependency resolution, image build, data fetch, upstream
reproduction, training, or evaluation was started for this update.

`IMPLEMENTATION_STATUS_2026-10-11.json` is the machine-readable boundary. It
marks exact official preprocessing, paper-faithful LANO/HS-FNO implementation,
external baseline reproduction, runtime tests, public-data evidence, and every
scientific/release claim as partial or blocked.

## Current verdict

The v2 scientific contract, complete-context alias task, streaming tensor
residual memory, controls, intervention evaluation, joint decision rule,
resource-guarded pipeline, artifact hashing, and tests are implemented.

Formatting, Ruff lint, bytecode compilation, the 10-point static contract audit,
and 10 runtime tests pass. The guarded pipeline admitted the tests at
587,972,608 free-plus-speculative bytes, then stopped before the experiment when
headroom fell to 527,171,584 bytes, below the immutable 536,870,912-byte floor.
Zero development seeds started.

The guard was subsequently strengthened to require zero pageout increase across
a stability interval. The latest check remained blocked at 94,060,544 bytes and
recorded two pageouts during the 0.25-second sample. This guard-only source
change is formatted, lint-clean, bytecode-compiled, and static-audit clean; the
exact runtime suite will rerun automatically at the next admitted pipeline.

No positive result may be claimed until the resource-guarded pipeline completes
all three declared seeds and the independent artifact verifier passes.

## Required execution

```bash
cd /Volumes/PRO-BLADE/GitHubRepos/Personal-Research/Fabric-Induced-Memory/successor/itfm_v2
rtk /opt/homebrew/opt/python@3.11/bin/python3.11 scripts/check_resource_admission.py
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 rtk nice -n 10 \
  /opt/homebrew/opt/python@3.11/bin/python3.11 scripts/run_pipeline.py
```

The pipeline stops before importing Torch when the host is below the immutable
512 MiB free-plus-speculative threshold.

