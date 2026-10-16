# Current iteration: ITFM v4 — 16 October 2026

The active status is [ITFM v4's research state](successor/itfm_v4/RESEARCH_STATE.json).
The contribution gate remains blocked. The new
[observation pipeline repair](successor/itfm_v4/repairs/observation_pipeline_v1/README.md)
adds tested observation-event handling and prediction-export validation, with
a target-free tensor bridge awaiting admitted runtime checks. No v4 performance
claim is supported.

The [durable evidence extension](successor/itfm_v4/repairs/observation_pipeline_v1/extension_20261015/REPORT.md)
adds persistent prediction/failure records, observation replay and frozen-plan
export reanalysis. Its combined software suite passes 69/69 tests. Tensor
execution remains resource-blocked; eight concept reviews have not established
a defensible new contribution. The report maps all 35 flaws and ten checkpoints.

The [round 6 runtime verification package](successor/itfm_v4/repairs/observation_pipeline_v1/round_6_20261016/REPORT.md)
adds strict test-inventory/source checks and an isolated worker deadline. The
combined scalar suite now passes 81/81 tests. Ten tensor tests are prepared in
a reviewed 21-file bundle; public upload and one remote job await approval.

Retained v2 remote replay passed 66/66 strict checks. V3 passed 12 runtime tests
but completed zero of 30 campaign cells before resource failure; its one-campaign
allowance is consumed. See the
[execution report](audits/seed307-diagnosis-20261014/REPORT.md).
The dated v3 introduction below is retained history, not current status.

# Historical ITFM v3 development successor — 13 October 2026

The separate [ITFM v3 study](successor/itfm_v3/README.md) now implements noisy
cues, drifting residuals, distractors and capacity stress, with simple lookup,
finite-history, GRU and attention baselines plus value interventions. Its source
and development protocol are frozen; numerical tests and experiments remain
resource-blocked. **No v3 performance result is claimed.**

The [execution report](audits/improvement-20261013/REPORT.md) records the actual
blocked v2 reproduction, passing static checks, and preservation of all 440
protected predecessor files. The original negative study and v2 synthetic
results remain unchanged. The existing manuscript status follows below.

# Current ITFM v2 paper workspace

The active synthetic manuscript for this local task is
`successor/itfm_v2/paper/main.tex`. Its recovered source, safe verification tools,
and readiness receipts are in
`successor/itfm_v2/publication/itfm-v2-paper-20261011`.
The current worktree includes untracked research and a preserved historical
relocation; the old HEAD and remote below do not identify these manuscript bytes.
No public release of this revision is claimed. The original FIM negative study
in `FIM/` and the prospective public ITFM-R campaign remain separate.

See [the revision report](successor/itfm_v2/publication/itfm-v2-paper-20261011/REVISION_REPORT.md)
and [publication checklist](PUBLICATION_READINESS_CHECKLIST_2026-10-11.md) for
the current work status. The manuscript is corrected but remains release-blocked.

## Historical repository notice

# Fabric-Induced Memory (FIM)

> **Legacy snapshot:** this repository is retained for provenance.  
> **Canonical current repository:** `THE-BU1LD/Fabric-Induced-Memory`  
> Do not use this repository as the default target for new FIM research, CI, paper, or release work unless a task explicitly refers to this historical snapshot.

FIM is a research framework for studying whether a structured latent memory can help neural sequence models retain and retrieve information over long horizons.

The implementation includes an adaptive memory bank, salience-based trace storage, retrieval-conditioned latent updates, scientific/sequence benchmarks, training and evaluation code, experiment utilities, tests, and paper assets.

## Scientific status

**The current repository should be treated as an experimental research codebase, not as evidence of a proven memory advantage.**

The compact paper-reference artifact currently records identical Lorenz ablation values for the full, no-memory, and no-retrieval variants (`1.317169` each), while the recorded no-spectral-mixing variant is `0.869918`. Those values do not establish a positive contribution from the memory or retrieval mechanism.

Before quoting or strengthening any result, read:

- `RESEARCH_TRUTH.md` — current supported and unsupported claims
- `EVIDENCE_LEDGER.md` — claim-to-artifact status and publication gate
- `paper_reference_results.json` — compact values transcribed from the current paper PDF

## Repository layout

```text
.
├── fim/                  # core model components
├── fim_experiments/      # experiment entry point, systems, benchmarks, training
├── configs/              # experiment configuration files
├── results/              # evaluation and result-processing code/artifacts
├── scripts/              # experiment helpers
├── tests/                # test suite
├── paper/                # current paper PDF
├── main.py               # current experiment entry point
└── paper_reference_results.json
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Inspect the current CLI:

```bash
python main.py --help
```

Run the test suite:

```bash
pytest -q
```

## Mechanism ablations

The repository now includes an evidence-bounded ablation path using the current experiment code rather than the legacy `run_train.py` interface.

```bash
bash scripts/run_ablation.sh
```

By default, this runs three predeclared seeds (`41 42 43`) for three mechanism variants:

- `full`: memory writes and retrieval enabled
- `no_memory`: memory writes and retrieval both disabled
- `no_retrieval`: memory writes enabled, retrieval and retrieval-conditioned feedback disabled

The runner writes resolved experiment outputs plus per-run provenance and a root `ablation_manifest.json`. It then produces `ablation_summary.json` and `ablation_summary.md` with descriptive across-seed mean, standard deviation, minimum, and maximum for scalar metrics.

The existence of this runner does **not** mean the paper result has been reproduced. A clean run still needs to complete successfully, outputs need to be inspected, and matched strong non-memory baselines remain required before stronger mechanism claims are safe.

## Reproducibility boundary

The next scientific milestone is a clean current-code ablation suite with explicit full / no-memory / no-retrieval controls, identical predeclared seeds and data settings, retained raw artifacts, provenance, and generated summaries tied to the exact code revision.

A matched strong non-memory baseline and an appropriate predeclared statistical comparison are still required for a paper-facing mechanism claim.

## Research integrity

Negative and null results belong in the repository. If repaired ablations continue to show no measurable benefit from memory or retrieval, that outcome should be reported directly.
