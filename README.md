# Fabric-Induced Memory (FIM)

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
