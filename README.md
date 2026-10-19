# Current research state — 19 October 2026

Read [CURRENT_RESEARCH_STATE.json](CURRENT_RESEARCH_STATE.json) before further
research work. It records version boundaries, budgets, protected studies,
claims, failures and the next action; update it at session close without
rewriting historical evidence.

The maintained v2 engineering repair passed **52 CPU tensor/runtime tests and
23 standard-library controls**, with **four CUDA skips**. One bounded synthetic
smoke completed. Full retrieval lost to last residual and persistence; this is
not a positive scientific-performance or novelty result. The
[execution report](audits/fim-causal-recovery-20261019/REPORT.md),
[flaw audit](audits/fim-causal-recovery-20261019/FLAW_AUDIT.md),
[exact specification](audits/fim-causal-recovery-20261019/TECHNICAL_SPECIFICATION.md)
and [prospective baseline-first protocol](audits/fim-causal-recovery-20261019/BASELINE_PROTOCOL_NOT_RUN.md)
separate what is implemented from what remains untested. The
[current launch handoff](successor/itfm_v2/kaggle/HANDOFF.md) binds the portable
package to its implementation and evidence. Original, historical v2, consumed
v3 and scientifically blocked v4 evidence remain separate.

The dated introductions below are preserved as history.

# Kaggle engineering preparation — 16 October 2026

The maintained `fim-sota` CLI now has a bounded engineering path for training,
checkpoint recovery, causal tensor-memory evaluation, finite campaigns and
artifact export. Start with [the handoff](successor/itfm_v2/kaggle/HANDOFF.md)
and [the readiness report](successor/itfm_v2/kaggle/GPU_READINESS.md).
Status is **BLOCKED** pending actual tensor correctness/recovery checks: local
resource admission refused model imports. This new synthetic diagnostic path
does not reopen the consumed v3 campaign or establish public-benchmark efficacy.
Earlier code is preserved under `audits/kaggle-readiness-20261016/before`;
all historical evidence and the v4 scientific contribution gate remain separate.

# Current iteration: ITFM v4 — 16 October 2026

The [v4 software bundle](https://github.com/THE-BU1LD/Fabric-Induced-Memory/tree/codex/itfm-v4-runtime-20261016) is published at commit `266fe56ff9897e57b7e610fb7afc369fe767855b`.
The [standard public macOS verification run](https://github.com/THE-BU1LD/Fabric-Induced-Memory/actions/runs/38847339155) passed **87 scalar tests and 10 tensor integration tests**, with zero failures, errors or skips. The code quality revision adds strict receipt validation, exact scalar test inventory and owned-worker cleanup on interruption.

The [publication report](successor/itfm_v4/repairs/observation_pipeline_v1/publication_20261016/REPORT.md) retains exact source identity, runtime environment, eight raw receipts, the initial setup failure and its correction. The tested bundle validates observation timing, owner isolation, durable scoring/replay and export integrity. Earlier sealed bundles remain unchanged.

The [canonical v4 research state](successor/itfm_v4/RESEARCH_STATE.json) remains scientifically blocked at the contribution gate. Eight concept reviews have not established a defensible new contribution; no v4 scientific trial, protected evaluation or forecasting-performance claim is supported. The [35-item checklist](successor/itfm_v4/repairs/observation_pipeline_v1/extension_20261015/ALL_ITEMS.json) remains the full research work map.

Retained v2 remote replay passed 66/66 strict checks. V3 passed 12 runtime tests but completed zero of 30 campaign cells before resource failure; its one-campaign allowance is consumed. See the [execution report](audits/seed307-diagnosis-20261014/REPORT.md).
The dated introductions below remain history.

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
The full research snapshot, including the preserved historical relocation, is
now tracked in Git. This archive does not constitute a validated scientific release. The original FIM negative study
in `FIM/` and the prospective public ITFM-R campaign remain separate.

See [the revision report](successor/itfm_v2/publication/itfm-v2-paper-20261011/REVISION_REPORT.md)
and [publication checklist](PUBLICATION_READINESS_CHECKLIST_2026-10-11.md) for
the current work status. The manuscript is corrected but remains release-blocked.

## Canonical FIM implementation

# Fabric-Induced Memory (FIM)

FIM is a research framework for neural sequence models with a structured latent memory fabric rather than only a larger context window or a single compressed hidden state.

The implementation explores local information propagation, persistent traces, decay, salience, retrieval, latent geometry, stochastic dynamics, and long-horizon forecasting.

## Evidence boundary

Read [`RESEARCH_TRUTH.md`](RESEARCH_TRUTH.md) before quoting results.

The public repository contains real source code, tests, experiment runners, stored checkpoints/logs/configs, a delayed-recall mini-suite, historical paper-reference values, and an existing PDF manuscript. These artifacts are not all equivalent forms of evidence.

In particular:

- `results/mini_processed/mini_suite_summary.md` is a persisted compact delayed-recall comparison. In that stored artifact, FIM is competitive but does **not** have the lowest rollout MSE; DeepONet does.
- `paper_reference_results.json` explicitly identifies its values as compact results transcribed from the existing paper. Treat those as historical paper-reference values unless they are freshly reproduced from the current commit/configuration.
- The repository must not be described as showing that the full FIM model consistently beats every baseline or ablation. The committed artifacts do not support that blanket statement.

Negative results and baseline wins are part of the scientific record and should be preserved.

## Canonical execution

Install the repository and test it before research runs:

```bash
python -m pip install -e . pytest
python -m compileall -q fim fim_experiments scripts tests
MPLBACKEND=Agg pytest -q
```

Run one maintained training/evaluation experiment:

```bash
DEVICE=auto \
BENCHMARK=lorenz96 \
MODEL=fim_plus \
EXP_NAME=lorenz_fim_seed42 \
SEED=42 \
bash scripts/run_train.sh
```

The experiment runner preserves resolved configuration, logs, metrics, figures and checkpoints under the selected results root. It does not rely on an ad-hoc root-level `model.pt`.

Evaluate an existing checkpoint without retraining:

```bash
CHECKPOINT=results/lorenz_fim_seed42/lorenz96_fim_plus/checkpoints/final_state.pt \
DEVICE=auto \
bash scripts/run_eval.sh
```

Run the supported model-comparison suite:

```bash
python scripts/run_full_suite.py --device auto
```

### Component-ablation warning

The historical `scripts/run_ablation.sh` used flags that were not connected to the maintained experiment CLI. It is intentionally disabled rather than producing misleading paper evidence. Component ablations must be implemented as explicit tested configuration switches with frozen semantics before they are run or quoted.

## Repository map

- `fim/` — core dynamics, geometry, memory, operators, stochastic modules, training, and utilities.
- `fim_experiments/` — benchmark systems, training/evaluation runners, and experiment orchestration.
- `tests/` and `fim/epistemic_tests_full/tests/` — implementation and scientific sanity checks.
- `scripts/` — maintained train/evaluation/suite entrypoints plus paper/research utilities.
- `results/` — stored run artifacts and processed summaries.
- `docs/` — architecture, mathematics, and evidence-bounded experiment documentation.
- `paper/` — existing manuscript artifact.
- `RESEARCH_TRUTH.md` — current evidence and claim boundary.

## Reproducibility rule

For new paper evidence, preserve the Git commit, dirty-tree state, resolved configuration, seed, benchmark/data identity, raw metrics/predictions, logs/checkpoints, and failure state. Paper tables and figures should be regenerated from those retained artifacts rather than manually copied from historical summaries.

## Current status

**IMPLEMENTED / EVIDENCE_PARTIAL.** The repository is a substantive research implementation with stored experimental artifacts, but a fresh frozen multi-seed benchmark-and-ablation reproduction is still required before stronger submission claims are justified.
