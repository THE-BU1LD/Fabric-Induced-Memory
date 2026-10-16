# ITFM implementation and research status — 16 October 2026

The durable evidence repair is implemented and verified at the scalar software
level. The full research pipeline is incomplete: the contribution gate remains
blocked, and tensor execution fails the existing resource rule.

## Implemented and verified

- `journal.py`: bounded records, durable non-overwriting publication, cooperative
  single-writer locking, sequence/hash validation and externally anchored recovery.
- `durable.py`: persistent actual prediction/scorer/failure recording, replay into
  the existing validator, sealed export recovery, and audit replay of observed
  coordinates using the original base predictions and actual arrival times.
- `verify_export.py`: executable raw-bundle reanalysis against separately frozen
  plan bytes; altered metrics, identities, schema and evidence are refused.
- `test_recovery.py` and `run_recovery_tests.py`: 33 added tests, 69 total with the
  existing pipeline tests. Final run: 69/69, zero failures/errors, 2.58 seconds,
  no torch/numpy/tensorflow imports. Ruff lint/format and AST parsing pass.

The first source revision passed 63 tests; the final revision adds strict export
schema/boolean checks and further negative controls. Both source revisions and
receipts remain retained. Abrupt process exit, disk failure before and after
publication, lost tail records, interior tampering, contention, incomplete
outputs and negative numerical gates are covered. These are software fixtures.

Read `../PERSISTENCE.md` for API usage and recovery limits. A restarted recorder
cannot generate new model predictions. Observation replay is an audit report,
not tensor-bank or optimizer restoration. Crash ambiguity after publication but
before anchor delivery requires inspection; history is never silently rolled
back. Hashes need a separately retained anchor and do not authenticate provenance.

## Verification and preservation

`FINAL_VERIFICATION.json` confirms all 440 protected predecessor files, all 12
frozen v3 sources, and all 44 files in the previous repair package including its
manifest remain byte-identical. The final source hashes match the tested files.
Historical invalid, negative, partial and resource-blocked evidence is retained.
No consumed v3 campaign was restarted.

The fresh tensor launcher refused admission before model imports: 247.6875 and
185.890625 MiB free-plus-speculative memory versus the unchanged 512 MiB floor;
pageouts increased by 3. Thus the existing six tensor tests remain unexecuted.
Full-shape memory/runtime and scientific runs were not started.

## Contribution review

Three additional bounded reviews are in `CONCEPT_REVIEW.json`:

1. Support-conditioned low-rank residual completion overlaps established masked
   least squares and [GROUSE](https://arxiv.org/abs/1006.4046); compressed residual
   correction also needs comparison with [EPOC](https://arxiv.org/html/2609.30929v2).
2. Doubly robust residual memory requires explicit selection assumptions and a
   material distinction from [missing-outcome kernel estimators](https://pmc.ncbi.nlm.nih.gov/articles/PMC3491912/).
3. A narrower study of spatial censoring and delay must separate usable-information
   differences from estimator quality. [The delay-aware adaptation audit](https://arxiv.org/html/2610.12232v1)
   already uses causal maturity, shared forecasts and per-coordinate controls.

These are bounded prior-art judgments, not exhaustive novelty certification.
None clears Checkpoint 1; no new forecasting architecture was implemented.
The complete 35-item flaw/cause/severity/repair/evidence inventory is retained in
`ALL_ITEMS.json`, with current partial and open dispositions rather than false
completion flags.

## Ten-checkpoint disposition

| Checkpoint | Current state | Remaining evidence |
|---|---|---|
| 1 Idea ready | BLOCKED | Material contribution, observation assumptions, discriminating test |
| 2 Model specification | NOT_EVALUATED | Paper-complete specification after a defensible idea |
| 3 Methodology | NOT_EVALUATED | Admitted data/terms/splits, matched controls, statistical and compute budgets |
| 4 Model implemented | NOT_EVALUATED | Paper-to-model equivalence, tensor/gradient/shape acceptance |
| 5 Experiment system | NOT_EVALUATED | Actual end-to-end public runner, baselines, profiling and analysis |
| 6 Run ready | NOT_EVALUATED | Admitted full pipeline dry run including tensor state recovery and measured cost |
| 7 Confirmatory freeze | NOT_EVALUATED | Prospective complete source/data/protocol/environment freeze |
| 8 Evidence locked | NOT_EVALUATED | Raw scientific results supporting each scoped claim |
| 9 Verified | NOT_EVALUATED | Clean-environment reproduction and independent adversarial verification |
| 10 Research complete | NOT_EVALUATED | Defensible terminal study plus paper and complete reproducibility package |

Canonical disposition remains NOT_EVALUATED for prospective v4. Earlier versions
retain their own outcomes. There is no new positive model result, SOTA result,
independent reproduction, publication-ready paper or authorized submission.

The renewed instruction extended the previously proposed bounds without resetting
history: five of six cumulative observation-repair rounds and eight of eight
concept reviews are consumed. One repair round remains. No paid compute,
protected evaluation or new scientific campaign was authorized by these bounds.
