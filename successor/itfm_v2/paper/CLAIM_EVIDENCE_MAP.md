# Revised manuscript claim-to-evidence map

Scope: route B, a constructed exact residual-reuse case study. This map describes retained evidence and analytic arguments. It does not assert raw-output replay, external reproduction, or publication readiness.

## Immutable source and evidence

The manifest records executed-source SHA-256 `5d01a2917fe78719e40fecbc702131e76a3b7f842986d79d47e5660c7221631a` and protocol SHA-256 `8cb1ea0493da40ece0663f1777dccf5bf01f367050b771434a9574b0ef52330b`. All 20 exact source files, including the original `pyproject.toml`, now exist in `../publication/itfm-v2-paper-20261011/frozen_source`. See `SOURCE_RECOVERY.json` there. The later public-stack project metadata remains unchanged in the live root. No complete historical dependency lock was recorded.

| Artifact (relative to synthetic-study root) | Verified SHA-256 |
|---|---|
| `results/development-v2/manifest.json` | `76c2bda7436f9b4798c423b02c99a3d65df3170e08629b1f847c0b6a3e4606bf` |
| `results/result_verification.json` | `fe7ab41ec8d9634e82c08b292cf4c1d0f4d2f7a7d91d9dbe2463fce23969c6ce` |
| `results/verification_receipt.json` | `4b91f3db87dedb80e2851b1b28427a33b9846c0ca3fd47c7f53de4cd64f30f79` |
| `results/development-v2/seed-101/result.json` | `8f03e8449a20fd05ea34c95a600d9e53ca195d145d1b543bbe1a948ab82f92b0` |
| `results/development-v2/seed-211/result.json` | `948bb206d63721a260e1742f3ce7e39d2771d07b5a3b9840535d3ceffc4f7867` |
| `results/development-v2/seed-307/result.json` | `f147987396d95af850d400bd6a7490cf3b11877aa91175a72085a270d942d165` |

The baseline receipt in the revision workspace hashes every pre-existing result file and all 20 live source paths. The current verifier checks those original bytes, all recovered source bytes and their exact file list, every prediction-record hash, record structure/finite values, reported summary metrics, decisions, citations, and the generated tables. Its current manuscript hashes bind the revision. The earlier paper's 19-check receipt and the historical 96-check result receipt remain historical; their successful status does not certify later files.

## Empirical claims

| Manuscript claim | Protocol/run/records | Analysis and presentation | Limitation |
|---|---|---|---|
| Three seeds, nine conditions; no excluded or failed retained runs | `protocols/development_v2.json`; `results/development-v2/manifest.json` | Current verifier seed identity, condition completeness and manifest completeness | Development design, no external preregistration |
| Base query NMSE 0.487685; tensor and oracle 0 | All `seed-*/predictions-{no_memory,tensor_memory,oracle_anchor}.json`; `result.json` summaries | Pool stored squared errors / element counts / recorded training variance; first generated table | Scalars are retained; target/prediction tensors were not originally saved |
| Erasure returns base error; sign flip about 1.950739 | All `predictions-{erase,sign_flip}.json` | Same pooled estimator, all three seed columns | Base and erase execute the same output ablation |
| Spatial mean, recent restriction, last residual have base-level query error | All `predictions-{vector_broadcast,recent_only,last_value}.json` | First table and numerical decision predicates | Recent restriction is empty, not a valid short-history competitor |
| Non-query NMSE unchanged, about 0.0541872 | `predictions-{no_memory,tensor_memory}.json`, 486 non-query rows per seed/condition | Second generated table | Same fixed synthetic generator and base |
| Matching-anchor precision 1 | Ordinary tensor records: source anchor versus query anchor on valid queries | Second table | Anchor agreement is not independent proof of useful memory |
| Swaps improve donor distance relative to base on every evaluated valid query | `predictions-paired_swap.json` stored `donor_direction_success`; frozen evaluator predicate | Second table; formula for D_j | Stored boolean, not independently recomputed raw distances; deliberate other-owner access |
| All eight numerical predicates pass | Per-seed `decision.gates`, frozen decision thresholds | Recomputed from saved scalars by verifier helpers | Not an additional experiment or external review |
| Five identifiability predicates and separate dataset checks were reported | Per-seed `identifiability.gates` and `dataset_checks` | Explicitly reported as stored historical flags | Raw generator/target replay is still required |

Each of the 27 condition files has 594 rows: 108 query and 486 non-query rows. Every target has 256 elements. No partial predecessor result supports a v2 claim. Values print to six decimal places; no artificial confidence intervals or population significance are inferred.

## Algorithm and analytic claims

| Claim | Direct source/proof | Scope |
|---|---|---|
| Full context key, capacity 128/owner, min age4, top1, cosine threshold0.999999, temperature0.05 | Frozen `memory.py`, `evaluation.py`, protocol | Equal repeated-context cosine comparisons; actual key storage is L times last-frame storage |
| Validation calibration uses query rows with valid retrieval; alpha clipped [0,1], zero if denominator0 | Frozen `evaluation.py` numerator/denominator and `calibrate_alpha` | No trained forecasting backbone |
| Equal contexts cannot reveal opposite residuals to identical context-only outputs | Proposition/proof in main.tex: expand the two squared norms | Does not constrain owner-aware/recurrent predictors |
| Erasure/base, fourfold sign-flip error, zero spatial-mean signal, donor swap expectation | Algebra in main.tex given exact matching value and alpha1 | Analytic expectation, not new measured robustness |
| Ignored retrieval and oracle shortcut can fool individual metrics | Explicit analytic counterexamples in main.tex | Illustrations proved by definitions; no unexecuted test is reported as a run |
| Equal per-owner/per-family sizes make pooled and hierarchical means agree | 6 query/27 non-query rows per owner, two owners/family, 256 elements/row | Only this balanced design |
| Event construction, owner parity leakage risk and predict-before-write ordering | Frozen `data.py` and `evaluation.py` | Events are not continuous overlapping trajectory windows |
| Retrieval cost O(N + N_o D_K + D_V) and storage O(N(D_K+D_V)) | Global owner filtering and top1 reference implementation | Asymptotic source analysis, no timing/efficiency measurement |

## Literature and release status

The six cited primary sources are audited in `../publication/itfm-v2-paper-20261011/CITATION_AUDIT.json`. Existing associative recall, external memory, synthetic retrieval evaluations, interchange interventions and retrieval forecasting are acknowledged. No referenced performance number is used as an ITFM comparison. This is a targeted comparison, not a systematic novelty proof.

The public campaign is explicitly deferred. Raw replay, clean-environment replay, external reproduction, independent scientific review, PDF build/render inspection, public archive, author metadata/rights and destination requirements remain pending or blocked in the release ledger. The prepared replay worker and internal editorial review do not close those gates.
