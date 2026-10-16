# Manuscript claim-to-evidence map

This file binds every quantitative or status-bearing manuscript claim to a
retained local artifact. It is not an independent reproduction receipt.

## Frozen identities

| Object | SHA-256 |
|---|---|
| Executed source bundle recorded by the run manifest | `5d01a2917fe78719e40fecbc702131e76a3b7f842986d79d47e5660c7221631a` |
| Frozen development protocol recorded by the run manifest | `8cb1ea0493da40ece0663f1777dccf5bf01f367050b771434a9574b0ef52330b` |
| `results/development-v2/manifest.json` | `76c2bda7436f9b4798c423b02c99a3d65df3170e08629b1f847c0b6a3e4606bf` |
| `results/result_verification.json` | `fe7ab41ec8d9634e82c08b292cf4c1d0f4d2f7a7d91d9dbe2463fce23969c6ce` |
| `results/verification_receipt.json` | `4b91f3db87dedb80e2851b1b28427a33b9846c0ca3fd47c7f53de4cd64f30f79` |
| Seed 101 result | `8f03e8449a20fd05ea34c95a600d9e53ca195d145d1b543bbe1a948ab82f92b0` |
| Seed 211 result | `948bb206d63721a260e1742f3ce7e39d2771d07b5a3b9840535d3ceffc4f7867` |
| Seed 307 result | `f147987396d95af850d400bd6a7490cf3b11877aa91175a72085a270d942d165` |

## Claims

| Claim in `main.tex` | Direct evidence | Verification |
|---|---|---|
| Three of three frozen seeds completed with zero failures | `results/development-v2/manifest.json` fields `completed_runs`, `expected_runs`, `failures`, and `status` | `results/result_verification.json` field `manifest_pass` |
| Base query NMSE is approximately 0.487685 in all seeds | `results/development-v2/seed-*/result.json` field `summaries.no_memory.query_nmse` | Verifier checks `metrics:<seed>:no_memory` |
| Tensor-memory query NMSE is 0 in all seeds | Seed result field `summaries.tensor_memory.query_nmse` | Verifier checks `metrics:<seed>:tensor_memory` |
| Erasure returns query NMSE to the base value | Seed result fields `summaries.erase.query_nmse` and `summaries.no_memory.query_nmse` | Decision field `erasure_gain_removal`; verifier decision checks |
| Sign flip produces query NMSE approximately 1.950739 | Seed result field `summaries.sign_flip.query_nmse` | Verifier metric checks |
| Broadcast, recent-only, and last-value controls do not reproduce the primary gain | Corresponding `summaries` fields in all three seed results | Decision gates `vector_not_equivalent` and `all_nonoracle_control_gains`; verifier metric checks |
| Non-query NMSE is unchanged | Seed result fields `summaries.tensor_memory.nonquery_nmse` and `summaries.no_memory.nonquery_nmse` | Decision field `nonquery_nmse_degradation` equals 0; verifier decision checks |
| Retrieval precision is 1.0 | Seed result field `summaries.tensor_memory.retrieval_anchor_precision` | Decision gate `retrieval_precision`; verifier metric checks |
| Paired swaps move every query toward the donor | Seed result field `summaries.paired_swap.swap_donor_direction` equals 1.0 | Decision gate `swap_direction`; verifier metric checks |
| All five pre-fit identifiability gates pass | Seed result field `identifiability.gates` | Verifier `identifiability:<seed>` checks |
| 96 of 96 checks pass | `results/result_verification.json` fields `checks` and `status` | File hash above and completed verification pipeline receipt |
| Public-data, SOTA, independent-reproduction, and release claims remain unavailable | `RESEARCH_STATE.json`, `SOTA_CAMPAIGN_STATE.json`, and `IMPLEMENTATION_STATUS_2026-10-11.json` | No public run manifest or independent reproduction receipt exists |

## Rounding policy

The table prints NMSE to six decimal places and binary diagnostic rates to
three. No significance, confidence interval, or natural-domain population claim
is inferred from this rounding. Source JSON retains full precision.

## Exclusions

- No partial or historical predecessor metric is used to support the synthetic
  v2 result.
- No public LANO, POBench-PDE, HS-FNO, or external-baseline metric is presented
  as a local result.
- No static check, dry-run manifest, source presence, or wrapper availability is
  treated as runtime or scientific evidence.
- No arXiv upload, publication, public release, or independent reproduction is
  claimed.
