# ITFM-R state-of-the-art conversion strategy

Date frozen: 2026-10-11

## Direct verdict

ITFM v2 is not currently state of the art. It is a verified synthetic mechanism
result. The shortest defensible route to SOTA is a backbone-agnostic residual
memory adapter evaluated on public tasks where incomplete observations or
non-Markovian dynamics create an information deficit that older observations
can actually close.

The target is a material win, not a leaderboard rounding error. The primary
gate requires at least 10% macro relative-error reduction over a locally
reproduced LANO frontier across POBench-PDE, at least 30 wins among 36 declared
cells, and no greater than 2% regression in any cell. The secondary gate
requires at least 15% lower aggregate rollout error than reproduced HS-FNO and
wins on at least four of five delay-PDE families.

## Why this wedge

POBench-PDE is the closest public partial-observation frontier. Its AAAI-26
LANO paper evaluates Navier-Stokes, Diffusion-Reaction, and ERA5 under point and
patch missingness and reports full-domain relative L2 over 36 settings. That is
the correct venue for testing whether old residual tensors add information
beyond a strong latent reconstruction model.

HS-FNO is the closest explicit non-Markovian frontier. Its suite covers five
delay or distributed-memory PDE families and uses ten seeds. ITFM-R must beat
the exact history-space inductive bias, not merely a current-frame FNO.

Large general PDE foundation models such as Poseidon and broad suites such as
The Well remain important stress tests, but they are not interchangeable with
partial-observation or delay-PDE tasks. They cannot be used to claim or deny
this narrower SOTA result without task-matched evaluation.

## Candidate mechanism

ITFM-R wraps a frozen or jointly trained strong backbone:

1. The backbone produces its ordinary forecast and a detached context key.
2. Once a target becomes observable, the observation-space residual is written
   to a same-trajectory, strictly past memory bank.
3. At a later forecast origin, top-k retrieval returns an old residual tensor.
4. A validation-only clipped gate adds the tensor to the backbone prediction.
5. Erasure, sign flip, spatial shuffle, wrong-trajectory, time-permutation,
   recent-only, and last-residual interventions test causal use and leakage.

This differs from merely enlarging the context window: it stores completed
forecast errors, enforces availability time, and exposes the retrieved value to
direct intervention. It also differs from MemNO and HS-FNO sequence state by
being an auditable episodic residual store that can augment either backbone.

## Fail-fast order

1. Reproduce upstream results at locked commits.
2. Run validation-only oracle-gap triage. Stop if old residuals cannot improve a
   strong backbone by at least 10% on enough task families.
3. Establish one-sequence memorization and eight-trajectory overfit gates.
4. Run a 64-trajectory development slice with every intervention.
5. Freeze the adapter and execute every public cell and declared seed once.
6. Repeat in a fresh environment under an independent operator.

No full sweep is admissible before the cheaper preceding gate passes.

## Frontier sources locked for reproduction

- LANO / POBench-PDE: `Kingyum-Hou/LANO` at
  `b53a62100b4705f1382a7c344b2d84afa255b96f`.
- HS-FNO: `lennonshikhman/hs-fno` at
  `9ebf2a7aa483c7f74769365c187c7d9492ecd6ef`.
- Neural-Solver-Library: `thuml/Neural-Solver-Library` at
  `db942b72c6b3972bad217300e3cbdb8004a1a80b`.
- The Well: `PolymathicAI/the_well` at
  `ad50de0e8861380d7c4d62c0503c654710f026e7`.
- Poseidon: `camlab-ethz/poseidon` at
  `b8fa28f59bd7f7673323f28d11a12c6f3a215c61`.

These locks identify source snapshots only. They do not imply successful local
reproduction, license admission, dataset availability, or endorsement of the
reported metrics.

No root license file was observed in the locked LANO or HS-FNO repositories on
2026-10-11. Their papers and public code can be studied and independently
reimplemented, but copying or redistributing their source remains blocked until
the authors or another authoritative source clarify applicable terms.

## Current execution boundary

The local host has enough memory for the lightweight v2 mechanism study but not
for a credible LANO/HS-FNO multiseed campaign. The public campaign therefore
remains blocked on dataset, license, accelerator, and cost admission. Lowering
model sizes, seeds, cells, or baselines to fit this laptop would change the
claim and is not an acceptable route to SOTA.
