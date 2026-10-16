# ITFM v2 literature and novelty audit

Date checked: 2026-10-11

This is a targeted prior-art audit, not a PRISMA systematic review. It covers
the closest mechanisms and evaluation settings needed to bound the current
claim. It does not establish that no uncatalogued paper overlaps the idea.

## Research question

Can an owner-safe, strictly past external tensor memory be shown to reuse a
specific spatial innovation that is unavailable in the visible context, with
forecast dependence localized through value-only interventions?

## Search and screening

The audit searched arXiv and linked primary paper/repository pages for:

- partial-observation neural operators and POBench-PDE;
- memory neural operators and Mori-Zwanzig learning;
- history-space or state-space neural operators;
- prediction/correction and data-assimilation neural operators;
- retrieval-augmented forecasting; and
- probabilistic operator learning under partial observations.

Works were retained when they materially overlapped the information problem,
model family, memory mechanism, public benchmark, or proposed claim. Broad PDE
foundation models were not treated as direct scientific comparators unless they
addressed partial observation or non-Markovian history.

## Closest work

| Work | What it already establishes | Consequence for ITFM |
|---|---|---|
| Li et al., [Fourier Neural Operator](https://arxiv.org/abs/2010.08895) | Fourier-domain operator learning for PDE families | A generic neural-operator contribution is not novel. |
| Lin et al., [Data-driven Mori-Zwanzig learning](https://arxiv.org/abs/2101.05873) | Learned separated Markov and memory operators from reduced observables | Learned memory for unresolved dynamics is established prior art. |
| Buitrago Ruiz et al., [MemNO](https://arxiv.org/abs/2409.02313) | State-space memory inside a neural operator; theory and experiments under low resolution/noise | A broad claim that memory helps PDE forecasting directly collides. |
| Singh et al., [NODA](https://arxiv.org/abs/2402.15656) | Recursive prediction plus observer correction from sparse noisy measurements | Prediction/correction under partial observation is not unique. |
| Koren and Lanthaler, [ST-SSM neural operator](https://arxiv.org/abs/2507.23428) | Factorized spatial/temporal SSM operator, universality, partial-observation evidence | State-space memory and long-range temporal modeling are occupied. |
| Han et al., [RAFT](https://arxiv.org/abs/2505.04163) | Retrieval of similar historical continuations for forecasting | Retrieval augmentation itself is not novel. |
| Hou et al., [LANO and POBench-PDE](https://arxiv.org/abs/2601.15547) | Partial-observation benchmark plus mask-to-predict and latent autoregressive propagation | LANO is the primary public frontier; exact reproduction is mandatory before candidate comparison. |
| Shikhman, [HS-FNO](https://arxiv.org/abs/2605.09523) | History-space neural operator with exact shift-append dynamics over five delay/memory PDE families | HS-FNO is the secondary non-Markovian frontier; simple FNO controls are insufficient. |
| Lara-Rangel and Guillas, [Neural Operator Processes](https://arxiv.org/abs/2606.22946) | Probabilistic full-field prediction from sparse or partial observations | Partial-observation uncertainty and context-query geometry are occupied. |

## Novelty boundary

The following are explicitly not claimed as novel:

- external memory;
- residual correction;
- partial-observation forecasting;
- non-Markovian neural operators;
- state-space or recurrent history models;
- retrieval-augmented forecasting;
- cosine nearest-neighbor lookup;
- validation-calibrated additive correction; or
- observation-space values in isolation.

The defensible contribution is the conjunction of:

1. exact full-context aliases with opposite futures;
2. a past-only, same-owner, inspectable spatial residual value;
3. a value that becomes available only after its target is observed;
4. inference-time erasure, sign, spatial, recency, nuisance, and paired-swap
   interventions on a fixed predictor; and
5. a joint decision rule that requires prediction, localization, integrity, and
   no-harm gates simultaneously.

This is principally an identifiability and evaluation contribution. The current
evidence establishes it only as a synthetic mechanism harness.

## Adversarial flaw register

| Flaw | Cause | Severity | Repair | Evidence required |
|---|---|---:|---|---|
| The synthetic task is solvable by exact lookup | Anchors repeat exactly and residuals repeat deterministically | Critical for external validity | Add noisy/approximate keys only in development, then freeze a public task | Public validation oracle gap and protected multiseed test |
| Perfect query error may be mistaken for model learning | The base is fixed and memory stores the exact residual | Critical | Describe the result as a mechanism unit test; never as general forecasting performance | Claim audit and reviewer-facing limitation |
| Memory may duplicate longer context | Old calibration information could be encoded by a sufficiently long sequence model | Critical | Match finite-history, recurrent, transformer, SSM, and history-space controls | Information- and compute-matched public comparison |
| Extra capacity or compute may explain a public gain | Adapter and bank add resources | Critical | Parameter-matched residual MLP, equal update/trial budgets, separate bank-byte accounting | Matched receipts and resource table |
| Retrieval may leak owner or future identity | Incorrect stream ownership or write/read order | Critical | Immutable trajectory IDs, target-after-prediction writes, event-time audit | Zero cross-owner and future retrieval in saved logs |
| Protected outcomes may guide development | Repeated test inspection changes the effective hypothesis | Critical | Validation-only oracle triage and cryptographically bound freeze receipt | Protected unlock receipt before S4 |
| Public preprocessing may differ from the frontier | Generic NPZ loading does not recreate official normalization and splits | Critical | Byte-for-byte comparison with official preprocessing outputs | Upstream validation reproduction receipt |
| Frontier wrappers may be mistaken for reproduced baselines | Interface compatibility is not algorithmic fidelity | Critical | Integrate admitted locked source or produce an independent paper-faithful implementation | Exact frontier reproduction within tolerance |
| Spatial residuals may fail under changing grids | Stored tensors assume compatible coordinates | High | Predeclare interpolation or coordinate-aware transport and test it separately | Resolution/OOD grid study |
| Exact search scales linearly with bank size | Reference retrieval is $O(ND)$ | Medium | Keep exact search at science scale; admit ANN only after equivalence | Recall, latency, memory, and outcome equivalence curves |
| Online writes can induce distribution feedback | Later predictions depend on earlier model errors | High | Separate frozen-bank and online-bank conditions; log bank drift | Seed stability and error-propagation analysis |
| A scalar gate may be too rigid or too favorable | One coefficient cannot express local reliability; a learned gate adds flexibility | High | Compare scalar, monotone, and uncertainty-conditioned gates under equal tuning budgets | Development-only calibration and frozen selection |
| Missing-data masks can become shortcuts | Mask geometry may correlate with targets or dataset identity | High | Randomize and stratify masks; include mask-only controls | Worst-group and permutation results |
| Pair-swap uses oracle matching | The swap diagnostic has access the deployed method does not | Medium | Keep it diagnostic and never include it in ordinary prediction | Separate intervention artifact and clear labeling |
| Three seeds are not public-domain statistical evidence | The synthetic system is nearly deterministic after normalization | Critical for broad claims | Use the frozen 5-seed and 10-seed public protocols with trajectory-first inference | Complete public matrix and clustered uncertainty |
| Independent verification is local only | The verifier shares repository assumptions and host | High | Clean environment, fresh operator, independently fetched sources | Independent reproduction receipt |

## Checkpoint decision

- Synthetic idea and protocol: **ready and executed** within the narrow mechanism scope.
- Public ITFM-R novelty: **plausible as an intervention-identifiability study,
  not established as an architectural novelty**.
- Public campaign execution: **blocked at S0** by absent admitted data/source
  licenses, missing frontier reproduction receipts, no authorized compute
  budget, and failed host resource admission.
- Broad SOTA or publication claim: **not allowed** from current evidence.
