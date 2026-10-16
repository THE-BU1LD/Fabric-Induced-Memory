# Literature and novelty boundary

Checked 2026-10-11. These citations bound the idea; they do not validate ITFM.

## Foundational external memory

- Graves, Wayne, and Danihelka, *Neural Turing Machines* (2014),
  <https://arxiv.org/abs/1410.5401>. Differentiable attention over external
  memory; establishes that learned read/write memory is not novel.
- Graves et al., *Hybrid computing using a neural network with dynamic external
  memory* (2016), <https://doi.org/10.1038/nature20101>. Differentiable Neural
  Computer with structured memory operations.
- Wu et al., *Memorizing Transformers* (2022),
  <https://arxiv.org/abs/2203.08913>. Non-differentiable k-nearest-neighbor
  key-value memory at inference.
- Bulatov et al., *Recurrent Memory Transformer* (2022),
  <https://arxiv.org/abs/2207.06881>. Recurrent memory tokens across segments.

## Memory for dynamics and operators

- Ruiz et al., *On the Benefits of Memory for Modeling Time-Dependent PDEs*
  (2024), <https://arxiv.org/abs/2409.02313>. MemNO combines SSM memory and FNO
  and directly addresses distorted or partially observed PDE evolution.
- Lin et al., *Data-driven learning for the Mori-Zwanzig formalism* (2021),
  <https://arxiv.org/abs/2101.05873>. Separates Markov and learned memory
  operators for reduced observables.
- Singh et al., *Learning Semilinear Neural Operators* (2024),
  <https://arxiv.org/abs/2402.15656>. Prediction/correction and data assimilation
  for sparsely observed semilinear PDEs.
- Koren and Lanthaler, *Merging Memory and Space* (2025),
  <https://arxiv.org/abs/2507.23428>. Spatiotemporal state-space neural operator
  with a universality result and partial-observability experiments.

## Spatial fields and partial observations

- Malhotra and Seghouani, *Neural Field Turing Machine* (2025),
  <https://arxiv.org/abs/2509.03370>. Continuous memory field with movable local
  read/write heads. This is a direct collision with broad "memory as a field."
- Hou et al., *Learning Neural Operators from Partial Observations via Latent
  Autoregressive Modeling* (2026), <https://arxiv.org/abs/2601.15547>. Partial
  observation operator learning with latent autoregressive propagation.
- Lara-Rangel and Guillas, *Neural Operator Processes for Probabilistic Operator
  Learning under Partial Observations* (2026),
  <https://arxiv.org/abs/2606.22946>. Query-aligned conditioning and the
  importance of preserving local context-query geometry.

## Registration, transport, and benchmark boundary

- Lewis, *Fast Normalized Cross-Correlation* (1995),
  <https://www.cs.swarthmore.edu/~turnbull/cs97/f08/paper/Lewis95.pdf>.
  Normalized cross-correlation and its efficient computation are established
  image-matching tools. ITFM's exhaustive bounded periodic search is a simple
  spatial-domain use of that established score, not a novel registration
  algorithm and not Fourier phase correlation.
- Takamoto et al., *PDEBench: An Extensive Benchmark for Scientific Machine
  Learning* (2022), <https://arxiv.org/abs/2210.07182>, with official dataset
  DOI <https://doi.org/10.18419/darus-2986>. PDEBench supplies public simulated
  PDE trajectories and strong conventional baselines. Its official registry
  and license are staged for loader validation, but no PDEBench task has been
  selected as protected confirmatory evidence.

## Surviving novelty claim

No reviewed source establishes that generic tensor-field memory is new. The
surviving candidate is the combination of:

1. observation-interpretable spatial residual values;
2. enforced owner/time eligibility;
3. matched information and compute controls;
4. memory-only counterfactual interventions;
5. a joint decision rule requiring forecast benefit and intervention response.
6. explicit transport alignment with transport-only, transport-off, and
   sham-information controls that can distinguish alignment from stored value.

This remains a candidate boundary until full texts, associated code, and later
work are reviewed before any submission claim.

