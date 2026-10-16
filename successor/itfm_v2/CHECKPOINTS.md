# ITFM v2 checkpoints

1. **Question and falsifiers — PASS.** Complete-context ambiguity, old-memory
   information, and intervention-localized use are explicit.
2. **Development protocol — PASS.** Dataset, controls, thresholds, seeds, and
   failure policy are machine-readable and frozen before outcomes.
3. **Implementation — PASS.** Generator, memory, interventions, evaluation,
   adjudication, artifact hashing, verifier, and guarded runner exist.
4. **Static audit — PASS.** Formatting, Ruff, bytecode compilation, JSON parsing,
   and all 10 contract checks pass.
5. **Runtime tests — PASS.** Ten tests passed inside the exact guarded pipeline
   under Python 3.11 on 2026-10-11.
6. **Identifiability execution — PASS.** All declared identifiability gates passed
   on all three frozen seeds.
7. **Three-seed development — PASS BOUNDED DEVELOPMENT.** Seeds 101, 211, and
   307 passed every predictive and intervention gate.
8. **Independent artifact verification — PASS.** All 96 decision, metric, hash,
   identifiability, and prediction-artifact checks passed.
9. **Public-domain confirmation — DESIGNED, NOT EXECUTED.** The frozen SOTA
   campaign is defined in `protocols/sota_confirmation_v1.json`; upstream data,
   reproduction, and accelerator admission remain pending.
10. **Research complete — NOT REACHED.** Requires protected confirmation and
    independent reproduction.

## Manuscript and release gate update

An evidence-bound manuscript source, bibliography, results table, literature
audit, claim-to-artifact map, and fail-closed release verifier were added on
2026-10-11. This does not advance Checkpoint 9 or 10. Two resource samples failed
the 512 MiB free-plus-speculative floor and pageouts increased, so no paper
compile, rendered inspection, public experiment, or release packaging was run.
The manuscript is an internal review artifact until those gates and the
authorship/license decision are closed.

