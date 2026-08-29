# Fabric-Induced Memory — Evidence Ledger

This ledger maps paper-facing claims to the checked-in evidence that currently supports or blocks them.

| Claim / question | Current evidence | Status | Next action |
|---|---|---|---|
| The repository implements a memory bank and retrieval path | `fim_experiments/systems.py` defines `AdaptiveMemoryBank` and `FIMSystem.step(...)` with trace storage and retrieval | Supported as an implementation claim | Keep covered by tests |
| The project has a reproducible experiment entry point | `main.py` delegates to `fim_experiments.main`; configuration is resolved and written with experiment outputs | Partially supported | Add a documented clean-environment smoke run and CI result |
| Current paper-reference metrics exist | `paper_reference_results.json` records compact results attributed to `paper/Final FIM.pdf` | Supported as a transcription/artifact claim | Link each paper table/figure to generated raw outputs |
| Memory improves Lorenz performance | Recorded full = `1.317169`; no_memory = `1.317169` | Not supported | Execute repaired ablations across predeclared seeds and inspect outputs |
| Retrieval improves Lorenz performance | Recorded full = `1.317169`; no_retrieval = `1.317169` | Not supported | Execute repaired retrieval-disable control and inspect outputs |
| Spectral mixing is necessary for the reported Lorenz result | no_spectral_mixing = `0.869918`, better than recorded full = `1.317169` if lower is better | Contradicted by current compact record | Verify metric direction and rerun from raw code |
| A current-code mechanism ablation runner exists | `fim_experiments/ablation.py` defines explicit `full`, `no_memory`, and `no_retrieval` semantics; `scripts/run_ablation.sh` invokes it | Implemented, not yet validated by a clean retained run | Run from a clean environment and retain logs/artifacts |
| Across-seed descriptive aggregation exists | `fim_experiments/ablation_report.py` reads completed summaries and emits mean/std/min/max without significance claims | Implemented, not evidence by itself | Generate reports only after successful retained runs |
| FIM beats matched non-memory baselines | No claim-to-artifact table is checked in at repository root | Unverified | Produce matched-capacity/compute baseline table with seeds and uncertainty |
| Results are robust across seeds | Compact paper-reference JSON contains no per-seed breakdown | Unverified | Use repaired runner with predeclared seeds and retain all per-seed outputs |
| Results are tied to a precise code/environment revision | Repaired runner writes per-run provenance including git SHA, Python/platform, Torch version, config, and control semantics | Infrastructure implemented; historical paper numbers remain unverified | Complete runs and bind generated tables to the retained provenance |
| The paper can safely claim a mechanism-level memory benefit | Current ablation record does not isolate a positive memory/retrieval contribution | Blocked | Keep claim conservative until repaired ablations and strong baselines support it |

## Publication gate

A mechanism claim should remain blocked until all of the following are true:

- [ ] Current ablation runner executes successfully from a clean environment.
- [x] Full / no-memory / no-retrieval variants are explicit in the current code path.
- [x] The default ablation command predeclares identical seeds across variants.
- [x] The runner is configured to retain resolved configs, per-run summaries, and provenance.
- [ ] Raw completed outputs have been inspected for all predeclared runs.
- [ ] Paper tables are generated from raw outputs, not manually transcribed values.
- [ ] At least one strong non-memory baseline is matched for parameter count or compute.
- [ ] Appropriate uncertainty/statistical comparisons are reported for headline comparisons.
- [ ] Current exact-equality ablation anomaly is explained or superseded by reproducible results.

Checked boxes above refer only to implemented infrastructure, not successful experiment outcomes.
