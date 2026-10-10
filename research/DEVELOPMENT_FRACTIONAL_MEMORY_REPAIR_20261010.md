# Fractional-memory runtime repair — 10 October 2026

## Research state and scope

Base: `84160703cc733c6deb150cf7b6ee5bb3a84d699b` on the maintained
`THE-BU1LD/Fabric-Induced-Memory` lineage. This is a prospective engineering
repair to the exported `fim.core.FractionalMemory` utility. The research question,
historical adverse and mixed comparisons, and held confirmatory protocols in
`RESEARCH_TRUTH.md` remain unchanged. This does not establish a memory advantage
or close the research project. The historical `build-the-future-11/FIM` sibling
is preserved.

The immediate implementation question is whether a finite-window
Grunwald–Letnikov (GL) memory preserves the declared observation geometry and
uses exactly the latest `max_history` observations. Validation is bounded to
artificial mathematical and repository fixtures; no benchmark campaign,
protected outcome access or paid compute is involved.

## Reproduced defects

The existing truncation `history[:, :, :, :, -max_history:]` selected a hardcoded
fifth axis, while `push` inserts time at axis `-2`. Once a window filled:

- vector and batched-vector observations raised `IndexError`;
- four-dimensional fields lost spatial width while retaining too many times;
- one-entry windows failed to replace their previous observation correctly.

The first observation also used `detach()` without copying storage, so changing
the caller's tensor rewrote the retained past. Before the repair, seven of the
eight initial mathematical fixtures failed on the base source. These are
implementation failures, not new scientific outcomes.

## Implemented behavior

`push` now admits a finite real observation with stable shape, device and dtype,
retains at most `max_history - 1` earlier times using the actual time axis, and
copies the new observation into owned storage. First observations and one-entry
windows are snapshots. A rejected push leaves the preceding history intact;
`reset()` permits a new trajectory with different geometry.

The returned quantity remains

`dt**(-alpha) * sum(w[k] * x[t-k] for k in the retained window)`,

where `w[0] = 1` and `w[k] = w[k-1] * (-(alpha-k+1)/k)`. Missing prehistory is
zero. This is the existing finite-window GL definition, not a new Caputo
initial-value correction. Direct `fractional_difference` retains autograd;
`FractionalMemory` intentionally stores detached observations.

Invalid fractional orders, nonpositive/nonfinite time steps, nonintegral window
lengths, and integer/complex history dtypes are rejected rather than silently
altering the intended real fractional coefficients. Retained valid short-window
arithmetic is unchanged. Long-window outputs and caller-alias behavior are
corrected, so any future outcome run must identify its actual source version.

## Verification

Environment: Python 3.12, Torch 2.5.1+cpu, CPU, one thread for BLAS/OpenMP.

- New mathematical/shape/state/autograd tests: **39 passed**.
- Complete maintained collection: **215 passed, 7 skipped, 10 subtests passed**
  in 10.88 seconds. Seven existing optional epistemic imports remain skipped.
- Targeted Ruff checks (`E4,E7,E9,F,I`), compilation and whitespace checks pass.
- The original frozen benchmark, recall metrics, protocols, workflows, retained
  outcomes, checkpoints and manuscripts are unchanged. The existing scientific
  workflow still requires an explicit manual dispatch.

Reproduce the new mathematical checks with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q tests/test_fractional_memory.py
```

This candidate has no changed-file overlap with the existing public-Trainer
repair or evidence-admission draft. Integration review remains distinct from
authorization for any protected scientific execution.
