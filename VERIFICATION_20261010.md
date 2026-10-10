# Analysis metric input contract — 10 October 2026

Parent: `9503099f9952ef3af6bb92f7e75d006e37ff414f` (existing FIM PR #27).
This is an engineering input-admission repair, not a scientific result.

`fim/analysis/metrics.py` previously allowed PyTorch broadcasting between a prediction
and a differently shaped target. Event metrics could also silently treat NaN comparisons
as false. Empty tensors and invalid denominator epsilons were not rejected consistently.

The five tensor metrics now require equal, nonempty, finite, real tensor inputs.
The three epsilon-using metrics require a finite positive real epsilon (not a Boolean
or string). Existing admitted formulas, threshold `> 0`, gradients and the no-events
zero convention are retained. The memory-efficiency formula is unchanged.

Verification uses only constructed tensors: **87 cases passed** after the repair;
the same tests against the authenticated parent metric blob produced **78 failures
and 9 passes**. Float32/float64 valid score and gradient equivalence is exact in the
included cases. Standalone source and test compilation passed. No full repository
pass or performance improvement is implied by this focused result. Finiteness checks
can synchronize an accelerator; do not compare runtime against old runs as equivalent
without measuring their effect under an authorized efficiency protocol.

Reproduce: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_analysis_metric_contracts.py`.
The test loads the actual module file directly, avoiding package initialization and
unrelated model imports. No benchmark, retained outcome, checkpoint or dataset is read.

No frozen protocol, retained result, model, training source, recall-specific frozen
metric source or workflow is modified. Existing scientific holds remain in force.
Normal repository tests are permitted; the scientific trajectory job remains dispatch-only.
A future authorized run must bind the actual new revision and cannot silently relabel
historical outputs. This document supplements, and does not replace, RESEARCH_TRUTH.md.
