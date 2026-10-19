# FIM one-shot admission preflight candidate

This separate candidate adds a deterministic, standard-library-only budget-plan validator for the stated shared ONCE envelope:

- 3,600 GPU-seconds
- 3,600 wall-clock seconds
- 7,200 aggregate CPU-seconds
- 8 GiB RAM
- 1 GiB cumulative output

It requires explicit entries for setup, tests, experiment, verification, and export, plus mandatory prior-consumption counters for GPU seconds, wall seconds, aggregate CPU seconds, and output bytes. Those cumulative counters are added to the new plan before checking finite caps, so a plan cannot appear to refill an already-spent allowance. RAM is a peak, not historical cumulative usage: planned phase peaks are summed conservatively when overlap has not been ruled out, then compared to 8 GiB.

Plan costs and prior-consumption counters are caller-declared, unverified inputs; the report labels them `SELF_DECLARED_UNVERIFIED`. This arithmetic does not prove actual usage, and a counter must not be treated as a verified accounting record. The validator never grants launch permission, even when a plan fits: its output is `WITHIN_DECLARED_BUDGET` plus `launch_admissible: false`, because no external enforcer or trustworthy accounting source is wired or verifiable here. It rejects free-form enforcement booleans and receipts rather than trusting user-supplied claims. In particular, it does not claim that process-local rlimits, cooperative deadlines, GPU inventory, post-write size checks, or this validator enforce aggregate GPU/CPU/RAM/output limits.

Known gap: before any real run, a platform-level mechanism must be identified and verified that applies the one-use GPU-hour lease, whole-session wall stop, aggregate CPU accounting/termination, memory cgroup ceiling, and pre-write output quota across setup, tests, experiment, verification, and export. This candidate cannot establish those controls, cannot reserve or observe the shared allowance, and contains no launcher.

`example_plan.json` is illustrative only; its self-declared counters are zero, its declared plan fits the arithmetic caps, and it remains blocked from launch by every enforcement gap above. Zero here is sample input, not evidence that no resources have already been consumed.

Run only the deterministic engineering tests with:

```sh
python -m unittest -v test_validator.py
```

No Kaggle access, notebook execution, ML import, or scientific workload is used by these tests.
