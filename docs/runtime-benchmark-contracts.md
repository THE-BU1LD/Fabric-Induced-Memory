# Opt-in native benchmark runtime contracts

`fim_experiments.runtime_benchmarks` exposes the retained Levy-stable and
delayed-recall runtime successors through normal package imports. These classes
are an engineering review candidate. The experiment CLI continues to import
`fim_experiments.benchmark`; the frozen scientific source, v2 protocol, seeds,
metrics, budgets, stored outcomes and manuscript are unchanged.

The preserved benchmark has Git blob
`c91a6afe9d86830c6bcfaa2d9187547d9688050d` at main commit
`08d2c3df4262612bffe38488f97e263769ff7ef6`. The existing frozen-protocol verifier
still checks that exact source identity and still reports
`execution_authorized: false` and `outcome_access_allowed: false`. There is no
automatic CLI selection of this successor or authorization to use it in an
outcome run. Repository-owner review and any scientific re-freeze are separate
decisions.

## Runtime behavior

The Levy successor validates configuration, batch size, real floating dtype,
state shape and rollout counts before sampling. A zero-step rollout still
validates its input, preserves the initial state and consumes no random draws.
It preserves the frozen default transform under the same seed and makes alpha-2
draws exactly beta-invariant. Location remains applied to every increment.

The delayed-recall successor validates non-Boolean positive integers for its
dimensions, delay and default step count, checks finite noise parameters, and
requires a non-empty rank-2 real floating hidden tensor. It preserves the
existing smooth reveal gate, hidden-memory transition, fresh observation noise,
input/target shift and clock channel.

Exact continuation requires all three pieces of state: the returned hidden
tensor, the next integer observation time, and the caller's generator state.
`rollout_segment` validates time against the hidden clock before consuming any
random draws. Its roundoff allowance is capped below one observation interval;
a low-precision clock cannot use a loose tolerance to admit a neighboring time.
If the clock dtype represents two adjacent observation times as the same value,
the request fails explicitly. Use a sufficiently precise state for the chosen
horizon; changing the dtype after information was lost cannot recover it.

An explicit small CPU fixture can import the successor as follows:

```python
import torch
from fim_experiments.runtime_benchmarks import (
    DelayedRecallBenchmark,
    DelayedRecallConfig,
)

benchmark = DelayedRecallBenchmark(DelayedRecallConfig())
generator = torch.Generator(device="cpu").manual_seed(73)
hidden = benchmark.sample_initial_state(2, generator=generator)
first, continuation = benchmark.rollout_segment(
    hidden, start_time=0, observation_count=6, generator=generator
)
second, final = benchmark.rollout_segment(
    continuation, start_time=6, observation_count=11, generator=generator
)
```

For a serialized resume, save and restore `generator.get_state()` together with
the hidden tensor and next observation time. A newly seeded generator supplies
different subsequent noise even if the hidden state is correct.

## Maintained verification

Run the bounded repository fixtures with a CPU PyTorch installation:

```sh
python -m pytest -q tests/test_benchmark_runtime_contracts.py
python -m pytest -q tests/test_delayed_recall_successor_v2.py
python -m compileall -q fim fim_experiments scripts tests
```

The tests import the actual package modules. They need no retained ZIP, source
snapshot, credentials, network, hosted compute or scientific output artifacts.
They cover default tensor parity against the preserved frozen Levy class,
alpha-2 invariance, invalid configurations/states, zero-step behavior, exact
continuation, source/target pairing, four real floating dtypes, non-finite or
ambiguous clocks, and rejection without RNG consumption.

The dispatch-only guard on `trajectory-isolated-v1` is the same protection
already proposed in PR 20. Ordinary pull requests keep repository tests enabled
while the frozen 40-cell matrix requires an explicit workflow dispatch. This
guard is not authorization to dispatch it.

These fixtures establish implementation behavior. They establish no memory
benefit, forecasting advantage, fractional-diffusion validity, delayed-recall
improvement or publication claim.
