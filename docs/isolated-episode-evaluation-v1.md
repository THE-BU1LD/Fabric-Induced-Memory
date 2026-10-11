# Isolated public-model episode evaluation v1

## Prospective engineering contract

This new adapter is for a validation interlude around the public `FIMModel` and
`Trainer`. The existing `Trainer.eval_step` intentionally continues the current
episode: `model.step` can update its recurrent cache, step counter and trace banks.
Its temporary EMA scope restores parameters, not those episode artifacts. Calling
it on an unrelated validation episode can therefore affect subsequent training.

The new `evaluate_isolated_episode_v1` runs one separate, explicitly observed-input
episode on an owned copy. The caller must choose raw or EMA weights. Input and
target sequences have shape `[1, T, C, H, W]`; every step receives its supplied
observation, so these are one-step validation losses, not autoregressive rollout
metrics. The copy starts with empty memory and state and retains its memory within
the episode. Returned values are the existing per-step `TrainStats` in time order.
There is no new aggregate metric or model-selection rule.

The copy avoids touching the original parameters, gradients, recurrent graph,
registered and nonpersistent buffers, modes, optimizer, scheduler and EMA. A live
nonleaf `_state` is deliberately omitted from copying, because the evaluated
episode begins empty. CPU Torch RNG is forked and restored on success or failure.
The adapter is explicitly CPU-only in this version and uses extra memory for one
model copy. It requires the concrete maintained `Trainer`/`FIMModel`, float32 or
float64 data matching its parameters, and no module execution hooks. The caller
must own the model exclusively during evaluation; arbitrary concurrent mutation
or external hook side effects are outside this API.

## Session boundary

Base: `92bcc3eb5322d967684673ca61f22a8bbebdc16b` (draft integration PR #38).
This is a new bounded engineering session under the user's request to find and
execute outstanding work. Earlier integration budgets are closed and preserved.
The local budget is one artificial reproducer and at most three focused test
commands, with one numeric thread and no more than 180 seconds per command. One
repair cycle is allowed if a concrete implementation failure is found. There
are no trained benchmark fits, retained development matrix reruns, protected
outcomes, paid compute, scientific workflow dispatches, merges or submissions.

The frozen 40-cell runner uses a different maintained experiment model. This
adapter neither changes that runner nor replaces a frozen protocol. Existing
adverse FIM results and protected execution/outcome holds remain authoritative.
Future scientific use requires prospective declaration of this API and its
observed-input, weight-selection and episode-isolation semantics.

## Use

```python
from fim.training.isolated_evaluation_v1 import evaluate_isolated_episode_v1

# Both tensors describe exactly one independent episode: [1, T, C, H, W].
per_step = evaluate_isolated_episode_v1(
    trainer, validation_inputs, validation_targets, weights="ema"
)
```

Call separately for each independent episode. A request for EMA weights fails
when the trainer has no EMA object. Validation and all evaluation steps must
complete before any result is returned.
