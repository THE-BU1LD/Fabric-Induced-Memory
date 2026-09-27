# FIM ablation semantics — `no_memory` vs `no_retrieval`

Status date: 2026-09-27  
Scope: static current-code interpretation only. No scientific outcome is rerun or reclassified by this document.

## Finding

The current maintained ablation switches are mechanically distinct:

| Variant | memory writes | retrieval | salience-gated storage |
|---|---:|---:|---:|
| `no_memory` | off | off | off |
| `no_retrieval` | on | off | on |

However, they are **not distinct prediction mechanisms** in the current architecture.

In `AblatedFIMSystem.step`:

1. the prediction latent is produced by encoder -> dynamics;
2. stored memory can affect that latent only through the retrieval-feedback block;
3. `no_retrieval` disables that retrieval-feedback block;
4. memory writes occur later under `torch.no_grad()`, after the prediction has been decoded;
5. therefore stored traces in `no_retrieval` cannot influence the current or later prediction while retrieval remains disabled.

Consequently, with identical trainable weights, inputs, RNG/data order, and optimizer history, `no_memory` and `no_retrieval` are expected to have the same prediction/loss path even though their memory-bank state differs.

A regression test now pins both facts simultaneously:
- `no_retrieval` populates the memory bank while `no_memory` does not;
- their predictions remain exactly identical for matched weights and a matched episode while retrieval is disabled.

## Scientific interpretation

Identical retained outputs for these two arms are therefore consistent with the current implementation semantics; they are not, by themselves, evidence of an execution failure.

But the two arms must **not** be presented as two independent predictive ablations:

- `full` vs `no_memory` tests the combined presence of storage + retrieval feedback.
- `full` vs `no_retrieval` also removes retrieval feedback; because storage-only state has no predictive path, it is behaviorally redundant for prediction under the current architecture.
- `no_retrieval` can still serve as a bookkeeping/state diagnostic showing that trace storage happens without feedback, but it does not isolate a separate predictive mechanism.

The frozen trajectory-isolated outcomes remain unchanged. This clarification narrows their interpretation; it does not authorize rescue tuning, a new retrieval path, or retroactive protocol changes.

## Claim boundary for paper/research truth

Safe:
- "The storage-only `no_retrieval` arm maintained a memory bank but, by construction, had no path for stored traces to affect predictions."
- "Its predictive equivalence with `no_memory` is expected under the current implementation."
- "The retained evidence does not support treating `no_memory` and `no_retrieval` as independent predictive controls."

Unsafe without a separately frozen successor:
- claiming two independent ablation replications from the identical arms;
- modifying `no_retrieval` after outcome access so storage influences predictions through a new path;
- rerunning until the two controls diverge.

Any successor that wants a distinct "write but do not retrieve" scientific control must define a mechanism where stored state has an independently specified causal role, freeze that mechanism before outcome access, and version it as a new protocol rather than rewriting the retained FIM result.
