# Delayed-recall episode bank v1 — implementation contract

Recorded before implementation on 2026-10-10. This is one newly authorized,
bounded engineering continuation of the completed fractional-history repair.
The previous repair, tests, negative research results, and frozen protocols remain
retained. No scientific checkpoint advances because of this contract.

## Concrete gap

`build_loader_static` in `fim_experiments/main.py` generates noisy trajectories
inside each invocation and uses the ambient RNG for its split. The successor-v2
protocol separately requires trajectory bytes to be generated and hashed before
any variant executes. Its current verifier checks that requirement is declared;
there is no retained episode-bank producer/consumer implementing it.

Implement a prospective **DEVELOPMENT-only** data path in new files. Do not alter
the existing benchmark, runtime benchmark, trainer, evaluator, frozen protocols,
analysis scripts, checkpoints, results, or open PR implementations. This path
does not authorize or replace the held successor or trajectory-isolated matrix.

## Exact mechanism and format

1. Generate delayed-recall observations on CPU from explicit float32 initial
   memory draws `N(0, I)` and a zero clock, using a dedicated `torch.Generator`.
   Reuse the maintained native `DelayedRecallBenchmark.rollout` for all transition
   and observation behavior. Do not mutate process-global RNG or default dtype.
2. Store one contiguous little-endian float32 observation array, shape
   `(episodes, horizon + 1, memory_dim + 1)`, as `observations.f32`. The last channel
   is the benchmark clock. Inputs and targets are adjacent temporal slices of
   this one array, preserving exact overlap without duplicate storage.
3. Store `manifest.json` with exact shape/dtype/byte length/data SHA-256, per-episode
   content identities, explicit nonoverlapping train/validation/development-eval
   indices, benchmark config, seed, runtime versions, source-file hashes, and a
   bank identity over canonical JSON. All splits are development data.
4. Require a caller-pinned expected bank SHA-256 before loading. Validate the exact
   schema, file identities, finite values, chronological clock, split partition,
   episode identities, and source hashes before exposing tensors. A self-consistent
   replacement still fails the externally pinned identity check.
5. Provide a real `torch.utils.data.Dataset` and DataLoader adapter with fixed
   batch size one, zero workers, explicit local shuffle seed, and fresh input/
   target clones per read. A consumer cannot mutate another consumer's evidence
   through returned tensor aliases. The caller still owns model-memory reset.
6. Reserve a fresh destination atomically. Publish the manifest only after the data
   are written and synchronized. Never overwrite a pre-existing destination.
   Retain failed/incomplete creation directories and a failure receipt when the
   filesystem permits; loaders reject incomplete/failed artifacts. Do not claim
   rollback when a final synchronization error follows installation.
7. Provide create/verify/demo CLI commands. The bounded demo uses a development
   seed and small real generator configuration, reloads two independent consumers
   around unrelated RNG activity, verifies identical ordered bytes and isolated
   tensors, and retains its artifact/receipt. It trains no model and computes no
   scientific model-performance claim.

## Bounds and scientific boundary

The producer admits only DEVELOPMENT, refuses the ten held v1/v2 scientific
seeds, and bounds a bank to 128 episodes, 64 transitions, 64 memory channels, and
1,000,000 stored float32 values. It admits no protected-data or protocol option.
The retained demo uses five episodes, four memory channels, and 12 transitions.
Source generation uses no network, accelerator, checkpoint, or learned model.

Future scientific adoption requires its own reviewed data/protocol integration
before outcome access. The bank guarantees retained byte identity and split
membership; it does not prove statistical independence, reset model memory, make
the full matrix executable, or establish memory benefit.

## Validation and stopping budget

One implementation workstream; one retained end-to-end development demonstration;
at most four focused test commands and 180 cumulative reported seconds for those
commands; at most two bounded source-review repair cycles. Artificial unit-test
fixtures stay within the producer's bounds. New model-training runs, protected
outcome runs, scientific matrix dispatches, and paid compute are all zero.

Required checks: exact native/frozen-reference data agreement at float32;
RNG/default-dtype isolation; repeated bank identity; temporal shift and clock
correctness; split completeness/disjointness; pinned-identity mismatch; changed,
missing, malformed or nonfinite bytes; clone isolation; local deterministic
shuffling; loader batch-size isolation; overwrite/collision refusal; retained
write failures; bounded real CLI demonstration. Record actual failures and costs.

The pass stops after independent review and publication of the bounded verified
artifact path. It does not continue until a model wins.
