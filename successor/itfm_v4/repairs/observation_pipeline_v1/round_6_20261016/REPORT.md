# ITFM round 6 — runtime verification prepared

Implemented an exact-source tensor-test supervisor and four actual tensor test
cases for durable prediction replay, observation-to-bank agreement, an empty
unrelated owner, and batch permutation. The original six tensor checks remain
unchanged. All ten are software acceptance tests, not scientific experiments.

The previous launcher accepted unittest success without requiring any discovered
tests. The replacement checks the complete test identity list, refuses skipped
and expected-failure tests, binds every selected source to an external manifest
hash, rechecks source after execution, and supervises a separate worker with a
30-second total deadline. Timeout preserves logs and a stale ownership lock for
auditing. Resource admission is repeated immediately before tensor import.

The combined standard-library suite passes 81/81 tests: the existing 69 plus
12 supervisor tests. Ruff lint/format, Python parsing and packaged workflow YAML
parsing pass. The full 21-file package manifest was verified without torch.

All 440 protected predecessor files, 12 frozen v3 sources, 44 original repair
files and 35 prior-extension manifest entries remain byte-identical. No historical
result, failure, checkpoint or budget marker was changed.

## Prepared remote action

- Repository: THE-BU1LD/Fabric-Induced-Memory (verified public and writable).
- New dedicated branch: codex/itfm-v4-runtime-20261016; main is unchanged.
- Exact upload: 21 allowlisted files, 173274 bytes, listed with hashes in
  DEPLOYMENT_REVIEW.json; the complete reviewable tree is remote_bundle/.
- Manifest SHA-256: d7ca447f6ec0204fbca892835b9186ad391064cb87d85ea56bd1b6abb9ab5597.
- One standard macos-26 Actions job; 15-minute outer setup limit and a 30-second
  tensor worker. The workflow rejects reruns and earlier runs under its identity.
- Python 3.11.15; torch 2.12.1 and numpy 2.4.6. Resolved transitive dependencies
  and actual runtime versions are retained. This environment is not yet tested.
- Receipts and failures are encoded with hashes in downloadable job logs.
  No Actions artifact/cache storage, paid runner or release upload is used.

The bundle includes software source and deterministic fixtures only. It excludes
research datasets/results, private project state, attached user instructions,
credentials and the rest of the dirty worktree. Checkout and Python setup actions
are pinned to exact commits. SOURCE_MANIFEST.json covers the selected source and
requirements; the deployment commit separately binds the workflow.

GitHub documents standard public-runner time as free:
https://docs.github.com/en/actions/reference/runners/github-hosted-runners .
The repository visibility and action SHA were rechecked during this round.

Publication of this newly prepared v4 source needs the user's approval. The
retained remote_execution/README.md records the earlier disclosure authorization
for the v2/v3 package; it does not establish authorization for this new v4 bundle.
No upload or remote job has occurred.

## Remaining research status

Local preflight again fails the 512 MiB/stable-pressure admission rule; the exact
packaged receipt is PACKAGED_PREFLIGHT.json. Tensor tests remain unexecuted. The
repair budget is now 6/6; concept reviews remain 8/8. No further implementation
or concept search is silently authorized by this preparation. Scientific trials
remain zero and protected outcomes remain unopened.

Checkpoint 1 still lacks a defensible contribution. Data and baseline admission,
full model/bank/optimizer recovery, measured full-shape cost, confirmation,
independent verification and the complete paper remain open. The 35-item
inventory from extension_20261015/ALL_ITEMS.json remains the complete scope map;
this round strengthens test acceptance and prepares the missing runtime evidence,
without asserting that the scientific or recovery requirements are now closed.
