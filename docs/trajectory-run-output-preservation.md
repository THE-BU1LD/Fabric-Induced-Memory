# Trajectory-run output preservation

The trajectory-isolated runner now requires a **fresh results directory and a
fresh manifest path** before installing execution hooks or attempting a cell.
An existing directory is rejected even if it is empty. Existing files,
directories and dangling symlinks at either destination are rejected.

This fixes an execution defect: a second invocation previously emptied the
prior manifest before training, reused its experiment destinations, and wrote
through a fixed `manifest.json.tmp` path. If that temporary path was a symlink,
the old writer could truncate an unrelated file.

## Publication behavior

- Initial manifest publication creates a complete staged JSON file and uses an
  exclusive sibling hard link. A destination created after preflight is retained
  and the new invocation fails.
- Subsequent snapshots use unique temporary siblings, flush and fsync the
  staged file, and replace the existing regular manifest atomically. An ordinary
  serialization, flush or replacement error preserves the preceding snapshot.
- Successful and failed cells continue to be written in their original order.
  All manifest fields, frozen protocol settings, and scientific computations are
  unchanged.
- A manifest outside the results directory remains supported. If custom
  destinations are needed for a separately authorized execution, set both
  `--results-root` and `--manifest`; the CLI defaults are unchanged.

The guarantee assumes one writer after the initial reservation. It does not
provide a cross-file transaction, resume a partial study, promise crash/power-loss
durability, or authorize scientific execution. A failure during initial setup
can retain an empty reserved directory; do not reuse that directory silently.
Manifest creation requires hard-link support in its destination filesystem.

## Verification and research boundary

`python -m pytest -q tests/test_trajectory_output_preservation.py` exercises real
temporary filesystem operations and the actual runner control flow with
artificial backend doubles. It covers retained files, directories and symlinks,
competing initial writers, stale temporary paths, injected publication failures,
and preservation of a failed cell before the next artificial cell.

These are engineering checks. They do not execute or read outcomes from the
frozen 40-cell study, clear its execution hold, or establish a memory benefit.
