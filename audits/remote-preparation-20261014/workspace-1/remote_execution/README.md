# Remote execution preparation

This directory adds orchestration around the unchanged v2/v3 studies. It does
not revise their algorithms, seeds, selection rules, budgets or guards. No
remote job is launched by these files. Do not place the workflow template under
`.github/workflows` until the destination and artifact storage are approved.

## Verified available route, 14 October 2026

- `THE-BU1LD/Fabric-Induced-Memory` is public, writable by the connected account,
  with Actions enabled and no registered self-hosted runners. Its inspected
  main SHA is `84160703cc733c6deb150cf7b6ee5bb3a84d699b`, with no `successor/` files.
- `build-the-future-11/Fabric-Induced-Memory` (local origin) is private and has
  no registered self-hosted runners. Private runner minutes are not assumed free.
- Standard public GitHub macOS runner time is free according to
  https://docs.github.com/en/actions/reference/runners/github-hosted-runners .
  Artifact storage is separate: review allowance/billing before upload; this
  preparation does not assert a zero-cost artifact-storage entitlement.
- The prepared target is standard `macos-15` ARM64, Python 3.11.15. PyPI metadata
  confirms a macOS ARM64 CPython 3.11 wheel for Torch 2.12.1 and wheels for the
  other exact direct replay requirements. Metadata availability is not a tested
  installation; transitive versions are recorded by `pip freeze --all` remotely.
- Google Cloud CLI is configured, but instance inventory failed with expired
  credentials (`invalid_grant`). No instance was started, changed or billed by
  this task. An SSH alias exists for a GCP runner; it does not establish a live
  compatible Mac. The unchanged guards reject Linux.

## Preparation and checks

From the repository root, use Python 3.11 or later for standard-library tooling:

```sh
rtk proxy python3 -B -m unittest discover -s remote_execution -p 'test_remote.py' -v
rtk proxy python3 -B remote_execution/prepare.py --output audits/remote-preparation-20261014/UNUSED-workspace
```

The allowlist copies exact v2 preservation-baseline inputs, recovered executed
source, paper verification inputs/tools, and the exact v3 development-2 source
and freeze. It excludes the old failed runs, secrets, unrelated projects and
public-benchmark stack. `REMOTE_MANIFEST.json` binds every packaged file; record
its SHA-256 outside the package before deployment. Package creation refuses an
existing output path, a consumed v3 campaign, or an existing v3 runner lock.

## Deployment gate and one-campaign ownership

Publishing the package to the public repository discloses presently local
synthetic source and results. Obtain the user's destination/publication choice
first. Prepare a dedicated branch in an isolated deployment checkout; do not
stage this dirty repository wholesale, replace main, or enable its unrelated
workflows. Copy `guarded-macos.yml` to `.github/workflows/itfm-frozen-remote.yml`
in that reviewed deployment. Record the exact commit and manifest digest.

Before dispatch, verify the local v3 campaign is still unconsumed, ensure no
other operator/run is active, and exclusively reserve this study for that one
deployment in a retained local receipt. Never run a parallel local campaign.
No reservation or campaign consumption has occurred during this preparation.
The template requires public visibility, storage approval, an exact commit,
global workflow serialization and first-dispatch-only validation. It refuses
GitHub reruns and any earlier dispatch under the same workflow ID, even if the
earlier dispatch failed before numerical work. Do not rename the workflow or
deploy another copy to bypass this check. Failed attempts require adjudication.

## Execution order and retained evidence

1. Verify exact deployment SHA, manifest hashes, macOS ARM64/Python identity,
   scratch space, source/scalar checks and the original two-sample memory guard.
2. Create a fresh environment; install exact direct requirements only after
   admission; record dependency checks and resolved versions. No substitutions.
3. Run the original isolated v2 wrapper with explicit interpreter. Require both
   `PASS_LOCAL_REPLAY` and raw-output PASS, all raw predicates and file hashes.
4. Run v3 numerical tests against unchanged development-2; require their exact
   source-bound receipt before the one five-seed/six-condition campaign.
5. Execute the unchanged v3 raw verifier. Keep all partial fits and failures.
6. Always collect outputs with hashes and one-day remote retention. Download
   immediately, verify transfer hashes, archive locally, and reconcile the
   v3 campaign-consumption marker before considering any future execution.

The original wrappers enforce their 300-second v2, 120-second v3 test and
1800-second v3 stage budgets, one thread and 512 MiB/stable-pageout admission.
Outer timeouts only allow wrapper cleanup; they do not extend worker budgets.
The output collector's 3 GiB ceiling covers v2 plus v3 and metadata; it does not
change the v3 2 GiB output budget. Unexpected/oversize files fail collection and
must be recovered/adjudicated, not silently discarded.

After artifact download into a fresh directory:

```sh
rtk proxy python3 -B remote_execution/collect.py verify \
  --output /absolute/path/to/downloaded-artifact --manifest-sha256 REVIEWED_DIGEST
```

Transfer hashes establish integrity, not scientific validity. Inspect every
stage status and numerical verification receipt. Remote execution by this
operator is not independent replication. A completed campaign may be negative;
report all cells and the original decision rule without retuning. Publication,
natural-domain efficacy and public-benchmark SOTA remain separate gates.
