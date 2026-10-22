# Maintained-source integration review — 22 October 2026

The PR's causal retrieval and optimizer-recovery fixes have been reviewed against
their complete surrounding implementations and regression tests. The runtime
source identity remains the same as the retained October 19 engineering evidence.
No historical snapshot, smoke outcome, checkpoint or protocol was regenerated.

The root pytest configuration previously searched the entire repository, including
duplicate FIM/QFIM trees and frozen v3/v4 campaign/source archives. It now declares
the maintained root tests and root epistemic tests explicitly. The maintained v2
workflow still runs its complete test directory. No assertion or science hold was
removed. The 40-cell scientific job remains byte-for-byte unchanged and requires
its existing separately authorized manual dispatch.

CPU jobs preinstall the same 2.11.0 PyTorch version used by the retained v2
environment, selecting its CPU build explicitly. Both Python 3.11 and 3.12 Linux
CPU wheels were verified on the official PyTorch wheel index. Frozen dependency
locks and source archives are unchanged. One-thread settings bound routine CPU
test parallelism. Existing format, lint, type, static-protocol, license and artifact
retention steps remain present.

The 23 standard-library controls pass. The attempted local tensor test command
stopped before importing tensors: measured headroom was 166–181 MiB against the
unchanged 512 MiB floor. It executed zero tests. This blocked receipt is retained,
and no local numerical rerun is claimed. Fresh hosted CI must establish the new
integration's status; an artifact quota failure is not a successful workflow.

The existing one-seed software fixture still shows full memory losing to the
last-residual and persistence controls. It is not a new scientific contribution.
The original 40-cell hold, consumed v3 campaign, v4 limits, historical v2 evidence,
public-data prerequisites and independent-review gates remain intact.
