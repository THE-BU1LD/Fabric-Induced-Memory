# ITFM v4 software verification bundle

This isolated branch publishes observation recording, durable journals, raw-export validation, and their tensor adapter integration. It does not contain a scientific campaign or a validated forecasting improvement.

The workflow executes 87 scalar tests and 10 tensor integration tests on one standard public macOS 26 runner. Each test invocation has a 30-second budget; the job has a 15-minute limit. Direct dependencies and Actions are pinned. Before heavy work, two resource samples and an immediate recheck must clear 512 MiB free-plus-speculative memory, stable pageouts/swap, and 1 GiB disk. Disposable runner caches are reclaimed once at each setup/install/runtime boundary; admission thresholds remain unchanged.

The source manifest freezes both test inventories and all runtime source files. The supervisor rejects zero tests, missing tests, skips, expected failures, contradictory receipts and source changes. It kills only its owned worker process group on timeout or interruption. Receipts include explicit RSS units and runtime identity. Scalar acceptance uses the same exact-inventory rule.

Results and dependency versions are preserved in downloadable job logs as SHA-256-bound base64 records. No Actions artifact or cache storage is requested. The first job stopped during Python setup before all tests because Python 3.11.15 had no macOS ARM build. This corrected bundle pins the catalogued Python 3.11.9 build. The workflow allows exactly that retained setup failure plus this one test execution; it rejects reruns and any other prior attempt.

Earlier local bundles and research evidence remain preserved separately. A green job establishes software test success only: novelty, model efficacy, independent reproduction and research completion remain unestablished.
