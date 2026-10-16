# V4 public bundle and macOS verification — 16 October 2026

Publication is authorized by the user's explicit instruction. The current published commit is `266fe56ff9897e57b7e610fb7afc369fe767855b` on `codex/itfm-v4-runtime-20261016`; the exact bundle manifest SHA-256 is `67a5253aab94d5f99d83fe3218db54670401e5fd8a502236123280b64af2849b`.

## Code quality changes

| Flaw | Cause | Repair | Evidence |
| --- | --- | --- | --- |
| Parent could accept a contradictory pass receipt | It trusted worker status without checking all outcome counts | Independent validation requires exact integer counts, test identities, zero skips/failures/errors/expected failures/unexpected successes, matching source identity and imported tensor environment | Six added regression tests, including malformed/missing/boolean outcomes |
| Interrupted supervisor could leave its worker running | Cleanup handled timeouts only | Terminate and reap only the owned process group on interruption; tolerate an exit race | Real deadline check plus interruption and exit-race regression tests |
| Scalar discovery could pass with an incomplete suite | Generic unittest success did not bind the inventory | Discover and execute exactly 87 source-manifest test IDs, with zero skips or expected failures | Final local receipt: 87/87 pass, 0.77 seconds |
| Dependency setup used an unavailable macOS ARM Python pin | Python source release availability was mistaken for Actions binary availability | Pin the verified catalog build 3.11.9; verify torch 2.12.1 and NumPy 2.4.6 CPython 3.11 ARM wheels before dispatch | PYTHON_BUILDS.json and DEPENDENCY_WHEELS.json |

The first local scalar attempt retained one failure and one error caused by subprocess startup timeouts. Isolated standard-library subprocesses (`-S`, `-B`) remove unrelated site initialization and bytecode writes. The final suite passes; the failure receipt remains. No tensor imports ran locally.

## Hosted execution

The first hosted job, 38847030101, passed the resource gate at approximately 3.4 GiB free-plus-speculative memory, stable pageouts and zero swap, then failed in Python setup. All tests were skipped. Its log and decoded resource receipt are retained. The correction changes only the workflow, public README and manifest; every Python implementation/test byte is unchanged.

The corrected [run 38847339155](https://github.com/THE-BU1LD/Fabric-Induced-Memory/actions/runs/38847339155) completed successfully against the exact published commit. All **87 scalar tests and 10 tensor tests passed**, with zero failures, errors, skips or expected failures. The scalar suite took 1.16 seconds; the complete tensor worker took 6.20 seconds and peaked at 250,380,288 bytes RSS. Runtime identity: macOS 26.6.2 ARM64, Python 3.11.9, torch 2.12.1, NumPy 2.4.6. All eight log-embedded receipt files were downloaded and SHA-256 verified. The parent receipt matches its worker, exact test inventories and manifest. Admission passed before setup, installation and tensor import; pip check passed.

GitHub documents standard public macOS execution as free: [runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners#standard-github-hosted-runners-for-public-repositories). Both runs use the standard public `macos-26` runner, with no paid runner, Actions artifact storage, cache storage or scientific campaign. The second workflow explicitly requires exactly the retained setup failure plus this run, and rejects any other attempt or rerun. The tensor worker has a 30-second cap; the hosted job has a 15-minute cap.

## Preservation and claim limits

All 440 protected predecessor files, 12 frozen v3 files, 42 sealed round-6 entries and all 123 prior pipeline artifacts match their retained hashes. Original source trees and prior bundles remain unchanged. The published quality revision lives in the separate publication bundle.

The scientific contribution gate remains BLOCKED, all later scientific checkpoints remain NOT_EVALUATED, concepts remain 8/8 reviewed, and v3 remains consumed. No protected outcome was opened and no scientific trial ran. Software verification cannot establish improved forecasting accuracy, novelty, SOTA, independent scientific reproduction, or completion of the research lifecycle.

## Final evidence

- `FINAL_VERIFICATION.json`: exact commit, source manifest, inventories, outcomes, environment and resource checks.
- `remote_run_2/`: eight verified raw receipts and runtime environment files.
- `REMOTE_RUN_1.log` and `remote_run_1/`: retained setup failure and admission receipt.
- `LOCAL_TESTS.json` and `LOCAL_TESTS_FINAL.json`: initial timeout failures and final 87/87 local pass.
- `PRESERVATION_VERIFIED.json`: unchanged predecessor and earlier package hashes.
- `setup_correction/remote_bundle/`: exact published 25-file source bundle.

The two quality rounds increased the cumulative repair count from 6 to 8 without resetting it. Two hosted setup attempts produced one actual test execution; the first attempt never reached any test. No further job was launched.
