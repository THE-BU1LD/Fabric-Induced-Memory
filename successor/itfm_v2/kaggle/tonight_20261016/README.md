# FIM private Kaggle launch package — 16 October 2026

**Prepared for a later GPU launch, as requested.** The current attempt ended **BLOCKED BEFORE EXECUTION**. No CPU numerical experiment or GPU job ran for this campaign. The executable notebook is a locally prepared, statically checked launch artifact, not a cloud execution receipt. The numerical implementation remains the preceding engineering snapshot, SHA256 `97a31bff8d785c37b0f2af6bfc374900670ed66bd554e17bb1d6525253a719f2`.

## Launch after the account/editor gates are resolved

1. Complete Kaggle phone verification in the existing `ryangomez010` account. The live UI currently disables GPU access and does not expose its remaining GPU balance. No quota value has been assumed.
2. Open the [private notebook draft](https://www.kaggle.com/code/ryangomez010/fim-private-readiness-2026-10-16/edit). The editor remained at “Editor Loading” after one reload; Run, Start session and Import Notebook were disabled. Resolve that loading problem in the browser, then use **File → Import Notebook** to upload `FIM_tonight_20261016.ipynb` from this directory. No source was uploaded during the blocked attempt.
3. Keep sharing **Private**, accelerator **None**, and the environment pinned. Run notebook sections1–6 in order. The embedded verified source ZIP avoids Git credentials, internet downloads and Mac-only paths. Required installed packages are torch>=2.6, NumPy and pytest; missing/incompatible packages stop the notebook on CPU. Their actual versions are recorded, not silently upgraded.
4. Test persistence before any long run: **Save Version → Quick Save → Advanced settings → Save output for this version when creating a Quick Save**. Open the saved private version, attach its outputs through Add Input, and hash-check the probe from `/kaggle/input`. Quick Save alone defaults to not saving output. Files-only session persistence is an extra best-effort convenience, not the required durable copy.
5. Read the actual remaining GPU allowance and Active Events immediately before allocation. Enter the real values, observation times and an explicit completion deadline in section7. The notebook chooses no more than4 quota-hours and never more than the preferred allowance. GPU readiness/profiling is capped at20% of that allowance; at least10% is reserved for variance/export/shutdown. An unknown or stale observation, other active sessions, expired deadline or zero allowance blocks launch. This account offered T4×2; if used, both allocated devices are conservatively charged while training uses device0 only.
6. Run the bounded CUDA checks and representative development profile. Only after all gates pass may section8 freeze the complete affordable five-seed DEVELOPMENT matrix. If the whole matrix does not fit, freeze is refused. The original candidate is not a frozen scientific protocol.
7. Run one admitted cell at a time. Save/retrieve each completed cell, refresh account observations, and repeat. Analyze all expected cells, preserve failures, export, verify remote retrieval, and explicitly stop the platform allocation. Record the observed final balance and stop time outside the stopped runtime.

The original planning target was **2026-10-16T16:30:00Z (22:00 IST)**. You subsequently chose preparation for a later GPU launch. The notebook therefore requires an explicitly entered future completion deadline when creating its first live account ledger and records that deadline in the later freeze. It does not schedule work, assume a new date, or extend an existing ledger. Preserve the original blocked-attempt ledger as historical evidence.

## Exact maintained commands

In the extracted source directory, CPU readiness uses:

```bash
python -u scripts/kaggle.py test --device cpu --receipt NEW_CPU_RECEIPT.json
python -u scripts/kaggle.py run --config kaggle/smoke-cpu.json --output NEW_CPU_SMOKE
```

The notebook invokes these through bounded supervisors. It uses only three selected CUDA tests, then the actual maintained training command with resolved development configuration. It does not call a hypothetical `train.py`, invent CLI flags, run distributed training or use Save & Run All as a persistence shortcut.

For local static controls, from this directory:

```bash
rtk proxy python3 -B -m unittest test_budget test_analysis test_workflow
rtk proxy python3 -B build_notebook.py
```

Those tests use explicitly simulated clocks and temporary fixture artifacts; they are software checks, not experimental results. Building the notebook embeds only the allowlisted source ZIP and campaign helpers. It performs no tensor import or model execution.

## Persistence and resumption

`workflow.archive` hashes and reads back an archive of the dedicated campaign directory. Save it as a private notebook output, retrieve it through Add Input, and use `workflow.verify_retrieval` to compare hashes. A ZIP in `/kaggle/working` alone does not pass.

In a fresh CPU session, restore with `workflow.restore(input_archive, NEW_destination, expected_sha256)`. It refuses unsafe archive paths, checksum mismatches and an existing destination. Keep attached input read-only. Reinstall exactly the recorded package versions only when necessary and reviewed. Reconstruct source from the notebook's identical source snapshot; verify source/config/data/evaluation identity before reuse.

For an interrupted training cell, the maintained continuation is:

```bash
python -u scripts/kaggle.py run --config ORIGINAL_CONFIG.json --output NEW_UNIQUE_RUN --resume RETRIEVED_PARENT/checkpoints/latest.pt
```

Reserve the continuation's account/time budget first. Retain the parent failure and checkpoint digest in a separate continuation lineage. The helper deliberately blocks silent retries and does not automatically turn an interrupted mandatory cell into a success. Full-state recovery still needs its first numerical CPU/CUDA verification. Evaluation replays each independent owner bank from the start.

The ledger is account-scoped in policy, but a local JSON file is not a distributed quota service. Every project must coordinate against one authoritative ledger and fresh account-wide observations. Do not give another project a fresh allowance or run multiple copies concurrently. Ledger estimates include setup, idle, failed work and shutdown delay until a platform stop is observed.

## Scientific scope

The current executable benchmark is a bounded synthetic periodic field fixture. A fixed trained MLP supplies predictions; residual retrieval uses fixed normalized context keys. Base, memory, erased, spatial-mean and last-residual controls share a frozen checkpoint; persistence is an untuned comparator. Scales are selected from a fixed validation grid. The test owners remain separate, but this is DEVELOPMENT evidence and cannot be promoted to protected confirmation.

The five proposed seeds are independent top-level repetitions with paired comparisons. Correlated origins and spatial points are not independent samples. The smallest attainable two-sided five-seed sign-enumeration p-value is0.0625; no p<.05 claim is possible. Additional fixed-scale interventions are not admitted until implemented and tested. The autonomous rollout is base-only and does not measure memory OOD performance.

No preprint results section or venue-specific submission claim is supported now. Prospective methods and scope are in `SCIENCE_SCOPE.md` and `protocol_candidate.json`. The historical v3 evidence and v4 scientific admission boundary remain intact.

## Platform evidence

The live UI confirmed private sharing, no active events, disabled GPU eligibility, and Quick Save output options. It did not provide quota, a working editor, executed output or persistence retrieval. [Kaggle's notebook documentation](https://www.kaggle.com/docs/notebooks) distinguishes Quick Save from a clean-session Save & Run All and documents the20GB output limit. The latter starts another session; this package selects interactive execution only. [Kaggle's GPU guidance](https://www.kaggle.com/docs/efficient-gpu-usage) describes checking usage and stopping sessions. [Kaggle staff's persistence announcement](https://www.kaggle.com/product-feedback/355440) describes interactive file persistence as best effort. Actual retrieval and stop observations remain mandatory.
