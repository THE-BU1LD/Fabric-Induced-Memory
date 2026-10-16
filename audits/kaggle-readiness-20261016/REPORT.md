# FIM Kaggle preparation receipt — 16 October 2026

**BLOCKED:** engineering implemented; actual numerical verification refused before imports.

Checkout HEAD: `32d8aa0b109f08a8ecd23c7bd97eaf29b5fe8989`. Executable source SHA-256: `97a31bff8d785c37b0f2af6bfc374900670ed66bd554e17bb1d6525253a719f2`.

23 control tests, 13 AST/source compilations, focused Ruff lint, CLI help, a three-cell manifest dry-run and isolated exported-bundle dry-run passed. The archive contains 30 hash-verified source/test/config/document files. No numerical CPU/GPU test passed; the CPU test command recorded zero executed tests because resource admission failed.

The final samples were 152.9 and 139.9 MiB free-plus-speculative, below the unchanged 512 MiB floor. No Torch import, GPU workload, upload, push, deployment or consumed v3 restart occurred. Pytest is absent in the laptop interpreter.

All 16 pre-edit backups match their initial hashes. Existing dirty changes were retained; no historical result was rewritten. Exact source and test hashes and command receipts are in FINAL_VERIFICATION.json and its linked receipts.

Changed existing files:

- `successor/itfm_v2/src/fim_sota/config.py`
- `successor/itfm_v2/src/fim_sota/memory.py`
- `successor/itfm_v2/src/fim_sota/adapter.py`
- `successor/itfm_v2/src/fim_sota/backbones.py`
- `successor/itfm_v2/src/fim_sota/cli.py`
- `successor/itfm_v2/src/fim_sota/provenance.py`
- `successor/itfm_v2/src/fim_sota/training.py`
- `README.md`

New maintained path, tests and handoff:

- `successor/itfm_v2/kaggle/CHANGE_LEDGER.md`
- `successor/itfm_v2/kaggle/GPU_READINESS.md`
- `successor/itfm_v2/kaggle/HANDOFF.md`
- `successor/itfm_v2/kaggle/development.json`
- `successor/itfm_v2/kaggle/requirements-runtime.txt`
- `successor/itfm_v2/kaggle/smoke-cpu.json`
- `successor/itfm_v2/kaggle/smoke-cuda.json`
- `successor/itfm_v2/scripts/kaggle.py`
- `successor/itfm_v2/src/fim_sota/engineering.py`
- `successor/itfm_v2/src/fim_sota/engineering_runtime.py`
- `successor/itfm_v2/tests/test_engineering_control.py`
- `successor/itfm_v2/tests/test_engineering_runtime.py`
- `successor/itfm_v2/tests/test_training_recovery.py`

Start with `successor/itfm_v2/scripts/kaggle.py preflight --config successor/itfm_v2/kaggle/smoke-cpu.json --receipt <fresh-path>`, then the CPU test command. After those pass and cloud use is separately authorized, the smallest CUDA step is `test --device cuda --receipt <fresh-path>`, preceded by explicit CUDA runtime preflight. HANDOFF.md gives every command.

The zip is `fim-kaggle-preflight.zip`; its hash is `9454e9cef6fba5ccf6057545ade46cd566167ad47570753623d789b5ffadfc46`. It is a local preparation artifact, not a published release. The generator is an engineering diagnostic. Public data/baseline admission, efficacy and v4 contribution remain unresolved.
