# Third-party notices and admission boundary

The `fim_sota` source in this directory is an independent implementation. It
does not vendor LANO, HS-FNO, POBench-PDE, ERA5, Poseidon, The Well, or
Neural-Solver-Library source or data.

Runtime dependencies are pinned in `locks/` and must pass
`scripts/scan_licenses.py` in the exact release environment. A dependency name
or public repository URL is not a license grant.

The frozen upstream registry records two unresolved source boundaries:

- LANO at `b53a62100b4705f1382a7c344b2d84afa255b96f`;
- HS-FNO at `9ebf2a7aa483c7f74769365c187c7d9492ecd6ef`.

Neither upstream source may be copied, redistributed, or packaged here until an
authoritative license receipt is stored under `admission/`. Dataset source,
use, redistribution, privacy, and security rights must be recorded separately.
