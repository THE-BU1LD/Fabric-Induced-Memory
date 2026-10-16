# Dataset staging

## Synthetic development data

`scripts/prepare_datasets.py` deterministically materializes
`aliased-advection-v1` into `datasets/materialized/aliased-advection-v1/` as a
compressed trajectory archive plus a hash-bound manifest. Materialization is
idempotent but fail-closed: incomplete or mismatched retained files are never
overwritten silently.

## PDEBench external data

`external/pdebench/pdebench_data_urls.csv` and `LICENSE.txt` are staged directly
from the official PDEBench repository. The registry contains the authoritative
per-file HTTPS URL and MD5 published by PDEBench. The dataset itself is licensed
CC BY 4.0 by its data repository; the staged code-repository license describes
PDEBench software.

The repository does not automatically download PDEBench arrays. The official
advection collection is approximately 47 GB. `PDEBenchRegistry` and
`download_pdebench_file` require an exact filename, an explicit destination,
and a hard byte limit, retain non-empty partial failures, and verify the
published checksum before promoting a file. Public HDF5 arrays are opened
lazily through `HDF5TrajectoryStore` with explicit axis semantics.

External public data remains development-only until a separately versioned
confirmatory protocol freezes exact files, hashes, splits, and observation
operators.

An exact shard can be requested explicitly after resource admission, for
example:

```bash
PYTHONPATH=src python scripts/prepare_datasets.py \
  --pdebench-equation Advection \
  --pdebench-file 1D_Advection_Sols_beta0.4.hdf5 \
  --max-download-bytes <explicit-byte-cap>
```

No default byte cap is guessed. A missing cap is an error, and a completed file
is not promoted until its MD5 equals the official registry value.
