from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/materialized/aliased-advection-v1"),
    )
    parser.add_argument("--pdebench-file")
    parser.add_argument("--pdebench-equation", default="Advection")
    parser.add_argument(
        "--pdebench-output",
        type=Path,
        default=Path("datasets/downloads/pdebench"),
    )
    parser.add_argument("--max-download-bytes", type=int)
    args = parser.parse_args()
    if args.pdebench_file and (
        args.max_download_bytes is None or args.max_download_bytes < 1
    ):
        parser.error("--pdebench-file requires a positive --max-download-bytes")
    guard = Path(__file__).with_name("check_resource_admission.py")
    admission = subprocess.run([sys.executable, str(guard)], check=False)
    if admission.returncode != 0:
        raise SystemExit(3)
    from fim_s1.data import AliasedAdvectionCorpus
    from fim_s1.datasets import (
        PDEBenchRegistry,
        download_pdebench_file,
        materialize_aliased_advection,
        md5_file,
    )

    if args.pdebench_file:
        registry = PDEBenchRegistry(
            Path("datasets/external/pdebench/pdebench_data_urls.csv")
        )
        selected = registry.select(
            equation=args.pdebench_equation,
            filename=args.pdebench_file,
        )
        if len(selected) != 1:
            raise RuntimeError("exact PDEBench selection did not resolve uniquely")
        downloaded = download_pdebench_file(
            selected[0],
            args.pdebench_output,
            max_bytes=int(args.max_download_bytes),
        )
        print(
            json.dumps(
                {
                    "dataset": "PDEBench",
                    "equation": args.pdebench_equation,
                    "file": str(downloaded),
                    "bytes": downloaded.stat().st_size,
                    "md5": md5_file(downloaded),
                },
                sort_keys=True,
            )
        )
        return

    corpus = AliasedAdvectionCorpus()
    manifest = materialize_aliased_advection(corpus, args.output)
    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset_id"],
                "archive_sha256": manifest["archive_sha256"],
                "archive_bytes": manifest["archive_bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
