from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "development_v1.json"


class ResourceBlocked(RuntimeError):
    pass


def run_step(name: str, command: list[str], receipt: dict[str, Any]) -> None:
    started = time.time()
    process = subprocess.run(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
    )
    step = {
        "name": name,
        "command": command,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    receipt["steps"].append(step)
    write_receipt(receipt)
    if process.returncode != 0:
        resource_blocked = name.startswith("resource_admission_") or (
            process.returncode == 3 and name.endswith("experiment")
        )
        receipt["status"] = "blocked_resource" if resource_blocked else "failed"
        receipt["failed_step"] = name
        write_receipt(receipt)
        if resource_blocked:
            raise ResourceBlocked(f"{name} stopped at the resource boundary")
        raise RuntimeError(f"{name} failed with exit code {process.returncode}")


def write_receipt(receipt: dict[str, Any]) -> None:
    destinations = [
        ROOT / str(receipt["receipt_file"]),
        ROOT / "results" / "verification_receipt.json",
    ]
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)


def admit(receipt: dict[str, Any], label: str) -> None:
    run_step(
        f"resource_admission_{label}",
        [sys.executable, str(ROOT / "scripts" / "check_resource_admission.py")],
        receipt,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-development",
        action="store_true",
        help="run the complete development matrix after the bounded smoke run",
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--niceness", type=int, default=10)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not 0 <= args.niceness <= 19:
        parser.error("--niceness must be between 0 and 19")
    if args.niceness:
        os.nice(args.niceness)

    started_at = time.time()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_unix": started_at,
        "receipt_file": (
            f"results/receipts/verification-{time.time_ns()}-pid-{os.getpid()}.json"
        ),
        "full_development_requested": args.full_development,
        "device": args.device,
        "threads": args.threads,
        "niceness": args.niceness,
        "steps": [],
    }
    write_receipt(receipt)
    try:
        admit(receipt, "before_tests")
        run_step(
            "tests",
            [sys.executable, "-m", "pytest"],
            receipt,
        )
        admit(receipt, "before_dataset_materialization")
        run_step(
            "prepare_synthetic_dataset",
            [sys.executable, str(ROOT / "scripts" / "prepare_datasets.py")],
            receipt,
        )
        admit(receipt, "before_smoke")
        smoke_output = ROOT / "results" / "smoke-development-v1"
        run_step(
            "smoke_experiment",
            [
                sys.executable,
                "-m",
                "fim_s1.experiment",
                "--protocol",
                str(PROTOCOL),
                "--output",
                str(smoke_output),
                "--device",
                args.device,
                "--threads",
                str(args.threads),
                "--smoke",
            ],
            receipt,
        )
        run_step(
            "smoke_analysis",
            [
                sys.executable,
                "-m",
                "fim_s1.analysis",
                str(smoke_output),
                "--resamples",
                "1000",
            ],
            receipt,
        )
        if args.full_development:
            admit(receipt, "before_full_development")
            development_output = ROOT / "results" / "development-v1"
            run_step(
                "full_development_experiment",
                [
                    sys.executable,
                    "-m",
                    "fim_s1.experiment",
                    "--protocol",
                    str(PROTOCOL),
                    "--output",
                    str(development_output),
                    "--device",
                    args.device,
                    "--threads",
                    str(args.threads),
                ],
                receipt,
            )
            run_step(
                "full_development_analysis",
                [
                    sys.executable,
                    "-m",
                    "fim_s1.analysis",
                    str(development_output),
                ],
                receipt,
            )
        receipt["status"] = "complete"
    except ResourceBlocked as error:
        receipt["error"] = str(error)
        receipt["status"] = "blocked_resource"
        receipt["finished_at_unix"] = time.time()
        write_receipt(receipt)
        print(json.dumps({"status": receipt["status"]}, sort_keys=True))
        raise SystemExit(3) from None
    except Exception as error:
        receipt.setdefault("error", str(error))
        receipt["status"] = "failed"
        write_receipt(receipt)
        raise
    receipt["finished_at_unix"] = time.time()
    write_receipt(receipt)
    print(json.dumps({"status": receipt["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
