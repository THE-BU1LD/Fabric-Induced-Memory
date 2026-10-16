from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_step(
    name: str, command: list[str], *, root: Path, env: dict[str, str]
) -> dict[str, Any]:
    started = time.time()
    completed = subprocess.run(
        command, cwd=root, env=env, capture_output=True, text=True
    )
    return {
        "name": name,
        "command": command,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(root / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    started = time.time()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "started_at_unix": started,
        "python": python,
        "niceness": os.nice(0),
        "threads": 1,
        "status": "running",
        "steps": [],
    }
    receipt_id = f"verification-{time.time_ns()}-pid-{os.getpid()}"
    receipt_path = root / "results/receipts" / f"{receipt_id}.json"
    latest_path = root / "results/verification_receipt.json"

    steps = [
        (
            "resource_admission_before_tests",
            [python, str(root / "scripts/check_resource_admission.py")],
        ),
        ("tests", [python, "-m", "pytest"]),
        ("static_contract_audit", [python, str(root / "scripts/audit_contract.py")]),
        (
            "resource_admission_before_experiment",
            [python, str(root / "scripts/check_resource_admission.py")],
        ),
        (
            "development_experiment",
            [
                python,
                "-m",
                "fim_v2.experiment",
                "--protocol",
                str(root / "protocols/development_v2.json"),
                "--output",
                str(root / "results/development-v2"),
            ],
        ),
        ("result_verification", [python, str(root / "scripts/verify_results.py")]),
    ]
    for name, command in steps:
        result = run_step(name, command, root=root, env=env)
        receipt["steps"].append(result)
        if result["exit_code"] != 0:
            if name.startswith("resource_admission") or result["exit_code"] == 3:
                receipt["status"] = "blocked_resource"
            else:
                receipt["status"] = "failed"
            receipt["failed_step"] = name
            receipt["finished_at_unix"] = time.time()
            atomic_json(receipt_path, receipt)
            atomic_json(latest_path, receipt)
            print(json.dumps({"status": receipt["status"], "failed_step": name}))
            raise SystemExit(result["exit_code"])

    receipt["status"] = "complete"
    receipt["finished_at_unix"] = time.time()
    atomic_json(receipt_path, receipt)
    atomic_json(latest_path, receipt)
    print(json.dumps({"status": "complete", "receipt": str(receipt_path)}))


if __name__ == "__main__":
    main()
