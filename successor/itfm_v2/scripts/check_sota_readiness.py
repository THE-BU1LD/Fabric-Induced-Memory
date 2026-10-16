from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    required_receipts = {
        "upstream_licenses": root / "admission/upstream_licenses.json",
        "dataset_hashes_and_splits": root / "admission/datasets.json",
        "accelerator_and_cost_budget": root / "admission/compute_budget.json",
        "frontier_reproduction": root / "results/sota/frontier_reproduction.json",
        "oracle_gap": root / "results/sota/oracle_gap.json",
    }
    checks = {name: path.is_file() for name, path in required_receipts.items()}
    resource = subprocess.run(
        [sys.executable, str(root / "scripts/check_resource_admission.py")],
        capture_output=True,
        text=True,
    )
    resource_payload = json.loads(resource.stdout)
    checks["local_resource_admission"] = resource.returncode == 0
    payload = {
        "status": "READY" if all(checks.values()) else "BLOCKED",
        "checks": checks,
        "resource": resource_payload,
        "missing": [name for name, passed in checks.items() if not passed],
        "note": "READY authorizes frontier reproduction only; it does not authorize protected-test execution.",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "READY" else 1)


if __name__ == "__main__":
    main()
