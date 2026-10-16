from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ALLOWED = {
    "Apache Software License",
    "Apache-2.0",
    "BSD License",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "MIT",
    "MIT License",
    "Python Software Foundation License",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/license-scan.json")
    )
    arguments = parser.parse_args()
    completed = subprocess.run(
        [sys.executable, "-m", "piplicenses", "--format=json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        payload = {
            "status": "BLOCKED_SCANNER_UNAVAILABLE",
            "stderr": completed.stderr[-2000:],
        }
    else:
        packages = json.loads(completed.stdout)
        blocked = [
            item
            for item in packages
            if str(item.get("License", "UNKNOWN")) not in ALLOWED
        ]
        payload = {
            "status": "PASS" if not blocked else "BLOCKED_LICENSE_REVIEW",
            "packages": packages,
            "blocked": blocked,
            "allowed_licenses": sorted(ALLOWED),
        }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(arguments.output)}))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
