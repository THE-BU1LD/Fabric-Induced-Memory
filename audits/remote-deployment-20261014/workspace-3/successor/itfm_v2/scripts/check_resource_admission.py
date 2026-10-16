from __future__ import annotations

import json
import subprocess
import sys
import time


def snapshot() -> tuple[int, int]:
    output = subprocess.run(
        ["vm_stat"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    page_size = 16384
    values: dict[str, int] = {}
    for line in output:
        if "page size of" in line:
            page_size = int(
                line.split("page size of", 1)[1].split("bytes", 1)[0].strip()
            )
        elif ":" in line:
            key, raw = line.split(":", 1)
            values[key.strip()] = int(raw.strip().rstrip("."))
    available = page_size * (
        values.get("Pages free", 0) + values.get("Pages speculative", 0)
    )
    return available, values.get("Pageouts", 0)


def main() -> None:
    if sys.platform != "darwin":
        print(json.dumps({"status": "UNKNOWN", "reason": "macOS-only guard"}))
        raise SystemExit(2)
    first_available, first_pageouts = snapshot()
    interval = 0.25
    time.sleep(interval)
    second_available, second_pageouts = snapshot()
    available = min(first_available, second_available)
    threshold = 512 * 1024 * 1024
    pageout_delta = second_pageouts - first_pageouts
    admitted = available >= threshold and pageout_delta == 0
    payload = {
        "status": "PASS" if admitted else "BLOCK",
        "free_plus_speculative_bytes": available,
        "threshold_bytes": threshold,
        "pageouts_cumulative": second_pageouts,
        "pageouts_during_stability_check": pageout_delta,
        "stability_interval_seconds": interval,
    }
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
