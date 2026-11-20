from __future__ import annotations

import subprocess
import sys
import time
from typing import Any


def _mac_snapshot() -> tuple[int, int]:
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


def resource_admission(
    threshold_bytes: int = 512 * 1024 * 1024,
    stability_interval_seconds: float = 0.25,
) -> dict[str, Any]:
    if sys.platform != "darwin":
        return {
            "status": "UNKNOWN",
            "reason": "resource guard is macOS-specific",
            "threshold_bytes": threshold_bytes,
            "checked_at_unix": time.time(),
        }
    first_available, first_pageouts = _mac_snapshot()
    time.sleep(stability_interval_seconds)
    second_available, second_pageouts = _mac_snapshot()
    available = min(first_available, second_available)
    pageout_delta = second_pageouts - first_pageouts
    admitted = available >= threshold_bytes and pageout_delta == 0
    return {
        "status": "PASS" if admitted else "BLOCK",
        "free_plus_speculative_bytes": available,
        "threshold_bytes": threshold_bytes,
        "pageouts_cumulative": second_pageouts,
        "pageouts_during_stability_check": pageout_delta,
        "stability_interval_seconds": stability_interval_seconds,
        "checked_at_unix": time.time(),
    }
