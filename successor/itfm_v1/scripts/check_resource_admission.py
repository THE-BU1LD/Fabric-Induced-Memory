from __future__ import annotations

import json
import subprocess
import sys


def main() -> None:
    if sys.platform != "darwin":
        print(
            json.dumps(
                {"status": "UNKNOWN", "reason": "resource guard is macOS-specific"}
            )
        )
        raise SystemExit(2)
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
    threshold = 512 * 1024 * 1024
    admitted = available >= threshold
    print(
        json.dumps(
            {
                "status": "PASS" if admitted else "BLOCK",
                "free_plus_speculative_bytes": available,
                "threshold_bytes": threshold,
                "pageouts_cumulative": values.get("Pageouts", 0),
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if admitted else 1)


if __name__ == "__main__":
    main()
