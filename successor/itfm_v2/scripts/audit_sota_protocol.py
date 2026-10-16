from __future__ import annotations

import json
from pathlib import Path

from fim_v2.sota import audit_sota_protocol


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "protocols/sota_confirmation_v1.json").read_text(encoding="utf-8")
    )
    frontier = json.loads(
        (root / "protocols/frontier_registry_2026-10-11.json").read_text(
            encoding="utf-8"
        )
    )
    checks = audit_sota_protocol(protocol, frontier)
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
    }
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
