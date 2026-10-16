from __future__ import annotations

import json
from pathlib import Path

from fim_v2.audit import audit


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = audit(root)
    output = root / "results/static_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
