"""Compatibility entry point for the revision's non-overwriting release tool."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(
        str(root / "publication/itfm-v2-paper-20261011/tools/release.py"),
        run_name="__main__",
    )
