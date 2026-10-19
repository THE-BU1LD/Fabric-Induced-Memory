"""Portable wrapper for the maintained fim-sota engineering subcommand."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    from fim_sota.cli import main

    sys.argv.insert(1, "engineering")
    main()
