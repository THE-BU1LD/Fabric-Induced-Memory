"""Run the preserved wrapper, supplementing only its omitted required README.

No numerical source, guard, threshold, command or worker is replaced. The
original wrapper's private copy helper adds one hash-bound documentation file
to its fresh workspace; the original raw worker executes unchanged.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "successor/itfm_v2"
PUB = V2 / "publication/itfm-v2-paper-20261011"


def supplemented_copy(source, destination):
    if Path(source).resolve() != (PUB / "frozen_source").resolve():
        raise RuntimeError("unexpected replay source")
    result = shutil.copytree(source, destination)
    target = Path(destination) / "README.md"
    if target.exists():
        raise RuntimeError("README already exists; refuse replacement")
    shutil.copyfile(V2 / "README.md", target)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--python", required=True)
    a = p.parse_args()
    sys.path.insert(0, str(PUB / "tools"))
    import reproduce
    import release

    # Scope the helper to this imported wrapper, not the process-wide shutil module.
    reproduce.shutil = SimpleNamespace(copytree=supplemented_copy)
    receipt = reproduce.execute(a.output, a.python)
    release.write_new(
        a.output / "DOCUMENTATION_SUPPLEMENT.json",
        {
            "source": "successor/itfm_v2/README.md",
            "sha256": release.sha(V2 / "README.md"),
            "reason": "Frozen static audit requires README; original 20-file replay copy omitted it",
            "scientific_source_or_guard_changed": False,
        },
    )
    print(json.dumps({"status": receipt["status"]}))
    raise SystemExit(0 if receipt["status"] == "PASS_LOCAL_REPLAY" else 3)
