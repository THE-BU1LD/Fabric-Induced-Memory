"""Build an allowlisted, hash-bound remote workspace without numerical imports."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = Path("successor/itfm_v2")
PUB = V2 / "publication/itfm-v2-paper-20261011"
V3 = Path("successor/itfm_v3")


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def safe(root, name):
    p = (root / name).resolve()
    if Path(name).is_absolute() or not p.is_relative_to(root.resolve()):
        raise ValueError("unsafe manifest path: " + str(name))
    return p


def prepare(output):
    if (ROOT / V3 / "runs/CAMPAIGN_CONSUMED.json").exists():
        raise RuntimeError("v3 campaign already consumed; adjudication required")
    if (ROOT / V3 / "RUNNER.lock").exists():
        raise RuntimeError("v3 runner lock exists")
    selected = set()
    baseline = json.loads((ROOT / PUB / "PRESERVATION_BASELINE.json").read_text())
    selected.update(V2 / name for name in baseline)
    selected.add(PUB / "PRESERVATION_BASELINE.json")
    selected.add(PUB / "REPLAY_REQUIREMENTS.txt")
    selected.add(V2 / "README.md")
    for folder in (PUB / "frozen_source", PUB / "tools", V2 / "paper"):
        selected.update(
            p.relative_to(ROOT)
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    frozen = json.loads((ROOT / V3 / "freezes/development-2.json").read_text())
    for name, expected in frozen["files"].items():
        if digest(safe(ROOT / V3, name)) != expected:
            raise RuntimeError("v3 source changed: " + name)
        selected.add(V3 / name)
    selected.add(V3 / "freezes/development-2.json")
    adjudication = Path("remote_execution/PREEXECUTION_ADJUDICATION.json")
    if (ROOT / adjudication).exists():
        selected.add(adjudication)
    selected.update(
        p.relative_to(ROOT)
        for p in (ROOT / "remote_execution").iterdir()
        if p.is_file() and p.suffix in {".py", ".md", ".yml"}
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for relative in sorted(selected):
        src = safe(ROOT, relative)
        if src.is_symlink() or not src.is_file():
            raise RuntimeError("not a regular file: " + str(relative))
        dest = safe(output, relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        files[relative.as_posix()] = digest(dest)
    manifest = {
        "schema": 1,
        "files": files,
        "v3_freeze": frozen,
        "scope": "Synthetic v2 replay then one frozen v3 development campaign",
        "guards": "Unchanged macOS source; no Linux shim or threshold override",
        "publication_authorized": False,
        "paid_compute_authorized": False,
    }
    (output / "REMOTE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "workspace": str(output),
                "files": len(files),
                "manifest_sha256": digest(output / "REMOTE_MANIFEST.json"),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args().output)
