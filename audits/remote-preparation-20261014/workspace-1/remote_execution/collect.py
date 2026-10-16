"""Collect bounded receipts and raw outputs without executing tensor files."""

import argparse
import json
import shutil
from pathlib import Path

from prepare import digest, safe


def collect(root, output):
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected = {root / "REMOTE_MANIFEST.json", root / "REMOTE_EXECUTION_STARTED.json"}
    for name in (
        "remote-preflight",
        "remote-install",
        "remote-run",
        "successor/itfm_v2/publication/itfm-v2-paper-20261011/replays/remote-1",
        "successor/itfm_v3/runs",
    ):
        folder = safe(root, name)
        if folder.exists():
            selected.update(p for p in folder.rglob("*") if p.is_file())
    files = {}
    total = 0
    for path in sorted(selected):
        if not path.exists():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError("unsafe result path")
        total += path.stat().st_size
        # This is a collection ceiling, never an expanded experiment budget.
        if total > 3 * 1024**3:
            raise RuntimeError("collection exceeds 3 GiB; partial collection retained")
        name = path.relative_to(root).as_posix()
        dest = safe(output, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        files[name] = {"sha256": digest(dest), "bytes": dest.stat().st_size}
    (output / "COLLECTION.json").write_text(
        json.dumps(
            {
                "status": "COLLECTED_NOT_SCIENTIFICALLY_CERTIFIED",
                "files": files,
                "bytes": total,
                "source_manifest_sha256": digest(root / "REMOTE_MANIFEST.json"),
            },
            indent=2,
        )
        + "\n"
    )


def verify(output, expected):
    output = output.resolve()
    data = json.loads((output / "COLLECTION.json").read_text())
    if data["source_manifest_sha256"] != expected:
        raise RuntimeError("wrong source identity")
    actual = {
        p.relative_to(output).as_posix()
        for p in output.rglob("*")
        if p.is_file() and p.name != "COLLECTION.json"
    }
    if actual != set(data["files"]):
        raise RuntimeError("unexpected or missing collection files")
    for name, item in data["files"].items():
        p = safe(output, name)
        if digest(p) != item["sha256"] or p.stat().st_size != item["bytes"]:
            raise RuntimeError("corrupted result: " + name)
    if digest(output / "REMOTE_MANIFEST.json") != expected:
        raise RuntimeError("source manifest corrupted")
    print(
        json.dumps(
            {
                "status": "PASS_COLLECTION_HASHES",
                "files": len(actual),
                "boundary": "Transfer integrity only; inspect numerical receipts separately",
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["collect", "verify"])
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest-sha256")
    a = p.parse_args()
    if a.command == "collect":
        collect(a.root.resolve(), a.output.resolve())
    else:
        if not a.manifest_sha256:
            p.error("--manifest-sha256 required for verify")
        verify(a.output.resolve(), a.manifest_sha256)
