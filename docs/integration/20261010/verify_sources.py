"""Verify the inspected source union and retained predecessor identities.

This checks file identity against the version-controlled inventory. It is not an
external signature, an execution receipt, or evidence of scientific efficacy.
Run from any directory: python docs/integration/20261010/verify_sources.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[3]
INVENTORY = Path(__file__).with_name("sources.json")


def blob_identity(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def inspect(path: str, record: dict) -> None:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe inventory path: {path}")
    local = ROOT / relative
    if local.is_symlink() or not local.is_file():
        raise ValueError(f"Expected regular file: {path}")
    data = local.read_bytes()
    if blob_identity(data) != record["git_blob"]:
        raise ValueError(f"Git blob mismatch: {path}")
    if "sha256" in record and hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")
    if "mode" in record:
        actual = "100755" if local.stat().st_mode & stat.S_IXUSR else "100644"
        if actual != record["mode"]:
            raise ValueError(f"Executable mode mismatch: {path}")


def main() -> None:
    inventory = json.loads(INVENTORY.read_text())
    for path, record in inventory["adopted_file_identities"].items():
        inspect(path, record)
    for path, record in inventory["unchanged_main_file_identities"].items():
        inspect(path, record)
    for record in inventory["preserved_metadata_snapshots"]:
        inspect(record["snapshot_path"], record)
    expected_states = [
        record for record in inventory["preserved_metadata_snapshots"]
        if record["original_path"] in {"RESEARCH_STATE.json", "research/RESEARCH_STATE.json"}
    ]
    pointer = json.loads((ROOT / "RESEARCH_STATE.json").read_text())
    if pointer["canonical_entry_point"] != "research/RESEARCH_STATE.json":
        raise ValueError("Root state must point to the single canonical research index")
    canonical = json.loads((ROOT / pointer["canonical_entry_point"]).read_text())
    if canonical["predecessor_records"] != expected_states:
        raise ValueError("Canonical predecessor index differs from exact source inventory")
    expected_root = next(record for record in expected_states if record["original_path"] == "RESEARCH_STATE.json")
    if pointer["previous_record_snapshot"] != expected_root["snapshot_path"]:
        raise ValueError("Root predecessor pointer does not resolve to its retained snapshot")
    print(json.dumps({
        "status": "VERIFIED_SOURCE_UNION",
        "adopted_files": len(inventory["adopted_file_identities"]),
        "unchanged_main_files": len(inventory["unchanged_main_file_identities"]),
        "predecessor_snapshots": len(inventory["preserved_metadata_snapshots"]),
        "canonical_links": "verified",
        "scientific_outcome_inferred": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
