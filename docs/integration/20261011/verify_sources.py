"""Verify the current FIM source union and preserved canonical histories.

This is a file-identity/integration check, not evidence that a scientific run
occurred. Pass --git-sources in a checkout containing all recorded source commits
to additionally verify every inventory statement against the original Git tree.
The 20261010 verifier is retained for its exact historical source scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess


ROOT = Path(__file__).resolve().parents[3]
INVENTORY = Path(__file__).with_name("sources.json")
STATE = "research/RESEARCH_STATE.json"


def blob_identity(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def inspect(path: str, record: dict) -> bytes:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe source path: {path}")
    local = ROOT / relative
    if any(parent.is_symlink() for parent in (local, *local.parents) if parent != ROOT.parent):
        raise ValueError(f"Symlink in source path: {path}")
    if not local.is_file():
        raise ValueError(f"Expected a regular source file: {path}")
    data = local.read_bytes()
    if blob_identity(data) != record["git_blob"]:
        raise ValueError(f"Git blob mismatch: {path}")
    if "sha256" in record and hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")
    if "mode" in record:
        mode = "100755" if local.stat().st_mode & stat.S_IXUSR else "100644"
        if mode != record["mode"]:
            raise ValueError(f"Executable-mode mismatch: {path}")
    return data


def git_tree(ref: str) -> dict:
    raw = subprocess.check_output(["git", "ls-tree", "-rz", ref], cwd=ROOT)
    leaves = {}
    for entry in raw.split(b"\0"):
        if entry:
            metadata, name = entry.split(b"\t", 1)
            mode, kind, sha = metadata.decode().split()
            if kind != "blob":
                raise ValueError(f"Unsupported source object: {name!r}")
            leaves[name.decode()] = {"mode": mode, "git_blob": sha}
    return leaves


def identity(record: dict) -> dict:
    return {key: record[key] for key in ("mode", "git_blob")}


def verify_git(inventory: dict) -> None:
    base = git_tree(inventory["base_commit"])
    adopted = inventory["adopted_file_identities"]
    source_states = {r["pr"]: r for r in inventory["canonical_predecessors"]}
    expected_union = {}
    for source in inventory["source_proposals"]:
        leaves = git_tree(source["head_commit"])
        actual_tree = subprocess.check_output(
            ["git", "rev-parse", source["head_commit"] + "^{tree}"], cwd=ROOT, text=True
        ).strip()
        if actual_tree != source["tree"]:
            raise ValueError(f"Source tree mismatch for PR{source['pr']}")
        if set(base) - set(leaves):
            raise ValueError(f"Source proposal removes inherited leaves: PR{source['pr']}")
        delta = {p: r for p, r in leaves.items() if base.get(p) != r}
        if sorted(delta) != source["changed_paths"]:
            raise ValueError(f"Source changed-path inventory mismatch: PR{source['pr']}")
        if leaves[STATE]["git_blob"] != source_states[source["pr"]]["git_blob"]:
            raise ValueError(f"Source canonical snapshot mismatch: PR{source['pr']}")
        for path, record in delta.items():
            if path == STATE:
                continue
            if path in expected_union and expected_union[path] != record:
                raise ValueError(f"Unresolved source conflict: {path}")
            expected_union[path] = record
            if identity(adopted[path]) != record or adopted[path]["source_commit"] != source["head_commit"]:
                raise ValueError(f"Adopted identity differs from source: {path}")
    if expected_union != {p: identity(r) for p, r in adopted.items()}:
        raise ValueError("The adopted source union is incomplete or contains an extra file")
    expected_unchanged = {p: r for p, r in base.items() if p not in adopted and p != STATE}
    if expected_unchanged != inventory["unchanged_base_file_identities"]:
        raise ValueError("The unchanged-base inventory is incomplete")
    expected_superseded = {p: base[p] for p in adopted if p in base}
    if expected_superseded != inventory["superseded_base_leaves"]:
        raise ValueError("The superseded-base inventory is incomplete")


def verify() -> dict:
    inventory = json.loads(INVENTORY.read_text())
    for bucket in ("adopted_file_identities", "unchanged_base_file_identities"):
        for path, record in inventory[bucket].items():
            inspect(path, record)
    states = {}
    for record in inventory["canonical_predecessors"]:
        states[record["pr"]] = json.loads(inspect(record["snapshot_path"], record))
    canonical = json.loads((ROOT / STATE).read_text())
    if canonical["followup_predecessor_records"] != inventory["canonical_predecessors"]:
        raise ValueError("Canonical predecessor membership differs from the inventory")
    base = states[38]
    for key, value in base.items():
        if key not in {"updated_on", "phase"} and canonical.get(key) != value:
            raise ValueError(f"Inherited canonical field changed: {key}")
    for pr, previous in states.items():
        if pr == 38:
            continue
        for key, value in previous.items():
            if key in base:
                if value != base[key] and key != "phase":
                    raise ValueError(f"Unexpected source state divergence: PR{pr} {key}")
                continue
            current = canonical.get(key)
            if isinstance(value, list):
                if not isinstance(current, list) or any(item not in current for item in value):
                    raise ValueError(f"Source session omitted: PR{pr} {key}")
            elif current != value:
                raise ValueError(f"Source-specific state lost: PR{pr} {key}")
    if canonical["protected_evaluation"] != base["protected_evaluation"]:
        raise ValueError("Scientific holds changed")
    archived_tests = [
        str(p.relative_to(ROOT))
        for p in (ROOT / "research").rglob("*.py")
        if p.name.startswith("test_") or p.name.endswith("_test.py")
    ]
    if archived_tests:
        raise ValueError(f"Archived tests would enter default collection: {archived_tests}")
    return {
        "status": "VERIFIED_FOLLOWUP_SOURCE_UNION",
        "adopted_paths": len(inventory["adopted_file_identities"]),
        "unchanged_base_leaves": len(inventory["unchanged_base_file_identities"]),
        "preserved_canonical_states": len(states),
        "scientific_hold": canonical["protected_evaluation"]["status"],
        "scientific_outcome_inferred": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-sources", action="store_true")
    args = parser.parse_args()
    result = verify()
    if args.git_sources:
        verify_git(json.loads(INVENTORY.read_text()))
        result["original_git_trees"] = "verified"
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
