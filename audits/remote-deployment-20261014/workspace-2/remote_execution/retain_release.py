"""Retain run outputs as draft-release assets, without Actions artifact billing."""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True)


def release_identity():
    repo = os.environ["GITHUB_REPOSITORY"]
    run = os.environ["GITHUB_RUN_ID"]
    sha = os.environ["GITHUB_SHA"]
    return repo, "itfm-execution-" + run, sha


def create():
    repo, tag, sha = release_identity()
    gh(
        "release",
        "create",
        tag,
        "--repo",
        repo,
        "--target",
        sha,
        "--draft",
        "--title",
        "ITFM execution receipts " + os.environ["GITHUB_RUN_ID"],
        "--notes",
        "Unpublished execution archive. Retains failures and partial outputs. "
        "No scientific success, publication readiness, or independent replication is implied.",
    )
    print("DRAFT_RETENTION_READY")


def split_archive(archive, directory, chunk_bytes=512 * 1024**2):
    parts = []
    with archive.open("rb") as source:
        index = 0
        while True:
            block = source.read(min(chunk_bytes, 1024**2))
            if not block:
                break
            part = directory / f"outputs.tar.gz.part-{index:03d}"
            checksum = hashlib.sha256()
            size = 0
            with part.open("xb") as target:
                while block:
                    target.write(block)
                    checksum.update(block)
                    size += len(block)
                    if size == chunk_bytes:
                        break
                    block = source.read(min(chunk_bytes - size, 1024**2))
            parts.append(
                {"file": part.name, "sha256": checksum.hexdigest(), "bytes": size}
            )
            index += 1
    return parts


def upload(root):
    repo, tag, sha = release_identity()
    info = json.loads(
        gh("release", "view", tag, "--repo", repo, "--json", "isDraft,targetCommitish")
    )
    if not info["isDraft"] or info["targetCommitish"] != sha:
        raise RuntimeError("draft retention target mismatch")
    directory = root / "release-assets"
    directory.mkdir(exist_ok=False)
    archive = directory / "outputs.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        tar.add(root / "remote-artifact", arcname="remote-artifact", recursive=True)
    parts = split_archive(archive, directory)
    manifest = directory / "ASSET_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "source_sha": sha,
                "run_id": os.environ["GITHUB_RUN_ID"],
                "parts": parts,
                "format": "concatenate parts in listed order, then gzip/tar",
            },
            indent=2,
        )
        + "\n"
    )
    # Never use --clobber: previous uploads and failures remain intact.
    for name in [p["file"] for p in parts] + [manifest.name]:
        gh("release", "upload", tag, str(directory / name), "--repo", repo)
    print(
        json.dumps({"status": "DRAFT_ASSETS_UPLOADED", "tag": tag, "parts": len(parts)})
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["create", "upload"])
    a = p.parse_args()
    if a.command == "create":
        create()
    else:
        upload(Path(__file__).resolve().parents[1])
