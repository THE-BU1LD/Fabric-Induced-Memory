from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .provenance import atomic_write_json, sha256_file


class VerificationError(RuntimeError):
    pass


def verify_bound_artifacts(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    verified: list[Mapping[str, Any]] = []
    paths: set[str] = set()
    for record in manifest.get("outputs", []):
        path = Path(record["path"])
        normalized = str(path.resolve())
        if normalized in paths:
            raise VerificationError(f"duplicate artifact path: {normalized}")
        paths.add(normalized)
        if not path.is_file():
            raise VerificationError(f"missing artifact: {path}")
        actual = sha256_file(path)
        if actual != record.get("sha256"):
            raise VerificationError(f"artifact hash mismatch: {path}")
        verified.append(
            {"path": normalized, "sha256": actual, "bytes": path.stat().st_size}
        )
    return verified


def recompute_scalar_metrics(prediction_path: str | Path) -> Mapping[str, float]:
    payload = json.loads(Path(prediction_path).read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise VerificationError("prediction artifact has no rows")
    squared_error = 0.0
    squared_target = 0.0
    count = 0
    for row in rows:
        predictions = row.get("prediction")
        targets = row.get("target")
        if not isinstance(predictions, list) or not isinstance(targets, list):
            raise VerificationError("prediction row lacks raw prediction/target arrays")
        if len(predictions) != len(targets) or not predictions:
            raise VerificationError("prediction/target array shape mismatch")
        for prediction, target in zip(predictions, targets, strict=True):
            squared_error += (float(prediction) - float(target)) ** 2
            squared_target += float(target) ** 2
            count += 1
    return {
        "mse": squared_error / count,
        "relative_l2": (squared_error / max(squared_target, 1e-24)) ** 0.5,
    }


def verify_metric_claims(
    claimed: Mapping[str, float],
    recomputed: Mapping[str, float],
    tolerance: float = 1e-10,
) -> None:
    for name, value in claimed.items():
        if name not in recomputed:
            raise VerificationError(f"claimed metric has no recomputation: {name}")
        if abs(float(value) - float(recomputed[name])) > tolerance:
            raise VerificationError(
                f"metric mismatch for {name}: claimed={value} "
                f"recomputed={recomputed[name]}"
            )


def decide_sota(
    *,
    protocol: Mapping[str, Any],
    primary_cells: Sequence[Mapping[str, Any]],
    secondary_families: Sequence[Mapping[str, Any]],
    independently_reproduced: bool,
) -> Mapping[str, Any]:
    primary_gate = protocol["primary_track"]["material_sota_gate"]
    secondary_gate = protocol["secondary_track"]["material_sota_gate"]
    wins = sum(bool(row.get("win")) for row in primary_cells)
    regressions = [float(row.get("relative_regression", 0.0)) for row in primary_cells]
    seed_wins = [float(row.get("seed_win_fraction", 0.0)) for row in primary_cells]
    macro_gain = sum(float(row["relative_improvement"]) for row in primary_cells) / max(
        len(primary_cells), 1
    )
    primary_pass = (
        len(primary_cells) == int(protocol["primary_track"]["cells"])
        and wins >= int(primary_gate["minimum_cells_won_of_36"])
        and macro_gain
        >= float(
            primary_gate["minimum_macro_relative_error_reduction_vs_reproduced_lano"]
        )
        and max(regressions, default=float("inf"))
        <= float(primary_gate["maximum_relative_regression_any_cell"])
        and min(seed_wins, default=0.0)
        >= float(primary_gate["minimum_seed_win_fraction"])
    )
    family_wins = sum(bool(row.get("win")) for row in secondary_families)
    secondary_gain = sum(
        float(row["relative_improvement"]) for row in secondary_families
    ) / max(len(secondary_families), 1)
    target_error = sum(
        float(row["candidate_error"]) for row in secondary_families
    ) / max(len(secondary_families), 1)
    secondary_pass = (
        len(secondary_families) == len(protocol["secondary_track"]["families"])
        and family_wins >= int(secondary_gate["minimum_family_wins_of_5"])
        and secondary_gain
        >= float(secondary_gate["minimum_relative_improvement_vs_reproduced_hs_fno"])
        and target_error <= float(secondary_gate["maximum_target_rollout_error"])
    )
    broad = primary_pass and secondary_pass and independently_reproduced
    return {
        "primary_track_pass": primary_pass,
        "secondary_track_pass": secondary_pass,
        "independently_reproduced": independently_reproduced,
        "broad_sota_allowed": broad,
        "claim": (
            "BROAD_SOTA_ALLOWED"
            if broad
            else "NO_BROAD_SOTA_CLAIM_REPORT_BOUNDED_OUTCOME"
        ),
        "primary": {
            "cells": len(primary_cells),
            "wins": wins,
            "macro_gain": macro_gain,
        },
        "secondary": {
            "families": len(secondary_families),
            "wins": family_wins,
            "macro_gain": secondary_gain,
            "candidate_error": target_error,
        },
    }


def verify_run(
    manifest_path: str | Path,
    receipt_path: str | Path,
) -> Mapping[str, Any]:
    source = Path(manifest_path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "manifest_schema": manifest.get("schema_version") == 1,
        "executed": manifest.get("status") == "PASS_EXECUTED_UNVERIFIED",
        "failures_empty": not manifest.get("failures"),
    }
    try:
        verified = verify_bound_artifacts(manifest)
        checks["artifact_hashes"] = True
    except VerificationError:
        verified = []
        checks["artifact_hashes"] = False
    payload = {
        "schema_version": 1,
        "status": "PASS_VERIFIED" if all(checks.values()) else "FAIL_VERIFICATION",
        "manifest": str(source.resolve()),
        "manifest_sha256": sha256_file(source),
        "checks": checks,
        "artifacts": verified,
    }
    atomic_write_json(receipt_path, payload, overwrite=False)
    return payload


def archival_manifest(root: str | Path, output: str | Path) -> Mapping[str, Any]:
    base = Path(root).resolve()
    destination = Path(output).resolve()
    files = [path for path in base.rglob("*") if path.is_file() and path != destination]
    records = [
        {
            "path": str(path.relative_to(base)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(files)
    ]
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["path"].encode())
        digest.update(record["sha256"].encode())
    payload = {
        "schema_version": 1,
        "root": str(base),
        "tree_sha256": digest.hexdigest(),
        "files": records,
    }
    atomic_write_json(destination, payload, overwrite=False)
    return payload


def build_verified_report(verification_receipt: str | Path, output: str | Path) -> None:
    receipt_path = Path(verification_receipt)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_VERIFIED":
        raise VerificationError("report generation requires PASS_VERIFIED evidence")
    lines = [
        "# ITFM-R verified run report",
        "",
        f"Verification receipt: `{receipt_path}`",
        f"Manifest SHA-256: `{receipt['manifest_sha256']}`",
        "",
        "## Evidence boundary",
        "",
        "This report records verified artifacts only. It does not, by itself,",
        "establish a state-of-the-art claim, publication readiness, or independent",
        "reproduction.",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(
        f"- `{record['path']}` — `{record['sha256']}` ({record['bytes']} bytes)"
        for record in receipt["artifacts"]
    )
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
