#!/usr/bin/env python3
"""Content-addressed validator for the persisted FIM delayed-recall mini-suite.

This script does not train or evaluate a model. It checks that the human-readable
summary and conservative qualitative claims remain exactly derivable from the
committed JSON artifact identified in a claim-lock specification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REQUIRED = (
    "model", "benchmark", "params", "train_last", "val_last",
    "rollout_mse", "final_step_mse", "rollout_score",
)
FLOAT_FIELDS = (
    "train_last", "val_last", "rollout_mse", "final_step_mse", "rollout_score",
)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| benchmark | model | params | train_last | val_last | rollout_mse | final_step_mse | rollout_score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['benchmark']} | {r['model']} | {r['params']} | "
            f"{r['train_last']} | {r['val_last']} | {r['rollout_mse']} | "
            f"{r['final_step_mse']} | {r['rollout_score']} |"
        )
    return "\n".join(lines) + "\n"

def validate_rows(rows: Any) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError("mini-suite JSON must be a non-empty list")
    identities: set[tuple[str, str]] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"row {i} is not an object")
        missing = [key for key in REQUIRED if key not in row]
        if missing:
            raise ValueError(f"row {i} missing keys: {missing}")
        ident = (str(row["benchmark"]), str(row["model"]))
        if ident in identities:
            raise ValueError(f"duplicate benchmark/model identity: {ident}")
        identities.add(ident)
        if isinstance(row["params"], bool) or int(row["params"]) <= 0:
            raise ValueError(f"invalid parameter count for {ident}: {row['params']!r}")
        for field in FLOAT_FIELDS:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {field} for {ident}")
        expected_score = 1.0 / (1.0 + float(row["rollout_mse"]))
        if not math.isclose(
            float(row["rollout_score"]), expected_score, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                f"rollout_score mismatch for {ident}: "
                f"stored={row['rollout_score']} expected={expected_score}"
            )

def claim_summary(rows: list[dict[str, Any]], claim: dict[str, Any]) -> dict[str, Any]:
    benchmark = claim["benchmark"]
    selected = [row for row in rows if row["benchmark"] == benchmark]
    if not selected:
        raise ValueError(f"claim benchmark absent: {benchmark}")
    by_model = {row["model"]: row for row in selected}
    if len(by_model) != len(selected):
        raise ValueError("duplicate model within claim benchmark")

    ranking = sorted(selected, key=lambda row: (float(row["rollout_mse"]), row["model"]))
    rank = {row["model"]: i + 1 for i, row in enumerate(ranking)}
    best = ranking[0]["model"]
    if best != claim["best_rollout_mse_model"]:
        raise ValueError(f"best-model claim drift: stored={best!r}")

    fim = claim["fim_model"]
    if fim not in by_model:
        raise ValueError(f"FIM claim model absent: {fim}")
    if rank[fim] != int(claim["fim_rollout_mse_rank"]):
        raise ValueError(f"FIM rank claim drift: stored={rank[fim]}")

    mlp = "ShapeAwareMLP"
    mlp_better = float(by_model[mlp]["rollout_mse"]) < float(by_model[fim]["rollout_mse"])
    if mlp_better is not bool(claim["shape_aware_mlp_better_than_fim"]):
        raise ValueError("ShapeAwareMLP/FIM ordering claim drift")

    for model in claim["fim_better_than"]:
        if not float(by_model[fim]["rollout_mse"]) < float(by_model[model]["rollout_mse"]):
            raise ValueError(f"FIM-better-than claim drift for {model}")

    return {
        "benchmark": benchmark,
        "ranking_by_rollout_mse": [
            {"rank": rank[row["model"]], "model": row["model"], "rollout_mse": row["rollout_mse"]}
            for row in ranking
        ],
        "best_rollout_mse_model": best,
        "fim_rollout_mse_rank": rank[fim],
        "fim_minus_best_rollout_mse": float(by_model[fim]["rollout_mse"]) - float(ranking[0]["rollout_mse"]),
        "fim_minus_shape_aware_mlp_rollout_mse": float(by_model[fim]["rollout_mse"]) - float(by_model[mlp]["rollout_mse"]),
    }

def verify(json_path: Path, md_path: Path, spec_path: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    artifacts = spec["source_artifacts"]

    json_key = "results/mini_processed/mini_suite_summary.json"
    md_key = "results/mini_processed/mini_suite_summary.md"
    got_json_sha = sha256_file(json_path)
    got_md_sha = sha256_file(md_path)
    if got_json_sha != artifacts[json_key]["sha256"]:
        raise ValueError(f"JSON SHA-256 mismatch: {got_json_sha}")
    if got_md_sha != artifacts[md_key]["sha256"]:
        raise ValueError(f"Markdown SHA-256 mismatch: {got_md_sha}")

    rows = json.loads(json_path.read_text(encoding="utf-8"))
    validate_rows(rows)

    regenerated = render_markdown(rows)
    committed = md_path.read_text(encoding="utf-8")
    if committed != regenerated:
        raise ValueError("committed Markdown is not an exact regeneration of JSON")

    summary = claim_summary(rows, spec["claim_lock"])
    return {
        "status": "PASS",
        "repository": spec["repository"],
        "source_commit": spec["source_commit"],
        "source_json": {
            "git_blob": artifacts[json_key]["git_blob"],
            "sha256": got_json_sha,
            "rows": len(rows),
        },
        "source_markdown": {
            "git_blob": artifacts[md_key]["git_blob"],
            "sha256": got_md_sha,
            "exactly_regenerated_from_json": True,
        },
        "claims": summary,
        "scientific_boundary": (
            "Saved-output verification only; no training/evaluation rerun and no causal "
            "memory-mechanism claim is established."
        ),
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    receipt = verify(args.json, args.markdown, args.spec)
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")

if __name__ == "__main__":
    main()
