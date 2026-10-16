from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .analysis import adjudicate, safe_relative_gain
from .data import EpisodicAliasCorpus
from .evaluation import (
    calibrate_alpha,
    evaluate_stream,
    summarize,
    training_target_variance,
)
from .resources import resource_admission


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def source_identity(root: Path) -> dict[str, object]:
    candidates = [
        root / "PROJECT_CONTRACT.md",
        root / "METHODOLOGY.md",
        root / "pyproject.toml",
        root / "protocols/development_v2.json",
    ]
    candidates.extend((root / "src").rglob("*.py"))
    candidates.extend((root / "tests").rglob("*.py"))
    candidates.extend((root / "scripts").rglob("*.py"))
    paths = sorted(path for path in candidates if path.is_file())
    files = [
        {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
        for path in paths
    ]
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item["path"]).encode())
        digest.update(str(item["sha256"]).encode())
    return {"sha256": digest.hexdigest(), "files": files}


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("protocol_id") != "itfm-development-v2":
        raise ValueError("unexpected protocol identity")
    if payload.get("status") != "development":
        raise ValueError("runner accepts development protocols only")
    return payload


def make_corpus(dataset: dict[str, Any], seed: int) -> EpisodicAliasCorpus:
    return EpisodicAliasCorpus(
        family_count=int(dataset["family_count"]),
        anchor_count=int(dataset["anchor_count"]),
        query_repeats=int(dataset["query_repeats"]),
        context_length=int(dataset["context_length"]),
        height=int(dataset["height"]),
        width=int(dataset["width"]),
        channels=int(dataset["channels"]),
        washout_events=int(dataset["washout_events"]),
        residual_scale=float(dataset["residual_scale"]),
        smoothness=float(dataset["smoothness"]),
        train_fraction=float(dataset["train_fraction"]),
        validation_fraction=float(dataset["validation_fraction"]),
        seed=int(seed),
    )


def serialize_rows(result) -> list[dict[str, object]]:
    return [dataclasses.asdict(row) for row in result.rows]


def run_seed(protocol: dict[str, Any], seed: int, output: Path) -> dict[str, object]:
    corpus = make_corpus(protocol["dataset"], seed)
    corpus_checks = corpus.validate()
    train_records = corpus.records("train")
    validation_records = corpus.records("validation")
    test_records = corpus.records("test")
    variance = training_target_variance(train_records)
    memory_config = protocol["memory"]

    validation_base = evaluate_stream(
        validation_records, corpus, memory_config, mode="no_memory", alpha=0.0
    )
    validation_oracle = evaluate_stream(
        validation_records, corpus, memory_config, mode="oracle_anchor", alpha=1.0
    )
    validation_base_summary = summarize(
        validation_base, normalization_variance=variance
    )
    validation_oracle_summary = summarize(
        validation_oracle, normalization_variance=variance
    )
    oracle_gain = safe_relative_gain(
        validation_base_summary["query_nmse"],
        validation_oracle_summary["query_nmse"],
    )
    full_context_static = max(
        float(
            (
                row.context[-corpus.channels :]
                - row.context[-2 * corpus.channels : -corpus.channels]
            )
            .abs()
            .max()
            .item()
        )
        for row in validation_records
    )
    identifiability = {
        "oracle_relative_gain": oracle_gain,
        "validation_base_query_nmse": validation_base_summary["query_nmse"],
        "validation_oracle_query_nmse": validation_oracle_summary["query_nmse"],
        "last_two_visible_frames_max_abs_difference": full_context_static,
        "gates": {
            "full_context_alias": all(
                float(corpus_checks[f"{split}_full_context_max_abs_difference"]) == 0.0
                for split in ("train", "validation", "test")
            ),
            "future_divergence": all(
                float(corpus_checks[f"{split}_paired_future_min_mse"]) > 0.0
                for split in ("train", "validation", "test")
            ),
            "age_beyond_context": all(
                int(corpus_checks[f"{split}_minimum_calibration_age"])
                > corpus.context_length
                for split in ("train", "validation", "test")
            ),
            "analytical_zero_shift_is_base": full_context_static == 0.0,
            "oracle_gap": oracle_gain
            >= float(protocol["identifiability"]["minimum_oracle_relative_gain"]),
        },
    }
    if not all(identifiability["gates"].values()):
        raise RuntimeError(f"identifiability gate failed: {identifiability}")

    validation_primary = evaluate_stream(
        validation_records,
        corpus,
        memory_config,
        mode="tensor_memory",
        alpha=1.0,
    )
    alpha = calibrate_alpha(validation_primary)
    modes = list(protocol["controls"])
    summaries: dict[str, dict[str, float]] = {}
    artifact_hashes: dict[str, str] = {}
    seed_root = output / f"seed-{seed}"
    for mode in modes:
        mode_alpha = 1.0 if mode == "oracle_anchor" else alpha
        result = evaluate_stream(
            test_records,
            corpus,
            memory_config,
            mode=mode,
            alpha=mode_alpha,
        )
        summaries[mode] = summarize(result, normalization_variance=variance)
        path = seed_root / f"predictions-{mode}.json"
        atomic_json(path, serialize_rows(result))
        artifact_hashes[str(path.relative_to(output))] = sha256_file(path)

    decision = adjudicate(summaries, protocol["decision"])
    result_payload: dict[str, object] = {
        "schema_version": 1,
        "seed": seed,
        "dataset_identity": corpus.identity(),
        "dataset_checks": corpus_checks,
        "normalization_variance": variance,
        "validation_alpha": alpha,
        "identifiability": identifiability,
        "summaries": summaries,
        "decision": decision,
        "prediction_artifacts": artifact_hashes,
    }
    result_path = seed_root / "result.json"
    atomic_json(result_path, result_payload)
    result_payload["result_sha256"] = sha256_file(result_path)
    return result_payload


def run(protocol_path: Path, output: Path) -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[2]
    protocol = load_protocol(protocol_path)
    source = source_identity(project_root)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "started_at_unix": time.time(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "source": source,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "runs": [],
        "failures": [],
        "resource_admission_history": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    atomic_json(manifest_path, manifest)

    for seed in protocol["dataset"]["seeds"]:
        admission = resource_admission()
        manifest["resource_admission_history"].append(admission)
        if admission["status"] != "PASS":
            manifest["status"] = "paused_resource"
            manifest["finished_at_unix"] = time.time()
            atomic_json(manifest_path, manifest)
            return manifest
        try:
            run_result = run_seed(protocol, int(seed), output)
            manifest["runs"].append(
                {
                    "seed": int(seed),
                    "status": run_result["decision"]["status"],
                    "result": f"seed-{seed}/result.json",
                    "result_sha256": run_result["result_sha256"],
                }
            )
        except Exception as error:
            manifest["failures"].append(
                {
                    "seed": int(seed),
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            manifest["status"] = "failed"
            manifest["finished_at_unix"] = time.time()
            atomic_json(manifest_path, manifest)
            raise
        atomic_json(manifest_path, manifest)

    statuses = [run["status"] for run in manifest["runs"]]
    manifest["status"] = (
        "PASS_BOUNDED_DEVELOPMENT"
        if statuses and all(status == "PASS_BOUNDED_DEVELOPMENT" for status in statuses)
        else "FAIL"
    )
    manifest["completed_runs"] = len(manifest["runs"])
    manifest["expected_runs"] = len(protocol["dataset"]["seeds"])
    manifest["finished_at_unix"] = time.time()
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=Path("protocols/development_v2.json")
    )
    parser.add_argument("--output", type=Path, default=Path("results/development-v2"))
    arguments = parser.parse_args()
    result = run(arguments.protocol.resolve(), arguments.output.resolve())
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    if result["status"] == "paused_resource":
        raise SystemExit(3)
    if result["status"] != "PASS_BOUNDED_DEVELOPMENT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
