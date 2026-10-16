"""Guarded, isolated replay of the frozen synthetic study.

No installs, parameter search, network calls, or writes into original evidence.
The optional worker captures tensors in a separate instrumented evaluator.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path

import release


def execute(output, python):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("output exists; every replay needs a fresh directory")
    if (
        output.is_relative_to(release.ROOT) and not output.is_relative_to(release.PUB)
    ) or any(
        output.is_relative_to(release.PUB / name)
        for name in ("before", "frozen_source")
    ):
        raise ValueError("refusing output in frozen/source directory")
    checks = release.verify()
    if checks["failed"]:
        raise RuntimeError("input verification failed")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "RUNNING",
        "checked_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "steps": [],
        "manuscript": checks["manuscript"],
        "boundary": (
            "Same host and operator. This is not external independent rep"
            "roduction or an isolated dependency installation."
        ),
        "budget": {
            "max_total_execution_seconds": 300,
            "seconds_per_step": 300,
            "threads": 1,
            "seeds": list(release.SEEDS),
            "paid_compute": 0,
        },
    }
    admission = release.resource_admission()
    receipt["admission"] = admission
    if admission["status"] != "PASS":
        receipt["status"] = "BLOCKED_RESOURCE"
        release.write_new(output / "REPLAY_RECEIPT.json", receipt)
        return receipt
    workspace = output / "workspace"
    shutil.copytree(release.PUB / "frozen_source", workspace)
    env = {
        **os.environ,
        "PYTHONPATH": str(workspace / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    commands = [
        (
            "frozen_tests_and_experiment",
            [python, str(workspace / "scripts/run_pipeline.py")],
        ),
        (
            "raw_capture_and_reference_check",
            [python, str(Path(__file__).resolve()), "worker", "--output", str(output)],
        ),
    ]
    deadline = time.monotonic() + 300
    for name, command in commands:
        admission = release.resource_admission()
        if admission["status"] != "PASS":
            receipt["status"] = "BLOCKED_RESOURCE"
            receipt["failed_step"] = name
            receipt["latest_admission"] = admission
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            receipt["status"] = "FAILED"
            receipt["failed_step"] = name
            receipt["error"] = "total 300-second execution budget exhausted"
            break
        try:
            step = release.bounded_command(
                command, cwd=workspace, env=env, timeout=min(300, remaining)
            )
            step["name"] = name
        except OSError as exc:
            step = {
                "name": name,
                "command": command,
                "exit_code": None,
                "error": str(exc),
            }
        receipt["steps"].append(step)
        if step["exit_code"] != 0:
            receipt["status"] = "FAILED"
            break
    else:
        receipt["status"] = "PASS_LOCAL_REPLAY"
    release.write_new(output / "REPLAY_RECEIPT.json", receipt)
    return receipt


def worker(output):
    """Only invoked after admission; reuse generator, independently check outputs."""
    admission = release.resource_admission()
    if admission["status"] != "PASS":
        raise RuntimeError("resource guard blocked before Torch import")
    import torch

    from fim_v2 import evaluation
    from fim_v2.data import EpisodicAliasCorpus
    from fim_v2.evaluation import (
        calibrate_alpha,
        evaluate_stream,
        summarize,
        training_target_variance,
    )
    from fim_v2.experiment import make_corpus

    output = Path(output).resolve()
    workspace = output / "workspace"
    protocol = release.read(workspace / "protocols/development_v2.json")
    if sys.version_info[:2] != (3, 11) or torch.__version__ != "2.12.1":
        raise RuntimeError(
            "raw replay requires the recorded Python 3.11 / Torch 2.12.1 environment"
        )
    torch.set_num_threads(1)
    raw_dir = output / "raw"
    raw_dir.mkdir(exist_ok=False)
    # Instrument a copy in memory. Original file and numerical statements are unchanged.
    tree = ast.parse((workspace / "src/fim_v2/evaluation.py").read_text())
    inserted = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        for i, stmt in enumerate(list(node.body)):
            if (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "squared_error"
            ):
                capture = ast.parse(
                    "_capture.append({'owner_id': row.owner_id, 'family_id': row."
                    "family_id, 'event_time': row.event_time, 'query': row.query,"
                    " 'base': base.detach().clone(), 'prediction': prediction.det"
                    "ach().clone(), 'target': row.target.detach().clone(), 'corre"
                    "ction': correction.detach().clone(), 'donor_target': target_"
                    "index[(row.family_id, -row.regime)].detach().clone(), 'valid"
                    "': retrieval.valid, 'source_time': retrieval.source_time})"
                ).body[0]
                node.body.insert(i, capture)
                inserted += 1
    if inserted != 1:
        raise RuntimeError("unexpected evaluator structure; no instrumentation applied")
    ast.fix_missing_locations(tree)
    instrumentation = ast.unparse(tree)
    (raw_dir / "instrumented_evaluation.py").write_text(instrumentation + "\n")
    namespace = dict(evaluation.__dict__)
    namespace["_capture"] = []
    exec(compile(tree, str(raw_dir / "instrumented_evaluation.py"), "exec"), namespace)
    checks = {}
    for seed in release.SEEDS:
        if release.resource_admission()["status"] != "PASS":
            raise RuntimeError("resource guard blocked between seeds")
        corpus = make_corpus(protocol["dataset"], seed)
        variance = training_target_variance(corpus.records("train"))
        validation = evaluate_stream(
            corpus.records("validation"),
            corpus,
            protocol["memory"],
            mode="tensor_memory",
            alpha=1.0,
        )
        alpha = calibrate_alpha(validation)
        old = release.read(
            release.ROOT / f"results/development-v2/seed-{seed}/result.json"
        )
        checks[f"variance:{seed}"] = (
            abs(variance - old["normalization_variance"]) < 1e-10
        )
        checks[f"calibration:{seed}"] = alpha == old["validation_alpha"]
        checks[f"data_integrity:{seed}"] = corpus.validate() == old["dataset_checks"]
        for mode in protocol["controls"]:
            namespace["_capture"] = []
            result = namespace["evaluate_stream"](
                corpus.records("test"),
                corpus,
                protocol["memory"],
                mode=mode,
                alpha=1.0 if mode == "oracle_anchor" else alpha,
            )
            captured = namespace["_capture"]
            raw_path = raw_dir / f"seed-{seed}-{mode}.pt"
            torch.save(captured, raw_path)
            # Reopen raw tensor output, recompute errors, and compare every row.
            raw = torch.load(raw_path, weights_only=True, map_location="cpu")
            old_rows = release.read(
                release.ROOT
                / f"results/development-v2/seed-{seed}/predictions-{mode}.json"
            )
            passed = len(raw) == len(old_rows) == 594
            for actual, record in zip(raw, old_rows, strict=True):
                passed &= (actual["owner_id"], actual["event_time"]) == (
                    record["owner_id"],
                    record["event_time"],
                )
                squared = float(
                    (actual["prediction"] - actual["target"]).square().sum()
                )
                passed &= abs(squared - record["squared_error"]) <= 1e-10
                if mode == "paired_swap" and actual["query"] and actual["valid"]:
                    predicate = float(
                        (actual["prediction"] - actual["donor_target"]).square().sum()
                    ) < float((actual["base"] - actual["donor_target"]).square().sum())
                    passed &= predicate == record["donor_direction_success"]
            checks[f"raw_rows:{seed}:{mode}"] = bool(passed)
            summary = summarize(result, normalization_variance=variance)
            checks[f"summary:{seed}:{mode}"] = all(
                abs(summary[k] - v) <= 1e-10 for k, v in old["summaries"][mode].items()
            )
            if mode == "tensor_memory":
                # Separate exact-key oracle for this repeated-key construction only.
                bank = {}
                reference_pass = True
                data_rows = corpus.records("test")
                for row, actual in zip(data_rows, raw, strict=True):
                    eligible = [
                        item
                        for item in bank.get(row.owner_id, [])
                        if row.event_time - item[0] >= 4
                        and torch.equal(item[1], row.context)
                    ]
                    base = EpisodicAliasCorpus.base_prediction(row, 1)
                    predicted = base + alpha * eligible[0][2] if eligible else base
                    reference_pass &= torch.equal(predicted, actual["prediction"])
                    bank.setdefault(row.owner_id, []).append(
                        (row.event_time, row.context.clone(), row.target - base)
                    )
                checks[f"separate_exact_key_reference:{seed}"] = bool(reference_pass)
        del corpus
    receipt = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checked_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "checks": checks,
        "manuscript": release.verify()["manuscript"],
        "raw_files": {p.name: release.sha(p) for p in raw_dir.iterdir()},
        "python": sys.version,
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pytest", "setuptools")
        },
        "boundary": (
            "Local raw-output replay with reused frozen generator and ind"
            "ependently computed tensor errors/exact-key reference. Same "
            "host and operator; not external reproduction or a clean depe"
            "ndency installation."
        ),
    }
    release.write_new(output / "RAW_OUTPUT_RECEIPT.json", receipt)
    if not all(checks.values()):
        raise RuntimeError("raw replay discrepancy retained")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("execute", "worker"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--python", default="/opt/homebrew/opt/python@3.11/bin/python3.11"
    )
    args = parser.parse_args()
    result = (
        execute(args.output, args.python)
        if args.command == "execute"
        else worker(args.output)
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "steps")}))
    if result["status"].startswith(("BLOCKED", "FAILED")):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
