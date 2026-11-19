"""Replay saved FIM development predictions without retraining or reselection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fim_sota.baseline_diagnostic import (
    Persistence,
    fit_linear,
    source_record,
    stream,
    verify_output,
)
from fim_sota.engineering import data_identity, digest
from fim_sota.engineering_runtime import _load_frozen_model, make_batches
from fim_sota.provenance import atomic_write_json, sha256_file


def replay(run: Path) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + 180
    torch.set_num_threads(1)
    verification = verify_output(run)
    summary = json.loads((run / "summary.json").read_text())
    if source_record()["sha256"] != summary["source_sha256"]:
        raise ValueError("replay source differs from frozen experiment source")

    def check_budget() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("180-second replay budget exhausted")

    counts = {"seeds": 0, "cells": 0, "forecasts": 0, "field_values": 0}
    for seed in summary["seeds"]:
        root = run / f"seed-{seed}"
        config = json.loads((root / "config.json").read_text())
        selection = json.loads((root / "selection.json").read_text())
        raw = json.loads((root / "raw_fixture.json").read_text())
        fixture = {
            split: {
                owner: torch.tensor(values, dtype=torch.float32)
                for owner, values in owners.items()
            }
            for split, owners in raw.items()
        }
        mlp, _ = _load_frozen_model(
            config,
            root / "mlp/checkpoints/best.pt",
            summary["source_sha256"],
            digest(config),
            data_identity(config),
        )
        models = {
            "persistence": Persistence(),
            "linear": fit_linear(
                make_batches(fixture, config, "train"),
                selection["linear_penalty"],
                (config["channels"], config["spatial_points"]),
            ),
            "mlp": mlp,
        }
        test = make_batches(fixture, config, "test", batch_size=1)
        for name, metadata in selection["validation_cells"].items():
            check_budget()
            expected = json.loads((root / "test" / f"{name}.json").read_text())
            actual, _ = stream(
                models[metadata["backbone"]],
                test,
                config,
                metadata["method"],
                metadata["penalty"],
                check_budget,
            )
            for actual_row, expected_row in zip(
                actual, expected["records"], strict=True
            ):
                for field in ("owner", "event", "split"):
                    if actual_row[field] != expected_row[field]:
                        raise ValueError(
                            f"replay identity mismatch: {seed}/{name}/{field}"
                        )
                for field in ("base", "correction", "target"):
                    a, b = (
                        np.asarray(actual_row[field]),
                        np.asarray(expected_row[field]),
                    )
                    if not np.array_equal(a, b):
                        raise ValueError(
                            f"replay value mismatch: {seed}/{name}/{field}"
                        )
                    counts["field_values"] += int(a.size)
                if actual_row["telemetry"] != expected_row["telemetry"]:
                    raise ValueError(
                        f"replay donor/solver telemetry mismatch: {seed}/{name}"
                    )
                counts["forecasts"] += 1
            counts["cells"] += 1
        counts["seeds"] += 1
    return {
        "status": "EXACT_SAVED_CHECKPOINT_AND_FIXED_RIDGE_REPLAY_PASSED",
        "counts": counts,
        "wall_seconds": time.monotonic() - started,
        "source_sha256": summary["source_sha256"],
        "replay_script_sha256": sha256_file(Path(__file__)),
        "artifact_verification": verification,
        "scope": (
            "Same-host forward prediction replay; no MLP retraining or reselection"
        ),
        "independent_retraining": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.run)
    atomic_write_json(args.receipt, result, overwrite=False)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
