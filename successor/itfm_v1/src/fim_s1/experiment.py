from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from .baselines import (
    ConvGRUPredictor,
    FiniteHistoryCNN,
    GatedDiagonalSSMPredictor,
    PeriodicTranslationPredictor,
    closest_width,
    trainable_parameter_count,
)
from .data import AliasedAdvectionCorpus, corrupt_records
from .metrics import summarize_rows, training_target_variance
from .model import InterventionalTensorFabricPredictor
from .provenance import atomic_json, environment_record, sha256_file, source_identity
from .training import (
    evaluate,
    evaluate_interventions,
    fit_model,
    set_deterministic_seed,
)


def _peak_memory_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _atomic_torch_save(payload: dict[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _resource_admission() -> dict[str, Any]:
    threshold = 512 * 1024 * 1024
    if sys.platform == "darwin":
        output = subprocess.run(
            ["vm_stat"], check=True, capture_output=True, text=True
        ).stdout.splitlines()
        page_size = 16384
        values: dict[str, int] = {}
        for line in output:
            if "page size of" in line:
                page_size = int(
                    line.split("page size of", 1)[1].split("bytes", 1)[0].strip()
                )
            elif ":" in line:
                key, raw = line.split(":", 1)
                values[key.strip()] = int(raw.strip().rstrip("."))
        available = page_size * (
            values.get("Pages free", 0) + values.get("Pages speculative", 0)
        )
        pageouts = values.get("Pageouts", 0)
    else:
        available = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        pageouts = None
    return {
        "status": "PASS" if available >= threshold else "BLOCK",
        "free_plus_speculative_bytes": available,
        "threshold_bytes": threshold,
        "pageouts_cumulative": pageouts,
        "checked_at_unix": time.time(),
    }


def _load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("status") != "development":
        raise ValueError("this runner accepts development protocols only")
    return protocol


def _make_corpus(config: dict[str, Any], smoke: bool) -> AliasedAdvectionCorpus:
    dataset = dict(config)
    if smoke:
        dataset.update(
            {"trajectory_count": 12, "time_steps": 12, "height": 8, "width": 8}
        )
    return AliasedAdvectionCorpus(
        trajectory_count=int(dataset["trajectory_count"]),
        time_steps=int(dataset["time_steps"]),
        height=int(dataset["height"]),
        width=int(dataset["width"]),
        channels=int(dataset["channels"]),
        context_length=int(dataset["context_length"]),
        train_fraction=float(dataset["train_fraction"]),
        validation_fraction=float(dataset["validation_fraction"]),
        seed=int(dataset["seed"]),
        observation_noise=float(dataset.get("observation_noise", 0.0)),
        speed=int(dataset.get("speed", 1)),
        smoothness=float(dataset.get("smoothness", 18.0)),
    )


def _batches_per_epoch(records, batch_size: int) -> int:
    counts = collections.Counter(row.event_time for row in records)
    return sum((count + batch_size - 1) // batch_size for count in counts.values())


def _make_model(
    name: str,
    protocol: dict[str, Any],
    corpus: AliasedAdvectionCorpus,
    target_parameters: int,
    overrides: dict[str, Any] | None = None,
) -> tuple[torch.nn.Module, bool, int | None]:
    model_cfg = dict(protocol["model"])
    model_cfg.update(overrides or {})
    context_channels = corpus.context_length * corpus.channels
    if name in {
        "itfm",
        "itfm_transport",
        "no_memory",
        "vector_broadcast",
        "sham_transport",
    } or name.startswith("itfm_"):
        return (
            InterventionalTensorFabricPredictor(
                context_channels=context_channels,
                output_channels=corpus.channels,
                height=corpus.height,
                width=corpus.width,
                hidden_channels=int(model_cfg["hidden_channels"]),
                key_channels=int(model_cfg["key_channels"]),
                memory_capacity=int(model_cfg["memory_capacity"]),
                topk=int(model_cfg["topk"]),
                temperature=float(model_cfg["temperature"]),
                max_feedback=float(model_cfg["max_feedback"]),
                memory_mode=(
                    "vector_broadcast" if name == "vector_broadcast" else "tensor"
                ),
                transport_mode=(
                    str(model_cfg["transport_mode"])
                    if name.startswith("itfm_transport") or name == "sham_transport"
                    else "identity"
                ),
                max_transport_shift=int(model_cfg["max_transport_shift"]),
            ),
            name != "no_memory",
            int(model_cfg["hidden_channels"]),
        )
    if name == "finite_history_cnn":
        model, width = closest_width(
            lambda candidate_width: FiniteHistoryCNN(
                context_channels=context_channels,
                output_channels=corpus.channels,
                hidden_channels=candidate_width,
            ),
            target_parameters,
        )
        return model, False, width
    if name == "conv_gru":
        model, width = closest_width(
            lambda candidate_width: ConvGRUPredictor(
                input_channels=corpus.channels,
                output_channels=corpus.channels,
                context_length=corpus.context_length,
                hidden_channels=candidate_width,
            ),
            target_parameters,
        )
        return model, False, width
    if name == "gated_diagonal_ssm":
        model, width = closest_width(
            lambda candidate_width: GatedDiagonalSSMPredictor(
                input_channels=corpus.channels,
                output_channels=corpus.channels,
                context_length=corpus.context_length,
                hidden_channels=candidate_width,
            ),
            target_parameters,
        )
        return model, False, width
    if name == "periodic_xcorr":
        return (
            PeriodicTranslationPredictor(
                input_channels=corpus.channels,
                context_length=corpus.context_length,
                max_shift=int(model_cfg["max_transport_shift"]),
            ),
            False,
            None,
        )
    raise ValueError(f"unsupported model: {name}")


def _robustness_records(
    protocol: dict[str, Any],
    validation_records,
    smoke: bool,
) -> dict[str, list]:
    conditions: dict[str, list] = {}
    for condition in protocol.get("robustness", []):
        name = str(condition["name"])
        kind = str(condition["kind"])
        if kind == "context_corruption":
            conditions[name] = corrupt_records(
                validation_records,
                noise_std=float(condition.get("noise_std", 0.0)),
                missing_fraction=float(condition.get("missing_fraction", 0.0)),
                seed=int(condition["seed"]),
            )
        elif kind == "speed_ood":
            dataset = dict(protocol["dataset"])
            dataset["speed"] = int(condition["speed"])
            dataset["seed"] = int(condition["seed"])
            shifted = _make_corpus(dataset, smoke)
            shifted.validate()
            conditions[name] = shifted.records("validation")
        else:
            raise ValueError(f"unsupported robustness condition: {kind}")
    return conditions


def _validated_resume(
    output: Path,
    *,
    protocol_hash: str,
    source_hash: str,
    smoke: bool,
    device: str,
) -> dict[str, Any] | None:
    path = output / "manifest.json"
    if not path.exists():
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("output directory is non-empty but has no manifest")
        return None
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {
        "protocol_sha256": protocol_hash,
        "source": source_hash,
        "smoke": smoke,
        "device": device,
    }
    observed = {
        "protocol_sha256": manifest.get("protocol_sha256"),
        "source": manifest.get("source", {}).get("sha256"),
        "smoke": manifest.get("smoke"),
        "device": manifest.get("device"),
    }
    if observed != expected:
        raise RuntimeError(f"resume identity mismatch: {observed} != {expected}")
    if manifest.get("failures"):
        raise RuntimeError("refusing silent retry of a manifest with retained failures")
    for run in manifest.get("runs", []):
        result_path = output / run["run_id"] / "result.json"
        if (
            not result_path.is_file()
            or sha256_file(result_path) != run["result_sha256"]
        ):
            raise RuntimeError(f"invalid cached result for {run['run_id']}")
        checkpoint = output / run["checkpoint"]
        predictions = output / run["predictions"]
        if (
            not checkpoint.is_file()
            or sha256_file(checkpoint) != run["checkpoint_sha256"]
        ):
            raise RuntimeError(f"invalid cached checkpoint for {run['run_id']}")
        if (
            not predictions.is_file()
            or sha256_file(predictions) != run["predictions_sha256"]
        ):
            raise RuntimeError(f"invalid cached predictions for {run['run_id']}")
        for path_key, hash_key in (
            ("robustness_predictions", "robustness_predictions_sha256"),
            ("intervention_predictions", "intervention_predictions_sha256"),
            ("sensitivity_predictions", "sensitivity_predictions_sha256"),
        ):
            if path_key not in run:
                continue
            artifact = output / run[path_key]
            if not artifact.is_file() or sha256_file(artifact) != run[hash_key]:
                raise RuntimeError(f"invalid cached {path_key} for {run['run_id']}")
    return manifest


def run(
    protocol_path: Path, output: Path, smoke: bool, device_name: str
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    protocol = _load_protocol(protocol_path)
    primary_model_name = str(protocol["primary_model"])
    corpus = _make_corpus(protocol["dataset"], smoke)
    dataset_checks = corpus.validate()
    train_records = corpus.records("train")
    validation_records = corpus.records("validation")
    robustness_records = _robustness_records(protocol, validation_records, smoke)
    normalization = training_target_variance([row.target for row in train_records])
    device = torch.device(device_name)
    if device.type not in {"cpu", "mps"}:
        raise ValueError("development compute is restricted to CPU or Apple MPS")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS was requested but is unavailable")
    training_cfg = protocol["training"]
    seeds = list(training_cfg["seeds"][:1] if smoke else training_cfg["seeds"])
    epochs = 1 if smoke else int(training_cfg["epochs"])

    protocol_hash = sha256_file(protocol_path)
    source = source_identity(project_root)
    resumed = _validated_resume(
        output,
        protocol_hash=protocol_hash,
        source_hash=source["sha256"],
        smoke=smoke,
        device=str(device),
    )
    manifest: dict[str, Any] = resumed or {
        "schema_version": 1,
        "status": "running",
        "started_at_unix": time.time(),
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "research_state_sha256": sha256_file(project_root / "RESEARCH_STATE.json"),
        "source": source,
        "environment": environment_record(),
        "smoke": smoke,
        "device": str(device),
        "primary_model": primary_model_name,
        "dataset_checks": dataset_checks,
        "dataset_identity": corpus.identity(),
        "normalization_variance": normalization,
        "runs": [],
        "failures": [],
    }
    manifest["status"] = "running"
    atomic_json(output / "manifest.json", manifest)

    reference_model, _, _ = _make_model(primary_model_name, protocol, corpus, 0)
    target_parameters = trainable_parameter_count(reference_model)
    del reference_model
    manifest["parameter_match_target"] = target_parameters
    completed_run_ids = {run["run_id"] for run in manifest["runs"]}

    run_cells = [
        {
            "name": model_name,
            "seed": int(seed),
            "overrides": {},
            "cell_class": "primary",
        }
        for seed in seeds
        for model_name in protocol["models"]
    ]
    if not smoke:
        run_cells.extend(
            {
                "name": str(cell["name"]),
                "seed": int(seeds[0]),
                "overrides": dict(cell["overrides"]),
                "cell_class": "sensitivity",
            }
            for cell in protocol.get("sensitivity", [])
        )
    expected = len(run_cells)
    manifest["expected_runs"] = expected
    manifest.setdefault("resource_admission_history", [])

    for cell in run_cells:
        model_name = cell["name"]
        seed = cell["seed"]
        retrieval_intervention = (
            "random_value" if model_name == "sham_transport" else "none"
        )
        run_id = f"{model_name}-seed-{seed}"
        if run_id in completed_run_ids:
            continue
        admission = _resource_admission()
        manifest["resource_admission_history"].append({"run_id": run_id, **admission})
        if admission["status"] != "PASS":
            manifest["status"] = "paused_resource"
            manifest["completed_runs"] = len(manifest["runs"])
            atomic_json(output / "manifest.json", manifest)
            return manifest
        started = time.perf_counter()
        try:
            set_deterministic_seed(int(seed))
            model, use_memory, selected_width = _make_model(
                model_name,
                protocol,
                corpus,
                target_parameters,
                overrides=cell["overrides"],
            )
            parameter_count = trainable_parameter_count(model)
            parameter_match_relative_error = (
                abs(parameter_count - target_parameters) / target_parameters
                if parameter_count and target_parameters
                else None
            )
            if model_name in {
                "finite_history_cnn",
                "conv_gru",
                "gated_diagonal_ssm",
            } and parameter_match_relative_error > float(
                protocol["model"]["parameter_match_relative_tolerance"]
            ):
                raise RuntimeError(
                    f"{model_name} exceeds the predeclared parameter-match tolerance"
                )
            if parameter_count:
                training_history = fit_model(
                    model,
                    train_records,
                    epochs=epochs,
                    batch_size=int(training_cfg["batch_size"]),
                    learning_rate=float(training_cfg["learning_rate"]),
                    weight_decay=float(training_cfg["weight_decay"]),
                    gradient_clip_norm=float(training_cfg["gradient_clip_norm"]),
                    device=device,
                    use_memory=use_memory,
                    intervention=retrieval_intervention,
                )
                effective_epochs = epochs
            else:
                model.to(device)
                training_history = []
                effective_epochs = 0
            rows = evaluate(
                model,
                validation_records,
                batch_size=int(training_cfg["batch_size"]),
                device=device,
                use_memory=use_memory,
                intervention=retrieval_intervention,
            )
            metrics = summarize_rows(rows, normalization)
            sensitivity_metrics = {}
            sensitivity_raw = {}
            if model_name == primary_model_name:
                original_topk = model.memory.topk
                original_max_shift = model.max_transport_shift
                for condition in protocol.get("evaluation_sensitivity", []):
                    name = str(condition["name"])
                    if condition["field"] == "topk":
                        model.memory.topk = int(condition["value"])
                    elif condition["field"] == "max_transport_shift":
                        model.max_transport_shift = int(condition["value"])
                    else:
                        raise ValueError(
                            f"unsupported sensitivity field: {condition['field']}"
                        )
                    sensitivity_rows = evaluate(
                        model,
                        validation_records,
                        batch_size=int(training_cfg["batch_size"]),
                        device=device,
                        use_memory=use_memory,
                    )
                    sensitivity_metrics[name] = summarize_rows(
                        sensitivity_rows, normalization
                    )
                    sensitivity_raw[name] = [
                        dataclasses.asdict(row) for row in sensitivity_rows
                    ]
                    model.memory.topk = original_topk
                    model.max_transport_shift = original_max_shift
            robustness_metrics = {}
            robustness_raw = {}
            for condition_name, condition_records in robustness_records.items():
                condition_rows = evaluate(
                    model,
                    condition_records,
                    batch_size=int(training_cfg["batch_size"]),
                    device=device,
                    use_memory=use_memory,
                    intervention=retrieval_intervention,
                )
                robustness_metrics[condition_name] = summarize_rows(
                    condition_rows, normalization
                )
                robustness_raw[condition_name] = [
                    dataclasses.asdict(row) for row in condition_rows
                ]
            intervention_metrics = None
            intervention_raw = None
            if model_name == primary_model_name:
                intervention_evaluation = evaluate_interventions(
                    model, validation_records, device=device
                )
                intervention_metrics = {
                    name: summarize_rows(intervention_rows, normalization)
                    for name, intervention_rows in intervention_evaluation[
                        "rows"
                    ].items()
                }
                direction_scores = intervention_evaluation[
                    "swap_donor_direction_scores"
                ]
                intervention_metrics["swap_donor_direction"] = {
                    "count": len(direction_scores),
                    "mean": (
                        sum(direction_scores) / len(direction_scores)
                        if direction_scores
                        else None
                    ),
                }
                intervention_raw = {
                    name: [dataclasses.asdict(row) for row in intervention_rows]
                    for name, intervention_rows in intervention_evaluation[
                        "rows"
                    ].items()
                }
                intervention_raw["swap_donor_direction_scores"] = direction_scores
            run_dir = output / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = run_dir / "model.pt"
            _atomic_torch_save(model.state_dict(), checkpoint)
            atomic_json(
                run_dir / "predictions.json",
                {"rows": [dataclasses.asdict(row) for row in rows]},
            )
            atomic_json(
                run_dir / "robustness_predictions.json",
                {"conditions": robustness_raw},
            )
            if intervention_raw is not None:
                atomic_json(run_dir / "intervention_predictions.json", intervention_raw)
            if sensitivity_raw:
                atomic_json(
                    run_dir / "sensitivity_predictions.json",
                    {"conditions": sensitivity_raw},
                )
            result = {
                "run_id": run_id,
                "status": "complete",
                "model": model_name,
                "cell_class": cell["cell_class"],
                "model_overrides": cell["overrides"],
                "seed": int(seed),
                "use_memory": use_memory,
                "retrieval_intervention": retrieval_intervention,
                "parameters": parameter_count,
                "parameter_match_target": target_parameters,
                "parameter_match_relative_error": parameter_match_relative_error,
                "selected_width": selected_width,
                "epochs": effective_epochs,
                "training_examples": len(train_records),
                "optimizer_steps": (
                    effective_epochs
                    * _batches_per_epoch(train_records, int(training_cfg["batch_size"]))
                ),
                "evaluation_rows": len(rows),
                "training_mse": training_history,
                "validation": metrics,
                "sensitivity": sensitivity_metrics,
                "robustness": robustness_metrics,
                "interventions": intervention_metrics,
                "runtime_seconds": time.perf_counter() - started,
                "process_peak_memory_bytes": _peak_memory_bytes(),
                "checkpoint": str(checkpoint.relative_to(output)),
                "checkpoint_sha256": sha256_file(checkpoint),
                "predictions": str((run_dir / "predictions.json").relative_to(output)),
                "predictions_sha256": sha256_file(run_dir / "predictions.json"),
                "robustness_predictions": str(
                    (run_dir / "robustness_predictions.json").relative_to(output)
                ),
                "robustness_predictions_sha256": sha256_file(
                    run_dir / "robustness_predictions.json"
                ),
            }
            if intervention_raw is not None:
                result["intervention_predictions"] = str(
                    (run_dir / "intervention_predictions.json").relative_to(output)
                )
                result["intervention_predictions_sha256"] = sha256_file(
                    run_dir / "intervention_predictions.json"
                )
            if sensitivity_raw:
                result["sensitivity_predictions"] = str(
                    (run_dir / "sensitivity_predictions.json").relative_to(output)
                )
                result["sensitivity_predictions_sha256"] = sha256_file(
                    run_dir / "sensitivity_predictions.json"
                )
            atomic_json(run_dir / "result.json", result)
            result["result_sha256"] = sha256_file(run_dir / "result.json")
            manifest["runs"].append(result)
        except Exception as error:
            failure = {
                "run_id": run_id,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "runtime_seconds": time.perf_counter() - started,
            }
            manifest["failures"].append(failure)
        atomic_json(output / "manifest.json", manifest)

    manifest["status"] = (
        "complete"
        if len(manifest["runs"]) == expected and not manifest["failures"]
        else "failed"
    )
    manifest["finished_at_unix"] = time.time()
    manifest["expected_runs"] = expected
    manifest["completed_runs"] = len(manifest["runs"])
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    manifest = run(args.protocol, args.output, args.smoke, args.device)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "completed_runs": manifest["completed_runs"],
                "expected_runs": manifest["expected_runs"],
                "failures": len(manifest["failures"]),
            },
            sort_keys=True,
        )
    )
    if manifest["status"] == "paused_resource":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
