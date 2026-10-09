from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fim_experiments.main import (  # noqa: E402
    build_system,
    get_device,
    load_yaml_config,
    select_benchmark,
    set_seed,
)
from fim_experiments.results.run_results import run  # noqa: E402


_CHECKPOINT_SPOOL_MAX_BYTES = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an existing FIM experiment checkpoint without retraining.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rollout_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help=(
            "New directory for this evaluation; existing paths are refused to "
            "preserve earlier results. Defaults to the run's evaluation directory."
        ),
    )
    parser.add_argument(
        "--weights", choices=("auto", "raw", "ema"), default="auto",
        help=(
            "auto uses the checkpoint's declared validation weights and keeps "
            "raw-weight behavior for legacy checkpoints without that metadata. "
            "Use raw or ema to select explicitly; the choice is recorded."
        ),
    )
    return parser.parse_args()


def _load_checkpoint(
    path: Path, device: torch.device, provenance: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if provenance is None:
        payload = torch.load(path, map_location=device)
    else:
        # Hash the same retained bytes that torch.load consumes, even if the
        # source path is replaced or rewritten before deserialization. Larger
        # snapshots spill to a temporary file instead of retaining an unbounded
        # checkpoint byte buffer in memory.
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source, tempfile.SpooledTemporaryFile(
            max_size=_CHECKPOINT_SPOOL_MAX_BYTES, mode="w+b",
        ) as snapshot:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
                snapshot.write(block)
            snapshot.seek(0)
            payload = torch.load(snapshot, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected checkpoint mapping, got {type(payload)!r}")
    if provenance is not None:
        provenance.update({
            "checkpoint_sha256": digest.hexdigest(),
            "checkpoint_size_bytes": size,
        })
    return payload


def _canonical_config_bytes(config: Dict[str, Any]) -> bytes:
    def check_keys(value: Any) -> None:
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("Resolved config keys must be strings for content identity")
            for item in value.values():
                check_keys(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check_keys(item)

    check_keys(config)
    return json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _resolve_config(
    args: argparse.Namespace, payload: Dict[str, Any],
    provenance: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    source_path = None
    if args.config is not None:
        config = load_yaml_config(args.config)
        source, source_path = "explicit_yaml", str(args.config)
    elif isinstance(payload.get("config"), dict):
        config = payload["config"]
        source = "checkpoint_embedded"
    else:
        candidate = args.checkpoint.parent.parent / "configs" / "resolved_config.yaml"
        if not candidate.exists():
            raise FileNotFoundError(
                "No experiment config was supplied or embedded. Pass --config explicitly, "
                f"or place resolved_config.yaml at {candidate}."
            )
        config = load_yaml_config(candidate)
        source, source_path = "sibling_yaml", str(candidate)

    if provenance is not None:
        provenance.update({
            "config_source": source,
            "config_source_path": source_path,
            "resolved_config_sha256": hashlib.sha256(_canonical_config_bytes(config)).hexdigest(),
        })
    return config


def _state_dict(payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("model_state_dict", "model"):
        state = payload.get(key)
        if isinstance(state, dict):
            return state
    raise KeyError("Checkpoint contains neither 'model_state_dict' nor trainer 'model' state.")


def _load_evaluation_weights(
    model: torch.nn.Module, payload: Dict[str, Any], requested: str = "auto",
) -> Dict[str, Any]:
    """Select raw/EMA inference weights without changing resume checkpoint state.

    Historical checkpoints do not identify which weights measured validation.
    Their automatic behavior stays raw, even if a config or EMA payload exists.
    EMA is a mapping of trainable named parameters, not a complete state dict;
    persistent buffers and frozen parameters still come from the raw state.
    """
    if requested not in ("auto", "raw", "ema"):
        raise ValueError("requested weights must be 'auto', 'raw', or 'ema'")
    declared = payload.get("validation_weight_source")
    if declared is not None and (
        not isinstance(declared, str) or declared not in ("raw", "ema")
    ):
        raise ValueError("Invalid checkpoint validation_weight_source metadata")

    if requested != "auto":
        selected, reason = requested, "explicit --weights selection"
    elif declared is not None:
        selected, reason = declared, "checkpoint validation-weight metadata"
    else:
        selected = "raw"
        reason = (
            "checkpoint does not declare validation weights"
            if "validation_weight_source" in payload
            else "legacy checkpoint without validation-weight metadata"
        )

    raw_state = _state_dict(payload)
    ema_state = payload.get("ema")
    parameters = {name: param for name, param in model.named_parameters() if param.requires_grad}
    if selected == "ema":
        if not isinstance(ema_state, dict):
            raise ValueError("EMA weights were selected but the checkpoint has no EMA mapping")
        if any(not isinstance(name, str) for name in ema_state):
            raise ValueError("EMA parameter names must be strings")
        missing = set(parameters) - set(ema_state)
        unexpected = set(ema_state) - set(parameters)
        if missing or unexpected:
            raise ValueError(
                f"EMA parameter mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        for name, parameter in parameters.items():
            value = ema_state[name]
            raw_value = raw_state.get(name)
            if not torch.is_tensor(value) or not torch.is_tensor(raw_value):
                raise ValueError(f"EMA/raw parameter {name!r} must be a tensor")
            if value.shape != parameter.shape or value.shape != raw_value.shape:
                raise ValueError(f"EMA parameter {name!r} has an incompatible shape")
            if value.dtype != raw_value.dtype:
                raise ValueError(f"EMA parameter {name!r} has a different dtype from raw weights")

    model.load_state_dict(raw_state, strict=True)
    if selected == "ema":
        # Copy through named parameters after loading buffers. Overlaying a
        # state_dict would let a later tied-parameter alias overwrite the EMA.
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(ema_state[name])

    return {
        "requested_weight_source": requested,
        "weight_source": selected,
        "weight_source_reason": reason,
        "validation_weight_source": declared,
    }


def main() -> None:
    args = parse_args()
    output_dir = (
        args.checkpoint.parent.parent / "evaluation"
        if args.output_dir is None else args.output_dir
    )
    # Refuse reuse before loading a checkpoint or constructing a benchmark.
    # is_symlink also catches dangling links, for which exists() is false.
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"Evaluation output already exists: {output_dir}. "
            "Choose a new --output_dir to preserve earlier results."
        )

    device = get_device(args.device)
    checkpoint_provenance: Dict[str, Any] = {}
    config_provenance: Dict[str, Any] = {}
    payload = _load_checkpoint(args.checkpoint, device, checkpoint_provenance)
    config = _resolve_config(args, payload, config_provenance)

    runtime = config.get("runtime", {}) or {}
    seed = int(runtime.get("seed", config.get("seed", 42)))
    set_seed(seed, deterministic=bool(runtime.get("deterministic", True)))

    benchmark = select_benchmark(config.get("benchmark", {}) or {})
    model = build_system(config, benchmark, device)
    weight_provenance = _load_evaluation_weights(model, payload, args.weights)

    # Exclusive creation also refuses a destination created after preflight.
    output_dir.mkdir(parents=True, exist_ok=False)

    eval_cfg = config.get("eval", {}) or {}
    benchmark_cfg = config.get("benchmark", {}) or {}
    rollout_steps = (
        int(args.rollout_steps)
        if args.rollout_steps is not None
        else int(eval_cfg.get("rollout_steps", benchmark_cfg.get("config", {}).get("steps", 50)))
    )

    metrics = run(
        model,
        benchmark,
        device=device,
        rollout_steps=rollout_steps,
        batch_size=args.batch_size,
        plot_path=output_dir / "rollout_comparison.pdf",
        save_results_path=output_dir / "metrics.json",
    )

    provenance = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config) if args.config is not None else "embedded-or-sibling-resolved-config",
        "device": str(device),
        "seed": seed,
        "rollout_steps": rollout_steps,
        "batch_size": int(args.batch_size),
        **checkpoint_provenance,
        **config_provenance,
        **weight_provenance,
    }
    with (output_dir / "evaluation_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)

    print(json.dumps({"metrics": metrics, "provenance": provenance}, indent=2))


if __name__ == "__main__":
    main()
