from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(config: Dict[str, Any]) -> str:
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping, got {type(data)!r}")
    return data


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - old PyTorch only
        raise RuntimeError(
            "PyTorch must support torch.load(..., weights_only=True); "
            "unrestricted pickle loading is intentionally refused."
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Expected checkpoint mapping, got {type(payload)!r}")
    return payload


def _state_dict_key(payload: Dict[str, Any]) -> str:
    for key in ("model_state_dict", "model"):
        if isinstance(payload.get(key), dict):
            return key
    raise KeyError("Checkpoint contains neither 'model_state_dict' nor trainer 'model' state.")


def _declared_source_sha(payload: Dict[str, Any]) -> str | None:
    for key in ("source_git_sha", "git_sha", "source_sha", "commit_sha"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def verify_checkpoint_identity(
    checkpoint: Path,
    config_path: Path | None = None,
    expected_source_sha: str | None = None,
    require_source_sha: bool = False,
) -> Dict[str, Any]:
    payload = _load_checkpoint(checkpoint)
    state_key = _state_dict_key(payload)

    embedded = payload.get("config")
    if embedded is not None and not isinstance(embedded, dict):
        raise TypeError(f"Embedded checkpoint config must be a mapping, got {type(embedded)!r}")

    external: Dict[str, Any] | None = None
    if config_path is not None:
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")
        external = _load_yaml(config_path)

    embedded_sha = _config_sha256(embedded) if isinstance(embedded, dict) else None
    external_sha = _config_sha256(external) if external is not None else None
    config_match = None
    if embedded_sha is not None and external_sha is not None:
        config_match = embedded_sha == external_sha
        if not config_match:
            raise ValueError(
                "External config does not match checkpoint-embedded config: "
                f"external_sha256={external_sha} embedded_sha256={embedded_sha}."
            )

    source_sha = _declared_source_sha(payload)
    if require_source_sha and source_sha is None:
        raise ValueError("Checkpoint does not declare a source Git SHA.")
    if expected_source_sha is not None:
        if source_sha is None:
            raise ValueError(
                "An expected source SHA was supplied, but the checkpoint declares no source Git SHA."
            )
        if source_sha != expected_source_sha:
            raise ValueError(
                f"Checkpoint source SHA mismatch: expected={expected_source_sha} declared={source_sha}."
            )

    return {
        "schema_version": 1,
        "checkpoint": {
            "sha256": _sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "state_dict_key": state_key,
            "declared_source_git_sha": source_sha,
        },
        "config": {
            "embedded_normalized_sha256": embedded_sha,
            "external_normalized_sha256": external_sha,
            "external_file_sha256": _sha256_file(config_path) if config_path is not None else None,
            "embedded_external_match": config_match,
        },
        "load_policy": "torch.load(weights_only=True)",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify FIM checkpoint/config identity without running training or evaluation."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--expected-source-sha", type=str, default=None)
    parser.add_argument("--require-source-sha", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = verify_checkpoint_identity(
        checkpoint=args.checkpoint,
        config_path=args.config,
        expected_source_sha=args.expected_source_sha,
        require_source_sha=args.require_source_sha,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
