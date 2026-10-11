"""Unambiguous JSON-object decoding for evidence and frozen-method admission.

Parsing is deliberately separate from each consumer's semantic validation. The
caller retains the original bytes for identity; this helper does not rewrite or
publish an artifact and does not grant permission to execute a scientific study.
"""

from __future__ import annotations

import json
import math
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite JSON number: {token}")
    return value


def _reject_constant(token: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {token}")


def decode_evidence_object(raw: bytes) -> dict[str, Any]:
    """Decode UTF-8 JSON with unique object keys and finite real numbers.

    Reject duplicate keys at every depth, including identical values and keys
    spelled with Unicode escapes. Integer values retain Python's exact integer
    representation. Valid float values keep ordinary JSON-decoder semantics.
    Semantic protocol/schema checks remain the responsibility of the caller.
    """
    if not isinstance(raw, bytes):
        raise TypeError("evidence JSON must be supplied as original bytes")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_float=_finite_float,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("evidence JSON must contain a top-level object")
    return value
