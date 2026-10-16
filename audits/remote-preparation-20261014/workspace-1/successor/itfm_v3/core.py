"""Standard-library contracts and inference summaries; no model imports."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def source_files():
    return sorted(
        p
        for p in ROOT.iterdir()
        if p.suffix in {".py", ".json", ".md", ".txt"}
        and p.name not in {"RESEARCH_STATE.json"}
    )


def identity():
    return {p.name: sha(p) for p in source_files()}


def validate(p):
    if not (len(p["seeds"]) == len(set(p["seeds"])) == 5):
        raise ValueError(
            "invalid protocol: len(p['seeds']) == len(set(p['seeds'])) == 5"
        )
    if not (p["memory"]["minimum_age"] <= p["memory"]["recent_max_age"]):
        raise ValueError(
            "invalid protocol: p['memory']['minimum_age'] <= p['memory']['recent_max_age']"
        )
    if not (p["memory"]["recent_max_age"] == p["training"]["history"]):
        raise ValueError(
            "invalid protocol: p['memory']['recent_max_age'] == p['training']['history']"
        )
    if not (p["memory"]["topk"] <= p["memory"]["capacity"]):
        raise ValueError(
            "invalid protocol: p['memory']['topk'] <= p['memory']['capacity']"
        )
    if not (p["data"]["warmup"] >= p["training"]["attention_history"]):
        raise ValueError(
            "invalid protocol: p['data']['warmup'] >= p['training']['attention_history']"
        )
    if not (p["data"]["steps"] > p["data"]["warmup"]):
        raise ValueError("invalid protocol: p['data']['steps'] > p['data']['warmup']")
    if not (p["data"]["cue_dim"] == p["data"]["value_dim"] == 16):
        raise ValueError(
            "invalid protocol: p['data']['cue_dim'] == p['data']['value_dim'] == 16"
        )
    if not (p["data"]["validation_owners"] == p["data"]["evaluation_owners"] == 16):
        raise ValueError(
            "invalid protocol: p['data']['validation_owners'] == p['data']['evaluation_owners'] == 16"
        )
    if not (len(set((c["name"] for c in p["conditions"]))) == 6):
        raise ValueError(
            "invalid protocol: len(set((c['name'] for c in p['conditions']))) == 6"
        )
    if not (
        {c["name"] for c in p["conditions"]}
        == {"clean", "noisy", "drift", "distractors", "capacity", "combined"}
    ):
        raise ValueError(
            "invalid protocol: {c['name'] for c in p['conditions']} == {'clean', 'noisy', 'drift', 'distractors', 'capacity', 'combined'}"
        )
    if not (any((c["anchors"] > p["memory"]["capacity"] for c in p["conditions"]))):
        raise ValueError(
            "invalid protocol: any((c['anchors'] > p['memory']['capacity'] for c in p['conditions']))"
        )
    if not (
        set(p["baselines"])
        == {
            "base",
            "last_value",
            "knn1",
            "knn4",
            "recent",
            "finite_history",
            "gru",
            "attention",
        }
    ):
        raise ValueError(
            "invalid protocol: set(p['baselines']) == {'base', 'last_value', 'knn1', 'knn4', 'recent', 'finite_history', 'gru', 'attention'}"
        )
    if not (
        set(p["interventions"])
        == {"erase", "sign_flip", "spatial_mean", "value_permutation"}
    ):
        raise ValueError(
            "invalid protocol: set(p['interventions']) == {'erase', 'sign_flip', 'spatial_mean', 'value_permutation'}"
        )
    if not (p["budget"]["minimum_free_mib"] == 512):
        raise ValueError("invalid protocol: p['budget']['minimum_free_mib'] == 512")
    if not (p["budget"]["paid_compute"] == 0 and p["budget"]["max_campaigns"] == 1):
        raise ValueError(
            "invalid protocol: p['budget']['paid_compute'] == 0 and p['budget']['max_campaigns'] == 1"
        )
    if not (not p["protected_evaluation"]["authorized"]):
        raise ValueError(
            "invalid protocol: not p['protected_evaluation']['authorized']"
        )
    return p


def admission():
    def sample():
        vm = subprocess.check_output(["vm_stat"], text=True)
        page = int(re.search(r"page size of (\d+)", vm)[1])

        def value(k):
            return int(re.search(re.escape(k) + r":\s+(\d+)", vm)[1])

        swap = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
        return {
            "free_bytes": page * (value("Pages free") + value("Pages speculative")),
            "pageouts": value("Pageouts"),
            "swap_mib": float(re.search(r"used = ([\d.]+)M", swap)[1]),
        }

    try:
        a = sample()
        time.sleep(0.5)
        b = sample()
        ok = min(a["free_bytes"], b["free_bytes"]) >= 512 * 1024**2
        ok &= a["pageouts"] == b["pageouts"] and b["swap_mib"] <= a["swap_mib"]
        return {"status": "PASS" if ok else "BLOCKED", "samples": [a, b]}
    except (OSError, subprocess.SubprocessError, ValueError, TypeError) as exc:
        return {"status": "BLOCKED", "reason": str(exc)}


def exact_sign_flip(differences):
    """Two-sided paired test across independent seeds, not individual timesteps."""
    if not differences or not all(math.isfinite(x) for x in differences):
        raise ValueError("finite, nonempty paired seed differences required")
    observed = abs(sum(differences))
    counts = sum(
        abs(sum(s * x for s, x in zip(signs, differences))) >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=len(differences))
    )
    return counts / 2 ** len(differences)


def decide(metrics, thresholds):
    base, full, oracle = (metrics[k] for k in ("base", "tensor", "oracle"))
    gap = base - oracle
    checks = {
        "oracle_gap": base > 0
        and gap / base >= thresholds["minimum_oracle_relative_gain"],
        "baseline_gain": metrics["strongest"] > 0
        and (metrics["strongest"] - full) / metrics["strongest"]
        >= thresholds["minimum_baseline_relative_gain"],
        "gap_recovery": gap > 0
        and (base - full) / gap >= thresholds["minimum_oracle_gap_recovery"],
        "erasure": base - full > 0
        and (metrics["erase"] - full) / (base - full)
        >= thresholds["minimum_erasure_gain_removal"],
        "nonprimary": metrics["nonprimary_delta"]
        <= thresholds["maximum_nonprimary_nmse_degradation"],
        "permutation_coverage": metrics["permutation_fraction"]
        >= thresholds["minimum_valid_permutation_fraction"],
        "permutation_effect": metrics["value_permutation"] - full
        >= thresholds["minimum_permutation_error_increase"],
    }
    return {
        "status": "SUPPORTED_BOUNDED_DEVELOPMENT"
        if all(checks.values())
        else "NOT_SUPPORTED",
        "checks": checks,
    }
