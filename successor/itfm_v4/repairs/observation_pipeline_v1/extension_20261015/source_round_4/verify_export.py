"""Recompute a saved bundle against separately frozen plan bytes, without torch."""

import argparse
import hashlib
import json
from pathlib import Path

from integrity import ev, VALIDATOR_SHA256
from journal import read_bounded
from recorder import PredictionRecorder


def verify(bundle, plan_bytes, *, expected_plan_sha256):
    recorder = PredictionRecorder(plan_bytes, expected_sha256=expected_plan_sha256)
    if (
        bundle["plan_sha256"] != recorder.plan_sha256
        or bundle["plan"] != recorder._plan
    ):
        raise ValueError("Bundle differs from externally frozen plan")
    if bundle["validator_sha256"] != VALIDATOR_SHA256:
        raise ValueError("Bundle validator identity mismatch")
    if bundle["claim_authorization"] != ev.CLAIM_BOUNDARY:
        raise ValueError("Bundle claim boundary changed")
    raw = json.dumps(bundle["evidence"], allow_nan=False, sort_keys=True).encode()
    if hashlib.sha256(raw).hexdigest() != bundle["evidence_sha256"]:
        raise ValueError("Raw evidence checksum mismatch")
    try:
        if len(raw) > ev.MAX_INPUT_BYTES:
            ev.reject("LIMIT", "Raw export exceeds validator 8 MiB limit")
        recomputed = ev.analyze(recorder._plan, bundle["evidence"])
    except ev.EvidenceError as error:
        recomputed = {
            "status": "FAIL_INTEGRITY",
            "error_code": error.code,
            "reason": str(error),
            "details": error.details,
        }
    if recomputed != bundle["result"]:
        raise ValueError("Saved analysis differs from raw evidence reanalysis")
    return {
        "status": "PASS_REANALYSIS",
        "evidence_status": recomputed["status"],
        "numerical_gates_pass": recomputed.get("numerical_gates_pass", False),
        "plan_sha256": recorder.plan_sha256,
        "evidence_sha256": bundle["evidence_sha256"],
        "validator_sha256": VALIDATOR_SHA256,
        "claim_authorization": ev.CLAIM_BOUNDARY,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError("Refusing existing verification receipt")
    try:
        # A bundle includes both plan and evidence; each is separately bounded.
        raw = read_bounded(args.bundle, 3 * ev.MAX_INPUT_BYTES)
        bundle = json.loads(
            raw,
            object_pairs_hook=unique_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
        result = verify(
            bundle,
            read_bounded(args.plan),
            expected_plan_sha256=args.expected_plan_sha256,
        )
        result["bundle_sha256"] = hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, TypeError, KeyError) as error:
        result = {"status": "FAIL_REANALYSIS", "reason": str(error)}
    ev.write_receipt(args.receipt, result)
    print(json.dumps(result))
    return 0 if result["status"] == "PASS_REANALYSIS" else 1


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


if __name__ == "__main__":
    raise SystemExit(main())
