"""Capture actual predictions and scorer truth separately; adjudicate raw exports."""

import copy
import hashlib
import json

from integrity import ev, VALIDATOR_SHA256


class PredictionRecorder:
    def __init__(self, plan_bytes, *, expected_sha256):
        if not isinstance(plan_bytes, bytes) or len(plan_bytes) > ev.MAX_INPUT_BYTES:
            raise ValueError("Bounded raw plan bytes required")
        self.plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
        if self.plan_sha256 != ev.digest(expected_sha256):
            raise ValueError("Plan SHA-256 mismatch")
        self._plan = ev.strict_json(plan_bytes)
        _, _, self._expected = ev.validate_plan(self._plan)
        self._methods = tuple(self._plan["methods"].values())
        self._predictions, self._targets, self._failures = {}, {}, {}
        self._finished = False
        self._bytes = len(plan_bytes)

    def _key(self, sample):
        if self._finished:
            raise ValueError("Recorder already finalized")
        ev.object_fields(sample, ("cell_id", "seed", "trajectory_id", "origin"))
        key = ev.sample_key(sample)
        if key not in self._expected:
            raise ValueError("Unexpected sample")
        return key

    def _reserve(self, value):
        size = len(json.dumps(value, allow_nan=False).encode())
        if self._bytes + size > ev.MAX_INPUT_BYTES:
            raise ValueError("Recorder 8 MiB budget exhausted")
        self._bytes += size

    def record_prediction(self, method, sample, prediction):
        key = self._key(sample)
        if method not in self._methods:
            raise ValueError("Unexpected method")
        identity = (method, *key)
        if (
            key in self._targets
            or identity in self._predictions
            or identity in self._failures
        ):
            raise ValueError("Prediction duplicate, failed, or supplied after scoring")
        prediction = ev.vector(list(prediction))
        other = [
            self._predictions[(m, *key)]
            for m in self._methods
            if (m, *key) in self._predictions
        ]
        if other and len(prediction) != len(other[0]):
            raise ValueError("Methods must predict the same vector shape")
        self._reserve([identity, prediction])
        self._predictions[identity] = prediction

    def record_target(self, sample, target):
        key = self._key(sample)
        if key in self._targets:
            raise ValueError("Duplicate scorer target")
        if any((m, *key) not in self._predictions for m in self._methods):
            raise ValueError("All method predictions must precede scorer truth")
        target = ev.vector(list(target))
        if any(len(self._predictions[(m, *key)]) != len(target) for m in self._methods):
            raise ValueError("Scorer target shape mismatch")
        self._reserve([key, target])
        self._targets[key] = target

    def record_failure(self, method, sample, error):
        key = self._key(sample)
        if (
            method not in self._methods
            or not isinstance(error, str)
            or not error
            or len(error) > 4096
        ):
            raise ValueError("Known method and bounded error text required")
        identity = (method, *key)
        if identity in self._failures or key in self._targets:
            raise ValueError("Duplicate failure or completed sample")
        self._reserve([identity, error])
        self._failures[identity] = error

    def payload(self):
        rows = []
        for key in sorted(self._expected):
            sample = dict(
                zip(("cell_id", "seed", "trajectory_id", "origin"), key, strict=True)
            )
            for method in self._methods:
                identity = (method, *key)
                row = dict(sample, method=method)
                if identity in self._failures:
                    row.update(status="FAILED", error=self._failures[identity])
                    if identity in self._predictions:
                        row["prediction"] = list(self._predictions[identity])
                elif identity not in self._predictions or key not in self._targets:
                    row.update(
                        status="INCOMPLETE", error="Prediction or scorer target missing"
                    )
                    if identity in self._predictions:
                        row["prediction"] = list(self._predictions[identity])
                else:
                    row.update(
                        status="PASS",
                        prediction=list(self._predictions[identity]),
                        target=list(self._targets[key]),
                    )
                rows.append(row)
        return {
            "schema_version": 1,
            "source_sha256": self._plan["source_sha256"],
            "data_sha256": self._plan["data_sha256"],
            "rows": rows,
        }

    def finalize(self, path):
        if self._finished:
            raise ValueError("Recorder already finalized")
        payload = self.payload()
        raw = json.dumps(payload, allow_nan=False, sort_keys=True).encode()
        try:
            if len(raw) > ev.MAX_INPUT_BYTES:
                ev.reject("LIMIT", "Raw export exceeds validator 8 MiB limit")
            result = ev.analyze(self._plan, payload)
        except ev.EvidenceError as error:
            result = {
                "status": "FAIL_INTEGRITY",
                "error_code": error.code,
                "reason": str(error),
                "details": error.details,
            }
        bundle = {
            "schema_version": 1,
            "plan": copy.deepcopy(self._plan),
            "plan_sha256": self.plan_sha256,
            "evidence_sha256": hashlib.sha256(raw).hexdigest(),
            "evidence_hash_encoding": "json sort_keys=True, default separators, UTF-8",
            "validator_sha256": VALIDATOR_SHA256,
            "evidence": payload,
            "result": result,
            "claim_authorization": ev.CLAIM_BOUNDARY,
        }
        ev.write_receipt(path, bundle)
        self._finished = True
        return copy.deepcopy(bundle)
