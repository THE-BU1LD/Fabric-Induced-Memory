"""Durable evidence recording; recovery does not restore a tensor model or bank."""

import copy
from dataclasses import asdict
import hashlib
from pathlib import Path

from integrity import ev, VALIDATOR_SHA256
from journal import Journal, digest
from observations import Forecast, Observation, ObservationLedger
from recorder import PredictionRecorder


def implementation_identity():
    base = Path(__file__).resolve().parent
    return {
        **{
            name: hashlib.sha256((base / name).read_bytes()).hexdigest()
            for name in (
                "journal.py",
                "durable.py",
                "observations.py",
                "recorder.py",
                "integrity.py",
            )
        },
        "evidence.py": VALIDATOR_SHA256,
    }


def validate_header(header, kind):
    if (
        header.get("kind") != kind
        or header.get("implementation") != implementation_identity()
    ):
        raise ValueError("Wrong journal kind or changed recovery implementation")


def apply_record(recorder, event):
    op = event["operation"]
    if op == "prediction":
        ev.object_fields(event, ("operation", "method", "sample", "values"))
        recorder.record_prediction(event["method"], event["sample"], event["values"])
    elif op == "target":
        ev.object_fields(event, ("operation", "sample", "values"))
        recorder.record_target(event["sample"], event["values"])
    elif op == "failure":
        ev.object_fields(event, ("operation", "method", "sample", "error"))
        recorder.record_failure(event["method"], event["sample"], event["error"])
    else:
        raise ValueError("Unknown recorder operation")


class DurableRecorder:
    """Drop-in recorder for a fresh bridge; resumed records are scorer-only.

    The caller must retain each returned journal anchor outside this directory.
    A crash after publication but before delivery of the anchor is an explicit
    ambiguous commit, requiring inspection rather than automatic rollback.
    """

    @classmethod
    def create(cls, directory, plan_bytes, *, expected_sha256):
        recorder = PredictionRecorder(plan_bytes, expected_sha256=expected_sha256)
        journal = Journal(directory, create=True)
        try:
            journal.append(
                {
                    "kind": "prediction-recorder-v1",
                    "plan_utf8": plan_bytes.decode("utf-8"),
                    "plan_sha256": recorder.plan_sha256,
                    "implementation": implementation_identity(),
                }
            )
        except BaseException:
            journal.close()
            raise
        return cls(journal, recorder, resumed=False)

    @classmethod
    def resume(cls, directory, *, expected_anchor, expected_plan_sha256):
        journal = Journal(directory, expected_anchor=expected_anchor)
        try:
            header, *events = journal.events
            validate_header(header, "prediction-recorder-v1")
            if header["plan_sha256"] != ev.digest(expected_plan_sha256):
                raise ValueError("Expected plan identity changed")
            recorder = PredictionRecorder(
                header["plan_utf8"].encode(), expected_sha256=expected_plan_sha256
            )
            sealed = False
            for event in events:
                if sealed:
                    raise ValueError("Event after sealed export")
                if event["operation"] == "seal":
                    ev.object_fields(event, ("operation", "payload_sha256"))
                    if event["payload_sha256"] != digest(recorder.payload()):
                        raise ValueError("Sealed payload differs from replay")
                    sealed = True
                else:
                    apply_record(recorder, event)
            result = cls(journal, recorder, resumed=True)
            result._sealed = sealed
            return result
        except BaseException:
            journal.close()
            raise

    def __init__(self, journal, recorder, *, resumed):
        self.journal, self._recorder = journal, recorder
        self._sealed = False
        self._resumed = resumed

    @property
    def _plan(self):
        return copy.deepcopy(self._recorder._plan)

    @property
    def _methods(self):
        return self._recorder._methods

    @property
    def anchor(self):
        return self.journal.anchor

    def _key(self, sample):
        if self._resumed:
            raise ValueError(
                "Recovered recorder cannot attach to a fresh model; only scoring/export may resume"
            )
        return self._recorder._key(sample)

    def _commit(self, event):
        if self._sealed:
            raise ValueError("Recorder sealed; no further events")
        candidate = copy.deepcopy(self._recorder)
        apply_record(candidate, event)
        anchor = self.journal.append(event)
        self._recorder = candidate
        return anchor

    def record_prediction(self, method, sample, prediction):
        if self._resumed:
            raise ValueError(
                "Model predictions cannot resume without a validated tensor checkpoint"
            )
        return self._commit(
            {
                "operation": "prediction",
                "method": method,
                "sample": sample,
                "values": list(prediction),
            }
        )

    def record_target(self, sample, target):
        return self._commit(
            {"operation": "target", "sample": sample, "values": list(target)}
        )

    def record_failure(self, method, sample, error):
        return self._commit(
            {"operation": "failure", "method": method, "sample": sample, "error": error}
        )

    def payload(self):
        return self._recorder.payload()

    def finalize(self, path):
        # Seal before export. If output publication fails, recovery can retry the
        # exact sealed payload; it cannot substitute predictions or fill gaps.
        if not self._sealed:
            self.journal.append(
                {"operation": "seal", "payload_sha256": digest(self.payload())}
            )
            self._sealed = True
        return self._recorder.finalize(path)

    def close(self):
        self.journal.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class DurableObservationLedger:
    """Fresh observation staging with durable provenance; replay is audit-only."""

    @classmethod
    def create(cls, directory, *, source_sha256, split, model_state_sha256, **limits):
        ledger = ObservationLedger(**limits)
        identity = {
            "source_sha256": ev.digest(source_sha256),
            "split": ev.identifier(split),
            "model_state_sha256": ev.digest(model_state_sha256),
        }
        journal = Journal(directory, create=True)
        try:
            journal.append(
                {
                    "kind": "observation-ledger-v1",
                    "identity": identity,
                    "limits": limits,
                    "implementation": implementation_identity(),
                }
            )
        except BaseException:
            journal.close()
            raise
        return cls(journal, ledger, identity)

    def __init__(self, journal, ledger, identity):
        self.journal, self._ledger, self._identity = journal, ledger, identity

    @property
    def anchor(self):
        return self.journal.anchor

    def check_prediction_time(self, now):
        self._ledger.check_prediction_time(now)

    def register_many(self, forecasts, *, now):
        forecasts = list(forecasts)
        if any(f.state_id != self._identity["model_state_sha256"] for f in forecasts):
            raise ValueError("Forecast state differs from bound journal")
        candidate = copy.deepcopy(self._ledger)
        candidate.register_many(forecasts, now=now)
        self.journal.append(
            {
                "operation": "forecasts",
                "now": now,
                "forecasts": [asdict(f) for f in forecasts],
            }
        )
        self._ledger = candidate

    def observe(self, event, *, now):
        candidate = copy.deepcopy(self._ledger)
        ready = candidate.observe(event, now=now)
        self.journal.append(
            {"operation": "observation", "now": now, "event": asdict(event)}
        )
        self._ledger = candidate
        return ready

    def checksum(self):
        return self._ledger.checksum()

    def close(self):
        self.journal.close()


def audit_observations(directory, *, expected_anchor, expected_identity):
    """Return a verified scalar audit, never a resumable model/ledger handle."""
    with Journal(directory, expected_anchor=expected_anchor) as journal:
        header, *events = journal.events
        validate_header(header, "observation-ledger-v1")
        if header["identity"] != expected_identity:
            raise ValueError("Observation source/split/model identity mismatch")
        ledger = ObservationLedger(**header["limits"])
        ready = []
        for event in events:
            if event["operation"] == "forecasts":
                ev.object_fields(event, ("operation", "now", "forecasts"))
                forecasts = [Forecast(**f) for f in event["forecasts"]]
                if any(
                    f.state_id != expected_identity["model_state_sha256"]
                    for f in forecasts
                ):
                    raise ValueError("Forecast model state mismatch")
                ledger.register_many(forecasts, now=event["now"])
            elif event["operation"] == "observation":
                ev.object_fields(event, ("operation", "now", "event"))
                result = ledger.observe(Observation(**event["event"]), now=event["now"])
                if result is not None:
                    ready.append(asdict(result))
            else:
                raise ValueError("Unknown observation operation")
        return {
            "status": "PASS_OBSERVATION_REPLAY_ONLY",
            "anchor": journal.anchor,
            "ledger_sha256": ledger.checksum(),
            "completed_residuals": ready,
            "boundary": "No tensor bank write or model recovery is verified.",
        }
