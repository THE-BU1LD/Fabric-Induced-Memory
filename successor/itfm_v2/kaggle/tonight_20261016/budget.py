"""Local shared-account admission ledger; never a guarantee of platform shutdown.

Pass the SAME ledger path for every project using the account. Quota/session
observations must come from the platform, not these estimates. No cloud API,
secret handling, process launch, or GPU import occurs in this module.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import time

MAX_OBSERVATION_AGE_SECONDS = 300
ABSOLUTE_FLOOR_HOURS = 20.0
TERMINAL_RUN_STATES = {"SUCCEEDED", "FAILED", "INTERRUPTED", "SKIPPED_BUDGET"}


class AdmissionError(RuntimeError):
    """The next allocation or child run is not admitted."""


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def quota_policy(q0_hours):
    if q0_hours is None:
        return {"preferred_budget_hours": None, "possible_expanded_budget_hours": None,
                "expanded_authorized": False}
    q0 = _number(q0_hours, "Q0")
    return {"preferred_budget_hours": max(0.0, min(6.0, q0 - 24.0)),
            "possible_expanded_budget_hours": max(0.0, min(10.0, q0 - 20.0)),
            "expanded_authorized": False}


def utc_timestamp(value):
    if isinstance(value, str):
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timestamps must include an explicit timezone")
        return parsed.timestamp()
    return _number(value, "timestamp")


def _save(path, state):
    descriptor, temporary = tempfile.mkstemp(prefix=".account-budget-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(state, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class AccountBudget:
    def __init__(self, path, *, clock=time.time):
        self.path, self.clock = Path(path), clock

    @contextlib.contextmanager
    def _transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with (self.path.parent / (self.path.name + ".lock")).open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = json.loads(self.path.read_text()) if self.path.exists() else None
            except (ValueError, OSError) as error:
                raise AdmissionError("shared account ledger is unreadable or malformed") from error
            if state is not None:
                self._require_state(state)
            yield state
            if state is not None:
                self._require_state(state)
                _save(self.path, state)

    @classmethod
    def create(cls, path, *, account_id, q0_hours=None, observed_at=None,
               active_sessions=None, sessions_observed_at=None, deadline_utc,
               authorized_hours=None, reserve_fraction=0.10, clock=time.time):
        ledger = cls(path, clock=clock)
        if not isinstance(account_id, str) or not account_id.strip():
            raise ValueError("account_id must be a non-secret stable account identifier")
        if not 0.10 <= _number(reserve_fraction, "reserve_fraction") < 1:
            raise ValueError("reserve_fraction must be at least 10 percent and below 100 percent")
        requested = None if authorized_hours is None else _number(authorized_hours, "authorized_hours")
        policy = quota_policy(q0_hours)
        if q0_hours is not None and requested is not None and requested > policy["preferred_budget_hours"]:
            raise AdmissionError("expanded allocation is not authorized")
        state = {"schema_version": 1, "account_id": account_id, "q0_hours": q0_hours,
                 "quota_observation": None, "session_observation": None,
                 "deadline_utc_timestamp": utc_timestamp(deadline_utc),
                 "requested_hours": requested, "authorized_hours": 0.0,
                 "reserve_fraction": float(reserve_fraction), "sessions": {},
                 "platform_limit_guaranteed": False, "platform_shutdown_requires_confirmation": True}
        if q0_hours is not None:
            state["authorized_hours"] = policy["preferred_budget_hours"] if requested is None else requested
        ledger._set_observation(state, q0_hours, observed_at, active_sessions, sessions_observed_at)
        with ledger._transaction() as existing:
            if existing is not None:
                raise FileExistsError(ledger.path)
            _save(ledger.path, state)
        return ledger

    def _set_observation(self, state, quota_hours, observed_at, active_sessions, sessions_observed_at):
        now = self.clock()
        for value in (observed_at, sessions_observed_at):
            if value is not None and utc_timestamp(value) > now:
                raise ValueError("observations cannot come from the future")
        if quota_hours is not None and observed_at is None:
            raise ValueError("quota observation requires its platform observation time")
        if active_sessions is not None and (not isinstance(active_sessions, list) or
                any(not isinstance(item, str) or not item for item in active_sessions) or
                len(set(active_sessions)) != len(active_sessions) or sessions_observed_at is None):
            raise ValueError("known active sessions need unique IDs and an observation time")
        state["quota_observation"] = (None if quota_hours is None else
            {"hours": _number(quota_hours, "quota_hours"), "observed_at": utc_timestamp(observed_at)})
        state["session_observation"] = (None if active_sessions is None else
            {"active_session_ids": list(active_sessions), "observed_at": utc_timestamp(sessions_observed_at)})

    def observe(self, *, quota_hours, observed_at, active_sessions, sessions_observed_at):
        with self._transaction() as state:
            self._require_state(state)
            self._set_observation(state, quota_hours, observed_at, active_sessions, sessions_observed_at)
            if state["q0_hours"] is None and quota_hours is not None:
                state["q0_hours"] = _number(quota_hours, "quota_hours")
                preferred = quota_policy(quota_hours)["preferred_budget_hours"]
                requested = state["requested_hours"]
                state["authorized_hours"] = preferred if requested is None else min(preferred, requested)

    def _require_state(self, state):
        if (not isinstance(state, dict) or type(state.get("schema_version")) is not int
                or state["schema_version"] != 1):
            raise AdmissionError("shared account ledger is absent or unsupported")
        try:
            self._validate_state(state)
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise AdmissionError(f"shared account ledger failed validation: {error}") from error

    def _validate_state(self, state):
        current = _number(self.clock(), "current clock")
        def timestamp(value, name):
            value = _number(value, name)
            if value > current:
                raise ValueError(f"{name} is in the future or clock moved backwards")
            return value
        def identifier(value):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("invalid account/session/run identifier")
        identifier(state["account_id"])
        q0 = state["q0_hours"]
        policy = quota_policy(q0)
        authorized = _number(state["authorized_hours"], "authorized_hours")
        requested = state["requested_hours"]
        if requested is not None and authorized > _number(requested, "requested_hours"):
            raise ValueError("authorization exceeds requested allocation")
        if authorized > (policy["preferred_budget_hours"] or 0.0):
            raise ValueError("authorization exceeds preferred allocation")
        reserve = _number(state["reserve_fraction"], "reserve_fraction")
        if not 0.10 <= reserve < 1:
            raise ValueError("protected reserve must be at least 10 percent")
        _number(state["deadline_utc_timestamp"], "deadline")
        if state["platform_limit_guaranteed"] is not False or state["platform_shutdown_requires_confirmation"] is not True:
            raise ValueError("unsupported platform guarantee or shutdown policy")
        observation = state["quota_observation"]
        if observation is not None:
            _number(observation["hours"], "observed quota")
            timestamp(observation["observed_at"], "quota observation")
        observation = state["session_observation"]
        if observation is not None:
            timestamp(observation["observed_at"], "session observation")
            active = observation["active_session_ids"]
            if not isinstance(active, list) or len(set(active)) != len(active):
                raise ValueError("invalid active session inventory")
            for name in active:
                identifier(name)
        if not isinstance(state["sessions"], dict):
            raise ValueError("invalid session inventory")
        active_count = 0
        for session_id, session in state["sessions"].items():
            identifier(session_id)
            identifier(session["project"])
            if q0 is None:
                raise ValueError("session reservation cannot precede observed Q0")
            status = session["state"]
            if status not in {"PENDING_ALLOCATION", "ALLOCATED_OBSERVED", "STOP_REQUESTED_NOT_CONFIRMED", "STOPPED_CONFIRMED"}:
                raise ValueError("invalid session lifecycle state")
            active_count += status != "STOPPED_CONFIRMED"
            count = session["allocated_device_count"]
            multiplier = _number(session["quota_multiplier"], "quota_multiplier")
            if type(count) is not int or count < 1 or multiplier < count:
                raise ValueError("quota charging must cover every allocated device")
            start = timestamp(session["reserved_at"], "session reservation")
            last_activity = start
            allocated, stopped = session["platform_allocation_observed_at"], session["platform_stop_observed_at"]
            if allocated is not None:
                if timestamp(allocated, "allocation observation") < start:
                    raise ValueError("allocation preceded reservation")
                last_activity = max(last_activity, allocated)
            if status == "ALLOCATED_OBSERVED" and allocated is None:
                raise ValueError("allocated session lacks platform observation")
            if status == "PENDING_ALLOCATION" and allocated is not None:
                raise ValueError("pending session already has an allocation observation")
            if (status == "STOPPED_CONFIRMED") != (stopped is not None):
                raise ValueError("stop state disagrees with platform observation")
            if "stop_requested_at" in session:
                requested_at = timestamp(session["stop_requested_at"], "stop request")
                if requested_at < start:
                    raise ValueError("stop request preceded reservation")
                last_activity = max(last_activity, requested_at)
            if not isinstance(session["runs"], dict):
                raise ValueError("invalid run inventory")
            running = 0
            for run_id, run in session["runs"].items():
                identifier(run_id)
                run_status = run["state"]
                if run_status not in TERMINAL_RUN_STATES | {"PENDING", "RUNNING"}:
                    raise ValueError("invalid run lifecycle state")
                created = timestamp(run["created_at"], "run creation")
                seconds = _number(run["estimated_seconds"], "run estimate")
                if created < start or seconds <= 0:
                    raise ValueError("invalid run creation or estimate")
                last_activity = max(last_activity, created)
                if run_status == "SKIPPED_BUDGET":
                    identifier(run["reason"])
                if run_status not in {"PENDING", "SKIPPED_BUDGET"}:
                    reserved = timestamp(run["reserved_at"], "run reservation")
                    if allocated is None or reserved < max(created, allocated):
                        raise ValueError("run preceded observed allocation")
                    timeout = _number(run["timeout_seconds"], "run timeout")
                    debit = _number(run["quota_reserved_estimate_hours"], "run quota reservation")
                    margin = _number(run["export_shutdown_margin_seconds"], "shutdown margin")
                    if (timeout != seconds or not math.isclose(debit, seconds * multiplier / 3600)
                            or margin < max(30.0, authorized * reserve * 3600 / multiplier)):
                        raise ValueError("run reservation, timeout or protected margin is inconsistent")
                    last_activity = max(last_activity, reserved)
                    if run_status == "RUNNING":
                        running += 1
                    else:
                        finished = timestamp(run["finished_at"], "run finish")
                        if finished < reserved:
                            raise ValueError("run ended before reservation")
                        if run_status == "SUCCEEDED" and run["artifacts_verified"] is not True:
                            raise ValueError("success requires verified artifacts")
                        last_activity = max(last_activity, finished)
            if running > 1 or (running and status == "STOPPED_CONFIRMED"):
                raise ValueError("contradictory running children or stopped session")
            if stopped is not None and timestamp(stopped, "platform stop") < last_activity:
                raise ValueError("platform stop preceded recorded session activity")
        if active_count > 1:
            raise ValueError("concurrent account allocations are not admitted")

    def _estimates(self, state):
        now = self.clock()
        observation = state["quota_observation"]
        device_hours = wall_seconds = since_observation = allocated_quota_hours = 0.0
        for session in state["sessions"].values():
            start, end = session["reserved_at"], session.get("platform_stop_observed_at") or now
            elapsed = max(0.0, end - start)
            wall_seconds += elapsed
            device_hours += elapsed * session["allocated_device_count"] / 3600
            allocated_quota_hours += elapsed * session["quota_multiplier"] / 3600
            if observation is not None:
                since_observation += max(0.0, end - max(start, observation["observed_at"])) * session["quota_multiplier"] / 3600
        observed_debit = (None if observation is None or state["q0_hours"] is None else
                          max(0.0, state["q0_hours"] - observation["hours"]))
        conservative_spent = max(allocated_quota_hours, (observed_debit or 0.0) + since_observation)
        estimated_remaining = None if observation is None else observation["hours"] - since_observation
        return {"wall_clock_allocated_seconds_estimated": wall_seconds,
                "physical_device_hours_estimated": device_hours,
                "reported_quota_debit_observed": None,
                "quota_balance_decrease_derived_from_observations": observed_debit,
                "quota_spent_conservative_estimate": conservative_spent,
                "quota_remaining_estimate": estimated_remaining}

    def _guard_observations(self, state, *, expected_active=()):
        self._require_state(state)
        now = self.clock()
        for name in ("quota_observation", "session_observation"):
            observation = state[name]
            if observation is None or not 0 <= now - observation["observed_at"] <= MAX_OBSERVATION_AGE_SECONDS:
                raise AdmissionError(f"{name} unknown or older than five minutes")
        active = state["session_observation"]["active_session_ids"]
        if set(active) - set(expected_active):
            raise AdmissionError("other active sessions require coordination; none will be terminated")
        if state["authorized_hours"] <= 0:
            raise AdmissionError("preferred allocation is zero; explicit revised allocation required")
        if now >= state["deadline_utc_timestamp"]:
            raise AdmissionError("absolute deadline has passed")

    def reserve_session(self, session_id, project, allocated_device_count):
        if not isinstance(session_id, str) or not session_id or not isinstance(project, str) or not project:
            raise ValueError("session_id and project are required")
        if type(allocated_device_count) is not int or allocated_device_count < 1:
            raise ValueError("allocated device count must be positive")
        with self._transaction() as state:
            self._guard_observations(state)
            if session_id in state["sessions"]:
                raise AdmissionError("session ID already reserved")
            if any(s["state"] != "STOPPED_CONFIRMED" for s in state["sessions"].values()):
                raise AdmissionError("another ledger allocation is not confirmed stopped")
            estimate = self._estimates(state)
            reserve_hours = state["authorized_hours"] * state["reserve_fraction"]
            margin_seconds = max(30.0, reserve_hours * 3600 / allocated_device_count)
            if self.clock() + 60.0 + margin_seconds > state["deadline_utc_timestamp"]:
                raise AdmissionError("insufficient deadline margin for setup and session shutdown")
            if estimate["quota_remaining_estimate"] <= ABSOLUTE_FLOOR_HOURS or (
                    estimate["quota_spent_conservative_estimate"] + 60 * allocated_device_count / 3600 >
                    state["authorized_hours"] - reserve_hours):
                raise AdmissionError("protected quota reserve leaves no new-session capacity")
            state["sessions"][session_id] = {
                "project": project, "state": "PENDING_ALLOCATION", "reserved_at": self.clock(),
                "allocated_device_count": allocated_device_count, "quota_multiplier": allocated_device_count,
                "charging_behavior": "UNKNOWN; conservatively charge every allocated device",
                "platform_allocation_observed_at": None, "platform_stop_observed_at": None,
                "runs": {}, "quota_accounting": "ESTIMATED_UNTIL_RECONCILED"}

    def confirm_allocation(self, session_id, *, observed_at):
        with self._transaction() as state:
            self._require_state(state)
            session = state["sessions"][session_id]
            self._guard_observations(state, expected_active=(session_id,))
            observed = utc_timestamp(observed_at)
            if session["state"] != "PENDING_ALLOCATION" or not session["reserved_at"] <= observed <= self.clock():
                raise AdmissionError("allocation observation does not match the reserved session")
            session.update(state="ALLOCATED_OBSERVED", platform_allocation_observed_at=observed)

    def reserve_run(self, session_id, run_id, *, estimated_seconds, export_shutdown_seconds):
        seconds, requested_margin = _number(estimated_seconds, "estimated_seconds"), _number(export_shutdown_seconds, "export_shutdown_seconds")
        if seconds <= 0 or requested_margin < 30 or not isinstance(run_id, str) or not run_id:
            raise ValueError("run needs positive estimate, ID and at least 30 seconds shutdown margin")
        rejection = None
        result = None
        with self._transaction() as state:
            self._require_state(state)
            session = state["sessions"][session_id]
            if run_id in session["runs"]:
                raise AdmissionError("run ID already reserved; no double reservation or silent retry")
            if any(run["state"] == "RUNNING" for run in session["runs"].values()):
                raise AdmissionError("concurrent child runs are not admitted")
            run = {"state": "PENDING", "estimated_seconds": seconds, "created_at": self.clock()}
            session["runs"][run_id] = run
            try:
                self._guard_observations(state, expected_active=(session_id,))
                if session["state"] != "ALLOCATED_OBSERVED":
                    raise AdmissionError("platform allocation has not been observed")
                estimate, multiplier = self._estimates(state), session["quota_multiplier"]
                reserve_hours = state["authorized_hours"] * state["reserve_fraction"]
                margin = max(requested_margin, reserve_hours * 3600 / multiplier)
                debit = seconds * multiplier / 3600
                remaining = estimate["quota_remaining_estimate"]
                spent = estimate["quota_spent_conservative_estimate"]
                if spent + debit > state["authorized_hours"] - reserve_hours:
                    raise AdmissionError("training would consume reserved variance/export/shutdown quota")
                if remaining - debit - margin * multiplier / 3600 < ABSOLUTE_FLOOR_HOURS:
                    raise AdmissionError("estimated quota would cross the absolute 20-hour floor")
                if self.clock() + seconds + margin > state["deadline_utc_timestamp"]:
                    raise AdmissionError("run cannot finish plus export/shutdown before deadline")
                run.update(state="RUNNING", reserved_at=self.clock(), quota_reserved_estimate_hours=debit,
                           timeout_seconds=seconds, export_shutdown_margin_seconds=margin)
                result = dict(run)
            except AdmissionError as error:
                rejection = str(error)
                run.update(state="SKIPPED_BUDGET", reason=rejection)
        if rejection is not None:
            raise AdmissionError(rejection)
        return result

    def finish_run(self, session_id, run_id, *, status, artifacts_verified=False):
        if status not in {"SUCCEEDED", "FAILED", "INTERRUPTED"}:
            raise ValueError("terminal run state must be SUCCEEDED, FAILED or INTERRUPTED")
        with self._transaction() as state:
            self._require_state(state)
            run = state["sessions"][session_id]["runs"][run_id]
            if run["state"] != "RUNNING" or (status == "SUCCEEDED" and artifacts_verified is not True):
                raise AdmissionError("success requires a running cell and verified required artifacts")
            run.update(state=status, finished_at=self.clock(), artifacts_verified=artifacts_verified,
                       timeout_exceeded=self.clock() > run["reserved_at"] + run["timeout_seconds"],
                       deadline_exceeded=self.clock() > state["deadline_utc_timestamp"])

    def request_stop(self, session_id):
        with self._transaction() as state:
            self._require_state(state)
            session = state["sessions"][session_id]
            if session["state"] == "STOPPED_CONFIRMED":
                raise AdmissionError("session already confirmed stopped")
            session["state"] = "STOP_REQUESTED_NOT_CONFIRMED"
            session["stop_requested_at"] = self.clock()

    def confirm_session_stopped(self, session_id, *, observed_at):
        with self._transaction() as state:
            self._require_state(state)
            session = state["sessions"][session_id]
            observed = utc_timestamp(observed_at)
            if not session["reserved_at"] <= observed <= self.clock():
                raise AdmissionError("invalid platform stop observation")
            if any(run["state"] == "RUNNING" for run in session["runs"].values()):
                raise AdmissionError("record child interruption/completion before confirming platform stop")
            if session["state"] == "STOPPED_CONFIRMED":
                raise AdmissionError("session already confirmed stopped")
            session.update(state="STOPPED_CONFIRMED", platform_stop_observed_at=observed)

    def snapshot(self):
        with self._transaction() as state:
            self._require_state(state)
            result = json.loads(json.dumps(state))
            result.update(policy=quota_policy(state["q0_hours"]), estimates=self._estimates(state))
            result["all_campaign_sessions_confirmed_stopped"] = (
                bool(state["sessions"]) and all(s["state"] == "STOPPED_CONFIRMED" for s in state["sessions"].values()))
            return result
