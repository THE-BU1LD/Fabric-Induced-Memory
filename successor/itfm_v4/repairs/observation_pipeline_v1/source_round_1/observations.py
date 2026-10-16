"""Bounded causal observation staging, independent of scorer truth and torch."""

from dataclasses import asdict, dataclass
import hashlib
import json

from integrity import ev


@dataclass(frozen=True)
class Forecast:
    forecast_id: str
    owner: str
    origin: int
    target_times: tuple[int, ...]
    base: tuple[float, ...]
    key: tuple[float, ...]
    state_id: str


@dataclass(frozen=True)
class Observation:
    event_id: str
    forecast_id: str
    owner: str
    arrived_at: int
    indices: tuple[int, ...]
    values: tuple[float, ...]
    available_at: tuple[int, ...]


@dataclass(frozen=True)
class ReadyResidual:
    forecast: Forecast
    residual: tuple[float, ...]
    available_at: int


def project_observed(*, event_id, forecast_id, owner, arrived_at, values, mask,
                     available_at):
    """Environment boundary: never index, convert or validate hidden values.

    The predictor receives only the resulting sparse Observation. Do not pass
    scorer arrays to a model merely because this projection exists.
    """
    if not mask or len(mask) != len(values) or len(mask) != len(available_at):
        raise ValueError("Mask, values and availability must have equal nonzero lengths")
    if any(type(x) is not bool for x in mask):
        raise ValueError("Observation mask must be strictly boolean")
    indices = tuple(i for i, visible in enumerate(mask) if visible)
    if not indices:
        raise ValueError("An observation event must reveal at least one coordinate")
    return Observation(event_id, forecast_id, owner, arrived_at, indices,
                       tuple(ev.finite(values[i]) for i in indices),
                       tuple(ev.integer(available_at[i], "availability", 0) for i in indices))


class ObservationLedger:
    def __init__(self, *, max_forecasts=1024, max_events=8192, max_scalars=100_000):
        self.max_forecasts = ev.integer(max_forecasts, "max forecasts", 1)
        self.max_events = ev.integer(max_events, "max events", 1)
        self.max_scalars = ev.integer(max_scalars, "max scalars", 1)
        self._forecasts = {}
        self._values = {}
        self._events = {}
        self._complete = set()
        self._clock = -1
        self._revealed = False
        self._scalars = 0

    def check_prediction_time(self, now):
        ev.integer(now, "prediction time", 0)
        if now < self._clock or (now == self._clock and self._revealed):
            raise ValueError("Predictions must precede observations at each timestamp")

    def register_many(self, forecasts, *, now):
        self.check_prediction_time(now)
        if not forecasts:
            raise ValueError("Empty forecast batch")
        if len(self._forecasts) + len(forecasts) > self.max_forecasts:
            raise ValueError("Forecast budget exhausted")
        staged = {}
        added = 0
        for f in forecasts:
            if not isinstance(f, Forecast):
                raise ValueError("Forecast object required")
            identity = ev.identifier(f.forecast_id)
            if identity in self._forecasts or identity in staged:
                raise ValueError("Duplicate forecast identity")
            owner = ev.identifier(f.owner)
            origin = ev.integer(f.origin, "origin", 0)
            if origin != now:
                raise ValueError("Forecast origin must equal registration time")
            base, key = ev.vector(list(f.base)), ev.vector(list(f.key))
            target_times = tuple(ev.integer(t, "target time", origin + 1) for t in f.target_times)
            if len(target_times) != len(base):
                raise ValueError("Target times must cover every coordinate")
            # Bound base, target times, observed values, and full event provenance.
            added += 6 * len(base) + len(key)
            staged[identity] = Forecast(identity, owner, origin, target_times, base, key, ev.digest(f.state_id))
        if self._scalars + added > self.max_scalars:
            raise ValueError("Scalar budget exhausted")
        self._forecasts.update(staged)
        self._values.update({identity: {} for identity in staged})
        self._scalars += added
        if now > self._clock:
            self._revealed = False
        self._clock = now

    def observe(self, event, *, now):
        if not isinstance(event, Observation):
            raise ValueError("Sparse observation event required")
        ev.integer(now, "observation time", 0)
        if now < self._clock or ev.integer(event.arrived_at, "arrival", 0) != now:
            raise ValueError("Observation must be processed at its actual monotonic arrival time")
        event_id = ev.identifier(event.event_id)
        if event_id in self._events:
            raise ValueError("Duplicate observation event")
        if len(self._events) >= self.max_events:
            raise ValueError("Event budget exhausted")
        f = self._forecasts.get(ev.identifier(event.forecast_id))
        if f is None or ev.identifier(event.owner) != f.owner:
            raise ValueError("Unknown forecast or wrong owner")
        if f.forecast_id in self._complete:
            raise ValueError("Forecast already released")
        if not event.indices or len(event.indices) != len(event.values) or len(event.indices) != len(event.available_at):
            raise ValueError("Observation coordinates, values and times must align")
        staged = dict(self._values[f.forecast_id])
        for index, value, available in zip(event.indices, event.values, event.available_at, strict=True):
            ev.integer(index, "coordinate", 0, len(f.base) - 1)
            if index in staged:
                raise ValueError("Duplicate coordinate observation")
            ev.integer(available, "availability", f.target_times[index], now)
            staged[index] = ev.finite(value, "observed value")
        ready = None
        if len(staged) == len(f.base):
            residual = tuple(ev.finite(staged[i] - base, "residual") for i, base in enumerate(f.base))
            ready = ReadyResidual(f, residual, now)
        # Commit only after every validation and arithmetic check succeeds.
        self._values[f.forecast_id] = staged
        self._events[event_id] = {
            "forecast_id": f.forecast_id, "owner": f.owner, "arrived_at": now,
            "indices": tuple(event.indices),
            "values": tuple(staged[i] for i in event.indices),
            "available_at": tuple(event.available_at),
        }
        self._clock, self._revealed = now, True
        if ready is not None:
            self._complete.add(f.forecast_id)
        return ready

    def checksum(self):
        payload = {
            "forecasts": {k: asdict(v) for k, v in self._forecasts.items()},
            "observations": self._values, "events": self._events,
            "complete": sorted(self._complete), "clock": self._clock,
            "revealed": self._revealed,
            "limits": [self.max_forecasts, self.max_events, self.max_scalars],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
