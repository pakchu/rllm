"""Pure, source-blind CVVH-432 clock mechanics.

This module deliberately has no filesystem, network, pandas, market-price,
funding, return, or portfolio dependencies.  It defines only the frozen
hourly state machine and deterministic scheduling rules used by later,
separately sealed stages.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Iterable, Literal, Sequence


CANDIDATE_ID = "CVVH-432"
FIVE_MINUTES = timedelta(minutes=5)
ONE_HOUR = timedelta(hours=1)
HOLD_BARS_5M = 432
HOLD_TIME = HOLD_BARS_5M * FIVE_MINUTES

PRIMARY = "primary"
DERIBIT_LED = "deribit_led"
BODY_LEAD_ONLY = "body_lead_only"
RANGE_LEAD_ONLY = "range_lead_only"
STALE_DERIBIT = "stale_deribit"

INDEPENDENT_CONTROLS = (
    DERIBIT_LED,
    BODY_LEAD_ONLY,
    RANGE_LEAD_ONLY,
    STALE_DERIBIT,
)
OWN_CLOCKS = (PRIMARY, *INDEPENDENT_CONTROLS)

DIRECTION_FLIP = "direction_flip"
DETERMINISTIC_RANDOM_SIDE = "deterministic_random_side"
CONSTANT_LONG = "constant_long"
CONSTANT_SHORT = "constant_short"
ONE_BAR_DELAYED_ENTRY = "one_bar_delayed_entry"
PARENT_SET_CONTROLS = (
    DIRECTION_FLIP,
    DETERMINISTIC_RANDOM_SIDE,
    CONSTANT_LONG,
    CONSTANT_SHORT,
    ONE_BAR_DELAYED_ENTRY,
)

Side = Literal["LONG", "SHORT"]
State = Side | None

UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MAX_DECIMAL_COEFFICIENT_DIGITS = 128
MIN_DECIMAL_EXPONENT = -128
MAX_DECIMAL_EXPONENT = 128


def _utc(value: datetime, label: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _rfc3339_z(value: datetime) -> str:
    checked = _utc(value, "timestamp")
    if checked.microsecond:
        raise ValueError("CVVH-432 identifiers require whole-second timestamps")
    return (
        f"{checked.year:04d}-{checked.month:02d}-{checked.day:02d}"
        f"T{checked.hour:02d}:{checked.minute:02d}:{checked.second:02d}Z"
    )


def _decimal(value: str | int | Decimal, label: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(f"{label} must be an exact decimal token")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} is not an exact decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{label} must be finite")
    _, digits, exponent = parsed.as_tuple()
    if (
        not isinstance(exponent, int)
        or len(digits) > MAX_DECIMAL_COEFFICIENT_DIGITS
        or exponent < MIN_DECIMAL_EXPONENT
        or exponent > MAX_DECIMAL_EXPONENT
    ):
        raise ValueError(f"{label} exceeds the frozen decimal token bounds")
    return parsed


@dataclass(frozen=True)
class Candle:
    """One exact-decimal completed-hour OHLC envelope."""

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal

    def __post_init__(self) -> None:
        for name in ("open", "high", "low", "close"):
            object.__setattr__(self, name, _decimal(getattr(self, name), name))
        self.validate()

    @classmethod
    def from_tokens(
        cls,
        *,
        open: str | int | Decimal,
        high: str | int | Decimal,
        low: str | int | Decimal,
        close: str | int | Decimal,
    ) -> "Candle":
        return cls(
            open=_decimal(open, "open"),
            high=_decimal(high, "high"),
            low=_decimal(low, "low"),
            close=_decimal(close, "close"),
        )

    def validate(self) -> None:
        values = (self.open, self.high, self.low, self.close)
        if any(not value.is_finite() or value <= 0 for value in values):
            raise ValueError("CVVH-432 OHLC values must be finite and strictly positive")
        if self.high < max(self.open, self.close):
            raise ValueError("CVVH-432 high violates the OHLC envelope")
        if self.low > min(self.open, self.close):
            raise ValueError("CVVH-432 low violates the OHLC envelope")


@dataclass(frozen=True)
class JoinedHour:
    """A UTC-aligned joined source hour.

    ``close_time`` is the completed-hour clock.  ``available_at`` is the later
    of the two source availability timestamps.  Invalid retained source hours
    use ``None`` candles and/or ``source_valid=False`` rather than imputation.
    """

    close_time: datetime
    available_at: datetime
    bvol: Candle | None
    dvol: Candle | None
    source_valid: bool = True

    def __post_init__(self) -> None:
        close_time = _utc(self.close_time, "close_time")
        available_at = _utc(self.available_at, "available_at")
        if close_time.minute or close_time.second or close_time.microsecond:
            raise ValueError("CVVH-432 close_time must be an exact UTC hour")
        if available_at < close_time:
            raise ValueError("CVVH-432 availability precedes the completed hour")
        if not isinstance(self.source_valid, bool):
            raise ValueError("CVVH-432 source_valid must be boolean")
        object.__setattr__(self, "close_time", close_time)
        object.__setattr__(self, "available_at", available_at)

    @property
    def base_valid(self) -> bool:
        return bool(
            self.source_valid
            and isinstance(self.bvol, Candle)
            and isinstance(self.dvol, Candle)
        )


@dataclass(frozen=True)
class ScheduledEvent:
    signal_id: str
    control: str
    signal_time: datetime
    available_at: datetime
    entry_time: datetime
    exit_time: datetime
    side: Side


def _opposite_nonzero(left: Fraction, right: Fraction) -> bool:
    return bool((left > 0 and right < 0) or (left < 0 and right > 0))


def _leader_side(body: Fraction) -> Side:
    if body > 0:
        return "SHORT"
    if body < 0:
        return "LONG"
    raise ValueError("a zero body has no CVVH-432 side")


def _fraction(value: Decimal) -> Fraction:
    """Convert a finite Decimal token to its exact integer-coefficient ratio."""

    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError("CVVH-432 finite Decimal exponent is not integral")
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + int(digit)
    if sign:
        coefficient = -coefficient
    if exponent >= 0:
        return Fraction(coefficient * (10**exponent), 1)
    return Fraction(coefficient, 10 ** (-exponent))


def _state_from_candles(
    bvol: Candle,
    dvol: Candle,
    control: str,
) -> State:
    bvol.validate()
    dvol.validate()
    b_open = _fraction(bvol.open)
    d_open = _fraction(dvol.open)
    b_body = _fraction(bvol.close) - b_open
    d_body = _fraction(dvol.close) - d_open
    if not _opposite_nonzero(b_body, d_body):
        return None

    b_body_leads = abs(b_body) * d_open > abs(d_body) * b_open
    d_body_leads = abs(d_body) * b_open > abs(b_body) * d_open
    b_range = _fraction(bvol.high) - _fraction(bvol.low)
    d_range = _fraction(dvol.high) - _fraction(dvol.low)
    b_range_leads = b_range * d_open > d_range * b_open
    d_range_leads = d_range * b_open > b_range * d_open

    if control in {PRIMARY, STALE_DERIBIT}:
        return _leader_side(b_body) if b_body_leads and b_range_leads else None
    if control == DERIBIT_LED:
        return _leader_side(d_body) if d_body_leads and d_range_leads else None
    if control == BODY_LEAD_ONLY:
        return _leader_side(b_body) if b_body_leads else None
    if control == RANGE_LEAD_ONLY:
        return _leader_side(b_body) if b_range_leads else None
    raise ValueError(f"unknown CVVH-432 own-clock control: {control}")


def state_for_hour(row: JoinedHour, control: str = PRIMARY) -> State:
    """Return the exact same-hour state for a primary/non-stale own clock."""

    if control == STALE_DERIBIT:
        raise ValueError("stale_deribit requires three-hour history")
    if control not in OWN_CLOCKS:
        raise ValueError(f"unknown CVVH-432 own-clock control: {control}")
    if not row.base_valid:
        return None
    assert row.bvol is not None and row.dvol is not None
    return _state_from_candles(row.bvol, row.dvol, control)


def ceil_to_five_minutes(value: datetime) -> datetime:
    checked = _utc(value, "availability")
    delta = checked - UNIX_EPOCH
    epoch_us = (
        (delta.days * 86_400 + delta.seconds) * 1_000_000
        + delta.microseconds
    )
    step_us = int(FIVE_MINUTES.total_seconds() * 1_000_000)
    rounded_us = ((epoch_us + step_us - 1) // step_us) * step_us
    return UNIX_EPOCH + timedelta(microseconds=rounded_us)


def _signal_id(control: str, signal_time: datetime) -> str:
    return f"{CANDIDATE_ID}|{control}|T={_rfc3339_z(signal_time)}"


def _event(row: JoinedHour, control: str, side: Side) -> ScheduledEvent:
    entry = ceil_to_five_minutes(row.available_at) + FIVE_MINUTES
    return ScheduledEvent(
        signal_id=_signal_id(control, row.close_time),
        control=control,
        signal_time=row.close_time,
        available_at=row.available_at,
        entry_time=entry,
        exit_time=entry + HOLD_TIME,
        side=side,
    )


def _consecutive(left: JoinedHour, right: JoinedHour) -> bool:
    return bool(
        left.base_valid
        and right.base_valid
        and right.close_time - left.close_time == ONE_HOUR
    )


def _validate_hour_sequence(rows: Sequence[JoinedHour]) -> None:
    previous: datetime | None = None
    for row in rows:
        if not isinstance(row, JoinedHour):
            raise ValueError("CVVH-432 rows must be JoinedHour objects")
        if previous is not None and row.close_time <= previous:
            raise ValueError("CVVH-432 joined hours must be unique and increasing")
        previous = row.close_time


def raw_candidates(
    rows: Sequence[JoinedHour],
    control: str = PRIMARY,
) -> tuple[ScheduledEvent, ...]:
    """Build source-only false-to-true/opposite-side onset candidates."""

    if control not in OWN_CLOCKS:
        raise ValueError(f"unknown CVVH-432 own-clock control: {control}")
    _validate_hour_sequence(rows)
    output: list[ScheduledEvent] = []
    for index, current in enumerate(rows):
        if control == STALE_DERIBIT:
            if index < 2:
                continue
            older, previous = rows[index - 2], rows[index - 1]
            if not (
                _consecutive(older, previous)
                and _consecutive(previous, current)
            ):
                continue
            assert (
                older.dvol is not None
                and previous.bvol is not None
                and previous.dvol is not None
                and current.bvol is not None
            )
            prior_state = _state_from_candles(
                previous.bvol, older.dvol, STALE_DERIBIT
            )
            current_state = _state_from_candles(
                current.bvol, previous.dvol, STALE_DERIBIT
            )
        else:
            if index < 1:
                continue
            previous = rows[index - 1]
            if not _consecutive(previous, current):
                continue
            prior_state = state_for_hour(previous, control)
            current_state = state_for_hour(current, control)

        if current_state is not None and current_state != prior_state:
            output.append(_event(current, control, current_state))
    return tuple(
        sorted(
            output,
            key=lambda row: (
                row.entry_time,
                row.signal_time,
                row.signal_id,
                row.side,
            ),
        )
    )


def reserve_globally(
    candidates: Iterable[ScheduledEvent],
) -> tuple[ScheduledEvent, ...]:
    """Accept one global position at a time; suppressed candidates are not queued."""

    ordered = sorted(
        candidates,
        key=lambda row: (
            row.entry_time,
            row.signal_time,
            row.signal_id,
            row.side,
        ),
    )
    accepted: list[ScheduledEvent] = []
    previous_exit: datetime | None = None
    for row in ordered:
        if row.exit_time - row.entry_time != HOLD_TIME:
            raise ValueError("CVVH-432 candidate hold drifted")
        if previous_exit is None or row.entry_time >= previous_exit:
            accepted.append(row)
            previous_exit = row.exit_time
    return tuple(accepted)


def build_clock(
    rows: Sequence[JoinedHour],
    control: str = PRIMARY,
) -> tuple[ScheduledEvent, ...]:
    """Build one independently reserved own clock."""

    return reserve_globally(raw_candidates(rows, control))


def _flip(side: Side) -> Side:
    return "SHORT" if side == "LONG" else "LONG"


def deterministic_random_side(primary_signal_id: str) -> Side:
    payload = (
        f"{CANDIDATE_ID}|{primary_signal_id}|RANDOM_SIDE".encode("utf-8")
    )
    return "LONG" if hashlib.sha256(payload).digest()[0] < 128 else "SHORT"


def derive_parent_set_control(
    primary_clock: Sequence[ScheduledEvent],
    control: str,
) -> tuple[ScheduledEvent, ...]:
    """Derive controls from the accepted primary parent set without re-reserving."""

    if control not in PARENT_SET_CONTROLS:
        raise ValueError(f"unknown CVVH-432 parent-set control: {control}")
    output: list[ScheduledEvent] = []
    for row in primary_clock:
        if row.control != PRIMARY:
            raise ValueError("CVVH-432 derived controls require primary parents")
        if control == DIRECTION_FLIP:
            side = _flip(row.side)
        elif control == DETERMINISTIC_RANDOM_SIDE:
            side = deterministic_random_side(row.signal_id)
        elif control == CONSTANT_LONG:
            side = "LONG"
        elif control == CONSTANT_SHORT:
            side = "SHORT"
        else:
            side = row.side
        delay = FIVE_MINUTES if control == ONE_BAR_DELAYED_ENTRY else timedelta(0)
        output.append(
            replace(
                row,
                signal_id=_signal_id(control, row.signal_time),
                control=control,
                entry_time=row.entry_time + delay,
                exit_time=row.exit_time + delay,
                side=side,
            )
        )
    return tuple(output)


__all__ = [
    "BODY_LEAD_ONLY",
    "CANDIDATE_ID",
    "CONSTANT_LONG",
    "CONSTANT_SHORT",
    "Candle",
    "DERIBIT_LED",
    "DETERMINISTIC_RANDOM_SIDE",
    "DIRECTION_FLIP",
    "FIVE_MINUTES",
    "HOLD_BARS_5M",
    "HOLD_TIME",
    "INDEPENDENT_CONTROLS",
    "JoinedHour",
    "ONE_BAR_DELAYED_ENTRY",
    "OWN_CLOCKS",
    "PARENT_SET_CONTROLS",
    "PRIMARY",
    "RANGE_LEAD_ONLY",
    "STALE_DERIBIT",
    "ScheduledEvent",
    "build_clock",
    "ceil_to_five_minutes",
    "derive_parent_set_control",
    "deterministic_random_side",
    "raw_candidates",
    "reserve_globally",
    "state_for_hour",
]
