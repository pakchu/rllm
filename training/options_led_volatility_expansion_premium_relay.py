"""Pure, source- and outcome-blind OVEPR-24 clock mechanics.

The module has no filesystem, network, dataframe, execution-price, funding,
return, portfolio, or Gross9 dependency.  It operates only on already joined,
completed source aggregates and emits deterministic schedules.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Literal, Sequence


CANDIDATE_ID = "OVEPR-24"
ONE_HOUR = timedelta(hours=1)
ENTRY_DELAY = timedelta(minutes=5)
HOLD_TIME = timedelta(hours=24)
PRIOR_WINDOW = timedelta(hours=720)
PRIOR_MIN_VALID = 672

PRIMARY = "primary"
NO_DERIBIT_LEAD = "no_deribit_lead"
DERIBIT_FALL_MIRROR = "deribit_fall_mirror"
NO_PREMIUM_EFFICIENCY = "no_premium_efficiency"
OWN_CLOCKS = (
    PRIMARY,
    NO_DERIBIT_LEAD,
    DERIBIT_FALL_MIRROR,
    NO_PREMIUM_EFFICIENCY,
)

DIRECTION_FLIP = "direction_flip"
EXTRA_LATENCY_1H = "extra_latency_1h"
DETERMINISTIC_RANDOM_SIDE = "deterministic_random_side"
PARENT_SET_CONTROLS = (
    DIRECTION_FLIP,
    EXTRA_LATENCY_1H,
    DETERMINISTIC_RANDOM_SIDE,
)

Side = Literal["LONG", "SHORT"]
MAX_DECIMAL_COEFFICIENT_DIGITS = 128
MIN_DECIMAL_EXPONENT = -128
MAX_DECIMAL_EXPONENT = 128


def _utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _rfc3339_z(value: datetime) -> str:
    checked = _utc(value, "timestamp")
    if checked.microsecond:
        raise ValueError("OVEPR-24 identifiers require whole-second timestamps")
    return checked.strftime("%Y-%m-%dT%H:%M:%SZ")


def _decimal(value: str | int | Decimal, label: str) -> Decimal:
    if isinstance(value, (bool, float)):
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


def _fraction(value: Decimal) -> Fraction:
    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError("finite Decimal exponent must be integral")
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    if sign:
        coefficient = -coefficient
    if exponent >= 0:
        return Fraction(coefficient * 10**exponent, 1)
    return Fraction(coefficient, 10 ** (-exponent))


@dataclass(frozen=True)
class VolatilityCandle:
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal

    def __post_init__(self) -> None:
        for name in ("open", "high", "low", "close"):
            object.__setattr__(self, name, _decimal(getattr(self, name), name))
        values = (self.open, self.high, self.low, self.close)
        if any(value <= 0 for value in values):
            raise ValueError("OVEPR-24 volatility OHLC must be strictly positive")
        if self.high < max(self.open, self.close):
            raise ValueError("OVEPR-24 volatility high violates the OHLC envelope")
        if self.low > min(self.open, self.close):
            raise ValueError("OVEPR-24 volatility low violates the OHLC envelope")

    @classmethod
    def from_tokens(
        cls,
        *,
        open: str | int | Decimal,
        high: str | int | Decimal,
        low: str | int | Decimal,
        close: str | int | Decimal,
    ) -> "VolatilityCandle":
        return cls(
            open=_decimal(open, "open"),
            high=_decimal(high, "high"),
            low=_decimal(low, "low"),
            close=_decimal(close, "close"),
        )

    @property
    def normalized_body(self) -> Fraction:
        open_value = _fraction(self.open)
        return (_fraction(self.close) - open_value) / open_value


@dataclass(frozen=True)
class PremiumPathHour:
    move: Decimal
    path_range: Decimal
    source_rows: int = 60
    source_valid: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "move", _decimal(self.move, "premium move"))
        object.__setattr__(
            self, "path_range", _decimal(self.path_range, "premium path range")
        )
        if self.path_range < 0:
            raise ValueError("OVEPR-24 premium path range cannot be negative")
        if not isinstance(self.source_rows, int) or isinstance(self.source_rows, bool):
            raise ValueError("OVEPR-24 premium source_rows must be an integer")
        if not isinstance(self.source_valid, bool):
            raise ValueError("OVEPR-24 premium source_valid must be boolean")

    @classmethod
    def from_tokens(
        cls,
        *,
        move: str | int | Decimal,
        path_range: str | int | Decimal,
        source_rows: int = 60,
        source_valid: bool = True,
    ) -> "PremiumPathHour":
        return cls(
            move=_decimal(move, "premium move"),
            path_range=_decimal(path_range, "premium path range"),
            source_rows=source_rows,
            source_valid=source_valid,
        )

    @property
    def valid(self) -> bool:
        return bool(
            self.source_valid
            and self.source_rows == 60
            and self.path_range > 0
        )

    @property
    def efficiency(self) -> Fraction:
        if not self.valid:
            raise ValueError("OVEPR-24 premium efficiency requires a valid hour")
        return abs(_fraction(self.move)) / _fraction(self.path_range)


@dataclass(frozen=True)
class JoinedHour:
    close_time: datetime
    bvol_available_at: datetime
    dvol_available_at: datetime
    premium_available_at: datetime
    bvol: VolatilityCandle | None
    dvol: VolatilityCandle | None
    premium: PremiumPathHour | None
    source_valid: bool = True

    def __post_init__(self) -> None:
        close_time = _utc(self.close_time, "close_time")
        if close_time.minute or close_time.second or close_time.microsecond:
            raise ValueError("OVEPR-24 close_time must be an exact UTC hour")
        object.__setattr__(self, "close_time", close_time)
        for name in (
            "bvol_available_at",
            "dvol_available_at",
            "premium_available_at",
        ):
            value = _utc(getattr(self, name), name)
            if value < close_time:
                raise ValueError(f"OVEPR-24 {name} precedes the completed hour")
            object.__setattr__(self, name, value)
        if not isinstance(self.source_valid, bool):
            raise ValueError("OVEPR-24 source_valid must be boolean")

    @property
    def available_at(self) -> datetime:
        return max(
            self.bvol_available_at,
            self.dvol_available_at,
            self.premium_available_at,
        )

    @property
    def base_valid(self) -> bool:
        return bool(
            self.source_valid
            and isinstance(self.bvol, VolatilityCandle)
            and isinstance(self.dvol, VolatilityCandle)
            and isinstance(self.premium, PremiumPathHour)
            and self.premium.valid
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


def _median(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise ValueError("OVEPR-24 median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def prior_efficiency_median(
    rows: Sequence[JoinedHour], current_index: int
) -> Fraction | None:
    current_time = rows[current_index].close_time
    lower = current_time - PRIOR_WINDOW
    values = [
        row.premium.efficiency
        for row in rows[:current_index]
        if lower <= row.close_time < current_time
        and row.base_valid
        and row.premium is not None
    ]
    if len(values) < PRIOR_MIN_VALID:
        return None
    return _median(values)


def _volatility_condition(row: JoinedHour, control: str) -> bool:
    assert row.bvol is not None and row.dvol is not None
    b_body = row.bvol.normalized_body
    d_body = row.dvol.normalized_body
    if control == NO_DERIBIT_LEAD:
        return b_body > 0 and d_body > 0
    if control == DERIBIT_FALL_MIRROR:
        return b_body < 0 and d_body < 0 and abs(d_body) > abs(b_body)
    return b_body > 0 and d_body > b_body


def setup_for_hour(
    rows: Sequence[JoinedHour], current_index: int, control: str = PRIMARY
) -> bool:
    if control not in OWN_CLOCKS:
        raise ValueError(f"unknown OVEPR-24 own-clock control: {control}")
    row = rows[current_index]
    if not row.base_valid:
        return False
    assert row.premium is not None
    if row.premium.move == 0 or not _volatility_condition(row, control):
        return False
    if control == NO_PREMIUM_EFFICIENCY:
        return True
    threshold = prior_efficiency_median(rows, current_index)
    return threshold is not None and row.premium.efficiency >= threshold


def _validate_rows(rows: Sequence[JoinedHour]) -> None:
    previous: datetime | None = None
    for row in rows:
        if not isinstance(row, JoinedHour):
            raise ValueError("OVEPR-24 rows must be JoinedHour objects")
        if previous is not None and row.close_time <= previous:
            raise ValueError("OVEPR-24 joined hours must be unique and increasing")
        previous = row.close_time


def _side(row: JoinedHour) -> Side:
    assert row.premium is not None
    if row.premium.move > 0:
        return "LONG"
    if row.premium.move < 0:
        return "SHORT"
    raise ValueError("zero premium move has no OVEPR-24 side")


def raw_candidates(
    rows: Sequence[JoinedHour], control: str = PRIMARY
) -> tuple[ScheduledEvent, ...]:
    """Emit only valid consecutive false-to-true setup onsets."""

    if control not in OWN_CLOCKS:
        raise ValueError(f"unknown OVEPR-24 own-clock control: {control}")
    _validate_rows(rows)
    setups = [setup_for_hour(rows, index, control) for index in range(len(rows))]
    output: list[ScheduledEvent] = []
    for index in range(1, len(rows)):
        previous, current = rows[index - 1], rows[index]
        consecutive = (
            previous.base_valid
            and current.base_valid
            and current.close_time - previous.close_time == ONE_HOUR
        )
        if not consecutive or setups[index - 1] or not setups[index]:
            continue
        side = _side(current)
        entry = current.close_time + ENTRY_DELAY
        if current.available_at >= entry:
            continue
        output.append(
            ScheduledEvent(
                signal_id=(
                    f"{CANDIDATE_ID}|{control}|T={_rfc3339_z(current.close_time)}"
                ),
                control=control,
                signal_time=current.close_time,
                available_at=current.available_at,
                entry_time=entry,
                exit_time=entry + HOLD_TIME,
                side=side,
            )
        )
    return tuple(output)


def reserve_nonoverlap(
    candidates: Sequence[ScheduledEvent],
) -> tuple[ScheduledEvent, ...]:
    """Apply one global, deterministic [entry, exit) reservation."""

    ordered = sorted(
        candidates,
        key=lambda event: (event.entry_time, event.signal_time, event.signal_id),
    )
    accepted: list[ScheduledEvent] = []
    reserved_until: datetime | None = None
    for event in ordered:
        if reserved_until is None or event.entry_time >= reserved_until:
            accepted.append(event)
            reserved_until = event.exit_time
    return tuple(accepted)


def primary_schedule(rows: Sequence[JoinedHour]) -> tuple[ScheduledEvent, ...]:
    return reserve_nonoverlap(raw_candidates(rows, PRIMARY))


def parent_control_schedule(
    primary_events: Sequence[ScheduledEvent], control: str
) -> tuple[ScheduledEvent, ...]:
    """Transform the reserved primary event set without clock repair."""

    if control not in PARENT_SET_CONTROLS:
        raise ValueError(f"unknown OVEPR-24 parent-set control: {control}")
    output: list[ScheduledEvent] = []
    for event in primary_events:
        side = event.side
        entry_time = event.entry_time
        exit_time = event.exit_time
        if control == DIRECTION_FLIP:
            side = "SHORT" if side == "LONG" else "LONG"
        elif control == EXTRA_LATENCY_1H:
            entry_time += ONE_HOUR
            exit_time += ONE_HOUR
        else:
            token = f"{CANDIDATE_ID}|{_rfc3339_z(event.signal_time)}".encode()
            side = "LONG" if hashlib.sha256(token).digest()[0] % 2 == 0 else "SHORT"
        output.append(
            replace(
                event,
                signal_id=f"{CANDIDATE_ID}|{control}|T={_rfc3339_z(event.signal_time)}",
                control=control,
                entry_time=entry_time,
                exit_time=exit_time,
                side=side,
            )
        )
    return tuple(output)
