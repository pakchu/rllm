"""One-shot, outcome-blind source-support evaluator for CVVH-432."""

from __future__ import annotations

import csv
import ctypes
import errno
import gzip
import hashlib
import io
import json
import os
import secrets
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from training import cross_venue_volatility_shape_handoff as mechanism
from training import preregister_cross_venue_volatility_shape_handoff as prereg


PROTOCOL_VERSION = (
    "cross_venue_volatility_shape_handoff_source_support_v1"
)
ATTEMPT_PROTOCOL = (
    "cross_venue_volatility_shape_handoff_source_support_attempt_v1"
)
FAILURE_PROTOCOL = (
    "cross_venue_volatility_shape_handoff_source_support_failure_v1"
)
PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "9e6901e67e36d6d9170dae54548419cbfb9178cf84338e3c5652012896ee6604"
)
PREREGISTRATION_MANIFEST_HASH = (
    "dd3e334939b24cb5508c66ed9787c2c3e1a0ad006dcadcbfb673e541d6520cad"
)

ATTEMPT_CLAIM = Path(
    "results/"
    "cross_venue_volatility_shape_handoff_source_support_"
    "attempt_claim_2026-07-30.json"
)
OUTPUT_DIRECTORY = Path(
    "results/"
    "cross_venue_volatility_shape_handoff_source_support_2026-07-30"
)
FAILURE_RECEIPT = Path(
    "results/"
    "cross_venue_volatility_shape_handoff_source_support_"
    "failure_2026-07-30.json"
)

FULL_START = datetime(2023, 6, 1, tzinfo=timezone.utc)
SELECTION_END = datetime(2025, 1, 1, tzinfo=timezone.utc)
FUTURE25_END = datetime(2026, 1, 1, tzinfo=timezone.utc)
FULL_END = datetime(2026, 6, 1, tzinfo=timezone.utc)
SOURCE_START = datetime(2023, 6, 20, tzinfo=timezone.utc)
SOURCE_END = datetime(2026, 7, 1, tzinfo=timezone.utc)

WINDOWS = {
    "selection": (FULL_START, SELECTION_END),
    "2023H2": (FULL_START, datetime(2024, 1, 1, tzinfo=timezone.utc)),
    "2024H1": (
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 7, 1, tzinfo=timezone.utc),
    ),
    "2024H2": (
        datetime(2024, 7, 1, tzinfo=timezone.utc),
        SELECTION_END,
    ),
    "future25": (SELECTION_END, FUTURE25_END),
    "future26": (FUTURE25_END, FULL_END),
    "full": (FULL_START, FULL_END),
}

EVALUATOR_PATHS = (
    "training/__init__.py",
    "training/cross_venue_volatility_shape_handoff.py",
    "training/preregister_cross_venue_volatility_shape_handoff.py",
    "training/evaluate_cross_venue_volatility_shape_handoff_source_support.py",
    "tests/conftest.py",
    "tests/test_cross_venue_volatility_shape_handoff.py",
    "tests/test_preregister_cross_venue_volatility_shape_handoff.py",
    "tests/test_cross_venue_volatility_shape_handoff_preregistration_artifact.py",
    "tests/test_evaluate_cross_venue_volatility_shape_handoff_source_support.py",
    (
        "docs/"
        "cross-venue-volatility-shape-handoff-source-support-"
        "evaluator-freeze-2026-07-30.md"
    ),
    str(PREREGISTRATION),
    "pyproject.toml",
    "uv.lock",
)

CLOCK_HEADER = [
    "policy_id",
    "control",
    "signal_id",
    "signal_time_utc",
    "source_available_at_utc",
    "entry_time_utc",
    "exit_time_utc",
    "side",
]


class SourceSupportTerminalError(RuntimeError):
    """An authoritative support-stage failure that cannot be retried."""


class SourceSupportPublicationCommittedError(RuntimeError):
    """Publication renamed successfully but a post-commit durability check failed."""


@dataclass(frozen=True)
class Config:
    bvol: str = str(
        prereg.SOURCE_ARTIFACTS["binance_btc_bvol_hourly"]["path"]
    )
    bvol_manifest: str = str(
        prereg.SOURCE_ARTIFACTS["binance_btc_bvol_manifest"]["path"]
    )
    dvol: str = str(
        prereg.SOURCE_ARTIFACTS["deribit_btc_dvol_hourly"]["path"]
    )
    dvol_summary: str = str(
        prereg.SOURCE_ARTIFACTS["deribit_btc_dvol_summary"]["path"]
    )
    preregistration: str = str(PREREGISTRATION)
    attempt_claim: str = str(ATTEMPT_CLAIM)
    output_directory: str = str(OUTPUT_DIRECTORY)
    failure_receipt: str = str(FAILURE_RECEIPT)


@dataclass(frozen=True)
class SourceCalendar:
    """Exact source grids and the strictly earlier mechanism cutoff."""

    start: datetime
    end: datetime
    mechanism_end: datetime

    def __post_init__(self) -> None:
        start = _utc_datetime(self.start, "source start")
        end = _utc_datetime(self.end, "source end")
        mechanism_end = _utc_datetime(
            self.mechanism_end, "mechanism end"
        )
        if (
            start.minute
            or start.second
            or start.microsecond
            or end.minute
            or end.second
            or end.microsecond
            or mechanism_end.minute
            or mechanism_end.second
            or mechanism_end.microsecond
        ):
            raise ValueError("CVVH-432 source calendar must use exact UTC hours")
        if end <= start + timedelta(hours=1):
            raise ValueError("CVVH-432 source calendar is too short")
        if mechanism_end <= start + timedelta(hours=1) or mechanism_end > end:
            raise ValueError("CVVH-432 mechanism cutoff is outside source grid")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "mechanism_end", mechanism_end)

    @property
    def bvol_rows(self) -> int:
        return _timedelta_microseconds(self.end - self.start) // 3_600_000_000

    @property
    def dvol_rows(self) -> int:
        return self.bvol_rows + 1


def _utc_datetime(value: datetime, label: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"CVVH-432 {label} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"CVVH-432 {label} must be UTC")
    return value.astimezone(timezone.utc)


def _timedelta_microseconds(value: timedelta) -> int:
    return (
        (value.days * 86_400 + value.seconds) * 1_000_000
        + value.microseconds
    )


CANONICAL_SOURCE_CALENDAR = SourceCalendar(
    start=SOURCE_START,
    end=SOURCE_END,
    mechanism_end=FULL_END,
)


@dataclass(frozen=True)
class CapturedSources:
    """Hash-bound compressed bytes retained before the write-once claim."""

    bindings: Mapping[str, Any]
    bvol_compressed: bytes
    dvol_compressed: bytes


def _timestamp(value: str) -> datetime:
    token = value.strip()
    if not token:
        raise SourceSupportTerminalError("CVVH-432 timestamp is empty")
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SourceSupportTerminalError(
            f"CVVH-432 timestamp is invalid: {token}"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    elif parsed.utcoffset() != timedelta(0):
        raise SourceSupportTerminalError("CVVH-432 timestamp is not UTC")
    return parsed.astimezone(timezone.utc)


def _format_time(value: datetime) -> str:
    checked = _utc_datetime(value, "output timestamp")
    if checked.microsecond:
        return checked.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return checked.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_bool(value: str, label: str) -> bool:
    token = value.strip().lower()
    if token == "true":
        return True
    if token == "false":
        return False
    raise SourceSupportTerminalError(f"CVVH-432 {label} is not boolean")


def _parse_int(value: str, label: str) -> int:
    token = value.strip()
    if not token or token.startswith("+"):
        raise SourceSupportTerminalError(f"CVVH-432 {label} is not canonical")
    try:
        parsed = int(token)
    except ValueError as exc:
        raise SourceSupportTerminalError(
            f"CVVH-432 {label} is not integer"
        ) from exc
    if str(parsed) != token:
        raise SourceSupportTerminalError(
            f"CVVH-432 {label} is not canonical integer"
        )
    return parsed


def _read_rows(
    compressed: bytes,
    label: str,
    header: Sequence[str],
) -> Iterable[dict[str, str]]:
    with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as binary:
        with io.TextIOWrapper(binary, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(header):
                raise SourceSupportTerminalError(
                    f"CVVH-432 source header drift: {label}"
                )
            for row in reader:
                if None in row or set(row) != set(header):
                    raise SourceSupportTerminalError(
                        f"CVVH-432 source row width drift: {label}"
                    )
                if any(not isinstance(value, str) for value in row.values()):
                    raise SourceSupportTerminalError(
                        f"CVVH-432 source row is malformed: {label}"
                    )
                yield {str(key): str(value) for key, value in row.items()}


def load_bvol_compressed(
    compressed: bytes,
    *,
    label: str,
    calendar: SourceCalendar = CANONICAL_SOURCE_CALENDAR,
) -> tuple[dict[datetime, mechanism.JoinedHour], dict[str, Any]]:
    spec = prereg.SOURCE_ARTIFACTS["binance_btc_bvol_hourly"]
    output: dict[datetime, mechanism.JoinedHour] = {}
    valid_rows = 0
    invalid_rows = 0
    expected_rows = calendar.bvol_rows
    for index, row in enumerate(_read_rows(compressed, label, spec["header"])):
        date = _timestamp(row["date"])
        expected_date = calendar.start + index * timedelta(hours=1)
        if date != expected_date:
            raise SourceSupportTerminalError("CVVH-432 BVOL hourly grid drift")
        available = _timestamp(row["feature_available_time_utc"])
        earliest = _timestamp(row["trade_earliest_time_utc"])
        if available != date + timedelta(hours=1) or earliest != available:
            raise SourceSupportTerminalError("CVVH-432 BVOL availability drift")
        source_rows = _parse_int(row["source_rows"], "BVOL source_rows")
        complete = _parse_bool(row["source_complete"], "BVOL source_complete")
        valid = _parse_bool(row["feature_valid"], "BVOL feature_valid")
        reason = row["feature_invalid_reason"].strip()
        if (
            source_rows < 0
            or source_rows > 3_600
            or complete != (source_rows == 3_600)
        ):
            raise SourceSupportTerminalError("CVVH-432 BVOL completeness drift")
        values = [row[name].strip() for name in ("open", "high", "low", "close")]
        candle: mechanism.Candle | None
        if valid:
            if not complete or reason != "ok" or any(not value for value in values):
                raise SourceSupportTerminalError("CVVH-432 BVOL valid row drift")
            candle = mechanism.Candle.from_tokens(
                open=values[0],
                high=values[1],
                low=values[2],
                close=values[3],
            )
            valid_rows += 1
        else:
            if any(values) or not reason or reason == "ok":
                raise SourceSupportTerminalError("CVVH-432 BVOL invalid row drift")
            candle = None
            invalid_rows += 1
        if available in output:
            raise SourceSupportTerminalError("CVVH-432 duplicate BVOL clock")
        output[available] = mechanism.JoinedHour(
            close_time=available,
            available_at=available,
            bvol=candle,
            dvol=None,
            source_valid=valid,
        )
    if len(output) != expected_rows:
        raise SourceSupportTerminalError("CVVH-432 BVOL row count drift")
    return output, {
        "rows": len(output),
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "first_close_time_utc": _format_time(min(output)),
        "last_close_time_utc": _format_time(max(output)),
    }


def load_bvol(
    path: Path,
    calendar: SourceCalendar = CANONICAL_SOURCE_CALENDAR,
) -> tuple[dict[datetime, mechanism.JoinedHour], dict[str, Any]]:
    return load_bvol_compressed(
        path.read_bytes(),
        label=str(path),
        calendar=calendar,
    )


def load_dvol_compressed(
    compressed: bytes,
    *,
    label: str,
    calendar: SourceCalendar = CANONICAL_SOURCE_CALENDAR,
) -> tuple[dict[datetime, mechanism.Candle], dict[str, Any]]:
    spec = prereg.SOURCE_ARTIFACTS["deribit_btc_dvol_hourly"]
    output: dict[datetime, mechanism.Candle] = {}
    expected_rows = calendar.dvol_rows
    for index, row in enumerate(_read_rows(compressed, label, spec["header"])):
        date = _timestamp(row["date"])
        expected_date = calendar.start + index * timedelta(hours=1)
        if date != expected_date:
            raise SourceSupportTerminalError("CVVH-432 DVOL hourly grid drift")
        close_time = _timestamp(row["close_time"])
        if close_time != date + timedelta(hours=1):
            raise SourceSupportTerminalError("CVVH-432 DVOL close clock drift")
        candle = mechanism.Candle.from_tokens(
            open=row["open"].strip(),
            high=row["high"].strip(),
            low=row["low"].strip(),
            close=row["close"].strip(),
        )
        if close_time in output:
            raise SourceSupportTerminalError("CVVH-432 duplicate DVOL clock")
        output[close_time] = candle
    if len(output) != expected_rows:
        raise SourceSupportTerminalError("CVVH-432 DVOL row count drift")
    return output, {
        "rows": len(output),
        "first_close_time_utc": _format_time(min(output)),
        "last_close_time_utc": _format_time(max(output)),
    }


def load_dvol(
    path: Path,
    calendar: SourceCalendar = CANONICAL_SOURCE_CALENDAR,
) -> tuple[dict[datetime, mechanism.Candle], dict[str, Any]]:
    return load_dvol_compressed(
        path.read_bytes(),
        label=str(path),
        calendar=calendar,
    )


def join_loaded_sources(
    bvol: Mapping[datetime, mechanism.JoinedHour],
    dvol: Mapping[datetime, mechanism.Candle],
    calendar: SourceCalendar,
) -> tuple[tuple[mechanism.JoinedHour, ...], dict[str, Any]]:
    mechanism_bvol = {
        close_time: row
        for close_time, row in bvol.items()
        if close_time < calendar.mechanism_end
    }
    eligible_dvol = {
        close_time: candle
        for close_time, candle in dvol.items()
        if close_time < calendar.end
    }
    missing_dvol = sorted(set(mechanism_bvol) - set(eligible_dvol))
    if missing_dvol:
        raise SourceSupportTerminalError("CVVH-432 exact join has missing DVOL")
    joined: list[mechanism.JoinedHour] = []
    for close_time in sorted(mechanism_bvol):
        left = mechanism_bvol[close_time]
        joined.append(
            mechanism.JoinedHour(
                close_time=close_time,
                available_at=max(left.available_at, close_time),
                bvol=left.bvol,
                dvol=eligible_dvol[close_time],
                source_valid=left.source_valid,
            )
        )
    if (
        not joined
        or joined[0].close_time != calendar.start + timedelta(hours=1)
    ):
        raise SourceSupportTerminalError("CVVH-432 joined source start drift")
    return tuple(joined), {
        "dvol_rows_after_required_close_before_source_end_filter": len(
            eligible_dvol
        ),
        "joined_rows_before_full_end": len(joined),
        "joined_valid_rows_before_full_end": sum(row.base_valid for row in joined),
        "join_missing_dvol_rows": 0,
        "fills_imputations_tolerance_or_nearest": 0,
    }


def load_joined_sources_for_calendar(
    repo: Path,
    cfg: Config,
    calendar: SourceCalendar,
) -> tuple[
    tuple[mechanism.JoinedHour, ...], dict[str, Any]
]:
    bvol, bvol_stats = load_bvol(repo / cfg.bvol, calendar)
    dvol, dvol_stats = load_dvol(repo / cfg.dvol, calendar)
    joined, join_stats = join_loaded_sources(bvol, dvol, calendar)
    return joined, {
        "bvol": bvol_stats,
        "dvol": dvol_stats,
        **join_stats,
    }


def load_captured_sources(
    captured: CapturedSources,
) -> tuple[tuple[mechanism.JoinedHour, ...], dict[str, Any]]:
    expected_snapshots = {
        "binance_btc_bvol_hourly": captured.bvol_compressed,
        "deribit_btc_dvol_hourly": captured.dvol_compressed,
    }
    for name, compressed in expected_snapshots.items():
        if (
            hashlib.sha256(compressed).hexdigest()
            != prereg.SOURCE_ARTIFACTS[name]["sha256"]
        ):
            raise SourceSupportTerminalError(
                f"CVVH-432 retained compressed snapshot drift: {name}"
            )
    bvol, bvol_stats = load_bvol_compressed(
        captured.bvol_compressed,
        label=str(prereg.SOURCE_ARTIFACTS["binance_btc_bvol_hourly"]["path"]),
    )
    dvol, dvol_stats = load_dvol_compressed(
        captured.dvol_compressed,
        label=str(prereg.SOURCE_ARTIFACTS["deribit_btc_dvol_hourly"]["path"]),
    )
    joined, join_stats = join_loaded_sources(
        bvol,
        dvol,
        CANONICAL_SOURCE_CALENDAR,
    )
    return joined, {
        "bvol": bvol_stats,
        "dvol": dvol_stats,
        **join_stats,
        "decoded_from_preclaim_compressed_snapshots": True,
        "source_paths_reopened_after_claim": 0,
        "decoded_compressed_snapshot_sha256": {
            "binance_btc_bvol_hourly": hashlib.sha256(
                captured.bvol_compressed
            ).hexdigest(),
            "deribit_btc_dvol_hourly": hashlib.sha256(
                captured.dvol_compressed
            ).hexdigest(),
        },
    }


def _contained(
    events: Sequence[mechanism.ScheduledEvent],
    start: datetime,
    end: datetime,
) -> tuple[mechanism.ScheduledEvent, ...]:
    return tuple(
        row
        for row in events
        if row.entry_time >= start and row.exit_time <= end
    )


def build_all_clocks(
    rows: Sequence[mechanism.JoinedHour],
) -> dict[str, tuple[mechanism.ScheduledEvent, ...]]:
    own = {
        control: mechanism.build_clock(rows, control)
        for control in mechanism.OWN_CLOCKS
    }
    primary = own[mechanism.PRIMARY]
    derived = {
        control: mechanism.derive_parent_set_control(primary, control)
        for control in mechanism.PARENT_SET_CONTROLS
    }
    return {**own, **derived}


def _event_entries(
    events: Sequence[mechanism.ScheduledEvent],
) -> tuple[datetime, ...]:
    return tuple(sorted({row.entry_time for row in events}))


def exact_entry_jaccard(
    left: Sequence[mechanism.ScheduledEvent],
    right: Sequence[mechanism.ScheduledEvent],
) -> Fraction:
    a = set(_event_entries(left))
    b = set(_event_entries(right))
    union = a | b
    if not union:
        raise SourceSupportTerminalError("CVVH-432 exact Jaccard is undefined")
    return Fraction(len(a & b), len(union))


def deterministic_one_to_one_pairs(
    left: Sequence[datetime],
    right: Sequence[datetime],
    tolerance: timedelta,
) -> tuple[tuple[datetime, datetime], ...]:
    """Maximum-cardinality, minimum-lag, lexicographically minimal matching."""

    tolerance_us = _timedelta_microseconds(tolerance)
    if tolerance_us < 0:
        raise ValueError("CVVH-432 matching tolerance must be nonnegative")
    a = tuple(sorted(_utc_datetime(value, "left match clock") for value in left))
    b = tuple(sorted(_utc_datetime(value, "right match clock") for value in right))
    if not a or not b:
        raise SourceSupportTerminalError("CVVH-432 one-to-one match is undefined")
    if len(set(a)) != len(a) or len(set(b)) != len(b):
        raise SourceSupportTerminalError(
            "CVVH-432 one-to-one match clock contains duplicates"
        )
    n, m = len(a), len(b)
    count = [[0] * (m + 1) for _ in range(n + 1)]
    cost = [[0] * (m + 1) for _ in range(n + 1)]

    def lag_us(i: int, j: int) -> int:
        return abs(_timedelta_microseconds(a[i] - b[j]))

    def better(
        first_count: int,
        first_cost: int,
        second_count: int,
        second_cost: int,
    ) -> bool:
        return bool(
            first_count > second_count
            or (first_count == second_count and first_cost < second_cost)
        )

    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best_count = count[i + 1][j]
            best_cost = cost[i + 1][j]
            if better(count[i][j + 1], cost[i][j + 1], best_count, best_cost):
                best_count, best_cost = count[i][j + 1], cost[i][j + 1]
            lag = lag_us(i, j)
            if lag <= tolerance_us:
                matched_count = 1 + count[i + 1][j + 1]
                matched_cost = lag + cost[i + 1][j + 1]
                if better(
                    matched_count, matched_cost, best_count, best_cost
                ):
                    best_count, best_cost = matched_count, matched_cost
            count[i][j], cost[i][j] = best_count, best_cost

    output: list[tuple[datetime, datetime]] = []
    i = j = 0
    while count[i][j] > 0:
        target = (count[i][j], cost[i][j])
        selected: tuple[int, int] | None = None
        for left_index in range(i, n):
            for right_index in range(j, m):
                lag = lag_us(left_index, right_index)
                if lag > tolerance_us:
                    continue
                suffix = (
                    1 + count[left_index + 1][right_index + 1],
                    lag + cost[left_index + 1][right_index + 1],
                )
                if suffix == target:
                    selected = (left_index, right_index)
                    break
            if selected is not None:
                break
        if selected is None:
            raise SourceSupportTerminalError(
                "CVVH-432 matching lexicographic reconstruction drift"
            )
        left_index, right_index = selected
        output.append((a[left_index], b[right_index]))
        i, j = left_index + 1, right_index + 1
    if len(output) != count[0][0]:
        raise SourceSupportTerminalError("CVVH-432 matching reconstruction drift")
    return tuple(output)


def _fraction(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def _clock_stats(
    events: Sequence[mechanism.ScheduledEvent],
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    rows = _contained(events, start, end)
    month_counts: dict[str, int] = {}
    side_counts = {"LONG": 0, "SHORT": 0}
    for row in rows:
        month = row.entry_time.strftime("%Y-%m")
        month_counts[month] = month_counts.get(month, 0) + 1
        side_counts[row.side] += 1
    maximum = (
        Fraction(max(month_counts.values()), len(rows)) if rows else None
    )
    crossing = sum(
        row.entry_time < end
        and row.exit_time > start
        and not (row.entry_time >= start and row.exit_time <= end)
        for row in events
    )
    return {
        "total": len(rows),
        "LONG": side_counts["LONG"],
        "SHORT": side_counts["SHORT"],
        "monthly_counts": dict(sorted(month_counts.items())),
        "maximum_month_share": _fraction(maximum) if maximum is not None else None,
        "boundary_crossing_excluded": crossing,
    }


def _maximum_same_side_run(
    events: Sequence[mechanism.ScheduledEvent],
) -> int:
    maximum = current = 0
    previous: str | None = None
    for row in sorted(events, key=lambda item: item.entry_time):
        current = current + 1 if row.side == previous else 1
        maximum = max(maximum, current)
        previous = row.side
    return maximum


def _maximum_gap_seconds(
    events: Sequence[mechanism.ScheduledEvent],
) -> int | None:
    entries = sorted(row.entry_time for row in events)
    if len(entries) < 2:
        return None
    return max(
        _timedelta_microseconds(right - left) // 1_000_000
        for left, right in zip(entries, entries[1:])
    )


def _pairs_hash(pairs: Sequence[tuple[datetime, datetime]]) -> str:
    payload = [
        [_format_time(left), _format_time(right)] for left, right in pairs
    ]
    return prereg.canonical_hash(payload)


def structural_control_novelty(
    primary: Sequence[mechanism.ScheduledEvent],
    control: Sequence[mechanism.ScheduledEvent],
) -> dict[str, Any]:
    left = _contained(primary, FULL_START, FULL_END)
    right = _contained(control, FULL_START, FULL_END)
    jaccard = exact_entry_jaccard(left, right)
    pairs = deterministic_one_to_one_pairs(
        _event_entries(left),
        _event_entries(right),
        timedelta(hours=24),
    )
    matched_share = Fraction(len(pairs), min(len(left), len(right)))
    total_lag_microseconds = sum(
        abs(_timedelta_microseconds(left_time - right_time))
        for left_time, right_time in pairs
    )
    checks = {
        "exact_entry_jaccard_strictly_below_9_over_10": (
            jaccard < Fraction(9, 10)
        ),
        "one_to_one_24h_max_matched_share_strictly_below_19_over_20": (
            matched_share < Fraction(19, 20)
        ),
    }
    return {
        "primary_events": len(left),
        "control_events": len(right),
        "exact_entry_jaccard": _fraction(jaccard),
        "one_to_one_24h": {
            "matches": len(pairs),
            "maximum_matched_share": _fraction(matched_share),
            "total_absolute_lag_microseconds": total_lag_microseconds,
            "pair_list_sha256": _pairs_hash(pairs),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _state_trace(
    rows: Sequence[mechanism.JoinedHour],
    cutoff: datetime,
) -> dict[str, Any]:
    ordered = tuple(rows)
    states: list[dict[str, Any]] = []
    stale_state = getattr(mechanism, "_state_from_candles")
    for index, row in enumerate(ordered):
        if row.close_time >= cutoff:
            continue
        row_states: dict[str, Any] = {}
        for control in (
            mechanism.PRIMARY,
            mechanism.DERIBIT_LED,
            mechanism.BODY_LEAD_ONLY,
            mechanism.RANGE_LEAD_ONLY,
        ):
            row_states[control] = mechanism.state_for_hour(row, control)
        stale: str | None = None
        if index and row.base_valid and ordered[index - 1].base_valid:
            previous = ordered[index - 1]
            if row.close_time - previous.close_time == timedelta(hours=1):
                assert row.bvol is not None and previous.dvol is not None
                stale = stale_state(
                    row.bvol, previous.dvol, mechanism.STALE_DERIBIT
                )
        row_states[mechanism.STALE_DERIBIT] = stale
        states.append(
            {
                "close_time_utc": _format_time(row.close_time),
                "base_valid": row.base_valid,
                "states": row_states,
            }
        )
    raw = {
        control: [
            event.signal_id
            for event in mechanism.raw_candidates(ordered, control)
            if event.signal_time < cutoff
        ]
        for control in mechanism.OWN_CLOCKS
    }
    clocks = build_all_clocks(ordered)
    accepted = {
        control: [
            {
                "signal_id": event.signal_id,
                "side": event.side,
                "entry_time_utc": _format_time(event.entry_time),
                "exit_time_utc": _format_time(event.exit_time),
            }
            for event in clock
            if event.signal_time < cutoff
        ]
        for control, clock in clocks.items()
    }
    return {
        "joined_validity_and_states": states,
        "raw_candidate_ids": raw,
        "accepted_ids_sides_entry_exit": accepted,
    }


def append_invariance(
    rows: Sequence[mechanism.JoinedHour],
) -> dict[str, Any]:
    full_trace = _state_trace(rows, SELECTION_END)
    prefix_rows = tuple(row for row in rows if row.close_time < SELECTION_END)
    prefix_trace = _state_trace(prefix_rows, SELECTION_END)
    full_bytes = json.dumps(
        full_trace, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    prefix_bytes = json.dumps(
        prefix_trace, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "cutoff_utc": _format_time(SELECTION_END),
        "full_trace_sha256": hashlib.sha256(full_bytes).hexdigest(),
        "prefix_trace_sha256": hashlib.sha256(prefix_bytes).hexdigest(),
        "byte_identical": full_bytes == prefix_bytes,
        "future_append_differences": 0 if full_bytes == prefix_bytes else 1,
    }


def evaluate_clock_support(
    clocks: Mapping[str, Sequence[mechanism.ScheduledEvent]],
    source_diagnostics: Mapping[str, Any],
    invariant: Mapping[str, Any],
) -> dict[str, Any]:
    expected_controls = (
        *mechanism.OWN_CLOCKS,
        *mechanism.PARENT_SET_CONTROLS,
    )
    if set(clocks) != set(expected_controls):
        raise SourceSupportTerminalError(
            "CVVH-432 support clocks do not match the frozen control set"
        )
    primary = clocks[mechanism.PRIMARY]
    stats = {
        name: _clock_stats(primary, *bounds) for name, bounds in WINDOWS.items()
    }
    full_primary = _contained(primary, FULL_START, FULL_END)
    maximum_gap = _maximum_gap_seconds(full_primary)
    maximum_run = _maximum_same_side_run(full_primary)
    distinctness = {
        control: structural_control_novelty(primary, clocks[control])
        for control in mechanism.INDEPENDENT_CONTROLS
    }
    def month_at_most(name: str, maximum: Fraction) -> bool:
        raw = stats[name]["maximum_month_share"]
        return bool(
            isinstance(raw, Mapping)
            and Fraction(int(raw["numerator"]), int(raw["denominator"]))
            <= maximum
        )

    checks = {
        "selection_total_min_45": stats["selection"]["total"] >= 45,
        "selection_2023H2_min_12": stats["2023H2"]["total"] >= 12,
        "selection_2024H1_min_12": stats["2024H1"]["total"] >= 12,
        "selection_2024H2_min_12": stats["2024H2"]["total"] >= 12,
        "selection_each_side_min_14": min(
            stats["selection"]["LONG"], stats["selection"]["SHORT"]
        )
        >= 14,
        "selection_maximum_month_share_at_most_1_over_5": month_at_most(
            "selection", Fraction(1, 5)
        ),
        "future25_total_min_30": stats["future25"]["total"] >= 30,
        "future25_each_side_min_8": min(
            stats["future25"]["LONG"], stats["future25"]["SHORT"]
        )
        >= 8,
        "future25_maximum_month_share_at_most_1_over_4": month_at_most(
            "future25", Fraction(1, 4)
        ),
        "future26_total_min_15": stats["future26"]["total"] >= 15,
        "future26_each_side_min_4": min(
            stats["future26"]["LONG"], stats["future26"]["SHORT"]
        )
        >= 4,
        "future26_maximum_month_share_at_most_3_over_10": month_at_most(
            "future26", Fraction(3, 10)
        ),
        "maximum_accepted_entry_gap_at_most_90_days": (
            maximum_gap is not None and maximum_gap <= 90 * 86_400
        ),
        "maximum_same_side_run_at_most_12": maximum_run <= 12,
        "selection_prefix_future_append_invariant": (
            invariant.get("byte_identical") is True
        ),
        "all_four_independent_controls_distinct": all(
            row["passed"] for row in distinctness.values()
        ),
    }
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": mechanism.CANDIDATE_ID,
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "comparator_rows_opened": 0,
        "gross9_rows_opened": 0,
        "btc_execution_rows_opened": 0,
        "funding_rows_opened": 0,
        "source_diagnostics": copy_json(source_diagnostics),
        "clock_counts": {
            control: len(clocks[control]) for control in expected_controls
        },
        "support_statistics": stats,
        "maximum_accepted_entry_gap_seconds": maximum_gap,
        "maximum_same_side_run": maximum_run,
        "structural_control_distinctness": distinctness,
        "selection_prefix_append_invariance": copy_json(invariant),
        "checks": checks,
        "passed": all(checks.values()),
        "failure_action": (
            None
            if all(checks.values())
            else "retire exact CVVH-432 unchanged before novelty"
        ),
    }
    return {**core, "support_manifest_hash": prereg.canonical_hash(core)}


def evaluate_support(
    rows: Sequence[mechanism.JoinedHour],
    source_diagnostics: Mapping[str, Any],
) -> tuple[
    dict[str, Any], dict[str, tuple[mechanism.ScheduledEvent, ...]]
]:
    clocks = build_all_clocks(rows)
    invariant = append_invariance(rows)
    return evaluate_clock_support(
        clocks,
        source_diagnostics,
        invariant,
    ), clocks


def copy_json(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _clock_rows(
    events: Sequence[mechanism.ScheduledEvent],
) -> list[dict[str, str]]:
    ordered = sorted(
        events,
        key=lambda row: (
            row.entry_time,
            row.signal_time,
            row.signal_id,
            row.side,
        ),
    )
    return [
        {
            "policy_id": mechanism.CANDIDATE_ID,
            "control": row.control,
            "signal_id": row.signal_id,
            "signal_time_utc": _format_time(row.signal_time),
            "source_available_at_utc": _format_time(row.available_at),
            "entry_time_utc": _format_time(row.entry_time),
            "exit_time_utc": _format_time(row.exit_time),
            "side": row.side,
        }
        for row in ordered
    ]


def _gzip_csv_bytes(events: Sequence[mechanism.ScheduledEvent]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=CLOCK_HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_clock_rows(events))
    raw = text.getvalue().encode("utf-8")
    output = io.BytesIO()
    with gzip.GzipFile(
        fileobj=output,
        mode="wb",
        filename="",
        mtime=0,
    ) as handle:
        handle.write(raw)
    return output.getvalue()


def _write_file(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_once(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any destination."""

    library = ctypes.CDLL(None, use_errno=True)
    renameat2: Any = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise SourceSupportTerminalError(
            "CVVH-432 renameat2(RENAME_NOREPLACE) is unavailable"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "CVVH-432 support output already exists",
            str(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def publish_bundle(
    output: Path,
    report: Mapping[str, Any],
    clocks: Mapping[str, Sequence[mechanism.ScheduledEvent]],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"CVVH-432 support output exists: {output}")
    expected_controls = (
        *mechanism.OWN_CLOCKS,
        *mechanism.PARENT_SET_CONTROLS,
    )
    if set(clocks) != set(expected_controls):
        raise SourceSupportTerminalError(
            "CVVH-432 bundle clocks do not match the frozen control set"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(
        f".{output.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    staging.mkdir()
    try:
        artifacts: dict[str, Any] = {}
        for control in expected_controls:
            clock = tuple(clocks[control])
            if any(row.control != control for row in clock):
                raise SourceSupportTerminalError(
                    f"CVVH-432 bundle control label drift: {control}"
                )
            relative = (
                "primary.csv.gz"
                if control == mechanism.PRIMARY
                else f"controls/{control}.csv.gz"
            )
            raw = _gzip_csv_bytes(clock)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_file(destination, raw)
            artifacts[control] = {
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "rows": len(clock),
                "header": CLOCK_HEADER,
            }
        report_core = {
            **copy_json(report),
            "clock_artifacts": artifacts,
        }
        final_report = {
            **report_core,
            "bundle_manifest_hash": prereg.canonical_hash(report_core),
        }
        raw_report = (
            json.dumps(
                final_report,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _write_file(staging / "report.json", raw_report)
        controls_directory = staging / "controls"
        if controls_directory.exists():
            _fsync_directory(controls_directory)
        _fsync_directory(staging)
        if output.exists():
            raise FileExistsError(f"CVVH-432 support output exists: {output}")
        _rename_directory_once(staging, output)
        try:
            _fsync_directory(output.parent)
        except BaseException as error:
            raise SourceSupportPublicationCommittedError(
                "CVVH-432 bundle was published but parent fsync failed"
            ) from error
        return final_report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_registration(repo: Path, cfg: Config) -> dict[str, Any]:
    path = repo / cfg.preregistration
    if prereg.sha256_file(path) != PREREGISTRATION_SHA256:
        raise SourceSupportTerminalError("CVVH-432 preregistration bytes drift")
    raw = json.loads(path.read_text(encoding="utf-8"))
    prereg.validate_registration(raw)
    if raw.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise SourceSupportTerminalError("CVVH-432 preregistration identity drift")
    return raw


def validate_config(cfg: Config) -> None:
    canonical = Config()
    for name in (
        "bvol",
        "bvol_manifest",
        "dvol",
        "dvol_summary",
        "preregistration",
        "attempt_claim",
        "output_directory",
        "failure_receipt",
    ):
        if getattr(cfg, name) != getattr(canonical, name):
            raise SourceSupportTerminalError(
                f"CVVH-432 authoritative path drift: {name}"
            )


def _gzip_header_from_compressed(
    compressed: bytes,
    label: str,
) -> tuple[list[str], str]:
    with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as handle:
        raw = handle.readline()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceSupportTerminalError(
            f"CVVH-432 non-UTF-8 gzip header: {label}"
        ) from exc
    if not text.endswith("\n"):
        raise SourceSupportTerminalError(
            f"CVVH-432 gzip header lacks newline: {label}"
        )
    return (
        text.rstrip("\r\n").split(","),
        hashlib.sha256(raw).hexdigest(),
    )


def validate_source_bindings(repo: Path, cfg: Config) -> CapturedSources:
    expected = {
        "binance_btc_bvol_hourly": cfg.bvol,
        "binance_btc_bvol_manifest": cfg.bvol_manifest,
        "deribit_btc_dvol_hourly": cfg.dvol,
        "deribit_btc_dvol_summary": cfg.dvol_summary,
    }
    output: dict[str, Any] = {}
    compressed_sources: dict[str, bytes] = {}
    for name, configured in expected.items():
        spec = prereg.SOURCE_ARTIFACTS[name]
        if configured != spec["path"]:
            raise SourceSupportTerminalError(
                f"CVVH-432 configured source path drift: {name}"
            )
        path = repo / configured
        raw = path.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        if observed != spec["sha256"]:
            raise SourceSupportTerminalError(
                f"CVVH-432 source hash drift: {name}"
            )
        row: dict[str, Any] = {
            "path": configured,
            "sha256": observed,
            "bytes": len(raw),
        }
        if "header" in spec:
            header, header_hash = _gzip_header_from_compressed(
                raw,
                configured,
            )
            if (
                header != spec["header"]
                or header_hash != spec["header_line_sha256"]
            ):
                raise SourceSupportTerminalError(
                    f"CVVH-432 source header drift: {name}"
                )
            row["header_line_sha256"] = header_hash
            row["rows_decoded_before_claim"] = 0
            compressed_sources[name] = raw
        output[name] = row
    return CapturedSources(
        bindings=copy_json(output),
        bvol_compressed=compressed_sources["binance_btc_bvol_hourly"],
        dvol_compressed=compressed_sources["deribit_btc_dvol_hourly"],
    )


def validate_preregistered_protocol_seal(
    repo: Path,
    registration: Mapping[str, Any],
) -> str:
    repository = registration.get("repository")
    if not isinstance(repository, Mapping):
        raise SourceSupportTerminalError(
            "CVVH-432 preregistered repository seal is missing"
        )
    seal = repository.get("protocol_seal")
    expected_hash = repository.get("protocol_seal_hash")
    if not isinstance(seal, Mapping) or not isinstance(expected_hash, str):
        raise SourceSupportTerminalError(
            "CVVH-432 preregistered protocol seal is malformed"
        )
    if prereg.canonical_hash(seal) != expected_hash:
        raise SourceSupportTerminalError(
            "CVVH-432 preregistered protocol seal hash drift"
        )
    for relative, raw_spec in seal.items():
        if not isinstance(relative, str) or not isinstance(raw_spec, Mapping):
            raise SourceSupportTerminalError(
                "CVVH-432 preregistered protocol seal row is malformed"
            )
        path = repo / relative
        if not path.is_file():
            raise SourceSupportTerminalError(
                f"CVVH-432 preregistered path missing: {relative}"
            )
        current_blob = prereg._run_git(
            repo,
            "rev-parse",
            f"HEAD:{relative}",
        )
        if (
            current_blob != raw_spec.get("git_blob")
            or prereg.sha256_file(path) != raw_spec.get("sha256")
        ):
            raise SourceSupportTerminalError(
                f"CVVH-432 preregistered protocol path drift: {relative}"
            )
    return expected_hash


def evaluator_identity(
    repo: Path,
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    repository = prereg.repository_identity(repo)
    preregistered_seal_hash = validate_preregistered_protocol_seal(
        repo,
        registration,
    )
    seal: dict[str, Any] = {}
    for relative in EVALUATOR_PATHS:
        path = repo / relative
        if not path.is_file():
            raise SourceSupportTerminalError(
                f"CVVH-432 evaluator path missing: {relative}"
            )
        prereg._run_git(repo, "ls-files", "--error-unmatch", relative)
        working_blob = prereg._run_git(repo, "hash-object", relative)
        committed_blob = prereg._run_git(repo, "rev-parse", f"HEAD:{relative}")
        if working_blob != committed_blob:
            raise SourceSupportTerminalError(
                f"CVVH-432 evaluator path is not committed: {relative}"
            )
        seal[relative] = {
            "git_blob": committed_blob,
            "sha256": prereg.sha256_file(path),
        }
    return {
        "repository": repository,
        "preregistered_protocol_seal_hash": preregistered_seal_hash,
        "evaluator_seal": seal,
        "evaluator_seal_hash": prereg.canonical_hash(seal),
    }


def attempt_claim_payload(
    *,
    cfg: Config,
    identity: Mapping[str, Any],
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "protocol_version": ATTEMPT_PROTOCOL,
        "policy_id": mechanism.CANDIDATE_ID,
        "status": "claimed_before_first_bvol_or_dvol_value_decode",
        "repository": copy_json(identity["repository"]),
        "evaluator_seal": copy_json(identity["evaluator_seal"]),
        "evaluator_seal_hash": identity["evaluator_seal_hash"],
        "preregistered_protocol_seal_hash": identity[
            "preregistered_protocol_seal_hash"
        ],
        "preregistration": {
            "path": cfg.preregistration,
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "sources": copy_json(sources),
        "source_transport": {
            "hash_header_and_snapshot_use_same_compressed_bytes": True,
            "compressed_snapshots_retained_before_claim": True,
            "value_rows_decompressed_only_after_claim": True,
            "source_paths_reopened_after_claim": False,
        },
        "source_calendar": {
            "source_start_utc": _format_time(CANONICAL_SOURCE_CALENDAR.start),
            "source_end_utc": _format_time(CANONICAL_SOURCE_CALENDAR.end),
            "bvol_expected_rows": CANONICAL_SOURCE_CALENDAR.bvol_rows,
            "dvol_expected_rows": CANONICAL_SOURCE_CALENDAR.dvol_rows,
            "dvol_join_close_time_operator": "<",
            "dvol_join_close_time_bound_utc": _format_time(
                CANONICAL_SOURCE_CALENDAR.end
            ),
            "mechanism_close_time_operator": "<",
            "mechanism_close_time_bound_utc": _format_time(
                CANONICAL_SOURCE_CALENDAR.mechanism_end
            ),
        },
        "protected_reads_at_claim": {
            "bvol_rows_decoded": 0,
            "dvol_rows_decoded": 0,
            "candidate_incidence_opened": False,
            "comparator_rows_opened": 0,
            "gross9_rows_opened": 0,
            "btc_execution_rows_opened": 0,
            "funding_rows_opened": 0,
            "outcomes_opened": False,
        },
        "authoritative_attempts_allowed": 1,
        "retry_resume_fallback_or_repair_after_claim": False,
        "verification_replay": False,
        "attempt_claim": cfg.attempt_claim,
        "output_directory": cfg.output_directory,
        "failure_receipt": cfg.failure_receipt,
    }
    return {**core, "claim_hash": prereg.canonical_hash(core)}


def _encoded_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _failure_payload(
    *,
    claim: Mapping[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    core = {
        "protocol_version": FAILURE_PROTOCOL,
        "policy_id": mechanism.CANDIDATE_ID,
        "status": "terminal_authoritative_source_support_failure",
        "claim_hash": claim["claim_hash"],
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "retry_resume_fallback_or_repair_allowed": False,
        "verification_replay_allowed": False,
        "later_stages_allowed": False,
    }
    return {**core, "failure_hash": prereg.canonical_hash(core)}


def run_authoritative(
    repo: Path,
    cfg: Config = Config(),
) -> dict[str, Any]:
    validate_config(cfg)
    claim_path = repo / cfg.attempt_claim
    output = repo / cfg.output_directory
    failure_path = repo / cfg.failure_receipt
    if claim_path.exists() or output.exists() or failure_path.exists():
        raise SourceSupportTerminalError(
            "CVVH-432 support claim/output/failure already exists; no retry"
        )
    registration = load_registration(repo, cfg)
    identity = evaluator_identity(repo, registration)
    captured = validate_source_bindings(repo, cfg)
    claim = attempt_claim_payload(
        cfg=cfg,
        identity=identity,
        sources=captured.bindings,
    )
    prereg.atomic_write_once(claim_path, _encoded_json(claim))
    try:
        rows, source_diagnostics = load_captured_sources(captured)
        report, clocks = evaluate_support(rows, source_diagnostics)
        report = {
            **report,
            "attempt_claim": {
                "path": cfg.attempt_claim,
                "sha256": prereg.sha256_file(claim_path),
                "claim_hash": claim["claim_hash"],
            },
            "preregistration": {
                "path": cfg.preregistration,
                "sha256": PREREGISTRATION_SHA256,
                "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
            },
            "authoritative_attempt": 1,
            "retry_resume_fallback_or_repair_used": False,
            "repository_commit": identity["repository"]["commit"],
            "evaluator_seal_hash": identity["evaluator_seal_hash"],
            "preregistered_protocol_seal_hash": identity[
                "preregistered_protocol_seal_hash"
            ],
        }
        return publish_bundle(output, report, clocks)
    except SourceSupportPublicationCommittedError:
        raise
    except BaseException as error:
        failure = _failure_payload(claim=claim, error=error)
        if not failure_path.exists():
            prereg.atomic_write_once(failure_path, _encoded_json(failure))
        raise


def main(arguments: Sequence[str] | None = None) -> int:
    received = tuple(sys.argv[1:] if arguments is None else arguments)
    if received:
        raise SystemExit("CVVH-432 source support takes no arguments")
    repo = Path(__file__).resolve().parents[1]
    report = run_authoritative(repo)
    print(
        json.dumps(
            {
                "status": "support_passed" if report["passed"] else "support_failed",
                "passed": report["passed"],
                "output_directory": str(OUTPUT_DIRECTORY),
                "bundle_manifest_hash": report["bundle_manifest_hash"],
                "outcomes_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_CLAIM",
    "CANONICAL_SOURCE_CALENDAR",
    "CapturedSources",
    "CLOCK_HEADER",
    "Config",
    "EVALUATOR_PATHS",
    "FAILURE_RECEIPT",
    "FULL_END",
    "FULL_START",
    "OUTPUT_DIRECTORY",
    "PREREGISTRATION_MANIFEST_HASH",
    "PREREGISTRATION_SHA256",
    "PROTOCOL_VERSION",
    "SELECTION_END",
    "SOURCE_END",
    "SOURCE_START",
    "SourceCalendar",
    "SourceSupportPublicationCommittedError",
    "SourceSupportTerminalError",
    "WINDOWS",
    "append_invariance",
    "attempt_claim_payload",
    "build_all_clocks",
    "deterministic_one_to_one_pairs",
    "evaluate_clock_support",
    "evaluate_support",
    "exact_entry_jaccard",
    "join_loaded_sources",
    "load_bvol",
    "load_bvol_compressed",
    "load_captured_sources",
    "load_dvol",
    "load_dvol_compressed",
    "load_joined_sources_for_calendar",
    "publish_bundle",
    "run_authoritative",
    "structural_control_novelty",
    "validate_config",
    "validate_preregistered_protocol_seal",
    "validate_source_bindings",
]
