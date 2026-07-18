"""Outcome-blind event clocks for Cleveland Fed CPI Surprise (CFCS-1)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

import pandas as pd

from training import build_cleveland_fed_cpi_surprise_panel as source_builder


SOURCE = Path(
    "data/cleveland_fed_cpi_surprise_2019_2023/"
    "cleveland_fed_cpi_surprise_2019_2023.csv.gz"
)
BUILD_MANIFEST = Path("data/cleveland_fed_cpi_surprise_2019_2023/build_manifest.json")
SOURCE_SHA256 = "e8755bfd15ec135b2a85cedada8880bf5d4518ed07f4eef43b4b3820211d508e"
BUILD_MANIFEST_SHA256 = (
    "33f6719bae4d0b9e6c1edb8e93adc3f0cdd60891c92ec05ab92e77287bd946e6"
)
DEFAULT_OUTPUT = (
    "results/cleveland_fed_cpi_surprise_preregistered_clock_2026-07-18.csv.gz"
)
NEW_YORK = ZoneInfo("America/New_York")

ClockMode = Literal[
    "primary",
    "headline_only",
    "core_only",
    "composite_no_concordance",
]


@dataclass(frozen=True)
class SourceRow:
    reference_month: str
    release_time: datetime
    latest_nowcast_date: date
    headline_nowcast_mom_pct: Decimal
    core_nowcast_mom_pct: Decimal
    headline_actual_mom_pct: Decimal
    core_actual_mom_pct: Decimal
    headline_surprise_pct: Decimal
    core_surprise_pct: Decimal
    composite_surprise_pct: Decimal
    surprise_sign_concordant: bool


@dataclass(frozen=True)
class Event:
    reference_month: str
    signal_time: str
    entry_time: str
    exit_time: str
    side: int
    clock_mode: str
    headline_nowcast_mom_pct: float
    core_nowcast_mom_pct: float
    headline_actual_mom_pct: float
    core_actual_mom_pct: float
    headline_surprise_pct: float
    core_surprise_pct: float
    composite_surprise_pct: float
    surprise_sign_concordant: int


EVENT_COLUMNS = tuple(Event.__dataclass_fields__)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decimal(value: Any, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"CFCS-1 {field} must be decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"CFCS-1 {field} must be finite")
    return parsed


def _timestamp(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError("CFCS-1 release timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("CFCS-1 release timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def verify_source() -> None:
    expected = {
        SOURCE: SOURCE_SHA256,
        BUILD_MANIFEST: BUILD_MANIFEST_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"CFCS-1 source hash mismatch: {path}")


def load_source(path: str | Path = SOURCE) -> list[SourceRow]:
    source = Path(path)
    if source == SOURCE:
        verify_source()
    frame = pd.read_csv(source, dtype=str)
    if tuple(frame.columns) != source_builder.PANEL_COLUMNS:
        raise ValueError("CFCS-1 source schema changed")
    rows: list[SourceRow] = []
    for _, raw in frame.iterrows():
        release = _timestamp(raw["release_time_utc"])
        local_release = release.astimezone(NEW_YORK)
        if local_release.time().replace(tzinfo=None) != time(8, 30):
            raise ValueError("CFCS-1 release is not 08:30 America/New_York")
        nowcast_date = date.fromisoformat(str(raw["latest_nowcast_date"]))
        if nowcast_date >= local_release.date():
            raise ValueError("CFCS-1 nowcast is not strictly pre-release")
        headline_nowcast = _decimal(
            raw["headline_nowcast_mom_pct"], field="headline nowcast"
        )
        core_nowcast = _decimal(raw["core_nowcast_mom_pct"], field="core nowcast")
        headline_actual = _decimal(
            raw["headline_actual_mom_pct"], field="headline actual"
        )
        core_actual = _decimal(raw["core_actual_mom_pct"], field="core actual")
        headline_surprise = _decimal(
            raw["headline_surprise_pct"], field="headline surprise"
        )
        core_surprise = _decimal(raw["core_surprise_pct"], field="core surprise")
        composite = _decimal(raw["composite_surprise_pct"], field="composite surprise")
        arithmetic_tolerance = Decimal("0.00000000000001")
        if (
            abs(headline_actual - headline_nowcast - headline_surprise)
            > arithmetic_tolerance
        ):
            raise ValueError("CFCS-1 headline surprise arithmetic changed")
        if abs(core_actual - core_nowcast - core_surprise) > arithmetic_tolerance:
            raise ValueError("CFCS-1 core surprise arithmetic changed")
        if (
            abs((headline_surprise + core_surprise) / Decimal(2) - composite)
            > arithmetic_tolerance
        ):
            raise ValueError("CFCS-1 composite surprise arithmetic changed")
        concordant = str(raw["surprise_sign_concordant"]) == "1"
        if concordant != (headline_surprise * core_surprise > 0):
            raise ValueError("CFCS-1 surprise concordance changed")
        rows.append(
            SourceRow(
                reference_month=str(raw["reference_month"]),
                release_time=release,
                latest_nowcast_date=nowcast_date,
                headline_nowcast_mom_pct=headline_nowcast,
                core_nowcast_mom_pct=core_nowcast,
                headline_actual_mom_pct=headline_actual,
                core_actual_mom_pct=core_actual,
                headline_surprise_pct=headline_surprise,
                core_surprise_pct=core_surprise,
                composite_surprise_pct=composite,
                surprise_sign_concordant=concordant,
            )
        )
    if len(rows) != 60:
        raise ValueError("CFCS-1 source row count changed")
    if any(
        current.release_time <= previous.release_time
        for previous, current in zip(rows, rows[1:])
    ):
        raise ValueError("CFCS-1 release clock is not strictly increasing")
    if rows[-1].release_time >= datetime(2024, 1, 1, tzinfo=timezone.utc):
        raise ValueError("CFCS-1 source escaped the frozen pre-2024 horizon")
    return rows


def _side(value: Decimal) -> int:
    if value < 0:
        return 1
    if value > 0:
        return -1
    return 0


def _signal_value(row: SourceRow, mode: ClockMode) -> Decimal | None:
    if mode == "primary":
        return row.composite_surprise_pct if row.surprise_sign_concordant else None
    if mode == "headline_only":
        return row.headline_surprise_pct
    if mode == "core_only":
        return row.core_surprise_pct
    if mode == "composite_no_concordance":
        return row.composite_surprise_pct
    raise ValueError(f"unknown CFCS-1 clock mode: {mode}")


def _execution_times(release: datetime, delay_days: int) -> tuple[datetime, datetime]:
    release_local = release.astimezone(NEW_YORK)
    execution_date = release_local.date() + timedelta(days=delay_days)
    entry = datetime.combine(execution_date, time(8, 35), tzinfo=NEW_YORK)
    exit_time = datetime.combine(execution_date, time(16, 0), tzinfo=NEW_YORK)
    if entry <= release and delay_days == 0:
        raise ValueError("CFCS-1 entry must follow the release")
    if exit_time <= entry:
        raise ValueError("CFCS-1 exit must follow entry")
    return entry.astimezone(timezone.utc), exit_time.astimezone(timezone.utc)


def build_events(
    rows: list[SourceRow],
    *,
    mode: ClockMode = "primary",
    threshold_pct: Decimal | str | float = Decimal("0.05"),
    delay_days: int = 0,
) -> list[Event]:
    threshold = _decimal(threshold_pct, field="threshold")
    if threshold <= 0 or threshold >= Decimal("1"):
        raise ValueError("CFCS-1 threshold must be in (0, 1) percentage points")
    if delay_days not in {0, 1, 7}:
        raise ValueError("CFCS-1 delay must be zero, one, or seven calendar days")
    events: list[Event] = []
    for row in rows:
        value = _signal_value(row, mode)
        if value is None or abs(value) < threshold:
            continue
        side = _side(value)
        if side == 0:
            continue
        entry, exit_time = _execution_times(row.release_time, delay_days)
        if delay_days == 1:
            clock_mode = "one_day_delay"
        elif delay_days == 7:
            clock_mode = "seven_day_placebo"
        else:
            clock_mode = mode
        events.append(
            Event(
                reference_month=row.reference_month,
                signal_time=row.release_time.isoformat(),
                entry_time=entry.isoformat(),
                exit_time=exit_time.isoformat(),
                side=side,
                clock_mode=clock_mode,
                headline_nowcast_mom_pct=float(row.headline_nowcast_mom_pct),
                core_nowcast_mom_pct=float(row.core_nowcast_mom_pct),
                headline_actual_mom_pct=float(row.headline_actual_mom_pct),
                core_actual_mom_pct=float(row.core_actual_mom_pct),
                headline_surprise_pct=float(row.headline_surprise_pct),
                core_surprise_pct=float(row.core_surprise_pct),
                composite_surprise_pct=float(row.composite_surprise_pct),
                surprise_sign_concordant=int(row.surprise_sign_concordant),
            )
        )
    events_frame(events)
    return events


def events_frame(events: list[Event]) -> pd.DataFrame:
    records = [asdict(event) for event in events]
    frame = (
        pd.DataFrame.from_records(records)
        if records
        else pd.DataFrame(
            {column: pd.Series(dtype="object") for column in EVENT_COLUMNS}
        )
    )
    if frame.empty:
        return frame
    for column in ("signal_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame = frame.sort_values(by=["entry_time"]).reset_index(drop=True)
    if not bool(frame["side"].isin((-1, 1)).all()):
        raise ValueError("CFCS-1 emitted invalid side")
    if not bool(frame["entry_time"].gt(frame["signal_time"]).all()):
        raise ValueError("CFCS-1 entry no longer follows signal")
    if not bool(frame["exit_time"].gt(frame["entry_time"]).all()):
        raise ValueError("CFCS-1 exit no longer follows entry")
    if len(frame) > 1:
        entries = frame["entry_time"].iloc[1:].reset_index(drop=True)
        exits = frame["exit_time"].iloc[:-1].reset_index(drop=True)
        if not bool(entries.ge(exits).all()):
            raise ValueError("CFCS-1 events overlap")
    return frame


def cast_mode(value: str) -> ClockMode:
    if value not in {
        "primary",
        "headline_only",
        "core_only",
        "composite_no_concordance",
    }:
        raise ValueError(f"unknown CFCS-1 clock mode: {value}")
    return cast(ClockMode, value)
