"""Outcome-blind event clock for IBRD-7 inflation breadth release drift."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd


SOURCE = Path(
    "data/bls_cpi_release_breadth_2019_2023/"
    "bls_cpi_release_breadth_2019_2023.csv.gz"
)
BUILD_MANIFEST = Path("data/bls_cpi_release_breadth_2019_2023/build_manifest.json")
SOURCE_MANIFEST = Path("data/bls_cpi_release_breadth_2019_2023/source_manifest.json")
SOURCE_SHA256 = "d199f409952d8cb83218864d0a96573bed82b59e649067b22fc97580a06d1059"
BUILD_MANIFEST_SHA256 = "fb546580e64a01a4247318c8d4dad87028686d190f51559a9162d3efa3235171"
SOURCE_MANIFEST_SHA256 = "7f889310707e4c490124ac2ce6817add7a227d6b0fa6d495c00405aba456aadc"

Component = Literal["headline", "core"]


@dataclass(frozen=True)
class SourceRow:
    reference_month: str
    release_time: pd.Timestamp
    headline_yoy_pct: Decimal
    core_yoy_pct: Decimal
    release_url: str


@dataclass(frozen=True)
class Event:
    reference_month: str
    signal_time: str
    entry_time: str
    exit_time: str
    side: int
    headline_yoy_pct: float
    core_yoy_pct: float
    headline_change_pct: float
    core_change_pct: float
    release_url: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result is pd.NaT:
        raise ValueError("IBRD-7 timestamp is NaT")
    return cast(pd.Timestamp, result)


def _decimal(value: Any) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("IBRD-7 CPI value is not finite")
    return result


def verify_source() -> None:
    expected = {
        SOURCE: SOURCE_SHA256,
        BUILD_MANIFEST: BUILD_MANIFEST_SHA256,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"IBRD-7 source hash mismatch: {path}")


def load_source() -> list[SourceRow]:
    verify_source()
    frame = pd.read_csv(SOURCE, dtype=str)
    required = {
        "reference_month",
        "release_time_utc",
        "headline_yoy_pct",
        "core_yoy_pct",
        "fred_crosscheck_passed",
        "release_url",
        "source_complete",
    }
    if not required.issubset(frame.columns):
        raise ValueError("IBRD-7 source schema changed")
    if not frame["source_complete"].str.lower().eq("true").all():
        raise ValueError("IBRD-7 source contains incomplete rows")
    if not frame["fred_crosscheck_passed"].str.lower().eq("true").all():
        raise ValueError("IBRD-7 source contains failed FRED cross-checks")

    rows = [
        SourceRow(
            reference_month=str(row["reference_month"]),
            release_time=_timestamp(row["release_time_utc"]),
            headline_yoy_pct=_decimal(row["headline_yoy_pct"]),
            core_yoy_pct=_decimal(row["core_yoy_pct"]),
            release_url=str(row["release_url"]),
        )
        for _, row in frame.iterrows()
    ]
    if len(rows) != 60:
        raise ValueError("IBRD-7 source row count changed")
    if any(current.release_time <= previous.release_time for previous, current in zip(rows, rows[1:])):
        raise ValueError("IBRD-7 release clock is not strictly increasing")
    return rows


def _side_from_change(change: Decimal) -> int:
    if change < 0:
        return 1
    if change > 0:
        return -1
    return 0


def _event(current: SourceRow, dh: Decimal, dc: Decimal, side: int) -> Event:
    signal = current.release_time
    entry = _timestamp(signal + pd.Timedelta(minutes=5))
    exit_time = _timestamp(entry + pd.Timedelta(days=7))
    return Event(
        reference_month=current.reference_month,
        signal_time=signal.isoformat(),
        entry_time=entry.isoformat(),
        exit_time=exit_time.isoformat(),
        side=side,
        headline_yoy_pct=float(current.headline_yoy_pct),
        core_yoy_pct=float(current.core_yoy_pct),
        headline_change_pct=float(dh),
        core_change_pct=float(dc),
        release_url=current.release_url,
    )


def build_events(rows: list[SourceRow], *, component: Component | None = None) -> list[Event]:
    events: list[Event] = []
    for previous, current in zip(rows, rows[1:]):
        dh = current.headline_yoy_pct - previous.headline_yoy_pct
        dc = current.core_yoy_pct - previous.core_yoy_pct
        if component == "headline":
            side = _side_from_change(dh)
        elif component == "core":
            side = _side_from_change(dc)
        else:
            headline_side = _side_from_change(dh)
            core_side = _side_from_change(dc)
            side = headline_side if headline_side == core_side else 0
        if side:
            events.append(_event(current, dh, dc, side))
    return events


def build_one_release_delay(rows: list[SourceRow]) -> list[Event]:
    primary_by_release = {
        event.signal_time: event for event in build_events(rows)
    }
    events: list[Event] = []
    for previous, current in zip(rows, rows[1:]):
        prior = primary_by_release.get(previous.release_time.isoformat())
        if prior is None:
            continue
        dh = current.headline_yoy_pct - previous.headline_yoy_pct
        dc = current.core_yoy_pct - previous.core_yoy_pct
        events.append(_event(current, dh, dc, prior.side))
    return events


def events_frame(events: list[Event]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(event) for event in events])
    if frame.empty:
        return frame
    for column in ("signal_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame = frame.sort_values(by=["entry_time"]).reset_index(drop=True)
    if not bool(frame["side"].isin((-1, 1)).all()):
        raise ValueError("IBRD-7 emitted invalid side")
    if not bool(
        frame["entry_time"].eq(frame["signal_time"] + pd.Timedelta(minutes=5)).all()
    ):
        raise ValueError("IBRD-7 entry delay changed")
    if not bool(
        frame["exit_time"].eq(frame["entry_time"] + pd.Timedelta(days=7)).all()
    ):
        raise ValueError("IBRD-7 hold changed")
    if len(frame) > 1:
        entries = frame["entry_time"].iloc[1:].reset_index(drop=True)
        exits = frame["exit_time"].iloc[:-1].reset_index(drop=True)
        if not bool(entries.ge(exits).all()):
            raise ValueError("IBRD-7 events overlap")
    return frame
