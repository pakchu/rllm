"""Outcome-blind event clock for EPSB-1 petroleum stock breadth."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd


SOURCE = Path(
    "data/eia_petroleum_stock_breadth_2019_2023/"
    "eia_petroleum_stock_breadth_2019_2023.csv.gz"
)
BUILD_MANIFEST = Path("data/eia_petroleum_stock_breadth_2019_2023/build_manifest.json")
SOURCE_MANIFEST = Path("data/eia_petroleum_stock_breadth_2019_2023/source_manifest.json")
SOURCE_SHA256 = "26cbe6a91079a64fd9bbcb1cb5e1f81e15df25e45ed2171f7c464d048b34757b"
BUILD_MANIFEST_SHA256 = "d6813b1a5677c9222a1197343900d6b03381f35ff9db8688892b77e4cd9c0661"
SOURCE_MANIFEST_SHA256 = "3969288900528d103016cdb0870a11269c1b352b9077faffdc61427f7fce29fb"

Mode = Literal["crude_only", "refined_products_only"]


@dataclass(frozen=True)
class SourceRow:
    release_date: str
    available_time: pd.Timestamp
    commercial_crude_change_mmbbl: Decimal
    gasoline_change_mmbbl: Decimal
    distillate_change_mmbbl: Decimal
    archive_page_url: str
    table1_csv_url: str
    source_complete: bool


@dataclass(frozen=True)
class Event:
    release_date: str
    signal_time: str
    entry_time: str
    exit_time: str
    side: int
    commercial_crude_change_mmbbl: float
    gasoline_change_mmbbl: float
    distillate_change_mmbbl: float
    archive_page_url: str
    table1_csv_url: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result is pd.NaT:
        raise ValueError("EPSB-1 timestamp is NaT")
    return cast(pd.Timestamp, result)


def _decimal(value: Any) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("EPSB-1 source value is not finite")
    return result


def verify_source() -> None:
    expected = {
        SOURCE: SOURCE_SHA256,
        BUILD_MANIFEST: BUILD_MANIFEST_SHA256,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"EPSB-1 source hash mismatch: {path}")


def load_source() -> list[SourceRow]:
    verify_source()
    frame = pd.read_csv(SOURCE, dtype=str)
    required = {
        "release_date",
        "available_time_utc",
        "commercial_crude_change_mmbbl",
        "gasoline_change_mmbbl",
        "distillate_change_mmbbl",
        "archive_page_url",
        "table1_csv_url",
        "source_complete",
    }
    if not required.issubset(frame.columns):
        raise ValueError("EPSB-1 source schema changed")
    rows = [
        SourceRow(
            release_date=str(row["release_date"]),
            available_time=_timestamp(row["available_time_utc"]),
            commercial_crude_change_mmbbl=_decimal(
                row["commercial_crude_change_mmbbl"]
            ),
            gasoline_change_mmbbl=_decimal(row["gasoline_change_mmbbl"]),
            distillate_change_mmbbl=_decimal(row["distillate_change_mmbbl"]),
            archive_page_url=str(row["archive_page_url"]),
            table1_csv_url=str(row["table1_csv_url"]),
            source_complete=str(row["source_complete"]).lower() == "true",
        )
        for _, row in frame.iterrows()
    ]
    if len(rows) != 259:
        raise ValueError("EPSB-1 source row count changed")
    if sum(not row.source_complete for row in rows) != 1:
        raise ValueError("EPSB-1 quarantine count changed")
    if any(
        current.available_time <= previous.available_time
        for previous, current in zip(rows, rows[1:])
    ):
        raise ValueError("EPSB-1 source clock is not strictly increasing")
    return rows


def _side_from_change(change: Decimal) -> int:
    if change > 0:
        return 1
    if change < 0:
        return -1
    return 0


def _event(current: SourceRow, side: int) -> Event:
    signal = current.available_time
    entry = _timestamp(signal + pd.Timedelta(minutes=5))
    exit_time = _timestamp(entry + pd.Timedelta(hours=72))
    return Event(
        release_date=current.release_date,
        signal_time=signal.isoformat(),
        entry_time=entry.isoformat(),
        exit_time=exit_time.isoformat(),
        side=side,
        commercial_crude_change_mmbbl=float(
            current.commercial_crude_change_mmbbl
        ),
        gasoline_change_mmbbl=float(current.gasoline_change_mmbbl),
        distillate_change_mmbbl=float(current.distillate_change_mmbbl),
        archive_page_url=current.archive_page_url,
        table1_csv_url=current.table1_csv_url,
    )


def build_events(rows: list[SourceRow], *, mode: Mode | None = None) -> list[Event]:
    events: list[Event] = []
    for current in rows:
        if not current.source_complete:
            continue
        crude = _side_from_change(current.commercial_crude_change_mmbbl)
        gasoline = _side_from_change(current.gasoline_change_mmbbl)
        distillate = _side_from_change(current.distillate_change_mmbbl)
        if mode == "crude_only":
            side = crude
        elif mode == "refined_products_only":
            side = gasoline if gasoline == distillate else 0
        else:
            side = crude if crude == gasoline == distillate else 0
        if side:
            events.append(_event(current, side))
    return events


def build_one_release_delay(rows: list[SourceRow]) -> list[Event]:
    complete = [row for row in rows if row.source_complete]
    primary_by_signal = {
        event.signal_time: event for event in build_events(complete)
    }
    events: list[Event] = []
    for previous, current in zip(complete, complete[1:]):
        prior = primary_by_signal.get(previous.available_time.isoformat())
        if prior is not None:
            events.append(_event(current, prior.side))
    return events


def events_frame(events: list[Event]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(event) for event in events])
    if frame.empty:
        return frame
    for column in ("signal_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame = frame.sort_values(by=["entry_time"]).reset_index(drop=True)
    if not bool(frame["side"].isin((-1, 1)).all()):
        raise ValueError("EPSB-1 emitted invalid side")
    if not bool(
        frame["entry_time"].eq(frame["signal_time"] + pd.Timedelta(minutes=5)).all()
    ):
        raise ValueError("EPSB-1 entry delay changed")
    if not bool(
        frame["exit_time"].eq(frame["entry_time"] + pd.Timedelta(hours=72)).all()
    ):
        raise ValueError("EPSB-1 hold changed")
    if len(frame) > 1:
        entries = frame["entry_time"].iloc[1:].reset_index(drop=True)
        exits = frame["exit_time"].iloc[:-1].reset_index(drop=True)
        if not bool(entries.ge(exits).all()):
            raise ValueError("EPSB-1 events overlap")
    return frame
