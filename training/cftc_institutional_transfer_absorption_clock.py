"""Outcome-blind event clocks for CITA-1 institutional transfer absorption."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd


SOURCE = Path(
    "data/cftc_institutional_transfer_absorption_2018_2023/"
    "cftc_institutional_transfer_absorption_2018_2023.csv.gz"
)
BUILD_MANIFEST = Path(
    "data/cftc_institutional_transfer_absorption_2018_2023/build_manifest.json"
)
SOURCE_MANIFEST = Path(
    "data/cftc_institutional_transfer_absorption_2018_2023/source_manifest.json"
)
SOURCE_SHA256 = "064eed3fa340b1701f4686d1176de2a10f39128abc5ebf846e8b6319b8144ee6"
BUILD_MANIFEST_SHA256 = "e9d4ca15da671c086265557e5d302518fd4b9a9ad59fe9d0ff1181d772d60406"
SOURCE_MANIFEST_SHA256 = "a594b02d1191c32f905c13be3faaa74ec2f3f0e04723d3b11b76ee8b454d6897"

Mode = Literal["asset_manager_only", "leveraged_contrarian_only"]


@dataclass(frozen=True)
class SourceRow:
    report_date: str
    available_time: pd.Timestamp
    asset_mgr_net_change: Decimal
    lev_money_net_change: Decimal
    official_zip_url: str
    special_publication_override: bool
    source_complete: bool


@dataclass(frozen=True)
class Event:
    report_date: str
    signal_time: str
    entry_time: str
    exit_time: str
    side: int
    asset_mgr_net_change: float
    lev_money_net_change: float
    official_zip_url: str
    special_publication_override: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result is pd.NaT:
        raise ValueError("CITA-1 timestamp is NaT")
    return cast(pd.Timestamp, result)


def _decimal(value: Any) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("CITA-1 source value is not finite")
    return result


def verify_source() -> None:
    expected = {
        SOURCE: SOURCE_SHA256,
        BUILD_MANIFEST: BUILD_MANIFEST_SHA256,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"CITA-1 source hash mismatch: {path}")


def load_source() -> list[SourceRow]:
    verify_source()
    frame = pd.read_csv(SOURCE, dtype=str)
    required = {
        "report_date",
        "available_time_utc",
        "asset_mgr_published_net_change",
        "lev_money_published_net_change",
        "official_zip_url",
        "special_publication_override",
        "source_complete",
    }
    if not required.issubset(frame.columns):
        raise ValueError("CITA-1 source schema changed")
    rows: list[SourceRow] = []
    for _, row in frame.iterrows():
        complete = str(row["source_complete"]).lower() == "true"
        rows.append(
            SourceRow(
                report_date=str(row["report_date"]),
                available_time=_timestamp(row["available_time_utc"]),
                asset_mgr_net_change=(
                    _decimal(row["asset_mgr_published_net_change"])
                    if complete
                    else Decimal(0)
                ),
                lev_money_net_change=(
                    _decimal(row["lev_money_published_net_change"])
                    if complete
                    else Decimal(0)
                ),
                official_zip_url=str(row["official_zip_url"]),
                special_publication_override=(
                    str(row["special_publication_override"]).lower() == "true"
                ),
                source_complete=complete,
            )
        )
    if len(rows) != 299:
        raise ValueError("CITA-1 source row count changed")
    if sum(not row.source_complete for row in rows) != 1:
        raise ValueError("CITA-1 quarantine count changed")
    if sum(row.special_publication_override for row in rows) != 7:
        raise ValueError("CITA-1 special-release count changed")
    if any(
        current.available_time <= previous.available_time
        for previous, current in zip(rows, rows[1:])
    ):
        raise ValueError("CITA-1 source availability is not strictly increasing")
    return rows


def _sign(value: Decimal) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def side_for_row(row: SourceRow, *, mode: Mode | None = None) -> int:
    asset = _sign(row.asset_mgr_net_change)
    leveraged = _sign(row.lev_money_net_change)
    if mode == "asset_manager_only":
        return asset
    if mode == "leveraged_contrarian_only":
        return -leveraged
    return asset if asset and asset == -leveraged else 0


def _event(row: SourceRow, side: int) -> Event:
    signal = row.available_time
    entry = _timestamp(signal + pd.Timedelta(minutes=5))
    exit_time = _timestamp(entry + pd.Timedelta(days=7))
    return Event(
        report_date=row.report_date,
        signal_time=signal.isoformat(),
        entry_time=entry.isoformat(),
        exit_time=exit_time.isoformat(),
        side=side,
        asset_mgr_net_change=float(row.asset_mgr_net_change),
        lev_money_net_change=float(row.lev_money_net_change),
        official_zip_url=row.official_zip_url,
        special_publication_override=row.special_publication_override,
    )


def _nonoverlapping(events: list[Event]) -> list[Event]:
    accepted: list[Event] = []
    previous_exit: pd.Timestamp | None = None
    for event in events:
        entry = _timestamp(event.entry_time)
        exit_time = _timestamp(event.exit_time)
        if previous_exit is not None and entry < previous_exit:
            continue
        accepted.append(event)
        previous_exit = exit_time
    return accepted


def build_events(rows: list[SourceRow], *, mode: Mode | None = None) -> list[Event]:
    candidates = [
        _event(row, side)
        for row in rows
        if row.source_complete and (side := side_for_row(row, mode=mode))
    ]
    return _nonoverlapping(candidates)


def build_one_release_delay(rows: list[SourceRow]) -> list[Event]:
    complete = [row for row in rows if row.source_complete]
    candidates: list[Event] = []
    for previous, current in zip(complete, complete[1:]):
        side = side_for_row(previous)
        if side:
            candidates.append(_event(current, side))
    return _nonoverlapping(candidates)


def events_frame(events: list[Event]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(event) for event in events])
    if frame.empty:
        return frame
    for column in ("signal_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame = frame.sort_values(by=["entry_time"]).reset_index(drop=True)
    if not bool(frame["side"].isin((-1, 1)).all()):
        raise ValueError("CITA-1 emitted invalid side")
    if not bool(
        frame["entry_time"].eq(frame["signal_time"] + pd.Timedelta(minutes=5)).all()
    ):
        raise ValueError("CITA-1 entry delay changed")
    if not bool(
        frame["exit_time"].eq(frame["entry_time"] + pd.Timedelta(days=7)).all()
    ):
        raise ValueError("CITA-1 hold changed")
    if len(frame) > 1:
        entries = frame["entry_time"].iloc[1:].reset_index(drop=True)
        exits = frame["exit_time"].iloc[:-1].reset_index(drop=True)
        if not bool(entries.ge(exits).all()):
            raise ValueError("CITA-1 events overlap")
    return frame
