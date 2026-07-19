"""Build a causal, price-free ETH COIN-M liquidation-snapshot panel.

The official ``ETHUSD_PERP`` daily archives contain a censored force-order
snapshot stream rather than a complete liquidation-fill tape.  This builder
checksum-verifies every published archive, removes exact duplicate snapshots,
marks missing archive days invalid, and retains only contract-count activity
needed by the preregistered ETH-to-BTC relay hypothesis.  Raw archives and
price fields are never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.build_binance_aggtrade_microstructure import (  # noqa: E402
    _fetch_bytes,
    _write_gzip_csv,
    expected_sha256,
    verify_sha256,
)


BASE_URL = "https://data.binance.vision/data/futures/cm/daily/liquidationSnapshot"
SYMBOL = "ETHUSD_PERP"
FIRST_ARCHIVE_DATE = date(2023, 6, 25)
LAST_ARCHIVE_EXCLUSIVE = date(2024, 10, 15)
RAW_COLUMNS = (
    "time",
    "side",
    "order_type",
    "time_in_force",
    "original_quantity",
    "price",
    "average_price",
    "order_status",
    "last_fill_quantity",
    "accumulated_fill_quantity",
)
NUMERIC_COLUMNS = (
    "original_quantity",
    "price",
    "average_price",
    "last_fill_quantity",
    "accumulated_fill_quantity",
)
FEATURE_COLUMNS = (
    "event_count",
    "short_liquidation_event_count",
    "long_liquidation_event_count",
    "short_liquidation_contracts",
    "long_liquidation_contracts",
    "total_liquidation_contracts",
    "signed_liquidation_contracts",
    "liquidation_imbalance",
)
OUTPUT_COLUMNS = (
    "date",
    "feature_available_time",
    "source_valid",
    *FEATURE_COLUMNS,
)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Config:
    start: str = FIRST_ARCHIVE_DATE.isoformat()
    end: str = LAST_ARCHIVE_EXCLUSIVE.isoformat()
    output_dir: str = "data/binance_coinm_liquidation_snapshot_eth_2023_2024"
    manifest: str = (
        "results/binance_coinm_liquidation_snapshot_eth_2023_2024_manifest.json"
    )
    workers: int = 12
    retries: int = 5
    timeout_seconds: int = 60


def archive_url(day: date) -> str:
    stem = f"{SYMBOL}-liquidationSnapshot-{day:%Y-%m-%d}.zip"
    return f"{BASE_URL}/{SYMBOL}/{stem}"


def checksum_url(day: date) -> str:
    return archive_url(day) + ".CHECKSUM"


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days)]


def _empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in RAW_COLUMNS})


def read_archive(payload: bytes) -> tuple[pd.DataFrame, int]:
    """Parse one archive and remove exact duplicate snapshots only."""

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one liquidation CSV, found {members}")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle, low_memory=False)

    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if tuple(frame.columns) != RAW_COLUMNS:
        raise ValueError(f"unexpected liquidation columns: {frame.columns.tolist()}")
    if frame.empty:
        return _empty_event_frame(), 0

    frame["time"] = cast(
        pd.Series, pd.to_numeric(frame["time"], errors="raise")
    ).astype("int64")
    for column in NUMERIC_COLUMNS:
        frame[column] = cast(
            pd.Series, pd.to_numeric(frame[column], errors="raise")
        ).astype(float)
    for column in ("side", "order_type", "time_in_force", "order_status"):
        frame[column] = frame[column].astype(str).str.strip().str.upper()

    if not bool(frame["side"].isin(("BUY", "SELL")).all()):
        raise ValueError("liquidation archive contains an unknown side")
    if not bool(frame["order_type"].eq("LIMIT").all()):
        raise ValueError("liquidation archive contains a non-LIMIT order")
    if not bool(frame["time_in_force"].eq("IOC").all()):
        raise ValueError("liquidation archive contains a non-IOC order")
    if not bool(frame["order_status"].eq("FILLED").all()):
        raise ValueError("liquidation archive contains a non-FILLED snapshot")
    values = frame[["time", *NUMERIC_COLUMNS]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("liquidation archive contains non-finite values")
    if bool((frame[list(NUMERIC_COLUMNS)] <= 0.0).any().any()):
        raise ValueError("liquidation archive contains non-positive values")

    quantity_columns = (
        "original_quantity",
        "last_fill_quantity",
        "accumulated_fill_quantity",
    )
    quantities = frame[list(quantity_columns)].to_numpy(float)
    if not np.equal(quantities, np.floor(quantities)).all():
        raise ValueError("ETHUSD_PERP contract quantities must be integers")
    if bool(
        (frame["accumulated_fill_quantity"] > frame["original_quantity"]).any()
    ):
        raise ValueError("accumulated fill exceeds original quantity")
    if bool(
        (frame["last_fill_quantity"] > frame["accumulated_fill_quantity"]).any()
    ):
        raise ValueError("last fill exceeds accumulated fill")
    buy = cast(pd.Series, frame["side"].eq("BUY"))
    if bool((frame.loc[buy, "average_price"] > frame.loc[buy, "price"]).any()):
        raise ValueError("BUY average price exceeds its limit")
    if bool((frame.loc[~buy, "average_price"] < frame.loc[~buy, "price"]).any()):
        raise ValueError("SELL average price is below its limit")

    before = len(frame)
    frame = frame.drop_duplicates(keep="first").copy()
    duplicates_removed = before - len(frame)
    frame = frame.sort_values("time", kind="stable").reset_index(drop=True)
    if not frame["time"].is_monotonic_increasing:
        raise ValueError("liquidation timestamps are not monotonic")
    return frame, int(duplicates_removed)


def _day_grid(day: date, *, source_valid: bool) -> pd.DataFrame:
    timestamps = pd.date_range(
        day, day + timedelta(days=1), freq="5min", inclusive="left"
    )
    output = pd.DataFrame({"date": timestamps})
    output["feature_available_time"] = output["date"] + pd.Timedelta(
        minutes=5, seconds=1
    )
    output["source_valid"] = source_valid
    for column in FEATURE_COLUMNS:
        if not source_valid:
            output[column] = np.nan
        elif column.endswith("event_count") or column == "event_count":
            output[column] = 0
        else:
            output[column] = 0.0
    return output.loc[:, list(OUTPUT_COLUMNS)]


def aggregate_day(events: pd.DataFrame, day: date) -> pd.DataFrame:
    """Aggregate completed snapshots to price-free five-minute activity bars."""

    output = _day_grid(day, source_valid=True)
    if events.empty:
        return output

    work = events.copy()
    work["date"] = (
        pd.to_datetime(work["time"], unit="ms", utc=True)
        .dt.tz_localize(None)
        .dt.floor("5min")
    )
    row_by_date = {timestamp: index for index, timestamp in enumerate(output["date"])}
    for timestamp, group in work.groupby("date", sort=True):
        index = row_by_date.get(timestamp)
        if index is None:
            raise ValueError(f"event falls outside UTC day {day}: {timestamp}")
        buy = cast(pd.Series, group["side"].eq("BUY"))
        sell = ~buy
        short_contracts = float(
            group.loc[buy, "accumulated_fill_quantity"].sum()
        )
        long_contracts = float(
            group.loc[sell, "accumulated_fill_quantity"].sum()
        )
        total_contracts = short_contracts + long_contracts
        output.loc[index, "event_count"] = int(len(group))
        output.loc[index, "short_liquidation_event_count"] = int(buy.sum())
        output.loc[index, "long_liquidation_event_count"] = int(sell.sum())
        output.loc[index, "short_liquidation_contracts"] = short_contracts
        output.loc[index, "long_liquidation_contracts"] = long_contracts
        output.loc[index, "total_liquidation_contracts"] = total_contracts
        output.loc[index, "signed_liquidation_contracts"] = (
            short_contracts - long_contracts
        )
        output.loc[index, "liquidation_imbalance"] = (
            (short_contracts - long_contracts) / total_contracts
        )
    return output


def _missing_day(day: date) -> dict[str, Any]:
    return {
        "date": day.isoformat(),
        "available": False,
        "reason": "official archive or checksum not published",
        "frame": _day_grid(day, source_valid=False),
    }


def process_day(
    day: date,
    cfg: Config,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    try:
        checksum_payload = fetcher(
            checksum_url(day), retries=cfg.retries, timeout=cfg.timeout_seconds
        )
        expected = expected_sha256(checksum_payload)
        payload = fetcher(
            archive_url(day), retries=cfg.retries, timeout=cfg.timeout_seconds
        )
    except FileNotFoundError:
        return _missing_day(day)

    archive_hash = verify_sha256(payload, expected)
    try:
        events, duplicates_removed = read_archive(payload)
    except Exception as error:
        raise ValueError(f"invalid liquidation archive for {day}") from error
    if not events.empty:
        timestamps = pd.to_datetime(
            events["time"], unit="ms", utc=True
        ).dt.tz_localize(None)
        day_start = pd.Timestamp(day)
        day_end = day_start + pd.Timedelta(days=1)
        if timestamps.lt(day_start).any() or timestamps.ge(day_end).any():
            raise ValueError(f"liquidation archive {day} contains another UTC date")
    frame = aggregate_day(events, day)
    return {
        "date": day.isoformat(),
        "available": True,
        "archive_url": archive_url(day),
        "checksum_url": checksum_url(day),
        "archive_sha256": archive_hash,
        "expected_archive_sha256": expected,
        "checksum_payload_sha256": hashlib.sha256(checksum_payload).hexdigest(),
        "raw_rows": int(len(events) + duplicates_removed),
        "snapshot_rows": int(len(events)),
        "duplicate_rows_removed": duplicates_removed,
        "event_bars": int(frame["event_count"].gt(0).sum()),
        "first_time_ms": None if events.empty else int(events["time"].min()),
        "last_time_ms": None if events.empty else int(events["time"].max()),
        "frame": frame,
    }


def _public_record(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "frame"}


def build(cfg: Config) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if start < FIRST_ARCHIVE_DATE or end > LAST_ARCHIVE_EXCLUSIVE:
        raise ValueError("liquidation build is physically bounded to the archive range")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {
            executor.submit(process_day, day, cfg): day for day in _days(start, end)
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["date"])

    panel = pd.concat([item["frame"] for item in results], ignore_index=True)
    expected_grid = pd.Series(
        pd.date_range(start, end, freq="5min", inclusive="left"), name="date"
    )
    if not panel["date"].equals(expected_grid):
        raise ValueError("combined liquidation panel has an invalid five-minute grid")
    if panel["date"].duplicated().any() or not panel["date"].is_monotonic_increasing:
        raise ValueError("combined liquidation timestamps are invalid")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_day = end - timedelta(days=1)
    output = output_dir / (
        f"{SYMBOL}_liquidation_5m_{start:%Y-%m-%d}_{last_day:%Y-%m-%d}.csv.gz"
    )
    _write_gzip_csv(panel, output)
    records = [_public_record(item) for item in results]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "name": "Binance ETH COIN-M force-order snapshot source",
            "outcomes_opened": False,
            "source_only": True,
            "prices_retained": False,
            "raw_archives_retained": False,
            "start_inclusive": str(pd.Timestamp(start)),
            "end_exclusive": str(pd.Timestamp(end)),
            "snapshot_censoring": (
                "latest liquidation order snapshot per symbol in each 1000ms "
                "interval; not a complete liquidation fill tape"
            ),
            "availability": (
                "completed five-minute bar plus one second; execution must use "
                "the next eligible market open"
            ),
            "side_mapping": {
                "BUY": "short position liquidation (forced buy)",
                "SELL": "long position liquidation (forced sell)",
            },
            "cross_symbol_scale_policy": (
                "retain native ETH contract counts; later relay features must "
                "normalize ETH and BTC against their own strictly-prior history"
            ),
        },
        "config": asdict(cfg),
        "archive_root": BASE_URL,
        "symbol": SYMBOL,
        "raw_columns": list(RAW_COLUMNS),
        "retained_columns": list(OUTPUT_COLUMNS),
        "missing_archive_dates": [
            item["date"] for item in records if not item["available"]
        ],
        "archives": records,
        "file": {
            "path": str(output),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "rows": int(len(panel)),
            "source_valid_rows": int(panel["source_valid"].sum()),
            "event_bars": int(panel["event_count"].fillna(0).gt(0).sum()),
            "snapshot_rows": int(sum(item.get("snapshot_rows", 0) for item in records)),
            "raw_rows": int(sum(item.get("raw_rows", 0) for item in records)),
            "duplicate_rows_removed": int(
                sum(item.get("duplicate_rows_removed", 0) for item in records)
            ),
            "first_date": str(panel["date"].min()),
            "last_date": str(panel["date"].max()),
        },
    }
    manifest_path = Path(cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=Config.start)
    parser.add_argument("--end", default=Config.end)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--manifest", default=Config.manifest)
    parser.add_argument("--workers", type=int, default=Config.workers)
    args = parser.parse_args()
    result = build(
        Config(
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            manifest=args.manifest,
            workers=args.workers,
        )
    )
    print(
        json.dumps(
            {
                "protocol": result["protocol"],
                "missing_archive_dates": result["missing_archive_dates"],
                "file": result["file"],
                "manifest": args.manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
