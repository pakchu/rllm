"""Build a source-only Binance USD-M/COIN-M BTC positioning panel.

The official Binance Vision daily ``metrics`` archives expose the same
five-minute positioning aggregates for USD-M ``BTCUSDT`` and COIN-M
``BTCUSD_PERP``.  This builder verifies every published archive checksum,
normalizes only exact duplicate rows, and writes a physically pre-2024 panel.
It never reads an executable price, future return, label, or strategy outcome.
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
from typing import Any, Callable

import numpy as np
import pandas as pd

from training.build_binance_aggtrade_microstructure import (
    _fetch_bytes,
    _write_gzip_csv,
    expected_sha256,
    verify_sha256,
)


BASE_URL = "https://data.binance.vision/data/futures"
VENUES = {
    "um": "BTCUSDT",
    "cm": "BTCUSD_PERP",
}
RAW_COLUMNS = (
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)
OPEN_INTEREST_COLUMNS = (
    "sum_open_interest",
    "sum_open_interest_value",
)
RATIO_COLUMNS = (
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)
REQUIRED_NUMERIC = (
    *OPEN_INTEREST_COLUMNS,
    "sum_taker_long_short_vol_ratio",
)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Config:
    start: str = "2021-07-08"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_cross_collateral_metrics_btc_2021_2023"
    manifest: str = (
        "results/binance_cross_collateral_metrics_btc_2021_2023_manifest.json"
    )
    workers: int = 12
    retries: int = 5
    timeout_seconds: int = 60


def archive_url(venue: str, symbol: str, day: date) -> str:
    stem = f"{symbol}-metrics-{day:%Y-%m-%d}.zip"
    return f"{BASE_URL}/{venue}/daily/metrics/{symbol}/{stem}"


def checksum_url(venue: str, symbol: str, day: date) -> str:
    return archive_url(venue, symbol, day) + ".CHECKSUM"


def _days(start: date, end: date) -> list[date]:
    values: list[date] = []
    current = start
    while current < end:
        values.append(current)
        current += timedelta(days=1)
    return values


def _deduplicate_exact(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicate_time = frame["create_time"].duplicated(keep=False)
    if not duplicate_time.any():
        return frame, 0
    duplicate_rows = frame.loc[duplicate_time]
    conflicts = duplicate_rows.groupby("create_time", sort=False).nunique(dropna=False)
    if conflicts.drop(columns=["create_time"], errors="ignore").gt(1).any().any():
        raise ValueError("metrics archive contains conflicting duplicate timestamps")
    output = frame.drop_duplicates(keep="first")
    removed = len(frame) - len(output)
    if output["create_time"].duplicated().any():
        raise ValueError("metrics archive duplicate timestamps were not exact rows")
    return output, int(removed)


def _coerce_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    blank = raw.isna() | raw.astype(str).str.strip().isin({"", "nan", "NaN"})
    if (numeric.isna() & ~blank).any():
        raise ValueError(f"metrics archive contains malformed {column}")
    return numeric


def read_archive(payload: bytes, *, symbol: str) -> tuple[pd.DataFrame, int]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one metrics CSV, found {members}")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle, low_memory=False)

    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if tuple(frame.columns) != RAW_COLUMNS:
        raise ValueError(f"unexpected metrics columns: {frame.columns.tolist()}")
    if frame.empty:
        raise ValueError("metrics archive is empty")
    if not frame["symbol"].eq(symbol).all():
        raise ValueError("metrics archive contains another symbol")

    frame["create_time"] = pd.to_datetime(
        frame["create_time"], utc=True, errors="raise"
    ).dt.tz_localize(None)
    for column in (*OPEN_INTEREST_COLUMNS, *RATIO_COLUMNS):
        frame[column] = _coerce_numeric(frame, column)
    open_interest = frame[list(OPEN_INTEREST_COLUMNS)].to_numpy(float)
    if np.isinf(open_interest).any() or (open_interest < 0.0).any():
        raise ValueError("metrics archive contains invalid open interest")
    unavailable_oi = frame[list(OPEN_INTEREST_COLUMNS)].isna().any(axis=1) | frame[
        list(OPEN_INTEREST_COLUMNS)
    ].le(0.0).any(axis=1)
    frame.loc[unavailable_oi, list(OPEN_INTEREST_COLUMNS)] = np.nan
    ratios = frame[list(RATIO_COLUMNS)].to_numpy(float)
    if np.isinf(ratios).any():
        raise ValueError("metrics archive contains infinite ratios")
    finite_ratios = np.isfinite(ratios)
    if (ratios[finite_ratios] < 0.0).any():
        raise ValueError("metrics archive contains negative ratios")
    zero_taker = frame["sum_taker_long_short_vol_ratio"].eq(0.0)
    frame.loc[zero_taker, "sum_taker_long_short_vol_ratio"] = np.nan

    frame, duplicate_rows_removed = _deduplicate_exact(frame)
    frame = frame.sort_values("create_time").reset_index(drop=True)
    if (
        frame["create_time"].duplicated().any()
        or not frame["create_time"].is_monotonic_increasing
    ):
        raise ValueError("metrics timestamps are invalid")
    minute = frame["create_time"].dt.minute
    if (
        frame["create_time"].dt.second.ne(0).any()
        or frame["create_time"].dt.microsecond.ne(0).any()
        or minute.mod(5).ne(0).any()
    ):
        raise ValueError("metrics timestamps are not aligned to five minutes")
    return frame, duplicate_rows_removed


def _empty_day(venue: str, symbol: str, day: date) -> dict[str, Any]:
    return {
        "venue": venue,
        "symbol": symbol,
        "date": day.isoformat(),
        "available": False,
        "reason": "official archive or checksum not published",
        "frame": pd.DataFrame(),
    }


def process_day(
    venue: str,
    symbol: str,
    day: date,
    cfg: Config,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    try:
        checksum_payload = fetcher(
            checksum_url(venue, symbol, day),
            retries=cfg.retries,
            timeout=cfg.timeout_seconds,
        )
        expected = expected_sha256(checksum_payload)
        payload = fetcher(
            archive_url(venue, symbol, day),
            retries=cfg.retries,
            timeout=cfg.timeout_seconds,
        )
    except FileNotFoundError:
        return _empty_day(venue, symbol, day)

    archive_hash = verify_sha256(payload, expected)
    try:
        frame, duplicates_removed = read_archive(payload, symbol=symbol)
    except Exception as error:
        raise ValueError(f"invalid {venue} metrics archive for {day}") from error
    day_start = pd.Timestamp(day)
    day_end = day_start + pd.Timedelta(days=1)
    if (
        frame["create_time"].lt(day_start).any()
        or frame["create_time"].ge(day_end).any()
    ):
        raise ValueError(f"{venue} metrics archive {day} contains another UTC date")
    expected_grid = pd.date_range(day_start, day_end, freq="5min", inclusive="left")
    missing = expected_grid.difference(frame["create_time"])
    invalid_oi_rows = int(frame[list(OPEN_INTEREST_COLUMNS)].isna().any(axis=1).sum())
    missing_taker_rows = int(frame["sum_taker_long_short_vol_ratio"].isna().sum())
    return {
        "venue": venue,
        "symbol": symbol,
        "date": day.isoformat(),
        "available": True,
        "archive_url": archive_url(venue, symbol, day),
        "checksum_url": checksum_url(venue, symbol, day),
        "archive_sha256": archive_hash,
        "expected_archive_sha256": expected,
        "checksum_payload_sha256": hashlib.sha256(checksum_payload).hexdigest(),
        "rows": int(len(frame)),
        "duplicate_rows_removed": duplicates_removed,
        "missing_five_minute_rows": int(len(missing)),
        "invalid_open_interest_rows": invalid_oi_rows,
        "missing_taker_ratio_rows": missing_taker_rows,
        "first_time": str(frame["create_time"].min()),
        "last_time": str(frame["create_time"].max()),
        "frame": frame,
    }


def _public_record(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "frame"}


def _prefix_frame(frame: pd.DataFrame, venue: str) -> pd.DataFrame:
    return frame.rename(
        columns={
            column: f"{venue}_{column}"
            for column in frame.columns
            if column != "create_time"
        }
    ).rename(columns={"create_time": "date"})


def build(cfg: Config) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if start < date(2021, 7, 8) or end > date(2024, 1, 1):
        raise ValueError("metrics build is physically bounded to 2021-07-08..2023")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")

    days = _days(start, end)
    tasks = [(venue, symbol, day) for venue, symbol in VENUES.items() for day in days]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {
            executor.submit(process_day, venue, symbol, day, cfg): (venue, day)
            for venue, symbol, day in tasks
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: (item["venue"], item["date"]))

    venue_panels: dict[str, pd.DataFrame] = {}
    for venue in VENUES:
        frames = [
            _prefix_frame(item["frame"], venue)
            for item in results
            if item["venue"] == venue and item["available"]
        ]
        venue_panels[venue] = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
        )
        if venue_panels[venue]["date"].duplicated().any():
            raise ValueError(f"{venue} metrics panel contains duplicate timestamps")

    full_grid = pd.DataFrame(
        {"date": pd.date_range(start, end, freq="5min", inclusive="left")}
    )
    panel = full_grid
    for venue in VENUES:
        panel = panel.merge(
            venue_panels[venue], on="date", how="left", validate="one_to_one"
        )
    required = [f"{venue}_{column}" for venue in VENUES for column in REQUIRED_NUMERIC]
    panel["source_complete"] = panel[required].notna().all(axis=1)
    if panel["date"].duplicated().any() or not panel["date"].is_monotonic_increasing:
        raise ValueError("combined metrics panel timestamps are invalid")
    if panel["date"].max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("combined metrics panel opened 2024+ rows")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "BTC_cross_collateral_metrics_5m_2021-07-08_2023-12-31.csv.gz"
    _write_gzip_csv(panel, output)
    output_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    records = [_public_record(item) for item in results]
    missing_archives = {
        venue: [
            item["date"]
            for item in records
            if item["venue"] == venue and not item["available"]
        ]
        for venue in VENUES
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "name": "Binance BTC USD-M/COIN-M positioning metrics pre-2024",
            "outcomes_opened": False,
            "start_inclusive": str(pd.Timestamp(start)),
            "end_exclusive": str(pd.Timestamp(end)),
            "post_2023_rows_requested": False,
            "source": "official public Binance Vision daily metrics archives",
            "source_only": True,
        },
        "config": asdict(cfg),
        "venues": VENUES,
        "archive_root": BASE_URL,
        "raw_columns": list(RAW_COLUMNS),
        "missing_archive_dates": missing_archives,
        "archives": records,
        "file": {
            "path": str(output),
            "sha256": output_hash,
            "rows": int(len(panel)),
            "source_complete_rows": int(panel["source_complete"].sum()),
            "first_date": str(panel["date"].min()),
            "last_date": str(panel["date"].max()),
            "duplicate_rows_removed": int(
                sum(item.get("duplicate_rows_removed", 0) for item in records)
            ),
            "invalid_open_interest_rows": {
                venue: int(
                    sum(
                        item.get("invalid_open_interest_rows", 0)
                        for item in records
                        if item["venue"] == venue
                    )
                )
                for venue in VENUES
            },
            "missing_taker_ratio_rows": {
                venue: int(
                    sum(
                        item.get("missing_taker_ratio_rows", 0)
                        for item in records
                        if item["venue"] == venue
                    )
                )
                for venue in VENUES
            },
            "missing_five_minute_rows": {
                venue: int(
                    sum(
                        item.get("missing_five_minute_rows", 0)
                        for item in records
                        if item["venue"] == venue
                    )
                )
                for venue in VENUES
            },
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
    parser.add_argument("--workers", type=int, default=Config.workers)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--manifest", default=Config.manifest)
    args = parser.parse_args()
    cfg = Config(
        workers=args.workers,
        output_dir=args.output_dir,
        manifest=args.manifest,
    )
    result = build(cfg)
    print(
        json.dumps(
            {
                "outcomes_opened": result["protocol"]["outcomes_opened"],
                "missing_archive_dates": result["missing_archive_dates"],
                "file": result["file"],
                "manifest": cfg.manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
