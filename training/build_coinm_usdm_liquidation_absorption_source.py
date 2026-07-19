#!/usr/bin/env python3
"""Build a checksum-verified Binance USD-M BTCUSDT 5m activity source.

This source is intentionally activity-only for COIN-M/USD-M liquidation
absorption research setup work: it uses official Binance Vision USD-M daily
kline archives, verifies each published checksum, derives only quote-volume and
taker-flow fields, and discards raw zip payloads immediately after parsing.
It retains no OHLC prices, returns, PnL fields, labels, or signals.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
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

BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
DEFAULT_START = "2023-06-25"
DEFAULT_END = "2024-10-15"
DEFAULT_OUTPUT_DIR = "data/binance_um_activity_5m_2023_2024"
DEFAULT_MANIFEST = "results/binance_um_activity_5m_2023_2024_manifest.json"
SCHEMA_VERSION = 1
PANDAS_FREQUENCY = "5min"
RAW_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
OUTPUT_COLUMNS = (
    "date",
    "feature_available_time",
    "quote_asset_volume",
    "taker_buy_quote",
    "taker_sell_quote",
    "taker_imbalance",
    "number_of_trades",
)


@dataclass(frozen=True)
class BuildConfig:
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    output_dir: str = DEFAULT_OUTPUT_DIR
    manifest: str = DEFAULT_MANIFEST
    workers: int = 4
    retries: int = 5
    timeout_seconds: int = 60


def archive_url(symbol: str, interval: str, day: date) -> str:
    stem = f"{symbol}-{interval}-{day:%Y-%m-%d}.zip"
    return f"{BASE_URL}/{symbol}/{interval}/{stem}"


def checksum_url(symbol: str, interval: str, day: date) -> str:
    return archive_url(symbol, interval, day) + ".CHECKSUM"


def _days(start: date, end: date) -> list[date]:
    current = start
    output: list[date] = []
    while current < end:
        output.append(current)
        current += timedelta(days=1)
    return output


def _normalize_header_columns(columns: pd.Index) -> tuple[str, ...]:
    return tuple(str(column).strip().lower() for column in columns)


def read_activity_archive(payload: bytes) -> pd.DataFrame:
    """Parse one official kline zip and return only activity fields."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one kline CSV, found {members}")
        with archive.open(members[0]) as handle:
            first_line = handle.readline()
        has_header = first_line.lower().startswith(b"open_time,")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(
                handle,
                header=0 if has_header else None,
                names=None if has_header else list(RAW_COLUMNS),
                low_memory=False,
            )
    if has_header:
        frame.columns = list(_normalize_header_columns(frame.columns))
    if tuple(frame.columns) != RAW_COLUMNS:
        raise ValueError(f"unexpected kline columns: {frame.columns.tolist()}")
    if frame.empty:
        raise ValueError("kline archive is empty")

    for column in RAW_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    numeric = frame.loc[:, list(RAW_COLUMNS)].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("kline archive contains non-finite values")
    if not frame["open_time"].is_unique or not frame["open_time"].is_monotonic_increasing:
        raise ValueError("kline open times are not strictly increasing")
    if not (frame["close_time"].astype("int64") == frame["open_time"].astype("int64") + 299_999).all():
        raise ValueError("kline close times are not 5m bar ends")

    output = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["open_time"].astype("int64"), unit="ms", utc=True).dt.tz_localize(None),
            "quote_asset_volume": frame["quote_volume"].astype(float),
            "taker_buy_quote": frame["taker_buy_quote_volume"].astype(float),
            "number_of_trades": frame["count"].astype("int64"),
        }
    )
    output["feature_available_time"] = output["date"] + pd.Timedelta(minutes=5, seconds=1)
    output["taker_sell_quote"] = output["quote_asset_volume"] - output["taker_buy_quote"]
    with np.errstate(divide="ignore", invalid="ignore"):
        output["taker_imbalance"] = np.where(
            output["quote_asset_volume"].to_numpy(float) > 0.0,
            (output["taker_buy_quote"].to_numpy(float) - output["taker_sell_quote"].to_numpy(float))
            / output["quote_asset_volume"].to_numpy(float),
            0.0,
        )
    output = output.loc[:, list(OUTPUT_COLUMNS)]
    validate_activity_frame(output, require_complete_grid=False)
    return output


def validate_activity_frame(frame: pd.DataFrame, *, require_complete_grid: bool = True) -> None:
    if tuple(frame.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"unexpected activity columns: {frame.columns.tolist()}")
    if frame.empty:
        raise ValueError("activity frame is empty")
    dates = pd.to_datetime(frame["date"], errors="raise")
    available = pd.to_datetime(frame["feature_available_time"], errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("activity timestamps are not strictly increasing")
    if not available.equals(dates + pd.Timedelta(minutes=5, seconds=1)):
        raise ValueError("feature_available_time must equal bar end plus one second")
    expected = pd.date_range(dates.min(), dates.max() + pd.Timedelta(minutes=5), freq=PANDAS_FREQUENCY, inclusive="left")
    if require_complete_grid and not dates.equals(pd.Series(expected, name="date")):
        raise ValueError("activity frame has incomplete 5m timestamp coverage")

    numeric = frame.drop(columns=["date", "feature_available_time"])
    values = numeric.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("activity frame contains non-finite values")
    if (frame[["quote_asset_volume", "taker_buy_quote", "number_of_trades"]] < 0.0).any().any():
        raise ValueError("activity frame contains negative activity")
    integral_trades = cast(
        pd.Series, frame["number_of_trades"].astype(float).mod(1).ne(0)
    )
    if bool(integral_trades.any()):
        raise ValueError("number_of_trades must be integral")
    absolute_tolerance = 1e-7
    relative_tolerance = 1e-12
    quote_tolerance = absolute_tolerance + (
        frame["quote_asset_volume"].abs() * relative_tolerance
    )
    if (
        frame["taker_buy_quote"]
        > frame["quote_asset_volume"] + quote_tolerance
    ).any():
        raise ValueError("taker_buy_quote exceeds quote_asset_volume")
    if (frame["taker_sell_quote"] < -quote_tolerance).any():
        raise ValueError("taker_sell_quote is negative")
    if (
        frame["taker_sell_quote"]
        > frame["quote_asset_volume"] + quote_tolerance
    ).any():
        raise ValueError("taker_sell_quote exceeds quote_asset_volume")
    if not np.allclose(
        frame["taker_buy_quote"] + frame["taker_sell_quote"],
        frame["quote_asset_volume"],
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    ):
        raise ValueError("taker buy/sell quote does not sum to quote_asset_volume")
    if (
        (frame["taker_imbalance"] < -1.0 - absolute_tolerance)
        | (frame["taker_imbalance"] > 1.0 + absolute_tolerance)
    ).any():
        raise ValueError("taker_imbalance is outside [-1, 1]")


def process_day(
    day: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    checksum = expected_sha256(
        fetcher(checksum_url(cfg.symbol, cfg.interval, day), retries=cfg.retries, timeout=cfg.timeout_seconds)
    )
    payload = fetcher(archive_url(cfg.symbol, cfg.interval, day), retries=cfg.retries, timeout=cfg.timeout_seconds)
    archive_hash = verify_sha256(payload, checksum)
    frame = read_activity_archive(payload)
    expected = pd.date_range(day, day + timedelta(days=1), inclusive="left", freq=PANDAS_FREQUENCY)
    if not frame["date"].equals(pd.Series(expected, name="date")):
        raise ValueError(f"kline day {day} has incomplete timestamp coverage")
    validate_activity_frame(frame, require_complete_grid=True)
    return {
        "date": day.isoformat(),
        "archive_sha256": archive_hash,
        "rows": int(len(frame)),
        "first_date": str(frame["date"].min()),
        "last_date": str(frame["date"].max()),
        "frame": frame,
    }


def validate_config(cfg: BuildConfig) -> tuple[date, date]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if cfg.symbol != "BTCUSDT":
        raise ValueError("this source is frozen to Binance USD-M BTCUSDT")
    if cfg.interval != "5m":
        raise ValueError("this source is frozen to 5m klines")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    return start, end


def build(cfg: BuildConfig, *, fetcher: Callable[..., bytes] = _fetch_bytes) -> dict[str, Any]:
    start, end = validate_config(cfg)
    days = _days(start, end)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {executor.submit(process_day, day, cfg, fetcher=fetcher): day for day in days}
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            records.append(result)
            if completed % 50 == 0:
                print(f"processed {completed}/{len(days)} days", flush=True)
    records.sort(key=lambda item: item["date"])
    combined = pd.concat([item["frame"] for item in records], ignore_index=True)
    expected = pd.date_range(start, end, inclusive="left", freq=PANDAS_FREQUENCY)
    if not combined["date"].equals(pd.Series(expected, name="date")):
        raise ValueError("combined activity source has incomplete timestamp coverage")
    validate_activity_frame(combined, require_complete_grid=True)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{cfg.symbol}_{cfg.interval}_activity_{cfg.start}_{cfg.end}_exclusive.csv.gz"
    _write_gzip_csv(combined, output_path)
    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()

    archives = [{key: item[key] for key in ("date", "archive_sha256", "rows", "first_date", "last_date")} for item in records]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "name": "coinm_usdm_liquidation_absorption_usdm_btcusdt_activity_source",
            "source": "official Binance Vision USD-M daily kline archives",
            "archive_root": BASE_URL,
            "archive_checksums_verified": True,
            "raw_archives_retained": False,
            "activity_only": True,
            "retained_columns": list(OUTPUT_COLUMNS),
            "end_is_exclusive": True,
            "feature_available_time": "bar end plus one second",
            "outcomes_opened": False,
            "returns_pnl_or_signals_included": False,
            "prices_retained": False,
            "start_inclusive": cfg.start,
            "end_exclusive": cfg.end,
        },
        "derivations": {
            "taker_sell_quote": "quote_asset_volume - taker_buy_quote",
            "taker_imbalance": "(taker_buy_quote - taker_sell_quote) / quote_asset_volume; zero when quote_asset_volume is zero",
        },
        "validation": {
            "complete_5m_grid": True,
            "expected_rows": int(len(expected)),
            "actual_rows": int(len(combined)),
            "taker_bounds_checked": True,
            "feature_available_time_checked": True,
        },
        "file": {
            "path": str(output_path),
            "sha256": output_sha256,
            "rows": int(len(combined)),
            "columns": list(combined.columns),
            "first_date": str(combined["date"].min()),
            "last_date": str(combined["date"].max()),
            "first_feature_available_time": str(combined["feature_available_time"].min()),
            "last_feature_available_time": str(combined["feature_available_time"].max()),
        },
        "archives": archives,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    manifest_path = Path(cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=BuildConfig.symbol)
    parser.add_argument("--interval", default=BuildConfig.interval)
    parser.add_argument("--start", default=BuildConfig.start)
    parser.add_argument("--end", default=BuildConfig.end)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--manifest", default=BuildConfig.manifest)
    parser.add_argument("--workers", type=int, default=BuildConfig.workers)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument("--timeout-seconds", type=int, default=BuildConfig.timeout_seconds)
    return parser.parse_args()


def main() -> None:
    manifest = build(BuildConfig(**vars(parse_args())))
    print(json.dumps({"file": manifest["file"], "archives": len(manifest["archives"])}, indent=2))


if __name__ == "__main__":
    main()
