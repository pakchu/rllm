"""Build a causal source-only panel of BTC stablecoin-quote spot flows.

The builder downloads official Binance Spot monthly hourly-kline archives,
verifies every published checksum, and retains only completed-hour base-volume,
trade-count, and taker-flow observables.  Price and quote-notional fields are
parsed solely to validate the upstream schema and are discarded before output.
No return, label, perpetual price, or funding field is read or computed here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

from training.build_binance_aggtrade_microstructure import (
    _fetch_bytes,
    _write_gzip_csv,
    expected_sha256,
    verify_sha256,
)


BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
INTERVAL = "1h"
SCHEMA_VERSION = 1
DEFAULT_SYMBOLS = ("BTCUSDT", "BTCUSDC", "BTCFDUSD")
ACTIVATION_UTC: dict[str, pd.Timestamp] = {
    "BTCUSDT": cast(pd.Timestamp, pd.Timestamp("2023-07-01T00:00:00Z")),
    "BTCUSDC": cast(pd.Timestamp, pd.Timestamp("2023-07-01T00:00:00Z")),
    "BTCFDUSD": cast(pd.Timestamp, pd.Timestamp("2023-08-04T08:00:00Z")),
}
RAW_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "base_volume",
    "close_time",
    "quote_notional",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
)
OUTPUT_COLUMNS = (
    "date",
    "symbol",
    "open_time_us",
    "close_time_us",
    "base_volume_btc",
    "trade_count",
    "taker_buy_base_btc",
    "taker_sell_base_btc",
    "signed_taker_flow_btc",
    "source_complete",
)


@dataclass(frozen=True)
class BuildConfig:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    start: str = "2023-07-01"
    end: str = "2026-07-01"
    output_dir: str = "data/binance_stablecoin_quote_flow_btc_2023_2026"
    workers: int = 8
    retries: int = 5
    timeout_seconds: int = 60


def _month_starts(start: date, end: date) -> list[date]:
    current = date(start.year, start.month, 1)
    months: list[date] = []
    while current < end:
        months.append(current)
        current = date(
            current.year + (current.month == 12),
            1 if current.month == 12 else current.month + 1,
            1,
        )
    return months


def archive_url(symbol: str, month: date) -> str:
    return (
        f"{BASE_URL}/{symbol}/{INTERVAL}/"
        f"{symbol}-{INTERVAL}-{month:%Y-%m}.zip"
    )


def checksum_url(symbol: str, month: date) -> str:
    return archive_url(symbol, month) + ".CHECKSUM"


def _timestamp_unit(values: pd.Series) -> tuple[str, int]:
    """Return Binance archive timestamp unit and one-hour step in that unit."""
    minimum = int(values.min())
    maximum = int(values.max())
    if minimum >= 10**15 and maximum < 10**17:
        return "us", 3_600_000_000
    if minimum >= 10**12 and maximum < 10**15:
        return "ms", 3_600_000
    raise ValueError("spot hourly-kline timestamps have an unsupported unit")


def read_archive(payload: bytes) -> tuple[pd.DataFrame, str]:
    """Read one official archive and fail closed on malformed source rows."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected exactly one CSV in archive, found {members}")
        with archive.open(members[0]) as handle:
            first_line = handle.readline().lower()
        has_header = first_line.startswith(b"open_time,")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(
                handle,
                header=0 if has_header else None,
                names=None if has_header else list(RAW_COLUMNS),
                dtype={
                    "open_time": "int64",
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "base_volume": "float64",
                    "close_time": "int64",
                    "quote_notional": "float64",
                    "trade_count": "int64",
                    "taker_buy_base": "float64",
                    "taker_buy_quote": "float64",
                    "ignore": "float64",
                },
                low_memory=False,
            )
    if has_header:
        frame.columns = [str(column).strip().lower() for column in frame.columns]
    if tuple(frame.columns) != RAW_COLUMNS:
        raise ValueError(f"unexpected spot kline columns: {frame.columns.tolist()}")
    if frame.empty:
        raise ValueError("spot hourly-kline archive is empty")
    if not frame["open_time"].is_monotonic_increasing or not frame["open_time"].is_unique:
        raise ValueError("spot hourly-kline open times are not strictly increasing")

    numeric = frame.loc[:, RAW_COLUMNS]
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("spot hourly-kline archive contains non-finite values")
    if (frame[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise ValueError("spot hourly-kline archive contains non-positive prices")
    if (frame[["base_volume", "quote_notional", "trade_count"]] < 0.0).any().any():
        raise ValueError("spot hourly-kline archive contains negative volume or count")
    tolerance = 1e-8
    if (
        (frame["taker_buy_base"] < -tolerance).any()
        or (frame["taker_buy_quote"] < -tolerance).any()
        or (frame["taker_buy_base"] > frame["base_volume"] + tolerance).any()
        or (frame["taker_buy_quote"] > frame["quote_notional"] + tolerance).any()
    ):
        raise ValueError("spot hourly-kline taker-buy fields violate total-volume bounds")

    open_time = cast(pd.Series, frame["open_time"])
    close_time = cast(pd.Series, frame["close_time"])
    unit, step = _timestamp_unit(open_time)
    open_times = pd.to_datetime(open_time, unit=unit, utc=True, errors="raise")
    aligned = (
        open_times.dt.minute.eq(0)
        & open_times.dt.second.eq(0)
        & open_times.dt.microsecond.eq(0)
    )
    if not bool(aligned.all()):
        raise ValueError("spot hourly-kline rows are not aligned to UTC hour opens")
    if not bool(close_time.eq(open_time + step - 1).all()):
        raise ValueError("spot hourly-kline close times do not span exact UTC hours")
    if not bool(open_time.diff().dropna().eq(step).all()):
        raise ValueError("spot hourly-kline archive has missing or non-hourly rows")
    return frame, unit


def source_panel(frame: pd.DataFrame, *, symbol: str, unit: str) -> pd.DataFrame:
    """Strip outcome-adjacent fields and expose only completed-hour flow."""
    multiplier = 1_000 if unit == "ms" else 1
    taker_sell = frame["base_volume"] - frame["taker_buy_base"]
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["open_time"], unit=unit, utc=True).dt.tz_localize(None),
            "symbol": symbol,
            "open_time_us": frame["open_time"].astype("int64") * multiplier,
            "close_time_us": frame["close_time"].astype("int64") * multiplier,
            "base_volume_btc": frame["base_volume"].astype(float),
            "trade_count": frame["trade_count"].astype("int64"),
            "taker_buy_base_btc": frame["taker_buy_base"].astype(float),
            "taker_sell_base_btc": taker_sell.clip(lower=0.0).astype(float),
            "signed_taker_flow_btc": (2.0 * frame["taker_buy_base"] - frame["base_volume"]).astype(float),
            "source_complete": True,
        }
    )
    output = output.loc[:, OUTPUT_COLUMNS]
    if not np.isfinite(output.loc[:, OUTPUT_COLUMNS[2:-1]].to_numpy(float)).all():
        raise ValueError("stablecoin-quote source rows contain non-finite values")
    return output


def _expected_hours(symbol: str, month: date) -> pd.DatetimeIndex:
    month_start = cast(pd.Timestamp, pd.Timestamp(month, tz="UTC"))
    month_end = cast(pd.Timestamp, month_start + pd.offsets.MonthBegin(1))
    start = max(month_start, ACTIVATION_UTC[symbol])
    if start >= month_end:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(start, month_end, freq="1h", inclusive="left")


def _process_archive(
    symbol: str,
    month: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checksum_payload = fetcher(
        checksum_url(symbol, month),
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    expected = expected_sha256(checksum_payload)
    payload = fetcher(
        archive_url(symbol, month),
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    archive_hash = verify_sha256(payload, expected)
    raw, unit = read_archive(payload)
    panel = source_panel(raw, symbol=symbol, unit=unit)
    observed = pd.DatetimeIndex(panel["date"]).tz_localize("UTC")
    expected_index = _expected_hours(symbol, month)
    if not observed.equals(expected_index):
        missing = expected_index.difference(observed)
        extra = observed.difference(expected_index)
        raise ValueError(
            f"{symbol} {month:%Y-%m} does not match its exact active UTC-hour grid; "
            f"missing={missing.tolist()[:10]}, extra={extra.tolist()[:10]}"
        )
    metadata = {
        "symbol": symbol,
        "month": f"{month:%Y-%m}",
        "archive_url": archive_url(symbol, month),
        "checksum_url": checksum_url(symbol, month),
        "archive_sha256": archive_hash,
        "timestamp_unit": unit,
        "rows": int(len(panel)),
        "first_date": panel["date"].min().isoformat(),
        "last_date": panel["date"].max().isoformat(),
    }
    return panel, metadata


def _validate_config(cfg: BuildConfig) -> tuple[date, date, tuple[str, ...]]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    symbols = tuple(symbol.strip().upper() for symbol in cfg.symbols)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if start.day != 1 or end.day != 1:
        raise ValueError("monthly spot build boundaries must be month starts")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    if not symbols or any(symbol not in ACTIVATION_UTC for symbol in symbols):
        raise ValueError("symbols must be non-empty members of the frozen quote basket")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique")
    return start, end, symbols


def build(
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    start, end, symbols = _validate_config(cfg)
    months = _month_starts(start, end)
    tasks = [
        (symbol, month)
        for symbol in symbols
        for month in months
        if len(_expected_hours(symbol, month)) > 0
    ]
    frames: list[pd.DataFrame] = []
    archives: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {
            executor.submit(_process_archive, symbol, month, cfg, fetcher=fetcher): (
                symbol,
                month,
            )
            for symbol, month in tasks
        }
        for future in as_completed(futures):
            symbol, month = futures[future]
            panel, metadata = future.result()
            frames.append(panel)
            archives.append(metadata)
            print(f"completed {symbol} {month:%Y-%m}: rows={len(panel)}", flush=True)

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    expected_pairs = [
        (timestamp.tz_localize(None), symbol)
        for symbol in sorted(symbols)
        for month in months
        for timestamp in _expected_hours(symbol, month)
    ]
    expected_index = pd.MultiIndex.from_tuples(
        sorted(expected_pairs), names=["date", "symbol"]
    )
    observed_frame = cast(pd.DataFrame, combined[["date", "symbol"]])
    observed_index = pd.MultiIndex.from_frame(observed_frame)
    if observed_index.has_duplicates or not observed_index.equals(expected_index):
        missing = expected_index.difference(observed_index)
        extra = observed_index.difference(expected_index)
        raise ValueError(
            "combined stablecoin-quote panel does not match the active date-symbol grid; "
            f"missing={missing.tolist()[:10]}, extra={extra.tolist()[:10]}"
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_hour = pd.Timestamp(end, tz="UTC") - pd.Timedelta(hours=1)
    combined_path = output_dir / (
        "BTC_stablecoin_quote_flow_1h_"
        f"{start.isoformat()}_{last_hour:%Y-%m-%dT%H}.csv.gz"
    )
    _write_gzip_csv(combined, combined_path)
    archives.sort(key=lambda item: (item["symbol"], item["month"]))
    config_record = asdict(cfg)
    config_record["symbols"] = list(symbols)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config": config_record,
        "activation_utc": {
            symbol: ACTIVATION_UTC[symbol].isoformat() for symbol in symbols
        },
        "protocol": {
            "source": "official Binance Spot monthly hourly-kline archives",
            "archive_root": BASE_URL,
            "archive_checksums_verified": True,
            "end_is_exclusive": True,
            "timestamp_transition": "milliseconds through 2024; microseconds from 2025; output normalized to microseconds",
            "hourly_bucket": "UTC open_time; exact completed one-hour source span",
            "launch_policy": "missing rows are allowed only before each frozen symbol activation; no fill or stale carry",
            "price_fields_retained": False,
            "quote_fields_retained": False,
            "raw_archives_persisted": False,
            "outcomes_opened": False,
        },
        "combined_output": str(combined_path),
        "combined_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        "rows": int(len(combined)),
        "complete_rows": int(combined["source_complete"].sum()),
        "expected_rows": int(len(expected_index)),
        "first_date": cast(pd.Timestamp, combined["date"].min()).isoformat(),
        "last_date": cast(pd.Timestamp, combined["date"].max()).isoformat(),
        "symbols": list(symbols),
        "columns": list(combined.columns),
        "archives": archives,
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--start", default=BuildConfig.start)
    parser.add_argument("--end", default=BuildConfig.end)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--workers", type=int, default=BuildConfig.workers)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument("--timeout-seconds", type=int, default=BuildConfig.timeout_seconds)
    args = parser.parse_args()
    manifest = build(
        BuildConfig(
            symbols=tuple(args.symbols),
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            workers=args.workers,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
        )
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "combined_output",
                    "combined_sha256",
                    "rows",
                    "complete_rows",
                    "first_date",
                    "last_date",
                    "symbols",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
