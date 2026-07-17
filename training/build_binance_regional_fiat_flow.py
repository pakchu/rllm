"""Build a causal source-only panel of regional BTC fiat spot flows.

The builder downloads official Binance Spot monthly daily-kline archives,
verifies every published checksum, and emits only same-day flow observables.
Price fields are parsed solely to validate the upstream archive schema and are
discarded before the panel is written.  No return, label, or forward field is
computed here.
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


BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
INTERVAL = "1d"
SCHEMA_VERSION = 1
DEFAULT_SYMBOLS = ("BTCUSDT", "BTCEUR", "BTCTRY", "BTCBRL")
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
    "open_time_ms",
    "close_time_ms",
    "base_volume_btc",
    "trade_count",
    "taker_buy_base_btc",
    "taker_sell_base_btc",
    "taker_buy_fraction",
    "source_complete",
)


@dataclass(frozen=True)
class BuildConfig:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    start: str = "2021-01-01"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_regional_fiat_flow_btc_2021_2023"
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


def read_archive(payload: bytes) -> pd.DataFrame:
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
        raise ValueError("spot daily-kline archive is empty")
    if not frame["open_time"].is_monotonic_increasing or not frame["open_time"].is_unique:
        raise ValueError("spot daily-kline open times are not strictly increasing")

    numeric = frame.loc[:, RAW_COLUMNS]
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("spot daily-kline archive contains non-finite values")
    prices = frame[["open", "high", "low", "close"]]
    if (prices <= 0.0).any().any():
        raise ValueError("spot daily-kline archive contains non-positive prices")
    if (frame[["base_volume", "quote_notional", "trade_count"]] < 0.0).any().any():
        raise ValueError("spot daily-kline archive contains negative volume or count")
    tolerance = 1e-8
    if (
        (frame["taker_buy_base"] < -tolerance).any()
        or (frame["taker_buy_quote"] < -tolerance).any()
        or (frame["taker_buy_base"] > frame["base_volume"] + tolerance).any()
        or (frame["taker_buy_quote"] > frame["quote_notional"] + tolerance).any()
    ):
        raise ValueError("spot daily-kline taker-buy fields violate total-volume bounds")

    open_times = pd.to_datetime(frame["open_time"], unit="ms", utc=True, errors="raise")
    if not (
        open_times.dt.hour.eq(0)
        & open_times.dt.minute.eq(0)
        & open_times.dt.second.eq(0)
        & open_times.dt.microsecond.eq(0)
    ).all():
        raise ValueError("spot daily-kline rows are not aligned to UTC day opens")
    expected_close = frame["open_time"] + 86_400_000 - 1
    if not frame["close_time"].eq(expected_close).all():
        raise ValueError("spot daily-kline close times do not span exact UTC days")
    expected_step = frame["open_time"].diff().dropna()
    if not expected_step.eq(86_400_000).all():
        raise ValueError("spot daily-kline archive has missing or non-daily rows")
    return frame


def source_panel(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """Strip price fields and expose only causal regional-flow observables."""
    taker_sell = frame["base_volume"] - frame["taker_buy_base"]
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.tz_localize(None),
            "symbol": symbol,
            "open_time_ms": frame["open_time"].astype("int64"),
            "close_time_ms": frame["close_time"].astype("int64"),
            "base_volume_btc": frame["base_volume"].astype(float),
            "trade_count": frame["trade_count"].astype("int64"),
            "taker_buy_base_btc": frame["taker_buy_base"].astype(float),
            "taker_sell_base_btc": taker_sell.astype(float),
            "taker_buy_fraction": frame["taker_buy_base"].divide(
                frame["base_volume"].replace(0.0, np.nan)
            ),
        }
    )
    output["source_complete"] = (
        output["base_volume_btc"].gt(0.0)
        & output["trade_count"].gt(0)
        & output["taker_buy_base_btc"].ge(0.0)
        & output["taker_sell_base_btc"].ge(-1e-8)
        & output["taker_buy_fraction"].between(0.0, 1.0, inclusive="both")
    )
    output["taker_sell_base_btc"] = output["taker_sell_base_btc"].clip(lower=0.0)
    output = output.loc[:, OUTPUT_COLUMNS]
    complete_numeric = output.loc[output["source_complete"], OUTPUT_COLUMNS[2:-1]]
    if not np.isfinite(complete_numeric.to_numpy(float)).all():
        raise ValueError("complete regional-flow rows contain non-finite values")
    return output


def _expected_month_dates(month: date) -> pd.DatetimeIndex:
    start = pd.Timestamp(month)
    end = start + pd.offsets.MonthBegin(1)
    return pd.date_range(start, end, freq="1D", inclusive="left")


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
    raw = read_archive(payload)
    panel = source_panel(raw, symbol=symbol)
    observed = pd.DatetimeIndex(panel["date"])
    month_dates = _expected_month_dates(month)
    if not observed.equals(month_dates):
        missing = month_dates.difference(observed)
        extra = observed.difference(month_dates)
        raise ValueError(
            f"{symbol} {month:%Y-%m} does not match the exact UTC daily grid; "
            f"missing={missing.tolist()}, extra={extra.tolist()}"
        )
    if not panel["source_complete"].all():
        bad = panel.loc[~panel["source_complete"], "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{symbol} {month:%Y-%m} has incomplete source days: {bad}")
    metadata = {
        "symbol": symbol,
        "month": f"{month:%Y-%m}",
        "archive_url": archive_url(symbol, month),
        "checksum_url": checksum_url(symbol, month),
        "archive_sha256": archive_hash,
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
    if not symbols or any(not symbol for symbol in symbols):
        raise ValueError("at least one non-empty symbol is required")
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
    tasks = [(symbol, month) for symbol in symbols for month in months]
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
    expected_dates = pd.date_range(start, end, freq="1D", inclusive="left")
    expected_index = pd.MultiIndex.from_product(
        [expected_dates, sorted(symbols)], names=["date", "symbol"]
    )
    observed_index = pd.MultiIndex.from_frame(combined[["date", "symbol"]])
    if observed_index.has_duplicates or not observed_index.equals(expected_index):
        missing = expected_index.difference(observed_index)
        extra = observed_index.difference(expected_index)
        raise ValueError(
            "combined regional-flow panel does not match the full date-symbol grid; "
            f"missing={missing.tolist()[:10]}, extra={extra.tolist()[:10]}"
        )
    if not combined["source_complete"].all():
        raise ValueError("combined regional-flow panel contains incomplete source rows")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_day = end - timedelta(days=1)
    combined_path = output_dir / (
        "BTC_regional_fiat_flow_1d_"
        f"{start.isoformat()}_{last_day.isoformat()}.csv.gz"
    )
    _write_gzip_csv(combined, combined_path)
    archives.sort(key=lambda item: (item["symbol"], item["month"]))
    config_record = asdict(cfg)
    config_record["symbols"] = list(symbols)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config": config_record,
        "protocol": {
            "source": "official Binance Spot monthly daily-kline archives",
            "archive_root": BASE_URL,
            "archive_checksums_verified": True,
            "end_is_exclusive": True,
            "time_unit": "milliseconds (panel ends before Binance's 2025 spot microsecond transition)",
            "daily_bucket": "UTC open_time; exact 24-hour source span",
            "source_complete": "positive BTC base volume and trade count with bounded taker-buy base volume",
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
        "first_date": combined["date"].min().isoformat(),
        "last_date": combined["date"].max().isoformat(),
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
    cfg = BuildConfig(
        symbols=tuple(args.symbols),
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        workers=args.workers,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
    )
    manifest = build(cfg)
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
