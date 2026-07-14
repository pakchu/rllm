"""Build a checksummed Binance USD-M 5m kline reference for data audits."""
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
from typing import Any, Callable

import numpy as np
import pandas as pd

from training.build_binance_aggtrade_microstructure import (
    _fetch_bytes,
    _month_starts,
    _write_gzip_csv,
    expected_sha256,
    verify_sha256,
)


BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
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
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
)
SCHEMA_VERSION = 1
PANDAS_FREQUENCY = "5min"


@dataclass(frozen=True)
class BuildConfig:
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    start: str = "2020-01-01"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_um_kline_reference_btc_2020_2023"
    workers: int = 4
    retries: int = 5
    timeout_seconds: int = 60
    overwrite: bool = False


def archive_url(symbol: str, interval: str, day: date) -> str:
    stem = f"{symbol}-{interval}-{day:%Y-%m-%d}.zip"
    return f"{BASE_URL}/{symbol}/{interval}/{stem}"


def checksum_url(symbol: str, interval: str, day: date) -> str:
    return archive_url(symbol, interval, day) + ".CHECKSUM"


def read_archive(payload: bytes) -> pd.DataFrame:
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
        frame.columns = [str(column).strip().lower() for column in frame.columns]
    if tuple(frame.columns) != RAW_COLUMNS:
        raise ValueError(f"unexpected kline columns: {frame.columns.tolist()}")

    for column in RAW_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.empty:
        raise ValueError("kline archive is empty")
    if not frame["open_time"].is_unique or not frame["open_time"].is_monotonic_increasing:
        raise ValueError("kline open times are not strictly increasing")
    if not np.isfinite(frame[list(RAW_COLUMNS)].to_numpy(float)).all():
        raise ValueError("kline archive contains non-finite values")

    frame["date"] = pd.to_datetime(frame["open_time"].astype("int64"), unit="ms")
    output = frame.rename(
        columns={
            "quote_volume": "quote_asset_volume",
            "count": "number_of_trades",
            "taker_buy_volume": "taker_buy_base",
            "taker_buy_quote_volume": "taker_buy_quote",
        }
    ).loc[:, OUTPUT_COLUMNS]
    prices = output[["open", "high", "low", "close"]]
    if (prices <= 0.0).any().any():
        raise ValueError("kline archive contains non-positive prices")
    if (output[["volume", "quote_asset_volume", "number_of_trades"]] < 0.0).any().any():
        raise ValueError("kline archive contains negative activity")
    if not (
        output["high"].ge(prices[["open", "close"]].max(axis=1)).all()
        and output["low"].le(prices[["open", "close"]].min(axis=1)).all()
    ):
        raise ValueError("kline OHLC envelope is invalid")
    return output


def _month_end(month: date) -> date:
    return date(
        month.year + (month.month == 12),
        1 if month.month == 12 else month.month + 1,
        1,
    )


def _month_days(month: date, start: date, end: date) -> list[date]:
    current = max(start, month)
    limit = min(end, _month_end(month))
    days: list[date] = []
    while current < limit:
        days.append(current)
        current += timedelta(days=1)
    return days


def _process_month(
    month: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    expected_days = _month_days(month, start, end)
    output_dir = Path(cfg.output_dir) / "monthly"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.symbol}_{cfg.interval}_{month:%Y-%m}"
    output_path = output_dir / f"{stem}.csv.gz"
    metadata_path = output_dir / f"{stem}.json"

    if output_path.exists() and metadata_path.exists() and not cfg.overwrite:
        metadata = json.loads(metadata_path.read_text())
        if _resume_is_current(
            metadata,
            cfg=cfg,
            month=month,
            expected_days=expected_days,
            output_path=output_path,
            fetcher=fetcher,
        ):
            return metadata

    frames: list[pd.DataFrame] = []
    archives: list[dict[str, Any]] = []
    for day in expected_days:
        current_checksum = expected_sha256(
            fetcher(
                checksum_url(cfg.symbol, cfg.interval, day),
                retries=cfg.retries,
                timeout=cfg.timeout_seconds,
            )
        )
        payload = fetcher(
            archive_url(cfg.symbol, cfg.interval, day),
            retries=cfg.retries,
            timeout=cfg.timeout_seconds,
        )
        archive_hash = verify_sha256(payload, current_checksum)
        frame = read_archive(payload)
        day_end = day + timedelta(days=1)
        expected = pd.date_range(
            day,
            day_end,
            inclusive="left",
            freq=PANDAS_FREQUENCY,
        )
        if not frame["date"].equals(pd.Series(expected, name="date")):
            raise ValueError(f"kline day {day} has incomplete timestamp coverage")
        frames.append(frame)
        archives.append(
            {
                "date": day.isoformat(),
                "archive_sha256": archive_hash,
                "rows": int(len(frame)),
            }
        )

    frame = pd.concat(frames, ignore_index=True)

    _write_gzip_csv(frame, output_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "month": f"{month:%Y-%m}",
        "symbol": cfg.symbol,
        "interval": cfg.interval,
        "requested_dates": [day.isoformat() for day in expected_days],
        "output": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "rows": int(len(frame)),
        "first_date": str(frame["date"].min()),
        "last_date": str(frame["date"].max()),
        "archives": archives,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def _resume_is_current(
    metadata: dict[str, Any],
    *,
    cfg: BuildConfig,
    month: date,
    expected_days: list[date],
    output_path: Path,
    fetcher: Callable[..., bytes],
) -> bool:
    expected_dates = [day.isoformat() for day in expected_days]
    archives = metadata.get("archives")
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("month") != f"{month:%Y-%m}"
        or metadata.get("symbol") != cfg.symbol
        or metadata.get("interval") != cfg.interval
        or metadata.get("requested_dates") != expected_dates
        or not isinstance(archives, list)
        or [archive.get("date") for archive in archives] != expected_dates
    ):
        return False
    if hashlib.sha256(output_path.read_bytes()).hexdigest() != metadata.get("output_sha256"):
        raise ValueError(f"kline resume artifact hash mismatch: {output_path}")
    for day, archive in zip(expected_days, archives, strict=True):
        current_checksum = expected_sha256(
            fetcher(
                checksum_url(cfg.symbol, cfg.interval, day),
                retries=cfg.retries,
                timeout=cfg.timeout_seconds,
            )
        )
        if current_checksum != archive.get("archive_sha256"):
            return False
    return True


def build(cfg: BuildConfig) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if cfg.interval != "5m":
        raise ValueError("this reference builder is frozen to 5m")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")

    months = _month_starts(start, end)
    metadata: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {executor.submit(_process_month, month, cfg): month for month in months}
        for future in as_completed(futures):
            result = future.result()
            metadata.append(result)
            print(f"completed {result['month']}: rows={result['rows']}", flush=True)
    metadata.sort(key=lambda item: item["month"])
    frames = [pd.read_csv(item["output"], compression="gzip", parse_dates=["date"]) for item in metadata]
    combined = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    expected = pd.date_range(start, end, inclusive="left", freq=PANDAS_FREQUENCY)
    if not combined["date"].equals(pd.Series(expected, name="date")):
        raise ValueError("combined kline reference has incomplete timestamp coverage")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / (
        f"{cfg.symbol}_{cfg.interval}_{cfg.start}_{end - timedelta(days=1)}.csv.gz"
    )
    _write_gzip_csv(combined, combined_path)
    manifest = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "source": "official Binance USD-M daily kline archives",
            "archive_checksums_verified": True,
            "end_is_exclusive": True,
            "outcomes_opened": False,
        },
        "combined_output": str(combined_path),
        "combined_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        "rows": int(len(combined)),
        "first_date": str(combined["date"].min()),
        "last_date": str(combined["date"].max()),
        "columns": list(combined.columns),
        "months": metadata,
    }
    (output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=BuildConfig.symbol)
    parser.add_argument("--interval", default=BuildConfig.interval)
    parser.add_argument("--start", default=BuildConfig.start)
    parser.add_argument("--end", default=BuildConfig.end)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--workers", type=int, default=BuildConfig.workers)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument("--timeout-seconds", type=int, default=BuildConfig.timeout_seconds)
    parser.add_argument("--overwrite", action="store_true")
    manifest = build(BuildConfig(**vars(parser.parse_args())))
    print(
        json.dumps(
            {
                "combined_output": manifest["combined_output"],
                "rows": manifest["rows"],
                "first_date": manifest["first_date"],
                "last_date": manifest["last_date"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
