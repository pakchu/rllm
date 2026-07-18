"""Build checksum-verified hourly BTCBVOLUSDT candles from Binance archives."""
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


BASE_URL = "https://data.binance.vision/data/option/daily/BVOLIndex"
SYMBOL = "BTCBVOLUSDT"
SCHEMA_VERSION = 1
SEALED_END_EXCLUSIVE = date(2024, 1, 1)
RAW_COLUMNS = ("calc_time", "symbol", "base_asset", "quote_asset", "index_value")
OUTPUT_COLUMNS = (
    "date",
    "feature_available_time_utc",
    "trade_earliest_time_utc",
    "open",
    "high",
    "low",
    "close",
    "source_rows",
    "source_complete",
    "feature_valid",
    "feature_invalid_reason",
)


@dataclass(frozen=True)
class BuildConfig:
    start: str = "2023-06-20"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_btc_bvol_hourly"
    workers: int = 4
    retries: int = 5
    timeout_seconds: int = 60
    overwrite: bool = False
    open_oos: bool = False


def archive_url(day: date) -> str:
    stem = f"{SYMBOL}-BVOLIndex-{day:%Y-%m-%d}.zip"
    return f"{BASE_URL}/{SYMBOL}/{stem}"


def checksum_url(day: date) -> str:
    return archive_url(day) + ".CHECKSUM"


def read_archive(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one BVOL CSV, found {members}")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle, low_memory=False)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if tuple(frame.columns) != RAW_COLUMNS:
        raise ValueError(f"unexpected BVOL columns: {frame.columns.tolist()}")
    if frame.empty:
        raise ValueError("BVOL archive is empty")
    if not frame["symbol"].eq(SYMBOL).all():
        raise ValueError("BVOL archive contains an unexpected symbol")
    if not frame["base_asset"].eq("BTCBVOL").all() or not frame["quote_asset"].eq("USDT").all():
        raise ValueError("BVOL archive contains unexpected assets")
    frame["calc_time"] = pd.to_numeric(frame["calc_time"], errors="raise").astype("int64")
    frame["index_value"] = pd.to_numeric(frame["index_value"], errors="raise")
    if not frame["calc_time"].is_unique or not frame["calc_time"].is_monotonic_increasing:
        raise ValueError("BVOL calculation times are duplicate or unordered")
    if not np.isfinite(frame["index_value"].to_numpy(float)).all() or (frame["index_value"] <= 0.0).any():
        raise ValueError("BVOL values must be finite and positive")
    microseconds = frame["calc_time"].abs().gt(100_000_000_000_000)
    if microseconds.any() and not microseconds.all():
        raise ValueError("BVOL timestamps mix milliseconds and microseconds")
    timestamp_ms = frame["calc_time"].floordiv(1_000) if microseconds.all() else frame["calc_time"]
    # Binance publishes one observation per UTC second, but calc_time can carry
    # a small sub-second calculation jitter.  Floor the label; never average or
    # forward-fill values.  More than one observation in a second fails closed.
    timestamp_ms = timestamp_ms.floordiv(1_000).mul(1_000)
    if timestamp_ms.duplicated().any() or not timestamp_ms.is_monotonic_increasing:
        raise ValueError("BVOL contains duplicate or unordered UTC seconds")
    frame["date"] = pd.to_datetime(timestamp_ms, unit="ms", utc=True).dt.tz_convert(None)
    return frame[["date", "index_value"]]


def aggregate_day(frame: pd.DataFrame, day: date) -> pd.DataFrame:
    start = pd.Timestamp(day)
    end = start + pd.Timedelta(days=1)
    if not frame["date"].between(start, end, inclusive="left").all():
        raise ValueError(f"BVOL archive contains rows outside {day}")
    expected = pd.date_range(start, end, inclusive="left", freq="1s")
    indexed = frame.set_index("date")["index_value"].reindex(expected)
    groups = indexed.index.floor("1h")
    grouped = indexed.groupby(groups, sort=True, observed=True)
    output = pd.DataFrame(
        {
            "date": pd.date_range(start, end, inclusive="left", freq="1h"),
            "open": grouped.first().to_numpy(float),
            "high": grouped.max().to_numpy(float),
            "low": grouped.min().to_numpy(float),
            "close": grouped.last().to_numpy(float),
            "source_rows": grouped.count().to_numpy(int),
        }
    )
    output["source_complete"] = output["source_rows"].eq(3_600)
    finite = np.isfinite(output[["open", "high", "low", "close"]].to_numpy(float)).all(axis=1)
    envelope = (
        output["high"].ge(output[["open", "close"]].max(axis=1))
        & output["low"].le(output[["open", "close"]].min(axis=1))
    )
    output["feature_valid"] = output["source_complete"] & finite & envelope
    output["feature_invalid_reason"] = np.select(
        [~output["source_complete"], ~finite, ~envelope],
        ["source_incomplete", "nonfinite_ohlc", "invalid_envelope"],
        default="ok",
    )
    output.loc[~output["feature_valid"], ["open", "high", "low", "close"]] = np.nan
    output["feature_available_time_utc"] = output["date"] + pd.Timedelta(hours=1)
    output["trade_earliest_time_utc"] = output["feature_available_time_utc"]
    return output.loc[:, OUTPUT_COLUMNS]


def _month_end(month: date) -> date:
    return date(month.year + (month.month == 12), 1 if month.month == 12 else month.month + 1, 1)


def _month_days(month: date, start: date, end: date) -> list[date]:
    current = max(month, start)
    limit = min(_month_end(month), end)
    output: list[date] = []
    while current < limit:
        output.append(current)
        current += timedelta(days=1)
    return output


def _process_month(
    month: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    days = _month_days(month, start, end)
    monthly_dir = Path(cfg.output_dir) / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{SYMBOL}_1h_{month:%Y-%m}"
    output_path = monthly_dir / f"{stem}.csv.gz"
    metadata_path = monthly_dir / f"{stem}.json"
    if output_path.exists() and metadata_path.exists() and not cfg.overwrite:
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("schema_version") == SCHEMA_VERSION
            and metadata.get("days") == [str(day) for day in days]
            and metadata.get("output_sha256") == hashlib.sha256(output_path.read_bytes()).hexdigest()
        ):
            return metadata

    frames: list[pd.DataFrame] = []
    archives: list[dict[str, Any]] = []
    for day in days:
        checksum = expected_sha256(
            fetcher(checksum_url(day), retries=cfg.retries, timeout=cfg.timeout_seconds)
        )
        payload = fetcher(archive_url(day), retries=cfg.retries, timeout=cfg.timeout_seconds)
        observed = verify_sha256(payload, checksum)
        raw = read_archive(payload)
        frames.append(aggregate_day(raw, day))
        archives.append(
            {
                "day": str(day),
                "archive_sha256": observed,
                "raw_rows": int(len(raw)),
            }
        )
    output = pd.concat(frames, ignore_index=True)
    expected_hours = pd.date_range(max(month, start), min(_month_end(month), end), inclusive="left", freq="1h")
    if not output["date"].equals(pd.Series(expected_hours, name="date")):
        raise ValueError(f"BVOL month {month:%Y-%m} has an invalid hourly grid")
    _write_gzip_csv(output, output_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "month": f"{month:%Y-%m}",
        "days": [str(day) for day in days],
        "rows": int(len(output)),
        "feature_valid_rows": int(output["feature_valid"].sum()),
        "first_date": str(output["date"].min()),
        "last_date": str(output["date"].max()),
        "output": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "archives": archives,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def build(cfg: BuildConfig) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if end > SEALED_END_EXCLUSIVE and not cfg.open_oos:
        raise ValueError("2024+ BVOL source is sealed; pass --open-oos only after candidate freeze")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    months = _month_starts(date(start.year, start.month, 1), end)
    metadata: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {executor.submit(_process_month, month, cfg): month for month in months}
        for future in as_completed(futures):
            month = futures[future]
            item = future.result()
            metadata.append(item)
            print(
                f"completed {month:%Y-%m}: rows={item['rows']} valid={item['feature_valid_rows']}",
                flush=True,
            )
    metadata.sort(key=lambda item: item["month"])
    frames = [
        pd.read_csv(
            item["output"],
            compression="gzip",
            parse_dates=["date", "feature_available_time_utc", "trade_earliest_time_utc"],
        )
        for item in metadata
    ]
    combined = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    expected = pd.date_range(start, end, inclusive="left", freq="1h")
    if not combined["date"].equals(pd.Series(expected, name="date")):
        raise ValueError("combined BVOL output has an invalid hourly grid")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_day = end - timedelta(days=1)
    combined_path = output_dir / f"{SYMBOL}_1h_{start:%Y-%m-%d}_{last_day:%Y-%m-%d}.csv.gz"
    _write_gzip_csv(combined, combined_path)
    manifest = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "source": "official Binance BTCBVOLUSDT daily one-second BVOLIndex archives",
            "archive_root": BASE_URL,
            "checksums_verified": True,
            "feature_available_time": "hour open plus one hour",
            "raw_archives_persisted": False,
            "outcomes_opened": False,
            "post2023_opened": bool(cfg.open_oos and end > SEALED_END_EXCLUSIVE),
        },
        "combined_output": str(combined_path),
        "combined_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        "rows": int(len(combined)),
        "feature_valid_rows": int(combined["feature_valid"].sum()),
        "first_date": str(combined["date"].min()),
        "last_date": str(combined["date"].max()),
        "months": metadata,
    }
    (output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=BuildConfig.start)
    parser.add_argument("--end", default=BuildConfig.end)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--workers", type=int, default=BuildConfig.workers)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument("--timeout-seconds", type=int, default=BuildConfig.timeout_seconds)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--open-oos", action="store_true")
    manifest = build(BuildConfig(**vars(parser.parse_args())))
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "combined_output",
                    "rows",
                    "feature_valid_rows",
                    "first_date",
                    "last_date",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
