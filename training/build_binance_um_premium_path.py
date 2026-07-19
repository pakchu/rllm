"""Build a checksummed, outcome-free BTCUSDT premium-index path panel.

The builder downloads official Binance Vision USD-M monthly
``premiumIndexKlines`` archives, verifies every published checksum, normalizes
the exchange's millisecond/microsecond timestamp transition, and writes only
premium-index OHLC plus causal availability timestamps.  BTC execution prices,
returns, funding and PnL are never read or persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.build_binance_aggtrade_microstructure import (  # noqa: E402
    _fetch_bytes,
    _month_starts,
    _write_gzip_csv,
    expected_sha256,
    verify_sha256,
)


BASE_URL = "https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines"
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
    "source_close_time",
    "feature_available_time",
    "source_valid",
    "premium_open",
    "premium_high",
    "premium_low",
    "premium_close",
)
SCHEMA_VERSION = 1
MICROSECOND_THRESHOLD = 100_000_000_000_000


@dataclass(frozen=True)
class Config:
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    start: str = "2020-01-01"
    end: str = "2026-07-01"
    output_dir: str = "data/binance_um_premium_path_btc_2020_2026"
    manifest: str = "results/binance_um_premium_path_btc_2020_2026_manifest.json"
    workers: int = 8
    retries: int = 5
    timeout_seconds: int = 60


def _next_month(month: date) -> date:
    return date(
        month.year + (month.month == 12),
        1 if month.month == 12 else month.month + 1,
        1,
    )


def archive_url(symbol: str, interval: str, month: date) -> str:
    stem = f"{symbol}-{interval}-{month:%Y-%m}.zip"
    return f"{BASE_URL}/{symbol}/{interval}/{stem}"


def checksum_url(symbol: str, interval: str, month: date) -> str:
    return archive_url(symbol, interval, month) + ".CHECKSUM"


def _timestamp_ms(values: pd.Series, *, label: str) -> pd.Series:
    integer = cast(pd.Series, pd.to_numeric(values, errors="raise")).astype("int64")
    microseconds = integer.abs().gt(MICROSECOND_THRESHOLD)
    if bool(microseconds.any()) and not bool(microseconds.all()):
        raise ValueError(f"{label} mixes millisecond and microsecond timestamps")
    if bool(microseconds.all()):
        if bool(integer.mod(1_000).ne(0).any()):
            raise ValueError(f"{label} microseconds are not millisecond aligned")
        integer = integer.floordiv(1_000)
    return integer


def read_archive(payload: bytes, *, interval_minutes: int = 1) -> pd.DataFrame:
    """Parse one official monthly premium-index archive."""

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one premium-index CSV, found {members}")
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
        raise ValueError(f"unexpected premium-index columns: {frame.columns.tolist()}")
    if frame.empty:
        raise ValueError("premium-index archive is empty")

    open_ms = _timestamp_ms(cast(pd.Series, frame["open_time"]), label="open_time")
    close_ms = _timestamp_ms(cast(pd.Series, frame["close_time"]), label="close_time")
    expected_close_ms = open_ms + interval_minutes * 60_000 - 1
    if not bool(close_ms.eq(expected_close_ms).all()):
        raise ValueError("premium-index close_time does not match its interval")
    if not bool(open_ms.is_monotonic_increasing) or not bool(open_ms.is_unique):
        raise ValueError("premium-index open times are not strictly increasing")

    prices = frame.loc[:, ["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(prices.to_numpy(float)).all():
        raise ValueError("premium-index OHLC contains non-finite values")
    if not (
        cast(pd.Series, prices["high"]).ge(prices[["open", "close"]].max(axis=1)).all()
        and cast(pd.Series, prices["low"]).le(prices[["open", "close"]].min(axis=1)).all()
    ):
        raise ValueError("premium-index OHLC envelope is invalid")

    opened = pd.to_datetime(open_ms, unit="ms", utc=True).dt.tz_localize(None)
    closed = pd.to_datetime(close_ms, unit="ms", utc=True).dt.tz_localize(None)
    output = pd.DataFrame(
        {
            "date": opened,
            "source_close_time": closed,
            "feature_available_time": opened
            + pd.Timedelta(minutes=interval_minutes, seconds=1),
            "source_valid": True,
            "premium_open": prices["open"].astype(float),
            "premium_high": prices["high"].astype(float),
            "premium_low": prices["low"].astype(float),
            "premium_close": prices["close"].astype(float),
        }
    )
    return output.loc[:, list(OUTPUT_COLUMNS)]


def _month_grid(month: date, cfg: Config) -> pd.DatetimeIndex:
    start = max(pd.Timestamp(cfg.start), pd.Timestamp(month))
    end = min(pd.Timestamp(cfg.end), pd.Timestamp(_next_month(month)))
    return pd.date_range(start, end, freq="1min", inclusive="left")


def _invalid_month(month: date, cfg: Config) -> pd.DataFrame:
    grid = _month_grid(month, cfg)
    output = pd.DataFrame({"date": grid})
    output["source_close_time"] = output["date"] + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1)
    output["feature_available_time"] = output["date"] + pd.Timedelta(minutes=1, seconds=1)
    output["source_valid"] = False
    for column in OUTPUT_COLUMNS[4:]:
        output[column] = np.nan
    return output.loc[:, list(OUTPUT_COLUMNS)]


def process_month(
    month: date,
    cfg: Config,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch, checksum and align one monthly archive to a complete 1m grid."""

    try:
        checksum_payload = fetcher(
            checksum_url(cfg.symbol, cfg.interval, month),
            retries=cfg.retries,
            timeout=cfg.timeout_seconds,
        )
        expected = expected_sha256(checksum_payload)
        payload = fetcher(
            archive_url(cfg.symbol, cfg.interval, month),
            retries=cfg.retries,
            timeout=cfg.timeout_seconds,
        )
    except FileNotFoundError:
        frame = _invalid_month(month, cfg)
        return frame, {
            "month": f"{month:%Y-%m}",
            "available": False,
            "reason": "official archive or checksum not published",
            "rows": int(len(frame)),
            "source_valid_rows": 0,
        }

    archive_hash = verify_sha256(payload, expected)
    parsed = read_archive(payload)
    month_grid = _month_grid(month, cfg)
    start = month_grid[0]
    end = month_grid[-1] + pd.Timedelta(minutes=1)
    parsed = parsed.loc[parsed["date"].ge(start) & parsed["date"].lt(end)].copy()
    if parsed["date"].duplicated().any():
        raise ValueError(f"premium-index archive {month:%Y-%m} has duplicate rows")

    grid = pd.DataFrame({"date": month_grid})
    frame = grid.merge(parsed, on="date", how="left", validate="one_to_one")
    missing = cast(pd.Series, frame["source_valid"]).isna()
    frame.loc[missing, "source_close_time"] = (
        frame.loc[missing, "date"] + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1)
    )
    frame.loc[missing, "feature_available_time"] = (
        frame.loc[missing, "date"] + pd.Timedelta(minutes=1, seconds=1)
    )
    frame["source_valid"] = cast(pd.Series, frame["source_valid"]).eq(True)
    return frame.loc[:, list(OUTPUT_COLUMNS)], {
        "month": f"{month:%Y-%m}",
        "available": True,
        "archive_sha256": archive_hash,
        "rows": int(len(frame)),
        "source_valid_rows": int(frame["source_valid"].sum()),
        "missing_rows": int(missing.sum()),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(cfg: Config) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if cfg.interval != "1m":
        raise ValueError("premium path source is frozen to one-minute bars")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")

    months = _month_starts(start, end)
    results: list[tuple[pd.DataFrame, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {
            executor.submit(process_month, month, cfg): month for month in months
        }
        for future in as_completed(futures):
            frame, metadata = future.result()
            results.append((frame, metadata))
            print(
                f"completed {metadata['month']}: valid={metadata['source_valid_rows']}/{metadata['rows']}",
                flush=True,
            )
    results.sort(key=lambda item: item[1]["month"])
    combined = pd.concat([item[0] for item in results], ignore_index=True)
    expected_grid = pd.Series(
        pd.date_range(cfg.start, cfg.end, freq="1min", inclusive="left"), name="date"
    )
    if not cast(pd.Series, combined["date"]).equals(expected_grid):
        raise ValueError("combined premium-index source grid is incomplete")
    if combined["date"].duplicated().any():
        raise ValueError("combined premium-index source contains duplicate timestamps")
    valid = cast(pd.Series, combined["source_valid"]).astype(bool)
    if not bool(combined.loc[valid, list(OUTPUT_COLUMNS[4:])].notna().all().all()):
        raise ValueError("valid premium-index rows contain missing OHLC")
    if not bool(combined.loc[~valid, list(OUTPUT_COLUMNS[4:])].isna().all().all()):
        raise ValueError("invalid premium-index rows contain OHLC")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "BTCUSDT_premium_path_1m_2020-01-01_2026-06-30.csv.gz"
    _write_gzip_csv(combined, data_path)
    archive_metadata = [item[1] for item in results]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "name": "official Binance USD-M BTCUSDT premium-index path source",
            "source": "Binance Vision monthly premiumIndexKlines",
            "source_only": True,
            "outcomes_opened": False,
            "archive_checksums_verified": True,
            "raw_archives_retained": False,
            "btc_execution_prices_retained": False,
            "returns_or_pnl_retained": False,
            "end_is_exclusive": True,
        },
        "config": asdict(cfg),
        "retained_columns": list(OUTPUT_COLUMNS),
        "archives": archive_metadata,
        "missing_archive_months": [
            item["month"] for item in archive_metadata if not item["available"]
        ],
        "file": {
            "path": str(data_path),
            "sha256": sha256_file(data_path),
            "rows": int(len(combined)),
            "source_valid_rows": int(valid.sum()),
            "first_date": str(combined["date"].min()),
            "last_date": str(combined["date"].max()),
        },
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = Path(cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=Config.start)
    parser.add_argument("--end", default=Config.end)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--manifest", default=Config.manifest)
    parser.add_argument("--workers", type=int, default=Config.workers)
    parser.add_argument("--retries", type=int, default=Config.retries)
    parser.add_argument("--timeout-seconds", type=int, default=Config.timeout_seconds)
    result = build(Config(**vars(parser.parse_args())))
    print(json.dumps(result["file"], indent=2))


if __name__ == "__main__":
    main()
