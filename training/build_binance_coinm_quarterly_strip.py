"""Build a checksum-verified COIN-M BTC front/next quarterly strip.

The builder uses contract-specific Binance Vision daily 5-minute klines.  A
fixed delivery calendar chooses the nearest two unexpired contracts at every
timestamp; missing front data is never replaced by a later maturity.  The
result contains only completed pre-2024 state and no strategy return.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from training.build_binance_aggtrade_microstructure import (
    _fetch_bytes,
    _write_gzip_csv,
    expected_sha256,
    verify_sha256,
)


ARCHIVE_ROOT = "https://data.binance.vision/data/futures/cm/daily/klines"
MONTHLY_ARCHIVE_ROOT = "https://data.binance.vision/data/futures/cm/monthly/klines"
S3_ROOT = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
SEALED_END_EXCLUSIVE = pd.Timestamp("2024-01-01")
ARCHIVE_COLUMNS = (
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
RAW_COLUMNS = tuple(
    {
        "quote_volume": "base_asset_volume",
        "taker_buy_quote_volume": "taker_buy_base_asset_volume",
    }.get(column, column)
    for column in ARCHIVE_COLUMNS
)
CONTRACTS = (
    "BTCUSD_200925",
    "BTCUSD_201225",
    "BTCUSD_210326",
    "BTCUSD_210625",
    "BTCUSD_210924",
    "BTCUSD_211231",
    "BTCUSD_220325",
    "BTCUSD_220624",
    "BTCUSD_220930",
    "BTCUSD_221230",
    "BTCUSD_230331",
    "BTCUSD_230630",
    "BTCUSD_230929",
    "BTCUSD_231229",
    "BTCUSD_240329",
    "BTCUSD_240628",
)
OUTPUT_NUMERIC_COLUMNS = tuple(
    f"{leg}_{column}"
    for leg in ("front", "next")
    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "base_asset_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_base_asset_volume",
    )
)


@dataclass(frozen=True)
class BuildConfig:
    start: str = "2020-07-01"
    end: str = "2023-12-31 23:55"
    output_dir: str = "data/binance_coinm_quarterly_strip_pre2024_v2"
    workers: int = 16
    retries: int = 5
    timeout_seconds: int = 60
    overwrite: bool = False
    open_oos: bool = False
    monthly_fallback: bool = True


def contract_delivery(symbol: str) -> pd.Timestamp:
    match = re.fullmatch(r"BTCUSD_(\d{6})", symbol)
    if match is None:
        raise ValueError(f"unexpected quarterly contract symbol: {symbol}")
    day = pd.to_datetime(match.group(1), format="%y%m%d", errors="raise")
    return pd.Timestamp(day) + pd.Timedelta(hours=8)


def listing_url(symbol: str) -> str:
    prefix = f"data/futures/cm/daily/klines/{symbol}/5m/"
    return f"{S3_ROOT}?{urlencode({'prefix': prefix})}"


def archive_url(symbol: str, day: str) -> str:
    stem = f"{symbol}-5m-{day}.zip"
    return f"{ARCHIVE_ROOT}/{symbol}/5m/{stem}"


def monthly_archive_url(symbol: str, month: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError(f"unexpected archive month: {month}")
    stem = f"{symbol}-5m-{month}.zip"
    return f"{MONTHLY_ARCHIVE_ROOT}/{symbol}/5m/{stem}"


def parse_listing(payload: bytes, symbol: str) -> list[str]:
    root = ElementTree.fromstring(payload)
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    if root.findtext("s3:IsTruncated", default="false", namespaces=namespace) == "true":
        raise ValueError(f"archive listing is unexpectedly truncated: {symbol}")
    pattern = re.compile(
        rf"^data/futures/cm/daily/klines/{re.escape(symbol)}/5m/"
        rf"{re.escape(symbol)}-5m-(\d{{4}}-\d{{2}}-\d{{2}})\.zip$"
    )
    days: list[str] = []
    for node in root.findall("s3:Contents", namespace):
        key = node.findtext("s3:Key", namespaces=namespace)
        if key is None:
            continue
        match = pattern.fullmatch(key)
        if match:
            days.append(match.group(1))
    if len(days) != len(set(days)):
        raise ValueError(f"duplicate archive day in listing: {symbol}")
    return sorted(days)


def _day_overlaps(day: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    day_start = pd.Timestamp(day)
    return day_start < end and day_start + pd.Timedelta(days=1) > start


def _read_csv_member(payload: bytes, symbol: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one CSV for {symbol}, found {members}")
        if not Path(members[0]).name.startswith(f"{symbol}-5m-"):
            raise ValueError(f"archive member does not match {symbol}: {members[0]}")
        with archive.open(members[0]) as handle:
            first_line = handle.readline().decode("utf-8").strip().lower()
        has_header = first_line.startswith("open_time,")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(
                handle,
                header=0 if has_header else None,
                names=None if has_header else list(ARCHIVE_COLUMNS),
                low_memory=False,
            )
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if tuple(frame.columns) != ARCHIVE_COLUMNS:
        raise ValueError(f"unexpected kline columns for {symbol}: {frame.columns.tolist()}")
    return frame.rename(
        columns={
            "quote_volume": "base_asset_volume",
            "taker_buy_quote_volume": "taker_buy_base_asset_volume",
        }
    )


def read_archive(
    payload: bytes,
    *,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    frame = _read_csv_member(payload, symbol)
    integer_columns = ("open_time", "close_time", "count")
    numeric_columns = tuple(column for column in RAW_COLUMNS if column != "ignore")
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in integer_columns:
        values = frame[column].to_numpy(float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"non-integral {column} in {symbol}")
        frame[column] = frame[column].astype("int64")
    frame["date"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    frame["feature_available_time_utc"] = pd.to_datetime(
        frame["close_time"] + 1, unit="ms", utc=True
    ).dt.tz_convert(None)
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError(f"duplicate or unordered klines in {symbol}")
    delivery = contract_delivery(symbol)
    frame = frame.loc[
        (frame["date"] >= start)
        & (frame["date"] < end)
        & (frame["feature_available_time_utc"] < delivery)
    ].copy()
    expected_close = frame["open_time"] + 299_999
    timing_valid = frame["close_time"].eq(expected_close)
    price = frame[["open", "high", "low", "close"]].to_numpy(float)
    finite = np.isfinite(price).all(axis=1)
    positive = (price > 0.0).all(axis=1)
    envelope = (
        frame["high"].ge(frame[["open", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "close"]].min(axis=1))
        & frame["high"].ge(frame["low"])
    )
    flow_columns = [
        "volume",
        "base_asset_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_base_asset_volume",
    ]
    flow_values = frame[flow_columns].to_numpy(float)
    nonnegative = np.isfinite(flow_values).all(axis=1) & frame[flow_columns].ge(0.0).all(axis=1)
    taker_bounded = (
        frame["taker_buy_volume"].le(frame["volume"] + 1e-8)
        & frame["taker_buy_base_asset_volume"].le(
            frame["base_asset_volume"] + 1e-8
        )
    )
    frame["row_valid"] = (
        timing_valid & finite & positive & envelope & nonnegative & taker_bounded
    )
    frame["symbol"] = symbol
    return frame[
        [
            "date",
            "feature_available_time_utc",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "base_asset_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_base_asset_volume",
            "row_valid",
        ]
    ].reset_index(drop=True)


def _cache_path(cache_dir: Path, url: str) -> Path:
    name = url.rsplit("/", 1)[-1]
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    return cache_dir / f"{key}_{name}"


def fetch_cached(
    url: str,
    *,
    cache_dir: Path,
    retries: int,
    timeout: int,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> bytes:
    path = _cache_path(cache_dir, url)
    if path.exists():
        return path.read_bytes()
    payload = fetcher(url, retries=retries, timeout=timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return payload


def build_fixed_strip(
    raw: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    contracts: tuple[str, ...] = CONTRACTS,
) -> pd.DataFrame:
    deliveries = np.asarray([contract_delivery(symbol).value for symbol in contracts], dtype=np.int64)
    if not np.all(deliveries[1:] > deliveries[:-1]):
        raise ValueError("quarterly contracts are not ordered by delivery")
    grid = pd.date_range(start, end, freq="5min", inclusive="left")
    available = grid + pd.Timedelta("5min")
    contract_index = np.searchsorted(deliveries, available.asi8, side="right")
    has_two = contract_index + 1 < len(contracts)
    symbols = np.asarray(contracts, dtype=object)
    front_index = np.minimum(contract_index, len(contracts) - 1)
    next_index = np.minimum(contract_index + 1, len(contracts) - 1)
    front_symbols = np.where(has_two, symbols[front_index], None)
    next_symbols = np.where(has_two, symbols[next_index], None)
    front_delivery_values = np.where(has_two, deliveries[front_index], pd.NaT.value)
    next_delivery_values = np.where(has_two, deliveries[next_index], pd.NaT.value)
    panel = pd.DataFrame(
        {
            "signal_bar_open_utc": grid,
            "feature_available_time_utc": available,
            "trade_earliest_time_utc": available,
            "front_symbol": front_symbols,
            "next_symbol": next_symbols,
            "front_delivery_utc": pd.to_datetime(front_delivery_values),
            "next_delivery_utc": pd.to_datetime(next_delivery_values),
        }
    )
    if raw.duplicated(["date", "symbol"]).any():
        raise ValueError("contract source contains duplicate date/symbol rows")
    value_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "base_asset_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_base_asset_volume",
        "row_valid",
    ]
    for leg in ("front", "next"):
        leg_frame = raw[["date", "symbol", *value_columns]].rename(
            columns={
                "date": "signal_bar_open_utc",
                "symbol": f"{leg}_symbol",
                **{column: f"{leg}_{column}" for column in value_columns},
            }
        )
        panel = panel.merge(
            leg_frame,
            on=["signal_bar_open_utc", f"{leg}_symbol"],
            how="left",
            sort=False,
            validate="one_to_one",
        )
    front_present = panel["front_row_valid"].notna()
    next_present = panel["next_row_valid"].notna()
    front_valid = panel["front_row_valid"].eq(True)
    next_valid = panel["next_row_valid"].eq(True)
    panel["feature_valid"] = (
        has_two
        & front_present
        & next_present
        & front_valid
        & next_valid
    )
    panel["feature_invalid_reason"] = np.select(
        [
            ~has_two,
            ~front_present,
            ~next_present,
            ~front_valid,
            ~next_valid,
        ],
        [
            "calendar_missing_two_contracts",
            "front_row_missing",
            "next_row_missing",
            "front_row_invalid",
            "next_row_invalid",
        ],
        default="ok",
    )
    panel["front_hours_to_delivery"] = (
        panel["front_delivery_utc"] - panel["feature_available_time_utc"]
    ).dt.total_seconds() / 3600.0
    panel["next_hours_to_delivery"] = (
        panel["next_delivery_utc"] - panel["feature_available_time_utc"]
    ).dt.total_seconds() / 3600.0
    panel.loc[~panel["feature_valid"], OUTPUT_NUMERIC_COLUMNS] = np.nan
    panel = panel.drop(columns=["front_row_valid", "next_row_valid"])
    if not panel["signal_bar_open_utc"].equals(
        pd.Series(grid, name="signal_bar_open_utc")
    ):
        raise ValueError("fixed strip is not a complete five-minute grid")
    return panel


def _download_archive(
    symbol: str,
    day: str,
    cfg: BuildConfig,
    cache_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    url = archive_url(symbol, day)
    payload = fetch_cached(
        url,
        cache_dir=cache_dir,
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    checksum_payload = fetch_cached(
        url + ".CHECKSUM",
        cache_dir=cache_dir,
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    expected = expected_sha256(checksum_payload)
    observed = verify_sha256(payload, expected)
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    frame = read_archive(payload, symbol=symbol, start=start, end=end)
    return frame, {
        "symbol": symbol,
        "day": day,
        "url": url,
        "archive_sha256": observed,
        "rows_in_sealed_interval": int(len(frame)),
        "valid_rows": int(frame["row_valid"].sum()),
    }


def _download_monthly_archive(
    symbol: str,
    month: str,
    cfg: BuildConfig,
    cache_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    url = monthly_archive_url(symbol, month)
    payload = fetch_cached(
        url,
        cache_dir=cache_dir,
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    checksum_payload = fetch_cached(
        url + ".CHECKSUM",
        cache_dir=cache_dir,
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    expected = expected_sha256(checksum_payload)
    observed = verify_sha256(payload, expected)
    frame = read_archive(
        payload,
        symbol=symbol,
        start=pd.Timestamp(cfg.start),
        end=pd.Timestamp(cfg.end),
    )
    return frame, {
        "symbol": symbol,
        "month": month,
        "url": url,
        "archive_sha256": observed,
        "rows_in_sealed_interval": int(len(frame)),
        "valid_rows": int(frame["row_valid"].sum()),
    }


def missing_contract_months(
    panel: pd.DataFrame,
    raw: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Return every calendar-required symbol/month absent from daily rows."""
    available = pd.MultiIndex.from_frame(raw[["date", "symbol"]])
    missing: set[tuple[str, str]] = set()
    for leg in ("front", "next"):
        required = panel[["signal_bar_open_utc", f"{leg}_symbol"]].dropna().copy()
        required.columns = ["date", "symbol"]
        absent = ~pd.MultiIndex.from_frame(required).isin(available)
        for row in required.loc[absent].itertuples(index=False):
            missing.add((str(row.symbol), pd.Timestamp(row.date).strftime("%Y-%m")))
    return sorted(missing)


def merge_daily_with_monthly_fallback(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, int, dict[str, Any]]:
    """Keep daily overlap, record official revisions, and add absent keys only."""
    keys = ["date", "symbol"]
    if daily.duplicated(keys).any() or monthly.duplicated(keys).any():
        raise ValueError("duplicate contract key before monthly fallback merge")
    daily_indexed = daily.set_index(keys).sort_index()
    monthly_indexed = monthly.set_index(keys).sort_index()
    overlap = daily_indexed.index.intersection(monthly_indexed.index)
    conflict_payload: list[dict[str, Any]] = []
    if len(overlap):
        daily_overlap = daily_indexed.loc[overlap]
        monthly_overlap = monthly_indexed.loc[overlap]
        mismatch = daily_overlap.astype(str).ne(monthly_overlap.astype(str))
        for key in mismatch.index[mismatch.any(axis=1)]:
            columns = mismatch.columns[mismatch.loc[key]].tolist()
            conflict_payload.append(
                {
                    "date": str(key[0]),
                    "symbol": str(key[1]),
                    "columns": [str(column) for column in columns],
                    "daily": {
                        str(column): str(daily_overlap.loc[key, column])
                        for column in columns
                    },
                    "monthly": {
                        str(column): str(monthly_overlap.loc[key, column])
                        for column in columns
                    },
                }
            )
    conflict_bytes = json.dumps(
        conflict_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    overlap_diagnostics = {
        "rows": int(len(overlap)),
        "conflict_rows": int(len(conflict_payload)),
        "conflict_fraction": (
            float(len(conflict_payload) / len(overlap)) if len(overlap) else 0.0
        ),
        "conflict_sha256": hashlib.sha256(conflict_bytes).hexdigest(),
        "resolution": "daily primary retained; monthly overlap never used",
    }
    additions = monthly_indexed.loc[~monthly_indexed.index.isin(daily_indexed.index)]
    combined = pd.concat([daily_indexed, additions]).reset_index()
    combined = combined.sort_values(keys).reset_index(drop=True)
    return combined, int(len(additions)), overlap_diagnostics


def build(cfg: BuildConfig) -> dict[str, Any]:
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if end >= SEALED_END_EXCLUSIVE and not cfg.open_oos:
        raise ValueError("2024+ strip is sealed; pass --open-oos only after candidate freeze")
    five_minutes_ns = pd.Timedelta("5min").value
    if start.value % five_minutes_ns or end.value % five_minutes_ns:
        raise ValueError("start/end must align to five-minute boundaries")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first_label = start.strftime("%Y%m%dT%H%M")
    last_label = (end - pd.Timedelta("5min")).strftime("%Y%m%dT%H%M")
    output_path = output_dir / (
        f"BTCUSD_front_next_quarterly_5m_{first_label}_{last_label}.csv.gz"
    )
    manifest_path = output_dir / "build_manifest.json"
    if (output_path.exists() or manifest_path.exists()) and not cfg.overwrite:
        raise FileExistsError("quarterly strip output already exists; use --overwrite explicitly")
    cache_dir = output_dir / "archive_cache"
    listings: list[dict[str, Any]] = []
    archives: list[tuple[str, str]] = []
    for symbol in CONTRACTS:
        url = listing_url(symbol)
        payload = fetch_cached(
            url,
            cache_dir=cache_dir,
            retries=cfg.retries,
            timeout=cfg.timeout_seconds,
        )
        days = [
            day
            for day in parse_listing(payload, symbol)
            if _day_overlaps(day, start, end)
        ]
        listings.append(
            {
                "symbol": symbol,
                "url": url,
                "listing_sha256": hashlib.sha256(payload).hexdigest(),
                "days": days,
            }
        )
        archives.extend((symbol, day) for day in days)
    frames: list[pd.DataFrame] = []
    archive_metadata: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {
            executor.submit(_download_archive, symbol, day, cfg, cache_dir): (symbol, day)
            for symbol, day in archives
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            frame, metadata = future.result()
            frames.append(frame)
            archive_metadata.append(metadata)
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"verified archives: {completed}/{len(futures)}",
                    flush=True,
                )
    daily_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if daily_raw.empty:
        raise ValueError("no quarterly contract rows were downloaded")
    daily_raw = daily_raw.sort_values(["date", "symbol"]).reset_index(drop=True)
    panel = build_fixed_strip(daily_raw, start=start, end=end)
    monthly_requests = (
        missing_contract_months(panel, daily_raw) if cfg.monthly_fallback else []
    )
    monthly_frames: list[pd.DataFrame] = []
    monthly_metadata: list[dict[str, Any]] = []
    if monthly_requests:
        with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {
                executor.submit(
                    _download_monthly_archive, symbol, month, cfg, cache_dir
                ): (symbol, month)
                for symbol, month in monthly_requests
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                frame, metadata = future.result()
                monthly_frames.append(frame)
                monthly_metadata.append(metadata)
                print(
                    f"verified monthly fallbacks: {completed}/{len(futures)}",
                    flush=True,
                )
    monthly_rows_added = 0
    monthly_overlap = {
        "rows": 0,
        "conflict_rows": 0,
        "conflict_fraction": 0.0,
        "conflict_sha256": hashlib.sha256(b"[]").hexdigest(),
        "resolution": "daily primary retained; monthly overlap never used",
    }
    raw = daily_raw
    if monthly_frames:
        monthly_raw = pd.concat(monthly_frames, ignore_index=True)
        raw, monthly_rows_added, monthly_overlap = merge_daily_with_monthly_fallback(
            daily_raw, monthly_raw
        )
        panel = build_fixed_strip(raw, start=start, end=end)
    _write_gzip_csv(panel, output_path)
    archive_metadata.sort(key=lambda item: (item["symbol"], item["day"]))
    monthly_metadata.sort(key=lambda item: (item["symbol"], item["month"]))
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "source": (
                "official checksum-verified Binance Vision COIN-M contract-specific "
                "daily klines with monthly missing-key fallback"
            ),
            "archive_root": ARCHIVE_ROOT,
            "monthly_archive_root": MONTHLY_ARCHIVE_ROOT,
            "source_precedence": (
                "daily overlap always wins; monthly official revisions are hashed; "
                "only absent keys are added"
            ),
            "selection": "fixed nearest two unexpired delivery symbols; missing front never promoted",
            "availability": "kline close_time + 1ms, equal to next five-minute boundary",
            "raw_archives_cached": True,
            "outcomes_opened": False,
            "post2023_opened": bool(cfg.open_oos and end >= SEALED_END_EXCLUSIVE),
        },
        "implementation": {
            "path": "training/build_binance_coinm_quarterly_strip.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "contracts": [
            {"symbol": symbol, "delivery_utc": str(contract_delivery(symbol))}
            for symbol in CONTRACTS
        ],
        "listings": listings,
        "archives": archive_metadata,
        "monthly_fallback_requests": [
            {"symbol": symbol, "month": month}
            for symbol, month in monthly_requests
        ],
        "monthly_fallback_archives": monthly_metadata,
        "monthly_rows_added": monthly_rows_added,
        "monthly_overlap_diagnostics": monthly_overlap,
        "output": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "rows": int(len(panel)),
        "valid_rows": int(panel["feature_valid"].sum()),
        "first_signal_bar": str(panel["signal_bar_open_utc"].iloc[0]),
        "last_signal_bar": str(panel["signal_bar_open_utc"].iloc[-1]),
        "invalid_reasons": {
            str(key): int(value)
            for key, value in panel["feature_invalid_reason"].value_counts().items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
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
    parser.add_argument("--no-monthly-fallback", action="store_true")
    args = vars(parser.parse_args())
    args["monthly_fallback"] = not args.pop("no_monthly_fallback")
    report = build(BuildConfig(**args))
    print(
        json.dumps(
            {
                "output": report["output"],
                "output_sha256": report["output_sha256"],
                "rows": report["rows"],
                "valid_rows": report["valid_rows"],
                "invalid_reasons": report["invalid_reasons"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
