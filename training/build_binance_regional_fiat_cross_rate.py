"""Build the outcome-blind RFXS regional-fiat daily-close source panel.

The builder downloads official Binance Spot monthly daily-kline archives,
verifies the published checksum filename and digest, validates the complete
UTC daily candle, and retains only the four closes allowed by the frozen RFXS
mechanism.  It never reads USD-M execution prices, funding, or returns.
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
)


BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
INTERVAL = "1d"
SCHEMA_VERSION = 1
MICROSECOND_TRANSITION = pd.Timestamp("2025-01-01", tz="UTC")
FROZEN_SOURCE_START = date(2020, 10, 1)
FROZEN_SOURCE_END_EXCLUSIVE = date(2024, 1, 1)
MECHANISM_COMMIT = "1d5805397ed72c98bc83597544b949b07d425f32"
MECHANISM_PATH = "docs/regional-fiat-cross-rate-stress-mechanism-decision-2026-07-20.md"
MECHANISM_SHA256 = "c3f7bcfd12c4412be0ad8696b2fa339c709fa94f1a5e61a22cf33c45e4d3ae89"
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
SOURCE_COLUMNS = (
    "date",
    "source_available_not_before",
    "symbol",
    "close",
    "source_complete",
)
OUTPUT_COLUMNS = (
    "date",
    "source_available_not_before",
    "BTCUSDT_close",
    "BTCEUR_close",
    "BTCTRY_close",
    "BTCBRL_close",
    "source_complete",
)


@dataclass(frozen=True)
class BuildConfig:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    start: str = "2020-10-01"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_regional_fiat_cross_rate_btc_2020_2023"
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


def expected_sha256(checksum_payload: bytes, *, expected_filename: str) -> str:
    """Parse one Binance checksum record and bind it to the requested ZIP."""
    try:
        text = checksum_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Binance checksum payload is not UTF-8") from exc
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("expected exactly one Binance checksum record")
    fields = lines[0].split()
    if len(fields) != 2 or len(fields[0]) != 64:
        raise ValueError("invalid Binance checksum payload")
    try:
        int(fields[0], 16)
    except ValueError as exc:
        raise ValueError("invalid Binance checksum digest") from exc
    published_filename = fields[1].removeprefix("*")
    if published_filename != expected_filename:
        raise ValueError(
            "Binance checksum filename mismatch: "
            f"expected {expected_filename}, got {published_filename}"
        )
    return fields[0].lower()


def _timestamp_unit(values: pd.Series) -> str:
    raw = values.to_numpy(dtype=np.int64, copy=False)
    if ((raw >= 10**12) & (raw < 10**14)).all():
        return "ms"
    if ((raw >= 10**15) & (raw < 10**17)).all():
        return "us"
    raise ValueError("spot daily-kline timestamps have mixed or unsupported units")


def read_archive(payload: bytes) -> pd.DataFrame:
    """Read one official Spot archive and fail closed on malformed candles."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise ValueError("Binance Spot archive is not a valid ZIP") from exc
    with archive:
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
    if (
        (frame["high"] < frame[["open", "close"]].max(axis=1)).any()
        or (frame["low"] > frame[["open", "close"]].min(axis=1)).any()
        or (frame["high"] < frame["low"]).any()
    ):
        raise ValueError("spot daily-kline archive violates OHLC bounds")
    if (frame["base_volume"] <= 0.0).any() or (frame["trade_count"] <= 0).any():
        raise ValueError("spot daily-kline archive contains an empty or stale candle")
    if (frame["quote_notional"] < 0.0).any():
        raise ValueError("spot daily-kline archive contains negative quote notional")
    tolerance = 1e-8
    if (
        (frame["taker_buy_base"] < -tolerance).any()
        or (frame["taker_buy_quote"] < -tolerance).any()
        or (frame["taker_buy_base"] > frame["base_volume"] + tolerance).any()
        or (frame["taker_buy_quote"] > frame["quote_notional"] + tolerance).any()
    ):
        raise ValueError("spot daily-kline taker-buy fields violate total-volume bounds")

    unit = _timestamp_unit(frame["open_time"])
    if _timestamp_unit(frame["close_time"]) != unit:
        raise ValueError("spot daily-kline open/close timestamp units differ")
    open_times = pd.to_datetime(frame["open_time"], unit=unit, utc=True, errors="raise")
    if not (
        open_times.dt.hour.eq(0)
        & open_times.dt.minute.eq(0)
        & open_times.dt.second.eq(0)
        & open_times.dt.microsecond.eq(0)
        & open_times.dt.nanosecond.eq(0)
    ).all():
        raise ValueError("spot daily-kline rows are not aligned to UTC day opens")
    if unit == "ms" and open_times.ge(MICROSECOND_TRANSITION).any():
        raise ValueError("2025+ Binance Spot archive must use microsecond timestamps")
    if unit == "us" and open_times.lt(MICROSECOND_TRANSITION).any():
        raise ValueError("pre-2025 Binance Spot archive must use millisecond timestamps")
    units_per_day = 86_400_000 if unit == "ms" else 86_400_000_000
    expected_close = frame["open_time"] + units_per_day - 1
    if not frame["close_time"].eq(expected_close).all():
        raise ValueError("spot daily-kline close times do not span exact UTC days")
    expected_step = frame["open_time"].diff().dropna()
    if not expected_step.eq(units_per_day).all():
        raise ValueError("spot daily-kline archive has missing or non-daily rows")
    frame.attrs["timestamp_unit"] = unit
    return frame


def source_row(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """Retain only one completed close per UTC day for the requested symbol."""
    symbol = symbol.strip().upper()
    if symbol not in DEFAULT_SYMBOLS:
        raise ValueError(f"unexpected RFXS symbol: {symbol}")
    unit = str(frame.attrs.get("timestamp_unit") or _timestamp_unit(frame["open_time"]))
    open_times = pd.to_datetime(frame["open_time"], unit=unit, utc=True, errors="raise")
    available = open_times + pd.Timedelta(days=1)
    output = pd.DataFrame(
        {
            "date": open_times.dt.strftime("%Y-%m-%d"),
            "source_available_not_before": available.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": symbol,
            "close": frame["close"].astype(float),
            "source_complete": True,
        }
    )
    if tuple(output.columns) != SOURCE_COLUMNS:
        raise AssertionError("internal RFXS source schema changed")
    return output


def _expected_month_dates(month: date) -> list[str]:
    start = pd.Timestamp(month)
    end = start + pd.offsets.MonthBegin(1)
    return pd.date_range(start, end, freq="1D", inclusive="left").strftime(
        "%Y-%m-%d"
    ).tolist()


def _process_archive(
    symbol: str,
    month: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    zip_filename = f"{symbol}-{INTERVAL}-{month:%Y-%m}.zip"
    checksum_payload = fetcher(
        checksum_url(symbol, month),
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    published_hash = expected_sha256(
        checksum_payload, expected_filename=zip_filename
    )
    payload = fetcher(
        archive_url(symbol, month),
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    local_hash = hashlib.sha256(payload).hexdigest()
    if local_hash != published_hash:
        raise ValueError(
            "archive checksum mismatch: "
            f"expected {published_hash}, got {local_hash}"
        )
    raw = read_archive(payload)
    panel = source_row(raw, symbol=symbol)
    expected_dates = _expected_month_dates(month)
    observed_dates = panel["date"].tolist()
    if observed_dates != expected_dates:
        missing = sorted(set(expected_dates).difference(observed_dates))
        extra = sorted(set(observed_dates).difference(expected_dates))
        raise ValueError(
            f"{symbol} {month:%Y-%m} does not match the exact UTC daily grid; "
            f"missing={missing}, extra={extra}"
        )
    if not panel["source_complete"].all():
        raise ValueError(f"{symbol} {month:%Y-%m} has incomplete source days")
    metadata = {
        "symbol": symbol,
        "month": f"{month:%Y-%m}",
        "archive_url": archive_url(symbol, month),
        "checksum_url": checksum_url(symbol, month),
        "checksum_response_sha256": hashlib.sha256(checksum_payload).hexdigest(),
        "published_archive_sha256": published_hash,
        "archive_sha256": local_hash,
        "timestamp_unit": raw.attrs["timestamp_unit"],
        "rows": int(len(panel)),
        "first_date": panel["date"].iloc[0],
        "last_date": panel["date"].iloc[-1],
    }
    return panel, metadata


def _validate_config(
    cfg: BuildConfig, *, allow_partial_fixture: bool = False
) -> tuple[date, date, tuple[str, ...]]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    symbols = tuple(symbol.strip().upper() for symbol in cfg.symbols)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if start.day != 1 or end.day != 1:
        raise ValueError("monthly Spot build boundaries must be month starts")
    if start < FROZEN_SOURCE_START:
        raise ValueError(
            f"RFXS source cannot begin before {FROZEN_SOURCE_START.isoformat()}"
        )
    if end > FROZEN_SOURCE_END_EXCLUSIVE:
        raise ValueError(
            "RFXS outcome-blind source is capped at exclusive end "
            f"{FROZEN_SOURCE_END_EXCLUSIVE.isoformat()}"
        )
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique")
    if set(symbols) != set(DEFAULT_SYMBOLS) or len(symbols) != len(DEFAULT_SYMBOLS):
        raise ValueError(f"RFXS requires exactly these symbols: {DEFAULT_SYMBOLS}")
    if not allow_partial_fixture and (
        start != FROZEN_SOURCE_START or end != FROZEN_SOURCE_END_EXCLUSIVE
    ):
        raise ValueError(
            "RFXS production source requires exact horizon "
            f"[{FROZEN_SOURCE_START.isoformat()}, "
            f"{FROZEN_SOURCE_END_EXCLUSIVE.isoformat()})"
        )
    return start, end, DEFAULT_SYMBOLS


def _verify_mechanism() -> tuple[Path, str]:
    source_path = Path(__file__).resolve()
    mechanism_path = source_path.parents[1] / MECHANISM_PATH
    mechanism_hash = hashlib.sha256(mechanism_path.read_bytes()).hexdigest()
    if mechanism_hash != MECHANISM_SHA256:
        raise ValueError("frozen RFXS mechanism document changed")
    return source_path, mechanism_hash


def _wide_panel(frames: list[pd.DataFrame], *, start: date, end: date) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "symbol"], kind="mergesort"
    )
    expected_dates = pd.date_range(start, end, freq="1D", inclusive="left").strftime(
        "%Y-%m-%d"
    )
    expected_index = pd.MultiIndex.from_product(
        [expected_dates, DEFAULT_SYMBOLS], names=["date", "symbol"]
    )
    observed_index = pd.MultiIndex.from_frame(combined[["date", "symbol"]])
    if observed_index.has_duplicates or set(observed_index) != set(expected_index):
        missing = expected_index.difference(observed_index)
        extra = observed_index.difference(expected_index)
        raise ValueError(
            "combined RFXS panel does not match the full date-symbol grid; "
            f"missing={missing.tolist()[:10]}, extra={extra.tolist()[:10]}"
        )
    if not combined["source_complete"].all():
        raise ValueError("combined RFXS panel contains incomplete source rows")
    availability_counts = combined.groupby("date", sort=True)[
        "source_available_not_before"
    ].nunique()
    if not availability_counts.eq(1).all():
        raise ValueError("RFXS symbols disagree on source availability boundary")

    close_wide = combined.pivot(index="date", columns="symbol", values="close")
    close_wide = close_wide.loc[list(expected_dates), list(DEFAULT_SYMBOLS)]
    close_wide.columns = [f"{symbol}_close" for symbol in close_wide.columns]
    availability = (
        combined.groupby("date", sort=True)["source_available_not_before"].first()
    )
    wide = close_wide.reset_index()
    wide.insert(
        1,
        "source_available_not_before",
        availability.reindex(expected_dates).to_numpy(),
    )
    wide["source_complete"] = True
    wide = wide.loc[:, OUTPUT_COLUMNS]
    close_columns = [f"{symbol}_close" for symbol in DEFAULT_SYMBOLS]
    if not np.isfinite(wide[close_columns].to_numpy(float)).all():
        raise ValueError("RFXS output contains non-finite closes")
    if (wide[close_columns] <= 0.0).any().any():
        raise ValueError("RFXS output contains non-positive closes")
    return wide


def build(
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
    _allow_partial_fixture: bool = False,
) -> dict[str, Any]:
    source_path, mechanism_hash = _verify_mechanism()
    start, end, symbols = _validate_config(
        cfg, allow_partial_fixture=_allow_partial_fixture
    )
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

    wide = _wide_panel(frames, start=start, end=end)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_day = end - timedelta(days=1)
    combined_path = output_dir / (
        "BTC_regional_fiat_cross_rate_1d_"
        f"{start.isoformat()}_{last_day.isoformat()}.csv.gz"
    )
    _write_gzip_csv(wide, combined_path)
    symbol_order = {symbol: index for index, symbol in enumerate(DEFAULT_SYMBOLS)}
    archives.sort(key=lambda item: (symbol_order[item["symbol"]], item["month"]))
    config_record = asdict(cfg)
    config_record["symbols"] = list(symbols)
    repository_root = source_path.parents[1]
    builder_record = str(source_path.relative_to(repository_root))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "builder": builder_record,
        "builder_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "mechanism": {
            "commit": MECHANISM_COMMIT,
            "path": MECHANISM_PATH,
            "sha256": mechanism_hash,
        },
        "config": config_record,
        "protocol": {
            "candidate": "RFXS-576",
            "fixture_only": bool(_allow_partial_fixture),
            "source": "official Binance Spot monthly daily-kline archives",
            "archive_root": BASE_URL,
            "archive_checksums_verified": True,
            "checksum_filenames_verified": True,
            "checksum_response_hashes_retained": True,
            "end_is_exclusive": True,
            "time_unit": "milliseconds before 2025; microseconds from 2025",
            "daily_bucket": "UTC open_time with exact 24-hour source span",
            "source_complete": (
                "finite positive OHLC, valid OHLC bounds, positive base volume "
                "and trade count, and bounded taker-buy fields"
            ),
            "signal_fields_retained": ["close"],
            "non_signal_fields_discarded": [
                "open",
                "high",
                "low",
                "base_volume",
                "quote_notional",
                "trade_count",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
            "raw_archives_persisted": False,
            "source_values_opened": True,
            "execution_ohlc_opened": False,
            "funding_opened": False,
            "outcomes_opened": False,
            "revision_policy": (
                "any changed companion or archive hash is a new source revision "
                "and must fail closed"
            ),
        },
        "combined_output": str(combined_path),
        "combined_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        "rows": int(len(wide)),
        "complete_rows": int(wide["source_complete"].sum()),
        "expected_rows": int(
            len(pd.date_range(start, end, freq="1D", inclusive="left"))
        ),
        "first_date": wide["date"].iloc[0],
        "last_date": wide["date"].iloc[-1],
        "symbols": list(symbols),
        "columns": list(wide.columns),
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
