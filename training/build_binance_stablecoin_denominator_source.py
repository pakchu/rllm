"""Build an outcome-blind BTC stablecoin-denominator cross-price panel.

The builder replays the same checksum-verified Binance Spot hourly kline
archives used by ``build_binance_stablecoin_quote_flow``.  It retains only
simultaneous cross-quote log price ratios and discards every raw BTC price and
flow field.  It never reads BTCUSDT perpetual OHLC, funding, future returns,
labels, or PnL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from training.build_binance_stablecoin_quote_flow import (
    ACTIVATION_UTC,
    DEFAULT_SYMBOLS,
    _expected_hours,
    _month_starts,
    archive_url,
    checksum_url,
    read_archive,
)


SCHEMA_VERSION = 1
COMMON_START_UTC = cast(pd.Timestamp, pd.Timestamp("2023-08-04T08:00:00Z"))
REFERENCE_MANIFEST = Path(
    "data/binance_stablecoin_quote_flow_btc_2023_2026/build_manifest.json"
)
REFERENCE_MANIFEST_SHA256 = (
    "9e6a82b9747df5c0ba1c9278e436551de03ef6136c0ad3aeb05f0a451ed12134"
)
OUTPUT_COLUMNS = (
    "date",
    "source_available_at",
    "usdc_vs_usdt",
    "fdusd_vs_usdt",
    "alt_consensus",
    "alt_disagreement",
    "source_complete",
)


@dataclass(frozen=True)
class BuildConfig:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    start: str = "2023-08-01"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_stablecoin_denominator_btc_2023"
    reference_manifest: str = str(REFERENCE_MANIFEST)
    workers: int = 8
    retries: int = 5
    timeout_seconds: int = 60


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_reference(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    manifest_path = Path(path)
    if sha256_file(manifest_path) != REFERENCE_MANIFEST_SHA256:
        raise ValueError("stablecoin quote-flow reference manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("stablecoin quote-flow reference opened outcomes")
    if protocol.get("price_fields_retained") is not False:
        raise ValueError("stablecoin quote-flow reference retained price fields")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for item in manifest.get("archives", []):
        key = (str(item["symbol"]), str(item["month"]))
        if key in records:
            raise ValueError(f"duplicate reference archive: {key}")
        records[key] = item
    return records


def price_panel(frame: pd.DataFrame, *, symbol: str, unit: str) -> pd.DataFrame:
    """Expose raw closes only inside the builder's ephemeral alignment step."""
    multiplier = 1_000 if unit == "ms" else 1
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(
                frame["open_time"], unit=unit, utc=True
            ).dt.tz_localize(None),
            "symbol": str(symbol),
            "close_time_us": frame["close_time"].astype("int64") * multiplier,
            "close": frame["close"].astype(float),
        }
    )
    if not np.isfinite(output[["close_time_us", "close"]].to_numpy(float)).all():
        raise ValueError("stablecoin denominator source contains non-finite values")
    if not output["close"].gt(0.0).all():
        raise ValueError("stablecoin denominator close prices must be positive")
    return output


def _process_archive(
    symbol: str,
    month: date,
    cfg: BuildConfig,
    reference: dict[str, Any],
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
    panel = price_panel(raw, symbol=symbol, unit=unit)
    observed = pd.DatetimeIndex(panel["date"]).tz_localize("UTC")
    expected_index = _expected_hours(symbol, month)
    if not observed.equals(expected_index):
        raise ValueError(f"{symbol} {month:%Y-%m} does not match exact active grid")
    if archive_hash != reference.get("archive_sha256"):
        raise ValueError(f"{symbol} {month:%Y-%m} archive differs from frozen reference")
    if int(len(panel)) != int(reference.get("rows", -1)):
        raise ValueError(f"{symbol} {month:%Y-%m} row count differs from frozen reference")
    metadata = {
        "symbol": symbol,
        "month": f"{month:%Y-%m}",
        "archive_url": archive_url(symbol, month),
        "checksum_url": checksum_url(symbol, month),
        "archive_sha256": archive_hash,
        "timestamp_unit": unit,
        "rows": int(len(panel)),
        "first_date": cast(pd.Timestamp, panel["date"].min()).isoformat(),
        "last_date": cast(pd.Timestamp, panel["date"].max()).isoformat(),
    }
    return panel, metadata


def cross_quote_panel(
    frames: list[pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate stablecoin denominator date-symbol row")
    required = set(DEFAULT_SYMBOLS)
    if set(combined["symbol"].unique()) != required:
        raise ValueError("stablecoin denominator source requires the frozen three-book basket")

    prices = combined.pivot(index="date", columns="symbol", values="close")
    close_times = combined.pivot(index="date", columns="symbol", values="close_time_us")
    common_start = max(cast(pd.Timestamp, start), COMMON_START_UTC.tz_localize(None))
    expected = pd.date_range(common_start, end, freq="1h", inclusive="left")
    prices = prices.reindex(expected)
    close_times = close_times.reindex(expected)
    if prices.isna().any().any() or close_times.isna().any().any():
        raise ValueError("stablecoin denominator common source grid is incomplete")
    if not close_times.nunique(axis=1).eq(1).all():
        raise ValueError("stablecoin denominator close timestamps are misaligned")

    usdc = np.log(prices["BTCUSDC"] / prices["BTCUSDT"])
    fdusd = np.log(prices["BTCFDUSD"] / prices["BTCUSDT"])
    output = pd.DataFrame(
        {
            "date": expected,
            "source_available_at": expected + pd.Timedelta(hours=1),
            "usdc_vs_usdt": usdc.to_numpy(float),
            "fdusd_vs_usdt": fdusd.to_numpy(float),
        }
    )
    output["alt_consensus"] = (
        output["usdc_vs_usdt"] + output["fdusd_vs_usdt"]
    ) / 2.0
    output["alt_disagreement"] = (
        output["usdc_vs_usdt"] - output["fdusd_vs_usdt"]
    ).abs()
    output["source_complete"] = True
    output = output.loc[:, OUTPUT_COLUMNS]
    values = output[list(OUTPUT_COLUMNS[2:-1])].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("stablecoin denominator ratios contain non-finite values")
    return output


def _validate_config(cfg: BuildConfig) -> tuple[date, date, tuple[str, ...]]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    symbols = tuple(str(symbol).strip().upper() for symbol in cfg.symbols)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if start.day != 1 or end.day != 1:
        raise ValueError("stablecoin denominator boundaries must be month starts")
    if start < date(2023, 8, 1) or end > date(2024, 1, 1):
        raise ValueError("source-only builder is frozen to the initial pre-2024 prefix")
    if symbols != DEFAULT_SYMBOLS:
        raise ValueError("symbols must equal the frozen three-book basket in order")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    return start, end, symbols


def build(
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    start, end, symbols = _validate_config(cfg)
    reference = _load_reference(cfg.reference_manifest)
    months = _month_starts(start, end)
    tasks = [
        (symbol, month)
        for symbol in symbols
        for month in months
        if len(_expected_hours(symbol, month)) > 0
    ]
    missing_reference = [
        (symbol, f"{month:%Y-%m}")
        for symbol, month in tasks
        if (symbol, f"{month:%Y-%m}") not in reference
    ]
    if missing_reference:
        raise ValueError(f"reference manifest is missing archives: {missing_reference}")

    frames: list[pd.DataFrame] = []
    archives: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {
            executor.submit(
                _process_archive,
                symbol,
                month,
                cfg,
                reference[(symbol, f"{month:%Y-%m}")],
                fetcher=fetcher,
            ): (symbol, month)
            for symbol, month in tasks
        }
        for future in as_completed(futures):
            symbol, month = futures[future]
            panel, metadata = future.result()
            frames.append(panel)
            archives.append(metadata)
            print(f"completed {symbol} {month:%Y-%m}: rows={len(panel)}", flush=True)

    panel = cross_quote_panel(
        frames,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first_date = cast(pd.Timestamp, panel["date"].min())
    last_date = cast(pd.Timestamp, panel["date"].max())
    output = output_dir / (
        "BTC_stablecoin_denominator_1h_"
        f"{first_date:%Y-%m-%dT%H}_{last_date:%Y-%m-%dT%H}.csv.gz"
    )
    _write_gzip_csv(panel, output)
    archives.sort(key=lambda item: (item["symbol"], item["month"]))
    config_record = asdict(cfg)
    config_record["symbols"] = list(symbols)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config": config_record,
        "protocol": {
            "source": "official Binance Spot monthly hourly-kline archives",
            "archive_checksums_verified": True,
            "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
            "initial_source_only_prefix": True,
            "end_is_exclusive": True,
            "hourly_bucket": "completed UTC hour; available at the next exact hour",
            "raw_btc_prices_retained": False,
            "flow_or_volume_fields_retained": False,
            "cross_quote_log_ratios_retained": True,
            "perpetual_ohlc_or_funding_opened": False,
            "future_returns_labels_or_pnl_opened": False,
            "post_2023_rows_requested": False,
            "outcomes_opened": False,
        },
        "feature_definition": {
            "usdc_vs_usdt": "log(BTCUSDC close / BTCUSDT close)",
            "fdusd_vs_usdt": "log(BTCFDUSD close / BTCUSDT close)",
            "alt_consensus": "arithmetic mean of the two simultaneous log ratios",
            "alt_disagreement": "absolute difference of the two simultaneous log ratios",
            "interpretation_caveat": "research proxy for relative quote-denominator value, not an official stablecoin FX rate",
        },
        "combined_output": str(output),
        "combined_sha256": sha256_file(output),
        "rows": int(len(panel)),
        "complete_rows": int(panel["source_complete"].sum()),
        "first_date": first_date.isoformat(),
        "last_date": last_date.isoformat(),
        "columns": list(panel.columns),
        "archives": archives,
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--workers", type=int, default=BuildConfig.workers)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument("--timeout-seconds", type=int, default=BuildConfig.timeout_seconds)
    args = parser.parse_args()
    manifest = build(
        BuildConfig(
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
                    "first_date",
                    "last_date",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
