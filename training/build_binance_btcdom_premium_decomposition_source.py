"""Build the checksummed, outcome-blind DLPD hourly premium source prefix."""

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
    _month_starts,
    _write_gzip_csv,
    expected_sha256,
    verify_sha256,
)
from training.build_binance_um_premium_path import read_archive


SCHEMA_VERSION = 1
BUILDER_PATH = Path("training/build_binance_btcdom_premium_decomposition_source.py")
INVENTORY_PATH = Path(
    "data/binance_btcdom_premium_decomposition_2021_2023/"
    "archive_checksums.json"
)
INVENTORY_SHA256 = (
    "96240ba01d4cd5720eefdc05aa3a15d94f9c494118815219a9fb3442981f200e"
)
SOURCE_DECISION = Path(
    "docs/btcdom-leverage-polarity-decomposition-source-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "ed402ef2a91e400b29b902154646637987318d57ab0543312f383f1193be3cf6"
)
SYMBOLS = ("BTCUSDT", "BTCDOMUSDT")
INTERVAL = "1h"
START = pd.Timestamp("2021-07-02T00:00:00")
END = pd.Timestamp("2024-01-01T00:00:00")
OUTPUT_COLUMNS = (
    "date",
    "source_close_time",
    "feature_available_time",
    "btcusdt_valid",
    "btcdomusdt_valid",
    "source_valid",
    "btcusdt_premium_close",
    "btcdomusdt_premium_close",
)


@dataclass(frozen=True)
class BuildConfig:
    symbols: tuple[str, ...] = SYMBOLS
    interval: str = INTERVAL
    start: str = START.isoformat()
    end: str = END.isoformat()
    output_dir: str = "data/binance_btcdom_premium_decomposition_2021_2023"
    workers: int = 8
    retries: int = 5
    timeout_seconds: int = 60


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inventory(path: str | Path = INVENTORY_PATH) -> dict[str, Any]:
    target = Path(path)
    if target == INVENTORY_PATH and sha256_file(target) != INVENTORY_SHA256:
        raise ValueError("DLPD checksum inventory hash mismatch")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("source_only") is not True:
        raise ValueError("DLPD checksum inventory is not source-only")
    for key in (
        "outcomes_opened",
        "archive_bytes_downloaded",
        "post_2023_rows_requested",
    ):
        if payload.get(key) is not False:
            raise ValueError(f"DLPD checksum inventory boundary changed: {key}")
    if payload.get("source_decision_sha256") != SOURCE_DECISION_SHA256:
        raise ValueError("DLPD checksum inventory source-decision hash changed")
    records = payload.get("records", [])
    expected = {
        (symbol, f"{month:%Y-%m}")
        for symbol in SYMBOLS
        for month in _month_starts(date(2021, 7, 1), date(2024, 1, 1))
    }
    observed = {(item.get("symbol"), item.get("month")) for item in records}
    if observed != expected or len(records) != len(expected):
        raise ValueError("DLPD checksum inventory coverage changed")
    if any(item.get("interval") != INTERVAL for item in records):
        raise ValueError("DLPD checksum inventory interval changed")
    if any(len(str(item.get("archive_sha256", ""))) != 64 for item in records):
        raise ValueError("DLPD checksum inventory contains an invalid hash")
    return payload


def _validate_config(cfg: BuildConfig) -> None:
    if cfg.symbols != SYMBOLS:
        raise ValueError("DLPD source symbols changed")
    if cfg.interval != INTERVAL:
        raise ValueError("DLPD source interval changed")
    if (pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)) != (START, END):
        raise ValueError("DLPD source prefix changed")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    if sha256_file(SOURCE_DECISION) != SOURCE_DECISION_SHA256:
        raise ValueError("DLPD source decision hash mismatch")


def _record_map(inventory: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(item["symbol"]), str(item["month"])): cast(dict[str, Any], item)
        for item in inventory["records"]
    }


def process_archive(
    symbol: str,
    month: date,
    record: dict[str, Any],
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if (record["symbol"], record["month"], record["interval"]) != (
        symbol,
        f"{month:%Y-%m}",
        INTERVAL,
    ):
        raise ValueError("DLPD archive record identity changed")
    checksum_payload = fetcher(
        record["checksum_url"],
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    published = expected_sha256(checksum_payload)
    frozen = str(record["archive_sha256"])
    if published != frozen:
        raise ValueError(f"published DLPD checksum changed: {symbol} {month:%Y-%m}")
    payload = fetcher(
        record["archive_url"],
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    archive_hash = verify_sha256(payload, frozen)
    parsed = read_archive(payload, interval_minutes=60)
    month_start = pd.Timestamp(month)
    next_month = month_start + pd.offsets.MonthBegin(1)
    if not parsed["date"].between(month_start, next_month, inclusive="left").all():
        raise ValueError(f"DLPD archive escapes its month: {symbol} {month:%Y-%m}")
    if parsed["date"].duplicated().any():
        raise ValueError(f"DLPD archive has duplicate hours: {symbol} {month:%Y-%m}")
    frame = parsed.loc[
        :, ["date", "source_close_time", "feature_available_time", "premium_close"]
    ].copy()
    frame.insert(1, "symbol", symbol)
    return frame, {
        "symbol": symbol,
        "month": f"{month:%Y-%m}",
        "archive_url": record["archive_url"],
        "checksum_url": record["checksum_url"],
        "archive_sha256": archive_hash,
        "rows": int(len(frame)),
        "first_date": cast(pd.Timestamp, frame["date"].min()).isoformat(),
        "last_date": cast(pd.Timestamp, frame["date"].max()).isoformat(),
    }


def pair_panel(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["date", "symbol"]).any():
        raise ValueError("DLPD source contains a duplicate date-symbol row")
    if set(combined["symbol"].unique()) != set(SYMBOLS):
        raise ValueError("DLPD source symbol set changed")

    expected = pd.date_range(START, END, freq="1h", inclusive="left")
    premium = combined.pivot(index="date", columns="symbol", values="premium_close")
    closes = combined.pivot(index="date", columns="symbol", values="source_close_time")
    available = combined.pivot(
        index="date", columns="symbol", values="feature_available_time"
    )
    premium = premium.reindex(expected)
    closes = closes.reindex(expected)
    available = available.reindex(expected)

    theoretical_close = expected + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    theoretical_available = expected + pd.Timedelta(hours=1, seconds=1)
    for symbol in SYMBOLS:
        present = premium[symbol].notna()
        if not closes.loc[present, symbol].equals(
            pd.Series(theoretical_close[present], index=expected[present], name=symbol)
        ):
            raise ValueError(f"DLPD {symbol} close timestamps are not hourly")
        if not available.loc[present, symbol].equals(
            pd.Series(
                theoretical_available[present], index=expected[present], name=symbol
            )
        ):
            raise ValueError(f"DLPD {symbol} availability timestamps changed")

    btc_valid = premium["BTCUSDT"].notna()
    dom_valid = premium["BTCDOMUSDT"].notna()
    output = pd.DataFrame(
        {
            "date": expected,
            "source_close_time": theoretical_close,
            "feature_available_time": theoretical_available,
            "btcusdt_valid": btc_valid.to_numpy(bool),
            "btcdomusdt_valid": dom_valid.to_numpy(bool),
            "source_valid": (btc_valid & dom_valid).to_numpy(bool),
            "btcusdt_premium_close": premium["BTCUSDT"].to_numpy(float),
            "btcdomusdt_premium_close": premium["BTCDOMUSDT"].to_numpy(float),
        }
    )
    valid_values = output.loc[
        output["source_valid"],
        ["btcusdt_premium_close", "btcdomusdt_premium_close"],
    ]
    if not np.isfinite(valid_values.to_numpy(float)).all():
        raise ValueError("DLPD valid premium rows contain non-finite values")
    if tuple(output.columns) != OUTPUT_COLUMNS:
        raise ValueError("DLPD source schema changed")
    return output


def _write_frozen_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if path.exists() and path.read_bytes() != encoded:
        raise FileExistsError(f"existing frozen DLPD manifest differs: {path}")
    path.write_bytes(encoded)


def build(
    cfg: BuildConfig = BuildConfig(),
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    _validate_config(cfg)
    inventory = load_inventory()
    records = _record_map(inventory)
    months = _month_starts(date(2021, 7, 1), date(2024, 1, 1))
    tasks = [(symbol, month) for symbol in SYMBOLS for month in months]

    frames: list[pd.DataFrame] = []
    archives: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {
            executor.submit(
                process_archive,
                symbol,
                month,
                records[(symbol, f"{month:%Y-%m}")],
                cfg,
                fetcher=fetcher,
            ): (symbol, month)
            for symbol, month in tasks
        }
        for future in as_completed(futures):
            frame, metadata = future.result()
            frames.append(frame)
            archives.append(metadata)
            print(
                f"completed {metadata['symbol']} {metadata['month']}: "
                f"rows={metadata['rows']}",
                flush=True,
            )

    panel = pair_panel(frames)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / (
        "BTCUSDT_BTCDOMUSDT_premium_close_1h_"
        "2021-07-02_2023-12-31.csv.gz"
    )
    _write_gzip_csv(panel, data_path)
    archives.sort(key=lambda item: (item["symbol"], item["month"]))
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "name": "DLPD outcome-blind BTC/BTCDOM premium decomposition source",
            "source": "official Binance Vision USD-M monthly premiumIndexKlines",
            "source_only": True,
            "outcomes_opened": False,
            "archive_checksums_verified": True,
            "published_hashes_frozen_before_archive_build": True,
            "post_2023_rows_requested": False,
            "btc_or_btcdom_contract_ohlc_retained": False,
            "btc_or_btcdom_index_prices_retained": False,
            "funding_returns_labels_or_pnl_retained": False,
            "premium_ohlc_paths_retained": False,
            "premium_closes_retained": True,
            "missing_rows_zero_filled": False,
            "end_is_exclusive": True,
        },
        "source_decision": str(SOURCE_DECISION),
        "source_decision_sha256": SOURCE_DECISION_SHA256,
        "checksum_inventory": str(INVENTORY_PATH),
        "checksum_inventory_sha256": INVENTORY_SHA256,
        "builder": str(BUILDER_PATH),
        "builder_sha256": sha256_file(BUILDER_PATH),
        "config": {**asdict(cfg), "symbols": list(cfg.symbols)},
        "columns": list(OUTPUT_COLUMNS),
        "combined_output": str(data_path),
        "combined_sha256": sha256_file(data_path),
        "rows": int(len(panel)),
        "valid_rows": int(panel["source_valid"].sum()),
        "btcusdt_valid_rows": int(panel["btcusdt_valid"].sum()),
        "btcdomusdt_valid_rows": int(panel["btcdomusdt_valid"].sum()),
        "first_date": cast(pd.Timestamp, panel["date"].min()).isoformat(),
        "last_date": cast(pd.Timestamp, panel["date"].max()).isoformat(),
        "archives": archives,
    }
    _write_frozen_json(output_dir / "build_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--workers", type=int, default=BuildConfig.workers)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument("--timeout-seconds", type=int, default=BuildConfig.timeout_seconds)
    args = parser.parse_args()
    result = build(
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
                key: result[key]
                for key in (
                    "combined_output",
                    "combined_sha256",
                    "rows",
                    "valid_rows",
                    "first_date",
                    "last_date",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
