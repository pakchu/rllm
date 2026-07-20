"""Build the outcome-blind direct stablecoin/USDT hourly source prefix."""

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
    _month_starts,
    archive_url,
    checksum_url,
    read_archive,
)


SCHEMA_VERSION = 1
DEFAULT_SYMBOLS = ("USDCUSDT", "TUSDUSDT", "USDPUSDT", "FDUSDUSDT")
MINIMUM_VALID_BREADTH = 3
EXPECTED_ARCHIVE_SHA256: dict[tuple[str, str], str] = {
    ("USDCUSDT", "2023-08"): "dfa83e699b157bfb5530871b984d2217a2160764bdeb1a37f7c28f4872ad44ab",
    ("USDCUSDT", "2023-09"): "b67bf5515102f2e98e8961d517fb1515862f7cdcd5e5e9d59409239219c8f78c",
    ("USDCUSDT", "2023-10"): "1586dcdc9eb86ed01d2ddef4d8e1a9b42720a6a6785c175d7687cd355adbaecb",
    ("USDCUSDT", "2023-11"): "1bdcee74530264ee01231ec2ebc5cfe8c467f3e53e348b98c878604f5f9b9277",
    ("USDCUSDT", "2023-12"): "0bbd6675316d248d8a00bd94a3394a0cf5a722a7bebf2e8355a83263443a1097",
    ("TUSDUSDT", "2023-08"): "9ef7e816f604af91bee2f6d6fff1fafa2a82c8ccf6c62dfd9ec34f52f9e6c22f",
    ("TUSDUSDT", "2023-09"): "0091b7a27c2012a145c17e917a4aa53b8ede53b8ba942f925cb4cb32acd23a4e",
    ("TUSDUSDT", "2023-10"): "b722f558037ef552f5e150fd6a986638371aaad35f0f1c0b00fb43c5304b321c",
    ("TUSDUSDT", "2023-11"): "222c53c248acb2ff320904c3ef7a8bed8ae971267fc8b1347b3d93848207163f",
    ("TUSDUSDT", "2023-12"): "ce9e7c714d0dfc2676dd327b00b027ec538f77e27c3395f3c8135781509eda03",
    ("USDPUSDT", "2023-08"): "6ba0652fd8912b7dd88f5a4dc3c21ae39a3c5f441f994b16cb62206c86f6b05d",
    ("USDPUSDT", "2023-09"): "8678f668927505ce3cc9264b1bf4db4d06c8efa239222e4d7d113e2ecab23b1f",
    ("USDPUSDT", "2023-10"): "ebfb03b121b105adbd203b71c1a4c725d195b2afde8ef097fda8ceda17567eb5",
    ("USDPUSDT", "2023-11"): "7ac750f6db0a93c85e15f4b3ba9fb26eceda5d2543e5f0b38cd587c7b275d4cd",
    ("USDPUSDT", "2023-12"): "cf5a6d729fac22fd716b2dd7c839acda1b13327303d30be16731292c5237238a",
    ("FDUSDUSDT", "2023-08"): "6abf7f6eae45d59b236ad0f0875a46a49dd384ec9342b9da80ea32b5b842e34f",
    ("FDUSDUSDT", "2023-09"): "40da78368c7cab3c521f468c921c2ea3954a28c40381d7532a60b9a258753616",
    ("FDUSDUSDT", "2023-10"): "44bdf02d4397bc96745e53b61813679e25f592f942924d6c37c4d6323dddf4e5",
    ("FDUSDUSDT", "2023-11"): "5f38d88e4bbde5006cd65c7afd4db4f8445a2d7937c29f87acd4f8632a88c3f4",
    ("FDUSDUSDT", "2023-12"): "63c4d2eb6a3c5e4011b93c3fc24432ab445771944ae06cdb33ed2bceb8675386",
}
LOG_COLUMNS = tuple(f"{symbol.lower()}_log_close" for symbol in DEFAULT_SYMBOLS)
VALID_COLUMNS = tuple(f"{symbol.lower()}_valid" for symbol in DEFAULT_SYMBOLS)
OUTPUT_COLUMNS = (
    "date",
    "source_available_at",
    *LOG_COLUMNS,
    *VALID_COLUMNS,
    "valid_breadth",
    "source_complete",
)


@dataclass(frozen=True)
class BuildConfig:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    start: str = "2023-08-01"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_usdt_collateral_breadth_2023"
    workers: int = 8
    retries: int = 5
    timeout_seconds: int = 60


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _expected_hours(month: date) -> pd.DatetimeIndex:
    start = cast(pd.Timestamp, pd.Timestamp(month, tz="UTC"))
    end = cast(pd.Timestamp, start + pd.offsets.MonthBegin(1))
    return pd.date_range(start, end, freq="1h", inclusive="left")


def pair_panel(frame: pd.DataFrame, *, symbol: str, unit: str) -> pd.DataFrame:
    """Retain a direct log close and validity while raw kline fields are ephemeral."""
    multiplier = 1_000 if unit == "ms" else 1
    close = frame["close"].astype(float)
    valid = (
        frame["base_volume"].astype(float).gt(0.0)
        & frame["quote_notional"].astype(float).gt(0.0)
        & frame["trade_count"].astype("int64").gt(0)
    )
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(
                frame["open_time"], unit=unit, utc=True
            ).dt.tz_localize(None),
            "symbol": symbol,
            "close_time_us": frame["close_time"].astype("int64") * multiplier,
            "log_close": np.log(close),
            "valid": valid,
        }
    )
    if not np.isfinite(output[["close_time_us", "log_close"]].to_numpy(float)).all():
        raise ValueError("direct stablecoin source contains non-finite values")
    return output


def _process_archive(
    symbol: str,
    month: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    key = (symbol, f"{month:%Y-%m}")
    frozen_hash = EXPECTED_ARCHIVE_SHA256.get(key)
    if frozen_hash is None:
        raise ValueError(f"archive is outside the frozen source prefix: {key}")
    checksum_payload = fetcher(
        checksum_url(symbol, month),
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    published_hash = expected_sha256(checksum_payload)
    if published_hash != frozen_hash:
        raise ValueError(f"published checksum changed after source decision: {key}")
    payload = fetcher(
        archive_url(symbol, month),
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    archive_hash = verify_sha256(payload, frozen_hash)
    raw, unit = read_archive(payload)
    panel = pair_panel(raw, symbol=symbol, unit=unit)
    observed = pd.DatetimeIndex(panel["date"]).tz_localize("UTC")
    if not observed.equals(_expected_hours(month)):
        raise ValueError(f"{symbol} {month:%Y-%m} does not match exact hourly grid")
    return panel, {
        "symbol": symbol,
        "month": f"{month:%Y-%m}",
        "archive_url": archive_url(symbol, month),
        "checksum_url": checksum_url(symbol, month),
        "archive_sha256": archive_hash,
        "timestamp_unit": unit,
        "rows": int(len(panel)),
        "valid_rows": int(panel["valid"].sum()),
        "first_date": cast(pd.Timestamp, panel["date"].min()).isoformat(),
        "last_date": cast(pd.Timestamp, panel["date"].max()).isoformat(),
    }


def breadth_panel(
    frames: list[pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate direct stablecoin date-symbol row")
    if set(combined["symbol"].unique()) != set(DEFAULT_SYMBOLS):
        raise ValueError("direct stablecoin source basket changed")
    expected = pd.date_range(start, end, freq="1h", inclusive="left")
    log_close = combined.pivot(index="date", columns="symbol", values="log_close")
    valid = combined.pivot(index="date", columns="symbol", values="valid")
    close_times = combined.pivot(
        index="date", columns="symbol", values="close_time_us"
    )
    log_close = log_close.reindex(expected)
    valid = valid.reindex(expected)
    close_times = close_times.reindex(expected)
    if log_close.isna().any().any() or valid.isna().any().any():
        raise ValueError("direct stablecoin common grid is incomplete")
    if close_times.isna().any().any() or not close_times.nunique(axis=1).eq(1).all():
        raise ValueError("direct stablecoin close timestamps are missing or misaligned")
    output = pd.DataFrame(
        {
            "date": expected,
            "source_available_at": expected + pd.Timedelta(hours=1),
        }
    )
    for symbol in DEFAULT_SYMBOLS:
        output[f"{symbol.lower()}_log_close"] = log_close[symbol].to_numpy(float)
        output[f"{symbol.lower()}_valid"] = valid[symbol].to_numpy(bool)
    output["valid_breadth"] = output.loc[:, VALID_COLUMNS].sum(axis=1).astype("int8")
    output["source_complete"] = output["valid_breadth"].ge(MINIMUM_VALID_BREADTH)
    output = output.loc[:, OUTPUT_COLUMNS]
    if not np.isfinite(output.loc[:, LOG_COLUMNS].to_numpy(float)).all():
        raise ValueError("direct stablecoin log-close panel contains non-finite values")
    return output


def _validate_config(cfg: BuildConfig) -> tuple[date, date, tuple[str, ...]]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    symbols = tuple(str(symbol).strip().upper() for symbol in cfg.symbols)
    if (start, end) != (date(2023, 8, 1), date(2024, 1, 1)):
        raise ValueError("direct stablecoin source is frozen to 2023-08 through 2023-12")
    if symbols != DEFAULT_SYMBOLS:
        raise ValueError("symbols must equal the frozen direct stablecoin basket")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    return start, end, symbols


def build(
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    start, end, symbols = _validate_config(cfg)
    months = _month_starts(start, end)
    tasks = [(symbol, month) for symbol in symbols for month in months]
    if set((symbol, f"{month:%Y-%m}") for symbol, month in tasks) != set(
        EXPECTED_ARCHIVE_SHA256
    ):
        raise ValueError("frozen direct stablecoin archive set changed")

    frames: list[pd.DataFrame] = []
    archives: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {
            executor.submit(
                _process_archive,
                symbol,
                month,
                cfg,
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

    panel = breadth_panel(
        frames,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first_date = cast(pd.Timestamp, panel["date"].min())
    last_date = cast(pd.Timestamp, panel["date"].max())
    output = output_dir / (
        "stablecoin_usdt_breadth_1h_"
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
            "archive_hashes_frozen_before_complete_build": True,
            "initial_source_only_prefix": True,
            "end_is_exclusive": True,
            "hourly_bucket": "completed UTC hour; available at the next exact hour",
            "minimum_valid_breadth": MINIMUM_VALID_BREADTH,
            "raw_ohlc_retained": False,
            "volume_trade_count_or_taker_flow_retained": False,
            "direct_stablecoin_log_closes_retained": True,
            "btc_prices_opened": False,
            "perpetual_ohlc_or_funding_opened": False,
            "future_returns_labels_or_pnl_opened": False,
            "post_2023_rows_requested": False,
            "outcomes_opened": False,
        },
        "feature_definition": {
            "log_close": "log(direct stablecoin/USDT hourly close)",
            "valid": "base volume, quote notional, and trade count are all positive in the current hour",
            "valid_breadth": "number of valid direct stablecoin books among the frozen four",
            "source_complete": f"valid_breadth >= {MINIMUM_VALID_BREADTH}",
            "interpretation_caveat": "direct Binance Spot prices are market observations, not official issuer redemption values",
        },
        "combined_output": str(output),
        "combined_sha256": sha256_file(output),
        "rows": int(len(panel)),
        "complete_rows": int(panel["source_complete"].sum()),
        "minimum_observed_breadth": int(panel["valid_breadth"].min()),
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
    parser = argparse.ArgumentParser(description=__doc__)
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
                    "complete_rows",
                    "minimum_observed_breadth",
                    "first_date",
                    "last_date",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
