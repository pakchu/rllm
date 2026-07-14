"""Build causal five-minute spot microstructure from Binance one-minute klines.

The official monthly spot kline archives are small enough to stream in memory.
Each archive is checksum-verified, aggregated without retaining raw payloads,
and written as a deterministic gzip resume point.  The output intentionally
contains observables only; no forward return or trading label is computed.
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
INTERVAL = "1m"
SCHEMA_VERSION = 1
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
    "first_open_time_ms",
    "last_close_time_ms",
    "spot_rows",
    "open",
    "high",
    "low",
    "close",
    "base_volume",
    "quote_notional",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "taker_sell_base",
    "taker_sell_quote",
    "signed_quote_notional",
    "flow_coherence",
    "vwap",
    "buyer_execution_centroid",
    "seller_execution_centroid",
    "buyer_seller_centroid_log_gap",
    "centroid_spread_bp",
    "close_vs_centroid_mid_bp",
    "mean_trade_notional",
    "micro_log_return",
    "signed_price_response",
    "minute_flow_sign_flip_rate",
    "minute_flow_price_alignment",
    "minute_price_path_efficiency",
    "minute_flow_path_efficiency",
    "source_complete",
)


@dataclass(frozen=True)
class BuildConfig:
    symbol: str = "BTCUSDT"
    start: str = "2020-01-01"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_spot_kline_microstructure_btc_2020_2023"
    workers: int = 4
    retries: int = 5
    timeout_seconds: int = 60
    overwrite: bool = False


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
        raise ValueError("spot kline archive is empty")
    if not frame["open_time"].is_monotonic_increasing or not frame["open_time"].is_unique:
        raise ValueError("spot kline open times are not strictly increasing")
    numeric = frame.loc[:, RAW_COLUMNS[1:]]
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("spot kline archive contains non-finite values")
    prices = frame[["open", "high", "low", "close"]]
    if (prices <= 0.0).any().any():
        raise ValueError("spot kline archive contains non-positive prices")
    if (frame[["base_volume", "quote_notional", "trade_count"]] < 0.0).any().any():
        raise ValueError("spot kline archive contains negative volume or count")
    tolerance = 1e-8
    if (
        (frame["taker_buy_base"] < -tolerance).any()
        or (frame["taker_buy_quote"] < -tolerance).any()
        or (frame["taker_buy_base"] > frame["base_volume"] + tolerance).any()
        or (frame["taker_buy_quote"] > frame["quote_notional"] + tolerance).any()
    ):
        raise ValueError("spot kline taker-buy fields violate total-volume bounds")
    if (frame["close_time"] < frame["open_time"]).any():
        raise ValueError("spot kline close time precedes open time")
    return frame


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float).divide(denominator.astype(float).replace(0.0, np.nan))


def aggregate_five_minute(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    minute_time = pd.to_datetime(work["open_time"], unit="ms", utc=True, errors="raise")
    work["date"] = minute_time.dt.floor("5min").dt.tz_localize(None)
    work["taker_sell_base"] = work["base_volume"] - work["taker_buy_base"]
    work["taker_sell_quote"] = work["quote_notional"] - work["taker_buy_quote"]
    work["signed_quote"] = work["taker_buy_quote"] - work["taker_sell_quote"]
    work["minute_log_return"] = np.log(work["close"] / work["open"])
    previous_close = work.groupby("date", sort=False, observed=True)["close"].shift()
    path_anchor = previous_close.where(previous_close.notna(), work["open"])
    work["minute_path_log_return"] = np.log(work["close"] / path_anchor)
    work["minute_flow_sign"] = np.sign(work["signed_quote"])
    work["minute_flow_price_relation"] = (
        work["minute_flow_sign"] * np.sign(work["minute_log_return"])
    )
    grouped = work.groupby("date", sort=True, observed=True)
    output = grouped.agg(
        first_open_time_ms=("open_time", "first"),
        last_close_time_ms=("close_time", "last"),
        spot_rows=("open_time", "size"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        base_volume=("base_volume", "sum"),
        quote_notional=("quote_notional", "sum"),
        trade_count=("trade_count", "sum"),
        taker_buy_base=("taker_buy_base", "sum"),
        taker_buy_quote=("taker_buy_quote", "sum"),
        taker_sell_base=("taker_sell_base", "sum"),
        taker_sell_quote=("taker_sell_quote", "sum"),
        signed_quote_notional=("signed_quote", "sum"),
        minute_abs_return_sum=(
            "minute_path_log_return", lambda x: float(np.abs(x).sum())
        ),
        minute_abs_flow_sum=("signed_quote", lambda x: float(np.abs(x).sum())),
        minute_flow_price_alignment=("minute_flow_price_relation", "mean"),
    )
    output["flow_coherence"] = _safe_divide(
        output["signed_quote_notional"].abs(), output["quote_notional"]
    )
    output["vwap"] = _safe_divide(output["quote_notional"], output["base_volume"])
    output["buyer_execution_centroid"] = _safe_divide(
        output["taker_buy_quote"], output["taker_buy_base"]
    )
    output["seller_execution_centroid"] = _safe_divide(
        output["taker_sell_quote"], output["taker_sell_base"]
    )
    output["buyer_seller_centroid_log_gap"] = np.log(
        output["buyer_execution_centroid"] / output["seller_execution_centroid"]
    )
    output["centroid_spread_bp"] = (
        output["buyer_seller_centroid_log_gap"].abs() * 10_000.0
    )
    centroid_mid = np.sqrt(
        output["buyer_execution_centroid"] * output["seller_execution_centroid"]
    )
    output["close_vs_centroid_mid_bp"] = np.log(output["close"] / centroid_mid) * 10_000.0
    output["mean_trade_notional"] = _safe_divide(
        output["quote_notional"], output["trade_count"]
    )
    output["micro_log_return"] = np.log(output["close"] / output["open"])
    output["signed_price_response"] = (
        np.sign(output["signed_quote_notional"]) * output["micro_log_return"]
    )

    group_change = work["date"].ne(work["date"].shift())
    sign_change = work["minute_flow_sign"].ne(work["minute_flow_sign"].shift())
    flip_count = (~group_change & sign_change).groupby(work["date"], observed=True).sum()
    output["minute_flow_sign_flip_rate"] = _safe_divide(
        flip_count, (output["spot_rows"] - 1).clip(lower=1)
    )
    output["minute_price_path_efficiency"] = _safe_divide(
        output["micro_log_return"].abs(), output.pop("minute_abs_return_sum")
    ).fillna(0.0).clip(0.0, 1.0)
    output["minute_flow_path_efficiency"] = _safe_divide(
        output["signed_quote_notional"].abs(), output.pop("minute_abs_flow_sum")
    ).fillna(0.0).clip(0.0, 1.0)

    expected_span = (output["spot_rows"] - 1) * 60_000
    actual_span = grouped["open_time"].max() - grouped["open_time"].min()
    centroid_ok = output[["buyer_execution_centroid", "seller_execution_centroid"]].gt(0.0).all(axis=1)
    output["source_complete"] = (
        output["spot_rows"].eq(5)
        & actual_span.eq(expected_span)
        & output["base_volume"].gt(0.0)
        & output["quote_notional"].gt(0.0)
        & output["trade_count"].gt(0)
        & centroid_ok
    )

    output = output.reset_index().loc[:, OUTPUT_COLUMNS]
    if output["date"].duplicated().any() or not output["date"].is_monotonic_increasing:
        raise ValueError("aggregated spot timestamps are duplicate or unordered")
    complete_numeric = output.loc[output["source_complete"], OUTPUT_COLUMNS[1:-1]]
    if not np.isfinite(complete_numeric.to_numpy(float)).all():
        raise ValueError("complete spot microstructure rows contain non-finite values")
    return output


def _resume_metadata_is_current(
    metadata: dict[str, Any],
    *,
    cfg: BuildConfig,
    month: date,
    output_path: Path,
    fetcher: Callable[..., bytes],
) -> bool:
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("month") != f"{month:%Y-%m}"
        or metadata.get("symbol") != cfg.symbol
    ):
        return False
    if hashlib.sha256(output_path.read_bytes()).hexdigest() != metadata.get("output_sha256"):
        raise ValueError(f"resume artifact hash mismatch: {output_path}")
    current = expected_sha256(
        fetcher(
            checksum_url(cfg.symbol, month),
            retries=cfg.retries,
            timeout=cfg.timeout_seconds,
        )
    )
    return current == metadata.get("archive_sha256")


def _process_month(
    month: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    monthly_dir = output_dir / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.symbol}_spot_kline_microstructure_5m_{month:%Y-%m}"
    output_path = monthly_dir / f"{stem}.csv.gz"
    metadata_path = monthly_dir / f"{stem}.json"
    if output_path.exists() and metadata_path.exists() and not cfg.overwrite:
        metadata = json.loads(metadata_path.read_text())
        if _resume_metadata_is_current(
            metadata,
            cfg=cfg,
            month=month,
            output_path=output_path,
            fetcher=fetcher,
        ):
            return metadata

    checksum_payload = fetcher(
        checksum_url(cfg.symbol, month),
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    expected = expected_sha256(checksum_payload)
    payload = fetcher(
        archive_url(cfg.symbol, month),
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    archive_hash = verify_sha256(payload, expected)
    raw = read_archive(payload)
    aggregated = aggregate_five_minute(raw)
    month_start = pd.Timestamp(month)
    next_month = pd.Timestamp(month_start) + pd.offsets.MonthBegin(1)
    if not (
        (aggregated["date"] >= month_start) & (aggregated["date"] < next_month)
    ).all():
        raise ValueError(f"archive contains timestamps outside {month:%Y-%m}")
    _write_gzip_csv(aggregated, output_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "month": f"{month:%Y-%m}",
        "symbol": cfg.symbol,
        "archive_sha256": archive_hash,
        "raw_rows": int(len(raw)),
        "rows": int(len(aggregated)),
        "complete_rows": int(aggregated["source_complete"].sum()),
        "first_date": str(aggregated["date"].min()),
        "last_date": str(aggregated["date"].max()),
        "output": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def build(cfg: BuildConfig) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if start.day != 1 or end.day != 1:
        raise ValueError("monthly spot build boundaries must be month starts")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    months = _month_starts(start, end)
    metadata: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {executor.submit(_process_month, month, cfg): month for month in months}
        for future in as_completed(futures):
            month = futures[future]
            result = future.result()
            metadata.append(result)
            print(f"completed {month:%Y-%m}: rows={result['rows']}", flush=True)
    metadata.sort(key=lambda item: item["month"])

    frames = [
        pd.read_csv(item["output"], compression="gzip", parse_dates=["date"])
        for item in metadata
    ]
    combined = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    if combined["date"].duplicated().any():
        raise ValueError("combined spot output has duplicate timestamps")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_month = date(end.year - (end.month == 1), 12 if end.month == 1 else end.month - 1, 1)
    combined_path = output_dir / (
        f"{cfg.symbol}_spot_kline_microstructure_5m_"
        f"{start:%Y-%m}_{last_month:%Y-%m}.csv.gz"
    )
    _write_gzip_csv(combined, combined_path)
    manifest = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "source": "official Binance Spot monthly one-minute kline archives",
            "archive_root": BASE_URL,
            "archive_checksums_verified": True,
            "end_is_exclusive": True,
            "five_minute_bin": "UTC floor of one-minute open_time",
            "source_complete": "exactly five contiguous minutes with positive two-sided centroid support",
            "raw_archives_persisted": False,
            "outcomes_opened": False,
        },
        "combined_output": str(combined_path),
        "combined_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        "rows": int(len(combined)),
        "complete_rows": int(combined["source_complete"].astype(bool).sum()),
        "first_date": str(combined["date"].min()),
        "last_date": str(combined["date"].max()),
        "columns": list(combined.columns),
        "months": metadata,
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=BuildConfig.symbol)
    parser.add_argument("--start", default=BuildConfig.start)
    parser.add_argument("--end", default=BuildConfig.end)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--workers", type=int, default=BuildConfig.workers)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument("--timeout-seconds", type=int, default=BuildConfig.timeout_seconds)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build(BuildConfig(**vars(args)))
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "combined_output",
                    "rows",
                    "complete_rows",
                    "first_date",
                    "last_date",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
