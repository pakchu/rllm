"""Freeze official BTCUSDT execution and funding sources for CLBR-24.

This source stage verifies every Binance Vision daily USD-M 5m kline archive,
downloads bounded official funding/mark history, and writes physically separated
train/test/eval files.  It never loads CLBR clocks or computes a return.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
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
from training.build_binance_um_kline_reference import (
    archive_url as kline_archive_url,
    checksum_url as kline_checksum_url,
    read_archive as read_kline_archive,
)
from training.preregister_coinm_liquidation_burst_release import SPLITS


SYMBOL = "BTCUSDT"
INTERVAL = "5m"
START = pd.Timestamp("2023-06-25 00:00:00")
END = pd.Timestamp("2024-10-15 00:00:00")
START_DATE = date(2023, 6, 25)
END_DATE = date(2024, 10, 15)
START_MS = int(datetime(2023, 6, 25, tzinfo=timezone.utc).timestamp() * 1_000)
END_MS = int(datetime(2024, 10, 15, tzinfo=timezone.utc).timestamp() * 1_000)
FUNDING_ENDPOINT = "https://fapi.binance.com/fapi/v1/fundingRate"
MARK_ENDPOINT = "https://fapi.binance.com/fapi/v1/markPriceKlines"
FUNDING_DOCS = (
    "https://developers.binance.com/en/docs/catalog/"
    "core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data"
    "#get-funding-rate-history"
)
MARK_DOCS = (
    "https://developers.binance.com/en/docs/catalog/"
    "core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data"
    "#mark-price-klinecandlestick-data"
)
MARK_INTERVAL = "8h"
MARK_STEP_MS = 8 * 60 * 60 * 1_000
SCHEMA_VERSION = 1
OpenJson = Callable[[str], Any]


@dataclass(frozen=True)
class Config:
    start: str = str(START)
    end: str = str(END)
    output_dir: str = "data/clbr_execution_sources_2023_2024"
    manifest: str = "results/clbr_execution_sources_2023_2024_manifest.json"
    workers: int = 12
    retries: int = 5
    timeout_seconds: int = 60
    retry_backoff_seconds: float = 0.5


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days)]


def process_market_day(
    day: date,
    cfg: Config,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    checksum_payload = fetcher(
        kline_checksum_url(SYMBOL, INTERVAL, day),
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    expected_hash = expected_sha256(checksum_payload)
    payload = fetcher(
        kline_archive_url(SYMBOL, INTERVAL, day),
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    archive_hash = verify_sha256(payload, expected_hash)
    frame = read_kline_archive(payload)
    expected_grid = pd.Series(
        pd.date_range(day, day + timedelta(days=1), freq="5min", inclusive="left"),
        name="date",
    )
    if not frame["date"].equals(expected_grid):
        raise ValueError(f"BTCUSDT market archive has incomplete day {day}")
    return {
        "date": day.isoformat(),
        "archive_url": kline_archive_url(SYMBOL, INTERVAL, day),
        "checksum_url": kline_checksum_url(SYMBOL, INTERVAL, day),
        "archive_sha256": archive_hash,
        "expected_archive_sha256": expected_hash,
        "checksum_payload_sha256": hashlib.sha256(checksum_payload).hexdigest(),
        "rows": int(len(frame)),
        "frame": frame[["date", "open", "high", "low", "close"]].copy(),
    }


def _default_open_json(url: str, cfg: Config) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": "rllm-clbr-source-freeze/1.0"}
    )
    error: BaseException | None = None
    for attempt in range(cfg.retries):
        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=cfg.timeout_seconds
            ) as response:
                return json.loads(response.read())
        except (OSError, TimeoutError, json.JSONDecodeError) as exc:
            error = exc
            if attempt + 1 < cfg.retries:
                time.sleep(cfg.retry_backoff_seconds * (2**attempt))
    raise RuntimeError(f"failed to download {url}") from error


def _request_json(
    endpoint: str,
    params: dict[str, Any],
    cfg: Config,
    opener: OpenJson | None,
) -> Any:
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    return opener(url) if opener is not None else _default_open_json(url, cfg)


def download_funding(cfg: Config, *, opener: OpenJson | None = None) -> tuple[pd.DataFrame, int]:
    start_ms = START_MS
    end_ms = END_MS
    rows: list[dict[str, Any]] = []
    pages = 0
    cursor = start_ms
    while cursor < end_ms:
        payload = _request_json(
            FUNDING_ENDPOINT,
            {
                "symbol": SYMBOL,
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": 1_000,
            },
            cfg,
            opener,
        )
        pages += 1
        if not isinstance(payload, list) or not all(
            isinstance(item, dict) for item in payload
        ):
            raise ValueError("funding response must be an array of objects")
        if not payload:
            break
        rows.extend(cast(list[dict[str, Any]], payload))
        last_time = int(payload[-1]["fundingTime"])
        if last_time < cursor:
            raise ValueError("funding pagination moved backward")
        cursor = last_time + 1
        if len(payload) < 1_000:
            break
    if not rows:
        raise ValueError("funding history is empty")

    frame = pd.DataFrame(rows)
    required = {"symbol", "fundingTime", "fundingRate", "markPrice"}
    if not required.issubset(frame.columns):
        raise ValueError("funding history lacks required fields")
    if not bool(cast(pd.Series, frame["symbol"]).eq(SYMBOL).all()):
        raise ValueError("funding history contains another symbol")
    funding_time = cast(
        pd.Series, pd.to_numeric(frame["fundingTime"], errors="raise")
    )
    funding_rate = cast(
        pd.Series, pd.to_numeric(frame["fundingRate"], errors="raise")
    )
    raw_recorded_mark = cast(pd.Series, frame["markPrice"])
    absent_recorded_mark = raw_recorded_mark.isna() | (
        raw_recorded_mark.astype("string").str.strip().eq("").fillna(False)
    )
    recorded_mark = pd.Series(np.nan, index=frame.index, dtype=float)
    try:
        parsed_recorded_mark = cast(
            pd.Series,
            pd.to_numeric(
                raw_recorded_mark.loc[~absent_recorded_mark], errors="raise"
            ),
        ).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("funding history contains an invalid recorded mark") from exc
    parsed_values = parsed_recorded_mark.to_numpy(float)
    if not np.isfinite(parsed_values).all() or (parsed_values <= 0.0).any():
        raise ValueError("funding history contains an invalid recorded mark")
    recorded_mark.loc[~absent_recorded_mark] = parsed_recorded_mark
    frame["funding_time_ms"] = funding_time.astype("int64")
    frame["funding_rate"] = funding_rate.astype(float)
    frame["recorded_mark_price"] = recorded_mark.astype(float)
    if bool(frame["funding_time_ms"].duplicated().any()):
        raise ValueError("funding history contains duplicate timestamps")
    frame = frame.sort_values("funding_time_ms").reset_index(drop=True)
    if not frame["funding_time_ms"].is_monotonic_increasing:
        raise ValueError("funding timestamps are invalid")
    if bool(frame["funding_time_ms"].lt(start_ms).any()) or bool(
        frame["funding_time_ms"].ge(end_ms).any()
    ):
        raise ValueError("funding response opened data outside the frozen range")
    if not np.isfinite(frame["funding_rate"].to_numpy(float)).all():
        raise ValueError("funding rates are non-finite")
    canonical = (
        frame["funding_time_ms"].to_numpy(np.int64) // MARK_STEP_MS * MARK_STEP_MS
    )
    expected_canonical = np.arange(start_ms, end_ms, MARK_STEP_MS, dtype=np.int64)
    if not np.array_equal(canonical, expected_canonical):
        raise ValueError("funding history has an incomplete canonical 8h grid")
    offsets = frame["funding_time_ms"].to_numpy(np.int64) - canonical
    if (offsets < 0).any() or (offsets > 60_000).any():
        raise ValueError("funding time is too far from its canonical boundary")
    return frame, pages


def download_mark_klines(
    cfg: Config, *, opener: OpenJson | None = None
) -> tuple[pd.DataFrame, int]:
    start_ms = START_MS
    end_ms = END_MS
    rows: list[list[Any]] = []
    pages = 0
    cursor = start_ms
    while cursor < end_ms:
        payload = _request_json(
            MARK_ENDPOINT,
            {
                "symbol": SYMBOL,
                "interval": MARK_INTERVAL,
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": 1_500,
            },
            cfg,
            opener,
        )
        pages += 1
        if not isinstance(payload, list) or not all(
            isinstance(item, list) and len(item) >= 7 for item in payload
        ):
            raise ValueError("mark-price response must contain kline arrays")
        if not payload:
            break
        rows.extend(cast(list[list[Any]], payload))
        last_open = int(payload[-1][0])
        if last_open < cursor:
            raise ValueError("mark-price pagination moved backward")
        cursor = last_open + MARK_STEP_MS
        if len(payload) < 1_500:
            break
    if not rows:
        raise ValueError("mark-price history is empty")
    frame = pd.DataFrame(
        {
            "mark_open_time_ms": [int(row[0]) for row in rows],
            "settlement_mark_price": pd.to_numeric(
                [row[1] for row in rows], errors="raise"
            ),
        }
    )
    if bool(frame["mark_open_time_ms"].duplicated().any()):
        raise ValueError("mark-price history contains duplicate timestamps")
    frame = frame.sort_values("mark_open_time_ms").reset_index(drop=True)
    expected = np.arange(start_ms, end_ms, MARK_STEP_MS, dtype=np.int64)
    if not np.array_equal(
        frame["mark_open_time_ms"].to_numpy(np.int64), expected
    ):
        raise ValueError("8h mark-price grid is incomplete")
    prices = frame["settlement_mark_price"].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("mark prices are invalid")
    return frame, pages


def compose_funding(
    funding: pd.DataFrame, marks: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = funding.copy()
    frame["mark_open_time_ms"] = (
        frame["funding_time_ms"].to_numpy(np.int64) // MARK_STEP_MS * MARK_STEP_MS
    )
    frame["funding_time_offset_ms"] = (
        frame["funding_time_ms"] - frame["mark_open_time_ms"]
    )
    if bool(frame["funding_time_offset_ms"].lt(0).any()) or bool(
        frame["funding_time_offset_ms"].gt(60_000).any()
    ):
        raise ValueError("funding time is too far from its canonical boundary")
    frame = frame.merge(
        marks, on="mark_open_time_ms", how="left", validate="many_to_one"
    )
    if bool(frame["settlement_mark_price"].isna().any()):
        raise ValueError("funding event lacks a settlement mark")
    frame = frame.rename(columns={"settlement_mark_price": "proxy_mark_price"})
    recorded = frame["recorded_mark_price"]
    overlap = recorded.notna()
    mark_error_bp = pd.Series(dtype=float)
    funding_cash_error_bp = pd.Series(dtype=float)
    if bool(overlap.any()):
        mark_error_bp = (
            frame.loc[overlap, "proxy_mark_price"] / recorded.loc[overlap] - 1.0
        ).abs() * 10_000.0
        funding_cash_error_bp = mark_error_bp * frame.loc[
            overlap, "funding_rate"
        ].abs()
        if float(funding_cash_error_bp.max()) > 0.1:
            raise ValueError("8h mark proxy funding-cash error exceeds 0.1bp")
    settlement_mark = recorded.where(overlap, frame["proxy_mark_price"])
    mark_source = pd.Series(
        np.where(
            overlap,
            "funding_history_recorded_mark",
            "binance_8h_mark_price_kline_open",
        ),
        index=frame.index,
    )
    output = pd.DataFrame(
        {
            "funding_time": pd.to_datetime(
                frame["funding_time_ms"], unit="ms", utc=True
            ).dt.tz_localize(None),
            "funding_time_ms": frame["funding_time_ms"].astype("int64"),
            "funding_rate": frame["funding_rate"].astype(float),
            "settlement_mark_price": settlement_mark.astype(float),
            "mark_source": mark_source,
            "funding_time_offset_ms": frame["funding_time_offset_ms"].astype("int64"),
        }
    )
    quality = {
        "events": int(len(output)),
        "recorded_mark_overlap_events": int(overlap.sum()),
        "mark_proxy_events": int((~overlap).sum()),
        "maximum_funding_time_offset_ms": int(output["funding_time_offset_ms"].max()),
        "maximum_recorded_mark_error_bp": (
            0.0 if mark_error_bp.empty else float(mark_error_bp.max())
        ),
        "maximum_funding_cash_error_bp_notional": (
            0.0
            if funding_cash_error_bp.empty
            else float(funding_cash_error_bp.max())
        ),
    }
    return output, quality


def _public_market_record(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "frame"}


def build(
    cfg: Config,
    *,
    funding_opener: OpenJson | None = None,
    mark_opener: OpenJson | None = None,
) -> dict[str, Any]:
    start = cast(pd.Timestamp, pd.Timestamp(cfg.start))
    end = cast(pd.Timestamp, pd.Timestamp(cfg.end))
    if start != START or end != END:
        raise ValueError("CLBR execution source range is immutable")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")

    market_days: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {
            executor.submit(process_market_day, day, cfg): day
            for day in _days(START_DATE, END_DATE)
        }
        for future in as_completed(futures):
            market_days.append(future.result())
    market_days.sort(key=lambda item: item["date"])
    market = pd.concat([item["frame"] for item in market_days], ignore_index=True)
    expected_market = pd.Series(
        pd.date_range(START, END, freq="5min", inclusive="left"), name="date"
    )
    if not market["date"].equals(expected_market):
        raise ValueError("combined BTCUSDT market grid is incomplete")

    funding_raw, funding_pages = download_funding(cfg, opener=funding_opener)
    marks, mark_pages = download_mark_klines(cfg, opener=mark_opener)
    funding, funding_quality = compose_funding(funding_raw, marks)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for split, (split_start, split_end) in SPLITS.items():
        left = pd.Timestamp(split_start)
        right = pd.Timestamp(split_end)
        market_mask = market["date"].ge(left) & market["date"].lt(right)
        funding_mask = funding["funding_time"].ge(left) & funding[
            "funding_time"
        ].lt(right)
        market_split = cast(pd.DataFrame, market.loc[market_mask].copy())
        funding_split = cast(pd.DataFrame, funding.loc[funding_mask].copy())
        market_path = output_dir / f"{split}_BTCUSDT_5m.csv.gz"
        funding_path = output_dir / f"{split}_BTCUSDT_funding.csv.gz"
        _write_gzip_csv(market_split, market_path)
        _write_gzip_csv(funding_split, funding_path)
        files[split] = {
            "start_inclusive": str(left),
            "end_exclusive": str(right),
            "market": {
                "path": str(market_path),
                "sha256": sha256_file(market_path),
                "rows": int(len(market_split)),
                "first_date": str(market_split["date"].min()),
                "last_date": str(market_split["date"].max()),
            },
            "funding": {
                "path": str(funding_path),
                "sha256": sha256_file(funding_path),
                "rows": int(len(funding_split)),
                "first_time": str(funding_split["funding_time"].min()),
                "last_time": str(funding_split["funding_time"].max()),
            },
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "name": "CLBR-24 official execution sources",
            "outcomes_opened": False,
            "strategy_returns_computed": False,
            "clbr_clocks_loaded": False,
            "timezone": "UTC represented as timezone-naive timestamps",
            "market_symbol": SYMBOL,
            "market_interval": INTERVAL,
            "market_kline_type": "USD-M futures trade-price kline",
            "market_missing_policy": "fail build on any missing or duplicate 5m bar",
            "funding_policy": (
                "exact returned fundingTime; 8h official mark-price open at the "
                "canonical funding boundary"
            ),
        },
        "config": asdict(cfg),
        "official_sources": {
            "market_archive_root": (
                "https://data.binance.vision/data/futures/um/daily/klines/"
                "BTCUSDT/5m/"
            ),
            "funding_endpoint": FUNDING_ENDPOINT,
            "funding_documentation": FUNDING_DOCS,
            "funding_pages": funding_pages,
            "mark_endpoint": MARK_ENDPOINT,
            "mark_documentation": MARK_DOCS,
            "mark_interval": MARK_INTERVAL,
            "mark_pages": mark_pages,
        },
        "market_archives": [_public_market_record(item) for item in market_days],
        "funding_quality": funding_quality,
        "files": files,
    }
    manifest_path = Path(cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--manifest", default=Config.manifest)
    parser.add_argument("--workers", type=int, default=Config.workers)
    args = parser.parse_args()
    result = build(
        Config(
            output_dir=args.output_dir,
            manifest=args.manifest,
            workers=args.workers,
        )
    )
    print(
        json.dumps(
            {
                "outcomes_opened": result["protocol"]["outcomes_opened"],
                "funding_quality": result["funding_quality"],
                "files": result["files"],
                "manifest": args.manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
