"""Build a verified pre-2024 Bybit BTCUSDT funding/premium panel.

Only public V5 market-data endpoints are used.  The builder is physically
bounded to 2021-2023 and writes deterministic gzip files plus a hash manifest.
No trade outcome or post-2023 row is requested or retained.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


FUNDING_ENDPOINT = "https://api.bybit.com/v5/market/funding/history"
PREMIUM_ENDPOINT = (
    "https://api.bybit.com/v5/market/premium-index-price-kline"
)


@dataclass(frozen=True)
class Config:
    start: str = "2021-01-01"
    end: str = "2024-01-01"
    symbol: str = "BTCUSDT"
    category: str = "linear"
    output_dir: str = "data/bybit_linear_aux_btc_2021_2023"
    manifest: str = "results/bybit_linear_aux_btc_2021_2023_manifest.json"
    funding_chunk_days: int = 60
    premium_chunk_days: int = 30
    request_pause_seconds: float = 0.05
    request_timeout_seconds: float = 30.0
    maximum_attempts: int = 5


def _milliseconds(timestamp: pd.Timestamp) -> int:
    value = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp
    return int(value.timestamp() * 1_000)


def chunk_intervals(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if start >= end:
        raise ValueError("chunk start must be before end")
    if days < 1:
        raise ValueError("chunk days must be positive")
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    delta = pd.Timedelta(days=days)
    while cursor < end:
        next_cursor = min(cursor + delta, end)
        intervals.append((cursor, next_cursor))
        cursor = next_cursor
    return intervals


def _request_json(
    endpoint: str,
    params: dict[str, Any],
    cfg: Config,
) -> dict[str, Any]:
    url = endpoint + "?" + urllib.parse.urlencode(params)
    last_error: Exception | None = None
    for attempt in range(cfg.maximum_attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "rllm-bybit-research/1.0"},
            )
            with urllib.request.urlopen(
                request,
                timeout=cfg.request_timeout_seconds,
            ) as response:
                payload = json.loads(response.read())
            if int(payload.get("retCode", -1)) != 0:
                raise RuntimeError(
                    f"Bybit API error {payload.get('retCode')}: "
                    f"{payload.get('retMsg')}"
                )
            time.sleep(cfg.request_pause_seconds)
            return payload
        except Exception as error:  # pragma: no cover - exercised by live retries
            last_error = error
            if attempt + 1 < cfg.maximum_attempts:
                time.sleep(min(2.0 ** attempt, 8.0))
    raise RuntimeError(f"Bybit request failed: {url}") from last_error


def parse_funding_rows(
    payload: dict[str, Any],
    *,
    symbol: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("result", {}).get("list", []):
        if item.get("symbol") != symbol:
            raise ValueError("Bybit funding response contains another symbol")
        timestamp = int(item["fundingRateTimestamp"])
        rows.append(
            {
                "date": pd.to_datetime(timestamp, unit="ms", utc=True).tz_localize(None),
                "symbol": symbol,
                "funding_rate": float(item["fundingRate"]),
                "funding_time": timestamp,
            }
        )
    return rows


def parse_premium_rows(
    payload: dict[str, Any],
    *,
    symbol: str,
) -> list[dict[str, Any]]:
    result = payload.get("result", {})
    if result.get("symbol") not in (None, symbol):
        raise ValueError("Bybit premium response contains another symbol")
    rows: list[dict[str, Any]] = []
    for item in result.get("list", []):
        if len(item) < 5:
            raise ValueError("Bybit premium candle is incomplete")
        timestamp = int(item[0])
        rows.append(
            {
                "date": pd.to_datetime(timestamp, unit="ms", utc=True).tz_localize(None),
                "symbol": symbol,
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "open_time": timestamp,
            }
        )
    return rows


def _within_bounds(
    rows: Iterable[dict[str, Any]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict[str, Any]]:
    return [row for row in rows if start <= row["date"] < end]


def fetch_funding(cfg: Config) -> tuple[pd.DataFrame, int]:
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    rows: list[dict[str, Any]] = []
    requests = 0
    for chunk_start, chunk_end in chunk_intervals(
        start, end, days=cfg.funding_chunk_days
    ):
        payload = _request_json(
            FUNDING_ENDPOINT,
            {
                "category": cfg.category,
                "symbol": cfg.symbol,
                "startTime": _milliseconds(chunk_start),
                "endTime": _milliseconds(chunk_end) - 1,
                "limit": 200,
            },
            cfg,
        )
        rows.extend(parse_funding_rows(payload, symbol=cfg.symbol))
        requests += 1
    frame = pd.DataFrame(_within_bounds(rows, start, end))
    return _validate_funding(frame, cfg), requests


def fetch_premium(cfg: Config) -> tuple[pd.DataFrame, int]:
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    rows: list[dict[str, Any]] = []
    requests = 0
    for chunk_start, chunk_end in chunk_intervals(
        start, end, days=cfg.premium_chunk_days
    ):
        payload = _request_json(
            PREMIUM_ENDPOINT,
            {
                "category": cfg.category,
                "symbol": cfg.symbol,
                "interval": "60",
                "start": _milliseconds(chunk_start),
                "end": _milliseconds(chunk_end) - 1,
                "limit": 1_000,
            },
            cfg,
        )
        rows.extend(parse_premium_rows(payload, symbol=cfg.symbol))
        requests += 1
    frame = pd.DataFrame(_within_bounds(rows, start, end))
    return _validate_premium(frame, cfg), requests


def _sorted_unique(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Bybit source returned no rows")
    output = frame.sort_values("date").drop_duplicates("date", keep="last")
    if output["date"].duplicated().any() or not output["date"].is_monotonic_increasing:
        raise ValueError("Bybit timestamps are invalid")
    return output.reset_index(drop=True)


def _validate_funding(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    output = _sorted_unique(frame)
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    if output["date"].min() != start:
        raise ValueError("Bybit funding does not start at the requested boundary")
    if output["date"].max() >= end:
        raise ValueError("Bybit funding opened post-boundary data")
    intervals = output["date"].diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta(hours=8)).all():
        raise ValueError("BTCUSDT funding is not a complete 8-hour grid")
    expected_last = end - pd.Timedelta(hours=8)
    if output["date"].max() != expected_last:
        raise ValueError("Bybit funding ends before the requested boundary")
    return output


def _validate_premium(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    output = _sorted_unique(frame)
    expected = pd.date_range(
        cfg.start,
        cfg.end,
        freq="h",
        inclusive="left",
    )
    actual = pd.DatetimeIndex(output["date"])
    if not actual.equals(expected):
        missing = expected.difference(actual)
        extra = actual.difference(expected)
        raise ValueError(
            "Bybit premium is not a complete hourly grid: "
            f"missing={len(missing)} extra={len(extra)}"
        )
    numeric = output[["open", "high", "low", "close"]]
    if not numeric.notna().all().all():
        raise ValueError("Bybit premium contains non-finite candles")
    # The historical endpoint sometimes carries the previous close as `open`
    # while high/low describe only updates within the requested hour.  Enforce
    # the invariant the API actually preserves and retain close as the frozen
    # decision value.
    if not output["high"].ge(output["low"]).all():
        raise ValueError("Bybit premium high is below low")
    return output


def _write_deterministic_gzip(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as stream:
        stream.write(csv_bytes)
    payload = buffer.getvalue()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def build(cfg: Config) -> dict[str, Any]:
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    if end > pd.Timestamp("2024-01-01"):
        raise ValueError("Bybit preselection builder cannot request 2024+ data")
    if cfg.category != "linear" or cfg.symbol != "BTCUSDT":
        raise ValueError("this frozen builder supports linear BTCUSDT only")

    funding, funding_requests = fetch_funding(cfg)
    premium, premium_requests = fetch_premium(cfg)
    output_dir = Path(cfg.output_dir)
    funding_path = output_dir / "BTCUSDT_funding_2021-01-01_2023-12-31.csv.gz"
    premium_path = output_dir / "BTCUSDT_premium_1h_2021-01-01_2023-12-31.csv.gz"
    funding_hash = _write_deterministic_gzip(funding, funding_path)
    premium_hash = _write_deterministic_gzip(premium, premium_path)

    manifest = {
        "protocol": {
            "name": "Bybit linear BTCUSDT pre-2024 auxiliary data",
            "outcomes_opened": False,
            "requested_start_inclusive": str(start),
            "requested_end_exclusive": str(end),
            "post_2023_rows_requested": False,
            "source": "official public Bybit V5 market-data API",
        },
        "config": asdict(cfg),
        "endpoints": {
            "funding": FUNDING_ENDPOINT,
            "premium": PREMIUM_ENDPOINT,
        },
        "request_counts": {
            "funding": funding_requests,
            "premium": premium_requests,
        },
        "files": {
            "funding": {
                "path": str(funding_path),
                "sha256": funding_hash,
                "rows": int(len(funding)),
                "first_date": str(funding["date"].min()),
                "last_date": str(funding["date"].max()),
            },
            "premium": {
                "path": str(premium_path),
                "sha256": premium_hash,
                "rows": int(len(premium)),
                "first_date": str(premium["date"].min()),
                "last_date": str(premium["date"].max()),
            },
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--manifest", default=Config.manifest)
    args = parser.parse_args()
    cfg = Config(output_dir=args.output_dir, manifest=args.manifest)
    result = build(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
