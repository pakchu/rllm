"""Download public Deribit BTC volatility-index candles with pagination.

Official endpoint:
https://docs.deribit.com/api-reference/market-data/public-get_volatility_index_data

The API returns at most 1,000 recent rows plus a continuation timestamp.  Candle
OHLC is not available to a strategy until ``timestamp + resolution``.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ENDPOINT = "https://www.deribit.com/api/v2/public/get_volatility_index_data"


@dataclass(frozen=True)
class DeribitVolatilityConfig:
    output_csv: str
    start: str = "2020-09-01"
    end: str = "2026-06-02"
    currency: str = "BTC"
    resolution: int = 3600
    timeout_sec: float = 30.0


def _timestamp_ms(value: str) -> int:
    return int(pd.Timestamp(value, tz="UTC").timestamp() * 1000)


def get_json(params: dict[str, Any], *, timeout_sec: float) -> dict[str, Any]:
    url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "rllm-research/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8"))


def download(
    cfg: DeribitVolatilityConfig,
    *,
    fetch: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    start_ms = _timestamp_ms(cfg.start)
    end_ms = _timestamp_ms(cfg.end)
    fetch = fetch or (lambda params: get_json(params, timeout_sec=cfg.timeout_sec))
    rows: list[list[Any]] = []
    cursor = end_ms
    seen_cursors: set[int] = set()
    while cursor >= start_ms:
        payload = fetch(
            {
                "currency": cfg.currency,
                "start_timestamp": start_ms,
                "end_timestamp": cursor,
                "resolution": cfg.resolution,
            }
        )
        if "error" in payload:
            raise RuntimeError(f"Deribit error: {payload['error']}")
        result = payload.get("result", {})
        batch = result.get("data", [])
        if not batch:
            break
        rows.extend(batch)
        continuation = result.get("continuation")
        if continuation is None:
            break
        continuation = int(continuation)
        if continuation in seen_cursors or continuation >= cursor:
            raise RuntimeError(f"non-decreasing Deribit continuation: {continuation} >= {cursor}")
        seen_cursors.add(continuation)
        cursor = continuation
    if not rows:
        return pd.DataFrame(columns=["date", "close_time", "open", "high", "low", "close"])
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    for column in ("timestamp", "open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    frame = frame[(frame["timestamp"] >= start_ms) & (frame["timestamp"] <= end_ms)].reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["timestamp"].astype("int64"), unit="ms", utc=True).dt.tz_convert(None)
    frame["close_time"] = frame["date"] + pd.to_timedelta(cfg.resolution, unit="s")
    return frame[["date", "close_time", "open", "high", "low", "close"]]


def run(cfg: DeribitVolatilityConfig) -> dict[str, Any]:
    frame = download(cfg)
    output = Path(cfg.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, compression="infer")
    report = {
        "config": asdict(cfg),
        "rows": int(len(frame)),
        "start": str(frame["date"].min()) if len(frame) else None,
        "end": str(frame["date"].max()) if len(frame) else None,
        "output_csv": str(output),
        "availability": "candle values join on close_time, never date/open time",
        "official_docs": "https://docs.deribit.com/api-reference/market-data/public-get_volatility_index_data",
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> DeribitVolatilityConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--start", default=DeribitVolatilityConfig.start)
    parser.add_argument("--end", default=DeribitVolatilityConfig.end)
    parser.add_argument("--currency", default=DeribitVolatilityConfig.currency)
    parser.add_argument("--resolution", type=int, default=DeribitVolatilityConfig.resolution)
    parser.add_argument("--timeout-sec", type=float, default=DeribitVolatilityConfig.timeout_sec)
    return DeribitVolatilityConfig(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
