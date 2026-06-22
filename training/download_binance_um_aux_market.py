"""Download Binance USD-M auxiliary public market data.

Sources checked against official Binance Open Platform docs (2026-06-23):
- Funding history: GET /fapi/v1/fundingRate, limit max 1000, ascending by time.
- Premium index klines: GET /fapi/v1/premiumIndexKlines, limit max 1500.

The downloader is intentionally public-data only: no API keys, no account endpoints.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

BASE_URL = "https://fapi.binance.com"
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


@dataclass(frozen=True)
class Cfg:
    symbols: list[str]
    output_dir: str
    start: str = "2023-01-01"
    end: str = "2026-06-01"
    kinds: list[str] | None = None
    premium_interval: str = "1h"
    sleep_sec: float = 0.05


def ms(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def get_json(path: str, params: dict[str, Any]) -> Any:
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "rllm-research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_funding(symbol: str, start_ms: int, end_ms: int, sleep_sec: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cur = start_ms
    while cur <= end_ms:
        batch = get_json("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        last = int(batch[-1]["fundingTime"])
        nxt = last + 1
        if nxt <= cur:
            break
        cur = nxt
        if last >= end_ms:
            break
        time.sleep(sleep_sec)
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "funding_rate", "funding_time", "mark_price"])
    df = pd.DataFrame(rows).drop_duplicates("fundingTime").sort_values("fundingTime")
    return pd.DataFrame({
        "date": pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True).dt.tz_convert(None),
        "symbol": df["symbol"].astype(str),
        "funding_rate": df["fundingRate"].astype(float),
        "funding_time": df["fundingTime"].astype("int64"),
        "mark_price": pd.to_numeric(df.get("markPrice", pd.Series([None] * len(df))), errors="coerce"),
    })


def download_premium_klines(symbol: str, interval: str, start_ms: int, end_ms: int, sleep_sec: float) -> pd.DataFrame:
    if interval not in INTERVAL_MS:
        raise ValueError(f"unsupported interval {interval}; expected one of {sorted(INTERVAL_MS)}")
    rows: list[list[Any]] = []
    cur = start_ms
    step = INTERVAL_MS[interval]
    while cur <= end_ms:
        batch = get_json(
            "/fapi/v1/premiumIndexKlines",
            {"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": 1500},
        )
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        nxt = last_open + step
        if nxt <= cur:
            break
        cur = nxt
        if last_open >= end_ms:
            break
        time.sleep(sleep_sec)
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "close_time"])
    df = pd.DataFrame(rows).drop_duplicates(0).sort_values(0)
    return pd.DataFrame({
        "date": pd.to_datetime(df[0].astype("int64"), unit="ms", utc=True).dt.tz_convert(None),
        "symbol": symbol,
        "open": df[1].astype(float),
        "high": df[2].astype(float),
        "low": df[3].astype(float),
        "close": df[4].astype(float),
        "close_time": df[6].astype("int64"),
    })


def run(cfg: Cfg) -> dict[str, Any]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_ms = ms(cfg.start)
    end_ms = ms(cfg.end)
    kinds = cfg.kinds or ["funding", "premium"]
    summary: dict[str, Any] = {
        "source": BASE_URL,
        "start": cfg.start,
        "end": cfg.end,
        "symbols": cfg.symbols,
        "kinds": kinds,
        "premium_interval": cfg.premium_interval,
        "files": [],
        "official_docs": {
            "funding_history": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History",
            "premium_index_klines": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Premium-Index-Kline-Data",
        },
    }
    for symbol in cfg.symbols:
        if "funding" in kinds:
            df = download_funding(symbol, start_ms, end_ms, cfg.sleep_sec)
            path = out_dir / f"{symbol}_funding_{cfg.start}_{cfg.end}.csv.gz"
            df.to_csv(path, index=False, compression="gzip")
            summary["files"].append({"kind": "funding", "symbol": symbol, "path": str(path), "rows": int(len(df)), "start": str(df["date"].min()) if len(df) else None, "end": str(df["date"].max()) if len(df) else None})
        if "premium" in kinds:
            df = download_premium_klines(symbol, cfg.premium_interval, start_ms, end_ms, cfg.sleep_sec)
            path = out_dir / f"{symbol}_premium_{cfg.premium_interval}_{cfg.start}_{cfg.end}.csv.gz"
            df.to_csv(path, index=False, compression="gzip")
            summary["files"].append({"kind": "premium", "symbol": symbol, "path": str(path), "rows": int(len(df)), "start": str(df["date"].min()) if len(df) else None, "end": str(df["date"].max()) if len(df) else None})
    sp = out_dir / f"download_summary_aux_{cfg.start}_{cfg.end}.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> Cfg:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start", default=Cfg.start)
    p.add_argument("--end", default=Cfg.end)
    p.add_argument("--kinds", nargs="+", choices=["funding", "premium"], default=None)
    p.add_argument("--premium-interval", default=Cfg.premium_interval)
    p.add_argument("--sleep-sec", type=float, default=Cfg.sleep_sec)
    return Cfg(**vars(p.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
