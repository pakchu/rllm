"""Freeze outcome-blind Yahoo daily SMH/XLU source rows for HVSUDR-24."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import urllib.request
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_semiconductor_utilities_disagreement_relay as prereg


SOURCE_DIR = Path("data/high_volatility_semiconductor_utilities_disagreement_relay_sources_2022_2026")
OUTPUT = SOURCE_DIR / "smh_xlu_sessions.csv.gz"
MANIFEST = SOURCE_DIR / "build_manifest.json"
BUILDER = Path("training/build_high_volatility_semiconductor_utilities_disagreement_relay_sources.py")
PREREG_SHA256 = "0aa3096ecb9cb5fffbe12e55cbcd33084e33b95904f7da1cdb64cec87a2abd56"
PERIOD1 = 1661990400  # 2022-09-01 UTC
PERIOD2 = 1785628800  # 2026-08-02 UTC, exclusive
URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?"
    f"period1={PERIOD1}&period2={PERIOD2}&interval=1d&events=history&"
    "includeAdjustedClose=false"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def parse_payload(raw: bytes, symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = json.loads(raw)
    chart = payload.get("chart", {})
    if chart.get("error") is not None:
        raise RuntimeError(f"HVSUDR Yahoo error for {symbol}: {chart['error']}")
    results = chart.get("result") or []
    if len(results) != 1:
        raise RuntimeError(f"HVSUDR Yahoo result count for {symbol}: {len(results)}")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = ((result.get("indicators") or {}).get("quote") or [])
    if not timestamps or len(quotes) != 1:
        raise RuntimeError(f"HVSUDR incomplete Yahoo payload for {symbol}")
    quote = quotes[0]
    opens, closes = quote.get("open"), quote.get("close")
    if opens is None or closes is None or len(opens) != len(timestamps) or len(closes) != len(timestamps):
        raise RuntimeError(f"HVSUDR Yahoo vector mismatch for {symbol}")
    timezone = str((result.get("meta") or {}).get("exchangeTimezoneName") or "")
    if timezone != "America/New_York":
        raise RuntimeError(f"HVSUDR timezone drift for {symbol}: {timezone}")
    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(timezone).tz_localize(None).normalize()
    frame = pd.DataFrame({"session_date": dates, "open": opens, "close": closes})
    frame[["open", "close"]] = frame[["open", "close"]].apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[(frame["session_date"] >= "2022-09-01") & (frame["session_date"] < "2026-08-01")].copy()
    valid = np.isfinite(frame[["open", "close"]]).all(axis=1) & frame[["open", "close"]].gt(0).all(axis=1)
    if not valid.all() or frame["session_date"].duplicated().any() or not frame["session_date"].is_monotonic_increasing:
        raise RuntimeError(f"HVSUDR invalid daily rows for {symbol}")
    metadata = {
        "symbol": symbol,
        "exchange_timezone": timezone,
        "rows": int(len(frame)),
        "first_session": str(frame["session_date"].iloc[0].date()),
        "last_session": str(frame["session_date"].iloc[-1].date()),
        "adjusted_close_requested": False,
        "adjusted_close_read": False,
    }
    return frame, metadata


def deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False, lineterminator="\n").encode()
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(payload)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 rllm-research"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA256:
        raise RuntimeError("HVSUDR preregistration hash drift")
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for symbol in ("SMH", "XLU"):
        raw_path = SOURCE_DIR / f"{symbol.lower()}_yahoo_chart.json"
        raw_path.write_bytes(fetch(URL.format(symbol=symbol)))
        frame, metadata = parse_payload(raw_path.read_bytes(), symbol)
        frames[symbol] = frame.rename(columns={"open": f"{symbol.lower()}_open", "close": f"{symbol.lower()}_close"})
        inputs[symbol] = {"path": str(raw_path), "sha256": sha256(raw_path), "url": URL.format(symbol=symbol), **metadata}
    panel = frames["SMH"].merge(frames["XLU"], on="session_date", how="inner", validate="one_to_one")
    if panel.empty or panel["session_date"].iloc[-1] != pd.Timestamp("2026-07-31"):
        raise RuntimeError("HVSUDR shared-session coverage drift")
    deterministic_gzip_csv(panel, OUTPUT)
    core = {
        "protocol_version": "hvsudr_24_sources_v1",
        "preregistration_sha256": PREREG_SHA256,
        "completed_preentry_sources_opened": True,
        "candidate_incidence_opened": False,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "inputs": inputs,
        "output": {"path": str(OUTPUT), "sha256": sha256(OUTPUT), "rows": int(len(panel))},
        "builder": {"path": str(BUILDER), "sha256": sha256(BUILDER)},
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    MANIFEST.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    built = run()
    print(json.dumps(built["output"], sort_keys=True))
