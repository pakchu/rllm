"""Freeze outcome-blind Yahoo daily KRE source rows for HVKRECLV-24."""
from __future__ import annotations
import gzip, hashlib, json, urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_regional_bank_close_location_relay as prereg

SOURCE_DIR = Path("data/high_volatility_regional_bank_close_location_relay_sources_2022_2026")
OUTPUT = SOURCE_DIR / "kre_sessions.csv.gz"
MANIFEST = SOURCE_DIR / "build_manifest.json"
BUILDER = Path("training/build_high_volatility_regional_bank_close_location_relay_sources.py")
PREREG_SHA256 = "a86e278fe212d64d58a45af97c5a3907dc77587d93bfaa4158de6bc0d334b4df"
PERIOD1 = 1661990400
PERIOD2 = 1785628800
URL = "https://query1.finance.yahoo.com/v8/finance/chart/KRE?" + f"period1={PERIOD1}&period2={PERIOD2}&interval=1d&events=history&includeAdjustedClose=false"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def parse_payload(raw: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = json.loads(raw)
    chart = payload.get("chart", {})
    if chart.get("error") is not None:
        raise RuntimeError(f"HVKRECLV Yahoo error: {chart['error']}")
    results = chart.get("result") or []
    if len(results) != 1:
        raise RuntimeError(f"HVKRECLV Yahoo result count: {len(results)}")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = ((result.get("indicators") or {}).get("quote") or [])
    if not timestamps or len(quotes) != 1:
        raise RuntimeError("HVKRECLV incomplete Yahoo payload")
    quote = quotes[0]
    vectors = {column: quote.get(column) for column in ("open", "high", "low", "close")}
    if any(values is None or len(values) != len(timestamps) for values in vectors.values()):
        raise RuntimeError("HVKRECLV Yahoo vector mismatch")
    timezone = str((result.get("meta") or {}).get("exchangeTimezoneName") or "")
    if timezone != "America/New_York":
        raise RuntimeError(f"HVKRECLV timezone drift: {timezone}")
    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(timezone).tz_localize(None).normalize()
    frame = pd.DataFrame({"session_date": dates, **vectors})
    columns = ["open", "high", "low", "close"]
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[(frame.session_date >= "2022-09-01") & (frame.session_date < "2026-08-01")].copy()
    valid = np.isfinite(frame[columns]).all(axis=1) & frame[columns].gt(0).all(axis=1) & frame.high.ge(frame[["open", "close"]].max(axis=1)) & frame.low.le(frame[["open", "close"]].min(axis=1)) & frame.high.gt(frame.low)
    if not valid.all() or frame.session_date.duplicated().any() or not frame.session_date.is_monotonic_increasing:
        raise RuntimeError("HVKRECLV invalid daily rows")
    metadata = {"symbol": "KRE", "exchange_timezone": timezone, "rows": int(len(frame)), "first_session": str(frame.session_date.iloc[0].date()), "last_session": str(frame.session_date.iloc[-1].date()), "adjusted_close_requested": False, "adjusted_close_read": False}
    return frame, metadata


def deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False, lineterminator="\n").encode()
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(payload)


def fetch() -> bytes:
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 rllm-research"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA256:
        raise RuntimeError("HVKRECLV preregistration hash drift")
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = SOURCE_DIR / "kre_yahoo_chart.json"
    raw_path.write_bytes(fetch())
    frame, metadata = parse_payload(raw_path.read_bytes())
    if frame.empty or frame.session_date.iloc[-1] != pd.Timestamp("2026-07-31"):
        raise RuntimeError("HVKRECLV coverage drift")
    deterministic_gzip_csv(frame.rename(columns={name: f"kre_{name}" for name in ("open", "high", "low", "close")}), OUTPUT)
    core = {
        "protocol_version": "hvkreclv_24_sources_v1", "preregistration_sha256": PREREG_SHA256,
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": False,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "input": {"path": str(raw_path), "sha256": sha256(raw_path), "url": URL, **metadata},
        "output": {"path": str(OUTPUT), "sha256": sha256(OUTPUT), "rows": int(len(frame))},
        "builder": {"path": str(BUILDER), "sha256": sha256(BUILDER)},
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    MANIFEST.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    built = run()
    print(json.dumps(built["output"], sort_keys=True))
