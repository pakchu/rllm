"""Freeze exact AFCH funding-event marks before any strategy PnL is opened."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from training.compose_alt_funding_carry_sources import LORE_DIR, SYMBOLS
from training.export_leave_one_out_residual_exhaustion_sources import (
    deterministic_csv_gz,
    sha256_file,
)
from training.preregister_alt_funding_carry_harvest import canonical_hash, protocol


START = pd.Timestamp("2023-01-01 00:00:00")
END = pd.Timestamp("2024-01-01 00:00:00")
MARK_START = START - pd.Timedelta(minutes=5)
STEP_MS = 300_000
LIMIT = 1_500
MAX_PROXY_FUNDING_CASH_ERROR_BP_NOTIONAL = 0.1
BASE_URL = "https://fapi.binance.com"
ENDPOINT = "/fapi/v1/markPriceKlines"
EXPECTED_PROTOCOL_HASH = "15a7d0adbace0255e1ea4359e4869154dfb34ad891a2125239340ff70c4e2a09"
SOURCE_MANIFEST = "results/leave_one_out_residual_exhaustion_v1_source_manifest_2026-07-17.json"
EXPECTED_SOURCE_MANIFEST_HASH = "1c54fddc45fcc516d8ce42741904e018e8d00e646eff40be514273cf10eee7ed"
DEFAULT_OUTPUT_DIR = "data/binance_um_afch_funding_marks_2023"
DEFAULT_MANIFEST = "results/alt_funding_carry_harvest_v1_funding_marks_2023_2026-07-17.json"
DEFAULT_DOCS = "docs/alt-funding-carry-harvest-v1-funding-marks-2023-2026-07-17.md"
FREEZER_PATH = "training/freeze_alt_funding_carry_marks.py"
TEST_PATH = "tests/test_freeze_alt_funding_carry_marks.py"


def _request_json(path: str, params: dict[str, Any]) -> Any:
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "rllm-research/1.0"})
    error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - exercised only on network failure
            error = exc
            time.sleep(2**attempt)
    assert error is not None
    raise error


def _git_attestation() -> dict[str, str]:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if status.strip():
        raise RuntimeError("repository must be clean before freezing AFCH marks")
    for path in (FREEZER_PATH, TEST_PATH):
        subprocess.check_call(["git", "ls-files", "--error-unmatch", path], stdout=subprocess.DEVNULL)
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "freezer_sha256": sha256_file(Path(FREEZER_PATH)),
        "test_sha256": sha256_file(Path(TEST_PATH)),
    }


def _manifest() -> dict[str, Any]:
    payload = json.loads(Path(SOURCE_MANIFEST).read_text())
    if payload.get("manifest_hash") != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("AFCH mark source manifest hash changed")
    body = {key: value for key, value in payload.items() if key not in {"manifest_hash", "created_at"}}
    if canonical_hash(body) != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("AFCH mark source manifest body changed")
    return payload


def download_mark_klines(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    request_json: Callable[[str, dict[str, Any]], Any] = _request_json,
    sleep_sec: float = 0.12,
) -> pd.DataFrame:
    start_ms = int(start.tz_localize("UTC").timestamp() * 1_000)
    end_ms = int(end.tz_localize("UTC").timestamp() * 1_000)
    rows: list[list[Any]] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = request_json(ENDPOINT, {
            "symbol": symbol,
            "interval": "5m",
            "startTime": cursor,
            "endTime": end_ms - 1,
            "limit": LIMIT,
        })
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + STEP_MS
        if next_cursor <= cursor:
            raise RuntimeError(f"{symbol} AFCH mark pagination stalled")
        cursor = next_cursor
        if cursor < end_ms and sleep_sec:
            time.sleep(sleep_sec)
    if not rows:
        raise RuntimeError(f"{symbol} empty AFCH mark-price history")
    frame = pd.DataFrame({
        "open_time": pd.to_datetime([int(row[0]) for row in rows], unit="ms"),
        "open": pd.to_numeric([row[1] for row in rows], errors="raise"),
        "high": pd.to_numeric([row[2] for row in rows], errors="raise"),
        "low": pd.to_numeric([row[3] for row in rows], errors="raise"),
        "close": pd.to_numeric([row[4] for row in rows], errors="raise"),
        "close_time": pd.to_datetime([int(row[6]) for row in rows], unit="ms"),
    }).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    frame = frame.loc[(frame["open_time"] >= start) & (frame["open_time"] < end)].reset_index(drop=True)
    timestamps = pd.DatetimeIndex(frame["open_time"])
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise RuntimeError(f"{symbol} AFCH mark-price timestamps are invalid")
    if ((timestamps.astype("int64") // 1_000_000) % STEP_MS != 0).any():
        raise RuntimeError(f"{symbol} AFCH mark-price timestamps are off-grid")
    expected_close = timestamps + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1)
    if not pd.DatetimeIndex(frame["close_time"]).equals(expected_close):
        raise RuntimeError(f"{symbol} AFCH mark-price close times are invalid")
    prices = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise RuntimeError(f"{symbol} AFCH mark-price values are invalid")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise RuntimeError(f"{symbol} AFCH mark-price high is invalid")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise RuntimeError(f"{symbol} AFCH mark-price low is invalid")
    return frame


def compose_event_marks(funding: pd.DataFrame, mark_klines: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = funding.copy()
    frame["event_time"] = pd.to_datetime(pd.to_numeric(frame["funding_time"], errors="raise"), unit="ms")
    frame["mark_open_time"] = frame["event_time"].dt.floor("5min")
    frame["recorded_mark"] = pd.to_numeric(frame["mark_price"], errors="coerce")
    frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="raise")
    frame = frame.loc[(frame["event_time"] >= START) & (frame["event_time"] < END)].copy()
    if frame.empty or frame["event_time"].duplicated().any():
        raise RuntimeError("invalid AFCH funding events for mark freeze")
    if not np.isfinite(frame["funding_rate"].to_numpy(dtype=float)).all():
        raise RuntimeError("invalid AFCH funding rates for mark freeze")
    current_open = mark_klines.set_index("open_time")["open"]
    prior_close = pd.Series(
        mark_klines["close"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(mark_klines["open_time"]) + pd.Timedelta(minutes=5),
    )
    frame["current_interval_open"] = frame["mark_open_time"].map(current_open)
    frame["prior_completed_mark"] = frame["mark_open_time"].map(prior_close)
    if frame[["current_interval_open", "prior_completed_mark"]].isna().any().any():
        raise RuntimeError("AFCH funding event lacks causal adjacent mark-price bars")
    recorded = frame["recorded_mark"].notna()
    if recorded.any():
        proxy_error_bp = (
            frame.loc[recorded, "prior_completed_mark"] / frame.loc[recorded, "recorded_mark"] - 1.0
        ).abs() * 10_000.0
        open_error_bp = (
            frame.loc[recorded, "current_interval_open"] / frame.loc[recorded, "recorded_mark"] - 1.0
        ).abs() * 10_000.0
        max_proxy_error_bp = float(proxy_error_bp.max())
        max_open_error_bp = float(open_error_bp.max())
        funding_cash_error_bp = proxy_error_bp * frame.loc[recorded, "funding_rate"].abs()
        max_funding_cash_error_bp = float(funding_cash_error_bp.max())
        if max_funding_cash_error_bp > MAX_PROXY_FUNDING_CASH_ERROR_BP_NOTIONAL:
            raise RuntimeError(
                f"causal funding cash proxy uncertainty: {max_funding_cash_error_bp:.9f} bp"
            )
    else:
        max_proxy_error_bp = max_open_error_bp = max_funding_cash_error_bp = 0.0
    frame["causal_mark_price"] = frame["recorded_mark"].fillna(frame["prior_completed_mark"])
    frame["mark_source"] = np.where(recorded, "funding_record", "prior_completed_mark_close")
    output = frame[["funding_time", "event_time", "causal_mark_price", "mark_source"]].reset_index(drop=True)
    stats = {
        "events": int(len(output)),
        "recorded_mark_events": int(recorded.sum()),
        "backfilled_mark_events": int((~recorded).sum()),
        "maximum_recorded_vs_prior_close_error_bp": max_proxy_error_bp,
        "maximum_recorded_vs_current_open_error_bp": max_open_error_bp,
        "maximum_proxy_funding_cash_error_bp_notional": max_funding_cash_error_bp,
    }
    return output, stats


def _markdown(result: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {row['symbol']} | {row['events']} | {row['recorded_mark_events']} | "
        f"{row['backfilled_mark_events']} | {row['missing_non_event_5m_mark_rows']} | "
        f"{row['maximum_recorded_vs_prior_close_error_bp']:.6f} | "
        f"{row['maximum_proxy_funding_cash_error_bp_notional']:.9f} |"
        for row in result["records"]
    )
    return f"""# AFCH v1 causal funding marks — 2026-07-17

> Outcome-blind source freeze only. No position return, PnL, CAGR, MDD, or gate was calculated.

Missing 2023 funding-record marks use the close of the last fully completed
Binance USD-M 5m mark-price interval before the funding event. This is a
causal proxy, not an exact settlement mark. On rows where the exact recorded
mark exists, mark error times absolute funding rate must imply no more than
`{MAX_PROXY_FUNDING_CASH_ERROR_BP_NOTIONAL:.1f} bp/notional` funding-cash error;
exact/proxy counts and worst overlap errors are frozen below. Official endpoint:
<https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price-Kline-Candlestick-Data>

| Symbol | Events | Exact recorded marks | Causal proxy marks | Missing non-event 5m bars | Max mark error bp | Max funding-cash error bp/notional |
|---|---:|---:|---:|---:|---:|---:|
{rows}

Manifest hash: `{result['manifest_hash']}`
"""


def run(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    manifest_path: str = DEFAULT_MANIFEST,
    docs_path: str = DEFAULT_DOCS,
    *,
    sleep_sec: float = 0.12,
) -> dict[str, Any]:
    if canonical_hash(protocol()) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("AFCH preregistration drifted before mark freeze")
    source = _manifest()
    attestation = _git_attestation()
    source_records = {str(row["symbol"]): row for row in source["records"]}
    target = Path(output_dir)
    records: list[dict[str, Any]] = []
    for symbol in sorted(SYMBOLS):
        funding_path = LORE_DIR / f"{symbol}_funding_2023_2024.csv.gz"
        if sha256_file(funding_path) != source_records[symbol]["output_funding_sha256"]:
            raise RuntimeError(f"{symbol} AFCH funding source changed before mark freeze")
        funding = pd.read_csv(funding_path, usecols=["funding_time", "funding_rate", "mark_price"])
        klines = download_mark_klines(symbol, MARK_START, END, sleep_sec=sleep_sec)
        event_marks, stats = compose_event_marks(funding, klines)
        expected_grid = pd.date_range(MARK_START, END - pd.Timedelta(minutes=5), freq="5min")
        missing_grid = expected_grid.difference(pd.DatetimeIndex(klines["open_time"]))
        output = target / f"{symbol}_funding_marks_2023.csv.gz"
        deterministic_csv_gz(event_marks, output)
        records.append({
            "symbol": symbol,
            **stats,
            "downloaded_5m_mark_rows": int(len(klines)),
            "missing_non_event_5m_mark_rows": int(len(missing_grid)),
            "downloaded_5m_mark_grid_coverage": float(len(klines) / len(expected_grid)),
            "output_path": str(output),
            "output_sha256": sha256_file(output),
        })
    result: dict[str, Any] = {
        "protocol_version": "afch_v1_causal_funding_marks_2023_2026-07-17",
        "preregistration_protocol_hash": EXPECTED_PROTOCOL_HASH,
        "source_manifest_hash": EXPECTED_SOURCE_MANIFEST_HASH,
        "source_endpoint": f"{BASE_URL}{ENDPOINT}",
        "official_documentation": (
            "https://developers.binance.com/docs/derivatives/usds-margined-futures/"
            "market-data/rest-api/Mark-Price-Kline-Candlestick-Data"
        ),
        "interval": "5m",
        "missing_mark_policy": "last fully completed mark-price 5m close before funding event",
        "maximum_proxy_funding_cash_error_bp_notional": MAX_PROXY_FUNDING_CASH_ERROR_BP_NOTIONAL,
        "period": [str(START), str(END)],
        "outcomes_opened": False,
        "pre_download_attestation": attestation,
        "records": records,
    }
    result["manifest_hash"] = canonical_hash(result)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    Path(docs_path).parent.mkdir(parents=True, exist_ok=True)
    Path(docs_path).write_text(_markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--docs", default=DEFAULT_DOCS)
    parser.add_argument("--sleep-sec", type=float, default=0.12)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.manifest, args.docs, sleep_sec=args.sleep_sec), indent=2))


if __name__ == "__main__":
    main()
