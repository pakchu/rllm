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
STEP_MS = 300_000
LIMIT = 1_500
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
        "mark_price": pd.to_numeric([row[1] for row in rows], errors="raise"),
    }).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    frame = frame.loc[(frame["open_time"] >= start) & (frame["open_time"] < end)].reset_index(drop=True)
    timestamps = pd.DatetimeIndex(frame["open_time"])
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise RuntimeError(f"{symbol} AFCH mark-price timestamps are invalid")
    if ((timestamps.astype("int64") // 1_000_000) % STEP_MS != 0).any():
        raise RuntimeError(f"{symbol} AFCH mark-price timestamps are off-grid")
    if not np.isfinite(frame["mark_price"].to_numpy(dtype=float)).all() or (frame["mark_price"] <= 0).any():
        raise RuntimeError(f"{symbol} AFCH mark-price values are invalid")
    return frame


def compose_event_marks(funding: pd.DataFrame, mark_klines: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = funding.copy()
    frame["event_time"] = pd.to_datetime(pd.to_numeric(frame["funding_time"], errors="raise"), unit="ms")
    frame["recorded_mark"] = pd.to_numeric(frame["mark_price"], errors="coerce")
    frame = frame.loc[(frame["event_time"] >= START) & (frame["event_time"] < END)].copy()
    if frame.empty or frame["event_time"].duplicated().any():
        raise RuntimeError("invalid AFCH funding events for mark freeze")
    mark_map = mark_klines.set_index("open_time")["mark_price"]
    frame["kline_open_mark"] = frame["event_time"].map(mark_map)
    if frame["kline_open_mark"].isna().any():
        raise RuntimeError("AFCH funding event lacks exact mark-price kline open")
    recorded = frame["recorded_mark"].notna()
    if recorded.any():
        error_bp = (
            frame.loc[recorded, "kline_open_mark"] / frame.loc[recorded, "recorded_mark"] - 1.0
        ).abs() * 10_000.0
        max_error_bp = float(error_bp.max())
        if max_error_bp > 1e-6:
            raise RuntimeError(f"recorded funding mark mismatch: {max_error_bp:.9f} bp")
    else:
        max_error_bp = 0.0
    frame["exact_mark_price"] = frame["recorded_mark"].fillna(frame["kline_open_mark"])
    frame["mark_source"] = np.where(recorded, "funding_record", "mark_price_kline_open")
    output = frame[["funding_time", "event_time", "exact_mark_price", "mark_source"]].reset_index(drop=True)
    stats = {
        "events": int(len(output)),
        "recorded_mark_events": int(recorded.sum()),
        "backfilled_mark_events": int((~recorded).sum()),
        "maximum_recorded_vs_kline_open_error_bp": max_error_bp,
    }
    return output, stats


def _markdown(result: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {row['symbol']} | {row['events']} | {row['recorded_mark_events']} | "
        f"{row['backfilled_mark_events']} | {row['missing_non_event_5m_mark_rows']} | "
        f"{row['maximum_recorded_vs_kline_open_error_bp']:.9f} |"
        for row in result["records"]
    )
    return f"""# AFCH v1 exact funding marks — 2026-07-17

> Outcome-blind source freeze only. No position return, PnL, CAGR, MDD, or gate was calculated.

Missing 2023 funding-record marks were filled from the open of Binance USD-M
5m mark-price klines at the exact funding timestamp. Where both sources exist,
they must agree within `1e-6 bp`. Official endpoint:
<https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price-Kline-Candlestick-Data>

| Symbol | Events | Funding-record marks | Backfilled exact opens | Missing non-event 5m bars | Max overlap error bp |
|---|---:|---:|---:|---:|---:|
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
        funding = pd.read_csv(funding_path, usecols=["funding_time", "mark_price"])
        klines = download_mark_klines(symbol, START, END, sleep_sec=sleep_sec)
        event_marks, stats = compose_event_marks(funding, klines)
        expected_grid = pd.date_range(START, END - pd.Timedelta(minutes=5), freq="5min")
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
        "protocol_version": "afch_v1_exact_funding_marks_2023_2026-07-17",
        "preregistration_protocol_hash": EXPECTED_PROTOCOL_HASH,
        "source_manifest_hash": EXPECTED_SOURCE_MANIFEST_HASH,
        "source_endpoint": f"{BASE_URL}{ENDPOINT}",
        "official_documentation": (
            "https://developers.binance.com/docs/derivatives/usds-margined-futures/"
            "market-data/rest-api/Mark-Price-Kline-Candlestick-Data"
        ),
        "interval": "5m",
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
