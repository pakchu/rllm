"""Freeze the already-physical 2023-2025 source composition used by AFCH."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from training.export_leave_one_out_residual_exhaustion_sources import sha256_file
from training.preregister_alt_funding_carry_harvest import canonical_hash, protocol


SYMBOLS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
START = pd.Timestamp("2023-01-01 00:00:00")
HANDOFF = pd.Timestamp("2024-01-01 00:00:00")
END = pd.Timestamp("2026-01-01 00:00:00")
EXPECTED_PROTOCOL_HASH = "15a7d0adbace0255e1ea4359e4869154dfb34ad891a2125239340ff70c4e2a09"
LORE_MANIFEST = "results/leave_one_out_residual_exhaustion_v1_source_manifest_2026-07-17.json"
LORE_MANIFEST_HASH = "1c54fddc45fcc516d8ce42741904e018e8d00e646eff40be514273cf10eee7ed"
LORC_MANIFEST = "results/leave_one_out_residual_continuation_v1_source_manifest_2026-07-17.json"
LORC_MANIFEST_HASH = "3ef36c5b77c6c2c48e77ab17af3b285152216b92ff031d6e496dc5255cd34a13"
LORE_DIR = Path("data/binance_um_lore_2023_2024")
LORC_DIR = Path("data/binance_um_lorc_2024_2025")
DEFAULT_OUTPUT = "results/alt_funding_carry_harvest_v1_source_composition_2026-07-17.json"


def _manifest(path: str, expected: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("manifest_hash") != expected:
        raise RuntimeError(f"source manifest hash changed: {path}")
    body = {key: value for key, value in payload.items() if key not in {"manifest_hash", "created_at"}}
    if canonical_hash(body) != expected:
        raise RuntimeError(f"source manifest body changed: {path}")
    return payload


def compose_market_dates(old: pd.Series, recent: pd.Series) -> pd.DatetimeIndex:
    left = pd.to_datetime(old, errors="raise")
    right = pd.to_datetime(recent, errors="raise")
    combined = pd.DatetimeIndex(pd.concat([
        left.loc[(left >= START) & (left < HANDOFF)],
        right.loc[(right >= HANDOFF) & (right < END)],
    ], ignore_index=True))
    if combined.duplicated().any() or not combined.is_monotonic_increasing:
        raise RuntimeError("AFCH composite market grid duplicate/order failure")
    expected = pd.date_range(START, END - pd.Timedelta(minutes=5), freq="5min")
    if not combined.equals(expected):
        raise RuntimeError("AFCH composite market grid is not exact 2023-2025")
    return combined


def run(output: str = DEFAULT_OUTPUT) -> dict[str, Any]:
    if canonical_hash(protocol()) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("AFCH preregistration drifted")
    lore = _manifest(LORE_MANIFEST, LORE_MANIFEST_HASH)
    lorc = _manifest(LORC_MANIFEST, LORC_MANIFEST_HASH)
    lore_records = {str(row["symbol"]): row for row in lore["records"]}
    lorc_records = {str(row["symbol"]): row for row in lorc["records"]}
    records: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        old_market = LORE_DIR / f"{symbol}_5m_2023_2024.csv.gz"
        recent_market = LORC_DIR / f"{symbol}_5m_2024_2025.csv.gz"
        old_funding = LORE_DIR / f"{symbol}_funding_2023_2024.csv.gz"
        recent_funding = LORC_DIR / f"{symbol}_funding_2024_2025.csv.gz"
        actual_hashes = {
            "old_market": sha256_file(old_market),
            "recent_market": sha256_file(recent_market),
            "old_funding": sha256_file(old_funding),
            "recent_funding": sha256_file(recent_funding),
        }
        expected_hashes = {
            "old_market": lore_records[symbol]["output_market_sha256"],
            "recent_market": lorc_records[symbol]["output_market_sha256"],
            "old_funding": lore_records[symbol]["output_funding_sha256"],
            "recent_funding": lorc_records[symbol]["output_funding_sha256"],
        }
        if actual_hashes != expected_hashes:
            raise RuntimeError(f"{symbol} AFCH source hash mismatch")
        old_market_frame = pd.read_csv(old_market, usecols=["date", "tic"])
        recent_market_frame = pd.read_csv(recent_market, usecols=["date", "tic"])
        if not old_market_frame["tic"].astype(str).eq(symbol).all() or not recent_market_frame["tic"].astype(str).eq(symbol).all():
            raise RuntimeError(f"{symbol} AFCH market identity mismatch")
        market_dates = compose_market_dates(old_market_frame["date"], recent_market_frame["date"])
        old_funding_frame = pd.read_csv(old_funding, usecols=["symbol", "funding_time"])
        recent_funding_frame = pd.read_csv(recent_funding, usecols=["symbol", "funding_time"])
        if not old_funding_frame["symbol"].astype(str).eq(symbol).all() or not recent_funding_frame["symbol"].astype(str).eq(symbol).all():
            raise RuntimeError(f"{symbol} AFCH funding identity mismatch")
        old_times = pd.to_datetime(pd.to_numeric(old_funding_frame["funding_time"], errors="raise"), unit="ms")
        recent_times = pd.to_datetime(pd.to_numeric(recent_funding_frame["funding_time"], errors="raise"), unit="ms")
        funding_times = pd.DatetimeIndex(pd.concat([
            old_times.loc[(old_times >= START) & (old_times < HANDOFF)],
            recent_times.loc[(recent_times >= HANDOFF) & (recent_times < END)],
        ], ignore_index=True))
        if funding_times.duplicated().any() or not funding_times.is_monotonic_increasing:
            raise RuntimeError(f"{symbol} AFCH funding composition failure")
        records.append({
            "symbol": symbol,
            "old_market_path": str(old_market),
            "recent_market_path": str(recent_market),
            "old_funding_path": str(old_funding),
            "recent_funding_path": str(recent_funding),
            **{f"{key}_sha256": value for key, value in actual_hashes.items()},
            "composite_market_rows": len(market_dates),
            "composite_market_min": str(market_dates.min()),
            "composite_market_max": str(market_dates.max()),
            "composite_funding_rows": len(funding_times),
            "composite_funding_min": str(funding_times.min()),
            "composite_funding_max": str(funding_times.max()),
        })
    result: dict[str, Any] = {
        "protocol_version": "afch_v1_source_composition_2026-07-17",
        "preregistration_protocol_hash": EXPECTED_PROTOCOL_HASH,
        "post_entry_returns_calculated": False,
        "physical_copy_created": False,
        "composition": {
            "start": str(START),
            "handoff": str(HANDOFF),
            "end_exclusive": str(END),
            "old_source_manifest_hash": LORE_MANIFEST_HASH,
            "recent_source_manifest_hash": LORC_MANIFEST_HASH,
            "future_2026_rows": 0,
        },
        "records": records,
    }
    result["manifest_hash"] = canonical_hash(result)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
