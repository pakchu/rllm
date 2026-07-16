"""Physically freeze the 2024 warmup plus calendar-2025 LORC source prefix."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from training.export_leave_one_out_residual_exhaustion_sources import (
    deterministic_csv_gz,
    resolve,
    sha256_file,
    validate_funding,
    validate_market,
)
from training.preregister_leave_one_out_residual_continuation import canonical_hash, protocol


SYMBOLS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
START = pd.Timestamp("2024-01-01 00:00:00")
HOLDOUT_START = pd.Timestamp("2025-01-01 00:00:00")
END = pd.Timestamp("2026-01-01 00:00:00")
EXPECTED_PROTOCOL_HASH = "151f7905b64a2eca471f56edf377a7b141f9ad8cb58fb7646c1f0b96a4a344ee"
DEFAULT_MARKET_DIR = "data/binance_um_pool_5m_2023_2026"
DEFAULT_AUX_DIR = "data/binance_um_aux_2023_2026"
DEFAULT_OUTPUT_DIR = "data/binance_um_lorc_2024_2025"
DEFAULT_MANIFEST = "results/leave_one_out_residual_continuation_v1_source_manifest_2026-07-17.json"


def run(
    market_dir: str = DEFAULT_MARKET_DIR,
    aux_dir: str = DEFAULT_AUX_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    manifest_path: str = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    if canonical_hash(protocol()) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("LORC preregistration protocol drifted")
    target = Path(output_dir)
    records: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        market_input = resolve(Path(market_dir) / f"{symbol}_5m_2023-01_2026-05.csv.gz")
        funding_input = resolve(Path(aux_dir) / f"{symbol}_funding_2023-01-01_2026-06-01.csv.gz")
        market = validate_market(pd.read_csv(market_input), symbol, START, END)
        funding = validate_funding(pd.read_csv(funding_input), symbol, START, END)
        market_output = target / f"{symbol}_5m_2024_2025.csv.gz"
        funding_output = target / f"{symbol}_funding_2024_2025.csv.gz"
        deterministic_csv_gz(market, market_output)
        deterministic_csv_gz(funding, funding_output)
        reread_market = validate_market(pd.read_csv(market_output), symbol, START, END)
        reread_funding = validate_funding(pd.read_csv(funding_output), symbol, START, END)
        records.append({
            "symbol": symbol,
            "input_market": str(market_input),
            "input_market_sha256": sha256_file(market_input),
            "output_market": str(market_output),
            "output_market_sha256": sha256_file(market_output),
            "market_rows": len(reread_market),
            "zero_quote_volume_bars": int((reread_market["quote_asset_volume"] == 0).sum()),
            "market_min": str(reread_market["date"].min()),
            "market_max": str(reread_market["date"].max()),
            "holdout_market_rows": int((reread_market["date"] >= HOLDOUT_START).sum()),
            "input_funding": str(funding_input),
            "input_funding_sha256": sha256_file(funding_input),
            "output_funding": str(funding_output),
            "output_funding_sha256": sha256_file(funding_output),
            "funding_rows": len(reread_funding),
            "funding_min": str(pd.to_datetime(reread_funding["funding_time"], unit="ms").min()),
            "funding_max": str(pd.to_datetime(reread_funding["funding_time"], unit="ms").max()),
            "holdout_funding_rows": int((pd.to_datetime(reread_funding["funding_time"], unit="ms") >= HOLDOUT_START).sum()),
        })
    manifest: dict[str, Any] = {
        "protocol_version": "lorc_v1_2025_source_freeze_2026-07-17",
        "preregistration_protocol_hash": EXPECTED_PROTOCOL_HASH,
        "post_entry_returns_calculated": False,
        "physical_prefix": {"start": str(START), "end_exclusive": str(END)},
        "warmup": {"start": str(START), "end_exclusive": str(HOLDOUT_START)},
        "holdout": {"start": str(HOLDOUT_START), "end_exclusive": str(END)},
        "future_2026_plus_rows_written": 0,
        "symbols": list(SYMBOLS),
        "market_contract": "exact complete 5m grid; no fill; valid OHLC/taker identity",
        "funding_contract": "exact reported event rows; no fill or synthetic rate",
        "records": records,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    out = Path(manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-dir", default=DEFAULT_MARKET_DIR)
    parser.add_argument("--aux-dir", default=DEFAULT_AUX_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(run(args.market_dir, args.aux_dir, args.output_dir, args.manifest), indent=2))


if __name__ == "__main__":
    main()
