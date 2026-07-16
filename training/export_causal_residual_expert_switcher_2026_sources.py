"""Freeze CRES-1 historical seed and exact 2025-2026H1 physical sources.

The exporter never calculates a 2026 strategy return.  Market files contain
the immutable OHLC/flow prefix required by the later frozen evaluator; the
separate training seed contains only already-opened 2023-2025 expert outcomes.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.develop_causal_residual_expert_switcher_pre2026 import (
    CURRENT_FEATURES,
    build_development_events,
)
from training.export_leave_one_out_residual_exhaustion_sources import (
    deterministic_csv_gz,
    resolve,
    sha256_file,
    validate_funding,
    validate_market,
)
from training.preregister_causal_residual_expert_switcher_2026 import (
    canonical_hash,
    protocol,
)


SYMBOLS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
START = pd.Timestamp("2025-01-01 00:00:00")
CONFIRMATION_START = pd.Timestamp("2026-01-01 00:00:00")
END = pd.Timestamp("2026-07-01 00:00:00")
EXPECTED_PROTOCOL_HASH = "101d08bebde054919ae17c2bfbfaa5e953b983179e8cc7fafc8031875aaaea24"

DEFAULT_BASE_MARKET_DIR = "data/binance_um_pool_5m_2023_2026"
DEFAULT_JUNE_MARKET_DIR = "data/binance_um_pool_5m_2026_06"
DEFAULT_FUNDING_DIR = "data/binance_um_aux_cres_2025_2026h1"
DEFAULT_OUTPUT_DIR = "data/binance_um_cres_2025_2026h1"
DEFAULT_SEED = "data/cres_v1_training_seed_2023_2025.csv.gz"
DEFAULT_MANIFEST = "results/causal_residual_expert_switcher_2026_source_manifest_2026-07-17.json"

SEED_COLUMNS = (
    "signal_time",
    "entry_time",
    "exit_time",
    *CURRENT_FEATURES,
    "range_risk",
    "continuation_net_log_return",
    "reversion_net_log_return",
    "edge",
)


def _market_inputs(base_dir: str, june_dir: str, symbol: str) -> tuple[Path, Path]:
    base = resolve(Path(base_dir) / f"{symbol}_5m_2023-01_2026-05.csv.gz")
    june = resolve(Path(june_dir) / f"{symbol}_5m_2026-06_2026-06.csv.gz")
    return base, june


def combine_market_prefix(base: pd.DataFrame, june: pd.DataFrame, symbol: str) -> pd.DataFrame:
    combined = pd.concat([base, june], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="raise")
    combined = combined.sort_values("date").drop_duplicates("date", keep="last")
    return validate_market(combined, symbol, START, END)


def build_training_seed(path: str = DEFAULT_SEED) -> dict[str, Any]:
    events, _, source_hashes = build_development_events()
    seed = events.loc[:, SEED_COLUMNS].copy()
    for column in ("signal_time", "entry_time", "exit_time"):
        seed[column] = pd.to_datetime(seed[column], errors="raise")
    if seed.empty or seed["signal_time"].duplicated().any():
        raise RuntimeError("CRES historical seed is empty or duplicated")
    if not (seed["exit_time"] < CONFIRMATION_START).all():
        raise RuntimeError("CRES historical seed crossed into confirmation outcomes")
    if not seed["signal_time"].is_monotonic_increasing:
        raise RuntimeError("CRES historical seed is not ordered")
    output = Path(path)
    deterministic_csv_gz(seed, output)
    reread = pd.read_csv(output, parse_dates=["signal_time", "entry_time", "exit_time"])
    if len(reread) != len(seed) or not (reread["exit_time"] < CONFIRMATION_START).all():
        raise RuntimeError("CRES historical seed roundtrip failed")
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "rows": int(len(seed)),
        "signal_min": str(seed["signal_time"].min()),
        "signal_max": str(seed["signal_time"].max()),
        "exit_max": str(seed["exit_time"].max()),
        "columns": list(seed.columns),
        "development_source_hashes": source_hashes,
        "contains_2026_rows": False,
    }


def run(
    base_market_dir: str = DEFAULT_BASE_MARKET_DIR,
    june_market_dir: str = DEFAULT_JUNE_MARKET_DIR,
    funding_dir: str = DEFAULT_FUNDING_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    seed_path: str = DEFAULT_SEED,
    manifest_path: str = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    if canonical_hash(protocol()) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("CRES preregistration protocol drifted")
    target = Path(output_dir)
    records: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        base_input, june_input = _market_inputs(base_market_dir, june_market_dir, symbol)
        funding_input = resolve(
            Path(funding_dir) / f"{symbol}_funding_2025-01-01_2026-07-01.csv.gz"
        )
        market = combine_market_prefix(
            pd.read_csv(base_input), pd.read_csv(june_input), symbol
        )
        funding = validate_funding(pd.read_csv(funding_input), symbol, START, END)
        market_output = target / f"{symbol}_5m_2025_2026h1.csv.gz"
        funding_output = target / f"{symbol}_funding_2025_2026h1.csv.gz"
        deterministic_csv_gz(market, market_output)
        deterministic_csv_gz(funding, funding_output)
        reread_market = validate_market(pd.read_csv(market_output), symbol, START, END)
        reread_funding = validate_funding(pd.read_csv(funding_output), symbol, START, END)
        records.append(
            {
                "symbol": symbol,
                "base_input_market": str(base_input),
                "base_input_market_sha256": sha256_file(base_input),
                "june_input_market": str(june_input),
                "june_input_market_sha256": sha256_file(june_input),
                "input_funding": str(funding_input),
                "input_funding_sha256": sha256_file(funding_input),
                "output_market": str(market_output),
                "output_market_sha256": sha256_file(market_output),
                "market_rows": int(len(reread_market)),
                "market_min": str(reread_market["date"].min()),
                "market_max": str(reread_market["date"].max()),
                "confirmation_market_rows": int(
                    (reread_market["date"] >= CONFIRMATION_START).sum()
                ),
                "zero_quote_volume_bars": int(
                    (reread_market["quote_asset_volume"] == 0.0).sum()
                ),
                "output_funding": str(funding_output),
                "output_funding_sha256": sha256_file(funding_output),
                "funding_rows": int(len(reread_funding)),
                "funding_min": str(
                    pd.to_datetime(reread_funding["funding_time"], unit="ms").min()
                ),
                "funding_max": str(
                    pd.to_datetime(reread_funding["funding_time"], unit="ms").max()
                ),
                "confirmation_funding_rows": int(
                    (
                        pd.to_datetime(reread_funding["funding_time"], unit="ms")
                        >= CONFIRMATION_START
                    ).sum()
                ),
            }
        )
    seed = build_training_seed(seed_path)
    manifest: dict[str, Any] = {
        "protocol_version": "cres_v1_2026_source_freeze_2026-07-17",
        "preregistration_protocol_hash": EXPECTED_PROTOCOL_HASH,
        "post_entry_2026_strategy_returns_calculated": False,
        "physical_prefix": {"start": str(START), "end_exclusive": str(END)},
        "warmup": {
            "start": str(START),
            "end_exclusive": str(CONFIRMATION_START),
        },
        "confirmation": {
            "start": str(CONFIRMATION_START),
            "end_exclusive": str(END),
        },
        "symbols": list(SYMBOLS),
        "market_contract": "exact complete 5m grid; no fill; valid OHLC/taker identity",
        "funding_contract": "exact reported event rows; no fill or synthetic rate",
        "historical_training_seed": seed,
        "records": records,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    output = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-market-dir", default=DEFAULT_BASE_MARKET_DIR)
    parser.add_argument("--june-market-dir", default=DEFAULT_JUNE_MARKET_DIR)
    parser.add_argument("--funding-dir", default=DEFAULT_FUNDING_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.base_market_dir,
                args.june_market_dir,
                args.funding_dir,
                args.output_dir,
                args.seed,
                args.manifest,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
