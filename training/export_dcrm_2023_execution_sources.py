"""Physically split DCRM-1's 2023 execution prefix before outcome evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.build_dispersion_conditioned_residual_momentum_support import SYMBOLS
from training.export_leave_one_out_residual_exhaustion_sources import (
    deterministic_csv_gz,
    resolve,
    sha256_file,
)
from training.preregister_dispersion_conditioned_residual_momentum import canonical_hash


START = pd.Timestamp("2023-01-01 00:00:00")
END = pd.Timestamp("2024-01-01 00:00:00")
MARKET_ROWS = 365 * 24 * 12
FUNDING_ROWS = 365 * 3
SOURCE_MANIFEST = Path(
    "results/leave_one_out_residual_exhaustion_v1_source_manifest_2026-07-17.json"
)
SOURCE_MANIFEST_SHA256 = "b3f5841f10b3e44ee47fb5d69c7acc6a2df0975596cdc0fd4019925f49b6eb66"
SOURCE_MANIFEST_HASH = "1c54fddc45fcc516d8ce42741904e018e8d00e646eff40be514273cf10eee7ed"
DEFAULT_INPUT_DIR = Path("data/binance_um_lore_2023_2024")
DEFAULT_OUTPUT_DIR = Path("data/dcrm_2023_execution")
DEFAULT_MANIFEST = Path("results/dcrm_2023_execution_source_manifest_2026-07-17.json")
DEFAULT_DOCS = Path("docs/dcrm-2023-execution-source-freeze-2026-07-17.md")
MARKET_COLUMNS = ("date", "open", "high", "low", "close", "tic")
FUNDING_COLUMNS = ("event_time", "funding_rate")


def _source_records() -> dict[str, dict[str, Any]]:
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("LORE source manifest file changed")
    payload = json.loads(SOURCE_MANIFEST.read_text())
    body = {key: value for key, value in payload.items() if key not in {"manifest_hash", "created_at"}}
    if payload.get("manifest_hash") != SOURCE_MANIFEST_HASH or canonical_hash(body) != SOURCE_MANIFEST_HASH:
        raise RuntimeError("LORE source manifest identity changed")
    return {str(row["symbol"]): row for row in payload["records"]}


def validate_market_prefix(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    missing = set(MARKET_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{symbol} missing market columns: {sorted(missing)}")
    out = frame.loc[:, MARKET_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="raise").dt.tz_convert(None)
    expected = pd.date_range(START, END - pd.Timedelta(minutes=5), freq="5min")
    if not pd.DatetimeIndex(out["date"]).equals(expected):
        raise ValueError(f"{symbol} 2023 market prefix/grid changed")
    if not out["tic"].astype(str).eq(symbol).all():
        raise ValueError(f"{symbol} market identity changed")
    numeric = ["open", "high", "low", "close"]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="raise")
    values = out[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{symbol} invalid 2023 OHLC")
    if not (out["date"] < END).all():
        raise ValueError(f"{symbol} future market row escaped prefix")
    return out


def validate_funding_prefix(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if "funding_time" in frame.columns:
        event_time = pd.to_datetime(
            pd.to_numeric(frame["funding_time"], errors="raise"), unit="ms", utc=True
        ).dt.tz_convert(None)
    elif "event_time" in frame.columns:
        event_time = pd.to_datetime(frame["event_time"], utc=True, errors="raise").dt.tz_convert(None)
    else:
        raise ValueError(f"{symbol} missing funding timestamp")
    rate = pd.to_numeric(frame["funding_rate"], errors="raise")
    expected = pd.date_range(START, END - pd.Timedelta(hours=8), freq="8h")
    if not pd.DatetimeIndex(event_time).equals(expected):
        raise ValueError(f"{symbol} 2023 funding prefix/grid changed")
    if not np.isfinite(rate).all():
        raise ValueError(f"{symbol} invalid 2023 funding rate")
    if not (event_time < END).all():
        raise ValueError(f"{symbol} future funding row escaped prefix")
    return pd.DataFrame({"event_time": event_time, "funding_rate": rate})


def markdown(result: dict[str, Any]) -> str:
    return f"""# DCRM-1 2023 execution-source freeze — 2026-07-17

- Outcome return/PnL calculated: **no**
- Market rows written per symbol: `{MARKET_ROWS}`
- Funding rows written per symbol: `{FUNDING_ROWS}`
- Maximum timestamp exclusive: `{END}`
- 2024 rows parsed or written: **0**
- Symbols: `{list(SYMBOLS)}`
- Manifest hash: `{result['manifest_hash']}`

The exporter reads fixed row counts from the already-frozen 2023–2024 source
and writes physically separate 2023-only files. It does not hash or parse the
combined source beyond those prefixes and calculates no return, PnL, label,
equity, or drawdown. The 2023 evaluator is required to read only these files.
"""


def run(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    docs_path: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    records = _source_records()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_records: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        input_market = resolve(Path(input_dir) / f"{symbol}_5m_2023_2024.csv.gz")
        input_funding = resolve(Path(input_dir) / f"{symbol}_funding_2023_2024.csv.gz")
        market = validate_market_prefix(
            pd.read_csv(input_market, usecols=MARKET_COLUMNS, nrows=MARKET_ROWS), symbol
        )
        funding = validate_funding_prefix(
            pd.read_csv(
                input_funding,
                usecols=["funding_time", "funding_rate"],
                nrows=FUNDING_ROWS,
            ),
            symbol,
        )
        market_output = output_root / f"{symbol}_5m_2023.csv.gz"
        funding_output = output_root / f"{symbol}_funding_2023.csv.gz"
        deterministic_csv_gz(market, market_output)
        deterministic_csv_gz(funding, funding_output)
        reread_market = validate_market_prefix(pd.read_csv(market_output), symbol)
        reread_funding = validate_funding_prefix(pd.read_csv(funding_output), symbol)
        frozen = records[symbol]
        output_records.append(
            {
                "symbol": symbol,
                "combined_input_market": str(input_market),
                "combined_input_market_sha256_from_prior_manifest": frozen[
                    "output_market_sha256"
                ],
                "combined_input_funding": str(input_funding),
                "combined_input_funding_sha256_from_prior_manifest": frozen[
                    "output_funding_sha256"
                ],
                "combined_inputs_rehashed": False,
                "market_output": str(market_output),
                "market_output_sha256": sha256_file(market_output),
                "market_rows": int(len(reread_market)),
                "market_max": reread_market["date"].iloc[-1].isoformat(),
                "funding_output": str(funding_output),
                "funding_output_sha256": sha256_file(funding_output),
                "funding_rows": int(len(reread_funding)),
                "funding_max": reread_funding["event_time"].iloc[-1].isoformat(),
                "rows_at_or_after_2024_parsed": 0,
                "rows_at_or_after_2024_written": 0,
            }
        )
    result: dict[str, Any] = {
        "protocol_version": "dcrm_2023_execution_source_freeze_v1_2026-07-17",
        "outcomes_calculated": False,
        "labels_or_equity_constructed": False,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_manifest_hash": SOURCE_MANIFEST_HASH,
        "selection_prefix": [str(START), str(END)],
        "market_rows_per_symbol": MARKET_ROWS,
        "funding_rows_per_symbol": FUNDING_ROWS,
        "2024_rows_parsed": 0,
        "2024_rows_written": 0,
        "records": output_records,
    }
    result["manifest_hash"] = canonical_hash(result)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest_output = Path(manifest_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    docs_output = Path(docs_path)
    docs_output.parent.mkdir(parents=True, exist_ok=True)
    docs_output.write_text(markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--docs", default=str(DEFAULT_DOCS))
    args = parser.parse_args()
    print(json.dumps(run(args.input_dir, args.output_dir, args.manifest, args.docs), indent=2))


if __name__ == "__main__":
    main()
