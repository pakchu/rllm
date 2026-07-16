"""Physically isolate CRRC-72's exact 2023 execution sources.

The exporter copies only calendar-2023 BTCUSDT 5m OHLC and exact Binance
funding settlement timestamps/rates into dedicated files.  It calculates no
return, label, signal, PnL, equity, CAGR, or drawdown.
"""
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

from training.export_leave_one_out_residual_exhaustion_sources import (
    deterministic_csv_gz,
    resolve,
    sha256_file,
)
from training.preregister_cross_venue_radial_refill_compression import canonical_hash


START = pd.Timestamp("2023-01-01 00:00:00")
END = pd.Timestamp("2024-01-01 00:00:00")
MARKET_ROWS = 365 * 24 * 12
FUNDING_ROWS = 365 * 3
SOURCE_MARKET_ROWS = 420_768
SOURCE_FUNDING_ROWS = 3_285

MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
MARKET_SOURCE = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_SOURCE_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)
FUNDING_MANIFEST = Path("results/binance_um_aux_btc_2021_2023_manifest.json")
FUNDING_MANIFEST_SHA256 = (
    "80c77f461be54b77c7554837a304a187321a052dd05cb39b4e0a3c80de5d2bdc"
)
FUNDING_SOURCE = Path(
    "data/binance_um_aux_btc_2021_2023/"
    "BTCUSDT_funding_2021-01-01_2023-12-31.csv.gz"
)
FUNDING_SOURCE_SHA256 = (
    "654c668e3aea344d5906465cbbd090f2e4ff0c47e9d4bd8cf3856c24549cfc97"
)
DEFAULT_OUTPUT_DIR = Path("data/crrc_2023_execution")
DEFAULT_MANIFEST = Path("results/crrc_2023_execution_source_manifest_2026-07-17.json")
DEFAULT_DOCS = Path("docs/crrc-2023-execution-source-freeze-2026-07-17.md")
MARKET_COLUMNS = ("date", "open", "high", "low", "close")
FUNDING_INPUT_COLUMNS = ("symbol", "funding_rate", "funding_time")
FUNDING_OUTPUT_COLUMNS = ("event_time", "funding_rate")


def _validate_source_contracts() -> dict[str, Any]:
    market_manifest_path = resolve(MARKET_MANIFEST)
    market_source_path = resolve(MARKET_SOURCE)
    funding_manifest_path = resolve(FUNDING_MANIFEST)
    funding_source_path = resolve(FUNDING_SOURCE)
    expected = (
        (market_manifest_path, MARKET_MANIFEST_SHA256, "market manifest"),
        (market_source_path, MARKET_SOURCE_SHA256, "market source"),
        (funding_manifest_path, FUNDING_MANIFEST_SHA256, "funding manifest"),
        (funding_source_path, FUNDING_SOURCE_SHA256, "funding source"),
    )
    for path, digest, label in expected:
        if sha256_file(path) != digest:
            raise RuntimeError(f"CRRC frozen {label} changed")

    market_manifest = json.loads(market_manifest_path.read_text())
    if market_manifest.get("combined_sha256") != MARKET_SOURCE_SHA256:
        raise RuntimeError("market manifest no longer binds the frozen source")
    if market_manifest.get("rows") != SOURCE_MARKET_ROWS:
        raise RuntimeError("market source row count drifted")
    if market_manifest.get("last_date") != "2023-12-31 23:55:00":
        raise RuntimeError("market source escaped pre-2024 boundary")
    if market_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise RuntimeError("market source manifest opened outcomes")

    funding_manifest = json.loads(funding_manifest_path.read_text())
    funding_file = funding_manifest.get("files", {}).get("funding", {})
    if funding_file.get("sha256") != FUNDING_SOURCE_SHA256:
        raise RuntimeError("funding manifest no longer binds the frozen source")
    if funding_file.get("rows") != SOURCE_FUNDING_ROWS:
        raise RuntimeError("funding source row count drifted")
    if funding_manifest.get("protocol", {}).get("post_2023_rows_written") is not False:
        raise RuntimeError("funding source manifest wrote a post-2023 row")
    return {
        "market_manifest": str(market_manifest_path),
        "market_manifest_sha256": MARKET_MANIFEST_SHA256,
        "market_source": str(market_source_path),
        "market_source_sha256": MARKET_SOURCE_SHA256,
        "funding_manifest": str(funding_manifest_path),
        "funding_manifest_sha256": FUNDING_MANIFEST_SHA256,
        "funding_source": str(funding_source_path),
        "funding_source_sha256": FUNDING_SOURCE_SHA256,
    }


def validate_market_2023(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(MARKET_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"BTCUSDT market misses columns: {sorted(missing)}")
    output = frame.loc[:, MARKET_COLUMNS].copy()
    output["date"] = pd.to_datetime(
        output["date"], utc=True, errors="raise"
    ).dt.tz_convert(None)
    expected = pd.date_range(START, END - pd.Timedelta(minutes=5), freq="5min")
    if not pd.DatetimeIndex(output["date"]).equals(expected):
        raise ValueError("BTCUSDT 2023 execution grid changed")
    for column in ("open", "high", "low", "close"):
        output[column] = pd.to_numeric(output[column], errors="raise")
    values = output[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("BTCUSDT 2023 OHLC is invalid")
    if (output["high"] < output[["open", "close"]].max(axis=1)).any():
        raise ValueError("BTCUSDT high is below open/close")
    if (output["low"] > output[["open", "close"]].min(axis=1)).any():
        raise ValueError("BTCUSDT low is above open/close")
    if not (output["date"] < END).all():
        raise ValueError("post-2023 market row escaped physical prefix")
    return output


def validate_funding_2023(frame: pd.DataFrame) -> pd.DataFrame:
    if "event_time" in frame.columns:
        event_time = pd.to_datetime(
            frame["event_time"], utc=True, errors="raise"
        ).dt.tz_convert(None)
    elif "funding_time" in frame.columns:
        event_time = pd.to_datetime(
            pd.to_numeric(frame["funding_time"], errors="raise"),
            unit="ms",
            utc=True,
        ).dt.tz_convert(None)
    else:
        raise ValueError("BTCUSDT funding misses its exact timestamp")
    if "symbol" in frame.columns and not frame["symbol"].astype(str).eq("BTCUSDT").all():
        raise ValueError("funding source contains another symbol")
    funding_rate = pd.to_numeric(frame["funding_rate"], errors="raise")
    if len(event_time) != FUNDING_ROWS:
        raise ValueError("BTCUSDT 2023 funding row count changed")
    if event_time.duplicated().any() or not event_time.is_monotonic_increasing:
        raise ValueError("BTCUSDT funding timestamps are duplicate or unsorted")
    expected = pd.date_range(START, END - pd.Timedelta(hours=8), freq="8h")
    jitter = pd.Series(
        (pd.DatetimeIndex(event_time) - expected).total_seconds(), dtype=float
    ).abs()
    if (jitter > 1.0).any():
        raise ValueError("BTCUSDT funding timestamp moved over one second from cadence")
    if not np.isfinite(funding_rate).all():
        raise ValueError("BTCUSDT funding rate is non-finite")
    if not ((event_time >= START) & (event_time < END)).all():
        raise ValueError("post-2023 funding row escaped physical prefix")
    return pd.DataFrame(
        {"event_time": event_time, "funding_rate": funding_rate},
        columns=FUNDING_OUTPUT_COLUMNS,
    )


def markdown(result: dict[str, Any]) -> str:
    return f"""# CRRC-72 2023 execution-source freeze — 2026-07-17

- Outcome return/PnL calculated: **no**
- Market rows: `{MARKET_ROWS}`
- Funding rows: `{FUNDING_ROWS}`
- Maximum timestamp exclusive: `{END}`
- 2024 rows parsed or written: **0**
- Exact funding timestamps preserved: **yes**
- Manifest hash: `{result['manifest_hash']}`

The exporter reads only the physically pre-2024 source tails and writes a
dedicated calendar-2023 BTCUSDT OHLC file plus exact millisecond funding
settlement timestamps/rates. It constructs no signal, return, label, equity,
CAGR, MDD, or PnL. The CRRC evaluator must load only these two outputs.
"""


def run(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    docs_path: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    source = _validate_source_contracts()
    market_source = Path(source["market_source"])
    funding_source = Path(source["funding_source"])

    # Keep the CSV header, skip every pre-2023 data row, and stop exactly at
    # the end of 2023.  Neither input contains a post-2023 row.
    market = validate_market_2023(
        pd.read_csv(
            market_source,
            usecols=MARKET_COLUMNS,
            skiprows=range(1, SOURCE_MARKET_ROWS - MARKET_ROWS + 1),
            nrows=MARKET_ROWS,
        )
    )
    funding = validate_funding_2023(
        pd.read_csv(
            funding_source,
            usecols=FUNDING_INPUT_COLUMNS,
            skiprows=range(1, SOURCE_FUNDING_ROWS - FUNDING_ROWS + 1),
            nrows=FUNDING_ROWS,
        )
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    market_output = root / "BTCUSDT_5m_2023.csv.gz"
    funding_output = root / "BTCUSDT_funding_2023.csv.gz"
    deterministic_csv_gz(market, market_output)
    deterministic_csv_gz(funding, funding_output)
    reread_market = validate_market_2023(pd.read_csv(market_output))
    reread_funding = validate_funding_2023(pd.read_csv(funding_output))

    core: dict[str, Any] = {
        "protocol_version": "crrc72_2023_execution_source_freeze_v1_2026-07-17",
        "outcomes_calculated": False,
        "signals_labels_or_equity_constructed": False,
        "source": source,
        "selection_prefix": [str(START), str(END)],
        "market_output": str(market_output),
        "market_output_sha256": sha256_file(market_output),
        "market_rows": int(len(reread_market)),
        "market_first": reread_market["date"].iloc[0].isoformat(),
        "market_last": reread_market["date"].iloc[-1].isoformat(),
        "funding_output": str(funding_output),
        "funding_output_sha256": sha256_file(funding_output),
        "funding_rows": int(len(reread_funding)),
        "funding_first_exact": reread_funding["event_time"].iloc[0].isoformat(),
        "funding_last_exact": reread_funding["event_time"].iloc[-1].isoformat(),
        "funding_nonzero_millisecond_offsets": int(
            (reread_funding["event_time"].dt.microsecond != 0).sum()
        ),
        "2024_rows_parsed": 0,
        "2024_rows_written": 0,
    }
    core["manifest_hash"] = canonical_hash(core)
    result = {**core, "created_at": datetime.now(timezone.utc).isoformat()}
    manifest_output = Path(manifest_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    docs_output = Path(docs_path)
    docs_output.parent.mkdir(parents=True, exist_ok=True)
    docs_output.write_text(markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--docs", default=str(DEFAULT_DOCS))
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.manifest, args.docs), indent=2))


if __name__ == "__main__":
    main()
