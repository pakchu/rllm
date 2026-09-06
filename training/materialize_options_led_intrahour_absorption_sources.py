"""Materialize only completed-hour intrahour BTC features for OLIAH-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.live_db_features import postgres_url_from_env
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


START = pd.Timestamp("2023-06-20T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
NONPRICE_MANIFEST = Path("data/options_crowding_deleveraging_relay_sources_v4_2023_2026/manifest.json")
OUTPUT_DIR = Path("data/options_led_intrahour_absorption_sources_2023_2026")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_completed_hours(env_file: str) -> pd.DataFrame:
    from sqlalchemy import create_engine, text

    engine = create_engine(
        postgres_url_from_env(env_file), connect_args={"connect_timeout": 10}
    )
    query = text(
        """
        SELECT
          date_bin('1 hour', ts, TIMESTAMPTZ '1970-01-01') AS hour_start,
          (array_agg(open ORDER BY ts))[1] AS hour_open,
          (array_agg(close ORDER BY ts))[30] AS first_half_close,
          (array_agg(open ORDER BY ts))[31] AS second_half_open,
          (array_agg(close ORDER BY ts DESC))[1] AS hour_close,
          count(*) AS source_rows,
          count(DISTINCT ts) AS distinct_timestamps
        FROM bars_binance
        WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
        GROUP BY 1
        ORDER BY 1
        """
    )
    with engine.connect() as connection:
        frame = pd.read_sql_query(
            query,
            connection,
            params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
        )
    engine.dispose()
    frame["hour_start"] = pd.to_datetime(frame["hour_start"], utc=True, format="mixed")
    grid = pd.DataFrame(
        {"hour_start": pd.date_range(START, END, freq="1h", inclusive="left")}
    )
    frame = grid.merge(frame, on="hour_start", how="left", validate="one_to_one")
    for column in ("source_rows", "distinct_timestamps"):
        frame[column] = frame[column].fillna(0).astype(int)
    price_columns = ("hour_open", "first_half_close", "second_half_open", "hour_close")
    for column in price_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["source_valid"] = (
        frame["source_rows"].eq(60)
        & frame["distinct_timestamps"].eq(60)
        & np.isfinite(frame[list(price_columns)]).all(axis=1)
        & frame[list(price_columns)].gt(0).all(axis=1)
    )
    frame.loc[~frame["source_valid"], list(price_columns)] = np.nan
    frame["decision_time"] = frame["hour_start"] + pd.Timedelta(hours=1)
    return frame


def run(env_file: str) -> dict[str, Any]:
    frame = load_completed_hours(env_file)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "btc_intrahour_path.csv.gz"
    _write_gzip_csv(frame, output)
    nonprice = json.loads(NONPRICE_MANIFEST.read_text())
    core = {
        "protocol_version": "oliah_6_source_snapshot_v1",
        "window": [START.isoformat(), END.isoformat()],
        "base_nonprice_manifest": {
            "path": str(NONPRICE_MANIFEST),
            "sha256": sha256(NONPRICE_MANIFEST),
            "manifest_hash": nonprice["manifest_hash"],
        },
        "feature_price_scope": "completed [T-1h,T) minute offsets 0,29,30,59 only",
        "post_entry_return_pnl_or_execution_price_opened": False,
        "candidate_incidence_opened": False,
        "output": {
            "path": str(output),
            "sha256": sha256(output),
            "rows": len(frame),
            "valid_rows": int(frame["source_valid"].sum()),
        },
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default="/home/pakchu/rllm/.env")
    args = parser.parse_args()
    print(json.dumps(run(args.env_file), indent=2))
