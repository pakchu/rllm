"""Materialize the preregistered OCDR-12C non-price source snapshot."""
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
OLD_BVOL = Path(
    "data/binance_btc_bvol_hourly_opdr_2023_2026/"
    "BTCBVOLUSDT_1h_2023-06-20_2026-06-30.csv.gz"
)
JULY_BVOL = Path(
    "data/binance_btc_bvol_hourly_ocdr_2026_07/"
    "BTCBVOLUSDT_1h_2026-07-01_2026-07-31.csv.gz"
)
OLD_DVOL = Path("data/deribit_btc_dvol_1h_2023-06-20_2026-07-01.csv.gz")
JULY_DVOL = Path("data/deribit_btc_dvol_1h_ocdr_2026-07-01_2026-08-01.csv.gz")
FROZEN_MARKS = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
FROZEN_MARKS_SHA = "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
OUTPUT_DIR = Path("data/options_crowding_deleveraging_relay_sources_v4_2023_2026")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def combine_volatility(first: Path, second: Path, *, time_column: str) -> pd.DataFrame:
    frames = [pd.read_csv(path, compression="gzip") for path in (first, second)]
    combined = pd.concat(frames, ignore_index=True)
    combined[time_column] = pd.to_datetime(combined[time_column], utc=True, errors="raise")
    combined = combined.sort_values(time_column).drop_duplicates(time_column, keep="last")
    combined = combined[(combined[time_column] >= START) & (combined[time_column] < END)]
    if combined[time_column].duplicated().any() or not combined[time_column].is_monotonic_increasing:
        raise RuntimeError("volatility clock is duplicate or unordered")
    return combined.reset_index(drop=True)


def query_postgres(env_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import create_engine, text

    engine = create_engine(postgres_url_from_env(env_file), connect_args={"connect_timeout": 10})
    oi_sql = text(
        """SELECT symbol,period,ts,sum_open_interest,sum_open_interest_value,source
        FROM open_interest_binance
        WHERE symbol='BTCUSDT' AND period='5m' AND source='open_interest_hist'
          AND ts>=:start AND ts<:end ORDER BY ts"""
    )
    funding_sql = text(
        """SELECT symbol,funding_time,funding_rate,mark_price
        FROM funding_rates_binance
        WHERE symbol='BTCUSDT' AND funding_time>=:start AND funding_time<:end
        ORDER BY funding_time"""
    )
    with engine.connect() as connection:
        params = {"start": START.to_pydatetime(), "end": END.to_pydatetime()}
        oi = pd.read_sql_query(oi_sql, connection, params=params)
        funding = pd.read_sql_query(funding_sql, connection, params=params)
    engine.dispose()
    return oi, funding


def validate_oi(frame: pd.DataFrame) -> None:
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="raise")
    if frame.empty or frame["ts"].duplicated().any() or not frame["ts"].is_monotonic_increasing:
        raise RuntimeError("OI source is empty, duplicate or unordered")
    values = frame[["sum_open_interest", "sum_open_interest_value"]].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("OI values must be finite and nonnegative")



def complete_funding_marks(frame: pd.DataFrame) -> pd.DataFrame:
    if sha256(FROZEN_MARKS) != FROZEN_MARKS_SHA:
        raise RuntimeError("frozen funding mark source drift")
    out = frame.copy()
    out["funding_time"] = pd.to_datetime(out["funding_time"], utc=True, errors="raise")
    marks = pd.read_csv(FROZEN_MARKS, compression="gzip")
    marks["mark_time"] = pd.to_datetime(marks["funding_time_utc"], utc=True, errors="raise")
    marks["frozen_rate"] = pd.to_numeric(marks["funding_rate"], errors="raise")
    marks["frozen_mark"] = pd.to_numeric(marks["settlement_mark_price"], errors="raise")
    joined = pd.merge_asof(
        out.sort_values("funding_time"),
        marks[["mark_time", "frozen_rate", "frozen_mark"]].sort_values("mark_time"),
        left_on="funding_time", right_on="mark_time", direction="nearest",
        tolerance=pd.Timedelta(seconds=1),
    )
    db_mark = pd.to_numeric(joined["mark_price"], errors="coerce")
    missing = ~np.isfinite(db_mark) | db_mark.le(0)
    rate = pd.to_numeric(joined["funding_rate"], errors="raise")
    if joined.loc[missing, "frozen_mark"].isna().any():
        raise RuntimeError("missing exact frozen settlement mark fallback")
    if not np.isclose(rate[missing], joined.loc[missing, "frozen_rate"], rtol=0, atol=1e-12).all():
        raise RuntimeError("funding rate disagrees with frozen mark source")
    joined["mark_source"] = np.where(missing, "frozen_official_binance_mark", "postgres_exact_mark")
    joined.loc[missing, "mark_price"] = joined.loc[missing, "frozen_mark"]
    return joined.drop(columns=["mark_time", "frozen_rate", "frozen_mark"])

def validate_funding(frame: pd.DataFrame) -> None:
    frame["funding_time"] = pd.to_datetime(frame["funding_time"], utc=True, errors="raise")
    if frame.empty or frame["funding_time"].duplicated().any() or not frame["funding_time"].is_monotonic_increasing:
        raise RuntimeError("funding source is empty, duplicate or unordered")
    values = frame[["funding_rate", "mark_price"]].to_numpy(float)
    if not np.isfinite(values).all() or (frame["mark_price"].astype(float) <= 0).any():
        raise RuntimeError("funding values are invalid")


def run(env_file: str, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    bvol = combine_volatility(OLD_BVOL, JULY_BVOL, time_column="date")
    dvol = combine_volatility(OLD_DVOL, JULY_DVOL, time_column="date")
    oi, funding = query_postgres(env_file)
    validate_oi(oi)
    funding = complete_funding_marks(funding)
    validate_funding(funding)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "bvol": output_dir / "bvol_hourly.csv.gz",
        "dvol": output_dir / "dvol_hourly.csv.gz",
        "oi": output_dir / "open_interest_5m.csv.gz",
        "funding": output_dir / "funding.csv.gz",
    }
    for name, frame in (("bvol", bvol), ("dvol", dvol), ("oi", oi), ("funding", funding)):
        _write_gzip_csv(frame, outputs[name])
    core = {
        "protocol_version": "ocdr_12c_source_snapshot_v1",
        "window": [START.isoformat(), END.isoformat()],
        "btc_price_or_return_opened": False,
        "candidate_incidence_opened": False,
        "inputs": {
            "old_bvol_sha256": sha256(OLD_BVOL),
            "july_bvol_sha256": sha256(JULY_BVOL),
            "old_dvol_sha256": sha256(OLD_DVOL),
            "july_dvol_sha256": sha256(JULY_DVOL),
            "postgres_tables": ["open_interest_binance", "funding_rates_binance"],
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path), "rows": len(frame)}
            for (name, path), frame in zip(outputs.items(), (bvol, dvol, oi, funding))
        },
        "invalid_source_rows_retained_without_imputation": {
            "oi_zero_rows": int(pd.to_numeric(oi["sum_open_interest"]).eq(0).sum()),
            "oi_off_grid_rows": int((oi["ts"].dt.floor("5min") != oi["ts"]).sum()),
            "funding_frozen_mark_fallback_rows": int(funding["mark_source"].eq("frozen_official_binance_mark").sum()),
        },
        "availability": {
            "bvol": "feature_available_time_utc",
            "dvol": "close_time",
            "oi": "raw observation ts; backward as-of maximum age 5m",
            "funding": "funding_time; positive DB mark else exact frozen official mark",
        },
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default="/home/pakchu/rllm/.env")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    report = run(args.env_file, args.output_dir)
    print(json.dumps(report["outputs"], indent=2))


if __name__ == "__main__":
    main()
