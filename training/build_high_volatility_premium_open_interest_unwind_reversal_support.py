"""Build outcome-blind source support for frozen HVPOIUR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_premium_open_interest_unwind_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "b69c04df556938efd5bb3856f65407f90c64dc10b5ad205a6d104e440475b75c"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
DVOL = Path("data/options_crowding_deleveraging_relay_sources_v4_2023_2026/dvol_hourly.csv.gz")
SOURCE_DIR = Path("data/high_volatility_premium_open_interest_unwind_reversal_sources_2023_2026")
FEATURES = SOURCE_DIR / "eight_hour_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_premium_open_interest_unwind_reversal_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_premium_open_interest_unwind_reversal_controls_2023_2026")
RESULT = Path("results/high_volatility_premium_open_interest_unwind_reversal_support_2026-08-11.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_dvol_gate", "no_premium_tail", "no_oi_contraction", "one_block_stale_features", "direction_flip", "forced_long")
PREMIUM_QUERY = """
SELECT date_bin('8 hours',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS block_start,
       count(*) AS premium_rows, count(DISTINCT ts) AS premium_distinct,
       min(ts) AS premium_first_ts, max(ts) AS premium_last_ts,
       (array_agg(open ORDER BY ts))[1] AS premium_open,
       (array_agg(close ORDER BY ts DESC))[1] AS premium_close
FROM bars_binance_premium
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
GROUP BY 1 ORDER BY 1
"""
OI_QUERY = """
SELECT date_bin('8 hours',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS block_start,
       count(*) AS oi_rows, count(DISTINCT ts) AS oi_distinct,
       min(ts) AS oi_first_ts, max(ts) AS oi_last_ts,
       (array_agg(sum_open_interest ORDER BY ts))[1] AS oi_first,
       (array_agg(sum_open_interest ORDER BY ts DESC))[1] AS oi_last
FROM open_interest_binance
WHERE symbol='BTCUSDT' AND period='5m' AND ts>=:start AND ts<:end
GROUP BY 1 ORDER BY 1
"""
CLOCK_COLUMNS = ("candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "premium_displacement", "premium_displacement_rank", "oi_change", "dvol_close", "dvol_level_rank")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def strict_prior_midrank(values: pd.Series, lookback: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-lookback:], dtype=float)
        if np.isfinite(current) and len(prior) >= minimum:
            output.at[index] = (np.sum(prior < current) + 0.5 * np.sum(prior == current)) / len(prior)
        if np.isfinite(current):
            history.append(float(current))
    return output


def load_database_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text
    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            premium = pd.read_sql_query(text(PREMIUM_QUERY), connection, params={"start": START, "end": END})
            oi = pd.read_sql_query(text(OI_QUERY), connection, params={"start": START, "end": END})
    finally:
        engine.dispose()
    for frame in (premium, oi): frame["block_start"] = pd.to_datetime(frame.block_start, utc=True)
    return premium, oi


def load_dvol() -> pd.DataFrame:
    frame = pd.read_csv(DVOL)
    frame["decision_time"] = pd.to_datetime(frame.close_time, utc=True)
    frame["dvol_close"] = pd.to_numeric(frame.close, errors="coerce")
    return frame[["decision_time", "dvol_close"]]


def build_features(premium: pd.DataFrame, oi: pd.DataFrame, dvol: pd.DataFrame) -> pd.DataFrame:
    frame = premium.merge(oi, on="block_start", how="inner", validate="one_to_one")
    frame["decision_time"] = frame.block_start + pd.Timedelta(hours=8)
    frame = frame.merge(dvol, on="decision_time", how="left", validate="one_to_one")
    frame["premium_displacement"] = pd.to_numeric(frame.premium_close, errors="coerce") - pd.to_numeric(frame.premium_open, errors="coerce")
    oi_first = pd.to_numeric(frame.oi_first, errors="coerce")
    oi_last = pd.to_numeric(frame.oi_last, errors="coerce")
    frame["oi_change"] = np.log((oi_last / oi_first).where(oi_first.gt(0) & oi_last.gt(0)))
    frame["source_valid"] = (
        frame.premium_rows.eq(480) & frame.premium_distinct.eq(480)
        & frame.oi_rows.eq(96) & frame.oi_distinct.eq(96)
        & frame.premium_first_ts.eq(frame.block_start)
        & frame.premium_last_ts.eq(frame.block_start + pd.Timedelta(hours=7, minutes=59))
        & frame.oi_first_ts.eq(frame.block_start)
        & frame.oi_last_ts.eq(frame.block_start + pd.Timedelta(hours=7, minutes=55))
        & frame.premium_displacement.ne(0) & np.isfinite(frame.premium_displacement)
        & np.isfinite(frame.oi_change) & oi_first.gt(0) & oi_last.gt(0)
        & frame.dvol_close.gt(0) & np.isfinite(frame.dvol_close)
    )
    frame = frame.sort_values("decision_time").reset_index(drop=True)
    frame["premium_displacement_rank"] = strict_prior_midrank(frame.premium_displacement.abs().where(frame.source_valid))
    frame["dvol_level_rank"] = strict_prior_midrank(frame.dvol_close.where(frame.source_valid))
    return frame


def conditions(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS): raise ValueError(control)
    used = frame.shift(1) if control == "one_block_stale_features" else frame
    active = used.source_valid.eq(True)
    if control != "no_dvol_gate": active &= used.dvol_level_rank.ge(0.60)
    if control != "no_premium_tail": active &= used.premium_displacement_rank.ge(0.60)
    if control != "no_oi_contraction": active &= used.oi_change.lt(0)
    side = -np.sign(used.premium_displacement).fillna(0).astype(int)
    if control == "direction_flip": side = -side
    elif control == "forced_long": side = pd.Series(1, index=frame.index)
    return active & side.ne(0), side, used


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides, used = conditions(features, control); rows: list[dict[str, Any]] = []
    for index in features.index[active]:
        decision = pd.Timestamp(features.at[index, "decision_time"]); entry = decision + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=8)
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        source = used.loc[index]
        rows.append({"candidate": "HVPOIUR-8", "control": control, "split": split, "decision_time": decision, "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]), "premium_displacement": float(source.premium_displacement), "premium_displacement_rank": float(source.premium_displacement_rank), "oi_change": float(source.oi_change), "dvol_close": float(source.dvol_close), "dvol_level_rank": float(source.dvol_level_rank)})
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    rows = clock[clock.split.eq(split)]
    if rows.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(rows.side.eq(1).sum()), int(rows.side.eq(-1).sum()); months = pd.to_datetime(rows.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(rows), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(rows), "max_month_share": int(months.max()) / len(rows)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA: raise RuntimeError("HVPOIUR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    premium, oi = load_database_sources(); features = build_features(premium, oi, load_dvol())
    primary = build_clock(features); controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True); CLOCK.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES); _write_gzip_csv(primary, CLOCK)
    for name, value in controls.items(): _write_gzip_csv(value, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {"protocol_version": "hvpoiur_8_sources_v1", "queries": {"premium": PREMIUM_QUERY, "oi": OI_QUERY}, "tables": ["bars_binance_premium", "open_interest_binance"], "dvol": {"path": str(DVOL), "sha256": sha(DVOL)}, "window": [START.isoformat(), END.isoformat()], "features": {"path": str(FEATURES), "sha256": sha(FEATURES), "rows": len(features), "valid_rows": int(features.source_valid.sum())}, "outcomes_opened": False, "gross9_rows_opened": False, "no_imputation": True}
    manifest = {**source_core, "manifest_hash": prereg.canonical_hash(source_core)}; SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {check: passed for name, item in support.items() for check, passed in ((f"{name}_minimum_events", item["events"] >= MINIMUM[name]), (f"{name}_side_balance", item["minority_side_share"] >= 0.20), (f"{name}_month_concentration", item["max_month_share"] <= 0.45))}
    passed = all(checks.values())
    core = {"protocol_version": "hvpoiur_8_source_support_v1", "policy_id": "HVPOIUR-8", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": manifest["manifest_hash"]}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(value), "promotion_authorized": False} for name, value in controls.items()}, "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}; RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); print(json.dumps(run()["support"], indent=2))
