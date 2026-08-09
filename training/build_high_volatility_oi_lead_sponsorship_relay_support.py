"""Build outcome-blind source support for frozen HVOILSR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_oi_lead_sponsorship_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "c79bb5e0ebdb0f1ee48f2d307f7648ac9b24782871de25f544438925dc43877c"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
RAW_START = START - pd.Timedelta(minutes=5)
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_lead_score_gate", "no_variation_gate", "contemporaneous_correlation",
    "one_day_stale_features", "direction_flip", "same_clock_forced_long",
)
BAR_QUERY = """
SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS bar_time,
       (array_agg(open ORDER BY ts))[1] AS bar_open,
       max(high) AS bar_high, min(low) AS bar_low,
       (array_agg(close ORDER BY ts DESC))[1] AS bar_close,
       count(*) AS source_rows, count(DISTINCT ts) AS distinct_rows,
       min(ts) AS first_ts, max(ts) AS last_ts,
       bool_and(open>0 AND high>0 AND low>0 AND close>0
                AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
GROUP BY 1 ORDER BY 1
"""
OI_QUERY = """
SELECT ts, sum_open_interest, count(*) OVER (PARTITION BY ts) AS duplicate_count
FROM open_interest_binance
WHERE symbol='BTCUSDT' AND period='5m' AND source='open_interest_hist'
  AND ts>=:start AND ts<:end
ORDER BY ts
"""
SOURCE_DIR = Path("data/high_volatility_oi_lead_sponsorship_relay_sources_2023_2026")
FEATURES = SOURCE_DIR / "daily_oi_lead_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_oi_lead_sponsorship_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_oi_lead_sponsorship_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_oi_lead_sponsorship_relay_support_2026-08-09.json")
FEATURE_COLUMNS = (
    "source_day", "feature_available_time", "source_valid", "day_return",
    "realized_variation", "lead_correlation", "directional_lead_score",
    "contemporaneous_correlation", "directional_contemporaneous_score",
    "lead_score_rank", "contemporaneous_score_rank", "variation_rank",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_day", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side",
    *FEATURE_COLUMNS[3:],
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chash(payload: Any) -> str:
    return prereg.canonical_hash(payload)


def strict_prior_midrank(values: pd.Series, lookback: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-lookback:], dtype=float)
        if math.isfinite(current) and len(prior) >= minimum:
            output.at[index] = (np.sum(prior < current) + 0.5 * np.sum(prior == current)) / len(prior)
        if math.isfinite(current):
            history.append(float(current))
    return output


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            bars = pd.read_sql_query(text(BAR_QUERY), connection, params={"start": START, "end": END})
            oi = pd.read_sql_query(text(OI_QUERY), connection, params={"start": RAW_START, "end": END})
    finally:
        engine.dispose()
    return bars, oi


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        return math.nan
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def build_features(bars: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
    required_bars = {"bar_time", "bar_open", "bar_high", "bar_low", "bar_close", "source_rows", "distinct_rows", "first_ts", "last_ts", "coherent"}
    required_oi = {"ts", "sum_open_interest", "duplicate_count"}
    if not required_bars.issubset(bars) or not required_oi.issubset(oi):
        raise ValueError("HVOILSR source schema drift")
    b = bars.copy(); o = oi.copy()
    for column in ("bar_time", "first_ts", "last_ts"):
        b[column] = pd.to_datetime(b[column], utc=True, errors="coerce")
    for column in ("bar_open", "bar_high", "bar_low", "bar_close", "source_rows", "distinct_rows"):
        b[column] = pd.to_numeric(b[column], errors="coerce")
    o["ts"] = pd.to_datetime(o.ts, utc=True, errors="coerce")
    o["sum_open_interest"] = pd.to_numeric(o.sum_open_interest, errors="coerce")
    o["duplicate_count"] = pd.to_numeric(o.duplicate_count, errors="coerce")
    b = b.sort_values("bar_time", kind="mergesort").set_index("bar_time")
    o = o[o.duplicate_count.eq(1)].sort_values("ts", kind="mergesort").set_index("ts")
    rows: list[dict[str, Any]] = []
    for source_day in pd.date_range(START, END, freq="1D", inclusive="left"):
        price_index = pd.date_range(source_day, source_day + pd.Timedelta(days=1), freq="5min", inclusive="left")
        oi_index = pd.date_range(source_day - pd.Timedelta(minutes=5), source_day + pd.Timedelta(days=1), freq="5min", inclusive="left")
        price = b.reindex(price_index); inventory = o.reindex(oi_index)
        expected_first = pd.Series(price_index, index=price_index)
        expected_last = pd.Series(price_index + pd.Timedelta(minutes=4), index=price_index)
        price_ok = (
            np.isfinite(price[["bar_open", "bar_high", "bar_low", "bar_close", "source_rows", "distinct_rows"]]).all(axis=1)
            & price.bar_open.gt(0) & price.bar_high.gt(0) & price.bar_low.gt(0) & price.bar_close.gt(0)
            & price.source_rows.eq(5) & price.distinct_rows.eq(5)
            & price.coherent.eq(True) & price.first_ts.eq(expected_first) & price.last_ts.eq(expected_last)
        )
        oi_ok = (
            np.isfinite(inventory[["sum_open_interest", "duplicate_count"]]).all(axis=1)
            & inventory.sum_open_interest.gt(0) & inventory.duplicate_count.eq(1)
        )
        valid = bool(len(price) == 288 and len(inventory) == 289 and price_ok.all() and oi_ok.all())
        day_return = variation = lead = contemporaneous = directional_lead = directional_contemporaneous = math.nan
        if valid:
            returns = np.log(price.bar_close.to_numpy(float) / price.bar_open.to_numpy(float))
            oi_changes = np.diff(np.log(inventory.sum_open_interest.to_numpy(float)))
            day_return = float(np.log(price.bar_close.iloc[-1] / price.bar_open.iloc[0]))
            variation = float(np.sqrt(np.square(returns).sum()))
            lead = _finite_correlation(oi_changes[:-1], returns[1:])
            contemporaneous = _finite_correlation(oi_changes, returns)
            valid = bool(day_return != 0 and variation > 0 and np.isfinite([day_return, variation, lead, contemporaneous]).all())
            if valid:
                directional_lead = float(np.sign(day_return) * lead)
                directional_contemporaneous = float(np.sign(day_return) * contemporaneous)
            else:
                day_return = variation = lead = contemporaneous = directional_lead = directional_contemporaneous = math.nan
        rows.append({
            "source_day": source_day, "feature_available_time": source_day + pd.Timedelta(days=1),
            "source_valid": valid, "day_return": day_return, "realized_variation": variation,
            "lead_correlation": lead, "directional_lead_score": directional_lead,
            "contemporaneous_correlation": contemporaneous,
            "directional_contemporaneous_score": directional_contemporaneous,
        })
    frame = pd.DataFrame(rows)
    frame["lead_score_rank"] = strict_prior_midrank(frame.directional_lead_score.where(frame.source_valid))
    frame["contemporaneous_score_rank"] = strict_prior_midrank(frame.directional_contemporaneous_score.where(frame.source_valid))
    frame["variation_rank"] = strict_prior_midrank(frame.realized_variation.where(frame.source_valid))
    return frame[list(FEATURE_COLUMNS)]


def conditions(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = frame.shift(1) if control == "one_day_stale_features" else frame
    valid = used.source_valid.eq(True) & used.day_return.ne(0)
    lead_gate = pd.Series(True, index=frame.index) if control == "no_lead_score_gate" else used.lead_score_rank.ge(0.75)
    if control == "contemporaneous_correlation":
        lead_gate = used.contemporaneous_score_rank.ge(0.75)
    variation_gate = pd.Series(True, index=frame.index) if control == "no_variation_gate" else used.variation_rank.ge(0.65)
    active = valid & lead_gate & variation_gate
    side = np.sign(used.day_return).fillna(0).astype(int)
    if control == "direction_flip": side = -side
    elif control == "same_clock_forced_long": side = pd.Series(1, index=frame.index)
    return active, side, used


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides, used = conditions(features, control); rows: list[dict[str, Any]] = []
    for index in features.index[active]:
        decision = pd.Timestamp(features.at[index, "source_day"]) + pd.Timedelta(days=1)
        entry = decision + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=12)
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        source = used.loc[index]
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "source_day": source.source_day, "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(sides.at[index]), **{column: source[column] for column in FEATURE_COLUMNS[3:]},
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    rows = clock[clock.split.eq(split)]
    if rows.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(rows.side.eq(1).sum()); shorts = int(rows.side.eq(-1).sum())
    months = pd.to_datetime(rows.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(rows), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(rows), "max_month_share": int(months.max()) / len(rows)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA: raise RuntimeError("HVOILSR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    bars, oi = load_sources(); features = build_features(bars, oi); primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True); CLOCK.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES); _write_gzip_csv(primary, CLOCK)
    for name, value in controls.items(): _write_gzip_csv(value, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "hvoilsr_12_sources_v1",
        "query_sha256": {"bars": hashlib.sha256(BAR_QUERY.encode()).hexdigest(), "oi": hashlib.sha256(OI_QUERY.encode()).hexdigest()},
        "tables": ["bars_binance", "open_interest_binance"], "window": [RAW_START.isoformat(), END.isoformat()],
        "physical_rows": {"bars_1m": int(pd.to_numeric(bars.source_rows).sum()), "oi_5m": len(oi)},
        "features": {"path": str(FEATURES), "sha256": sha(FEATURES), "rows": len(features), "valid_rows": int(features.source_valid.sum())},
        "outcomes_opened": False, "gross9_rows_opened": False, "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": chash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}; checks: dict[str, bool] = {}
    for name, item in support.items():
        checks[f"{name}_minimum_events"] = item["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = item["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = item["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvoilsr_12_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(value), "promotion_authorized": False} for name, value in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": chash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
