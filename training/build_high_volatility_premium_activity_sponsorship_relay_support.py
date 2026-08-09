"""Build outcome-blind source support for frozen HVPASR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_premium_activity_sponsorship_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "d428a7065e47203e0daf9f99d2862280e9d147569fa5b737703338506802824d"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR = Path("data/high_volatility_premium_activity_sponsorship_relay_sources_2023_2026")
FEATURES = SOURCE_DIR / "daily_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_premium_activity_sponsorship_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_premium_activity_sponsorship_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_premium_activity_sponsorship_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_premium_activity_gate",
    "no_btc_variation_gate",
    "one_day_stale_features",
    "direction_flip",
    "same_clock_forced_long",
)
PREMIUM_QUERY = """
WITH minute_path AS (
 SELECT ts, date_trunc('day',ts) AS source_day, close,
        lag(close) OVER (PARTITION BY date_trunc('day',ts) ORDER BY ts) AS previous_close
 FROM bars_binance_premium
 WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
)
SELECT source_day, count(*) AS source_rows, count(DISTINCT ts) AS distinct_timestamps,
       min(ts) AS first_ts, max(ts) AS last_ts,
       sum(abs(close-previous_close)) FILTER (WHERE previous_close IS NOT NULL) AS premium_total_variation,
       count(previous_close) AS adjacent_pairs
FROM minute_path GROUP BY source_day ORDER BY source_day
"""
BTC_QUERY = """
WITH five_minute AS (
 SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS bar_time,
        (array_agg(open ORDER BY ts))[1] AS open, max(high) AS high, min(low) AS low,
        (array_agg(close ORDER BY ts DESC))[1] AS close, count(*) AS source_rows
 FROM bars_binance
 WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
 GROUP BY 1
)
SELECT date_trunc('day',bar_time) AS source_day, count(*) AS bars_5m,
       sum(source_rows) AS source_rows_1m, min(bar_time) AS first_bar, max(bar_time) AS last_bar,
       (array_agg(open ORDER BY bar_time))[1] AS day_open,
       (array_agg(close ORDER BY bar_time DESC))[1] AS day_close,
       sqrt(sum(power(ln(close/open),2))) AS btc_realized_variation,
       bool_and(source_rows=5 AND open>0 AND high>=greatest(open,close,low)
                AND low<=least(open,close,high) AND close>0) AS coherent
FROM five_minute GROUP BY 1 ORDER BY 1
"""
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_day", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side",
    "premium_total_variation", "premium_activity_rank", "btc_return",
    "btc_realized_variation", "btc_variation_rank",
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
        if np.isfinite(current) and len(prior) >= minimum:
            output.at[index] = (np.sum(prior < current) + 0.5 * np.sum(prior == current)) / len(prior)
        if np.isfinite(current):
            history.append(float(current))
    return output


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_daily_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            premium = pd.read_sql_query(text(PREMIUM_QUERY), connection, params={"start": START, "end": END})
            btc = pd.read_sql_query(text(BTC_QUERY), connection, params={"start": START, "end": END})
    finally:
        engine.dispose()
    for frame in (premium, btc):
        frame["source_day"] = pd.to_datetime(frame.source_day, utc=True)
    return premium, btc


def build_features(premium: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    frame = premium.merge(btc, on="source_day", how="inner", validate="one_to_one")
    numeric = (
        (frame.source_rows == 1440)
        & (frame.distinct_timestamps == 1440)
        & (frame.adjacent_pairs == 1439)
        & (frame.bars_5m == 288)
        & (frame.source_rows_1m == 1440)
        & frame.coherent.astype(bool)
        & pd.to_numeric(frame.premium_total_variation, errors="coerce").gt(0)
        & pd.to_numeric(frame.btc_realized_variation, errors="coerce").gt(0)
        & pd.to_numeric(frame.day_open, errors="coerce").gt(0)
        & pd.to_numeric(frame.day_close, errors="coerce").gt(0)
    )
    expected_last = frame.source_day + pd.Timedelta(hours=23, minutes=59)
    expected_last_bar = frame.source_day + pd.Timedelta(hours=23, minutes=55)
    frame["source_valid"] = (
        numeric
        & frame.first_ts.eq(frame.source_day)
        & frame.last_ts.eq(expected_last)
        & frame.first_bar.eq(frame.source_day)
        & frame.last_bar.eq(expected_last_bar)
    )
    frame["btc_return"] = np.log(
        pd.to_numeric(frame.day_close, errors="coerce") / pd.to_numeric(frame.day_open, errors="coerce")
    ).where(frame.source_valid)
    frame["premium_activity_rank"] = strict_prior_midrank(
        pd.to_numeric(frame.premium_total_variation, errors="coerce").where(frame.source_valid)
    )
    frame["btc_variation_rank"] = strict_prior_midrank(
        pd.to_numeric(frame.btc_realized_variation, errors="coerce").where(frame.source_valid)
    )
    return frame.sort_values("source_day").reset_index(drop=True)


def conditions(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = frame.shift(1) if control == "one_day_stale_features" else frame
    valid = used.source_valid.fillna(False).astype(bool) & used.btc_return.ne(0)
    premium_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_premium_activity_gate"
        else used.premium_activity_rank.ge(0.75)
    )
    variation_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_btc_variation_gate"
        else used.btc_variation_rank.ge(0.65)
    )
    active = valid & premium_gate & variation_gate
    side = np.sign(used.btc_return).fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1, index=frame.index)
    return active, side, used


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides, used = conditions(features, control)
    rows: list[dict[str, Any]] = []
    for index in features.index[active]:
        decision = pd.Timestamp(features.at[index, "source_day"]) + pd.Timedelta(days=1)
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        source = used.loc[index]
        rows.append({
            "candidate": "HVPASR-12", "control": control, "split": split,
            "source_day": source.source_day, "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(sides.at[index]),
            "premium_total_variation": float(source.premium_total_variation),
            "premium_activity_rank": float(source.premium_activity_rank),
            "btc_return": float(source.btc_return),
            "btc_realized_variation": float(source.btc_realized_variation),
            "btc_variation_rank": float(source.btc_variation_rank),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    rows = clock[clock.split.eq(split)]
    if rows.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(rows.side.eq(1).sum()), int(rows.side.eq(-1).sum())
    months = pd.to_datetime(rows.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(rows), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(rows), "max_month_share": int(months.max()) / len(rows)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVPASR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    premium, btc = load_daily_sources()
    features = build_features(premium, btc)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True); CLOCK.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES); _write_gzip_csv(primary, CLOCK)
    for name, control in controls.items():
        _write_gzip_csv(control, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "hvpasr_12_sources_v1",
        "queries": {"premium_daily": PREMIUM_QUERY, "btc_daily": BTC_QUERY},
        "tables": ["bars_binance_premium", "bars_binance"],
        "window": [START.isoformat(), END.isoformat()],
        "features": {"path": str(FEATURES), "sha256": sha(FEATURES), "rows": len(features), "valid_rows": int(features.source_valid.sum())},
        "outcomes_opened": False, "gross9_rows_opened": False, "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": chash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {
        check: passed
        for name, item in support.items()
        for check, passed in (
            (f"{name}_minimum_events", item["events"] >= MINIMUM[name]),
            (f"{name}_side_balance", item["minority_side_share"] >= 0.20),
            (f"{name}_month_concentration", item["max_month_share"] <= 0.45),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "hvpasr_12_source_support_v1", "policy_id": "HVPASR-12",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(control), "promotion_authorized": False} for name, control in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": chash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run(); print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
