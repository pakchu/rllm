"""Materialize outcome-blind source support for frozen HVRLXC-12."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_realized_leverage_cross_moment_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_realized_leverage_cross_moment_relay_support.py")
PREREG_SHA = "2d6d0e27aa90f054f0c3955fb7c1862d48128fcae045d9b8ce6886b54044f9e4"
SOURCE_DIR = Path("data/high_volatility_realized_leverage_cross_moment_relay_sources_2023_2026")
FIVE_PANEL = SOURCE_DIR / "causal_five_minute_moments.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "four_hour_preentry_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_realized_leverage_cross_moment_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_realized_leverage_cross_moment_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_realized_leverage_cross_moment_relay_support_2026-08-12.json")
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_variation_gate", "leverage_direction_flip", "one_decision_stale_cross_moment", "contemporaneous_return_variance_moment", "same_clock_forced_long")
CLOCK_COLUMNS = ("candidate", "control", "split", "decision_time", "entry_time", "exit_time", "side", "leverage_cross_moment", "magnitude_rank", "btc_realized_variation", "btc_variation_rank")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g", date_format="%Y-%m-%dT%H:%M:%SZ")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(text.encode())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())


def strict_prior_midrank(values: pd.Series, valid: pd.Series, lookback: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if bool(valid.at[index]) and np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float)
            output.at[index] = (np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)) / len(array)
        if bool(valid.at[index]) and np.isfinite(current):
            history.append(float(current))
    return output


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def source_query() -> str:
    return """SELECT date_bin(interval '5 minutes', ts, timestamptz '2023-01-01 00:00:00+00') AS bar_time,
count(*) source_rows, count(DISTINCT ts) distinct_timestamps, min(ts) first_ts, max(ts) last_ts,
(array_agg(open ORDER BY ts))[1] open, max(high) high, min(low) low, (array_agg(close ORDER BY ts DESC))[1] close,
bool_and(open>0 AND high>0 AND low>0 AND close>0 AND low<=least(open,close) AND high>=greatest(open,close) AND high>=low) coherent
FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m'
AND ts>='2023-01-01T00:00:00Z' AND ts<'2026-08-01T00:00:00Z'
GROUP BY 1 ORDER BY 1"""


def load_five_minute() -> tuple[pd.DataFrame, str]:
    from sqlalchemy import text
    query = source_query()
    engine = postgres_engine()
    try:
        raw = pd.read_sql_query(text(query), engine)
    finally:
        engine.dispose()
    raw.bar_time = pd.to_datetime(raw.bar_time, utc=True, errors="raise")
    if raw.bar_time.duplicated().any() or not raw.bar_time.is_monotonic_increasing:
        raise RuntimeError("HVRLXC five-minute identity drift")
    grid = pd.DataFrame({"bar_time": pd.date_range(START, END, inclusive="left", freq="5min")})
    frame = grid.merge(raw, on="bar_time", how="left", validate="one_to_one")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["valid"] = frame.source_rows.eq(5) & frame.distinct_timestamps.eq(5) & frame.coherent.eq(True)
    frame["valid"] &= pd.to_datetime(frame.first_ts, utc=True).eq(frame.bar_time)
    frame["valid"] &= pd.to_datetime(frame.last_ts, utc=True).eq(frame.bar_time + pd.Timedelta(minutes=4))
    frame["valid"] &= np.isfinite(frame[["open", "high", "low", "close"]]).all(axis=1) & frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
    frame["return"] = np.where(frame.valid, np.log(frame.close / frame.open), np.nan)
    return frame, query


def derive_moments(five: pd.DataFrame) -> pd.DataFrame:
    frame = five.copy()
    returns = frame["return"]
    complete = frame.valid.rolling(288, min_periods=288).sum().eq(288)
    pair_numerator = (returns.shift(1) * returns.pow(2)).rolling(287, min_periods=287).sum()
    left_second = returns.shift(1).pow(2).rolling(287, min_periods=287).sum()
    next_fourth = returns.pow(4).rolling(287, min_periods=287).sum()
    denominator = np.sqrt(left_second * next_fourth)
    frame["leverage_cross_moment"] = np.where(complete & denominator.gt(0), pair_numerator / denominator, np.nan)
    contemp_denominator = np.sqrt(returns.pow(2).rolling(288, min_periods=288).sum() * returns.pow(4).rolling(288, min_periods=288).sum())
    frame["contemporaneous_moment"] = np.where(complete & contemp_denominator.gt(0), returns.pow(3).rolling(288, min_periods=288).sum() / contemp_denominator, np.nan)
    frame["btc_realized_variation"] = np.where(complete, returns.pow(2).rolling(288, min_periods=288).sum().pow(0.5), np.nan)
    decisions = frame[(frame.bar_time + pd.Timedelta(minutes=5)).dt.minute.eq(0) & (frame.bar_time + pd.Timedelta(minutes=5)).dt.hour.mod(4).eq(0)].copy()
    decisions["decision_time"] = decisions.bar_time + pd.Timedelta(minutes=5)
    decisions.reset_index(drop=True, inplace=True)
    valid = np.isfinite(decisions.leverage_cross_moment) & np.isfinite(decisions.btc_realized_variation) & decisions.btc_realized_variation.gt(0)
    decisions["magnitude_rank"] = strict_prior_midrank(decisions.leverage_cross_moment.abs(), valid)
    decisions["btc_variation_rank"] = strict_prior_midrank(decisions.btc_realized_variation, valid)
    contemp_valid = np.isfinite(decisions.contemporaneous_moment) & np.isfinite(decisions.btc_realized_variation) & decisions.btc_realized_variation.gt(0)
    decisions["contemporaneous_magnitude_rank"] = strict_prior_midrank(decisions.contemporaneous_moment.abs(), contemp_valid)
    return decisions[["decision_time", "leverage_cross_moment", "contemporaneous_moment", "magnitude_rank", "contemporaneous_magnitude_rank", "btc_realized_variation", "btc_variation_rank"]]


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    moment = features.leverage_cross_moment.copy()
    magnitude_rank = features.magnitude_rank.copy()
    side = np.sign(moment).fillna(0).astype(int)
    if control == "leverage_direction_flip":
        side = -side
    elif control == "one_decision_stale_cross_moment":
        moment = moment.shift(1)
        magnitude_rank = magnitude_rank.shift(1)
        side = np.sign(moment).fillna(0).astype(int)
    elif control == "contemporaneous_return_variance_moment":
        moment = features.contemporaneous_moment.copy()
        magnitude_rank = features.contemporaneous_magnitude_rank.copy()
        side = np.sign(moment).fillna(0).astype(int)
    eligible = side.ne(0) & magnitude_rank.ge(0.80) & features.btc_variation_rank.ge(0.65)
    if control == "no_variation_gate":
        eligible = side.ne(0) & magnitude_rank.ge(0.80)
    if control == "same_clock_forced_long":
        side = side.where(~eligible, 1)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in features.index[eligible]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({"candidate": "HVRLXC-12", "control": control, "split": split, "decision_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "leverage_cross_moment": float(moment.at[index]), "magnitude_rank": float(magnitude_rank.at[index]), "btc_realized_variation": float(features.at[index, "btc_realized_variation"]), "btc_variation_rank": float(features.at[index, "btc_variation_rank"])})
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock.split.eq(split)].copy()
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    entries = pd.to_datetime(subset.entry_time, utc=True)
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVRLXC preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    five, query = load_five_minute()
    features = derive_moments(five)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    write_gzip_csv(five, FIVE_PANEL)
    write_gzip_csv(features, FEATURE_PANEL)
    write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {"protocol_version": "hvrlxc_12_sources_v1", "btc_query": query, "source_counts": {"five_minute_rows": len(five), "valid_five_minute_rows": int(five.valid.sum()), "four_hour_decisions": len(features), "finite_cross_moments": int(np.isfinite(features.leverage_cross_moment).sum())}, "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)}, "outputs": {"five_minute_panel": {"path": str(FIVE_PANEL), "sha256": sha(FIVE_PANEL), "rows": len(five)}, "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)}}, "candidate_outcomes_opened": False, "execution_prices_opened": False, "no_imputation": True}
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support_values = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support_values.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {"protocol_version": "hvrlxc_12_source_support_v1", "policy_id": "HVRLXC-12", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()}, "support": support_values, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
