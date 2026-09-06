"""Materialize outcome-blind source support for frozen FARR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_funding_acceleration_reversal_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_funding_acceleration_reversal_relay_support.py")
PREREG_SHA = "da543375cf42c905c3c193dcdab6c6fa1f8f498bc4ba53594072ec35fb93ef64"
SOURCE_DIR = Path("data/funding_acceleration_reversal_relay_sources_2023_2026")
FEATURES = SOURCE_DIR / "farr_preentry_features.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/funding_acceleration_reversal_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/funding_acceleration_reversal_relay_controls_2023_2026")
RESULT = Path("results/funding_acceleration_reversal_relay_support_2026-08-09.json")
BTC_START = pd.Timestamp("2022-12-31T08:00:00Z")
SETTLEMENT_START = pd.Timestamp("2023-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_volatility_gate", "no_change_tail", "funding_level", "one_day_stale_change", "direction_flip")
COLUMNS = (
    "candidate", "control", "split", "settlement_date", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side", "funding_rate_00",
    "funding_rate_08", "funding_change", "absolute_change_rank",
    "btc_realized_variation", "btc_variation_rank",
)
BTC_QUERY = (
    "SELECT ts,open,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' "
    "AND ts>=:start AND ts<:end ORDER BY ts"
)
SETTLEMENT_QUERY = (
    "SELECT funding_time,funding_rate FROM funding_rates_binance WHERE symbol='BTCUSDT' "
    "AND funding_time>=:start AND funding_time<:end ORDER BY funding_time"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_prior_rank(values: pd.Series, lookback: int = 252, minimum: int = 126) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (
                np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)
            ) / len(array)
        if np.isfinite(current):
            history.append(float(current))
    return output


def engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text

    connection_engine = engine()
    try:
        btc = pd.read_sql_query(
            text(BTC_QUERY), connection_engine,
            params={"start": BTC_START.to_pydatetime(), "end": SOURCE_END.to_pydatetime()},
        )
        settlements = pd.read_sql_query(
            text(SETTLEMENT_QUERY), connection_engine,
            params={"start": SETTLEMENT_START.to_pydatetime(), "end": SOURCE_END.to_pydatetime()},
        )
    finally:
        connection_engine.dispose()
    if btc.columns.tolist() != ["ts", "open", "close"]:
        raise RuntimeError("FARR BTC schema drift")
    if settlements.columns.tolist() != ["funding_time", "funding_rate"]:
        raise RuntimeError("FARR settlement schema drift")
    btc["ts"] = pd.to_datetime(btc.ts, utc=True, errors="raise")
    btc.sort_values("ts", inplace=True)
    btc.reset_index(drop=True, inplace=True)
    if btc.ts.duplicated().any():
        raise RuntimeError("FARR BTC duplicate timestamp")
    for column in ("open", "close"):
        btc[column] = pd.to_numeric(btc[column], errors="coerce")
    if not np.isfinite(btc[["open", "close"]]).all(axis=None) or not btc[["open", "close"]].gt(0).all(axis=None):
        raise RuntimeError("FARR invalid BTC price")
    expected = pd.date_range(BTC_START, SOURCE_END, freq="1min", inclusive="left")
    if len(btc) != len(expected) or not btc.ts.equals(pd.Series(expected, name="ts")):
        raise RuntimeError("FARR BTC source not exact 1m grid")
    settlements["funding_time"] = pd.to_datetime(settlements.funding_time, utc=True, errors="raise")
    settlements["funding_rate"] = pd.to_numeric(settlements.funding_rate, errors="coerce")
    if settlements.funding_time.duplicated().any():
        raise RuntimeError("FARR duplicate settlement timestamp")
    if not np.isfinite(settlements.funding_rate).all():
        raise RuntimeError("FARR invalid funding rate")
    return btc.set_index("ts"), settlements


def build_features(btc: pd.DataFrame, settlements: pd.DataFrame) -> pd.DataFrame:
    rows = []
    minute = settlements.funding_time.dt.floor("min")
    selected = settlements[(minute.dt.minute == 0) & minute.dt.hour.isin([0, 8])].copy()
    selected["boundary"] = minute[selected.index]
    if selected.boundary.duplicated().any():
        raise RuntimeError("FARR multiple rows for one funding boundary")
    selected["settlement_date"] = selected.boundary.dt.floor("D")
    by_boundary = selected.set_index("boundary")
    for day in pd.date_range("2023-01-01", "2026-07-31", freq="D", tz="UTC"):
        zero = day
        decision = day + pd.Timedelta(hours=8)
        if zero not in by_boundary.index or decision not in by_boundary.index:
            continue
        row_00 = by_boundary.loc[zero]
        row_08 = by_boundary.loc[decision]
        if not (zero <= row_00.funding_time < zero + pd.Timedelta(minutes=1)):
            raise RuntimeError("FARR late 00:00 funding timestamp")
        if not (decision <= row_08.funding_time < decision + pd.Timedelta(minutes=1)):
            raise RuntimeError("FARR late 08:00 funding timestamp")
        window = btc.loc[decision - pd.Timedelta(hours=24): decision - pd.Timedelta(minutes=1)]
        if len(window) != 1440:
            continue
        rate_00 = float(row_00.funding_rate)
        rate_08 = float(row_08.funding_rate)
        change = rate_08 - rate_00
        variation = float(np.sqrt(np.square(np.log(window.close.to_numpy() / window.open.to_numpy())).sum()))
        rows.append({
            "settlement_date": decision.floor("D"), "decision_time": decision,
            "feature_available_time": pd.Timestamp(row_08.funding_time),
            "funding_rate_00": rate_00, "funding_rate_08": rate_08,
            "funding_change": change,
            "btc_realized_variation": variation,
        })
    features = pd.DataFrame(rows)
    features["absolute_change_rank"] = strict_prior_rank(features.funding_change.abs())
    features["btc_variation_rank"] = strict_prior_rank(features.btc_realized_variation)
    return features


def signal(features: pd.DataFrame, control: str) -> pd.Series:
    change = features.funding_change
    side = -np.sign(change).astype("Int64").fillna(0).astype(int)
    eligible = change.ne(0) & features.absolute_change_rank.ge(0.75) & features.btc_variation_rank.ge(0.65)
    if control == "no_volatility_gate":
        eligible = change.ne(0) & features.absolute_change_rank.ge(0.75)
    elif control == "no_change_tail":
        eligible = change.ne(0) & features.btc_variation_rank.ge(0.65)
    elif control == "funding_level":
        level = features.funding_rate_08
        side = -np.sign(level).astype("Int64").fillna(0).astype(int)
        eligible = level.ne(0) & features.btc_variation_rank.ge(0.65)
    elif control == "one_day_stale_change":
        change = change.shift(1)
        side = -np.sign(change).astype("Int64").fillna(0).astype(int)
        eligible = change.ne(0) & features.absolute_change_rank.shift(1).ge(0.75) & features.btc_variation_rank.ge(0.65)
    side = side.where(eligible, 0)
    return -side if control == "direction_flip" else side


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    sides = signal(features, control)
    rows = []
    next_allowed = None
    for index in features.index[sides.ne(0)]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        available = pd.Timestamp(features.at[index, "feature_available_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=6)
        if available >= entry:
            continue
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({
            "candidate": "FARR-6", "control": control, "split": split,
            "settlement_date": pd.Timestamp(features.at[index, "settlement_date"]),
            "decision_time": decision, "feature_available_time": available,
            "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
            "funding_rate_00": float(features.at[index, "funding_rate_00"]),
            "funding_rate_08": float(features.at[index, "funding_rate_08"]),
            "funding_change": float(features.at[index, "funding_change"]),
            "absolute_change_rank": float(features.at[index, "absolute_change_rank"]),
            "btc_realized_variation": float(features.at[index, "btc_realized_variation"]),
            "btc_variation_rank": float(features.at[index, "btc_variation_rank"]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)].copy()
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    entries = pd.to_datetime(selected.entry_time, utc=True)
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    return {
        "events": len(selected), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("FARR prereg drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    btc, settlements = load_sources()
    features = build_features(btc, settlements)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "farr_6_sources_v1",
        "queries": {"btc": BTC_QUERY, "settlement": SETTLEMENT_QUERY},
        "windows": {"btc": [BTC_START.isoformat(), SOURCE_END.isoformat()], "settlement": [SETTLEMENT_START.isoformat(), SOURCE_END.isoformat()]},
        "rows": {"btc": len(btc), "settlements": len(settlements), "features": len(features)},
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "feature_output": {"path": str(FEATURES), "sha256": sha(FEATURES)},
        "funding_rate_signal_column_opened": True, "mark_price_signal_column_opened": False,
        "candidate_outcomes_opened": False, "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": chash(source_core)}
    MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.2
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "farr_6_source_support_v1", "policy_id": "FARR-6",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "funding_rate_signal_column_opened": True,
        "mark_price_signal_column_opened": False,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": chash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    outcome = run()
    print(json.dumps({"passed": outcome["support_passed"], "support": outcome["support"]}, indent=2))
