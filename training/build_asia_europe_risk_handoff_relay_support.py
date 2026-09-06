"""Materialize outcome-blind source support for frozen AERHR-12."""
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

from training import preregister_asia_europe_risk_handoff_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "1620fd6138e4d5d63cd75da5f0598fd312ee591429bd2b0334688dfb6feb348e"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_variation_gate", "any_reversal", "asia_continuation", "direction_flip")
QUERY = """SELECT date_bin('4 hours',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS block_start,(array_agg(open ORDER BY ts))[1] AS block_open,max(high) AS block_high,min(low) AS block_low,(array_agg(close ORDER BY ts DESC))[1] AS block_close,sum(power(ln(close/open),2)) AS minute_variation_sq,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close) AND low<=least(open,close) AND high>=low) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end AND EXTRACT(hour FROM ts)<12 GROUP BY block_start ORDER BY block_start"""
SOURCE_DIR = Path("data/asia_europe_risk_handoff_relay_sources_2023_2026")
FEATURES = SOURCE_DIR / "daily_regional_handoff_features.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/asia_europe_risk_handoff_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/asia_europe_risk_handoff_relay_controls_2023_2026")
RESULT = Path("results/asia_europe_risk_handoff_relay_support_2026-08-10.json")
FEATURE_COLUMNS = (
    "source_day", "decision_time", "feature_available_time", "source_valid",
    "asia_return", "europe_return", "path_variation", "variation_rank", "reversal", "handoff",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_day", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "asia_return", "europe_return", "path_variation",
    "variation_rank", "reversal", "handoff",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_prior_rank(values: pd.Series, lookback: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    output = np.full(len(numeric), np.nan)
    history: list[float] = []
    for index, current in enumerate(numeric):
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output[index] = (np.sum(array < current) + 0.5 * np.sum(array == current)) / len(array)
        if np.isfinite(current):
            history.append(float(current))
    return pd.Series(output, index=values.index)


def engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_source() -> pd.DataFrame:
    from sqlalchemy import text

    connection_engine = engine()
    try:
        with connection_engine.connect() as connection:
            return pd.read_sql_query(
                text(QUERY), connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        connection_engine.dispose()


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    required = [
        "block_start", "block_open", "block_high", "block_low", "block_close",
        "minute_variation_sq", "source_rows", "distinct_rows", "first_ts", "last_ts", "coherent",
    ]
    if not set(required).issubset(raw.columns):
        raise ValueError("AERHR source schema drift")
    frame = raw[required].copy()
    for column in ("block_start", "first_ts", "last_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    numeric = (
        "block_open", "block_high", "block_low", "block_close", "minute_variation_sq",
        "source_rows", "distinct_rows",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("block_start", kind="mergesort").set_index("block_start")
    rows = []
    for day in pd.date_range(START, END, freq="D", inclusive="left"):
        expected = pd.DatetimeIndex([day, day + pd.Timedelta(hours=4), day + pd.Timedelta(hours=8)])
        window = frame.reindex(expected)
        prices = window[["block_open", "block_high", "block_low", "block_close"]]
        valid = bool(
            np.isfinite(window[list(numeric)]).all(axis=1).all()
            and prices.gt(0).all(axis=1).all()
            and window.source_rows.eq(240).all()
            and window.distinct_rows.eq(240).all()
            and window.coherent.fillna(False).astype(bool).all()
            and window.first_ts.equals(pd.Series(expected, index=expected))
            and window.last_ts.equals(pd.Series(expected + pd.Timedelta(minutes=239), index=expected))
        )
        if valid:
            asia_return = float(np.log(window.block_close.iloc[1] / window.block_open.iloc[0]))
            europe_return = float(np.log(window.block_close.iloc[2] / window.block_open.iloc[2]))
            path_variation = float(np.sqrt(window.minute_variation_sq.sum()))
            reversal = bool(asia_return != 0 and europe_return != 0 and np.sign(asia_return) != np.sign(europe_return))
            handoff = bool(reversal and abs(europe_return) >= abs(asia_return))
        else:
            asia_return = europe_return = path_variation = np.nan
            reversal = handoff = False
        rows.append({
            "source_day": day, "decision_time": day + pd.Timedelta(hours=12),
            "feature_available_time": day + pd.Timedelta(hours=12), "source_valid": valid,
            "asia_return": asia_return, "europe_return": europe_return,
            "path_variation": path_variation, "reversal": reversal, "handoff": handoff,
        })
    features = pd.DataFrame(rows)
    features["variation_rank"] = strict_prior_rank(features.path_variation)
    return features[list(FEATURE_COLUMNS)]


def signal(features: pd.DataFrame, control: str = "primary") -> pd.Series:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    variation_ok = features.variation_rank.ge(0.65)
    if control == "no_variation_gate":
        eligible = features.source_valid & features.handoff
        side = np.sign(features.europe_return)
    elif control == "any_reversal":
        eligible = features.source_valid & features.reversal & variation_ok
        side = np.sign(features.europe_return)
    elif control == "asia_continuation":
        eligible = features.source_valid & features.handoff & variation_ok
        side = np.sign(features.asia_return)
    else:
        eligible = features.source_valid & features.handoff & variation_ok
        side = np.sign(features.europe_return) * (-1 if control == "direction_flip" else 1)
    return pd.Series(side, index=features.index).astype("Int64").fillna(0).astype(int).where(eligible, 0)


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    sides = signal(features, control)
    rows = []
    reserved_until = None
    for index in features.index[sides.ne(0)]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None
        )
        if split is None:
            continue
        reserved_until = exit_time
        rows.append({
            "candidate": "AERHR-12", "control": control, "split": split,
            "source_day": features.at[index, "source_day"], "decision_time": decision,
            "feature_available_time": features.at[index, "feature_available_time"],
            "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
            **{column: features.at[index, column] for column in CLOCK_COLUMNS[9:]},
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("AERHR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    raw = load_source()
    features = build_features(raw)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "aerhr_12_source_v1",
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "window": [START.isoformat(), END.isoformat()],
        "physical_rows": int(raw.source_rows.sum()), "aggregate_rows": len(raw),
        "features": {"path": str(FEATURES), "sha256": sha(FEATURES), "rows": len(features)},
        "completed_preentry_sources_opened": True, "execution_prices_opened": False,
        "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "rv20_opened": False,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {
        key: value
        for name, values in support.items()
        for key, value in (
            (f"{name}_minimum_events", values["events"] >= MINIMUM[name]),
            (f"{name}_side_balance", values["minority_side_share"] >= 0.2),
            (f"{name}_month_concentration", values["max_month_share"] <= 0.45),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "aerhr_12_source_support_v1", "policy_id": "AERHR-12",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "execution_prices_opened": False,
        "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "rv20_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False}
            for name, frame in controls.items()
        },
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    outcome = run()
    print(json.dumps({"passed": outcome["support_passed"], "support": outcome["support"]}, indent=2))
