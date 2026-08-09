"""Build source-only AARMR-8 clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_adjacent_auction_range_migration_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "0444afeb6c0742f4217016b7712788e72a4806fb71c3df04e6b254ea46b0d4ce"
START = pd.Timestamp("2023-04-01T00:00Z")
END = pd.Timestamp("2026-08-01T00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00Z"), pd.Timestamp("2024-01-01T00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2025-01-01T00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00Z"), pd.Timestamp("2026-01-01T00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("range_tail_only", "low_overlap_only", "one_hour_stale_geometry", "midpoint_fade")
QUERY = """SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS bar_time,(array_agg(open ORDER BY ts))[1] AS bar_open,max(high) AS bar_high,min(low) AS bar_low,(array_agg(close ORDER BY ts DESC))[1] AS bar_close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close) AND low<=least(open,close) AND high>=low) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY bar_time ORDER BY bar_time"""
SOURCE_DIR = Path("data/adjacent_auction_range_migration_relay_sources_2023_2026")
FEATURES = SOURCE_DIR / "hourly_adjacent_ranges.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/adjacent_auction_range_migration_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/adjacent_auction_range_migration_relay_controls_2023_2026")
RESULT = Path("results/adjacent_auction_range_migration_relay_support_2026-08-09.json")
FEATURE_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "previous_range",
    "current_range", "overlap_ratio", "midpoint_displacement",
    "current_range_rank", "overlap_rank",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", *FEATURE_COLUMNS[3:],
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 1440, minimum: int = 960) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float); history: list[float] = []
    for index, current in pd.to_numeric(values, errors="coerce").items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (np.sum(array < current) + 0.5 * np.sum(array == current)) / len(array)
        if math.isfinite(current):
            history.append(float(current))
    return output


def engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_source() -> pd.DataFrame:
    from sqlalchemy import text
    db = engine()
    try:
        with db.connect() as connection:
            return pd.read_sql_query(text(QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()})
    finally:
        db.dispose()


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    required = ["bar_time", "bar_open", "bar_high", "bar_low", "bar_close", "source_rows", "distinct_rows", "first_ts", "last_ts", "coherent"]
    if not set(required).issubset(raw.columns):
        raise ValueError("AARMR schema drift")
    frame = raw[required].copy()
    for column in ("bar_time", "first_ts", "last_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    for column in ("bar_open", "bar_high", "bar_low", "bar_close", "source_rows", "distinct_rows"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("bar_time", kind="mergesort").set_index("bar_time")
    rows = []
    for decision in pd.date_range(START + pd.Timedelta(hours=2), END, freq="1h", inclusive="left"):
        expected = pd.date_range(decision - pd.Timedelta(hours=2), decision, freq="5min", inclusive="left")
        window = frame.reindex(expected)
        prices = window[["bar_open", "bar_high", "bar_low", "bar_close"]]
        valid = bool(
            np.isfinite(window[["bar_open", "bar_high", "bar_low", "bar_close", "source_rows", "distinct_rows"]]).all(axis=1).all()
            and prices.gt(0).all(axis=1).all()
            and window.source_rows.eq(5).all() and window.distinct_rows.eq(5).all()
            and window.coherent.fillna(False).astype(bool).all()
            and window.first_ts.equals(pd.Series(expected, index=expected))
            and window.last_ts.equals(pd.Series(expected + pd.Timedelta(minutes=4), index=expected))
        )
        if valid:
            previous, current = window.iloc[:12], window.iloc[12:]
            low_previous, high_previous = float(previous.bar_low.min()), float(previous.bar_high.max())
            low_current, high_current = float(current.bar_low.min()), float(current.bar_high.max())
            previous_range, current_range = high_previous - low_previous, high_current - low_current
            overlap = max(0.0, min(high_previous, high_current) - max(low_previous, low_current))
            overlap_ratio = overlap / min(previous_range, current_range) if min(previous_range, current_range) > 0 else np.nan
            displacement = 0.5 * (high_current + low_current - high_previous - low_previous)
            valid = bool(
                np.isfinite([previous_range, current_range, overlap_ratio, displacement]).all()
                and previous_range > 0 and current_range > 0
                and 0 <= overlap_ratio <= 1 + 1e-12 and displacement != 0
            )
            overlap_ratio = float(np.clip(overlap_ratio, 0, 1)) if valid else np.nan
        else:
            previous_range = current_range = overlap_ratio = displacement = np.nan
        rows.append({
            "decision_time": decision, "feature_available_time": decision, "source_valid": valid,
            "previous_range": previous_range, "current_range": current_range,
            "overlap_ratio": overlap_ratio, "midpoint_displacement": displacement,
        })
    output = pd.DataFrame(rows)
    output["current_range_rank"] = strict_prior_midrank(output.current_range.where(output.source_valid))
    output["overlap_rank"] = strict_prior_midrank(output.overlap_ratio.where(output.source_valid))
    return output[list(FEATURE_COLUMNS)]


def onset(state: pd.Series) -> pd.Series:
    return state.fillna(False) & ~state.shift(1, fill_value=False)


def active_and_side(features: pd.DataFrame, control: str = "primary"):
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    primary = features.source_valid & features.current_range_rank.ge(0.80) & features.overlap_rank.le(0.20)
    displacement = features.midpoint_displacement
    if control == "range_tail_only":
        state = features.source_valid & features.current_range_rank.ge(0.80)
    elif control == "low_overlap_only":
        state = features.source_valid & features.overlap_rank.le(0.20)
    elif control == "one_hour_stale_geometry":
        state = primary.shift(1, fill_value=False); displacement = displacement.shift(1)
    else:
        state = primary
    side = np.sign(displacement)
    if control == "midpoint_fade":
        side = -side
    active = onset(state) & pd.Series(displacement, index=features.index).ne(0)
    return active, pd.Series(side, index=features.index).astype("Int64").fillna(0).astype(int)


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides = active_and_side(features, control); rows = []; reserved = None
    for index in features.index[active & sides.ne(0)]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=8)
        if reserved is not None and entry < reserved:
            continue
        split = next((name for name, (left, right) in SPLITS.items() if entry >= left and exit_time <= right), None)
        if split is None:
            continue
        reserved = exit_time
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "decision_time": decision, "feature_available_time": features.at[index, "feature_available_time"],
            "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
            **{column: features.at[index, column] for column in CLOCK_COLUMNS[8:]},
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    frame = clock[clock.split.eq(split)]
    if frame.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(frame.side.eq(1).sum()); shorts = int(frame.side.eq(-1).sum())
    months = pd.to_datetime(frame.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(frame), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(frame), "max_month_share": int(months.max()) / len(frame)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("AARMR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    raw = load_source(); features = build_features(raw); primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES); _write_gzip_csv(primary, CLOCK)
    for name, clock in controls.items():
        _write_gzip_csv(clock, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "aarmr_8_source_v1", "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "window": [START.isoformat(), END.isoformat()], "physical_rows": int(raw.source_rows.sum()),
        "aggregate_rows": len(raw), "features": {"path": str(FEATURES), "sha256": sha(FEATURES), "rows": len(features)},
        "completed_preentry_sources_opened": True, "execution_prices_opened": False,
        "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "rv20_opened": False,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}; checks = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.2
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "aarmr_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "execution_prices_opened": False,
        "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "rv20_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(clock), "promotion_authorized": False} for name, clock in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
