"""Materialize outcome-blind source support for frozen HVTDS-S9-24."""
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

from training import preregister_high_volatility_tom_demark_setup_completion_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_tom_demark_setup_completion_relay_support.py")
PREREG_SHA = "9ec7e0e26aceb21cabf1d1d854950bf83349ea6721d6b8e6458d713c12682613"
SOURCE_DIR = Path("data/high_volatility_tom_demark_setup_completion_relay_sources_2023_2026")
FIVE_MINUTE_PANEL = SOURCE_DIR / "causal_five_minute_bars.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "four_hour_setup_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_tom_demark_setup_completion_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_tom_demark_setup_completion_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_tom_demark_setup_completion_relay_support_2026-08-12.json")
START = pd.Timestamp("2020-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_variation_gate", "raw_four_bar_reversal", "one_bar_stale_completion", "direction_flip")
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "entry_time", "exit_time", "side",
    "setup_side", "four_bar_relation", "btc_realized_variation", "btc_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g", date_format="%Y-%m-%dT%H:%M:%SZ")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(text.encode())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())


def strict_prior_midrank(values: pd.Series, valid: pd.Series, lookback: int = 180, minimum: int = 120) -> pd.Series:
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
    return """SELECT date_bin(interval '5 minutes', ts, timestamptz '2020-01-01 00:00:00+00') AS bar_time,
count(*) AS source_rows, count(DISTINCT ts) AS distinct_timestamps, min(ts) AS first_ts, max(ts) AS last_ts,
(array_agg(open ORDER BY ts))[1] AS open, max(high) AS high, min(low) AS low,
(array_agg(close ORDER BY ts DESC))[1] AS close,
bool_and(open > 0 AND high > 0 AND low > 0 AND close > 0 AND low <= least(open, close)
         AND high >= greatest(open, close) AND high >= low) AS coherent
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts >= '2020-01-01T00:00:00Z' AND ts < '2026-08-01T00:00:00Z'
GROUP BY 1 ORDER BY 1"""


def load_five_minute_bars() -> tuple[pd.DataFrame, str]:
    from sqlalchemy import text

    query = source_query()
    engine = postgres_engine()
    try:
        raw = pd.read_sql_query(text(query), engine)
    finally:
        engine.dispose()
    raw.bar_time = pd.to_datetime(raw.bar_time, utc=True, errors="raise")
    if raw.bar_time.duplicated().any() or not raw.bar_time.is_monotonic_increasing:
        raise RuntimeError("HVTDS five-minute aggregation identity drift")
    grid = pd.DataFrame({"bar_time": pd.date_range(START, END, inclusive="left", freq="5min")})
    frame = grid.merge(raw, on="bar_time", how="left", validate="one_to_one")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["valid"] = frame.source_rows.eq(5) & frame.distinct_timestamps.eq(5) & frame.coherent.eq(True)
    frame["valid"] &= pd.to_datetime(frame.first_ts, utc=True).eq(frame.bar_time)
    frame["valid"] &= pd.to_datetime(frame.last_ts, utc=True).eq(frame.bar_time + pd.Timedelta(minutes=4))
    frame["valid"] &= np.isfinite(frame[["open", "high", "low", "close"]]).all(axis=1)
    frame["valid"] &= frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
    frame["log_return_sq"] = np.where(frame.valid, np.square(np.log(frame.close / frame.open)), np.nan)
    frame["variation_24h"] = frame.log_return_sq.rolling(288, min_periods=288).sum().pow(0.5)
    return frame, query


def derive_setup_states(five: pd.DataFrame) -> pd.DataFrame:
    work = five.copy()
    work["four_hour_start"] = work.bar_time.dt.floor("4h")
    rows: list[dict[str, Any]] = []
    for start, group in work.groupby("four_hour_start", sort=True):
        expected = pd.date_range(start, start + pd.Timedelta(hours=4), inclusive="left", freq="5min")
        complete = len(group) == 48 and group.bar_time.reset_index(drop=True).equals(pd.Series(expected, name="bar_time"))
        valid = bool(complete and group.valid.all())
        rows.append({
            "decision_time": start + pd.Timedelta(hours=4),
            "four_hour_valid": valid,
            "four_hour_open": float(group.open.iloc[0]) if valid else np.nan,
            "four_hour_high": float(group.high.max()) if valid else np.nan,
            "four_hour_low": float(group.low.min()) if valid else np.nan,
            "four_hour_close": float(group.close.iloc[-1]) if valid else np.nan,
            "btc_realized_variation": float(group.variation_24h.iloc[-1]) if valid else np.nan,
        })
    frame = pd.DataFrame(rows)
    five_valid = frame.four_hour_valid.rolling(5, min_periods=5).sum().eq(5)
    lag_close = frame.four_hour_close.shift(4)
    frame["four_bar_relation"] = np.where(
        five_valid & frame.four_hour_close.gt(lag_close), 1,
        np.where(five_valid & frame.four_hour_close.lt(lag_close), -1, 0),
    ).astype(int)
    setup_side: list[int] = []
    active_relation = 0
    count = 0
    previous_relation = 0
    for valid, relation in zip(five_valid, frame.four_bar_relation, strict=True):
        event = 0
        if not bool(valid) or relation == 0:
            active_relation = count = previous_relation = 0
            setup_side.append(event)
            continue
        if active_relation:
            if relation == active_relation:
                count += 1
                if count == 9:
                    event = -relation
                    active_relation = count = 0
            else:
                active_relation = count = 0
        if not active_relation and previous_relation == -relation:
            active_relation, count = relation, 1
        previous_relation = relation
        setup_side.append(event)
    frame["setup_side"] = setup_side
    rank_valid = five_valid & np.isfinite(frame.btc_realized_variation) & frame.btc_realized_variation.gt(0)
    frame["btc_variation_rank"] = strict_prior_midrank(frame.btc_realized_variation, rank_valid)
    return frame


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    side = features.setup_side.copy()
    if control == "raw_four_bar_reversal":
        side = -features.four_bar_relation
    elif control == "one_bar_stale_completion":
        side = side.shift(1, fill_value=0)
    elif control == "direction_flip":
        side = -side
    eligible = side.ne(0) & features.btc_variation_rank.ge(0.65)
    if control == "no_variation_gate":
        eligible = side.ne(0)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in features.index[eligible]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        source_index = index - 1 if control == "one_bar_stale_completion" else index
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "decision_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(side.at[index]), "setup_side": int(features.at[source_index, "setup_side"]),
            "four_bar_relation": int(features.at[source_index, "four_bar_relation"]),
            "btc_realized_variation": float(features.at[index, "btc_realized_variation"]),
            "btc_variation_rank": float(features.at[index, "btc_variation_rank"]),
        })
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
        raise RuntimeError("HVTDS preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    five, query = load_five_minute_bars()
    features = derive_setup_states(five)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    write_gzip_csv(five, FIVE_MINUTE_PANEL)
    write_gzip_csv(features, FEATURE_PANEL)
    write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "hvtds_s9_24_sources_v1", "btc_query": query,
        "source_counts": {"five_minute_rows": len(five), "valid_five_minute_rows": int(five.valid.sum()), "four_hour_decisions": len(features), "setup_completions": int(features.setup_side.ne(0).sum())},
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "outputs": {"five_minute_panel": {"path": str(FIVE_MINUTE_PANEL), "sha256": sha(FIVE_MINUTE_PANEL), "rows": len(five)}, "feature_panel": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)}},
        "candidate_outcomes_opened": False, "execution_prices_opened": False, "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support_values = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support_values.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvtds_s9_24_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support_values, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
