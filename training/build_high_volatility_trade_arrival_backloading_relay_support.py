"""Build source-only support evidence for preregistered HVTAB-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.live_db_features import postgres_url_from_env
from training import preregister_high_volatility_trade_arrival_backloading_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
BLOCK = pd.Timedelta(hours=4)
ENTRY_DELAY = pd.Timedelta(minutes=5)
HOLD = pd.Timedelta(hours=6)
HISTORY = 540
MIN_HISTORY = 360
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
CONTROLS = ("no_variation_gate", "no_arrival_tail", "one_block_stale_features", "direction_flip", "forced_long")
SOURCE_DIR = Path("data/high_volatility_trade_arrival_backloading_sources_2023_2026")
STATES = SOURCE_DIR / "trade_arrival_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_trade_arrival_backloading_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_trade_arrival_backloading_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_trade_arrival_backloading_relay_support_2026-08-10.json")
QUERY = """
SELECT ts,open,high,low,close,number_of_trades
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side",
    "block_return", "late_return", "late_arrival_share", "late_share_rank", "realized_variation", "variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def postgres_engine(env_file: str = ENV_FILE):
    from sqlalchemy import create_engine

    return create_engine(postgres_url_from_env(env_file), connect_args={"connect_timeout": 10})


def query_bars(env_file: str = ENV_FILE) -> pd.DataFrame:
    from sqlalchemy import text

    engine = postgres_engine(env_file)
    try:
        with engine.connect() as connection:
            frame = pd.read_sql_query(
                text(QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()}
            )
    finally:
        engine.dispose()
    return frame


def strict_prior_midrank(values: pd.Series) -> pd.Series:
    array = values.to_numpy(float)
    output = np.full(len(array), np.nan)
    for index, current in enumerate(array):
        if not np.isfinite(current):
            continue
        prior = array[max(0, index - HISTORY):index]
        prior = prior[np.isfinite(prior)]
        if len(prior) < MIN_HISTORY:
            continue
        output[index] = (np.sum(prior < current) + 0.5 * np.sum(prior == current)) / len(prior)
    return pd.Series(output, index=values.index)


def build_states(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, format="mixed")
    if frame.ts.duplicated().any() or not frame.ts.is_monotonic_increasing:
        raise RuntimeError("HVTAB source timestamp contract failed")
    numeric = ["open", "high", "low", "close", "number_of_trades"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["block_start"] = frame.ts.dt.floor("4h")
    rows: list[dict[str, Any]] = []
    for block_start, window in frame.groupby("block_start", sort=True):
        expected_end = block_start + BLOCK - pd.Timedelta(minutes=1)
        coherent = (
            len(window) == 240
            and window.ts.nunique() == 240
            and window.ts.iloc[0] == block_start
            and window.ts.iloc[-1] == expected_end
            and np.isfinite(window[numeric].to_numpy(float)).all()
            and window[["open", "high", "low", "close"]].gt(0).all().all()
            and window.high.ge(window[["open", "close"]].max(axis=1)).all()
            and window.low.le(window[["open", "close"]].min(axis=1)).all()
            and window.high.ge(window.low).all()
            and window.number_of_trades.ge(0).all()
            and window.number_of_trades.eq(np.floor(window.number_of_trades)).all()
            and window.number_of_trades.sum() > 0
        )
        row: dict[str, Any] = {"block_start": block_start, "decision_time": block_start + BLOCK, "source_valid": coherent}
        if coherent:
            closes = window.close.to_numpy(float)
            variation = float(np.sqrt(np.square(np.diff(np.log(closes))).sum()))
            total_trades = float(window.number_of_trades.sum())
            row.update(
                block_return=float(np.log(window.close.iloc[-1] / window.open.iloc[0])),
                late_return=float(np.log(window.close.iloc[-1] / window.open.iloc[-60])),
                late_arrival_share=float(window.number_of_trades.iloc[-60:].sum() / total_trades),
                realized_variation=variation,
            )
        else:
            row.update(block_return=np.nan, late_return=np.nan, late_arrival_share=np.nan, realized_variation=np.nan)
        rows.append(row)
    states = pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)
    states["late_share_rank"] = strict_prior_midrank(states.late_arrival_share.where(states.source_valid))
    states["variation_rank"] = strict_prior_midrank(states.realized_variation.where(states.source_valid))
    return states


def eligibility(states: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series]:
    source_valid = states.source_valid.fillna(False)
    block_return = states.block_return
    late_return = states.late_return
    late_rank = states.late_share_rank
    variation_rank = states.variation_rank
    if control == "one_block_stale_features":
        source_valid = source_valid.shift(1, fill_value=False)
        block_return = block_return.shift(1)
        late_return = late_return.shift(1)
        late_rank = late_rank.shift(1)
        variation_rank = variation_rank.shift(1)
    direction = np.sign(block_return)
    agreement = block_return.ne(0) & late_return.ne(0) & np.sign(late_return).eq(direction)
    arrival = pd.Series(True, index=states.index) if control == "no_arrival_tail" else late_rank.ge(0.75)
    variation = pd.Series(True, index=states.index) if control == "no_variation_gate" else variation_rank.ge(0.65)
    eligible = source_valid & np.isfinite(block_return) & np.isfinite(late_return) & agreement & arrival & variation
    onset = eligible & ~eligible.shift(1, fill_value=False) & source_valid.shift(1, fill_value=False) & states.decision_time.diff().eq(BLOCK)
    return onset, direction


def build_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    onset, direction = eligibility(states, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in states.index[onset]:
        decision = pd.Timestamp(states.at[index, "decision_time"])
        entry = decision + ENTRY_DELAY
        exit_time = entry + HOLD
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        side = int(direction.at[index])
        if control == "direction_flip":
            side = -side
        elif control == "forced_long":
            side = 1
        next_allowed = exit_time
        rows.append({
            "candidate": "HVTAB-6", "control": control, "split": split, "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time, "side": side,
            "block_return": float(states.at[index, "block_return"]), "late_return": float(states.at[index, "late_return"]),
            "late_arrival_share": float(states.at[index, "late_arrival_share"]), "late_share_rank": float(states.at[index, "late_share_rank"]),
            "realized_variation": float(states.at[index, "realized_variation"]), "variation_rank": float(states.at[index, "variation_rank"]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run(env_file: str = ENV_FILE) -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVTAB preregistration manifest drift")
    states = build_states(query_bars(env_file))
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, STATES)
    source_core = {"protocol_version": "hvtab_source_v1", "query": QUERY.strip(), "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "window": [START.isoformat(), END.isoformat()], "outcomes_opened": False, "candidate_incidence_opened_before_materialization": False, "output": {"path": str(STATES), "sha256": sha(STATES), "rows": len(states), "valid_rows": int(states.source_valid.sum())}}
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    primary = build_clock(states)
    controls = {name: build_clock(states, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    preregistration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "high_volatility_trade_arrival_backloading_relay_support_v1", "policy_id": "HVTAB-6",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha(prereg.DEFAULT_OUTPUT), "manifest_hash": preregistration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


PREREG_SHA = "abbd1443802bf4ce596a9cbe1d0578ad2ae55e835403773baa57e25f07b8fcd7"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=ENV_FILE)
    args = parser.parse_args()
    result = run(args.env_file)
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
