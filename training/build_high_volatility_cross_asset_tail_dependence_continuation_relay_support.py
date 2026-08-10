"""Materialize source-only HVCTDC-8 clocks before Gross9 or economics."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_cross_asset_tail_dependence_continuation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "73a1a21e2d3fb64f2824d74182fe3fff2f00b2366e0e40df764ad9265ec455a7"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
ALT_SYMBOLS = SYMBOLS[1:]
BLOCK_QUERY = """
SELECT symbol,
       date_bin('4 hours', ts, TIMESTAMPTZ '1970-01-01 00:00:00+00') AS block_start,
       count(*) AS rows,
       min(ts) AS first_ts,
       max(ts) AS last_ts,
       (array_agg(open ORDER BY ts))[1] AS first_open,
       (array_agg(close ORDER BY ts DESC))[1] AS last_close,
       bool_and(open>0 AND high>0 AND low>0 AND close>0
                AND high>=greatest(open,close) AND low<=least(open,close) AND high>=low) AS values_valid
FROM bars_binance
WHERE interval='1m' AND symbol = ANY(:symbols) AND ts>=:start AND ts<:end
GROUP BY symbol, block_start
ORDER BY block_start, symbol
"""
BTC_QUERY = """
SELECT ts,open,high,low,close FROM bars_binance
WHERE interval='1m' AND symbol='BTCUSDT' AND ts>=:start AND ts<:end ORDER BY ts
"""
SOURCE_DIR = Path("data/high_volatility_cross_asset_tail_dependence_continuation_relay_sources_2023_2026")
STATE = SOURCE_DIR / "four_hour_tail_dependence_states.csv.gz"
CLOCK = Path("data/high_volatility_cross_asset_tail_dependence_continuation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_cross_asset_tail_dependence_continuation_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_cross_asset_tail_dependence_continuation_relay_support_2026-08-10.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_volatility_gate", "no_asymmetry_magnitude_gate", "no_current_tail_confirmation", "one_block_stale_geometry", "direction_flip", "forced_long")
COLUMNS = (
    "candidate", "control", "split", "block_start", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "tail_asymmetry", "asymmetry_magnitude_rank",
    "current_btc_tail_rank", "btc_variation", "btc_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_prior_midrank(values: pd.Series, maximum: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=numeric.index, dtype=float); history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-maximum:], dtype=float)
        if math.isfinite(current) and len(prior) >= minimum:
            result.at[index] = ((prior < current).sum() + 0.5 * (prior == current).sum()) / len(prior)
        if math.isfinite(current): history.append(float(current))
    return result


def empirical_midranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.asarray([((array < value).sum() + 0.5 * (array == value).sum()) / len(array) for value in array])


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def joint_blocks(aggregates: pd.DataFrame) -> pd.DataFrame:
    frame = aggregates.copy(); frame["block_start"] = pd.to_datetime(frame.block_start, utc=True)
    frame["first_ts"] = pd.to_datetime(frame.first_ts, utc=True); frame["last_ts"] = pd.to_datetime(frame.last_ts, utc=True)
    for column in ("rows", "first_open", "last_close"): frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["block_valid"] = (
        frame.rows.eq(240) & frame.first_ts.eq(frame.block_start)
        & frame.last_ts.eq(frame.block_start + pd.Timedelta(minutes=239))
        & frame.values_valid.eq(True) & np.isfinite(frame[["first_open", "last_close"]]).all(axis=1)
        & frame.first_open.gt(0) & frame.last_close.gt(0)
    )
    if frame.duplicated(["block_start", "symbol"], keep=False).any(): raise RuntimeError("duplicate aggregate block/symbol")
    expected = pd.date_range(START, END, freq="4h", inclusive="left")
    output = pd.DataFrame({"block_start": expected, "decision_time": expected + pd.Timedelta(hours=4)})
    for symbol in SYMBOLS:
        selected = frame[frame.symbol.eq(symbol)].set_index("block_start").reindex(expected)
        output[f"{symbol}_return"] = np.log(selected.last_close.to_numpy(float) / selected.first_open.to_numpy(float))
        output[f"{symbol}_valid"] = selected.block_valid.fillna(False).to_numpy(bool)
    output["joint_valid"] = output[[f"{symbol}_valid" for symbol in SYMBOLS]].all(axis=1)
    return output


def add_variation(blocks: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    market = btc.copy(); market["ts"] = pd.to_datetime(market.ts, utc=True)
    for column in ("open", "high", "low", "close"): market[column] = pd.to_numeric(market[column], errors="coerce")
    if market.duplicated("ts", keep=False).any(): raise RuntimeError("duplicate BTC minute")
    market = market.set_index("ts").sort_index(); variations = []
    for decision in blocks.decision_time:
        expected = pd.date_range(decision - pd.Timedelta(days=1), decision, freq="1min", inclusive="left")
        window = market.reindex(expected); ohlc = window[["open", "high", "low", "close"]]
        good = bool(np.isfinite(ohlc).all(axis=1).all() and ohlc.gt(0).all(axis=1).all() and window.high.ge(window[["open", "close"]].max(axis=1)).all() and window.low.le(window[["open", "close"]].min(axis=1)).all() and window.high.ge(window.low).all())
        variations.append(float(np.square(np.diff(np.log(window.close.to_numpy(float)))).sum()) if good else float("nan"))
    output = blocks.copy(); output["btc_variation"] = variations
    output["btc_variation_rank"] = strict_prior_midrank(output.btc_variation.where(output.joint_valid))
    return output


def add_tail_dependence(blocks: pd.DataFrame) -> pd.DataFrame:
    output = blocks.copy(); returns = output[[f"{symbol}_return" for symbol in SYMBOLS]].to_numpy(float)
    valid_indices: list[int] = []; asymmetry = np.full(len(output), np.nan); current_rank = np.full(len(output), np.nan)
    for index in range(len(output)):
        if not bool(output.at[index, "joint_valid"]): continue
        history_indices = valid_indices[-270:]
        if len(history_indices) >= 180:
            history = returns[history_indices]
            ranks = np.column_stack([empirical_midranks(history[:, column]) for column in range(history.shape[1])])
            btc_upper, btc_lower = ranks[:, 0] >= 0.80, ranks[:, 0] <= 0.20
            if btc_upper.any() and btc_lower.any():
                upper = [np.mean(btc_upper & (ranks[:, column] >= 0.80)) / np.mean(btc_upper) for column in range(1, len(SYMBOLS))]
                lower = [np.mean(btc_lower & (ranks[:, column] <= 0.20)) / np.mean(btc_lower) for column in range(1, len(SYMBOLS))]
                asymmetry[index] = float(np.mean(upper) - np.mean(lower))
                btc_prior = history[:, 0]; value = returns[index, 0]
                current_rank[index] = ((btc_prior < value).sum() + 0.5 * (btc_prior == value).sum()) / len(btc_prior)
        valid_indices.append(index)
    output["tail_asymmetry"] = asymmetry; output["current_btc_tail_rank"] = current_rank
    output["asymmetry_magnitude_rank"] = strict_prior_midrank(output.tail_asymmetry.abs())
    return output


def build_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    used = states.shift(1) if control == "one_block_stale_geometry" else states
    asymmetry = used.tail_asymmetry; magnitude = used.asymmetry_magnitude_rank
    current_tail = used.current_btc_tail_rank; variation_rank = used.btc_variation_rank
    valid = used.joint_valid.eq(True) & np.isfinite(asymmetry) & asymmetry.ne(0)
    vol_gate = pd.Series(True, index=states.index) if control == "no_volatility_gate" else variation_rank.ge(0.65)
    mag_gate = pd.Series(True, index=states.index) if control == "no_asymmetry_magnitude_gate" else magnitude.ge(0.80)
    tail_gate = pd.Series(True, index=states.index) if control == "no_current_tail_confirmation" else ((asymmetry.gt(0) & current_tail.ge(0.80)) | (asymmetry.lt(0) & current_tail.le(0.20)))
    eligible = valid & vol_gate & mag_gate & tail_gate
    onset = eligible & ~eligible.shift(1, fill_value=False)
    side = np.sign(asymmetry).fillna(0).astype(int)
    if control == "direction_flip": side = -side
    elif control == "forced_long": side = pd.Series(1, index=states.index)
    rows: list[dict[str, Any]] = []; reserved_until = None
    for index in states.index[onset]:
        decision = pd.Timestamp(states.at[index, "decision_time"]); entry = decision + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=8)
        if reserved_until is not None and entry < reserved_until: continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        reserved_until = exit_time
        rows.append({"candidate": "HVCTDC-8", "control": control, "split": split, "block_start": states.at[index, "block_start"], "decision_time": decision, "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "tail_asymmetry": float(asymmetry.at[index]), "asymmetry_magnitude_rank": float(magnitude.at[index]), "current_btc_tail_rank": float(current_tail.at[index]), "btc_variation": float(used.at[index, "btc_variation"]), "btc_variation_rank": float(variation_rank.at[index])})
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = clock[clock.split.eq(split)]
    if selected.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected.side.eq(1).sum()), int(selected.side.eq(-1).sum()); months = selected.entry_time.dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(selected), "max_month_share": float(months.max() / len(selected))}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA: raise RuntimeError("HVCTDC preregistration drift")
    engine = postgres_engine()
    try:
        aggregates = pd.read_sql_query(BLOCK_QUERY, engine, params={"symbols": list(SYMBOLS), "start": START, "end": END})
        btc = pd.read_sql_query(BTC_QUERY, engine, params={"start": START, "end": END})
    finally: engine.dispose()
    states = add_tail_dependence(add_variation(joint_blocks(aggregates), btc)); primary = build_clock(states)
    controls = {name: build_clock(states, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CLOCK.parent.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, STATE); _write_gzip_csv(primary, CLOCK)
    for name, clock in controls.items(): _write_gzip_csv(clock, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: passed for name, values in support.items() for key, passed in ((f"{name}_minimum_events", values["events"] >= MINIMUM[name]), (f"{name}_side_balance", values["minority_side_share"] >= 0.20), (f"{name}_month_concentration", values["max_month_share"] <= 0.45))}
    passed = all(checks.values()); registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {"protocol_version": "hvctdc_8_source_support_v1", "policy_id": "HVCTDC-8", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source": {"block_query_sha256": hashlib.sha256(BLOCK_QUERY.encode()).hexdigest(), "btc_query_sha256": hashlib.sha256(BTC_QUERY.encode()).hexdigest(), "aggregate_rows": len(aggregates), "btc_rows": len(btc), "state": {"path": str(STATE), "sha256": sha(STATE), "rows": len(states)}}, "completed_preentry_sources_opened": True, "candidate_incidence_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(clock), "promotion_authorized": False} for name, clock in controls.items()}, "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}; RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); return result


if __name__ == "__main__":
    report = run(); print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
