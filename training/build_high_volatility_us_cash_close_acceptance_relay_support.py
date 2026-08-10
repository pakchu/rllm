"""Materialize source-only HVUSCCA-8 clocks before Gross9 or economics."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_us_cash_close_acceptance_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z"); END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "0536fdaba8cac6f4fe98f875ada3ac3768f937cee27e4c310b985764038949f7"
QUERY = """SELECT ts,open,high,low,close FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
SOURCE_DIR = Path("data/high_volatility_us_cash_close_acceptance_relay_sources_2023_2026")
STATE = SOURCE_DIR / "us_cash_session_states.csv.gz"
CLOCK = Path("data/high_volatility_us_cash_close_acceptance_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_us_cash_close_acceptance_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_us_cash_close_acceptance_relay_support_2026-08-10.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_volatility_gate", "no_efficiency_gate", "no_terminal_acceptance_gate", "one_session_stale_geometry", "direction_flip", "forced_long")
COLUMNS = ("candidate", "control", "split", "session_date", "session_start", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "session_return", "session_efficiency", "terminal_location", "absolute_return_rank", "efficiency_rank", "btc_variation", "btc_variation_rank")


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_prior_midrank(values: pd.Series, maximum: int = 252, minimum: int = 126) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float); result = pd.Series(np.nan, index=numeric.index); history = []
    for index, current in numeric.items():
        prior = np.asarray(history[-maximum:], dtype=float)
        if math.isfinite(current) and len(prior) >= minimum: result.at[index] = ((prior < current).sum() + 0.5 * (prior == current).sum()) / len(prior)
        if math.isfinite(current): history.append(float(current))
    return result


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE); return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def eligible_local_dates() -> list[pd.Timestamp]:
    excluded = set(prereg.EXCLUDED_LOCAL_DATES)
    return [date for date in pd.date_range("2023-01-01", "2026-07-31", freq="D") if date.weekday() < 5 and date.strftime("%Y-%m-%d") not in excluded]


def session_bounds(local_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    midnight = pd.Timestamp(local_date.date()).tz_localize("America/New_York", ambiguous="raise", nonexistent="raise")
    return (midnight + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC"), (midnight + pd.Timedelta(hours=16)).tz_convert("UTC")


def build_states(bars: pd.DataFrame) -> pd.DataFrame:
    market = bars.copy(); market["ts"] = pd.to_datetime(market.ts, utc=True)
    for column in ("open", "high", "low", "close"): market[column] = pd.to_numeric(market[column], errors="coerce")
    if market.duplicated("ts", keep=False).any(): raise RuntimeError("duplicate BTC minute")
    market = market.set_index("ts").sort_index(); rows = []
    for local_date in eligible_local_dates():
        start, decision = session_bounds(local_date); expected = pd.date_range(start, decision, freq="1min", inclusive="left"); window = market.reindex(expected)
        ohlc = window[["open", "high", "low", "close"]]
        valid = bool(len(window) == 390 and np.isfinite(ohlc).all(axis=1).all() and ohlc.gt(0).all(axis=1).all() and window.high.ge(window[["open", "close"]].max(axis=1)).all() and window.low.le(window[["open", "close"]].min(axis=1)).all() and window.high.ge(window.low).all())
        session_return = efficiency = location = float("nan")
        if valid:
            path = np.r_[float(window.open.iloc[0]), window.close.to_numpy(float)]; increments = np.diff(np.log(path)); session_return = float(np.log(path[-1] / path[0])); travel = float(np.abs(increments).sum()); low, high = float(window.low.min()), float(window.high.max())
            valid = travel > 0 and high > low and session_return != 0
            if valid: efficiency = abs(session_return) / travel; location = (path[-1] - low) / (high - low)
        variation_window = market.reindex(pd.date_range(decision - pd.Timedelta(days=1), decision, freq="1min", inclusive="left")); variation_ohlc = variation_window[["open", "high", "low", "close"]]
        variation_valid = bool(np.isfinite(variation_ohlc).all(axis=1).all() and variation_ohlc.gt(0).all(axis=1).all() and variation_window.high.ge(variation_window[["open", "close"]].max(axis=1)).all() and variation_window.low.le(variation_window[["open", "close"]].min(axis=1)).all() and variation_window.high.ge(variation_window.low).all())
        variation = float(np.square(np.diff(np.log(variation_window.close.to_numpy(float)))).sum()) if variation_valid else float("nan")
        rows.append({"session_date": local_date.strftime("%Y-%m-%d"), "session_start": start, "decision_time": decision, "source_valid": valid and variation_valid and variation > 0, "session_return": session_return, "session_efficiency": efficiency, "terminal_location": location, "btc_variation": variation})
    frame = pd.DataFrame(rows); valid = frame.source_valid
    frame["absolute_return_rank"] = strict_prior_midrank(frame.session_return.abs().where(valid)); frame["efficiency_rank"] = strict_prior_midrank(frame.session_efficiency.where(valid)); frame["btc_variation_rank"] = strict_prior_midrank(frame.btc_variation.where(valid)); return frame


def build_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    used = states.shift(1) if control == "one_session_stale_geometry" else states
    session_return, efficiency_rank, location, return_rank, variation_rank = used.session_return, used.efficiency_rank, used.terminal_location, used.absolute_return_rank, used.btc_variation_rank
    valid = used.source_valid.eq(True) & np.isfinite(session_return) & session_return.ne(0)
    vol_gate = pd.Series(True, index=states.index) if control == "no_volatility_gate" else variation_rank.ge(0.65)
    eff_gate = pd.Series(True, index=states.index) if control == "no_efficiency_gate" else efficiency_rank.ge(0.65)
    terminal_gate = pd.Series(True, index=states.index) if control == "no_terminal_acceptance_gate" else ((session_return.gt(0) & location.ge(0.80)) | (session_return.lt(0) & location.le(0.20)))
    eligible = valid & return_rank.ge(0.75) & vol_gate & eff_gate & terminal_gate; onset = eligible & ~eligible.shift(1, fill_value=False)
    side = np.sign(session_return).fillna(0).astype(int)
    if control == "direction_flip": side = -side
    elif control == "forced_long": side = pd.Series(1, index=states.index)
    rows: list[dict[str, Any]] = []; reserved_until = None
    for index in states.index[onset]:
        decision = pd.Timestamp(states.at[index, "decision_time"]); entry = decision + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=8)
        if reserved_until is not None and entry < reserved_until: continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        reserved_until = exit_time
        rows.append({"candidate": "HVUSCCA-8", "control": control, "split": split, "session_date": used.at[index, "session_date"], "session_start": used.at[index, "session_start"], "decision_time": decision, "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "session_return": float(session_return.at[index]), "session_efficiency": float(used.at[index, "session_efficiency"]), "terminal_location": float(location.at[index]), "absolute_return_rank": float(return_rank.at[index]), "efficiency_rank": float(efficiency_rank.at[index]), "btc_variation": float(used.at[index, "btc_variation"]), "btc_variation_rank": float(variation_rank.at[index])})
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = clock[clock.split.eq(split)]
    if selected.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected.side.eq(1).sum()), int(selected.side.eq(-1).sum()); months = selected.entry_time.dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(selected), "max_month_share": float(months.max() / len(selected))}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA: raise RuntimeError("HVUSCCA preregistration drift")
    from sqlalchemy import text
    engine = postgres_engine()
    try: bars = pd.read_sql_query(text(QUERY), engine, params={"start": START - pd.Timedelta(days=1), "end": END})
    finally: engine.dispose()
    states = build_states(bars); primary = build_clock(states); controls = {name: build_clock(states, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CLOCK.parent.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True); _write_gzip_csv(states, STATE); _write_gzip_csv(primary, CLOCK)
    for name, clock in controls.items(): _write_gzip_csv(clock, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}; checks = {key: passed for name, values in support.items() for key, passed in ((f"{name}_minimum_events", values["events"] >= MINIMUM[name]), (f"{name}_side_balance", values["minority_side_share"] >= 0.20), (f"{name}_month_concentration", values["max_month_share"] <= 0.45))}; passed = all(checks.values()); registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {"protocol_version": "hvuscca_8_source_support_v1", "policy_id": "HVUSCCA-8", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source": {"query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(), "physical_rows": len(bars), "state": {"path": str(STATE), "sha256": sha(STATE), "rows": len(states)}}, "completed_preentry_sources_opened": True, "candidate_incidence_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(clock), "promotion_authorized": False} for name, clock in controls.items()}, "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}; RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); return result


if __name__ == "__main__":
    report = run(); print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
