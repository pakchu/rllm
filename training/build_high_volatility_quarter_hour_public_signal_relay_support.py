"""Materialize source-only support clocks for frozen HVQHPS-12."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_quarter_hour_public_signal_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_quarter_hour_opening_imbalance_relay_support import strict_prior_midrank


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "0191d4e3d90e918eeb5bf5716eaae656d7f7544f20427136b7a68e7a8e799bdd"
SOURCE_DIR = Path("data/high_volatility_quarter_hour_public_signal_relay_sources_2023_2026")
FEATURES = SOURCE_DIR / "quarter_hour_public_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_quarter_hour_public_signal_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_quarter_hour_public_signal_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_quarter_hour_public_signal_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_quarter_hour_public_signal_relay_support_2026-08-13.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "lagged_flow_component", "include_ols_intercept", "no_public_strength_tail",
    "no_variation_gate", "latest_indicator_bar_included",
    "one_quarter_stale_public_component", "direction_flip", "same_clock_forced_long",
)
QUERY = """SELECT ts,open,high,low,close,volume,taker_buy_base FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
FEATURE_COLUMNS = (
    "decision_time", "source_valid", "public_component", "lagged_component", "ols_intercept",
    "latest_public_component", "model_valid", "latest_model_valid", "public_strength_rank",
    "lagged_strength_rank", "intercept_strength_rank", "latest_public_strength_rank",
    "realized_variation", "variation_rank",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time", "entry_time",
    "exit_time", "side", "public_component", "public_strength_rank", "realized_variation",
    "variation_rank",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def validate_bars(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    required = ["ts", "open", "high", "low", "close", "volume", "taker_buy_base"]
    if list(bars.columns) != required:
        raise RuntimeError(f"HVQHPS source schema must be exactly {required}")
    frame = bars.copy(); frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="raise")
    if frame["ts"].duplicated().any(): raise RuntimeError("HVQHPS source has duplicate minutes")
    frame = frame.sort_values("ts").reset_index(drop=True)
    expected = pd.date_range(start, end, freq="1min", inclusive="left")
    if len(frame) != len(expected) or not frame["ts"].equals(pd.Series(expected, name="ts")):
        raise RuntimeError("HVQHPS source is not the exact requested one-minute grid")
    for column in required[1:]: frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(frame[required[1:]]).all(axis=1)
    coherent = (
        frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & frame["high"].ge(frame[["open", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "close"]].min(axis=1))
        & frame["volume"].ge(0) & frame["taker_buy_base"].ge(0)
        & frame["taker_buy_base"].le(frame["volume"])
    )
    if not bool((finite & coherent).all()): raise RuntimeError("HVQHPS source has invalid OHLCV or flow")
    return frame.set_index("ts")


def wilder_rsi(close: pd.Series, period: int = 24) -> pd.Series:
    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan); delta = np.diff(values)
    if len(values) <= period: return pd.Series(out, index=close.index)
    gains, losses = np.maximum(delta, 0), np.maximum(-delta, 0)
    avg_gain, avg_loss = float(gains[:period].mean()), float(losses[:period].mean())
    def score(gain: float, loss: float) -> float:
        if loss == 0: return 100.0 if gain > 0 else math.nan
        return 100.0 - 100.0 / (1.0 + gain / loss)
    out[period] = score(avg_gain, avg_loss)
    for position in range(period + 1, len(values)):
        avg_gain = ((period - 1) * avg_gain + gains[position - 1]) / period
        avg_loss = ((period - 1) * avg_loss + losses[position - 1]) / period
        out[position] = score(avg_gain, avg_loss)
    return pd.Series(out, index=close.index)


def technical_indicators(bars15: pd.DataFrame) -> pd.DataFrame:
    close, high, low, volume = (bars15[name].astype(float) for name in ("close", "high", "low", "volume"))
    output: dict[str, pd.Series] = {}
    rsi = wilder_rsi(close, 24); output["rsi_24"] = rsi
    lowest, highest = low.rolling(24, min_periods=24).min(), high.rolling(24, min_periods=24).max()
    stochastic = 100 * (close - lowest) / (highest - lowest)
    output["stochastic_k_24"] = stochastic
    output["stochastic_d_6"] = stochastic.rolling(6, min_periods=6).mean()
    rsi_low, rsi_high = rsi.rolling(24, min_periods=24).min(), rsi.rolling(24, min_periods=24).max()
    stoch_rsi = 100 * (rsi - rsi_low) / (rsi_high - rsi_low)
    output["stoch_rsi_double_6"] = stoch_rsi.rolling(6, min_periods=6).mean().rolling(6, min_periods=6).mean()
    typical = (high + low + close) / 3
    typical_mean = typical.rolling(24, min_periods=24).mean()
    mad = typical.rolling(24, min_periods=24).apply(lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True)
    output["cci_24"] = (typical - typical_mean) / (0.015 * mad)
    for window in (4, 6, 12, 20, 32, 48, 96): output[f"price_sma_{window}"] = close / close.rolling(window, min_periods=window).mean() - 1
    price_macd = (close.ewm(span=8, adjust=False, min_periods=8).mean() - close.ewm(span=32, adjust=False, min_periods=32).mean()) / close
    output["price_macd"] = price_macd; output["price_macd_diff"] = price_macd - price_macd.ewm(span=6, adjust=False, min_periods=6).mean()
    for window in (4, 6, 12, 16, 24, 32, 48): output[f"volume_sma_{window}"] = volume / volume.rolling(window, min_periods=window).mean() - 1
    volume_macd = (volume.ewm(span=8, adjust=False, min_periods=8).mean() - volume.ewm(span=32, adjust=False, min_periods=32).mean()) / volume
    output["volume_macd"] = volume_macd; output["volume_macd_diff"] = volume_macd - volume_macd.ewm(span=6, adjust=False, min_periods=6).mean()
    spread = high - low; multiplier = pd.Series(np.where(spread.gt(0), (2 * close - high - low) / spread, 0.0), index=close.index)
    adl = (multiplier * volume).cumsum()
    output["chaikin_normalized"] = (adl.ewm(span=4, adjust=False, min_periods=4).mean() - adl.ewm(span=32, adjust=False, min_periods=32).mean()) / volume.rolling(32, min_periods=32).mean()
    middle = close.rolling(24, min_periods=24).mean(); std = close.rolling(24, min_periods=24).std(ddof=0)
    lower, upper = middle - 2 * std, middle + 2 * std
    output["bollinger_lower_distance"] = (close - lower) / lower
    output["bollinger_middle_distance"] = (close - middle) / middle
    output["bollinger_upper_distance"] = (close - upper) / upper
    output["bollinger_width"] = (upper - lower) / middle
    result = pd.DataFrame(output, index=bars15.index)
    if result.shape[1] != 28: raise RuntimeError("HVQHPS TI28 column-count drift")
    return result.replace([np.inf, -np.inf], np.nan)


def rolling_components(
    response: pd.Series, indicators: pd.DataFrame, *, lag_count: int = 12,
    lookback: int = 8640, minimum: int = 5760,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    numeric = pd.to_numeric(response, errors="coerce").astype(float)
    lags = pd.concat([numeric.shift(k).rename(f"lag_{k}") for k in range(1, lag_count + 1)], axis=1)
    if len(indicators) != len(numeric) or not indicators.index.equals(numeric.index): raise ValueError("component inputs misaligned")
    dimension = 1 + lag_count + indicators.shape[1]
    public = pd.Series(np.nan, index=numeric.index); lagged = public.copy(); intercept = public.copy()
    history: deque[tuple[np.ndarray, float]] = deque(); xtx = np.zeros((dimension, dimension)); xty = np.zeros(dimension)
    inverse: np.ndarray | None = None; updates = 0
    for index in numeric.index:
        lag_values = lags.loc[index].to_numpy(dtype=float); ti_values = indicators.loc[index].to_numpy(dtype=float)
        predictor_valid = np.isfinite(lag_values).all() and np.isfinite(ti_values).all()
        if predictor_valid and len(history) >= minimum:
            if inverse is None:
                if np.linalg.matrix_rank(xtx) == dimension: inverse = np.linalg.inv(xtx)
            if inverse is not None:
                coefficients = inverse @ xty
                values = (float(coefficients[lag_count + 1:] @ ti_values), float(coefficients[1:lag_count + 1] @ lag_values), float(coefficients[0]))
                if all(math.isfinite(value) for value in values): public.at[index], lagged.at[index], intercept.at[index] = values
        current = float(numeric.at[index])
        if predictor_valid and math.isfinite(current):
            design = np.concatenate(([1.0], lag_values, ti_values))
            if len(history) == lookback:
                old, old_y = history.popleft(); xtx -= np.outer(old, old); xty -= old * old_y
                if inverse is not None:
                    vector = inverse @ old; denominator = 1.0 - float(old @ vector)
                    inverse = None if abs(denominator) < 1e-10 else inverse + np.outer(vector, vector) / denominator
            history.append((design, current)); xtx += np.outer(design, design); xty += design * current
            if inverse is not None:
                vector = inverse @ design; inverse -= np.outer(vector, vector) / (1.0 + float(design @ vector))
            updates += 1
            if updates % 96 == 0: inverse = None; xtx = (xtx + xtx.T) / 2
    return public, lagged, intercept


def derive_features(bars: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp, lookback: int = 8640, minimum: int = 5760) -> pd.DataFrame:
    source = validate_bars(bars, start, end); decisions = pd.date_range(start, end, freq="15min", inclusive="left")
    opening = source.reindex(decisions); volumes = opening["volume"].to_numpy(dtype=float); source_valid = volumes > 0
    imbalance_values = np.full(len(decisions), np.nan)
    np.divide(2 * opening["taker_buy_base"].to_numpy(dtype=float) - volumes, volumes, out=imbalance_values, where=source_valid)
    imbalance = pd.Series(imbalance_values, index=decisions)
    grouped = source.resample("15min", origin="epoch", closed="left", label="left")
    bars15 = grouped.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), rows=("open", "count"))
    if not bars15["rows"].eq(15).all() or not bars15.index.equals(decisions): raise RuntimeError("HVQHPS 15-minute aggregation incomplete")
    ti = technical_indicators(bars15.drop(columns="rows")); frozen_ti = ti.reindex(decisions - pd.Timedelta(minutes=30)).set_axis(decisions)
    latest_ti = ti.reindex(decisions - pd.Timedelta(minutes=15)).set_axis(decisions)
    public, lagged, intercept = rolling_components(imbalance, frozen_ti, lookback=lookback, minimum=minimum)
    latest_public, _, _ = rolling_components(imbalance, latest_ti, lookback=lookback, minimum=minimum)
    squared = np.log(source["close"] / source["open"]).pow(2)
    variation = np.sqrt(squared.shift(1).rolling(1440, min_periods=1440).sum()).reindex(decisions)
    frame = pd.DataFrame({
        "decision_time": decisions, "source_valid": source_valid, "public_component": public.to_numpy(),
        "lagged_component": lagged.to_numpy(), "ols_intercept": intercept.to_numpy(),
        "latest_public_component": latest_public.to_numpy(), "model_valid": np.isfinite(public),
        "latest_model_valid": np.isfinite(latest_public),
        "public_strength_rank": strict_prior_midrank(public.abs(), lookback=lookback, minimum=minimum).to_numpy(),
        "lagged_strength_rank": strict_prior_midrank(lagged.abs(), lookback=lookback, minimum=minimum).to_numpy(),
        "intercept_strength_rank": strict_prior_midrank((public + intercept).abs(), lookback=lookback, minimum=minimum).to_numpy(),
        "latest_public_strength_rank": strict_prior_midrank(latest_public.abs(), lookback=lookback, minimum=minimum).to_numpy(),
        "realized_variation": variation.to_numpy(),
        "variation_rank": strict_prior_midrank(variation, lookback=lookback, minimum=minimum, update_mask=source_valid).to_numpy(),
    })
    return frame.loc[:, FEATURE_COLUMNS]


def materialize_features() -> dict[str, Any]:
    from sqlalchemy import text
    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            bars = pd.read_sql_query(text(QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()})
    finally: engine.dispose()
    frame = derive_features(bars, start=START, end=END); SOURCE_DIR.mkdir(parents=True, exist_ok=True); _write_gzip_csv(frame, FEATURES)
    core = {"protocol_version": "hvqhps_12_btc_source_v1", "query": QUERY, "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "window": [START.isoformat(), END.isoformat()], "exact_minute_grid": True, "ti28_count": 28, "no_imputation": True, "outcomes_opened": False, "candidate_incidence_opened": True, "output": {"path": str(FEATURES), "sha256": sha256(FEATURES), "rows": len(frame), "model_valid_rows": int(frame["model_valid"].sum())}}
    payload = {**core, "manifest_hash": canonical_hash(core)}; SOURCE_MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); return payload


def load_features() -> pd.DataFrame:
    frame = pd.read_csv(FEATURES, compression="gzip"); frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True)
    for column in ("source_valid", "model_valid", "latest_model_valid"): frame[column] = frame[column].astype(str).str.lower().eq("true")
    for column in set(FEATURE_COLUMNS) - {"decision_time", "source_valid", "model_valid", "latest_model_valid"}: frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def onset_after_previous_valid(valid: pd.Series, eligible: pd.Series) -> pd.Series:
    active = pd.Series(False, index=eligible.index); previous = False
    for index in eligible.index:
        if not bool(valid.at[index]): continue
        current = bool(eligible.at[index]); active.at[index] = current and not previous; previous = current
    return active


def selected_series(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    component, rank, valid = frame["public_component"].copy(), frame["public_strength_rank"].copy(), frame["model_valid"].copy()
    if control == "lagged_flow_component": component, rank = frame["lagged_component"].copy(), frame["lagged_strength_rank"].copy()
    elif control == "include_ols_intercept": component, rank = component + frame["ols_intercept"], frame["intercept_strength_rank"].copy()
    elif control == "latest_indicator_bar_included": component, rank, valid = frame["latest_public_component"].copy(), frame["latest_public_strength_rank"].copy(), frame["latest_model_valid"].copy()
    elif control == "one_quarter_stale_public_component": component, rank, valid = component.shift(1), rank.shift(1), valid.shift(1, fill_value=False)
    return component, rank, valid


def active_and_side(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS): raise ValueError(control)
    component, rank, valid = selected_series(frame, control)
    tail = pd.Series(True, index=frame.index) if control == "no_public_strength_tail" else rank.ge(0.95)
    variation = pd.Series(True, index=frame.index) if control == "no_variation_gate" else frame["variation_rank"].ge(0.65)
    eligible = valid & np.isfinite(component) & component.ne(0) & np.isfinite(rank) & tail & np.isfinite(frame["realized_variation"]) & np.isfinite(frame["variation_rank"]) & variation
    active = onset_after_previous_valid(valid, eligible)
    side = pd.Series(np.where(component.gt(0), 1, np.where(component.lt(0), -1, 0)), index=frame.index, dtype=int)
    if control == "direction_flip": side = -side
    elif control == "same_clock_forced_long": side = pd.Series(1, index=frame.index, dtype=int)
    return active, side


def make_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = active_and_side(frame, control); component, rank, _ = selected_series(frame, control)
    rows: list[dict[str, Any]] = []; next_allowed: pd.Timestamp | None = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"]); entry = decision + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=12)
        if next_allowed is not None and entry < next_allowed: continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        next_allowed = exit_time
        rows.append({"candidate": prereg.POLICY_ID, "control": control, "split": split, "decision_time": decision, "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "public_component": float(component.at[index]), "public_strength_rank": float(rank.at[index]), "realized_variation": float(frame.at[index, "realized_variation"]), "variation_rank": float(frame.at[index, "variation_rank"])})
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    frame = clock[clock["split"].eq(split)]
    if frame.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(frame["side"].eq(1).sum()), int(frame["side"].eq(-1).sum()); months = frame["entry_time"].dt.strftime("%Y-%m").value_counts()
    return {"events": len(frame), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(frame), "max_month_share": int(months.max()) / len(frame)}


def support_checks(values: dict[str, dict[str, Any]]) -> dict[str, bool]:
    checks = {}
    for split, row in values.items():
        checks[f"{split}_minimum_events"] = row["events"] >= MINIMUM[split]; checks[f"{split}_side_balance"] = row["minority_side_share"] >= 0.20; checks[f"{split}_month_concentration"] = row["max_month_share"] <= 0.45
    return checks


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA: raise RuntimeError("HVQHPS preregistration hash drift")
    source_manifest = materialize_features(); frame = load_features(); primary = make_clock(frame); controls = {name: make_clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True); SPLIT_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True); _write_gzip_csv(primary, CLOCK)
    for split in SPLITS: _write_gzip_csv(primary[primary["split"].eq(split)].reset_index(drop=True), SPLIT_DIR / f"{split}.csv.gz")
    for name, value in controls.items(): _write_gzip_csv(value, CONTROL_DIR / f"{name}.csv.gz")
    values = {split: support_stats(primary, split) for split in SPLITS}; checks = support_checks(values); passed = all(checks.values()); registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {"protocol_version": "hvqhps_12_source_support_v1", "policy_id": prereg.POLICY_ID, "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha256(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]}, "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha256(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)}, "split_clocks": {split: {"path": str(SPLIT_DIR / f"{split}.csv.gz"), "sha256": sha256(SPLIT_DIR / f"{split}.csv.gz"), "rows": int(primary["split"].eq(split).sum())} for split in SPLITS}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(value), "promotion_authorized": False} for name, value in controls.items()}, "support": values, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    payload = {**core, "manifest_hash": canonical_hash(core)}; RESULT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); return payload


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); result = run(); print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
