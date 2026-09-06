"""Materialize source-only support clocks for frozen HVVIXTE-24."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_vix_transfer_entropy_forecast_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_quarter_hour_opening_imbalance_relay_support import strict_prior_midrank

ENV_FILE = "/home/pakchu/rllm/.env"
VIX = Path("data/cboe_volatility_surface_2021_2026/cboe_volatility_surface_2021-01-01_2026-08-07.csv.gz")
VIX_SHA = "42eb1093f5167aec9c71a4733ab3451e40807c81dc7cb49568a6a0c634267ba0"
PREREG_SHA = "64c0eba3c8630ce3d46797a9d0f5695e1af5efb83611457a1ad6fe0c3f705c6b"
START = pd.Timestamp("2021-01-01T00:00:00Z"); END = pd.Timestamp("2026-08-11T00:00:00Z")
SOURCE_DIR = Path("data/high_volatility_vix_transfer_entropy_forecast_relay_sources_2021_2026")
FEATURES = SOURCE_DIR / "decision_features.csv.gz"; SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_vix_transfer_entropy_forecast_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_vix_transfer_entropy_forecast_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_vix_transfer_entropy_forecast_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_vix_transfer_entropy_forecast_relay_support_2026-08-14.json")
SPLITS = {"train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")), "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")), "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")), "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z"))}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("vix_sign_direct_fade", "unconditional_btc_transition", "no_strength_tail", "no_variation_gate", "one_session_stale_forecast", "direction_flip", "same_clock_forced_long")
QUERY = """SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS ts,
(array_agg(open ORDER BY ts))[1] AS open,max(high) AS high,min(low) AS low,
(array_agg(close ORDER BY ts DESC))[1] AS close,count(*) AS source_rows
FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
GROUP BY 1 ORDER BY 1"""
FEATURE_COLUMNS = ("source_date", "decision_time", "source_valid", "vix_state", "btc_state", "target_state", "target_completion_time", "conditioning_count", "target_positive_count", "target_negative_count", "conditional_probability", "forecast_strength", "strength_rank", "unconditional_probability", "unconditional_strength", "unconditional_strength_rank", "realized_variation", "variation_rank", "forecast_valid")
CLOCK_COLUMNS = ("candidate", "control", "split", "source_date", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "conditional_probability", "forecast_strength", "strength_rank", "realized_variation", "variation_rank")


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical_hash(value: Any) -> str: return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE); return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def cboe_decisions(vix: pd.DataFrame) -> pd.DataFrame:
    required = ["observation_date", "VIX_close"]
    if not set(required).issubset(vix): raise RuntimeError("HVVIXTE Cboe schema drift")
    frame = vix[required].copy(); frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="raise")
    frame["VIX_close"] = pd.to_numeric(frame["VIX_close"], errors="coerce")
    if frame["observation_date"].duplicated().any() or not frame["observation_date"].is_monotonic_increasing: raise RuntimeError("HVVIXTE Cboe dates invalid")
    if not bool((np.isfinite(frame["VIX_close"]) & frame["VIX_close"].gt(0)).all()): raise RuntimeError("HVVIXTE VIX close invalid")
    local = [pd.Timestamp(day).tz_localize(ZoneInfo("America/New_York")) + pd.Timedelta(hours=9, minutes=35) for day in frame["observation_date"]]
    decisions = pd.Series(local).dt.tz_convert("UTC")
    output = pd.DataFrame({"source_date": frame["observation_date"].dt.strftime("%Y-%m-%d"), "vix_close": frame["VIX_close"], "decision_time": decisions.shift(-1)})
    output["vix_state"] = np.sign(np.log(output["vix_close"] / output["vix_close"].shift(1)))
    return output.iloc[1:-1].reset_index(drop=True)


def validate_market(market: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    required = ["ts", "open", "high", "low", "close", "source_rows"]
    if list(market.columns) != required: raise RuntimeError("HVVIXTE market schema drift")
    frame = market.copy(); frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="raise"); frame = frame.sort_values("ts").reset_index(drop=True)
    expected = pd.date_range(start, end, freq="5min", inclusive="left")
    if len(frame) != len(expected) or not frame["ts"].equals(pd.Series(expected, name="ts")): raise RuntimeError("HVVIXTE market is not exact five-minute grid")
    for column in required[1:]: frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not frame["source_rows"].eq(5).all(): raise RuntimeError("HVVIXTE five-minute source incomplete")
    valid = np.isfinite(frame[["open", "high", "low", "close"]]).all(axis=1) & frame[["open", "high", "low", "close"]].gt(0).all(axis=1) & frame["high"].ge(frame[["open", "close"]].max(axis=1)) & frame["low"].le(frame[["open", "close"]].min(axis=1))
    if not bool(valid.all()): raise RuntimeError("HVVIXTE market values invalid")
    return frame.set_index("ts")


def transition_forecasts(frame: pd.DataFrame, *, lookback: int = 756, minimum: int = 252, cell_minimum: int = 30, target_minimum: int = 5) -> pd.DataFrame:
    output = frame.copy(); fields = ["conditioning_count", "target_positive_count", "target_negative_count", "conditional_probability", "forecast_strength", "unconditional_probability", "unconditional_strength"]
    for field in fields: output[field] = np.nan
    for position in range(len(output)):
        current_time = pd.Timestamp(output.at[position, "decision_time"])
        prior = output.iloc[:position]
        prior = prior[prior["target_completion_time"].le(current_time) & prior["source_valid"] & prior["target_state"].isin((-1, 1))].tail(lookback)
        if len(prior) < minimum: continue
        x, z = output.at[position, "vix_state"], output.at[position, "btc_state"]
        cell = prior[prior["vix_state"].eq(x) & prior["btc_state"].eq(z)]
        positive, negative = int(cell["target_state"].eq(1).sum()), int(cell["target_state"].eq(-1).sum())
        output.at[position, "conditioning_count"] = len(cell); output.at[position, "target_positive_count"] = positive; output.at[position, "target_negative_count"] = negative
        if len(cell) >= cell_minimum and min(positive, negative) >= target_minimum:
            probability = positive / len(cell); output.at[position, "conditional_probability"] = probability; output.at[position, "forecast_strength"] = abs(probability - 0.5)
        unconditional = prior[prior["btc_state"].eq(z)]
        u_positive, u_negative = int(unconditional["target_state"].eq(1).sum()), int(unconditional["target_state"].eq(-1).sum())
        if len(unconditional) >= cell_minimum and min(u_positive, u_negative) >= target_minimum:
            probability = u_positive / len(unconditional); output.at[position, "unconditional_probability"] = probability; output.at[position, "unconditional_strength"] = abs(probability - 0.5)
    output["strength_rank"] = strict_prior_midrank(output["forecast_strength"], lookback=lookback, minimum=minimum)
    output["unconditional_strength_rank"] = strict_prior_midrank(output["unconditional_strength"], lookback=lookback, minimum=minimum)
    output["forecast_valid"] = np.isfinite(output["conditional_probability"]) & output["conditional_probability"].ne(0.5) & np.isfinite(output["strength_rank"])
    return output


def derive_features(vix: pd.DataFrame, market: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp, lookback: int = 756, minimum: int = 252) -> pd.DataFrame:
    prices = validate_market(market, start, end); decisions = cboe_decisions(vix)
    decisions = decisions[decisions["decision_time"].between(start, end, inclusive="left")].reset_index(drop=True)
    opens = prices["open"].reindex(pd.DatetimeIndex(decisions["decision_time"])); decisions["decision_open"] = opens.to_numpy()
    decisions["btc_state"] = np.sign(np.log(decisions["decision_open"] / decisions["decision_open"].shift(1)))
    decisions["target_state"] = decisions["btc_state"].shift(-1); decisions["target_completion_time"] = decisions["decision_time"].shift(-1)
    bar_sq = np.log(prices["close"] / prices["open"]).pow(2); variations = [math.sqrt(float(bar_sq.loc[(bar_sq.index >= previous) & (bar_sq.index < current)].sum())) if position > 0 else math.nan for position, (previous, current) in enumerate(zip(decisions["decision_time"].shift(1), decisions["decision_time"]))]
    decisions["realized_variation"] = variations
    decisions["source_valid"] = decisions["vix_state"].isin((-1, 1)) & decisions["btc_state"].isin((-1, 1)) & np.isfinite(decisions["realized_variation"]) & decisions["realized_variation"].gt(0)
    decisions["variation_rank"] = strict_prior_midrank(decisions["realized_variation"].where(decisions["source_valid"]), lookback=lookback, minimum=minimum)
    decisions = transition_forecasts(decisions, lookback=lookback, minimum=minimum)
    return decisions.loc[:, FEATURE_COLUMNS]


def materialize_features() -> dict[str, Any]:
    from sqlalchemy import text
    if sha256(VIX) != VIX_SHA: raise RuntimeError("HVVIXTE VIX source drift")
    vix = pd.read_csv(VIX, compression="gzip"); engine = postgres_engine()
    try:
        with engine.connect() as connection: market = pd.read_sql_query(text(QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()})
    finally: engine.dispose()
    frame = derive_features(vix, market, start=START, end=END); SOURCE_DIR.mkdir(parents=True, exist_ok=True); _write_gzip_csv(frame, FEATURES)
    core = {"protocol_version": "hvvixte_24_source_v1", "vix": {"path": str(VIX), "sha256": VIX_SHA}, "query": QUERY, "btc_table": "bars_binance", "window": [START.isoformat(), END.isoformat()], "outcomes_opened": False, "candidate_incidence_opened": True, "no_imputation": True, "output": {"path": str(FEATURES), "sha256": sha256(FEATURES), "rows": len(frame), "forecast_valid_rows": int(frame["forecast_valid"].sum())}}
    payload = {**core, "manifest_hash": canonical_hash(core)}; SOURCE_MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); return payload


def load_features() -> pd.DataFrame:
    frame = pd.read_csv(FEATURES, compression="gzip"); frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True); frame["target_completion_time"] = pd.to_datetime(frame["target_completion_time"], utc=True)
    for column in ("source_valid", "forecast_valid"): frame[column] = frame[column].astype(str).str.lower().eq("true")
    for column in set(FEATURE_COLUMNS) - {"source_date", "decision_time", "target_completion_time", "source_valid", "forecast_valid"}: frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def onset(valid: pd.Series, eligible: pd.Series) -> pd.Series:
    output = pd.Series(False, index=eligible.index); previous = False
    for index in eligible.index:
        if not bool(valid.at[index]): continue
        current = bool(eligible.at[index]); output.at[index] = current and not previous; previous = current
    return output


def active_and_side(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS): raise ValueError(control)
    probability, strength, valid = frame["conditional_probability"].copy(), frame["strength_rank"].copy(), frame["forecast_valid"].copy()
    if control == "unconditional_btc_transition": probability, strength = frame["unconditional_probability"].copy(), frame["unconditional_strength_rank"].copy(); valid = np.isfinite(probability) & probability.ne(0.5) & np.isfinite(strength)
    elif control == "one_session_stale_forecast": probability, strength, valid = probability.shift(1), strength.shift(1), valid.shift(1, fill_value=False)
    tail = pd.Series(True, index=frame.index) if control in ("no_strength_tail", "vix_sign_direct_fade") else strength.ge(0.75)
    variation = pd.Series(True, index=frame.index) if control == "no_variation_gate" else frame["variation_rank"].ge(0.65)
    eligible = valid & np.isfinite(probability) & probability.ne(0.5) & np.isfinite(frame["variation_rank"]) & tail & variation
    active = onset(valid, eligible); side = pd.Series(np.where(probability.gt(0.5), 1, np.where(probability.lt(0.5), -1, 0)), index=frame.index, dtype=int)
    if control == "vix_sign_direct_fade": side = -frame["vix_state"].fillna(0).astype(int)
    elif control == "direction_flip": side = -side
    elif control == "same_clock_forced_long": side = pd.Series(1, index=frame.index, dtype=int)
    return active, side, probability


def make_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side, probability = active_and_side(frame, control); rows = []; next_allowed = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"]); entry = decision + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed: continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        next_allowed = exit_time
        rows.append({"candidate": prereg.POLICY_ID, "control": control, "split": split, "source_date": frame.at[index, "source_date"], "decision_time": decision, "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "conditional_probability": float(probability.at[index]), "forecast_strength": float(abs(probability.at[index] - 0.5)), "strength_rank": float(frame.at[index, "strength_rank"]), "realized_variation": float(frame.at[index, "realized_variation"]), "variation_rank": float(frame.at[index, "variation_rank"])})
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    frame = clock[clock["split"].eq(split)]
    if frame.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(frame["side"].eq(1).sum()), int(frame["side"].eq(-1).sum()); months = frame["entry_time"].dt.strftime("%Y-%m").value_counts()
    return {"events": len(frame), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(frame), "max_month_share": int(months.max()) / len(frame)}


def checks(values: dict[str, dict[str, Any]]) -> dict[str, bool]:
    output = {}
    for split, row in values.items(): output[f"{split}_minimum_events"] = row["events"] >= MINIMUM[split]; output[f"{split}_side_balance"] = row["minority_side_share"] >= 0.20; output[f"{split}_month_concentration"] = row["max_month_share"] <= 0.45
    return output


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA: raise RuntimeError("HVVIXTE preregistration hash drift")
    source = materialize_features(); frame = load_features(); primary = make_clock(frame); controls = {name: make_clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True); SPLIT_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True); _write_gzip_csv(primary, CLOCK)
    for split in SPLITS: _write_gzip_csv(primary[primary["split"].eq(split)].reset_index(drop=True), SPLIT_DIR / f"{split}.csv.gz")
    for name, value in controls.items(): _write_gzip_csv(value, CONTROL_DIR / f"{name}.csv.gz")
    support = {split: stats(primary, split) for split in SPLITS}; support_checks = checks(support); passed = all(support_checks.values()); registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {"protocol_version": "hvvixte_24_source_support_v1", "policy_id": prereg.POLICY_ID, "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha256(SOURCE_MANIFEST), "manifest_hash": source["manifest_hash"]}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)}, "split_clocks": {split: {"path": str(SPLIT_DIR / f"{split}.csv.gz"), "sha256": sha256(SPLIT_DIR / f"{split}.csv.gz"), "rows": int(primary["split"].eq(split).sum())} for split in SPLITS}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(value), "promotion_authorized": False} for name, value in controls.items()}, "support": support, "support_checks": support_checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    payload = {**core, "manifest_hash": canonical_hash(core)}; RESULT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); return payload


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); result = run(); print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
