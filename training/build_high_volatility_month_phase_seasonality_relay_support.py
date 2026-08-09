"""Materialize source-only OOS HVMPS-12 clocks."""
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

from training import preregister_high_volatility_month_phase_seasonality_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market


PREREG_SHA = "2e1f9577317c7ec77c49a35b6e0ae781ebc322beb5a8d207296c8ae63f3b734e"
MODEL = Path("data/high_volatility_month_phase_seasonality_relay_model_2026-08-10.json")
MODEL_SHA = "0437809c4469087d3465b7c11e2df7c6445bda199f5dfef9854b99b1cd744f73"
MODEL_FREEZE = Path("results/high_volatility_month_phase_seasonality_relay_model_freeze_2026-08-10.json")
MODEL_FREEZE_SHA = "3f970e348cd750cf53db5cece65b247f16eb62da86d714826db245abea645eb3"
MARKET_HELPER = Path("training/build_scheduled_trend_concordance_relay_support.py")
MARKET_HELPER_SHA = "8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f"
STATE = Path("data/high_volatility_month_phase_seasonality_relay_states_2023_2026.csv.gz")
CLOCK = Path("data/high_volatility_month_phase_seasonality_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_month_phase_seasonality_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_month_phase_seasonality_relay_support_2026-08-10.json")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {"train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")), "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")), "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")), "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END)}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_variation_gate", "constant_selected_date_long", "one_day_stale_phase", "direction_flip", "same_clock_forced_long")
COLUMNS = ("candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "day_of_month", "fit_mean_log_return", "btc_variation", "btc_variation_rank")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_prior_midrank(values: pd.Series, maximum: int = 270, minimum: int = 180) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce"); ranked = pd.Series(np.nan, index=values.index, dtype=float); history: list[float] = []
    for index, current in values.items():
        prior = np.asarray(history[-maximum:], dtype=float)
        if np.isfinite(current) and len(prior) >= minimum:
            ranked.at[index] = ((prior < current).sum() + 0.5 * (prior == current).sum()) / len(prior)
        if np.isfinite(current): history.append(float(current))
    return ranked


def daily_states(market: pd.DataFrame) -> pd.DataFrame:
    candles = market.sort_values("date").set_index("date"); close = pd.to_numeric(candles.close, errors="coerce")
    valid = np.isfinite(close) & close.gt(0); contiguous = candles.index.to_series().diff().eq(pd.Timedelta(minutes=5))
    variation = np.sqrt(np.log(close / close.shift()).pow(2).rolling(2016, min_periods=2016).sum())
    complete = valid.rolling(2017, min_periods=2017).sum().eq(2017) & contiguous.rolling(2016, min_periods=2016).sum().eq(2016)
    decisions = pd.date_range(candles.index.min().ceil("D"), END, freq="1D", inclusive="left")
    states = pd.DataFrame({"decision_time": decisions, "btc_variation": variation.where(complete).reindex(decisions - pd.Timedelta(minutes=5)).to_numpy()})
    states["btc_variation_rank"] = strict_prior_midrank(states.btc_variation); states["day_of_month"] = states.decision_time.dt.day
    return states


def build_clock(states: pd.DataFrame, model: dict[str, Any], control: str = "primary") -> pd.DataFrame:
    selected = {int(item["day_of_month"]): item for item in model["selected"]}
    stale = control == "one_day_stale_phase"
    phase = states.day_of_month.shift(1) if stale else states.day_of_month
    side = phase.map({day: int(item["side"]) for day, item in selected.items()})
    score = phase.map({day: float(item["fit_mean_log_return"]) for day, item in selected.items()})
    active = side.notna() & np.isfinite(states.btc_variation_rank)
    if control not in ("no_variation_gate", "constant_selected_date_long"):
        active &= states.btc_variation_rank.ge(0.65)
    if control in ("constant_selected_date_long", "same_clock_forced_long"):
        side = pd.Series(1, index=states.index)
    elif control == "direction_flip":
        side = -side
    rows: list[dict[str, Any]] = []
    for index in states.index[active & states.decision_time.ge(SPLITS["train"][0])]:
        decision = pd.Timestamp(states.at[index, "decision_time"]); entry = decision + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=12)
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        rows.append({"candidate": prereg.POLICY_ID, "control": control, "split": split, "decision_time": decision, "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "day_of_month": int(phase.at[index]), "fit_mean_log_return": float(score.at[index]), "btc_variation": float(states.at[index, "btc_variation"]), "btc_variation_rank": float(states.at[index, "btc_variation_rank"])})
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected.side.eq(1).sum()), int(selected.side.eq(-1).sum()); months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(selected), "max_month_share": int(months.max()) / len(selected)}


def run() -> dict[str, Any]:
    bindings = {prereg.DEFAULT_OUTPUT: PREREG_SHA, MODEL: MODEL_SHA, MODEL_FREEZE: MODEL_FREEZE_SHA, MARKET_HELPER: MARKET_HELPER_SHA, prereg.MARKET: prereg.MARKET_SHA}
    for path, expected in bindings.items():
        if sha(path) != expected: raise RuntimeError(f"HVMPS binding drift: {path}")
    model = json.loads(MODEL.read_text()); freeze = json.loads(MODEL_FREEZE.read_text())
    if freeze.get("oos_source_incidence_opened") is not False or freeze.get("oos_outcomes_opened") is not False: raise RuntimeError("HVMPS model was not frozen before OOS")
    market, market_source = load_market(); states = daily_states(market); primary = build_clock(states, model); controls = {name: build_clock(states, model, name) for name in CONTROLS}
    STATE.parent.mkdir(parents=True, exist_ok=True); CLOCK.parent.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, STATE); _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items(): _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: passed for name, values in support.items() for key, passed in ((f"{name}_minimum_events", values["events"] >= MINIMUM[name]), (f"{name}_side_balance", values["minority_side_share"] >= 0.2), (f"{name}_month_concentration", values["max_month_share"] <= 0.45))}; passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {"protocol_version": "hvmps_12_source_support_v1", "policy_id": prereg.POLICY_ID, "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "bindings": {str(path): expected for path, expected in bindings.items()}, "frozen_model": {"path": str(MODEL), "sha256": MODEL_SHA, "manifest_hash": model["manifest_hash"], "selected": model["selected"]}, "market_source": market_source, "completed_preentry_sources_opened": True, "oos_candidate_incidence_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "state": {"path": str(STATE), "sha256": sha(STATE), "rows": len(states)}, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(candidate), "promotion_authorized": False} for name, candidate in controls.items()}, "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}; RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); report = run(); print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
