"""Materialize source-only HVKRECLV-24 clocks before Gross9 or economics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_regional_bank_close_location_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market


PREREG_SHA = "a86e278fe212d64d58a45af97c5a3907dc77587d93bfaa4158de6bc0d334b4df"
SOURCE_SHA = "e6d9b6e7414430927d59c93472da3e33d3b2ca7a89e36b220860cac591432671"
SOURCE = Path("data/high_volatility_regional_bank_close_location_relay_sources_2022_2026/kre_sessions.csv.gz")
SOURCE_MANIFEST = SOURCE.parent / "build_manifest.json"
SOURCE_MANIFEST_SHA = "d4b3a97d5e353ea2e25c8c85e56849a8659e57463ec3a00bba2081609e6315d5"
MARKET_HELPER = Path("training/build_scheduled_trend_concordance_relay_support.py")
MARKET_HELPER_SHA = "8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f"
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
STATE = SOURCE.parent / "daily_kre_states.csv.gz"
CLOCK = Path("data/high_volatility_regional_bank_close_location_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_regional_bank_close_location_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_regional_bank_close_location_relay_support_2026-08-11.json")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_variation_gate", "any_nonzero_close_location", "cash_return_direction", "one_session_stale_location", "direction_flip", "forced_long")
COLUMNS = (
    "candidate", "control", "split", "session_date", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side", "kre_cash_return",
    "close_location", "btc_variation", "btc_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_prior_midrank(values: pd.Series, maximum: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-maximum:], dtype=float)
        if np.isfinite(current) and len(prior) >= minimum:
            result.at[index] = ((prior < current).sum() + 0.5 * (prior == current).sum()) / len(prior)
        if np.isfinite(current):
            history.append(float(current))
    return result


def variation_states(market: pd.DataFrame) -> pd.DataFrame:
    candles = market.copy().sort_values("date").set_index("date")
    open_ = pd.to_numeric(candles.open, errors="coerce")
    close = pd.to_numeric(candles.close, errors="coerce")
    valid = np.isfinite(open_) & np.isfinite(close) & open_.gt(0) & close.gt(0)
    contiguous = candles.index.to_series().diff().eq(pd.Timedelta(minutes=5))
    squared = np.log(close / open_).pow(2)
    variation = np.sqrt(squared.rolling(288, min_periods=288).sum())
    complete = valid.rolling(288, min_periods=288).sum().eq(288) & contiguous.rolling(288, min_periods=288).sum().eq(288)
    decisions = pd.date_range(candles.index.min().ceil("D") + pd.Timedelta(hours=23), END, freq="1D", inclusive="left")
    lookup = variation.where(complete).rename("btc_variation")
    states = pd.DataFrame({"decision_time": decisions})
    states["btc_variation"] = lookup.reindex(decisions - pd.Timedelta(minutes=5)).to_numpy()
    states["btc_variation_rank"] = strict_prior_midrank(states.btc_variation)
    return states


def score_sessions(source: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    frame["session_date"] = pd.to_datetime(frame.session_date)
    if frame.session_date.duplicated().any() or not frame.session_date.is_monotonic_increasing:
        raise RuntimeError("HVKRECLV source session order drift")
    columns = ("kre_open", "kre_high", "kre_low", "kre_close")
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(frame[list(columns)]).all(axis=1)
    geometry = frame.kre_high.ge(frame[["kre_open", "kre_close"]].max(axis=1)) & frame.kre_low.le(frame[["kre_open", "kre_close"]].min(axis=1)) & frame.kre_high.gt(frame.kre_low)
    frame["kre_cash_return"] = np.log(frame.kre_close / frame.kre_open)
    frame["close_location"] = ((frame.kre_close - frame.kre_low) - (frame.kre_high - frame.kre_close)) / (frame.kre_high - frame.kre_low)
    frame["decision_time"] = (frame.session_date + pd.Timedelta(hours=23)).dt.tz_localize("UTC")
    frame = frame.merge(variation_states(market), on="decision_time", how="left", validate="one_to_one")
    frame["source_valid"] = finite & geometry & np.isfinite(frame[["kre_cash_return", "close_location"]]).all(axis=1) & frame.close_location.between(-1.000000001, 1.000000001)
    return frame


def build_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = states.shift(1) if control == "one_session_stale_location" else states
    location = used.close_location
    cash_return = used.kre_cash_return
    valid = used.source_valid.eq(True) & np.isfinite(location) & np.isfinite(cash_return)
    relation = location.abs().ge(0.5)
    side = np.sign(location).fillna(0).astype(int)
    if control == "any_nonzero_close_location":
        relation = location.ne(0)
    elif control == "cash_return_direction":
        relation = cash_return.ne(0)
        side = np.sign(cash_return).fillna(0).astype(int)
    variation_gate = pd.Series(True, index=states.index) if control == "no_variation_gate" else states.btc_variation_rank.ge(0.65)
    active = valid & relation & variation_gate & np.isfinite(states.btc_variation_rank) & side.ne(0)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=states.index)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in states.index[active]:
        decision = pd.Timestamp(states.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        reserved_until = exit_time
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "session_date": states.at[index, "session_date"], "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(side.at[index]), "kre_cash_return": float(cash_return.at[index]),
            "close_location": float(location.at[index]),
            "btc_variation": float(states.at[index, "btc_variation"]),
            "btc_variation_rank": float(states.at[index, "btc_variation_rank"]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected.side.eq(1).sum()), int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(selected), "max_month_share": int(months.max()) / len(selected)}


def run() -> dict[str, Any]:
    bindings = {prereg.DEFAULT_OUTPUT: PREREG_SHA, SOURCE: SOURCE_SHA, SOURCE_MANIFEST: SOURCE_MANIFEST_SHA, MARKET_HELPER: MARKET_HELPER_SHA, MARKET: MARKET_SHA}
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise RuntimeError(f"HVKRECLV binding drift: {path}")
    market, market_source = load_market()
    states = score_sessions(pd.read_csv(SOURCE), market)
    primary = build_clock(states)
    controls = {name: build_clock(states, name) for name in CONTROLS}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, STATE)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: passed for name, values in support.items() for key, passed in (
        (f"{name}_minimum_events", values["events"] >= MINIMUM[name]),
        (f"{name}_side_balance", values["minority_side_share"] >= 0.2),
        (f"{name}_month_concentration", values["max_month_share"] <= 0.45),
    )}
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    core = {
        "protocol_version": "hvkreclv_24_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "bindings": {str(path): expected for path, expected in bindings.items()},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": SOURCE_MANIFEST_SHA, "manifest_hash": source_manifest["manifest_hash"]},
        "market_source": market_source, "completed_preentry_sources_opened": True,
        "candidate_incidence_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "state": {"path": str(STATE), "sha256": sha(STATE), "rows": len(states)},
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(candidate), "promotion_authorized": False} for name, candidate in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
