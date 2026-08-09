"""Materialize source-only HVQGR-12 clocks before Gross9 or economics."""
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

from training import preregister_high_volatility_qqq_gold_rotation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market


PREREG_SHA = "f0dd9c8660aed4391157dd9eea0e30a84ffbe0bacdd4b4bdf835016d02374e08"
SOURCE_SHA = "17a0e631b90d911c81310184734361432a7b1eb2e9a145c638abfe3762a53fc0"
SOURCE_MANIFEST = prereg.SOURCE.parent / "build_manifest.json"
SOURCE_MANIFEST_SHA = "7947396e12aed9264aaabbc7f56173848cf550a25303fcdf2a7bf5d64b7e57a3"
MARKET_HELPER = Path("training/build_scheduled_trend_concordance_relay_support.py")
MARKET_HELPER_SHA = "8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f"
STATE = prereg.SOURCE.parent / "daily_rotation_states.csv.gz"
CLOCK = Path("data/high_volatility_qqq_gold_rotation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_qqq_gold_rotation_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_qqq_gold_rotation_relay_support_2026-08-09.json")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_variation_gate", "qqq_only", "gld_inverse_only", "one_session_stale_rotation", "direction_flip", "same_clock_forced_long")
COLUMNS = (
    "candidate", "control", "split", "session_date", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side", "qqq_return",
    "gld_return", "btc_variation", "btc_variation_rank",
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
    close = pd.to_numeric(candles.close, errors="coerce")
    valid = np.isfinite(close) & close.gt(0)
    contiguous = candles.index.to_series().diff().eq(pd.Timedelta(minutes=5))
    squared = np.log(close / close.shift(1)).pow(2)
    variation = np.sqrt(squared.rolling(2016, min_periods=2016).sum())
    complete = valid.rolling(2017, min_periods=2017).sum().eq(2017) & contiguous.rolling(2016, min_periods=2016).sum().eq(2016)
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
        raise RuntimeError("HVQGR source session order drift")
    for column in ("qqq_open", "qqq_close", "gld_open", "gld_close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["qqq_return"] = np.log(frame.qqq_close / frame.qqq_open)
    frame["gld_return"] = np.log(frame.gld_close / frame.gld_open)
    frame["decision_time"] = (frame.session_date + pd.Timedelta(hours=23)).dt.tz_localize("UTC")
    frame = frame.merge(variation_states(market), on="decision_time", how="left", validate="one_to_one")
    frame["source_valid"] = np.isfinite(frame[["qqq_return", "gld_return"]]).all(axis=1) & frame.qqq_return.ne(0) & frame.gld_return.ne(0)
    frame["rotation"] = np.sign(frame.qqq_return).ne(np.sign(frame.gld_return))
    return frame


def build_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = states.shift(1) if control == "one_session_stale_rotation" else states
    qqq, gld = used.qqq_return, used.gld_return
    valid = used.source_valid.eq(True) & np.isfinite(qqq) & np.isfinite(gld)
    if control == "qqq_only":
        relation = qqq.ne(0)
    elif control == "gld_inverse_only":
        relation = gld.ne(0)
    else:
        relation = np.sign(qqq).ne(np.sign(gld))
    variation_gate = pd.Series(True, index=states.index) if control == "no_variation_gate" else states.btc_variation_rank.ge(0.65)
    active = valid & relation & variation_gate & np.isfinite(states.btc_variation_rank)
    side = np.sign(qqq).fillna(0).astype(int)
    if control == "gld_inverse_only":
        side = -np.sign(gld).fillna(0).astype(int)
    elif control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1, index=states.index)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in states.index[active]:
        decision = pd.Timestamp(states.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
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
            "side": int(side.at[index]), "qqq_return": float(qqq.at[index]), "gld_return": float(gld.at[index]),
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
    bindings = {prereg.DEFAULT_OUTPUT: PREREG_SHA, prereg.SOURCE: SOURCE_SHA, SOURCE_MANIFEST: SOURCE_MANIFEST_SHA, MARKET_HELPER: MARKET_HELPER_SHA, prereg.MARKET: prereg.MARKET_SHA}
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise RuntimeError(f"HVQGR binding drift: {path}")
    market, market_source = load_market()
    states = score_sessions(pd.read_csv(prereg.SOURCE), market)
    primary = build_clock(states)
    controls = {name: build_clock(states, name) for name in CONTROLS}
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
        "protocol_version": "hvqgr_12_source_support_v1", "policy_id": prereg.POLICY_ID,
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
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
