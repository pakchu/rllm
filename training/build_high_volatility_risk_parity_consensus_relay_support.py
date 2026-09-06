"""Materialize source-only HVRPC-24 clocks before Gross9 or economics."""
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

from training import preregister_high_volatility_risk_parity_consensus_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market


PREREG_SHA = "fbf24f786677c02ad63e91329c96487cebe545e61cb4d31500114203d54ca8da"
SOURCE_SHA = "a49da1ed3d1f1065656a473f2e102dc4602a7ff72508e4df2a74ace9abcaf6eb"
SOURCE_MANIFEST = prereg.SOURCE.parent / "build_manifest.json"
SOURCE_MANIFEST_SHA = "0ca68700205a365dc239e10c2997b7f02f49cb73e20030f2e4acab15de286dd9"
MARKET_HELPER = Path("training/build_scheduled_trend_concordance_relay_support.py")
MARKET_HELPER_SHA = "8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f"
STATE = prereg.SOURCE.parent / "daily_consensus_states.csv.gz"
CLOCK = Path("data/high_volatility_risk_parity_consensus_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_risk_parity_consensus_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_risk_parity_consensus_relay_support_2026-08-10.json")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_variation_gate", "spy_only", "tlt_only", "one_session_stale_consensus", "direction_flip", "same_clock_forced_long")
COLUMNS = (
    "candidate", "control", "split", "session_date", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side", "spy_return",
    "tlt_return", "btc_variation", "btc_variation_rank",
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
        raise RuntimeError("HVRPC source session order drift")
    for column in ("spy_open", "spy_close", "tlt_open", "tlt_close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["spy_return"] = np.log(frame.spy_close / frame.spy_open)
    frame["tlt_return"] = np.log(frame.tlt_close / frame.tlt_open)
    frame["decision_time"] = (frame.session_date + pd.Timedelta(hours=23)).dt.tz_localize("UTC")
    frame = frame.merge(variation_states(market), on="decision_time", how="left", validate="one_to_one")
    frame["source_valid"] = np.isfinite(frame[["spy_return", "tlt_return"]]).all(axis=1) & frame.spy_return.ne(0) & frame.tlt_return.ne(0)
    frame["consensus"] = np.sign(frame.spy_return).eq(np.sign(frame.tlt_return))
    return frame


def build_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = states.shift(1) if control == "one_session_stale_consensus" else states
    spy, tlt = used.spy_return, used.tlt_return
    valid = used.source_valid.eq(True) & np.isfinite(spy) & np.isfinite(tlt)
    if control == "spy_only":
        relation = spy.ne(0)
    elif control == "tlt_only":
        relation = tlt.ne(0)
    else:
        relation = np.sign(spy).eq(np.sign(tlt))
    variation_gate = pd.Series(True, index=states.index) if control == "no_variation_gate" else states.btc_variation_rank.ge(0.65)
    active = valid & relation & variation_gate & np.isfinite(states.btc_variation_rank)
    side = np.sign(spy).fillna(0).astype(int)
    if control == "tlt_only":
        side = np.sign(tlt).fillna(0).astype(int)
    elif control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
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
            "side": int(side.at[index]), "spy_return": float(spy.at[index]), "tlt_return": float(tlt.at[index]),
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
            raise RuntimeError(f"HVRPC binding drift: {path}")
    market, market_source = load_market()
    states = score_sessions(pd.read_csv(prereg.SOURCE), market)
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
        "protocol_version": "hvrpc_24_source_support_v1", "policy_id": prereg.POLICY_ID,
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
