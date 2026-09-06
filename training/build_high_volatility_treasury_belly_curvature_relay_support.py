"""Build source-only HVTBCR-24 clocks before Gross9 or economics."""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_treasury_belly_curvature_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market


PREREG_SHA = "a7cf756abf1d5ccd096bb8697ac1740abc015bcc1af46e09468a93411994670e"
HELPER = Path("training/build_scheduled_trend_concordance_relay_support.py")
HELPER_SHA = "8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f"
END = pd.Timestamp("2026-08-01T00:00:00Z")
STATE = Path("data/high_volatility_treasury_belly_curvature_relay_sources_2023_2026/daily_states.csv.gz")
CLOCK = Path("data/high_volatility_treasury_belly_curvature_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_treasury_belly_curvature_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_treasury_belly_curvature_relay_support_2026-08-10.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_variation_gate",
    "curvature_level",
    "one_observation_stale_curvature_change",
    "direction_flip",
    "same_clock_forced_long",
)
COLUMNS = (
    "candidate", "control", "split", "source_day", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side", "yield_2y",
    "yield_5y", "yield_10y", "curvature", "curvature_change", "btc_variation",
    "btc_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_treasury_xml(paths: list[Path]) -> pd.DataFrame:
    namespace = {"atom": "http://www.w3.org/2005/Atom", "d": "http://schemas.microsoft.com/ado/2007/08/dataservices"}
    rows: list[dict[str, Any]] = []
    for path in paths:
        root = ET.parse(path).getroot()
        for entry in root.findall("atom:entry", namespace):
            properties = entry.find("atom:content/m:properties", {
                **namespace, "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
            })
            if properties is None:
                continue
            values = {child.tag.rsplit("}", 1)[-1]: child.text for child in properties}
            rows.append({
                "source_day": values.get("NEW_DATE"),
                "yield_2y": values.get("BC_2YEAR"),
                "yield_5y": values.get("BC_5YEAR"),
                "yield_10y": values.get("BC_10YEAR"),
                "source_file": path.name,
            })
    frame = pd.DataFrame(rows)
    frame["source_day"] = pd.to_datetime(frame.source_day, utc=True, errors="coerce")
    for column in ("yield_2y", "yield_5y", "yield_10y"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("source_day").reset_index(drop=True)
    if frame.source_day.isna().any() or frame.source_day.duplicated().any():
        raise RuntimeError("HVTBCR invalid or duplicate Treasury source day")
    frame["source_valid"] = np.isfinite(frame[["yield_2y", "yield_5y", "yield_10y"]]).all(axis=1)
    frame["source_valid"] &= frame[["yield_2y", "yield_5y", "yield_10y"]].gt(0).all(axis=1)
    gaps = frame.source_day.diff().dt.days
    frame["transition_valid"] = frame.source_valid & frame.source_valid.shift(1, fill_value=False) & gaps.between(1, 5)
    frame["curvature"] = 2.0 * frame.yield_5y - frame.yield_2y - frame.yield_10y
    frame["curvature_change"] = frame.curvature.diff().where(frame.transition_valid)
    return frame


def strict_prior_rank(values: pd.Series, maximum: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-maximum:], dtype=float)
        if np.isfinite(current) and len(prior) >= minimum:
            output.at[index] = ((prior < current).sum() + 0.5 * (prior == current).sum()) / len(prior)
        if np.isfinite(current):
            history.append(float(current))
    return output


def score_states(source: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    states = source.copy().sort_values("source_day").reset_index(drop=True)
    states["decision_time"] = states.source_day + pd.Timedelta(hours=36)
    states["curvature_change_valid"] = (
        states.transition_valid.astype(bool)
        & np.isfinite(states.curvature_change)
        & states.curvature_change.ne(0)
    )
    candles = market.copy()
    candles["date"] = pd.to_datetime(candles.date, utc=True)
    candles = candles.sort_values("date").set_index("date")
    close = pd.to_numeric(candles.close, errors="coerce")
    valid = np.isfinite(close) & close.gt(0)
    contiguous = candles.index.to_series().diff().eq(pd.Timedelta(minutes=5))
    squared_returns = np.log(close / close.shift(1)).pow(2)
    variation = np.sqrt(squared_returns.rolling(2016, min_periods=2016).sum())
    complete = valid.rolling(2017, min_periods=2017).sum().eq(2017) & contiguous.rolling(2016, min_periods=2016).sum().eq(2016)
    lookup = pd.DataFrame({
        "decision_time": candles.index + pd.Timedelta(minutes=5),
        "btc_variation": variation.where(complete).to_numpy(),
    })
    daily_times = pd.date_range(candles.index.min().normalize() + pd.Timedelta(hours=12), END, freq="1D")
    daily = pd.DataFrame({"decision_time": daily_times}).merge(lookup, on="decision_time", how="left", validate="one_to_one")
    daily["btc_variation_rank"] = strict_prior_rank(daily.btc_variation)
    states = states.merge(daily, on="decision_time", how="left", validate="one_to_one")
    states["state_valid"] = states.curvature_change_valid & np.isfinite(states.btc_variation)
    return states


def build_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    if control == "one_observation_stale_curvature_change":
        signal = -states.curvature_change.shift(1)
        signal_valid = states.curvature_change_valid.shift(1, fill_value=False)
    elif control == "curvature_level":
        signal = -states.curvature
        signal_valid = states.source_valid.astype(bool) & np.isfinite(signal) & signal.ne(0)
    else:
        signal = -states.curvature_change
        signal_valid = states.curvature_change_valid.astype(bool)
    valid = signal_valid & np.isfinite(signal) & signal.ne(0) & np.isfinite(states.btc_variation)
    variation_gate = pd.Series(True, index=states.index) if control == "no_variation_gate" else states.btc_variation_rank.ge(0.65)
    active = valid & variation_gate
    side = np.sign(signal).fillna(0).astype(int)
    if control == "direction_flip":
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
        row = states.loc[index]
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "source_day": row.source_day, "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(side.at[index]), "yield_2y": float(row.yield_2y),
            "yield_5y": float(row.yield_5y), "yield_10y": float(row.yield_10y),
            "curvature": float(row.curvature), "curvature_change": float(row.curvature_change),
            "btc_variation": float(row.btc_variation),
            "btc_variation_rank": float(row.btc_variation_rank),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected.side.eq(1).sum()), int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts,
            "minority_side_share": min(longs, shorts) / len(selected),
            "max_month_share": int(months.max()) / len(selected)}


def run() -> dict[str, Any]:
    bindings: dict[Path, str] = {
        prereg.DEFAULT_OUTPUT: PREREG_SHA,
        prereg.SOURCE_MANIFEST: prereg.SOURCE_MANIFEST_SHA,
        HELPER: HELPER_SHA,
        prereg.MARKET: prereg.MARKET_SHA,
        **{prereg.SOURCE_DIR / name: digest for name, digest in prereg.SOURCE_HASHES.items()},
    }
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise RuntimeError(f"HVTBCR binding drift: {path}")
    source = load_treasury_xml(list(prereg.SOURCE_DIR / name for name in sorted(prereg.SOURCE_HASHES)))
    market, market_source = load_market()
    states = score_states(source, market)
    primary = build_clock(states)
    controls = {name: build_clock(states, name) for name in CONTROLS}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, STATE)
    _write_gzip_csv(primary, CLOCK)
    for name, clock in controls.items():
        _write_gzip_csv(clock, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: value for name, values in support.items() for key, value in (
        (f"{name}_minimum_events", values["events"] >= MINIMUM[name]),
        (f"{name}_side_balance", values["minority_side_share"] >= 0.2),
        (f"{name}_month_concentration", values["max_month_share"] <= 0.45),
    )}
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvtbcr_24_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA,
                            "manifest_hash": registration["manifest_hash"]},
        "bindings": {str(path): expected for path, expected in bindings.items()},
        "market_source": market_source, "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "source_state": {"path": str(STATE), "sha256": sha(STATE), "rows": len(states)},
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"),
                            "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(clock),
                            "promotion_authorized": False} for name, clock in controls.items()},
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
