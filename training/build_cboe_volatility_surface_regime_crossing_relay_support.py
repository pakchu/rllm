"""Build outcome-blind source-support clocks for preregistered CVSRC-24."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from training import preregister_cboe_volatility_surface_regime_crossing_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

CLOCK = Path("data/cboe_volatility_surface_regime_crossing_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/cboe_volatility_surface_regime_crossing_relay_controls_2023_2026")
RESULT = Path("results/cboe_volatility_surface_regime_crossing_relay_support_2026-08-08.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_vix_high", "term_only", "tail_only", "outer_state_onset", "direction_flip")
NEW_YORK = ZoneInfo("America/New_York")
ECONOMIC_OUTCOMES_AUTHORIZED = False
COLUMNS = (
    "candidate", "control", "split", "observation_date", "previous_observation_date",
    "decision_time", "feature_available_time", "entry_time", "exit_time", "side",
    "skew_rank", "vvix_relative_rank", "front_rank", "broad_rank", "tail", "term",
    "surface", "previous_surface", "vix_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 252, minimum: int = 126) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            result.at[index] = (np.sum(array < current) + 0.5 * np.sum(array == current)) / len(array)
        if math.isfinite(current):
            history.append(current)
    return result


def features(source: Path = prereg.SOURCE) -> pd.DataFrame:
    if sha(source) != prereg.SOURCE_SHA256 or sha(prereg.SOURCE_MANIFEST) != prereg.SOURCE_MANIFEST_SHA256:
        raise RuntimeError("CVSRC frozen source binding changed")
    frame = pd.read_csv(source, compression="gzip")
    expected = ["observation_date", "SKEW_close", "VVIX_close", "VIX9D_close", "VIX_close", "VIX3M_close"]
    if frame.columns.tolist() != expected:
        raise RuntimeError("CVSRC source schema changed")
    frame["observation_date"] = pd.to_datetime(frame.observation_date, format="%Y-%m-%d")
    if not frame.observation_date.is_monotonic_increasing or frame.observation_date.duplicated().any():
        raise RuntimeError("CVSRC source dates are not strictly increasing")
    for column in expected[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[expected[1:]]).all().all() or not frame[expected[1:]].gt(0).all().all():
        raise RuntimeError("CVSRC source values must be finite and positive")
    raw = {
        "skew_rank": np.log(frame.SKEW_close / 100.0),
        "vvix_relative_rank": np.log(frame.VVIX_close / frame.VIX_close),
        "front_rank": np.log(frame.VIX9D_close / frame.VIX_close),
        "broad_rank": np.log(frame.VIX_close / frame.VIX3M_close),
        "vix_rank": np.log(frame.VIX_close),
    }
    for name, values in raw.items():
        frame[name] = strict_prior_midrank(values)
    frame["tail"] = 0.5 * (frame.skew_rank + frame.vvix_relative_rank)
    frame["term"] = 0.5 * (frame.front_rank + frame.broad_rank)
    frame["surface"] = 0.5 * (frame.tail + frame.term)
    frame["previous_surface"] = frame.surface.shift(1)
    frame["previous_tail"] = frame.tail.shift(1)
    frame["previous_term"] = frame.term.shift(1)
    return frame


def _side(current: pd.Series, previous: pd.Series, mode: str) -> pd.Series:
    if mode == "outer_state_onset":
        long = current.le(0.25) & previous.gt(0.25)
        short = current.ge(0.75) & previous.lt(0.75)
    else:
        long = current.le(0.25) & previous.gt(0.50)
        short = current.ge(0.75) & previous.lt(0.50)
    side = pd.Series(0, index=current.index, dtype=int)
    side.loc[long] = 1
    side.loc[short] = -1
    return side


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown CVSRC control: {control}")
    current_name = "term" if control == "term_only" else "tail" if control == "tail_only" else "surface"
    previous_name = "previous_term" if control == "term_only" else "previous_tail" if control == "tail_only" else "previous_surface"
    side = _side(frame[current_name], frame[previous_name], control)
    eligible = side.ne(0) & frame.vix_rank.notna()
    if control != "no_vix_high":
        eligible &= frame.vix_rank.ge(0.60)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[eligible]:
        if index + 1 >= len(frame):
            continue
        observation = frame.at[index, "observation_date"]
        next_date = frame.at[index + 1, "observation_date"]
        entry = pd.Timestamp(next_date.date()).tz_localize(NEW_YORK) + pd.Timedelta(hours=9, minutes=35)
        entry = entry.tz_convert("UTC")
        exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        selected_side = int(side.at[index])
        if control == "direction_flip":
            selected_side = -selected_side
        next_allowed = exit_time
        rows.append({
            "candidate": "CVSRC-24", "control": control, "split": split,
            "observation_date": observation.date().isoformat(),
            "previous_observation_date": frame.at[index - 1, "observation_date"].date().isoformat(),
            "decision_time": pd.Timestamp(observation.date()).tz_localize(NEW_YORK) + pd.Timedelta(hours=16),
            "feature_available_time": entry, "entry_time": entry, "exit_time": exit_time, "side": selected_side,
            "skew_rank": float(frame.at[index, "skew_rank"]), "vvix_relative_rank": float(frame.at[index, "vvix_relative_rank"]),
            "front_rank": float(frame.at[index, "front_rank"]), "broad_rank": float(frame.at[index, "broad_rank"]),
            "tail": float(frame.at[index, "tail"]), "term": float(frame.at[index, "term"]),
            "surface": float(frame.at[index, "surface"]), "previous_surface": float(frame.at[index, "previous_surface"]),
            "vix_rank": float(frame.at[index, "vix_rank"]),
        })
    result = pd.DataFrame(rows, columns=COLUMNS)
    for column in ("decision_time", "feature_available_time", "entry_time", "exit_time"):
        if not result.empty:
            result[column] = pd.to_datetime(result[column], utc=True)
    return result


def stats(clock_frame: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock_frame[clock_frame.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    preregistration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(preregistration)
    feature_frame = features()
    primary = clock(feature_frame)
    controls = {name: clock(feature_frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, control in controls.items():
        _write_gzip_csv(control, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, summary in support.items():
        checks[f"{name}_minimum_events"] = summary["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = summary["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = summary["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "cvsrc_24_source_support_v1", "policy_id": "CVSRC-24",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha(prereg.DEFAULT_OUTPUT), "manifest_hash": preregistration["manifest_hash"]},
        "source_manifest": {"path": str(prereg.SOURCE_MANIFEST), "sha256": sha(prereg.SOURCE_MANIFEST)},
        "source_panel": {"path": str(prereg.SOURCE), "sha256": sha(prereg.SOURCE)},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(value)} for name, value in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": ECONOMIC_OUTCOMES_AUTHORIZED,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    payload = run()
    print(json.dumps({"passed": payload["support_passed"], "support": payload["support"]}, indent=2))
