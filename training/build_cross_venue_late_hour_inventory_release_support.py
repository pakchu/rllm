"""Build source-support clocks for CVLIR-6 without post-entry outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import build_options_led_intrahour_absorption_support as intrahour
from training import preregister_cross_venue_late_hour_inventory_release as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


CLOCK = Path("data/cross_venue_late_hour_inventory_release_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/cross_venue_late_hour_inventory_release_controls_2023_2026")
RESULT = Path("results/cross_venue_late_hour_inventory_release_support_2026-08-08.json")
SPLITS = base.SPLITS
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_vol_disagreement", "no_quiet_first_half", "no_late_tail", "no_abs_oi_tail", "direction_flip")
ECONOMIC_OUTCOMES_AUTHORIZED = False
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "bvol_body", "dvol_body",
    "first_half_return", "prior_abs_first_half_q40", "second_half_return",
    "prior_abs_second_half_q90", "oi_change", "prior_abs_oi_q75",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def features() -> pd.DataFrame:
    frame = intrahour.features().copy()
    first = frame["first_half_return"].abs().where(frame["price_valid"])
    second = frame["second_half_return"].abs().where(frame["price_valid"])
    absolute_oi = frame["oi_change"].abs().where(frame["source_valid"])
    frame["first_half_q40"] = first.shift(1).rolling(720, min_periods=672).quantile(0.40)
    frame["second_half_q90"] = second.shift(1).rolling(720, min_periods=672).quantile(0.90)
    frame["abs_oi_q75"] = absolute_oi.shift(1).rolling(720, min_periods=672).quantile(0.75)
    relevant = ["bvol_body", "dvol_body", "first_half_return", "second_half_return", "oi_change"]
    frame["base_valid"] = (
        frame["source_valid"] & frame["price_valid"]
        & np.isfinite(frame[relevant]).all(axis=1)
        & frame[["bvol_open", "bvol_close", "dvol_open", "dvol_close", "oi_current", "oi_prior"]].gt(0).all(axis=1)
    )
    return frame


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control == "no_vol_disagreement":
        volatility = frame["bvol_body"].ne(0) & frame["dvol_body"].ne(0)
    else:
        volatility = (
            frame["bvol_body"].ne(0) & frame["dvol_body"].ne(0)
            & np.sign(frame["bvol_body"]).eq(-np.sign(frame["dvol_body"]))
        )
    quiet = frame["first_half_return"].notna()
    if control != "no_quiet_first_half":
        quiet &= frame["first_half_q40"].notna() & frame["first_half_return"].abs().le(frame["first_half_q40"])
    late = frame["second_half_return"].ne(0)
    if control != "no_late_tail":
        late &= frame["second_half_q90"].notna() & frame["second_half_return"].abs().ge(frame["second_half_q90"])
    oi_gate = frame["oi_change"].ne(0)
    if control != "no_abs_oi_tail":
        oi_gate &= frame["abs_oi_q75"].notna() & frame["oi_change"].abs().ge(frame["abs_oi_q75"])
    active = frame["base_valid"] & volatility & quiet & late & oi_gate
    onset = (
        active & ~active.shift(1, fill_value=False)
        & frame["base_valid"].shift(1, fill_value=False)
        & frame["decision_time"].diff().eq(pd.Timedelta(hours=1))
    )
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[onset]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=6)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        side = int(np.sign(frame.at[index, "second_half_return"]))
        if control == "direction_flip":
            side *= -1
        next_allowed = exit_time
        rows.append({
            "candidate": "CVLIR-6", "control": control, "split": split,
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": side,
            "bvol_body": float(frame.at[index, "bvol_body"]),
            "dvol_body": float(frame.at[index, "dvol_body"]),
            "first_half_return": float(frame.at[index, "first_half_return"]),
            "prior_abs_first_half_q40": float(frame.at[index, "first_half_q40"]),
            "second_half_return": float(frame.at[index, "second_half_return"]),
            "prior_abs_second_half_q90": float(frame.at[index, "second_half_q90"]),
            "oi_change": float(frame.at[index, "oi_change"]),
            "prior_abs_oi_q75": float(frame.at[index, "abs_oi_q75"]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def split_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock["split"].eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset["side"].eq(1).sum())
    shorts = int(subset["side"].eq(-1).sum())
    months = subset["entry_time"].dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    frame = features()
    primary = build_clock(frame)
    controls = {name: build_clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, control in controls.items():
        _write_gzip_csv(control, CONTROL_DIR / f"{name}.csv.gz")
    statistics = {name: split_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, stats in statistics.items():
        checks[f"{name}_minimum_events"] = stats["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = stats["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = stats["max_month_share"] <= 0.45
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    source_manifest = intrahour.PRICE_DIR / "manifest.json"
    passed = all(checks.values())
    core = {
        "protocol_version": "cvlir_6_source_support_v1", "policy_id": "CVLIR-6",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha256(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(source_manifest), "sha256": sha256(source_manifest)},
        "completed_preentry_feature_price_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(control)} for name, control in controls.items()},
        "support": statistics, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": ECONOMIC_OUTCOMES_AUTHORIZED,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
