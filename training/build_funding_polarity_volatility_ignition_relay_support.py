"""Build outcome-blind source-support clocks for FPVIR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import preregister_funding_polarity_volatility_ignition_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


SOURCE_DIR = base.SOURCE_DIR
CLOCK = Path("data/funding_polarity_volatility_ignition_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/funding_polarity_volatility_ignition_relay_controls_2023_2026")
RESULT = Path("results/funding_polarity_volatility_ignition_relay_support_2026-08-08.json")
SPLITS = base.SPLITS
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_high_volatility", "no_sign_rotation", "no_amplitude_expansion",
    "one_settlement_stale_volatility", "direction_flip",
)
COLUMNS = (
    "candidate", "control", "split", "settlement_time", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side",
    "funding_rate", "previous_funding_rate", "funding_amplitude_ratio",
    "bvol_close", "dvol_close", "prior_bvol_q60", "prior_dvol_q60",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def features(source_dir: Path = SOURCE_DIR) -> pd.DataFrame:
    bvol = pd.read_csv(source_dir / "bvol_hourly.csv.gz", compression="gzip")
    dvol = pd.read_csv(source_dir / "dvol_hourly.csv.gz", compression="gzip")
    funding = pd.read_csv(source_dir / "funding.csv.gz", compression="gzip")
    b = pd.DataFrame({
        "decision_time": pd.to_datetime(bvol["feature_available_time_utc"], utc=True, format="mixed"),
        "bvol_close": pd.to_numeric(bvol["close"], errors="coerce"),
        "bvol_valid": bvol["feature_valid"].astype(str).str.lower().eq("true"),
    })
    d = pd.DataFrame({
        "decision_time": pd.to_datetime(dvol["close_time"], utc=True, format="mixed"),
        "dvol_close": pd.to_numeric(dvol["close"], errors="coerce"),
    })
    vol = b.merge(d, on="decision_time", validate="one_to_one").sort_values("decision_time")
    vol["vol_valid"] = (
        vol["bvol_valid"] & np.isfinite(vol[["bvol_close", "dvol_close"]]).all(axis=1)
        & vol[["bvol_close", "dvol_close"]].gt(0).all(axis=1)
    )
    vol["prior_bvol_q60"] = (
        vol["bvol_close"].where(vol["vol_valid"]).shift(1)
        .rolling(720, min_periods=672).quantile(0.60)
    )
    vol["prior_dvol_q60"] = (
        vol["dvol_close"].where(vol["vol_valid"]).shift(1)
        .rolling(720, min_periods=672).quantile(0.60)
    )
    funding = pd.DataFrame({
        "settlement_time": pd.to_datetime(funding["funding_time"], utc=True, format="mixed"),
        "funding_rate": pd.to_numeric(funding["funding_rate"], errors="coerce"),
    }).sort_values("settlement_time").reset_index(drop=True)
    if funding["settlement_time"].duplicated().any():
        raise RuntimeError("FPVIR duplicate funding settlement")
    funding["previous_funding_rate"] = funding["funding_rate"].shift(1)
    funding["decision_time"] = funding["settlement_time"]
    joined = funding.merge(vol, on="decision_time", how="inner", validate="many_to_one")
    joined["funding_amplitude_ratio"] = (
        joined["funding_rate"].abs() / joined["previous_funding_rate"].abs()
    )
    joined["signal_valid"] = (
        joined["vol_valid"]
        & np.isfinite(joined[[
            "funding_rate", "previous_funding_rate", "funding_amplitude_ratio",
            "prior_bvol_q60", "prior_dvol_q60",
        ]]).all(axis=1)
        & joined["funding_rate"].ne(0) & joined["previous_funding_rate"].ne(0)
    )
    joined["high_volatility"] = (
        joined["prior_bvol_q60"].notna() & joined["prior_dvol_q60"].notna()
        & joined["bvol_close"].ge(joined["prior_bvol_q60"])
        & joined["dvol_close"].ge(joined["prior_dvol_q60"])
    )
    return joined.sort_values("settlement_time").reset_index(drop=True)


def conditions(frame: pd.DataFrame, control: str) -> pd.Series:
    sign_rotation = np.sign(frame["funding_rate"]).eq(-np.sign(frame["previous_funding_rate"]))
    if control == "no_sign_rotation":
        sign_rotation = pd.Series(True, index=frame.index)
    amplitude = frame["funding_amplitude_ratio"].ge(1.0)
    if control == "no_amplitude_expansion":
        amplitude = pd.Series(True, index=frame.index)
    if control == "no_high_volatility":
        high_volatility = pd.Series(True, index=frame.index)
    elif control == "one_settlement_stale_volatility":
        high_volatility = frame["high_volatility"].shift(1, fill_value=False)
    else:
        high_volatility = frame["high_volatility"]
    stale_valid = (
        frame["signal_valid"].shift(1, fill_value=False)
        if control == "one_settlement_stale_volatility"
        else pd.Series(True, index=frame.index)
    )
    return frame["signal_valid"] & stale_valid & sign_rotation & amplitude & high_volatility


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active = conditions(frame, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[active]:
        settlement = pd.Timestamp(frame.at[index, "settlement_time"])
        entry = settlement + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=6)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((
            name for name, (start, end) in SPLITS.items()
            if entry >= start and exit_time <= end
        ), None)
        if split is None:
            continue
        side = int(np.sign(frame.at[index, "funding_rate"]))
        if control == "direction_flip":
            side *= -1
        next_allowed = exit_time
        rows.append({
            "candidate": "FPVIR-6", "control": control, "split": split,
            "settlement_time": settlement, "decision_time": settlement,
            "feature_available_time": settlement, "entry_time": entry,
            "exit_time": exit_time, "side": side,
            "funding_rate": float(frame.at[index, "funding_rate"]),
            "previous_funding_rate": float(frame.at[index, "previous_funding_rate"]),
            "funding_amplitude_ratio": float(frame.at[index, "funding_amplitude_ratio"]),
            "bvol_close": float(frame.at[index, "bvol_close"]),
            "dvol_close": float(frame.at[index, "dvol_close"]),
            "prior_bvol_q60": float(frame.at[index, "prior_bvol_q60"]),
            "prior_dvol_q60": float(frame.at[index, "prior_dvol_q60"]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def split_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock["split"].eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts,
            "minority_side_share": min(longs, shorts) / len(subset),
            "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    frame = features()
    primary = build_clock(frame)
    controls = {name: build_clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, control in controls.items():
        _write_gzip_csv(control, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: split_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, stats in support.items():
        checks[f"{name}_minimum_events"] = stats["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = stats["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = stats["max_month_share"] <= 0.45
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    source_manifest = SOURCE_DIR / "manifest.json"
    passed = all(checks.values())
    core = {
        "protocol_version": "fpvir_6_source_support_v1", "policy_id": "FPVIR-6",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha256(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(source_manifest), "sha256": sha256(source_manifest)},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(control), "promotion_authorized": False} for name, control in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
