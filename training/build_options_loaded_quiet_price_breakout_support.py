"""Build source-support clocks for OLQPB-6 without post-entry outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import preregister_options_loaded_quiet_price_breakout as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


NONPRICE_DIR = Path("data/options_crowding_deleveraging_relay_sources_v4_2023_2026")
PRICE_DIR = Path("data/options_oi_chase_exhaustion_sources_2023_2026")
CLOCK = Path("data/options_loaded_quiet_price_breakout_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/options_loaded_quiet_price_breakout_controls_2023_2026")
RESULT = Path("results/options_loaded_quiet_price_breakout_support_2026-08-08.json")
SPLITS = base.SPLITS
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_bvol_level", "no_dvol_level", "no_oi_tail", "no_funding_neutral", "no_quiet_price", "direction_flip")
ECONOMIC_OUTCOMES_AUTHORIZED = False
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "bvol_close", "prior_bvol_q75",
    "dvol_close", "prior_dvol_q75", "oi_change", "prior_oi_q75",
    "hour_return", "prior_abs_return_q40", "funding_rate", "prior_abs_funding_q50",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def features() -> pd.DataFrame:
    bvol, dvol, oi, funding = base.load_sources(NONPRICE_DIR)
    joined = base.joined_features(bvol, dvol, oi, funding)
    joined["bvol_level_tail"] = (
        joined["bvol_close"].where(joined["bvol_valid"])
        .shift(1).rolling(720, min_periods=672).quantile(0.75)
    )
    dvol_valid = np.isfinite(joined["dvol_close"]) & joined["dvol_close"].gt(0)
    joined["dvol_level_tail"] = (
        joined["dvol_close"].where(dvol_valid)
        .shift(1).rolling(720, min_periods=672).quantile(0.75)
    )

    funding = funding.copy()
    funding["funding_time"] = pd.to_datetime(funding["funding_time"], utc=True, format="mixed")
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="coerce")
    funding["funding_neutral_tail"] = (
        funding["funding_rate"].abs().shift(1).rolling(270, min_periods=252).quantile(0.50)
    )
    joined = joined.merge(
        funding[["funding_time", "funding_neutral_tail"]],
        on="funding_time", how="left", validate="many_to_one",
    )

    price = pd.read_csv(PRICE_DIR / "btc_completed_hour.csv.gz", compression="gzip")
    price["decision_time"] = pd.to_datetime(price["decision_time"], utc=True, format="mixed")
    price["open"] = pd.to_numeric(price["open"], errors="coerce")
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price["price_valid"] = price["source_valid"].astype(str).str.lower().eq("true")
    price["hour_return"] = price["close"] / price["open"] - 1.0
    price["quiet_tail"] = (
        price["hour_return"].abs().where(price["price_valid"])
        .shift(1).rolling(720, min_periods=672).quantile(0.40)
    )
    joined = joined.merge(
        price[["decision_time", "price_valid", "hour_return", "quiet_tail"]],
        on="decision_time", validate="one_to_one",
    )
    relevant = [
        "bvol_close", "dvol_close", "oi_current", "oi_prior", "oi_change",
        "funding_rate", "hour_return",
    ]
    joined["base_valid"] = (
        joined["bvol_valid"] & joined["price_valid"]
        & np.isfinite(joined[relevant]).all(axis=1)
        & joined[["bvol_close", "dvol_close", "oi_current", "oi_prior"]].gt(0).all(axis=1)
        & joined["funding_rate"].ne(0) & joined["hour_return"].ne(0)
    )
    return joined


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    bvol_gate = pd.Series(True, index=frame.index) if control == "no_bvol_level" else (
        frame["bvol_level_tail"].notna() & frame["bvol_close"].ge(frame["bvol_level_tail"])
    )
    dvol_gate = pd.Series(True, index=frame.index) if control == "no_dvol_level" else (
        frame["dvol_level_tail"].notna() & frame["dvol_close"].ge(frame["dvol_level_tail"])
    )
    oi_gate = frame["oi_change"].gt(0)
    if control != "no_oi_tail":
        oi_gate &= frame["oi_tail"].notna() & frame["oi_change"].ge(frame["oi_tail"])
    funding_gate = frame["funding_rate"].ne(0)
    if control != "no_funding_neutral":
        funding_gate &= frame["funding_neutral_tail"].notna() & frame["funding_rate"].abs().le(frame["funding_neutral_tail"])
    quiet_gate = frame["hour_return"].ne(0)
    if control != "no_quiet_price":
        quiet_gate &= frame["quiet_tail"].notna() & frame["hour_return"].abs().le(frame["quiet_tail"])
    active = frame["base_valid"] & bvol_gate & dvol_gate & oi_gate & funding_gate & quiet_gate
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
        side = int(np.sign(frame.at[index, "hour_return"]))
        if control == "direction_flip":
            side *= -1
        next_allowed = exit_time
        rows.append({
            "candidate": "OLQPB-6", "control": control, "split": split,
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": side,
            "bvol_close": float(frame.at[index, "bvol_close"]),
            "prior_bvol_q75": float(frame.at[index, "bvol_level_tail"]),
            "dvol_close": float(frame.at[index, "dvol_close"]),
            "prior_dvol_q75": float(frame.at[index, "dvol_level_tail"]),
            "oi_change": float(frame.at[index, "oi_change"]),
            "prior_oi_q75": float(frame.at[index, "oi_tail"]),
            "hour_return": float(frame.at[index, "hour_return"]),
            "prior_abs_return_q40": float(frame.at[index, "quiet_tail"]),
            "funding_rate": float(frame.at[index, "funding_rate"]),
            "prior_abs_funding_q50": float(frame.at[index, "funding_neutral_tail"]),
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
    nonprice_manifest = NONPRICE_DIR / "manifest.json"
    price_manifest = PRICE_DIR / "manifest.json"
    passed = all(checks.values())
    core = {
        "protocol_version": "olqpb_6_source_support_v1", "policy_id": "OLQPB-6",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha256(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]},
        "source_manifests": {
            "nonprice": {"path": str(nonprice_manifest), "sha256": sha256(nonprice_manifest)},
            "completed_hour_price": {"path": str(price_manifest), "sha256": sha256(price_manifest)},
        },
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
