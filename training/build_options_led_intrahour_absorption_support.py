"""Build source-support clocks for OLIAH-6 without post-entry outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import preregister_options_led_intrahour_absorption_handoff as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


NONPRICE_DIR = Path("data/options_crowding_deleveraging_relay_sources_v4_2023_2026")
PRICE_DIR = Path("data/options_led_intrahour_absorption_sources_2023_2026")
CLOCK = Path("data/options_led_intrahour_absorption_handoff_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/options_led_intrahour_absorption_handoff_controls_2023_2026")
RESULT = Path("results/options_led_intrahour_absorption_handoff_support_2026-08-08.json")
SPLITS = base.SPLITS
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_deribit_lead", "no_oi_contraction", "no_first_half_tail", "no_open_reclaim", "direction_flip")
ECONOMIC_OUTCOMES_AUTHORIZED = False
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "bvol_body", "dvol_body", "oi_change",
    "first_half_return", "second_half_return", "hour_return",
    "prior_abs_first_half_q75", "absorption_ratio",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def features() -> pd.DataFrame:
    bvol, dvol, oi, funding = base.load_sources(NONPRICE_DIR)
    joined = base.joined_features(bvol, dvol, oi, funding)
    relevant = ["bvol_open", "bvol_close", "dvol_open", "dvol_close", "oi_current", "oi_prior", "oi_change"]
    joined["source_valid"] = (
        joined["bvol_valid"]
        & np.isfinite(joined[relevant]).all(axis=1)
        & joined[["bvol_open", "bvol_close", "dvol_open", "dvol_close", "oi_current", "oi_prior"]].gt(0).all(axis=1)
    )
    price = pd.read_csv(PRICE_DIR / "btc_intrahour_path.csv.gz", compression="gzip")
    price["decision_time"] = pd.to_datetime(price["decision_time"], utc=True, format="mixed")
    price_columns = ["hour_open", "first_half_close", "second_half_open", "hour_close"]
    for column in price_columns:
        price[column] = pd.to_numeric(price[column], errors="coerce")
    price["price_valid"] = price["source_valid"].astype(str).str.lower().eq("true")
    price["first_half_return"] = price["first_half_close"] / price["hour_open"] - 1.0
    price["second_half_return"] = price["hour_close"] / price["second_half_open"] - 1.0
    price["hour_return"] = price["hour_close"] / price["hour_open"] - 1.0
    price["first_half_tail"] = (
        price["first_half_return"].abs().where(price["price_valid"])
        .shift(1).rolling(720, min_periods=672).quantile(0.75)
    )
    joined = joined.merge(
        price[["decision_time", "price_valid", "first_half_return", "second_half_return", "hour_return", "first_half_tail"]],
        on="decision_time", validate="one_to_one",
    )
    joined["base_valid"] = (
        joined["source_valid"] & joined["price_valid"]
        & np.isfinite(joined[["first_half_return", "second_half_return", "hour_return"]]).all(axis=1)
    )
    return joined


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    bvol = frame["bvol_body"]
    dvol = frame["dvol_body"]
    volatility = bvol.gt(0) & dvol.gt(0) if control == "no_deribit_lead" else bvol.gt(0) & dvol.gt(bvol)
    oi_gate = pd.Series(True, index=frame.index) if control == "no_oi_contraction" else frame["oi_change"].lt(0)
    shock = frame["first_half_return"].ne(0)
    if control != "no_first_half_tail":
        shock &= frame["first_half_tail"].notna() & frame["first_half_return"].abs().ge(frame["first_half_tail"])
    opposite = frame["second_half_return"].ne(0) & np.sign(frame["first_half_return"]).eq(-np.sign(frame["second_half_return"]))
    ratio = frame["second_half_return"].abs() / frame["first_half_return"].abs()
    absorption = opposite & ratio.ge(0.5)
    if control != "no_open_reclaim":
        absorption &= frame["hour_return"].ne(0) & np.sign(frame["hour_return"]).eq(np.sign(frame["second_half_return"]))
    active = frame["base_valid"] & volatility & oi_gate & shock & absorption
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
            "candidate": "OLIAH-6", "control": control, "split": split,
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": side,
            "bvol_body": float(bvol.at[index]), "dvol_body": float(dvol.at[index]),
            "oi_change": float(frame.at[index, "oi_change"]),
            "first_half_return": float(frame.at[index, "first_half_return"]),
            "second_half_return": float(frame.at[index, "second_half_return"]),
            "hour_return": float(frame.at[index, "hour_return"]),
            "prior_abs_first_half_q75": float(frame.at[index, "first_half_tail"]),
            "absorption_ratio": float(ratio.at[index]),
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
    source_manifest = PRICE_DIR / "manifest.json"
    passed = all(checks.values())
    core = {
        "protocol_version": "oliah_6_source_support_v1", "policy_id": "OLIAH-6",
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
