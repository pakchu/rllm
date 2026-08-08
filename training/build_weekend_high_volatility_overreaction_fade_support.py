"""Build outcome-blind source-support clocks for WHVOF-6."""
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
from training import preregister_weekend_high_volatility_overreaction_fade as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


CLOCK = Path("data/weekend_high_volatility_overreaction_fade_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/weekend_high_volatility_overreaction_fade_controls_2023_2026")
RESULT = Path("results/weekend_high_volatility_overreaction_fade_support_2026-08-08.json")
SPLITS = base.SPLITS
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_weekend_gate", "no_high_volatility", "no_return_tail",
    "one_block_stale_volatility", "direction_flip",
)
COLUMNS = (
    "candidate", "control", "split", "block_start_time", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side",
    "block_return", "prior_abs_block_q60", "bvol_close", "prior_bvol_q60",
    "dvol_close", "prior_dvol_q60",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def features() -> pd.DataFrame:
    volatility = intrahour.features().drop(columns=["price_valid"], errors="ignore")
    price = pd.read_csv(intrahour.PRICE_DIR / "btc_intrahour_path.csv.gz", compression="gzip")
    price["decision_time"] = pd.to_datetime(price["decision_time"], utc=True, format="mixed")
    price["hour_open"] = pd.to_numeric(price["hour_open"], errors="coerce")
    price["hour_close"] = pd.to_numeric(price["hour_close"], errors="coerce")
    price["price_valid"] = price["source_valid"].astype(str).str.lower().eq("true")
    frame = volatility.merge(
        price[["decision_time", "hour_open", "hour_close", "price_valid"]],
        on="decision_time", validate="one_to_one",
    ).sort_values("decision_time").reset_index(drop=True)
    vol_valid = (
        frame["bvol_valid"]
        & np.isfinite(frame[["bvol_close", "dvol_close"]]).all(axis=1)
        & frame[["bvol_close", "dvol_close"]].gt(0).all(axis=1)
    )
    for name in ("bvol", "dvol"):
        frame[f"prior_{name}_q60"] = (
            frame[f"{name}_close"].where(vol_valid).shift(1)
            .rolling(720, min_periods=672).quantile(0.60)
        )
    price_valid = (
        frame["price_valid"]
        & np.isfinite(frame[["hour_open", "hour_close"]]).all(axis=1)
        & frame[["hour_open", "hour_close"]].gt(0).all(axis=1)
    )
    consecutive = frame["decision_time"].diff().eq(pd.Timedelta(hours=1))
    frame["twelve_valid"] = (
        price_valid.rolling(12, min_periods=12).sum().eq(12)
        & consecutive.rolling(11, min_periods=11).sum().eq(11)
    )
    frame["block_return"] = frame["hour_close"] / frame["hour_open"].shift(11) - 1.0
    frame["vol_valid"] = vol_valid
    blocks = frame[frame["decision_time"].dt.hour.isin([0, 12])].copy().reset_index(drop=True)
    blocks["prior_abs_block_q60"] = (
        blocks["block_return"].abs().where(blocks["twelve_valid"])
        .shift(1).rolling(540, min_periods=504).quantile(0.60)
    )
    blocks["weekend"] = blocks["decision_time"].dt.dayofweek.isin([5, 6])
    blocks["block_valid"] = (
        blocks["twelve_valid"] & blocks["vol_valid"]
        & np.isfinite(blocks[["block_return", "bvol_close", "dvol_close"]]).all(axis=1)
        & blocks["block_return"].ne(0)
    )
    return blocks


def conditions(frame: pd.DataFrame, control: str) -> pd.Series:
    weekend = pd.Series(True, index=frame.index) if control == "no_weekend_gate" else frame["weekend"]
    if control == "no_high_volatility":
        high_volatility = pd.Series(True, index=frame.index)
    elif control == "one_block_stale_volatility":
        high_volatility = (
            frame["bvol_close"].shift(1).ge(frame["prior_bvol_q60"].shift(1))
            & frame["dvol_close"].shift(1).ge(frame["prior_dvol_q60"].shift(1))
        )
    else:
        high_volatility = (
            frame["bvol_close"].ge(frame["prior_bvol_q60"])
            & frame["dvol_close"].ge(frame["prior_dvol_q60"])
        )
    tail = frame["prior_abs_block_q60"].notna()
    if control != "no_return_tail":
        tail &= frame["block_return"].abs().ge(frame["prior_abs_block_q60"])
    stale_valid = (
        frame["vol_valid"].shift(1, fill_value=False)
        if control == "one_block_stale_volatility"
        else pd.Series(True, index=frame.index)
    )
    return frame["block_valid"] & stale_valid & weekend & high_volatility & tail


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active = conditions(frame, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=6)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        side = -int(np.sign(frame.at[index, "block_return"]))
        if control == "direction_flip":
            side *= -1
        next_allowed = exit_time
        rows.append({
            "candidate": "WHVOF-6", "control": control, "split": split,
            "block_start_time": decision - pd.Timedelta(hours=12),
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": side,
            "block_return": float(frame.at[index, "block_return"]),
            "prior_abs_block_q60": float(frame.at[index, "prior_abs_block_q60"]),
            "bvol_close": float(frame.at[index, "bvol_close"]),
            "prior_bvol_q60": float(frame.at[index, "prior_bvol_q60"]),
            "dvol_close": float(frame.at[index, "dvol_close"]),
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
    vol_manifest = intrahour.NONPRICE_DIR / "manifest.json"
    price_manifest = intrahour.PRICE_DIR / "manifest.json"
    passed = all(checks.values())
    core = {
        "protocol_version": "whvof_6_source_support_v1", "policy_id": "WHVOF-6",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha256(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]},
        "source_manifests": {"volatility": {"path": str(vol_manifest), "sha256": sha256(vol_manifest)}, "completed_price": {"path": str(price_manifest), "sha256": sha256(price_manifest)}},
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
