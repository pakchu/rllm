"""Build source-support clocks for OIFAR-6 without post-entry outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import preregister_options_implied_vol_flush_absorption_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


NONPRICE_DIR = Path("data/options_crowding_deleveraging_relay_sources_v4_2023_2026")
PRICE_DIR = Path("data/options_oi_chase_exhaustion_sources_2023_2026")
PREREGISTRATION = prereg.DEFAULT_OUTPUT
CLOCK = Path("data/options_implied_vol_flush_absorption_reversal_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/options_implied_vol_flush_absorption_reversal_controls_2023_2026")
RESULT = Path("results/options_implied_vol_flush_absorption_reversal_support_2026-08-08.json")
SPLITS = base.SPLITS
MINIMUM_EVENTS = {"train": 12, "test": 18, "eval": 18, "final": 12}
CONTROLS = ("no_deribit_lead", "no_oi_flush", "no_return_shock", "direction_flip")
ECONOMIC_OUTCOMES_AUTHORIZED = False
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "bvol_body", "dvol_body", "oi_change",
    "prior_oi_q25", "hour_return", "prior_abs_return_q75",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def features() -> pd.DataFrame:
    bvol, dvol, oi, funding = base.load_sources(NONPRICE_DIR)
    joined = base.joined_features(bvol, dvol, oi, funding)
    relevant = [
        "bvol_open", "bvol_close", "dvol_open", "dvol_close", "oi_current",
        "oi_prior", "oi_change",
    ]
    joined["source_valid"] = (
        joined["bvol_valid"]
        & np.isfinite(joined[relevant]).all(axis=1)
        & joined[["bvol_open", "bvol_close", "dvol_open", "dvol_close", "oi_current", "oi_prior"]]
        .gt(0)
        .all(axis=1)
    )
    joined["oi_floor"] = (
        joined["oi_change"]
        .where(joined["source_valid"])
        .shift(1)
        .rolling(720, min_periods=672)
        .quantile(0.25)
    )

    price = pd.read_csv(PRICE_DIR / "btc_completed_hour.csv.gz", compression="gzip")
    price["decision_time"] = pd.to_datetime(price["decision_time"], utc=True, format="mixed")
    price["open"] = pd.to_numeric(price["open"], errors="coerce")
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price["price_valid"] = price["source_valid"].astype(str).str.lower().eq("true")
    price["hour_return"] = price["close"] / price["open"] - 1.0
    price["return_tail"] = (
        price["hour_return"]
        .abs()
        .where(price["price_valid"])
        .shift(1)
        .rolling(720, min_periods=672)
        .quantile(0.75)
    )
    joined = joined.merge(
        price[["decision_time", "price_valid", "hour_return", "return_tail"]],
        on="decision_time",
        validate="one_to_one",
    )
    joined["base_valid"] = (
        joined["source_valid"]
        & joined["price_valid"]
        & np.isfinite(joined[["hour_return"]]).all(axis=1)
        & joined["hour_return"].ne(0)
    )
    return joined


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    bvol_body = frame["bvol_body"]
    dvol_body = frame["dvol_body"]
    if control == "no_deribit_lead":
        volatility_gate = bvol_body.gt(0) & dvol_body.gt(0)
    else:
        volatility_gate = bvol_body.gt(0) & dvol_body.gt(bvol_body)

    oi_gate = frame["oi_change"].lt(0)
    if control != "no_oi_flush":
        oi_gate &= frame["oi_floor"].notna() & frame["oi_change"].le(frame["oi_floor"])

    return_gate = frame["hour_return"].ne(0)
    if control != "no_return_shock":
        return_gate &= (
            frame["return_tail"].notna()
            & frame["hour_return"].abs().ge(frame["return_tail"])
        )

    active = frame["base_valid"] & volatility_gate & oi_gate & return_gate
    onset = (
        active
        & ~active.shift(1, fill_value=False)
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
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_time <= end
            ),
            None,
        )
        if split is None:
            continue
        side = -int(np.sign(frame.at[index, "hour_return"]))
        if control == "direction_flip":
            side *= -1
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "OIFAR-6",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": side,
                "bvol_body": float(bvol_body.at[index]),
                "dvol_body": float(dvol_body.at[index]),
                "oi_change": float(frame.at[index, "oi_change"]),
                "prior_oi_q25": (
                    float(frame.at[index, "oi_floor"])
                    if pd.notna(frame.at[index, "oi_floor"])
                    else None
                ),
                "hour_return": float(frame.at[index, "hour_return"]),
                "prior_abs_return_q75": (
                    float(frame.at[index, "return_tail"])
                    if pd.notna(frame.at[index, "return_tail"])
                    else None
                ),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def split_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock["split"].eq(split)]
    if subset.empty:
        return {
            "events": 0, "longs": 0, "shorts": 0,
            "minority_side_share": 0.0, "max_month_share": 0.0,
        }
    longs = int(subset["side"].eq(1).sum())
    shorts = int(subset["side"].eq(-1).sum())
    months = subset["entry_time"].dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


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

    registration = json.loads(PREREGISTRATION.read_text())
    nonprice_manifest = NONPRICE_DIR / "manifest.json"
    price_manifest = PRICE_DIR / "manifest.json"
    passed = all(checks.values())
    core = {
        "protocol_version": "oifar_6_source_support_v1",
        "policy_id": "OIFAR-6",
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": sha256(PREREGISTRATION),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifests": {
            "nonprice": {"path": str(nonprice_manifest), "sha256": sha256(nonprice_manifest)},
            "completed_hour_price": {"path": str(price_manifest), "sha256": sha256(price_manifest)},
        },
        "completed_preentry_feature_price_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(control),
            }
            for name, control in controls.items()
        },
        "support": statistics,
        "support_checks": checks,
        "support_passed": passed,
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
