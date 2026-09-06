"""Build source-support clocks for HVOPCR-6 without post-entry outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import build_options_oi_chase_exhaustion_support as pricebase
from training import preregister_high_volatility_oi_purge_continuation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


SOURCE_DIR = Path("data/options_crowding_deleveraging_relay_sources_v4_2023_2026")
PRICE_DIR = Path("data/options_oi_chase_exhaustion_sources_2023_2026")
CLOCK = Path("data/high_volatility_oi_purge_continuation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_oi_purge_continuation_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_oi_purge_continuation_relay_support_2026-08-08.json")
SPLITS = base.SPLITS
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate",
    "no_oi_purge",
    "no_price_shock",
    "one_hour_stale_regime",
    "direction_flip",
)
ECONOMIC_OUTCOMES_AUTHORIZED = False
COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "bvol_close",
    "prior_bvol_q60",
    "dvol_close",
    "prior_dvol_q60",
    "oi_current_time",
    "oi_prior_time",
    "oi_change",
    "prior_abs_oi_change_q75",
    "hour_return",
    "prior_abs_return_q60",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def features() -> pd.DataFrame:
    bvol, dvol, oi, funding = base.load_sources(SOURCE_DIR)
    joined = base.joined_features(bvol, dvol, oi, funding)
    price = pd.read_csv(PRICE_DIR / "btc_completed_hour.csv.gz", compression="gzip")
    price["decision_time"] = pd.to_datetime(price.decision_time, utc=True, format="mixed")
    price["open"] = pd.to_numeric(price.open, errors="coerce")
    price["close"] = pd.to_numeric(price.close, errors="coerce")
    price["price_valid"] = price.source_valid.astype(str).str.lower().eq("true")
    price["hour_return"] = price.close / price.open - 1.0
    joined = (
        joined.merge(
            price[["decision_time", "price_valid", "hour_return"]],
            on="decision_time",
            validate="one_to_one",
        )
        .sort_values("decision_time")
        .reset_index(drop=True)
    )
    vol_valid = (
        joined.bvol_valid
        & np.isfinite(joined[["bvol_close", "dvol_close"]]).all(axis=1)
        & joined[["bvol_close", "dvol_close"]].gt(0).all(axis=1)
    )
    for name in ("bvol", "dvol"):
        joined[f"prior_{name}_q60"] = (
            joined[f"{name}_close"]
            .where(vol_valid)
            .shift(1)
            .rolling(720, min_periods=672)
            .quantile(0.60)
        )
    current_age = joined.decision_time - joined.oi_current_time
    prior_target = joined.decision_time - pd.Timedelta(hours=1)
    prior_age = prior_target - joined.oi_prior_time
    oi_valid = (
        np.isfinite(joined[["oi_current", "oi_prior", "oi_change"]]).all(axis=1)
        & joined[["oi_current", "oi_prior"]].gt(0).all(axis=1)
        & current_age.between(pd.Timedelta(0), pd.Timedelta(minutes=5))
        & prior_age.between(pd.Timedelta(0), pd.Timedelta(minutes=5))
    )
    joined["prior_abs_oi_change_q75"] = (
        joined.oi_change.abs()
        .where(oi_valid)
        .shift(1)
        .rolling(720, min_periods=672)
        .quantile(0.75)
    )
    price_valid = joined.price_valid & np.isfinite(joined.hour_return) & joined.hour_return.ne(0)
    joined["prior_abs_return_q60"] = (
        joined.hour_return.abs()
        .where(price_valid)
        .shift(1)
        .rolling(720, min_periods=672)
        .quantile(0.60)
    )
    joined["source_valid"] = vol_valid & oi_valid & price_valid
    return joined


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    regime = frame.shift(1) if control == "one_hour_stale_regime" else frame
    volatility = (
        pd.Series(True, index=frame.index)
        if control == "no_volatility_gate"
        else regime.bvol_close.ge(regime.prior_bvol_q60)
        & regime.dvol_close.ge(regime.prior_dvol_q60)
    )
    purge = frame.oi_change.lt(0)
    if control != "no_oi_purge":
        purge &= frame.prior_abs_oi_change_q75.notna() & frame.oi_change.abs().ge(
            frame.prior_abs_oi_change_q75
        )
    shock = frame.hour_return.ne(0)
    if control != "no_price_shock":
        shock &= frame.prior_abs_return_q60.notna() & frame.hour_return.abs().ge(
            frame.prior_abs_return_q60
        )
    previous = frame.shift(1)
    valid = (
        frame.source_valid
        & previous.source_valid
        & frame.decision_time.diff().eq(pd.Timedelta(hours=1))
        & regime.prior_bvol_q60.notna()
        & regime.prior_dvol_q60.notna()
    )
    return valid & volatility & purge & shock, np.sign(frame.hour_return)


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(frame, control)
    onset = active & ~active.shift(1, fill_value=False)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[onset]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=6)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        signal_side = int(side.at[index])
        if control == "direction_flip":
            signal_side = -signal_side
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "HVOPCR-6",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": signal_side,
                "bvol_close": float(frame.at[index, "bvol_close"]),
                "prior_bvol_q60": float(frame.at[index, "prior_bvol_q60"]),
                "dvol_close": float(frame.at[index, "dvol_close"]),
                "prior_dvol_q60": float(frame.at[index, "prior_dvol_q60"]),
                "oi_current_time": frame.at[index, "oi_current_time"],
                "oi_prior_time": frame.at[index, "oi_prior_time"],
                "oi_change": float(frame.at[index, "oi_change"]),
                "prior_abs_oi_change_q75": float(frame.at[index, "prior_abs_oi_change_q75"]),
                "hour_return": float(frame.at[index, "hour_return"]),
                "prior_abs_return_q60": float(frame.at[index, "prior_abs_return_q60"]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def split_stats(clocks: pd.DataFrame, split: str) -> dict[str, int | float]:
    selected = clocks[clocks.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    monthly = selected.entry_time.dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(monthly.max()) / len(selected),
    }


def run() -> dict[str, Any]:
    frame = features()
    primary = clock(frame)
    controls = {name: clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, control_clocks in controls.items():
        _write_gzip_csv(control_clocks, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: split_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, stats in support.items():
        checks[f"{name}_minimum_events"] = stats["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = stats["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = stats["max_month_share"] <= 0.45
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    source_manifest = SOURCE_DIR / "manifest.json"
    price_manifest = PRICE_DIR / "manifest.json"
    passed = all(checks.values())
    core = {
        "protocol_version": "hvopcr_6_source_support_v1",
        "policy_id": "HVOPCR-6",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifests": {
            "volatility_oi": {"path": str(source_manifest), "sha256": sha(source_manifest)},
            "completed_price": {"path": str(price_manifest), "sha256": sha(price_manifest)},
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(control_clocks),
                "promotion_authorized": False,
            }
            for name, control_clocks in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": ECONOMIC_OUTCOMES_AUTHORIZED,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    output = run()
    print(json.dumps({"passed": output["support_passed"], "support": output["support"]}, indent=2))
