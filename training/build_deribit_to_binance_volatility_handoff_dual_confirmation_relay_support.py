"""Build outcome-blind source-support clocks for DBVHDR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_deribit_to_binance_volatility_handoff_dual_confirmation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


VOL_DIR = Path("data/options_crowding_deleveraging_relay_sources_v4_2023_2026")
PRICE_DIR = Path("data/options_oi_chase_exhaustion_sources_2023_2026")
CLOCK = Path("data/deribit_to_binance_volatility_handoff_dual_confirmation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/deribit_to_binance_volatility_handoff_dual_confirmation_relay_controls_2023_2026")
RESULT = Path("results/deribit_to_binance_volatility_handoff_dual_confirmation_relay_support_2026-08-08.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_ordered_handoff", "leader_hour_only", "handoff_hour_only", "one_hour_stale_handoff", "direction_flip")
ECONOMIC_OUTCOMES_AUTHORIZED = False
COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side",
    "prior_bvol_body", "prior_dvol_body", "bvol_body", "dvol_body", "prior_hour_return", "hour_return",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def features() -> pd.DataFrame:
    bvol = pd.read_csv(VOL_DIR / "bvol_hourly.csv.gz", compression="gzip")
    dvol = pd.read_csv(VOL_DIR / "dvol_hourly.csv.gz", compression="gzip")
    price = pd.read_csv(PRICE_DIR / "btc_completed_hour.csv.gz", compression="gzip")
    b = pd.DataFrame({
        "decision_time": pd.to_datetime(bvol["feature_available_time_utc"], utc=True, format="mixed"),
        "bvol_open": pd.to_numeric(bvol["open"], errors="coerce"),
        "bvol_close": pd.to_numeric(bvol["close"], errors="coerce"),
        "bvol_valid": bvol["feature_valid"].astype(str).str.lower().eq("true"),
    })
    d = pd.DataFrame({
        "decision_time": pd.to_datetime(dvol["close_time"], utc=True, format="mixed"),
        "dvol_open": pd.to_numeric(dvol["open"], errors="coerce"),
        "dvol_close": pd.to_numeric(dvol["close"], errors="coerce"),
    })
    p = pd.DataFrame({
        "decision_time": pd.to_datetime(price["decision_time"], utc=True, format="mixed"),
        "price_open": pd.to_numeric(price["open"], errors="coerce"),
        "price_close": pd.to_numeric(price["close"], errors="coerce"),
        "price_valid": price["source_valid"].astype(str).str.lower().eq("true"),
    })
    joined = b.merge(d, on="decision_time", validate="one_to_one").merge(p, on="decision_time", validate="one_to_one")
    joined["bvol_body"] = joined["bvol_close"] / joined["bvol_open"] - 1.0
    joined["dvol_body"] = joined["dvol_close"] / joined["dvol_open"] - 1.0
    joined["hour_return"] = joined["price_close"] / joined["price_open"] - 1.0
    numeric = ["bvol_open", "bvol_close", "dvol_open", "dvol_close", "price_open", "price_close", "bvol_body", "dvol_body", "hour_return"]
    joined["base_valid"] = (
        joined["bvol_valid"] & joined["price_valid"] & np.isfinite(joined[numeric]).all(axis=1)
        & joined[["bvol_open", "bvol_close", "dvol_open", "dvol_close", "price_open", "price_close"]].gt(0).all(axis=1)
        & joined["hour_return"].ne(0)
    )
    return joined.sort_values("decision_time").reset_index(drop=True)


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    current_valid = frame["base_valid"]
    prior_valid = current_valid.shift(1, fill_value=False)
    consecutive = frame["decision_time"].diff().eq(pd.Timedelta(hours=1))
    b, d, ret = frame["bvol_body"], frame["dvol_body"], frame["hour_return"]
    prior_b, prior_d, prior_ret = b.shift(1), d.shift(1), ret.shift(1)
    same_price_direction = ret.ne(0) & prior_ret.ne(0) & np.sign(ret).eq(np.sign(prior_ret))
    leader = prior_d.gt(0) & prior_b.lt(0)
    handoff = d.gt(0) & b.gt(0)
    if control == "no_ordered_handoff":
        active = current_valid & prior_valid & consecutive & handoff & same_price_direction
    elif control == "leader_hour_only":
        active = current_valid & prior_valid & consecutive & leader
    elif control == "handoff_hour_only":
        active = current_valid & handoff
    else:
        active = current_valid & prior_valid & consecutive & leader & handoff & same_price_direction
    if control == "one_hour_stale_handoff":
        active = active.shift(1, fill_value=False) & current_valid & consecutive
    onset = active & ~active.shift(1, fill_value=False)
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
        source_index = index - 1 if control == "one_hour_stale_handoff" else index
        if control == "leader_hour_only":
            side = int(np.sign(prior_ret.at[index]))
        else:
            side = int(np.sign(ret.at[source_index]))
        if side == 0:
            continue
        if control == "direction_flip":
            side *= -1
        next_allowed = exit_time
        rows.append({
            "candidate": "DBVHDR-6", "control": control, "split": split, "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time, "side": side,
            "prior_bvol_body": float(prior_b.at[source_index]), "prior_dvol_body": float(prior_d.at[source_index]),
            "bvol_body": float(b.at[source_index]), "dvol_body": float(d.at[source_index]),
            "prior_hour_return": float(prior_ret.at[source_index]), "hour_return": float(ret.at[source_index]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def split_stats(clock_frame: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = clock_frame[clock_frame["split"].eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected["side"].eq(1).sum())
    shorts = int(selected["side"].eq(-1).sum())
    months = selected["entry_time"].dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(selected), "max_month_share": int(months.max()) / len(selected)}


def run() -> dict[str, Any]:
    frame = features()
    primary = clock(frame)
    controls = {name: clock(frame, name) for name in CONTROLS}
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
    passed = all(checks.values())
    core = {
        "protocol_version": "dbvhdr_6_source_support_v1", "policy_id": "DBVHDR-6",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha256(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]},
        "source_manifests": {
            "volatility": {"path": str(VOL_DIR / "manifest.json"), "sha256": sha256(VOL_DIR / "manifest.json")},
            "completed_hour_price": {"path": str(PRICE_DIR / "manifest.json"), "sha256": sha256(PRICE_DIR / "manifest.json")},
        },
        "completed_preentry_feature_price_opened": True, "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(control)} for name, control in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": ECONOMIC_OUTCOMES_AUTHORIZED,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    payload = run()
    print(json.dumps({"passed": payload["support_passed"], "support": payload["support"]}, indent=2))
