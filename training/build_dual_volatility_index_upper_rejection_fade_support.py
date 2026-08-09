"""Build source-only DVURF-6 clocks before Gross9 or outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_dual_volatility_index_upper_rejection_fade as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

PREREG_SHA = "79da6f3d3b37826b84a8b3fdd2b51fe00d0ae45f022e0f762a6c2d7bf89f3b77"
VOL_DIR = Path("data/options_crowding_deleveraging_relay_sources_v4_2023_2026")
BTC_DIR = Path("data/options_oi_chase_exhaustion_sources_2023_2026")
SNAPSHOT = Path("data/dual_volatility_index_upper_rejection_fade_sources_2023_2026/paired_hourly_features.csv.gz")
CLOCK = Path("data/dual_volatility_index_upper_rejection_fade_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/dual_volatility_index_upper_rejection_fade_controls_2023_2026")
RESULT = Path("results/dual_volatility_index_upper_rejection_fade_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("btc_shock_only", "bvol_rejection_only", "dvol_rejection_only", "no_joint_range_floor", "one_hour_stale_index_geometry", "direction_flip", "forced_long")
CLOCK_COLUMNS = ("candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "btc_hour_return", "bvol_net_upper_rejection", "dvol_net_upper_rejection", "joint_rejection", "joint_range", "joint_rejection_rank", "joint_range_rank", "btc_absolute_return_rank")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 720, minimum: int = 672) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)) / len(array)
        if np.isfinite(current):
            history.append(float(current))
    return output


def _vol_frame(path: Path, prefix: str, time_column: str, validity: pd.Series | None = None) -> pd.DataFrame:
    raw = pd.read_csv(path, compression="gzip")
    frame = pd.DataFrame({"decision_time": pd.to_datetime(raw[time_column], utc=True, format="mixed")})
    for column in ("open", "high", "low", "close"):
        frame[f"{prefix}_{column}"] = pd.to_numeric(raw[column], errors="coerce")
    frame[f"{prefix}_valid"] = validity if validity is not None else True
    return frame


def features() -> pd.DataFrame:
    braw = pd.read_csv(VOL_DIR / "bvol_hourly.csv.gz", compression="gzip")
    bvalid = braw["feature_valid"].astype(str).str.lower().eq("true")
    bvol = pd.DataFrame({"decision_time": pd.to_datetime(braw["feature_available_time_utc"], utc=True, format="mixed"), "bvol_valid": bvalid})
    for column in ("open", "high", "low", "close"):
        bvol[f"bvol_{column}"] = pd.to_numeric(braw[column], errors="coerce")
    dvol = _vol_frame(VOL_DIR / "dvol_hourly.csv.gz", "dvol", "close_time")
    btcraw = pd.read_csv(BTC_DIR / "btc_completed_hour.csv.gz", compression="gzip")
    btc = pd.DataFrame({
        "decision_time": pd.to_datetime(btcraw["decision_time"], utc=True, format="mixed"),
        "btc_valid": btcraw["source_valid"].astype(str).str.lower().eq("true"),
        "btc_open": pd.to_numeric(btcraw["open"], errors="coerce"),
        "btc_close": pd.to_numeric(btcraw["close"], errors="coerce"),
    })
    frame = bvol.merge(dvol, on="decision_time", validate="one_to_one").merge(btc, on="decision_time", validate="one_to_one").sort_values("decision_time").reset_index(drop=True)
    numeric = [f"{prefix}_{column}" for prefix in ("bvol", "dvol") for column in ("open", "high", "low", "close")]
    coherent = pd.Series(True, index=frame.index)
    for prefix in ("bvol", "dvol"):
        coherent &= frame[f"{prefix}_high"].ge(frame[[f"{prefix}_open", f"{prefix}_close", f"{prefix}_low"]].max(axis=1))
        coherent &= frame[f"{prefix}_low"].le(frame[[f"{prefix}_open", f"{prefix}_close", f"{prefix}_high"]].min(axis=1))
        span = frame[f"{prefix}_high"] - frame[f"{prefix}_low"]
        upper = frame[f"{prefix}_high"] - frame[[f"{prefix}_open", f"{prefix}_close"]].max(axis=1)
        lower = frame[[f"{prefix}_open", f"{prefix}_close"]].min(axis=1) - frame[f"{prefix}_low"]
        frame[f"{prefix}_net_upper_rejection"] = (upper - lower) / span
        frame[f"{prefix}_normalized_range"] = span / frame[f"{prefix}_open"]
    frame["source_valid"] = frame["bvol_valid"] & frame["dvol_valid"] & frame["btc_valid"] & np.isfinite(frame[numeric + ["btc_open", "btc_close"]]).all(axis=1) & frame[numeric + ["btc_open", "btc_close"]].gt(0).all(axis=1) & coherent & frame["bvol_high"].gt(frame["bvol_low"]) & frame["dvol_high"].gt(frame["dvol_low"])
    frame["btc_hour_return"] = np.log(frame["btc_close"] / frame["btc_open"])
    frame["joint_rejection"] = frame[["bvol_net_upper_rejection", "dvol_net_upper_rejection"]].min(axis=1)
    frame["joint_range"] = np.sqrt(frame["bvol_normalized_range"] * frame["dvol_normalized_range"])
    valid = frame["source_valid"]
    for column in ("joint_rejection", "joint_range"):
        frame[f"{column}_rank"] = strict_prior_midrank(frame[column].where(valid))
    frame["bvol_rejection_rank"] = strict_prior_midrank(frame["bvol_net_upper_rejection"].where(valid))
    frame["dvol_rejection_rank"] = strict_prior_midrank(frame["dvol_net_upper_rejection"].where(valid))
    frame["btc_absolute_return_rank"] = strict_prior_midrank(frame["btc_hour_return"].abs().where(valid))
    return frame


def conditions(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series]:
    work = frame.copy()
    if control == "one_hour_stale_index_geometry":
        columns = ["bvol_net_upper_rejection", "dvol_net_upper_rejection", "joint_rejection", "joint_range", "joint_rejection_rank", "joint_range_rank", "bvol_rejection_rank", "dvol_rejection_rank"]
        work[columns] = work[columns].shift(1)
    shock = work["btc_hour_return"].ne(0) & work["btc_absolute_return_rank"].ge(0.75)
    if control == "btc_shock_only":
        rejection = pd.Series(True, index=work.index)
    elif control == "bvol_rejection_only":
        rejection = work["bvol_net_upper_rejection"].gt(0) & work["bvol_rejection_rank"].ge(0.75)
    elif control == "dvol_rejection_only":
        rejection = work["dvol_net_upper_rejection"].gt(0) & work["dvol_rejection_rank"].ge(0.75)
    else:
        rejection = work["joint_rejection"].gt(0) & work["joint_rejection_rank"].ge(0.75)
    range_gate = pd.Series(True, index=work.index) if control in ("btc_shock_only", "no_joint_range_floor") else work["joint_range_rank"].ge(0.50)
    active = work["source_valid"] & shock & rejection & range_gate
    side = -np.sign(work["btc_hour_return"])
    if control == "direction_flip":
        side = -side
    if control == "forced_long":
        side = pd.Series(1.0, index=work.index)
    return active, side


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(frame, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry, exit_time = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=6, minutes=5)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({"candidate": "DVURF-6", "control": control, "split": split, "decision_time": decision, "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "btc_hour_return": float(frame.at[index, "btc_hour_return"]), "bvol_net_upper_rejection": float(frame.at[index, "bvol_net_upper_rejection"]), "dvol_net_upper_rejection": float(frame.at[index, "dvol_net_upper_rejection"]), "joint_rejection": float(frame.at[index, "joint_rejection"]), "joint_range": float(frame.at[index, "joint_range"]), "joint_rejection_rank": float(frame.at[index, "joint_rejection_rank"]), "joint_range_rank": float(frame.at[index, "joint_range_rank"]), "btc_absolute_return_rank": float(frame.at[index, "btc_absolute_return_rank"])})
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    rows = clock[clock.split.eq(split)]
    if rows.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(rows.side.eq(1).sum()), int(rows.side.eq(-1).sum())
    months = pd.to_datetime(rows.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(rows), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(rows), "max_month_share": int(months.max()) / len(rows)}


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("DVURF preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    frame = features()
    primary = build_clock(frame)
    controls = {name: build_clock(frame, name) for name in CONTROLS}
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(frame, SNAPSHOT); _write_gzip_csv(primary, CLOCK)
    for name, rows in controls.items(): _write_gzip_csv(rows, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {check: passed for name, item in support.items() for check, passed in ((f"{name}_minimum_events", item["events"] >= MINIMUM[name]), (f"{name}_side_balance", item["minority_side_share"] >= 0.20), (f"{name}_month_concentration", item["max_month_share"] <= 0.45))}
    passed = all(checks.values())
    core = {"protocol_version": "dvurf_6_source_support_v1", "policy_id": "DVURF-6", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source_manifests": {"volatility": {"path": str(VOL_DIR / "manifest.json"), "sha256": sha256(VOL_DIR / "manifest.json")}, "btc_completed_hour": {"path": str(BTC_DIR / "manifest.json"), "sha256": sha256(BTC_DIR / "manifest.json")}}, "source_snapshot": {"path": str(SNAPSHOT), "sha256": sha256(SNAPSHOT), "rows": len(frame)}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(rows), "promotion_authorized": False} for name, rows in controls.items()}, "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    report = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
