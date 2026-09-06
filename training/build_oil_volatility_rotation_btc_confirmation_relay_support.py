"""Build outcome-blind source support for preregistered OVRCR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from training import preregister_oil_volatility_rotation_btc_confirmation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_cboe_volatility_surface_regime_crossing_relay_support import strict_prior_midrank

CLOCK = Path("data/oil_volatility_rotation_btc_confirmation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/oil_volatility_rotation_btc_confirmation_relay_controls_2023_2026")
RESULT = Path("results/oil_volatility_rotation_btc_confirmation_relay_support_2026-08-08.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("ovx_change_only", "vix_change_only", "no_btc_confirmation", "one_session_stale_rotation", "direction_flip")
NY = ZoneInfo("America/New_York")
ECONOMIC_OUTCOMES_AUTHORIZED = False
COLUMNS = (
    "candidate", "control", "split", "cboe_observation_date", "next_cboe_source_date",
    "overnight_start_time", "decision_time", "feature_available_time", "entry_time", "exit_time",
    "side", "delta_log_relative", "absolute_rotation_rank", "delta_log_ovx", "delta_log_vix", "overnight_btc_return",
)
OVX = Path("data/cboe_ovx_2021_2026/source/OVX_History.csv")
VIX = Path("data/cboe_volatility_surface_2021_2026/cboe_volatility_surface_2021-01-01_2026-08-07.csv.gz")
PRICE = Path("data/options_led_intrahour_absorption_sources_2023_2026/btc_intrahour_path.csv.gz")
HASHES = {
    OVX: "77f872f1e069cc93554fe6d80dc6f9d44d0a798ad0a906202a570ad81f73417a",
    VIX: "42eb1093f5167aec9c71a4733ab3451e40807c81dc7cb49568a6a0c634267ba0",
    PRICE: "7413b1dde7148efbc30eccd077e4f7430d57be87836bbd4551df9f6f85609579",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_features() -> pd.DataFrame:
    for raw, expected in HASHES.items():
        if sha(raw) != expected:
            raise RuntimeError(f"OVRCR frozen source changed: {raw}")
    vix = pd.read_csv(VIX, compression="gzip", usecols=["observation_date", "VIX_close"])
    ovx = pd.read_csv(OVX, usecols=["DATE", "OVX"]).rename(columns={"DATE": "observation_date", "OVX": "OVX_close"})
    vix["observation_date"] = pd.to_datetime(vix.observation_date, format="%Y-%m-%d")
    ovx["observation_date"] = pd.to_datetime(ovx.observation_date, format="%m/%d/%Y")
    cboe = vix.merge(ovx, on="observation_date", validate="one_to_one").sort_values("observation_date").reset_index(drop=True)
    cboe["VIX_close"] = pd.to_numeric(cboe.VIX_close, errors="coerce")
    cboe["OVX_close"] = pd.to_numeric(cboe.OVX_close, errors="coerce")
    if not cboe.observation_date.is_monotonic_increasing or cboe.observation_date.duplicated().any():
        raise RuntimeError("OVRCR Cboe dates invalid")
    if not np.isfinite(cboe[["VIX_close", "OVX_close"]]).all(axis=None) or not cboe[["VIX_close", "OVX_close"]].gt(0).all(axis=None):
        raise RuntimeError("OVRCR volatility values invalid")
    cboe["delta_log_vix"] = np.log(cboe.VIX_close).diff()
    cboe["delta_log_ovx"] = np.log(cboe.OVX_close).diff()
    cboe["delta_log_relative"] = np.log(cboe.OVX_close / cboe.VIX_close).diff()
    cboe["absolute_rotation_rank"] = strict_prior_midrank(cboe.delta_log_relative.abs())
    cboe["absolute_ovx_rank"] = strict_prior_midrank(cboe.delta_log_ovx.abs())
    cboe["absolute_vix_rank"] = strict_prior_midrank(cboe.delta_log_vix.abs())

    price = pd.read_csv(PRICE, compression="gzip")
    if price.columns.tolist() != ["hour_start", "hour_open", "first_half_close", "second_half_open", "hour_close", "source_rows", "distinct_timestamps", "source_valid", "decision_time"]:
        raise RuntimeError("OVRCR completed-price schema changed")
    price["hour_start"] = pd.to_datetime(price.hour_start, utc=True)
    price["decision_time"] = pd.to_datetime(price.decision_time, utc=True)
    price["hour_open"] = pd.to_numeric(price.hour_open, errors="coerce")
    price["hour_close"] = pd.to_numeric(price.hour_close, errors="coerce")
    price["valid"] = price.source_valid.astype(str).str.lower().eq("true") & price.source_rows.eq(60) & price.distinct_timestamps.eq(60)
    if price.hour_start.duplicated().any() or not price.hour_start.is_monotonic_increasing:
        raise RuntimeError("OVRCR completed-price clock invalid")
    by_start = price.set_index("hour_start")
    rows: list[dict[str, Any]] = []
    for index in range(1, len(cboe) - 1):
        observation = cboe.at[index, "observation_date"].date()
        next_date = cboe.at[index + 1, "observation_date"].date()
        start = pd.Timestamp(observation).tz_localize(NY) + pd.Timedelta(hours=16)
        final_hour = pd.Timestamp(next_date).tz_localize(NY) + pd.Timedelta(hours=9)
        start, final_hour = start.tz_convert("UTC"), final_hour.tz_convert("UTC")
        if start not in by_start.index or final_hour not in by_start.index:
            continue
        first, last = by_start.loc[start], by_start.loc[final_hour]
        valid = bool(first.valid and last.valid and np.isfinite([first.hour_open, last.hour_close]).all() and first.hour_open > 0 and last.hour_close > 0)
        overnight = float(last.hour_close / first.hour_open - 1.0) if valid else np.nan
        rows.append({
            "source_index": index, "observation_date": observation, "next_source_date": next_date,
            "overnight_start_time": start, "decision_time": final_hour + pd.Timedelta(hours=1),
            "delta_log_relative": float(cboe.at[index, "delta_log_relative"]),
            "absolute_rotation_rank": float(cboe.at[index, "absolute_rotation_rank"]),
            "absolute_ovx_rank": float(cboe.at[index, "absolute_ovx_rank"]),
            "absolute_vix_rank": float(cboe.at[index, "absolute_vix_rank"]),
            "delta_log_ovx": float(cboe.at[index, "delta_log_ovx"]),
            "delta_log_vix": float(cboe.at[index, "delta_log_vix"]),
            "overnight_btc_return": overnight, "valid": valid,
        })
    return pd.DataFrame(rows)


def signal(frame: pd.DataFrame, control: str) -> pd.Series:
    dv, btc, rank = frame.delta_log_relative, frame.overnight_btc_return, frame.absolute_rotation_rank
    valid = frame.valid & dv.ne(0) & btc.ne(0) & rank.notna()
    confirmed = np.sign(btc).eq(-np.sign(dv))
    side = np.sign(btc).astype("Int64").fillna(0).astype(int)
    if control == "ovx_change_only":
        eligible = frame.valid & frame.delta_log_ovx.ne(0) & frame.absolute_ovx_rank.ge(0.60)
        side = -np.sign(frame.delta_log_ovx).astype("Int64").fillna(0).astype(int)
    elif control == "vix_change_only":
        eligible = frame.valid & frame.delta_log_vix.ne(0) & frame.absolute_vix_rank.ge(0.60)
        side = -np.sign(frame.delta_log_vix).astype("Int64").fillna(0).astype(int)
    elif control == "no_btc_confirmation":
        eligible = frame.valid & dv.ne(0) & rank.ge(0.60)
        side = -np.sign(dv).astype("Int64").fillna(0).astype(int)
    elif control == "one_session_stale_rotation":
        stale_dv, stale_rank = dv.shift(1), rank.shift(1)
        eligible = frame.valid & btc.ne(0) & stale_dv.ne(0) & stale_rank.ge(0.60) & np.sign(btc).eq(-np.sign(stale_dv))
    else:
        eligible = valid & confirmed & rank.ge(0.60)
    side = side.where(eligible, 0)
    return -side if control == "direction_flip" else side


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    sides = signal(frame, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[sides.ne(0)]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry, exit_time = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=6, minutes=5)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({
            "candidate": "OVRCR-6", "control": control, "split": split,
            "cboe_observation_date": frame.at[index, "observation_date"].isoformat(),
            "next_cboe_source_date": frame.at[index, "next_source_date"].isoformat(),
            "overnight_start_time": frame.at[index, "overnight_start_time"], "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
            "delta_log_relative": float(frame.at[index, "delta_log_relative"]),
            "absolute_rotation_rank": float(frame.at[index, "absolute_rotation_rank"]),
            "delta_log_ovx": float(frame.at[index, "delta_log_ovx"]),
            "delta_log_vix": float(frame.at[index, "delta_log_vix"]),
            "overnight_btc_return": float(frame.at[index, "overnight_btc_return"]),
        })
    result = pd.DataFrame(rows, columns=COLUMNS)
    for column in ("overnight_start_time", "decision_time", "feature_available_time", "entry_time", "exit_time"):
        if not result.empty:
            result[column] = pd.to_datetime(result[column], utc=True)
    return result


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    registration_core = {key: value for key, value in registration.items() if key != "manifest_hash"}
    if registration.get("manifest_hash") != prereg.canonical_hash(registration_core):
        raise RuntimeError("OVRCR preregistration drift")
    feature_frame = load_features(); primary = build_clock(feature_frame); controls = {name: build_clock(feature_frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True); _write_gzip_csv(primary, CLOCK)
    for name, value in controls.items(): _write_gzip_csv(value, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}; checks: dict[str, bool] = {}
    for name, value in support.items():
        checks[f"{name}_minimum_events"] = value["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = value["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = value["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {"protocol_version": "ovrcr_6_source_support_v1", "policy_id": "OVRCR-6", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]}, "source_bindings": {str(path): digest for path, digest in HASHES.items()}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(value)} for name, value in controls.items()}, "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": ECONOMIC_OUTCOMES_AUTHORIZED, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": canonical_hash(core)}; RESULT.parent.mkdir(parents=True, exist_ok=True); RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); payload = run(); print(json.dumps({"passed": payload["support_passed"], "support": payload["support"]}, indent=2))
