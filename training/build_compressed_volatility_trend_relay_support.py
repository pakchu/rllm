"""Build source-support clocks for CVTR-12 without post-entry outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import preregister_compressed_volatility_trend_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


NONPRICE_DIR = Path("data/options_crowding_deleveraging_relay_sources_v4_2023_2026")
PRICE_DIR = Path("data/options_oi_chase_exhaustion_sources_2023_2026")
CLOCK = Path("data/compressed_volatility_trend_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/compressed_volatility_trend_relay_controls_2023_2026")
RESULT = Path("results/compressed_volatility_trend_relay_support_2026-08-08.json")
SPLITS = base.SPLITS
MINIMUM_EVENTS = {"train": 12, "test": 18, "eval": 18, "final": 12}
CONTROLS = ("no_bvol_contraction", "no_dvol_contraction", "no_two_hour_agreement", "no_oi_stability", "no_funding_neutral", "direction_flip")
ECONOMIC_OUTCOMES_AUTHORIZED = False
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "bvol_body", "dvol_body",
    "previous_hour_return", "hour_return", "prior_abs_return_q40",
    "prior_abs_return_q75", "oi_change", "prior_oi_q35", "prior_oi_q65",
    "funding_rate", "prior_abs_funding_q50",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def features() -> pd.DataFrame:
    bvol, dvol, oi, funding = base.load_sources(NONPRICE_DIR)
    joined = base.joined_features(bvol, dvol, oi, funding)
    finite_oi = joined["oi_change"].where(np.isfinite(joined["oi_change"]))
    joined["oi_q35"] = finite_oi.shift(1).rolling(720, min_periods=672).quantile(0.35)
    joined["oi_q65"] = finite_oi.shift(1).rolling(720, min_periods=672).quantile(0.65)
    funding = funding.copy()
    funding["funding_time"] = pd.to_datetime(funding["funding_time"], utc=True, format="mixed")
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="coerce")
    funding["funding_q50"] = funding["funding_rate"].abs().shift(1).rolling(270, min_periods=252).quantile(0.50)
    joined = joined.merge(funding[["funding_time", "funding_q50"]], on="funding_time", how="left", validate="many_to_one")
    price = pd.read_csv(PRICE_DIR / "btc_completed_hour.csv.gz", compression="gzip")
    price["decision_time"] = pd.to_datetime(price["decision_time"], utc=True, format="mixed")
    price["open"] = pd.to_numeric(price["open"], errors="coerce")
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price["price_valid"] = price["source_valid"].astype(str).str.lower().eq("true")
    price["hour_return"] = price["close"] / price["open"] - 1.0
    abs_return = price["hour_return"].abs().where(price["price_valid"])
    price["return_q40"] = abs_return.shift(1).rolling(720, min_periods=672).quantile(0.40)
    price["return_q75"] = abs_return.shift(1).rolling(720, min_periods=672).quantile(0.75)
    joined = joined.merge(price[["decision_time", "price_valid", "hour_return", "return_q40", "return_q75"]], on="decision_time", validate="one_to_one")
    relevant = ["bvol_body", "dvol_body", "oi_change", "funding_rate", "hour_return"]
    joined["base_valid"] = (
        joined["bvol_valid"] & joined["price_valid"]
        & np.isfinite(joined[relevant]).all(axis=1)
        & joined[["bvol_open", "bvol_close", "dvol_open", "dvol_close", "oi_current", "oi_prior"]].gt(0).all(axis=1)
        & joined["funding_rate"].ne(0) & joined["hour_return"].ne(0)
    )
    return joined


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    bvol_gate = pd.Series(True, index=frame.index) if control == "no_bvol_contraction" else frame["bvol_body"].lt(0)
    dvol_gate = pd.Series(True, index=frame.index) if control == "no_dvol_contraction" else frame["dvol_body"].lt(0)
    middle = (
        frame["return_q40"].notna() & frame["return_q75"].notna()
        & frame["hour_return"].abs().ge(frame["return_q40"])
        & frame["hour_return"].abs().le(frame["return_q75"])
    )
    if control == "no_two_hour_agreement":
        trend = middle
    else:
        trend = middle & middle.shift(1, fill_value=False) & np.sign(frame["hour_return"]).eq(np.sign(frame["hour_return"].shift(1)))
    oi_gate = pd.Series(True, index=frame.index) if control == "no_oi_stability" else (
        frame["oi_q35"].notna() & frame["oi_q65"].notna()
        & frame["oi_change"].ge(frame["oi_q35"]) & frame["oi_change"].le(frame["oi_q65"])
    )
    funding_gate = frame["funding_rate"].ne(0)
    if control != "no_funding_neutral":
        funding_gate &= frame["funding_q50"].notna() & frame["funding_rate"].abs().le(frame["funding_q50"])
    active = frame["base_valid"] & bvol_gate & dvol_gate & trend & oi_gate & funding_gate
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
        exit_time = entry + pd.Timedelta(hours=12)
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
            "candidate": "CVTR-12", "control": control, "split": split,
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": side,
            "bvol_body": float(frame.at[index, "bvol_body"]),
            "dvol_body": float(frame.at[index, "dvol_body"]),
            "previous_hour_return": float(frame.at[index - 1, "hour_return"]),
            "hour_return": float(frame.at[index, "hour_return"]),
            "prior_abs_return_q40": float(frame.at[index, "return_q40"]),
            "prior_abs_return_q75": float(frame.at[index, "return_q75"]),
            "oi_change": float(frame.at[index, "oi_change"]),
            "prior_oi_q35": float(frame.at[index, "oi_q35"]),
            "prior_oi_q65": float(frame.at[index, "oi_q65"]),
            "funding_rate": float(frame.at[index, "funding_rate"]),
            "prior_abs_funding_q50": float(frame.at[index, "funding_q50"]),
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
    passed = all(checks.values())
    core = {
        "protocol_version": "cvtr_12_source_support_v1", "policy_id": "CVTR-12",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha256(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]},
        "source_manifests": {
            "nonprice": {"path": str(NONPRICE_DIR / "manifest.json"), "sha256": sha256(NONPRICE_DIR / "manifest.json")},
            "completed_hour_price": {"path": str(PRICE_DIR / "manifest.json"), "sha256": sha256(PRICE_DIR / "manifest.json")},
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
