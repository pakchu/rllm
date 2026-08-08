"""Fit and freeze HVHOW-6 using pre-2023 labels and 2023H1 sources only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training import preregister_high_volatility_hour_of_week_relay as prereg
from training.long_regime_combo_scan import LongComboScanConfig, _load_market


MODEL = Path("data/high_volatility_hour_of_week_relay_model_2026-08-09.joblib")
RESULT = Path("results/high_volatility_hour_of_week_relay_model_freeze_2026-08-09.json")
MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
FIT_START = pd.Timestamp("2020-01-01T00:00:00Z")
FIT_END = pd.Timestamp("2023-01-01T00:00:00Z")
CALIBRATION_START = FIT_END
CALIBRATION_END = pd.Timestamp("2023-07-01T00:00:00Z")
HOLD_BARS = 72
MIN_SLOT_ROWS = 100


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return prereg.canonical_hash(payload)


def load_source() -> tuple[pd.DataFrame, np.ndarray]:
    market = _load_market(
        LongComboScanConfig(input_csv=MARKET, output="", exclude_from="2023-07-01")
    )
    market["date"] = pd.to_datetime(market["date"], utc=True)
    range_vol = pd.to_numeric(
        build_market_feature_frame(market, window_size=144)["range_vol"],
        errors="coerce",
    ).to_numpy(float)
    return market, range_vol


def completed_hour_anchors(market: pd.DataFrame) -> np.ndarray:
    dates = pd.DatetimeIndex(market["date"])
    positions = np.flatnonzero(dates.minute.to_numpy() == 55).astype(np.int64)
    positions = positions[positions + 1 < len(market)]
    exact_next_hour = dates[positions + 1] == dates[positions] + pd.Timedelta(minutes=5)
    return positions[np.asarray(exact_next_hour)]


def hour_of_week(dates: pd.DatetimeIndex) -> np.ndarray:
    return (dates.weekday.to_numpy() * 24 + dates.hour.to_numpy()).astype(np.int64)


def run() -> dict[str, Any]:
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    market, range_vol = load_source()
    dates = pd.DatetimeIndex(market["date"])
    opens = pd.to_numeric(market["open"], errors="coerce").to_numpy(float)
    anchors = completed_hour_anchors(market)
    signal_dates = dates[anchors]

    fit_anchors = anchors[(signal_dates >= FIT_START) & (signal_dates < FIT_END)]
    fit_anchors = fit_anchors[fit_anchors + 1 + HOLD_BARS < len(market)]
    fit_entries = fit_anchors + 1
    fit_exits = fit_entries + HOLD_BARS
    complete_fit = (
        (dates[fit_exits] < FIT_END)
        & np.isfinite(opens[fit_entries])
        & np.isfinite(opens[fit_exits])
        & (opens[fit_entries] > 0.0)
        & (opens[fit_exits] > 0.0)
        & np.isfinite(range_vol[fit_anchors])
    )
    fit_anchors = fit_anchors[np.asarray(complete_fit)]
    fit_entries = fit_anchors + 1
    fit_exits = fit_entries + HOLD_BARS
    if len(fit_anchors) < 20_000:
        raise RuntimeError("HVHOW fit row floor failed")

    fit_volatility_threshold = float(np.quantile(range_vol[fit_anchors], 0.60))
    high_volatility = range_vol[fit_anchors] >= fit_volatility_threshold
    high_vol_anchors = fit_anchors[high_volatility]
    high_vol_entries = high_vol_anchors + 1
    high_vol_exits = high_vol_entries + HOLD_BARS
    labels = np.log(opens[high_vol_exits] / opens[high_vol_entries])
    slots = hour_of_week(dates[high_vol_entries])
    frame = pd.DataFrame({"slot": slots, "label": labels})
    summary = frame.groupby("slot", sort=True)["label"].agg(["count", "mean"])
    eligible = summary[summary["count"] >= MIN_SLOT_ROWS].copy()
    positive = eligible[eligible["mean"] > 0.0].sort_values(
        ["mean"], ascending=[False], kind="stable"
    )
    negative = eligible[eligible["mean"] < 0.0].sort_values(
        ["mean"], ascending=[True], kind="stable"
    )
    if len(positive) < 2 or len(negative) < 2:
        core = {
            "protocol_version": "hvhow_6_model_freeze_v1",
            "policy_id": "HVHOW-6",
            "preregistration": {
                "path": str(prereg.DEFAULT_OUTPUT),
                "sha256": sha256(prereg.DEFAULT_OUTPUT),
                "manifest_hash": registration["manifest_hash"],
            },
            "source_bindings_verified": True,
            "pretraining_outcomes_opened": True,
            "calibration_sources_opened": True,
            "calibration_outcomes_opened": False,
            "oos_source_incidence_opened": False,
            "oos_post_entry_outcomes_opened": False,
            "gross9_rows_opened": False,
            "fit": {
                "window": [str(FIT_START), str(FIT_END)],
                "complete_rows": int(len(fit_anchors)),
                "high_volatility_rows": int(len(high_vol_anchors)),
                "range_vol_q60": fit_volatility_threshold,
                "slot_floor": MIN_SLOT_ROWS,
                "maximum_slot_rows": int(summary["count"].max()),
                "eligible_slots": int(len(eligible)),
                "eligible_positive_slots": int(len(positive)),
                "eligible_negative_slots": int(len(negative)),
            },
            "model_created": False,
            "advance_to_oos_source_support": False,
            "terminal_failure": True,
            "decision": "fit_slot_floor_failed",
            "failure_reason": (
                "No hour-of-week slot reached the preregistered 100-row floor among "
                "pre-2023 high-volatility fit labels."
            ),
        }
        report = {**core, "manifest_hash": canonical_hash(core)}
        RESULT.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        return report
    positive_slots = [int(value) for value in positive.index[:2]]
    negative_slots = [int(value) for value in negative.index[:2]]
    selected_slots = positive_slots + negative_slots
    slot_sides = {slot: 1 for slot in positive_slots} | {slot: -1 for slot in negative_slots}

    calibration_anchors = anchors[
        (signal_dates >= CALIBRATION_START) & (signal_dates < CALIBRATION_END)
    ]
    calibration_values = range_vol[calibration_anchors]
    calibration_values = calibration_values[np.isfinite(calibration_values)]
    if len(calibration_values) < 4_000:
        raise RuntimeError("HVHOW calibration source row floor failed")
    calibration_threshold = float(np.quantile(calibration_values, 0.60))

    artifact = {
        "policy_id": "HVHOW-6",
        "selected_slots": selected_slots,
        "slot_sides": slot_sides,
        "fit_range_vol_q60": fit_volatility_threshold,
        "range_vol_threshold": calibration_threshold,
        "window_size": 144,
        "hold_bars": HOLD_BARS,
    }
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL, compress=3)
    selected = [
        {
            "slot": slot,
            "weekday": int(slot // 24),
            "hour": int(slot % 24),
            "side": int(slot_sides[slot]),
            "fit_rows": int(summary.loc[slot, "count"]),
            "fit_mean_log_return": float(summary.loc[slot, "mean"]),
        }
        for slot in selected_slots
    ]
    core = {
        "protocol_version": "hvhow_6_model_freeze_v1",
        "policy_id": "HVHOW-6",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha256(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_bindings_verified": True,
        "pretraining_outcomes_opened": True,
        "calibration_sources_opened": True,
        "calibration_outcomes_opened": False,
        "oos_source_incidence_opened": False,
        "oos_post_entry_outcomes_opened": False,
        "gross9_rows_opened": False,
        "fit": {
            "window": [str(FIT_START), str(FIT_END)],
            "complete_rows": int(len(fit_anchors)),
            "high_volatility_rows": int(len(high_vol_anchors)),
            "range_vol_q60": fit_volatility_threshold,
            "eligible_slots": int(len(eligible)),
            "selected": selected,
        },
        "calibration": {
            "window": [str(CALIBRATION_START), str(CALIBRATION_END)],
            "source_rows": int(len(calibration_values)),
            "range_vol_q60": calibration_threshold,
        },
        "model": {"path": str(MODEL), "sha256": sha256(MODEL)},
        "advance_to_oos_source_support": True,
        "decision": "model_frozen_before_oos_incidence",
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = run()
    print(
        json.dumps(
            {
                "fit": report["fit"],
                "decision": report["decision"],
                "terminal_failure": report.get("terminal_failure", False),
                "model": report.get("model"),
            },
            indent=2,
        )
    )
