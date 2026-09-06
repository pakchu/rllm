"""Fit and freeze HVHGC-12 before opening OOS source incidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from preprocessing.market_features import build_market_feature_frame
from training import preregister_high_volatility_histogram_classifier_relay as prereg
from training.long_regime_combo_scan import LongComboScanConfig, _load_market
from training.long_regime_interest_gate_validation import build_interest_features


MODEL = Path("data/high_volatility_histogram_classifier_relay_model_2026-08-09.joblib")
RESULT = Path("results/high_volatility_histogram_classifier_relay_model_freeze_2026-08-09.json")
MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
FUNDING = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
PREMIUM = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
FIT_START = pd.Timestamp("2020-01-01T00:00:00Z")
FIT_END = pd.Timestamp("2023-01-01T00:00:00Z")
CALIBRATION_START = FIT_END
CALIBRATION_END = pd.Timestamp("2023-07-01T00:00:00Z")
HOLD_BARS = 144


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_training_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    market = _load_market(
        LongComboScanConfig(
            input_csv=MARKET,
            output="",
            funding_csv=FUNDING,
            premium_csv=PREMIUM,
            exclude_from="2023-07-01",
        )
    )
    market["date"] = pd.to_datetime(market["date"], utc=True)
    base = build_market_feature_frame(market, window_size=144)
    features = pd.concat([base, build_interest_features(market, base)], axis=1)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    ordered = list(prereg.COMPACT_FEATURES)
    missing = sorted(set(ordered) - set(features.columns))
    if missing:
        raise RuntimeError(f"HVHGC pretraining feature drift: {missing}")
    return market, features.loc[:, ordered]


def hourly_anchors(market: pd.DataFrame) -> np.ndarray:
    dates = pd.DatetimeIndex(market["date"])
    positions = np.flatnonzero(dates.minute.to_numpy() == 55).astype(np.int64)
    positions = positions[positions + 1 < len(market)]
    exact = dates[positions + 1] == dates[positions] + pd.Timedelta(minutes=5)
    return positions[np.asarray(exact)]


def run() -> dict[str, Any]:
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    market, features = load_training_frame()
    dates = pd.DatetimeIndex(market["date"])
    opens = pd.to_numeric(market["open"], errors="coerce").to_numpy(float)
    matrix = features.to_numpy(float)
    matrix[~np.isfinite(matrix)] = np.nan
    range_index = list(prereg.COMPACT_FEATURES).index("range_vol")
    anchors = hourly_anchors(market)
    signal_dates = dates[anchors]

    fit_anchors = anchors[(signal_dates >= FIT_START) & (signal_dates < FIT_END)]
    fit_anchors = fit_anchors[fit_anchors + 1 + HOLD_BARS < len(market)]
    entries = fit_anchors + 1
    exits = entries + HOLD_BARS
    complete = (
        (dates[exits] < FIT_END)
        & np.isfinite(opens[entries])
        & np.isfinite(opens[exits])
        & (opens[entries] > 0.0)
        & (opens[exits] > 0.0)
        & np.isfinite(matrix[fit_anchors, range_index])
    )
    fit_anchors = fit_anchors[np.asarray(complete)]
    if len(fit_anchors) < 20_000:
        raise RuntimeError("HVHGC fit row floor failed")
    fit_volatility = matrix[fit_anchors, range_index]
    fit_volatility_threshold = float(np.quantile(fit_volatility, 0.60))
    eligible_fit = fit_anchors[fit_volatility >= fit_volatility_threshold]
    fit_entries = eligible_fit + 1
    fit_exits = fit_entries + HOLD_BARS
    labels = (np.log(opens[fit_exits] / opens[fit_entries]) > 0.0).astype(np.int64)
    if len(eligible_fit) < 8_000 or len(np.unique(labels)) != 2:
        raise RuntimeError("HVHGC eligible fit class floor failed")

    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.03,
            max_iter=400,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=809,
        ),
    )
    pipeline.fit(matrix[eligible_fit], labels)

    calibration_anchors = anchors[
        (signal_dates >= CALIBRATION_START) & (signal_dates < CALIBRATION_END)
    ]
    calibration_volatility = matrix[calibration_anchors, range_index]
    calibration_valid = np.isfinite(calibration_volatility)
    calibration_anchors = calibration_anchors[calibration_valid]
    calibration_volatility = calibration_volatility[calibration_valid]
    if len(calibration_anchors) < 4_000:
        raise RuntimeError("HVHGC calibration source row floor failed")
    probabilities = pipeline.predict_proba(matrix[calibration_anchors])[:, 1]
    low, high = np.quantile(probabilities, (0.10, 0.90))
    volatility_threshold = float(np.quantile(calibration_volatility, 0.60))
    if not (
        np.isfinite(low)
        and 0.0 <= low < high <= 1.0
        and np.isfinite(volatility_threshold)
    ):
        raise RuntimeError("HVHGC calibration threshold drift")

    artifact = {
        "policy_id": "HVHGC-12",
        "pipeline": pipeline,
        "ordered_features": list(prereg.COMPACT_FEATURES),
        "probability_low": float(low),
        "probability_high": float(high),
        "fit_range_vol_q60": fit_volatility_threshold,
        "range_vol_threshold": volatility_threshold,
        "hold_bars": HOLD_BARS,
    }
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL, compress=3)
    core = {
        "protocol_version": "hvhgc_12_model_freeze_v1",
        "policy_id": "HVHGC-12",
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
            "range_vol_q60": fit_volatility_threshold,
            "eligible_high_volatility_rows": int(len(eligible_fit)),
            "positive_labels": int(labels.sum()),
            "negative_labels": int(len(labels) - labels.sum()),
        },
        "calibration": {
            "window": [str(CALIBRATION_START), str(CALIBRATION_END)],
            "source_rows": int(len(calibration_anchors)),
            "probability_low_q10": float(low),
            "probability_high_q90": float(high),
            "range_vol_q60": volatility_threshold,
        },
        "model": {
            "path": str(MODEL),
            "sha256": sha256(MODEL),
            "sklearn_version": sklearn.__version__,
            "ordered_features": list(prereg.COMPACT_FEATURES),
        },
        "advance_to_oos_source_support": True,
        "decision": "model_frozen_before_oos_incidence",
    }
    report = {**core, "manifest_hash": prereg.canonical_hash(core)}
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(
        json.dumps(
            {"fit": report["fit"], "calibration": report["calibration"], "model": report["model"]},
            indent=2,
        )
    )
