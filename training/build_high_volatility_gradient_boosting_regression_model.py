"""Fit and freeze HVGBR-72 using only pre-OOS labels and calibration sources."""
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from preprocessing.market_features import build_market_feature_frame
from training import preregister_high_volatility_gradient_boosting_regression_relay as prereg
from training.long_regime_combo_scan import LongComboScanConfig, _load_market
from training.long_regime_interest_gate_validation import build_interest_features


MODEL = Path("data/high_volatility_gradient_boosting_regression_relay_model_2026-08-09.joblib")
RESULT = Path("results/high_volatility_gradient_boosting_regression_relay_model_freeze_2026-08-09.json")
MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
FUNDING = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
PREMIUM = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
FIT_START = pd.Timestamp("2020-01-01T00:00:00Z")
FIT_END = pd.Timestamp("2023-01-01T00:00:00Z")
CALIBRATION_START = FIT_END
CALIBRATION_END = pd.Timestamp("2023-07-01T00:00:00Z")
HOLD_BARS = 864
ANCHOR_STRIDE = 72
ANCHOR_OFFSET = 143


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return prereg.canonical_hash(payload)


def load_training_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = LongComboScanConfig(
        input_csv=MARKET,
        output="",
        funding_csv=FUNDING,
        premium_csv=PREMIUM,
        exclude_from="2023-07-01",
    )
    market = _load_market(cfg)
    market["date"] = pd.to_datetime(market["date"], utc=True)
    base = build_market_feature_frame(market, window_size=144)
    features = pd.concat([base, build_interest_features(market, base)], axis=1)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    ordered = list(prereg.COMPACT_FEATURES)
    missing = sorted(set(ordered) - set(features.columns))
    if missing:
        raise RuntimeError(f"HVGBR pretraining feature drift: {missing}")
    return market, features.loc[:, ordered]


def anchor_rows(market: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.arange(
        ANCHOR_OFFSET,
        len(market) - HOLD_BARS - 2,
        ANCHOR_STRIDE,
        dtype=np.int64,
    )
    opens = pd.to_numeric(market["open"], errors="coerce").to_numpy(float)
    entry = positions + 1
    exit_position = entry + HOLD_BARS
    valid = (
        np.isfinite(opens[entry])
        & np.isfinite(opens[exit_position])
        & (opens[entry] > 0)
        & (opens[exit_position] > 0)
    )
    positions = positions[valid]
    labels = np.log(opens[positions + 1 + HOLD_BARS] / opens[positions + 1])
    dates = pd.to_datetime(market["date"], utc=True).to_numpy()
    return positions, labels, dates


def run() -> dict[str, Any]:
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    market, features = load_training_frame()
    positions, labels, dates = anchor_rows(market)
    signal_dates = pd.to_datetime(dates[positions], utc=True)
    exit_dates = pd.to_datetime(dates[positions + 1 + HOLD_BARS], utc=True)
    fit_mask = (
        (signal_dates >= FIT_START)
        & (signal_dates < FIT_END)
        & (exit_dates < FIT_END)
    )
    calibration_mask = (
        (signal_dates >= CALIBRATION_START)
        & (signal_dates < CALIBRATION_END)
        & (exit_dates < CALIBRATION_END)
    )
    if int(fit_mask.sum()) < 1000 or int(calibration_mask.sum()) < 500:
        raise RuntimeError("HVGBR pretraining/calibration row floor failed")
    matrix = features.to_numpy(float)
    matrix[~np.isfinite(matrix)] = np.nan
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=0.5,
            random_state=809,
        ),
    )
    fit_positions = positions[fit_mask]
    calibration_positions = positions[calibration_mask]
    pipeline.fit(matrix[fit_positions], labels[fit_mask])
    predictions = pipeline.predict(matrix[calibration_positions])
    low, high = np.quantile(predictions, (0.15, 0.85))
    range_index = list(prereg.COMPACT_FEATURES).index("range_vol")
    calibration_volatility = matrix[calibration_positions, range_index]
    calibration_volatility = calibration_volatility[np.isfinite(calibration_volatility)]
    if len(calibration_volatility) < 500:
        raise RuntimeError("HVGBR calibration volatility floor failed")
    volatility_threshold = float(np.quantile(calibration_volatility, 0.70))
    if not (np.isfinite(low) and low < high and np.isfinite(high) and np.isfinite(volatility_threshold)):
        raise RuntimeError("HVGBR calibration threshold drift")
    artifact = {
        "policy_id": "HVGBR-72",
        "pipeline": pipeline,
        "ordered_features": list(prereg.COMPACT_FEATURES),
        "prediction_low": float(low),
        "prediction_high": float(high),
        "range_vol_threshold": volatility_threshold,
        "anchor_offset": ANCHOR_OFFSET,
        "anchor_stride": ANCHOR_STRIDE,
        "hold_bars": HOLD_BARS,
    }
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL, compress=3)
    core = {
        "protocol_version": "hvgbr_72_model_freeze_v1",
        "policy_id": "HVGBR-72",
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
            "rows": int(fit_mask.sum()),
            "target_mean": float(np.mean(labels[fit_mask])),
            "target_std": float(np.std(labels[fit_mask])),
            "positive_targets": int(np.count_nonzero(labels[fit_mask] > 0)),
            "negative_targets": int(np.count_nonzero(labels[fit_mask] <= 0)),
        },
        "calibration": {
            "window": [str(CALIBRATION_START), str(CALIBRATION_END)],
            "rows": int(calibration_mask.sum()),
            "prediction_low_q15": float(low),
            "prediction_high_q85": float(high),
            "range_vol_q70": volatility_threshold,
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
                "calibration": report["calibration"],
                "model_sha256": report["model"]["sha256"],
            },
            indent=2,
        )
    )
