"""Fit and freeze HVETCR-48 using only pre-OOS labels and calibration sources."""
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from preprocessing.market_features import build_market_feature_frame
from training import preregister_high_volatility_extra_trees_classification_relay as prereg
from training.long_regime_combo_scan import LongComboScanConfig, _load_market
from training.long_regime_interest_gate_validation import build_interest_features


MODEL = Path("data/high_volatility_extra_trees_classification_relay_model_2026-08-09.joblib")
RESULT = Path("results/high_volatility_extra_trees_classification_relay_model_freeze_2026-08-09.json")
MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
FUNDING = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
PREMIUM = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
FIT_START = pd.Timestamp("2020-01-01T00:00:00Z")
FIT_END = pd.Timestamp("2023-01-01T00:00:00Z")
CALIBRATION_START = FIT_END
CALIBRATION_END = pd.Timestamp("2023-07-01T00:00:00Z")
HOLD_BARS = 576
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
        raise RuntimeError(f"HVETCR pretraining feature drift: {missing}")
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
    labels = (np.log(opens[positions + 1 + HOLD_BARS] / opens[positions + 1]) > 0).astype(int)
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
        raise RuntimeError("HVETCR pretraining/calibration row floor failed")
    matrix = features.to_numpy(float)
    matrix[~np.isfinite(matrix)] = np.nan
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=8,
            max_features=0.7,
            class_weight="balanced",
            n_jobs=-1,
            random_state=809,
        ),
    )
    fit_positions = positions[fit_mask]
    calibration_positions = positions[calibration_mask]
    pipeline.fit(matrix[fit_positions], labels[fit_mask])
    probabilities = pipeline.predict_proba(matrix[calibration_positions])[:, 1]
    low, high = np.quantile(probabilities, (0.20, 0.80))
    range_index = list(prereg.COMPACT_FEATURES).index("range_vol")
    calibration_volatility = matrix[calibration_positions, range_index]
    calibration_volatility = calibration_volatility[np.isfinite(calibration_volatility)]
    if len(calibration_volatility) < 500:
        raise RuntimeError("HVETCR calibration volatility floor failed")
    volatility_threshold = float(np.quantile(calibration_volatility, 0.65))
    if not (0.0 <= low < high <= 1.0) or not np.isfinite(volatility_threshold):
        raise RuntimeError("HVETCR calibration threshold drift")
    artifact = {
        "policy_id": "HVETCR-48",
        "pipeline": pipeline,
        "ordered_features": list(prereg.COMPACT_FEATURES),
        "probability_low": float(low),
        "probability_high": float(high),
        "range_vol_threshold": volatility_threshold,
        "anchor_offset": ANCHOR_OFFSET,
        "anchor_stride": ANCHOR_STRIDE,
        "hold_bars": HOLD_BARS,
    }
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL, compress=3)
    core = {
        "protocol_version": "hvetcr_48_model_freeze_v1",
        "policy_id": "HVETCR-48",
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
            "positive_labels": int(labels[fit_mask].sum()),
            "negative_labels": int(fit_mask.sum() - labels[fit_mask].sum()),
        },
        "calibration": {
            "window": [str(CALIBRATION_START), str(CALIBRATION_END)],
            "rows": int(calibration_mask.sum()),
            "probability_low_q20": float(low),
            "probability_high_q80": float(high),
            "range_vol_q65": volatility_threshold,
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
