"""Fit and freeze HVKMR-24 using only pre-OOS labels and source calibration."""
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
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing.market_features import build_market_feature_frame
from training import preregister_high_volatility_kmeans_regime_relay as prereg
from training.long_regime_combo_scan import LongComboScanConfig, _load_market
from training.long_regime_interest_gate_validation import build_interest_features


MODEL = Path("data/high_volatility_kmeans_regime_relay_model_2026-08-09.joblib")
RESULT = Path("results/high_volatility_kmeans_regime_relay_model_freeze_2026-08-09.json")
MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
FUNDING = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
PREMIUM = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
PREREG_SHA = "31cfe3033739c1647c5cd259fa6f9bcc8701289d2f99ba2827d4ff4555080e1f"
FIT_START = pd.Timestamp("2020-01-01T00:00:00Z")
FIT_END = pd.Timestamp("2023-01-01T00:00:00Z")
CALIBRATION_START = FIT_END
CALIBRATION_END = pd.Timestamp("2023-07-01T00:00:00Z")
HOLD_BARS = 288
ANCHOR_STRIDE = 36
ANCHOR_OFFSET = 143


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
    missing = sorted(set(prereg.FEATURES) - set(features.columns))
    if missing:
        raise RuntimeError(f"HVKMR pretraining feature drift: {missing}")
    return market, features.loc[:, list(prereg.FEATURES)]


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


def choose_state_map(states: np.ndarray, labels: np.ndarray) -> tuple[dict[int, int], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for state in range(12):
        selected = labels[states == state]
        rows.append(
            {
                "state": state,
                "rows": int(len(selected)),
                "mean_forward_log_return": float(np.mean(selected)) if len(selected) else None,
            }
        )
    eligible = [row for row in rows if row["rows"] >= 100]
    positive = sorted(
        (row for row in eligible if row["mean_forward_log_return"] > 0),
        key=lambda row: row["mean_forward_log_return"],
        reverse=True,
    )
    negative = sorted(
        (row for row in eligible if row["mean_forward_log_return"] < 0),
        key=lambda row: row["mean_forward_log_return"],
    )
    if len(positive) < 2 or len(negative) < 2:
        raise RuntimeError("HVKMR state-side mapping floor failed")
    mapping = {int(row["state"]): 1 for row in positive[:2]}
    mapping.update({int(row["state"]): -1 for row in negative[:2]})
    return mapping, rows


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVKMR preregistration drift")
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
    if int(fit_mask.sum()) < 5000 or int(calibration_mask.sum()) < 1000:
        raise RuntimeError("HVKMR pretraining/calibration row floor failed")
    matrix = features.to_numpy(float)
    matrix[~np.isfinite(matrix)] = np.nan
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KMeans(n_clusters=12, n_init=30, max_iter=500, random_state=809),
    )
    fit_positions = positions[fit_mask]
    calibration_positions = positions[calibration_mask]
    pipeline.fit(matrix[fit_positions])
    fit_states = pipeline.predict(matrix[fit_positions])
    state_side_map, state_stats = choose_state_map(fit_states, labels[fit_mask])
    range_index = list(prereg.FEATURES).index("range_vol")
    calibration_volatility = matrix[calibration_positions, range_index]
    calibration_volatility = calibration_volatility[np.isfinite(calibration_volatility)]
    if len(calibration_volatility) < 1000:
        raise RuntimeError("HVKMR calibration volatility floor failed")
    volatility_threshold = float(np.quantile(calibration_volatility, 0.60))
    if not np.isfinite(volatility_threshold):
        raise RuntimeError("HVKMR calibration threshold drift")
    artifact = {
        "policy_id": "HVKMR-24",
        "pipeline": pipeline,
        "ordered_features": list(prereg.FEATURES),
        "state_side_map": state_side_map,
        "range_vol_threshold": volatility_threshold,
        "anchor_offset": ANCHOR_OFFSET,
        "anchor_stride": ANCHOR_STRIDE,
        "hold_bars": HOLD_BARS,
    }
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL, compress=3)
    core = {
        "protocol_version": "hvkmr_24_model_freeze_v1",
        "policy_id": "HVKMR-24",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
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
            "state_statistics": state_stats,
            "state_side_map": {str(key): value for key, value in sorted(state_side_map.items())},
        },
        "calibration": {
            "window": [str(CALIBRATION_START), str(CALIBRATION_END)],
            "rows": int(calibration_mask.sum()),
            "range_vol_q60": volatility_threshold,
        },
        "model": {
            "path": str(MODEL),
            "sha256": sha256(MODEL),
            "sklearn_version": sklearn.__version__,
            "ordered_features": list(prereg.FEATURES),
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
            {
                "fit_rows": report["fit"]["rows"],
                "state_side_map": report["fit"]["state_side_map"],
                "range_vol_q60": report["calibration"]["range_vol_q60"],
                "model_sha256": report["model"]["sha256"],
            },
            indent=2,
        )
    )
