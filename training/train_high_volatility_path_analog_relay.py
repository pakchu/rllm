"""Build HVPAR-8 source features and freeze its pre-2023 analog model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_path_analog_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2020-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
LABEL_START = pd.Timestamp("2021-01-01T00:00:00Z")
LABEL_END = pd.Timestamp("2023-01-01T00:00:00Z")
SOURCE_DIR = Path("data/high_volatility_path_analog_relay_sources_2020_2026")
PANEL = SOURCE_DIR / "eight_hour_feature_panel.csv.gz"
TRAINING = SOURCE_DIR / "pre2023_training_rows.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
MODEL = Path("results/high_volatility_path_analog_relay_model_freeze_2026-08-10.json")
PREREG_SHA256 = "73d207071f55dcd1995ca41489bd2113a160c0b348e17e32477cbcf15f080d37"
FEATURES = prereg.FEATURES
NEIGHBORS = 64
QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (np.sum(array < current) + 0.5 * np.sum(array == current)) / len(array)
        if math.isfinite(current):
            history.append(current)
    return output


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop_duplicates("ts", keep=False).set_index("ts").sort_index()


def _ohlc_valid(window: pd.DataFrame) -> bool:
    ohlc = window[["open", "high", "low", "close"]]
    return bool(
        len(window) == 480
        and np.isfinite(ohlc).all(axis=1).all()
        and ohlc.gt(0).all(axis=1).all()
        and window.high.ge(window[["open", "close"]].max(axis=1)).all()
        and window.low.le(window[["open", "close"]].min(axis=1)).all()
        and window.high.ge(window.low).all()
    )


def path_features(window: pd.DataFrame) -> tuple[dict[str, float], float]:
    close = window.close.to_numpy(float)
    log_close = np.log(close)
    minute_returns = np.diff(log_close)
    squared = np.square(minute_returns)
    full_variation = float(squared.sum())
    if full_variation <= 0:
        raise ValueError("HVPAR realized variation invalid")
    root_variation = math.sqrt(full_variation)
    boundary_levels = np.r_[float(window.open.iloc[0]), close[np.arange(29, 480, 30)]]
    normalized_path = np.diff(np.log(boundary_levels)) / root_variation
    later_close_positions = np.arange(1, 480)
    segment_index = np.minimum(later_close_positions // 30, 15)
    segment_variation = np.bincount(segment_index, weights=squared, minlength=16)
    variation_shares = segment_variation / full_variation
    values = {
        **{name: float(value) for name, value in zip(prereg.PATH_FEATURES, normalized_path, strict=True)},
        **{name: float(value) for name, value in zip(prereg.VARIATION_FEATURES, variation_shares, strict=True)},
    }
    if len(boundary_levels) != 17 or not np.isfinite(list(values.values())).all():
        raise ValueError("HVPAR path features invalid")
    return values, full_variation


def feature_panel(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = _prepare(bars)
    decisions = pd.date_range(START.ceil("8h"), END, freq="8h", inclusive="left")
    rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for decision in decisions:
        expected = pd.date_range(decision - pd.Timedelta(hours=8), decision, freq="1min", inclusive="left")
        window = bars.reindex(expected)
        valid = _ohlc_valid(window)
        values = {feature: float("nan") for feature in FEATURES if feature != "variation_rank"}
        full_variation = float("nan")
        if valid:
            try:
                values, full_variation = path_features(window)
            except ValueError:
                valid = False
        rows.append({"decision_time": decision, "source_valid": valid, "full_variation": full_variation, **values})
        entry = decision + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=8)
        if decision >= LABEL_START and exit_ < LABEL_END:
            entry_open = float(bars.open.get(entry, np.nan))
            exit_open = float(bars.open.get(exit_, np.nan))
            label = math.log(exit_open / entry_open) if entry_open > 0 and exit_open > 0 else float("nan")
            labels.append({"decision_time": decision, "entry_time": entry, "exit_time": exit_, "label": label})
    panel = pd.DataFrame(rows)
    panel["variation_rank"] = strict_prior_midrank(panel.full_variation.where(panel.source_valid))
    return panel, pd.DataFrame(labels)


def analog_prediction(
    query: np.ndarray,
    references: np.ndarray,
    labels: np.ndarray,
    decision_order: np.ndarray,
    excluded: np.ndarray | None = None,
) -> float:
    distances = np.linalg.norm(references - query, axis=1)
    if excluded is not None:
        excluded = np.asarray(excluded, dtype=bool)
        if excluded.shape != distances.shape:
            raise ValueError("HVPAR exclusion mask shape invalid")
        distances[excluded] = np.inf
    order = np.lexsort((decision_order, distances))
    selected = order[:NEIGHBORS]
    if len(selected) != NEIGHBORS or not np.isfinite(distances[selected]).all():
        raise RuntimeError("HVPAR neighbor population incomplete")
    weights = 1.0 / np.maximum(distances[selected], 1e-12)
    weights /= weights.sum()
    return float(weights @ labels[selected])


def purged_reference_mask(query_time: pd.Timestamp, reference_times: pd.DatetimeIndex) -> np.ndarray:
    label_start = query_time + pd.Timedelta(minutes=5)
    label_end = label_start + pd.Timedelta(hours=8)
    feature_start = reference_times - pd.Timedelta(hours=8)
    overlap = (reference_times > label_start) & (feature_start < label_end)
    return np.asarray(overlap | (reference_times == query_time), dtype=bool)


def fit_model(panel: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    training = panel.merge(labels, on="decision_time", how="inner", validate="one_to_one")
    finite = np.isfinite(training[list(FEATURES) + ["label"]]).all(axis=1)
    training = training.loc[training.source_valid & finite].copy()
    matrix = training.loc[:, FEATURES].to_numpy(float)
    target = training.label.to_numpy(float)
    if len(training) == 0:
        raise RuntimeError("HVPAR training population empty")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise RuntimeError("HVPAR training standardization invalid")
    standardized = (matrix - mean) / scale
    reference_mask = training.variation_rank.ge(0.65).to_numpy()
    reference_positions = np.flatnonzero(reference_mask)
    if len(reference_positions) <= NEIGHBORS:
        raise RuntimeError("HVPAR high-volatility reference population incomplete")
    reference_matrix = standardized[reference_positions]
    reference_labels = target[reference_positions]
    reference_datetimes = pd.DatetimeIndex(training.decision_time.iloc[reference_positions])
    reference_times = reference_datetimes.astype("int64").to_numpy()
    loo = np.asarray([
        analog_prediction(
            reference_matrix[index],
            reference_matrix,
            reference_labels,
            reference_times,
            purged_reference_mask(reference_datetimes[index], reference_datetimes),
        )
        for index in range(len(reference_positions))
    ])
    threshold = float(np.quantile(np.abs(loo), 0.75, method="linear"))
    if not math.isfinite(threshold) or threshold <= 0:
        raise RuntimeError("HVPAR prediction-strength threshold invalid")
    training["reference_row"] = reference_mask
    training["leave_one_out_prediction"] = np.nan
    training.loc[training.index[reference_positions], "leave_one_out_prediction"] = loo
    training = training[[
        "decision_time", "entry_time", "exit_time", *FEATURES, "label", "reference_row", "leave_one_out_prediction"
    ]]
    model = {
        "feature_order": list(FEATURES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "neighbors": NEIGHBORS,
        "variation_rank_min": 0.65,
        "distance": "standardized_euclidean",
        "weight": "inverse_distance_floor_1e-12",
        "tie_break": "distance_then_decision_time",
        "calibration_purge": "current_reference_and_label_overlapping_feature_windows",
        "prediction_strength_threshold": threshold,
        "training_rows": len(training),
        "reference_rows": int(reference_mask.sum()),
        "training_label_mean": float(target.mean()),
        "training_label_std": float(target.std(ddof=0)),
    }
    return training, model


def run() -> dict[str, Any]:
    from sqlalchemy import text

    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA256:
        raise RuntimeError("HVPAR preregistration artifact drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    if registration != prereg.build():
        raise RuntimeError("HVPAR preregistration payload drift")
    database = postgres_engine()
    with database.connect() as connection:
        bars = pd.read_sql_query(
            text(QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()}
        )
    database.dispose()
    panel, labels = feature_panel(bars)
    training, model = fit_model(panel, labels)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    _write_gzip_csv(training, TRAINING)
    source_core = {
        "protocol_version": "hvpar_8_source_and_pretraining_v1",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA256, "manifest_hash": registration["manifest_hash"]},
        "queries": {"bars": QUERY},
        "tables": ["bars_binance"],
        "symbol": "BTCUSDT", "interval": "1m",
        "window": [START.isoformat(), END.isoformat()],
        "pretraining_label_window": [LABEL_START.isoformat(), LABEL_END.isoformat()],
        "oos_incidence_opened": False, "oos_outcomes_opened": False, "no_imputation": True,
        "feature_panel": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())},
        "pretraining_rows": {"path": str(TRAINING), "sha256": sha(TRAINING), "rows": len(training), "reference_rows": model["reference_rows"]},
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    model_core = {
        "protocol_version": "hvpar_8_model_freeze_v1", "policy_id": "HVPAR-8",
        "preregistration": source_manifest["preregistration"],
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "feature_panel": source_manifest["feature_panel"], "pretraining_rows": source_manifest["pretraining_rows"],
        "estimator": {"class": "frozen_inverse_distance_nearest_analogs", "neighbors": NEIGHBORS, "hyperparameter_grid": False, "feature_selection": False},
        **model, "oos_incidence_opened": False, "oos_outcomes_opened": False, "refit_authorized": False,
    }
    result = {**model_core, "manifest_hash": canonical_hash(model_core)}
    MODEL.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"training_rows": result["training_rows"], "reference_rows": result["reference_rows"], "threshold": result["prediction_strength_threshold"], "oos_incidence_opened": result["oos_incidence_opened"]}, indent=2))
