"""Build HVLPSR-8 source features and freeze its pre-2023 ridge model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_leverage_premium_state_ridge_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2020-09-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
LABEL_START = pd.Timestamp("2021-01-01T00:00:00Z")
LABEL_END = pd.Timestamp("2023-01-01T00:00:00Z")
SOURCE_DIR = Path("data/high_volatility_leverage_premium_state_ridge_relay_sources_2020_2026")
PANEL = SOURCE_DIR / "eight_hour_feature_panel.csv.gz"
TRAINING = SOURCE_DIR / "pre2023_training_rows.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
MODEL = Path("results/high_volatility_leverage_premium_state_ridge_relay_model_freeze_2026-08-10.json")
PREREG_SHA256 = "4c393fa82a306f52c0fe42e7e2dab5cf5af47d95209b4745ddf1e7d8f04069d3"
FEATURES = tuple(prereg.build()["feature_contract"]["ordered_features"])
BAR_QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
PREMIUM_QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance_premium
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
OI_QUERY = """
SELECT ts,sum_open_interest
FROM open_interest_binance
WHERE symbol='BTCUSDT' AND period='5m' AND source='open_interest_hist'
  AND ts>=:start AND ts<:end
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


def _prepare_oi(frame: pd.DataFrame) -> pd.Series:
    frame = frame.copy()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True)
    frame["sum_open_interest"] = pd.to_numeric(frame.sum_open_interest, errors="coerce")
    return frame.drop_duplicates("ts", keep=False).set_index("ts").sort_index().sum_open_interest


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


def _premium_valid(window: pd.DataFrame) -> bool:
    ohlc = window[["open", "high", "low", "close"]]
    return bool(
        len(window) == 480
        and np.isfinite(ohlc).all(axis=1).all()
        and window.high.ge(window[["open", "close"]].max(axis=1)).all()
        and window.low.le(window[["open", "close"]].min(axis=1)).all()
        and window.high.ge(window.low).all()
    )


def leverage_premium_state_features(
    price: pd.DataFrame, premium: pd.DataFrame, oi: pd.Series
) -> tuple[dict[str, float], float]:
    price_open = price.open.to_numpy(float).reshape(96, 5)[:, 0]
    price_close = price.close.to_numpy(float).reshape(96, 5)[:, -1]
    returns = np.log(price_close / price_open)
    full_variation = float(np.square(returns).sum())
    root_variation = math.sqrt(full_variation)
    full_return = math.log(float(price.close.iloc[-1]) / float(price.open.iloc[0]))
    late_return = math.log(float(price.close.iloc[-1]) / float(price.open.iloc[360]))
    oi_values = oi.to_numpy(float)
    oi_changes = np.diff(np.log(oi_values))
    full_oi_change = float(oi_changes.sum())
    late_oi_change = float(oi_changes[-24:].sum())
    oi_path_length = float(np.abs(oi_changes).sum())
    oi_range = float(oi_values.max() - oi_values.min())
    premium_values = premium.close.to_numpy(float)
    premium_change = float(premium_values[-1] - premium_values[0])
    premium_range = float(premium_values.max() - premium_values.min())
    if root_variation <= 0 or oi_path_length <= 0 or oi_range <= 0 or premium_range <= 0:
        raise ValueError("leverage-premium feature denominator invalid")
    normalized_full_return = full_return / root_variation
    values = {
        "normalized_full_return": normalized_full_return,
        "normalized_late_return": late_return / root_variation,
        "full_oi_log_change": full_oi_change,
        "late_oi_log_change": late_oi_change,
        "oi_path_efficiency": abs(full_oi_change) / oi_path_length,
        "oi_change_variation": math.sqrt(float(np.square(oi_changes).sum())),
        "oi_terminal_location": 2.0 * (float(oi_values[-1]) - float(oi_values.min())) / oi_range - 1.0,
        "full_premium_mean": float(premium_values.mean()),
        "late_premium_mean": float(premium_values[-120:].mean()),
        "terminal_premium": float(premium_values[-1]),
        "premium_change": premium_change,
        "premium_range": premium_range,
        "return_oi_sponsorship": normalized_full_return * full_oi_change,
        "return_premium_dislocation": normalized_full_return * premium_change,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("leverage-premium features nonfinite")
    return values, full_variation


def feature_panel(bars: pd.DataFrame, premium: pd.DataFrame, oi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = _prepare(bars)
    premium = _prepare(premium)
    oi_series = _prepare_oi(oi)
    decisions = pd.date_range(START.ceil("8h"), END, freq="8h", inclusive="left")
    rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for decision in decisions:
        expected = pd.date_range(decision - pd.Timedelta(hours=8), decision, freq="1min", inclusive="left")
        window = bars.reindex(expected)
        premium_window = premium.reindex(expected)
        oi_expected = pd.date_range(decision - pd.Timedelta(hours=8, minutes=5), decision, freq="5min", inclusive="left")
        oi_window = oi_series.reindex(oi_expected)
        valid = _ohlc_valid(window) and _premium_valid(premium_window) and bool(
            len(oi_window) == 97 and np.isfinite(oi_window).all() and oi_window.gt(0).all()
        )
        values = {feature: float("nan") for feature in FEATURES if feature != "variation_rank"}
        full_variation = float("nan")
        if valid:
            try:
                values, full_variation = leverage_premium_state_features(window, premium_window, oi_window)
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


def fit_model(panel: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    training = panel.merge(labels, on="decision_time", how="inner", validate="one_to_one")
    finite = np.isfinite(training[list(FEATURES) + ["label"]]).all(axis=1)
    training = training.loc[training.source_valid & finite].copy()
    matrix = training.loc[:, FEATURES].to_numpy(float)
    target = training.label.to_numpy(float)
    if len(training) == 0:
        raise RuntimeError("HVLPSR training population empty")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise RuntimeError("HVLPSR training standardization invalid")
    standardized = (matrix - mean) / scale
    target_mean = float(target.mean())
    gram = standardized.T @ standardized + 100.0 * np.eye(standardized.shape[1])
    coefficient = np.linalg.solve(gram, standardized.T @ (target - target_mean))
    fitted = standardized @ coefficient + target_mean
    high_volatility = training.variation_rank.ge(0.65).to_numpy()
    if high_volatility.sum() == 0:
        raise RuntimeError("HVLPSR training high-volatility population empty")
    threshold = float(np.quantile(np.abs(fitted[high_volatility]), 0.75, method="linear"))
    if not math.isfinite(threshold) or threshold <= 0:
        raise RuntimeError("HVLPSR prediction-strength threshold invalid")
    training["fitted_prediction"] = fitted
    training = training[["decision_time", "entry_time", "exit_time", *FEATURES, "label", "fitted_prediction"]]
    model = {
        "feature_order": list(FEATURES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": coefficient.astype(float).tolist(),
        "intercept": target_mean,
        "prediction_strength_threshold": threshold,
        "training_rows": len(training),
        "training_high_volatility_rows": int(high_volatility.sum()),
        "training_label_mean": target_mean,
        "training_label_std": float(target.std(ddof=0)),
    }
    return training, model


def run() -> dict[str, Any]:
    from sqlalchemy import text

    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA256:
        raise RuntimeError("HVLPSR preregistration artifact drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    if registration != prereg.build():
        raise RuntimeError("HVLPSR preregistration payload drift")
    database = postgres_engine()
    with database.connect() as connection:
        bars = pd.read_sql_query(
            text(BAR_QUERY),
            connection,
            params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
        )
        premium = pd.read_sql_query(
            text(PREMIUM_QUERY), connection,
            params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
        )
        oi = pd.read_sql_query(
            text(OI_QUERY), connection,
            params={"start": (START - pd.Timedelta(minutes=5)).to_pydatetime(), "end": END.to_pydatetime()},
        )
    database.dispose()
    panel, labels = feature_panel(bars, premium, oi)
    training, model = fit_model(panel, labels)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    _write_gzip_csv(training, TRAINING)
    source_core = {
        "protocol_version": "hvlpsr_8_source_and_pretraining_v1",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA256,
            "manifest_hash": registration["manifest_hash"],
        },
        "queries": {"bars": BAR_QUERY, "premium": PREMIUM_QUERY, "open_interest": OI_QUERY},
        "tables": ["bars_binance", "bars_binance_premium", "open_interest_binance"],
        "symbol": "BTCUSDT",
        "interval": "1m",
        "window": [START.isoformat(), END.isoformat()],
        "pretraining_label_window": [LABEL_START.isoformat(), LABEL_END.isoformat()],
        "oos_incidence_opened": False,
        "oos_outcomes_opened": False,
        "no_imputation": True,
        "feature_panel": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())},
        "pretraining_rows": {"path": str(TRAINING), "sha256": sha(TRAINING), "rows": len(training)},
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    model_core = {
        "protocol_version": "hvlpsr_8_model_freeze_v1",
        "policy_id": "HVLPSR-8",
        "preregistration": source_manifest["preregistration"],
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "feature_panel": source_manifest["feature_panel"],
        "pretraining_rows": source_manifest["pretraining_rows"],
        "estimator": {
            "class": "closed_form_standardized_ridge",
            "alpha": 100.0,
            "fit_intercept": True,
            "sample_weight": None,
            "hyperparameter_grid": False,
            "feature_selection": False,
        },
        **model,
        "oos_incidence_opened": False,
        "oos_outcomes_opened": False,
        "refit_authorized": False,
    }
    result = {**model_core, "manifest_hash": canonical_hash(model_core)}
    MODEL.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({
        "training_rows": result["training_rows"],
        "high_volatility_rows": result["training_high_volatility_rows"],
        "threshold": result["prediction_strength_threshold"],
        "oos_incidence_opened": result["oos_incidence_opened"],
    }, indent=2))
