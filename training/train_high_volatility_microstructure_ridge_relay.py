"""Build HVMRR-6 source features and freeze its pre-2023H2 ridge model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from training import preregister_high_volatility_microstructure_ridge_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2020-10-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
LABEL_START = pd.Timestamp("2021-01-01T00:00:00Z")
LABEL_END = pd.Timestamp("2023-07-01T00:00:00Z")
SOURCE_DIR = Path("data/high_volatility_microstructure_ridge_relay_sources_2020_2026")
PANEL = SOURCE_DIR / "eight_hour_feature_panel.csv.gz"
TRAINING = SOURCE_DIR / "pre2023h2_training_rows.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
MODEL = Path("results/high_volatility_microstructure_ridge_relay_model_freeze_2026-08-09.json")
FEATURES = tuple(prereg.build()["feature_contract"]["ordered_features"])
PERP_QUERY = """
SELECT ts,open,high,low,close,quote_asset_volume,taker_buy_quote
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
SPOT_QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance_spot
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


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


def _prepare(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = frame.copy()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True)
    for column in columns:
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


def feature_panel(perpetual: pd.DataFrame, spot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    perpetual = _prepare(perpetual, ("open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote"))
    spot = _prepare(spot, ("open", "high", "low", "close"))
    decisions = pd.date_range(START.ceil("8h"), END, freq="8h", inclusive="left")
    rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for decision in decisions:
        expected = pd.date_range(decision - pd.Timedelta(hours=8), decision, freq="1min", inclusive="left")
        perp_window = perpetual.reindex(expected)
        spot_window = spot.reindex(expected)
        flow_valid = bool(
            np.isfinite(perp_window[["quote_asset_volume", "taker_buy_quote"]]).all(axis=1).all()
            and perp_window[["quote_asset_volume", "taker_buy_quote"]].ge(0).all(axis=1).all()
            and perp_window.taker_buy_quote.le(perp_window.quote_asset_volume).all()
        )
        valid = _ohlc_valid(perp_window) and _ohlc_valid(spot_window) and flow_valid
        values = {feature: float("nan") for feature in FEATURES if feature != "variation_rank"}
        full_variation = float("nan")
        if valid:
            close_logs = np.log(perp_window.close.astype(float))
            minute_returns = close_logs.diff().dropna()
            full_variation = float(minute_returns.pow(2).sum())
            root_variation = math.sqrt(full_variation) if full_variation > 0 else float("nan")
            path_length = float(minute_returns.abs().sum())
            full_return = float(np.log(float(perp_window.close.iloc[-1]) / float(perp_window.open.iloc[0])))
            late_return = float(np.log(float(perp_window.close.iloc[-1]) / float(perp_window.open.iloc[360])))
            full_quote = float(perp_window.quote_asset_volume.sum())
            full_buy = float(perp_window.taker_buy_quote.sum())
            late = perp_window.iloc[360:]
            late_quote = float(late.quote_asset_volume.sum())
            late_buy = float(late.taker_buy_quote.sum())
            basis_start = float(np.log(float(perp_window.open.iloc[360]) / float(spot_window.open.iloc[360])))
            basis_end = float(np.log(float(perp_window.close.iloc[-1]) / float(spot_window.close.iloc[-1])))
            hour = decision.hour
            valid = all(value > 0 for value in (root_variation, path_length, full_quote, late_quote))
            if valid:
                values = {
                    "normalized_full_return": full_return / root_variation,
                    "normalized_late_return": late_return / root_variation,
                    "path_efficiency": abs(full_return) / path_length,
                    "full_taker_imbalance": (2 * full_buy - full_quote) / full_quote,
                    "late_taker_imbalance": (2 * late_buy - late_quote) / late_quote,
                    "late_quote_volume_share": late_quote / full_quote,
                    "normalized_cash_basis_change": (basis_end - basis_start) / root_variation,
                    "decision_hour_sin": math.sin(2 * math.pi * hour / 24),
                    "decision_hour_cos": math.cos(2 * math.pi * hour / 24),
                }
        rows.append({"decision_time": decision, "source_valid": valid, "full_variation": full_variation, **values})
        entry = decision + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=6)
        if decision >= LABEL_START and exit_ < LABEL_END:
            entry_open = float(perpetual.open.get(entry, np.nan))
            exit_open = float(perpetual.open.get(exit_, np.nan))
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
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise RuntimeError("HVMRR training standardization invalid")
    standardized = (matrix - mean) / scale
    estimator = Ridge(alpha=10.0, fit_intercept=True, solver="svd")
    estimator.fit(standardized, target)
    fitted = estimator.predict(standardized)
    high_volatility = training.variation_rank.ge(0.65).to_numpy()
    if high_volatility.sum() == 0:
        raise RuntimeError("HVMRR training high-volatility population empty")
    threshold = float(np.quantile(np.abs(fitted[high_volatility]), 0.80, method="linear"))
    training["fitted_prediction"] = fitted
    training = training[["decision_time", "entry_time", "exit_time", *FEATURES, "label", "fitted_prediction"]]
    model = {
        "feature_order": list(FEATURES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": estimator.coef_.astype(float).tolist(),
        "intercept": float(estimator.intercept_),
        "prediction_strength_threshold": threshold,
        "training_rows": len(training),
        "training_high_volatility_rows": int(high_volatility.sum()),
        "training_label_mean": float(target.mean()),
        "training_label_std": float(target.std(ddof=0)),
    }
    return training, model


def run() -> dict[str, Any]:
    from sqlalchemy import text

    database = postgres_engine()
    with database.connect() as connection:
        perpetual = pd.read_sql_query(text(PERP_QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()})
        spot = pd.read_sql_query(text(SPOT_QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()})
    database.dispose()
    panel, labels = feature_panel(perpetual, spot)
    training, model = fit_model(panel, labels)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    _write_gzip_csv(training, TRAINING)
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    source_core = {
        "protocol_version": "hvmrr_6_source_and_pretraining_v1",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]},
        "queries": {"perpetual": PERP_QUERY, "spot": SPOT_QUERY},
        "tables": ["bars_binance", "bars_binance_spot"],
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
        "protocol_version": "hvmrr_6_model_freeze_v1",
        "policy_id": "HVMRR-6",
        "preregistration": source_manifest["preregistration"],
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "feature_panel": source_manifest["feature_panel"],
        "pretraining_rows": source_manifest["pretraining_rows"],
        "estimator": {"class": "sklearn.linear_model.Ridge", "alpha": 10.0, "fit_intercept": True, "solver": "svd"},
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
    print(json.dumps({"training_rows": result["training_rows"], "high_volatility_rows": result["training_high_volatility_rows"], "threshold": result["prediction_strength_threshold"], "oos_incidence_opened": result["oos_incidence_opened"]}, indent=2))
