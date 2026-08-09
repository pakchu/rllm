"""Freeze HVKAR-12 fit analogs and build source-support clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_knn_analog_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = Path("/home/pakchu/rllm/.env")
END = pd.Timestamp("2026-08-01T00:00:00Z")
EXTENSION_START = pd.Timestamp("2026-05-01T00:00:00Z")
FIT_START = pd.Timestamp("2021-01-01T00:00:00Z")
FIT_END = pd.Timestamp("2023-07-01T00:00:00Z")
FEATURES = (
    "day_return", "realized_variation", "log_range", "semivariance_balance",
    "taker_imbalance", "close_vwap_gap", "sign_entropy", "log_quote_volume",
    "log_trade_count",
)
SOURCE_DIR = Path("data/high_volatility_knn_analog_relay_sources_2020_2026")
PANEL = SOURCE_DIR / "daily_feature_panel.csv.gz"
TRAINING = SOURCE_DIR / "fit_analog_rows.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
MODEL = Path("results/high_volatility_knn_analog_relay_model_freeze_2026-08-09.json")
CLOCK = Path("data/high_volatility_knn_analog_relay_clocks_2023_2026.csv.gz")
RESULT = Path("results/high_volatility_knn_analog_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
LIVE_QUERY = """
SELECT ts,open,high,low,close,quote_asset_volume,taker_buy_quote,number_of_trades
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
FUNDING_QUERY = """
SELECT funding_time,funding_rate
FROM funding_rates_binance
WHERE symbol='BTCUSDT' AND funding_time>=:start AND funding_time<:end
ORDER BY funding_time
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def postgres_engine():
    from preprocessing.live_db_features import sqlalchemy_engine_from_env
    return sqlalchemy_engine_from_env(ENV_FILE)


def _five_minute_extension(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame.pop("ts"), utc=True)
    numeric = ("open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote", "number_of_trades")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.date.duplicated().any():
        raise RuntimeError("HVKAR live source has duplicate timestamps")
    frame = frame.sort_values("date").set_index("date")
    expected = pd.date_range(EXTENSION_START, END, freq="1min", inclusive="left")
    if not frame.index.equals(expected):
        raise RuntimeError("HVKAR live source is not an exact one-minute grid")
    valid = (
        np.isfinite(frame[list(numeric)]).all(axis=1)
        & frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & frame.high.ge(frame[["open", "close"]].max(axis=1))
        & frame.low.le(frame[["open", "close"]].min(axis=1))
        & frame.high.ge(frame.low)
        & frame[["quote_asset_volume", "taker_buy_quote", "number_of_trades"]].ge(0).all(axis=1)
        & frame.taker_buy_quote.le(frame.quote_asset_volume)
    )
    if not bool(valid.all()):
        raise RuntimeError("HVKAR live source validity drift")
    grouped = frame.resample("5min", origin="epoch", closed="left", label="left")
    bars = grouped.agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("quote_asset_volume", "sum"),
        quote_asset_volume=("quote_asset_volume", "sum"),
        taker_buy_quote=("taker_buy_quote", "sum"), number_of_trades=("number_of_trades", "sum"),
        source_rows=("open", "size"),
    ).reset_index()
    if not bars.source_rows.eq(5).all():
        raise RuntimeError("HVKAR live five-minute aggregation is incomplete")
    return bars.drop(columns="source_rows")


def load_market_and_funding() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    historical = pd.read_csv(
        prereg.SOURCE_PATH,
        usecols=["date", "open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote", "number_of_trades"],
    )
    historical["date"] = pd.to_datetime(historical.date, utc=True)
    from sqlalchemy import text
    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            raw = pd.read_sql_query(text(LIVE_QUERY), connection, params={"start": EXTENSION_START.to_pydatetime(), "end": END.to_pydatetime()})
            funding = pd.read_sql_query(text(FUNDING_QUERY), connection, params={"start": FIT_START.to_pydatetime(), "end": FIT_END.to_pydatetime()})
    finally:
        engine.dispose()
    live = _five_minute_extension(raw)
    market = pd.concat([historical, live], ignore_index=True, sort=False).sort_values("date").drop_duplicates("date", keep="last")
    market = market[market.date.lt(END)].reset_index(drop=True)
    if market.date.duplicated().any() or not market.date.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise RuntimeError("HVKAR combined market continuity drift")
    funding["funding_time"] = pd.to_datetime(funding.funding_time, utc=True)
    funding["funding_rate"] = pd.to_numeric(funding.funding_rate, errors="raise")
    if funding.funding_time.duplicated().any() or not np.isfinite(funding.funding_rate).all():
        raise RuntimeError("HVKAR fit funding source drift")
    return market, funding, {
        "historical_path": str(prereg.SOURCE_PATH), "historical_sha256": sha(prereg.SOURCE_PATH),
        "historical_rows": len(historical), "live_query_sha256": hashlib.sha256(LIVE_QUERY.encode()).hexdigest(),
        "funding_query_sha256": hashlib.sha256(FUNDING_QUERY.encode()).hexdigest(),
        "live_one_minute_rows": len(raw), "live_five_minute_rows": len(live), "combined_rows": len(market),
        "first": str(market.date.iloc[0]), "last": str(market.date.iloc[-1]),
    }


def _entropy(returns: np.ndarray) -> float:
    chunks = np.array_split(np.sign(returns), 8)
    values = []
    for chunk in chunks:
        positive = float(np.mean(chunk > 0))
        negative = float(np.mean(chunk < 0))
        zero = max(0.0, 1.0 - positive - negative)
        probabilities = np.asarray([positive, negative, zero])
        probabilities = probabilities[probabilities > 0]
        values.append(float(-(probabilities * np.log(probabilities)).sum() / math.log(3.0)))
    return float(np.mean(values))


def build_panel(market: pd.DataFrame) -> pd.DataFrame:
    frame = market.copy().set_index("date").sort_index()
    rows: list[dict[str, Any]] = []
    for day, window in frame.groupby(frame.index.floor("D"), sort=True):
        expected = pd.date_range(day, day + pd.Timedelta(days=1), freq="5min", inclusive="left")
        window = window.reindex(expected)
        numeric = window[["open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote", "number_of_trades"]]
        valid = len(window) == 288 and bool(np.isfinite(numeric).all(axis=1).all())
        values = {name: float("nan") for name in FEATURES}
        day_close = float("nan")
        if valid:
            prices = window[["open", "high", "low", "close"]]
            valid = bool(
                prices.gt(0).all(axis=1).all()
                and window.high.ge(window[["open", "close"]].max(axis=1)).all()
                and window.low.le(window[["open", "close"]].min(axis=1)).all()
                and window.high.ge(window.low).all()
                and window.quote_asset_volume.gt(0).sum() > 0
                and window.taker_buy_quote.ge(0).all()
                and window.taker_buy_quote.le(window.quote_asset_volume).all()
                and window.number_of_trades.ge(0).all()
            )
        if valid:
            returns = np.diff(np.log(window.close.to_numpy(float)))
            squared = returns * returns
            total_sq = float(squared.sum())
            upside = float(squared[returns > 0].sum())
            downside = float(squared[returns < 0].sum())
            quote = float(window.quote_asset_volume.sum())
            buy = float(window.taker_buy_quote.sum())
            typical = (window.high + window.low + window.close) / 3.0
            vwap = float((typical * window.quote_asset_volume).sum() / quote)
            day_close = float(window.close.iloc[-1])
            values = {
                "day_return": float(np.log(day_close / float(window.open.iloc[0]))),
                "realized_variation": math.sqrt(total_sq),
                "log_range": float(np.log(float(window.high.max()) / float(window.low.min()))),
                "semivariance_balance": (upside - downside) / total_sq if total_sq > 0 else 0.0,
                "taker_imbalance": (2.0 * buy / quote) - 1.0,
                "close_vwap_gap": float(np.log(day_close / vwap)),
                "sign_entropy": _entropy(returns),
                "log_quote_volume": math.log(quote),
                "log_trade_count": math.log1p(float(window.number_of_trades.sum())),
            }
        rows.append({"source_day": day, "decision_time": day + pd.Timedelta(days=1), "source_valid": valid, "day_close": day_close, **values})
    panel = pd.DataFrame(rows)
    panel["daily_close_return"] = np.log(panel.day_close / panel.day_close.shift(1))
    panel["rv20"] = panel.daily_close_return.rolling(20, min_periods=20).apply(lambda x: math.sqrt(365.0 * float(np.mean(np.square(x)))), raw=True)
    panel["rv20_threshold"] = panel.rv20.rolling(756, min_periods=756).quantile(0.90, interpolation="linear").shift(1)
    panel["high_volatility"] = panel.rv20.ge(panel.rv20_threshold)
    return panel


def fit_analogs(panel: pd.DataFrame, market: pd.DataFrame, funding: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    bars = market.set_index("date").sort_index()
    funding_times = pd.DatetimeIndex(funding.funding_time)
    funding_values = funding.funding_rate.to_numpy(float)
    fit = panel[panel.decision_time.ge(FIT_START) & panel.decision_time.lt(FIT_END)].copy()
    labels = []
    for decision in fit.decision_time:
        entry = decision + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=12)
        entry_open = float(bars.open.get(entry, np.nan))
        exit_open = float(bars.open.get(exit_, np.nan))
        mask = (funding_times >= entry) & (funding_times < exit_)
        label = 0.5 * (exit_open / entry_open - 1.0) - 0.0006 - 0.5 * float(funding_values[mask].sum()) if entry_open > 0 and exit_open > 0 and exit_ < FIT_END else np.nan
        labels.append(label)
    fit["label"] = labels
    finite = np.isfinite(fit[list(FEATURES) + ["label"]]).all(axis=1)
    fit = fit.loc[fit.source_valid & finite].copy()
    matrix = fit.loc[:, FEATURES].to_numpy(float)
    median = np.median(matrix, axis=0)
    q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0, method="linear")
    scale = np.maximum(q75 - q25, 1e-12)
    fit.loc[:, [f"z_{name}" for name in FEATURES]] = (matrix - median) / scale
    return fit, {"feature_order": list(FEATURES), "median": median.tolist(), "scale": scale.tolist(), "fit_rows": len(fit)}


def predict_clock(panel: pd.DataFrame, fit: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    z_columns = [f"z_{name}" for name in FEATURES]
    fit_matrix = fit[z_columns].to_numpy(float)
    fit_labels = fit.label.to_numpy(float)
    fit_times = pd.DatetimeIndex(fit.decision_time)
    median = np.asarray(model["median"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    rows = []
    for _, item in panel[panel.decision_time.ge(SPLITS["train"][0]) & panel.decision_time.lt(END)].iterrows():
        values = item.loc[list(FEATURES)].to_numpy(float)
        if not bool(item.source_valid and item.high_volatility and np.isfinite(values).all()):
            continue
        distances = np.sqrt(np.square(fit_matrix - ((values - median) / scale)).sum(axis=1))
        order = np.lexsort((fit_times.asi8, distances))[:21]
        prediction = float(np.median(fit_labels[order]))
        if prediction == 0 or not math.isfinite(prediction):
            continue
        decision = pd.Timestamp(item.decision_time)
        entry, exit_ = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=12, minutes=5)
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end), None)
        if split is None:
            continue
        rows.append({
            "candidate": prereg.POLICY_ID, "split": split, "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_,
            "side": 1 if prediction > 0 else -1, "rv20": float(item.rv20),
            "rv20_threshold": float(item.rv20_threshold), "neighbor_median_label": prediction,
            "nearest_distance": float(distances[order[0]]), "farthest_selected_distance": float(distances[order[-1]]),
        })
    return pd.DataFrame(rows)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = clock[clock.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    months = pd.to_datetime(subset.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    market, funding, source = load_market_and_funding()
    panel = build_panel(market)
    fit, model_values = fit_analogs(panel, market, funding)
    clock = predict_clock(panel, fit, model_values)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel.drop(columns=["daily_close_return"]), PANEL)
    _write_gzip_csv(fit[["decision_time", *FEATURES, *[f"z_{name}" for name in FEATURES], "label"]], TRAINING)
    _write_gzip_csv(clock, CLOCK)
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    source_core = {
        "protocol_version": "hvkar_12_source_and_fit_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]},
        "source": source, "fit_funding_rows": len(funding), "oos_postentry_returns_computed": False,
        "gross9_rows_opened": False, "panel": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel)},
        "training": {"path": str(TRAINING), "sha256": sha(TRAINING), "rows": len(fit)},
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    model_core = {
        "protocol_version": "hvkar_12_model_freeze_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": source_core["preregistration"],
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "training": source_core["training"], "neighbors": 21, "distance": "euclidean", "prediction": "median_label",
        **model_values, "refit_authorized": False, "oos_incidence_opened": False, "oos_outcomes_opened": False,
    }
    model = {**model_core, "manifest_hash": canonical_hash(model_core)}
    MODEL.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
    support = {name: stats(clock, name) for name in SPLITS}
    checks = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvkar_12_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": source_core["preregistration"],
        "source_manifest": model_core["source_manifest"],
        "model_freeze": {"path": str(MODEL), "sha256": sha(MODEL), "manifest_hash": model["manifest_hash"]},
        "completed_preentry_sources_opened": True, "fit_period_labels_opened": True,
        "oos_postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(clock)},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
