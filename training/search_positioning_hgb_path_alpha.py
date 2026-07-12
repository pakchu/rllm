"""Risk-aware nonlinear BTC alpha from delayed Binance positioning metrics.

Two histogram-gradient-boosting regressors predict executable long and short
path utility.  The model can learn interactions between price regime, OI,
volume/taker flow, macro context, and futures positioning disagreement.  The
experiment writes a 2023 Top-10 manifest before any 2024+ feature, prediction,
target, or metric is computed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from training.search_positioning_disagreement_alpha import (
    WINDOWS,
    _attach_delayed_metrics,
    _future_extreme,
    _simulate_no_stop,
    _window_mask,
    build_positioning_features,
)


FIT_START = "2020-10-15"
FIT_END = "2023-01-01"
SELECTION_END = "2024-01-01"
ANCHOR_STRIDE = 6
WINDOWS["model_fit"] = (FIT_START, FIT_END)


@dataclass(frozen=True)
class HgbPathSpec:
    name: str
    hold_bars: int
    risk_lambda: float
    loss: str
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float


MODEL_SPECS = (
    HgbPathSpec("hgb_h144_mean_lam05", 144, 0.5, "squared_error", 15, 120, 3.0),
    HgbPathSpec("hgb_h144_robust_lam10", 144, 1.0, "absolute_error", 7, 180, 10.0),
    HgbPathSpec("hgb_h288_mean_lam05", 288, 0.5, "squared_error", 15, 120, 3.0),
    HgbPathSpec("hgb_h288_robust_lam10", 288, 1.0, "absolute_error", 7, 180, 10.0),
    HgbPathSpec("hgb_h576_mean_lam05", 576, 0.5, "squared_error", 15, 120, 3.0),
    HgbPathSpec("hgb_h576_robust_lam10", 576, 1.0, "absolute_error", 7, 180, 10.0),
)


@dataclass(frozen=True)
class PositioningHgbConfig:
    input_csv: str
    metrics_csv: str
    output: str
    manifest_output: str
    top_n: int = 10
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    metrics_tolerance: str = "10min"
    source_delay_bars: int = 1
    min_select_trades: int = 24
    min_half_trades: int = 8
    random_state: int = 42


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    minimum = max(24, window // 2)
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
    return (series - mean) / std


def build_model_features(market: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(market["close"], errors="coerce")
    log_close = np.log(close.where(close > 0.0))
    high = pd.to_numeric(market["high"], errors="coerce")
    low = pd.to_numeric(market["low"], errors="coerce")
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    taker_imbalance = (2.0 * taker_buy / quote.replace(0.0, np.nan) - 1.0).clip(-1.0, 1.0)
    open_interest = np.log(pd.to_numeric(market["sum_open_interest"], errors="coerce").where(lambda x: x > 0.0))

    columns: dict[str, pd.Series | np.ndarray] = {}
    for window in (12, 48, 144, 288, 2016, 8640, 25920):
        returns = log_close - log_close.shift(window)
        columns[f"price_return_{window}"] = returns
        columns[f"price_return_z_{window}"] = _rolling_z(returns, max(288, min(8640, window * 4)))
        rolling_high = high.rolling(window, min_periods=max(12, window // 2)).max()
        rolling_low = low.rolling(window, min_periods=max(12, window // 2)).min()
        columns[f"range_position_{window}"] = (close - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan) - 0.5
    one_bar_return = log_close.diff()
    for window in (48, 288, 2016, 8640):
        columns[f"realized_vol_{window}"] = one_bar_return.rolling(window, min_periods=max(24, window // 2)).std(ddof=0)
        columns[f"quote_volume_z_{window}"] = _rolling_z(np.log1p(quote), window)
    for window in (12, 48, 144, 288):
        columns[f"taker_imbalance_{window}"] = taker_imbalance.rolling(window, min_periods=window).mean()
        oi_return = open_interest - open_interest.shift(window)
        columns[f"oi_return_{window}"] = oi_return
        columns[f"oi_price_divergence_{window}"] = oi_return - (log_close - log_close.shift(window))

    dates = pd.to_datetime(market["date"])
    minute_of_day = dates.dt.hour * 60 + dates.dt.minute
    columns["time_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    columns["time_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    columns["week_sin"] = np.sin(2.0 * np.pi * dates.dt.dayofweek / 7.0)
    columns["week_cos"] = np.cos(2.0 * np.pi * dates.dt.dayofweek / 7.0)
    columns["positioning_available"] = pd.to_numeric(market["positioning_available"], errors="coerce")
    for source in (
        "dxy_zscore",
        "dxy_momentum",
        "kimchi_premium_zscore",
        "kimchi_premium_change",
        "usdkrw_zscore",
        "usdkrw_momentum",
    ):
        if source in market.columns:
            columns[source] = pd.to_numeric(market[source], errors="coerce")

    positioning = build_positioning_features(market)
    frame = pd.concat([pd.DataFrame(columns, index=market.index), positioning], axis=1)
    return frame.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def _feature_hash(features: pd.DataFrame, dates: pd.Series | None = None) -> str:
    values = features.to_numpy(dtype="<f4", copy=True)
    values = np.nan_to_num(values, nan=3.4028233e38, posinf=3.4028231e38, neginf=-3.4028231e38)
    digest = hashlib.sha256()
    digest.update("\n".join(features.columns).encode("utf-8"))
    if dates is not None:
        timestamps = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]").astype("<i8", copy=False)
        digest.update(timestamps.tobytes(order="C"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def executable_path_utilities(
    market: pd.DataFrame,
    positions: np.ndarray,
    *,
    hold_bars: int,
    risk_lambda: float,
    leverage: float,
    side_cost: float,
    available_before_position: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    opens = market["open"].to_numpy(float)
    future_low = _future_extreme(market["low"].to_numpy(float), hold_bars, "min")
    future_high = _future_extreme(market["high"].to_numpy(float), hold_bars, "max")
    utilities = np.full((len(positions), 2), np.nan, dtype=np.float32)
    for row, position in enumerate(positions):
        entry_position = int(position) + 1
        exit_position = entry_position + int(hold_bars)
        if exit_position >= min(len(market), int(available_before_position)):
            continue
        entry = opens[entry_position]
        exit_price = opens[exit_position]
        if not np.isfinite(entry) or not np.isfinite(exit_price) or entry <= 0.0:
            continue
        simple_return = exit_price / entry - 1.0
        long_mae = max(0.0, 1.0 - future_low[entry_position] / entry)
        short_mae = max(0.0, future_high[entry_position] / entry - 1.0)
        long_net = (1.0 - side_cost) * (1.0 + leverage * simple_return) * (1.0 - side_cost) - 1.0
        short_net = (1.0 - side_cost) * (1.0 - leverage * simple_return) * (1.0 - side_cost) - 1.0
        utilities[row, 0] = long_net - float(risk_lambda) * leverage * long_mae
        utilities[row, 1] = short_net - float(risk_lambda) * leverage * short_mae
    return utilities


def _fit_model(x: np.ndarray, y: np.ndarray, spec: HgbPathSpec, random_state: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss=spec.loss,
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=spec.max_leaf_nodes,
        min_samples_leaf=spec.min_samples_leaf,
        l2_regularization=spec.l2_regularization,
        max_bins=127,
        early_stopping=False,
        random_state=random_state,
    )
    model.fit(x, y)
    return model


def fit_path_models(
    market: pd.DataFrame,
    features: pd.DataFrame,
    spec: HgbPathSpec,
    *,
    cfg: PositioningHgbConfig,
) -> tuple[tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor], dict[str, Any]]:
    dates = pd.to_datetime(market["date"])
    positions = np.arange(0, len(market), ANCHOR_STRIDE, dtype=np.int64)
    fit_end_position = int(np.searchsorted(dates.to_numpy(), np.datetime64(pd.Timestamp(FIT_END)), side="left"))
    utilities = executable_path_utilities(
        market,
        positions,
        hold_bars=spec.hold_bars,
        risk_lambda=spec.risk_lambda,
        leverage=cfg.leverage,
        side_cost=(cfg.fee_rate + cfg.slippage_rate) * cfg.leverage,
        available_before_position=fit_end_position,
    )
    signal_dates = dates.iloc[positions].to_numpy()
    fit_mask = (
        (signal_dates >= np.datetime64(pd.Timestamp(FIT_START)))
        & (signal_dates < np.datetime64(pd.Timestamp(FIT_END)))
        & np.isfinite(utilities).all(axis=1)
    )
    fit_positions = positions[fit_mask]
    x = features.iloc[fit_positions].to_numpy(np.float32)
    y = utilities[fit_mask]
    lower = np.nanquantile(y, 0.005, axis=0)
    upper = np.nanquantile(y, 0.995, axis=0)
    y = np.clip(y, lower, upper)
    long_model = _fit_model(x, y[:, 0], spec, cfg.random_state)
    short_model = _fit_model(x, y[:, 1], spec, cfg.random_state + 1)
    return (long_model, short_model), {
        "fit_rows": int(len(x)),
        "fit_start": FIT_START,
        "fit_end": FIT_END,
        "anchor_stride": ANCHOR_STRIDE,
        "target_clip_0p5_99p5": {"lower": lower.tolist(), "upper": upper.tolist()},
        "feature_count": int(features.shape[1]),
        "model": asdict(spec),
    }


def predict_utilities(
    models: tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor],
    features: pd.DataFrame,
    *,
    chunk_size: int = 100_000,
) -> np.ndarray:
    output = np.full((len(features), 2), np.nan, dtype=np.float32)
    for start in range(0, len(features), chunk_size):
        stop = min(len(features), start + chunk_size)
        x = features.iloc[start:stop].to_numpy(np.float32)
        output[start:stop, 0] = models[0].predict(x).astype(np.float32)
        output[start:stop, 1] = models[1].predict(x).astype(np.float32)
    return output


def causal_rolling_threshold(scores: np.ndarray, *, window_bars: int, quantile: float) -> np.ndarray:
    series = pd.Series(np.asarray(scores, dtype=float)).shift(1)
    minimum = min(window_bars, max(2, window_bars // 4))
    return series.rolling(window_bars, min_periods=minimum).quantile(quantile).to_numpy(float)


def _policy_masks(
    predictions: np.ndarray,
    threshold: np.ndarray,
    *,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    long_score = predictions[:, 0]
    short_score = predictions[:, 1]
    confidence = np.maximum(long_score, short_score)
    eligible = np.isfinite(confidence) & np.isfinite(threshold) & (confidence >= threshold) & (confidence > 0.0)
    long_active = eligible & (long_score > short_score) & (side in {"long", "both"})
    short_active = eligible & (short_score > long_score) & (side in {"short", "both"})
    return long_active, short_active


def _prediction_hash(predictions: np.ndarray) -> str:
    values = np.nan_to_num(np.asarray(predictions, dtype="<f4"), nan=3.4028233e38)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _read_before(path: str, date_column: str, cutoff: str) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    boundary = pd.Timestamp(cutoff)
    for chunk in pd.read_csv(path, compression="infer", chunksize=100_000):
        timestamps = pd.to_datetime(chunk[date_column], utc=True, errors="raise", format="mixed").dt.tz_convert(None)
        keep = timestamps < boundary
        if keep.any():
            chunks.append(chunk.loc[keep].copy())
        if (~keep).any():
            break
    if not chunks:
        raise ValueError(f"no rows before {cutoff} in {path}")
    return pd.concat(chunks, ignore_index=True)


def _load_attached(cfg: PositioningHgbConfig, *, cutoff: str | None = None) -> pd.DataFrame:
    if cutoff is None:
        market = pd.read_csv(cfg.input_csv, compression="infer")
        metrics = pd.read_csv(cfg.metrics_csv, compression="infer")
    else:
        market = _read_before(cfg.input_csv, "date", cutoff)
        metrics = _read_before(cfg.metrics_csv, "create_time", cutoff)
    return _attach_delayed_metrics(
        market,
        metrics,
        tolerance=cfg.metrics_tolerance,
        delay_bars=cfg.source_delay_bars,
    )


def run(cfg: PositioningHgbConfig) -> dict[str, Any]:
    # Phase 1: the process is physically limited to pre-2024 rows.
    phase1_market = _load_attached(cfg, cutoff=SELECTION_END)
    phase1_dates = pd.to_datetime(phase1_market["date"])
    phase1_features = build_model_features(phase1_market)
    phase1_feature_hash = _feature_hash(phase1_features, phase1_dates)

    trained: dict[str, tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor]] = {}
    training_meta: dict[str, Any] = {}
    phase1_predictions: dict[str, np.ndarray] = {}
    candidates: list[dict[str, Any]] = []
    extremes_phase1 = {
        spec.hold_bars: (
            _future_extreme(phase1_market["low"].to_numpy(float), spec.hold_bars, "min"),
            _future_extreme(phase1_market["high"].to_numpy(float), spec.hold_bars, "max"),
        )
        for spec in MODEL_SPECS
    }
    for spec in MODEL_SPECS:
        models, meta = fit_path_models(phase1_market, phase1_features, spec, cfg=cfg)
        predictions = predict_utilities(models, phase1_features)
        trained[spec.name] = models
        training_meta[spec.name] = meta
        phase1_predictions[spec.name] = predictions
        confidence = np.nanmax(predictions, axis=1)
        for history_days in (90, 180, 365):
            history_bars = history_days * 288
            for quantile in (0.80, 0.90, 0.95):
                threshold = causal_rolling_threshold(confidence, window_bars=history_bars, quantile=quantile)
                for side in ("both", "long", "short"):
                    long_active, short_active = _policy_masks(predictions, threshold, side=side)
                    for stride_bars in (6, 12, 24):
                        kwargs = {
                            "hold_bars": spec.hold_bars,
                            "stride_bars": stride_bars,
                            "leverage": cfg.leverage,
                            "fee_rate": cfg.fee_rate,
                            "slippage_rate": cfg.slippage_rate,
                            "extremes": extremes_phase1[spec.hold_bars],
                        }
                        selection = _simulate_no_stop(
                            phase1_market,
                            phase1_dates,
                            long_active,
                            short_active,
                            window="select2023",
                            **kwargs,
                        )
                        halves = [
                            _simulate_no_stop(
                                phase1_market,
                                phase1_dates,
                                long_active,
                                short_active,
                                window=name,
                                **kwargs,
                            )
                            for name in ("select2023_h1", "select2023_h2")
                        ]
                        if selection["trades"] < cfg.min_select_trades or min(half["trades"] for half in halves) < cfg.min_half_trades:
                            continue
                        candidates.append(
                            {
                                "model": spec.name,
                                "hold_bars": spec.hold_bars,
                                "history_days": history_days,
                                "score_quantile": quantile,
                                "side": side,
                                "stride_bars": stride_bars,
                                "select2023": selection,
                                "select2023_halves": halves,
                                "selection_score": {
                                    "positive_halves": sum(half["return_pct"] > 0.0 for half in halves),
                                    "min_half_ratio": min(half["ratio"] for half in halves),
                                    "full_ratio": selection["ratio"],
                                },
                            }
                        )
    candidates.sort(
        key=lambda row: (
            row["selection_score"]["positive_halves"],
            row["selection_score"]["min_half_ratio"],
            row["selection_score"]["full_ratio"],
            row["select2023"]["return_pct"],
        ),
        reverse=True,
    )
    selected = candidates[: cfg.top_n]
    manifest_payload = [
        {key: row[key] for key in ("model", "hold_bars", "history_days", "score_quantile", "side", "stride_bars")}
        for row in selected
    ]
    manifest_json = json.dumps(manifest_payload, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "pre_future_manifest",
        "selection_end": SELECTION_END,
        "feature_hash": phase1_feature_hash,
        "manifest_hash": manifest_hash,
        "policies": manifest_payload,
        "training": training_meta,
        "phase1_prediction_hashes": {name: _prediction_hash(values) for name, values in phase1_predictions.items()},
    }
    manifest_path = Path(cfg.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # Phase 2 starts only after the manifest exists.
    if not manifest_path.exists():  # pragma: no cover
        raise RuntimeError("Top-10 manifest was not persisted before phase 2")
    full_attached = _load_attached(cfg)
    full_dates = pd.to_datetime(full_attached["date"])
    full_features = build_model_features(full_attached)
    full_prefix_dates = full_dates.iloc[: len(phase1_dates)].reset_index(drop=True)
    if not full_prefix_dates.equals(phase1_dates.reset_index(drop=True)):
        raise RuntimeError("pre-2024 timestamp prefix changed after future rows were admitted")
    if _feature_hash(full_features.iloc[: len(phase1_features)], full_prefix_dates) != phase1_feature_hash:
        raise RuntimeError("pre-2024 feature prefix changed after future rows were admitted")
    extremes_full = {
        spec.hold_bars: (
            _future_extreme(full_attached["low"].to_numpy(float), spec.hold_bars, "min"),
            _future_extreme(full_attached["high"].to_numpy(float), spec.hold_bars, "max"),
        )
        for spec in MODEL_SPECS
    }
    full_predictions: dict[str, np.ndarray] = {}
    for model_name in {row["model"] for row in selected}:
        values = predict_utilities(trained[model_name], full_features)
        if _prediction_hash(values[: len(phase1_features)]) != manifest["phase1_prediction_hashes"][model_name]:
            raise RuntimeError(f"pre-2024 prediction prefix changed for {model_name}")
        full_predictions[model_name] = values
    for row in selected:
        predictions = full_predictions[row["model"]]
        confidence = np.nanmax(predictions, axis=1)
        threshold = causal_rolling_threshold(
            confidence,
            window_bars=row["history_days"] * 288,
            quantile=row["score_quantile"],
        )
        long_active, short_active = _policy_masks(predictions, threshold, side=row["side"])
        for window in ("model_fit", "test2024", "eval2025", "ytd2026"):
            output_key = "train_in_sample" if window == "model_fit" else window
            row[output_key] = _simulate_no_stop(
                full_attached,
                full_dates,
                long_active,
                short_active,
                window=window,
                hold_bars=row["hold_bars"],
                stride_bars=row["stride_bars"],
                leverage=cfg.leverage,
                fee_rate=cfg.fee_rate,
                slippage_rate=cfg.slippage_rate,
                extremes=extremes_full[row["hold_bars"]],
            )
        row["passes_alpha_target"] = (
            row["test2024"]["ratio"] >= 3.0
            and row["eval2025"]["ratio"] >= 3.0
            and row["test2024"]["trades"] >= 16
            and row["eval2025"]["trades"] >= 16
        )
        row["passes_live_target"] = (
            row["passes_alpha_target"]
            and row["ytd2026"]["ratio"] >= 5.0
            and row["ytd2026"]["trades"] >= 8
            and row["ytd2026"]["return_pct"] > 0.0
        )
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "fit": (FIT_START, FIT_END),
            "selection": WINDOWS["select2023"],
            "manifest_written_before_future": True,
            "future_windows": {name: WINDOWS[name] for name in ("test2024", "eval2025", "ytd2026")},
            "source_delay_bars": cfg.source_delay_bars,
            "target": "account-level next-open return after two side costs minus lambda * leveraged side-specific intratrade MAE",
            "execution": "next-bar-open entry; fixed hold; full-window CAGR; strict intraposition MDD",
        },
        "input": {
            "rows": int(len(full_attached)),
            "features": int(full_features.shape[1]),
            "start": str(full_dates.min()),
            "end": str(full_dates.max()),
        },
        "tested": int(len(candidates)),
        "manifest_hash": manifest_hash,
        "selected": selected,
        "alpha_qualifiers": [row for row in selected if row["passes_alpha_target"]],
        "live_qualifiers": [row for row in selected if row["passes_live_target"]],
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> PositioningHgbConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--top-n", type=int, default=PositioningHgbConfig.top_n)
    parser.add_argument("--leverage", type=float, default=PositioningHgbConfig.leverage)
    parser.add_argument("--fee-rate", type=float, default=PositioningHgbConfig.fee_rate)
    parser.add_argument("--slippage-rate", type=float, default=PositioningHgbConfig.slippage_rate)
    parser.add_argument("--metrics-tolerance", default=PositioningHgbConfig.metrics_tolerance)
    parser.add_argument("--source-delay-bars", type=int, default=PositioningHgbConfig.source_delay_bars)
    parser.add_argument("--min-select-trades", type=int, default=PositioningHgbConfig.min_select_trades)
    parser.add_argument("--min-half-trades", type=int, default=PositioningHgbConfig.min_half_trades)
    parser.add_argument("--random-state", type=int, default=PositioningHgbConfig.random_state)
    return PositioningHgbConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "tested": report["tested"],
                "manifest_hash": report["manifest_hash"],
                "alpha_qualifiers": len(report["alpha_qualifiers"]),
                "live_qualifiers": len(report["live_qualifiers"]),
                "selected": report["selected"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
