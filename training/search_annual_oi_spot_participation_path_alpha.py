"""Pre-2025 annual search for the preregistered OI/spot participation alpha.

AOSP-1 fits separate low-complexity HGB regressors for executable long and
short path utility.  Every prediction is annual walk-forward: the 2023 model
uses targets whose exits precede 2023, and the 2024 model uses targets whose
exits precede 2024.  The selection process physically truncates both market and
spot sources before 2025-01-01 and cannot open later rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from training.search_positioning_disagreement_alpha import (
    _future_extreme,
    _simulate_no_stop,
)
from training.search_positioning_hgb_path_alpha import _read_before


PREREGISTRATION = Path(
    "results/annual_oi_spot_participation_path_preregistration_2026-07-28.json"
)
OUTPUT = Path(
    "results/annual_oi_spot_participation_path_pre2025_selection_2026-07-28.json"
)
DOCS = Path(
    "docs/annual-oi-spot-participation-path-pre2025-selection-2026-07-28.md"
)
SELECTION_CUTOFF = "2025-01-01"
FIT_START = pd.Timestamp("2020-10-15")
FOLDS = {
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
}
WINDOWS = {
    "2023": FOLDS["2023"],
    "2023_h1": ("2023-01-01", "2023-07-01"),
    "2023_h2": ("2023-07-01", "2024-01-01"),
    "2024": FOLDS["2024"],
    "2024_h1": ("2024-01-01", "2024-07-01"),
    "2024_h2": ("2024-07-01", "2025-01-01"),
}
FEATURE_COLUMNS = (
    "spot_perp_basis",
    "spot_perp_return_gap_12",
    "basis_change_12",
    "spot_participation_12",
    "oi_return_12",
    "oi_perp_divergence_12",
    "oi_spot_divergence_12",
    "spot_perp_return_gap_48",
    "basis_change_48",
    "spot_participation_48",
    "oi_return_48",
    "oi_perp_divergence_48",
    "oi_spot_divergence_48",
    "spot_perp_return_gap_288",
    "basis_change_288",
    "spot_participation_288",
    "oi_return_288",
    "oi_perp_divergence_288",
    "oi_spot_divergence_288",
    "spot_notional_z_2016",
    "perp_notional_z_2016",
    "spot_notional_z_8640",
    "perp_notional_z_8640",
    "spot_participation_acceleration",
    "oi_value_to_spot_notional_24h",
    "perp_taker_imbalance_12",
    "perp_taker_imbalance_48",
    "perp_realized_vol_48",
    "perp_realized_vol_288",
    "perp_range_position_144",
    "perp_range_position_2016",
    "time_sin",
    "time_cos",
    "week_sin",
    "week_cos",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hold_bars: int
    risk_lambda: float
    loss: str
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float


@dataclass(frozen=True)
class Config:
    market_csv: str = (
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
    )
    spot_csv: str = (
        "data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz"
    )
    preregistration: str = str(PREREGISTRATION)
    output: str = str(OUTPUT)
    docs_output: str = str(DOCS)
    leverage: float = 0.5
    cost_rate: float = 0.0006
    learning_rate: float = 0.05
    max_iter: int = 120
    max_bins: int = 127
    random_state: int = 715


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_preregistration(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("name") != "annual_oi_spot_participation_path":
        raise RuntimeError("unexpected preregistration name")
    if payload.get("status") != "preregistered_before_outcome_scan":
        raise RuntimeError("preregistration is not outcome-blind")
    grid = payload["frozen_grid"]
    count = (
        len(grid["model_specs"])
        * len(grid["score_quantile"])
        * len(grid["allowed_side"])
    )
    if count != grid.get("model_policy_cells") or count != 24:
        raise RuntimeError("frozen policy grid drifted")
    if (
        payload["selection_contract"]["future_veto"].get("future_can_rerank")
        is not False
    ):
        raise RuntimeError("future reranking is not disabled")
    return payload


def model_specs(preregistration: dict[str, Any]) -> list[ModelSpec]:
    return [
        ModelSpec(**row)
        for row in preregistration["frozen_grid"]["model_specs"]
    ]


def load_sources(cfg: Config, *, cutoff: str = SELECTION_CUTOFF) -> pd.DataFrame:
    market = _read_before(cfg.market_csv, "date", cutoff)
    spot = _read_before(cfg.spot_csv, "date", cutoff)
    for frame in (market, spot):
        frame["date"] = pd.to_datetime(
            frame["date"], utc=True, errors="raise", format="mixed"
        ).dt.tz_convert(None)
    market = (
        market.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    spot = (
        spot.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    columns = ["date", "spot_close", "spot_volume", "spot_rows"]
    merged = market.merge(spot[columns], on="date", how="left", validate="one_to_one")
    dates = pd.to_datetime(merged["date"])
    if len(dates) and dates.max() >= pd.Timestamp(cutoff):
        raise RuntimeError("selection source was not physically truncated")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise RuntimeError("market source is not a complete five-minute grid")
    return merged.reset_index(drop=True)


def prior_z(values: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    prior = numeric.shift(1)
    minimum = max(288, window // 2)
    mean = prior.rolling(window, min_periods=minimum).mean()
    std = prior.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
    return ((numeric - mean) / std).clip(-20.0, 20.0)


def build_features(market: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the frozen causal 35-column feature matrix."""
    perp_close = pd.to_numeric(market["close"], errors="coerce")
    spot_close = pd.to_numeric(market["spot_close"], errors="coerce")
    log_perp = np.log(perp_close.where(perp_close > 0.0))
    log_spot = np.log(spot_close.where(spot_close > 0.0))
    basis = log_spot - log_perp
    perp_notional = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    spot_volume = pd.to_numeric(market["spot_volume"], errors="coerce")
    spot_notional = spot_close * spot_volume
    observed_oi = pd.to_numeric(market["open_interest"], errors="coerce").shift(1)
    observed_oi_value = pd.to_numeric(
        market["open_interest_value"], errors="coerce"
    ).shift(1)
    log_oi = np.log(observed_oi.where(observed_oi > 0.0))
    quote = perp_notional.replace(0.0, np.nan)
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    taker_imbalance = (2.0 * taker_buy / quote - 1.0).clip(-1.0, 1.0)
    columns: dict[str, pd.Series | np.ndarray] = {"spot_perp_basis": basis}
    participation: dict[int, pd.Series] = {}
    for horizon in (12, 48, 288):
        perp_return = log_perp - log_perp.shift(horizon)
        spot_return = log_spot - log_spot.shift(horizon)
        oi_return = log_oi - log_oi.shift(horizon)
        spot_sum = spot_notional.rolling(horizon, min_periods=horizon).sum()
        perp_sum = perp_notional.rolling(horizon, min_periods=horizon).sum()
        participation[horizon] = np.log(
            spot_sum.where(spot_sum > 0.0) / perp_sum.where(perp_sum > 0.0)
        )
        columns[f"spot_perp_return_gap_{horizon}"] = spot_return - perp_return
        columns[f"basis_change_{horizon}"] = basis - basis.shift(horizon)
        columns[f"spot_participation_{horizon}"] = participation[horizon]
        columns[f"oi_return_{horizon}"] = oi_return
        columns[f"oi_perp_divergence_{horizon}"] = oi_return - perp_return
        columns[f"oi_spot_divergence_{horizon}"] = oi_return - spot_return
    columns["spot_notional_z_2016"] = prior_z(np.log1p(spot_notional), 2016)
    columns["perp_notional_z_2016"] = prior_z(np.log1p(perp_notional), 2016)
    columns["spot_notional_z_8640"] = prior_z(np.log1p(spot_notional), 8640)
    columns["perp_notional_z_8640"] = prior_z(np.log1p(perp_notional), 8640)
    columns["spot_participation_acceleration"] = (
        participation[12] - participation[288]
    )
    spot_notional_24h = spot_notional.rolling(288, min_periods=288).sum()
    columns["oi_value_to_spot_notional_24h"] = np.log(
        observed_oi_value.where(observed_oi_value > 0.0)
        / spot_notional_24h.where(spot_notional_24h > 0.0)
    )
    columns["perp_taker_imbalance_12"] = taker_imbalance.rolling(
        12, min_periods=12
    ).mean()
    columns["perp_taker_imbalance_48"] = taker_imbalance.rolling(
        48, min_periods=48
    ).mean()
    log_return = log_perp.diff()
    columns["perp_realized_vol_48"] = log_return.rolling(
        48, min_periods=48
    ).std(ddof=0)
    columns["perp_realized_vol_288"] = log_return.rolling(
        288, min_periods=288
    ).std(ddof=0)
    high = pd.to_numeric(market["high"], errors="coerce")
    low = pd.to_numeric(market["low"], errors="coerce")
    for horizon in (144, 2016):
        rolling_high = high.rolling(horizon, min_periods=horizon).max()
        rolling_low = low.rolling(horizon, min_periods=horizon).min()
        columns[f"perp_range_position_{horizon}"] = (
            (perp_close - rolling_low)
            / (rolling_high - rolling_low).replace(0.0, np.nan)
            - 0.5
        )
    dates = pd.to_datetime(market["date"])
    minute_of_day = dates.dt.hour * 60 + dates.dt.minute
    columns["time_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    columns["time_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    columns["week_sin"] = np.sin(2.0 * np.pi * dates.dt.dayofweek / 7.0)
    columns["week_cos"] = np.cos(2.0 * np.pi * dates.dt.dayofweek / 7.0)
    frame = pd.DataFrame(columns, index=market.index).loc[:, FEATURE_COLUMNS]
    source_valid = (
        pd.to_numeric(market["spot_rows"], errors="coerce").eq(5)
        & spot_close.gt(0.0)
        & spot_volume.gt(0.0)
        & observed_oi.gt(0.0)
        & observed_oi_value.gt(0.0)
    ).to_numpy(bool)
    frame = frame.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    source_valid &= np.isfinite(frame.to_numpy(float)).all(axis=1)
    return frame, source_valid


def decision_mask(market: pd.DataFrame, source_valid: np.ndarray) -> np.ndarray:
    dates = pd.to_datetime(market["date"])
    return (
        (dates.dt.minute.to_numpy(int) == 0)
        & np.asarray(source_valid, dtype=bool)
    )


def executable_utilities(
    market: pd.DataFrame,
    positions: np.ndarray,
    *,
    spec: ModelSpec,
    leverage: float,
    cost_rate: float,
    available_before_position: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.int64)
    opens = pd.to_numeric(market["open"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(market["low"], errors="coerce").to_numpy(float)
    highs = pd.to_numeric(market["high"], errors="coerce").to_numpy(float)
    future_low = _future_extreme(lows, int(spec.hold_bars), "min")
    future_high = _future_extreme(highs, int(spec.hold_bars), "max")
    utilities = np.full((len(positions), 2), np.nan, dtype=np.float32)
    exits = np.full(len(positions), len(market), dtype=np.int64)
    side_cost = float(cost_rate) * float(leverage)
    for index, raw_position in enumerate(positions):
        position = int(raw_position)
        entry = position + 1
        exit_position = entry + int(spec.hold_bars)
        exits[index] = exit_position
        if exit_position >= min(len(market), int(available_before_position)):
            continue
        entry_price = float(opens[entry])
        exit_price = float(opens[exit_position])
        if not (
            np.isfinite(entry_price)
            and np.isfinite(exit_price)
            and entry_price > 0.0
        ):
            continue
        simple_return = exit_price / entry_price - 1.0
        long_mae = max(0.0, 1.0 - float(future_low[entry]) / entry_price)
        short_mae = max(0.0, float(future_high[entry]) / entry_price - 1.0)
        long_net = (
            (1.0 - side_cost)
            * (1.0 + float(leverage) * simple_return)
            * (1.0 - side_cost)
            - 1.0
        )
        short_net = (
            (1.0 - side_cost)
            * (1.0 - float(leverage) * simple_return)
            * (1.0 - side_cost)
            - 1.0
        )
        utilities[index, 0] = (
            long_net
            - float(spec.risk_lambda) * float(leverage) * long_mae
        )
        utilities[index, 1] = (
            short_net
            - float(spec.risk_lambda) * float(leverage) * short_mae
        )
    return utilities, exits


def fold_fit_mask(
    dates: pd.Series,
    positions: np.ndarray,
    exits: np.ndarray,
    utilities: np.ndarray,
    *,
    fit_end: str,
) -> np.ndarray:
    cutoff = pd.Timestamp(fit_end)
    signal_dates = dates.iloc[positions].to_numpy(dtype="datetime64[ns]")
    exit_dates = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_exit = exits < len(dates)
    exit_dates[valid_exit] = dates.iloc[exits[valid_exit]].to_numpy(
        dtype="datetime64[ns]"
    )
    return np.asarray(
        (signal_dates >= np.datetime64(FIT_START))
        & (signal_dates < np.datetime64(cutoff))
        & valid_exit
        & (exit_dates < np.datetime64(cutoff))
        & np.isfinite(utilities).all(axis=1),
        dtype=bool,
    )


def _fit_one(
    x: np.ndarray,
    y: np.ndarray,
    spec: ModelSpec,
    cfg: Config,
    *,
    seed_offset: int,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss=spec.loss,
        learning_rate=float(cfg.learning_rate),
        max_iter=int(cfg.max_iter),
        max_leaf_nodes=int(spec.max_leaf_nodes),
        min_samples_leaf=int(spec.min_samples_leaf),
        l2_regularization=float(spec.l2_regularization),
        max_bins=int(cfg.max_bins),
        early_stopping=False,
        random_state=int(cfg.random_state) + int(seed_offset),
    )
    model.fit(x, y)
    return model


def fit_fold(
    market: pd.DataFrame,
    features: pd.DataFrame,
    valid_decisions: np.ndarray,
    spec: ModelSpec,
    *,
    fit_end: str,
    predict_start: str,
    predict_end: str,
    cfg: Config,
) -> dict[str, Any]:
    dates = pd.to_datetime(market["date"])
    positions = np.flatnonzero(valid_decisions)
    cutoff_position = int(
        np.searchsorted(
            dates.to_numpy(dtype="datetime64[ns]"),
            np.datetime64(pd.Timestamp(fit_end)),
            side="left",
        )
    )
    utilities, exits = executable_utilities(
        market,
        positions,
        spec=spec,
        leverage=cfg.leverage,
        cost_rate=cfg.cost_rate,
        available_before_position=cutoff_position,
    )
    fit = fold_fit_mask(
        dates, positions, exits, utilities, fit_end=fit_end
    )
    fit_positions = positions[fit]
    if len(fit_positions) < 2_000:
        raise RuntimeError(
            f"insufficient fit decisions for {spec.name}/{fit_end}: {len(fit_positions)}"
        )
    x_fit = features.iloc[fit_positions].to_numpy(np.float32)
    y_fit = utilities[fit].astype(np.float64)
    lower = np.quantile(y_fit, 0.005, axis=0)
    upper = np.quantile(y_fit, 0.995, axis=0)
    y_fit = np.clip(y_fit, lower, upper)
    models = (
        _fit_one(x_fit, y_fit[:, 0], spec, cfg, seed_offset=0),
        _fit_one(x_fit, y_fit[:, 1], spec, cfg, seed_offset=1),
    )
    fit_predictions = np.column_stack(
        [models[0].predict(x_fit), models[1].predict(x_fit)]
    )
    predict_mask = (
        (dates >= pd.Timestamp(predict_start))
        & (dates < pd.Timestamp(predict_end))
    ).to_numpy(bool)
    predict_positions = np.flatnonzero(valid_decisions & predict_mask)
    x_predict = features.iloc[predict_positions].to_numpy(np.float32)
    predictions = np.column_stack(
        [models[0].predict(x_predict), models[1].predict(x_predict)]
    )
    return {
        "fit_positions": fit_positions,
        "fit_predictions": fit_predictions,
        "predict_positions": predict_positions,
        "predictions": predictions,
        "meta": {
            "fit_start": str(FIT_START),
            "fit_end_exclusive": fit_end,
            "predict_start": predict_start,
            "predict_end_exclusive": predict_end,
            "fit_rows": int(len(fit_positions)),
            "predict_rows": int(len(predict_positions)),
            "target_clip": {"lower": lower.tolist(), "upper": upper.tolist()},
            "fit_position_hash": _array_hash(
                np.asarray(fit_positions, dtype="<i8")
            ),
            "predict_position_hash": _array_hash(
                np.asarray(predict_positions, dtype="<i8")
            ),
            "fit_prediction_hash": _array_hash(
                np.asarray(fit_predictions, dtype="<f8")
            ),
            "prediction_hash": _array_hash(
                np.asarray(predictions, dtype="<f8")
            ),
        },
    }


def policy_masks(
    length: int,
    fitted: dict[str, Any],
    *,
    quantile: float,
    allowed_side: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fit_predictions = np.asarray(fitted["fit_predictions"], dtype=float)
    predictions = np.asarray(fitted["predictions"], dtype=float)
    threshold = float(
        np.quantile(np.max(fit_predictions, axis=1), float(quantile))
    )
    confidence = np.max(predictions, axis=1)
    eligible = (
        np.isfinite(confidence)
        & (confidence >= threshold)
        & (confidence > 0.0)
    )
    local_long = (
        eligible
        & (predictions[:, 0] > predictions[:, 1])
        & (allowed_side in {"both", "long"})
    )
    local_short = (
        eligible
        & (predictions[:, 1] > predictions[:, 0])
        & (allowed_side in {"both", "short"})
    )
    long_active = np.zeros(length, dtype=bool)
    short_active = np.zeros(length, dtype=bool)
    positions = np.asarray(fitted["predict_positions"], dtype=np.int64)
    long_active[positions] = local_long
    short_active[positions] = local_short
    activation = np.column_stack(
        [
            positions[local_long | local_short],
            np.where(local_long[local_long | local_short], 1, -1),
        ]
    ).astype("<i8", copy=False)
    return long_active, short_active, {
        "threshold": threshold,
        "eligible": int(np.sum(eligible)),
        "long_signals": int(np.sum(local_long)),
        "short_signals": int(np.sum(local_short)),
        "activation_hash": _array_hash(activation),
    }


def _cell_passes(stats: dict[str, dict[str, Any]]) -> bool:
    full = [stats["2023"], stats["2024"]]
    halves = [
        stats["2023_h1"],
        stats["2023_h2"],
        stats["2024_h1"],
        stats["2024_h2"],
    ]
    return bool(
        all(row["return_pct"] > 0.0 for row in full)
        and all(row["trades"] >= 12 for row in full)
        and all(row["trades"] >= 4 for row in halves)
        and all(row["strict_mdd_pct"] <= 15.0 for row in full)
        and min(row["ratio"] for row in full) >= 1.5
    )


def _cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    stats = row["stats"]
    ratios = [stats["2023"]["ratio"], stats["2024"]["ratio"]]
    half_returns = [
        stats["2023_h1"]["return_pct"],
        stats["2023_h2"]["return_pct"],
        stats["2024_h1"]["return_pct"],
        stats["2024_h2"]["return_pct"],
    ]
    return (
        min(ratios),
        math.sqrt(max(0.0, ratios[0]) * max(0.0, ratios[1])),
        min(half_returns),
        stats["2023"]["return_pct"] + stats["2024"]["return_pct"],
        -int(row["spec"]["max_leaf_nodes"]),
        -int(row["spec"]["hold_bars"]),
        str(row["name"]),
    )


def _feature_hash(features: pd.DataFrame, dates: pd.Series) -> str:
    values = features.to_numpy(dtype="<f4", copy=True)
    values = np.nan_to_num(
        values,
        nan=3.4028233e38,
        posinf=3.4028231e38,
        neginf=-3.4028231e38,
    )
    digest = hashlib.sha256()
    digest.update("\n".join(features.columns).encode())
    digest.update(
        pd.to_datetime(dates)
        .to_numpy(dtype="datetime64[ns]")
        .astype("<i8", copy=False)
        .tobytes()
    )
    digest.update(values.tobytes())
    return digest.hexdigest()


def _coverage(
    market: pd.DataFrame, features: pd.DataFrame, source_valid: np.ndarray
) -> dict[str, Any]:
    dates = pd.to_datetime(market["date"])
    output: dict[str, Any] = {}
    finite = np.isfinite(features.to_numpy(float)).all(axis=1)
    for year in range(2020, 2025):
        mask = np.asarray(
            (dates >= pd.Timestamp(f"{year}-01-01"))
            & (dates < pd.Timestamp(f"{year + 1}-01-01")),
            dtype=bool,
        )
        output[str(year)] = {
            "rows": int(np.sum(mask)),
            "spot_complete_fraction": float(
                pd.to_numeric(market.loc[mask, "spot_rows"], errors="coerce")
                .eq(5)
                .mean()
            ),
            "delayed_oi_available_fraction": float(
                pd.to_numeric(market["open_interest"], errors="coerce")
                .shift(1)
                .loc[mask]
                .gt(0.0)
                .mean()
            ),
            "full_feature_fraction": float(np.mean(finite[mask])),
            "source_valid_fraction": float(np.mean(source_valid[mask])),
        }
    return output


def _public_fold(fold: dict[str, Any]) -> dict[str, Any]:
    return dict(fold["meta"])


def _metric_cell(metric: dict[str, Any]) -> str:
    return (
        f"{metric['return_pct']:.2f}/{metric['cagr_pct']:.2f}/"
        f"{metric['strict_mdd_pct']:.2f}/{metric['ratio']:.2f}/"
        f"{metric['trades']} ({metric['longs']}/{metric['shorts']})"
    )


def _render(payload: dict[str, Any]) -> str:
    displayed = payload["shortlist"] or payload["diagnostic_top"]
    table_label = (
        "frozen shortlist"
        if payload["shortlist"]
        else "top failed cells (diagnostic only; no cell admitted)"
    )
    lines = [
        "# AOSP-1 pre-2025 annual selection",
        "",
        "Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades (long/short)`.",
        "",
        f"- features: {payload['feature_count']}",
        f"- policy cells: {payload['tested']}",
        f"- standalone passers: {payload['passed']}",
        f"- decision: **{payload['decision']}**",
        f"- table: {table_label}",
        "",
        "| # | policy | pass | 2023 | 2023 H1 | 2023 H2 | 2024 | 2024 H1 | 2024 H2 |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(displayed, start=1):
        stats = row["stats"]
        cells = [
            _metric_cell(stats[name])
            for name in (
                "2023",
                "2023_h1",
                "2023_h2",
                "2024",
                "2024_h1",
                "2024_h2",
            )
        ]
        lines.append(
            f"| {index} | `{row['name']}` | {'Y' if row['passes'] else 'N'} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "## Leakage and execution",
        "",
        "- Both market and spot frames are physically truncated before 2025.",
        "- 2023 and 2024 use separate annual expanding fits.",
        "- Every fit target is purged unless its exit is strictly before the fold cutoff.",
        "- OI is delayed one complete 5m bar; incomplete spot or missing OI fails closed.",
        "- Entry is the next 5m open; costs, idle calendar time, non-overlap, and split-contained exits are included.",
        "- 2025/2026 and Gross9 portfolio marginal outcomes are not opened here.",
    ]
    return "\n".join(lines) + "\n"


def _atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def run_pre2025(cfg: Config) -> dict[str, Any]:
    preregistration = load_preregistration(cfg.preregistration)
    market = load_sources(cfg)
    dates = pd.to_datetime(market["date"])
    features, source_valid = build_features(market)
    valid_decisions = decision_mask(market, source_valid)
    specs = model_specs(preregistration)
    fold_bank: dict[tuple[str, str], dict[str, Any]] = {}
    fold_meta: dict[str, dict[str, Any]] = {}
    for spec in specs:
        fold_meta[spec.name] = {}
        for fold_name, (start, end) in FOLDS.items():
            fitted = fit_fold(
                market,
                features,
                valid_decisions,
                spec,
                fit_end=start,
                predict_start=start,
                predict_end=end,
                cfg=cfg,
            )
            fold_bank[(spec.name, fold_name)] = fitted
            fold_meta[spec.name][fold_name] = _public_fold(fitted)

    grid = preregistration["frozen_grid"]
    rows: list[dict[str, Any]] = []
    extremes = {
        spec.hold_bars: (
            _future_extreme(
                pd.to_numeric(market["low"], errors="coerce").to_numpy(float),
                spec.hold_bars,
                "min",
            ),
            _future_extreme(
                pd.to_numeric(market["high"], errors="coerce").to_numpy(float),
                spec.hold_bars,
                "max",
            ),
        )
        for spec in specs
    }
    for spec in specs:
        for quantile in grid["score_quantile"]:
            for allowed_side in grid["allowed_side"]:
                stats: dict[str, dict[str, Any]] = {}
                policy_meta: dict[str, Any] = {}
                for fold_name in FOLDS:
                    fitted = fold_bank[(spec.name, fold_name)]
                    long_active, short_active, meta = policy_masks(
                        len(market),
                        fitted,
                        quantile=float(quantile),
                        allowed_side=str(allowed_side),
                    )
                    policy_meta[fold_name] = meta
                    metric_names = (
                        (fold_name, f"{fold_name}_h1", f"{fold_name}_h2")
                    )
                    for metric_name in metric_names:
                        stats[metric_name] = _simulate_no_stop(
                            market,
                            dates,
                            long_active,
                            short_active,
                            window=metric_name,
                            hold_bars=spec.hold_bars,
                            stride_bars=int(grid["decision_stride_bars"]),
                            leverage=cfg.leverage,
                            fee_rate=cfg.cost_rate,
                            slippage_rate=0.0,
                            extremes=extremes[spec.hold_bars],
                            windows=WINDOWS,
                        )
                name = f"aosp_{spec.name}_q{int(round(float(quantile) * 100))}_{allowed_side}"
                row = {
                    "name": name,
                    "spec": asdict(spec),
                    "score_quantile": float(quantile),
                    "allowed_side": str(allowed_side),
                    "policy_meta": policy_meta,
                    "stats": stats,
                }
                row["passes"] = _cell_passes(stats)
                rows.append(row)
    if len(rows) != int(grid["model_policy_cells"]):
        raise RuntimeError("evaluated policy count drifted")
    passed = sorted(
        (row for row in rows if row["passes"]), key=_cell_key, reverse=True
    )
    max_shortlist = int(
        preregistration["selection_contract"]["standalone_pre2025"][
            "shortlist_max"
        ]
    )
    shortlist = passed[:max_shortlist]
    diagnostic_top = sorted(rows, key=_cell_key, reverse=True)[:max_shortlist]
    freeze = {
        "preregistration_sha256": _sha256(cfg.preregistration),
        "source_prefix": {
            "cutoff": SELECTION_CUTOFF,
            "rows": len(market),
            "first": str(dates.iloc[0]),
            "last": str(dates.iloc[-1]),
            "market_sha256": _sha256(cfg.market_csv),
            "spot_sha256": _sha256(cfg.spot_csv),
        },
        "feature_hash": _feature_hash(features, dates),
        "fold_meta": fold_meta,
        "shortlist": shortlist,
    }
    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre2025_annual_selection",
        "config": asdict(cfg),
        "preregistration": cfg.preregistration,
        "preregistration_sha256": freeze["preregistration_sha256"],
        "source_prefix": freeze["source_prefix"],
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_hash": freeze["feature_hash"],
        "coverage": _coverage(market, features, source_valid),
        "fold_meta": fold_meta,
        "tested": len(rows),
        "passed": len(passed),
        "shortlist": shortlist,
        "diagnostic_top": diagnostic_top,
        "decision": "open_gross9_marginal" if shortlist else "reject_pre2025",
        "gross9_marginal_opened": False,
        "future_opened": False,
        "freeze_hash": _json_hash(freeze),
        "all_pre2025_rows": rows,
    }
    _atomic_json(cfg.output, payload)
    docs = Path(cfg.docs_output)
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(_render(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("pre2025",))
    parser.add_argument("--market-csv", default=Config.market_csv)
    parser.add_argument("--spot-csv", default=Config.spot_csv)
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    args = parser.parse_args()
    payload = run_pre2025(
        Config(
            market_csv=args.market_csv,
            spot_csv=args.spot_csv,
            preregistration=args.preregistration,
            output=args.output,
            docs_output=args.docs_output,
        )
    )
    print(
        json.dumps(
            {
                "phase": payload["phase"],
                "tested": payload["tested"],
                "passed": payload["passed"],
                "decision": payload["decision"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
