#!/usr/bin/env python3
"""Run the preregistered RREM-1 residual-recovery Gross9 marginal battery.

The learner sees only causal spot-perpetual and delayed inventory-conservation
residuals.  Every model and threshold is annual expanding.  This module exposes
only the physically truncated pre-2025 candidate graph and the frozen Gross9
selection windows; it has no future-evaluation phase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

import training.audit_gross9_fixed_candidate_state_substitution as gross9_context
import training.portfolio_opt_added_alpha_update as portfolio
from preprocessing.binance_aux_features import (
    attach_binance_um_aux_frames,
    normalise_funding_history_frame,
    normalise_premium_index_frame,
)
from training.audit_gross9_invariant_ensemble_top10_marginal import (
    SELECTION_REX_PATHS,
    append_rex_taker_policy_selection_only,
)
from training.audit_gross9_oi_pullback_marginal import (
    _atomic_json,
    _sha256,
)
from training.search_inventory_conservation_residual_alpha import (
    _attach_delayed_metrics,
    _build_features as build_inventory_features,
    _frame_hash,
    _read_premium_before,
)
from training.search_positioning_hgb_path_alpha import _read_before
from training.search_spot_perp_absorption_alpha import (
    SpotPerpConfig,
    _load_market_spot,
    build_features as build_spot_perp_features,
)


PREREGISTRATION = Path(
    "results/"
    "residual_recovery_extratrees_marginal_preregistration_2026-07-28.json"
)
OUTPUT = Path(
    "results/"
    "gross9_residual_recovery_extratrees_marginal_pre2025_2026-07-28.json"
)
DOCS = Path(
    "docs/"
    "gross9-residual-recovery-extratrees-marginal-pre2025-2026-07-28.md"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "498f546d0ce44013d6ef3f9ee7aa05ddbcdffe250c0b2d9ba3039d946930d540"
)
SELECTION_CUTOFF = "2025-01-01"
SELECTION_SPLITS = ("train", "test2024")
FIT_START = pd.Timestamp("2020-10-15")
FEATURE_COLUMNS = (
    "spa_residual_z_2016",
    "spa_residual_z_8640",
    "spa_lead_residual_z_2016",
    "spa_lead_residual_z_8640",
    "spa_flow_z_2016",
    "spa_flow_z_8640",
    "spa_spot_share_z_2016",
    "spa_spot_share_z_8640",
    "resid_z48",
    "resid_z144",
    "resid_z288",
    "carry_z",
)
HOLDS = (72, 144)
COORDINATION_MODES = (
    "unrestricted",
    "gross9_flat_at_signal",
    "gross9_drawdown_ge_5pct",
)
WEIGHTS = (0.25, 0.5, 0.75, 1.0)
SEEDS = (7, 71, 715)
CANDIDATE_NAMES = tuple(
    f"residual_recovery_h{hold:03d}_{mode}"
    for hold in HOLDS
    for mode in COORDINATION_MODES
)
INPUT_KEYS = (
    "market",
    "market_with_oi",
    "spot_premium_5m",
    "binance_metrics_5m",
    "funding",
    "premium_1h",
    "gross9_pre2025_anchor",
    "rank7_capacity_evidence",
    "spot_perp_feature_builder",
    "inventory_feature_builder",
    "gross9_context_builder",
    "gross9_portfolio_engine",
)


@dataclass(frozen=True)
class Config:
    preregistration: str = str(PREREGISTRATION)
    output: str = str(OUTPUT)
    docs_output: str = str(DOCS)
    market_csv: str = portfolio.Config.input_csv
    market_with_oi_csv: str = (
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
    )
    spot_csv: str = (
        "data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz"
    )
    metrics_csv: str = (
        "data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz"
    )
    funding_csv: str = portfolio.Config.funding_csv
    premium_csv: str = portfolio.Config.premium_csv
    gross9_pre2025_anchor: str = (
        "results/gross9_pre2025_authoritative_anchor_2026-07-28.json"
    )
    rank7_capacity_evidence: str = (
        "results/expanding_extratrees_rank7_leverage_battery_2026-07-27.json"
    )
    cutoff: str = SELECTION_CUTOFF
    cost_rate: float = 0.0006
    stress_cost_rate: float = 0.001
    leverage: float = 0.5
    stride_bars: int = 12
    metrics_delay_bars: int = 1
    metrics_tolerance: str = "5min"
    funding_tolerance: str = "12h"
    premium_tolerance: str = "65min"


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode("ascii"))
    h.update(np.asarray(array.shape, dtype="<i8").tobytes())
    h.update(array.tobytes())
    return h.hexdigest()


def load_preregistration(path: str | Path) -> dict[str, Any]:
    observed = _sha256(path)
    if observed != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError(
            f"preregistration hash drifted: {observed} != "
            f"{EXPECTED_PREREGISTRATION_SHA256}"
        )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("name") != (
        "residual_recovery_extratrees_gross9_marginal_battery"
    ):
        raise RuntimeError("unexpected preregistration")
    if payload.get("physical_selection_cutoff") != SELECTION_CUTOFF:
        raise RuntimeError("selection cutoff drifted")
    if tuple(payload["feature_contract"]["columns"]) != FEATURE_COLUMNS:
        raise RuntimeError("feature contract drifted")
    universe = payload["candidate_universe"]
    if tuple(universe["hold_bars"]) != HOLDS:
        raise RuntimeError("hold grid drifted")
    if tuple(row["name"] for row in universe["coordination_modes"]) != (
        COORDINATION_MODES
    ):
        raise RuntimeError("coordination grid drifted")
    if tuple(float(value) for value in universe["candidate_weight_grid"]) != (
        WEIGHTS
    ):
        raise RuntimeError("weight grid drifted")
    if int(universe["portfolio_cells"]) != len(CANDIDATE_NAMES) * len(WEIGHTS):
        raise RuntimeError("portfolio cell count drifted")
    learner = payload["learner_contract"]
    expected = {
        "n_estimators": 256,
        "max_depth": 4,
        "min_samples_leaf": 128,
        "max_features": 0.75,
        "risk_lambda": 0.5,
        "seed_uncertainty_lambda": 0.5,
        "score_quantile": 0.8,
        "anchor_stride_bars": 12,
        "entry_delay_bars": 1,
        "unit_leverage": 0.5,
    }
    for key, value in expected.items():
        if learner.get(key) != value:
            raise RuntimeError(f"learner contract drifted: {key}")
    if tuple(learner["seeds"]) != SEEDS:
        raise RuntimeError("seed contract drifted")
    if payload["selection_contract"].get("top1_only") is not True:
        raise RuntimeError("top1-only contract drifted")
    future = payload["future_veto_contract"]
    if future.get("future_can_rerank") is not False:
        raise RuntimeError("future reranking is enabled")
    if future.get("future_can_repair") is not False:
        raise RuntimeError("future repair is enabled")
    return payload


def configured_inputs(cfg: Config) -> dict[str, str]:
    return {
        "market": cfg.market_csv,
        "market_with_oi": cfg.market_with_oi_csv,
        "spot_premium_5m": cfg.spot_csv,
        "binance_metrics_5m": cfg.metrics_csv,
        "funding": cfg.funding_csv,
        "premium_1h": cfg.premium_csv,
        "gross9_pre2025_anchor": cfg.gross9_pre2025_anchor,
        "rank7_capacity_evidence": cfg.rank7_capacity_evidence,
        "spot_perp_feature_builder": (
            "training/search_spot_perp_absorption_alpha.py"
        ),
        "inventory_feature_builder": (
            "training/search_inventory_conservation_residual_alpha.py"
        ),
        "gross9_context_builder": (
            "training/audit_gross9_fixed_candidate_state_substitution.py"
        ),
        "gross9_portfolio_engine": (
            "training/portfolio_opt_added_alpha_update.py"
        ),
    }


def validate_inputs(
    cfg: Config,
    preregistration: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    configured = configured_inputs(cfg)
    if tuple(configured) != INPUT_KEYS:
        raise RuntimeError("configured input order drifted")
    provenance = preregistration["input_provenance"]
    if tuple(provenance) != INPUT_KEYS:
        raise RuntimeError("preregistered input order drifted")
    records: dict[str, dict[str, Any]] = {}
    for name, path in configured.items():
        configured_path = Path(path).resolve(strict=True)
        frozen_path = Path(provenance[name]["path"]).resolve(strict=True)
        if configured_path != frozen_path:
            raise RuntimeError(f"input path drifted for {name}")
        observed = _sha256(configured_path)
        expected = str(provenance[name]["sha256"])
        if observed != expected:
            raise RuntimeError(
                f"input hash drifted for {name}: {observed} != {expected}"
            )
        records[name] = {
            "path": str(configured_path),
            "sha256": observed,
        }
    return records


def _load_inventory_frame(
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    market_raw = _read_before(cfg.market_with_oi_csv, "date", cfg.cutoff)
    metrics_raw = _read_before(cfg.metrics_csv, "create_time", cfg.cutoff)
    funding_raw = _read_before(cfg.funding_csv, "date", cfg.cutoff)
    premium_raw = _read_premium_before(cfg.premium_csv, cfg.cutoff)
    boundary = pd.Timestamp(cfg.cutoff)

    market = market_raw.copy()
    market["date"] = pd.to_datetime(
        market["date"], utc=True, errors="raise"
    ).dt.tz_convert(None)
    market = (
        market.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if len(market) and market["date"].max() >= boundary:
        raise RuntimeError("inventory market breached pre-2025 cutoff")
    market = attach_binance_um_aux_frames(
        market,
        funding_frame=normalise_funding_history_frame(funding_raw),
        premium_frame=normalise_premium_index_frame(premium_raw),
        funding_tolerance=cfg.funding_tolerance,
        premium_tolerance=cfg.premium_tolerance,
    )
    market = _attach_delayed_metrics(
        market,
        metrics_raw,
        tolerance=cfg.metrics_tolerance,
        delay_bars=cfg.metrics_delay_bars,
    )
    source_times = pd.to_datetime(
        market.loc[
            market["positioning_source_time"].notna(),
            "positioning_source_time",
        ]
    )
    if len(source_times) and source_times.max() >= boundary:
        raise RuntimeError("delayed metrics source breached cutoff")
    dates = pd.to_datetime(market["date"])
    features, residual_meta = build_inventory_features(market, dates)
    audit = {
        "source_prefix_hashes": {
            "market_with_oi": _frame_hash(market_raw),
            "metrics": _frame_hash(metrics_raw),
            "funding": _frame_hash(funding_raw),
            "premium": _frame_hash(premium_raw),
        },
        "source_max": {
            "market": str(dates.max()),
            "metrics_available": (
                str(source_times.max()) if len(source_times) else None
            ),
        },
        "residual_fit": residual_meta,
        "metrics_delay_bars": cfg.metrics_delay_bars,
    }
    return market, features, audit


def build_residual_features(
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spot_cfg = SpotPerpConfig(
        input_csv=cfg.market_csv,
        spot_csv=cfg.spot_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        output="",
        manifest_output="",
        docs_output="",
    )
    spot_market, spot_hashes = _load_market_spot(
        spot_cfg, cutoff=cfg.cutoff
    )
    spot_features = build_spot_perp_features(spot_market)
    inventory_market, inventory_features, inventory_audit = (
        _load_inventory_frame(cfg)
    )
    spot_dates = pd.to_datetime(spot_market["date"]).reset_index(drop=True)
    inventory_dates = pd.to_datetime(
        inventory_market["date"]
    ).reset_index(drop=True)
    if not spot_dates.equals(inventory_dates):
        raise RuntimeError("spot-perp and inventory clocks differ")
    intervals = spot_dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise RuntimeError("residual feature clock is not complete 5m")

    frame = pd.concat(
        [
            spot_dates.rename("date"),
            spot_features.loc[:, list(FEATURE_COLUMNS[:8])].reset_index(
                drop=True
            ),
            inventory_features.loc[:, list(FEATURE_COLUMNS[8:])].reset_index(
                drop=True
            ),
        ],
        axis=1,
    )
    if tuple(frame.columns[1:]) != FEATURE_COLUMNS:
        raise RuntimeError("residual feature order drifted")
    feature_hash = _frame_hash(frame)
    audit = {
        "physical_cutoff": cfg.cutoff,
        "rows": int(len(frame)),
        "first": str(frame["date"].iloc[0]),
        "last": str(frame["date"].iloc[-1]),
        "spot_prefix_hashes": spot_hashes,
        "inventory": inventory_audit,
        "feature_frame_hash": feature_hash,
        "finite_rows": int(
            np.isfinite(
                frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(float)
            )
            .all(axis=1)
            .sum()
        ),
    }
    return spot_market, frame, audit


def _install_candidate_universe() -> None:
    names = list(portfolio.SLEEVES)
    for name in CANDIDATE_NAMES:
        if name not in names:
            names.append(name)
    portfolio.SLEEVES = tuple(names)
    portfolio.FAMILIES = {
        **portfolio.FAMILIES,
        **{name: "residual_recovery" for name in CANDIDATE_NAMES},
    }


def build_gross9_context(
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    base_cfg = gross9_context.Config(
        market_csv=cfg.market_csv,
        market_with_oi_csv=cfg.market_with_oi_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        gross9_pre2025_anchor=cfg.gross9_pre2025_anchor,
        rank7_capacity_evidence=cfg.rank7_capacity_evidence,
        cost_rate=cfg.cost_rate,
    )
    original = portfolio.append_rex_taker_policy
    portfolio.append_rex_taker_policy = append_rex_taker_policy_selection_only
    try:
        market, masks, events, _features, source_meta = (
            gross9_context.build_gross9_selection_context(base_cfg)
        )
    finally:
        portfolio.append_rex_taker_policy = original
    dates = pd.to_datetime(market["date"])
    keep = dates < pd.Timestamp(cfg.cutoff)
    if not keep.any() or not np.all(keep[: int(keep.sum())]):
        raise RuntimeError("Gross9 market cutoff is not a contiguous prefix")
    length = int(keep.sum())
    market = market.iloc[:length].reset_index(drop=True)
    masks = {
        split: np.asarray(mask[:length], dtype=bool)
        for split, mask in masks.items()
    }
    if tuple(masks) != SELECTION_SPLITS:
        raise RuntimeError("future Gross9 masks entered RREM selection")
    source_meta["rrem_selection_boundary"] = {
        "market_prefix_rows": length,
        "market_prefix_last": str(pd.to_datetime(market["date"]).iloc[-1]),
        "future_masks_exposed": False,
        "rex_paths_opened": list(SELECTION_REX_PATHS),
        "rex_future_path_opened": False,
    }
    return market, masks, events, source_meta


def align_features_to_market(
    market: pd.DataFrame,
    residual_frame: pd.DataFrame,
) -> pd.DataFrame:
    dates = pd.to_datetime(market["date"])
    lookup = residual_frame.set_index("date")
    if lookup.index.has_duplicates:
        raise RuntimeError("residual feature dates are duplicated")
    aligned = lookup.reindex(pd.DatetimeIndex(dates)).reset_index(drop=True)
    if tuple(aligned.columns) != FEATURE_COLUMNS:
        raise RuntimeError("aligned residual feature order drifted")
    return aligned.astype(np.float64)


def exact_no_stop_targets(
    market: pd.DataFrame,
    anchors: np.ndarray,
    *,
    hold: int,
    cost_rate: float,
    leverage: float,
) -> np.ndarray:
    """Vectorized labels matching the canonical no-stop bar accounting."""
    anchors = np.asarray(anchors, dtype=np.int64)
    entry = anchors + 1
    exit_position = entry + int(hold)
    if len(anchors) and (
        anchors.min() < 0 or exit_position.max(initial=-1) >= len(market)
    ):
        raise ValueError("target anchor is outside market")
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    bar_open = opens[:-1]
    next_open = opens[1:]
    long_bar = float(leverage) * (next_open - bar_open) / bar_open
    short_bar = float(leverage) * (bar_open - next_open) / bar_open
    cost = float(cost_rate) * float(leverage)

    def window_product(values: np.ndarray) -> np.ndarray:
        factors = 1.0 + values
        if np.any(factors <= 0.0):
            raise RuntimeError("non-positive no-stop bar factor")
        prefix = np.r_[0.0, np.cumsum(np.log(factors))]
        raw = np.exp(prefix[entry + hold] - prefix[entry])
        entry_adjustment = (
            factors[entry] - cost
        ) / factors[entry]
        return raw * entry_adjustment * (1.0 - cost) - 1.0

    long_return = window_product(long_bar)
    short_return = window_product(short_bar)
    long_adverse_bar = np.maximum(
        0.0,
        float(leverage) * (bar_open - lows[:-1]) / bar_open,
    )
    short_adverse_bar = np.maximum(
        0.0,
        float(leverage) * (highs[:-1] - bar_open) / bar_open,
    )
    long_windows = np.lib.stride_tricks.sliding_window_view(
        long_adverse_bar, int(hold)
    )
    short_windows = np.lib.stride_tricks.sliding_window_view(
        short_adverse_bar, int(hold)
    )
    long_adverse = long_windows[entry].max(axis=1)
    short_adverse = short_windows[entry].max(axis=1)
    return np.column_stack(
        [long_return, long_adverse, short_return, short_adverse]
    )


def _model_hash(model: ExtraTreesRegressor) -> str:
    return hashlib.sha256(
        pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _fit_predict_ensemble(
    matrix: np.ndarray,
    targets: np.ndarray,
    fit: np.ndarray,
    predict: np.ndarray,
    preregistration: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if int(fit.sum()) < 1_000:
        raise RuntimeError(f"insufficient RREM fit rows: {int(fit.sum())}")
    if not predict.any():
        raise RuntimeError("empty RREM prediction fold")
    x_fit = np.asarray(matrix[fit], dtype=np.float64)
    y_fit = np.asarray(targets[fit], dtype=np.float64)
    x_predict = np.asarray(matrix[predict], dtype=np.float64)
    medians = np.nanmedian(x_fit, axis=0)
    if not np.isfinite(medians).all():
        raise RuntimeError("non-finite RREM training median")
    x_fit = np.where(np.isfinite(x_fit), x_fit, medians)
    x_predict = np.where(np.isfinite(x_predict), x_predict, medians)
    learner = preregistration["learner_contract"]
    predictions: list[np.ndarray] = []
    model_hashes: list[str] = []
    for seed in SEEDS:
        model = ExtraTreesRegressor(
            n_estimators=int(learner["n_estimators"]),
            max_depth=int(learner["max_depth"]),
            min_samples_leaf=int(learner["min_samples_leaf"]),
            max_features=float(learner["max_features"]),
            bootstrap=False,
            random_state=int(seed),
            n_jobs=-1,
        ).fit(x_fit, y_fit)
        predictions.append(model.predict(x_predict))
        model_hashes.append(_model_hash(model))
    stacked = np.stack(predictions, axis=0)
    return stacked, {
        "fit_rows": int(fit.sum()),
        "predict_rows": int(predict.sum()),
        "training_median": medians.tolist(),
        "model_hashes": model_hashes,
        "prediction_hash": _array_hash(stacked),
    }


def adjusted_scores(
    predictions: np.ndarray,
    *,
    risk_lambda: float = 0.5,
    uncertainty_lambda: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 4:
        raise ValueError("predictions must have shape seeds x rows x 4")
    long_utility = values[:, :, 0] - float(risk_lambda) * values[:, :, 1]
    short_utility = values[:, :, 2] - float(risk_lambda) * values[:, :, 3]
    long_score = long_utility.mean(axis=0) - float(
        uncertainty_lambda
    ) * long_utility.std(axis=0, ddof=0)
    short_score = short_utility.mean(axis=0) - float(
        uncertainty_lambda
    ) * short_utility.std(axis=0, ddof=0)
    choose_long = long_score >= short_score
    score = np.where(choose_long, long_score, short_score)
    side = np.where(choose_long, 1, -1).astype(np.int8)
    return score, side


def annual_masks(
    signal_dates: pd.Series,
    exit_dates: pd.Series,
    finite_fit_rows: np.ndarray,
    finite_predict_rows: np.ndarray,
    *,
    prediction_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    start = pd.Timestamp(f"{prediction_year}-01-01")
    end = pd.Timestamp(f"{prediction_year + 1}-01-01")
    fit_finite = np.asarray(finite_fit_rows, dtype=bool)
    predict_finite = np.asarray(finite_predict_rows, dtype=bool)
    fit = np.asarray(
        (signal_dates >= FIT_START)
        & (exit_dates < start)
        & fit_finite,
        dtype=bool,
    )
    predict = np.asarray(
        (signal_dates >= start)
        & (signal_dates < end)
        & (exit_dates < end)
        & predict_finite,
        dtype=bool,
    )
    return fit, predict


def annual_residual_predictions(
    market: pd.DataFrame,
    features: pd.DataFrame,
    preregistration: dict[str, Any],
    *,
    hold: int,
    cfg: Config,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    dates = pd.to_datetime(market["date"]).reset_index(drop=True)
    matrix = features.to_numpy(np.float64)
    anchors = np.arange(
        143,
        len(market) - int(hold) - 1,
        int(cfg.stride_bars),
        dtype=np.int64,
    )
    targets = exact_no_stop_targets(
        market,
        anchors,
        hold=hold,
        cost_rate=cfg.cost_rate,
        leverage=cfg.leverage,
    )
    signal_dates = dates.iloc[anchors].reset_index(drop=True)
    exit_dates = dates.iloc[anchors + 1 + int(hold)].reset_index(drop=True)
    finite_features = np.isfinite(matrix[anchors]).all(axis=1)
    finite_targets = np.isfinite(targets).all(axis=1)
    outputs: dict[int, dict[str, np.ndarray]] = {}
    meta: dict[str, Any] = {
        "hold_bars": int(hold),
        "anchors": int(len(anchors)),
        "finite_feature_anchors": int(finite_features.sum()),
        "target_hash": _array_hash(targets),
        "annual_models": {},
    }
    learner = preregistration["learner_contract"]
    for year in (2022, 2023, 2024):
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year + 1}-01-01")
        fit, predict = annual_masks(
            signal_dates,
            exit_dates,
            finite_features & finite_targets,
            finite_features,
            prediction_year=year,
        )
        predictions, model_meta = _fit_predict_ensemble(
            matrix[anchors],
            targets,
            fit,
            predict,
            preregistration,
        )
        score, side = adjusted_scores(
            predictions,
            risk_lambda=float(learner["risk_lambda"]),
            uncertainty_lambda=float(
                learner["seed_uncertainty_lambda"]
            ),
        )
        outputs[year] = {
            "anchors": anchors[predict],
            "score": score,
            "side": side,
        }
        model_meta.update(
            {
                "fit_end_exclusive": str(start.date()),
                "prediction_start": str(start.date()),
                "prediction_end_exclusive": str(end.date()),
                "score_hash": _array_hash(score),
                "side_hash": _array_hash(side),
            }
        )
        meta["annual_models"][str(year)] = model_meta

    for year in (2023, 2024):
        calibration = outputs[year - 1]["score"]
        finite = calibration[np.isfinite(calibration)]
        if len(finite) < 1_000:
            raise RuntimeError("insufficient RREM calibration scores")
        threshold = max(
            float(learner["minimum_score"]),
            float(np.quantile(finite, float(learner["score_quantile"]))),
        )
        outputs[year]["threshold"] = np.asarray(threshold)
        meta["annual_models"][str(year)]["calibration"] = {
            "source_prediction_year": int(year - 1),
            "scores": int(len(finite)),
            "quantile": float(learner["score_quantile"]),
            "raw_quantile": float(
                np.quantile(finite, float(learner["score_quantile"]))
            ),
            "threshold": threshold,
        }
    return outputs, meta


def event_occupancy(event: dict[str, Any], length: int) -> np.ndarray:
    """Reconstruct exact held intervals; zero-range bars remain occupied."""
    occupied = np.zeros(int(length), dtype=bool)
    raw_entries = event.get("entry_positions")
    if raw_entries is None:
        if "signal_pos" not in event:
            raise RuntimeError("event has neither entries nor signal position")
        entries = [int(event["signal_pos"]) + 1]
    else:
        entries = sorted({int(value) for value in raw_entries})
    support = np.zeros(int(length), dtype=bool)
    for key in ("ret", "adv", "fav", "low", "high"):
        if key not in event:
            continue
        values = np.asarray(event[key], dtype=np.float64)
        if len(values) < length:
            raise RuntimeError(f"event {key} path is shorter than market")
        support |= np.abs(values[:length]) > 1e-15
    for index, entry in enumerate(entries):
        if entry < 0 or entry >= length:
            raise RuntimeError("event entry is outside market")
        next_entry = entries[index + 1] if index + 1 < len(entries) else length
        candidates = np.flatnonzero(support[entry:next_entry])
        if not len(candidates):
            raise RuntimeError("event entry has no executable path support")
        exit_position = entry + int(candidates[-1])
        if exit_position < entry:
            raise RuntimeError("event exit precedes entry")
        occupied[entry:exit_position] = True
    return occupied


def gross9_state(
    market: pd.DataFrame,
    masks: dict[str, np.ndarray],
    base_events: list[dict[str, Any]],
    baseline_weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    arrays = portfolio.split_arrays(base_events, market, masks)
    length = len(market)
    flat = np.zeros(length, dtype=bool)
    drawdown = np.zeros(length, dtype=np.float64)
    meta: dict[str, Any] = {}
    vector = np.asarray(
        [baseline_weights.get(name, 0.0) for name in portfolio.SLEEVES],
        dtype=np.float64,
    )
    global_returns = np.zeros(length, dtype=np.float64)
    global_low = np.zeros(length, dtype=np.float64)
    global_high = np.zeros(length, dtype=np.float64)
    global_occupied = np.zeros(length, dtype=bool)
    bounds: dict[str, tuple[int, int]] = {}
    for split in SELECTION_SPLITS:
        data = arrays[split]
        positions = np.flatnonzero(masks[split])
        start, end = int(positions[0]), int(positions[-1]) + 1
        bounds[split] = (start, end)
        global_returns[start:end] = vector @ data["R"]
        global_low[start:end] = vector @ data["L"]
        global_high[start:end] = vector @ data["H"]
    for event in base_events:
        sleeve = str(event.get("sleeve", ""))
        split = str(event.get("split", ""))
        if baseline_weights.get(sleeve, 0.0) <= 0.0:
            continue
        if split not in bounds:
            continue
        global_occupied |= event_occupancy(event, length)

    calendar_start = bounds["train"][0]
    calendar_end = bounds["test2024"][1]
    returns = global_returns[calendar_start:calendar_end]
    low_path = global_low[calendar_start:calendar_end]
    high_path = global_high[calendar_start:calendar_end]
    equity_after = np.cumprod(np.maximum(0.0, 1.0 + returns))
    equity_before = np.r_[1.0, equity_after[:-1]]
    low_value = equity_before * np.maximum(0.0, 1.0 + low_path)
    high_value = equity_before * np.maximum(0.0, 1.0 + high_path)
    upper = np.maximum.reduce(
        [equity_before, equity_after, low_value, high_value]
    )
    peak = np.maximum.accumulate(upper)
    current_drawdown = 1.0 - equity_after / np.maximum(peak, 1e-12)
    flat[calendar_start:calendar_end] = ~global_occupied[
        calendar_start:calendar_end
    ]
    drawdown[calendar_start:calendar_end] = current_drawdown
    for split in SELECTION_SPLITS:
        start, end = bounds[split]
        local_start, local_end = (
            start - calendar_start,
            end - calendar_start,
        )
        occupied = global_occupied[start:end]
        split_drawdown = current_drawdown[local_start:local_end]
        meta[split] = {
            "gross9_flat_fraction": float(np.mean(~occupied)),
            "gross9_drawdown_ge_5pct_fraction": float(
                np.mean(split_drawdown >= 0.05)
            ),
            "state_hash": _array_hash(
                np.column_stack([~occupied, split_drawdown])
            ),
        }
    meta["continuous_clock"] = {
        "start": str(pd.to_datetime(market["date"]).iloc[calendar_start]),
        "end": str(pd.to_datetime(market["date"]).iloc[calendar_end - 1]),
        "reset_at_2024": False,
    }
    return flat, drawdown, meta


def build_candidate_masks(
    market: pd.DataFrame,
    features: pd.DataFrame,
    preregistration: dict[str, Any],
    flat: np.ndarray,
    drawdown: np.ndarray,
    cfg: Config,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    model_meta: dict[str, Any] = {}
    for hold in HOLDS:
        annual, hold_meta = annual_residual_predictions(
            market,
            features,
            preregistration,
            hold=hold,
            cfg=cfg,
        )
        base_long = np.zeros(len(market), dtype=bool)
        base_short = np.zeros(len(market), dtype=bool)
        thresholds: dict[str, float] = {}
        for year in (2023, 2024):
            rows = annual[year]
            anchors = rows["anchors"]
            score = rows["score"]
            side = rows["side"]
            threshold = float(rows["threshold"])
            active = np.isfinite(score) & (score >= threshold)
            base_long[anchors] = active & (side > 0)
            base_short[anchors] = active & (side < 0)
            thresholds[str(year)] = threshold
        for mode in COORDINATION_MODES:
            if mode == "unrestricted":
                gate = np.ones(len(market), dtype=bool)
            elif mode == "gross9_flat_at_signal":
                gate = flat
            elif mode == "gross9_drawdown_ge_5pct":
                gate = drawdown >= 0.05
            else:
                raise KeyError(mode)
            name = f"residual_recovery_h{hold:03d}_{mode}"
            long_active = base_long & gate
            short_active = base_short & gate
            candidates[name] = {
                "hold": int(hold),
                "mode": mode,
                "long_active": long_active,
                "short_active": short_active,
                "activation_hash": _array_hash(
                    np.column_stack([long_active, short_active])
                ),
                "raw_active_anchors": int(
                    np.sum(long_active | short_active)
                ),
                "raw_long_anchors": int(long_active.sum()),
                "raw_short_anchors": int(short_active.sum()),
                "thresholds": thresholds,
            }
        model_meta[str(hold)] = hold_meta
    if tuple(candidates) != CANDIDATE_NAMES:
        raise RuntimeError("RREM candidate order drifted")
    return candidates, model_meta


def append_rrem_mask_policy(
    events: list[dict[str, Any]],
    market: pd.DataFrame,
    masks: dict[str, np.ndarray],
    **kwargs: Any,
) -> dict[str, int]:
    """Use the canonical engine while admitting its contract-valid final bar.

    The shared historical engine intentionally remains hash-frozen for older
    studies and stops one anchor early.  A never-traded sentinel bar widens only
    its iterator bound; every executable RREM path remains inside the original
    market and is sliced back to the original clock immediately afterward.
    """
    if not len(market):
        raise ValueError("cannot append RREM policy to an empty market")
    sentinel = market.iloc[[-1]].copy()
    sentinel["date"] = pd.to_datetime(sentinel["date"]) + pd.Timedelta(
        "5min"
    )
    padded_market = pd.concat([market, sentinel], ignore_index=True)
    padded_masks = {
        split: np.r_[np.asarray(mask, dtype=bool), False]
        for split, mask in masks.items()
    }
    before = len(events)
    counts = portfolio.append_mask_policy(
        events,
        padded_market,
        padded_masks,
        **kwargs,
    )
    for event in events[before:]:
        for key in ("ret", "adv", "fav", "low", "high"):
            event[key] = np.asarray(event[key])[: len(market)]
    return counts


def append_candidates(
    events: list[dict[str, Any]],
    market: pd.DataFrame,
    masks: dict[str, np.ndarray],
    candidates: dict[str, dict[str, Any]],
    *,
    cost_rate: float,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for name, candidate in candidates.items():
        counts[name] = append_rrem_mask_policy(
            events,
            market,
            masks,
            name=name,
            long_active=candidate["long_active"],
            short_active=candidate["short_active"],
            hold=int(candidate["hold"]),
            stride=12,
            cost_rate=float(cost_rate),
        )
    return counts


def _metric(
    arrays: dict[str, dict[str, Any]],
    split: str,
    weights: dict[str, float],
) -> dict[str, Any]:
    return portfolio.strict_metric(
        arrays[split], portfolio.years_for(split), weights
    )


def _slice_2023(data: dict[str, Any]) -> dict[str, Any]:
    dates = pd.DatetimeIndex(data["dates"])
    keep = (dates >= pd.Timestamp("2023-01-01")) & (
        dates < pd.Timestamp("2024-01-01")
    )
    if not keep.any():
        raise RuntimeError("empty 2023 standalone slice")
    return {
        **data,
        "R": data["R"][:, keep],
        "A": data["A"][:, keep],
        "U": data["U"][:, keep],
        "L": data["L"][:, keep],
        "H": data["H"][:, keep],
        "dates": dates[keep],
    }


def standalone_metrics(
    arrays: dict[str, dict[str, Any]],
    candidate: str,
) -> dict[str, dict[str, Any]]:
    weights = {candidate: 1.0}
    return {
        "selection_2023": portfolio.strict_metric(
            _slice_2023(arrays["train"]),
            (
                pd.Timestamp("2024-01-01") - pd.Timestamp("2023-01-01")
            ).total_seconds()
            / (365.25 * 86_400.0),
            weights,
        ),
        "selection_2024": _metric(arrays, "test2024", weights),
    }


def validate_baseline(
    arrays: dict[str, dict[str, Any]],
    anchor: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    weights = {str(k): float(v) for k, v in anchor["weights"].items()}
    observed = {
        split: _metric(arrays, split, weights)
        for split in SELECTION_SPLITS
    }
    expected = anchor["selection_stats"]
    for split in SELECTION_SPLITS:
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
        ):
            if not np.isclose(
                float(observed[split][key]),
                float(expected[split][key]),
                rtol=0.0,
                atol=1e-10,
            ):
                raise RuntimeError(f"Gross9 drift in {split}/{key}")
        if int(observed[split]["trades"]) != int(expected[split]["trades"]):
            raise RuntimeError(f"Gross9 trade count drift in {split}")
    return observed


def entry_jaccards(
    arrays: dict[str, dict[str, Any]],
    market: pd.DataFrame,
    baseline_weights: dict[str, float],
    candidate: str,
) -> dict[str, float]:
    dates = pd.to_datetime(market["date"])
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2025-01-01")

    def pooled(name: str) -> set[int]:
        values: set[int] = set()
        for split in SELECTION_SPLITS:
            for raw in arrays[split]["entry_positions"][name]:
                position = int(raw)
                if start <= dates.iloc[position] < end:
                    values.add(position)
        return values

    candidate_entries = pooled(candidate)
    output: dict[str, float] = {}
    for sleeve, weight in baseline_weights.items():
        if weight <= 0.0:
            continue
        entries = pooled(sleeve)
        union = candidate_entries | entries
        output[sleeve] = (
            float(len(candidate_entries & entries) / len(union))
            if union
            else 0.0
        )
    return output


def signed_geometric_mean(left: float, right: float) -> float:
    product = float(left) * float(right)
    if product == 0.0:
        return 0.0
    if float(left) > 0.0 and float(right) > 0.0:
        return math.sqrt(abs(product))
    return -math.sqrt(abs(product))


def same_gross_weights(
    baseline_weights: dict[str, float],
    candidate: str,
    candidate_weight: float,
) -> tuple[dict[str, float], dict[str, float]]:
    weight = float(candidate_weight)
    combined = {**baseline_weights, candidate: weight}
    comparator = {
        sleeve: float(value) * (9.0 + weight) / 9.0
        for sleeve, value in baseline_weights.items()
    }
    return combined, comparator


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: str(row["candidate_name"]))
    ranked.sort(
        key=lambda row: tuple(float(x) for x in row["selection_key"]),
        reverse=True,
    )
    return ranked


def selection_row(
    preregistration: dict[str, Any],
    *,
    candidate: str,
    candidate_meta: dict[str, Any],
    weight: float,
    baseline: dict[str, dict[str, Any]],
    stats: dict[str, dict[str, Any]],
    comparator: dict[str, dict[str, Any]],
    standalone: dict[str, dict[str, Any]],
    stressed_stats: dict[str, dict[str, Any]],
    stressed_standalone: dict[str, dict[str, Any]],
    max_jaccard: float,
) -> dict[str, Any]:
    standalone_req = preregistration["selection_contract"][
        "standalone_requirements"
    ]
    portfolio_req = preregistration["selection_contract"][
        "portfolio_requirements"
    ]
    improvement = {
        split: float(stats[split]["cagr_to_strict_mdd"])
        - float(comparator[split]["cagr_to_strict_mdd"])
        for split in SELECTION_SPLITS
    }
    retention = {
        split: float(stats[split]["absolute_return_pct"])
        / float(baseline[split]["absolute_return_pct"])
        for split in SELECTION_SPLITS
    }
    mdd_reduction = {
        split: float(baseline[split]["strict_mdd_pct"])
        - float(stats[split]["strict_mdd_pct"])
        for split in SELECTION_SPLITS
    }
    standalone_checks = {
        "standalone_positive": all(
            standalone[window]["absolute_return_pct"] > 0.0
            for window in ("selection_2023", "selection_2024")
        ),
        "standalone_ratio": all(
            standalone[window]["cagr_to_strict_mdd"]
            >= float(
                standalone_req["minimum_cagr_to_strict_mdd_each_year"]
            )
            for window in ("selection_2023", "selection_2024")
        ),
        "standalone_mdd": all(
            standalone[window]["strict_mdd_pct"]
            <= float(standalone_req["maximum_strict_mdd_pct_each_year"])
            for window in ("selection_2023", "selection_2024")
        ),
        "standalone_trades": all(
            standalone[window]["trades"]
            >= int(standalone_req["minimum_trades_each_year"])
            for window in ("selection_2023", "selection_2024")
        ),
        "stressed_standalone_positive": all(
            stressed_standalone[window]["absolute_return_pct"] > 0.0
            for window in ("selection_2023", "selection_2024")
        ),
    }
    checks = {
        **standalone_checks,
        "train_mdd_cap": stats["train"]["strict_mdd_pct"]
        <= float(portfolio_req["train_strict_mdd_cap_pct"]),
        "selection_2024_mdd_cap": stats["test2024"]["strict_mdd_pct"]
        <= float(portfolio_req["selection_2024_strict_mdd_cap_pct"]),
        "return_retention": all(
            retention[split]
            >= float(
                portfolio_req[
                    "absolute_return_retention_floor_vs_unscaled_gross9"
                ]
            )
            for split in SELECTION_SPLITS
        ),
        "same_gross_improvement": all(
            improvement[split]
            >= float(
                portfolio_req[
                    "minimum_cagr_mdd_improvement_vs_same_gross_prorata_gross9_each_window"
                ]
            )
            for split in SELECTION_SPLITS
        ),
        "mdd_reduction": any(
            mdd_reduction[split] > 0.0 for split in SELECTION_SPLITS
        ),
        "entry_jaccard": max_jaccard
        <= float(
            portfolio_req[
                "maximum_exact_entry_jaccard_vs_any_gross9_sleeve"
            ]
        ),
        "stress_portfolio_positive": all(
            stressed_stats[split]["absolute_return_pct"] > 0.0
            for split in SELECTION_SPLITS
        ),
    }
    mode_ordinal = COORDINATION_MODES.index(str(candidate_meta["mode"]))
    key = (
        int(all(checks.values())),
        min(improvement.values()),
        signed_geometric_mean(
            improvement["train"], improvement["test2024"]
        ),
        mdd_reduction["test2024"],
        min(
            stressed_standalone[window]["absolute_return_pct"]
            for window in ("selection_2023", "selection_2024")
        ),
        -float(weight),
        -int(candidate_meta["hold"]),
        -int(mode_ordinal),
    )
    return {
        "candidate_name": candidate,
        "hold_bars": int(candidate_meta["hold"]),
        "coordination_mode": str(candidate_meta["mode"]),
        "coordination_mode_ordinal": int(mode_ordinal),
        "candidate_weight": float(weight),
        "gross": 9.0 + float(weight),
        "passes": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "stats": stats,
        "same_gross_comparator": comparator,
        "standalone": standalone,
        "stressed_stats": stressed_stats,
        "stressed_standalone": stressed_standalone,
        "same_gross_ratio_improvement": improvement,
        "absolute_return_retention": retention,
        "mdd_reduction_vs_unscaled_gross9": mdd_reduction,
        "max_entry_jaccard": float(max_jaccard),
        "selection_key": list(key),
    }


def _metric_cell(metric: dict[str, Any]) -> str:
    return (
        f"{metric['absolute_return_pct']:.2f}/"
        f"{metric['cagr_pct']:.2f}/"
        f"{metric['strict_mdd_pct']:.2f}/"
        f"{metric['cagr_to_strict_mdd']:.2f}/"
        f"{metric['trades']}"
    )


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Gross9 residual-recovery ExtraTrees pre-2025 marginal battery",
        "",
        "Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        f"- tested portfolio cells: `{payload['tested']}`",
        f"- passing cells: `{payload['passed']}`",
        f"- decision: **{payload['decision']}**",
        f"- future opened: `{payload['future_opened']}`",
        "",
        "## Frozen Gross9",
        "",
        f"- train: `{_metric_cell(payload['baseline']['train'])}`",
        f"- selection 2024: `{_metric_cell(payload['baseline']['test2024'])}`",
        "",
        "## Ranked cells",
        "",
        "| # | candidate | w | pass | standalone 2023 | standalone 2024 | portfolio train | portfolio 2024 | same-gross Δ train/2024 | MDD Δ train/2024 |",
        "|---:|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(payload["rows"], start=1):
        lines.append(
            "| {rank} | `{name}` | {weight:.2f} | {passed} | {s23} | "
            "{s24} | {train} | {test} | {it:.3f}/{iv:.3f} | "
            "{mt:.2f}/{mv:.2f} |".format(
                rank=rank,
                name=row["candidate_name"],
                weight=row["candidate_weight"],
                passed="Y" if row["passes"] else "N",
                s23=_metric_cell(row["standalone"]["selection_2023"]),
                s24=_metric_cell(row["standalone"]["selection_2024"]),
                train=_metric_cell(row["stats"]["train"]),
                test=_metric_cell(row["stats"]["test2024"]),
                it=row["same_gross_ratio_improvement"]["train"],
                iv=row["same_gross_ratio_improvement"]["test2024"],
                mt=row["mdd_reduction_vs_unscaled_gross9"]["train"],
                mv=row["mdd_reduction_vs_unscaled_gross9"]["test2024"],
            )
        )
    lines.extend(
        [
            "",
            "## Model admission diagnosis",
            "",
            "| hold | prior-OOS q80 for 2023 | prior-OOS q80 for 2024 | frozen threshold 2023/2024 | unrestricted active anchors |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for hold in HOLDS:
        models = payload["model_meta"][str(hold)]["annual_models"]
        candidate = payload["candidates"][
            f"residual_recovery_h{hold:03d}_unrestricted"
        ]
        lines.append(
            "| {hold} | {q23:.6f} | {q24:.6f} | {t23:.6f}/{t24:.6f} | {active} |".format(
                hold=hold,
                q23=models["2023"]["calibration"]["raw_quantile"],
                q24=models["2024"]["calibration"]["raw_quantile"],
                t23=models["2023"]["calibration"]["threshold"],
                t24=models["2024"]["calibration"]["threshold"],
                active=candidate["raw_active_anchors"],
            )
        )
    if not any(
        payload["candidates"][name]["raw_active_anchors"]
        for name in CANDIDATE_NAMES
    ):
        lines.extend(
            [
                "",
                "Every prior-year OOS q80 score was negative. The preregistered "
                "`max(0, q80)` admission floor therefore produced zero entries. "
                "This is a fail-closed rejection of executable edge, not missing "
                "market data or an execution failure.",
            ]
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Candidate market, spot, OI metrics, funding, and premium inputs were physically truncated before `2025-01-01`.",
            "- OI metrics are delayed one complete 5m source bar; spot-perp regressions use history ending at t-1.",
            "- Annual prediction models are purged by time exit; thresholds use only the prior year's OOS score distribution.",
            "- 2024 is part of pre-2025 selection despite the portfolio engine's historical `test2024` array key.",
            "- No 2025/2026 candidate result was opened. Future cannot rerank or repair this battery.",
            "",
        ]
    )
    return "\n".join(lines)


def run_pre2025(cfg: Config) -> dict[str, Any]:
    preregistration = load_preregistration(cfg.preregistration)
    input_identity = validate_inputs(cfg, preregistration)
    _install_candidate_universe()
    feature_market, residual_frame, feature_audit = (
        build_residual_features(cfg)
    )
    market, masks, base_events, gross9_meta = build_gross9_context(cfg)
    if not pd.to_datetime(feature_market["date"]).equals(
        pd.to_datetime(residual_frame["date"])
    ):
        raise RuntimeError("feature market and feature frame clocks differ")
    features = align_features_to_market(market, residual_frame)

    anchor = json.loads(
        Path(cfg.gross9_pre2025_anchor).read_text(encoding="utf-8")
    )
    baseline_weights = {
        str(name): float(weight) for name, weight in anchor["weights"].items()
    }
    flat, drawdown, state_meta = gross9_state(
        market, masks, base_events, baseline_weights
    )
    candidates, model_meta = build_candidate_masks(
        market,
        features,
        preregistration,
        flat,
        drawdown,
        cfg,
    )

    normal_events = list(base_events)
    normal_counts = append_candidates(
        normal_events,
        market,
        masks,
        candidates,
        cost_rate=cfg.cost_rate,
    )
    normal_arrays = portfolio.split_arrays(normal_events, market, masks)
    stress_events = list(base_events)
    stress_counts = append_candidates(
        stress_events,
        market,
        masks,
        candidates,
        cost_rate=cfg.stress_cost_rate,
    )
    stress_arrays = portfolio.split_arrays(stress_events, market, masks)
    if normal_counts != stress_counts:
        raise RuntimeError("candidate stress changed executable schedules")
    for name in CANDIDATE_NAMES:
        for split in SELECTION_SPLITS:
            if not np.array_equal(
                normal_arrays[split]["entry_positions"][name],
                stress_arrays[split]["entry_positions"][name],
            ):
                raise RuntimeError(
                    f"candidate stress changed entries for {name}/{split}"
                )

    baseline = validate_baseline(normal_arrays, anchor)
    standalone = {
        name: standalone_metrics(normal_arrays, name)
        for name in CANDIDATE_NAMES
    }
    stressed_standalone = {
        name: standalone_metrics(stress_arrays, name)
        for name in CANDIDATE_NAMES
    }
    jaccards = {
        name: entry_jaccards(
            normal_arrays, market, baseline_weights, name
        )
        for name in CANDIDATE_NAMES
    }
    rows: list[dict[str, Any]] = []
    for name in CANDIDATE_NAMES:
        for weight in WEIGHTS:
            combined_weights, control_weights = same_gross_weights(
                baseline_weights, name, float(weight)
            )
            stats = {
                split: _metric(normal_arrays, split, combined_weights)
                for split in SELECTION_SPLITS
            }
            comparator = {
                split: _metric(normal_arrays, split, control_weights)
                for split in SELECTION_SPLITS
            }
            stressed_stats = {
                split: _metric(stress_arrays, split, combined_weights)
                for split in SELECTION_SPLITS
            }
            rows.append(
                selection_row(
                    preregistration,
                    candidate=name,
                    candidate_meta=candidates[name],
                    weight=float(weight),
                    baseline=baseline,
                    stats=stats,
                    comparator=comparator,
                    standalone=standalone[name],
                    stressed_stats=stressed_stats,
                    stressed_standalone=stressed_standalone[name],
                    max_jaccard=max(jaccards[name].values(), default=0.0),
                )
            )
    if len(rows) != 24:
        raise RuntimeError(f"RREM tested-cell drift: {len(rows)}")
    rows = rank_rows(rows)
    passed = [row for row in rows if row["passes"]]
    frozen_top1 = passed[0] if passed else None
    freeze = {
        "preregistration_sha256": _sha256(cfg.preregistration),
        "baseline_weights": baseline_weights,
        "candidate_activation_hashes": {
            name: candidates[name]["activation_hash"]
            for name in CANDIDATE_NAMES
        },
        "normal_counts": normal_counts,
        "rows": rows,
        "frozen_top1": frozen_top1,
    }
    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre2025_residual_recovery_extratrees_gross9_marginal",
        "config": asdict(cfg),
        "preregistration": cfg.preregistration,
        "preregistration_sha256": _sha256(cfg.preregistration),
        "input_identity": input_identity,
        "feature_audit": feature_audit,
        "gross9_source_meta": gross9_meta,
        "gross9_observable_state": state_meta,
        "market": {
            "rows": int(len(market)),
            "first": str(pd.to_datetime(market["date"]).iloc[0]),
            "last": str(pd.to_datetime(market["date"]).iloc[-1]),
        },
        "model_meta": model_meta,
        "candidates": {
            name: {
                key: value
                for key, value in candidates[name].items()
                if key not in ("long_active", "short_active")
            }
            for name in CANDIDATE_NAMES
        },
        "normal_counts": normal_counts,
        "stress_counts": stress_counts,
        "baseline": baseline,
        "standalone": standalone,
        "stressed_standalone": stressed_standalone,
        "entry_jaccards": jaccards,
        "tested": len(rows),
        "passed": len(passed),
        "rows": rows,
        "frozen_top1": frozen_top1,
        "decision": (
            "open_frozen_top1_future_veto"
            if frozen_top1 is not None
            else "reject_battery"
        ),
        "future_opened": False,
        "future_can_rerank": False,
        "future_can_repair": False,
        "freeze_hash": _json_hash(freeze),
    }
    _atomic_json(cfg.output, payload)
    docs = Path(cfg.docs_output)
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(render(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("pre2025",))
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    args = parser.parse_args()
    payload = run_pre2025(
        Config(
            preregistration=args.preregistration,
            output=args.output,
            docs_output=args.docs_output,
        )
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
