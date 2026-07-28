#!/usr/bin/env python3
"""Run the final preregistered residual-family direction classifier battery."""
from __future__ import annotations

import argparse
import hashlib
import json
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
import sklearn
from sklearn.ensemble import ExtraTreesClassifier

import training.audit_gross9_residual_recovery_extratrees_marginal as runtime
import training.portfolio_opt_added_alpha_update as portfolio
from preprocessing.binance_aux_features import (
    attach_binance_um_aux_frames,
    normalise_funding_history_frame,
    normalise_premium_index_frame,
)
from training.audit_gross9_oi_pullback_marginal import _atomic_json, _sha256
from training.search_inventory_conservation_residual_alpha import (
    _attach_delayed_metrics,
    _fit_linear_residual,
    _frame_hash,
    _read_premium_before,
    _rolling_z,
)
from training.search_positioning_hgb_path_alpha import _read_before
from training.search_spot_perp_absorption_alpha import (
    SpotPerpConfig,
    _load_market_spot,
    build_features as build_spot_perp_features,
)


PREREGISTRATION = Path(
    "results/"
    "residual_direction_classifier_marginal_preregistration_2026-07-28.json"
)
OUTPUT = Path(
    "results/"
    "gross9_residual_direction_classifier_marginal_pre2025_2026-07-28.json"
)
DOCS = Path(
    "docs/"
    "gross9-residual-direction-classifier-marginal-pre2025-2026-07-28.md"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "321933024ae00aa74753117b3c87411c4cba18a4818f7f5acda836d8248d3383"
)
EXPECTED_SKLEARN_VERSION = "1.7.2"
SELECTION_CUTOFF = "2025-01-01"
HOLDS = (72, 144)
COORDINATION_MODES = runtime.COORDINATION_MODES
WEIGHTS = runtime.WEIGHTS
SEEDS = runtime.SEEDS
SPOT_FEATURES = runtime.FEATURE_COLUMNS[:8]
INVENTORY_FEATURES = runtime.FEATURE_COLUMNS[8:]
FEATURE_COLUMNS = runtime.FEATURE_COLUMNS
SELECTION_SPLITS = runtime.SELECTION_SPLITS
FIT_START = runtime.FIT_START
CANDIDATE_NAMES = tuple(
    f"residual_direction_h{hold:03d}_{mode}"
    for hold in HOLDS
    for mode in COORDINATION_MODES
)
INPUT_KEYS = runtime.INPUT_KEYS + ("residual_runtime",)


@dataclass(frozen=True)
class Config(runtime.Config):
    preregistration: str = str(PREREGISTRATION)
    output: str = str(OUTPUT)
    docs_output: str = str(DOCS)


def load_preregistration(path: str | Path) -> dict[str, Any]:
    observed = _sha256(path)
    if observed != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError(
            f"preregistration hash drifted: {observed} != "
            f"{EXPECTED_PREREGISTRATION_SHA256}"
        )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("name") != (
        "residual_direction_confidence_classifier_gross9_marginal_battery"
    ):
        raise RuntimeError("unexpected RDC preregistration")
    if payload.get("physical_selection_cutoff") != SELECTION_CUTOFF:
        raise RuntimeError("RDC cutoff drifted")
    if tuple(payload["feature_contract"]["columns"]) != FEATURE_COLUMNS:
        raise RuntimeError("RDC feature contract drifted")
    learner = payload["learner_contract"]
    expected = {
        "estimator": "sklearn.ensemble.ExtraTreesClassifier",
        "sklearn_version": EXPECTED_SKLEARN_VERSION,
        "n_estimators": 256,
        "max_depth": 4,
        "min_samples_leaf": 128,
        "max_features": 0.75,
        "bootstrap": False,
        "class_weight": "balanced",
        "score_quantile": 0.9,
        "anchor_stride_bars": 12,
        "entry_delay_bars": 1,
        "unit_leverage": 0.5,
    }
    for key, value in expected.items():
        if learner.get(key) != value:
            raise RuntimeError(f"RDC learner contract drifted: {key}")
    if tuple(learner["seeds"]) != SEEDS:
        raise RuntimeError("RDC seed contract drifted")
    universe = payload["candidate_universe"]
    if tuple(universe["hold_bars"]) != HOLDS:
        raise RuntimeError("RDC holds drifted")
    if tuple(row["name"] for row in universe["coordination_modes"]) != (
        COORDINATION_MODES
    ):
        raise RuntimeError("RDC coordination modes drifted")
    if tuple(float(value) for value in universe["candidate_weight_grid"]) != (
        WEIGHTS
    ):
        raise RuntimeError("RDC weight grid drifted")
    if int(universe["portfolio_cells"]) != 24:
        raise RuntimeError("RDC portfolio-cell count drifted")
    boundary = payload["prior_battery_boundary"]
    if boundary.get("familywise_stop", "").find("final battery") < 0:
        raise RuntimeError("RDC familywise stop drifted")
    future = payload["future_veto_contract"]
    if future.get("future_can_rerank") is not False:
        raise RuntimeError("RDC future reranking enabled")
    if future.get("future_can_repair") is not False:
        raise RuntimeError("RDC future repair enabled")
    return payload


def configured_inputs(cfg: Config) -> dict[str, str]:
    configured = runtime.configured_inputs(cfg)
    configured["residual_runtime"] = (
        "training/audit_gross9_residual_recovery_extratrees_marginal.py"
    )
    return configured


def validate_inputs(
    cfg: Config,
    preregistration: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    configured = configured_inputs(cfg)
    provenance = preregistration["input_provenance"]
    if tuple(configured) != INPUT_KEYS or tuple(provenance) != INPUT_KEYS:
        raise RuntimeError("RDC input order drifted")
    records: dict[str, dict[str, Any]] = {}
    for name in INPUT_KEYS:
        configured_path = Path(configured[name]).resolve(strict=True)
        frozen_path = Path(provenance[name]["path"]).resolve(strict=True)
        if configured_path != frozen_path:
            raise RuntimeError(f"RDC input path drifted for {name}")
        observed = _sha256(configured_path)
        expected = str(provenance[name]["sha256"])
        if observed != expected:
            raise RuntimeError(
                f"RDC input hash drifted for {name}: "
                f"{observed} != {expected}"
            )
        records[name] = {
            "path": str(configured_path),
            "sha256": observed,
        }
    return records


def _load_inventory_source(
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
        raise RuntimeError("RDC inventory source breached cutoff")
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
        raise RuntimeError("RDC delayed OI source breached cutoff")
    audit = {
        "source_prefix_hashes": {
            "market_with_oi": _frame_hash(market_raw),
            "metrics": _frame_hash(metrics_raw),
            "funding": _frame_hash(funding_raw),
            "premium": _frame_hash(premium_raw),
        },
        "last_market": str(pd.to_datetime(market["date"]).max()),
        "last_delayed_metric": (
            str(source_times.max()) if len(source_times) else None
        ),
        "metrics_delay_bars": int(cfg.metrics_delay_bars),
    }
    return market, audit


def inventory_fit_mask(
    dates: pd.Series,
    available: np.ndarray,
    fit_end_exclusive: str | pd.Timestamp,
) -> np.ndarray:
    end = pd.Timestamp(fit_end_exclusive)
    return np.asarray(
        (dates >= FIT_START) & (dates < end) & np.asarray(available, bool),
        dtype=bool,
    )


def build_fold_inventory_features(
    market: pd.DataFrame,
    *,
    fit_end_exclusive: str | pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dates = pd.to_datetime(market["date"]).reset_index(drop=True)
    close = pd.to_numeric(market["close"], errors="coerce")
    log_close = np.log(close.where(close > 0.0))
    oi = np.log(
        pd.to_numeric(
            market["sum_open_interest"], errors="coerce"
        ).where(lambda value: value > 0.0)
    )
    funding = pd.to_numeric(market.get("funding_rate"), errors="coerce")
    premium = pd.to_numeric(market.get("premium_index"), errors="coerce")
    premium_change = pd.to_numeric(
        market.get("premium_index_change"), errors="coerce"
    )
    available = (
        pd.to_numeric(
            market.get("positioning_available", 0.0), errors="coerce"
        ).fillna(0.0)
        > 0.5
    ) & (
        pd.to_numeric(
            market.get("funding_available", 0.0), errors="coerce"
        ).fillna(0.0)
        > 0.5
    ) & (
        pd.to_numeric(
            market.get("premium_available", 0.0), errors="coerce"
        ).fillna(0.0)
        > 0.5
    )
    fit = inventory_fit_mask(
        dates, available.to_numpy(bool), fit_end_exclusive
    )
    carry = (
        _rolling_z(funding, 2016).fillna(0.0)
        + _rolling_z(premium, 2016).fillna(0.0)
        + 0.5 * _rolling_z(premium_change, 2016).fillna(0.0)
    )
    columns: dict[str, pd.Series] = {}
    meta: dict[str, Any] = {
        "fit_end_exclusive": str(pd.Timestamp(fit_end_exclusive).date()),
        "fit_rows": int(fit.sum()),
        "windows": {},
    }
    for window in (48, 144, 288):
        oi_change = oi - oi.shift(window)
        price_return = log_close - log_close.shift(window)
        regressors = pd.DataFrame(
            {
                "abs_price_ret": price_return.abs(),
                "price_ret": price_return,
                "funding": funding,
                "premium": premium,
                "premium_change": premium_change,
                "abs_carry": carry.abs(),
            }
        )
        beta, residual = _fit_linear_residual(
            oi_change.to_numpy(float),
            regressors.to_numpy(float),
            fit,
        )
        residual_series = pd.Series(residual, index=market.index)
        columns[f"resid_z{window}"] = _rolling_z(
            residual_series, 2016
        ).where(available)
        meta["windows"][str(window)] = {
            "beta_columns": ["intercept", *regressors.columns.tolist()],
            "beta": beta.tolist(),
        }
    columns["carry_z"] = carry.where(available)
    frame = (
        pd.DataFrame(columns, index=market.index)
        .replace([np.inf, -np.inf], np.nan)
        .loc[:, list(INVENTORY_FEATURES)]
        .astype(np.float64)
    )
    meta["feature_hash"] = _frame_hash(
        pd.concat([dates.rename("date"), frame], axis=1)
    )
    meta["finite_rows"] = int(np.isfinite(frame).all(axis=1).sum())
    return frame, meta


def load_feature_sources(
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
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
    spot = build_spot_perp_features(spot_market).loc[
        :, list(SPOT_FEATURES)
    ]
    inventory_market, inventory_audit = _load_inventory_source(cfg)
    spot_dates = pd.to_datetime(spot_market["date"]).reset_index(drop=True)
    inventory_dates = pd.to_datetime(
        inventory_market["date"]
    ).reset_index(drop=True)
    if not spot_dates.equals(inventory_dates):
        raise RuntimeError("RDC spot and inventory clocks differ")
    intervals = spot_dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise RuntimeError("RDC source clock is not complete 5m")
    audit = {
        "physical_cutoff": cfg.cutoff,
        "rows": int(len(spot_market)),
        "first": str(spot_dates.iloc[0]),
        "last": str(spot_dates.iloc[-1]),
        "spot_prefix_hashes": spot_hashes,
        "inventory_source": inventory_audit,
        "spot_feature_hash": _frame_hash(
            pd.concat([spot_dates.rename("date"), spot], axis=1)
        ),
    }
    return spot_market, spot.reset_index(drop=True), inventory_market, audit


def align_fold_features(
    market: pd.DataFrame,
    source_dates: pd.Series,
    spot_features: pd.DataFrame,
    inventory_features: pd.DataFrame,
) -> pd.DataFrame:
    source = pd.concat(
        [
            pd.to_datetime(source_dates).rename("date").reset_index(drop=True),
            spot_features.reset_index(drop=True),
            inventory_features.reset_index(drop=True),
        ],
        axis=1,
    ).set_index("date")
    if source.index.has_duplicates:
        raise RuntimeError("RDC source feature dates are duplicated")
    aligned = source.reindex(
        pd.DatetimeIndex(pd.to_datetime(market["date"]))
    ).reset_index(drop=True)
    if tuple(aligned.columns) != FEATURE_COLUMNS:
        raise RuntimeError("RDC feature order drifted")
    return aligned.astype(np.float64)


def empirical_confidence(
    p_long_by_seed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(p_long_by_seed, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("p_long must have shape seeds x rows")
    mean_long = values.mean(axis=0)
    uncertainty = values.std(axis=0, ddof=0)
    choose_long = mean_long >= 0.5
    score = np.maximum(mean_long, 1.0 - mean_long) - 0.5 * uncertainty
    side = np.where(choose_long, 1, -1).astype(np.int8)
    return score, side


def calibration_threshold(scores: np.ndarray) -> float:
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 1_000:
        raise RuntimeError("insufficient RDC calibration scores")
    return float(np.quantile(finite, 0.90, method="linear"))


def _fit_predict_classifier(
    matrix: np.ndarray,
    labels: np.ndarray,
    fit: np.ndarray,
    predict: np.ndarray,
    preregistration: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if int(fit.sum()) < 1_000:
        raise RuntimeError(f"insufficient RDC fit rows: {int(fit.sum())}")
    if not predict.any():
        raise RuntimeError("empty RDC prediction fold")
    x_fit = np.asarray(matrix[fit], dtype=np.float64)
    y_fit = np.asarray(labels[fit], dtype=np.int8)
    x_predict = np.asarray(matrix[predict], dtype=np.float64)
    if set(np.unique(y_fit).tolist()) != {0, 1}:
        raise RuntimeError("RDC fit does not contain both classes")
    if not np.isfinite(x_fit).all() or not np.isfinite(x_predict).all():
        raise RuntimeError("RDC admitted a non-finite feature row")
    learner = preregistration["learner_contract"]
    predictions: list[np.ndarray] = []
    model_hashes: list[str] = []
    for seed in SEEDS:
        model = ExtraTreesClassifier(
            n_estimators=int(learner["n_estimators"]),
            max_depth=int(learner["max_depth"]),
            min_samples_leaf=int(learner["min_samples_leaf"]),
            max_features=float(learner["max_features"]),
            bootstrap=False,
            class_weight=str(learner["class_weight"]),
            random_state=int(seed),
            n_jobs=-1,
        ).fit(x_fit, y_fit)
        matches = np.flatnonzero(model.classes_ == 1)
        if len(matches) != 1:
            raise RuntimeError("RDC p_long class column is ambiguous")
        predictions.append(model.predict_proba(x_predict)[:, int(matches[0])])
        model_hashes.append(
            hashlib.sha256(
                pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            ).hexdigest()
        )
    stacked = np.stack(predictions, axis=0)
    return stacked, {
        "fit_rows": int(fit.sum()),
        "fit_class_counts": {
            "short_0": int(np.sum(y_fit == 0)),
            "long_1": int(np.sum(y_fit == 1)),
        },
        "predict_rows": int(predict.sum()),
        "model_hashes": model_hashes,
        "prediction_hash": runtime._array_hash(stacked),
    }


def annual_classifier_predictions(
    market: pd.DataFrame,
    spot_features: pd.DataFrame,
    inventory_market: pd.DataFrame,
    preregistration: dict[str, Any],
    *,
    hold: int,
    cfg: Config,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    dates = pd.to_datetime(market["date"]).reset_index(drop=True)
    source_dates = pd.to_datetime(
        inventory_market["date"]
    ).reset_index(drop=True)
    anchors = np.arange(
        143,
        len(market) - int(hold) - 1,
        int(cfg.stride_bars),
        dtype=np.int64,
    )
    targets = runtime.exact_no_stop_targets(
        market,
        anchors,
        hold=hold,
        cost_rate=cfg.cost_rate,
        leverage=cfg.leverage,
    )
    labels = (targets[:, 0] >= targets[:, 2]).astype(np.int8)
    signal_dates = dates.iloc[anchors].reset_index(drop=True)
    exit_dates = dates.iloc[anchors + 1 + int(hold)].reset_index(drop=True)
    outputs: dict[int, dict[str, np.ndarray]] = {}
    meta: dict[str, Any] = {
        "hold_bars": int(hold),
        "anchors": int(len(anchors)),
        "label_hash": runtime._array_hash(labels),
        "annual_models": {},
    }
    for year in (2022, 2023, 2024):
        inventory, feature_meta = build_fold_inventory_features(
            inventory_market,
            fit_end_exclusive=f"{year}-01-01",
        )
        fold_features = align_fold_features(
            market,
            source_dates,
            spot_features,
            inventory,
        )
        matrix = fold_features.to_numpy(np.float64)
        finite_features = np.isfinite(matrix[anchors]).all(axis=1)
        finite_targets = np.isfinite(targets).all(axis=1)
        fit, predict = runtime.annual_masks(
            signal_dates,
            exit_dates,
            finite_features & finite_targets,
            finite_features,
            prediction_year=year,
        )
        p_long, model_meta = _fit_predict_classifier(
            matrix[anchors],
            labels,
            fit,
            predict,
            preregistration,
        )
        score, side = empirical_confidence(p_long)
        outputs[year] = {
            "anchors": anchors[predict],
            "score": score,
            "side": side,
        }
        model_meta.update(
            {
                "fit_end_exclusive": f"{year}-01-01",
                "prediction_start": f"{year}-01-01",
                "prediction_end_exclusive": f"{year + 1}-01-01",
                "feature_state": feature_meta,
                "score_hash": runtime._array_hash(score),
                "side_hash": runtime._array_hash(side),
            }
        )
        meta["annual_models"][str(year)] = model_meta
    for year in (2023, 2024):
        threshold = calibration_threshold(outputs[year - 1]["score"])
        outputs[year]["threshold"] = np.asarray(threshold)
        meta["annual_models"][str(year)]["calibration"] = {
            "source_prediction_year": int(year - 1),
            "population": (
                "all finite hourly split-contained anchors before "
                "coordination and non-overlap"
            ),
            "scores": int(len(outputs[year - 1]["score"])),
            "quantile": 0.9,
            "method": "linear",
            "threshold": threshold,
            "admission_operator": ">=",
        }
    return outputs, meta


def build_candidate_masks(
    market: pd.DataFrame,
    spot_features: pd.DataFrame,
    inventory_market: pd.DataFrame,
    preregistration: dict[str, Any],
    flat: np.ndarray,
    drawdown: np.ndarray,
    cfg: Config,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    model_meta: dict[str, Any] = {}
    for hold in HOLDS:
        annual, hold_meta = annual_classifier_predictions(
            market,
            spot_features,
            inventory_market,
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
            name = f"residual_direction_h{hold:03d}_{mode}"
            long_active = base_long & gate
            short_active = base_short & gate
            candidates[name] = {
                "hold": int(hold),
                "mode": mode,
                "long_active": long_active,
                "short_active": short_active,
                "activation_hash": runtime._array_hash(
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
        raise RuntimeError("RDC candidate order drifted")
    return candidates, model_meta


def _install_candidate_universe() -> None:
    names = list(portfolio.SLEEVES)
    for name in CANDIDATE_NAMES:
        if name not in names:
            names.append(name)
    portfolio.SLEEVES = tuple(names)
    portfolio.FAMILIES = {
        **portfolio.FAMILIES,
        **{name: "residual_direction" for name in CANDIDATE_NAMES},
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
        "# Gross9 residual-direction classifier pre-2025 marginal battery",
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
            "## Classifier admission",
            "",
            "| hold | 2023 threshold from 2022 | 2024 threshold from 2023 | unrestricted active anchors |",
            "|---:|---:|---:|---:|",
        ]
    )
    for hold in HOLDS:
        models = payload["model_meta"][str(hold)]["annual_models"]
        candidate = payload["candidates"][
            f"residual_direction_h{hold:03d}_unrestricted"
        ]
        lines.append(
            "| {hold} | {t23:.6f} | {t24:.6f} | {active} |".format(
                hold=hold,
                t23=models["2023"]["calibration"]["threshold"],
                t24=models["2024"]["calibration"]["threshold"],
                active=candidate["raw_active_anchors"],
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Every inventory residual beta is fold-local and fitted strictly before its scoring year.",
            "- OI source bars are delayed one complete 5m bar; spot-perp regressions use history ending at t-1.",
            "- 2023 and 2024 are both pre-2025 selection. No 2025/2026 candidate result was opened.",
            "- This is the last battery for the exact 12-feature residual family. Even a survivor remains research shadow until the immutable 90-day prospective gate passes.",
            "",
        ]
    )
    return "\n".join(lines)


def run_pre2025(cfg: Config) -> dict[str, Any]:
    preregistration = load_preregistration(cfg.preregistration)
    if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
        raise RuntimeError(
            f"sklearn version drifted: {sklearn.__version__} != "
            f"{EXPECTED_SKLEARN_VERSION}"
        )
    input_identity = validate_inputs(cfg, preregistration)
    _install_candidate_universe()
    source_market, spot_features, inventory_market, feature_audit = (
        load_feature_sources(cfg)
    )
    market, masks, base_events, gross9_meta = runtime.build_gross9_context(
        cfg
    )
    if not pd.to_datetime(source_market["date"]).equals(
        pd.to_datetime(inventory_market["date"])
    ):
        raise RuntimeError("RDC source-market clocks differ")
    anchor = json.loads(
        Path(cfg.gross9_pre2025_anchor).read_text(encoding="utf-8")
    )
    baseline_weights = {
        str(name): float(weight) for name, weight in anchor["weights"].items()
    }
    flat, drawdown, state_meta = runtime.gross9_state(
        market, masks, base_events, baseline_weights
    )
    candidates, model_meta = build_candidate_masks(
        market,
        spot_features,
        inventory_market,
        preregistration,
        flat,
        drawdown,
        cfg,
    )
    normal_events = list(base_events)
    normal_counts = runtime.append_candidates(
        normal_events,
        market,
        masks,
        candidates,
        cost_rate=cfg.cost_rate,
    )
    normal_arrays = portfolio.split_arrays(normal_events, market, masks)
    stress_events = list(base_events)
    stress_counts = runtime.append_candidates(
        stress_events,
        market,
        masks,
        candidates,
        cost_rate=cfg.stress_cost_rate,
    )
    stress_arrays = portfolio.split_arrays(stress_events, market, masks)
    if normal_counts != stress_counts:
        raise RuntimeError("RDC stress changed executable schedules")
    for name in CANDIDATE_NAMES:
        for split in SELECTION_SPLITS:
            if not np.array_equal(
                normal_arrays[split]["entry_positions"][name],
                stress_arrays[split]["entry_positions"][name],
            ):
                raise RuntimeError(
                    f"RDC stress changed entries for {name}/{split}"
                )
    baseline = runtime.validate_baseline(normal_arrays, anchor)
    standalone = {
        name: runtime.standalone_metrics(normal_arrays, name)
        for name in CANDIDATE_NAMES
    }
    stressed_standalone = {
        name: runtime.standalone_metrics(stress_arrays, name)
        for name in CANDIDATE_NAMES
    }
    jaccards = {
        name: runtime.entry_jaccards(
            normal_arrays, market, baseline_weights, name
        )
        for name in CANDIDATE_NAMES
    }
    rows: list[dict[str, Any]] = []
    for name in CANDIDATE_NAMES:
        for weight in WEIGHTS:
            combined_weights, control_weights = runtime.same_gross_weights(
                baseline_weights, name, float(weight)
            )
            stats = {
                split: runtime._metric(
                    normal_arrays, split, combined_weights
                )
                for split in SELECTION_SPLITS
            }
            comparator = {
                split: runtime._metric(
                    normal_arrays, split, control_weights
                )
                for split in SELECTION_SPLITS
            }
            stressed_stats = {
                split: runtime._metric(
                    stress_arrays, split, combined_weights
                )
                for split in SELECTION_SPLITS
            }
            rows.append(
                runtime.selection_row(
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
        raise RuntimeError(f"RDC tested-cell drift: {len(rows)}")
    rows = runtime.rank_rows(rows)
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
        "phase": "pre2025_residual_direction_classifier_gross9_marginal",
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
            "open_frozen_top1_historical_future_veto"
            if frozen_top1 is not None
            else "reject_and_close_residual_family"
        ),
        "future_opened": False,
        "future_can_rerank": False,
        "future_can_repair": False,
        "family_closed_if_rejected": True,
        "freeze_hash": runtime._json_hash(freeze),
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
