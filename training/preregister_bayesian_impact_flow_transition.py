"""Support-only preregistration for BIFT.

BIFT detects abrupt changes in the relation between public trade flow, price
impact, and event intensity.  A fixed three-hour transition then distinguishes
flow propagation from price absorption.  This module deliberately contains no
future-return labels, trade PnL, or backtest statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln

from training.preregister_metaorder_fragmentation_impact_curvature import (
    Config as SourceConfig,
    load_causal_frame,
    nonoverlapping_schedule,
)


SELECTION_END = pd.Timestamp("2024-01-01")
SUPPORT_CALIBRATION_GRID = (0.90, 0.925, 0.95, 0.975, 0.99)
BOCPD_FEATURE_COLUMNS = (
    "flow_imbalance",
    "impact_alignment",
    "log_event_count",
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_bayesian_impact_flow_transition.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/bayesian-impact-flow-transition-preregistration-2026-07-14.md"
)


@dataclass(frozen=True)
class Config:
    output: str = (
        "results/bayesian_impact_flow_transition_support_2026-07-14.json"
    )
    change_quantile: float = 0.925
    minimum_abs_flow_imbalance: float = 0.02
    robust_baseline_hours: int = 720
    robust_min_periods: int = 168
    change_baseline_hours: int = 4_320
    change_min_periods: int = 720
    hazard_lambda_hours: float = 168.0
    max_run_length_hours: int = 672
    short_run_horizon_hours: int = 6
    confirmation_hours: int = 3
    hold_bars: int = 144
    minimum_hour_bars: int = 12
    minimum_nonoverlap_total: int = 250
    minimum_nonoverlap_per_year: int = 40
    minimum_nonoverlap_per_2023_half: int = 30
    minimum_side_share: float = 0.25
    minimum_branch_share: float = 0.25


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_config(cfg: Config) -> None:
    if not 0.0 <= cfg.change_quantile <= 1.0:
        raise ValueError("change quantile must be in [0, 1]")
    if cfg.minimum_abs_flow_imbalance < 0.0:
        raise ValueError("minimum flow imbalance cannot be negative")
    if not 1 <= cfg.robust_min_periods <= cfg.robust_baseline_hours:
        raise ValueError("robust baseline periods are invalid")
    if not 1 <= cfg.change_min_periods <= cfg.change_baseline_hours:
        raise ValueError("change baseline periods are invalid")
    if cfg.hazard_lambda_hours <= 1.0 or cfg.max_run_length_hours < 2:
        raise ValueError("BOCPD hazard or run-length cap is invalid")
    if cfg.confirmation_hours < 1 or cfg.hold_bars < 1:
        raise ValueError("confirmation and holding periods must be positive")
    if cfg.minimum_hour_bars != 12:
        raise ValueError("BIFT requires twelve 5m bars per completed hour")


def aggregate_hourly(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Aggregate the verified 5m source into causal, completed-hour features."""
    _validate_config(cfg)
    required = {
        "date",
        "open",
        "close",
        "quote_notional",
        "signed_quote_notional",
        "agg_trade_count",
        "signed_event_imbalance",
        "quarantined",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"BIFT source columns are missing: {missing}")

    working = frame.loc[:, list(required)].copy()
    working["hour"] = pd.to_datetime(working["date"]).dt.floor("h")
    working["weighted_event_imbalance"] = (
        working["signed_event_imbalance"].astype(float)
        * working["agg_trade_count"].astype(float)
    )
    grouped = working.groupby("hour", sort=True, observed=True)
    hourly = grouped.agg(
        date=("date", "max"),
        first_date=("date", "min"),
        bar_count=("date", "size"),
        quarantined_bars=("quarantined", "sum"),
        open=("open", "first"),
        close=("close", "last"),
        quote_notional=("quote_notional", "sum"),
        signed_quote_notional=("signed_quote_notional", "sum"),
        agg_trade_count=("agg_trade_count", "sum"),
        weighted_event_imbalance=("weighted_event_imbalance", "sum"),
    ).reset_index()

    expected_first = hourly["hour"]
    expected_last = hourly["hour"] + pd.Timedelta(minutes=55)
    hourly["clean"] = (
        hourly["bar_count"].eq(cfg.minimum_hour_bars)
        & hourly["quarantined_bars"].eq(0)
        & pd.to_datetime(hourly["first_date"]).eq(expected_first)
        & pd.to_datetime(hourly["date"]).eq(expected_last)
        & hourly["quote_notional"].gt(0.0)
        & hourly["agg_trade_count"].gt(0.0)
        & hourly["open"].gt(0.0)
        & hourly["close"].gt(0.0)
    )
    hourly["flow_imbalance"] = hourly["signed_quote_notional"].divide(
        hourly["quote_notional"].replace(0.0, np.nan)
    )
    hourly["event_imbalance"] = hourly["weighted_event_imbalance"].divide(
        hourly["agg_trade_count"].replace(0.0, np.nan)
    )
    hourly["price_return"] = np.log(
        hourly["close"].astype(float) / hourly["open"].astype(float)
    )
    hourly["impact_alignment"] = (
        np.sign(hourly["flow_imbalance"])
        * hourly["price_return"]
        / np.sqrt(hourly["flow_imbalance"].abs().clip(lower=0.01))
    )
    hourly["log_event_count"] = np.log1p(hourly["agg_trade_count"].astype(float))
    return hourly


def lagged_robust_zscore(
    values: pd.Series,
    clean: pd.Series,
    *,
    window: int,
    minimum: int,
) -> pd.Series:
    """Return a prefix-invariant robust score from strictly prior clean hours."""
    if not 1 <= minimum <= window:
        raise ValueError("robust score periods are invalid")
    prior = values.astype(float).where(clean.astype(bool)).shift(1)
    center = prior.rolling(window, min_periods=minimum).median()
    mad = (prior - center).abs().rolling(window, min_periods=minimum).median()
    scale = 1.4826 * mad.replace(0.0, np.nan)
    return ((values.astype(float) - center) / scale).clip(-12.0, 12.0)


def _student_t_log_predictive(
    observation: np.ndarray,
    mean: np.ndarray,
    kappa: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    degrees = 2.0 * alpha
    scale2 = beta * (kappa[:, None] + 1.0) / (alpha * kappa[:, None])
    centered2 = (observation[None, :] - mean) ** 2
    per_dimension = (
        gammaln((degrees + 1.0) / 2.0)
        - gammaln(degrees / 2.0)
        - 0.5 * (np.log(degrees * np.pi) + np.log(scale2))
        - 0.5
        * (degrees + 1.0)
        * np.log1p(centered2 / (degrees * scale2))
    )
    return per_dimension.sum(axis=1)


def _posterior_update(
    observation: np.ndarray,
    mean: np.ndarray,
    kappa: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    next_kappa = kappa + 1.0
    next_mean = (
        kappa[:, None] * mean + observation[None, :]
    ) / next_kappa[:, None]
    next_alpha = alpha + 0.5
    next_beta = beta + 0.5 * (
        kappa[:, None]
        * (observation[None, :] - mean) ** 2
        / next_kappa[:, None]
    )
    return next_mean, next_kappa, next_alpha, next_beta


def bocpd_student_t(
    observations: np.ndarray,
    *,
    hazard_lambda: float,
    max_run_length: int,
    short_run_horizon: int,
    prior_kappa: float = 0.1,
    prior_alpha: float = 2.0,
    prior_beta: float = 1.0,
) -> dict[str, np.ndarray]:
    """Causal Adams-MacKay run-length recursion with independent dimensions."""
    values = np.asarray(observations, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("observations must be a non-empty matrix")
    if not np.isfinite(values).all():
        raise ValueError("BOCPD observations must be finite")
    if hazard_lambda <= 1.0 or max_run_length < 2:
        raise ValueError("invalid BOCPD hazard or run-length cap")

    dimensions = values.shape[1]
    hazard = 1.0 / float(hazard_lambda)
    weights = np.array([1.0])
    mean = np.zeros((1, dimensions), dtype=float)
    kappa = np.array([prior_kappa], dtype=float)
    alpha = np.full((1, dimensions), prior_alpha, dtype=float)
    beta = np.full((1, dimensions), prior_beta, dtype=float)

    expected_run = np.empty(len(values), dtype=float)
    short_mass = np.empty(len(values), dtype=float)
    run_drop = np.empty(len(values), dtype=float)
    surprise = np.empty(len(values), dtype=float)
    previous_expected = 0.0

    prior_mean = np.zeros((1, dimensions), dtype=float)
    prior_kappa_array = np.array([prior_kappa], dtype=float)
    prior_alpha_array = np.full((1, dimensions), prior_alpha, dtype=float)
    prior_beta_array = np.full((1, dimensions), prior_beta, dtype=float)

    for position, observation in enumerate(values):
        log_predictive = _student_t_log_predictive(
            observation, mean, kappa, alpha, beta
        )
        log_joint = np.log(np.maximum(weights, 1e-300)) + log_predictive
        offset = float(np.max(log_joint))
        joint = np.exp(log_joint - offset)
        surprise[position] = -(offset + np.log(np.sum(joint)))

        next_weights = np.r_[
            hazard * float(np.sum(joint)),
            (1.0 - hazard) * joint,
        ]
        reset = _posterior_update(
            observation,
            prior_mean,
            prior_kappa_array,
            prior_alpha_array,
            prior_beta_array,
        )
        growth = _posterior_update(
            observation, mean, kappa, alpha, beta
        )
        next_mean = np.vstack([reset[0], growth[0]])
        next_kappa = np.r_[reset[1], growth[1]]
        next_alpha = np.vstack([reset[2], growth[2]])
        next_beta = np.vstack([reset[3], growth[3]])

        keep = min(len(next_weights), max_run_length + 1)
        weights = next_weights[:keep]
        weights /= np.sum(weights)
        mean = next_mean[:keep]
        kappa = next_kappa[:keep]
        alpha = next_alpha[:keep]
        beta = next_beta[:keep]

        run_axis = np.arange(keep, dtype=float)
        current_expected = float(weights @ run_axis)
        expected_run[position] = current_expected
        short_mass[position] = float(
            weights[: min(short_run_horizon + 1, keep)].sum()
        )
        expected_without_reset = previous_expected + 1.0
        run_drop[position] = max(
            0.0, expected_without_reset - current_expected
        ) / max(expected_without_reset, 1.0)
        previous_expected = current_expected

    return {
        "expected_run": expected_run,
        "short_mass": short_mass,
        "run_drop": run_drop,
        "surprise": surprise,
    }


def segmented_bocpd(
    observations: np.ndarray,
    available: np.ndarray,
    cfg: Config,
) -> dict[str, np.ndarray]:
    """Reset the posterior at every unavailable or quarantined hour."""
    values = np.asarray(observations, dtype=float)
    valid = np.asarray(available, dtype=bool)
    if values.ndim != 2 or len(values) != len(valid):
        raise ValueError("segmented BOCPD inputs have incompatible shapes")
    if np.any(valid & ~np.isfinite(values).all(axis=1)):
        raise ValueError("available BOCPD rows must be finite")

    output = {
        name: np.full(len(values), np.nan, dtype=float)
        for name in ("expected_run", "short_mass", "run_drop", "surprise")
    }
    boundaries = np.flatnonzero(
        np.r_[True, valid[1:] != valid[:-1], True]
    )
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if not valid[start]:
            continue
        segment = bocpd_student_t(
            values[start:end],
            hazard_lambda=cfg.hazard_lambda_hours,
            max_run_length=cfg.max_run_length_hours,
            short_run_horizon=cfg.short_run_horizon_hours,
        )
        for name in output:
            output[name][start:end] = segment[name]
    return output


def build_hourly_diagnostics(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    hourly = aggregate_hourly(frame, cfg)
    scores: list[pd.Series] = []
    for column in BOCPD_FEATURE_COLUMNS:
        score = lagged_robust_zscore(
            hourly[column],
            hourly["clean"],
            window=cfg.robust_baseline_hours,
            minimum=cfg.robust_min_periods,
        )
        hourly[f"{column}_robust_z"] = score
        scores.append(score)
    observations = np.column_stack(scores)
    detector_available = (
        hourly["clean"].to_numpy(bool)
        & np.isfinite(observations).all(axis=1)
    )
    detector = segmented_bocpd(observations, detector_available, cfg)
    hourly["detector_available"] = detector_available
    for name, values in detector.items():
        hourly[name] = values
    return hourly


def _lagged_quantile(
    values: pd.Series,
    available: pd.Series,
    *,
    quantile: float,
    window: int,
    minimum: int,
) -> pd.Series:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if not 1 <= minimum <= window:
        raise ValueError("quantile baseline periods are invalid")
    return (
        values.where(available.astype(bool))
        .shift(1)
        .rolling(window, min_periods=minimum)
        .quantile(quantile)
    )


def classify_candidates(
    hourly: pd.DataFrame,
    cfg: Config,
    *,
    change_quantile: float | None = None,
) -> pd.DataFrame:
    """Apply the frozen setup, confirmation, branch, and side rules."""
    quantile = cfg.change_quantile if change_quantile is None else change_quantile
    available = hourly["detector_available"].astype(bool)
    drop_baseline = _lagged_quantile(
        hourly["run_drop"],
        available,
        quantile=quantile,
        window=cfg.change_baseline_hours,
        minimum=cfg.change_min_periods,
    )
    surprise_baseline = _lagged_quantile(
        hourly["surprise"],
        available,
        quantile=quantile,
        window=cfg.change_baseline_hours,
        minimum=cfg.change_min_periods,
    )
    setup = (
        hourly["clean"].astype(bool)
        & available
        & hourly["flow_imbalance"].abs().ge(
            cfg.minimum_abs_flow_imbalance
        )
        & hourly["run_drop"].ge(drop_baseline)
        & hourly["surprise"].ge(surprise_baseline)
    )

    confirmation = cfg.confirmation_hours
    reference = np.sign(hourly["flow_imbalance"]).shift(confirmation)
    post_flow = hourly["flow_imbalance"].rolling(
        confirmation, min_periods=confirmation
    ).sum()
    post_price = np.log(
        hourly["close"].astype(float)
        / hourly["close"].astype(float).shift(confirmation)
    )
    clean_transition = (
        hourly["clean"]
        .astype(np.int16)
        .rolling(confirmation + 1, min_periods=confirmation + 1)
        .sum()
        .eq(confirmation + 1)
    )
    origin_setup = setup.shift(confirmation, fill_value=False)
    persistent_flow = (reference * post_flow).gt(0.0)
    resolved_price = post_price.notna() & post_price.ne(0.0)
    candidate = (
        origin_setup & clean_transition & persistent_flow & resolved_price
    )
    propagation = candidate & (reference * post_price).gt(0.0)
    absorption = candidate & ~propagation

    side = pd.Series(0, index=hourly.index, dtype=np.int8)
    side.loc[propagation] = reference.loc[propagation].astype(np.int8)
    side.loc[absorption] = -reference.loc[absorption].astype(np.int8)
    branch = pd.Series("none", index=hourly.index, dtype="string")
    branch.loc[propagation] = "propagation"
    branch.loc[absorption] = "absorption"
    origin_position = pd.Series(
        np.arange(len(hourly), dtype=np.int64) - confirmation,
        index=hourly.index,
        dtype=np.int64,
    )

    return pd.DataFrame(
        {
            "date": hourly["date"],
            "setup": setup,
            "candidate": candidate,
            "origin_hour_position": origin_position,
            "reference_direction": reference.fillna(0.0).astype(np.int8),
            "post_flow_sum": post_flow,
            "post_price_return": post_price,
            "run_drop": hourly["run_drop"],
            "run_drop_baseline": drop_baseline,
            "surprise": hourly["surprise"],
            "surprise_baseline": surprise_baseline,
            "side": side,
            "branch": branch,
            "hold_bars": np.where(side.ne(0), cfg.hold_bars, 0).astype(
                np.int16
            ),
        }
    )


def project_to_five_minute(
    hourly_state: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Place each completed-hour decision on that hour's final 5m bar."""
    date_index = pd.Index(pd.to_datetime(frame["date"]))
    positions = date_index.get_indexer(pd.to_datetime(hourly_state["date"]))
    if np.any(positions < 0):
        raise ValueError("hourly BIFT decision does not map to the 5m grid")

    signal = pd.DataFrame(
        {
            "date": frame["date"],
            "side": np.zeros(len(frame), dtype=np.int8),
            "hold_bars": np.zeros(len(frame), dtype=np.int16),
            "branch": pd.Series("none", index=frame.index, dtype="string"),
            "origin_position": np.full(len(frame), -1, dtype=np.int64),
        }
    )
    signal.loc[positions, "side"] = hourly_state["side"].to_numpy(np.int8)
    signal.loc[positions, "hold_bars"] = hourly_state["hold_bars"].to_numpy(
        np.int16
    )
    signal.loc[positions, "branch"] = hourly_state["branch"].astype(str).to_numpy()
    hourly_origins = hourly_state["origin_hour_position"].to_numpy(np.int64)
    valid = hourly_origins >= 0
    mapped_origins = np.full(len(hourly_state), -1, dtype=np.int64)
    mapped_origins[valid] = positions[hourly_origins[valid]]
    signal.loc[positions, "origin_position"] = mapped_origins
    return signal


def nonoverlapping_bift_schedule(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp = SELECTION_END,
) -> pd.DataFrame:
    """Build a clean next-open schedule whose setup origin is in the split."""
    start_timestamp = frame["date"].min() if start is None else pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    period = frame["date"].ge(start_timestamp) & frame["date"].lt(end_timestamp)
    origins = signal["origin_position"].to_numpy(np.int64)
    valid_origin = np.zeros(len(signal), dtype=bool)
    in_range = (origins >= 0) & (origins < len(frame))
    valid_origin[in_range] = period.to_numpy(bool)[origins[in_range]]
    eligible = signal.copy()
    eligible.loc[~valid_origin, "side"] = 0
    schedule = nonoverlapping_schedule(
        eligible,
        frame,
        start=start_timestamp,
        end=end_timestamp,
    )
    if not schedule.empty:
        schedule["origin_position"] = [
            int(signal.loc[position, "origin_position"])
            for position in schedule["signal_position"]
        ]
    return schedule


def support_summary(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
    cfg: Config,
) -> dict[str, Any]:
    annual = {
        str(year): nonoverlapping_bift_schedule(
            signal,
            frame,
            start=f"{year}-01-01",
            end=f"{year + 1}-01-01",
        )
        for year in range(2020, 2024)
    }
    schedule = pd.concat(annual.values(), ignore_index=True)
    by_year = {year: len(rows) for year, rows in annual.items()}
    h1 = len(
        nonoverlapping_bift_schedule(
            signal, frame, start="2023-01-01", end="2023-07-01"
        )
    )
    h2 = len(
        nonoverlapping_bift_schedule(
            signal, frame, start="2023-07-01", end="2024-01-01"
        )
    )
    total = len(schedule)
    long_share = float(schedule["side"].gt(0).mean()) if total else 0.0
    short_share = float(schedule["side"].lt(0).mean()) if total else 0.0
    propagation_share = (
        float(schedule["branch"].eq("propagation").mean()) if total else 0.0
    )
    absorption_share = (
        float(schedule["branch"].eq("absorption").mean()) if total else 0.0
    )
    passes = (
        total >= cfg.minimum_nonoverlap_total
        and all(
            value >= cfg.minimum_nonoverlap_per_year
            for value in by_year.values()
        )
        and h1 >= cfg.minimum_nonoverlap_per_2023_half
        and h2 >= cfg.minimum_nonoverlap_per_2023_half
        and min(long_share, short_share) >= cfg.minimum_side_share
        and min(propagation_share, absorption_share)
        >= cfg.minimum_branch_share
    )
    return {
        "nonoverlap_total": int(total),
        "by_year": by_year,
        "2023_h1": int(h1),
        "2023_h2": int(h2),
        "long_share": long_share,
        "short_share": short_share,
        "propagation_share": propagation_share,
        "absorption_share": absorption_share,
        "passes_support": bool(passes),
    }


def _selected_support_quantile(trials: list[dict[str, Any]]) -> float | None:
    passing = [
        float(trial["change_quantile"])
        for trial in trials
        if trial["passes_support"]
    ]
    return max(passing) if passing else None


def run_support(cfg: Config) -> dict[str, Any]:
    frame, source = load_causal_frame(SourceConfig())
    hourly = build_hourly_diagnostics(frame, cfg)
    trials: list[dict[str, Any]] = []
    selected_state: pd.DataFrame | None = None
    selected_signal: pd.DataFrame | None = None
    selected_support: dict[str, Any] | None = None

    for quantile in SUPPORT_CALIBRATION_GRID:
        hourly_state = classify_candidates(
            hourly, cfg, change_quantile=quantile
        )
        signal = project_to_five_minute(hourly_state, frame)
        support = support_summary(signal, frame, cfg)
        trials.append(
            {
                "change_quantile": quantile,
                "raw_setup_count": int(hourly_state["setup"].sum()),
                "raw_candidate_count": int(hourly_state["candidate"].sum()),
                **support,
            }
        )
        if quantile == cfg.change_quantile:
            selected_state = hourly_state
            selected_signal = signal
            selected_support = support

    selected = _selected_support_quantile(trials)
    if selected != cfg.change_quantile:
        raise ValueError("configured BIFT quantile violates support stopping rule")
    if selected_state is None or selected_signal is None or selected_support is None:
        raise AssertionError("configured BIFT state was not evaluated")

    selected_schedule = pd.concat(
        [
            nonoverlapping_bift_schedule(
                selected_signal,
                frame,
                start=f"{year}-01-01",
                end=f"{year + 1}-01-01",
            )
            for year in range(2020, 2024)
        ],
        ignore_index=True,
    )
    detector_available = hourly["detector_available"].astype(bool)
    reset_count = int(
        (
            detector_available
            & ~detector_available.shift(1, fill_value=False)
        ).sum()
    )
    branch_counts = {
        name: int(value)
        for name, value in selected_schedule["branch"].value_counts().items()
    }

    return {
        "protocol": {
            "name": "BIFT — Bayesian Impact-Flow Transition",
            "support_only": True,
            "outcomes_opened_for_bift": False,
            "selection_end_exclusive": "2024-01-01 00:00:00",
            "event_clock": "completed hourly BOCPD change event plus fixed 3h confirmation",
            "signal_availability": "after the confirmation hour closes at :55; enter next 5m open",
            "branch_rule": {
                "propagation": "persistent flow and price alignment; follow reference flow",
                "absorption": "persistent flow without price alignment; fade reference flow",
            },
            "candidate_clock": "fixed before any outcome; both branches share one event clock",
            "holding_rule": "fixed 144 completed 5m bars; scheduled-open exit",
            "source_gap_policy": "verified missing/full-gap day plus 24-bar quarantine; BOCPD posterior reset",
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
        },
        "config": asdict(cfg),
        "frozen_artifacts": {
            "preregistration_source": str(PREREGISTRATION_SOURCE),
            "preregistration_source_sha256": _sha256(PREREGISTRATION_SOURCE),
            "preregistration_document": str(PREREGISTRATION_DOCUMENT),
            "preregistration_document_sha256": _sha256(
                PREREGISTRATION_DOCUMENT
            ),
            "feature_manifest_sha256": _sha256(
                SourceConfig().feature_manifest
            ),
            "market_manifest_sha256": _sha256(
                SourceConfig().market_manifest
            ),
        },
        "source": source,
        "detector": {
            "features": list(BOCPD_FEATURE_COLUMNS),
            "hourly_rows": int(len(hourly)),
            "clean_hours": int(hourly["clean"].sum()),
            "detector_available_hours": int(detector_available.sum()),
            "posterior_segment_count": reset_count,
            "standardization": "strictly lagged rolling median and recursive MAD; clip [-12, 12]",
            "bocpd": "multivariate independent-dimension Student-t predictive recursion",
        },
        "support_calibration": {
            "outcomes_opened_for_bift": False,
            "tested_change_quantiles": list(SUPPORT_CALIBRATION_GRID),
            "all_other_parameters_fixed": True,
            "stopping_rule": "highest tested quantile passing every frozen support floor",
            "selected_change_quantile": selected,
            "further_support_repairs_allowed": False,
            "trials": trials,
        },
        "raw_setup_count": int(selected_state["setup"].sum()),
        "raw_candidate_count": int(selected_state["candidate"].sum()),
        "scheduled_branch_counts": branch_counts,
        "support": selected_support,
        "all_support_gates_pass": bool(selected_support["passes_support"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Config.output)
    args = parser.parse_args()
    cfg = Config(output=args.output)
    result = run_support(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "outcomes_opened_for_bift": result["protocol"][
                    "outcomes_opened_for_bift"
                ],
                "selected_change_quantile": result["support_calibration"][
                    "selected_change_quantile"
                ],
                "support": result["support"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
