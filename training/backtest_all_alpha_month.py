#!/usr/bin/env python3
"""Replay every frozen atomic alpha contract on one completed calendar window.

Universe rules
--------------
The battery includes every distinct atomic strategy reachable from
``configs/live/*.json``, ``configs/shadow/*.json``, and
``research/pools/alphas/*.json`` at the pinned repository revision. Portfolio
mixes are not additional alphas, selector-only portfolio overlays are not
signal generators, and raw scan rows without a frozen execution contract are
inventory-only.

All standalone rows use the same comparison contract:

* completed 5-minute bars and next-bar-open entry;
* the frozen research clock (explicit offset, otherwise ``stride - 1``);
* 6 bp per notional side and 0.5x unit leverage;
* one position per alpha, with overlapping signals suppressed;
* source availability flags fail closed, except the explicitly frozen legacy
  REX weekend-FX fallback, which is reported with a strict sensitivity row; and
* same-BTC upper-before-lower strict MDD.

The July window is retrospective and is not pristine OOS evidence.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.portfolio_live import build_live_portfolio_frames
from execution.rex_llm_live import RexLivePolicyConfig, _is_weekend_or_fx_closed
from preprocessing.binance_aux_features import normalise_funding_history_frame
from preprocessing.live_db_features import (
    LiveDbFeatureConfig,
    sqlalchemy_engine_from_env,
)
from training.backtest_added_alpha_month import (
    BASE_LEVERAGE,
    COST_RATE,
    INTERVAL_MINUTES,
    _availability_summary,
    _barrier_arrays,
    _empty_arrays,
    _fixed_hold_arrays,
    _frame_hash,
    _fresh_signal,
    _interval_slots,
    _load_json,
    _markov_signal,
    _naive,
    _rank7_signal,
    _read_frame,
    _rex_signal,
    _sha256,
    _strict_metric,
    _utc,
    _vector_gate_clauses,
    _vector_gate_pass,
)
from training.evaluate_oi_llm_selector import _context_id, _tokens
from training.event_candidate_pool_probe import _feature_candidates
from training.long_regime_interest_gate_validation import (
    build_interest_features,
)
from training.long_regime_score_gate_validation import (
    _build_score_frame,
    _score_variant,
)
from training.portfolio_opt_added_alpha_update import favorable_path, funding_lr_active
from training.portfolio_opt_new_alpha_pool import _event_path
from training.search_bocpd_state_gated_alpha import (
    _map_output as bocpd_map_output,
)
from training.search_bocpd_state_gated_alpha import (
    _state_from_mapped as bocpd_state_from_mapped,
)
from training.search_bocpd_state_gated_alpha import bocpd_student_t
from training.search_calendar_oi_funding_alpha import add_calendar_features
from training.search_gaussian_hmm_regime_alpha import hourly_features
from training.search_kalman_state_gated_alpha import (
    kalman_local_linear,
    map_hourly_state,
)
from training.search_semimarkov_duration_alpha import (
    duration_key,
    map_hourly_key,
)

DEFAULT_OUTPUT = Path(
    "results/all_alpha_july_2026_performance_2026-07-27.json"
)
DEFAULT_DOCS = Path(
    "docs/all-alpha-july-2026-performance-2026-07-27.md"
)
DEFAULT_HISTORY = Path(
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
)
PROMOTED_PORTFOLIO = Path(
    "configs/live/portfolio_added_alpha_mainnet_live_2026-07-18.json"
)

PROMOTED = (
    "fresh_kimchi_fx",
    "frozen_annual_rank7",
    "rex_taker_low_range_position",
    "cand_rex_veto_7",
    "markov_transition_long",
)

SOURCE_PATHS = {
    "pb30_base": "configs/live/bullish_pb30_addon_returnz_htf1w_candidate.json",
    "pb30_addon": "configs/live/bullish_pb30_addon_returnz_htf1w_candidate.json",
    "nonpb30_taker": "configs/live/nonpb30_taker_returnz_rangevol_htf4hrange_h72_candidate.json",
    "oi_divergence_pullback": "configs/live/oi_divergence_pullback_range_rsi_h96_s6_candidate.json",
    "oi_divergence_highfreq": "configs/live/oi_divergence_sma24_highfreq_h30_s6_candidate.json",
    "oi_divergence_highfreq_selector": "configs/live/oi_divergence_sma24_highfreq_h30_s6_llm_selector_overlay.json",
    "oi_upbit_ratio288_low": "configs/live/oi_upbit_ratio288_low_candidate.json",
    "oi_alt_ratio72_dynamic_exit": "configs/live/oi_alt_ratio72_dyn_exit_candidate.json",
    "short_kimchi3d": "configs/live/short_kimchi3d_candidate.json",
    "short_premium_panic": "configs/live/short_premium_panic_candidate.json",
    "new_long_minimal_funding_premium": "configs/live/new_long_minimal_funding_premium_candidate.json",
    "funding_premium_lr_impact_central": "configs/live/funding_premium_lr_impact_central_research_candidate.json",
    "calendar_oi_funding_friday_asia_long": "research/pools/alphas/calendar_oi_funding_friday_asia_long_20260712.json",
    "kalman_funding_premium_long": "research/pools/alphas/kalman_top10_funding_premium_long_20260713.json",
    "bocpd_funding_premium_long": "research/pools/alphas/bocpd_top10_funding_premium_long_20260713.json",
    "semimarkov_funding_premium_long": "research/pools/alphas/semimarkov_top10_funding_premium_long_20260713.json",
    "rex_htf_range_veto": "research/pools/alphas/rex_htf_range_veto_alpha_20260712.json",
    "legacy_rex_dual_regime_auto": "execution/rex_llm_live.py",
    "legacy_rex_dual_regime_short": "execution/rex_llm_live.py",
    "fresh_kimchi_fx": "configs/shadow/fresh_kimchi_fx_2026-07-16.json",
    "frozen_annual_rank7": "configs/shadow/frozen_annual_rank7_2026-07-16.json",
    "rex_taker_low_range_position": "configs/shadow/rex_taker_low_range_position_2026-07-16.json",
    "cand_rex_veto_7": "configs/live/rex_veto_7_candidate.json",
    "markov_transition_long": "configs/shadow/markov_transition_long_2026-07-16.json",
}

FAMILIES = {
    "pb30_base": "pb30",
    "pb30_addon": "pb30",
    "nonpb30_taker": "price_flow",
    "oi_divergence_pullback": "open_interest",
    "oi_divergence_highfreq": "open_interest",
    "oi_divergence_highfreq_selector": "open_interest_selector",
    "oi_upbit_ratio288_low": "cross_venue_oi",
    "oi_alt_ratio72_dynamic_exit": "cross_asset_oi",
    "short_kimchi3d": "kimchi_short",
    "short_premium_panic": "premium_short",
    "new_long_minimal_funding_premium": "funding_premium",
    "funding_premium_lr_impact_central": "funding_premium",
    "calendar_oi_funding_friday_asia_long": "calendar_derivatives",
    "kalman_funding_premium_long": "state_model",
    "bocpd_funding_premium_long": "state_model",
    "semimarkov_funding_premium_long": "state_model",
    "rex_htf_range_veto": "rex",
    "legacy_rex_dual_regime_auto": "rex_legacy",
    "legacy_rex_dual_regime_short": "rex_legacy",
    "fresh_kimchi_fx": "kimchi_fx",
    "frozen_annual_rank7": "rank7_ml",
    "rex_taker_low_range_position": "rex",
    "cand_rex_veto_7": "rex",
    "markov_transition_long": "state_model",
}

STATUSES = {
    "pb30_base": "legacy_candidate",
    "pb30_addon": "legacy_candidate",
    "nonpb30_taker": "paper_candidate",
    "oi_divergence_pullback": "paper_candidate",
    "oi_divergence_highfreq": "paper_candidate",
    "oi_divergence_highfreq_selector": "paper_selector_proxy",
    "oi_upbit_ratio288_low": "legacy_live_candidate",
    "oi_alt_ratio72_dynamic_exit": "legacy_live_candidate",
    "short_kimchi3d": "paper_zero_weight",
    "short_premium_panic": "legacy_live_candidate",
    "new_long_minimal_funding_premium": "superseded_live_candidate",
    "funding_premium_lr_impact_central": "research_shadow_required",
    "calendar_oi_funding_friday_asia_long": "weak_research_candidate",
    "kalman_funding_premium_long": "research_shadow_required",
    "bocpd_funding_premium_long": "research_shadow_required",
    "semimarkov_funding_premium_long": "research_shadow_required",
    "rex_htf_range_veto": "weak_research_candidate",
    "legacy_rex_dual_regime_auto": "superseded_legacy",
    "legacy_rex_dual_regime_short": "superseded_legacy",
    "fresh_kimchi_fx": "promoted",
    "frozen_annual_rank7": "promoted",
    "rex_taker_low_range_position": "promoted",
    "cand_rex_veto_7": "promoted",
    "markov_transition_long": "promoted",
}


@dataclass(frozen=True)
class Config:
    env_path: Path = Path("/home/pakchu/rllm/.env")
    output: Path = DEFAULT_OUTPUT
    docs_output: Path = DEFAULT_DOCS
    historical_context: Path = DEFAULT_HISTORY
    start: str = "2026-07-01T00:00:00Z"
    end: str = "2026-08-01T00:00:00Z"
    asof: str = "2026-07-27T15:03:00Z"
    lookback_minutes: int = 150_000
    enriched_cache: Path | None = None
    features_cache: Path | None = None
    funding_cache: Path | None = None


async def _query_frames(
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    engine = sqlalchemy_engine_from_env(cfg.env_path)
    live_cfg = LiveDbFeatureConfig(
        lookback_minutes=int(cfg.lookback_minutes),
        include_spot_source=True,
    )
    asof = _utc(cfg.asof)
    enriched, features = await build_live_portfolio_frames(
        engine=engine,
        asof=asof,
        cfg=live_cfg,
        live_oi_snapshot_cutoff=asof + pd.Timedelta(minutes=2),
        include_activity_flow=False,
        include_alt_pool=True,
    )
    from sqlalchemy import text

    with engine.connect() as conn:
        funding = pd.read_sql_query(
            text(
                """
                SELECT funding_time AS date, funding_rate
                FROM funding_rates_binance
                WHERE symbol = 'BTCUSDT'
                  AND funding_time <= :asof
                ORDER BY funding_time
                """
            ),
            conn,
            params={"asof": asof.to_pydatetime()},
        )
    return enriched, features, funding, engine


def _load_frames(
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any | None]:
    cache_paths = (cfg.enriched_cache, cfg.features_cache, cfg.funding_cache)
    if any(cache_paths):
        if not all(cache_paths):
            raise ValueError("all three frame caches must be supplied together")
        assert cfg.enriched_cache is not None
        assert cfg.features_cache is not None
        assert cfg.funding_cache is not None
        return (
            _read_frame(cfg.enriched_cache),
            _read_frame(cfg.features_cache),
            _read_frame(cfg.funding_cache),
            None,
        )
    return asyncio.run(_query_frames(cfg))


def _cap_frames_asof(
    market: pd.DataFrame,
    features: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    asof: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(market) != len(features):
        raise RuntimeError("market/features length mismatch before asof cap")
    asof_utc = _utc(asof)
    completed_end = asof_utc.floor(f"{INTERVAL_MINUTES}min")
    market_dates = pd.to_datetime(market["date"], utc=True)
    keep = (market_dates < completed_end).to_numpy(bool)
    capped_market = market.loc[keep].reset_index(drop=True)
    capped_features = features.loc[keep].reset_index(drop=True)

    capped_funding = funding.copy()
    funding_discarded = 0
    if not capped_funding.empty and "date" in capped_funding:
        funding_dates = pd.to_datetime(capped_funding["date"], utc=True)
        funding_keep = (funding_dates <= asof_utc).to_numpy(bool)
        funding_discarded = int((~funding_keep).sum())
        capped_funding = capped_funding.loc[funding_keep].reset_index(drop=True)
    return capped_market, capped_features, capped_funding, {
        "asof": str(asof_utc),
        "completed_end_exclusive": str(completed_end),
        "market_rows_discarded_after_asof": int((~keep).sum()),
        "funding_rows_discarded_after_asof": funding_discarded,
    }


def _frozen_activity_flow_htf(
    market: pd.DataFrame,
    history_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_quote",
    ]
    historical = pd.read_csv(
        history_path,
        compression="infer",
        usecols=columns,
    )
    current = market.loc[:, columns].copy()
    combined = pd.concat([historical, current], ignore_index=True)
    combined["date"] = pd.to_datetime(
        combined["date"], utc=True
    ).dt.tz_convert(None)
    combined = (
        combined.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    base = pd.DataFrame(index=combined.index)
    quote = pd.to_numeric(
        combined["quote_asset_volume"], errors="coerce"
    ).replace(0.0, np.nan)
    taker_buy = pd.to_numeric(
        combined["taker_buy_quote"], errors="coerce"
    )
    base["taker_imbalance"] = (2.0 * taker_buy / quote - 1.0).clip(-1.0, 1.0)
    interest = build_interest_features(combined, base)
    raw = _build_score_frame(combined, base, interest)
    train_mask = (
        (combined["date"] >= pd.Timestamp("2020-01-01"))
        & (combined["date"] < pd.Timestamp("2024-01-01"))
    ).to_numpy(bool)
    scored = _score_variant(raw, train_mask, "activity_flow_htf")
    if scored is None:
        raise RuntimeError("could not rebuild frozen activity_flow_htf")
    score, stats = scored
    mapped = pd.DataFrame(
        {
            "date": combined["date"],
            "activity_flow_htf": score.to_numpy(float),
        }
    ).set_index("date")["activity_flow_htf"].reindex(
        pd.to_datetime(market["date"])
    )
    if mapped.isna().any():
        raise RuntimeError("activity_flow_htf did not map to every market row")
    train_score = score.to_numpy(float)[train_mask]
    return mapped.to_numpy(float), {
        "fit_window": "2020-01-01/2024-01-01",
        "fit_rows": int(train_mask.sum()),
        "history_path": str(history_path),
        "score_stats": stats,
        "train_quantiles": {
            "q40": float(np.quantile(train_score, 0.40)),
            "q50": float(np.quantile(train_score, 0.50)),
        },
    }


def _research_offset(stride: int, configured: int | None = None) -> int:
    if configured is not None:
        return int(configured) % int(stride)
    return max(0, int(stride) - 1)


def _gated_signal(
    dates: pd.Series,
    frame: pd.DataFrame,
    *,
    gates: list[dict[str, Any]] | None = None,
    gate_clauses: list[list[dict[str, Any]]] | None = None,
    side: int,
    stride: int,
    offset: int | None = None,
    extra: np.ndarray | None = None,
) -> np.ndarray:
    if gate_clauses is not None:
        active = _vector_gate_clauses(frame, gate_clauses)
    else:
        active = _vector_gate_pass(frame, gates or [])
    if extra is not None:
        active &= np.asarray(extra, dtype=bool)
    active &= _interval_slots(
        dates,
        int(stride),
        _research_offset(int(stride), offset),
    )
    return np.where(active, int(side), 0).astype(np.int8)


def _selector_proxy_signal(
    market: pd.DataFrame,
    features: pd.DataFrame,
    base_signal: np.ndarray,
    overlay: dict[str, Any],
) -> np.ndarray:
    proxy = overlay["symbolic_proxy"]
    blocked = {str(row["context_id"]) for row in proxy["blocked_contexts"]}
    keys = tuple(str(value) for value in proxy["context_keys"])
    signal = np.asarray(base_signal, dtype=np.int8).copy()
    for position in np.flatnonzero(signal):
        context = _context_id(
            _tokens(int(position), market=market, feat=features),
            keys,
        )
        if context in blocked:
            signal[int(position)] = 0
    return signal


def _legacy_rex_signals(
    market: pd.DataFrame,
    features: pd.DataFrame,
    *,
    allow_weekend_fallback: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    policy = RexLivePolicyConfig()
    strength, direction = _feature_candidates(features)[policy.family]
    positive = pd.Series(np.asarray(strength, dtype=float)).where(
        np.asarray(strength, dtype=float) > 0.0
    )
    threshold = (
        positive.expanding(min_periods=int(policy.min_positive_strengths))
        .quantile(float(policy.strength_quantile))
        .to_numpy(float)
    )

    def gate_mask(gates: tuple[Any, ...]) -> np.ndarray:
        return _vector_gate_pass(
            features,
            [
                {
                    "feature": gate.feature,
                    "op": gate.op,
                    "threshold": float(gate.value),
                }
                for gate in gates
            ],
        )

    gate_sets = [gate_mask(policy.gates)]
    gate_sets.extend(gate_mask(gates) for gates in policy.alternate_gate_sets)
    gates_pass = np.logical_or.reduce(gate_sets)
    dates = pd.to_datetime(market["date"])
    weekend = np.asarray(
        [_is_weekend_or_fx_closed(value) for value in dates],
        dtype=bool,
    )
    core_available = np.logical_and.reduce(
        [
            pd.to_numeric(features[key], errors="coerce")
            .fillna(0.0)
            .to_numpy(float)
            > 0.5
            for key in ("dxy_available", "kimchi_available", "usdkrw_available")
        ]
    )
    quality = core_available
    if allow_weekend_fallback:
        quality |= (
            weekend & bool(policy.allow_missing_core_external_on_weekend)
        )
    active = (
        np.isfinite(threshold)
        & (np.asarray(strength, dtype=float) > threshold)
        & (np.asarray(direction, dtype=float) != 0.0)
        & gates_pass
        & quality
    )
    auto = np.where(active, np.sign(direction), 0).astype(np.int8)
    short = np.where(auto < 0, -1, 0).astype(np.int8)
    return auto, short


def _state_context(
    market: pd.DataFrame,
    history_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "quote_asset_volume",
        "taker_buy_quote",
    ]
    historical = pd.read_csv(
        history_path,
        compression="infer",
        usecols=required,
    )
    current = market.loc[:, required].copy()
    combined = pd.concat([historical, current], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], utc=True).dt.tz_convert(
        None
    )
    combined = (
        combined.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    return hourly_features(combined)


def _state_model_signals(
    market: pd.DataFrame,
    features: pd.DataFrame,
    history_path: Path,
) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(market["date"])
    base_cfg = _load_json(
        "configs/live/new_long_minimal_funding_premium_candidate.json"
    )
    base = _vector_gate_clauses(features, base_cfg["gate_clauses"])
    hourly, hourly_feature = _state_context(market, history_path)
    slots = _interval_slots(dates, 12, 11)

    kalman_pool = _load_json(
        "research/pools/alphas/kalman_top10_funding_premium_long_20260713.json"
    )
    kalman_scan = _load_json(
        "results/kalman_state_gated_alpha_scan_2026-07-13.json"
    )
    kalman_rank = int(kalman_pool["selection"]["pre_evaluation_rank"])
    kalman_row = kalman_scan["selected"][kalman_rank - 1]
    log_price = np.log(hourly["close"].to_numpy(float))
    train = (
        (hourly.index >= pd.Timestamp("2020-01-01"))
        & (hourly.index < pd.Timestamp("2024-01-01"))
    )
    train_returns = np.diff(log_price)[np.asarray(train, dtype=bool)[1:]]
    train_var = float(np.nanvar(train_returns))
    filtered = kalman_local_linear(
        log_price,
        float(kalman_row["q_level"]),
        float(kalman_row["q_slope"]),
        float(kalman_row["r_obs"]),
        train_var,
    )
    thresholds = kalman_row["state_thresholds"]
    slope = filtered[:, 3]
    innovation = filtered[:, 2]
    slope_bucket = np.where(
        slope <= float(thresholds["slope_low"]),
        0,
        np.where(slope >= float(thresholds["slope_high"]), 2, 1),
    )
    innovation_bucket = np.where(
        innovation <= float(thresholds["innovation_low"]),
        0,
        np.where(
            innovation >= float(thresholds["innovation_high"]),
            2,
            1,
        ),
    )
    kalman_hourly = pd.DataFrame(
        {
            "date": hourly.index.to_numpy(),
            "state": slope_bucket * 3 + innovation_bucket,
        }
    )
    kalman_state = map_hourly_state(dates, kalman_hourly)

    bocpd_pool = _load_json(
        "research/pools/alphas/bocpd_top10_funding_premium_long_20260713.json"
    )
    bocpd_scan = _load_json(
        "results/bocpd_state_gated_alpha_scan_2026-07-13.json"
    )
    bocpd_rank = int(bocpd_pool["selection"]["pre_evaluation_rank"])
    bocpd_row = bocpd_scan["selected"][bocpd_rank - 1]
    model = bocpd_row["model"]
    columns = tuple(str(value) for value in model["columns"])
    good = hourly_feature[list(columns)].notna().all(axis=1).to_numpy()
    raw = hourly_feature.loc[good, list(columns)].to_numpy(float)
    mean = np.asarray(model["train_standardization_mean"], dtype=float)
    std = np.asarray(model["train_standardization_std"], dtype=float)
    standardized = ((raw - mean) / std).clip(-12, 12)
    posterior = bocpd_student_t(
        standardized,
        hazard_lambda=int(model["hazard_lambda_hours"]),
        max_run_length=int(model["max_run_length"]),
    )
    secondary_index = columns.index("flow24")
    bocpd_output = pd.DataFrame(
        {
            "date": hourly_feature.index[good].to_numpy(),
            "primary": posterior["posterior_mean"][:, 0],
            "short_mass": posterior["short_mass"],
            "run_drop": posterior["run_drop"],
            "secondary": posterior["posterior_mean"][:, secondary_index],
            "surprise": posterior["surprise"],
        }
    )
    bocpd_state = bocpd_state_from_mapped(
        bocpd_map_output(dates, bocpd_output),
        bocpd_row["state_thresholds"],
    )

    semimarkov_pool = _load_json(
        "research/pools/alphas/semimarkov_top10_funding_premium_long_20260713.json"
    )
    semimarkov_scan = _load_json(
        "results/semimarkov_duration_alpha_scan_2026-07-13.json"
    )
    semimarkov_rank = int(
        semimarkov_pool["selection"]["pre_evaluation_rank"]
    )
    semimarkov_row = semimarkov_scan["selected"][semimarkov_rank - 1]
    sem_thresholds = semimarkov_row["state_thresholds"]
    trend = hourly_feature["trend24"].to_numpy(float)
    volatility = hourly_feature["vol24"].to_numpy(float)
    flow = hourly_feature["flow24"].to_numpy(float)
    trend_bucket = np.where(
        trend <= float(sem_thresholds["trend_low"]),
        0,
        np.where(trend >= float(sem_thresholds["trend_high"]), 2, 1),
    )
    sem_state = (
        trend_bucket * 4
        + (volatility >= float(sem_thresholds["vol_median"])).astype(int) * 2
        + (flow >= float(sem_thresholds["flow_median"])).astype(int)
    )
    sem_state = np.where(
        np.isfinite(trend) & np.isfinite(volatility) & np.isfinite(flow),
        sem_state,
        -1,
    )
    sem_key, _ = duration_key(
        sem_state,
        tuple(int(value) for value in semimarkov_row["duration_cutpoints_hours"]),
        timestamps=hourly_feature.index,
    )
    mapped_sem_key = map_hourly_key(
        dates,
        hourly_feature.index,
        sem_key,
    )

    return {
        "kalman_funding_premium_long": np.where(
            slots
            & base
            & np.isin(
                kalman_state,
                np.asarray(kalman_row["allowed_states"], dtype=int),
            ),
            1,
            0,
        ).astype(np.int8),
        "bocpd_funding_premium_long": np.where(
            slots
            & base
            & np.isin(
                bocpd_state,
                np.asarray(bocpd_row["allowed_states"], dtype=int),
            ),
            1,
            0,
        ).astype(np.int8),
        "semimarkov_funding_premium_long": np.where(
            slots
            & base
            & np.isin(
                mapped_sem_key,
                np.asarray(semimarkov_row["allowed_keys"], dtype=int),
            ),
            1,
            0,
        ).astype(np.int8),
    }


def _dynamic_exit_arrays(
    market: pd.DataFrame,
    features: pd.DataFrame,
    signal: np.ndarray,
    *,
    name: str,
    hold_bars: int,
    dynamic: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    dates = pd.to_datetime(market["date"])
    output = _empty_arrays(len(market))
    next_allowed = 0
    min_bars = int(dynamic.get("min_bars", 0))
    gates = list(dynamic.get("gates", []))
    dynamic_active = _vector_gate_pass(features, gates)
    for raw_position in np.flatnonzero(signal):
        position = int(raw_position)
        if not (start <= dates.iloc[position] < end):
            continue
        if position < next_allowed:
            output["skipped_overlap"] += 1
            continue
        entry = position + 1
        max_exit = entry + int(hold_bars)
        if max_exit >= len(market):
            output["skipped_boundary"] += 1
            continue
        exit_position = max_exit
        exit_reason = "max_hold"
        for observed in range(entry, max_exit):
            bars_elapsed = observed - position
            if bars_elapsed >= min_bars and dynamic_active[observed]:
                exit_position = observed + 1
                exit_reason = "dynamic_exit"
                break
        if not (dates.iloc[exit_position] < end):
            output["skipped_boundary"] += 1
            continue
        actual_hold = exit_position - entry
        side = "long" if int(signal[position]) > 0 else "short"
        path = _event_path(
            market,
            position,
            side=side,
            hold=actual_hold,
            cost_rate=COST_RATE,
            entry_delay=1,
            leverage=BASE_LEVERAGE,
        )
        if path is None:
            output["skipped_boundary"] += 1
            continue
        event_return, event_adverse, realized = path
        event_favorable = favorable_path(
            market,
            signal_position=position,
            exit_position=exit_position,
            side=side,
            leverage=BASE_LEVERAGE,
        )
        output["R"] += event_return
        if side == "long":
            output["L"] += event_adverse
            output["H"] += event_favorable
        else:
            output["L"] += event_favorable
            output["H"] += event_adverse
        output["trades"].append(
            {
                "sleeve": name,
                "signal_date": str(dates.iloc[position]),
                "entry_date": str(dates.iloc[entry]),
                "exit_date": str(dates.iloc[exit_position]),
                "side": side.upper(),
                "net_return": float(realized),
                "source": exit_reason,
            }
        )
        next_allowed = exit_position + 1
    return output


def _digest_array(values: np.ndarray) -> str:
    array = np.nan_to_num(
        np.asarray(values, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return hashlib.sha256(np.round(array, 12).tobytes()).hexdigest()


def _signal_digest(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.int8)
    if not np.isin(array, (-1, 0, 1)).all():
        raise ValueError("signal values must be -1, 0, or 1")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _duplicate_groups(
    signals: dict[str, np.ndarray],
    arrays: dict[str, dict[str, Any]],
    window_mask: np.ndarray,
) -> dict[str, Any]:
    signal_buckets: dict[str, list[str]] = {}
    path_buckets: dict[str, list[str]] = {}
    zero_signal: list[str] = []
    zero_path: list[str] = []
    for name, signal in signals.items():
        window_signal = np.asarray(signal[window_mask], dtype=np.int8)
        if np.count_nonzero(window_signal) == 0:
            zero_signal.append(name)
        else:
            signal_hash = _signal_digest(window_signal)
            signal_buckets.setdefault(signal_hash, []).append(name)
        path_active = any(
            np.count_nonzero(np.asarray(arrays[name][key])[window_mask])
            for key in ("R", "L", "H")
        )
        if not path_active:
            zero_path.append(name)
            continue
        path_hash = hashlib.sha256(
            (
                _digest_array(arrays[name]["R"][window_mask])
                + _digest_array(arrays[name]["L"][window_mask])
                + _digest_array(arrays[name]["H"][window_mask])
            ).encode()
        ).hexdigest()
        path_buckets.setdefault(path_hash, []).append(name)
    return {
        "exact_signal": sorted(
            [names for names in signal_buckets.values() if len(names) > 1]
        ),
        "exact_path": sorted(
            [names for names in path_buckets.values() if len(names) > 1]
        ),
        "zero_signal": sorted(zero_signal),
        "zero_path": sorted(zero_path),
    }


def _inventory() -> dict[str, Any]:
    atomic_configs = sorted(
        str(path)
        for folder in (Path("configs/live"), Path("configs/shadow"))
        for path in folder.glob("*.json")
        if not path.name.startswith("portfolio_")
        and path.name != "rex_llm_binance_testnet_bear_pilot.json"
    )
    research_pool_files = sorted(
        str(path) for path in Path("research/pools/alphas").glob("*.json")
    )
    portfolio_configs = sorted(
        str(path)
        for folder in (Path("configs/live"), Path("configs/shadow"))
        for path in folder.glob("portfolio_*.json")
    )
    raw_scan_artifacts = sorted(
        str(path)
        for path in Path("results").glob("*alpha*scan*.json")
        if path.is_file()
    )
    non_replayed = [
        {
            "path": "configs/live/bullish_pb30_base_candidate.json",
            "reason": "deduplicated wrapper; the same fixed base module is replayed from the PB30 base/addon source",
        },
        {
            "path": "configs/live/bullish_pb30_funding_activity_flow_htf_candidate.json",
            "reason": "quantile-only thresholds are not frozen; PB30 fixed base/addon modules are replayed instead",
        },
        {
            "path": "configs/live/rex_taker_low_range_position_research_candidate.json",
            "reason": "deduplicated alias; productionized frozen shadow contract is replayed",
        },
        {
            "path": "research/pools/alphas/markov_persistent_funding_premium_long_20260712.json",
            "reason": "deduplicated research record; the productionized frozen Markov shadow contract is replayed",
        },
    ]
    scored_source_files = sorted(
        {
            str(path)
            for path in SOURCE_PATHS.values()
            if Path(path).suffix == ".json"
        }
    )
    accounted = set(scored_source_files) | {
        str(row["path"]) for row in non_replayed
    }
    unaccounted = sorted(
        (set(atomic_configs) | set(research_pool_files)) - accounted
    )
    if unaccounted:
        raise RuntimeError(
            f"unaccounted atomic alpha files: {unaccounted}"
        )
    return {
        "scored_atomic_alphas": len(SOURCE_PATHS),
        "atomic_config_files": atomic_configs,
        "research_pool_files": research_pool_files,
        "scored_source_files": scored_source_files,
        "portfolio_configs_excluded_as_compositions": portfolio_configs,
        "runtime_configs_excluded": [
            "configs/live/rex_llm_binance_testnet_bear_pilot.json"
        ],
        "raw_scan_artifacts_inventory_only": raw_scan_artifacts,
        "raw_scan_artifact_count": len(raw_scan_artifacts),
        "non_replayed_or_deduplicated": non_replayed,
        "unaccounted_atomic_files": unaccounted,
        "replayed_variant_notes": [
            {
                "name": "oi_divergence_highfreq_selector",
                "note": "bounded frozen symbolic selector proxy; cannot create entries",
            },
            {
                "name": "legacy_rex_dual_regime_auto",
                "note": "code-frozen legacy expanding-quantile policy",
            },
            {
                "name": "legacy_rex_dual_regime_short",
                "note": "short-only projection of the code-frozen legacy policy",
            },
        ],
    }


def _render_docs(report: dict[str, Any]) -> str:
    def cell(row: dict[str, Any]) -> str:
        return (
            f"{row['absolute_return_pct']:.4f}% / {row['cagr_pct']:.2f}% / "
            f"{row['strict_mdd_pct']:.4f}% / "
            f"{row['cagr_to_strict_mdd']:.2f} / {row['trades']}"
        )

    metrics = report["metrics"]
    ordered = sorted(
        metrics,
        key=lambda name: (
            name not in PROMOTED,
            -metrics[name]["absolute_return_pct"],
            name,
        ),
    )
    lines = [
        "# 전체 atomic alpha — 2026년 7월 성과 감사",
        "",
        f"- 데이터: `{report['window']['start']}` ~ `{report['window']['end_exclusive']}` UTC",
        f"- 완결 5분봉: **{report['window']['bars']:,}개**",
        f"- 전수 범위: **{len(metrics)}개 atomic alpha/variant**; 포트폴리오 조합과 raw scan row는 제외.",
        "- 공통 계약: 0.5x, 6 bp/notional/side, next-bar open, overlap suppression, strict MDD.",
        "- 셀: `절대수익 / 관측기간 연율화 CAGR / strict MDD / CAGR-MDD / 거래수`.",
        "- 한 달 미만 CAGR은 불안정하므로 절대수익·MDD·거래수를 우선한다.",
        "- 단일 알파 최대 거래수도 14회라 통계적 승격 근거로는 부족하다.",
        "",
        "## 전수 결과",
        "",
        "| 알파 | 상태 | 계열 | 성과 | L/S | 승률 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name in ordered:
        row = metrics[name]
        lines.append(
            f"| `{name}` | {STATUSES[name]} | {FAMILIES[name]} | "
            f"{cell(row)} | {row['longs']}/{row['shorts']} | "
            f"{row['win_rate'] * 100:.1f}% |"
        )
    portfolio = report["promoted_gross8"]
    legacy_notes = []
    for name, sensitivity in report[
        "legacy_rex_availability_sensitivity"
    ].items():
        strict = sensitivity["strict_fail_closed_metric"]
        legacy_notes.append(
            f"- `{name}` 주말 FX fallback 차이 신호 "
            f"{sensitivity['weekend_fallback_signal_rows']}개; strict 결과 "
            f"`{cell(strict)}`."
        )
    lines.extend(
        [
            "",
            "## 현재 승격 Gross 8",
            "",
            f"- 성과: **{cell(portfolio)}**",
            (
                f"- 방향/승률: {portfolio['longs']}/{portfolio['shorts']}, "
                f"{portfolio['win_rate'] * 100:.1f}%"
            ),
            "",
            "## 전수성 및 계약 주의",
            "",
            (
                f"- `results/*alpha*scan*.json` raw scan 산출물 "
                f"{report['inventory']['raw_scan_artifact_count']}개는 고정 "
                "실행계약이 아니므로 통계에 섞지 않았다."
            ),
            (
                "- PB30 `activity_flow_htf`는 July 전체 통계가 아니라 "
                "2020~2023 고정 통계로 재구축했다."
            ),
            "- PB30 quantile-only 파일은 threshold가 완결되지 않아 제외하고, 고정 base/addon module을 각각 재현했다.",
            "- OI selector는 실제 LLM이 아니라 동결된 symbolic ALLOW/BLOCK proxy다.",
            "- legacy 후보의 누락 stride offset은 원 연구 clock과 맞는 `stride-1`로 복구했다. 구형 generic live default 0과는 parity 위험이 있다.",
            (
                "- 캐시 사용 시에도 `asof` 뒤의 미완결/미래 market 및 "
                "funding row를 강제 절단한다."
            ),
            *legacy_notes,
            "- BOCPD/Kalman/Semi-Markov는 2019-12-31부터의 5분봉을 causal warm-up으로 사용하고 July에는 backward-as-of state만 매핑했다.",
            "- 전체 구간은 이미 연구에서 관찰된 retrospective 진단이며 pristine OOS가 아니다.",
            "",
            "## 중복",
            "",
            "```json",
            json.dumps(report["duplicate_groups"], indent=2, ensure_ascii=False),
            "```",
        ]
    )
    return "\n".join(lines)


def run(cfg: Config) -> dict[str, Any]:
    market, features, raw_funding, _ = _load_frames(cfg)
    loaded_market_rows = len(market)
    loaded_funding_rows = len(raw_funding)
    market, features, raw_funding, asof_cap = _cap_frames_asof(
        market,
        features,
        raw_funding,
        asof=cfg.asof,
    )
    market = market.copy()
    market["date"] = pd.to_datetime(
        market["date"], utc=True
    ).dt.tz_convert(None)
    features = features.reset_index(drop=True)
    if len(market) != len(features):
        raise RuntimeError("market/features length mismatch")
    dates = pd.to_datetime(market["date"])
    if not dates.diff().dropna().eq(
        pd.Timedelta(minutes=INTERVAL_MINUTES)
    ).all():
        raise RuntimeError("market frame is not a complete 5-minute grid")
    required_columns = {
        "alt_pool_available",
        "upbit_volume_available",
    }
    missing = sorted(required_columns - set(features.columns))
    if missing:
        raise RuntimeError(f"full-alpha feature cache is incomplete: {missing}")
    activity_flow, activity_flow_meta = _frozen_activity_flow_htf(
        market,
        cfg.historical_context,
    )
    features["activity_flow_htf"] = activity_flow

    requested_start = _naive(cfg.start)
    requested_end = _naive(cfg.end)
    data_end = dates.iloc[-1] + pd.Timedelta(minutes=INTERVAL_MINUTES)
    start = requested_start
    end = min(requested_end, data_end)
    window_mask = ((dates >= start) & (dates < end)).to_numpy(bool)
    expected_bars = int(
        (end - start) / pd.Timedelta(minutes=INTERVAL_MINUTES)
    )
    if int(window_mask.sum()) != expected_bars:
        raise RuntimeError("July window is incomplete")
    funding = normalise_funding_history_frame(raw_funding)

    signals: dict[str, np.ndarray] = {}
    pb30 = _load_json(SOURCE_PATHS["pb30_base"])
    frozen_activity_thresholds = {
        "q50": float(pb30["base_module"]["gates"][-1]["threshold"]),
        "q40": float(pb30["addon_module"]["gates"][1]["threshold"]),
    }
    activity_flow_meta["frozen_thresholds"] = frozen_activity_thresholds
    activity_flow_meta["threshold_drift"] = {
        key: float(activity_flow_meta["train_quantiles"][key] - value)
        for key, value in frozen_activity_thresholds.items()
    }
    if any(
        abs(value) > 5e-4
        for value in activity_flow_meta["threshold_drift"].values()
    ):
        raise RuntimeError(
            "PB30 frozen activity score drift exceeds tolerance"
        )
    for name, module in (
        ("pb30_base", pb30["base_module"]),
        ("pb30_addon", pb30["addon_module"]),
    ):
        stride = int(module["stride_bars_5m"])
        signals[name] = _gated_signal(
            dates,
            features,
            gates=module["gates"],
            side=1,
            stride=stride,
        )

    nonpb = _load_json(SOURCE_PATHS["nonpb30_taker"])["signal"]
    signals["nonpb30_taker"] = _gated_signal(
        dates,
        features,
        gates=nonpb["gates"],
        side=1,
        stride=int(nonpb["stride_bars_5m"]),
    )
    oi_pullback = _load_json(SOURCE_PATHS["oi_divergence_pullback"])[
        "signal"
    ]
    signals["oi_divergence_pullback"] = _gated_signal(
        dates,
        features,
        gates=oi_pullback["gates"],
        side=1,
        stride=int(oi_pullback["stride_bars_5m"]),
    )
    oi_high = _load_json(SOURCE_PATHS["oi_divergence_highfreq"])
    signals["oi_divergence_highfreq"] = _gated_signal(
        dates,
        features,
        gates=oi_high["gates"],
        side=1,
        stride=int(oi_high["stride_bars"]),
    )
    oi_overlay = _load_json(
        SOURCE_PATHS["oi_divergence_highfreq_selector"]
    )
    signals["oi_divergence_highfreq_selector"] = _selector_proxy_signal(
        market,
        features,
        signals["oi_divergence_highfreq"],
        oi_overlay,
    )

    for name in ("oi_upbit_ratio288_low", "oi_alt_ratio72_dynamic_exit"):
        source = _load_json(SOURCE_PATHS[name])
        signals[name] = _gated_signal(
            dates,
            features,
            gates=source["gates"],
            side=1,
            stride=int(source["stride_bars"]),
            offset=source.get("stride_offset_bars"),
        )

    for name in ("short_kimchi3d", "short_premium_panic"):
        source = _load_json(SOURCE_PATHS[name])
        signals[name] = _gated_signal(
            dates,
            features,
            gates=source["gates"],
            side=-1,
            stride=int(source["stride_bars"]),
            offset=source.get("stride_offset_bars"),
        )

    new_long = _load_json(
        SOURCE_PATHS["new_long_minimal_funding_premium"]
    )
    signals["new_long_minimal_funding_premium"] = _gated_signal(
        dates,
        features,
        gate_clauses=new_long["gate_clauses"],
        side=1,
        stride=int(new_long["stride_bars"]),
        offset=int(new_long["stride_offset_bars"]),
    )

    funding_active, funding_meta = funding_lr_active(market)
    signals["funding_premium_lr_impact_central"] = np.where(
        funding_active & _interval_slots(dates, 12, 11),
        1,
        0,
    ).astype(np.int8)

    calendar_pool = _load_json(
        SOURCE_PATHS["calendar_oi_funding_friday_asia_long"]
    )
    calendar_frame = add_calendar_features(market, features)
    calendar_gates = [
        {
            "feature": row["feature"],
            "op": row["op"],
            "threshold": row["threshold"],
        }
        for row in calendar_pool["entry_logic"]["terms"]
    ]
    calendar_sources = np.logical_and.reduce(
        [
            pd.to_numeric(features[key], errors="coerce")
            .fillna(0.0)
            .to_numpy(float)
            > 0.5
            for key in (
                "funding_available",
                "premium_available",
                "open_interest_available",
            )
        ]
    )
    signals["calendar_oi_funding_friday_asia_long"] = _gated_signal(
        dates,
        calendar_frame,
        gates=calendar_gates,
        side=1,
        stride=int(calendar_pool["exit_logic"]["stride_bars"]),
        extra=calendar_sources,
    )

    signals.update(
        _state_model_signals(market, features, cfg.historical_context)
    )

    legacy_auto, legacy_short = _legacy_rex_signals(market, features)
    strict_legacy_auto, strict_legacy_short = _legacy_rex_signals(
        market,
        features,
        allow_weekend_fallback=False,
    )
    signals["legacy_rex_dual_regime_auto"] = legacy_auto
    signals["legacy_rex_dual_regime_short"] = legacy_short

    promoted_cfg = _load_json(PROMOTED_PORTFOLIO)
    promoted_sources = {
        row["name"]: _load_json(row["source"])
        for row in promoted_cfg["base_sleeves"]
    }
    signals["fresh_kimchi_fx"] = _fresh_signal(
        market,
        features,
        promoted_sources["fresh_kimchi_fx"],
    )
    rank7_signal, rank7_lifecycles, rank7_diagnostics = _rank7_signal(
        market,
        promoted_sources["frozen_annual_rank7"],
    )
    signals["frozen_annual_rank7"] = rank7_signal
    signals["rex_taker_low_range_position"] = _rex_signal(
        market,
        features,
        promoted_sources["rex_taker_low_range_position"],
    )
    signals["cand_rex_veto_7"] = _rex_signal(
        market,
        features,
        promoted_sources["cand_rex_veto_7"],
    )
    signals["markov_transition_long"] = _markov_signal(
        market,
        features,
        promoted_sources["markov_transition_long"],
    )

    rex_pool = _load_json(SOURCE_PATHS["rex_htf_range_veto"])
    rex_pool_cfg = copy.deepcopy(promoted_sources["cand_rex_veto_7"])
    rex_pool_cfg["gates"] = list(rex_pool["entry_logic"]["keep_gates"])
    signals["rex_htf_range_veto"] = _rex_signal(
        market,
        features,
        rex_pool_cfg,
    )

    holds = {
        "pb30_base": int(pb30["base_module"]["hold_bars_5m"]),
        "pb30_addon": int(pb30["addon_module"]["hold_bars_5m"]),
        "nonpb30_taker": int(nonpb["hold_bars_5m"]),
        "oi_divergence_pullback": int(oi_pullback["hold_bars_5m"]),
        "oi_divergence_highfreq": int(oi_high["hold_bars"]),
        "oi_divergence_highfreq_selector": int(oi_high["hold_bars"]),
        "oi_upbit_ratio288_low": int(
            _load_json(SOURCE_PATHS["oi_upbit_ratio288_low"])["hold_bars"]
        ),
        "short_kimchi3d": int(
            _load_json(SOURCE_PATHS["short_kimchi3d"])["hold_bars"]
        ),
        "short_premium_panic": int(
            _load_json(SOURCE_PATHS["short_premium_panic"])["hold_bars"]
        ),
        "new_long_minimal_funding_premium": int(new_long["hold_bars"]),
        "funding_premium_lr_impact_central": 576,
        "calendar_oi_funding_friday_asia_long": int(
            calendar_pool["exit_logic"]["hold_bars_5m"]
        ),
        "kalman_funding_premium_long": 576,
        "bocpd_funding_premium_long": 576,
        "semimarkov_funding_premium_long": 576,
        "rex_htf_range_veto": 144,
        "legacy_rex_dual_regime_auto": 144,
        "legacy_rex_dual_regime_short": 144,
        "rex_taker_low_range_position": int(
            promoted_sources["rex_taker_low_range_position"]["hold_bars"]
        ),
        "cand_rex_veto_7": int(
            promoted_sources["cand_rex_veto_7"]["hold_bars"]
        ),
        "markov_transition_long": int(
            promoted_sources["markov_transition_long"]["hold_bars"]
        ),
    }
    arrays = {
        name: _fixed_hold_arrays(
            market,
            signals[name],
            name=name,
            hold_bars=hold,
            start=start,
            end=end,
        )
        for name, hold in holds.items()
    }
    oi_dynamic = _load_json(
        SOURCE_PATHS["oi_alt_ratio72_dynamic_exit"]
    )
    arrays["oi_alt_ratio72_dynamic_exit"] = _dynamic_exit_arrays(
        market,
        features,
        signals["oi_alt_ratio72_dynamic_exit"],
        name="oi_alt_ratio72_dynamic_exit",
        hold_bars=int(oi_dynamic["hold_bars"]),
        dynamic=oi_dynamic["dynamic_exit"],
        start=start,
        end=end,
    )
    fresh_cfg = promoted_sources["fresh_kimchi_fx"]
    arrays["fresh_kimchi_fx"] = _barrier_arrays(
        market,
        funding,
        signals["fresh_kimchi_fx"],
        name="fresh_kimchi_fx",
        lifecycle=lambda _position: {
            "hold_bars": int(fresh_cfg["hold_bars"]),
            "take_bps": float(fresh_cfg["take_bps"]),
            "stop_bps": float(fresh_cfg["stop_bps"]),
            "source": None,
        },
        start=start,
        end=end,
    )
    arrays["frozen_annual_rank7"] = _barrier_arrays(
        market,
        funding,
        signals["frozen_annual_rank7"],
        name="frozen_annual_rank7",
        lifecycle=lambda position: rank7_lifecycles[int(position)],
        start=start,
        end=end,
    )
    strict_legacy_signals = {
        "legacy_rex_dual_regime_auto": strict_legacy_auto,
        "legacy_rex_dual_regime_short": strict_legacy_short,
    }
    strict_legacy_arrays = {
        name: _fixed_hold_arrays(
            market,
            signal,
            name=name,
            hold_bars=144,
            start=start,
            end=end,
        )
        for name, signal in strict_legacy_signals.items()
    }
    legacy_availability_sensitivity = {
        name: {
            "weekend_fallback_signal_rows": int(
                np.count_nonzero(
                    signals[name][window_mask]
                    != strict_legacy_signals[name][window_mask]
                )
            ),
            "strict_fail_closed_metric": _strict_metric(
                strict_legacy_arrays,
                {name: 1.0},
                dates=dates,
                start=start,
                end=end,
            ),
        }
        for name in strict_legacy_signals
    }
    if set(arrays) != set(SOURCE_PATHS):
        raise RuntimeError(
            f"alpha registry mismatch: {sorted(set(SOURCE_PATHS) - set(arrays))}"
        )

    metrics = {
        name: _strict_metric(
            arrays,
            {name: 1.0},
            dates=dates,
            start=start,
            end=end,
        )
        for name in SOURCE_PATHS
    }
    promoted_weights = {
        str(name): float(weight)
        for name, weight in promoted_cfg["weights"].items()
    }
    promoted_metric = _strict_metric(
        arrays,
        promoted_weights,
        dates=dates,
        start=start,
        end=end,
    )
    signal_diagnostics = {
        name: {
            "raw": int(np.count_nonzero(signal[window_mask])),
            "raw_longs": int(np.count_nonzero(signal[window_mask] > 0)),
            "raw_shorts": int(np.count_nonzero(signal[window_mask] < 0)),
            "accepted_trades": len(arrays[name]["trades"]),
            "skipped_overlap": int(arrays[name]["skipped_overlap"]),
            "skipped_boundary": int(arrays[name]["skipped_boundary"]),
        }
        for name, signal in signals.items()
    }
    sources = sorted(
        {
            str(path)
            for path in SOURCE_PATHS.values()
            if Path(path).is_file()
        }
        | {
            str(PROMOTED_PORTFOLIO),
            str(cfg.historical_context),
            "results/kalman_state_gated_alpha_scan_2026-07-13.json",
            "results/bocpd_state_gated_alpha_scan_2026-07-13.json",
            "results/semimarkov_duration_alpha_scan_2026-07-13.json",
            "results/funding_premium_independent_gate_top10_manifest_2026-07-13.json",
        }
    )
    report = {
        "schema_version": 1,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "mode": "all_frozen_atomic_alpha_completed_bar_monthly_replay",
        "accounting_version": "same_btc_low_high_v1",
        "retrospective_not_pristine_oos": True,
        "config": {
            **asdict(cfg),
            "env_path": "<redacted>",
            "output": str(cfg.output),
            "docs_output": str(cfg.docs_output),
            "historical_context": str(cfg.historical_context),
            "enriched_cache": (
                None if cfg.enriched_cache is None else str(cfg.enriched_cache)
            ),
            "features_cache": (
                None if cfg.features_cache is None else str(cfg.features_cache)
            ),
            "funding_cache": (
                None if cfg.funding_cache is None else str(cfg.funding_cache)
            ),
        },
        "window": {
            "requested_start": str(requested_start),
            "requested_end_exclusive": str(requested_end),
            "start": str(start),
            "end_exclusive": str(end),
            "last_completed_bar": str(dates.iloc[-1]),
            "bars": int(window_mask.sum()),
            "calendar_days": (end - start).total_seconds() / 86_400.0,
        },
        "comparison_contract": {
            "unit_leverage": BASE_LEVERAGE,
            "cost_rate_each_side": COST_RATE,
            "entry": "next_completed_5m_bar_open",
            "stride": "absolute UTC research clock; explicit offset else stride-1",
            "overlap": "one open trade per atomic alpha",
            "source_availability": (
                "fail closed; code-frozen legacy REX retains its explicit "
                "weekend FX-calendar fallback and reports strict sensitivity"
            ),
            "strict_mdd": "same BTC upper-before-lower intrabar envelope",
            "split_boundary": "flat start; only split-contained exits admitted",
            "fixed_hold_funding": "excluded for parity with frozen event-path research accounting",
            "fresh_rank7_funding": "realized funding included",
        },
        "inventory": _inventory(),
        "data_quality": {
            "loaded_market_rows_before_asof_cap": loaded_market_rows,
            "loaded_funding_rows_before_asof_cap": loaded_funding_rows,
            "asof_cap": asof_cap,
            "market_rows_with_warmup": len(market),
            "market_start": str(dates.iloc[0]),
            "market_end": str(dates.iloc[-1]),
            "feature_columns": len(features.columns),
            "availability": _availability_summary(market, window_mask),
            "alt_pool_coverage_pct": float(
                100.0
                * (
                    pd.to_numeric(
                        features.loc[window_mask, "alt_pool_available"],
                        errors="coerce",
                    )
                    > 0.5
                ).mean()
            ),
            "upbit_volume_coverage_pct": float(
                100.0
                * (
                    pd.to_numeric(
                        features.loc[window_mask, "upbit_volume_available"],
                        errors="coerce",
                    )
                    > 0.5
                ).mean()
            ),
            "window_market_hash": _frame_hash(
                market.loc[window_mask].reset_index(drop=True),
                [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "open_interest",
                    "funding_rate",
                    "premium_index",
                    "usdkrw",
                    "kimchi_premium",
                ],
            ),
        },
        "activity_flow_frozen_fit": activity_flow_meta,
        "metrics": metrics,
        "promoted_gross8": promoted_metric,
        "promoted_weights": promoted_weights,
        "signal_diagnostics": signal_diagnostics,
        "duplicate_groups": _duplicate_groups(
            signals,
            arrays,
            window_mask,
        ),
        "rank7_diagnostics": rank7_diagnostics,
        "funding_lr_diagnostics": funding_meta,
        "legacy_rex_availability_sensitivity": (
            legacy_availability_sensitivity
        ),
        "trades": {
            name: arrays[name]["trades"] for name in SOURCE_PATHS
        },
        "source_sha256": {path: _sha256(path) for path in sources},
        "interpretation": {
            "primary": "absolute_return_pct, strict_mdd_pct, and trades",
            "partial_period_cagr_warning": (
                "CAGR annualizes 26.625 days and is not statistically stable"
            ),
            "universe_boundary": (
                "all frozen atomic contracts; portfolio combinations and "
                "unfrozen raw scan rows excluded"
            ),
            "statistical_power": (
                "26.625 days and at most 14 accepted trades per alpha are "
                "insufficient for standalone promotion"
            ),
        },
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    cfg.docs_output.parent.mkdir(parents=True, exist_ok=True)
    cfg.docs_output.write_text(_render_docs(report) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="/home/pakchu/rllm/.env")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--docs-output", default=str(DEFAULT_DOCS))
    parser.add_argument("--historical-context", default=str(DEFAULT_HISTORY))
    parser.add_argument("--start", default=Config.start)
    parser.add_argument("--end", default=Config.end)
    parser.add_argument("--asof", default=Config.asof)
    parser.add_argument(
        "--lookback-minutes",
        type=int,
        default=Config.lookback_minutes,
    )
    parser.add_argument("--enriched-cache", default="")
    parser.add_argument("--features-cache", default="")
    parser.add_argument("--funding-cache", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(
        Config(
            env_path=Path(args.env),
            output=Path(args.output),
            docs_output=Path(args.docs_output),
            historical_context=Path(args.historical_context),
            start=str(args.start),
            end=str(args.end),
            asof=str(args.asof),
            lookback_minutes=int(args.lookback_minutes),
            enriched_cache=(
                Path(args.enriched_cache) if args.enriched_cache else None
            ),
            features_cache=(
                Path(args.features_cache) if args.features_cache else None
            ),
            funding_cache=(
                Path(args.funding_cache) if args.funding_cache else None
            ),
        )
    )
    ranked = sorted(
        (
            {
                "name": name,
                "absolute_return_pct": row["absolute_return_pct"],
                "strict_mdd_pct": row["strict_mdd_pct"],
                "trades": row["trades"],
            }
            for name, row in report["metrics"].items()
        ),
        key=lambda row: row["absolute_return_pct"],
        reverse=True,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "docs": str(args.docs_output),
                "window": report["window"],
                "scored_alphas": len(report["metrics"]),
                "promoted_gross8": report["promoted_gross8"],
                "ranked": ranked,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
