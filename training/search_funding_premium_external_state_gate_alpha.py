"""Gate the fixed funding/premium squeeze alpha with external BTC derivative states.

The base long alpha is fixed before this experiment.  External feature
thresholds are fitted on 2021-04-15..2023-01-01, variants are selected on full
2023 plus both 2023 half-years, and a Top-10 manifest is frozen while market,
Binance USD-M metrics, and Deribit DVOL inputs are physically truncated before
2024.  The frozen manifest is then replayed unchanged on 2024, 2025, 2026 YTD,
and combined 2024-2026.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from preprocessing.binance_aux_features import (
    attach_binance_um_aux_frames,
    normalise_funding_history_frame,
    normalise_premium_index_frame,
)
from preprocessing.market_features import _completed_timeframe_features
from training.search_deribit_dvol_alpha import attach_dvol, build_dvol_features
from training.search_funding_premium_independent_gate_alpha import (
    DAILY_MOMENTUM_THRESHOLD,
    FUNDING_THRESHOLD,
    PREMIUM_CHANGE_THRESHOLD,
    TREND_96_THRESHOLD,
    _apply_gate,
    _gate_mask,
)
from training.search_liquidity_recovery_bidirectional_alpha import features as build_liquidity_features
from training.search_positioning_disagreement_alpha import (
    _attach_delayed_metrics,
    _future_extreme,
    _simulate_no_stop,
    build_positioning_features,
)
from training.search_positioning_hgb_path_alpha import _feature_hash, _read_before


FIT_START = "2021-04-15"
FIT_END = "2023-01-01"
SELECTION_END = "2024-01-01"
HOLD_BARS = 576
STRIDE_BARS = 12

WINDOWS = {
    "fit": (FIT_START, FIT_END),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
    "test_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026": ("2026-01-01", "2026-06-02"),
    "oos_2024_2026": ("2024-01-01", "2026-06-02"),
}

QUARTER_WINDOWS = {
    "2024Q1": ("2024-01-01", "2024-04-01"),
    "2024Q2": ("2024-04-01", "2024-07-01"),
    "2024Q3": ("2024-07-01", "2024-10-01"),
    "2024Q4": ("2024-10-01", "2025-01-01"),
    "2025Q1": ("2025-01-01", "2025-04-01"),
    "2025Q2": ("2025-04-01", "2025-07-01"),
    "2025Q3": ("2025-07-01", "2025-10-01"),
    "2025Q4": ("2025-10-01", "2026-01-01"),
    "2026Q1": ("2026-01-01", "2026-04-01"),
    "2026Q2_to_Jun02": ("2026-04-01", "2026-06-02"),
}

BASE_ADMISSION_FEATURES = (
    "btc_funding_rate",
    "btc_premium_index_change",
    "btc_trend_96",
    "btc_daily_mom4",
    "btc_lr_impact_72",
)

POSITIONING_DISAGREEMENT_FEATURES = (
    "smart_size_z144",
    "smart_size_z2016",
    "smart_size_z8640",
    "smart_retail_z144",
    "smart_retail_z2016",
    "smart_retail_z8640",
    "topacct_retail_z144",
    "topacct_retail_z2016",
    "smart_retail_chg2016",
    "smart_size_chg2016",
    "smart_absorb_144",
    "top_acct_z144",
    "top_acct_z2016",
    "global_acct_z144",
    "global_acct_z2016",
    "top_acct_chg2016",
    "global_acct_chg2016",
    "crowding_144",
)

OI_FEATURES = (
    "oi_z288",
    "oi_z2016",
    "oi_z8640",
    "oi_logchg288",
    "oi_logchg2016",
    "oi_logchg8640",
)

DVOL_FEATURES = (
    "dvol_z2016",
    "dvol_z8640",
    "dvol_z25920",
    "dvol_logchg2016",
    "dvol_logchg8640",
    "dvol_logchg25920",
)


@dataclass(frozen=True)
class ExternalStateGateConfig:
    input_csv: str
    metrics_csv: str
    dvol_csv: str
    funding_csv: str
    premium_csv: str
    output: str
    manifest_output: str
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    quantiles: str = "0.10,0.20,0.30"
    top_n: int = 10
    top_per_feature: int = 2
    min_fit_observations: int = 20_000
    min_fit_trades: int = 80
    min_select_trades: int = 24
    min_half_trades: int = 8
    max_abs_spearman: float = 0.30
    metrics_tolerance: str = "5min"
    metrics_delay_bars: int = 1
    dvol_tolerance: str = "65min"
    funding_tolerance: str = "12h"
    premium_tolerance: str = "65min"
    refresh_manifest: bool = False


def _parse_csv(raw: str, cast: Any) -> list[Any]:
    return [cast(part.strip()) for part in str(raw).split(",") if part.strip()]


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    minimum = max(24, window // 2)
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
    return (series - mean) / std


def _window_mask(dates: pd.Series, name: str) -> np.ndarray:
    start, end = WINDOWS[name]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(cfg: ExternalStateGateConfig) -> dict[str, str]:
    return {
        str(Path(path)): _file_sha256(path)
        for path in (cfg.input_csv, cfg.metrics_csv, cfg.dvol_csv, cfg.funding_csv, cfg.premium_csv)
    }


def _frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(map(str, frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="<u8").tobytes())
    return digest.hexdigest()


def _read_premium_before(path: str, cutoff: str) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    boundary = pd.Timestamp(cutoff)
    for chunk in pd.read_csv(path, compression="infer", chunksize=100_000):
        close_time = pd.to_datetime(
            pd.to_numeric(chunk["close_time"], errors="raise"),
            unit="ms",
            utc=True,
        ).dt.tz_convert(None)
        keep = close_time < boundary
        if keep.any():
            chunks.append(chunk.loc[keep].copy())
        if (~keep).any():
            break
    if not chunks:
        raise ValueError(f"no premium rows before {cutoff} in {path}")
    return pd.concat(chunks, ignore_index=True)


def _load_bundle(
    cfg: ExternalStateGateConfig,
    *,
    cutoff: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, str]]:
    boundary = pd.Timestamp(cutoff)
    market_raw = _read_before(cfg.input_csv, "date", cutoff)
    funding_raw = _read_before(cfg.funding_csv, "date", cutoff)
    premium_raw = _read_premium_before(cfg.premium_csv, cutoff)
    metrics = _read_before(cfg.metrics_csv, "create_time", cutoff)
    dvol = _read_before(cfg.dvol_csv, "close_time", cutoff)
    source_prefix_hashes = {
        "market": _frame_hash(market_raw),
        "funding": _frame_hash(funding_raw),
        "premium": _frame_hash(premium_raw),
        "metrics": _frame_hash(metrics),
        "dvol": _frame_hash(dvol),
    }

    market = market_raw
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    market = attach_binance_um_aux_frames(
        market,
        funding_frame=normalise_funding_history_frame(funding_raw),
        premium_frame=normalise_premium_index_frame(premium_raw),
        funding_tolerance=cfg.funding_tolerance,
        premium_tolerance=cfg.premium_tolerance,
    )

    metrics_times = pd.to_datetime(metrics["create_time"], utc=True, errors="raise").dt.tz_convert(None)
    if len(metrics_times) and metrics_times.max() >= boundary:
        raise RuntimeError("metrics source rows were not physically truncated before cutoff")
    market = _attach_delayed_metrics(
        market,
        metrics,
        tolerance=cfg.metrics_tolerance,
        delay_bars=cfg.metrics_delay_bars,
    )
    market["oi_available"] = (
        pd.to_numeric(market.get("sum_open_interest"), errors="coerce").notna()
        & pd.to_datetime(market.get("positioning_source_time"), errors="coerce").notna()
    ).astype(float)

    dvol_times = pd.to_datetime(dvol["close_time"], utc=True, errors="raise").dt.tz_convert(None)
    if len(dvol_times) and dvol_times.max() >= boundary:
        raise RuntimeError("DVOL source rows were not physically truncated before cutoff")
    market = attach_dvol(market, dvol, tolerance=cfg.dvol_tolerance)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= boundary:
        raise RuntimeError("market rows were not physically truncated before cutoff")

    features, availability = build_external_state_features(market)
    base_features = build_base_admission_features(market)
    return market, dates, features, base_features, availability, source_prefix_hashes


def _build_base_components(market: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    close = pd.to_numeric(market["close"], errors="coerce")
    trend_96 = close / close.shift(95).replace(0.0, np.nan) - 1.0
    daily = _completed_timeframe_features(market, prefix="htf_1d", resample_rule="1D", min_source_rows=24 * 60 * 4)
    funding = pd.to_numeric(market["funding_rate"], errors="coerce")
    premium_change = pd.to_numeric(market["premium_index_change"], errors="coerce")
    funding_available = pd.to_numeric(market.get("funding_available", 0.0), errors="coerce").fillna(0.0)
    premium_available = pd.to_numeric(market.get("premium_available", 0.0), errors="coerce").fillna(0.0)
    funding_component = (
        (funding_available.to_numpy(float) > 0.5)
        & (funding.to_numpy(float) <= FUNDING_THRESHOLD)
        & (trend_96.to_numpy(float) >= TREND_96_THRESHOLD)
    )
    premium_component = (
        (premium_available.to_numpy(float) > 0.5)
        & (premium_change.to_numpy(float) <= PREMIUM_CHANGE_THRESHOLD)
        & (daily["htf_1d_return_4"].to_numpy(float) >= DAILY_MOMENTUM_THRESHOLD)
    )
    return funding_component, premium_component


def build_base_admission_features(market: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(market["close"], errors="coerce")
    daily = _completed_timeframe_features(market, prefix="htf_1d", resample_rule="1D", min_source_rows=24 * 60 * 4)
    liquidity = build_liquidity_features(market, pd.DataFrame(index=market.index))
    return pd.DataFrame(
        {
            "btc_funding_rate": pd.to_numeric(market.get("funding_rate"), errors="coerce"),
            "btc_premium_index_change": pd.to_numeric(market.get("premium_index_change"), errors="coerce"),
            "btc_trend_96": close / close.shift(95).replace(0.0, np.nan) - 1.0,
            "btc_daily_mom4": daily["htf_1d_return_4"],
            "btc_lr_impact_72": liquidity["lr_impact_72"],
        },
        index=market.index,
    ).replace([np.inf, -np.inf], np.nan)


def build_external_state_features(market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    columns: dict[str, pd.Series] = {}
    availability: dict[str, str] = {}

    positioning_available = pd.to_numeric(market.get("positioning_available", 0.0), errors="coerce").fillna(0.0)
    oi_available = pd.to_numeric(market.get("oi_available", 0.0), errors="coerce").fillna(0.0)
    oi = np.log(pd.to_numeric(market.get("sum_open_interest"), errors="coerce").where(lambda values: values > 0.0))
    for window in (288, 2016, 8640):
        z_name = f"oi_z{window}"
        chg_name = f"oi_logchg{window}"
        columns[z_name] = _rolling_z(oi, window).where(oi_available > 0.5)
        columns[chg_name] = (oi - oi.shift(window)).where(oi_available > 0.5)
        availability[z_name] = "oi_available"
        availability[chg_name] = "oi_available"

    positioning = build_positioning_features(market)
    missing_positioning = [name for name in POSITIONING_DISAGREEMENT_FEATURES if name not in positioning.columns]
    if missing_positioning:
        raise RuntimeError(f"missing positioning disagreement features: {missing_positioning}")
    for name in POSITIONING_DISAGREEMENT_FEATURES:
        columns[f"pos_{name}"] = positioning[name].where(positioning_available > 0.5)
        availability[f"pos_{name}"] = "positioning_available"

    dvol_features = build_dvol_features(market)
    missing_dvol = [name for name in DVOL_FEATURES if name not in dvol_features.columns]
    if missing_dvol:
        raise RuntimeError(f"missing DVOL features: {missing_dvol}")
    for name in DVOL_FEATURES:
        columns[name] = dvol_features[name].where(dvol_features["dvol_available"] > 0.5)
        availability[name] = "dvol_available"

    frame = pd.DataFrame(columns, index=market.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return frame, availability


def feature_admission(
    features: pd.DataFrame,
    base_features: pd.DataFrame,
    availability: dict[str, str],
    fit_mask: np.ndarray,
    *,
    max_abs_spearman: float,
    min_observations: int,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    admitted: list[str] = []
    audit: dict[str, dict[str, Any]] = {}
    for feature in features.columns:
        values = pd.to_numeric(features[feature], errors="coerce")
        finite_fit = fit_mask & np.isfinite(values.to_numpy(float))
        correlations: dict[str, float] = {}
        pair_counts: dict[str, int] = {}
        for base in BASE_ADMISSION_FEATURES:
            paired = finite_fit & np.isfinite(base_features[base].to_numpy(float))
            pair_counts[base] = int(paired.sum())
            correlations[base] = (
                float(values.loc[paired].corr(base_features.loc[paired, base], method="spearman"))
                if pair_counts[base] >= 100
                else float("nan")
            )
        finite_correlations = [abs(value) for value in correlations.values() if np.isfinite(value)]
        max_abs = max(finite_correlations, default=float("inf"))
        observations = int(finite_fit.sum())
        passes = observations >= min_observations and max_abs < max_abs_spearman and feature in availability
        audit[feature] = {
            "availability_column": availability.get(feature),
            "fit_observations": observations,
            "pair_counts": pair_counts,
            "spearman": correlations,
            "max_abs_spearman": float(max_abs),
            "passes": bool(passes),
        }
        if passes:
            admitted.append(feature)
    return admitted, audit


def _fit_threshold(values: np.ndarray, fit_mask: np.ndarray, quantile: float, *, min_observations: int) -> float:
    reference = values[fit_mask & np.isfinite(values)]
    if len(reference) < min_observations:
        raise ValueError(f"insufficient observations for threshold: {len(reference)}")
    return float(np.quantile(reference, quantile))


def _gate_specs(features: pd.DataFrame, admitted: Iterable[str], fit_mask: np.ndarray, quantiles: list[float], *, min_observations: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for feature, tail, gate_mode, target_component in itertools.product(
        admitted,
        quantiles,
        ("lower", "upper", "central", "outer"),
        ("all", "funding", "premium"),
    ):
        values = features[feature].to_numpy(float)
        specs.append(
            {
                "feature": feature,
                "tail": float(tail),
                "lower": _fit_threshold(values, fit_mask, tail, min_observations=min_observations),
                "upper": _fit_threshold(values, fit_mask, 1.0 - tail, min_observations=min_observations),
                "gate_mode": gate_mode,
                "target_component": target_component,
            }
        )
    return specs


def external_gate_mask(features: pd.DataFrame, spec: dict[str, Any], availability_frame: pd.DataFrame, availability: dict[str, str]) -> np.ndarray:
    feature = spec["feature"]
    active = _gate_mask(features[feature].to_numpy(float), spec)
    availability_column = availability[feature]
    available = pd.to_numeric(availability_frame[availability_column], errors="coerce").fillna(0.0).to_numpy(float) > 0.5
    return active & available


def _activation_hash(active: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(active, dtype=bool)).tobytes()).hexdigest()


def _simulate(market: pd.DataFrame, dates: pd.Series, active: np.ndarray, cfg: ExternalStateGateConfig, *, window: str, extremes: tuple[np.ndarray, np.ndarray], windows: dict[str, tuple[str, str]] = WINDOWS) -> dict[str, Any]:
    return _simulate_no_stop(
        market,
        dates,
        active,
        np.zeros(len(active), dtype=bool),
        window=window,
        hold_bars=HOLD_BARS,
        stride_bars=STRIDE_BARS,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        extremes=extremes,
        windows=windows,
    )


def _selection_score(stats: dict[str, dict[str, Any]], cfg: ExternalStateGateConfig) -> float:
    fit = stats["fit"]
    select = stats["select_2023"]
    h1 = stats["select_2023_h1"]
    h2 = stats["select_2023_h2"]
    if fit["trades"] < cfg.min_fit_trades or select["trades"] < cfg.min_select_trades:
        return -1e12
    if h1["trades"] < cfg.min_half_trades or h2["trades"] < cfg.min_half_trades:
        return -1e12
    if min(fit["cagr_pct"], select["cagr_pct"], h1["cagr_pct"], h2["cagr_pct"]) <= 0.0:
        return -1e12
    if fit["strict_mdd_pct"] > 25.0 or select["strict_mdd_pct"] > 15.0:
        return -1e12
    ratios = np.asarray([fit["ratio"], select["ratio"], h1["ratio"], h2["ratio"]], dtype=float)
    return float(np.min(ratios) + 0.30 * np.median(ratios) + min(0.25, select["trades"] / 200.0))


def _select_top(rows: list[dict[str, Any]], *, top_n: int, top_per_feature: int) -> list[dict[str, Any]]:
    rows = sorted(
        rows,
        key=lambda row: (
            row["selection_score"],
            row["selection_stats"]["select_2023"]["ratio"],
            row["selection_stats"]["select_2023"]["return_pct"],
            row["feature"],
            row["gate_mode"],
            row["target_component"],
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in rows:
        feature = row["feature"]
        if counts.get(feature, 0) >= top_per_feature:
            continue
        selected.append(row)
        counts[feature] = counts.get(feature, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def _availability_frame(market: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "positioning_available": pd.to_numeric(market.get("positioning_available", 0.0), errors="coerce").fillna(0.0),
            "oi_available": pd.to_numeric(market.get("oi_available", 0.0), errors="coerce").fillna(0.0),
            "dvol_available": pd.to_numeric(market.get("dvol_available", 0.0), errors="coerce").fillna(0.0),
        },
        index=market.index,
    )


def _manifest_core_hash(core: dict[str, Any]) -> str:
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_manifest(manifest: dict[str, Any]) -> None:
    core = {key: value for key, value in manifest.items() if key not in {"as_of", "sha256"}}
    if manifest.get("sha256") != _manifest_core_hash(core):
        raise RuntimeError("manifest content does not match its frozen SHA-256")


def _select_manifest(cfg: ExternalStateGateConfig) -> dict[str, Any]:
    market, dates, features, base_features, availability, source_prefix_hashes = _load_bundle(
        cfg,
        cutoff=SELECTION_END,
    )
    availability_frame = _availability_frame(market)
    fit_mask = _window_mask(dates, "fit")
    admitted, correlation_audit = feature_admission(
        features,
        base_features,
        availability,
        fit_mask,
        max_abs_spearman=cfg.max_abs_spearman,
        min_observations=cfg.min_fit_observations,
    )
    specs = _gate_specs(features, admitted, fit_mask, _parse_csv(cfg.quantiles, float), min_observations=cfg.min_fit_observations)
    funding_component, premium_component = _build_base_components(market)
    extremes = (_future_extreme(market["low"].to_numpy(float), HOLD_BARS, "min"), _future_extreme(market["high"].to_numpy(float), HOLD_BARS, "max"))
    base_active = funding_component | premium_component
    baseline = {window: _simulate(market, dates, base_active, cfg, window=window, extremes=extremes) for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        gate = external_gate_mask(features, spec, availability_frame, availability)
        active = _apply_gate(funding_component, premium_component, gate, spec["target_component"])
        activation_hash = _activation_hash(active)
        if activation_hash in seen:
            continue
        seen.add(activation_hash)
        stats = {window: _simulate(market, dates, active, cfg, window=window, extremes=extremes) for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")}
        score = _selection_score(stats, cfg)
        if score <= -1e11:
            continue
        rows.append({**spec, "activation_hash": activation_hash, "selection_score": score, "selection_stats": stats})
    selected = _select_top(rows, top_n=cfg.top_n, top_per_feature=cfg.top_per_feature)
    core = {
        "protocol": {
            "base_alpha": "fixed funding squeeze OR premium squeeze long; thresholds inherited from independent-gate search",
            "threshold_fit": (FIT_START, FIT_END),
            "selection": {name: WINDOWS[name] for name in ("select_2023", "select_2023_h1", "select_2023_h2")},
            "all_future_market_and_aux_rows_physically_excluded_before_manifest": True,
            "external_feature_sources": "delayed Binance UM positioning/OI plus Deribit DVOL only",
            "feature_admission": f"fit Spearman max |rho| < {cfg.max_abs_spearman} versus {BASE_ADMISSION_FEATURES}; finite per-feature availability required",
            "included_feature_families": {"oi": OI_FEATURES, "positioning_disagreement": POSITIONING_DISAGREEMENT_FEATURES, "dvol": DVOL_FEATURES},
            "entry": "next 5m open",
            "exit": f"fixed {HOLD_BARS} bars; stride {STRIDE_BARS}; no TP/SL",
            "cost": "5bp fee + 1bp slippage per side at 0.5x",
            "mdd": "strict favorable-high-water then adverse OHLC extreme",
            "status_ceiling": "shadow research: 2024+ has been inspected by the broader program",
        },
        "source_prefix_hashes": source_prefix_hashes,
        "external_feature_hash": _feature_hash(features, dates),
        "base_feature_hash": _feature_hash(base_features, dates),
        "availability_hash": _feature_hash(availability_frame, dates),
        "feature_availability": availability,
        "feature_admission_audit": correlation_audit,
        "search_space": {"admitted_features": admitted, "raw_specs": len(specs), "effective_unique_masks": len(seen), "eligible_variants": len(rows), "top_n": cfg.top_n, "top_per_feature": cfg.top_per_feature},
        "baseline_selection_stats": baseline,
        "selected": selected,
    }
    manifest = {"as_of": datetime.now(timezone.utc).isoformat(), "sha256": _manifest_core_hash(core), **core}
    path = Path(cfg.manifest_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    return manifest


def _replay(cfg: ExternalStateGateConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_manifest(manifest)
    _, _, _, _, _, source_prefix_hashes = _load_bundle(cfg, cutoff=SELECTION_END)
    if source_prefix_hashes != manifest["source_prefix_hashes"]:
        raise RuntimeError("pre-2024 source prefixes changed after manifest freeze")
    market, dates, features, base_features, availability, _ = _load_bundle(cfg, cutoff=cfg.exclude_from)
    availability_frame = _availability_frame(market)
    prefix = dates < pd.Timestamp(SELECTION_END)
    prefix_dates = dates.loc[prefix].reset_index(drop=True)
    if _feature_hash(features.loc[prefix].reset_index(drop=True), prefix_dates) != manifest["external_feature_hash"]:
        raise RuntimeError("pre-2024 external feature prefix changed during full replay")
    if _feature_hash(base_features.loc[prefix].reset_index(drop=True), prefix_dates) != manifest["base_feature_hash"]:
        raise RuntimeError("pre-2024 BTC admission-feature prefix changed during full replay")
    if _feature_hash(availability_frame.loc[prefix].reset_index(drop=True), prefix_dates) != manifest["availability_hash"]:
        raise RuntimeError("pre-2024 external availability prefix changed during full replay")
    if availability != manifest["feature_availability"]:
        raise RuntimeError("feature availability mapping changed during replay")

    funding_component, premium_component = _build_base_components(market)
    extremes = (_future_extreme(market["low"].to_numpy(float), HOLD_BARS, "min"), _future_extreme(market["high"].to_numpy(float), HOLD_BARS, "max"))
    base_active = funding_component | premium_component
    baseline = {window: _simulate(market, dates, base_active, cfg, window=window, extremes=extremes) for window in WINDOWS}
    selected: list[dict[str, Any]] = []
    for rank, frozen in enumerate(manifest["selected"], start=1):
        spec = {key: frozen[key] for key in ("feature", "tail", "lower", "upper", "gate_mode", "target_component")}
        gate = external_gate_mask(features, spec, availability_frame, availability)
        active = _apply_gate(funding_component, premium_component, gate, spec["target_component"])
        if _activation_hash(active[prefix.to_numpy(bool)]) != frozen["activation_hash"]:
            raise RuntimeError(f"pre-2024 activation drift at rank {rank}")
        stats = {window: _simulate(market, dates, active, cfg, window=window, extremes=extremes) for window in WINDOWS}
        for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2"):
            if stats[window] != frozen["selection_stats"][window]:
                raise RuntimeError(f"selection replay drift rank={rank} window={window}")
        quarterly = {window: _simulate(market, dates, active, cfg, window=window, extremes=extremes, windows=QUARTER_WINDOWS) for window in QUARTER_WINDOWS}
        test, evaluation, holdout, combined = stats["test_2024"], stats["eval_2025"], stats["holdout_2026"], stats["oos_2024_2026"]
        enough = test["trades"] >= 20 and evaluation["trades"] >= 20 and holdout["trades"] >= 10
        passes_alpha_pool = enough and test["ratio"] >= 2.5 and evaluation["ratio"] >= 2.5
        passes_live_grade = passes_alpha_pool and holdout["ratio"] >= 3.0 and combined["ratio"] >= 3.0 and combined["p_value_mean_return_approx"] <= 0.05
        selected.append({"manifest_rank": rank, **frozen, "stats": stats, "quarterly_stats": quarterly, "quarterly_summary": {"positive_return_quarters": sum(row["return_pct"] > 0.0 for row in quarterly.values()), "flat_quarters": sum(row["trades"] == 0 for row in quarterly.values()), "negative_return_quarters": sum(row["return_pct"] < 0.0 for row in quarterly.values()), "total_quarters": len(quarterly)}, "passes_alpha_pool": bool(passes_alpha_pool), "passes_live_grade": bool(passes_live_grade)})
    return {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "manifest": cfg.manifest_output, "manifest_sha256": manifest["sha256"], "protocol": manifest["protocol"], "source_file_hashes_after_manifest_freeze": _source_hashes(cfg), "feature_admission_audit": manifest["feature_admission_audit"], "baseline": baseline, "selected": selected, "alpha_pool_qualifiers": [row for row in selected if row["passes_alpha_pool"]], "live_grade": [row for row in selected if row["passes_live_grade"]]}


def run(cfg: ExternalStateGateConfig) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest_output)
    if manifest_path.exists() and not cfg.refresh_manifest:
        manifest = json.loads(manifest_path.read_text())
        _validate_manifest(manifest)
    else:
        manifest = _select_manifest(cfg)
    report = _replay(cfg, manifest)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    return report


def parse_args() -> ExternalStateGateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--dvol-csv", required=True)
    parser.add_argument("--funding-csv", required=True)
    parser.add_argument("--premium-csv", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-from", default=ExternalStateGateConfig.exclude_from)
    parser.add_argument("--quantiles", default=ExternalStateGateConfig.quantiles)
    parser.add_argument("--top-n", type=int, default=ExternalStateGateConfig.top_n)
    parser.add_argument("--top-per-feature", type=int, default=ExternalStateGateConfig.top_per_feature)
    parser.add_argument("--max-abs-spearman", type=float, default=ExternalStateGateConfig.max_abs_spearman)
    parser.add_argument("--refresh-manifest", action="store_true")
    return ExternalStateGateConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"manifest_sha256": report["manifest_sha256"], "selected": len(report["selected"]), "alpha_pool_qualifiers": len(report["alpha_pool_qualifiers"]), "live_grade": len(report["live_grade"]), "top": report["selected"][:3]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
