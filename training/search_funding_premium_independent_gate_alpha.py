"""Gate the fixed non-REX funding/premium squeeze with independent features.

The base long alpha is fixed before this experiment.  Gate thresholds are fit
on 2020-2022 and gate variants are selected on 2023 plus both 2023 half-years.
The Top-10 manifest is written while 2024+ market rows are physically absent,
then replayed unchanged on 2024, 2025 and 2026 YTD.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import _completed_timeframe_features
from training.long_regime_combo_scan import LongComboScanConfig, _load_market
from training.search_crossvenue_microstructure_consensus_alpha import (
    WINDOWS,
    _activation_hash,
    _executed_signal_dates,
    _jaccard,
    _load_reference_dates,
)
from training.search_jump_variation_bidirectional_alpha import features as build_jump_features
from training.search_liquidity_recovery_bidirectional_alpha import features as build_liquidity_features
from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop
from training.search_volume_clock_bidirectional_alpha import features as build_volume_clock_features


FUNDING_THRESHOLD = -0.0000167
TREND_96_THRESHOLD = 0.007485218212390219
PREMIUM_CHANGE_THRESHOLD = -0.00023471
DAILY_MOMENTUM_THRESHOLD = 0.0940403008961932
HOLD_BARS = 576
STRIDE_BARS = 12

GATE_FEATURES = (
    "vc_ret_0p25",
    "vc_duration_0p25",
    "vc_imbalance_0p25",
    "vc_flow_speed_0p25",
    "vc_ret_0p5",
    "vc_duration_0p5",
    "vc_imbalance_0p5",
    "vc_flow_speed_0p5",
    "lr_flow_recovery",
    "lr_signed_eff_24",
    "lr_signed_eff_72",
    "lr_signed_eff_144",
    "lr_flow_24",
    "lr_flow_72",
    "lr_flow_144",
    "lr_impact_72",
    "jv_jump_ratio_24",
    "jv_jump_ratio_72",
    "jv_signed_jump_24",
    "jv_signed_jump_72",
    "jv_flow_recovery",
    "jv_vov",
)


@dataclass(frozen=True)
class IndependentGateConfig(LongComboScanConfig):
    manifest_output: str = ""
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    quantiles: str = "0.10,0.20,0.30"
    top_n: int = 10
    top_per_feature: int = 2
    min_fit_trades: int = 80
    min_select_trades: int = 24
    min_half_trades: int = 8


def _parse_csv(raw: str, cast: Any) -> list[Any]:
    return [cast(part.strip()) for part in str(raw).split(",") if part.strip()]


def _window_mask(dates: pd.Series, name: str) -> np.ndarray:
    start, end = WINDOWS[name]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)


def _build_base_components(market: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    close = pd.to_numeric(market["close"], errors="coerce")
    trend_96 = close / close.shift(95).replace(0.0, np.nan) - 1.0
    daily = _completed_timeframe_features(
        market,
        prefix="htf_1d",
        resample_rule="1D",
        min_source_rows=24 * 60 * 4,
    )
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


def _build_gate_features(market: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=market.index)
    features = build_volume_clock_features(market, features)
    features = build_jump_features(market, features)
    features = build_liquidity_features(market, features)
    missing = [name for name in GATE_FEATURES if name not in features.columns]
    if missing:
        raise RuntimeError(f"missing pre-registered gate features: {missing}")
    forbidden = [name for name in GATE_FEATURES if "rex" in name.lower()]
    if forbidden:
        raise RuntimeError(f"REX feature entered independent gate search: {forbidden}")
    return features


def _fit_quantile(values: np.ndarray, fit_mask: np.ndarray, quantile: float) -> float:
    reference = values[fit_mask & np.isfinite(values)]
    if len(reference) < 50_000:
        raise ValueError(f"insufficient gate fit observations: {len(reference)}")
    return float(np.quantile(reference, quantile))


def _gate_mask(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    finite = np.isfinite(values)
    mode = spec["gate_mode"]
    if mode == "lower":
        return finite & (values <= spec["lower"])
    if mode == "upper":
        return finite & (values >= spec["upper"])
    if mode == "central":
        return finite & (values >= spec["lower"]) & (values <= spec["upper"])
    if mode == "outer":
        return finite & ((values <= spec["lower"]) | (values >= spec["upper"]))
    raise ValueError(f"unknown gate mode: {mode}")


def _apply_gate(
    funding_component: np.ndarray,
    premium_component: np.ndarray,
    gate: np.ndarray,
    target_component: str,
) -> np.ndarray:
    if target_component == "all":
        return (funding_component | premium_component) & gate
    if target_component == "funding":
        return (funding_component & gate) | premium_component
    if target_component == "premium":
        return funding_component | (premium_component & gate)
    raise ValueError(f"unknown target component: {target_component}")


def _gate_specs(features: pd.DataFrame, fit_mask: np.ndarray, quantiles: list[float]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for feature, tail, gate_mode, target_component in itertools.product(
        GATE_FEATURES,
        quantiles,
        ("lower", "upper", "central", "outer"),
        ("all", "funding", "premium"),
    ):
        values = features[feature].to_numpy(float)
        specs.append(
            {
                "feature": feature,
                "tail": float(tail),
                "lower": _fit_quantile(values, fit_mask, tail),
                "upper": _fit_quantile(values, fit_mask, 1.0 - tail),
                "gate_mode": gate_mode,
                "target_component": target_component,
            }
        )
    return specs


def _simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    cfg: IndependentGateConfig,
    window: str,
    extremes: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    return _simulate_no_stop(
        market,
        dates,
        long_active,
        np.zeros(len(market), dtype=bool),
        window=window,
        hold_bars=HOLD_BARS,
        stride_bars=STRIDE_BARS,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        extremes=extremes,
        windows=WINDOWS,
    )


def _selection_score(stats: dict[str, dict[str, Any]], cfg: IndependentGateConfig) -> float:
    fit = stats["fit_2020_2022"]
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


def _select_manifest(cfg: IndependentGateConfig) -> dict[str, Any]:
    selection_cfg = replace(cfg, exclude_from="2024-01-01")
    market = _load_market(selection_cfg)
    if pd.Timestamp(market["date"].max()) >= pd.Timestamp("2024-01-01"):
        raise RuntimeError("selection phase contains post-2023 rows")
    dates = pd.to_datetime(market["date"])
    funding_component, premium_component = _build_base_components(market)
    features = _build_gate_features(market)
    fit_mask = _window_mask(dates, "fit_2020_2022")
    specs = _gate_specs(features, fit_mask, _parse_csv(cfg.quantiles, float))
    extremes = (
        _future_extreme(market["low"].to_numpy(float), HOLD_BARS, "min"),
        _future_extreme(market["high"].to_numpy(float), HOLD_BARS, "max"),
    )
    base_active = funding_component | premium_component
    baseline = {
        window: _simulate(market, dates, base_active, cfg, window, extremes)
        for window in ("fit_2020_2022", "select_2023", "select_2023_h1", "select_2023_h2")
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        gate = _gate_mask(features[spec["feature"]].to_numpy(float), spec)
        active = _apply_gate(funding_component, premium_component, gate, spec["target_component"])
        mask_hash = _activation_hash(active, np.zeros(len(active), dtype=bool))
        if mask_hash in seen:
            continue
        seen.add(mask_hash)
        stats = {
            window: _simulate(market, dates, active, cfg, window, extremes)
            for window in ("fit_2020_2022", "select_2023", "select_2023_h1", "select_2023_h2")
        }
        score = _selection_score(stats, cfg)
        if score <= -1e11:
            continue
        rows.append({**spec, "activation_hash": mask_hash, "selection_score": score, "selection_stats": stats})
    rows.sort(
        key=lambda row: (
            row["selection_score"],
            row["selection_stats"]["select_2023"]["ratio"],
            row["selection_stats"]["select_2023"]["return_pct"],
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in rows:
        feature = row["feature"]
        if counts.get(feature, 0) >= cfg.top_per_feature:
            continue
        selected.append(row)
        counts[feature] = counts.get(feature, 0) + 1
        if len(selected) >= cfg.top_n:
            break
    core = {
        "protocol": {
            "base_alpha": "fixed funding10_trend70 OR premium20_mom90 long",
            "fit": WINDOWS["fit_2020_2022"],
            "selection": WINDOWS["select_2023"],
            "future_rows_physically_excluded_before_manifest": True,
            "gate_sources": "promoted volume-clock, jump-variation and liquidity-recovery features only",
            "entry": "next 5m open",
            "exit": f"fixed {HOLD_BARS} bars; stride {STRIDE_BARS}; no TP/SL",
            "cost": "6bp/side at 0.5x",
            "mdd": "strict favorable-high-water then adverse-extreme path",
        },
        "search_space": {
            "raw_specs": len(specs),
            "effective_unique_masks": len(seen),
            "eligible_variants": len(rows),
            "top_n": cfg.top_n,
            "top_per_feature": cfg.top_per_feature,
        },
        "baseline_selection_stats": baseline,
        "selected": selected,
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manifest = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        **core,
    }
    path = Path(cfg.manifest_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def _replay(
    cfg: IndependentGateConfig,
    manifest: dict[str, Any],
    rex_reference_jsonl: list[str],
) -> dict[str, Any]:
    market = _load_market(cfg)
    dates = pd.to_datetime(market["date"])
    funding_component, premium_component = _build_base_components(market)
    features = _build_gate_features(market)
    extremes = (
        _future_extreme(market["low"].to_numpy(float), HOLD_BARS, "min"),
        _future_extreme(market["high"].to_numpy(float), HOLD_BARS, "max"),
    )
    base_active = funding_component | premium_component
    baseline = {window: _simulate(market, dates, base_active, cfg, window, extremes) for window in WINDOWS}
    reference_dates = _load_reference_dates(rex_reference_jsonl)
    selected: list[dict[str, Any]] = []
    for rank, frozen in enumerate(manifest["selected"], start=1):
        spec = {key: frozen[key] for key in ("feature", "tail", "lower", "upper", "gate_mode", "target_component")}
        gate = _gate_mask(features[spec["feature"]].to_numpy(float), spec)
        active = _apply_gate(funding_component, premium_component, gate, spec["target_component"])
        stats = {window: _simulate(market, dates, active, cfg, window, extremes) for window in WINDOWS}
        for window in ("fit_2020_2022", "select_2023", "select_2023_h1", "select_2023_h2"):
            if stats[window] != frozen["selection_stats"][window]:
                raise RuntimeError(f"selection replay drift rank={rank} window={window}")
        overlap = {}
        for window in ("test_2024", "eval_2025", "holdout_2026"):
            candidate_dates = _executed_signal_dates(
                market,
                dates,
                active,
                np.zeros(len(active), dtype=bool),
                window=window,
                hold_bars=HOLD_BARS,
                stride_bars=STRIDE_BARS,
            )
            start, end = WINDOWS[window]
            reference = {date for date in reference_dates if pd.Timestamp(start) <= date < pd.Timestamp(end)}
            overlap[window] = {
                "candidate_signals": len(candidate_dates),
                "rex_reference_signals": len(reference),
                "exact_intersection": len(candidate_dates & reference),
                "jaccard": _jaccard(candidate_dates, reference),
            }
        test, evaluation, holdout, combined = (
            stats["test_2024"],
            stats["eval_2025"],
            stats["holdout_2026"],
            stats["oos_2024_2026"],
        )
        enough = test["trades"] >= 20 and evaluation["trades"] >= 20 and holdout["trades"] >= 12
        passes_alpha_pool = enough and test["ratio"] >= 2.5 and evaluation["ratio"] >= 2.5
        passes_live_grade = (
            passes_alpha_pool
            and holdout["ratio"] >= 3.0
            and combined["ratio"] >= 3.0
            and combined["p_value_mean_return_approx"] <= 0.05
            and max(item["jaccard"] for item in overlap.values()) < 0.05
        )
        selected.append(
            {
                "manifest_rank": rank,
                **frozen,
                "stats": stats,
                "rex_activation_overlap": overlap,
                "passes_alpha_pool": passes_alpha_pool,
                "passes_live_grade": passes_live_grade,
            }
        )
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "manifest": cfg.manifest_output,
        "manifest_sha256": manifest["sha256"],
        "protocol": manifest["protocol"],
        "baseline": baseline,
        "selected": selected,
        "alpha_pool_qualifiers": [row for row in selected if row["passes_alpha_pool"]],
        "live_grade": [row for row in selected if row["passes_live_grade"]],
    }


def run(cfg: IndependentGateConfig, rex_reference_jsonl: list[str]) -> dict[str, Any]:
    if not cfg.funding_csv or not cfg.premium_csv:
        raise ValueError("funding_csv and premium_csv are required")
    if not cfg.manifest_output:
        raise ValueError("manifest_output is required")
    manifest = _select_manifest(cfg)
    report = _replay(cfg, manifest, rex_reference_jsonl)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def parse_args() -> tuple[IndependentGateConfig, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--funding-csv", required=True)
    parser.add_argument("--premium-csv", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-from", default="2026-06-02")
    parser.add_argument("--quantiles", default="0.10,0.20,0.30")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-per-feature", type=int, default=2)
    parser.add_argument("--rex-reference-jsonl", action="append", default=[])
    args = parser.parse_args()
    references = list(args.rex_reference_jsonl)
    delattr(args, "rex_reference_jsonl")
    return IndependentGateConfig(**vars(args)), references


def main() -> None:
    cfg, references = parse_args()
    report = run(cfg, references)
    print(
        json.dumps(
            {
                "manifest_sha256": report["manifest_sha256"],
                "baseline": report["baseline"],
                "selected": len(report["selected"]),
                "alpha_pool_qualifiers": len(report["alpha_pool_qualifiers"]),
                "live_grade": len(report["live_grade"]),
                "top": report["selected"][:3],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
