"""Fold-safe regime-conditioned selector over event candidate families.

This is a diagnostic bridge between raw candidate-pool probes and LLM/RL
selection.  Each family is treated as an expert.  For every target fold, family
thresholds are fit only on rows before the fold, previous fold outcomes are the
only performance memory, and the target fold outcome is stitched after the
selection has been made.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.event_candidate_pool_probe import _candidate_rows_for_family, _feature_candidates, _load_market, _simulate_rows, _split_mask


@dataclass(frozen=True)
class RegimeFamilySelectorConfig:
    input_csv: str
    output: str
    train_start: str = "2020-01-01"
    eval_start: str = "2023-01-01"
    eval_end: str = "2026-06-01"
    fold_months: int = 6
    hold_bars: int = 288
    entry_delay_bars: int = 1
    window_size: int = 144
    stride_bars: int = 24
    quantile: float = 0.80
    min_train_trades: int = 80
    min_fold_trades: int = 20
    memory_folds: int = 3
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    family_include: str = ""


def _folds(start: str, end: str, months: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    cur = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    idx = 0
    while cur < stop:
        nxt = min(cur + pd.DateOffset(months=int(months)), stop)
        out.append({"name": f"eval_{cur:%Y%m}_{nxt:%Y%m}", "start": str(cur.date()), "end": str(nxt.date())})
        cur = nxt
        idx += 1
    return out


def _safe_sim_score(split: dict[str, Any], *, min_trades: int) -> float:
    sim = split.get("sim", {})
    stats = split.get("trade_stats", {})
    trades = int(sim.get("trade_entries", 0) or 0)
    if trades < int(min_trades):
        return -1e9 + trades
    cagr = float(sim.get("cagr_pct", 0.0) or 0.0)
    mdd = float(sim.get("strict_mdd_pct", 0.0) or 0.0)
    ratio = float(sim.get("cagr_to_strict_mdd", -1e9) or -1e9)
    p = float(stats.get("p_value_mean_ret_approx", 1.0) or 1.0)
    if not np.isfinite(ratio):
        ratio = 0.0
    # Prefer positive CAGR/MDD, but penalize noisy tiny-sample/p-value wins.
    return ratio + 0.02 * cagr - 0.01 * mdd - 0.25 * max(0.0, p - 0.25)


def _regime_vector(features: pd.DataFrame, dates: pd.Series, fold_start: str, *, lookback_days: int = 30) -> dict[str, float]:
    end = pd.Timestamp(fold_start)
    start = end - pd.Timedelta(days=int(lookback_days))
    mask = np.asarray((dates >= start) & (dates < end), dtype=bool)
    if int(mask.sum()) == 0:
        mask = np.asarray(dates < end, dtype=bool)
    names = [
        "trend_96",
        "range_vol",
        "window_drawdown",
        "htf_1d_return_4",
        "htf_1w_return_4",
        "kimchi_premium_zscore",
        "dxy_zscore",
        "usdkrw_zscore",
        "funding_zscore",
        "taker_imbalance",
    ]
    out: dict[str, float] = {}
    for name in names:
        if name not in features:
            out[name] = 0.0
            continue
        vals = features.loc[mask, name].replace([np.inf, -np.inf], np.nan).dropna()
        out[name] = float(vals.mean()) if len(vals) else 0.0
    return out


def _distance(a: dict[str, float], b: dict[str, float], scale: dict[str, float]) -> float:
    total = 0.0
    n = 0
    for k, av in a.items():
        s = max(float(scale.get(k, 1.0) or 1.0), 1e-9)
        bv = float(b.get(k, 0.0) or 0.0)
        total += ((float(av) - bv) / s) ** 2
        n += 1
    return math.sqrt(total / max(1, n))


def _metric_row(result: dict[str, Any]) -> dict[str, Any]:
    sim = result.get("sim", {})
    stats = result.get("trade_stats", {})
    return {
        "cagr_pct": sim.get("cagr_pct"),
        "strict_mdd_pct": sim.get("strict_mdd_pct"),
        "cagr_to_strict_mdd": sim.get("cagr_to_strict_mdd"),
        "trade_entries": sim.get("trade_entries"),
        "side_counts": sim.get("side_counts"),
        "p_value_mean_ret_approx": stats.get("p_value_mean_ret_approx"),
        "mean_trade_ret_pct": stats.get("mean_trade_ret_pct"),
    }


def run(cfg: RegimeFamilySelectorConfig) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    dates = pd.to_datetime(market["date"])
    families = _feature_candidates(features)
    if cfg.family_include:
        needles = [x.strip() for x in str(cfg.family_include).split(",") if x.strip()]
        families = {name: value for name, value in families.items() if any(needle in name for needle in needles)}
        if not families:
            raise ValueError(f"no candidate families matched --family-include={cfg.family_include!r}")

    folds = _folds(cfg.eval_start, cfg.eval_end, cfg.fold_months)
    all_regimes = [_regime_vector(features, dates, f["start"]) for f in folds]
    scales = {k: float(np.std([r.get(k, 0.0) for r in all_regimes]) or 1.0) for k in all_regimes[0]}

    family_fold_results: dict[str, dict[str, Any]] = {}
    initial_prior_results: dict[str, dict[str, Any]] = {}
    fold_rows: list[dict[str, Any]] = []
    selected_events: list[dict[str, Any]] = []
    q = float(np.clip(cfg.quantile, 0.5, 0.99))

    for fold_idx, fold in enumerate(folds):
        train_mask = _split_mask(dates, cfg.train_start, fold["start"])
        fold_mask = _split_mask(dates, fold["start"], fold["end"])
        fold_family_results: dict[str, dict[str, Any]] = {}
        fold_family_rows: dict[str, list[dict[str, Any]]] = {}
        for family, (strength, direction) in families.items():
            x = strength[train_mask & np.isfinite(strength) & (strength > 0.0)]
            if x.size < 100:
                continue
            threshold = float(np.quantile(x, q))
            fold_rows_for_family = _candidate_rows_for_family(market, strength, direction, family=family, threshold=threshold, mask=fold_mask, cfg=cfg)  # type: ignore[arg-type]
            fold_result = _simulate_rows(fold_rows_for_family, market, cfg)  # type: ignore[arg-type]
            if fold_idx == 0:
                train_rows = _candidate_rows_for_family(market, strength, direction, family=family, threshold=threshold, mask=train_mask, cfg=cfg)  # type: ignore[arg-type]
                train_result = _simulate_rows(train_rows, market, cfg)  # type: ignore[arg-type]
                initial_prior_results[family] = {"threshold": threshold, "train": train_result}
            fold_family_results[family] = {"threshold": threshold, "fold": fold_result, "candidate_count": len(fold_rows_for_family)}
            fold_family_rows[family] = fold_rows_for_family

        if not fold_family_results:
            fold_rows.append({"fold": fold, "selected_family": None, "skip": "no_family_results"})
            continue

        if fold_idx == 0:
            selected_family = max(
                fold_family_results,
                key=lambda fam: _safe_sim_score(initial_prior_results.get(fam, {}).get("train", {}), min_trades=int(cfg.min_train_trades)),
            )
            selector_mode = "prefold_prior"
            selector_evidence: list[dict[str, Any]] = []
        else:
            current_regime = all_regimes[fold_idx]
            prev = []
            for j in range(max(0, fold_idx - int(cfg.memory_folds)), fold_idx):
                dist = _distance(current_regime, all_regimes[j], scales)
                prev.append((dist, folds[j]))
            prev.sort(key=lambda x: x[0])
            nearest = prev[: max(1, min(len(prev), int(cfg.memory_folds)))]
            scores: dict[str, float] = {}
            selector_evidence = []
            for fam in fold_family_results:
                vals = []
                for dist, pfold in nearest:
                    hist = family_fold_results.get(pfold["name"], {}).get(fam)
                    if hist is None:
                        continue
                    vals.append(_safe_sim_score(hist["fold"], min_trades=int(cfg.min_fold_trades)) / (1.0 + dist))
                if vals:
                    scores[fam] = float(np.mean(vals))
                else:
                    scores[fam] = -1e9
            selected_family = max(scores, key=scores.get)
            selector_mode = "nearest_prior_regime_folds"
            selector_evidence = [{"fold": pfold["name"], "distance": dist} for dist, pfold in nearest]

        selected_events.extend(fold_family_rows.get(selected_family, []))
        family_fold_results[fold["name"]] = fold_family_results
        fold_rows.append(
            {
                "fold": fold,
                "selector_mode": selector_mode,
                "selected_family": selected_family,
                "selector_evidence": selector_evidence,
                "selected_threshold": fold_family_results[selected_family]["threshold"],
                "selected_metrics": _metric_row(fold_family_results[selected_family]["fold"]),
                "top_fold_diagnostic_not_for_selection": [
                    {"family": fam, "metrics": _metric_row(row["fold"])}
                    for fam, row in sorted(
                        fold_family_results.items(),
                        key=lambda kv: _safe_sim_score(kv[1]["fold"], min_trades=1),
                        reverse=True,
                    )[:5]
                ],
            }
        )

    selected_events.sort(key=lambda r: (str(r.get("entry_date")), str(r.get("family"))))
    final = _simulate_rows(selected_events, market, cfg)  # type: ignore[arg-type]
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "family_count": len(families),
        "folds": fold_rows,
        "final": {"sim": final["sim"], "trade_stats": final["trade_stats"]},
        "leakage_guard": {
            "family_thresholds_fit_before_each_fold": True,
            "family_selection_uses_only_prefold_prior_or_previous_fold_outcomes": True,
            "target_fold_outcome_not_used_for_selection": True,
            "regime_vectors_use_pre_fold_lookback_only": True,
            "zero_strength_candidates_excluded": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fold-safe regime-conditioned candidate-family selector")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-start", default=RegimeFamilySelectorConfig.train_start)
    p.add_argument("--eval-start", default=RegimeFamilySelectorConfig.eval_start)
    p.add_argument("--eval-end", default=RegimeFamilySelectorConfig.eval_end)
    p.add_argument("--fold-months", type=int, default=RegimeFamilySelectorConfig.fold_months)
    p.add_argument("--hold-bars", type=int, default=RegimeFamilySelectorConfig.hold_bars)
    p.add_argument("--entry-delay-bars", type=int, default=RegimeFamilySelectorConfig.entry_delay_bars)
    p.add_argument("--window-size", type=int, default=RegimeFamilySelectorConfig.window_size)
    p.add_argument("--stride-bars", type=int, default=RegimeFamilySelectorConfig.stride_bars)
    p.add_argument("--quantile", type=float, default=RegimeFamilySelectorConfig.quantile)
    p.add_argument("--min-train-trades", type=int, default=RegimeFamilySelectorConfig.min_train_trades)
    p.add_argument("--min-fold-trades", type=int, default=RegimeFamilySelectorConfig.min_fold_trades)
    p.add_argument("--memory-folds", type=int, default=RegimeFamilySelectorConfig.memory_folds)
    p.add_argument("--leverage", type=float, default=RegimeFamilySelectorConfig.leverage)
    p.add_argument("--fee-rate", type=float, default=RegimeFamilySelectorConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=RegimeFamilySelectorConfig.slippage_rate)
    p.add_argument("--wave-trading-root", default=RegimeFamilySelectorConfig.wave_trading_root)
    p.add_argument("--external-tolerance", default=RegimeFamilySelectorConfig.external_tolerance)
    p.add_argument("--family-include", default=RegimeFamilySelectorConfig.family_include)
    return p.parse_args()


def main() -> None:
    rep = run(RegimeFamilySelectorConfig(**vars(parse_args())))
    print(json.dumps({"final": rep["final"], "folds": [{"fold": f["fold"], "selected_family": f.get("selected_family"), "selected_metrics": f.get("selected_metrics")} for f in rep["folds"]]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
