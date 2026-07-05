"""Monthly diagnostics for long-regime score/sizing alpha candidates.

Purpose: find whether the current long alpha is a real long-regime building
block or only a few lucky half-year clusters. Uses full calendar month CAGR,
absolute return, strict MDD, and past-only/train-only score thresholds.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market, _split_mask, _strict_long_sim
from training.long_regime_gate_scan import LongRegimeGateConfig, _entry_active, _gate_active
from training.long_regime_score_gate_validation import _build_score_frame, _score_variant
from training.long_regime_score_sizing_validation import _fit_tier_leverage, _parse_tier_spec, _strict_long_sim_variable_leverage
from training.long_regime_interest_gate_validation import build_interest_features
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class LongMonthlyDiagnosticConfig(LongRegimeGateConfig):
    score_variants: str = "activity_flow_htf,activity_flow_deriv_htf,activity_flow,activity"
    score_quantiles: str = "0.5,0.6,0.7"
    tier_specs: str = "0.6:0.25:0.8:0.5,0.5:0.2:0.8:0.5,0.5:0.25:0.7:0.5"
    entry_quantile: float = 0.8
    premium_quantile: float = 0.8
    funding_quantile: float = 0.7
    hold_bars: int = 144
    stride_bars: int = 12
    train_start: str = "2020-01-01"
    monthly_start: str = "2024-01-01"
    monthly_end: str = "2026-06-02"
    fixed_leverage: float = 0.5
    mode: str = "both"  # gate, sizing, both


def _parse_list(raw: str, cast: Any) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _month_folds(start: str, end: str, train_start: str) -> list[tuple[str, str, str, str]]:
    cur = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    folds: list[tuple[str, str, str, str]] = []
    while cur < stop:
        nxt = min(cur + pd.DateOffset(months=1), stop)
        folds.append((str(pd.Timestamp(train_start).date()), str(cur.date()), str(cur.date()), str(nxt.date())))
        cur = nxt
    return folds


def _score_summary(month_rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = [r for r in month_rows if int(r["sim"]["trade_entries"]) > 0]
    if not active:
        return {"months": len(month_rows), "active_months": 0, "positive_months": 0, "total_trades": 0, "sum_return_pct": 0.0}
    rets = [float(r["sim"]["total_return_pct"]) for r in active]
    ratios = [float(r["sim"]["cagr_to_strict_mdd"]) for r in active if np.isfinite(float(r["sim"]["cagr_to_strict_mdd"]))]
    return {
        "months": len(month_rows),
        "active_months": len(active),
        "positive_months": sum(1 for x in rets if x > 0.0),
        "negative_months": sum(1 for x in rets if x < 0.0),
        "hit_month_rate": float(sum(1 for x in rets if x > 0.0) / len(active)),
        "total_trades": int(sum(int(r["sim"]["trade_entries"]) for r in active)),
        "sum_return_pct": float(sum(rets)),
        "mean_month_return_pct": float(np.mean(rets)),
        "median_month_return_pct": float(np.median(rets)),
        "min_month_return_pct": float(min(rets)),
        "max_month_return_pct": float(max(rets)),
        "max_mdd_pct": float(max(float(r["sim"]["strict_mdd_pct"]) for r in active)),
        "median_cagr_mdd": float(np.median(ratios)) if ratios else 0.0,
    }


def _run_gate_months(market: pd.DataFrame, features: pd.DataFrame, score_raw: pd.DataFrame, cfg: LongMonthlyDiagnosticConfig, variant: str, q: float, folds: list[tuple[str, str, str, str]]) -> dict[str, Any] | None:
    dates = pd.to_datetime(market["date"])
    rows = []
    for train_start, train_end, val_start, val_end in folds:
        train_mask = _split_mask(dates, train_start, train_end)
        val_mask = _split_mask(dates, val_start, val_end)
        entry = _entry_active(features, rule="pb30_funding", train_mask=train_mask, quantile=float(cfg.entry_quantile))
        premium = _gate_active(features, feature="premium_index_zscore", op="ge", train_mask=train_mask, quantile=float(cfg.premium_quantile))
        funding = _gate_active(features, feature="funding_zscore", op="ge", train_mask=train_mask, quantile=float(cfg.funding_quantile))
        score_pack = _score_variant(score_raw, train_mask, variant)
        if entry is None or premium is None or funding is None or score_pack is None:
            return None
        score, _ = score_pack
        arr = score.to_numpy(float)
        ref = arr[train_mask & np.isfinite(arr)]
        if ref.size < 500:
            return None
        threshold = float(np.quantile(ref, q))
        entry_mask, _ = entry; premium_mask, _ = premium; funding_mask, _ = funding
        active = entry_mask & premium_mask & funding_mask & (arr >= threshold) & np.isfinite(arr)
        positions = np.arange(max(0, int(cfg.window_size) - 1), max(0, len(market) - int(cfg.hold_bars) - int(cfg.entry_delay_bars) - 1), int(cfg.stride_bars), dtype=np.int64)
        p = positions[active[positions] & val_mask[positions]]
        sim, returns = _strict_long_sim(p, market=market, hold_bars=int(cfg.hold_bars), entry_delay_bars=int(cfg.entry_delay_bars), leverage=float(cfg.fixed_leverage), fee_rate=float(cfg.fee_rate), slippage_rate=float(cfg.slippage_rate), annualization_start=val_start, annualization_end=val_end)
        rows.append({"month": val_start[:7], "fold": {"train_start": train_start, "train_end": train_end, "validation_start": val_start, "validation_end": val_end}, "threshold": threshold, "sim": sim, "trade_stats": _trade_stats(returns)})
    return {"mode": "gate", "score_variant": variant, "score_quantile": q, "summary": _score_summary(rows), "months": rows}


def _run_sizing_months(market: pd.DataFrame, features: pd.DataFrame, score_raw: pd.DataFrame, cfg: LongMonthlyDiagnosticConfig, variant: str, tier: dict[str, float], folds: list[tuple[str, str, str, str]]) -> dict[str, Any] | None:
    dates = pd.to_datetime(market["date"])
    rows = []
    for train_start, train_end, val_start, val_end in folds:
        train_mask = _split_mask(dates, train_start, train_end)
        val_mask = _split_mask(dates, val_start, val_end)
        entry = _entry_active(features, rule="pb30_funding", train_mask=train_mask, quantile=float(cfg.entry_quantile))
        premium = _gate_active(features, feature="premium_index_zscore", op="ge", train_mask=train_mask, quantile=float(cfg.premium_quantile))
        funding = _gate_active(features, feature="funding_zscore", op="ge", train_mask=train_mask, quantile=float(cfg.funding_quantile))
        score_pack = _score_variant(score_raw, train_mask, variant)
        if entry is None or premium is None or funding is None or score_pack is None:
            return None
        score, _ = score_pack
        tier_pack = _fit_tier_leverage(score, train_mask, tier)
        if tier_pack is None:
            return None
        leverage_by_pos, tier_spec = tier_pack
        entry_mask, _ = entry; premium_mask, _ = premium; funding_mask, _ = funding
        active = entry_mask & premium_mask & funding_mask & (leverage_by_pos > 0.0)
        positions = np.arange(max(0, int(cfg.window_size) - 1), max(0, len(market) - int(cfg.hold_bars) - int(cfg.entry_delay_bars) - 1), int(cfg.stride_bars), dtype=np.int64)
        p = positions[active[positions] & val_mask[positions]]
        sim, returns = _strict_long_sim_variable_leverage(p, leverage_by_pos, market=market, hold_bars=int(cfg.hold_bars), entry_delay_bars=int(cfg.entry_delay_bars), fee_rate=float(cfg.fee_rate), slippage_rate=float(cfg.slippage_rate), annualization_start=val_start, annualization_end=val_end)
        rows.append({"month": val_start[:7], "fold": {"train_start": train_start, "train_end": train_end, "validation_start": val_start, "validation_end": val_end}, "tier_spec": tier_spec, "sim": sim, "trade_stats": _trade_stats(returns)})
    return {"mode": "sizing", "score_variant": variant, "tier": tier, "summary": _score_summary(rows), "months": rows}


def run(cfg: LongMonthlyDiagnosticConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    interest = build_interest_features(market, features)
    score_raw = _build_score_frame(market, features, interest)
    folds = _month_folds(cfg.monthly_start, cfg.monthly_end, cfg.train_start)
    rows = []
    mode = str(cfg.mode).lower()
    for variant in _parse_list(cfg.score_variants, str):
        if mode in ("gate", "both"):
            for q in _parse_list(cfg.score_quantiles, float):
                r = _run_gate_months(market, features, score_raw, cfg, variant, q, folds)
                if r is not None:
                    rows.append(r)
        if mode in ("sizing", "both"):
            for tier in _parse_list(cfg.tier_specs, _parse_tier_spec):
                r = _run_sizing_months(market, features, score_raw, cfg, variant, tier, folds)
                if r is not None:
                    rows.append(r)
    rows.sort(key=lambda r: (r["summary"].get("positive_months", 0), r["summary"].get("min_month_return_pct", -999), r["summary"].get("sum_return_pct", -999), r["summary"].get("total_trades", 0)), reverse=True)
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])}, "folds": [{"train_start": a, "train_end": b, "validation_start": c, "validation_end": d} for a,b,c,d in folds], "top": rows[:50], "all_count": len(rows), "leakage_guard": {"train_only_thresholds": True, "full_calendar_month_cagr": True, "absolute_return_primary": True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--funding-csv", default="")
    p.add_argument("--premium-csv", default="")
    for field in ("score-variants", "score-quantiles", "tier-specs", "train-start", "monthly-start", "monthly-end", "mode", "exclude-from"):
        p.add_argument(f"--{field}", default=getattr(LongMonthlyDiagnosticConfig, field.replace("-", "_")))
    p.add_argument("--entry-quantile", type=float, default=LongMonthlyDiagnosticConfig.entry_quantile)
    p.add_argument("--premium-quantile", type=float, default=LongMonthlyDiagnosticConfig.premium_quantile)
    p.add_argument("--funding-quantile", type=float, default=LongMonthlyDiagnosticConfig.funding_quantile)
    p.add_argument("--hold-bars", type=int, default=LongMonthlyDiagnosticConfig.hold_bars)
    p.add_argument("--stride-bars", type=int, default=LongMonthlyDiagnosticConfig.stride_bars)
    p.add_argument("--window-size", type=int, default=LongMonthlyDiagnosticConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=LongMonthlyDiagnosticConfig.entry_delay_bars)
    p.add_argument("--fixed-leverage", type=float, default=LongMonthlyDiagnosticConfig.fixed_leverage)
    p.add_argument("--fee-rate", type=float, default=LongMonthlyDiagnosticConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=LongMonthlyDiagnosticConfig.slippage_rate)
    return p.parse_args()


def main() -> None:
    report = run(LongMonthlyDiagnosticConfig(**vars(parse_args())))
    compact = []
    for r in report["top"][:10]:
        compact.append({k: r[k] for k in r if k not in {"months"}} | {"months": [{"month": m["month"], "ret_pct": m["sim"]["total_return_pct"], "cagr_pct": m["sim"]["cagr_pct"], "mdd_pct": m["sim"]["strict_mdd_pct"], "trades": m["sim"]["trade_entries"]} for m in r["months"]]})
    print(json.dumps({"output": report["config"]["output"], "input": report["input"], "all_count": report["all_count"], "top": compact}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
