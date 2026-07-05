"""Search long-regime alpha candidates by final full-period CAGR/strict-MDD ratio.

The selection criterion is explicitly full-period CAGR / strict MDD >= target.
This avoids earlier mistakes where sparse trade clusters made CAGR look better
than the real calendar-window result.
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
from training.long_regime_interest_gate_validation import build_interest_features
from training.long_regime_score_gate_validation import _build_score_frame, _score_variant
from training.long_regime_score_sizing_validation import _fit_tier_leverage, _parse_tier_spec, _strict_long_sim_variable_leverage
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class LongFinalRatioSearchConfig(LongRegimeGateConfig):
    search_start: str = "2024-01-01"
    search_end: str = "2026-06-02"
    train_start: str = "2020-01-01"
    train_end: str = "2024-01-01"
    target_ratio: float = 3.0
    score_variants: str = "activity_flow_deriv_htf,activity_flow_htf,activity_flow,activity"
    score_quantiles: str = "0.4,0.5,0.6,0.7,0.8"
    tier_specs: str = "0.4:0.15:0.7:0.5,0.5:0.15:0.75:0.5,0.5:0.2:0.8:0.5,0.6:0.25:0.8:0.5"
    entry_quantiles: str = "0.75,0.8,0.85"
    premium_quantiles: str = "0.6,0.7,0.8"
    funding_quantiles: str = "0.5,0.7,0.8"
    htf_veto_quantiles: str = ""  # optional bear_pressure <= train quantile, e.g. 0.7,0.8
    hold_bars: str = "72,144,216"
    stride_bars: str = "6,12,24"
    fixed_leverage: float = 0.5
    min_trades: int = 40
    max_mdd_pct: float = 15.0


def _parse_list(raw: str, cast: Any) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _fit_le_gate(values: pd.Series, train_mask: np.ndarray, q: float) -> tuple[np.ndarray, dict[str, Any]] | None:
    arr = values.to_numpy(float)
    ref = arr[train_mask & np.isfinite(arr)]
    if ref.size < 500:
        return None
    thr = float(np.quantile(ref, q))
    return (arr <= thr) & np.isfinite(arr), {"op": "le", "quantile": float(q), "threshold": thr}


def _positions(n: int, window: int, hold: int, delay: int, stride: int) -> np.ndarray:
    return np.arange(max(0, window - 1), max(0, n - hold - delay - 1), stride, dtype=np.int64)


def _candidate_score(sim: dict[str, Any], stats: dict[str, Any], cfg: LongFinalRatioSearchConfig) -> float:
    ratio = float(sim.get("cagr_to_strict_mdd", -999.0))
    ret = float(sim.get("total_return_pct", sim.get("ret_pct", 0.0)))
    trades = int(sim.get("trade_entries", 0))
    mdd = float(sim.get("strict_mdd_pct", 999.0))
    p = float(stats.get("p_value_mean_ret_approx", 1.0) or 1.0)
    penalty = 0.0
    if trades < int(cfg.min_trades):
        penalty += (int(cfg.min_trades) - trades) * 0.05
    if mdd > float(cfg.max_mdd_pct):
        penalty += (mdd - float(cfg.max_mdd_pct)) * 0.2
    return ratio + 0.03 * ret + min(1.0, trades / 120.0) - p - penalty


def _run_gate_candidate(*, market: pd.DataFrame, features: pd.DataFrame, score_raw: pd.DataFrame, cfg: LongFinalRatioSearchConfig, variant: str, score_q: float, entry_q: float, premium_q: float, funding_q: float, htf_q: float | None, hold: int, stride: int) -> dict[str, Any] | None:
    dates = pd.to_datetime(market["date"])
    train_mask = _split_mask(dates, cfg.train_start, cfg.train_end)
    eval_mask = _split_mask(dates, cfg.search_start, cfg.search_end)
    entry = _entry_active(features, rule="pb30_funding", train_mask=train_mask, quantile=entry_q)
    premium = _gate_active(features, feature="premium_index_zscore", op="ge", train_mask=train_mask, quantile=premium_q)
    funding = _gate_active(features, feature="funding_zscore", op="ge", train_mask=train_mask, quantile=funding_q)
    score_pack = _score_variant(score_raw, train_mask, variant)
    if entry is None or premium is None or funding is None or score_pack is None:
        return None
    score, _score_stats = score_pack
    score_arr = score.to_numpy(float)
    ref = score_arr[train_mask & np.isfinite(score_arr)]
    if ref.size < 500:
        return None
    score_thr = float(np.quantile(ref, score_q))
    active = entry[0] & premium[0] & funding[0] & (score_arr >= score_thr) & np.isfinite(score_arr)
    htf_gate = None
    if htf_q is not None:
        htf_gate = _fit_le_gate(score_raw["bear_pressure"], train_mask, htf_q)
        if htf_gate is None:
            return None
        active &= htf_gate[0]
    pos = _positions(len(market), int(cfg.window_size), hold, int(cfg.entry_delay_bars), stride)
    p = pos[active[pos] & eval_mask[pos]]
    sim, rets = _strict_long_sim(p, market=market, hold_bars=hold, entry_delay_bars=int(cfg.entry_delay_bars), leverage=float(cfg.fixed_leverage), fee_rate=float(cfg.fee_rate), slippage_rate=float(cfg.slippage_rate), annualization_start=cfg.search_start, annualization_end=cfg.search_end)
    stats = _trade_stats(rets)
    return {"mode": "gate", "score_variant": variant, "score_quantile": score_q, "score_threshold": score_thr, "entry_quantile": entry_q, "premium_quantile": premium_q, "funding_quantile": funding_q, "htf_veto": htf_gate[1] if htf_gate else None, "hold_bars": hold, "hold_hours": hold * 5 / 60, "stride_bars": stride, "sim": sim, "trade_stats": stats, "selection_score": _candidate_score(sim, stats, cfg)}


def _run_sizing_candidate(*, market: pd.DataFrame, features: pd.DataFrame, score_raw: pd.DataFrame, cfg: LongFinalRatioSearchConfig, variant: str, tier: dict[str, float], entry_q: float, premium_q: float, funding_q: float, htf_q: float | None, hold: int, stride: int) -> dict[str, Any] | None:
    dates = pd.to_datetime(market["date"])
    train_mask = _split_mask(dates, cfg.train_start, cfg.train_end)
    eval_mask = _split_mask(dates, cfg.search_start, cfg.search_end)
    entry = _entry_active(features, rule="pb30_funding", train_mask=train_mask, quantile=entry_q)
    premium = _gate_active(features, feature="premium_index_zscore", op="ge", train_mask=train_mask, quantile=premium_q)
    funding = _gate_active(features, feature="funding_zscore", op="ge", train_mask=train_mask, quantile=funding_q)
    score_pack = _score_variant(score_raw, train_mask, variant)
    if entry is None or premium is None or funding is None or score_pack is None:
        return None
    score, _score_stats = score_pack
    tier_pack = _fit_tier_leverage(score, train_mask, tier)
    if tier_pack is None:
        return None
    leverage_by_pos, tier_spec = tier_pack
    active = entry[0] & premium[0] & funding[0] & (leverage_by_pos > 0.0)
    htf_gate = None
    if htf_q is not None:
        htf_gate = _fit_le_gate(score_raw["bear_pressure"], train_mask, htf_q)
        if htf_gate is None:
            return None
        active &= htf_gate[0]
    pos = _positions(len(market), int(cfg.window_size), hold, int(cfg.entry_delay_bars), stride)
    p = pos[active[pos] & eval_mask[pos]]
    sim, rets = _strict_long_sim_variable_leverage(p, leverage_by_pos, market=market, hold_bars=hold, entry_delay_bars=int(cfg.entry_delay_bars), fee_rate=float(cfg.fee_rate), slippage_rate=float(cfg.slippage_rate), annualization_start=cfg.search_start, annualization_end=cfg.search_end)
    stats = _trade_stats(rets)
    return {"mode": "sizing", "score_variant": variant, "tier": tier_spec, "entry_quantile": entry_q, "premium_quantile": premium_q, "funding_quantile": funding_q, "htf_veto": htf_gate[1] if htf_gate else None, "hold_bars": hold, "hold_hours": hold * 5 / 60, "stride_bars": stride, "sim": sim, "trade_stats": stats, "selection_score": _candidate_score(sim, stats, cfg)}


def run(cfg: LongFinalRatioSearchConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    interest = build_interest_features(market, features)
    score_raw = _build_score_frame(market, features, interest)
    rows = []
    htf_qs: list[float | None] = [None] + _parse_list(cfg.htf_veto_quantiles, float)
    for variant in _parse_list(cfg.score_variants, str):
        for entry_q in _parse_list(cfg.entry_quantiles, float):
            for premium_q in _parse_list(cfg.premium_quantiles, float):
                for funding_q in _parse_list(cfg.funding_quantiles, float):
                    for htf_q in htf_qs:
                        for hold in _parse_list(cfg.hold_bars, int):
                            for stride in _parse_list(cfg.stride_bars, int):
                                for score_q in _parse_list(cfg.score_quantiles, float):
                                    r = _run_gate_candidate(market=market, features=features, score_raw=score_raw, cfg=cfg, variant=variant, score_q=score_q, entry_q=entry_q, premium_q=premium_q, funding_q=funding_q, htf_q=htf_q, hold=hold, stride=stride)
                                    if r is not None:
                                        rows.append(r)
                                for tier in _parse_list(cfg.tier_specs, _parse_tier_spec):
                                    r = _run_sizing_candidate(market=market, features=features, score_raw=score_raw, cfg=cfg, variant=variant, tier=tier, entry_q=entry_q, premium_q=premium_q, funding_q=funding_q, htf_q=htf_q, hold=hold, stride=stride)
                                    if r is not None:
                                        rows.append(r)
    rows.sort(key=lambda r: float(r["selection_score"]), reverse=True)
    qualified = [r for r in rows if float(r["sim"].get("cagr_to_strict_mdd", -999)) >= float(cfg.target_ratio) and int(r["sim"].get("trade_entries", 0)) >= int(cfg.min_trades) and float(r["sim"].get("strict_mdd_pct", 999)) <= float(cfg.max_mdd_pct)]
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])}, "selection_protocol": "Thresholds fit on train window only; final score is full-period search window CAGR/strict-MDD with absolute return shown.", "top": rows[:100], "qualified": qualified[:100], "all_count": len(rows), "qualified_count": len(qualified), "leakage_guard": {"train_only_thresholds": True, "full_period_cagr": True, "absolute_return_reported": True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--funding-csv", default="")
    p.add_argument("--premium-csv", default="")
    for field in ("search-start", "search-end", "train-start", "train-end", "score-variants", "score-quantiles", "tier-specs", "entry-quantiles", "premium-quantiles", "funding-quantiles", "htf-veto-quantiles", "hold-bars", "stride-bars", "exclude-from"):
        p.add_argument(f"--{field}", default=getattr(LongFinalRatioSearchConfig, field.replace("-", "_")))
    p.add_argument("--target-ratio", type=float, default=LongFinalRatioSearchConfig.target_ratio)
    p.add_argument("--fixed-leverage", type=float, default=LongFinalRatioSearchConfig.fixed_leverage)
    p.add_argument("--min-trades", type=int, default=LongFinalRatioSearchConfig.min_trades)
    p.add_argument("--max-mdd-pct", type=float, default=LongFinalRatioSearchConfig.max_mdd_pct)
    p.add_argument("--window-size", type=int, default=LongFinalRatioSearchConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=LongFinalRatioSearchConfig.entry_delay_bars)
    p.add_argument("--fee-rate", type=float, default=LongFinalRatioSearchConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=LongFinalRatioSearchConfig.slippage_rate)
    return p.parse_args()


def main() -> None:
    report = run(LongFinalRatioSearchConfig(**vars(parse_args())))
    compact=[]
    for r in report["top"][:20]:
        sim=r["sim"]
        compact.append({"mode": r["mode"], "score_variant": r["score_variant"], "q_or_tier": r.get("score_quantile") or r.get("tier"), "entry_q": r["entry_quantile"], "premium_q": r["premium_quantile"], "funding_q": r["funding_quantile"], "htf_veto": r["htf_veto"], "hold_bars": r["hold_bars"], "stride_bars": r["stride_bars"], "ret_pct": sim.get("total_return_pct"), "cagr_pct": sim.get("cagr_pct"), "mdd_pct": sim.get("strict_mdd_pct"), "ratio": sim.get("cagr_to_strict_mdd"), "trades": sim.get("trade_entries"), "score": r["selection_score"]})
    print(json.dumps({"output": report["config"]["output"], "all_count": report["all_count"], "qualified_count": report["qualified_count"], "top": compact}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
