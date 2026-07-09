"""Scan distinct long-regime alpha families with full-period CAGR/MDD scoring.

This is intentionally not another pullback/activity optimizer.  It tests separate
long-side hypotheses over the same leak-safe market feature frame:

- HTF momentum continuation
- volatility expansion / range breakout
- compression breakout
- macro relief / weak-dollar beta
- derivatives squeeze / funding-premium context
- oversold weekly rebound

All thresholds are fitted on the train window only.  The reported CAGR is over
the full search window and must be read with absolute return, MDD, and trades.
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
from training.long_regime_combo_scan import LongComboScanConfig, _load_market, _split_mask, _strict_long_sim
from training.long_regime_interest_gate_validation import build_interest_features
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class LongAlphaFamilyScanConfig(LongComboScanConfig):
    train_start: str = "2020-01-01"
    train_end: str = "2024-01-01"
    search_start: str = "2024-01-01"
    search_end: str = "2026-06-02"
    hold_bars: str = "144,216,288,432,576,720"
    stride_bars: str = "12,24"
    leverage: float = 0.5
    target_ratio: float = 3.0
    min_trades: int = 15
    max_mdd_pct: float = 15.0


def _parse_list(raw: str, cast: Any) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _fit_mask(features: pd.DataFrame, train: np.ndarray, feature: str, op: str, q: float) -> tuple[np.ndarray, dict[str, Any]] | None:
    if feature not in features.columns:
        return None
    values = features[feature].to_numpy(float)
    ref = values[train & np.isfinite(values)]
    if ref.size < 500 or float(np.nanstd(ref)) <= 1e-12:
        return None
    thr = float(np.quantile(ref, q))
    mask = (values <= thr if op == "le" else values >= thr) & np.isfinite(values)
    return mask, {"feature": feature, "op": op, "quantile": float(q), "threshold": thr}


def _candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    # 1) Trend/momentum continuation: avoid pullback thesis; buy persistent HTF strength.
    for f in ["htf_1d_return_4", "htf_3d_return_1", "htf_3d_return_4", "weekly_return_1w", "weekly_return_4w"]:
        for q in [0.8, 0.85, 0.9, 0.95]:
            specs.append({"family": "htf_momentum", "conds": [(f, "ge", q)]})
            for pos_f in ["rex_8640_range_pos", "htf_1w_range_pos", "weekly_range_pos"]:
                for pq in [0.5, 0.7, 0.8]:
                    specs.append({"family": "htf_momentum_position", "conds": [(f, "ge", q), (pos_f, "ge", pq)]})
    # 2) Breakout / volatility expansion.
    for width in ["rex_144_range_width_pct", "rex_576_range_width_pct", "rex_2016_range_width_pct", "range_vol"]:
        for pos in ["rex_144_range_pos", "rex_576_range_pos", "rex_2016_range_pos", "bb_z", "close_zscore_48"]:
            for wq in [0.7, 0.8, 0.9]:
                for pq in [0.7, 0.8, 0.9]:
                    specs.append({"family": "range_breakout", "conds": [(width, "ge", wq), (pos, "ge", pq)]})
    # 3) Compression then direction: low longer range + positive short trend.
    for comp in ["rex_2016_range_width_pct", "rex_8640_range_width_pct", "weekly_range_1w"]:
        for trend in ["trend_24", "trend_96", "htf_4h_return_4", "htf_1d_return_1"]:
            for cq in [0.05, 0.1, 0.2]:
                for tq in [0.7, 0.8, 0.9]:
                    specs.append({"family": "compression_breakout", "conds": [(comp, "le", cq), (trend, "ge", tq)]})
    # 4) Macro relief / dollar liquidity beta.
    for macro in ["dxy_momentum", "dxy_zscore", "usdkrw_momentum", "usdkrw_zscore"]:
        for trend in ["htf_1d_return_1", "htf_3d_return_1", "weekly_return_1w"]:
            for mq in [0.1, 0.2, 0.3]:
                for tq in [0.6, 0.7, 0.8]:
                    specs.append({"family": "macro_relief_momentum", "conds": [(macro, "le", mq), (trend, "ge", tq)]})
    # 5) Derivatives squeeze / under-positioned long: low funding/premium plus positive price action.
    for deriv in ["funding_zscore", "funding_rate", "premium_index_zscore", "premium_index_change"]:
        for trend in ["htf_1d_return_4", "trend_96", "weekly_return_1w", "rex_576_range_pos"]:
            for dq in [0.05, 0.1, 0.2, 0.3]:
                for tq in [0.7, 0.8, 0.9]:
                    specs.append({"family": "deriv_squeeze_long", "conds": [(deriv, "le", dq), (trend, "ge", tq)]})
    # 6) Weekly oversold rebound: separate from pb30/activity, longer-horizon drawdown/return stretch.
    for stretch in ["weekly_return_1w", "htf_1w_return_1", "htf_3d_return_4"]:
        for confirm in ["trend_24", "taker_imbalance", "premium_index_change", "quote_vol_z_1d"]:
            for sq in [0.05, 0.1, 0.2]:
                for cq in [0.6, 0.7, 0.8]:
                    specs.append({"family": "weekly_rebound", "conds": [(stretch, "le", sq), (confirm, "ge", cq)]})
    # de-dup
    seen = set(); out = []
    for s in specs:
        key = (s["family"], tuple(s["conds"]))
        if key not in seen:
            seen.add(key); out.append(s)
    return out


def run(cfg: LongAlphaFamilyScanConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    base_features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    features = pd.concat([base_features, build_interest_features(market, base_features)], axis=1)
    dates = pd.to_datetime(market["date"])
    train = _split_mask(dates, cfg.train_start, cfg.train_end)
    search = _split_mask(dates, cfg.search_start, cfg.search_end)
    specs = _candidate_specs()
    rows: list[dict[str, Any]] = []
    for spec in specs:
        active = np.ones(len(market), dtype=bool)
        fitted = []
        ok = True
        for feature, op, q in spec["conds"]:
            fm = _fit_mask(features, train, feature, op, q)
            if fm is None:
                ok = False; break
            mask, meta = fm
            active &= mask
            fitted.append(meta)
        if not ok or int((active & train).sum()) < 200:
            continue
        for hold in _parse_list(cfg.hold_bars, int):
            for stride in _parse_list(cfg.stride_bars, int):
                positions = np.arange(max(0, int(cfg.window_size) - 1), max(0, len(market) - hold - int(cfg.entry_delay_bars) - 1), stride, dtype=np.int64)
                p = positions[active[positions] & search[positions]]
                if len(p) < int(cfg.min_trades):
                    continue
                sim, returns = _strict_long_sim(
                    p,
                    market=market,
                    hold_bars=hold,
                    entry_delay_bars=int(cfg.entry_delay_bars),
                    leverage=float(cfg.leverage),
                    fee_rate=float(cfg.fee_rate),
                    slippage_rate=float(cfg.slippage_rate),
                    annualization_start=cfg.search_start,
                    annualization_end=cfg.search_end,
                )
                stats = _trade_stats(returns)
                score = float(sim["cagr_to_strict_mdd"]) + 0.03 * float(sim["total_return_pct"]) + min(1.0, int(sim["trade_entries"]) / 100.0) - float(stats.get("p_value_mean_ret_approx", 1.0) or 1.0)
                rows.append({"family": spec["family"], "conditions": fitted, "hold_bars": hold, "hold_hours": hold * 5 / 60, "stride_bars": stride, "sim": sim, "trade_stats": stats, "selection_score": score})
    rows.sort(key=lambda r: (float(r["sim"]["cagr_to_strict_mdd"]), float(r["sim"]["total_return_pct"]), int(r["sim"]["trade_entries"])), reverse=True)
    qualified = [r for r in rows if float(r["sim"]["cagr_to_strict_mdd"]) >= float(cfg.target_ratio) and float(r["sim"]["strict_mdd_pct"]) <= float(cfg.max_mdd_pct)]
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "selection_protocol": "All condition thresholds fit on train only; search window CAGR is full-period; absolute return and strict MDD are reported.",
        "families": sorted({s["family"] for s in specs}),
        "top": rows[:200],
        "qualified": qualified[:200],
        "all_count": len(rows),
        "qualified_count": len(qualified),
        "leakage_guard": {"train_only_thresholds": True, "full_period_cagr": True, "future_prices_only_in_backtest": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--funding-csv", default="")
    p.add_argument("--premium-csv", default="")
    for field in ("train-start", "train-end", "search-start", "search-end", "hold-bars", "stride-bars", "exclude-from"):
        p.add_argument(f"--{field}", default=getattr(LongAlphaFamilyScanConfig, field.replace("-", "_")))
    p.add_argument("--leverage", type=float, default=LongAlphaFamilyScanConfig.leverage)
    p.add_argument("--target-ratio", type=float, default=LongAlphaFamilyScanConfig.target_ratio)
    p.add_argument("--min-trades", type=int, default=LongAlphaFamilyScanConfig.min_trades)
    p.add_argument("--max-mdd-pct", type=float, default=LongAlphaFamilyScanConfig.max_mdd_pct)
    p.add_argument("--window-size", type=int, default=LongAlphaFamilyScanConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=LongAlphaFamilyScanConfig.entry_delay_bars)
    p.add_argument("--fee-rate", type=float, default=LongAlphaFamilyScanConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=LongAlphaFamilyScanConfig.slippage_rate)
    return p.parse_args()


def main() -> None:
    report = run(LongAlphaFamilyScanConfig(**vars(parse_args())))
    top = []
    for r in report["top"][:20]:
        s = r["sim"]
        top.append({"family": r["family"], "conditions": r["conditions"], "hold_bars": r["hold_bars"], "stride_bars": r["stride_bars"], "ret_pct": s["total_return_pct"], "cagr_pct": s["cagr_pct"], "mdd_pct": s["strict_mdd_pct"], "ratio": s["cagr_to_strict_mdd"], "trades": s["trade_entries"], "p": r["trade_stats"].get("p_value_mean_ret_approx")})
    print(json.dumps({"output": report["config"]["output"], "all_count": report["all_count"], "qualified_count": report["qualified_count"], "top": top}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
