"""Scan bearish-regime short alpha families with full-period CAGR/MDD scoring.

The scan mirrors the successful long-regime alpha-family workflow, but targets
persistent downtrends: downside momentum, range breakdowns, failed bounces,
overheated funding/premium with weak price action, macro pressure, and kimchi
stress.  Thresholds are fit on train only; all reported CAGR values use the full
calendar window including idle/cash periods.
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
from training.long_regime_combo_scan import LongComboScanConfig, _load_market, _split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class ShortAlphaFamilyScanConfig(LongComboScanConfig):
    train_start: str = "2020-01-01"
    train_end: str = "2024-01-01"
    search_start: str = "2024-01-01"
    search_end: str = "2026-06-02"
    hold_bars: str = "72,144,216,288,432,576"
    stride_bars: str = "12,24"
    leverage: float = 0.5
    target_ratio: float = 3.0
    min_trades: int = 20
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


def _strict_short_sim(
    signal_positions: np.ndarray,
    *,
    market: pd.DataFrame,
    hold_bars: int,
    entry_delay_bars: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
    annualization_start: str | None = None,
    annualization_end: str | None = None,
) -> tuple[dict[str, Any], list[float]]:
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    dates = pd.to_datetime(market["date"]).to_numpy()
    cost = (float(fee_rate) + float(slippage_rate)) * float(leverage)
    signal_positions = np.asarray(signal_positions, dtype=np.int64)
    eq = peak = 1.0
    max_dd = 0.0
    next_allowed = 0
    trade_returns: list[float] = []
    first_signal: int | None = None
    last_signal: int | None = None
    for pos in signal_positions:
        if pos < next_allowed:
            continue
        entry_pos = int(pos) + int(entry_delay_bars)
        exit_pos = entry_pos + int(hold_bars)
        if entry_pos >= len(market) - 1 or exit_pos >= len(market):
            continue
        first_signal = int(pos) if first_signal is None else first_signal
        last_signal = int(pos)
        entry_eq = eq
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - max(0.0, eq) / peak if peak > 0.0 else 0.0)
        for j in range(entry_pos, exit_pos):
            open_j = float(opens[j])
            if open_j <= 0.0:
                continue
            adverse_ret = (float(highs[j]) - open_j) / open_j
            max_dd = max(max_dd, 1.0 - max(0.0, eq * (1.0 - float(leverage) * adverse_ret)) / peak)
            close_ret = (float(opens[j + 1]) - open_j) / open_j
            eq *= max(0.0, 1.0 - float(leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - max(0.0, eq) / peak if peak > 0.0 else 0.0)
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        next_allowed = exit_pos
        if eq <= 0.0:
            break
    trade_start_dt = pd.Timestamp(dates[first_signal]).to_pydatetime() if first_signal is not None else None
    trade_end_dt = pd.Timestamp(dates[last_signal]).to_pydatetime() if last_signal is not None else None
    if annualization_start is not None and annualization_end is not None:
        start_dt = pd.Timestamp(annualization_start).to_pydatetime()
        end_dt = pd.Timestamp(annualization_end).to_pydatetime()
    elif trade_start_dt is not None and trade_end_dt is not None:
        start_dt = trade_start_dt
        end_dt = trade_end_dt
    else:
        start_dt = end_dt = datetime.now()
    years = max(1.0 / 365.25, (end_dt - start_dt).total_seconds() / (365.25 * 24 * 3600))
    ret_pct = (eq - 1.0) * 100.0
    cagr_pct = ((eq ** (1.0 / years) - 1.0) * 100.0) if eq > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    return {
        "period": {"start": str(start_dt), "end": str(end_dt), "years": years},
        "trade_period": {"start": str(trade_start_dt), "end": str(trade_end_dt)},
        "cagr_pct": cagr_pct,
        "strict_mdd_pct": mdd_pct,
        "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else (float("inf") if cagr_pct > 0 else 0.0),
        "trade_entries": len(trade_returns),
        "win_rate": sum(1 for r in trade_returns if r > 0.0) / len(trade_returns) if trade_returns else 0.0,
        "total_return_pct": ret_pct,
        "hold_bars": int(hold_bars),
        "entry_delay_bars": int(entry_delay_bars),
        "return_application": "short_only_actual_ohlc_strict_mdd",
    }, trade_returns


def _candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    # Downside momentum continuation.
    for f in ["htf_1d_return_4", "htf_3d_return_1", "htf_3d_return_4", "weekly_return_1w", "weekly_return_4w", "trend_96"]:
        for q in [0.05, 0.1, 0.15, 0.2]:
            specs.append({"family": "downside_momentum", "conds": [(f, "le", q)]})
            for pos_f in ["rex_8640_range_pos", "htf_1w_range_pos", "weekly_range_pos"]:
                for pq in [0.2, 0.3, 0.5]:
                    specs.append({"family": "downside_momentum_position", "conds": [(f, "le", q), (pos_f, "le", pq)]})
    # Range breakdown / volatility expansion.
    for width in ["rex_144_range_width_pct", "rex_576_range_width_pct", "rex_2016_range_width_pct", "range_vol"]:
        for pos in ["rex_144_range_pos", "rex_576_range_pos", "rex_2016_range_pos", "bb_z", "close_zscore_48"]:
            for wq in [0.7, 0.8, 0.9]:
                for pq in [0.1, 0.2, 0.3]:
                    specs.append({"family": "range_breakdown", "conds": [(width, "ge", wq), (pos, "le", pq)]})
    # Failed bounce: high short-term bounce/position in weak HTF trend.
    for weak in ["htf_1d_return_4", "htf_3d_return_4", "weekly_return_4w"]:
        for bounce in ["trend_24", "close_zscore_48", "rex_144_range_pos", "bb_z"]:
            for wq in [0.1, 0.2, 0.3]:
                for bq in [0.7, 0.8, 0.9]:
                    specs.append({"family": "failed_bounce_short", "conds": [(weak, "le", wq), (bounce, "ge", bq)]})
    # Funding / premium overheat while price action weak.
    for deriv in ["funding_zscore", "funding_rate", "premium_index_zscore", "premium_index_change"]:
        for weak in ["htf_1d_return_4", "trend_96", "weekly_return_1w", "rex_576_range_pos"]:
            for dq in [0.7, 0.8, 0.9, 0.95]:
                for wq in [0.1, 0.2, 0.3]:
                    specs.append({"family": "deriv_overheat_short", "conds": [(deriv, "ge", dq), (weak, "le", wq)]})
    # Macro pressure: dollar/KRW strength plus weak BTC.
    for macro in ["dxy_momentum", "dxy_zscore", "usdkrw_momentum", "usdkrw_zscore"]:
        for weak in ["htf_1d_return_1", "htf_3d_return_1", "weekly_return_1w", "trend_96"]:
            for mq in [0.7, 0.8, 0.9]:
                for wq in [0.1, 0.2, 0.3]:
                    specs.append({"family": "macro_pressure_short", "conds": [(macro, "ge", mq), (weak, "le", wq)]})
    # Kimchi stress/unwind proxies if present.
    for kimchi in ["kimchi_premium", "kimchi_premium_change", "kimchi_zscore"]:
        for weak in ["htf_1d_return_4", "htf_3d_return_1", "trend_96"]:
            for kq in [0.1, 0.2, 0.8, 0.9]:
                kop = "le" if kq <= 0.2 else "ge"
                for wq in [0.1, 0.2, 0.3]:
                    specs.append({"family": "kimchi_stress_short", "conds": [(kimchi, kop, kq), (weak, "le", wq)]})
    seen = set(); out = []
    for s in specs:
        key = (s["family"], tuple(s["conds"]))
        if key not in seen:
            seen.add(key); out.append(s)
    return out


def run(cfg: ShortAlphaFamilyScanConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    base_features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    features = pd.concat([base_features, build_interest_features(market, base_features)], axis=1)
    dates = pd.to_datetime(market["date"])
    train = _split_mask(dates, cfg.train_start, cfg.train_end)
    search = _split_mask(dates, cfg.search_start, cfg.search_end)
    rows: list[dict[str, Any]] = []
    for spec in _candidate_specs():
        active = np.ones(len(market), dtype=bool)
        fitted: list[dict[str, Any]] = []
        ok = True
        for feature, op, q in spec["conds"]:
            fm = _fit_mask(features, train, feature, op, q)
            if fm is None:
                ok = False; break
            mask, meta = fm
            active &= mask
            fitted.append(meta)
        if not ok or int((active & train).sum()) < 100:
            continue
        for hold in _parse_list(cfg.hold_bars, int):
            for stride in _parse_list(cfg.stride_bars, int):
                positions = np.arange(max(0, int(cfg.window_size) - 1), max(0, len(market) - hold - int(cfg.entry_delay_bars) - 1), stride, dtype=np.int64)
                p = positions[active[positions] & search[positions]]
                if len(p) < int(cfg.min_trades):
                    continue
                sim, returns = _strict_short_sim(
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
                rows.append({"family": spec["family"], "conditions": fitted, "hold_bars": hold, "hold_hours": hold * 5 / 60, "stride_bars": stride, "sim": sim, "trade_stats": stats})
    rows.sort(key=lambda r: (float(r["sim"]["cagr_to_strict_mdd"]), float(r["sim"]["total_return_pct"]), int(r["sim"]["trade_entries"])), reverse=True)
    qualified = [r for r in rows if float(r["sim"]["cagr_to_strict_mdd"]) >= float(cfg.target_ratio) and float(r["sim"]["strict_mdd_pct"]) <= float(cfg.max_mdd_pct)]
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "selection_protocol": "All thresholds fit on train only; search CAGR is full-period; absolute return, strict MDD, trades are reported.",
        "top": rows[:300],
        "qualified": qualified[:300],
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
        p.add_argument(f"--{field}", default=getattr(ShortAlphaFamilyScanConfig, field.replace("-", "_")))
    p.add_argument("--leverage", type=float, default=ShortAlphaFamilyScanConfig.leverage)
    p.add_argument("--target-ratio", type=float, default=ShortAlphaFamilyScanConfig.target_ratio)
    p.add_argument("--min-trades", type=int, default=ShortAlphaFamilyScanConfig.min_trades)
    p.add_argument("--max-mdd-pct", type=float, default=ShortAlphaFamilyScanConfig.max_mdd_pct)
    p.add_argument("--window-size", type=int, default=ShortAlphaFamilyScanConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=ShortAlphaFamilyScanConfig.entry_delay_bars)
    p.add_argument("--fee-rate", type=float, default=ShortAlphaFamilyScanConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=ShortAlphaFamilyScanConfig.slippage_rate)
    return p.parse_args()


def main() -> None:
    report = run(ShortAlphaFamilyScanConfig(**vars(parse_args())))
    top = []
    for r in report["top"][:25]:
        s = r["sim"]
        top.append({"family": r["family"], "conditions": r["conditions"], "hold_bars": r["hold_bars"], "stride_bars": r["stride_bars"], "ret_pct": s["total_return_pct"], "cagr_pct": s["cagr_pct"], "mdd_pct": s["strict_mdd_pct"], "ratio": s["cagr_to_strict_mdd"], "trades": s["trade_entries"], "p": r["trade_stats"].get("p_value_mean_ret_approx")})
    print(json.dumps({"output": report["config"]["output"], "all_count": report["all_count"], "qualified_count": report["qualified_count"], "top": top}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
