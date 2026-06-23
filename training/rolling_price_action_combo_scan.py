"""Rolling causal validation for price-action weak-alpha bundles.

Each target month is unseen during candidate selection:
- train window: older history ending before validation window;
- validation window: recent history immediately before target month;
- target month: traded with the selected frozen ridge+threshold rule.

The final report simulates all monthly selected signals through one global strict
bar-by-bar engine so drawdown and non-overlap are not reset at month boundaries.
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

from preprocessing.binance_aux_features import attach_binance_um_aux_features
from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.alpha_feature_backtest import FeatureRuleConfig, fit_rule, _signal_for_value
from training.alpha_linear_combo_scan import _fit_ridge_predict, _forward_return, _load_market, _parse_list, _standardize_train
from training.price_action_combo_scan import _feature_groups
from training.price_action_extreme_feature_scan import build_price_action_extreme_features
from training.strict_bar_backtest import BarExecutionConfig, _drawdown_from_trough, _trade_stats


@dataclass(frozen=True)
class RollingPriceActionComboCfg:
    input_csv: str
    output: str
    eval_start: str = "2024-01-01"
    eval_end: str = "2026-06-01"
    train_days: int = 1095
    validation_days: int = 180
    pa_windows: str = "36,72,144,288,576,2016"
    horizons: str = "72,144,288"
    quantiles: str = "0.05,0.10,0.20"
    ridge_l2s: str = "10,100,1000"
    window_size: int = 144
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    cooldown_bars: int = 0
    min_validation_trades: int = 20
    max_validation_mdd: float = 25.0
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    binance_funding_csv: str = ""
    binance_premium_csv: str = ""
    binance_funding_tolerance: str = "12h"
    binance_premium_tolerance: str = "2h"


def _month_starts(start: str, end: str) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start).normalize().replace(day=1)
    end_ts = pd.Timestamp(end)
    return [m for m in pd.date_range(start_ts, end_ts, freq="MS") if m < end_ts]


def _score_validation(report: dict[str, Any], *, min_trades: int, max_mdd: float) -> float:
    sim = report["sim"]
    stats = report["trade_stats"]
    trades = int(sim.get("trade_entries", 0))
    if trades < int(min_trades):
        return -1e9
    mdd = float(sim.get("strict_mdd_pct", 999.0))
    if mdd > float(max_mdd):
        return -1e9
    cagr = float(sim.get("cagr_pct", -100.0))
    ratio = float(sim.get("cagr_to_strict_mdd", -999.0))
    p = float(stats.get("p_value_mean_ret_approx", 1.0))
    mean = float(stats.get("mean_trade_ret_pct", 0.0))
    return ratio + 0.01 * cagr + 0.5 * mean + min(1.0, trades / 100.0) - p


def _compact_sim(report: dict[str, Any]) -> dict[str, Any]:
    sim = report["sim"]
    stats = report["trade_stats"]
    return {
        "ret_pct": sim["ret_pct"],
        "cagr_pct": sim["cagr_pct"],
        "strict_mdd_pct": sim["strict_mdd_pct"],
        "cagr_to_strict_mdd": sim["cagr_to_strict_mdd"],
        "trade_entries": sim["trade_entries"],
        "side_counts": sim["side_counts"],
        "mean_trade_ret_pct": stats.get("mean_trade_ret_pct"),
        "p_value_mean_ret_approx": stats.get("p_value_mean_ret_approx"),
    }




def _fast_validation_report(
    *,
    dates: pd.Series,
    feature_values: np.ndarray,
    forward_returns: np.ndarray,
    rule: dict[str, Any],
    horizon: int,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    cfg: RollingPriceActionComboCfg,
) -> dict[str, Any]:
    """Fast non-overlap validation scorer for candidate selection only.

    This intentionally uses close-to-close forward-return labels instead of the
    strict intra-trade bar path.  It is only used before target-month selection;
    the final selected policy is still evaluated by `_dynamic_strict_simulate`.
    """
    mask = np.asarray((dates >= eval_start) & (dates < eval_end), dtype=bool)
    idxs = np.flatnonzero(mask)
    next_allowed = 0
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    trade_returns: list[float] = []
    side_counts = {"LONG": 0, "SHORT": 0}
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * float(cfg.leverage) * 2.0
    hold_bars = max(1, int(horizon))
    for pos in idxs:
        pos = int(pos)
        if pos < next_allowed:
            continue
        sig = int(_signal_for_value(float(feature_values[pos]), rule))
        if sig == 0 or not np.isfinite(forward_returns[pos]):
            continue
        raw = float(forward_returns[pos]) * (1.0 if sig > 0 else -1.0)
        trade_ret = float(cfg.leverage) * raw - cost
        entry_eq = eq
        eq *= max(0.0, 1.0 + trade_ret)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        side_counts["LONG" if sig > 0 else "SHORT"] += 1
        next_allowed = pos + hold_bars + max(0, int(cfg.cooldown_bars))
    if len(idxs) == 0:
        years = 1.0 / 365.25
        start, end = str(eval_start), str(eval_end)
    else:
        start, end = str(dates.iloc[idxs[0]]), str(dates.iloc[idxs[-1]])
        years = max(1.0 / 365.25, float((pd.Timestamp(end) - pd.Timestamp(start)).days) / 365.25)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    return {
        "period": {"start": start, "end": end, "years": years},
        "sim": {
            "ret_pct": ret_pct,
            "cagr_pct": cagr_pct,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else (999.0 if cagr_pct > 0 else -999.0),
            "trade_entries": len(trade_returns),
            "side_counts": side_counts,
            "samples": int(len(idxs)),
            "return_application": "fast_forward_return_candidate_selection_only",
        },
        "trade_stats": _trade_stats(trade_returns),
    }

def _dynamic_strict_simulate(
    *,
    market: pd.DataFrame,
    dates: pd.Series,
    signals: np.ndarray,
    horizons: np.ndarray,
    cfg: RollingPriceActionComboCfg,
) -> dict[str, Any]:
    eval_mask = (dates >= pd.Timestamp(cfg.eval_start)) & (dates < pd.Timestamp(cfg.eval_end))
    eval_indices = np.flatnonzero(np.asarray(eval_mask, dtype=bool))
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    exec_cfg = BarExecutionConfig(
        leverage=float(cfg.leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        drawdown_stop=1.0,
        pause_bars=0,
        monthly_loss_stop=1.0,
        entry_delay_bars=int(cfg.entry_delay_bars),
    )
    cost = (float(exec_cfg.fee_rate) + float(exec_cfg.slippage_rate)) * float(exec_cfg.leverage)
    next_allowed = 0
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    trade_returns: list[float] = []
    executed: list[dict[str, Any]] = []
    side_counts = {"LONG": 0, "SHORT": 0}
    skipped = 0

    for pos in eval_indices:
        pos = int(pos)
        if pos < next_allowed:
            continue
        signal = int(signals[pos])
        if signal == 0:
            continue
        hold_bars = int(horizons[pos])
        if hold_bars <= 0:
            continue
        entry_pos = pos + int(cfg.entry_delay_bars)
        exit_pos = entry_pos + hold_bars
        if entry_pos >= len(market) - 1 or exit_pos >= len(market):
            skipped += 1
            continue
        side_counts["LONG" if signal > 0 else "SHORT"] += 1
        entry_eq = eq
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        for j in range(entry_pos, exit_pos):
            open_j = float(opens[j])
            if open_j <= 0.0:
                continue
            if signal > 0:
                adverse_ret = (float(lows[j]) - open_j) / open_j
                close_ret = (float(opens[j + 1]) - open_j) / open_j
            else:
                adverse_ret = (open_j - float(highs[j])) / open_j
                close_ret = (open_j - float(opens[j + 1])) / open_j
            adverse_eq = eq * (1.0 + float(cfg.leverage) * adverse_ret)
            max_dd = max(max_dd, _drawdown_from_trough(peak, adverse_eq))
            eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        executed.append({
            "signal_date": str(dates.iloc[pos]),
            "entry_date": str(dates.iloc[entry_pos]),
            "exit_date": str(dates.iloc[exit_pos]),
            "side": "LONG" if signal > 0 else "SHORT",
            "horizon": hold_bars,
            "executed_ret_pct": (eq / entry_eq - 1.0) * 100.0,
        })
        next_allowed = exit_pos + max(0, int(cfg.cooldown_bars))
        if eq <= 0.0:
            break

    if len(eval_indices) == 0:
        raise ValueError("no eval rows")
    eval_dates = dates.iloc[eval_indices]
    start_dt = pd.Timestamp(eval_dates.iloc[0]).to_pydatetime()
    end_dt = pd.Timestamp(eval_dates.iloc[-1]).to_pydatetime()
    years = max(1.0 / 365.25, float((end_dt - start_dt).days) / 365.25)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    return {
        "period": {"start": str(eval_dates.iloc[0]), "end": str(eval_dates.iloc[-1]), "years": years},
        "sim": {
            "ret_pct": ret_pct,
            "cagr_pct": cagr_pct,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf"),
            "trade_entries": len(trade_returns),
            "side_counts": side_counts,
            "samples": int(len(eval_indices)),
            "skipped_missing_bars": skipped,
            "entry_delay_bars": int(cfg.entry_delay_bars),
            "return_application": "rolling_monthly_selected_dynamic_horizon_strict_mdd",
        },
        "trade_stats": _trade_stats(trade_returns),
        "executed": executed,
    }


def run(cfg: RollingPriceActionComboCfg) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    if cfg.binance_funding_csv or cfg.binance_premium_csv:
        market = attach_binance_um_aux_features(
            market,
            funding_csv=cfg.binance_funding_csv or None,
            premium_csv=cfg.binance_premium_csv or None,
            funding_tolerance=cfg.binance_funding_tolerance,
            premium_tolerance=cfg.binance_premium_tolerance,
        )
    base = build_market_feature_frame(market, window_size=int(cfg.window_size))
    pa = build_price_action_extreme_features(market, _parse_list(cfg.pa_windows, int))
    features = pd.concat([base, pa], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    dates = pd.to_datetime(market["date"])
    columns = [c for c in features.columns if np.nanstd(features[c].to_numpy(dtype=float)) > 1e-12]
    groups = _feature_groups(columns)
    signals = np.zeros(len(market), dtype=np.int8)
    selected_horizons = np.zeros(len(market), dtype=np.int32)
    months: list[dict[str, Any]] = []

    for month_start in _month_starts(cfg.eval_start, cfg.eval_end):
        month_end = min(month_start + pd.offsets.MonthBegin(1), pd.Timestamp(cfg.eval_end))
        val_start = month_start - pd.Timedelta(days=int(cfg.validation_days))
        train_start = val_start - pd.Timedelta(days=int(cfg.train_days))
        train_mask = np.asarray((dates >= train_start) & (dates < val_start), dtype=bool)
        month_mask = np.asarray((dates >= month_start) & (dates < month_end), dtype=bool)
        if not month_mask.any():
            continue
        candidate_count = 0
        viable_candidate_count = 0
        error_count = 0
        best: dict[str, Any] | None = None
        best_pred: np.ndarray | None = None
        for horizon in _parse_list(cfg.horizons, int):
            fwd = _forward_return(market["open"].astype(float), horizon=horizon, entry_delay_bars=cfg.entry_delay_bars)
            for group_name, cols in groups.items():
                Xraw = features[cols].to_numpy(dtype=float)
                try:
                    X, _, _ = _standardize_train(Xraw, train_mask)
                except Exception:
                    error_count += len(_parse_list(cfg.ridge_l2s, float)) * len(_parse_list(cfg.quantiles, float))
                    continue
                for l2 in _parse_list(cfg.ridge_l2s, float):
                    try:
                        pred, fit_info = _fit_ridge_predict(X, fwd, train_mask, l2)
                    except Exception:
                        error_count += len(_parse_list(cfg.quantiles, float))
                        continue
                    for q in _parse_list(cfg.quantiles, float):
                        candidate_count += 1
                        rule_cfg = FeatureRuleConfig(
                            input_csv=cfg.input_csv,
                            output="",
                            feature=f"rolling_ridge:{group_name}",
                            horizon=int(horizon),
                            fit_start=str(train_start),
                            fit_end=str(val_start - pd.Timedelta(minutes=5)),
                            eval_start=str(val_start),
                            eval_end=str(month_start - pd.Timedelta(minutes=5)),
                            quantile=float(q),
                            window_size=cfg.window_size,
                            entry_delay_bars=cfg.entry_delay_bars,
                            leverage=cfg.leverage,
                            fee_rate=cfg.fee_rate,
                            slippage_rate=cfg.slippage_rate,
                        )
                        try:
                            rule = fit_rule(dates=dates, feature_values=pred, forward_returns=fwd, cfg=rule_cfg)
                            val_report = _fast_validation_report(
                                dates=dates,
                                feature_values=pred,
                                forward_returns=fwd,
                                rule=rule,
                                horizon=int(horizon),
                                eval_start=val_start,
                                eval_end=month_start,
                                cfg=cfg,
                            )
                            score = _score_validation(val_report, min_trades=cfg.min_validation_trades, max_mdd=cfg.max_validation_mdd)
                            if score > -1e8:
                                viable_candidate_count += 1
                            if best is None or float(score) > float(best.get("score", -1e18)):
                                best = {
                                    "group": group_name,
                                    "features": cols,
                                    "n_features": len(cols),
                                    "horizon": int(horizon),
                                    "quantile": float(q),
                                    "ridge_l2": float(l2),
                                    "fit_info": fit_info,
                                    "rule": rule,
                                    "validation": _compact_sim(val_report),
                                    "score": float(score),
                                }
                                best_pred = pred.copy()
                        except Exception:
                            error_count += 1
        selected = best if best is not None and float(best.get("score", -1e9)) > -1e8 else None
        month_indices = np.flatnonzero(month_mask)
        month_signal_counts = {"LONG": 0, "SHORT": 0, "NO_TRADE": 0}
        if selected is not None and best_pred is not None:
            rule = selected["rule"]
            for idx in month_indices:
                sig = int(_signal_for_value(float(best_pred[int(idx)]), rule))
                signals[int(idx)] = sig
                selected_horizons[int(idx)] = int(selected["horizon"])
                if sig > 0:
                    month_signal_counts["LONG"] += 1
                elif sig < 0:
                    month_signal_counts["SHORT"] += 1
                else:
                    month_signal_counts["NO_TRADE"] += 1
        else:
            month_signal_counts["NO_TRADE"] = int(len(month_indices))
        month_report = {
            "month": str(month_start.date())[:7],
            "train_start": str(train_start),
            "validation_start": str(val_start),
            "selection_cutoff_exclusive": str(month_start),
            "target_start": str(month_start),
            "target_end_exclusive": str(month_end),
            "candidate_count": candidate_count,
            "viable_candidate_count": viable_candidate_count,
            "error_count": error_count,
            "selected": selected,
            "raw_signal_counts": month_signal_counts,
        }
        months.append(month_report)
        print(
            json.dumps(
                {
                    "month": month_report["month"],
                    "candidates": candidate_count,
                    "viable": viable_candidate_count,
                    "selected": None if selected is None else {
                        "group": selected["group"],
                        "h": selected["horizon"],
                        "q": selected["quantile"],
                        "l2": selected["ridge_l2"],
                        "score": selected["score"],
                    },
                    "raw_signal_counts": month_signal_counts,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    bt = _dynamic_strict_simulate(market=market, dates=dates, signals=signals, horizons=selected_horizons, cfg=cfg)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "feature_columns": columns,
        "groups": {k: len(v) for k, v in groups.items()},
        "months": months,
        "result": {"period": bt["period"], "sim": bt["sim"], "trade_stats": bt["trade_stats"]},
        "executed_tail": bt["executed"][-20:],
        "selection_protocol": "each target month selects candidate on immediately prior fast validation only; train fit uses only rows before validation; final backtest uses one global strict simulator",
        "leakage_guard": {
            "target_month_not_used_for_selection": True,
            "standardization_fit_train_only_per_month": True,
            "ridge_fit_train_only_per_month": True,
            "rule_fit_train_only_per_month": True,
            "validation_used_for_candidate_selection_only": True,
            "validation_scoring_is_fast_forward_return_not_final_strict_bt": True,
            "global_strict_mdd_not_reset_monthly": True,
            "entry_after_signal_by_bars": int(cfg.entry_delay_bars),
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling causal validation for price-action weak-alpha bundles")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--eval-start", default=RollingPriceActionComboCfg.eval_start)
    p.add_argument("--eval-end", default=RollingPriceActionComboCfg.eval_end)
    p.add_argument("--train-days", type=int, default=RollingPriceActionComboCfg.train_days)
    p.add_argument("--validation-days", type=int, default=RollingPriceActionComboCfg.validation_days)
    p.add_argument("--pa-windows", default=RollingPriceActionComboCfg.pa_windows)
    p.add_argument("--horizons", default=RollingPriceActionComboCfg.horizons)
    p.add_argument("--quantiles", default=RollingPriceActionComboCfg.quantiles)
    p.add_argument("--ridge-l2s", default=RollingPriceActionComboCfg.ridge_l2s)
    p.add_argument("--window-size", type=int, default=RollingPriceActionComboCfg.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=RollingPriceActionComboCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=RollingPriceActionComboCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=RollingPriceActionComboCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=RollingPriceActionComboCfg.slippage_rate)
    p.add_argument("--cooldown-bars", type=int, default=RollingPriceActionComboCfg.cooldown_bars)
    p.add_argument("--min-validation-trades", type=int, default=RollingPriceActionComboCfg.min_validation_trades)
    p.add_argument("--max-validation-mdd", type=float, default=RollingPriceActionComboCfg.max_validation_mdd)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=RollingPriceActionComboCfg.external_tolerance)
    p.add_argument("--binance-funding-csv", default="")
    p.add_argument("--binance-premium-csv", default="")
    p.add_argument("--binance-funding-tolerance", default=RollingPriceActionComboCfg.binance_funding_tolerance)
    p.add_argument("--binance-premium-tolerance", default=RollingPriceActionComboCfg.binance_premium_tolerance)
    return p.parse_args()


def main() -> None:
    report = run(RollingPriceActionComboCfg(**vars(parse_args())))
    sim = report["result"]["sim"]
    stats = report["result"]["trade_stats"]
    print(json.dumps({
        "output": report["config"]["output"],
        "months": len(report["months"]),
        "selected_months": sum(1 for m in report["months"] if m.get("selected")),
        "groups": report["groups"],
        "result": {
            "cagr_pct": sim["cagr_pct"],
            "strict_mdd_pct": sim["strict_mdd_pct"],
            "cagr_to_strict_mdd": sim["cagr_to_strict_mdd"],
            "trade_entries": sim["trade_entries"],
            "side_counts": sim["side_counts"],
            "mean_trade_ret_pct": stats.get("mean_trade_ret_pct"),
            "p_value_mean_ret_approx": stats.get("p_value_mean_ret_approx"),
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
