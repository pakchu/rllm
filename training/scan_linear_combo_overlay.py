"""Search linear-combo feature rules with execution overlays in the loop.

Unlike alpha_linear_combo_scan, this ranks candidates after exporting signals to
the strict online overlay backtester.  The selection objective can therefore
prefer low-MDD candidates rather than high raw CAGR with unacceptable drawdown.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.alpha_feature_backtest import FeatureRuleConfig, _forward_return, _signal_for_value, fit_rule
from training.alpha_linear_combo_scan import _feature_groups, _fit_ridge_predict, _load_market, _parse_list, _standardize_train
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class OverlayScanConfig:
    input_csv: str
    output: str
    work_dir: str
    train_start: str = "2023-01-01"
    train_end: str = "2024-06-30 23:59:59"
    test_start: str = "2024-07-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01 00:00:00"
    groups: str = "external_plus_market,external,kimchi_plus_trend,trend,range_reversion,candle_flow,all"
    horizons: str = "36,72,144,288"
    quantiles: str = "0.05,0.10,0.15,0.20,0.25,0.30"
    leverages: str = "0.15,0.20,0.25,0.30,0.40"
    pause_after_losses: str = "0,3,4,5"
    monthly_loss_stops: str = "0,4,6,8"
    take_profits: str = "0,2,4"
    ridge_l2: float = 10.0
    window_size: int = 144
    entry_delay_bars: int = 1
    min_test_trades: int = 80
    max_test_mdd: float = 15.0
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    top_k: int = 50


NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "linear_combo", "confidence": "HIGH"}


def _write_predictions(path: str | Path, *, dates: pd.Series, values: np.ndarray, rule: dict[str, Any], horizon: int, group: str, start: str, end: str) -> dict[str, Any]:
    mask = np.asarray((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)), dtype=bool)
    rows: list[dict[str, Any]] = []
    side_counts = {"LONG": 0, "SHORT": 0}
    trade_rows = 0
    for pos in np.flatnonzero(mask):
        sig = _signal_for_value(float(values[pos]), rule)
        if sig > 0:
            pred = {"gate": "TRADE", "side": "LONG", "hold_bars": int(horizon), "family": f"linear_combo:{group}", "confidence": "HIGH"}
        elif sig < 0:
            pred = {"gate": "TRADE", "side": "SHORT", "hold_bars": int(horizon), "family": f"linear_combo:{group}", "confidence": "HIGH"}
        else:
            pred = dict(NO_TRADE)
        if pred["gate"] == "TRADE":
            trade_rows += 1
            side_counts[pred["side"]] += 1
        rows.append({"date": str(dates.iloc[pos]), "signal_pos": int(pos), "prediction": pred, "feature_value": float(values[pos])})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")
    return {"rows": len(rows), "trade_rows": trade_rows, "side_counts": side_counts, "path": str(path)}


def _selection_score(test: dict[str, Any], *, min_trades: int, max_mdd: float) -> float:
    sim = test["sim"]
    ts = test["trade_stats"]
    trades = int(sim["trade_entries"])
    cagr = float(sim["cagr_pct"])
    mdd = float(sim["strict_mdd_pct"])
    if trades < int(min_trades) or cagr <= 0 or mdd > float(max_mdd):
        return -1000.0 + trades / 10000.0 + cagr / 1000.0 - max(0.0, mdd - max_mdd) / 100.0
    return float(sim["cagr_to_strict_mdd"]) + min(1.0, trades / 300.0) - float(ts.get("p_value_mean_ret_approx", 1.0))


def run_scan(cfg: OverlayScanConfig) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    dates = pd.to_datetime(market["date"])
    columns = [c for c in features.columns if np.nanstd(features[c].to_numpy(dtype=float)) > 1e-12]
    all_groups = _feature_groups(columns)
    group_names = [g for g in _parse_list(cfg.groups, str) if g in all_groups]
    train_mask = np.asarray((dates >= pd.Timestamp(cfg.train_start)) & (dates <= pd.Timestamp(cfg.train_end)), dtype=bool)
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for group in group_names:
        cols = all_groups[group]
        Xraw = features[cols].to_numpy(dtype=float)
        X, _, _ = _standardize_train(Xraw, train_mask)
        for horizon in _parse_list(cfg.horizons, int):
            fwd = _forward_return(market["open"].astype(float), horizon=int(horizon), entry_delay_bars=int(cfg.entry_delay_bars))
            try:
                pred_values, fit_info = _fit_ridge_predict(X, fwd, train_mask, cfg.ridge_l2)
            except Exception as exc:
                rows.append({"group": group, "horizon": horizon, "error": str(exc)})
                continue
            for q in _parse_list(cfg.quantiles, float):
                rule_cfg = FeatureRuleConfig(input_csv=cfg.input_csv, output="", feature="linear_combo", horizon=int(horizon), fit_start=cfg.train_start, fit_end=cfg.train_end, eval_start=cfg.test_start, eval_end=cfg.test_end, quantile=float(q), window_size=int(cfg.window_size), entry_delay_bars=int(cfg.entry_delay_bars), wave_trading_root=cfg.wave_trading_root, external_tolerance=cfg.external_tolerance)
                try:
                    rule = fit_rule(dates=dates, feature_values=pred_values, forward_returns=fwd, cfg=rule_cfg)
                except Exception as exc:
                    rows.append({"group": group, "horizon": horizon, "quantile": q, "error": str(exc)})
                    continue
                tag = f"{group}_h{horizon}_q{str(q).replace('.', 'p')}"
                test_pred = _write_predictions(work / f"{tag}_test.jsonl", dates=dates, values=pred_values, rule=rule, horizon=horizon, group=group, start=cfg.test_start, end=cfg.test_end)
                eval_pred = _write_predictions(work / f"{tag}_eval.jsonl", dates=dates, values=pred_values, rule=rule, horizon=horizon, group=group, start=cfg.eval_start, end=cfg.eval_end)
                for lev in _parse_list(cfg.leverages, float):
                    for pal in _parse_list(cfg.pause_after_losses, int):
                        for ml in _parse_list(cfg.monthly_loss_stops, float):
                            for tp in _parse_list(cfg.take_profits, float):
                                otag = f"{tag}_lev{str(lev).replace('.', 'p')}_pal{pal}_ml{str(ml).replace('.', 'p')}_tp{str(tp).replace('.', 'p')}"
                                test_bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=test_pred["path"], market_csv=cfg.input_csv, output=str(work / f"{otag}_test.bt.json"), leverage=float(lev), pause_after_losses=int(pal), pause_bars=288, monthly_loss_stop_pct=float(ml), trade_take_profit_pct=float(tp)))
                                score = _selection_score(test_bt, min_trades=int(cfg.min_test_trades), max_mdd=float(cfg.max_test_mdd))
                                # Only spend eval time on potentially useful or at least informative configs.
                                eval_bt = None
                                if score > -999.0 or (float(test_bt["sim"]["cagr_pct"]) > 0 and float(test_bt["sim"]["strict_mdd_pct"]) <= 25.0):
                                    eval_bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=eval_pred["path"], market_csv=cfg.input_csv, output=str(work / f"{otag}_eval.bt.json"), leverage=float(lev), pause_after_losses=int(pal), pause_bars=288, monthly_loss_stop_pct=float(ml), trade_take_profit_pct=float(tp)))
                                rows.append({
                                    "group": group,
                                    "features": cols,
                                    "horizon": int(horizon),
                                    "quantile": float(q),
                                    "fit_info": fit_info,
                                    "rule": rule,
                                    "overlay": {"leverage": float(lev), "pause_after_losses": int(pal), "monthly_loss_stop_pct": float(ml), "trade_take_profit_pct": float(tp)},
                                    "test_signal": test_pred,
                                    "eval_signal": eval_pred,
                                    "test": {"period": test_bt["period"], "sim": test_bt["sim"], "trade_stats": test_bt["trade_stats"]},
                                    "eval": None if eval_bt is None else {"period": eval_bt["period"], "sim": eval_bt["sim"], "trade_stats": eval_bt["trade_stats"]},
                                    "selection_score": score,
                                })
    ranked = sorted(rows, key=lambda r: (float(r.get("selection_score", -1e9)), float((r.get("eval") or {"sim": {"cagr_to_strict_mdd": -999}})["sim"].get("cagr_to_strict_mdd", -999))), reverse=True)
    report = {
        "config": asdict(cfg),
        "feature_columns": columns,
        "selection_protocol": "fit linear combo and quantile rule on train; select by test overlay MDD/ratio; eval only for selected-like candidates",
        "top_by_selection": ranked[: int(cfg.top_k)],
        "all_count": len(rows),
        "leakage_guard": {"train_fit_only": True, "test_selection_only": True, "eval_not_used_for_selection": True, "external_join": "backward_asof_no_future" if cfg.wave_trading_root else "disabled"},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search linear combo feature rules with overlays")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--train-start", default="2023-01-01")
    p.add_argument("--train-end", default="2024-06-30 23:59:59")
    p.add_argument("--test-start", default="2024-07-01")
    p.add_argument("--test-end", default="2025-12-31 23:59:59")
    p.add_argument("--eval-start", default="2026-01-01")
    p.add_argument("--eval-end", default="2026-06-01 00:00:00")
    p.add_argument("--groups", default=OverlayScanConfig.groups)
    p.add_argument("--horizons", default=OverlayScanConfig.horizons)
    p.add_argument("--quantiles", default=OverlayScanConfig.quantiles)
    p.add_argument("--leverages", default=OverlayScanConfig.leverages)
    p.add_argument("--pause-after-losses", default=OverlayScanConfig.pause_after_losses)
    p.add_argument("--monthly-loss-stops", default=OverlayScanConfig.monthly_loss_stops)
    p.add_argument("--take-profits", default=OverlayScanConfig.take_profits)
    p.add_argument("--ridge-l2", type=float, default=10.0)
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--min-test-trades", type=int, default=80)
    p.add_argument("--max-test-mdd", type=float, default=15.0)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default="30min")
    p.add_argument("--top-k", type=int, default=50)
    return p.parse_args()


def main() -> None:
    report = run_scan(OverlayScanConfig(**vars(parse_args())))
    for row in report["top_by_selection"][:20]:
        if "test" not in row:
            print(json.dumps(row, ensure_ascii=False))
            continue
        ts = row["test"]["sim"]
        es = (row.get("eval") or {"sim": {}})["sim"]
        print(json.dumps({
            "group": row["group"], "h": row["horizon"], "q": row["quantile"], "overlay": row["overlay"], "score": row["selection_score"],
            "test": {"cagr": ts.get("cagr_pct"), "mdd": ts.get("strict_mdd_pct"), "ratio": ts.get("cagr_to_strict_mdd"), "trades": ts.get("trade_entries")},
            "eval": {"cagr": es.get("cagr_pct"), "mdd": es.get("strict_mdd_pct"), "ratio": es.get("cagr_to_strict_mdd"), "trades": es.get("trade_entries")},
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
