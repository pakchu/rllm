"""Leak-safe linear feature-combination alpha scan.

Fits simple ridge-style linear predictors on a chronological train window, freezes
quantile thresholds/direction from that same train window, ranks candidates on a
later test window, and reports final eval-window results without using eval for
selection.

No sklearn dependency is required; the ridge solver uses numpy only.
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
from training.alpha_feature_backtest import FeatureRuleConfig, fit_rule, simulate_rule


@dataclass(frozen=True)
class LinearComboScanConfig:
    input_csv: str
    output: str
    train_start: str = "2023-01-01"
    train_end: str = "2024-06-30 23:59:59"
    test_start: str = "2024-07-01"
    test_end: str = "2025-08-31 23:59:59"
    eval_start: str = "2025-09-01"
    eval_end: str = "2026-02-28 23:59:59"
    horizons: str = "72,144,288"
    quantiles: str = "0.10,0.15,0.20"
    window_size: int = 144
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    binance_funding_csv: str = ""
    binance_premium_csv: str = ""
    binance_funding_tolerance: str = "12h"
    binance_premium_tolerance: str = "2h"
    ridge_l2: float = 10.0
    top_k: int = 30


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _forward_return(open_: pd.Series, *, horizon: int, entry_delay_bars: int) -> np.ndarray:
    entry = open_.shift(-int(entry_delay_bars))
    exit_ = open_.shift(-(int(entry_delay_bars) + int(horizon)))
    return ((exit_ - entry) / entry.replace(0.0, np.nan)).to_numpy(dtype=float)


def _parse_list(s: str, cast):
    return [cast(x.strip()) for x in str(s).split(",") if x.strip()]


def _feature_groups(columns: list[str]) -> dict[str, list[str]]:
    groups = {
        "external": ["dxy_zscore", "dxy_momentum", "kimchi_premium_zscore", "kimchi_premium_change", "usdkrw_zscore", "usdkrw_momentum", "dxy_available", "kimchi_available", "usdkrw_available", "external_any_available"],
        "kimchi_only": ["kimchi_premium_zscore", "kimchi_premium_change"],
        "trend": ["trend_12", "trend_24", "trend_96", "sma12_ratio", "sma24_ratio", "sma48_ratio", "bb_z", "close_zscore_48", "return_zscore_48"],
        "range_reversion": ["range_vol", "range_pos", "window_drawdown", "rsi_norm", "mfi_norm"],
        "candle_flow": ["body_ratio", "upper_shadow", "lower_shadow", "candle_range", "body_to_range", "shadow_imbalance", "volume_ratio", "volume_zscore", "trades_ratio", "taker_buy_ratio", "taker_imbalance"],
        "derivatives_aux": ["funding_rate", "funding_zscore", "funding_available", "premium_index", "premium_index_zscore", "premium_index_change", "premium_available", "binance_aux_any_available"],
    }
    groups["kimchi_plus_trend"] = groups["kimchi_only"] + groups["trend"]
    groups["kimchi_plus_range"] = groups["kimchi_only"] + groups["range_reversion"]
    groups["external_plus_market"] = groups["external"] + groups["trend"] + groups["range_reversion"] + groups["candle_flow"]
    groups["market_derivatives"] = groups["trend"] + groups["range_reversion"] + groups["candle_flow"] + groups.get("derivatives_aux", [])
    groups["external_market_derivatives"] = groups["external_plus_market"] + groups.get("derivatives_aux", [])
    groups["all"] = columns
    return {k: [c for c in v if c in columns] for k, v in groups.items() if any(c in columns for c in v)}


def _standardize_train(X: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmedian(X[train_mask], axis=0)
    q75 = np.nanpercentile(X[train_mask], 75, axis=0)
    q25 = np.nanpercentile(X[train_mask], 25, axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (np.abs(scale) > 1e-12), scale, 1.0)
    Z = (X - mu) / scale
    Z = np.where(np.isfinite(Z), np.clip(Z, -10.0, 10.0), 0.0)
    return Z, mu, scale


def _fit_ridge_predict(X: np.ndarray, y: np.ndarray, train_mask: np.ndarray, l2: float) -> tuple[np.ndarray, dict[str, Any]]:
    valid = train_mask & np.isfinite(y)
    Xt = X[valid]
    yt = y[valid]
    if Xt.shape[0] < max(100, Xt.shape[1] * 5):
        raise ValueError(f"not enough train rows: rows={Xt.shape[0]} cols={Xt.shape[1]}")
    # Add intercept; do not regularize intercept.
    A = np.c_[np.ones(len(Xt)), Xt]
    reg = np.eye(A.shape[1]) * float(l2)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ yt)
    pred = np.c_[np.ones(len(X)), X] @ coef
    return pred.astype(float), {"intercept": float(coef[0]), "coef_norm": float(np.linalg.norm(coef[1:])), "n_train": int(valid.sum())}


def _score_result(result: dict[str, Any]) -> float:
    sim = result["sim"]
    stats = result["trade_stats"]
    trades = float(sim["trade_entries"])
    if trades < 30:
        return -1e9
    ratio = float(sim["cagr_to_strict_mdd"])
    cagr = float(sim["cagr_pct"])
    p = float(stats.get("p_value_mean_ret_approx", 1.0))
    return ratio + 0.02 * cagr + min(1.0, trades / 150.0) - p


def run_scan(cfg: LinearComboScanConfig) -> dict[str, Any]:
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
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    dates = pd.to_datetime(market["date"])
    columns = [c for c in features.columns if np.nanstd(features[c].to_numpy(dtype=float)) > 1e-12]
    groups = _feature_groups(columns)
    train_mask = np.asarray((dates >= pd.Timestamp(cfg.train_start)) & (dates <= pd.Timestamp(cfg.train_end)), dtype=bool)
    test_cfg_base = FeatureRuleConfig(
        input_csv=cfg.input_csv, output="", feature="linear_combo", horizon=1,
        fit_start=cfg.train_start, fit_end=cfg.train_end, eval_start=cfg.test_start, eval_end=cfg.test_end,
        quantile=0.2, window_size=cfg.window_size, entry_delay_bars=cfg.entry_delay_bars,
        leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate,
        wave_trading_root=cfg.wave_trading_root, external_tolerance=cfg.external_tolerance,
        binance_funding_csv=cfg.binance_funding_csv, binance_premium_csv=cfg.binance_premium_csv,
        binance_funding_tolerance=cfg.binance_funding_tolerance, binance_premium_tolerance=cfg.binance_premium_tolerance,
    )
    rows: list[dict[str, Any]] = []
    for horizon in _parse_list(cfg.horizons, int):
        fwd = _forward_return(market["open"].astype(float), horizon=horizon, entry_delay_bars=cfg.entry_delay_bars)
        for group_name, cols in groups.items():
            Xraw = features[cols].to_numpy(dtype=float)
            X, _, _ = _standardize_train(Xraw, train_mask)
            try:
                pred, fit_info = _fit_ridge_predict(X, fwd, train_mask, cfg.ridge_l2)
            except Exception as exc:
                rows.append({"group": group_name, "horizon": horizon, "error": str(exc)})
                continue
            for q in _parse_list(cfg.quantiles, float):
                base = asdict(test_cfg_base)
                base.update({"horizon": horizon, "quantile": q, "eval_start": cfg.test_start, "eval_end": cfg.test_end})
                test_cfg = FeatureRuleConfig(**base)
                rule = fit_rule(dates=dates, feature_values=pred, forward_returns=fwd, cfg=test_cfg)
                test_report = simulate_rule(market=market, feature_values=pred, dates=dates, rule=rule, cfg=test_cfg)
                eval_base = dict(base)
                eval_base.update({"eval_start": cfg.eval_start, "eval_end": cfg.eval_end})
                eval_cfg = FeatureRuleConfig(**eval_base)
                eval_report = simulate_rule(market=market, feature_values=pred, dates=dates, rule=rule, cfg=eval_cfg)
                rows.append({
                    "group": group_name,
                    "features": cols,
                    "n_features": len(cols),
                    "horizon": horizon,
                    "quantile": q,
                    "fit_info": fit_info,
                    "rule": rule,
                    "test": test_report,
                    "eval": eval_report,
                    "test_score": _score_result(test_report),
                })
    ranked = sorted(rows, key=lambda r: float(r.get("test_score", -1e9)), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "feature_columns": columns,
        "selection_protocol": "fit model and thresholds on train only; rank by test only; eval reported as untouched holdout",
        "top_by_test": ranked[: int(cfg.top_k)],
        "all_count": len(rows),
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leak-safe linear combo alpha scan")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-start", default="2023-01-01")
    p.add_argument("--train-end", default="2024-06-30 23:59:59")
    p.add_argument("--test-start", default="2024-07-01")
    p.add_argument("--test-end", default="2025-08-31 23:59:59")
    p.add_argument("--eval-start", default="2025-09-01")
    p.add_argument("--eval-end", default="2026-02-28 23:59:59")
    p.add_argument("--horizons", default="72,144,288")
    p.add_argument("--quantiles", default="0.10,0.15,0.20")
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default="30min")
    p.add_argument("--binance-funding-csv", default="")
    p.add_argument("--binance-premium-csv", default="")
    p.add_argument("--binance-funding-tolerance", default=LinearComboScanConfig.binance_funding_tolerance)
    p.add_argument("--binance-premium-tolerance", default=LinearComboScanConfig.binance_premium_tolerance)
    p.add_argument("--ridge-l2", type=float, default=10.0)
    p.add_argument("--top-k", type=int, default=30)
    return p.parse_args()


def main() -> None:
    report = run_scan(LinearComboScanConfig(**vars(parse_args())))
    for row in report["top_by_test"][:10]:
        if "test" not in row:
            print(json.dumps(row, ensure_ascii=False))
            continue
        ts, tt = row["test"]["sim"], row["test"]["trade_stats"]
        es, et = row["eval"]["sim"], row["eval"]["trade_stats"]
        print(json.dumps({
            "group": row["group"], "h": row["horizon"], "q": row["quantile"], "n_features": row["n_features"],
            "test": {"cagr": ts["cagr_pct"], "mdd": ts["strict_mdd_pct"], "ratio": ts["cagr_to_strict_mdd"], "trades": ts["trade_entries"], "mean": tt["mean_trade_ret_pct"], "p": tt["p_value_mean_ret_approx"]},
            "eval": {"cagr": es["cagr_pct"], "mdd": es["strict_mdd_pct"], "ratio": es["cagr_to_strict_mdd"], "trades": es["trade_entries"], "mean": et["mean_trade_ret_pct"], "p": et["p_value_mean_ret_approx"]},
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
