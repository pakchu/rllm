"""Wave-trading-inspired compact feature ridge policy.

No new dependencies: this approximates wave_trading's strongest non-wavelet and
flow ideas (efficiency, GK vol, VWAP deviation, CVD/flow, candle shape) and emits
strict-backtest prediction streams.  It is deliberately linear/regularized to
avoid the tree/overfit failure mode documented in wave_trading.
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
from training.alpha_feature_backtest import _forward_return, _signal_for_value, fit_rule, FeatureRuleConfig
from training.alpha_linear_combo_scan import _load_market, _parse_list, _standardize_train
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


def _clean(x: pd.Series | np.ndarray, clip: float | None = None) -> pd.Series:
    s = pd.Series(x).replace([np.inf, -np.inf], np.nan)
    if clip is not None:
        s = s.clip(-float(clip), float(clip))
    return s.fillna(0.0)


def _ret(close: pd.Series, n: int) -> pd.Series:
    return _clean(close / close.shift(max(1, int(n))).replace(0.0, np.nan) - 1.0, clip=5.0)


def _rolling_z(s: pd.Series, n: int) -> pd.Series:
    n = max(2, int(n))
    m = s.rolling(n, min_periods=max(5, n // 3)).mean()
    sd = s.rolling(n, min_periods=max(5, n // 3)).std(ddof=0)
    return _clean((s - m) / sd.replace(0.0, np.nan), clip=5.0)


def _price_efficiency(close: pd.Series, n: int) -> pd.Series:
    direct = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n, min_periods=max(5, n // 3)).sum()
    return _clean(direct / path.replace(0.0, np.nan), clip=1.0)


def _garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    log_hl = np.log(high / low.replace(0.0, np.nan))
    log_co = np.log(close / open_.replace(0.0, np.nan))
    var = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
    return _clean(np.sqrt(var.abs().rolling(n, min_periods=max(5, n // 3)).mean()), clip=1.0)


def build_wave_feature_frame(market: pd.DataFrame, *, window: int = 96) -> pd.DataFrame:
    df = market.copy()
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    quote_vol = df.get("quote_asset_volume", volume * close).astype(float) if "quote_asset_volume" in df else volume * close
    taker_buy_quote = df.get("taker_buy_quote", quote_vol * 0.5).astype(float) if "taker_buy_quote" in df else quote_vol * 0.5
    n_trades = df.get("number_of_trades", pd.Series(0.0, index=df.index)).astype(float) if "number_of_trades" in df else pd.Series(0.0, index=df.index)

    tp = (high + low + close) / 3.0
    vwap = (tp * volume).rolling(16, min_periods=4).sum() / volume.rolling(16, min_periods=4).sum().replace(0.0, np.nan)
    taker_ratio = taker_buy_quote / quote_vol.replace(0.0, np.nan)
    taker_imb = _clean(taker_ratio * 2.0 - 1.0, clip=1.0)
    cvd = _clean((2.0 * taker_buy_quote - quote_vol).cumsum())
    vol_mean = volume.rolling(window, min_periods=max(5, window // 3)).mean()
    vol_std = volume.rolling(window, min_periods=max(5, window // 3)).std(ddof=0)
    body = close - open_
    candle_range = (high - low).replace(0.0, np.nan)
    flow_mom = taker_imb.rolling(max(4, window // 4), min_periods=4).mean()

    out = pd.DataFrame(index=df.index)
    for n in (12, 24, 48, 96, 144, 288):
        out[f"mom_{n}"] = _ret(close, n)
    for n in (28, 55, 96, 150):
        out[f"eff_{n}"] = _price_efficiency(close, n)
        out[f"gk_{n}"] = _garman_klass(open_, high, low, close, n)
        out[f"cvd_mom_{n}"] = _ret(cvd.abs() + 1.0, n) * np.sign(cvd.diff(n).fillna(0.0))
        out[f"vol_price_corr_{n}"] = close.pct_change().rolling(n, min_periods=max(5, n // 3)).corr(volume.pct_change())
    out["vwap_dev"] = _clean(close / vwap.replace(0.0, np.nan) - 1.0, clip=1.0)
    out["vwap_dev_z"] = _rolling_z(out["vwap_dev"], window)
    out["vol_spike"] = _clean((volume - vol_mean) / vol_std.replace(0.0, np.nan), clip=5.0)
    out["vol_regime"] = _clean(volume / vol_mean.replace(0.0, np.nan), clip=20.0)
    out["flow_mom"] = _clean(flow_mom, clip=1.0)
    out["taker_imbalance"] = taker_imb
    out["trade_intensity"] = _clean(volume / n_trades.replace(0.0, np.nan), clip=1e6)
    out["trade_intensity_z"] = _rolling_z(out["trade_intensity"], window)
    out["candle_body"] = _clean(body / close.replace(0.0, np.nan), clip=1.0)
    out["candle_upper"] = _clean((high - np.maximum(open_, close)) / close.replace(0.0, np.nan), clip=1.0)
    out["candle_lower"] = _clean((np.minimum(open_, close) - low) / close.replace(0.0, np.nan), clip=1.0)
    out["body_to_range"] = _clean(body.abs() / candle_range, clip=10.0)
    out["vol_price_div"] = _clean(_ret(close, 24) * -_rolling_z(volume, 48), clip=5.0)

    for col in ("dxy_zscore", "dxy_momentum", "kimchi_premium_zscore", "kimchi_premium_change", "usdkrw_zscore", "usdkrw_momentum", "dxy_available", "kimchi_available", "usdkrw_available", "external_any_available"):
        if col in df.columns:
            out[col] = _clean(df[col].astype(float), clip=5.0)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _groups(cols: list[str]) -> dict[str, list[str]]:
    wave_core = [c for c in cols if c.startswith(("mom_", "eff_", "gk_", "vwap_", "candle_", "body_", "vol_price_div"))]
    flow = [c for c in cols if c.startswith(("cvd_", "vol_", "flow_", "taker_", "trade_"))]
    external = [c for c in cols if c.startswith(("dxy", "kimchi", "usdkrw", "external"))]
    return {
        "wave_core": wave_core,
        "wave_flow": flow,
        "wave_all": sorted(set(wave_core + flow)),
        "wave_external": sorted(set(wave_core + flow + external)),
    }


def _fit_ridge(X: np.ndarray, y: np.ndarray, train_mask: np.ndarray, l2: float) -> tuple[np.ndarray, dict[str, Any]]:
    valid = train_mask & np.isfinite(y)
    Xt, yt = X[valid], y[valid]
    if len(Xt) < max(100, Xt.shape[1] * 5):
        raise ValueError(f"not enough train rows: {len(Xt)} cols={Xt.shape[1]}")
    A = np.c_[np.ones(len(Xt)), Xt]
    reg = np.eye(A.shape[1]) * float(l2)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ yt)
    pred = np.c_[np.ones(len(X)), X] @ coef
    return pred.astype(float), {"n_train": int(valid.sum()), "coef_norm": float(np.linalg.norm(coef[1:])), "intercept": float(coef[0])}


def _write_predictions(path: str | Path, *, dates: pd.Series, values: np.ndarray, rule: dict[str, Any], horizon: int, group: str, start: str, end: str) -> dict[str, Any]:
    mask = np.asarray((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)), dtype=bool)
    rows = []
    trade_rows = 0
    side_counts = {"LONG": 0, "SHORT": 0}
    for pos in np.flatnonzero(mask):
        sig = _signal_for_value(float(values[pos]), rule)
        if sig > 0:
            pred = {"gate": "TRADE", "side": "LONG", "hold_bars": int(horizon), "family": f"wave_ridge:{group}", "confidence": "HIGH"}
        elif sig < 0:
            pred = {"gate": "TRADE", "side": "SHORT", "hold_bars": int(horizon), "family": f"wave_ridge:{group}", "confidence": "HIGH"}
        else:
            pred = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "wave_ridge", "confidence": "HIGH"}
        if pred["gate"] == "TRADE":
            trade_rows += 1
            side_counts[pred["side"]] += 1
        rows.append({"date": str(dates.iloc[pos]), "signal_pos": int(pos), "prediction": pred, "feature_value": float(values[pos])})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")
    return {"path": str(path), "rows": len(rows), "trade_rows": trade_rows, "side_counts": side_counts}


@dataclass(frozen=True)
class WaveRidgeScanConfig:
    input_csv: str
    output: str
    work_dir: str
    train_start: str = "2023-01-01"
    train_end: str = "2024-06-30 23:59:59"
    test_start: str = "2024-07-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01 00:00:00"
    groups: str = "wave_core,wave_flow,wave_all,wave_external"
    horizons: str = "72,144,288"
    quantiles: str = "0.10,0.20,0.30"
    leverages: str = "0.20,0.30"
    pause_after_losses: str = "0,4"
    ridge_l2: float = 25.0
    window_size: int = 144
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    min_test_trades: int = 100
    max_test_mdd: float = 15.0
    top_k: int = 60


def _selection_score(bt: dict[str, Any], *, min_trades: int, max_mdd: float) -> float:
    s = bt["sim"]
    t = bt["trade_stats"]
    trades = int(s["trade_entries"])
    cagr = float(s["cagr_pct"])
    mdd = float(s["strict_mdd_pct"])
    if trades < min_trades or cagr <= 0 or mdd > max_mdd:
        return -1000 + trades / 10000 + cagr / 1000 - max(0.0, mdd - max_mdd) / 100
    return float(s["cagr_to_strict_mdd"]) + min(1.0, trades / 300.0) - float(t.get("p_value_mean_ret_approx", 1.0))


def run_scan(cfg: WaveRidgeScanConfig) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    # Include existing market feature frame too; wave features are compact but this
    # lets htf/external availability columns be reused when present.
    base = build_market_feature_frame(market, window_size=int(cfg.window_size))
    wave = build_wave_feature_frame(market, window=int(cfg.window_size))
    features = pd.concat([base, wave], axis=1)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    dates = pd.to_datetime(market["date"])
    columns = [c for c in features.columns if np.nanstd(features[c].to_numpy(dtype=float)) > 1e-12]
    groups = _groups(columns)
    group_names = [g for g in _parse_list(cfg.groups, str) if g in groups and groups[g]]
    train_mask = np.asarray((dates >= pd.Timestamp(cfg.train_start)) & (dates <= pd.Timestamp(cfg.train_end)), dtype=bool)
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for group in group_names:
        cols = groups[group]
        Xraw = features[cols].to_numpy(dtype=float)
        X, _, _ = _standardize_train(Xraw, train_mask)
        for horizon in _parse_list(cfg.horizons, int):
            fwd = _forward_return(market["open"].astype(float), horizon=int(horizon), entry_delay_bars=1)
            try:
                values, fit_info = _fit_ridge(X, fwd, train_mask, float(cfg.ridge_l2))
            except Exception as exc:
                rows.append({"group": group, "horizon": horizon, "error": str(exc)})
                continue
            for q in _parse_list(cfg.quantiles, float):
                rule_cfg = FeatureRuleConfig(input_csv=cfg.input_csv, output="", feature="wave_ridge", horizon=int(horizon), fit_start=cfg.train_start, fit_end=cfg.train_end, eval_start=cfg.test_start, eval_end=cfg.test_end, quantile=float(q), window_size=int(cfg.window_size), entry_delay_bars=1, wave_trading_root=cfg.wave_trading_root, external_tolerance=cfg.external_tolerance)
                rule = fit_rule(dates=dates, feature_values=values, forward_returns=fwd, cfg=rule_cfg)
                tag = f"{group}_h{horizon}_q{str(q).replace('.', 'p')}"
                test_pred = _write_predictions(work / f"{tag}_test.jsonl", dates=dates, values=values, rule=rule, horizon=int(horizon), group=group, start=cfg.test_start, end=cfg.test_end)
                eval_pred = _write_predictions(work / f"{tag}_eval.jsonl", dates=dates, values=values, rule=rule, horizon=int(horizon), group=group, start=cfg.eval_start, end=cfg.eval_end)
                for lev in _parse_list(cfg.leverages, float):
                    for pal in _parse_list(cfg.pause_after_losses, int):
                        test_bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=test_pred["path"], market_csv=cfg.input_csv, output=str(work / f"{tag}_lev{lev}_pal{pal}_test.bt.json"), leverage=float(lev), pause_after_losses=int(pal), pause_bars=288))
                        score = _selection_score(test_bt, min_trades=int(cfg.min_test_trades), max_mdd=float(cfg.max_test_mdd))
                        eval_bt = None
                        if score > -999 or (float(test_bt["sim"]["cagr_pct"]) > 0 and float(test_bt["sim"]["strict_mdd_pct"]) <= 25):
                            eval_bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=eval_pred["path"], market_csv=cfg.input_csv, output=str(work / f"{tag}_lev{lev}_pal{pal}_eval.bt.json"), leverage=float(lev), pause_after_losses=int(pal), pause_bars=288))
                        rows.append({"group": group, "features": cols, "horizon": int(horizon), "quantile": float(q), "fit_info": fit_info, "rule": rule, "overlay": {"leverage": float(lev), "pause_after_losses": int(pal)}, "test_signal": test_pred, "eval_signal": eval_pred, "test": {"period": test_bt["period"], "sim": test_bt["sim"], "trade_stats": test_bt["trade_stats"]}, "eval": None if eval_bt is None else {"period": eval_bt["period"], "sim": eval_bt["sim"], "trade_stats": eval_bt["trade_stats"]}, "selection_score": score})
    ranked = sorted(rows, key=lambda r: (float(r.get("selection_score", -1e9)), float((r.get("eval") or {"sim": {"cagr_to_strict_mdd": -999}})["sim"].get("cagr_to_strict_mdd", -999))), reverse=True)
    report = {"config": asdict(cfg), "feature_columns": columns, "top_by_selection": ranked[: int(cfg.top_k)], "all_count": len(rows), "selection_protocol": "ridge fit and quantile rule on train; overlay-selected on test; eval reported only after test-like pass", "leakage_guard": {"train_fit_only": True, "test_selection_only": True, "eval_not_used_for_selection": True, "external_join": "backward_asof_no_future" if cfg.wave_trading_root else "disabled"}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan wave-trading-inspired ridge policy")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", required=True)
    for field, default in [("train-start", WaveRidgeScanConfig.train_start), ("train-end", WaveRidgeScanConfig.train_end), ("test-start", WaveRidgeScanConfig.test_start), ("test-end", WaveRidgeScanConfig.test_end), ("eval-start", WaveRidgeScanConfig.eval_start), ("eval-end", WaveRidgeScanConfig.eval_end), ("groups", WaveRidgeScanConfig.groups), ("horizons", WaveRidgeScanConfig.horizons), ("quantiles", WaveRidgeScanConfig.quantiles), ("leverages", WaveRidgeScanConfig.leverages), ("pause-after-losses", WaveRidgeScanConfig.pause_after_losses)]:
        p.add_argument(f"--{field}", default=default)
    p.add_argument("--ridge-l2", type=float, default=WaveRidgeScanConfig.ridge_l2)
    p.add_argument("--window-size", type=int, default=WaveRidgeScanConfig.window_size)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=WaveRidgeScanConfig.external_tolerance)
    p.add_argument("--min-test-trades", type=int, default=WaveRidgeScanConfig.min_test_trades)
    p.add_argument("--max-test-mdd", type=float, default=WaveRidgeScanConfig.max_test_mdd)
    p.add_argument("--top-k", type=int, default=WaveRidgeScanConfig.top_k)
    ns = p.parse_args()
    data = vars(ns)
    data = {k.replace("_", "_"): v for k, v in data.items()}
    return ns


def main() -> None:
    rep = run_scan(WaveRidgeScanConfig(**vars(parse_args())))
    for row in rep["top_by_selection"][:20]:
        ts = row.get("test", {}).get("sim", {})
        es = (row.get("eval") or {"sim": {}})["sim"]
        print(json.dumps({"group": row.get("group"), "h": row.get("horizon"), "q": row.get("quantile"), "overlay": row.get("overlay"), "score": row.get("selection_score"), "test": {"cagr": ts.get("cagr_pct"), "mdd": ts.get("strict_mdd_pct"), "ratio": ts.get("cagr_to_strict_mdd"), "trades": ts.get("trade_entries")}, "eval": {"cagr": es.get("cagr_pct"), "mdd": es.get("strict_mdd_pct"), "ratio": es.get("cagr_to_strict_mdd"), "trades": es.get("trade_entries")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
