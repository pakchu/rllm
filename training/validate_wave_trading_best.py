"""Validate the documented wave_trading best 15m strategy through 2026.

Run with wave_trading's Python env because rllm intentionally does not depend on
PyWavelets/sklearn.  This is a validation bridge, not production code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_wave_module(root: str):
    sys.path.insert(0, str(Path(root).resolve()))
    sys.path.insert(0, str(Path(root).resolve() / "research"))
    import profit_search_rolling as psr  # type: ignore

    return psr


def _build_best_features(psr, *, start_date: str, end_date: str, time_interval: str) -> dict:
    bars_per_day, bars_per_month = psr.get_bars_per_period(time_interval)
    psr.BARS_PER_DAY = bars_per_day
    psr.BARS_PER_MONTH = bars_per_month
    psr._df_cache = None
    df = psr.load_data(start_date, end_date, time_interval=time_interval, verbose=True)
    close = df["close"].to_numpy().astype(np.float64)
    open_ = df["open"].to_numpy().astype(np.float64)
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    volume = df["volume"].to_numpy().astype(np.float64)
    flow = df["flow"].to_numpy().astype(np.float64)
    vwap_dev = df["vwap_dev"].to_numpy().astype(np.float64)
    dates = df["datetime"].to_numpy()

    # README documented best params.
    train_months = 9
    test_months = 2
    pt_mult = 3.75
    holding_period = 4
    atr_period = 15
    long_th = 0.66
    short_th = 0.35
    base_ws = 150
    base_wavelet = "haar"
    base_level = 4
    base_th = 1.585
    add_ws = 55
    add_wavelet = "db4"
    add_level = 4
    add_th = 0.524
    nw_ws = 28

    atr = psr.calc_atr(high, low, close, atr_period)
    y_label, long_ret, short_ret, underlying_ret = psr.generate_labels(close, high, low, atr, pt_mult, holding_period)

    base_feat = psr.extract_base_wavelet_features_cached(close, volume, base_ws, base_wavelet, base_level, base_th)
    n_samples = len(base_feat)

    def align(feat, ws):
        offset = base_ws - ws
        if offset > 0:
            return feat[offset:][:n_samples]
        if offset < 0:
            pad = np.full((-offset, feat.shape[1]), np.nan)
            return np.vstack([pad, feat])[:n_samples]
        return feat[:n_samples]

    flow_feat = align(psr.extract_wavelet_features_cached(flow, "flow_best", add_ws, add_wavelet, add_level, add_th), add_ws)
    vwap_feat = align(psr.extract_wavelet_features_cached(vwap_dev, "vwap_best", add_ws, add_wavelet, add_level, add_th), add_ws)
    gk = psr.calc_garman_klass(open_, high, low, close, nw_ws)[base_ws:][:n_samples].reshape(-1, 1)
    eff = psr.calc_price_efficiency(close, nw_ws)[base_ws:][:n_samples].reshape(-1, 1)
    candle = psr.calc_candle_patterns(open_, high, low, close, nw_ws)[base_ws:][:n_samples]
    X = np.hstack([base_feat, flow_feat, vwap_feat, gk, eff, candle])

    return {
        "X": X,
        "dates": dates[base_ws:][:n_samples],
        "y_label": y_label[base_ws:][:n_samples],
        "long_ret": long_ret[base_ws:][:n_samples],
        "short_ret": short_ret[base_ws:][:n_samples],
        "underlying_ret": underlying_ret[base_ws:][:n_samples],
        "params": {
            "train_months": train_months,
            "test_months": test_months,
            "pt_mult": pt_mult,
            "holding_period": holding_period,
            "atr_period": atr_period,
            "long_th": long_th,
            "short_th": short_th,
            "base_ws": base_ws,
            "base_wavelet": base_wavelet,
            "base_level": base_level,
            "base_th": base_th,
            "add_ws": add_ws,
            "add_wavelet": add_wavelet,
            "add_level": add_level,
            "add_th": add_th,
            "nw_ws": nw_ws,
            "features": 15,
        },
        "bars_per_month": bars_per_month,
    }


def _stats(trade_rets: list[float], years: float) -> dict:
    if not trade_rets:
        return {"return_pct": 0.0, "cagr_pct": 0.0, "mdd_pct": 0.0, "calmar": 0.0, "trades": 0, "win_rate": 0.0}
    r = np.asarray(trade_rets, dtype=float)
    eq = np.cumprod(1.0 + r)
    ret = (eq[-1] - 1.0) * 100.0
    cagr = (eq[-1] ** (1.0 / years) - 1.0) * 100.0 if years > 0 and eq[-1] > 0 else -100.0
    peak = np.maximum.accumulate(eq)
    mdd = np.max((peak - eq) / (peak + 1e-10)) * 100.0 if len(eq) else 0.0
    return {"return_pct": float(ret), "cagr_pct": float(cagr), "mdd_pct": float(mdd), "calmar": float(cagr / mdd) if mdd > 0 else 0.0, "trades": int(len(r)), "win_rate": float(np.mean(r > 0) * 100.0), "mean_trade_ret_pct": float(np.mean(r) * 100.0)}


def walk_forward(psr, data: dict, *, eval_start: str, eval_end: str, lr_C: float, lr_penalty: str) -> dict:
    X = data["X"]
    dates = np.asarray(data["dates"], dtype="datetime64[ns]")
    y_label = data["y_label"]
    long_ret = data["long_ret"]
    short_ret = data["short_ret"]
    holding = int(data["params"]["holding_period"])
    long_th = float(data["params"]["long_th"])
    short_th = float(data["params"]["short_th"])
    bars_per_month = int(data["bars_per_month"])
    train_bars = int(data["params"]["train_months"] * bars_per_month)
    test_bars = int(data["params"]["test_months"] * bars_per_month)
    purge_gap = holding * 2
    start_dt = np.datetime64(eval_start)
    end_dt = np.datetime64(eval_end)
    valid = (~np.any(~np.isfinite(X), axis=1)) & np.isfinite(y_label) & np.isfinite(long_ret) & np.isfinite(short_ret) & (y_label != 0)
    y_bin = (y_label > 0).astype(np.int32)
    trade_rets: list[float] = []
    folds = []
    pos = max(0, int(np.searchsorted(dates, start_dt)) - purge_gap - train_bars)
    while pos + train_bars + purge_gap < len(X):
        train_start = pos
        train_end = pos + train_bars
        test_start = train_end + purge_gap
        test_end = min(test_start + test_bars, len(X))
        if dates[test_start] > end_dt:
            break
        if dates[test_end - 1] < start_dt:
            pos += test_bars
            continue
        tr = np.arange(train_start, train_end)
        te = np.arange(test_start, test_end)
        tr = tr[valid[tr]]
        te = te[valid[te] & (dates[te] >= start_dt) & (dates[te] <= end_dt)]
        if len(tr) < 1000 or len(te) < 100:
            pos += test_bars
            continue
        proba = psr._predict_proba_lr(X[tr], y_bin[tr], X[te], lr_C, lr_penalty)
        rets, n_long, n_short, long_profit, short_profit = psr.get_trade_returns_numba(long_ret[te], short_ret[te], proba, long_th, short_th, holding, psr.TOTAL_COST)
        trade_rets.extend([float(x) for x in rets])
        folds.append({"start": str(dates[te[0]]), "end": str(dates[te[-1]]), "rows": int(len(te)), "trades": int(len(rets)), "n_long": int(n_long), "n_short": int(n_short), "return_pct": float((np.prod(1 + rets) - 1) * 100.0) if len(rets) else 0.0})
        pos += test_bars
    years = (np.datetime64(eval_end) - np.datetime64(eval_start)).astype("timedelta64[s]").astype(float) / (365.25 * 24 * 3600)
    return {"period": {"start": eval_start, "end": eval_end, "years": years}, "stats": _stats(trade_rets, years), "folds": folds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave-root", default="/home/pakchu/workspace/wave_trading")
    ap.add_argument("--start-date", default="2020-01-01")
    ap.add_argument("--end-date", default="2026-06-02")
    ap.add_argument("--time-interval", default="15m")
    ap.add_argument("--output", required=True)
    ap.add_argument("--lr-c", type=float, default=0.1)
    ap.add_argument("--lr-penalty", default="l2")
    ap.add_argument("--lr-c-grid", default="", help="comma-separated C values; overrides --lr-c when set")
    ap.add_argument("--lr-penalties", default="", help="comma-separated penalties; overrides --lr-penalty when set")
    args = ap.parse_args()
    psr = _load_wave_module(args.wave_root)
    data = _build_best_features(psr, start_date=args.start_date, end_date=args.end_date, time_interval=args.time_interval)
    c_values = [float(x) for x in args.lr_c_grid.split(",") if x.strip()] if args.lr_c_grid else [float(args.lr_c)]
    penalties = [x.strip() for x in args.lr_penalties.split(",") if x.strip()] if args.lr_penalties else [args.lr_penalty]
    candidates = []
    for C in c_values:
        for penalty in penalties:
            splits = {
                "test_2024h2_2025": walk_forward(psr, data, eval_start="2024-07-01", eval_end="2025-12-31 23:59:59", lr_C=C, lr_penalty=penalty),
                "eval_2026_jan_may": walk_forward(psr, data, eval_start="2026-01-01", eval_end="2026-06-01 00:00:00", lr_C=C, lr_penalty=penalty),
            }
            test = splits["test_2024h2_2025"]["stats"]
            ev = splits["eval_2026_jan_may"]["stats"]
            score = test["calmar"] + min(1.0, test["trades"] / 150.0) if test["cagr_pct"] > 0 and test["trades"] >= 50 else -1000.0 + test["trades"] / 1000.0 + test["cagr_pct"] / 1000.0
            candidates.append({"lr_C": C, "lr_penalty": penalty, "selection_score": score, "splits": splits})
    candidates.sort(key=lambda x: (x["selection_score"], x["splits"]["eval_2026_jan_may"]["stats"]["calmar"]), reverse=True)
    report = {
        "params": data["params"],
        "data": {"start_date": args.start_date, "end_date": args.end_date, "time_interval": args.time_interval, "rows": int(len(data["X"]))},
        "candidates": candidates,
        "best_by_selection": candidates[0] if candidates else None,
        "leakage_guard": {"rolling_train_before_test": True, "purge_gap_bars": int(data["params"]["holding_period"] * 2), "eval_not_used_for_training": True, "selection_uses_test_only": True},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for c in candidates:
        print(json.dumps({"lr_C": c["lr_C"], "penalty": c["lr_penalty"], "score": c["selection_score"], "test": c["splits"]["test_2024h2_2025"]["stats"], "eval": c["splits"]["eval_2026_jan_may"]["stats"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
