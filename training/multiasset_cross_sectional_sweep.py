"""Causal cross-sectional multi-asset futures sweeps with bar-level portfolio accounting.

The script intentionally avoids the earlier shared-direction trend assumption.  It builds
ranked portfolios from information available at the rebalance close, applies the new
positions from the next close-to-close interval, charges turnover costs, and selects
parameters only on the validation window before reporting the final holdout.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MOM_WINDOWS = [48, 288, 576, 2016]
VOL_WINDOWS = [288, 2016]
STRIDES = [96, 288]
TOP_KS = [1, 2]
SCORE_MODES = ["mom", "vol_adj_mom", "dual_mom_reversal"]
PORTFOLIOS = ["long_top", "long_short"]
REGIME_FILTERS = ["none", "market_mom_pos", "dispersion_high"]


@dataclass(frozen=True)
class Cfg:
    data_dir: str
    output: str
    val_start: str = "2023-01-01"
    val_end: str = "2024-12-31 23:59:59"
    eval_start: str = "2025-01-01"
    eval_end: str = "2026-06-01"
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    total_leverage: float = 1.0
    min_val_rebalances: int = 100
    val_mdd_ceiling: float = 0.0
    top_n: int = 30


def _normal_p_value(t: float) -> float:
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))


def load_close_matrix(data_dir: str) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for p in sorted(Path(data_dir).glob("*USDT_5m_*.csv.gz")):
        sym = p.name.split("_")[0]
        df = pd.read_csv(p, usecols=["date", "close"], parse_dates=["date"])
        ser = (
            df.drop_duplicates("date")
            .sort_values("date")
            .set_index("date")["close"]
            .astype(float)
            .rename(sym)
        )
        frames.append(ser)
    if not frames:
        raise FileNotFoundError(f"no *USDT_5m_*.csv.gz in {data_dir}")
    close = pd.concat(frames, axis=1).sort_index()
    close = close.ffill(limit=3).dropna(how="any")
    if close.empty:
        raise ValueError("aligned close matrix is empty")
    return close


def build_feature_cache(close: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    market = close.pct_change().mean(axis=1).add(1.0).cumprod()
    cache: dict[str, pd.DataFrame | pd.Series] = {"returns": returns, "market": market}
    for w in MOM_WINDOWS:
        cache[f"mom_{w}"] = (close / close.shift(w) - 1.0).replace([np.inf, -np.inf], np.nan)
        cache[f"mkt_mom_{w}"] = (market / market.shift(w) - 1.0).replace([np.inf, -np.inf], np.nan)
    for w in VOL_WINDOWS:
        cache[f"vol_{w}"] = returns.rolling(w, min_periods=max(12, w // 4)).std()
    cache["dispersion_288"] = cache["mom_288"].std(axis=1)  # type: ignore[union-attr]
    cache["dispersion_2016"] = cache["mom_2016"].std(axis=1)  # type: ignore[union-attr]
    return cache


def compute_score(cache: dict[str, pd.DataFrame | pd.Series], params: dict[str, Any]) -> pd.DataFrame:
    mom = cache[f"mom_{params['mom_window']}"]
    assert isinstance(mom, pd.DataFrame)
    if params["score_mode"] == "mom":
        return mom
    if params["score_mode"] == "vol_adj_mom":
        vol = cache[f"vol_{params['vol_window']}"]
        assert isinstance(vol, pd.DataFrame)
        return mom / (vol * math.sqrt(params["mom_window"]) + 1e-9)
    if params["score_mode"] == "short_reversal":
        short = cache["mom_48"]
        assert isinstance(short, pd.DataFrame)
        return -short
    if params["score_mode"] == "dual_mom_reversal":
        long_mom = cache[f"mom_{params['mom_window']}"]
        short = cache["mom_48"]
        assert isinstance(long_mom, pd.DataFrame) and isinstance(short, pd.DataFrame)
        return long_mom - short
    raise ValueError(f"unknown score_mode {params['score_mode']}")


def regime_allows(cache: dict[str, pd.DataFrame | pd.Series], dt: pd.Timestamp, params: dict[str, Any]) -> bool:
    filt = params["regime_filter"]
    if filt == "none":
        return True
    if filt in {"market_mom_pos", "market_mom_neg"}:
        m = cache[f"mkt_mom_{params['regime_window']}"]
        assert isinstance(m, pd.Series)
        v = float(m.loc[dt]) if dt in m.index and pd.notna(m.loc[dt]) else 0.0
        return v > 0 if filt == "market_mom_pos" else v < 0
    if filt == "dispersion_high":
        d = cache["dispersion_288"]
        assert isinstance(d, pd.Series)
        hist = d.loc[:dt].tail(params["regime_window"])
        if len(hist) < max(50, params["regime_window"] // 4):
            return False
        return float(hist.iloc[-1]) > float(hist.quantile(0.60))
    raise ValueError(f"unknown regime_filter {filt}")


def target_weights(scores: pd.Series, params: dict[str, Any], leverage: float) -> pd.Series:
    scores = scores.dropna().sort_values(ascending=False)
    w = pd.Series(0.0, index=scores.index)
    if len(scores) < 2:
        return w
    k = min(int(params["top_k"]), len(scores) // 2 if params["portfolio"] == "long_short" else len(scores))
    if k <= 0:
        return w
    if params["portfolio"] == "long_top":
        w.loc[scores.index[:k]] = leverage / k
    elif params["portfolio"] == "short_bottom":
        w.loc[scores.index[-k:]] = -leverage / k
    elif params["portfolio"] == "long_short":
        w.loc[scores.index[:k]] = leverage * 0.5 / k
        w.loc[scores.index[-k:]] = -leverage * 0.5 / k
    else:
        raise ValueError(f"unknown portfolio {params['portfolio']}")
    return w


def summarize_equity(equity: pd.Series, rebalance_returns: list[float], rebalance_count: int, turnover_sum: float, cfg: Cfg, start: str, end: str) -> dict[str, Any]:
    eq = equity.dropna()
    if eq.empty:
        eq = pd.Series([1.0])
    peak = eq.cummax()
    mdd = float((1.0 - eq / peak).max()) if len(eq) else 0.0
    days = max(1, (pd.Timestamp(end) - pd.Timestamp(start)).days)
    final = float(eq.iloc[-1])
    cagr = (final ** (365.25 / days) - 1.0) * 100.0 if final > 0 else -100.0
    daily = eq.resample("1D").last().pct_change().dropna() if isinstance(eq.index, pd.DatetimeIndex) else pd.Series(dtype=float)
    if len(daily) > 1 and float(daily.std(ddof=1)) > 1e-12:
        t_daily = float(daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily))))
        p_daily = _normal_p_value(t_daily)
    else:
        t_daily = 0.0
        p_daily = 1.0
    rr = np.asarray(rebalance_returns, dtype=float)
    if len(rr) > 1 and float(rr.std(ddof=1)) > 1e-12:
        t_reb = float(rr.mean() / (rr.std(ddof=1) / math.sqrt(len(rr))))
        p_reb = _normal_p_value(t_reb)
    else:
        t_reb = 0.0
        p_reb = 1.0
    mdd_pct = mdd * 100.0
    return {
        "sim": {
            "ret_pct": (final - 1.0) * 100.0,
            "cagr_pct": cagr,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": cagr / mdd_pct if mdd_pct > 1e-9 else 0.0,
            "rebalance_count": int(rebalance_count),
            "avg_turnover_per_rebalance": float(turnover_sum / rebalance_count) if rebalance_count else 0.0,
        },
        "stats": {
            "daily_obs": int(len(daily)),
            "daily_t_stat_like": t_daily,
            "daily_p_value_mean_ret_approx": p_daily,
            "rebalance_obs": int(len(rr)),
            "rebalance_mean_ret_pct": float(rr.mean() * 100.0) if len(rr) else 0.0,
            "rebalance_t_stat_like": t_reb,
            "rebalance_p_value_mean_ret_approx": p_reb,
        },
    }


def _regime_mask(cache: dict[str, pd.DataFrame | pd.Series], dates: pd.DatetimeIndex, params: dict[str, Any]) -> np.ndarray:
    filt = params["regime_filter"]
    if filt == "none":
        return np.ones(len(dates), dtype=bool)
    if filt in {"market_mom_pos", "market_mom_neg"}:
        m = cache[f"mkt_mom_{params['regime_window']}"]
        assert isinstance(m, pd.Series)
        vals = m.reindex(dates).fillna(0.0).to_numpy(dtype=float)
        return vals > 0 if filt == "market_mom_pos" else vals < 0
    if filt == "dispersion_high":
        d = cache["dispersion_288"]
        assert isinstance(d, pd.Series)
        vals = d.reindex(dates)
        q = vals.rolling(params["regime_window"], min_periods=max(50, params["regime_window"] // 4)).quantile(0.60)
        return (vals > q).fillna(False).to_numpy(dtype=bool)
    raise ValueError(f"unknown regime_filter {filt}")


def _weights_from_scores(score_vec: np.ndarray, params: dict[str, Any], leverage: float) -> np.ndarray:
    valid = np.isfinite(score_vec)
    w = np.zeros(score_vec.shape[0], dtype=float)
    if valid.sum() < 2:
        return w
    valid_idx = np.flatnonzero(valid)
    order = valid_idx[np.argsort(score_vec[valid_idx])[::-1]]
    k = min(int(params["top_k"]), len(order) // 2 if params["portfolio"] == "long_short" else len(order))
    if k <= 0:
        return w
    if params["portfolio"] == "long_top":
        w[order[:k]] = leverage / k
    elif params["portfolio"] == "short_bottom":
        w[order[-k:]] = -leverage / k
    elif params["portfolio"] == "long_short":
        w[order[:k]] = leverage * 0.5 / k
        w[order[-k:]] = -leverage * 0.5 / k
    else:
        raise ValueError(f"unknown portfolio {params['portfolio']}")
    return w



def summarize_stream(final_eq: float, mdd: float, rebalance_returns: list[float], rebalance_count: int, turnover_sum: float, cfg: Cfg, start: str, end: str) -> dict[str, Any]:
    days = max(1, (pd.Timestamp(end) - pd.Timestamp(start)).days)
    cagr = (final_eq ** (365.25 / days) - 1.0) * 100.0 if final_eq > 0 else -100.0
    rr = np.asarray(rebalance_returns, dtype=float)
    if len(rr) > 1 and float(rr.std(ddof=1)) > 1e-12:
        t_reb = float(rr.mean() / (rr.std(ddof=1) / math.sqrt(len(rr))))
        p_reb = _normal_p_value(t_reb)
    else:
        t_reb = 0.0
        p_reb = 1.0
    mdd_pct = float(mdd) * 100.0
    return {
        "sim": {
            "ret_pct": (final_eq - 1.0) * 100.0,
            "cagr_pct": cagr,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": cagr / mdd_pct if mdd_pct > 1e-9 else 0.0,
            "rebalance_count": int(rebalance_count),
            "avg_turnover_per_rebalance": float(turnover_sum / rebalance_count) if rebalance_count else 0.0,
        },
        "stats": {
            "rebalance_obs": int(len(rr)),
            "rebalance_mean_ret_pct": float(rr.mean() * 100.0) if len(rr) else 0.0,
            "rebalance_t_stat_like": t_reb,
            "rebalance_p_value_mean_ret_approx": p_reb,
        },
    }

def backtest(close: pd.DataFrame, cache: dict[str, pd.DataFrame | pd.Series], start: str, end: str, params: dict[str, Any], cfg: Cfg) -> dict[str, Any]:
    score = compute_score(cache, params).reindex(close.index).reindex(columns=close.columns)
    returns = cache["returns"]
    assert isinstance(returns, pd.DataFrame)
    returns = returns.reindex(close.index).reindex(columns=close.columns).fillna(0.0)

    mask = (close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))
    pos = np.flatnonzero(mask)
    if len(pos) < 2:
        return summarize_stream(1.0, 0.0, [], 0, 0.0, cfg, start, end)

    dates = close.index
    ret_values = returns.to_numpy(dtype=float)
    score_values = score.to_numpy(dtype=float)
    allow = _regime_mask(cache, dates, params)
    stride = int(params["stride"])
    cost_rate = cfg.fee_rate + cfg.slippage_rate
    current = np.zeros(close.shape[1], dtype=float)
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    rebalance_returns: list[float] = []
    rebalance_count = 0
    turnover_sum = 0.0

    for start_i in pos[:-1:stride]:
        end_i = min(start_i + stride, pos[-1])
        if allow[start_i]:
            target = _weights_from_scores(score_values[start_i], params, cfg.total_leverage)
        else:
            target = np.zeros_like(current)
        turnover = float(np.abs(target - current).sum())
        if turnover > 1e-12:
            eq *= max(0.0, 1.0 - turnover * cost_rate)
            turnover_sum += turnover
            peak = max(peak, eq)
            if peak > 0:
                mdd = max(mdd, 1.0 - eq / peak)
        current = target
        rebalance_count += 1
        before = eq
        if end_i > start_i:
            block = ret_values[start_i + 1 : end_i + 1] @ current
            gross = np.maximum(0.0, 1.0 + block.astype(float))
            path = eq * np.cumprod(gross)
            if len(path):
                running_peak = np.maximum.accumulate(np.maximum(path, peak))
                drawdowns = 1.0 - path / running_peak
                mdd = max(mdd, float(np.nanmax(drawdowns)))
                peak = max(peak, float(np.nanmax(path)))
                eq = float(path[-1])
        rebalance_returns.append(eq / before - 1.0 if before > 0 else -1.0)
        if eq <= 0:
            mdd = 1.0
            break
    return summarize_stream(eq, mdd, rebalance_returns, rebalance_count, turnover_sum, cfg, start, end)

def param_grid() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for score_mode in SCORE_MODES:
        for mom_window in MOM_WINDOWS:
            if score_mode in {"mom", "vol_adj_mom"} and mom_window == 48:
                continue
            for vol_window in VOL_WINDOWS:
                if score_mode != "vol_adj_mom" and vol_window != VOL_WINDOWS[0]:
                    continue
                for stride in STRIDES:
                    for top_k in TOP_KS:
                        for portfolio in PORTFOLIOS:
                            for regime_filter in REGIME_FILTERS:
                                for regime_window in [288, 2016]:
                                    if regime_filter == "none" and regime_window != 288:
                                        continue
                                    out.append({
                                        "score_mode": score_mode,
                                        "mom_window": mom_window,
                                        "vol_window": vol_window,
                                        "stride": stride,
                                        "top_k": top_k,
                                        "portfolio": portfolio,
                                        "regime_filter": regime_filter,
                                        "regime_window": regime_window,
                                    })
    return out


def run(cfg: Cfg) -> dict[str, Any]:
    close = load_close_matrix(cfg.data_dir)
    cache = build_feature_cache(close)
    rows: list[dict[str, Any]] = []
    for params in param_grid():
        res = backtest(close, cache, cfg.val_start, cfg.val_end, params, cfg)
        score = float(res["sim"]["cagr_to_strict_mdd"])
        if int(res["sim"]["rebalance_count"]) < cfg.min_val_rebalances:
            score -= 1000.0
        if cfg.val_mdd_ceiling > 0 and float(res["sim"]["strict_mdd_pct"]) > cfg.val_mdd_ceiling:
            score -= 1000.0 + (float(res["sim"]["strict_mdd_pct"]) - cfg.val_mdd_ceiling)
        if float(res["stats"]["rebalance_p_value_mean_ret_approx"]) > 0.10:
            score -= 0.25
        rows.append({"score": score, "params": params, **res})
    rows.sort(key=lambda r: r["score"], reverse=True)
    selected = rows[0]
    ev = backtest(close, cache, cfg.eval_start, cfg.eval_end, selected["params"], cfg)
    report = {
        "config": cfg.__dict__,
        "assets": list(close.columns),
        "date_range": {"start": str(close.index.min()), "end": str(close.index.max()), "bars": int(len(close))},
        "searched": len(rows),
        "top_val": rows[: cfg.top_n],
        "selected": selected,
        "eval": ev,
        "leakage_guard": "scores use rolling past close data; positions decided at rebalance close and applied to next bar; params selected on validation only; eval is final holdout",
        "accounting_note": "bar-level portfolio equity with turnover fee+slippage on weight changes; MDD from full bar equity path",
    }
    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> Cfg:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--val-start", default=Cfg.val_start)
    p.add_argument("--val-end", default=Cfg.val_end)
    p.add_argument("--eval-start", default=Cfg.eval_start)
    p.add_argument("--eval-end", default=Cfg.eval_end)
    p.add_argument("--fee-rate", type=float, default=Cfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=Cfg.slippage_rate)
    p.add_argument("--total-leverage", type=float, default=Cfg.total_leverage)
    p.add_argument("--min-val-rebalances", type=int, default=Cfg.min_val_rebalances)
    p.add_argument("--val-mdd-ceiling", type=float, default=Cfg.val_mdd_ceiling)
    p.add_argument("--top-n", type=int, default=Cfg.top_n)
    return Cfg(**vars(p.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
