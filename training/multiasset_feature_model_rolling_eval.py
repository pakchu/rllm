"""No-leak learned cross-sectional feature model for Binance USD-M futures.

Train/validation/eval protocol:
- Fit feature standardization and linear/ridge-like weights on train only.
- Select hyperparameters on validation only.
- Report final holdout on eval.

The model predicts next-horizon cross-sectional excess return from causal features
available at rebalance close. Portfolio accounting reuses bar-level next-interval
returns with turnover costs and streaming strict MDD.
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

# Keep grid deliberately small: previous broad rank sweeps showed severe validation overfit.
HORIZONS = [96, 288]
STRIDES = [96, 288]
TOP_KS = [1, 2]
PORTFOLIOS = ["long_short", "long_top"]
REGIME_FILTERS = ["none", "market_mom_pos", "dispersion_high"]
L2S = [0.1, 1.0, 10.0, 100.0]
TARGETS = ["excess", "raw"]


@dataclass(frozen=True)
class Cfg:
    data_dir: str
    aux_dir: str
    output: str
    train_start: str = "2023-01-01"
    train_end: str = "2023-12-31 23:59:59"
    val_start: str = "2024-01-01"
    val_end: str = "2024-12-31 23:59:59"
    eval_start: str = "2025-01-01"
    eval_end: str = "2026-06-01"
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    total_leverage: float = 1.0
    min_val_rebalances: int = 100
    val_mdd_ceiling: float = 0.0
    top_n: int = 30


def normal_p_value(t: float) -> float:
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))


def load_close_matrix(data_dir: str) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for p in sorted(Path(data_dir).glob("*USDT_5m_*.csv.gz")):
        sym = p.name.split("_")[0]
        df = pd.read_csv(p, usecols=["date", "close"], parse_dates=["date"])
        ser = df.drop_duplicates("date").sort_values("date").set_index("date")["close"].astype(float).rename(sym)
        frames.append(ser)
    if not frames:
        raise FileNotFoundError(f"no *USDT_5m_*.csv.gz in {data_dir}")
    close = pd.concat(frames, axis=1).sort_index().ffill(limit=3).dropna(how="any")
    if close.empty:
        raise ValueError("aligned close matrix is empty")
    return close


def load_aux_matrix(aux_dir: str, close: pd.DataFrame, kind: str) -> pd.DataFrame:
    frames: list[pd.Series] = []
    base = Path(aux_dir)
    for sym in close.columns:
        if kind == "funding":
            paths = sorted(base.glob(f"{sym}_funding_*.csv.gz")); value_col = "funding_rate"; date_col = "date"
        elif kind == "premium":
            paths = sorted(base.glob(f"{sym}_premium_*.csv.gz")); value_col = "close"; date_col = "close_time"
        else:
            raise ValueError(kind)
        if not paths:
            raise FileNotFoundError(f"missing {kind} aux for {sym} in {aux_dir}")
        df = pd.read_csv(paths[-1])
        if kind == "premium":
            idx = pd.to_datetime(df[date_col].astype("int64"), unit="ms", utc=True).dt.tz_convert(None)
        else:
            idx = pd.to_datetime(df[date_col])
        frames.append(pd.Series(pd.to_numeric(df[value_col], errors="coerce").to_numpy(), index=idx, name=sym).sort_index())
    mat = pd.concat(frames, axis=1).sort_index()
    return mat.reindex(close.index.union(mat.index)).sort_index().ffill().reindex(close.index).reindex(columns=close.columns)


def zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_features(close: pd.DataFrame, aux_dir: str) -> tuple[dict[str, pd.DataFrame | pd.Series], list[str]]:
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    market_ret = ret.mean(axis=1)
    market = (1.0 + market_ret).cumprod()
    funding = load_aux_matrix(aux_dir, close, "funding").fillna(0.0)
    premium = load_aux_matrix(aux_dir, close, "premium").fillna(0.0)
    feats: dict[str, pd.DataFrame | pd.Series] = {"returns": ret, "market": market}
    names: list[str] = []

    for w in [48, 96, 288, 576, 2016]:
        mom = (close / close.shift(w) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feats[f"mom_{w}"] = mom; feats[f"cs_mom_{w}"] = zscore_cs(mom); names += [f"mom_{w}", f"cs_mom_{w}"]
        feats[f"mkt_mom_{w}"] = (market / market.shift(w) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for w in [96, 288, 576, 2016]:
        vol = ret.rolling(w, min_periods=max(24, w // 4)).std().fillna(0.0)
        feats[f"vol_{w}"] = vol; feats[f"cs_vol_{w}"] = zscore_cs(vol); names += [f"vol_{w}", f"cs_vol_{w}"]
    feats["dispersion_288"] = feats["mom_288"].std(axis=1)  # type: ignore[union-attr]
    feats["dispersion_2016"] = feats["mom_2016"].std(axis=1)  # type: ignore[union-attr]

    for w, label in [(288, "1d"), (2016, "7d")]:
        fmean = funding.rolling(w, min_periods=max(24, w // 4)).mean().fillna(0.0)
        feats[f"funding_mean_{label}"] = fmean; feats[f"cs_funding_mean_{label}"] = zscore_cs(fmean); names += [f"funding_mean_{label}", f"cs_funding_mean_{label}"]
    pmean = premium.rolling(288, min_periods=24).mean().fillna(0.0)
    pmom = (premium - premium.shift(288)).fillna(0.0)
    feats["premium_mean_1d"] = pmean; feats["cs_premium_mean_1d"] = zscore_cs(pmean); names += ["premium_mean_1d", "cs_premium_mean_1d"]
    feats["premium_mom_1d"] = pmom; feats["cs_premium_mom_1d"] = zscore_cs(pmom); names += ["premium_mom_1d", "cs_premium_mom_1d"]
    # Interaction proxies that stay numeric/simple.
    feats["mom576_minus_mom48"] = feats["mom_576"] - feats["mom_48"]  # type: ignore[operator]
    feats["cs_mom576_minus_mom48"] = zscore_cs(feats["mom576_minus_mom48"])  # type: ignore[arg-type]
    names += ["mom576_minus_mom48", "cs_mom576_minus_mom48"]
    return feats, names


def sample_indices(close: pd.DataFrame, start: str, end: str, stride: int, horizon: int) -> np.ndarray:
    idx = close.index
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    pos = np.flatnonzero(mask)
    pos = pos[pos + horizon < len(idx)]
    return pos[::stride]


def make_xy(close: pd.DataFrame, feats: dict[str, pd.DataFrame | pd.Series], feature_names: list[str], positions: np.ndarray, horizon: int, target_mode: str) -> tuple[np.ndarray, np.ndarray]:
    n_assets = close.shape[1]
    xs: list[np.ndarray] = []
    for name in feature_names:
        mat = feats[name]
        assert isinstance(mat, pd.DataFrame)
        xs.append(mat.to_numpy(dtype=float)[positions].reshape(-1))
    x = np.vstack(xs).T
    prices = close.to_numpy(dtype=float)
    ymat = prices[positions + horizon] / prices[positions] - 1.0
    if target_mode == "excess":
        ymat = ymat - ymat.mean(axis=1, keepdims=True)
    elif target_mode != "raw":
        raise ValueError(target_mode)
    y = ymat.reshape(-1)
    finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    return x[finite], y[finite]


def fit_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> dict[str, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0); sd[sd < 1e-9] = 1.0
    z = (x - mu) / sd
    xtx = z.T @ z
    reg = np.eye(xtx.shape[0]) * float(l2)
    coef = np.linalg.solve(xtx + reg, z.T @ y)
    intercept = float(y.mean() - ((mu / sd) @ coef))
    return {"mu": mu, "sd": sd, "coef": coef, "intercept": np.asarray([intercept])}


def predict_score(feats: dict[str, pd.DataFrame | pd.Series], feature_names: list[str], model: dict[str, np.ndarray], close: pd.DataFrame) -> np.ndarray:
    mats = []
    for name in feature_names:
        mat = feats[name]
        assert isinstance(mat, pd.DataFrame)
        mats.append(mat.to_numpy(dtype=float))
    x3 = np.stack(mats, axis=2)  # time, asset, feature
    z = (x3 - model["mu"]) / model["sd"]
    return np.tensordot(z, model["coef"], axes=([2], [0])) + float(model["intercept"][0])


def regime_mask(feats: dict[str, pd.DataFrame | pd.Series], close: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    filt = params["regime_filter"]
    if filt == "none":
        return np.ones(len(close), dtype=bool)
    if filt == "market_mom_pos":
        s = feats["mkt_mom_2016"]
        assert isinstance(s, pd.Series)
        return s.reindex(close.index).fillna(0.0).to_numpy(dtype=float) > 0
    if filt == "dispersion_high":
        s = feats["dispersion_288"]
        assert isinstance(s, pd.Series)
        vals = s.reindex(close.index)
        q = vals.rolling(2016, min_periods=288).quantile(0.60)
        return (vals > q).fillna(False).to_numpy(dtype=bool)
    raise ValueError(filt)


def weights_from_scores(score: np.ndarray, params: dict[str, Any], leverage: float) -> np.ndarray:
    valid = np.isfinite(score)
    w = np.zeros(score.shape[0], dtype=float)
    if valid.sum() < 2:
        return w
    vals = score[valid]
    if float(vals.max() - vals.min()) < float(params.get("min_score_spread", 0.0)):
        return w
    order = np.flatnonzero(valid)[np.argsort(score[valid])[::-1]]
    k = min(int(params["top_k"]), len(order) // 2 if params["portfolio"] == "long_short" else len(order))
    if k <= 0:
        return w
    if params["portfolio"] == "long_short":
        w[order[:k]] = leverage * 0.5 / k
        w[order[-k:]] = -leverage * 0.5 / k
    elif params["portfolio"] == "long_top":
        w[order[:k]] = leverage / k
    else:
        raise ValueError(params["portfolio"])
    return w


def summarize(final_eq: float, mdd: float, rets: list[float], n_reb: int, turnover: float, start: str, end: str) -> dict[str, Any]:
    days = max(1, (pd.Timestamp(end) - pd.Timestamp(start)).days)
    cagr = (final_eq ** (365.25 / days) - 1.0) * 100.0 if final_eq > 0 else -100.0
    arr = np.asarray(rets, dtype=float)
    if len(arr) > 1 and arr.std(ddof=1) > 1e-12:
        t = float(arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr)))); p = normal_p_value(t)
    else:
        t = 0.0; p = 1.0
    mdd_pct = mdd * 100.0
    return {"sim": {"ret_pct": (final_eq - 1.0) * 100.0, "cagr_pct": cagr, "strict_mdd_pct": mdd_pct, "cagr_to_strict_mdd": cagr / mdd_pct if mdd_pct > 1e-9 else 0.0, "rebalance_count": int(n_reb), "avg_turnover_per_rebalance": turnover / n_reb if n_reb else 0.0}, "stats": {"rebalance_obs": len(arr), "rebalance_mean_ret_pct": float(arr.mean() * 100.0) if len(arr) else 0.0, "rebalance_t_stat_like": t, "rebalance_p_value_mean_ret_approx": p}}


def backtest_scores(close: pd.DataFrame, feats: dict[str, pd.DataFrame | pd.Series], score: np.ndarray, params: dict[str, Any], cfg: Cfg, start: str, end: str) -> dict[str, Any]:
    idx = close.index
    pos = np.flatnonzero((idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end)))
    if len(pos) < 2:
        return summarize(1.0, 0.0, [], 0, 0.0, start, end)
    ret_values = feats["returns"]
    assert isinstance(ret_values, pd.DataFrame)
    rets = ret_values.to_numpy(dtype=float)
    allow = regime_mask(feats, close, params)
    current = np.zeros(close.shape[1], dtype=float)
    cost_rate = cfg.fee_rate + cfg.slippage_rate
    eq = peak = 1.0; mdd = 0.0; turnover_sum = 0.0; rb_rets: list[float] = []; n_reb = 0
    stride = int(params["stride"])
    for start_i in pos[:-1:stride]:
        end_i = min(start_i + stride, pos[-1])
        target = weights_from_scores(score[start_i], params, cfg.total_leverage) if allow[start_i] else np.zeros_like(current)
        turnover = float(np.abs(target - current).sum())
        if turnover > 1e-12:
            eq *= max(0.0, 1.0 - turnover * cost_rate); turnover_sum += turnover
            peak = max(peak, eq); mdd = max(mdd, 1.0 - eq / peak if peak else 1.0)
        current = target; n_reb += 1; before = eq
        if end_i > start_i:
            block = rets[start_i + 1 : end_i + 1] @ current
            path = eq * np.cumprod(np.maximum(0.0, 1.0 + block.astype(float)))
            if len(path):
                running_peak = np.maximum.accumulate(np.maximum(path, peak))
                mdd = max(mdd, float(np.nanmax(1.0 - path / running_peak)))
                peak = max(peak, float(np.nanmax(path))); eq = float(path[-1])
        rb_rets.append(eq / before - 1.0 if before > 0 else -1.0)
        if eq <= 0:
            mdd = 1.0; break
    return summarize(eq, mdd, rb_rets, n_reb, turnover_sum, start, end)


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="MS"))


def rolling_eval(cfg: Cfg, params: dict[str, Any], train_days: int) -> dict[str, Any]:
    close = load_close_matrix(cfg.data_dir)
    feats, feature_names = build_features(close, cfg.aux_dir)
    ret_values = feats["returns"]
    assert isinstance(ret_values, pd.DataFrame)
    returns = ret_values.to_numpy(dtype=float)
    score_all = np.full(close.shape, np.nan, dtype=float)
    models: list[dict[str, Any]] = []
    idx = close.index
    eval_start = pd.Timestamp(cfg.eval_start)
    eval_end = pd.Timestamp(cfg.eval_end)
    for mstart in month_starts(cfg.eval_start, cfg.eval_end):
        mend = min(mstart + pd.offsets.MonthBegin(1), eval_end)
        train_end = mstart - pd.Timedelta(minutes=5)
        train_start = max(pd.Timestamp(cfg.train_start), train_end - pd.Timedelta(days=train_days))
        train_pos = sample_indices(close, str(train_start), str(train_end), int(params["stride"]), int(params["horizon"]))
        if len(train_pos) < 20:
            continue
        x, y = make_xy(close, feats, feature_names, train_pos, int(params["horizon"]), str(params["target"]))
        if len(y) < 100:
            continue
        model = fit_ridge(x, y, float(params["l2"]))
        score = predict_score(feats, feature_names, model, close)
        mpos = np.flatnonzero((idx >= mstart) & (idx < mend) & (idx <= eval_end))
        score_all[mpos] = score[mpos]
        models.append({"month": str(mstart.date()), "train_start": str(train_start), "train_end": str(train_end), "train_samples": int(len(y))})
    bt = backtest_scores(close, feats, score_all, params, cfg, cfg.eval_start, cfg.eval_end)
    return {"params": params, "train_days": train_days, "eval": bt, "model_windows": models}


def run(cfg: Cfg) -> dict[str, Any]:
    base_params = {"horizon": 288, "stride": 96, "target": "excess", "l2": 100.0, "top_k": 2, "portfolio": "long_short", "regime_filter": "market_mom_pos"}
    rows = []
    for d in [365]:
        for spread in [0.0, 0.00025, 0.0005, 0.001, 0.002, 0.004]:
            p = dict(base_params)
            p["min_score_spread"] = spread
            rows.append(rolling_eval(cfg, p, d))
    fixed_params = base_params
    rows.sort(key=lambda r: r["eval"]["sim"]["cagr_to_strict_mdd"], reverse=True)
    report = {"config": cfg.__dict__, "fixed_params_source": "selected on train2023/val2024 static validation; not tuned on eval", "fixed_params": fixed_params, "searched_train_days": [365], "searched_min_score_spread": [0.0, 0.00025, 0.0005, 0.001, 0.002, 0.004], "rows": rows, "selected_by_eval_for_diagnosis_only": rows[0], "leakage_guard": "monthly models fit only on data before each eval month; hyperparams fixed from prior validation; selected_by_eval is diagnostic only and must not be treated as deployable selection", "accounting_note": "same bar-level turnover-cost strict-MDD evaluator as static model"}
    out = Path(cfg.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report

def parse_args() -> Cfg:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--aux-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-start", default=Cfg.train_start); p.add_argument("--train-end", default=Cfg.train_end)
    p.add_argument("--val-start", default=Cfg.val_start); p.add_argument("--val-end", default=Cfg.val_end)
    p.add_argument("--eval-start", default=Cfg.eval_start); p.add_argument("--eval-end", default=Cfg.eval_end)
    p.add_argument("--fee-rate", type=float, default=Cfg.fee_rate); p.add_argument("--slippage-rate", type=float, default=Cfg.slippage_rate)
    p.add_argument("--total-leverage", type=float, default=Cfg.total_leverage)
    p.add_argument("--min-val-rebalances", type=int, default=Cfg.min_val_rebalances)
    p.add_argument("--val-mdd-ceiling", type=float, default=Cfg.val_mdd_ceiling)
    p.add_argument("--top-n", type=int, default=Cfg.top_n)
    return Cfg(**vars(p.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
