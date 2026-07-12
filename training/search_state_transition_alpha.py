"""Leak-aware standalone BTC alpha scan based on market-state transitions.

Candidate thresholds are fitted only on train (<2024). Candidate ranking and
selection use test2024 only; eval2025 and ytd2026 are attached after selection.
The scan uses 6bp/side (5bp fee + 1bp slippage), next-open entries, full-window
CAGR, and strict intrabar MDD with period-contained exits.
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

from training.long_regime_combo_scan import LongComboScanConfig, _load_market, _split_mask, _strict_long_sim
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class Config(LongComboScanConfig):
    exclude_from: str = "2026-06-02"
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    leverage: float = 1.0
    hold_bars: str = "24,48,72,144,216,288"
    stride_bars: str = "6,12"
    min_test_trades: int = 25
    top_n: int = 400
    oi_csv: str = ""


WINDOWS = {
    "train": ("2020-01-01", "2024-01-01"),
    "test2024": ("2024-01-01", "2025-01-01"),
    "eval2025": ("2025-01-01", "2026-01-01"),
    "ytd2026": ("2026-01-01", "2026-06-02"),
}


def _z(s: pd.Series, n: int) -> pd.Series:
    mean = s.rolling(n, min_periods=n).mean()
    std = s.rolling(n, min_periods=n).std().replace(0.0, np.nan)
    return (s - mean) / std


def _features(m: pd.DataFrame) -> pd.DataFrame:
    close = m["close"].astype(float)
    op = m["open"].astype(float)
    high = m["high"].astype(float)
    low = m["low"].astype(float)
    ret = np.log(close).diff()
    oi = np.log(m["open_interest"].astype(float).replace(0.0, np.nan)).diff()
    qv = m["quote_asset_volume"].astype(float).replace(0.0, np.nan)
    buy = m["taker_buy_quote"].astype(float)
    imb = (2.0 * buy / qv - 1.0).clip(-1.0, 1.0)
    tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1) / close.shift()
    body = (close-op).abs().replace(0.0, np.nan)
    f = pd.DataFrame(index=m.index)
    # Directional efficiency: persistent displacement rather than raw momentum.
    for n in (24, 72, 144):
        f[f"ret_{n}"] = np.log(close / close.shift(n))
        f[f"eff_{n}"] = f[f"ret_{n}"] / ret.abs().rolling(n, min_periods=n).sum().replace(0.0, np.nan)
    # Volatility transition: recent realized range expansion relative to prior state.
    f["rv_12_144"] = tr.rolling(12, min_periods=12).mean() / tr.rolling(144, min_periods=144).mean().replace(0.0, np.nan)
    f["rv_24_288"] = tr.rolling(24, min_periods=24).mean() / tr.rolling(288, min_periods=288).mean().replace(0.0, np.nan)
    f["compression_144"] = tr.rolling(144, min_periods=144).mean() / tr.rolling(576, min_periods=576).mean().replace(0.0, np.nan)
    # Order-flow persistence and its change; no future bars are used.
    f["imb_12"] = imb.rolling(12, min_periods=12).mean()
    f["imb_72"] = imb.rolling(72, min_periods=72).mean()
    f["imb_accel"] = f["imb_12"] - f["imb_72"]
    f["volume_z_288"] = _z(np.log1p(qv), 288)
    # OI impulse versus price: participation-confirmed move and squeeze release.
    f["oi_ret_12"] = oi.rolling(12, min_periods=12).sum()
    f["oi_ret_72"] = oi.rolling(72, min_periods=72).sum()
    f["oi_z_12"] = _z(f["oi_ret_12"], 288)
    f["oi_price_gap"] = _z(f["oi_ret_72"], 576) - _z(f["ret_72"], 576)
    # Rejection/absorption descriptors.
    f["lower_wick_body"] = (np.minimum(op, close)-low).clip(lower=0.0) / body
    f["upper_wick_body"] = (high-np.maximum(op, close)).clip(lower=0.0) / body
    f["flow_price_div"] = _z(f["imb_72"], 576) - _z(f["ret_72"], 576)
    return f.replace([np.inf, -np.inf], np.nan)


def _regimes(m: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fixed past-only bullish regimes, independent of T/T/E outcomes."""
    close = m["close"].astype(float)
    sma30 = close.rolling(8640, min_periods=8640).mean()
    sma90 = close.rolling(25920, min_periods=25920).mean()
    return {
        "all": np.ones(len(m), dtype=bool),
        "above_sma90": (close > sma90).fillna(False).to_numpy(bool),
        "sma30_rising": ((close > sma30) & (sma30 > sma30.shift(2016))).fillna(False).to_numpy(bool),
        "bull_stack": ((close > sma30) & (sma30 > sma90) & (sma90 > sma90.shift(2016))).fillna(False).to_numpy(bool),
    }


def _q(f: pd.DataFrame, train: np.ndarray, name: str, q: float) -> float | None:
    if name not in f.columns:
        return None
    x = f.loc[train, name].dropna().to_numpy(float)
    x = x[np.isfinite(x)]
    if len(x) < 100 or np.nanstd(x) <= 1e-12:
        return None
    return float(np.quantile(x, q))


def _specs(f: pd.DataFrame, train: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    def add(family: str, conds: list[tuple[str, str, float]]) -> None:
        built = []
        for n, op, q in conds:
            thr = _q(f, train, n, q)
            if thr is None:
                return
            built.append({"feature": n, "op": op, "q": q, "threshold": thr})
        out.append({"family": family, "conditions": built})
    # Predeclared families; the grid broadens thresholds, not formulas.
    for n in (24,72,144):
        for eq in (0.7,0.8,0.9):
            for rq in (0.6,0.7,0.8): add("directional_efficiency", [(f"eff_{n}","ge",eq),(f"ret_{n}","ge",rq)])
    for rv in ("rv_12_144","rv_24_288"):
        for vq in (0.7,0.8,0.9):
            for rq in (0.6,0.7,0.8): add("volatility_release", [(rv,"ge",vq),("ret_24","ge",rq),("compression_144","le",0.4)])
    for iq in (0.7,0.8,0.9):
        for vq in (0.6,0.7,0.8):
            for rq in (0.5,0.6,0.7): add("flow_acceleration", [("imb_accel","ge",iq),("volume_z_288","ge",vq),("ret_24","ge",rq)])
    for oq in (0.7,0.8,0.9):
        for iq in (0.6,0.7,0.8):
            for rq in (0.5,0.6,0.7): add("oi_flow_confirmation", [("oi_z_12","ge",oq),("imb_12","ge",iq),("ret_24","ge",rq)])
    for gq in (0.1,0.2,0.3):
        for iq in (0.7,0.8,0.9):
            for rq in (0.6,0.7,0.8): add("oi_squeeze_release", [("oi_price_gap","le",gq),("imb_accel","ge",iq),("ret_24","ge",rq)])
    for wq in (0.8,0.9,0.95):
        for dq in (0.1,0.2,0.3):
            for iq in (0.6,0.7,0.8): add("sell_absorption_reversal", [("lower_wick_body","ge",wq),("flow_price_div","le",dq),("imb_accel","ge",iq)])
    return out


def _mask(f: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    m = np.ones(len(f), bool)
    for c in spec["conditions"]:
        x=f[c["feature"]].to_numpy(float); m &= np.isfinite(x) & ((x>=c["threshold"]) if c["op"]=="ge" else (x<=c["threshold"]))
    return m


def _score(market: pd.DataFrame, dates: pd.Series, active: np.ndarray, cfg: Config, hold: int, stride: int, window: str) -> dict[str, Any]:
    start,end=WINDOWS[window]; wm=_split_mask(dates,start,end)
    pos=np.arange(max(576,cfg.window_size-1),len(market)-hold-cfg.entry_delay_bars-1,stride,dtype=np.int64)
    p=pos[active[pos]&wm[pos]]
    sim,rets=_strict_long_sim(p,market=market,hold_bars=hold,entry_delay_bars=cfg.entry_delay_bars,leverage=cfg.leverage,fee_rate=cfg.fee_rate,slippage_rate=cfg.slippage_rate,annualization_start=start,annualization_end=end)
    ts=_trade_stats(rets)
    sharpe_like = float(ts["effect_size_d"]) * np.sqrt(max(0, int(ts["n_trades"])))
    return {"return_pct":sim["total_return_pct"],"cagr_pct":sim["cagr_pct"],"strict_mdd_pct":sim["strict_mdd_pct"],"ratio":sim["cagr_to_strict_mdd"],"trades":sim["trade_entries"],"win_rate":sim["win_rate"],"sharpe_like":sharpe_like,"p":ts.get("p_value_mean_ret_approx"),"signals":len(p)}


def run(cfg: Config) -> dict[str, Any]:
    market=_load_market(cfg)
    if getattr(cfg, "oi_csv", "") and Path(str(cfg.oi_csv)).exists():
        oi = pd.read_csv(str(cfg.oi_csv))
        oi["date"] = pd.to_datetime(oi["date"], utc=True, errors="raise").dt.tz_convert(None)
        oi_col = next((c for c in ("open_interest", "sum_open_interest", "openInterest") if c in oi.columns), None)
        if oi_col is None:
            raise ValueError(f"OI CSV has no supported open-interest column: {list(oi.columns)}")
        market = market.merge(oi[["date", oi_col]].rename(columns={oi_col: "open_interest"}), on="date", how="left")
    market["open_interest"] = market.get("open_interest", pd.Series(0.0, index=market.index)).astype(float).replace(0.0, np.nan).ffill().fillna(0.0)
    dates=pd.to_datetime(market.date); f=_features(market); regimes=_regimes(market); train=_split_mask(dates,*WINDOWS["train"])
    specs=_specs(f,train); holds=[int(x) for x in cfg.hold_bars.split(',')]; strides=[int(x) for x in cfg.stride_bars.split(',')]
    test_rows=[]
    for spec in specs:
        base=_mask(f,spec)
        for regime_name, regime in regimes.items():
            active=base & regime
            if int((active&train).sum())<200: continue
            for hold in holds:
                for stride in strides:
                    s=_score(market,dates,active,cfg,hold,stride,"test2024")
                    if s["trades"]>=cfg.min_test_trades:
                        test_rows.append({"family":spec["family"],"regime":regime_name,"conditions":spec["conditions"],"hold_bars":hold,"stride_bars":stride,"test2024":s})
    test_rows.sort(key=lambda r:(r["test2024"]["ratio"],r["test2024"]["return_pct"],r["test2024"]["trades"]),reverse=True)
    selected=test_rows[:cfg.top_n]
    for r in selected:
        active=_mask(f,r) & regimes[r["regime"]]
        for w in ("train","eval2025","ytd2026"):
            r[w]=_score(market,dates,active,cfg,r["hold_bars"],r["stride_bars"],w)
        r["passes_alpha_pool"] = r["test2024"]["ratio"]>=2.5 and r["eval2025"]["ratio"]>=2.5
        r["passes_live_grade"] = r["test2024"]["ratio"]>=3 and r["eval2025"]["ratio"]>=3
        r["passes_2026_target"] = r["ytd2026"]["ratio"]>=5
    report={"as_of":datetime.now(timezone.utc).isoformat(),"config":asdict(cfg),"protocol":"train-only thresholds; ranked by test2024 only; eval2025/ytd2026 attached after selection; 6bp/side; full-window CAGR; strict intrabar MDD", "input":{"rows":len(market),"start":str(dates.iloc[0]),"end":str(dates.iloc[-1])},"tested":len(test_rows),"selected":selected,"alpha_pool_qualifiers":[r for r in selected if r["passes_alpha_pool"]],"live_grade":[r for r in selected if r["passes_live_grade"]]}
    Path(cfg.output).parent.mkdir(parents=True,exist_ok=True); Path(cfg.output).write_text(json.dumps(report,indent=2,ensure_ascii=False)); return report


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--input-csv',required=True); p.add_argument('--output',required=True); p.add_argument('--funding-csv',default=''); p.add_argument('--premium-csv',default=''); p.add_argument('--oi-csv',default=Config.oi_csv); p.add_argument('--exclude-from',default=Config.exclude_from)
    a=p.parse_args(); r=run(Config(**vars(a))); print(json.dumps({"output":a.output,"tested":r["tested"],"alpha_pool":len(r["alpha_pool_qualifiers"]),"live_grade":len(r["live_grade"]),"top":r["selected"][:10]},ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
