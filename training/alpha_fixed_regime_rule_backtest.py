"""Backtest one fixed regime-conditioned feature rule across train/test/eval splits."""
from __future__ import annotations

import argparse, json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.alpha_feature_backtest import FeatureRuleConfig, fit_rule, simulate_rule


@dataclass(frozen=True)
class FixedRegimeRuleConfig:
    input_csv: str
    output: str
    regime_col: str
    regime_side: str
    signal_col: str
    horizon: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    eval_start: str
    eval_end: str
    regime_quantile: float = 0.25
    signal_quantile: float = 0.25
    window_size: int = 144
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    wave_trading_root: str = ""
    external_tolerance: str = "30min"


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _forward_return(open_: pd.Series, *, horizon: int, entry_delay_bars: int) -> np.ndarray:
    entry = open_.shift(-int(entry_delay_bars))
    exit_ = open_.shift(-(int(entry_delay_bars) + int(horizon)))
    return ((exit_ - entry) / entry.replace(0.0, np.nan)).to_numpy(dtype=float)


def _monthly_trade_replay(market: pd.DataFrame, features: pd.DataFrame, dates: pd.Series, gated: np.ndarray, rule: dict[str, Any], cfg: FixedRegimeRuleConfig, start: str, end: str) -> dict[str, Any]:
    opens=market["open"].to_numpy(float); highs=market["high"].to_numpy(float); lows=market["low"].to_numpy(float)
    idx=np.flatnonzero(np.asarray((dates>=pd.Timestamp(start)) & (dates<=pd.Timestamp(end)), dtype=bool))
    cost=(float(cfg.fee_rate)+float(cfg.slippage_rate))*float(cfg.leverage)
    hold=int(cfg.horizon); next_allowed=0; eq=1.0; peak=1.0; maxdd=0.0; trades=[]
    def sig(v: float) -> int:
        if not np.isfinite(v): return 0
        if v >= float(rule["high_threshold"]): return 1 if rule["high_side"]=="LONG" else -1
        if v <= float(rule["low_threshold"]): return 1 if rule["low_side"]=="LONG" else -1
        return 0
    from training.strict_bar_backtest import _drawdown_from_trough
    for pos in idx:
        if int(pos)<next_allowed: continue
        s=sig(float(gated[pos]))
        if s==0: continue
        entry=int(pos)+int(cfg.entry_delay_bars); exit_=entry+hold
        if exit_>=len(market): continue
        entry_eq=eq; eq*=max(0.0,1.0-cost); maxdd=max(maxdd,_drawdown_from_trough(peak,eq))
        for j in range(entry, exit_):
            o=float(opens[j])
            if o<=0: continue
            if s>0:
                adv=(float(lows[j])-o)/o; close=(float(opens[j+1])-o)/o
            else:
                adv=(o-float(highs[j]))/o; close=(o-float(opens[j+1]))/o
            maxdd=max(maxdd,_drawdown_from_trough(peak, eq*(1.0+float(cfg.leverage)*adv)))
            eq*=max(0.0,1.0+float(cfg.leverage)*close); peak=max(peak,eq)
        eq*=max(0.0,1.0-cost); maxdd=max(maxdd,_drawdown_from_trough(peak,eq)); peak=max(peak,eq)
        trades.append({"signal_date": str(dates.iloc[pos]), "entry_date": str(dates.iloc[entry]), "exit_date": str(dates.iloc[exit_]), "side": "LONG" if s>0 else "SHORT", "ret_pct": (eq/entry_eq-1.0)*100.0})
        next_allowed=exit_
    if not trades:
        return {"trade_count":0,"months":{},"last_trades":[]}
    df=pd.DataFrame(trades); df["month"]=pd.to_datetime(df["entry_date"]).dt.to_period("M").astype(str)
    months={}
    for m,g in df.groupby("month"):
        months[m]={"n":int(len(g)),"mean_trade_ret_pct":float(g.ret_pct.mean()),"sum_trade_ret_pct":float(g.ret_pct.sum()),"long_n":int((g.side=="LONG").sum()),"short_n":int((g.side=="SHORT").sum())}
    return {"trade_count":int(len(df)),"months":months,"last_trades":trades[-10:]}


def run(cfg: FixedRegimeRuleConfig) -> dict[str, Any]:
    market=_load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market=attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    features=build_market_feature_frame(market, window_size=cfg.window_size)
    for col in [cfg.regime_col,cfg.signal_col]:
        if col not in features.columns: raise ValueError(f"missing feature: {col}")
    dates=pd.to_datetime(market["date"])
    train_mask=np.asarray((dates>=pd.Timestamp(cfg.train_start)) & (dates<=pd.Timestamp(cfg.train_end)), dtype=bool)
    rv=features[cfg.regime_col].to_numpy(float); sv=features[cfg.signal_col].to_numpy(float)
    train_rv=rv[train_mask & np.isfinite(rv)]
    q=float(cfg.regime_quantile)
    regime_thr=float(np.quantile(train_rv, q if cfg.regime_side=="low" else 1.0-q))
    regime_mask=rv<=regime_thr if cfg.regime_side=="low" else rv>=regime_thr
    gated=np.where(regime_mask, sv, np.nan)
    fwd=_forward_return(market["open"].astype(float), horizon=cfg.horizon, entry_delay_bars=cfg.entry_delay_bars)
    base=FeatureRuleConfig(input_csv=cfg.input_csv, output="", feature=cfg.signal_col, horizon=cfg.horizon, fit_start=cfg.train_start, fit_end=cfg.train_end, eval_start=cfg.test_start, eval_end=cfg.test_end, quantile=cfg.signal_quantile, window_size=cfg.window_size, entry_delay_bars=cfg.entry_delay_bars, leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate, wave_trading_root=cfg.wave_trading_root, external_tolerance=cfg.external_tolerance)
    rule=fit_rule(dates=dates, feature_values=gated, forward_returns=fwd, cfg=base)
    test=simulate_rule(market=market, feature_values=gated, dates=dates, rule=rule, cfg=base)
    ed=asdict(base); ed.update({"eval_start":cfg.eval_start,"eval_end":cfg.eval_end})
    eval_cfg=FeatureRuleConfig(**ed)
    ev=simulate_rule(market=market, feature_values=gated, dates=dates, rule=rule, cfg=eval_cfg)
    report={"as_of":datetime.now(timezone.utc).isoformat(),"config":asdict(cfg),"fitted_regime_threshold":regime_thr,"fitted_signal_rule":rule,"test":test,"eval":ev,"monthly":{"test":_monthly_trade_replay(market,features,dates,gated,rule,cfg,cfg.test_start,cfg.test_end),"eval":_monthly_trade_replay(market,features,dates,gated,rule,cfg,cfg.eval_start,cfg.eval_end)},"leakage_guard":{"regime_threshold_fit_train_only":True,"signal_thresholds_and_direction_fit_train_only":True,"test_and_eval_after_train":True,"external_join":"backward_asof_with_tolerance","entry_delay_bars":cfg.entry_delay_bars}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True); Path(cfg.output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def parse_args():
    p=argparse.ArgumentParser()
    for x in ["input_csv","output","regime_col","regime_side","signal_col","train_start","train_end","test_start","test_end","eval_start","eval_end"]: p.add_argument(f"--{x.replace('_','-')}", required=True)
    p.add_argument("--horizon",type=int,required=True); p.add_argument("--regime-quantile",type=float,default=0.25); p.add_argument("--signal-quantile",type=float,default=0.25)
    p.add_argument("--window-size",type=int,default=144); p.add_argument("--entry-delay-bars",type=int,default=1); p.add_argument("--leverage",type=float,default=0.5); p.add_argument("--fee-rate",type=float,default=0.0004); p.add_argument("--slippage-rate",type=float,default=0.0001); p.add_argument("--wave-trading-root",default=""); p.add_argument("--external-tolerance",default="30min")
    return p.parse_args()


def main():
    r=run(FixedRegimeRuleConfig(**vars(parse_args())))
    for name in ["test","eval"]:
        sim=r[name]["sim"]; stats=r[name]["trade_stats"]
        print(json.dumps({name:{"cagr":sim["cagr_pct"],"mdd":sim["strict_mdd_pct"],"ratio":sim["cagr_to_strict_mdd"],"trades":sim["trade_entries"],"mean":stats["mean_trade_ret_pct"],"p":stats["p_value_mean_ret_approx"]}}, ensure_ascii=False))

if __name__=="__main__": main()
