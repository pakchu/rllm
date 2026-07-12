"""Standalone BTC channel-breakout alpha with heterogeneous exits.

Unlike prior quantile/fixed-hold scans, this uses deterministic price-channel
events and path-dependent exits: shifted Donchian structure exits and ATR
trailing stops. Selection is test2024-only; eval2025/ytd2026 remain reporting
windows. Costs are 6bp/side and strict MDD includes intrabar adverse excursion.
"""
from __future__ import annotations

import argparse, json, math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    input_csv: str
    output: str
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    leverage: float = 1.0
    exclude_from: str = "2026-06-02"


WINDOWS = {
    "train": ("2020-01-01", "2024-01-01"),
    "test2024": ("2024-01-01", "2025-01-01"),
    "eval2025": ("2025-01-01", "2026-01-01"),
    "ytd2026": ("2026-01-01", "2026-06-02"),
}


def _load(cfg: Config) -> pd.DataFrame:
    m = pd.read_csv(cfg.input_csv, parse_dates=["date"], compression="infer")
    m["date"] = pd.to_datetime(m["date"], utc=True).dt.tz_convert(None)
    return m.sort_values("date").drop_duplicates("date", keep="last").query("date < @cfg.exclude_from").reset_index(drop=True)


def _indicators(m: pd.DataFrame, entry_n: int, exit_n: int, atr_n: int) -> dict[str, np.ndarray]:
    h=m.high.astype(float); l=m.low.astype(float); c=m.close.astype(float)
    pc=c.shift(1); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return {
        "upper": h.rolling(entry_n,min_periods=entry_n).max().shift(1).to_numpy(float),
        "lower": l.rolling(entry_n,min_periods=entry_n).min().shift(1).to_numpy(float),
        "exit_low": l.rolling(exit_n,min_periods=exit_n).min().shift(1).to_numpy(float),
        "exit_high": h.rolling(exit_n,min_periods=exit_n).max().shift(1).to_numpy(float),
        "atr": tr.ewm(alpha=1/atr_n,adjust=False,min_periods=atr_n).mean().shift(1).to_numpy(float),
    }


def _simulate(m: pd.DataFrame, ind: dict[str,np.ndarray], cfg: Config, start: str, end: str, side_mode: str, atr_mult: float, max_hold: int) -> dict[str,Any]:
    dates=pd.to_datetime(m.date); idx=np.flatnonzero(((dates>=start)&(dates<end)).to_numpy())
    if len(idx)<2: return {}
    first,last=int(idx[0]),int(idx[-1]); o=m.open.to_numpy(float); h=m.high.to_numpy(float); l=m.low.to_numpy(float); c=m.close.to_numpy(float)
    cost=(cfg.fee_rate+cfg.slippage_rate)*cfg.leverage
    eq=peak=1.0; mdd=0.0; side=0; entry_i=-1; entry_eq=1.0; trail=np.nan; trades=[]; wins=0; exits={"channel":0,"trail":0,"max_hold":0,"period_end":0}; long_n=short_n=0
    def dd(x: float) -> None:
        nonlocal mdd; mdd=max(mdd,1.0-max(0.0,x)/peak)
    for i in range(first,last):
        # Existing position is marked open-to-open; intrabar extreme is included.
        if side:
            adverse=(l[i]/o[i]-1.0) if side>0 else (1.0-h[i]/o[i])
            dd(eq*max(0.0,1.0+cfg.leverage*adverse))
            exit_px=None; reason=None
            if side>0:
                if np.isfinite(trail) and l[i]<=trail: exit_px=min(o[i],trail); reason="trail"
                elif np.isfinite(ind["exit_low"][i]) and c[i]<ind["exit_low"][i]: exit_px=o[i+1]; reason="channel"
                elif np.isfinite(ind["atr"][i]): trail=max(trail,h[i]-atr_mult*ind["atr"][i])
            else:
                if np.isfinite(trail) and h[i]>=trail: exit_px=max(o[i],trail); reason="trail"
                elif np.isfinite(ind["exit_high"][i]) and c[i]>ind["exit_high"][i]: exit_px=o[i+1]; reason="channel"
                elif np.isfinite(ind["atr"][i]): trail=min(trail,l[i]+atr_mult*ind["atr"][i])
            if reason is None and max_hold>0 and i-entry_i>=max_hold: exit_px=o[i+1]; reason="max_hold"
            if reason:
                r=side*(exit_px/o[i]-1.0); eq*=max(0.0,1.0+cfg.leverage*r); eq*=max(0.0,1.0-cost); dd(eq); peak=max(peak,eq)
                tr=eq/entry_eq-1.0; trades.append(tr); wins+=tr>0; exits[reason]+=1; side=0
            else:
                r=side*(o[i+1]/o[i]-1.0); eq*=max(0.0,1.0+cfg.leverage*r); peak=max(peak,eq)
        # Signal from completed bar i, entry at i+1 open.
        if side==0 and i+1<=last:
            sig=0
            long_cross = i > first and np.isfinite(ind["upper"][i]) and np.isfinite(ind["upper"][i-1]) and c[i] > ind["upper"][i] and c[i-1] <= ind["upper"][i-1]
            short_cross = i > first and np.isfinite(ind["lower"][i]) and np.isfinite(ind["lower"][i-1]) and c[i] < ind["lower"][i] and c[i-1] >= ind["lower"][i-1]
            if side_mode in ("long","dual") and long_cross: sig=1
            if side_mode in ("short","dual") and short_cross: sig=-1
            if sig:
                side=sig; entry_i=i+1; entry_eq=eq; eq*=max(0.0,1.0-cost); dd(eq); peak=max(peak,eq)
                trail=o[i+1]-atr_mult*ind["atr"][i] if sig>0 else o[i+1]+atr_mult*ind["atr"][i]
                long_n+=sig>0; short_n+=sig<0
    if side:
        # Force liquidation at period-end open, including costs.
        eq*=max(0.0,1.0-cost); dd(eq); peak=max(peak,eq); tr=eq/entry_eq-1.0; trades.append(tr); wins+=tr>0; exits["period_end"]+=1
    years=(pd.Timestamp(end)-pd.Timestamp(start)).total_seconds()/(365.25*86400); ret=(eq-1)*100; cagr=(eq**(1/years)-1)*100 if eq>0 else -100; md=mdd*100
    mean=float(np.mean(trades)) if trades else 0.; std=float(np.std(trades,ddof=1)) if len(trades)>1 else 0.; sharpe=mean/std*math.sqrt(len(trades)) if std>0 else 0.
    return {"return_pct":ret,"cagr_pct":cagr,"strict_mdd_pct":md,"ratio":cagr/md if md>1e-12 else 0.,"trades":len(trades),"win_rate":wins/len(trades) if trades else 0.,"sharpe_like":sharpe,"longs":long_n,"shorts":short_n,"exits":exits}


def run(cfg: Config) -> dict[str,Any]:
    m=_load(cfg); rows=[]
    # Fixed, economically interpretable grid; 12h to 14d entry channels.
    for entry_n in (288,864,2016,4032):
      for exit_frac in (0.25,0.5):
       exit_n=max(24,int(entry_n*exit_frac))
       for atr_n in (72,288):
        ind=_indicators(m,entry_n,exit_n,atr_n)
        for atr_mult in (3.0,4.0,6.0):
         for max_hold in (0,):
          for side_mode in ("long","short","dual"):
           s=_simulate(m,ind,cfg,*WINDOWS["test2024"],side_mode,atr_mult,max_hold)
           if s.get("trades",0)>=12: rows.append({"entry_bars":entry_n,"exit_bars":exit_n,"atr_bars":atr_n,"atr_mult":atr_mult,"max_hold_bars":max_hold,"side_mode":side_mode,"test2024":s})
    rows.sort(key=lambda r:(r["test2024"]["ratio"],r["test2024"]["return_pct"]),reverse=True)
    selected=rows[:100]
    for r in selected:
        ind=_indicators(m,r["entry_bars"],r["exit_bars"],r["atr_bars"])
        for w in ("train","eval2025","ytd2026"): r[w]=_simulate(m,ind,cfg,*WINDOWS[w],r["side_mode"],r["atr_mult"],r["max_hold_bars"])
        r["passes_alpha_pool"]=r["test2024"]["ratio"]>=2.5 and r["eval2025"]["ratio"]>=2.5
        r["passes_live_grade"]=r["test2024"]["ratio"]>=3 and r["eval2025"]["ratio"]>=3
        r["passes_2026_target"]=r["ytd2026"]["ratio"]>=5
    out={"as_of":datetime.now(timezone.utc).isoformat(),"config":asdict(cfg),"protocol":"deterministic shifted channels; test2024-only selection; sealed eval2025/ytd2026; 6bp/side; no fixed hold unless max_hold set; strict intrabar MDD; forced period-end close","input":{"rows":len(m),"start":str(m.date.iloc[0]),"end":str(m.date.iloc[-1])},"tested":len(rows),"selected":selected,"alpha_pool_qualifiers":[r for r in selected if r["passes_alpha_pool"]],"live_grade":[r for r in selected if r["passes_live_grade"]]}
    encoder = lambda x: x.item() if isinstance(x, np.generic) else str(x)
    Path(cfg.output).parent.mkdir(parents=True,exist_ok=True); Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=encoder)); return out


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--input-csv',required=True); p.add_argument('--output',required=True); p.add_argument('--exclude-from',default=Config.exclude_from); a=p.parse_args(); out=run(Config(**vars(a))); print(json.dumps({"tested":out["tested"],"alpha_pool":len(out["alpha_pool_qualifiers"]),"live_grade":len(out["live_grade"]),"top":out["selected"][:5]},indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))


if __name__=='__main__': main()
