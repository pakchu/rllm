"""Shared-parameter multi-asset trend/reversion sweep for Binance UM futures data."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math
import numpy as np
import pandas as pd

MA_WINDOWS=[576,2016]
MOM_WINDOWS=[96,576,2016]
STRIDES=[48,96,288]
MODES=["follow","fade"]
THRESHOLDS=[0.0,0.005,0.01]
COMBINES=["ma","mom","agree"]

@dataclass(frozen=True)
class Cfg:
    data_dir: str
    output: str
    val_start: str = "2023-01-01"
    val_end: str = "2024-12-31 23:59:59"
    eval_start: str = "2025-01-01"
    eval_end: str = "2026-06-01"
    hold_bars: int = 288
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    total_leverage: float = 1.0
    min_val_trades: int = 100
    top_n: int = 20


def load_assets(data_dir: str) -> dict[str,pd.DataFrame]:
    out={}
    for p in sorted(Path(data_dir).glob("*USDT_5m_*.csv.gz")):
        sym=p.name.split('_')[0]
        df=pd.read_csv(p, parse_dates=['date']).sort_values('date').drop_duplicates('date').reset_index(drop=True)
        close=df['close'].astype(float)
        for w in MA_WINDOWS:
            ma=close.rolling(w,min_periods=max(10,w//4)).mean()
            df[f'ma_gap_{w}']=(close/ma-1).replace([np.inf,-np.inf],np.nan).fillna(0.0)
        for w in MOM_WINDOWS:
            df[f'mom_{w}']=(close/close.shift(w)-1).replace([np.inf,-np.inf],np.nan).fillna(0.0)
        out[sym]=df
    if not out: raise FileNotFoundError(f"no *USDT_5m_*.csv.gz in {data_dir}")
    return out


def signal(row, params):
    ma=float(row[f"ma_gap_{params['ma_window']}"]); mom=float(row[f"mom_{params['mom_window']}"])
    if params['combine']=='ma': sig=ma
    elif params['combine']=='mom': sig=mom
    else: sig=(ma+mom)/2 if ma*mom>0 else 0.0
    if abs(sig)<params['threshold'] or abs(sig)<=1e-12: return None
    if params['mode']=='follow': return 'LONG' if sig>0 else 'SHORT'
    return 'SHORT' if sig>0 else 'LONG'


def backtest(assets: dict[str,pd.DataFrame], start: str, end: str, params: dict[str,Any], cfg: Cfg) -> dict[str,Any]:
    lev=cfg.total_leverage/max(1,len(assets)); cost=(cfg.fee_rate+cfg.slippage_rate)*lev
    trades=[]
    for sym,df in assets.items():
        sub_idx=df.index[(df['date']>=pd.Timestamp(start)) & (df['date']<=pd.Timestamp(end))].to_numpy()
        if len(sub_idx)==0: continue
        next_allowed=-1
        stride=int(params['stride'])
        for idx in sub_idx[::stride]:
            if idx<next_allowed: continue
            side=signal(df.iloc[idx],params)
            if side is None: continue
            entry=idx+1; exit_i=entry+cfg.hold_bars
            if exit_i>=len(df): continue
            ep=float(df.iloc[entry]['open']); xp=float(df.iloc[exit_i]['open'])
            if ep<=0: continue
            raw=(xp/ep-1.0) if side=='LONG' else (ep/xp-1.0)
            ret=lev*raw - 2*cost
            path=df.iloc[entry:exit_i+1]
            if side=='LONG': adverse=(float(path['low'].min())/ep-1.0)*lev - cost
            else: adverse=(ep/float(path['high'].max())-1.0)*lev - cost
            trades.append({'symbol':sym,'entry_date':str(df.iloc[entry]['date']),'exit_date':str(df.iloc[exit_i]['date']),'ret':float(ret),'adverse':float(adverse),'side':side})
            next_allowed=exit_i
    trades.sort(key=lambda x:x['entry_date'])
    eq=peak=1.0; mdd=0.0; rets=[]
    for t in trades:
        adverse_eq=eq*max(0.0,1.0+t['adverse']); mdd=max(mdd,1.0-adverse_eq/peak if peak>0 else 1.0)
        eq*=max(0.0,1.0+t['ret']); peak=max(peak,eq); mdd=max(mdd,1.0-eq/peak if peak>0 else 1.0); rets.append(t['ret'])
    days=max(1,(pd.Timestamp(end)-pd.Timestamp(start)).days)
    cagr=(eq**(365.25/days)-1.0)*100.0
    mdd_pct=mdd*100.0
    arr=np.asarray(rets,float)
    if len(arr)>1 and arr.std()>1e-12:
        tstat=float(arr.mean()/(arr.std(ddof=1)/np.sqrt(len(arr)))); p=float(2*(1-0.5*(1+math.erf(abs(tstat)/math.sqrt(2)))))
    else:
        tstat=0.0; p=1.0
    return {'sim':{'ret_pct':(eq-1)*100.0,'cagr_pct':cagr,'strict_mdd_pct':mdd_pct,'cagr_to_strict_mdd':cagr/mdd_pct if mdd_pct>1e-9 else 0.0,'trade_entries':len(trades),'assets':len(assets)},'trade_stats':{'n_trades':len(arr),'mean_trade_ret_pct':float(arr.mean()*100) if len(arr) else 0.0,'std_trade_ret_pct':float(arr.std(ddof=1)*100) if len(arr)>1 else 0.0,'t_stat_like':tstat,'p_value_mean_ret_approx':p},'trade_counts_by_symbol':{str(k): int(v) for k,v in (pd.Series([t['symbol'] for t in trades]).value_counts().to_dict()).items()} if trades else {}}


def run(cfg: Cfg) -> dict[str,Any]:
    assets=load_assets(cfg.data_dir)
    rows=[]
    for ma in MA_WINDOWS:
      for mom in MOM_WINDOWS:
       for stride in STRIDES:
        for mode in MODES:
         for th in THRESHOLDS:
          for combine in COMBINES:
           params={'ma_window':ma,'mom_window':mom,'stride':stride,'mode':mode,'threshold':th,'combine':combine}
           res=backtest(assets,cfg.val_start,cfg.val_end,params,cfg)
           sc=float(res['sim']['cagr_to_strict_mdd'])
           if res['sim']['trade_entries']<cfg.min_val_trades: sc-=1000
           rows.append({'score':sc,'params':params,**res})
    rows.sort(key=lambda r:r['score'], reverse=True); sel=rows[0]
    ev=backtest(assets,cfg.eval_start,cfg.eval_end,sel['params'],cfg)
    report={'config':cfg.__dict__,'assets':sorted(assets),'searched':len(rows),'top_val':rows[:cfg.top_n],'selected':sel,'eval':ev,'leakage_guard':'shared params selected on validation; eval final holdout; rolling features causal'}
    Path(cfg.output).parent.mkdir(parents=True,exist_ok=True); Path(cfg.output).write_text(json.dumps(report,indent=2,ensure_ascii=False, default=lambda o: int(o) if hasattr(o, "item") else str(o))); return report


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--output',required=True); p.add_argument('--val-start',default=Cfg.val_start); p.add_argument('--val-end',default=Cfg.val_end); p.add_argument('--eval-start',default=Cfg.eval_start); p.add_argument('--eval-end',default=Cfg.eval_end); p.add_argument('--hold-bars',type=int,default=Cfg.hold_bars); p.add_argument('--total-leverage',type=float,default=Cfg.total_leverage); p.add_argument('--min-val-trades',type=int,default=Cfg.min_val_trades); p.add_argument('--top-n',type=int,default=Cfg.top_n)
    return Cfg(**vars(p.parse_args()))

def main(): print(json.dumps(run(parse_args()),indent=2,ensure_ascii=False))
if __name__=='__main__': main()
