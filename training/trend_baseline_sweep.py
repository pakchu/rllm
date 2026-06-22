"""Leak-free broad BTC trend/reversion baseline outside the event pool.

Generates periodic predictions from causal close-vs-MA and momentum features.
Validation selects parameters; eval is final holdout.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

@dataclass(frozen=True)
class Cfg:
    market_csv: str
    output: str
    work_dir: str = "results/trend_baseline_sweep"
    val_start: str = "2023-01-01"
    val_end: str = "2024-12-31 23:59:59"
    eval_start: str = "2025-01-01"
    eval_end: str = "2026-06-01"
    min_val_trades: int = 50
    top_n: int = 20

MA_WINDOWS=[576,2016]
MOM_WINDOWS=[96,576,2016]
STRIDES=[48,96,288]
MODES=["follow","fade"]
THRESHOLDS=[0.0,0.005,0.01]


def load_market(path: str) -> pd.DataFrame:
    df=pd.read_csv(path)
    df['date']=pd.to_datetime(df['date'])
    df=df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    close=df['close'].astype(float)
    for w in MA_WINDOWS:
        ma=close.rolling(w,min_periods=max(10,w//4)).mean()
        df[f'ma_gap_{w}']=(close/ma-1.0).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    for w in MOM_WINDOWS:
        df[f'mom_{w}']=(close/close.shift(w)-1.0).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    return df


def side_from_signal(v: float, mode: str) -> str | None:
    if abs(v)<=1e-12: return None
    if mode=='follow': return 'LONG' if v>0 else 'SHORT'
    return 'SHORT' if v>0 else 'LONG'


def make_predictions(df: pd.DataFrame, start: str, end: str, params: dict[str, Any], path: str) -> dict[str, Any]:
    s=pd.Timestamp(start); e=pd.Timestamp(end)
    sub=df[(df['date']>=s)&(df['date']<=e)].copy()
    rows=[]; counts={'TRADE':0,'NO_TRADE':0,'LONG':0,'SHORT':0}
    ma_col=f"ma_gap_{params['ma_window']}"; mom_col=f"mom_{params['mom_window']}"
    stride=int(params['stride']); thr=float(params['threshold']); mode=params['mode']; combine=params['combine']
    idxs=list(range(0,len(sub),stride))
    for j in idxs:
        r=sub.iloc[j]
        ma=float(r[ma_col]); mom=float(r[mom_col])
        if combine=='ma': sig=ma
        elif combine=='mom': sig=mom
        else:
            sig=0.5*np.sign(ma)*abs(ma)+0.5*np.sign(mom)*abs(mom) if ma*mom>0 else 0.0
        side=side_from_signal(sig,mode) if abs(sig)>=thr else None
        if side:
            pred={'gate':'TRADE','side':side,'hold_bars':288,'confidence':'HIGH','family':'trend_baseline'}; scale=0.5; counts['TRADE']+=1; counts[side]+=1
        else:
            pred={'gate':'NO_TRADE','side':'NONE','hold_bars':0,'confidence':'LOW','family':'trend_baseline'}; scale=0.0; counts['NO_TRADE']+=1
        rows.append({'date':str(r['date']),'signal_pos':int(r.name),'prediction':pred,'position_scale':scale,'signal':float(sig),'ma_gap':ma,'momentum':mom})
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text('\n'.join(json.dumps(x,sort_keys=True) for x in rows)+'\n')
    return {'rows':len(rows),'counts':counts,'output':path,'params':params}


def run(c: Cfg) -> dict[str, Any]:
    df=prepare(load_market(c.market_csv)); Path(c.work_dir).mkdir(parents=True,exist_ok=True)
    vals=[]
    for ma in MA_WINDOWS:
        for mom in MOM_WINDOWS:
            for stride in STRIDES:
                for mode in MODES:
                    for th in THRESHOLDS:
                        for combine in ['ma','mom','agree']:
                            params={'ma_window':ma,'mom_window':mom,'stride':stride,'mode':mode,'threshold':th,'combine':combine}
                            pp=str(Path(c.work_dir)/f"val_{len(vals):04d}.jsonl")
                            ps=make_predictions(df,c.val_start,c.val_end,params,pp)
                            bt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=pp,market_csv=c.market_csv,output=str(Path(c.work_dir)/f"val_{len(vals):04d}.bt.json"),leverage=1.0,entry_delay_bars=1))
                            score=float(bt['sim']['cagr_to_strict_mdd'])
                            if int(bt['sim']['trade_entries'])<c.min_val_trades: score-=1000.0
                            vals.append({'score':score,'prediction_summary':ps,'val_sim':bt['sim'],'val_trade_stats':bt['trade_stats']})
    vals.sort(key=lambda r:r['score'],reverse=True); sel=vals[0]
    ep=str(Path(c.work_dir)/'selected_eval_predictions.jsonl')
    eps=make_predictions(df,c.eval_start,c.eval_end,sel['prediction_summary']['params'],ep)
    ebt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=ep,market_csv=c.market_csv,output=str(Path(c.work_dir)/'selected_eval_backtest.json'),leverage=1.0,entry_delay_bars=1))
    rep={'config':c.__dict__,'searched':len(vals),'top_val':vals[:c.top_n],'selected':sel,'eval_prediction_summary':eps,'eval_backtest':{'sim':ebt['sim'],'trade_stats':ebt['trade_stats']},'leakage_guard':'signals use rolling closes up to prediction timestamp; validation selects params; eval final'}
    Path(c.output).parent.mkdir(parents=True,exist_ok=True); Path(c.output).write_text(json.dumps(rep,indent=2,ensure_ascii=False)); return rep


def parse_args() -> Cfg:
    p=argparse.ArgumentParser(); p.add_argument('--market-csv',required=True); p.add_argument('--output',required=True); p.add_argument('--work-dir',default=Cfg.work_dir)
    p.add_argument('--val-start',default=Cfg.val_start); p.add_argument('--val-end',default=Cfg.val_end); p.add_argument('--eval-start',default=Cfg.eval_start); p.add_argument('--eval-end',default=Cfg.eval_end); p.add_argument('--min-val-trades',type=int,default=Cfg.min_val_trades); p.add_argument('--top-n',type=int,default=Cfg.top_n)
    return Cfg(**vars(p.parse_args()))

def main(): print(json.dumps(run(parse_args()),indent=2,ensure_ascii=False))
if __name__=='__main__': main()
