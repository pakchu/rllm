"""Sweep gate bias for an already-scored two-step REX policy.

Bias is selected on train only; test/eval are reported as untouched validation.
"""
from __future__ import annotations
import argparse,json
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Any
import numpy as np
from training.event_candidate_pool_probe import EventPoolConfig,_load_market,_simulate_rows

@dataclass(frozen=True)
class Cfg:
    scores_json: str
    output_json: str
    market_csv: str
    bias_min: float = -3.0
    bias_max: float = 3.0
    bias_step: float = 0.1
    hold_bars: int = 144
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    max_train_mdd: float = 20.0
    min_train_trades: int = 50


def _bt(rows:list[dict[str,Any]], bias:float, pri:dict[str,float], market, cfg:Cfg)->dict[str,Any]:
    trades=[]
    for r in rows:
        gs=r['gate_scores']; ss=r['side_scores']
        d=(float(gs['TRADE'])-float(pri.get('TRADE',0.0)))-(float(gs['NO_TRADE'])-float(pri.get('NO_TRADE',0.0)))+float(bias)
        gate='TRADE' if d>=0 else 'NO_TRADE'
        side=max(['LONG','SHORT'], key=lambda c: float(ss[c]))
        if gate=='TRADE':
            trades.append({'date':r['date'],'signal_date':r['date'],'side':side,'family':'rex_two_step_gate_bias','strength':abs(d),'score_mean':abs(d)})
    ecfg=EventPoolConfig(input_csv=cfg.market_csv,output='',hold_bars=cfg.hold_bars,entry_delay_bars=cfg.entry_delay_bars,leverage=cfg.leverage,fee_rate=cfg.fee_rate,slippage_rate=cfg.slippage_rate)
    res=_simulate_rows(trades,market,ecfg)
    return {'predicted_trade_rows':len(trades),'sim':res.get('sim',{}),'trade_stats':res.get('trade_stats',{})}


def _score_train(x:dict[str,Any], cfg:Cfg)->tuple:
    sim=x['splits']['train']['sim']; ts=x['splits']['train']['trade_stats']
    r=float(sim.get('cagr_to_strict_mdd',-999) or -999)
    c=float(sim.get('cagr_pct',0) or 0); m=float(sim.get('strict_mdd_pct',999) or 999); n=int(ts.get('n_trades',0) or 0)
    ok=(m<=cfg.max_train_mdd and n>=cfg.min_train_trades and c>0)
    return (1 if ok else 0, r, c, -m, n)


def run(cfg:Cfg)->dict[str,Any]:
    obj=json.load(open(cfg.scores_json)); rows=obj['score_rows']; pri=obj.get('gate_prior') or {'TRADE':0.0,'NO_TRADE':0.0}
    market=_load_market(cfg.market_csv)
    by={sp:[r for r in rows if r.get('split')==sp] for sp in ['train','test','eval']}
    trials=[]
    biases=np.arange(cfg.bias_min, cfg.bias_max+cfg.bias_step/2, cfg.bias_step)
    for b in biases:
        item={'bias':round(float(b),6),'splits':{sp:_bt(part,float(b),pri,market,cfg) for sp,part in by.items()}}
        item['rank_tuple']=_score_train(item,cfg)
        trials.append(item)
    trials.sort(key=lambda x:x['rank_tuple'], reverse=True)
    report={'config':asdict(cfg),'source_scores':cfg.scores_json,'gate_prior':pri,'selection_protocol':'bias selected on train only; test/eval untouched','top_train':trials[:30]}
    Path(cfg.output_json).parent.mkdir(parents=True,exist_ok=True)
    Path(cfg.output_json).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return {k:v for k,v in report.items() if k!='top_train'} | {'best':trials[0] if trials else None}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--scores-json',required=True); p.add_argument('--output-json',required=True); p.add_argument('--market-csv',required=True)
    p.add_argument('--bias-min',type=float,default=-3.0); p.add_argument('--bias-max',type=float,default=3.0); p.add_argument('--bias-step',type=float,default=0.1)
    print(json.dumps(run(Cfg(**vars(p.parse_args()))),indent=2,ensure_ascii=False))
