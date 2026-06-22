"""No-leak signal-level logistic baseline for event preference actions."""
from __future__ import annotations

import argparse, json, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

LABELS = ["NO_TRADE", "LONG", "SHORT"]

@dataclass(frozen=True)
class Cfg:
    train_candidates: str
    eval_candidates: str
    output: str
    work_dir: str = "results/event_signal_logistic"
    validation_start: str = "2023-01-01"
    validation_end: str = "2024-12-31 23:59:59"
    market_csv: str = "data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz"
    min_trade_conf: str = "0.34,0.4,0.45,0.5,0.55,0.6"
    min_val_trades: int = 50


def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def date(r): return str(r.get('date',''))
def group(rows):
    d={}
    for r in rows: d.setdefault(int(r['signal_pos']),[]).append(r)
    return [d[k] for k in sorted(d)]
def best_label(rows):
    acts=[]
    for r in rows:
        rew=r.get('reward',{}); net=float(rew.get('net_return_pct',0)); util=float(rew.get('utility',net)); side=r['side']
        scale=1.0 if net>=1.2 and util>=0.5 else (0.5 if net>=0.25 and util>=0 else 0.5)
        rank=util+0.05*net
        acts.append((rank,side))
    acts.append((0.0,'NO_TRADE'))
    return max(acts)[1]
def names(groups):
    ks=set(); toks=set()
    for g in groups:
        r=g[0]; ks.update((r.get('feature_snapshot') or {}).keys())
        for k,v in (r.get('state_tokens') or {}).items(): toks.add(f'{k}={v}')
    return sorted(ks), sorted(toks)
def xy(groups, ns, cs):
    ci={c:i for i,c in enumerate(cs)}; X=np.zeros((len(groups),len(ns)+len(cs))); y=np.zeros(len(groups),int)
    for i,g in enumerate(groups):
        r=g[0]; snap=r.get('feature_snapshot') or {}; toks=r.get('state_tokens') or {}
        X[i,:len(ns)]=[float(snap.get(n,0) or 0) for n in ns]
        base=len(ns)
        for k,v in toks.items():
            j=ci.get(f'{k}={v}')
            if j is not None: X[i,base+j]=1
        y[i]=LABELS.index(best_label(g))
    return X,y
def standardize(a,b):
    mu=a.mean(0); sd=a.std(0); sd=np.where(sd<1e-9,1,sd); return (a-mu)/sd,(b-mu)/sd
def train_softmax(X,y,lr=0.1,epochs=400,l2=1e-3):
    n,k=X.shape[0],len(LABELS); W=np.zeros((X.shape[1]+1,k)); Xb=np.c_[np.ones(n),X]
    Y=np.eye(k)[y]
    for _ in range(epochs):
        z=Xb@W; z-=z.max(1,keepdims=True); P=np.exp(z); P/=P.sum(1,keepdims=True)
        grad=Xb.T@(P-Y)/n + l2*W; grad[0]*=0
        W-=lr*grad
    return W
def pred(X,W):
    Xb=np.c_[np.ones(len(X)),X]; z=Xb@W; z-=z.max(1,keepdims=True); P=np.exp(z); P/=P.sum(1,keepdims=True); return P
def write(groups,P,path,thr):
    rows=[]; counts={"TRADE":0,"NO_TRADE":0,"LONG":0,"SHORT":0}
    for g,p in zip(groups,P):
        j=int(np.argmax(p)); lab=LABELS[j]; conf=float(p[j]); r=g[0]
        if lab!='NO_TRADE' and conf>=thr:
            pr={"gate":"TRADE","side":lab,"hold_bars":288,"confidence":"HIGH","family":"event_signal_logistic"}; scale=0.5; counts['TRADE']+=1; counts[lab]+=1
        else:
            pr={"gate":"NO_TRADE","side":"NONE","hold_bars":0,"confidence":"LOW","family":"event_signal_logistic"}; scale=0; counts['NO_TRADE']+=1
        rows.append({'date':r['date'],'signal_pos':r['signal_pos'],'prediction':pr,'position_scale':scale,'probs':dict(zip(LABELS,map(float,p)))})
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text('\n'.join(json.dumps(x,sort_keys=True) for x in rows)+'\n')
    return {'rows':len(rows),'counts':counts,'threshold':thr,'output':path}

def run(c:Cfg):
    allg=group(load(c.train_candidates)); evg=group(load(c.eval_candidates))
    fit=[g for g in allg if date(g[0])<c.validation_start]; val=[g for g in allg if c.validation_start<=date(g[0])<=c.validation_end]
    ns,cs=names(fit); Xf,yf=xy(fit,ns,cs); Xv,yv=xy(val,ns,cs); Xe,ye=xy(evg,ns,cs)
    Xfz,Xvz=standardize(Xf,Xv); _,Xez=standardize(Xf,Xe); W=train_softmax(Xfz,yf)
    Pv=pred(Xvz,W); Pe=pred(Xez,W); Path(c.work_dir).mkdir(parents=True,exist_ok=True)
    vals=[]
    for t in [float(x) for x in c.min_trade_conf.split(',') if x.strip()]:
        pp=str(Path(c.work_dir)/f'val_t{t}.jsonl'); ps=write(val,Pv,pp,t); bt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=pp,market_csv=c.market_csv,output=str(Path(c.work_dir)/f'val_t{t}.bt.json'),leverage=1.0,entry_delay_bars=1))
        score=bt['sim']['cagr_to_strict_mdd'];
        if bt['sim']['trade_entries']<c.min_val_trades: score-=1000
        vals.append({'threshold':t,'prediction_summary':ps,'val_sim':bt['sim'],'val_trade_stats':bt['trade_stats'],'score':score})
    vals.sort(key=lambda x:x['score'],reverse=True); sel=vals[0]
    ep=str(Path(c.work_dir)/'selected_eval_predictions.jsonl'); eps=write(evg,Pe,ep,sel['threshold']); ebt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=ep,market_csv=c.market_csv,output=str(Path(c.work_dir)/'selected_eval_backtest.json'),leverage=1.0,entry_delay_bars=1))
    rep={'config':c.__dict__,'rows':{'fit':len(fit),'val':len(val),'eval':len(evg)},'features':{'numeric':len(ns),'categorical':len(cs)},'label_counts':{'fit':dict(zip(LABELS,np.bincount(yf,minlength=3).tolist())),'val':dict(zip(LABELS,np.bincount(yv,minlength=3).tolist())),'eval':dict(zip(LABELS,np.bincount(ye,minlength=3).tolist()))},'top_val':vals,'selected':sel,'eval_prediction_summary':eps,'eval_backtest':{'sim':ebt['sim'],'trade_stats':ebt['trade_stats']}}
    Path(c.output).write_text(json.dumps(rep,indent=2)); return rep

def main():
    p=argparse.ArgumentParser();
    for f in Cfg.__dataclass_fields__.values(): p.add_argument('--'+f.name.replace('_','-'), default=f.default, required=f.default is None)
    a=vars(p.parse_args()); a['min_val_trades']=int(a['min_val_trades']); print(json.dumps(run(Cfg(**a)),indent=2))
if __name__=='__main__': main()
